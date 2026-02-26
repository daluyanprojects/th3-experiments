# hyetograph.py
"""
Fixed parameters (same for all 50 training scenarios):
    D_hr_effective : 1.0 hour
    dt_min         : 5 minutes
    n_real_steps   : 12  (5-min blocks covering 1 hour)
    n_steps        : 13  (12 real + 1 trailing zero appended during training)

User-defined parameters:
    storm_type : one of 'front-loaded', 'balanced', 'back-loaded', 'triangular'
    depth_mm   : total storm depth P in mm
    tpeak      : peak time fraction (only for triangular; None for SCS types)
"""
import math
import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional, Dict, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
D_HR        = 1.0          
DT_MIN      = 5.0          
DT_HR       = DT_MIN / 60 
N_REAL      = 12           # real intensity blocks
N_STEPS     = 13           # N_REAL + 1 trailing zero (matches training CSV format)

# Beta distribution parameters per SCS storm type
# pattern: (alpha, beta)
SCS_BETA_PARAMS: Dict[str, Tuple[float, float]] = {
    'front-loaded': (2.0, 5.0),
    'balanced':     (2.5, 2.5),
    'back-loaded':  (5.0, 2.0),
}

# Storm type → one-hot index (0-indexed)
STORM_TYPE_INDEX: Dict[str, int] = {
    'front-loaded': 0,
    'balanced':     1,
    'back-loaded':  2,
    'triangular':   3,
}

# Training distribution ranges from info.csv — used for validation warnings
TRAINING_RANGES: Dict[str, Dict] = {
    'front-loaded': {'depth_mm': (6,  78),  'tpeak': None},
    'balanced':     {'depth_mm': (19, 78),  'tpeak': None},
    'back-loaded':  {'depth_mm': (7,  75),  'tpeak': None},
    'triangular':   {'depth_mm': (5,  77),  'tpeak': (0.1, 0.9)},
}


# ── Input Validation ──────────────────────────────────────────────────────────
def validate_user_inputs(storm_type : str, depth_mm: float, tpeak: Optional[float] = None) -> list:
    warnings = []

    if storm_type not in STORM_TYPE_INDEX:
        raise ValueError(
            f"Unknown storm_type '{storm_type}'. "
            f"Must be one of: {list(STORM_TYPE_INDEX.keys())}"
        )

    ranges = TRAINING_RANGES[storm_type]

    # Depth range check
    d_min, d_max = ranges['depth_mm']
    if not (d_min <= depth_mm <= d_max):
        warnings.append(
            f"depth_mm={depth_mm} is outside the training range "
            f"[{d_min}, {d_max}] mm for '{storm_type}' storms. "
            f"Predictions may be less reliable."
        )

    # tpeak checks
    if storm_type == 'triangular':
        if tpeak is None:
            raise ValueError("tpeak is required for triangular storms.")
        tp_min, tp_max = ranges['tpeak']
        if not (tp_min <= tpeak <= tp_max):
            warnings.append(
                f"tpeak={tpeak} is outside the training range "
                f"[{tp_min}, {tp_max}] for triangular storms. "
                f"Predictions may be less reliable."
            )
    else:
        if tpeak is not None:
            warnings.append(
                f"tpeak={tpeak} was provided but will be ignored "
                f"for '{storm_type}' storms."
            )

    return warnings

def _generate_scs_hyetograph(depth_mm: float, alpha: float, beta: float) -> np.ndarray:
    """
    Generate 12-step SCS hyetograph using Beta PDF evaluated at block midpoints.
    The tool evaluates the Beta distribution at t_mid of each block, then scales so total depth sums exactly to depth_mm.
    """

    # Midpoints of each block as fractions of duration
    # Block i spans [i/12, (i+1)/12], midpoint = (i + 0.5) / 12
    t_mids = np.array([(i + 0.5) / N_REAL for i in range(N_REAL)])

    # Beta PDF values at midpoints (unnormalized weights)
    pdf_vals = beta_dist.pdf(t_mids, alpha, beta)

    # Normalize weights so they sum to 1 → depth fractions per block
    weights = pdf_vals / pdf_vals.sum()

    # Depth per block (mm)
    block_depths = weights * depth_mm

    # Intensity per block (mm/hr)
    intensities = block_depths / DT_HR

    return intensities.astype(np.float32)

def _generate_triangular_hyetograph(depth_mm: float, tpeak: float) -> np.ndarray:
    n_rising  = math.floor(tpeak * N_REAL) + 1  
    n_falling = N_REAL - n_rising

    # Guard against degenerate cases
    if n_rising == 0:
        n_rising = 1
        n_falling = N_REAL - 1
    if n_falling == 0:
        n_falling = 1
        n_rising = N_REAL - 1

    # Solving for peak:
    peak_intensity = depth_mm / (DT_HR * ((n_rising + 1) / 2.0 + n_falling / 2.0))

    # Rising: increment to peak
    rising_increment = peak_intensity / n_rising
    rising = np.array([rising_increment * (i + 1) for i in range(n_rising)],
                      dtype=np.float32)

    # Falling: (peak - decrement) down to decrement — never reaches zero
    falling_decrement = peak_intensity / (n_falling + 1)                        
    falling = np.array([peak_intensity - falling_decrement * (i + 1)
                        for i in range(n_falling)],
                       dtype=np.float32)

    intensities = np.concatenate([rising, falling])
    assert len(intensities) == N_REAL
    return intensities

# ── Main Generator ────────────────────────────────────────────────────────────
def generate_hyetograph(storm_type : str, depth_mm: float, tpeak: Optional[float] = None,) -> np.ndarray:
    if storm_type in SCS_BETA_PARAMS:
        alpha, beta = SCS_BETA_PARAMS[storm_type]
        intensities = _generate_scs_hyetograph(depth_mm, alpha, beta)
    elif storm_type == 'triangular':
        intensities = _generate_triangular_hyetograph(depth_mm, tpeak)
    else:
        raise ValueError(f"Unknown storm_type: '{storm_type}'")
    sequence = np.append(intensities, 0.0).astype(np.float32)
    assert sequence.shape == (N_STEPS,), f"Expected (13,), got {sequence.shape}"
    return sequence

# ── Normalizer ────────────────────────────────────────────────────────────────
def normalize_hyetograph(raw_sequence : np.ndarray, rain_min: float, rain_max: float) -> np.ndarray:
    return ((raw_sequence - rain_min) / (rain_max - rain_min + 1e-8)).astype(np.float32)

# ── One-Hot Encoder ───────────────────────────────────────────────────────────
def get_storm_type_onehot(storm_type: str) -> np.ndarray:
    onehot = np.zeros(4, dtype=np.float32)
    onehot[STORM_TYPE_INDEX[storm_type]] = 1.0
    return onehot

# ── Main Entry Point ──────────────────────────────────────────────────────────
def build_conditioning_vector(
    storm_type  : str,
    depth_mm    : float,
    rain_min    : float,
    rain_max    : float,
    depth_min   : float,
    depth_max   : float,
    hasDrainage : bool,
    hasSoil     : bool,
    tpeak       : Optional[float] = None,
) -> Tuple[np.ndarray, list]:

    # 1. Validate
    warnings = validate_user_inputs(storm_type, depth_mm, tpeak)

    # 2. Generate raw (13,) hyetograph
    raw_sequence = generate_hyetograph(storm_type, depth_mm, tpeak)

    # 3. Normalize → (13,)
    norm_sequence = normalize_hyetograph(raw_sequence, rain_min, rain_max)

    # 4. Storm type one-hot → (4,)
    onehot = get_storm_type_onehot(storm_type)

    # 5. Availability flags → (2,)
    flags = np.array([float(hasDrainage), float(hasSoil)], dtype=np.float32)

    # 6. tpeak and depth_norm → (2,)
    tpeak_val  = float(tpeak) if (storm_type == 'triangular' and tpeak is not None) else 0.0
    depth_norm = float(np.clip(
        (depth_mm - depth_min) / (depth_max - depth_min + 1e-8),
        -0.1, 1.1
    ))
    extra = np.array([tpeak_val, depth_norm], dtype=np.float32)

    # 7. Concatenate all → (21,)
    conditioning = np.concatenate([norm_sequence, onehot, flags, extra])

    assert conditioning.shape == (21,), f"Expected (21,), got {conditioning.shape}"

    return conditioning, warnings


def verify_against_csv(storm_type: str, depth_mm: float, csv_sequence : np.ndarray, tpeak: Optional[float] = None, tolerance: float = 0.5) -> None:
    generated = generate_hyetograph(storm_type, depth_mm, tpeak)

    print(f"\nVerification — {storm_type}, {depth_mm}mm"
          + (f", tpeak={tpeak}" if tpeak else ""))
    print(f"{'Step':<6} {'Generated':>12} {'CSV':>12} {'Diff':>10} {'Status':>8}")
    print("-" * 54)

    all_pass = True
    for i, (gen, csv) in enumerate(zip(generated, csv_sequence)):
        diff   = abs(gen - csv)
        status = "✓" if diff <= tolerance else "✗ FAIL"
        if diff > tolerance:
            all_pass = False
        print(f"{i+1:<6} {gen:>12.4f} {csv:>12.4f} {diff:>10.4f} {status:>8}")

    print("-" * 54)
    print(f"Max diff : {np.max(np.abs(generated - csv_sequence)):.4f} mm/hr")
    print(f"Result   : {'ALL PASS ✓' if all_pass else 'FAILED ✗ — check Beta params or triangular formula'}")