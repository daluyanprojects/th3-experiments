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

Conditioning vector format (matches training):
    Shape  : (4,)
    dims   : ['pattern_type', 'depth_normalized', 'tpeak', 'has_tpeak']
    ──────────────────────────────────────────────────────────────────
    [0] pattern_type    : float index  0=front-loaded, 1=balanced,
                                       2=back-loaded,  3=triangular
    [1] depth_normalized: (depth_mm − DEPTH_MIN) / (DEPTH_MAX − DEPTH_MIN)
    [2] tpeak           : fractional peak time [0,1]; 0.0 for SCS storms
    [3] has_tpeak       : 1.0 if triangular, 0.0 otherwise
"""

import math
import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional, Dict, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────
D_HR    = 1.0
DT_MIN  = 5.0
DT_HR   = DT_MIN / 60
N_REAL  = 12          
N_STEPS = 13         

# Beta distribution parameters per SCS storm type
SCS_BETA_PARAMS: Dict[str, Tuple[float, float]] = {
    'front-loaded': (2.0, 5.0),
    'balanced':     (2.5, 2.5),
    'back-loaded':  (5.0, 2.0),
}

STORM_TYPE_INDEX: Dict[str, int] = {
    'front-loaded': 0,
    'balanced':     1,
    'back-loaded':  2,
    'triangular':   3,
}

DEPTH_MIN: float = 5.0  
DEPTH_MAX: float = 78.0  

RAIN_MIN: float = 0.0
RAIN_MAX: float = 189.85  

TRAINING_RANGES: Dict[str, Dict] = {
    'front-loaded': {'depth_mm': (6,  78),  'tpeak': None},
    'balanced':     {'depth_mm': (19, 78),  'tpeak': None},
    'back-loaded':  {'depth_mm': (7,  75),  'tpeak': None},
    'triangular':   {'depth_mm': (5,  77),  'tpeak': (0.1, 0.9)},
}


# ── Input Validation ───────────────────────────────────────────────────────────
def validate_user_inputs(
    storm_type: str,
    depth_mm: float,
    tpeak: Optional[float] = None,
) -> list:
    warnings = []

    if storm_type not in STORM_TYPE_INDEX:
        raise ValueError(
            f"Unknown storm_type '{storm_type}'. "
            f"Must be one of: {list(STORM_TYPE_INDEX.keys())}"
        )

    ranges = TRAINING_RANGES[storm_type]

    d_min, d_max = ranges['depth_mm']
    if not (d_min <= depth_mm <= d_max):
        warnings.append(
            f"depth_mm={depth_mm} is outside the training range "
            f"[{d_min}, {d_max}] mm for '{storm_type}' storms. "
            f"Predictions may be less reliable."
        )

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


# ── Hyetograph Generators ──────────────────────────────────────────────────────
def _generate_scs_hyetograph(depth_mm: float, alpha: float, beta: float) -> np.ndarray:
    t_mids    = np.array([(i + 0.5) / N_REAL for i in range(N_REAL)])
    pdf_vals  = beta_dist.pdf(t_mids, alpha, beta)
    weights   = pdf_vals / pdf_vals.sum()
    block_depths = weights * depth_mm
    intensities  = block_depths / DT_HR
    return intensities.astype(np.float32)


def _generate_triangular_hyetograph(depth_mm: float, tpeak: float) -> np.ndarray:
    n_rising  = math.floor(tpeak * N_REAL) + 1
    n_falling = N_REAL - n_rising

    # Guard degenerate cases
    if n_rising == 0:
        n_rising, n_falling = 1, N_REAL - 1
    if n_falling == 0:
        n_falling, n_rising = 1, N_REAL - 1

    peak_intensity   = depth_mm / (DT_HR * ((n_rising + 1) / 2.0 + n_falling / 2.0))
    rising_increment = peak_intensity / n_rising
    falling_decrement = peak_intensity / (n_falling + 1)

    rising  = np.array([rising_increment * (i + 1)
                        for i in range(n_rising)],  dtype=np.float32)
    falling = np.array([peak_intensity - falling_decrement * (i + 1)
                        for i in range(n_falling)], dtype=np.float32)

    intensities = np.concatenate([rising, falling])
    assert len(intensities) == N_REAL
    return intensities


# ── Main Hyetograph Generator ──────────────────────────────────────────────────
def generate_hyetograph(
    storm_type: str,
    depth_mm: float,
    tpeak: Optional[float] = None,
) -> np.ndarray:

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


# ── Normaliser ─────────────────────────────────────────────────────────────────
def normalize_hyetograph(
    raw_sequence: np.ndarray,
    rain_min: float = RAIN_MIN,
    rain_max: float = RAIN_MAX,
) -> np.ndarray:
    return ((raw_sequence - rain_min) / (rain_max - rain_min + 1e-8)).astype(np.float32)


# ── One-Hot Helper (kept for downstream use if needed) ─────────────────────────
def get_storm_type_onehot(storm_type: str) -> np.ndarray:
    onehot = np.zeros(4, dtype=np.float32)
    onehot[STORM_TYPE_INDEX[storm_type]] = 1.0
    return onehot


# ── Conditioning Vector ────────────────────────────────────────────────────────
def build_conditioning_vector(
    storm_type: str,
    depth_mm: float,
    tpeak: Optional[float] = None,
) -> Tuple[np.ndarray, list]:

    # 1. Validate inputs and collect OOD warnings
    warn_list = validate_user_inputs(storm_type, depth_mm, tpeak)

    # 2. Compute each dimension
    pattern_type = float(STORM_TYPE_INDEX[storm_type])

    depth_normalized = float(np.clip(
        (depth_mm - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN + 1e-8),
        0.0, 1.0,
    ))

    tpeak_val = float(tpeak) if (storm_type == 'triangular' and tpeak is not None) else 0.0

    has_tpeak = 1.0 if storm_type == 'triangular' else 0.0

    # 3. Assemble
    conditioning = np.array(
        [pattern_type, depth_normalized, tpeak_val, has_tpeak],
        dtype=np.float32,
    )

    assert conditioning.shape == (4,), f"Expected (4,), got {conditioning.shape}"
    return conditioning, warn_list


# ── Convenience: build both hyetograph + conditioning in one call ──────────────
def build_inference_inputs(
    storm_type: str,
    depth_mm: float,
    tpeak: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:

    conditioning, warn_list = build_conditioning_vector(storm_type, depth_mm, tpeak)
    raw_sequence  = generate_hyetograph(storm_type, depth_mm, tpeak)
    rainfall_norm = normalize_hyetograph(raw_sequence)
    return rainfall_norm, conditioning, warn_list


# ── Verification Helper ────────────────────────────────────────────────────────
def verify_against_csv(
    storm_type: str,
    depth_mm: float,
    csv_sequence: np.ndarray,
    tpeak: Optional[float] = None,
    tolerance: float = 0.5,
) -> None:

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