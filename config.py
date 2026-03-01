from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class SVMConfig:
    patch_size            : int   = 4
    spatial_channels      : int   = 3        # DEM, infiltration, land use
    rainfall_timesteps    : int   = 13
    conditioning_dim      : int   = 4
    n_features            : int   = 35

    compute_dem_slope     : bool  = True
    entropy_bins          : int   = 16

    include_raw_timesteps : bool  = True     # 13 raw timestep values
    include_rain_derived  : bool  = True     # total_depth, peak, tpeak_idx

    # ── Data ──────────────────────────────────────────────────────────────────
    num_classes           : int   = 5
    map_h                 : int   = 1152
    map_w                 : int   = 1152
    ignore_index          : int   = -1       # outside-mask pixels excluded from metrics

    # ── SVM hyperparameters ───────────────────────────────────────────────────
    # Kernel: 'rbf' | 'linear' | 'poly' | 'sigmoid'
    kernel                : str   = 'rbf'
    C                     : float = 10.0     # regularisation strength (inverse)
    gamma                 : str   = 'scale'  # 'scale' | 'auto' | float
    degree                : int   = 3        # only for 'poly' kernel
    coef0                 : float = 0.0      # 'poly' and 'sigmoid' free term
    tol                   : float = 1e-3
    max_iter              : int   = -1       # -1 = no limit

    # Decision function shape for multiclass
    # 'ovr' = one-vs-rest  |  'ovo' = one-vs-one (default for SVC)
    decision_function_shape : str = 'ovr'

    # Probability calibration — required for predict_proba()
    # Adds Platt scaling via 5-fold internal CV; slightly slower to train
    probability           : bool  = True

    # ── Feature scaling ───────────────────────────────────────────────────────
    # SVM is sensitive to scale — always StandardScaler before fitting
    scale_features        : bool  = True

    # ── Class weighting ───────────────────────────────────────────────────────
    # 'balanced' = sklearn auto-weighting  |  None = uniform
    class_weight          : str   = 'balanced'

    # ── Subsampling for tractability ──────────────────────────────────────────
    # SVM is O(n²–n³) — use a random subsample of training patches if n > max_train_samples
    max_train_samples     : int   = 150_000   # set None to disable
    subsample_seed        : int   = 42

    # ── Cross-validation ──────────────────────────────────────────────────────
    num_folds             : int   = 2         # mirrors ViT/XGBoost benchmark

    # ── Evaluation ────────────────────────────────────────────────────────────
    checkpoint_metric     : str   = 'macro_f1'

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir            : Path  = Path('./outputs_svm')

    random_state          : int   = 42

    def save(self, path: Path):
        d = asdict(self)
        d['output_dir'] = str(d['output_dir'])
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'SVMConfig':
        with open(path) as f:
            d = json.load(f)
        d['output_dir'] = Path(d['output_dir'])
        return cls(**d)

    def summary(self):
        print("\nSVMConfig")
        print("=" * 60)
        print(f"  Features       : {self.n_features} total per patch")
        print(f"    DEM          : mean, std, min, max, slope_mean, slope_std, entropy  (7)")
        print(f"    Infiltration : mean, std, min, max, entropy                         (5)")
        print(f"    Land use     : mode, n_unique, entropy                              (3)")
        print(f"    Rainfall     : 13 timesteps + total / peak / tpeak_idx             (16)")
        print(f"    Conditioning : pattern_type, depth_norm, tpeak, has_tpeak           (4)")
        print(f"  Kernel         : {self.kernel}  |  C={self.C}  |  gamma={self.gamma}")
        if self.kernel == 'poly':
            print(f"  Poly degree    : {self.degree}  coef0={self.coef0}")
        print(f"  Multiclass     : {self.decision_function_shape} (one-vs-rest)")
        print(f"  Probability    : {'✓ (Platt scaling)' if self.probability else '✗'}")
        print(f"  Feature scale  : {'✓ StandardScaler' if self.scale_features else '✗'}")
        print(f"  Class weight   : {self.class_weight}")
        print(f"  Max train n    : {self.max_train_samples:,}" if self.max_train_samples else "  Max train n    : unlimited")
        print(f"  CV strategy    : {self.num_folds}-fold on patch indices")
        print(f"  Output dir     : {self.output_dir}")
        print("=" * 60)
