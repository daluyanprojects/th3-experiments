from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class XGBConfig:
    # ── Feature engineering ───────────────────────────────────────────────────
    patch_size            : int   = 4
    spatial_channels      : int   = 3        # DEM, infiltration, land use
    rainfall_timesteps    : int   = 13
    conditioning_dim      : int   = 4
    n_features            : int   = 35

    compute_dem_slope     : bool  = True     # slope from full DEM before patchification
    entropy_bins          : int   = 16       # histogram bins for Shannon entropy

    include_raw_timesteps : bool  = True     # 13 raw timestep values
    include_rain_derived  : bool  = True     # total_depth, peak, tpeak_idx

    # ── Data ──────────────────────────────────────────────────────────────────
    num_classes           : int   = 5
    map_h                 : int   = 1152
    map_w                 : int   = 1152
    ignore_index          : int   = -1       # outside-mask pixels excluded from metrics

    # ── XGBoost hyperparameters ───────────────────────────────────────────────
    n_estimators          : int   = 500
    max_depth             : int   = 6
    min_child_weight      : int   = 5
    subsample             : float = 0.8
    colsample_bytree      : float = 0.8
    colsample_bylevel     : float = 1.0
    gamma                 : float = 0.1      # min loss reduction to make a split
    reg_alpha             : float = 0.0      # L1
    reg_lambda            : float = 1.0      # L2

    learning_rate         : float = 0.1
    early_stopping_rounds : int   = 30

    objective             : str   = 'multi:softmax'
    eval_metric           : str   = 'mlogloss'

    tree_method           : str   = 'hist'   # fast CPU+GPU (XGBoost ≥ 2.0)
    device                : str   = 'cpu'    # set 'cuda' if GPU available
    n_jobs                : int   = -1
    random_state          : int   = 42

    # ── Class weighting ───────────────────────────────────────────────────────
    # XGBoost uses sample_weight=(N,) array — no class_weight dict
    # w_c = (N / (C × count_c)) ^ weight_power, normalised so mean = 1
    use_class_weights     : bool  = True
    weight_power          : float = 0.5

    # ── Cross-validation ──────────────────────────────────────────────────────
    # 2-fold on patch indices — mirrors ViT benchmarking condition
    num_folds             : int   = 2
    fold_on_scenarios     : bool  = False    

    # ── Evaluation ────────────────────────────────────────────────────────────
    checkpoint_metric     : str   = 'macro_f1'

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir            : Path  = Path('./outputs_xgb')

    def save(self, path: Path):
        d = asdict(self)
        d['output_dir'] = str(d['output_dir'])
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'XGBConfig':
        with open(path) as f:
            d = json.load(f)
        d['output_dir'] = Path(d['output_dir'])
        return cls(**d)

    def summary(self):
        print("\nXGBConfig")
        print("=" * 56)
        print(f"  Features       : {self.n_features} total per patch")
        print(f"    DEM          : mean, std, min, max, slope_mean, slope_std, entropy  (7)")
        print(f"    Infiltration : mean, std, min, max, entropy                         (5)")
        print(f"    Land use     : mode, n_unique, entropy                              (3)")
        print(f"    Rainfall     : 13 timesteps + total / peak / tpeak_idx             (16)")
        print(f"    Conditioning : pattern_type, depth_norm, tpeak, has_tpeak           (4)")
        print(f"  Estimators     : {self.n_estimators}  (early_stop={self.early_stopping_rounds})")
        print(f"  Max depth      : {self.max_depth}")
        print(f"  LR (eta)       : {self.learning_rate}")
        print(f"  Subsample      : rows={self.subsample}  cols={self.colsample_bytree}")
        print(f"  Regularisation : gamma={self.gamma}  L1={self.reg_alpha}  L2={self.reg_lambda}")
        print(f"  Class weights  : {'✓' if self.use_class_weights else '✗'}  power={self.weight_power}")
        print(f"  CV strategy    : {self.num_folds}-fold on patch indices  ⚠ val F1 optimistic")
        print(f"  Device         : {self.device}  tree_method={self.tree_method}")
        print(f"  Output dir     : {self.output_dir}")
        print("=" * 56)