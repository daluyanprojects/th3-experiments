from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class RFConfig:
    patch_size            : int   = 4
    spatial_channels      : int   = 3      
    rainfall_timesteps    : int   = 13
    conditioning_dim      : int   = 4
    n_features            : int   = 35

    compute_dem_slope     : bool  = True
    entropy_bins          : int   = 16

    include_raw_timesteps : bool  = True
    include_rain_derived  : bool  = True

    # -- Data ------------------------------------------------------------------
    num_classes           : int   = 5
    map_h                 : int   = 1152
    map_w                 : int   = 1152
    ignore_index          : int   = -1

    # -- Random Forest hyperparameters -----------------------------------------
    n_estimators          : int   = 500      # number of trees
    max_depth             : int   = 20       # None = grow until pure; 20 is a sensible cap
    min_samples_split     : int   = 10       # min samples required to split a node
    min_samples_leaf      : int   = 5        # min samples required in a leaf
    max_features          : str   = 'sqrt'   # features per split: 'sqrt', 'log2', int, float
    max_samples           : float = 0.8      # bootstrap sample fraction (bagging)
    bootstrap             : bool  = True     # use bootstrap sampling
    oob_score             : bool  = True     # out-of-bag score (free val estimate)

    # -- Regularisation --------------------------------------------------------
    min_impurity_decrease : float = 0.0      # split only if impurity drops >= this
    ccp_alpha             : float = 0.0      # minimal cost-complexity pruning (0=off)

    # -- Runtime ---------------------------------------------------------------
    n_jobs                : int   = -1       # use all CPU cores
    random_state          : int   = 42
    verbose               : int   = 1

    # -- Class weighting -------------------------------------------------------
    use_class_weights     : bool  = True
    weight_power          : float = 0.5      # 0=none, 1=full inverse-freq

    # -- Cross-validation ------------------------------------------------------
    num_folds             : int   = 2
    fold_on_scenarios     : bool  = False    # patch-level split -> val F1 optimistic

    # -- Evaluation ------------------------------------------------------------
    checkpoint_metric     : str   = 'macro_f1'

    # -- Output ----------------------------------------------------------------
    output_dir            : Path  = Path('./outputs_rf')

    def save(self, path: Path):
        d = asdict(self)
        d['output_dir'] = str(d['output_dir'])
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'RFConfig':
        with open(path) as f:
            d = json.load(f)
        d['output_dir'] = Path(d['output_dir'])
        return cls(**d)

    def summary(self):
        print("\nRFConfig")
        print("=" * 56)
        print(f"  Features       : {self.n_features} total per patch")
        print(f"    DEM          : mean, std, min, max, slope_mean, slope_std, entropy  (7)")
        print(f"    Infiltration : mean, std, min, max, entropy                         (5)")
        print(f"    Land use     : mode, n_unique, entropy                              (3)")
        print(f"    Rainfall     : 13 timesteps + total / peak / tpeak_idx             (16)")
        print(f"    Conditioning : pattern_type, depth_norm, tpeak, has_tpeak           (4)")
        print(f"  Trees          : {self.n_estimators}")
        print(f"  Max depth      : {self.max_depth}")
        print(f"  Max features   : {self.max_features}  (per split)")
        print(f"  Bootstrap      : {self.bootstrap}  sample_frac={self.max_samples}")
        print(f"  OOB score      : {self.oob_score}")
        print(f"  Min split      : {self.min_samples_split}  min_leaf={self.min_samples_leaf}")
        print(f"  CCP alpha      : {self.ccp_alpha}")
        print(f"  Class weights  : {'yes' if self.use_class_weights else 'no'}  power={self.weight_power}")
        print(f"  CV strategy    : {self.num_folds}-fold on patch indices  (val F1 optimistic)")
        print(f"  n_jobs         : {self.n_jobs}")
        print(f"  Output dir     : {self.output_dir}")
        print("=" * 56)
