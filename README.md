# XGBoost — Metro Manila Flood Prediction

## 1. Project Overview

The XGBoost branch predicts **flood inundation class** for every 4×4-pixel patch across Metro Manila's core, given a rainfall scenario. It is a classical ML baseline designed to benchmark against the ViT (patch transformer) and CNN (full-map U-Net) branches in the same suite.

Rather than feeding raw patch tensors into a neural network, this branch hand-engineers a **35-dimensional feature vector** per patch — capturing DEM statistics, infiltration capacity, land-use composition, the full hyetograph, and the storm conditioning descriptor — then trains an `XGBClassifier` on the resulting tabular matrix.

**Key properties:**

- Operates on the exact same patch-level arrays as the ViT branch — no spatial reconstruction required
- Outputs one flood class per patch, identical in structure to ViT predictions
- Provides native gain-based feature importance — interpretable insight into which inputs drive predictions
- No GPU required; runs entirely on CPU (`tree_method='hist'`, GPU optional via `device='cuda'`)
- 2-fold CV on patch indices mirrors the ViT benchmarking condition exactly

---

## 2. Repository Structure

```
root/
│
├── COP-30m-GMM-ManilaMaskOut/by_box/         ← Train rasters (GMM region)
│   ├── greater_mm_bbox_dem_cop_noManila_box.tif
│   ├── GM_Infilt_noManila_box.tif
│   ├── GM_LU_noManila_box.tif
│   └── RS01..RS50_flood.tif                  (50 flood simulation maps)
│
├── COP-30m-ManilaOnly/
│   ├── manila_box_3857.tif                   ← Manila boundary mask
│   └── exp-des-4/                            ← Test rasters (Manila core)
│       ├── manila_bbox_dem_cop.tif
│       ├── GM_Infilt_Manila_box.tif
│       ├── GM_LU_Manila_box.tif
│       └── RS01..RS50_flood.tif
│
├── mm_hr_scenarios/                          ← Rainfall scenario files + info.csv
│
├── SHARED MODULES
│   ├── data_loader.py           # load_dem, load_infiltration_map, load_landuse_map,
│   │                            # load_manila_mask, load_all_flood_maps,
│   │                            # load_all_rainfall_scenarios
│   ├── data_preprocessing.py    # preprocess_spatial_data → (1152,1152) normalised rasters
│   ├── rainfall.py              # preprocess_rainfall_sequences → sequences + conditioning
│   ├── flood_maps.py            # resize_flood_maps, categorize_flood_maps
│   └── data_splitting.py        # split_spatial_data → split_results
│   ├── config.py                # XGBConfig — all hyperparameters in one dataclass
│   ├── feature_engineering.py   # compute_slope_map, build_feature_matrix,
│   │                            # get_feature_names, validate_feature_matrix
│   ├── training.py              # train_kfold_xgb, evaluate_xgb,
│   │                            # plot_feature_importance, plot_fold_histories_xgb
│   ├── prediction.py            # reconstruct_ground_truth_maps_xgb,
│   │                            # generate_prediction_maps_xgb,
│   │                            # build_eval_results, plot_flood_on_dem,
│   │                            # plot_best_worst_dem
│   └── xgboost.ipynb            # End-to-end notebook — Cells 0–33
│
└── outputs_xgb/                 ← All artefacts written here (auto-created)
    ├── checkpoints/
    ├── logs/
    ├── predictions/
    └── test_patches.npz
```

---

## 3. Flood Classification Schema

Five flood depth classes are used throughout the pipeline. Labels are assigned by `categorize_flood_maps` using `majority_vote` over each 4×4 patch.

| Class | Label | Depth Range | Notes |
|---|---|---|---|
| 0 | No Flood | < 0.15 m | Dominant class (~73% of patches) |
| 1 | Light | 0.15 – 0.24 m | |
| 2 | Moderate | 0.24 – 0.46 m | |
| 3 | Heavy | 0.46 – 0.68 m | |
| 4 | Extreme | > 0.68 m | |
| –1 | Outside mask | — | Excluded from all metrics and loss |

**Visualization palette:**

| Class | Color | Hex |
|---|---|---|
| No Flood | White | `#FFFFFF` |
| Light | Yellow | `#FFEB3B` |
| Moderate | Orange | `#FF9800` |
| Heavy | Red | `#F44336` |
| Extreme | Purple | `#9C27B0` |

---

## 4. Data Splits at a Glance

The dataset is divided along two orthogonal axes simultaneously.

**Spatial axis** — hard geographic boundary, zero overlap:

| Split | Region | Inclusion rule | Valid patches / scenario |
|---|---|---|---|
| Train | GMM outskirts (`mask = 0`) | All 16 pixels outside Manila | 76,553 |
| Test | Manila core (`mask = 1`) | All 16 pixels inside Manila | 6,075 |

**Scenario axis** — random 70/30 split (`random_seed=42`):

| Split | Scenarios | Total patches |
|---|---|---|
| Train | 35 | 2,679,355 |
| Test | 15 | 91,125 |

**Array layout — scenarios-outer, patches-inner.** Rows `0..6,074` belong to scenario 0; rows `6,075..12,149` to scenario 1; and so on. This ordering is preserved across every patch-level array produced by `split_results`.

---

## 5. Data Loading


Three raster layers are loaded per region at native resolution `(1224, 1125)`:

| Layer | Train file | Test file |
|---|---|---|
| DEM | `greater_mm_bbox_dem_cop_noManila_box.tif` | `manila_bbox_dem_cop.tif` |
| Infiltration | `GM_Infilt_noManila_box.tif` | `GM_Infilt_Manila_box.tif` |
| Land use | `GM_LU_noManila_box.tif` | `GM_LU_Manila_box.tif` |

50 flood simulation maps are loaded per region (`NUM_SCENARIO = 50`). The Manila boundary mask (`manila_box_3857.tif`) is loaded separately and resized to `(1224, 1125)` using nearest-neighbour interpolation (`scipy.ndimage.zoom`, `order=0`) to keep binary pixel values intact.

---

## 6. Spatial Preprocessing


All six rasters are resized and normalised to `(1152, 1152)`. The target shape `1152 = 288 × 4` is exactly divisible by the 4-pixel patch stride — no partial boundary patches, no remainder rows or columns.

`nodata_method='interpolate'` fills NoData holes before normalisation. `norm_method='minmax'` scales each layer independently to `[0, 1]` using its own observed min/max.



## 7. Rainfall Preprocessing

50 rainfall scenarios are encoded. Each has 13 timesteps (12 × 5-minute intensity readings plus one trailing zero).

`norm_method='global_max'` divides all values by the peak intensity across all scenarios and all timesteps:

```
RAIN_MAX = 189.853086 mm/hr    ← rainfall_results['normalization_params']['global_max']
```

Each scenario also produces a **conditioning vector** `(4,)` that gets broadcast to every patch belonging to that scenario during Phase 5:

| Dim | Name | Description | Range |
|---|---|---|---|
| 0 | `pattern_type` | Storm shape class | `{0, 1, 2, 3}` |
| 1 | `depth_normalized` | `depth_mm / 78.0` | `[0, 1]` |
| 2 | `tpeak` | Normalised time of peak intensity | `[0, 1]` or `0` if N/A |
| 3 | `has_tpeak` | 1 if triangular storm, else 0 | `{0, 1}` |

Storm pattern types:

| Index | Name | Shape |
|---|---|---|
| 0 | Front-loaded (SCS Type II) | Beta(2.0, 5.0) — early peak |
| 1 | Balanced (SCS intermediate) | Beta(2.5, 2.5) — symmetric |
| 2 | Back-loaded (SCS Type I) | Beta(5.0, 2.0) — late peak |
| 3 | Triangular / Chicago-style | Linear ramp with user-defined `tpeak` |

```
DEPTH_MAX = 78.0    ← rainfall_results['encoding_info']['max_depth_mm']
```

---

## 8. Ground Truth Categorization

Each flood simulation map is resized to `(1152, 1152)` then thresholded into 5 depth classes. `majority_vote` assigns the modal class among the 16 pixels inside each 4×4 window as the ground truth label for that patch. Patches outside the corresponding mask receive label `–1` and are excluded from all loss and metric computations.

---

## 9. Data Splitting

The XGBoost branch uses the patch-level arrays **directly** from `split_results` — no reconstruction step is needed:

```python
# Training — 35 scenarios × 76,553 patches
X_train_spatial      = split_results['train']['spatial_patches']       # (2,679,355, 3, 4, 4)
X_train_rainfall     = split_results['train']['rainfall_sequences']    # (2,679,355, 13)
X_train_conditioning = split_results['train']['conditioning_vectors']  # (2,679,355, 4)
y_train              = split_results['train']['labels']                 # (2,679,355,)
train_scenario_meta  = split_results['train']['scenario_metadata']     # list[dict], 2,679,355 entries
train_scenario_ids   = split_results['train']['scenario_ids']          # list[int], 35 entries

# Test — 15 scenarios × 6,075 patches
X_test_spatial       = split_results['test']['spatial_patches']        # (91,125, 3, 4, 4)
X_test_rainfall      = split_results['test']['rainfall_sequences']     # (91,125, 13)
X_test_conditioning  = split_results['test']['conditioning_vectors']   # (91,125, 4)
y_test               = split_results['test']['labels']                  # (91,125,)
test_scenario_meta   = split_results['test']['scenario_metadata']      # list[dict], 91,125 entries
test_scenario_ids    = split_results['test']['scenario_ids']           # list[int], 15 entries
```

Each dict in `scenario_metadata` has the form `{'scenario_id': int, 'patch_coord': (i, j)}` where `(i, j)` is the patch grid coordinate. Multiply by `PATCH_SIZE` to get pixel coordinates.

---

## 10. Patch Index 

**Cell 20**

The Manila patch pixel coordinates are re-derived from the mask and saved alongside the unique spatial patches for later use in prediction reconstruction.

**Why `mean() == 1.0`:** All 16 pixels in the 4×4 window must fall inside Manila. This strict rule yields exactly 6,075 patches and matches the criterion used internally by `split_spatial_data`. Using `> 0.5` instead yields 6,306 patches — 231 boundary straddlers are incorrectly included.

**Why `X_test_spatial[:6075]`:** The array is scenarios-outer, patches-inner, so the first 6,075 rows are the patches for scenario 0. DEM, infiltration, and land use are static (scene-independent), so one scenario's slice is sufficient for reconstruction.

---

## 11. Feature Engineering

This phase converts raw patch tensors and rainfall arrays into a flat `(N, 35)` tabular matrix that XGBoost can consume.

### 11.1 Slope Map — computed once on the full DEM


The DEM slope is computed on the **full normalised 1152×1152 map** using `np.gradient` before any patchification:

```python
dy, dx    = np.gradient(dem_map.astype(np.float64))
slope_map = np.sqrt(dx**2 + dy**2).astype(np.float32)
```

This must happen before patch extraction. Computing gradient on isolated 4×4 patches gives incorrect boundary values because the neighbouring pixel context is unavailable at the patch edges. The resulting `slope_map` is passed into `build_feature_matrix` which extracts per-patch statistics from it.

### 11.2 Feature Matrix Construction

```python
from feature_engineering import build_feature_matrix, get_feature_names

feature_names = get_feature_names()   # list of 35 strings

X_train_feat, _ = build_feature_matrix(
    spatial      = X_train_spatial,       # (2,679,355, 3, 4, 4)
    rainfall     = X_train_rainfall,      # (2,679,355, 13)
    conditioning = X_train_conditioning,  # (2,679,355, 4)
    slope_map    = slope_map,             # (1152, 1152)
    patch_coords = None,
    entropy_bins = 16,
    chunk_size   = 250_000,
    verbose      = True,
)
# X_train_feat : (2,679,355, 35) float32

X_test_feat, _ = build_feature_matrix(
    spatial      = X_test_spatial,
    rainfall     = X_test_rainfall,
    conditioning = X_test_conditioning,
    slope_map    = slope_map,
    patch_coords = None,
    entropy_bins = 16,
    chunk_size   = 100_000,
    verbose      = True,
)
# X_test_feat : (91,125, 35) float32
```

Processing runs in `chunk_size` batches to cap RAM. The full train matrix occupies ~375 MB at float32 (2,679,355 × 35 × 4 bytes).

### 11.3 The 35-Feature Vector

Every row encodes one patch:

**DEM — 7 features** (channel 0 of the `(3, 4, 4)` spatial patch):

| # | Name | Description |
|---|---|---|
| 0 | `dem_mean` | Mean elevation across 16 pixels |
| 1 | `dem_std` | Standard deviation |
| 2 | `dem_min` | Minimum elevation |
| 3 | `dem_max` | Maximum elevation |
| 4 | `dem_slope_mean` | Mean gradient magnitude from full-map slope |
| 5 | `dem_slope_std` | Std of gradient magnitude |
| 6 | `dem_entropy` | Shannon entropy of elevation values (16 bins) |

**Infiltration — 5 features** (channel 1):

| # | Name | Description |
|---|---|---|
| 7 | `infilt_mean` | Mean infiltration capacity |
| 8 | `infilt_std` | Standard deviation |
| 9 | `infilt_min` | Minimum |
| 10 | `infilt_max` | Maximum |
| 11 | `infilt_entropy` | Shannon entropy (16 bins) |

**Land use — 3 features** (channel 2, categorical):

| # | Name | Description |
|---|---|---|
| 12 | `landuse_mode` | Modal land use class among 16 pixels |
| 13 | `landuse_nunique` | Count of distinct land use classes in patch |
| 14 | `landuse_entropy` | Shannon entropy over 11 land use classes |

**Rainfall — 16 features** (same for all patches within a scenario):

| # | Name | Description |
|---|---|---|
| 15–27 | `rain_t00` … `rain_t12` | Raw normalised hyetograph values |
| 28 | `rain_total` | Sum across all 13 timesteps |
| 29 | `rain_peak` | Maximum intensity across 13 timesteps |
| 30 | `rain_tpeak_idx` | `argmax / 12` — normalised time of peak, `[0, 1]` |

**Conditioning — 4 features** (same for all patches within a scenario):

| # | Name | Description |
|---|---|---|
| 31 | `cond_pattern_type` | Storm shape class `{0,1,2,3}` |
| 32 | `cond_depth_norm` | `depth_mm / 78.0`, range `[0, 1]` |
| 33 | `cond_tpeak` | Fractional peak time `[0, 1]` or `0` if not triangular |
| 34 | `cond_has_tpeak` | `1` = triangular storm, `0` = otherwise |

### 11.4 Entropy Computation

Shannon entropy is computed per patch using histogram binning:

```
H = −Σ pᵢ · log₂(pᵢ)   where pᵢ = normalised bin count
```

For continuous channels (DEM, infiltration) `entropy_bins=16` bins are used. For land use (categorical) `n_bins=11` is used. Two safety guards prevent crashes on near-constant patches, which are common for infiltration in dense urban areas where all 16 pixels carry nearly identical float32 values:

1. If `vmax − vmin < 1e-7`, return `0.0` immediately (zero information content).
2. `safe_bins = min(n_bins, len(np.unique(values)))` — bin count is clamped to never exceed the number of distinct values, so `np.histogram` never receives more bins than it can create.

### 11.5 Validation

```python
from feature_engineering import validate_feature_matrix

validate_feature_matrix(X_train_feat, y_train, feature_names, split='train')
validate_feature_matrix(X_test_feat,  y_test,  feature_names, split='test')
```

Checks shape `(N, 35)`, zero NaN, zero Inf, all labels in `[0, 4]`, slope features `≥ 0`, and prints class distribution.

---

## 12. Training

### 12.1 Configuration

```python
from config import XGBConfig

cfg_xgb = XGBConfig()
cfg_xgb.summary()
```

All hyperparameters are in `XGBConfig`:

| Parameter | Value | Notes |
|---|---|---|
| `n_estimators` | 500 | Maximum boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `min_child_weight` | 5 | Minimum sample-weight sum in a leaf |
| `learning_rate` | 0.1 | Shrinkage (eta) |
| `subsample` | 0.8 | Row fraction sampled per tree |
| `colsample_bytree` | 0.8 | Feature fraction sampled per tree |
| `colsample_bylevel` | 1.0 | Feature fraction per tree level |
| `gamma` | 0.1 | Minimum loss reduction required to split |
| `reg_alpha` | 0.0 | L1 regularisation |
| `reg_lambda` | 1.0 | L2 regularisation |
| `objective` | `multi:softmax` | 5-class classification |
| `eval_metric` | `mlogloss` | Early-stopping monitor |
| `early_stopping_rounds` | 30 | Stop if no improvement for 30 rounds |
| `tree_method` | `hist` | Fast histogram algorithm (CPU + GPU) |
| `device` | `cpu` | Set `'cuda'` to use GPU |
| `num_folds` | 2 | K-fold cross-validation |
| `weight_power` | 0.5 | Class-weight smoothing exponent |
| `output_dir` | `./outputs_xgb` | All checkpoints, logs, and plots |

GPU: uncomment `cfg_xgb.device = 'cuda'` before training. `tree_method='hist'` supports CPU and GPU without further changes (XGBoost ≥ 2.0).

### 12.2 Class Weighting

XGBoost's `fit()` accepts `sample_weight=(N,)` — there is no `class_weight` dict. Weights are computed per class then expanded to a per-sample array:

```
w_c = (freq_c)^(−weight_power)    where freq_c = count_c / N
```

Results are normalised so mean weight across all training samples equals 1.0. With `weight_power=0.5` this applies square-root smoothing — less aggressive than full inverse-frequency but still enough to up-weight the minority flood classes.

### 12.3 Cross-Validation

`KFold(n_splits=2, shuffle=True, random_state=42)` splits the 2,679,355 patch indices randomly. Each fold trains on ~1.34M patches and validates on ~1.34M patches.

> **Validation F1 is optimistically inflated.** The split is random across patches — patches from the same scenario appear in both train and val halves. A model that memorises one scenario's spatial patterns will still score well on its val patches. The held-out test set (Manila core, 15 scenarios never seen during training) is the only reliable benchmarking metric.

Per fold the training loop:
1. Computes `sample_weight` from the fold's training label distribution
2. Fits `XGBClassifier` with `eval_set=[(X_val, y_val)]` and `mlogloss` early stopping
3. Evaluates val fold: accuracy, macro F1, weighted F1, per-class P/R/F1, critical recall
4. Logs top-5 feature importances by gain
5. Saves the model to `outputs_xgb/checkpoints/fold_N_best.json` (native XGBoost JSON — portable across versions and platforms)


### 12.5 Fold Summary Plot

Two-panel chart: val Macro F1 per fold with mean line, and number of boosting rounds at early stopping per fold. Unlike the ViT/CNN branches there are no epoch-level training curves — XGBoost produces one summary metric per fold.

---

## 13. Phase 9 — Evaluation & Prediction Maps

### 13.1 Test Set Evaluation

Metrics (identical suite to ViT and CNN for fair comparison): accuracy, Macro F1, weighted F1, critical recall (`recall(Heavy ∪ Extreme)`), per-class precision/recall/F1, confusion matrix.

### 13.2 Feature Importance

`importance_type='gain'` is the average reduction in multi-class log-loss per split on that feature — the most meaningful importance metric. `'weight'` (split count) and `'cover'` (average sample coverage) are also available by changing the argument.

### 13.3 Ground Truth Reconstruction

Reassembles the flat `y_test` into per-scenario spatial maps by placing each patch at its `patch_coord[i] × PATCH_SIZE, patch_coord[j] × PATCH_SIZE` pixel position.

### 13.4 Prediction Map Generation

For each scenario, patches are placed at their `(i × 4, j × 4)` pixel coordinates. Confidence is `max(softmax)` per patch. The function auto-saves per-scenario prediction maps, confidence heatmaps, the 15-scenario grid, and the confidence histogram to `outputs_xgb/predictions/`.

### 13.5 Spatial Evaluation and DEM Overlays

`build_eval_results` computes per-scenario accuracy, Macro F1, IoU, macro precision, and macro recall across all 15 test scenarios. `plot_best_worst_dem` produces a 2×2 DEM-overlay figure: best scenario GT, best scenario predicted, worst scenario GT, worst scenario predicted.

`eval_results_xgb` structure:

```python
{
    'gt_maps'      : list[(H,W) int8],
    'pred_maps'    : list[(H,W) int8],
    'per_scenario' : [
        {'scenario_id': int, 'accuracy': float, 'f1_macro': float,
         'iou_macro': float, 'precision_macro': float, 'recall_macro': float},
        ...   # 15 entries, one per test scenario
    ],
    'scenario_ids' : list[int],
    'f1_scores'    : list[float],
    'aggregate'    : {
        'best_scenario' : int,
        'worst_scenario': int,
        'best_idx'      : int,
        'worst_idx'     : int,
    },
}
```

---

## 14. Output Directory

After all phases complete, `outputs_xgb/` contains:

```
outputs_xgb/
│
├── checkpoints/
│   ├── fold_1_best.json            # XGBoost native JSON — portable across versions
│   └── fold_2_best.json
│
├── logs/
│   ├── fold_results.json           # per-fold metrics, best_ntree, feature importances
│   ├── test_results.json           # accuracy, macro_f1, per-class P/R/F1, critical_recall
│   ├── fold_summary.png            # val F1 + early-stopping rounds bar chart
│   ├── feature_importance_gain.png # top-20 features ranked by gain
│   └── test_confusion_matrix.png
│
├── predictions/
│   ├── prediction_grid.png                  # 3×5 grid of all 15 test scenario maps
│   ├── eval_dem_best_worst_grid.png         # 2×2 DEM overlay (best/worst × GT/pred)
│   ├── prediction_scenario_{sid}.png        # individual flood map per scenario
│   └── confidence_scenario_{sid}.png        # confidence + low-confidence region map
│
└── test_patches.npz
    ├── spatial        (6,075, 3, 4, 4)      # scene-independent Manila patches
    ├── patch_indices  (6,075, 2)            # pixel coordinates of each patch
    └── dem            (1152, 1152)          # Manila DEM for DEM-overlay plots
```

Model checkpoints are **XGBoost JSON** (`.json`), not PyTorch `.pth`. There is no torch dependency for loading them.

---

## 15. Known Limitations & Design Notes

**Scenario leakage in validation.** The 2-fold patch split is random — patches from the same storm scenario land in both train and val halves. Validation Macro F1 is inflated relative to true generalisation. Only test set results (15 unseen Manila scenarios, separate geography) are valid for benchmarking.

**No spatial context.** Like the ViT branch, XGBoost classifies each 4×4 patch in complete isolation. It cannot learn flood routing — water accumulating in low-lying areas, spreading along drainage corridors, or propagating from upstream to downstream across patch boundaries. The CNN U-Net captures all of this through its full spatial receptive field.

**Patch-level, not scenario-level, arrays.** `X_train_conditioning` is `(2,679,355, 4)` — each scenario's conditioning vector has been tiled across all 76,553 of its patches. Do not call `reconstruct_for_cnn()` before feature engineering; this would collapse the array to `(35, 4)` and break `build_feature_matrix`.

**Memory.** The full train feature matrix is ~375 MB. The default `chunk_size=250_000` caps per-chunk allocation to ~27 MB. Reduce to `100_000` on machines with less than 8 GB free RAM.

**Class imbalance.** No Flood accounts for ~73% of training patches. Even with `weight_power=0.5`, the minority classes Light (~2.6%) and Heavy (~2.8%) are severely under-represented. Macro F1 will be depressed by these classes regardless of boosting rounds.

---