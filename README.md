
---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Flood Classification Schema](#3-flood-classification-schema)
4. [Shared Pipeline — Phases 1–5](#4-shared-pipeline--phases-15)
   - Phase 1: Data Loading
   - Phase 2: Spatial Preprocessing
   - Phase 3: Rainfall Preprocessing
   - Phase 4: Flood Map Categorization
   - Phase 5: Spatial + Scenario Split
5. [Branch A — ViT Pipeline](#5-branch-a--vit-pipeline)
   - Phase 6: Patch Extraction
   - Phase 7: Dataset Assembly
   - Phase 8: Training
   - Inference Setup
   - Inference
6. [Branch B — CNN Pipeline](#6-branch-b--cnn-pipeline)
   - Phase 6: Map Reconstruction
   - Phase 7: Dataset Assembly
   - Phase 8: Training
7. [Shared Evaluation](#7-shared-evaluation)
8. [Model Architectures](#8-model-architectures)
9. [Benchmarking: ViT vs CNN](#9-benchmarking-vit-vs-cnn)
10. [Normalization Constants](#10-normalization-constants)
11. [Known Limitations](#11-known-limitations)
12. [Reproducing Results](#12-reproducing-results)

---

## 1. Project Overview

FloodCast predicts **flood inundation categories** across Metro Manila for any user-defined rainfall scenario. The pipeline trains on the Greater Metro Manila (GMM) region and evaluates on the Manila core — a spatially held-out area never seen during training.

Two model branches are implemented for benchmarking:

| Branch | Model | Input unit | Output unit |
|---|---|---|---|
| **A** | Patch-based ViT | 4×4 pixel patch | Class per patch |
| **B** | EfficientNet-B3 U-Net | Full 1152×1152 map | Class per pixel |

Both branches share identical data preprocessing (Phases 1–5) and evaluation logic. They diverge at Phase 6 where ViT requires patchification and CNN operates on full spatial maps.

**Three inputs per prediction:**
- **Spatial context** — DEM, infiltration capacity, land use
- **Rainfall sequence** — 13-step hyetograph (5-min intervals, 1 hour total)
- **Conditioning vector** — 4D descriptor: `[pattern_type, depth_norm, tpeak, has_tpeak]`

---

## 2. Repository Structure

```
FloodCast/
│
├── SHARED (Phases 1–5)
│   ├── data_preprocessing.py       # Raster normalization + resizing
│   ├── data_splitting.py           # Spatial + scenario train/test split
│   ├── flood_maps.py               # Depth thresholding → 5-class labels
│   ├── rainfall.py                 # Hyetograph normalization + conditioning
│   └── hyetograph.py               # Inference-time storm input builder
│
├── BRANCH A — ViT
│   ├── patchify.py                 # 4×4 patch extraction
│   ├── dataset.py                  # create_complete_dataset (patch-level)
│   ├── vit.py                      # ViTFloodClassifier architecture
│   ├── config.py                   # TrainConfig
│   ├── training.py                 # FloodPatchDataset, train_kfold, evaluate
│   ├── prediction.py               # reconstruct_ground_truth_maps, generate_prediction_maps
│   ├── save_test_patches_inline.py # One-time static patch save
│   ├── inference_engine.py         # InferenceEngine, predict_with_confidence
│   ├── visualize_inference.py      # Inference visualization utilities
│   └── outputs/
│       ├── checkpoints/            # fold_N_best.pth
│       ├── logs/                   # config.json, fold_curves.png, test_results.json
│       ├── predictions/            # Per-scenario flood maps
│       └── test_patches.npz        # Static Manila patches for inference
│
└── BRANCH B — CNN
    ├── reconstruct.py              # reconstruct_for_cnn → (S,3,H,W) maps
    ├── dataset_cnn.py              # FloodMapDataset (scenario-level)
    ├── cnn.py                      # CNNFloodModel (EfficientNet-B3 + FiLM)
    ├── config_cnn.py               # CNNTrainConfig
    ├── training_cnn.py             # train_kfold_cnn, evaluate_cnn
    ├── prediction_cnn.py           # reconstruct_ground_truth_maps_cnn, generate_prediction_maps_cnn
    └── outputs_cnn/
        ├── checkpoints/            # fold_N_best.pth
        ├── logs/                   # config.json, fold_curves.png, test_results.json
        └── predictions/            # Per-scenario flood maps
```

---

## 3. Flood Classification Schema

Both branches use the same 5-class flood depth schema:

| Class | Label | Depth Range | Training % |
|---|---|---|---|
| 0 | No Flood | < 0.15 m | 72.8% |
| 1 | Light | 0.15 – 0.24 m | 2.6% |
| 2 | Moderate | 0.24 – 0.46 m | 4.9% |
| 3 | Heavy | 0.46 – 0.68 m | 2.8% |
| 4 | Extreme | > 0.68 m | 16.9% |

**Color scheme for visualization:**

| Class | Color | Hex |
|---|---|---|
| No Flood | White | `#FFFFFF` |
| Light | Yellow | `#FFEB3B` |
| Moderate | Orange | `#FF9800` |
| Heavy | Red | `#F44336` |
| Extreme | Purple | `#9C27B0` |
| Outside mask | Transparent | — |

**Outside mask pixels** (`-1`) are excluded from all loss computations and metrics in both branches.

---

## 4. Shared Pipeline — Phases 1–5

Phases 1–5 are **identical** across both branches. Run them once; the outputs feed into both ViT and CNN training.

### Phase 1 — Data Loading

Two spatially disjoint regions are loaded:

| Split | Region | Area fraction | Scenarios |
|---|---|---|---|
| Train | GMM (excluding Manila box) | 92.5% | 35 |
| Test | Manila core (inside Manila box) | 7.5% | 15 |

**Raster inputs per region** (native resolution `1224×1125`):
- `DEM` — Copernicus Digital Elevation Model
- `Infiltration` — soil infiltration capacity
- `Land Use` — categorical land use classification

Plus **50 flood simulation maps** per region (one per rainfall scenario).

---

### Phase 2 — Spatial Preprocessing

All rasters normalized and resized to a common shape:

```python
preprocessed = preprocess_spatial_data(
    train_dem, train_infilt, train_landuse,
    test_dem,  test_infilt,  test_landuse,
    mask=manila_box_mask,
    target_shape=(1152, 1152),
    nodata_method='interpolate',
    norm_method='minmax'
)
```

**Target shape: `(1152, 1152)`** — chosen because `1152 = 288 × 4`, making it evenly divisible by the 4×4 patch size with zero remainder. This ensures complete patch coverage with no boundary artifacts in the ViT branch.

**Manila mask:** A boolean `(1152, 1152)` array stored in `split_results['masks']['manila_mask']` after Phase 5. Used to separate Manila core from GMM outskirts.

---

### Phase 3 — Rainfall Scenario Preprocessing

50 rainfall scenarios are encoded. Each has **13 timesteps** (12 × 5-minute intensity blocks + 1 trailing zero).

**Normalization:** `global_max` — all sequences divided by the global peak intensity.

```
RAIN_MAX = 189.853086 mm/hr   ← normalization_params['global_max']
RAIN_MIN = 0.0 mm/hr
```

**Conditioning vector** `(4,)` per scenario:

| Dim | Name | Description | Range |
|---|---|---|---|
| 0 | `pattern_type` | Storm class integer | `{0,1,2,3}` |
| 1 | `depth_normalized` | `depth_mm / 78.0` | `[0, 1]` |
| 2 | `tpeak` | Fractional peak time | `[0, 1]` or `0` |
| 3 | `has_tpeak` | Triangular storm flag | `{0, 1}` |

**Storm pattern types:**

| Index | Name | Shape | Peak |
|---|---|---|---|
| 0 | Front-loaded (SCS Type II) | Beta(2.0, 5.0) | t=2 |
| 1 | Balanced (SCS intermediate) | Beta(2.5, 2.5) | t=5 |
| 2 | Back-loaded (SCS Type I) | Beta(5.0, 2.0) | t=9 |
| 3 | Triangular / Chicago-style | Linear ramp | user-defined `tpeak` |

**Normalization constants verified from training data:**
```
DEPTH_MAX = 78.0    ← encoding_info['max_depth_mm']
DEPTH_MIN = 0.0
```

---

### Phase 4 — Ground Truth Flood Map Categorization

Flood simulation maps are resized to `(1152, 1152)` and thresholded into 5 classes.

**Depth thresholds:** `[0.15, 0.24, 0.46, 0.68]` meters

**ViT:** Majority vote over each 4×4 patch → single class label per patch.
**CNN:** Per-pixel classification → `(1152, 1152)` label map directly.

---

### Phase 5 — Spatial + Scenario Split

The dataset is split along **two independent axes simultaneously:**

**Spatial axis** (hard geographic split, no overlap):
```
Train: GMM region       → 76,553 valid patches  (92.5% of Manila area)
Test:  Manila core      →  6,075 valid patches   (7.5% of Manila area)
```

**Scenario axis** (random split, `seed=42`):
```
Train: 35 scenarios  (70%)
Test:  15 scenarios  (30%)
```

**Patch inclusion rule:** `manila_mask[r:r+4, c:c+4].mean() == 1.0`
All 16 pixels in the patch must be inside Manila. Patches straddling the boundary (75 border patches) are excluded. This is the exact rule Phase 5 used and must match exactly when deriving patch indices at inference time.

**This split is orthogonal** — the model is evaluated on a geography it has never seen, under rainfall scenarios it has never seen. This tests true generalization, not interpolation.

---

## 5. Branch A — ViT Pipeline

### Phase 6 — Patch Extraction

Each `(1152, 1152)` spatial map is divided into non-overlapping 4×4 patches:

```
Patches per dimension: 1152 / 4 = 288
Total possible patches: 288 × 288 = 82,944
Valid patches (inside mask): 76,553 (train) / 6,075 (test)
```

**Final array shapes after patchification:**
```
X_train_spatial      : (2,679,355, 3, 4, 4)   — 35 scenarios × 76,553 patches
X_train_rainfall     : (2,679,355, 13)
X_train_conditioning : (2,679,355, 4)
y_train              : (2,679,355,)

X_test_spatial       : (91,125, 3, 4, 4)       — 15 scenarios × 6,075 patches
X_test_rainfall      : (91,125, 13)
X_test_conditioning  : (91,125, 4)
y_test               : (91,125,)
```

**Dataset layout:** scenarios-outer, patches-inner. Rows `0..6074` = all patches for scenario 0, rows `6075..12149` = all patches for scenario 1, etc.

---

### Phase 7 — Dataset Assembly

```python
from dataset import create_complete_dataset
dataset = create_complete_dataset(vit_results)
```

Validates shapes, NaN values, label ranges, and conditioning vector format. Returns a dict with `train` and `test` keys.

---

### Phase 8 — ViT Training

```python
from config import TrainConfig
from training import FloodPatchDataset, train_kfold, plot_fold_histories

cfg = TrainConfig()
cfg.use_conditioning = True

full_dataset = FloodPatchDataset(
    X_train_spatial, y_train, X_train_rainfall, X_train_conditioning
)
fold_histories, best_fold_idx = train_kfold(full_dataset, y_train, cfg)
```

**Key hyperparameters:**
```
batch_size    : 1024
num_epochs    : 20
num_folds     : 2
lr            : 3e-4
warmup_epochs : 10
loss          : CombinedLoss(ce=0.9, dice=0.1)
weight_power  : 0.5
```

**Class weights (power=0.5 smoothing):**
```
No Flood  : 0.5243    Extreme  : 1.0867
Light     : 2.7562    Heavy    : 2.6842
Moderate  : 2.0192
```

**K-fold strategy:** `KFold(n_splits=2)` on 2,679,355 patch indices. Each fold trains on ~1.34M patches and validates on ~1.34M patches.

**Results:**
```
Fold 1 — val macro_f1 : 0.3701
Fold 2 — val macro_f1 : 0.3782  ← best

Test Accuracy          : 0.7917
Test Macro F1          : 0.3994
Test Weighted F1       : 0.7825
Critical Recall        : 0.5491  (Heavy + Extreme)

Class       Precision   Recall    F1
No Flood      0.8953    0.9374  0.9159
Light         0.1553    0.4104  0.2254
Moderate      0.1721    0.1032  0.1290
Heavy         0.1461    0.0044  0.0085   ← known weakness
Extreme       0.8116    0.6437  0.7180
```

---

### Inference Setup (ViT, one-time)

After training, save the static Manila patches so inference works without the training notebook in memory:

```python
# save_test_patches_inline.py
# Requires: split_results, X_test_spatial, test_dem_preprocessed

manila_mask = split_results['masks']['manila_mask']

patch_rows, patch_cols = [], []
for r in range(0, 1152 - 4 + 1, 4):
    for c in range(0, 1152 - 4 + 1, 4):
        if manila_mask[r:r+4, c:c+4].mean() == 1.0:   # strict inclusion
            patch_rows.append(r)
            patch_cols.append(c)

np.savez('outputs/test_patches.npz',
    spatial       = X_test_spatial[:6075],    # (6075, 3, 4, 4)
    patch_indices = np.stack([patch_rows, patch_cols], axis=1),  # (6075, 2)
    dem           = test_dem_preprocessed,    # (1152, 1152)
)
```

**Why `mean() == 1.0`?** Border patches where only some pixels fall inside Manila are excluded — matching the exact Phase 5 criterion. Using `> 0.5` gives 6306 patches (231 too many); using `== 1.0` gives exactly 6075.

**Why `X_test_spatial[:6075]`?** The patch array is scenarios-outer, patches-inner. The first 6075 rows are scenario 0's patches. Spatial features (DEM, infiltration, land use) are identical across all scenarios so one slice is sufficient.

---

### Inference (ViT)

```python
from inference_engine import InferenceEngine, predict_with_confidence

engine = InferenceEngine()   # loads model + patches once

result = predict_with_confidence(
    engine, storm_type='triangular', depth_mm=50, tpeak=0.4
)
```

**Inference data flow:**
```
User input: storm_type, depth_mm, tpeak
    ↓ build_inference_inputs (hyetograph.py)
rainfall_seq (13,)  +  conditioning (4,)
    ↓ tile across 6,075 Manila patches
rainfall_tiled (6075,13)  +  conditioning_tiled (6075,4)
    ↓ + unique_spatial (6075,3,4,4) from test_patches.npz
    ↓ model forward pass (fold_N_best.pth)
predictions (6075,)  +  probabilities (6075,5)
    ↓ reconstruct using patch_indices
flood_map (1152,1152)   ← final output
```

**Result dict keys:**

| Key | Shape | Description |
|---|---|---|
| `flood_map` | `(1152,1152) int8` | Spatial map; -1 = outside Manila |
| `predictions` | `(6075,) int` | Patch-level class labels |
| `probabilities` | `(6075,5) float` | Softmax probabilities |
| `confidence` | `(6075,) float` | `max(softmax)` per patch |
| `rainfall_sequence` | `(13,) float` | Normalized hyetograph |
| `conditioning` | `(4,) float` | Conditioning vector |
| `summary` | `dict` | Class counts, percentages, mean confidence |
| `warnings` | `list[str]` | OOD alerts |

**Visualization:**
```python
from visualize_inference import (
    plot_flood_map, plot_confidence,
    plot_comparison, plot_class_distribution, plot_hyetograph
)

plot_flood_map(result, engine)          # DEM + flood map
plot_confidence(result, engine)         # uncertainty heatmap
plot_comparison([result1, result2], engine)  # multi-storm comparison
```

---

## 6. Branch B — CNN Pipeline

### Phase 6 — Map Reconstruction

Unlike ViT which patchifies, CNN operates on **full spatial maps**. The patch arrays from Phase 5 are reassembled back into `(S, H, W)` label maps:

```python
from reconstruct import reconstruct_for_cnn, validate_reconstruction

cnn_train, cnn_test = reconstruct_for_cnn(
    split_results,
    patch_size=4, map_h=1152, map_w=1152,
    nodata_label=-1,
)
```

**Output shapes:**
```
X_train_spatial      : (35, 3, 1152, 1152)   — one map per scenario
y_train              : (35, 1152, 1152)        — label map, -1 = outside mask
X_train_rainfall     : (35, 13)
X_train_conditioning : (35, 4)

X_test_spatial       : (15, 3, 1152, 1152)
y_test               : (15, 1152, 1152)
X_test_rainfall      : (15, 13)
X_test_conditioning  : (15, 4)
```

**Critical difference from ViT:** The CNN dataset has 35 items (scenarios), not 2,679,355 items (patches). `__getitem__` returns one full `(3,1152,1152)` map.

---

### Phase 7 — Dataset Assembly

```python
from dataset_cnn import create_cnn_datasets

train_dataset, test_dataset = create_cnn_datasets(cnn_train, cnn_test)
```

`FloodMapDataset.__getitem__` returns a 4-tuple:
```python
(spatial, rainfall, conditioning, labels)
# (3,H,W)   (13,)    (4,)           (H,W)
```

---

### Phase 8 — CNN Training

```python
from config_cnn import CNNTrainConfig
from training_cnn import train_kfold_cnn, evaluate_cnn, plot_fold_histories

cfg = CNNTrainConfig()
cfg.summary()

fold_histories, best_fold_idx = train_kfold_cnn(train_dataset, y_train, cfg)
```

**Key hyperparameters:**
```
encoder          : efficientnet-b3 (ImageNet pretrained)
decoder_channels : (256, 128, 64, 32, 16)
batch_size       : 2  (full maps are memory-heavy)
grad_accum_steps : 8  (effective batch = 16)
num_epochs       : 50
num_folds        : 5
lr               : 1e-4
warmup_epochs    : 5
loss             : CombinedLoss(ce=0.9, dice=0.1, ignore_index=-1)
weight_power     : 0.5
```

**K-fold strategy:** `KFold(n_splits=5)` on **35 scenario indices** (not pixels). Each fold: 28 train / 7 val scenarios. This is a coarser split than ViT but necessary — with only 35 total items, 2-fold would give 17/18 which is too coarse to detect overfitting.

**Gradient accumulation:** `batch_size=2 × grad_accum=8 = effective_batch=16`. Required because each `(2, 3, 1152, 1152)` batch consumes ~2.5GB GPU memory at fp16.

**ignore_index=-1:** Outside-mask pixels (`-1` labels) are excluded from both CE loss and Dice loss. This is the most critical difference from ViT training where labels were always `[0,4]` and no masking was needed.

---

## 7. Shared Evaluation

Both branches use identical evaluation logic after the forward pass.

**Test evaluation:**
```python
# ViT
from training import evaluate
metrics = evaluate(best_model, test_loader, cfg, split_name='Test')

# CNN
from training_cnn import evaluate_cnn
metrics = evaluate_cnn(best_model, test_loader, cfg, split_name='Test')
```

**Metrics computed:**
- Pixel/patch accuracy
- Macro F1 (unweighted average across 5 classes)
- Weighted F1
- Critical Recall — `recall(Heavy + Extreme)` — most operationally important metric
- Per-class Precision / Recall / F1
- Confusion matrix

**Prediction maps and DEM overlays:**
```python
# ViT
from prediction import (
    reconstruct_ground_truth_maps, generate_prediction_maps,
    build_eval_results, plot_flood_on_dem, plot_best_worst_dem
)

# CNN
from prediction_cnn import (
    reconstruct_ground_truth_maps_cnn, generate_prediction_maps_cnn,
    build_eval_results, plot_flood_on_dem, plot_best_worst_dem
)
```

`build_eval_results`, `plot_flood_on_dem`, and `plot_best_worst_dem` are **identical** in both branches — they operate on `(H,W)` numpy arrays from `eval_results` regardless of how they were generated.

---

## 8. Model Architectures

### Branch A — ViT Flood Classifier

```
Spatial patches (B, 3, 4, 4)
    ↓ Linear projection → patch embeddings (B, N, D)
    ↓ + learnable positional encoding
    ↓ Transformer encoder (4 layers, 8 heads, D=256)
    ↓ CLS token extraction
    ↓ MLP head → (B, 5) logits

Rainfall sequence (B, 13)
    ↓ 1D Conv → (B, 64) embedding

Conditioning vector (B, 4)
    ↓ MLP (4→64→64) → (B, 64) embedding

[CLS + rainfall_emb + cond_emb] → concat → classifier → (B, 5)
```

| Parameter | Value |
|---|---|
| `embed_dim` | 256 |
| `num_layers` | 4 |
| `num_heads` | 8 |
| `mlp_ratio` | 2.0 |
| `dropout` | 0.1 |
| `rainfall_method` | conv |
| `conditioning_hidden` | 64 |

---

### Branch B — CNN Flood Model (EfficientNet-B3 U-Net + FiLM)

```
Rainfall (B,13) + Conditioning (B,4)
    ↓ RainfallConditioningMLP (17→128→128→256)
    ↓ context vector (B, 256)
    ↓ → γ,β per decoder stage via FiLM projection

Spatial (B, 3, 1152, 1152)
    ↓ EfficientNet-B3 Encoder
    skip₁ (B, 24,  576, 576)
    skip₂ (B, 32,  288, 288)
    skip₃ (B, 48,  144, 144)
    skip₄ (B,136,   72,  72)
    bottleneck (B, 384, 36, 36)
    ↓ U-Net Decoder (upsample + skip concat)
    Stage 1: (B, 256, 72,  72)  → FiLM(γ₁,β₁)
    Stage 2: (B, 128, 144, 144) → FiLM(γ₂,β₂)
    Stage 3: (B,  64, 288, 288) → FiLM(γ₃,β₃)
    Stage 4: (B,  32, 576, 576) → FiLM(γ₄,β₄)
    Stage 5: (B,  16,1152,1152) → FiLM(γ₅,β₅)
    ↓ 1×1 conv → (B, 5, 1152, 1152) logits
```

**FiLM (Feature-wise Linear Modulation):**
```
out = γ * x + β
```
γ and β are derived per-stage from the context vector via a linear projection. This allows the rainfall/conditioning to modulate the spatial feature maps at every scale of the decoder — a much stronger conditioning mechanism than ViT's simple concatenation.

| Parameter | Value |
|---|---|
| `encoder` | efficientnet-b3 (ImageNet) |
| `decoder_channels` | (256,128,64,32,16) |
| `context_dim` | 256 |
| `mlp_hidden` | 128 |
| FiLM stages | 5 (every decoder level) |

---

## 9. Benchmarking: ViT vs CNN

| Aspect | ViT (Branch A) | CNN (Branch B) |
|---|---|---|
| Input unit | 4×4 patch | Full 1152×1152 map |
| Dataset size | 2,679,355 samples | 35 samples |
| Spatial context | Patch only (no neighbors) | Full receptive field |
| Conditioning | Concat to CLS token | FiLM at every decoder stage |
| K-fold split | On patch indices | On scenario indices |
| Batch size | 1024 patches | 2 maps (+ grad accum ×8) |
| Epochs | 20 | 50 |
| Pretrained weights | None | ImageNet (EfficientNet-B3) |
| Flood routing | ❌ No spatial context | ✅ Captures upstream/downstream |
| Training time | ~20 min/epoch | ~5–10 min/epoch |
| Inference complexity | Tile → forward → reconstruct | Single forward pass |

**Expected CNN advantage:** The U-Net encoder-decoder can learn spatial relationships between neighboring pixels — e.g., water flowing from high to low elevation, urban drainage corridors. The ViT classifies each 4×4 patch in complete isolation and cannot learn these patterns.

---

## 10. Normalization Constants

All constants are verified from `rainfall_results` in the training pipeline:

```python
# Rainfall normalization
RAIN_MAX  = 189.853086   # normalization_params['global_max']
RAIN_MIN  = 0.0

# Conditioning normalization
DEPTH_MAX = 78.0         # encoding_info['max_depth_mm']
DEPTH_MIN = 0.0

# OOD training ranges (per storm type)
TRAINING_RANGES = {
    'front-loaded': {'depth': (6,  78), 'tpeak': None},
    'balanced'    : {'depth': (19, 78), 'tpeak': None},
    'back-loaded' : {'depth': (7,  75), 'tpeak': None},
    'triangular'  : {'depth': (5,  77), 'tpeak': (0.1, 0.9)},
}
```

Predictions outside training ranges trigger OOD warnings but still run. Treat OOD predictions with reduced confidence.

---

## 11. Known Limitations

**Heavy class underperformance (ViT F1=0.008).** The Heavy class (0.46–0.68m) is squeezed between Moderate and Extreme with only 2.8% training representation. The model tends to skip it entirely, predicting Extreme directly. Potential fixes: focal loss, Heavy+Extreme merge, or oversampling.

**Single epoch ViT training.** Only 1 epoch was run per fold due to compute constraints. Minority class performance (Light, Moderate, Heavy) would improve significantly with 10–20 epochs and a proper learning rate schedule.

**Spatial generalization gap.** Training on GMM outskirts and testing on Manila core introduces a distribution shift — different urbanization density, drainage infrastructure, and impervious surface fraction. Both models are tested under this harder-than-typical condition.

**Patch independence in ViT.** Each 4×4 patch is classified with no knowledge of adjacent patches. Flood routing (water flowing from high to low elevation across patch boundaries) is entirely invisible to the ViT.

**OOD storm configurations.** Balanced storms only appeared at depths ≥19mm in training. Predicting balanced storms below 19mm is extrapolation.

**Small CNN training set.** With only 35 training scenarios, the CNN is data-limited. Augmentation (random flips, slight rotations, cutout on spatial inputs) could help regularize the model across 50 epochs.

---

## 12. Reproducing Results

### ViT Branch

```python
# Phases 1–8
exec(open('data_preprocessing.py').read())   # phase 2
exec(open('rainfall.py').read())             # phase 3
exec(open('flood_maps.py').read())           # phase 4
exec(open('data_splitting.py').read())       # phase 5
exec(open('patchify.py').read())             # phase 6

from config import TrainConfig
from training import FloodPatchDataset, train_kfold

cfg = TrainConfig()
cfg.use_conditioning = True
full_dataset = FloodPatchDataset(X_train_spatial, y_train, X_train_rainfall, X_train_conditioning)
fold_histories, best_fold_idx = train_kfold(full_dataset, y_train, cfg)

# Save test patches (once)
exec(open('save_test_patches_inline.py').read())

# Inference
from inference_engine import InferenceEngine, predict_with_confidence
engine = InferenceEngine()
result = predict_with_confidence(engine, storm_type='triangular', depth_mm=50, tpeak=0.4)
```

### CNN Branch

```python
# After running Phases 1–5 above:
from reconstruct import reconstruct_for_cnn
cnn_train, cnn_test = reconstruct_for_cnn(split_results, patch_size=4, map_h=1152, map_w=1152)

from dataset_cnn import create_cnn_datasets
train_dataset, test_dataset = create_cnn_datasets(cnn_train, cnn_test)

from config_cnn import CNNTrainConfig
from training_cnn import train_kfold_cnn, evaluate_cnn

cfg = CNNTrainConfig()
fold_histories, best_fold_idx = train_kfold_cnn(train_dataset, y_train, cfg)

# Evaluate
import torch
from training_cnn import make_model_cnn
from torch.utils.data import DataLoader

best_model = make_model_cnn(cfg).to('cuda')
ckpt = torch.load(cfg.output_dir / 'checkpoints' / f'fold_{best_fold_idx+1}_best.pth',
                  map_location='cuda', weights_only=False)
best_model.load_state_dict(ckpt['model_state_dict'])

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
evaluate_cnn(best_model, test_loader, cfg, split_name='Test')
```
