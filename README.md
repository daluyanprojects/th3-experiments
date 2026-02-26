# EXP-DES-2 — Flood Inundation Prediction with Vision Transformer
## Metro Manila · 11-Channel Spatial Input · Optional Drainage & Soil · Web Inference

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Overview](#3-data-overview)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Stage 1 — Metadata & Scenario Loading](#5-stage-1--metadata--scenario-loading)
6. [Stage 2 — Spatial Data Preparation](#6-stage-2--spatial-data-preparation)
7. [Stage 3 — Conditioning Vector Design](#7-stage-3--conditioning-vector-design)
8. [Stage 4 — Channel Masking](#8-stage-4--channel-masking)
9. [Stage 5 — Dataset Construction](#9-stage-5--dataset-construction)
10. [Stage 6 — Model Architecture](#10-stage-6--model-architecture)
11. [Stage 7 — Training](#11-stage-7--training)
12. [Stage 8 — Evaluation](#12-stage-8--evaluation)
13. [Stage 9 — Saving Inference Artifacts](#13-stage-9--saving-inference-artifacts)
14. [Stage 10 — Hyetograph Generator](#14-stage-10--hyetograph-generator)
15. [Stage 11 — Web Inference Engine](#15-stage-11--web-inference-engine)
16. [Stage 12 — Output: GeoTIFF + Visualization](#16-stage-12--output-geotiff--visualization)
17. [Configuration System](#17-configuration-system)
18. [Key Design Decisions & Rationale](#18-key-design-decisions--rationale)
19. [Dependencies](#19-dependencies)
20. [Quick Start for New Developers](#20-quick-start-for-new-developers)

---

## 1. Project Overview

EXP-DES-2 trains a single **Vision Transformer (ViT)** to predict patch-level flood inundation class across a **320×320 raster grid of Metro Manila** given a design storm scenario. The model is explicitly designed for **graceful degradation**: it accepts optional drainage and soil spatial inputs at inference time, handling all four availability combinations (full / drainage-only / soil-only / neither) **without retraining**.

The pipeline is end-to-end: raw DEM and scenario CSVs go in, a deployable web inference engine comes out. A co-developer can run the notebook top-to-bottom and have a working `FloodInferenceEngine` that accepts user-facing storm parameters and returns geo-referenced flood maps with per-patch confidence scores.

### Flood Severity Classes

| Class | Label    | Color   | Hex     |
|-------|----------|---------|---------|
| 0     | No Flood | White   | #FFFFFF |
| 1     | Light    | Yellow  | #FFEB3B |
| 2     | Moderate | Orange  | #FF9800 |
| 3     | Heavy    | Red     | #F44336 |
| 4     | Extreme  | Purple  | #9C27B0 |

### What Changed from the Baseline

| Feature               | EXP-DES-2                       |
|-------------------------|--------------------------------------------------------|
| Conditioning vector              | `(21,)` = rainfall + storm type + flags + tpeak + depth|
| Storm type identity      | Explicit one-hot at `[13:17]`                          |
| Storm depth              | Explicit `depth_norm` scalar at `[20]`                 |
| Peak time fraction       | Explicit `tpeak` scalar at `[19]`                      |
| Drainage input            | Optional — `hasDrainage` flag at `[17]`                |
| Soil input                | Optional — `hasSoil` flag at `[18]`                    |
| Models needed             | 1 (single model, all combinations)                     |
| Rainfall encoder          | MLP — handles mixed-type conditioning                  |
| Confidence scoring        | Softmax max-probability per patch                       |
| Web inference             | `FloodInferenceEngine` + GeoTIFF output                |
| Hyetograph from scratch   | `hyetograph.py` replicates tool exactly                |


| Version   | `conditioning_dim` | Key addition over previous                          |
|-----------|--------------------|-----------------------------------------------------|
| EXP-DES-2 | **21**        | Storm type one-hot + flags, then storm depth + tpeak |

EXP-DES-2 at `(21,)`includes all config that a user controls from the web interface.

---

## 2. Repository Structure

```
project/
│
├── exp-des-2.ipynb                         ← Main training + evaluation notebook
│
├── config.py                               ← DatasetConfig, ModelConfig, TrainConfig
├── vit.py                                  ← ViTFloodClassifier + ConditioningEncoder
├── model_config.py                         ← make_model(), model_factory(), summary
├── training.py                             ← FloodPatchDataset, train_kfold(), run_epoch()
├── testing.py                              ← evaluate_all_scenarios(), predict_scenario(),
│                                             plot_flood_on_dem(), plot_confidence_on_dem()
├── inference.py                            ← FloodInferenceEngine for web deployment
├── hyetograph.py                           ← Design storm hyetograph + conditioning builder
├── data_loader.py                          ← DEM / raster loading utilities
├── data_preprocessing.py                  ← Normalization, patch extraction
├── flood_maps.py                           ← Flood map loading and patching
├── RC_randomizer.py                        ← Stratified 70/30 scenario split utility
│
├── COP-30m-GMM/                            ← Copernicus DEM source tiles
├── COP-30m-GMM-ManilaMaskOut/              ← DEM masked to Manila extent
├── COP-30m-ManilaOnly/                     ← DEM clipped to Manila boundary
├── drainage/                               ← Drainage network rasters (4 layers)
├── soil_types/                             ← Soil classification rasters (4 layers)
├── mm_hr_scenarios/                        ← Original 50 rainfall scenario CSVs
└── rc_randomized_mm_hr_scenarios/          ← Post-split scenario CSVs
    ├── train/                              ← 35 training scenario CSVs
    └── test/                               ← 15 test scenario CSVs
```

**Generated after training:**

```
checkpoints/
├── config.json                             ← Full DatasetConfig + ModelConfig snapshot
├── inference_config.json                   ← Slim config for web backend
├── spatial_data.npz                        ← Pre-extracted spatial patches for inference
├── fold_1/best_model.pt
├── fold_2/best_model.pt
│   ...
└── fold_N/best_model.pt
```

---

## 3. Data Overview

### 3.1 Spatial Layers — 11 Channels, 320×320 Grid

All layers are rasterized to a 320×320 grid covering Metro Manila, min-max normalized, and stacked into an 11-channel spatial array.

| Channel Index | Layer                              | Group    | Optional |
|---------------|------------------------------------|----------|----------|
| 0             | DEM elevation                      | Terrain  | No       |
| 1             | Infiltration rate                  | Terrain  | No       |
| 2             | Land use / LULC                    | Terrain  | No       |
| 3–6           | Drainage network (4 layers)        | Drainage | Yes      |
| 7–10          | Soil properties (4 layers)         | Soil     | Yes      |

Channels 0–2 are always present. Channels 3–6 (`DRAIN_CHANNELS = [3,4,5,6]`) and 7–10 (`SOIL_CHANNELS = [7,8,9,10]`) are **zeroed on-the-fly** when the corresponding flag is disabled — the stored data is never modified.

**Drainage layers (ch 3–6):** `manila_riverbank_new`, `manila_streams_aligned_new`, `waterway_canals_new`, `waterway_streams_new`

**Soil layers (ch 7–10):** `Clay_Content_Percent_Manila`, `Organic_Carbon_Content_Manila`, `Sand_Content_Percent_Manila`, `Soil_pH_H2O_Manila`

### 3.2 Rainfall Scenarios — 50 Total

Scenarios are drawn from 4 design storm families generated by the Design Hyetograph Generator tool and stored in `info.csv`:

| Type ID | Label        | Scenario IDs | Formula                               |
|---------|--------------|--------------|---------------------------------------|
| 1       | front-loaded | 1–10         | SCS Beta PDF α=2.0, β=5.0             |
| 2       | balanced     | 11–20        | SCS Beta PDF α=2.5, β=2.5             |
| 3       | back-loaded  | 21–30        | SCS Beta PDF α=5.0, β=2.0             |
| 4       | triangular   | 31–50        | Linear ramps at peak fraction `tpeak` |

Each scenario CSV has a 13-row `intensity_mmhr` column — 12 real 5-minute block intensities plus one trailing zero.

`info.csv` also stores `depth_mm` and `tpeak` per scenario. These are extracted to build `tpeak_arr` and `depth_norm_arr` for `FloodPatchDataset` — see Stage 5.

### 3.3 Train / Test Split — Scenario Level

| Split | Scenarios | Storm type breakdown                   | Samples           |
|-------|-----------|----------------------------------------|-------------------|
| Train | 35        | 7 each from types 1–3, 14 from type 4 | 224,000 per epoch |
| Test  | 15        | 3 each from types 1–3, 6 from type 4  | 96,000 for eval   |

Splitting at the scenario level prevents spatial leakage — no patches from the same storm appear in both train and test.

**Critical — numerical sort when loading CSVs:**

```python
# os.listdir sorts lexicographically: Scenario_10 before Scenario_2
sorted(files, key=lambda f: int(re.search(r'Scenario_(\d+)', f).group(1)))
```

### 3.4 Normalization Statistics

All normalization stats are computed from the **training set only** and stored in `inference_config.json` for use at web inference time.

**Rainfall:**
```python
rain_min = train_sequences.min()   # 0.0
rain_max = train_sequences.max()   # 189.8531 mm/hr
```

**Storm depth:**
```python
train_depths = [scenario_metadata[sid]['depth_mm'] for sid in train_scenario_ids]
depth_min = min(train_depths)
depth_max = max(train_depths)
```

`tpeak` requires no normalization — it is bounded `[0.1, 0.9]` by definition and is used as-is.

### 3.5 Flood Maps (Ground Truth Labels)

One 320×320 flood inundation raster per scenario. Each raster is divided into 6,400 non-overlapping patches. Each patch receives a single flood class label (0–4) via majority vote across its pixels.

---

## 4. Pipeline Architecture


```
info.csv ──→ scenario_metadata ──────────────────────────────────────────┐
             depth_mm, tpeak, storm_type per scenario                    │
                   │                                                     │
mm_hr_scenarios/ ──→ CSVs ──→ normalize ──→ (13,) rainfall seq           │
                                                  │                      │
                                     storm_type one-hot (4)              │
                                                  │                      │
                                     hasDrainage / hasSoil flags (2)     │
                                                  │                      │
                                     tpeak scalar (1) ───────────────────┤
                                                  │                      │
                                     depth_norm scalar (1) ──────────────┘
                                                  │
                                     conditioning (21,)
                                                  │
DEM ──────────────────────────────────────────────│──→ (11, 4, 4) patch
infiltration ──→ normalize ──→ stack 11 channels  │     + random masking
drainage ×4 ──→ (optional mask)                   │       during training
soil ×4 ──────→ (optional mask)                   │
                                                  ▼
flood maps ──→ patch ──→ majority vote ──→ FloodPatchDataset
                                          (spatial, cond_21, label)
                                                  │
                                                  ▼
                                         ViTFloodClassifier
                                         ConditioningEncoder MLP (21→embed_dim)
                                         PatchEmbedding Conv2d
                                         TransformerEncoder
                                         ClassificationHead
                                                  │
                                                  ▼
                                         train_kfold()
                                         K-fold, scenario-aware splits
                                         CombinedLoss (CE + Dice)
                                         WarmupCosine LR
                                                  │
                               ┌──────────────────┴──────────────────┐
                               ▼                                      ▼
                        evaluate_all_scenarios()              Save artifacts
                        4-pass: Full/Drain/Soil/Neither       spatial_data.npz
                        + confidence maps per scenario        inference_config.json
                               │                              best_model.pt
                               ▼
                        FloodInferenceEngine
                        hyetograph.py ──→ (21,) cond
                        batch forward ──→ flood_map + conf_map
                               │
                               ▼
                        2-band GeoTIFF
                        Band 1: flood class (0–4)
                        Band 2: confidence (0.0–1.0)
```

---

## 5. Stage 1 — Metadata & Scenario Loading

**Source files:** `info.csv`, `rc_randomized_mm_hr_scenarios/train/`, `.../test/`

`info.csv` is the metadata backbone used in two places: building the conditioning vector during training, and at web inference via `hyetograph.py`.

| Column                           | Used for                                      |
|----------------------------------|-----------------------------------------------|
| `No`                             | Scenario ID                                   |
| `Pattern Type`                   | Storm type → one-hot encoding                 |
| `Total storm depth P (mm)`       | `depth_norm` at position `[20]`               |
| `Peak time fraction r (tpeak/D)` | `tpeak` at position `[19]` (triangular only)  |

```python
scenario_metadata = load_scenario_metadata('info.csv')
# Returns: {sid: {'type_id', 'label', 'depth_mm', 'tpeak'}}
```

Normalization stats computed immediately after loading, train set only:

```python
rain_min = train_sequences.min()
rain_max = train_sequences.max()

train_depths = [scenario_metadata[sid]['depth_mm'] for sid in train_scenario_ids]
depth_min    = min(train_depths)
depth_max    = max(train_depths)
```

---

## 6. Stage 2 — Spatial Data Preparation

**Source files:** `COP-30m-ManilaOnly/`, `drainage/`, `soil_types/`

The DEM is loaded via `data_loader.py`, cleaned, and clipped to the 320×320 Manila grid. Drainage (4 rasters) and soil (4 rasters) layers are loaded separately and resized to match.

All 11 layers are individually min-max normalized, then stacked into `(320, 320, 11)` and patched into `(6400, 11, 4, 4)`.

**`dem_meta` is retained from rasterio** — its `transform` and `crs` are passed directly to `save_result_as_tif()` so output GeoTIFFs are georeferenced without any manual coordinate entry.

---

## 7. Stage 3 — Conditioning Vector Design

The model receives a **`(21,)` conditioning vector** per sample, assembled in `FloodPatchDataset.__getitem__`:

```
Position  Segment              Dim   Description
────────  ───────────────────  ───   ────────────────────────────────────────────
[0:13]    Rainfall sequence     13   Normalized intensity per 5-min timestep
[13:17]   Storm type one-hot     4   [1,0,0,0]=front-loaded … [0,0,0,1]=triangular
[17]      hasDrainage flag        1   1.0 = available,  0.0 = zeroed out
[18]      hasSoil flag            1   1.0 = available,  0.0 = zeroed out
[19]      tpeak                   1   Peak time fraction (0.0 for non-triangular)
[20]      depth_norm              1   Normalized total storm depth
```

### Why Each Segment Exists

**`[0:13]` Rainfall sequence** — the primary temporal flood-driving signal. Normalized using `rain_min`/`rain_max` from training data only.

**`[13:17]` Storm type one-hot** — resolves ambiguity between families. Front-loaded and balanced storms can overlap in sequence shape at different depths. The one-hot gives the model an unambiguous family label.

**`[17:18]` Availability flags** — explicitly signal which spatial channel groups are zeroed vs genuinely absent. Without these, the model cannot distinguish "no drainage data provided" from "no drainage infrastructure present."

**`[19]` tpeak** — directly encodes where rainfall peaks within the storm duration. Triangular storms with different `tpeak` values look very different in the sequence; providing it explicitly lets the model disentangle timing from magnitude. Non-triangular storms always receive `0.0`.

**`[20]` depth_norm** — directly encodes total storm volume. Two storms with the same shape but different depths look only proportionally scaled in the sequence. Providing depth as an explicit scalar gives the model an independent handle on total volume, separate from timing. Normalized using `depth_min`/`depth_max` from training scenarios only.

### Depth Normalization

```python
depth_norm = (depth_mm - depth_min) / (depth_max - depth_min + 1e-8)
# Clipped to [-0.1, 1.1] — allows slight out-of-range at inference
```

Test scenarios with depths slightly outside the training range produce values just below 0 or above 1 — this is expected. The confirmed test range is `[-0.014, 0.958]`.

### Building `tpeak_arr` and `depth_norm_arr`

These per-scenario arrays are precomputed from `scenario_metadata` before `FloodPatchDataset` is constructed:

```python
def build_storm_param_arrays(scenario_ids, scenario_metadata, depth_min, depth_max):
    tpeak_arr      = np.zeros(len(scenario_ids), dtype=np.float32)
    depth_norm_arr = np.zeros(len(scenario_ids), dtype=np.float32)
    for i, sid in enumerate(scenario_ids):
        meta = scenario_metadata[sid]
        tpeak_arr[i]      = float(meta['tpeak']) if meta['tpeak'] is not None else 0.0
        depth_norm_arr[i] = (meta['depth_mm'] - depth_min) / (depth_max - depth_min + 1e-8)
    return tpeak_arr, depth_norm_arr

train_tpeak_arr, train_depth_norm_arr = build_storm_param_arrays(
    train_scenario_ids, scenario_metadata, depth_min, depth_max
)
test_tpeak_arr, test_depth_norm_arr = build_storm_param_arrays(
    test_scenario_ids, scenario_metadata, depth_min, depth_max
)
```

---

## 8. Stage 4 — Channel Masking

The model is trained with **random independent channel group dropout** so it learns to operate under all 4 input combinations:

| hasDrainage | hasSoil | Training frequency |
|-------------|---------|-------------------|
| True        | True    | ~25% of samples   |
| True        | False   | ~25% of samples   |
| False       | True    | ~25% of samples   |
| False       | False   | ~25% of samples   |

When a group is masked, two things happen simultaneously in `__getitem__`:
1. The corresponding spatial channels are **zeroed** in the patch tensor
2. The availability flag in the conditioning vector is set to `0.0`

The model sees both signals at once and learns their joint meaning. At **evaluation and inference**, masking is deterministic — set by explicit `hasDrainage`/`hasSoil` arguments.

```python
if self.training:
    has_drain = bool(np.random.randint(0, 2))
    has_soil  = bool(np.random.randint(0, 2))
else:
    has_drain = self.hasDrainage
    has_soil  = self.hasSoil

if not has_drain:
    spatial_patch[DRAIN_CHANNELS] = 0.0
if not has_soil:
    spatial_patch[SOIL_CHANNELS]  = 0.0
```

---

## 9. Stage 5 — Dataset Construction

**Key class:** `FloodPatchDataset` in `training.py`

```python
train_pytorch_dataset = FloodPatchDataset(
    dataset_dict   = train_dataset,
    training       = True,
    tpeak_arr      = train_tpeak_arr,        # (35,) float32
    depth_norm_arr = train_depth_norm_arr,   # (35,) float32
)

test_pytorch_dataset = FloodPatchDataset(
    dataset_dict   = test_dataset,
    training       = False,
    hasDrainage    = True,
    hasSoil        = True,
    tpeak_arr      = test_tpeak_arr,         # (15,) float32
    depth_norm_arr = test_depth_norm_arr,    # (15,) float32
)
```

### Internal Storage

`tpeak_arr` and `depth_norm_arr` are expanded from `(N_scenarios,)` to `(total_samples,)` using `np.repeat(..., patches_per_scenario)` so each flat sample index directly maps to its scenario's values:

```python
self.tpeak_flat = np.repeat(tpeak_arr, self.patches_per_scenario)   # (total_samples,)
self.depth_flat = np.repeat(depth_norm_arr, self.patches_per_scenario)
```

### `__getitem__` — `(21,)` Assembly

```python
rain_and_type = self.rain_and_type_flat[idx]              # (17,) rainfall + onehot
flags         = [float(has_drain), float(has_soil)]       # (2,)
extra         = [self.tpeak_flat[idx], self.depth_flat[idx]]  # (2,)
conditioning  = np.concatenate([rain_and_type, flags, extra]) # (21,)
```

### Confirmed Output

```
FloodPatchDataset initialized:
  Mode              : TRAINING (random masking)
  Total samples     : 224,000
  Conditioning dim  : (21,)  ✓
  tpeak range       : [0.000, 0.900]
  depth_norm range  : [0.000, 1.000]

FloodPatchDataset initialized:
  Mode              : EVAL/INFERENCE (fixed masking)
  Total samples     : 96,000
  Conditioning dim  : (21,)  ✓
  tpeak range       : [0.000, 0.700]
  depth_norm range  : [-0.014, 0.958]   ← slight extrapolation, expected
```

---

## 10. Stage 6 — Model Architecture

**File:** `vit.py`

### Token Flow

```
spatial_patch (B, 11, 4, 4)          conditioning (B, 21)
        │                                    │
        ▼                                    ▼
PatchEmbedding                       ConditioningEncoder
Conv2d(11 → embed_dim)               MLP: 21 → 128 → 256 → embed_dim
→ patch_token (B, 1, embed_dim)      → cond_token (B, 1, embed_dim)
        │                                    │
        └──────────── concat ────────────────┘
                           │  (B, 2, embed_dim)
              + positional encoding (learnable, 2 positions)
                           │
               TransformerEncoder × num_layers
                           │
               ClassificationHead
               reads token at index 1 (patch token position)
               Linear(embed_dim → num_classes)
                           │
               logits (B, 5)
```

### `ConditioningEncoder` — MLP not Conv1d

The `(21,)` conditioning vector is heterogeneous: temporal intensities, a categorical one-hot, binary flags, a bounded scalar, and a normalized scalar. Conv1d assumes translational invariance along the sequence dimension — meaningless for positions 13–20. An MLP treats all 21 dimensions correctly as a flat feature vector.

```
(21,) → Linear(21→128) → GELU → Dropout → Linear(128→256) → GELU → Dropout → Linear(256→embed_dim)
```

### Default Hyperparameters

| Parameter          | Default | Description                         |
|--------------------|---------|-------------------------------------|
| `embed_dim`        | 256     | Token embedding size                |
| `num_layers`       | 4       | Transformer encoder depth           |
| `num_heads`        | 8       | Self-attention heads                |
| `mlp_ratio`        | 2.0     | MLP hidden / embed ratio            |
| `dropout`          | 0.1     | Dropout rate                        |
| `conditioning_dim` | **21**  | Input dim to ConditioningEncoder    |

`conditioning_dim` is read from `cfg.data.conditioning_dim` — no hardcoded dimension anywhere in `vit.py`. Changing it in config is sufficient.

---

## 11. Stage 7 — Training

**File:** `training.py`

### Loss Function

```python
CombinedLoss = 0.9 × CrossEntropyLoss(class_weighted) + 0.1 × DiceLoss
```

Class weights are computed per fold from that fold's training labels using sklearn's `balanced` strategy, addressing the severe class imbalance between No Flood and flooded classes.

### K-Fold Cross-Validation

```python
fold_histories, best_fold_idx = train_kfold(
    model_factory      = lambda: make_model(cfg.data, cfg.model),
    full_dataset       = train_pytorch_dataset,
    scenario_to_samples= scenario_to_samples,
    cfg                = cfg.train,
    device             = device,
)
```

Folds are split at the **patch level**. Each fold uses `SubsetRandomSampler` over flat indices. Random channel masking runs independently each epoch, so the model sees different masking combinations throughout training.

### Optimizer & Scheduler

```python
optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = WarmupCosineScheduler(optimizer, cfg)
# Linear warmup for cfg.warmup_epochs, then cosine decay to cfg.min_lr
```

Gradient clipping: max norm 1.0 per step. Early stopping patience: 10 epochs on validation macro F1.

### Checkpoint Format

```python
# checkpoints/fold_{n}/best_model.pt
{
    'model_state_dict': OrderedDict,
    'best_macro_f1':    float,
    'fold':             int,
}
```

`best_fold_idx` (0-indexed) identifies the fold with highest validation macro F1. Stored as `best_fold = best_fold_idx + 1` (1-indexed) in `inference_config.json` for automatic model selection.

---

## 12. Stage 8 — Evaluation

**File:** `testing.py`

### `predict_scenario()`

Runs inference for one scenario across all test patches, returning both predictions and confidence:

```python
preds, confs = predict_scenario(
    model, X_test, conditioning_vectors,
    scenario_idx, batch_size, device
)
# preds : (N_patches,) int64   — flood class per patch
# confs : (N_patches,) float32 — softmax max-probability per patch
```

### `evaluate_all_scenarios()`

```python
results = {
    'per_scenario': [{'scenario_id', 'accuracy', 'f1_macro', 'iou_macro',
                       'preds_flat', 'confs_flat', ...}, ...],
    'aggregate':    {'accuracy': (mean, std), 'f1_macro': (mean, std), ...
                     'best_scenario': int, 'worst_scenario': int},
    'pred_maps':    (N_scenarios, n_h, n_w),
    'conf_maps':    (N_scenarios, n_h, n_w),
    'gt_maps':      (N_scenarios, n_h, n_w),
}
```

### Metrics Per Scenario

| Metric            | Description                               |
|-------------------|-------------------------------------------|
| `accuracy`        | Overall patch classification accuracy     |
| `f1_macro`        | Mean F1 across 5 classes — primary metric |
| `iou_macro`       | Mean IoU across 5 classes                 |
| `precision_macro` | Mean precision across 5 classes           |
| `recall_macro`    | Mean recall across 5 classes              |
| `confs_flat`      | `(N_patches,)` softmax confidence scores  |

### Confidence Scoring

Confidence = softmax max-probability. Below 0.5 means no class exceeds 50% probability — the model is genuinely uncertain. These patches are highlighted in `plot_confidence_on_dem()`.

---

## 13. Stage 9 — Saving Inference Artifacts

Three artifacts saved to `checkpoints/` after training:

### `inference_config.json`

```python
inference_config = {
    'rain_min'        : cfg.data.rain_min,        # 0.0
    'rain_max'        : cfg.data.rain_max,        # 189.8531
    'depth_min'       : float(depth_min),         # min depth from train scenarios
    'depth_max'       : float(depth_max),         # max depth from train scenarios
    'patch_size'      : cfg.data.patch_size,      # 4
    'num_classes'     : cfg.data.num_classes,     # 5
    'input_channels'  : cfg.data.input_channels,  # 11
    'conditioning_dim': cfg.data.conditioning_dim, # 21
    'drain_channels'  : cfg.data.drain_channels,  # [3,4,5,6]
    'soil_channels'   : cfg.data.soil_channels,   # [7,8,9,10]
    'best_fold'       : best_fold_idx + 1,         # 1-indexed
    'n_steps'         : 13,
    'n_real'          : 12,
    'dt_min'          : 5.0,
    'duration_hr'     : 1.0,
}
```

`depth_min` and `depth_max` are stored here because the web backend must normalize user-supplied `depth_mm` identically to training. `tpeak` is **not** stored — it is a raw user input already in `[0.1, 0.9]` and requires no normalization.

### `spatial_data.npz`

```python
np.savez_compressed('checkpoints/spatial_data.npz',
                     spatial_patches=spatial_patches)  # (6400, 11, 4, 4)
```

### `fold_{best}/best_model.pt`

Already saved by `train_kfold`. The `best_fold` key in `inference_config.json` tells `FloodInferenceEngine` which fold to load automatically.

---

## 14. Stage 10 — Hyetograph Generator

**File:** `hyetograph.py`

The web backend has no access to scenario CSVs. `hyetograph.py` reconstructs the exact same `(13,)` intensity sequence from user inputs that would have been in the training CSV.

### SCS Storm Types (front-loaded, balanced, back-loaded)

Beta PDF midpoint method — identical to the Design Hyetograph Generator tool:

```python
t_mids   = [(i + 0.5) / 12 for i in range(12)]      # block midpoints as fractions
pdf_vals = beta_dist.pdf(t_mids, alpha, beta)          # Beta PDF at midpoints
weights  = pdf_vals / pdf_vals.sum()                   # normalize → depth fractions
intensities = (weights * depth_mm) / (5/60)           # convert depth/block → mm/hr
```

Beta parameters: front-loaded (2.0, 5.0), balanced (2.5, 2.5), back-loaded (5.0, 2.0).

### Triangular Storm Type

```python
n_rising  = math.floor(tpeak * 12) + 1               # floor+1 rule — matches tool
n_falling = 12 - n_rising
peak_intensity    = depth_mm / (DT_HR * ((n_rising+1)/2 + n_falling/2))
rising_increment  = peak_intensity / n_rising
falling_decrement = peak_intensity / (n_falling + 1)  # never reaches zero
```

The `floor(tpeak × 12) + 1` and `/(n_falling + 1)` are critical — they exactly replicate the source tool.

### Trailing Zero

```python
sequence = np.append(intensities, 0.0)   # (13,) — matches training CSV format
```

### `build_conditioning_vector()` — Full `(21,)` Assembly

```python
def build_conditioning_vector(
    storm_type  : str,
    depth_mm    : float,
    rain_min    : float,
    rain_max    : float,
    depth_min   : float,       # from inference_config.json
    depth_max   : float,       # from inference_config.json
    hasDrainage : bool,
    hasSoil     : bool,
    tpeak       : Optional[float] = None,   # user-defined for triangular
) -> Tuple[np.ndarray, list]:

    warnings      = validate_user_inputs(storm_type, depth_mm, tpeak)
    raw_sequence  = generate_hyetograph(storm_type, depth_mm, tpeak)   # (13,)
    norm_sequence = normalize_hyetograph(raw_sequence, rain_min, rain_max)
    onehot        = get_storm_type_onehot(storm_type)                   # (4,)
    flags         = [float(hasDrainage), float(hasSoil)]                # (2,)
    tpeak_val     = float(tpeak) if storm_type == 'triangular' else 0.0
    depth_norm    = clip((depth_mm - depth_min) / (depth_max - depth_min), -0.1, 1.1)
    extra         = [tpeak_val, depth_norm]                             # (2,)

    conditioning  = np.concatenate([norm_sequence, onehot, flags, extra])  # (21,)
    assert conditioning.shape == (21,)
    return conditioning, warnings
```

### Input Validation Ranges

| Storm Type   | depth_mm  | tpeak    |
|--------------|-----------|----------|
| front-loaded | 6–78 mm   | N/A      |
| balanced     | 19–78 mm  | N/A      |
| back-loaded  | 7–75 mm   | N/A      |
| triangular   | 5–77 mm   | 0.1–0.9  |

Out-of-range inputs generate non-blocking warnings shown as banners in the visualization.

---

## 15. Stage 11 — Web Inference Engine

**File:** `inference.py`

### Initialization

```python
engine = FloodInferenceEngine(
    inference_config_path = 'checkpoints/inference_config.json',
    spatial_data_path     = 'checkpoints/spatial_data.npz',
)
```

On init the engine loads all normalization constants (including `depth_min`/`depth_max`), spatial patches, and the best fold model auto-selected via `best_fold` in the config.

### `predict()` — Internal Flow

```
1. build_conditioning_vector(
       storm_type, depth_mm, rain_min, rain_max,
       depth_min, depth_max,          ← from inference_config.json
       hasDrainage, hasSoil,
       tpeak                          ← user-defined, passed directly
   ) → (21,) conditioning vector

2. np.tile(cond, (6400, 1))  → (6400, 21)

3. _apply_channel_masking() per patch
   → (6400, 11, 4, 4)

4. Batch forward pass
   → softmax(logits) → probs (B, 5)
   → argmax → preds  (B,)   flood class
   → max    → confs  (B,)   confidence score

5. _reconstruct_map()
   → reshape + upsample → (320, 320)
```

### Usage from Notebook

```python
result = predict_with_confidence(
    engine,
    storm_type  = 'triangular',
    depth_mm    = 10,          # user-defined → normalized internally
    tpeak       = 0.5,         # user-defined → used as-is at position [19]
    hasDrainage = True,
    hasSoil     = False,
)

# result['flood_map']   — (320, 320) flood class map
# result['conf_map']    — (320, 320) confidence map
# result['patch_preds'] — (6400,) flat predictions
# result['patch_confs'] — (6400,) flat confidence scores
# result['warnings']    — list of out-of-range warnings
```

---

## 16. Stage 12 — Output: GeoTIFF + Visualization

### GeoTIFF Format

Each prediction is saved as a **2-band GeoTIFF** georeferenced using `dem_meta` from rasterio:

```
Band 1 — Flood Hazard Class : float32, values 0–4
Band 2 — Model Confidence   : float32, values 0.0–1.0
```

```python
save_result_as_tif(
    flood_map = result['flood_map'],
    conf_map  = result['conf_map'],
    config    = config,
    save_path = str(cfg.train.output_dir / f'web_config{i+1}_result.tif'),
    transform = dem_meta['transform'],   # exact affine from DEM
    crs       = dem_meta['crs'],
    map_size  = (320, 320),
)
```

File-level tags store `storm_type`, `depth_mm`, `tpeak` so the file is self-documenting when opened in QGIS/ArcGIS.

### PNG Figures

**Figure 1 — `web_config{n}_flood_map.png`:**
- Left: flood class map colored by severity
- Right: horizontal bar chart of class coverage percentages

**Figure 2 — `web_config{n}_confidence_map.png`:**
- Left: RdYlGn heatmap (red=uncertain, green=confident)
- Left overlay: semi-transparent blue on patches with conf < 0.5
- Left annotation: % uncertain (red if > 20%)
- Right: histogram of patch confidence distribution with mean and 0.5 threshold lines

---

## 17. Configuration System

**File:** `config.py`

### `DatasetConfig`

```python
@dataclass
class DatasetConfig:
    patch_size        : int   = 4
    num_classes       : int   = 5
    rainfall_timesteps: int   = 13
    num_storm_types   : int   = 4
    num_channel_flags : int   = 2
    conditioning_dim  : int   = 21      # 13 + 4 + 2 + 2
    input_channels    : int   = None
    drain_channels    : list  = [3, 4, 5, 6]
    soil_channels     : list  = [7, 8, 9, 10]
    rain_min          : float = 0.0
    rain_max          : float = 189.8531
```

`conditioning_dim` is derived in `populate()` as:
```python
self.conditioning_dim = rain_and_type_dim + self.num_channel_flags + 2  # +2 for tpeak, depth_norm
```

### `ModelConfig`

```python
embed_dim   = 256
num_layers  = 4
num_heads   = 8
mlp_ratio   = 2.0
dropout     = 0.1
```

### `TrainConfig`

```python
batch_size          = 1024
num_epochs          = 50
num_folds           = 5
lr                  = 1e-4
weight_decay        = 0.05
warmup_epochs       = 10
min_lr              = 1e-6
output_dir          = Path('./checkpoints')
```

---

## 18. Key Design Decisions & Rationale

**Why extend to `(21,)` from `(19,)`?**
The original `(19,)` vector encoded `depth_mm` and `tpeak` only implicitly via sequence shape and magnitude. Two triangular storms with the same `tpeak` but different depths look only proportionally scaled — the model must infer total volume from the sequence alone. Adding explicit scalars at positions `[19]` and `[20]` gives the model independent handles on timing and magnitude. This mirrors the actual web UI: the user directly controls `storm_type`, `depth_mm`, `tpeak`, `hasDrainage`, and `hasSoil` — all five are now explicitly represented in the conditioning vector.

**Why is `tpeak = 0.0` for non-triangular storms?**
Non-triangular storms have no concept of a peak time fraction — SCS storms are fully defined by storm type and depth. Using `0.0` keeps the input within the observed training distribution for that position. A sentinel like `-1.0` would introduce values never seen during training.

**Why store `depth_min`/`depth_max` in config but not `tpeak` bounds?**
`depth_mm` must be normalized to match the training-time scaling before the model sees it. `tpeak` is already in `[0.1, 0.9]` by definition — it requires no normalization and is passed directly to position `[19]`.

**Why scenario-level train/test splitting?**
Patch-level splitting would allow patches from the same storm to appear in both train and test. The model would memorize spatial patterns per conditioning vector rather than generalizing to new rainfall events. Scenario-level splitting ensures the test set contains truly unseen storms.

**Why MLP for `ConditioningEncoder` instead of Conv1d?**
The `(21,)` vector is heterogeneous: temporal intensities + categorical one-hot + binary flags + bounded scalar + normalized scalar. Conv1d assumes translational invariance along the sequence dimension — meaningless for positions 13–20. An MLP treats all 21 dimensions correctly as a flat feature vector.

**Why random channel masking during training instead of 4 separate models?**
Random masking teaches shared representations across all input combinations. Four separate models would require 4× compute, 4× storage, and would not share terrain learning across configurations.

**Why reuse `dem_meta['transform']` for GeoTIFF output?**
Reusing the rasterio affine transform from the DEM guarantees pixel-perfect spatial alignment with the input DEM in any GIS application. Manual coordinate computation introduces rounding errors.

---

## 19. Dependencies

```
torch          >= 2.0
numpy          >= 1.24
pandas
scipy                    # Beta distribution for SCS hyetograph generation
scikit-learn             # metrics, class weights, KFold
matplotlib
rasterio                 # GeoTIFF read/write with georeferencing
```

---

## 20. Quick Start for New Developers

**Step 1 — Install:**
```bash
pip install torch numpy pandas scipy scikit-learn matplotlib rasterio
```

**Step 2 — Verify data is in place:**
```
info.csv                          ← 50 scenarios with depth_mm and tpeak
rc_randomized_mm_hr_scenarios/    ← train/ and test/ CSV folders
COP-30m-ManilaOnly/               ← DEM
drainage/  soil_types/            ← optional spatial layers
```

**Step 3 — Run notebook top to bottom. Watch for:**

```
DatasetConfig populated:
  conditioning_dim : 21   ← must be 21

FloodPatchDataset initialized:
  Conditioning dim : (21,)  ✓
  tpeak range      : [0.000, 0.900]
  depth_norm range : [0.000, 1.000]
```

**Step 4 — After training, verify `checkpoints/inference_config.json`:**
```json
{
  "rain_min": 0.0,
  "rain_max": 189.8531,
  "depth_min": <from train scenarios>,
  "depth_max": <from train scenarios>,
  "conditioning_dim": 21,
  "best_fold": <N>,
  ...
}
```

**Step 5 — Run web inference:**
```python
engine = FloodInferenceEngine(
    inference_config_path = 'checkpoints/inference_config.json',
    spatial_data_path     = 'checkpoints/spatial_data.npz',
)

# Triangular storm — user provides all three storm parameters
result = predict_with_confidence(
    engine,
    storm_type  = 'triangular',
    depth_mm    = 10,
    tpeak       = 0.5,
    hasDrainage = True,
    hasSoil     = True,
)

# Non-triangular storm — tpeak not needed, auto-set to 0.0 internally
result = predict_with_confidence(
    engine,
    storm_type  = 'balanced',
    depth_mm    = 29,
    hasDrainage = True,
    hasSoil     = False,
)
```

**Step 6 — Check outputs in `checkpoints/`:**
```
web_config1_flood_map.png       ← flood class map + distribution bar chart
web_config1_confidence_map.png  ← confidence heatmap + histogram
web_config1_result.tif          ← 2-band GeoTIFF, open directly in QGIS
```

**Step 7 — Open `.tif` in QGIS:**
- Band 1: categorized symbology, values 0–4, flood class colors
- Band 2: singleband pseudocolor, RdYlGn, range 0.0–1.0
- Both layers align automatically with the input DEM — no manual georeferencing needed

---

*Pipeline verified end-to-end: DEM + info.csv → `(21,)` conditioning → ViT training → scenario evaluation → web inference → 2-band GeoTIFF.*