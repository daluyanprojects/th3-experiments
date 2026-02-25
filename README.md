# EXP-DES-2: Vision Transformer Flood Inundation Prediction
## Metro Manila — Design Storm Flood Mapping Pipeline

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Overview](#3-data-overview)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Stage 1 — Metadata Loading](#5-stage-1--metadata-loading)
6. [Stage 2 — Spatial Data Preparation](#6-stage-2--spatial-data-preparation)
7. [Stage 3 — Rainfall Data & Normalization](#7-stage-3--rainfall-data--normalization)
8. [Stage 4 — Conditioning Vector Design](#8-stage-4--conditioning-vector-design)
9. [Stage 5 — Dataset Construction](#9-stage-5--dataset-construction)
10. [Stage 6 — Model Architecture](#10-stage-6--model-architecture)
11. [Stage 7 — Training](#11-stage-7--training)
12. [Stage 8 — Evaluation (4-Pass)](#12-stage-8--evaluation-4-pass)
13. [Stage 9 — Web Inference Pipeline](#13-stage-9--web-inference-pipeline)
14. [Hyetograph Generator](#14-hyetograph-generator)
15. [Configuration System](#15-configuration-system)
16. [Key Design Decisions](#16-key-design-decisions)
17. [Dependencies](#17-dependencies)
18. [Quick Start for New Developers](#18-quick-start-for-new-developers)

---

## 1. Project Overview

EXP-DES-2 trains a **Vision Transformer (ViT)** to predict flood inundation class per spatial patch given:

- A **design storm** defined by type, total depth (mm), and peak time fraction
- **Spatial context** layers including DEM, infiltration, land use, drainage networks, and soil properties
- **Optional availability** of drainage and soil data (the model handles missing layers gracefully via channel masking)

The output is a **320×320 flood class map** with 5 classes:

| Class | Label    | Meaning                    |
|-------|----------|----------------------------|
| 0     | No Flood | Dry                        |
| 1     | Light    | Minor inundation           |
| 2     | Moderate | Significant inundation     |
| 3     | Heavy    | Severe inundation          |
| 4     | Extreme  | Catastrophic inundation    |

The model is designed for **web deployment** — a user provides storm parameters and data availability toggles, and the model returns a flood map in real time without requiring the original scenario CSVs.

---

## 2. Repository Structure

## 2. Repository Structure
```
EXP-DES-2/
│
├── exp-des-2.ipynb             ← Main training notebook (all stages run here)
│
├── config.py                   ← DatasetConfig, ModelConfig, TrainConfig dataclasses
├── vit.py                      ← ViTFloodClassifier architecture
├── model_config.py             ← make_model() factory function
├── training.py                 ← train_epoch(), validate(), train_kfold()
├── testing.py                  ← evaluate_test_scenarios(), plot_confidence_map()
├── hyetograph.py               ← Hyetograph generator for web inference
├── inference.py                ← FloodInferenceEngine for web backend
├── data_loader.py              ← Data loading utilities
├── data_preprocessing.py       ← Spatial raster preprocessing
├── flood_maps.py               ← Flood map categorization and patch labeling
├── RC_randomizer.py            ← Rainfall scenario randomization utilities
│
├── info.csv                    ← Scenario metadata (storm type, depth, tpeak)
├── README.md                   ← This file
├── .gitignore
│
├── COP-30m-GMM/                
├── COP-30m-GMM-ManilaMaskOut/  
├── COP-30m-ManilaOnly/         
├── drainage/                   ← Drainage rasters
├── soil_types/                 ← Soil rasters
├── mm_hr_scenarios/            ← Rainfall scenario CSVs
├── rc_randomized_mm_hr_scenarios/ ← Randomized rainfall scenario variants
│
└── checkpoints/                ← Generated after training
    ├── config.json             ← Full training configuration
    ├── inference_config.json   ← Web inference parameters
    ├── spatial_data.npz        ← Extracted spatial patches for inference
    ├── fold_1/                 ← Best model checkpoint from fold 1
    ├── fold_2/                 ← Best model checkpoint from fold 2
    └── test_results/           ← Per-scenario evaluation outputs
```

---

## 3. Data Overview

### 3.1 Rainfall Scenarios

50 design storm scenarios generated using the **Design Hyetograph Generator** tool (SCS-style Beta distribution and Triangular/Chicago method). Each scenario is a CSV named `Rainfall_Scenario_N_mmhr.csv` with 13 rows of `intensity_mmhr` values — 12 real timesteps at 5-minute intervals over 1 hour, plus one appended trailing zero row.

**Important:** Files are loaded numerically sorted by scenario ID, not lexicographically. `Rainfall_Scenario_1_mmhr.csv` comes before `Rainfall_Scenario_10_mmhr.csv`. The loader must use `sort(key=lambda f: int(re.search(r'Scenario_(\d+)', f).group(1)))` to match the numerical order produced by `get_scenario_ids_from_dir()`.

**Scenario split (scenario-level, not patch-level):**

| Split | Count | Scenario IDs                                          |
|-------|-------|-------------------------------------------------------|
| Train | 35    | 2,3,5,6,7,8,9,12,13,14,15,16,18,19,21,22,23,25,26,28,30,31,32,33,34,36,38,40,42,45,46,47,48,49,50 |
| Test  | 15    | 1,4,10,11,17,20,24,27,29,35,37,39,41,43,44            |

Scenario-level splitting prevents spatial leakage — all 6,400 patches from the same scenario share identical conditioning, so patch-level splitting would allow memorization rather than generalization.

**Storm type distribution:**

| Type | Label        | Total | Train | Test | IDs    | Formula        |
|------|--------------|-------|-------|------|--------|----------------|
| 1    | front-loaded | 10    | 7     | 3    | 1–10   | Beta α=2, β=5  |
| 2    | balanced     | 10    | 7     | 3    | 11–20  | Beta α=2.5, β=2.5 |
| 3    | back-loaded  | 10    | 7     | 3    | 21–30  | Beta α=5, β=2  |
| 4    | triangular   | 20    | 14    | 6    | 31–50  | Linear ramps   |

### 3.2 Spatial Layers

All rasters are 320×320 pixels covering Metro Manila. All layers are min-max normalized before patch extraction.

| Index | Channel                          | Group      | Zeroed when         |
|-------|----------------------------------|------------|---------------------|
| 0     | DEM (Digital Elevation Model)    | Base       | Never               |
| 1     | Infiltration rate                | Base       | Never               |
| 2     | Land use classification          | Base       | Never               |
| 3     | Drain: Manila riverbank          | Drainage   | hasDrainage=False   |
| 4     | Drain: Manila streams aligned    | Drainage   | hasDrainage=False   |
| 5     | Drain: Waterway canals           | Drainage   | hasDrainage=False   |
| 6     | Drain: Waterway streams          | Drainage   | hasDrainage=False   |
| 7     | Soil: Clay content (%)           | Soil       | hasSoil=False       |
| 8     | Soil: Organic carbon content     | Soil       | hasSoil=False       |
| 9     | Soil: Sand content (%)           | Soil       | hasSoil=False       |
| 10    | Soil: Soil pH H2O                | Soil       | hasSoil=False       |

### 3.3 Flood Maps

Ground truth flood inundation rasters per scenario. Each 320×320 map is divided into 6,400 patches of size 4×4. Each patch receives a single flood class label via **majority vote** across its 16 pixels.

---

## 4. Pipeline Architecture

The full pipeline runs sequentially in `notebook.ipynb`:

```
info.csv ──────────────────────────────────────────────────────────────┐
                                                                       │ scenario_metadata
Spatial rasters ──→ normalize ──→ extract_spatial_patches()            │ (type_id, depth_mm, tpeak)
                                  (6400, 11, 4, 4)                     │
                                  ↓ saved to spatial_data.npz          │
                                                                       ▼
Scenario CSVs ──→ extract_rainfall_sequence() ──→ (13,) raw          Stage 4:
               ──→ normalize_sequence()        ──→ (13,) normalized   Conditioning
               ──→ get_storm_type_onehot()     ──→ (4,)  one-hot      Vector
                                                                       (19,) per patch
Flood maps ────→ categorize ──→ patch majority vote                    │
               (35 or 15 scenarios × 6400 patches)                     │
                                                                       ▼
                              create_patch_dataset()
                              ↓
                              FloodPatchDataset
                              (random masking during training,
                               fixed masking during eval)
                                                                       │
                                                                       ▼
                              ViTFloodClassifier
                              Input:  (B, 11, 4, 4) spatial patch
                                    + (B, 19)       conditioning
                              Output: (B, 5)         logits
                                                                       │
                                                                       ▼
                              train_kfold()
                              K-fold, scenario-aware splits
                              Best model saved per fold
                                                                       │
                                                                       ▼
                              evaluate_test_scenarios()
                              4-pass evaluation:
                              Full / Drainage only / Soil only / Neither
                                                                       │
                                                                       ▼
                              inference_config.json + spatial_data.npz
                              ↓
                              FloodInferenceEngine (web backend)
```

---

## 5. Stage 1 — Metadata Loading

**Source:** `info.csv`
**Key function:** `load_scenario_metadata()`

Reads `info.csv` and builds a dict keyed by scenario ID (1–50):

```python
scenario_metadata[sid] = {
    'type_id'  : int,          # 1=front-loaded, 2=balanced, 3=back-loaded, 4=triangular
    'label'    : str,          # human-readable label
    'depth_mm' : float,        # total storm depth P (mm)
    'tpeak'    : float or None # peak time fraction — only for triangular (type 4)
}
```

`depth_mm` and `tpeak` are **not injected into the model during training**. They are implicit in the intensity sequences from the scenario CSVs. These values are stored in metadata solely for use by the web inference hyetograph generator, which reconstructs the intensity sequence from these parameters.

**Storm type parsing** handles Unicode dashes and BOM characters — the CSV uses non-breaking hyphens in pattern names which must be normalized before string matching.

---

## 6. Stage 2 — Spatial Data Preparation

**Key function:** `extract_spatial_patches(dem, infiltration, landuse, drainage, soil, patch_size)`

```
320×320 raster ──→ extract_patches_2d() ──→ (6400, 4, 4)
                                              ↑ one layer

11 layers stacked ──→ (6400, 11, 4, 4)
```

`extract_patches_2d()` reshapes using numpy without loops:
```python
patches = image.reshape(n_h, patch_size, n_w, patch_size)
patches = patches.transpose(0, 2, 1, 3)   # (n_h, n_w, ps, ps)
patches = patches.reshape(-1, patch_size, patch_size)  # (6400, ps, ps)
```

Spatial patches are **shared across all scenarios** — the spatial context of Metro Manila is identical regardless of storm. Per-scenario variation comes entirely from the conditioning vector.

Spatial patches are saved to disk after extraction for web inference:
```python
from inference import save_spatial_data
save_spatial_data(spatial_patches, 'checkpoints/spatial_data.npz')
```

---

## 7. Stage 3 — Rainfall Data & Normalization

**Key functions:** `extract_rainfall_sequence()`, `normalize_sequence()`

Each scenario CSV contains 13 rows. The `intensity_mmhr` column is extracted as a `(13,)` float32 numpy array. The 13th value is always 0.0 (the appended trailing zero).

**Normalization is computed from the training set only** to prevent test data leakage:

```python
rain_min = 0.0          # mm/hr — global min across all 35 train sequences
rain_max = 189.8531     # mm/hr — global max across all 35 train sequences

normalized = (raw - rain_min) / (rain_max - rain_min + 1e-8)
```

These exact values are stored in `cfg.data.rain_min` / `cfg.data.rain_max` and written to `inference_config.json`. The web inference engine must use these same values — using different statistics would place the conditioning vector on a different scale than what the model trained on.

---

## 8. Stage 4 — Conditioning Vector Design

The conditioning vector is the **central design innovation** of EXP-DES-2. Every patch in every scenario receives a `(19,)` vector:

```
conditioning[0:13]  = normalized rainfall intensity sequence
conditioning[13:17] = storm type one-hot encoding
conditioning[17:19] = data availability flags [hasDrainage, hasSoil]
```

### Why 19 dimensions?

| Component              | Dims | Values              | Purpose                              |
|------------------------|------|---------------------|--------------------------------------|
| Rainfall sequence      | 13   | [0.0, 1.0]          | Storm temporal profile               |
| Storm type one-hot     | 4    | {0,1}               | Physical storm mechanism             |
| hasDrainage flag       | 1    | {0.0, 1.0}          | Signal that channels 3–6 are zeroed  |
| hasSoil flag           | 1    | {0.0, 1.0}          | Signal that channels 7–10 are zeroed |

### Why include storm type one-hot?

The rainfall sequence alone is ambiguous. Two scenarios with different storm types can produce similar intensity profiles if they have different depths. More critically, triangular storms with different `tpeak` values have sequences that shift left/right — the model cannot reliably infer the physical mechanism from shape alone. The one-hot gives an explicit, unambiguous signal.

```python
front-loaded: [1, 0, 0, 0]
balanced:     [0, 1, 0, 0]
back-loaded:  [0, 0, 1, 0]
triangular:   [0, 0, 0, 1]
```

### Why include data availability flags?

Without flags, the model cannot distinguish between "drainage data is unavailable" and "this location genuinely has no drainage infrastructure." The flags provide a dual signal: the channels are zeroed AND the model is told why. This allows it to learn conditional flood prediction strategies per availability combination.

### Channel masking during training

```python
# FloodPatchDataset.__getitem__() — training mode
has_drain = bool(np.random.randint(0, 2))   # 50% per sample, independent
has_soil  = bool(np.random.randint(0, 2))

if not has_drain:
    spatial_patch[3:7] = 0.0    # zero drainage channels
if not has_soil:
    spatial_patch[7:11] = 0.0   # zero soil channels

flags        = [float(has_drain), float(has_soil)]
conditioning = concat([rain_and_type(17), flags(2)])  # → (19,)
```

The model sees all 4 availability combinations during training with approximately equal frequency. At evaluation and inference, flags are fixed and explicit.

---

## 9. Stage 5 — Dataset Construction

**Key functions:** `create_patch_dataset()`, `FloodPatchDataset`

### `create_patch_dataset()`

Tiles spatial patches and conditioning across all scenarios into aligned arrays:

```python
dataset = {
    'spatial_patches': np.tile(spatial_patches, (num_scenarios, 1, 1, 1, 1)),
    # shape: (num_scenarios, 6400, 11, 4, 4)

    'rain_and_type': np.tile(rain_and_type[:, np.newaxis, :], (1, 6400, 1)),
    # shape: (num_scenarios, 6400, 17)  ← rainfall(13) + onehot(4), NO flags yet

    'labels': np.stack(patch_labels, axis=0),
    # shape: (num_scenarios, 6400)
}
```

Flags are NOT stored here — they are generated dynamically in `__getitem__()` because they vary randomly per sample during training.

### `FloodPatchDataset`

Returns `(spatial_patch, conditioning, label)` per index:

```python
# Training: random masking each sample
train_pytorch_dataset = FloodPatchDataset(train_dataset, training=True)

# Eval: fixed flags
test_pytorch_dataset = FloodPatchDataset(test_dataset, training=False,
                                          hasDrainage=True, hasSoil=True)
```

**Final dataset sizes:**
- Training: 35 × 6,400 = **224,000 samples**
- Test: 15 × 6,400 = **96,000 samples**

---

## 10. Stage 6 — Model Architecture

**File:** `vit.py`
**Factory:** `model_config.make_model(data_cfg, model_cfg)`

### Token Flow

```
spatial_patch (B, 11, 4, 4)         conditioning (B, 19)
        │                                    │
        ▼                                    ▼
PatchEmbedding                       ConditioningEncoder
Conv2d(11 → embed_dim)               MLP: 19 → 128 → embed_dim
→ (B, embed_dim) spatial token       → (B, embed_dim) context token
        │                                    │
        └──────────── add ──────────────────┘
                          │
                          ▼
                + learnable positional encoding
                          │
                          ▼
              TransformerEncoder × num_layers
              (embed_dim=256, heads=8, mlp_ratio=2.0, dropout=0.1)
                          │
                          ▼
              ClassificationHead
              Linear(embed_dim → num_classes)
                          │
                          ▼
              logits (B, 5)
```

### ConditioningEncoder — Why MLP?

The `(19,)` conditioning vector is **heterogeneous**: temporal intensities at positions 0–12, categorical one-hot at positions 13–16, binary flags at positions 17–18. An MLP treats each position independently and learns arbitrary mappings from the full vector.

```python
ConditioningEncoder(
    Linear(conditioning_dim → 128),
    GELU(),
    Dropout(0.1),
    Linear(128 → embed_dim),   # projects to 256
)
```

### Model Config Defaults

```python
embed_dim   = 256
num_layers  = 4
num_heads   = 8
mlp_ratio   = 2.0
dropout     = 0.1
```

These defaults must match at inference time. `FloodInferenceEngine._load_model()` instantiates `ModelConfig()` with defaults — if you trained with non-default values, pass a custom `ModelConfig` to the engine.

---

## 11. Stage 7 — Training

**File:** `training.py`

### K-Fold Cross-Validation

Training uses **scenario-aware k-fold splitting**. The `scenario_to_samples` dict maps each scenario ID to its patch indices. Folds are built by grouping complete scenarios — no scenario is split across folds.

```python
# Build scenario → sample index mapping
scenario_to_samples = {
    sid: list(range(idx * patches_per_scenario, (idx + 1) * patches_per_scenario))
    for idx, sid in enumerate(train_scenario_ids)
}

fold_histories, best_fold = train_kfold(
    model               = model_factory(cfg.data, cfg.model),
    full_dataset        = train_pytorch_dataset,
    scenario_to_samples = scenario_to_samples,
    cfg                 = cfg.train,
    device              = device,
)
```

`best_fold` is an integer (1-indexed) identifying the fold with the highest validation macro F1. It is saved to `inference_config.json` so `FloodInferenceEngine` automatically selects the best checkpoint.

### Optimization

| Component       | Setting                        |
|-----------------|--------------------------------|
| Loss            | CrossEntropyLoss + (0.9) DiceLoss(0.1)               |
| Optimizer       | AdamW                          |
| Learning rate   | 1e-4 with cosine annealing     |
| Warmup          | 10 epochs linear warmup        |
| Min LR          | 1e-6                           |
| Weight decay    | 0.05                           |
| Gradient clip   | max norm 1.0                   |
| Batch size      | 1024                           |

### Checkpoint Format

Each fold saves `checkpoints/fold_{n}/best_model.pt`:
```python
{
    'model_state_dict': OrderedDict,   # model weights
    'best_macro_f1':    float,         # validation metric at save time
    'epoch':            int,           # epoch when saved
}
```

---

## 12. Stage 8 — Evaluation (4-Pass)

**File:** `testing.py`

The test set is evaluated under **4 input configurations** to measure the performance cost of missing spatial data. This quantifies how much the model degrades when drainage or soil data is unavailable — critical information for real-world deployment scenarios.

| Config Label            | hasDrainage | hasSoil | Expected Performance |
|-------------------------|-------------|---------|----------------------|
| Full (drainage + soil)  | True        | True    | Highest              |
| Drainage only           | True        | False   | Medium-high          |
| Soil only               | False       | True    | Medium-low           |
| Neither (minimal)       | False       | False   | Lowest               |

### `predict_scenario_with_flags()`

Runs patch-by-patch inference for one scenario. Returns:
- `patch_predictions`: `(6400,)` array of predicted flood class per patch
- `patch_confidences`: `(6400,)` array of softmax max-probability per patch (model confidence)

### Metrics Per Scenario

Each scenario × config combination produces:
```python
{
    'accuracy':            float,       # overall patch classification accuracy
    'mean_iou':            float,       # mean IoU across 5 classes
    'iou_per_class':       dict,        # per-class IoU
    'precision_per_class': list,        # per-class precision
    'recall_per_class':    list,        # per-class recall
    'f1_per_class':        list,        # per-class F1
    'confusion_matrix':    list,        # 5×5 confusion matrix
    'mean_confidence':     float,       # average softmax confidence
    'min_confidence':      float,
    'max_confidence':      float,
    'std_confidence':      float,
    'frac_low_confidence': float,       # fraction of patches with confidence < 0.5
}
```

### Confidence Map Visualization

```python
from testing import plot_confidence_map

plot_confidence_map(
    confidence_map = all_confidences['Full (drainage + soil)'][scenario_id],
    predicted_map  = all_predictions['Full (drainage + soil)'][scenario_id],
    scenario_id    = scenario_id,
    config_label   = 'Full (drainage + soil)',
    metrics        = all_results['Full (drainage + soil)'][scenario_id],
    save_path      = f'checkpoints/test_results/confidence_RS{scenario_id}.png',
)
```

Produces a side-by-side visualization: predicted flood class map (left) and model confidence per patch (right), with metric summary. Saved to `test_results/`.

---

## 13. Stage 9 — Web Inference Pipeline

**File:** `inference.py`

### Prerequisites — What Must Be Saved After Training

Three files must exist before `FloodInferenceEngine` can run:

**In the training notebook, after `train_kfold` completes:**

```python
# 1. Save inference config
import json
inference_config = {
    'rain_min'        : cfg.data.rain_min,
    'rain_max'        : cfg.data.rain_max,
    'patch_size'      : cfg.data.patch_size,
    'num_classes'     : cfg.data.num_classes,
    'input_channels'  : cfg.data.input_channels,
    'conditioning_dim': cfg.data.conditioning_dim,
    'drain_channels'  : cfg.data.drain_channels,
    'soil_channels'   : cfg.data.soil_channels,
    'n_steps'         : 13,
    'n_real'          : 12,
    'dt_min'          : 5.0,
    'duration_hr'     : 1.0,
    'best_fold'       : best_fold,
}
with open(cfg.train.output_dir / 'inference_config.json', 'w') as f:
    json.dump(inference_config, f, indent=2)

# 2. Save spatial patches
from inference import save_spatial_data
save_spatial_data(spatial_patches, str(cfg.train.output_dir / 'spatial_data.npz'))
```

**Final checkpoints directory:**
```
checkpoints/
  config.json             ← saved by cfg.save() during setup
  inference_config.json   ← saved after training (includes best_fold)
  spatial_data.npz        ← saved after training
  fold_1/best_model.pt
  fold_2/best_model.pt
```

### FloodInferenceEngine

```python
from inference import FloodInferenceEngine

engine = FloodInferenceEngine(
    inference_config_path = 'checkpoints/inference_config.json',
    spatial_data_path     = 'checkpoints/spatial_data.npz',
)
# Automatically loads checkpoints/fold_{best_fold}/best_model.pt
```

### Running Predictions

```python
# User config 1: Triangular, 9mm, tpeak=0.5, hasSoil=True, hasDrainage=False
result = engine.predict(
    storm_type  = 'triangular',
    depth_mm    = 9,
    tpeak       = 0.5,
    hasDrainage = False,
    hasSoil     = True,
)

# User config 2: Front-loaded, 9mm, no spatial data
result = engine.predict(
    storm_type  = 'front-loaded',
    depth_mm    = 9,
    hasDrainage = False,
    hasSoil     = False,
)

flood_map = result['flood_map']    # (320, 320) numpy array, values 0–4
warnings  = result['warnings']     # list of str — out-of-distribution alerts
```

### Internal Predict Flow

```
User inputs (storm_type, depth_mm, tpeak, hasDrainage, hasSoil)
        │
        ▼ hyetograph.build_conditioning_vector()
        │
        ├─ validate_user_inputs()        → warnings (non-blocking)
        ├─ generate_hyetograph()         → raw (13,) mm/hr sequence
        ├─ normalize_hyetograph()        → normalized (13,) using saved rain_min/max
        ├─ get_storm_type_onehot()       → (4,) one-hot vector
        ├─ flags = [hasDrainage, hasSoil] → (2,) float array
        └─ concat                        → (19,) conditioning vector
        │
        ▼
Apply channel masking to spatial_patches (6400, 11, 4, 4)
  channels 3–6  → zeroed if hasDrainage=False
  channels 7–10 → zeroed if hasSoil=False
        │
        ▼
Batch forward pass (default batch_size=512, tune for GPU memory)
  model(spatial_batch, cond_batch) → logits (B, 5)
  softmax.argmax                   → predictions (6400,)
        │
        ▼
_reconstruct_map()
  (6400,) patch predictions → (80, 80) patch grid → (320, 320) pixel map
  each patch prediction is repeated across its 4×4 pixel area
        │
        ▼
result = {
    'flood_map'  : (320, 320) ndarray,
    'patch_preds': (6400,) ndarray,
    'warnings'   : list of str,
    'config'     : dict of inputs used,
}
```

---

## 14. Hyetograph Generator

**File:** `hyetograph.py`

Replicates the **Design Hyetograph Generator** tool exactly. Given user-defined storm parameters, produces the same `(13,)` intensity sequence as the training CSVs. This is the bridge between web user inputs and model-ready conditioning.

### Fixed Parameters (Same for All 50 Training Scenarios)

| Parameter      | Value        |
|----------------|--------------|
| Duration       | 1.0 hour     |
| Timestep       | 5 minutes    |
| Real blocks    | 12           |
| Total steps    | 13 (12 + trailing zero) |

### SCS-Style Storms (Types 1–3) — Beta PDF Midpoint Method

The tool evaluates the Beta distribution PDF at each block midpoint, then scales to match `depth_mm`:

```python
t_mids    = [(i + 0.5) / 12 for i in range(12)]      # midpoints as duration fractions
pdf_vals  = scipy.stats.beta.pdf(t_mids, alpha, beta) # unnormalized weights
weights   = pdf_vals / pdf_vals.sum()                 # normalized to sum=1
depths    = weights * depth_mm                        # mm per block
intensity = depths / (5/60)                           # convert to mm/hr
```

| Storm Type   | α   | β   | Peak Location |
|--------------|-----|-----|---------------|
| front-loaded | 2.0 | 5.0 | Early (~block 3) |
| balanced     | 2.5 | 2.5 | Middle (symmetric) |
| back-loaded  | 5.0 | 2.0 | Late (~block 10) |

### Triangular Storms (Type 4) — Linear Ramp Method

```python
n_rising  = math.floor(tpeak * 12) + 1   # ← floor+1
n_falling = 12 - n_rising

# Peak intensity from depth constraint
peak = depth_mm / (DT_HR × ((n_rising+1)/2 + n_falling/2))

# Rising limb: evenly spaced from increment to peak
rising_increment = peak / n_rising
rising  = [rising_increment × (i+1) for i in range(n_rising)]

# Falling limb: evenly spaced from (peak - decrement) to decrement. never reaches zero — last value = peak / (n_falling + 1)
falling_decrement = peak / (n_falling + 1)
falling = [peak - falling_decrement × (i+1) for i in range(n_falling)]
```

### Verification Against Training CSVs

Verify the generator matches all 4 storm types:

```python
from hyetograph import verify_against_csv

# Verify all storm types
verify_against_csv('front-loaded', 38,
    extract_rainfall_sequence(test_rs_lookup[1]), tolerance=0.01)

verify_against_csv('balanced', 35,
    extract_rainfall_sequence(test_rs_lookup[11]), tolerance=0.01)

verify_against_csv('back-loaded', 14,
    extract_rainfall_sequence(train_rs_lookup[21]), tolerance=0.01)

verify_against_csv('triangular', 8,
    extract_rainfall_sequence(train_rs_lookup[31]),
    tpeak=0.4, tolerance=0.01)
```

All should print `ALL PASS ✓` with `Max diff: 0.0000 mm/hr`.

### Input Validation

`validate_user_inputs()` checks against training distribution ranges and returns warnings (non-blocking):

| Storm Type   | Depth range | tpeak range |
|--------------|-------------|-------------|
| front-loaded | 6–78 mm     | N/A         |
| balanced     | 19–78 mm    | N/A         |
| back-loaded  | 7–75 mm     | N/A         |
| triangular   | 5–77 mm     | 0.1–0.9     |

Inputs outside these ranges still produce predictions but with a warning string in `result['warnings']`. The web UI should display these warnings to the user.

---

## 15. Configuration System

**File:** `config.py`

### `DatasetConfig`

Populated from live notebook variables via `cfg.data.populate(...)`. Stores all dataset-derived constants. Key fields relevant to web inference:

```python
patch_size        : 4        # spatial patch size
num_classes       : 5        # flood severity classes
rainfall_timesteps: 13       # steps per sequence including trailing zero
conditioning_dim  : 19       # total conditioning vector size
input_channels    : 11       # spatial channels per patch
drain_channels    : [3,4,5,6]
soil_channels     : [7,8,9,10]
rain_min          : 0.0      # mm/hr — normalization min
rain_max          : 189.8531 # mm/hr — normalization max
```

`populate()` call signature:

```python
cfg.data.populate(
    spatial_patches  = spatial_patches,
    train_dataset    = train_dataset,
    test_dataset     = test_dataset,
    train_data       = train_data,
    drainage_resized = drainage_resized,
    soil_resized     = soil_resized,
    rain_min         = rain_min,
    rain_max         = rain_max,
    drain_channels   = DRAIN_CHANNELS,
    soil_channels    = SOIL_CHANNELS,
)
```

### `ModelConfig`

```python
embed_dim   = 256    # transformer embedding dimension
num_layers  = 4      # number of transformer encoder layers
num_heads   = 8      # attention heads
mlp_ratio   = 2.0    # MLP hidden dim = embed_dim × mlp_ratio
dropout     = 0.1
```

### `TrainConfig`

```python
batch_size           = 1024
num_epochs           = 1       # increase for real training
num_folds            = 2
lr                   = 1e-4
weight_decay         = 0.05
warmup_epochs        = 10
min_lr               = 1e-6
grad_clip_norm       = 1.0
early_stop_patience  = 10
checkpoint_metric    = 'macro_f1'
output_dir           = Path('./checkpoints')
```

The full config is saved via `cfg.save(cfg.train.output_dir / 'config.json')` and can be reloaded for reproducibility.

---

## 16. Key Design Decisions

### Scenario-Level Train/Test Split
All 6,400 patches from one scenario share identical conditioning. Patch-level splitting would allow the model to see the conditioning for test scenarios during training and memorize rather than generalize. Scenario-level splitting ensures the test set contains entirely unseen storms.

### Single Model for All Input Combinations
Random channel masking during training — independent 50/50 for each channel group — teaches the model to handle all 4 combinations equally. One model file, no retraining, no separate branches.

### Explicit Storm Type One-Hot
Solves the fundamental ambiguity of the rainfall sequence alone. Two storms with different physical mechanisms can produce overlapping intensity profiles at different depths. The one-hot provides a direct, unambiguous signal about the storm family.

### MLP  for Conditioning
The conditioning vector is heterogeneous (temporal + categorical + binary). MLP treats each position independently.

### Trailing Zero in Rainfall Sequences
All training CSVs have 13 rows (12 real + 1 trailing zero added during data preparation). The hyetograph generator replicates this so web-generated sequences are format-identical to training sequences. Omitting the trailing zero would shift the normalized sequence by one position relative to what the model trained on.

### rain_max Computed from Train Set Only
The normalization max of 189.8531 mm/hr comes from the training scenarios only. Using the full dataset (including test) would constitute data leakage. If a web user provides a storm with higher intensity than 189.8531 mm/hr, the normalized value will exceed 1.0 — `validate_user_inputs()` warns about this but does not block the prediction.

---

## 18. Quick Start for New Developers

**Step 1 — Environment setup**
```bash
pip install <dependencies> ....
```

**Step 2 — Run the training notebook top to bottom**

All stages are sequential cells. Each stage prints a validation summary. Do not skip cells — later stages depend on variables from earlier ones. Watch for these key outputs:
- `DatasetConfig populated` — confirms all shapes and dimensions
- `FloodPatchDataset initialized` — confirms conditioning shape is `(19,)`
- `train_kfold` outputs — fold-by-fold validation metrics
- `best_fold: N` — identifies which fold to use for inference

**Step 3 — Save inference artifacts** (cells after `train_kfold`)
```python
# Cell 1: save inference_config.json
# Cell 2: save spatial_data.npz via save_spatial_data()
```
Confirm both files appear in `checkpoints/`.

**Step 4 — Verify hyetograph generator**
```python
from hyetograph import verify_against_csv
# Run verify_against_csv() for at least one scenario per storm type
# All must print ALL PASS ✓ with Max diff: 0.0000
```

**Step 5 — Test web inference end-to-end**
```python
from inference import FloodInferenceEngine

engine = FloodInferenceEngine(
    inference_config_path = 'checkpoints/inference_config.json',
    spatial_data_path     = 'checkpoints/spatial_data.npz',
)

result = engine.predict(
    storm_type  = 'front-loaded',
    depth_mm    = 38,
    hasDrainage = True,
    hasSoil     = True,
)
print(result['flood_map'].shape)   # → (320, 320)
print(result['warnings'])          # → [] (38mm front-loaded is in training range)
```

**Step 6 — Review 4-pass evaluation**

Check `checkpoints/test_results/` for the cross-config comparison table showing accuracy/mIoU/F1 degradation as spatial inputs are removed. This tells you how much the model degrades for users who lack drainage or soil data.

---

*Last updated: EXP-DES-2, hyetograph generator verified against all 50 training scenarios.*