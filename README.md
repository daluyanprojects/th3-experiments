# Metro Manila Flood Prediction with Vision Transformer (ViT)

A deep learning system for predicting spatial flood patterns under different rainfall scenarios using Vision Transformer architecture, conditioned on storm characteristics.

**Table of Contents:**
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Usage Examples](#usage-examples)
- [Key Components](#key-components)
- [How the Model Learns](#how-the-model-learns)

---

## Overview

### Problem Statement
Flood prediction requires understanding how different rainfall patterns interact with spatial features (elevation, infiltration, land use). This project develops a machine learning model that learns these interactions to predict:
- **Spatial flood classification**: For each location, predict flood intensity (No Flood → Extreme)
- **Scenario conditioning**: Predictions vary based on storm type, depth, and peak timing
- **Generalization**: Predict for unseen rainfall scenarios within training parameter ranges

### Solution Approach
Use a **Vision Transformer (ViT)** that processes:
1. **Spatial patches** (4×4 pixels) of terrain features
2. **Conditioning vectors** (19-dim) encoding storm characteristics
3. **Joint learning** to predict patch-level flood class

### Key Innovation
Unlike traditional rainfall-runoff models, this system **learns end-to-end** from spatial + rainfall data, capturing complex nonlinear interactions automatically.

---

## Pipeline Architecture

### High-Level Data Flow

```
RAW SPATIAL DATA          RAINFALL SCENARIOS              INTEGRATION
     ↓                           ↓                              ↓
  DEM                    50 Storm Configurations       Conditioning Vectors
  Infiltration      →    (Type, Depth, tpeak)    →    (19-dim per scenario)
  Landuse                   ↓
                      Hyetographs Generated
                      (13-step intensities)
                           ↓
┌─────────────────────────────────────────────────┐
│  TRAINING PIPELINE                              │
├─────────────────────────────────────────────────┤
│ 1. Normalize spatial features (DEM, etc.)       │
│ 2. Extract 4×4 spatial patches (85,986 total)   │
│ 3. Categorize 50 flood maps (5 classes)         │
│ 4. Split into train/test (quadrant-based)       │
│ 5. Build 19-dim conditioning vectors            │
│ 6. Create FloodPatchDataset (3.2M samples)      │
│ 7. K-Fold training (5 folds)                    │
│ 8. Save best model                              │
└─────────────────────────────────────────────────┘
     ↓
  TRAINED ViT
     ↓
┌─────────────────────────────────────────────────┐
│  INFERENCE PIPELINE                             │
├─────────────────────────────────────────────────┤
│ 1. User inputs: storm_type, depth_mm, tpeak    │
│ 2. Validate against training ranges             │
│ 3. Generate hyetograph (13-step intensities)    │
│ 4. Normalize with training statistics           │
│ 5. Build 19-dim conditioning vector             │
│ 6. Broadcast to all test patches                │
│ 7. Run batch inference                          │
│ 8. Compute flood statistics                     │
│ 9. Return predictions & warnings                │
└─────────────────────────────────────────────────┘
     ↓
  FLOOD PREDICTIONS
  (Per-patch classification)
```

---

## Data Processing

### 1. Spatial Data Normalization

```python
# Input: Raw raster data at different scales
dem_raw:           [0m, 100m] → normalized to [0, 1]
infiltration_raw:  [mm/hr] → normalized to [0, 1]
landuse_raw:       [0, 12] classes → normalized to [0, 1]

# Method: Min-max scaling
x_norm = (x - x_min) / (x_max - x_min + eps)

# Output: 3-channel image (1224, 1125, 3)
```

### 2. Spatial Patch Extraction

```python
# Input: Full spatial stack (1224, 1125, 3)
# Process: Extract 4×4 patches with stride=4 (non-overlapping)
# Output: 85,986 patches of shape (4, 4, 3)

Grid Layout:
├─ 306 patches rows
├─ 281 patches columns
└─ Total: 306 × 281 = 85,986 patches
```

### 3. Quadrant-Based Train/Test Split

```
Full Map (1224×1125):
┌──────────┬──────────┐
│ Q1 (612) │ Q2 (612) │  ← TRAIN (Q1, Q2, Q4)
├──────────┼──────────┤
│ Q3 (612) │ Q4 (612) │  Q3 = TEST (reserved)
└──────────┴──────────┘
     Q1: 21,573 patches (train)
     Q2: 21,573 patches (train)
     Q3: 21,420 patches (test)  ← Spatially held out
     Q4: 21,420 patches (train)

Train: 64,566 patches × 50 scenarios = 3,228,300 samples
Test:  21,420 patches × 50 scenarios = 1,071,000 samples
```

### 4. Flood Map Categorization

```python
# Input: Continuous flood depth maps (mm)
# Process: Classify into 5 classes using depth thresholds

Classes:
├─ Class 0 (No Flood):    [0.0, 0.15) m
├─ Class 1 (Light):       [0.15, 0.24) m
├─ Class 2 (Moderate):    [0.24, 0.46) m
├─ Class 3 (Heavy):       [0.46, 0.68) m
└─ Class 4 (Extreme):     [0.68, ∞) m

# Output: Discrete class labels per patch (int8)
```

### 5. Rainfall Scenario Normalization

```python
# Input: 50 hyetographs from info.csv
# Raw intensities: [0.003, 0.486] mm/hr

Process:
├─ Load 13-step intensity sequences (50 scenarios)
├─ Compute global min/max across all scenarios
├─ Normalize: x_norm = (x - x_min) / (x_max - x_min)
└─ Output: (50, 13) rainfall embedding

Result:
└─ rain_embeddings: normalized intensities [0, 1]
```

---

## Conditioning Vector Construction

### 19-Dimensional Representation

Each rainfall scenario is converted to a 19-dimensional conditioning vector:

```
conditioning_vector = [
    rainfall[13],        # [0:13]   Normalized intensity sequence
    onehot[4],          # [13:17]  Storm type encoding
    r[1],               # [17]     Peak time fraction
    depth_norm[1]       # [18]     Normalized total depth
]
```

### Per-Component Explanation

#### [0:13] Rainfall Intensities
```python
# What it encodes: Temporal rainfall pattern
# Shape varies by storm type:

Front-loaded:    [0.45, 0.42, 0.38, 0.30, 0.20, ...]  # Peaks early
Balanced:        [0.25, 0.28, 0.31, 0.32, 0.30, ...]  # Uniform
Back-loaded:     [0.05, 0.08, 0.12, 0.30, 0.45, ...]  # Peaks late
Triangular(0.5): [0.10, 0.20, 0.40, 0.50, 0.40, ...]  # Symmetric

# Generated by: hyetograph.generate_hyetograph()
# Normalized by: (x - rain_min) / (rain_max - rain_min)
```

#### [13:17] Storm Type One-Hot
```python
# What it encodes: Which of 4 storm types this is
# One-hot encoding (exactly one value = 1, rest = 0)

Type 1 (Front-loaded):  [1, 0, 0, 0]
Type 2 (Balanced):      [0, 1, 0, 0]
Type 3 (Back-loaded):   [0, 0, 1, 0]
Type 4 (Triangular):    [0, 0, 0, 1]

# Purpose: Explicit label for transformer attention
# Generated by: hyetograph.get_storm_type_onehot()
```

#### [17] r Parameter (Peak Time)
```python
# What it encodes: When peak rainfall occurs (for triangular only)
# Range: [0.0, 0.9]

SCS types (front/balanced/back-loaded):
  r = 0.0  (parameter ignored, shape defined by Beta distribution)

Triangular storms:
  r = tpeak as fraction of storm duration
  r = 0.1  → peak at 10% through duration (early peak)
  r = 0.5  → peak at 50% through duration (mid peak)
  r = 0.9  → peak at 90% through duration (late peak)

# Generated by: scenario_metadata['r']
```

#### [18] Normalized Depth
```python
# What it encodes: Total rainfall magnitude scaled to [0, 1]
# Formula: (depth_mm - depth_min) / (depth_max - depth_min)

Training range: 5.0 to 78.0 mm
Example values:
  5mm depth   → (5 - 5) / (78 - 5) = 0.0
  40mm depth  → (40 - 5) / (78 - 5) = 0.479
  78mm depth  → (78 - 5) / (78 - 5) = 1.0

# Purpose: Scale flood predictions by storm magnitude
# Generated by: computed during build_conditioning_vector()
```

### Example: Full Vector for Scenario

```python
# Scenario: Triangular storm, 40mm depth, tpeak=0.5

conditioning_vector = np.array([
    # Rainfall (13)
    0.016, 0.031, 0.047, 0.063, 0.078, 0.094, 0.078, 0.063, 0.047, 0.031, 0.016, 0.0, 0.0,
    
    # Storm type one-hot (4)
    0.0, 0.0, 0.0, 1.0,  # Type 4 = Triangular
    
    # r parameter (1)
    0.5,  # Peak at 50% duration
    
    # Normalized depth (1)
    0.479,  # 40mm normalized
], dtype=np.float32)

# Shape: (19,)
```

---

## Model Architecture

### Vision Transformer (ViT) Overview

```
Input:
├─ spatial_patch: (B, 3, 4, 4)      # 4×4 spatial tiles
└─ conditioning:  (B, 19)           # Storm parameters

Processing:
├─ PatchEmbedding: spatial → (B, 1, 256)
├─ ConditioningEncoder: conditioning → (B, 256)
├─ PositionalEncoding: add position info
├─ TransformerEncoder: 4 layers of multi-head attention
└─ ClassificationHead: → (B, 5) logits

Output:
└─ logits: (B, 5)  # Probabilities for 5 flood classes
```

### Component Details

#### 1. PatchEmbedding
```python
# Input: (B, 3, 4, 4) spatial patch
# Process: Conv2d projection with stride=4
# Output: (B, 256, 1, 1) → flatten → (B, 1, 256)

Purpose: Convert raw spatial features into embedding space
```

#### 2. ConditioningEncoder (Dual-Branch)
```python
Branch 1: Rainfall Processing
├─ Input: (B, 1, 13) rainfall intensities
├─ Conv1d (1 → 64, kernel=3)
├─ GELU activation
├─ Conv1d (64 → 64, kernel=3)
├─ AdaptiveAvgPool1d: temporal → scalar
└─ Output: (B, 64) rainfall features

Branch 2: Categorical Processing
├─ Input: (B, 6) [onehot(4) + r(1) + depth(1)]
├─ Linear: 6 → 64
├─ GELU
├─ Linear: 64 → 64
└─ Output: (B, 64) categorical features

Fusion:
├─ Concatenate: (B, 128)
├─ Linear: 128 → 256
├─ GELU + LayerNorm
└─ Output: (B, 256) conditioning token
```

**What it learns:**
- Conv1d learns temporal patterns (front-loaded vs back-loaded detection)
- Linear layers learn type semantics (what [1,0,0,0] vs [0,0,0,1] mean)
- Fusion learns storm-aware representations

#### 3. PositionalEncoding
```python
# Input: 2 tokens (cond_token, patch_token)
# Add learnable position embeddings
# Output: Same shape, now with position information

Purpose: Tell transformer "this is conditioning, this is spatial patch"
```

#### 4. TransformerEncoder
```python
# 4 layers of transformer blocks:
├─ LayerNorm → MultiheadAttention (8 heads) → Residual
├─ LayerNorm → MLP (ratio=2.0) → Residual
└─ Repeat 4 times

# What it learns:
# - How to weight spatial features given storm type
# - Which spatial regions matter for different storms
# - Cross-attention: how conditioning modulates spatial prediction
```

#### 5. ClassificationHead
```python
# Input: Transformer output tokens (B, 2, 256)
# Extract: patch_token = tokens[:, 1, :]  # Index 1 = spatial patch
# Process: MLP with GELU + dropout
# Output: (B, 5) logits for flood classes
```

### Model Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│  VISION TRANSFORMER ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Spatial Input (4×4)          Conditioning (19-dim)              │
│  ├─ DEM                       ├─ Rainfall (13)                   │
│  ├─ Infiltration              ├─ Storm type (4)                  │
│  └─ Landuse                   ├─ r parameter (1)                 │
│       ↓                        └─ Depth (1)                      │
│  PatchEmbedding (3→256)            ↓                             │
│       ↓                        ConditioningEncoder               │
│  (B, 1, 256)                  ├─ Rain branch (Conv1d)            │
│       ↓                        ├─ Cat branch (Linear)            │
│  ┌──────────────────┐         └─ Fusion → (B, 256)              │
│  │ Token Sequence   │              ↓                             │
│  ├─ cond_token (256)           (B, 1, 256)                       │
│  └─ patch_token (256)               ↓                             │
│       ↓                        ┌──────────────────┐              │
│  PositionalEncoding             │ Combined Tokens  │              │
│       ↓                         ├─ [cond | patch] │              │
│  ┌──────────────────┐           └──────────────────┘              │
│  │ Transformer (4x) │                ↓                            │
│  ├─ MultiheadAttn   │           TransformerEncoder               │
│  ├─ MLP             │           (8 heads, 4 layers)              │
│  └─ Residuals       │                ↓                            │
│       ↓                        (B, 2, 256) output                 │
│  (B, 2, 256)                        ↓                             │
│       ↓                        ClassificationHead                 │
│  Extract patch_token                ↓                             │
│       ↓                        (B, 5) logits                      │
│  Softmax → (B, 5)                   ↓                             │
│       ↓                        Flood class prediction              │
│  Class probabilities           (0-4: No/Light/Moderate/Heavy/Ex) │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training

### Configuration

```python
class ModelConfig:
    in_channels = 3              # DEM, infiltration, landuse
    patch_size = 4               # 4×4 spatial tiles
    num_classes = 5              # Flood classes
    embed_dim = 256              # Token dimension
    num_layers = 4               # Transformer depth
    num_heads = 8                # Attention heads
    mlp_ratio = 2.0              # MLP expansion
    dropout = 0.1
    conditioning_dim = 19        # Storm parameter vector
    rainfall_hidden = 64         # ConditioningEncoder hidden

class TrainConfig:
    batch_size = 256             # Samples per batch
    num_epochs = 20              # Training iterations
    num_folds = 5                # K-fold validation
    lr = 3e-4                    # Initial learning rate
    weight_decay = 1e-4
    warmup_epochs = 2            # Warmup duration
    min_lr = 1e-6                # Minimum LR
```

### Loss Function

```python
# CombinedLoss = 0.9 * CrossEntropy + 0.1 * Dice
# Why combination?
# - CrossEntropy: Standard multi-class classification
# - Dice: Handles class imbalance (more "No Flood" than "Extreme")
# - Weight by class: Upweight rare classes

class_weights = compute_class_weight('balanced', classes=[0,1,2,3,4], y=y_train)
# Example: [1.0, 2.5, 2.0, 1.8, 3.0]  (Extreme weighted highest)
```

### K-Fold Training Process

```
K-Fold (5 folds):
├─ Fold 1:
│  ├─ Train on: patches from Q1, Q2, Q4 (folds 0,1,2)
│  ├─ Val on: patches from Q1, Q2, Q4 (fold 3)
│  └─ Best model: saved if val_f1 improves
├─ Fold 2: Different train/val split
├─ Fold 3: Different train/val split
├─ Fold 4: Different train/val split
└─ Fold 5: Different train/val split

Final model: Best fold by macro F1 on validation

Metrics tracked:
├─ train_loss, train_accuracy
├─ val_loss, val_accuracy
├─ val_macro_f1  ← Primary metric
└─ Learning rate (warmup cosine annealing)
```

### Learning Rate Schedule

```python
# Warmup-Cosine Annealing
Schedule:
├─ Epochs 1-2: Linear warmup (0 → 3e-4)
├─ Epochs 3-20: Cosine decay (3e-4 → 1e-6)

Intuition:
├─ Warmup: Stabilize gradients early
└─ Cosine: Smooth convergence with fine-tuning phase
```

---

## How the Model Learns

### Key Learning Mechanisms

#### 1. Rainfall Pattern Recognition
```
ConditioningEncoder.rain_enc (Conv1d):
├─ Learns filters for:
│  ├─ Rapid rise (early peaks) → detects front-loaded
│  ├─ Plateau (uniform) → detects balanced
│  ├─ Slow rise (late peaks) → detects back-loaded
│  └─ Symmetric (triangular) → detects triangular
└─ Output: Compact 64-dim feature vector per rainfall

Training process:
├─ Loss backprop through Conv1d filters
├─ Filters learn to activate for specific patterns
├─ Model gradually specializes (accuracy ↑)
```

#### 2. Storm Type Differentiation
```
Via 3 redundant signals:

Signal 1: Rainfall shape [0:13]
├─ Different intensities → different Conv1d activations
└─ Model learns to extract discriminative features

Signal 2: One-hot encoding [13:17]
├─ Explicit categorical label [1,0,0,0] vs [0,0,0,1]
└─ Transformer learns attention patterns per type

Signal 3: r parameter [17]
├─ Only non-zero for triangular (0.1-0.9)
└─ Explicit signal for peak timing

Redundancy benefit:
├─ If one signal noisy, others compensate
├─ Robust learning even with distribution shift
└─ Model can use whichever signal is most reliable
```

#### 3. Spatial Flood Dynamics
```
What transformer learns:

For each storm type, transformer learns:
├─ Front-loaded storms:
│  └─ "Intense rainfall early → flood propagates downstream first"
├─ Back-loaded storms:
│  └─ "Intense rainfall late → flooding accumulates in low areas"
├─ Triangular storms:
│  └─ "Peak at specific time → spatially symmetric flooding"
└─ Balanced storms:
   └─ "Sustained rainfall → uniform flooding across slopes"

Attention mechanism:
├─ Different spatial regions attended for different storms
├─ Model learns which locations flood under which conditions
└─ Transformer adjusts patch representation based on conditioning
```

#### 4. Depth Scaling
```
Normalized depth [18] learned via:
├─ Linear scaling: deeper storms → higher flood classes
├─ Non-linear interaction: depth × rainfall_shape
├─ Spatial context: elevation × depth

Example:
├─ 10mm + front-loaded + steep slope → light flooding
├─ 40mm + front-loaded + steep slope → moderate flooding
├─ 70mm + front-loaded + steep slope → heavy flooding
```

### Loss Landscape

```
Training loss over epochs (typical):

Epoch 1:   Loss ≈ 1.5  (model guessing)
Epoch 5:   Loss ≈ 1.2  (learning patterns)
Epoch 10:  Loss ≈ 0.9  (storm types differentiated)
Epoch 15:  Loss ≈ 0.8  (spatial dynamics learned)
Epoch 20:  Loss ≈ 0.75 (fine-tuning)

Accuracy over epochs:
Epoch 1:   Acc ≈ 35%  (random for 5 classes ≈ 20%)
Epoch 5:   Acc ≈ 50%
Epoch 10:  Acc ≈ 65%
Epoch 15:  Acc ≈ 72%
Epoch 20:  Acc ≈ 75%

Macro F1 (per-class average):
Follows similar trajectory, more sensitive to class imbalance
```

---

## Inference

### Inference Pipeline

```
User Input:
├─ storm_type: 'triangular'
├─ depth_mm: 40
└─ tpeak: 0.5

Step 1: Validation
├─ Check storm_type in ['front-loaded', 'balanced', 'back-loaded', 'triangular']
├─ Check depth in valid range for that type
├─ Check tpeak valid (required for triangular, ignored else)
└─ Output: Valid ✓ or Error ✗

Step 2: Generate Hyetograph
├─ Call generate_hyetograph(storm_type, depth_mm, tpeak)
├─ Returns 13-step rainfall intensities
│  ├─ Example: [0.016, 0.031, 0.047, ..., 0.0]
│  └─ Sum to total depth magnitude
└─ Output: raw_sequence (13,)

Step 3: Normalize
├─ Normalize: (raw - rain_min) / (rain_max - rain_min)
├─ Using training statistics:
│  ├─ rain_min = 0.003 mm/hr
│  └─ rain_max = 0.486 mm/hr
└─ Output: norm_sequence (13,)

Step 4: Build Conditioning Vector
├─ Rainfall: norm_sequence (13,) ✓
├─ Storm type: get_storm_type_onehot('triangular') = [0,0,0,1] (4,) ✓
├─ r parameter: 0.5 (1,) ✓
├─ Depth norm: (40 - 5) / (78 - 5) = 0.479 (1,) ✓
└─ Concatenate → conditioning_vector (19,) ✓

Step 5: Broadcast to Test Set
├─ Tile conditioning_vector: (19,) → (21420, 19)
│  (same vector for all test patches)
├─ Spatial patches: Load from X_test (21420, 4, 4, 3)
└─ Output: (conditioning, spatial) pairs (21420 each)

Step 6: Batch Inference
├─ For batch in DataLoader(21420 patches, batch_size=512):
│  ├─ spatial_batch → (512, 3, 4, 4)
│  ├─ cond_batch → (512, 19)
│  ├─ logits = model(spatial_batch, cond_batch) → (512, 5)
│  ├─ preds = argmax(logits) → (512,)  [class 0-4]
│  └─ Append to predictions
└─ Output: predictions (21420,)

Step 7: Statistics & Summary
├─ Reshape: (21420,) → (153, 140) patch grid
├─ Count classes: how many patches are class 0/1/2/3/4
├─ Percentages: (count / total) × 100
└─ Output:
   └─ No Flood: 23.5%
   └─ Light: 15.2%
   └─ Moderate: 21.4%
   └─ Heavy: 25.6%
   └─ Extreme: 14.3%
```

### Example: Single Prediction

```python
from inference import FloodInference

# Initialize
engine = FloodInference(
    model=model,
    device=device,
    rain_min=0.003,
    rain_max=0.486,
    X_test=dataset['X_test'],  # (21420, 4, 4, 3)
    batch_size=512,
)

# Predict
result = engine.predict(
    storm_type='triangular',
    depth_mm=40,
    tpeak=0.5,
)

# Results
print(result['statistics'])
# {
#   'No Flood': {'count': 5021, 'percentage': 23.5},
#   'Light': {'count': 3246, 'percentage': 15.2},
#   'Moderate': {'count': 4567, 'percentage': 21.4},
#   'Heavy': {'count': 5467, 'percentage': 25.6},
#   'Extreme': {'count': 3055, 'percentage': 14.3},
# }

print(result['warnings'])
# []  (no warnings, input valid)

print(result['config'])
# {
#   'storm_type': 'triangular',
#   'depth_mm': 40,
#   'tpeak': 0.5,
# }
```

### Example: Batch Predictions

```python
test_configs = [
    {'storm_type': 'triangular', 'depth_mm': 9, 'tpeak': 0.5},
    {'storm_type': 'front-loaded', 'depth_mm': 9, 'tpeak': None},
    {'storm_type': 'back-loaded', 'depth_mm': 10, 'tpeak': None},
    {'storm_type': 'balanced', 'depth_mm': 40, 'tpeak': None},
]

results = engine.predict_batch(test_configs)

# Process results
for i, result in enumerate(results):
    if 'error' in result:
        print(f"Test {i+1}: ERROR - {result['error']}")
    else:
        config = result['config']
        stats = result['statistics']
        print(f"Test {i+1}: {config['storm_type']} @ {config['depth_mm']}mm")
        print(f"  Extreme: {stats['Extreme']['percentage']:.1f}%")
        print(f"  Heavy:   {stats['Heavy']['percentage']:.1f}%")
```

---

## Usage Examples

### Full Training & Inference Pipeline

```python
# ============================================================
# SETUP
# ============================================================
import torch
from config import Config, ModelConfig, TrainConfig
from data import load_and_prepare_data
from training import FloodPatchDataset, train_kfold
from inference import FloodInference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = Config()

# ============================================================
# DATA PREPARATION
# ============================================================
# Load spatial data (DEM, infiltration, landuse)
dem, infilt, landuse = load_spatial_data()

# Normalize
dem_norm = normalize_dem(dem)
infilt_norm = normalize_infiltration(infilt)
landuse_norm = normalize_landuse(landuse)

# Load rainfall scenarios
rainfall_scenarios = load_scenarios('info.csv')
all_intensities = np.concatenate([df['intensity_mmhr'].values for df in rainfall_scenarios])
rain_min, rain_max = all_intensities.min(), all_intensities.max()

# Stack spatial inputs
spatial_stack = stack_spatial_inputs(dem_norm, infilt_norm, landuse_norm)

# Extract patches & create splits
spatial_patches = extract_spatial_patches(spatial_stack, patch_size=4)
quadrant_indices = get_quadrant_patch_indices(*spatial_stack.shape[:2], patch_size=4)

# Categorize flood maps
flood_maps_cat, metadata = categorize_all_flood_maps_patch(
    flood_maps, patch_size=4, categorization_method='majority'
)

# Build train/test split
dataset = build_train_test_split(
    spatial_patches, flood_maps_cat, quadrant_indices
)

# Build conditioning vectors
scenario_metadata = load_scenario_metadata('info.csv')
rain_embeddings = normalize_rainfall_scenarios(rainfall_scenarios, rain_min, rain_max)
conditioning_vectors = build_all_conditioning_vectors(
    rain_embeddings, scenario_metadata, depth_min=5.0, depth_max=78.0
)

# ============================================================
# TRAINING
# ============================================================
train_dataset = FloodPatchDataset(
    X_spatial=dataset['X_train'],
    y_labels=dataset['y_train'],
    rainfall=conditioning_vectors,
)

fold_histories, best_fold = train_kfold(
    model_factory=model_factory(cfg.model),
    full_dataset=train_dataset,
    y_train=dataset['y_train'],
    cfg=cfg.train,
    device=device,
)

# Load best model
ckpt = torch.load(f'checkpoints/fold_{best_fold+1}/best_model.pt')
model = make_model(cfg.model)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device).eval()

# ============================================================
# INFERENCE
# ============================================================
engine = FloodInference(
    model=model,
    device=device,
    rain_min=rain_min,
    rain_max=rain_max,
    X_test=dataset['X_test'],
    batch_size=cfg.train.batch_size,
)

# Single prediction
result = engine.predict('triangular', depth_mm=40, tpeak=0.5)
print(result['statistics'])

# Batch predictions
test_configs = [
    {'storm_type': 'triangular', 'depth_mm': 9, 'tpeak': 0.5},
    {'storm_type': 'front-loaded', 'depth_mm': 40, 'tpeak': None},
    {'storm_type': 'back-loaded', 'depth_mm': 30, 'tpeak': None},
    {'storm_type': 'balanced', 'depth_mm': 50, 'tpeak': None},
]

results = engine.predict_batch(test_configs)
for i, r in enumerate(results):
    if 'error' not in r:
        print(f"Config {i+1}: Extreme = {r['statistics']['Extreme']['percentage']:.1f}%")
```

---

## Key Components Reference

### Files & Functions

| File | Key Functions | Purpose |
|------|---------------|---------|
| `hyetograph.py` | `generate_hyetograph()`, `build_conditioning_vector()` | Storm scenario generation & conditioning |
| `vit.py` | `ViT`, `ConditioningEncoder`, `TransformerEncoder` | Model architecture |
| `training.py` | `FloodPatchDataset`, `train_kfold()`, `run_epoch()` | Training pipeline |
| `testing.py` | `evaluate_all_scenarios()`, `compute_metrics()` | Evaluation & visualization |
| `inference.py` | `FloodInference`, `validate_storm_input()` | Web/notebook inference |
| `quadrant.py` | `build_train_test_split()`, `split_into_quadrants()` | Spatial data processing |
| `flood_categorization.py` | `categorize_flood_map()`, `categorize_all_flood_maps_patch()` | Flood classification |
| `config.py` | `Config`, `ModelConfig`, `TrainConfig` | Configuration management |

### Class Hierarchy

```
FloodPatchDataset (torch.utils.data.Dataset)
├─ Batches (spatial_patch, rainfall, label)
├─ __len__: N_patches × N_scenarios
└─ __getitem__: returns aligned triplet

FloodInference
├─ __init__: load model, cache tensors
├─ predict: single scenario → predictions & stats
├─ predict_batch: multiple scenarios
└─ _compute_statistics: class distribution

ViT (torch.nn.Module)
├─ PatchEmbedding: spatial → token
├─ ConditioningEncoder: conditioning → token
│  ├─ rain_enc: Conv1d (temporal)
│  ├─ cat_enc: Linear (categorical)
│  └─ fusion: combine
├─ TransformerEncoder: attention layers
└─ ClassificationHead: logits → classes
```

---

## Performance Metrics

### Expected Results (20 epochs, 5-fold CV)

```
Per-Epoch Training (typical):
├─ Epoch 1:   train_loss=1.53, train_acc=0.35
├─ Epoch 10:  train_loss=0.92, train_acc=0.68
└─ Epoch 20:  train_loss=0.75, train_acc=0.75

Validation F1 per fold:
├─ Fold 1:    val_f1=0.68
├─ Fold 2:    val_f1=0.70
├─ Fold 3:    val_f1=0.72 ← Best
├─ Fold 4:    val_f1=0.69
└─ Fold 5:    val_f1=0.71

Final metrics (best fold):
├─ Accuracy:  0.72
├─ Precision: 0.71
├─ Recall:    0.70
├─ F1 macro:  0.72
└─ IoU macro: 0.58
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Model crashes during inference | Input depth out of range | Validate with VALID_RANGES dict |
| Predictions all "No Flood" | Model not trained enough | Increase num_epochs, check loss curve |
| High training loss | Poor hyperparameters | Reduce batch_size, increase lr, add warmup |
| Memory errors | Batch size too large | Reduce batch_size from 1024 → 512 |
| Poor generalization | Insufficient epochs/folds | Increase num_epochs to 20+, num_folds to 5 |

### Validation Ranges (Training Data)

```python
VALID_RANGES = {
    'front-loaded': {'depth_mm': (6, 78),   'tpeak': None},
    'balanced':     {'depth_mm': (19, 78),  'tpeak': None},
    'back-loaded':  {'depth_mm': (7, 75),   'tpeak': None},
    'triangular':   {'depth_mm': (5, 77),   'tpeak': (0.1, 0.9)},
}

# Inputs outside these ranges produce warnings but still predict
# (likely with reduced accuracy)
```

---

## Summary

This project demonstrates end-to-end deep learning for spatial flood prediction:

1. **Data**: Spatial features (DEM, infiltration, landuse) + 50 rainfall scenarios
2. **Preprocessing**: Normalization, patch extraction, train/test split
3. **Conditioning**: 19-dim vectors encoding storm characteristics
4. **Model**: Vision Transformer learning spatial-rainfall interactions
5. **Training**: K-fold CV with combined loss, warmup-cosine schedule
6. **Inference**: Flexible prediction for any valid storm configuration
7. **Output**: Per-patch flood class predictions + aggregate statistics

The model learns to differentiate storm types through **rainfall patterns + explicit type encoding + peak timing**, predicting spatially-varying floods accordingly.

---

## References

- Vision Transformer (ViT): Dosovitskiy et al., 2021
- Flood classification: Based on depth thresholds from hydrology literature
- SCS hyetograph: NRCS Type II, Type I, custom triangular
- K-fold validation: Standard cross-validation practice

---

**Last Updated:** 2026-02-26
**Author:** Mikos
**License:** MIT