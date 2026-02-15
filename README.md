# th3-experiments

'''
## **PROJECT OVERVIEW**

### Study Design
- **Study Area**: Greater Metro Manila (GMM) for training, Metro Manila core for testing
- **Task**: Multi-class flood susceptibility classification (5 classes)
- **Model**: Vision Transformer (ViT) with temporal rainfall encoding
- **Key Innovation**: Spatial generalization test - train on GMM outskirts, test on Manila core

### Data Characteristics
- **Train Set**: 
  - Spatial inputs: (1224, 1125), 1,273,959 valid pixels (GMM - outside Manila box)
  - 20 flood maps with data outside Manila, NoData inside Manila
- **Test Set**: 
  - Spatial inputs: (1224, 1125), 103,041 valid pixels (Manila core only)
  - 20 flood maps with data inside Manila, NoData outside Manila
- **Inverse Mask Pattern**: Train and test are spatially complementary

---
## **PHASE 1: DATA LOADING**

### 1.1 Load Training Data (GMM)
- **Spatial inputs** (constant across scenarios):
  - Load DEM: (1224, 1125), float32, range [-4.9, 466.4]
  - Load Infiltration: (1224, 1125), float64, range [0.0, 3.68]
  - Load Landuse: (1224, 1125), float64, range [1.0, 11.0]
  - Valid pixels: 1,273,959 / 1,377,000 (92.5% coverage - outside Manila)
- **Temporal inputs**:
  - Load 20 rainfall scenario CSVs
  - Each: 13 timesteps with [hour, intensity_mmhr]
- **Ground truth**:
  - Load 20 flood maps: (1224, 1125), continuous depth values
  - Valid region: Outside Manila box (GMM areas)
  - NoData region: Inside Manila box

### 1.2 Load Test Data (Manila Core)
- **Spatial inputs** (same files, different valid region):
  - Load DEM: (1224, 1125), float32, range [-9.1, 46.7]
  - Load Infiltration: (1224, 1125), float64, range [0.0, 2.77]
  - Load Landuse: (1224, 1125), float64, range [1.0, 11.0]
  - Valid pixels: 103,041 / 1,377,000 (7.5% coverage - inside Manila)
- **Temporal inputs**:
  - Load same 20 rainfall scenarios
- **Ground truth**:
  - Load 20 flood maps: (1224, 1125)
  - Valid region: Inside Manila box only
  - NoData region: Outside Manila box

### 1.3 Load Manila Box Mask
- Load mask defining Manila boundary
- Values: 1 = inside Manila, 0 = outside Manila
- Shape: (1224, 1125)
- Critical for spatial split understanding

### 1.4 Verify Data Integrity
- Check all files loaded successfully
- Verify shapes match across inputs
- Confirm inverse mask pattern (train vs test)
- Check rainfall scenario completeness (all 20 scenarios × 13 timesteps)

---

## **PHASE 2: SPATIAL INPUT PREPROCESSING**

### 2.1 Handle NoData Values
- **DEM**:
  - Identify NoData: NaN, -9999, extreme negatives
  - Fill method: mean, median, or nearest neighbor interpolation
  - Preserve spatial mask information
- **Infiltration**:
  - Identify NoData values
  - Fill method: mean or interpolation
  - Ensure non-negative values
- **Landuse**:
  - Identify NoData values
  - Fill method: nearest neighbor interpolation (categorical)
  - Preserve integer class values

### 2.2 Resize to Target Dimension
- **Current**: (1224, 1125) - irregular dimensions
- **Options**:
  - Option A: Pad to (1224, 1224) - maintains resolution
  - Option B: Resize to (1152, 1152) - divisible by 16, 32, 64
  - Option C: Resize to (1024, 1024) - standard, divisible by many patch sizes
  - **Recommended**: Resize to (1152, 1152) for 16×16 patches
- **Method**: Bilinear interpolation for continuous, nearest neighbor for categorical
- Apply same resize to ALL inputs and masks for consistency

### 2.3 Normalize Spatial Inputs
- **DEM**:
  - Method: Min-max normalization to [0, 1]
  - Calculate global min/max from training set
  - Apply same parameters to test set
  - Save normalization parameters for inference
- **Infiltration**:
  - Method: Min-max to [0, 1]
  - Global normalization across train set
- **Landuse**:
  - Method: Min-max to [0, 1] 
  - Consider categorical nature
  - Recommended: Normalized continuous for simplicity

### 2.5 Quality Checks
- Visualize preprocessed rasters
- Check value distributions
- Verify no NaN values remain
- Confirm dimensions: (1152, 1152)

---

## **PHASE 3: RAINFALL TEMPORAL SEQUENCE EXTRACTION**

### 3.1 Extract Rainfall Sequences
- **For each scenario**:
  - Load CSV file
  - Extract intensity_mmhr column
  - Create sequence array: (13,) timesteps
- **Output**: 20 sequences, each (13,)

### 3.2 Normalize Rainfall Sequences
- Calculate global maximum intensity across ALL scenarios
- Normalize each sequence: intensity / max_intensity
- Range: [0, 1]
- Preserve temporal patterns (peak timing, shape, duration)

### 3.3 Rainfall Features (Optional Analysis)
- Extract summary statistics per scenario:
  - Total rainfall (mm)
  - Peak intensity (mm/hr)
  - Peak timing (hour)
  - Mean intensity
  - Duration
  - Intensity gradient
- Use for exploratory data analysis

### 3.4 Verify Temporal Patterns
- Visualize all 20 hyetographs
- Identify early-peaked vs late-peaked events
- Confirm sequence diversity
- Check for data quality issues

---

## **PHASE 4: FLOOD MAP CATEGORIZATION**

### 4.1 Define Flood Susceptibility Classes
- **Class 0 - No Flood**: [0.00, 0.15) meters
- **Class 1 - Light**: [0.15, 0.24) meters
- **Class 2 - Moderate**: [0.24, 0.46) meters
- **Class 3 - Heavy**: [0.46, 0.68) meters
- **Class 4 - Extreme**: [0.68, ∞) meters

### 4.2 Patch-Based Categorization
- **Patch size decision**:
  - Options: 16×16, 32×32, 64×64
  - For (1152, 1152): 16×16 → 5,184 patches, 32×32 → 1,296 patches
  - Recommended: Start with 16×16 for fine detail
- **Categorization methods**:
  - **Majority vote**: Most frequent class in patch (recommended)
  - **Center pixel**: Use center pixel class
  - **Weighted majority**: Prioritize flood classes
- **Process**:
  - Extract patches from continuous flood maps
  - Apply categorization rule per patch
  - Assign single class label (0-4) to each patch
  - Handle NoData patches: label as -1 or exclude

### 4.3 Categorize All 20 Flood Maps
- **Train flood maps** (15 scenarios):
  - Categorize patches in GMM region (outside Manila)
  - Output: 15 arrays of patch labels
  - Each array: (num_valid_patches,) with values 0-4
- **Test flood maps** (5 scenarios):
  - Categorize patches in Manila region (inside Manila box)
  - Output: 5 arrays of patch labels
  - Smaller number of patches (only Manila core)

### 4.4 Class Distribution Analysis
- Calculate class distribution per scenario
- Aggregate distribution across train/test
- Check for class imbalance
- Identify if certain classes are rare
- Plan for weighted loss or oversampling if needed

### 4.5 Visualization
- Visualize categorized maps (reconstructed from patches)
- Compare original continuous vs categorized
- Show class distribution histograms
- Identify spatial patterns in classification

---

## **PHASE 5: SPATIAL DATA SPLITTING**

### 5.1 Understand Spatial Split Logic
- **Train region**: Greater Metro Manila (outside Manila box)
  - Valid pixels in train spatial inputs: ~1.27M
  - Flood maps have data outside Manila, NoData inside
- **Test region**: Metro Manila core (inside Manila box)
  - Valid pixels in test spatial inputs: ~103K
  - Flood maps have data inside Manila, NoData outside
- **Key insight**: Train and test are spatially complementary, NOT overlapping

### 5.2 Identify Spatial Masks
- **Create GMM mask**: Where train data is valid (0 in Manila box mask)
- **Create Manila mask**: Where test data is valid (1 in Manila box mask)
- Verify masks sum correctly
- Check alignment with raster data

### 5.3 Random Scenario Assignment
- **Total scenarios**: 20
- **Training**: 15 scenarios (randomly selected)
- **Testing**: 5 scenarios (remaining)
- Set random seed for reproducibility
- **Note**: All 20 scenarios used for BOTH regions, but:
  - Train scenarios contribute GMM patches
  - Test scenarios contribute Manila patches

### 5.4 Document Split
- Save train scenario IDs
- Save test scenario IDs
- Save spatial mask definitions
- Record split strategy and rationale
- Store in metadata JSON

---

## **PHASE 6: PATCH EXTRACTION FOR ViT**

### 6.1 Determine Patch Configuration
- **Target dimension**: (1152, 1152) after preprocessing
- **Patch size**: 4x4 (recommended for ViT)
- **Patches per dimension**: 1152 / 4 
- **Stride**: 4 (non-overlapping, standard for ViT)

### 6.2 Extract Spatial Patches (Shared)
- **Stack spatial inputs**:
  - Combine DEM + Infiltration + Landuse
  - Shape: (1152, 1152, 3) → (3, 1152, 1152) for PyTorch
- **Extract patches**:
  - Use sliding window or reshape operation
  - Output: (5184, 3, 16, 16)
  - Each patch: 3-channel, 16×16 spatial feature
- **Note**: These patches are SHARED across all scenarios (spatial inputs don't change)

### 6.3 Apply Spatial Masks to Patches
- **Create patch-level masks**:
  - Downsample GMM mask to patch resolution
  - Downsample Manila mask to patch resolution
  - Each patch labeled: inside/outside Manila
- **For training**:
  - Keep only patches where GMM mask == 1 (outside Manila)
  - Results in ~X patches (92.5% of 5184)
- **For testing**:
  - Keep only patches where Manila mask == 1 (inside Manila)
  - Results in ~Y patches (7.5% of 5184)

### 6.4 Organize Rainfall Sequences
- **For each scenario**:
  - Rainfall sequence: (13,) normalized values
  - Repeat for all valid patches in that scenario
- **Training set**:
  - 15 scenarios × num_gmm_patches
  - Each sample: spatial patch + rainfall sequence
- **Test set**:
  - 5 scenarios × num_manila_patches
  - Each sample: spatial patch + rainfall sequence

### 6.5 Organize Ground Truth Labels
- **Training labels**:
  - 15 categorized flood maps
  - Extract only GMM patches (outside Manila)
  - Each patch label: 0-4
- **Test labels**:
  - 5 categorized flood maps
  - Extract only Manila patches (inside Manila)
  - Each patch label: 0-4

---

## **PHASE 7: CREATE COMPLETE DATASET**

### 7.1 Training Dataset Structure
- **Spatial patches**: (num_train_scenarios, num_gmm_patches, 3, 16, 16)
  - 15 scenarios × ~4,800 GMM patches = ~72,000 samples
- **Rainfall sequences**: (num_train_scenarios, num_gmm_patches, 13)
  - Each GMM patch paired with its scenario's rainfall
- **Labels**: (num_train_scenarios, num_gmm_patches)
  - Flood class for each GMM patch
- **Metadata**: scenario IDs, patch locations, spatial region

### 7.2 Test Dataset Structure
- **Spatial patches**: (num_test_scenarios, num_manila_patches, 3, 16, 16)
  - 5 scenarios × ~390 Manila patches = ~1,950 samples
- **Rainfall sequences**: (num_test_scenarios, num_manila_patches, 13)
  - Each Manila patch paired with its scenario's rainfall
- **Labels**: (num_test_scenarios, num_manila_patches)
  - Flood class for each Manila patch
- **Metadata**: scenario IDs, patch locations, spatial region

### 7.3 Flatten for DataLoader
- **Training**:
  - Flatten to (72000, 3, 16, 16) spatial
  - Flatten to (72000, 13) rainfall
  - Flatten to (72000,) labels
- **Testing**:
  - Flatten to (1950, 3, 16, 16) spatial
  - Flatten to (1950, 13) rainfall
  - Flatten to (1950,) labels

### 7.4 Data Validation
- Check shapes are consistent
- Verify no NaN values in inputs
- Confirm label range: 0-4
- Check class distribution
- Validate spatial-temporal alignment

### 7.5 Save Processed Dataset
- Save as NPY files or HDF5 for fast loading
- Include metadata JSON
- Document preprocessing steps
- Store normalization parameters

---

## **PHASE 8: PYTORCH DATASET & DATALOADER**

### 8.1 Create PyTorch Dataset Class
- **Inputs per sample**:
  - Spatial patch: (3, 16, 16) tensor
  - Rainfall sequence: (13,) tensor
  - Label: scalar (0-4)
- **Implement**:
  - `__init__`: Load flattened data
  - `__len__`: Return total samples
  - `__getitem__`: Return (spatial, rainfall, label) tuple
- **Optional**: Add data augmentation transforms

### 8.2 Data Augmentation (Optional)
- **Spatial augmentations**:
  - Random horizontal/vertical flip
  - Random rotation (90°, 180°, 270°)
  - Random brightness/contrast adjustment
- **Constraints**:
  - DO NOT augment rainfall (must stay realistic)
  - DO NOT augment labels
  - Apply consistently to spatial patches only

### 8.3 Create DataLoaders
- **Training DataLoader**:
  - Batch size: 32-128 (depending on GPU memory)
  - Shuffle: True
  - Num workers: 4-8 (parallel data loading)
  - Pin memory: True (faster GPU transfer)
  - Drop last: True (consistent batch sizes)
- **Test DataLoader**:
  - Batch size: Same as training
  - Shuffle: False (preserve order for reconstruction)
  - Num workers: 4-8
  - Pin memory: True

### 8.4 Test Batch Loading
- Iterate one batch from train loader
- Verify shapes: (batch, 3, 16, 16), (batch, 13), (batch,)
- Check data types: float32, long
- Confirm values in expected ranges
- Test on both CPU and GPU

---

## **PHASE 9: VIT ARCHITECTURE DESIGN**

### 9.1 Model Components

#### A. Spatial Patch Embedding
- **Input**: (batch, 3, 16, 16) spatial patch
- **Method**: Conv2d projection
  - Kernel size: 16×16
  - Stride: 16
  - Output channels: embed_dim (e.g., 256, 512)
- **Output**: (batch, 1, embed_dim) - single patch token

#### B. Rainfall Sequence Embedding
- **Input**: (batch, 13) temporal sequence
- **Methods** (choose one):
  - **1D Convolution** (recommended):
    - Conv1d layers to capture temporal patterns
    - Global pooling
    - Linear projection to embed_dim
  - **MLP**:
    - Flatten sequence
    - Linear(13 → hidden → embed_dim)
  - **Temporal Attention**:
    - Self-attention over 13 timesteps
    - Pool and project
- **Output**: (batch, 1, embed_dim) - single rainfall token

#### C. Token Combination
- **Concatenate**: [rainfall_token, patch_token]
- **Shape**: (batch, 2, embed_dim)
- **Interpretation**: 2-token sequence per sample

#### D. Positional Encoding
- **Method**: Learnable or sinusoidal
- **Positions**: 2 (one for rainfall, one for patch)
- **Add to tokens**: tokens + positional_embeddings

#### E. Transformer Encoder
- **Layers**: 6-12 transformer blocks
- **Each block contains**:
  - Multi-head self-attention (8-16 heads)
  - Layer normalization
  - Feed-forward network (MLP)
  - Residual connections
  - Dropout (0.1-0.25)
- **Output**: (batch, 2, embed_dim) encoded tokens

#### F. Classification Head
- **Pooling**: Mean pool over tokens OR use first token
- **MLP**: Linear → GELU → Dropout → Linear
- **Output**: (batch, 5) class logits

### 9.2 Model Configurations

#### Tiny (Debug/Baseline)
- Embed dim: 128
- Layers: 4
- Heads: 4
- Parameters: ~500K

#### Small (Recommended Start)
- Embed dim: 256
- Layers: 6
- Heads: 8
- Parameters: ~3M

#### Medium
- Embed dim: 384
- Layers: 8
- Heads: 8
- Parameters: ~8M

#### Base (If needed)
- Embed dim: 512
- Layers: 12
- Heads: 8
- Parameters: ~20M

### 9.3 Model Initialization
- Xavier/Kaiming initialization for weights
- Zero initialization for biases
- Truncated normal for embeddings

### 9.4 Test Forward Pass
- Create dummy inputs
- Run forward pass
- Check output shape: (batch, 5)
- Verify gradient flow
- Count parameters

---

## **PHASE 10: TRAINING SETUP**

### 10.1 Loss Function
- **Base**: Cross-Entropy Loss
- **Class Imbalance Handling**:
  - Calculate class weights from training set
  - Weight = 1 / class_frequency
  - Normalize weights
  - Use weighted CE loss
- **Alternative**: Focal Loss (for severe imbalance)
  - Alpha: class weights
  - Gamma: 2.0 (focus on hard examples)

### 10.2 Optimizer
- **AdamW** (recommended):
  - Learning rate: 1e-4 to 5e-4
  - Weight decay: 0.05-0.1 (regularization)
  - Betas: (0.9, 0.999)
  - Eps: 1e-8

### 10.3 Learning Rate Scheduler
- **Warmup Phase**:
  - Linear warmup: 5-10 epochs
  - Start LR: 1e-6
  - End LR: target LR
- **Main Schedule**:
  - Cosine annealing: smooth decay
  - T_max: total_epochs - warmup
  - Min LR: 1e-6
- **Alternative**: ReduceLROnPlateau
  - Monitor: validation loss
  - Factor: 0.5
  - Patience: 10 epochs

### 10.4 Regularization
- **Dropout**: 0.1-0.25 in transformer and classifier
- **Weight decay**: 0.05-0.1 in optimizer
- **Gradient clipping**: max_norm = 1.0
- **Label smoothing**: 0.1 (optional)
- **Stochastic depth**: 0.1 (optional, advanced)

### 10.5 Training Configuration
- **Epochs**: 100-300
- **Batch size**: 32-128 (GPU dependent)
- **Early stopping**:
  - Patience: 20-30 epochs
  - Monitor: validation accuracy or F1
- **Checkpoint saving**:
  - Save best model (highest val accuracy)
  - Save every N epochs
  - Save optimizer state for resuming
- **Mixed precision**: FP16 training for speed

### 10.6 Evaluation Metrics
- **Primary**: Accuracy, Macro F1-score
- **Per-class**: Precision, Recall, F1 for each class
- **Confusion matrix**: Visualize misclassifications
- **Additional**:
  - IoU per class
  - Weighted F1 (by class frequency)
  - Critical class recall (Heavy, Extreme)

---

## **PHASE 11: TRAINING LOOP**

### 11.1 Training Epoch
- **For each batch**:
  - Load spatial patches, rainfall sequences, labels
  - Move to device (GPU)
  - Forward pass through model
  - Calculate loss
  - Backward pass (compute gradients)
  - Clip gradients (max norm 1.0)
  - Optimizer step
  - Update learning rate
- **Track metrics**:
  - Batch loss
  - Batch accuracy
  - Running averages
- **Logging**: Every N batches

### 11.2 Validation Epoch
- **Set model to eval mode**
- **For each batch**:
  - Load data
  - Forward pass (no grad)
  - Calculate loss
  - Collect predictions and labels
- **Calculate metrics**:
  - Validation loss
  - Accuracy
  - Per-class precision/recall/F1
  - Confusion matrix
- **Return**: Aggregated metrics

### 11.3 Main Training Loop
- **For each epoch**:
  - Run training epoch
  - Run validation epoch
  - Log metrics (console, tensorboard, wandb)
  - Update learning rate scheduler
  - Check early stopping condition
  - Save checkpoint if best model
  - Save periodic checkpoints
- **Monitor**:
  - Train/val loss curves
  - Train/val accuracy curves
  - Learning rate schedule
  - Class-wise performance

### 11.4 Monitoring & Logging
- **TensorBoard/Weights & Biases**:
  - Loss curves
  - Accuracy curves
  - Learning rate
  - Gradient norms
  - Per-class metrics
- **Console logging**:
  - Epoch summary
  - Best model updates
  - Time per epoch
- **CSV logs**: Metrics per epoch for analysis

### 11.5 Checkpointing
- **Save best model**:
  - Monitor: validation accuracy
  - Save: model state, optimizer state, epoch, metrics
  - Filename: `best_model.pth`
- **Save periodic**:
  - Every 10-20 epochs
  - Filename: `checkpoint_epoch_{N}.pth`
- **Save final**:
  - At end of training
  - Filename: `final_model.pth`

---

## **PHASE 12: TESTING & EVALUATION**

### 12.1 Load Best Model
- Load checkpoint with highest validation accuracy
- Move model to eval mode
- Move to appropriate device

### 12.2 Test Set Prediction
- **For each batch in test loader**:
  - Load spatial patches, rainfall, labels
  - Forward pass (no grad)
  - Collect predictions
  - Collect true labels
  - Store probabilities (for confidence analysis)

### 12.3 Comprehensive Metrics
- **Overall**:
  - Test accuracy
  - Macro-averaged F1
  - Weighted F1
- **Per-class**:
  - Precision for each class (0-4)
  - Recall for each class
  - F1-score for each class
  - Support (number of samples)
- **Confusion Matrix**:
  - 5×5 matrix
  - Visualize heatmap
  - Identify common misclassifications
- **Critical Metrics**:
  - Recall for Heavy and Extreme (safety-critical)
  - Precision for No Flood (avoid false alarms)

### 12.4 Spatial Generalization Analysis
- **Question**: How well does GMM training transfer to Manila?
- **Analysis**:
  - Compare train accuracy (GMM) vs test accuracy (Manila)
  - Gap indicates generalization difficulty
  - Per-class transfer: Which classes generalize better?
  - Spatial error patterns: Where does model fail in Manila?

### 12.5 Error Analysis
- **Identify failure cases**:
  - Samples with highest prediction error
  - Consistently misclassified classes
  - Spatial patterns in errors
- **Visualize**:
  - Example patches: true vs predicted
  - Confidence scores for errors
  - Attention maps (if using attention visualization)

### 12.6 Per-Scenario Performance
- **For each test scenario**:
  - Calculate accuracy, F1
  - Reconstruct full prediction map
  - Compare with ground truth
  - Identify scenario-specific patterns
- **Correlation analysis**:
  - Does performance correlate with rainfall characteristics?
  - Early vs late peak scenarios

---

## **PHASE 13: PREDICTION PIPELINE**

### 13.1 Reconstruct Prediction Maps
- **For each test scenario**:
  - Collect all patch predictions
  - Arrange in spatial grid (based on patch locations)
  - Reconstruct full (1152, 1152) map
  - Only fill Manila region (test area)
  - Leave GMM region as NoData

### 13.2 Visualization
- **Create comparison plots**:
  - Ground truth flood map
  - Predicted flood map (categorical)
  - Difference map
- **Color-code classes**:
  - No Flood: White
  - Light: Yellow
  - Moderate: Orange
  - Heavy: Red
  - Extreme: Purple
- **Overlay on DEM**: For spatial context

### 13.3 Confidence Mapping
- **For each patch**:
  - Calculate prediction confidence (max softmax probability)
  - Create confidence map
  - Identify low-confidence regions
- **Use cases**:
  - Flag uncertain predictions
  - Guide data collection
  - Identify out-of-distribution areas

### 13.4 Save Predictions
- **As rasters**:
  - Categorical prediction map (GeoTIFF)
  - Confidence map (GeoTIFF)
  - Preserve geospatial metadata
- **As tables**:
  - CSV with patch locations, predictions, confidence
- **As visualizations**:
  - PNG/PDF comparison figures
  - Interactive HTML maps (optional)

---

## **PHASE 14: INFERENCE PIPELINE (NEW DATA)**

### 14.1 Preprocessing New Data
- **Load new spatial inputs** (DEM, infiltration, landuse)
- Apply same preprocessing:
  - Handle NoData (same methods)
  - Resize to (1152, 1152)
  - Normalize using saved parameters
- **Load new rainfall scenario**:
  - Extract 13-timestep sequence
  - Normalize using saved max_intensity

### 14.2 Patch Extraction
- Extract spatial patches: (5184, 3, 16, 16)
- Repeat rainfall sequence for all patches
- Create dataset with no labels

### 14.3 Prediction
- Load trained model
- Run inference on all patches
- Collect predictions and confidence scores

### 14.4 Post-Processing
- Reconstruct full prediction map
- Apply spatial filters if needed (median filter for smoothing)
- Convert to desired output format
- Save results

### 14.5 Validation (If Ground Truth Available)
- Compare with actual flood data
- Calculate metrics
- Update model if performance degrades

---

## **PHASE 15: MODEL INTERPRETATION & ANALYSIS**

### 15.1 Attention Visualization
- Extract attention weights from transformer layers
- Visualize which tokens attend to each other
- Analyze: Does model focus more on rainfall or spatial features?
- Scenario-specific attention patterns

### 15.2 Feature Importance
- **Spatial features**:
  - Which channel is most important? (DEM vs infiltration vs landuse)
  - Ablation study: Remove one channel, measure performance drop
- **Temporal features**:
  - Which timesteps are most important?
  - Early vs late rainfall impact

### 15.3 Gradient-Based Attribution
- Compute gradients w.r.t. inputs
- Generate saliency maps
- Identify critical spatial regions
- Identify critical temporal phases

### 15.4 Failure Case Analysis
- **Systematic errors**:
  - Does model always overpredict/underpredict certain classes?
  - Spatial bias in errors?
- **Edge cases**:
  - Extreme rainfall values
  - Unusual spatial configurations
  - Boundary regions

### 15.5 Learned Representations
- Extract patch embeddings
- Visualize using t-SNE/UMAP
- Cluster similar patches
- Interpret spatial-temporal relationships

---

## **PHASE 16: MODEL OPTIMIZATION & REFINEMENT**

### 16.1 Hyperparameter Tuning
- **Grid/random search** on:
  - Learning rate
  - Batch size
  - Embed dimension
  - Number of layers
  - Dropout rate
  - Weight decay
- **Use validation set** for selection
- Document best configuration

### 16.2 Architecture Ablations
- **Test variants**:
  - Different rainfall embedding methods (conv vs MLP vs attention)
  - Different patch sizes (16 vs 32 vs 64)
  - Deeper vs wider transformers
  - With/without positional encoding
- **Compare**:
  - Accuracy
  - Training time
  - Inference speed
  - Model size

### 16.3 Ensemble Methods
- **Train multiple models**:
  - Different random seeds
  - Different architectures
  - Different data splits
- **Combine predictions**:
  - Average probabilities
  - Voting
  - Weighted ensemble
- **Expected improvement**: 1-3% accuracy boost

### 16.4 Transfer Learning (Future)
- **Pre-train on related tasks**:
  - Other flood datasets
  - Remote sensing tasks
  - Generic vision tasks
- **Fine-tune on Manila data**
- **Compare**: From-scratch vs transfer learning

---

## **SUCCESS CRITERIA**

### Minimum Viable Product (MVP)
- ✅ Train accuracy > 75%
- ✅ Test accuracy > 60% (given spatial generalization challenge)
- ✅ F1-score > 0.55 (macro-average)
- ✅ Recall for Extreme class > 50%
- ✅ Model trains in < 24 hours
- ✅ Inference time < 1 second per scenario

### Stretch Goals
- 🎯 Test accuracy > 70%
- 🎯 F1-score > 0.65
- 🎯 Recall for Extreme class > 70%
- 🎯 Successful spatial transfer (GMM → Manila)
- 🎯 Interpretable attention patterns
- 🎯 Real-time prediction capability

---

## **RISK MITIGATION**

### Potential Issues & Solutions

#### 1. Severe Class Imbalance
- **Risk**: "No Flood" dominates, model ignores rare classes
- **Solutions**: Weighted loss, focal loss, oversampling, SMOTE

#### 2. Poor Spatial Generalization
- **Risk**: GMM patterns don't transfer to Manila
- **Solutions**: Domain adaptation, semi-supervised learning, transfer learning

#### 3. Overfitting
- **Risk**: High train accuracy, low test accuracy
- **Solutions**: Stronger regularization, data augmentation, simpler model

#### 4. Insufficient Training Data
- **Risk**: 15 scenarios may be too few
- **Solutions**: Data augmentation, synthetic scenarios, transfer learning

#### 5. Computational Constraints
- **Risk**: Model too large for available GPU
- **Solutions**: Smaller model, gradient accumulation, mixed precision

#### 6. NoData Handling Issues
- **Risk**: Improper fill affects predictions
- **Solutions**: Test multiple fill methods, analyze sensitivity

---


'''