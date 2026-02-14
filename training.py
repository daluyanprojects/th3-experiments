import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


# =============================================================================
# 6.1 LOSS FUNCTION: Combined CE + Dice
# =============================================================================

class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice Loss (70% CE + 30% Dice)"""
    
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.smooth = 1.0
    
    def dice_loss(self, pred, target):
        """Dice loss for spatial consistency"""
        pred = torch.softmax(pred, dim=1)
        target_onehot = torch.nn.functional.one_hot(target, num_classes=5).float()
        
        intersection = (pred * target_onehot).sum(dim=0)
        union = pred.sum(dim=0) + target_onehot.sum(dim=0)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return 0.7 * ce_loss + 0.3 * dice_loss


# =============================================================================
# 6.2 CLASS WEIGHTING: Handle Imbalance with SMOTE + Class Weights
# =============================================================================

def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculate balanced class weights using sklearn"""
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.FloatTensor(weights)


def apply_smote_to_fold(dataset, indices, random_state: int = 42):
    """
    Apply SMOTE to balance classes in one fold's training data.
    
    Args:
        dataset: The full dataset
        indices: Training sample indices for this fold
        random_state: Random seed
    
    Returns:
        resampled_data: List of (spatial, rainfall, label) tuples
    """
    print("  Applying SMOTE to balance training classes...")
    
    # Extract data from this fold
    spatial_patches = []
    rainfall_values = []
    labels = []
    
    for idx in indices:
        s, r, l = dataset[idx]
        spatial_patches.append(s.numpy())
        rainfall_values.append(r.numpy())
        labels.append(l.item())
    
    spatial_patches = np.array(spatial_patches)
    rainfall_values = np.array(rainfall_values)
    labels = np.array(labels)
    
    # Flatten spatial patches for SMOTE (batch, 3, 4, 4) -> (batch, 48)
    n_samples = spatial_patches.shape[0]
    spatial_flat = spatial_patches.reshape(n_samples, -1)
    
    # Concatenate spatial + rainfall features
    X = np.concatenate([spatial_flat, rainfall_values], axis=1)  # (batch, 49)
    y = labels
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split back into spatial and rainfall
    spatial_resampled = X_resampled[:, :-1].reshape(-1, 3, 4, 4)
    rainfall_resampled = X_resampled[:, -1:]
    
    print(f"    Original: {len(y):,} samples")
    print(f"    After SMOTE: {len(y_resampled):,} samples")
    
    # Create resampled dataset as list of tuples
    resampled_data = [
        (torch.FloatTensor(s), torch.FloatTensor(r), torch.LongTensor([l])[0])
        for s, r, l in zip(spatial_resampled, rainfall_resampled, y_resampled)
    ]
    
    return resampled_data


# =============================================================================
# 6.3 OPTIMIZER: AdamW
# =============================================================================

def create_optimizer(model: nn.Module) -> optim.Optimizer:
    """Create AdamW optimizer with ViT-optimized settings"""
    return optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )


# =============================================================================
# 6.4 LEARNING RATE SCHEDULE: Warmup + Cosine Annealing
# =============================================================================

class WarmupCosineScheduler:
    """Linear warmup (10%) + Cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = 1e-6
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


# =============================================================================
# 6.5 TRAINING UTILITIES: Early Stopping & Gradient Clipping
# =============================================================================

class EarlyStopping:
    """Stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """Clip gradients to prevent exploding gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


# =============================================================================
# 6.6 K-FOLD CROSS-VALIDATION: 3-Fold on Scenarios
# =============================================================================

def create_kfold_splits(scenario_ids: List[int], num_folds: int) -> List[Tuple[List[int], List[int]]]:
    np.random.seed(42)
    
    # Shuffle the actual scenario IDs
    shuffled_ids = np.array(scenario_ids.copy())
    np.random.shuffle(shuffled_ids)
    
    folds = []
    fold_size = len(shuffled_ids) // num_folds
    
    for i in range(num_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < num_folds - 1 else len(shuffled_ids)
        val_scenarios = shuffled_ids[val_start:val_end].tolist()
        
        train_scenarios = np.concatenate([
            shuffled_ids[:val_start],
            shuffled_ids[val_end:]
        ]).tolist()
        
        folds.append((train_scenarios, val_scenarios))
    
    return folds

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    print(f"    [train_epoch] Starting training epoch...")
    start_time = time.time()
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # from tqdm import tqdm
    
    for batch_idx, (spatial, rainfall, labels) in enumerate(loader):
        if batch_idx == 0:
            print(f"    [train_epoch] Processing first batch...")
        
        spatial = spatial.to(device)
        rainfall = rainfall.to(device)
        labels = labels.to(device)
        
        # Forward
        logits, _ = model(spatial, rainfall)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model, max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 1000 == 0:
            print(f"    [train_epoch] Batch {batch_idx + 1}/{len(loader)} - {time.time() - start_time:.1f}s elapsed")
        
    elapsed = time.time() - start_time
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validate model"""
    print(f"    [validate] Starting validation...")
    start_time = time.time()
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (spatial, rainfall, labels) in enumerate(loader):
            if batch_idx == 0:
                print(f"    [validate] Processing first batch...")
            
            spatial = spatial.to(device)
            rainfall = rainfall.to(device)
            labels = labels.to(device)
            
            logits, _ = model(spatial, rainfall)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    elapsed = time.time() - start_time
    print(f"    [validate] Completed in {elapsed:.1f}s")
    return total_loss / len(loader), correct / total


def train_one_fold(model, train_loader, val_loader, fold_idx: int, num_epochs: int = 100, device: str = 'cuda'):  
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}/3")
    print(f"{'='*70}")
    
    # Get class weights efficiently from the dataset (not by iterating loader!)
    print(f"[train_one_fold] Extracting labels from dataset...")
    start_time = time.time()
    
    train_dataset = train_loader.dataset
    
    # Extract labels efficiently
    if hasattr(train_dataset, 'dataset'):
        # It's a Subset - get the underlying dataset
        base_dataset = train_dataset.dataset
        indices = train_dataset.indices
        print(f"[train_one_fold] Extracting {len(indices)} labels from Subset...")
        train_labels = np.array([base_dataset[i][2].item() for i in indices])
    else:
        # It's the full dataset
        print(f"[train_one_fold] Extracting {len(train_dataset)} labels from full dataset...")
        train_labels = np.array([train_dataset[i][2].item() for i in range(len(train_dataset))])
    
    print(f"[train_one_fold] Labels extracted in {time.time() - start_time:.1f}s")
    
    class_weights = calculate_class_weights(train_labels).to(device)
    print(f"\nClass distribution:")
    for i in range(5):
        count = (train_labels == i).sum()
        pct = 100 * count / len(train_labels)
        print(f"  Class {i}: {count:>6,} samples ({pct:>5.2f}%) - weight: {class_weights[i]:.3f}")
    
    # Setup training
    print(f"[train_one_fold] Setting up training components...")
    criterion = CombinedLoss(class_weights=class_weights)
    optimizer = create_optimizer(model)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=num_epochs)
    early_stop = EarlyStopping(patience=20)
    print(f"[train_one_fold] Training setup complete!")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    print(f"[train_one_fold] Starting training loop...\n")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"[Epoch {epoch+1}/{num_epochs}] Starting...")
        
        lr = scheduler.step()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"LR: {lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  → New best! Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if early_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\nFold {fold_idx + 1} complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final val acc: {max(history['val_acc']):.4f}")
    
    return history, best_val_loss


# =============================================================================
# MAIN K-FOLD TRAINING FUNCTION
# =============================================================================

def train_kfold(model_factory,full_dataset, scenario_to_samples: Dict[int, List[int]], batch_size: int, num_epochs: int, device: str, use_smote: bool):
    
    # Get the actual scenario IDs from the mapping
    scenario_ids = list(scenario_to_samples.keys())
    # Create 3-fold splits using actual scenario IDs
    folds = create_kfold_splits(scenario_ids=scenario_ids, num_folds=3)
    
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SETUP")
    print("="*70)
    print(f"SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    for i, (train_scen, val_scen) in enumerate(folds):
        print(f"\nFold {i+1}: Train on {len(train_scen)} scenarios, Validate on {len(val_scen)} scenarios")
        print(f"  Train: {sorted(train_scen)}")
        print(f"  Val:   {sorted(val_scen)}")
    
    fold_histories = []
    fold_best_losses = []
    
    for fold_idx, (train_scenarios, val_scenarios) in enumerate(folds):
        # Get sample indices for this fold
        train_indices = []
        val_indices = []
        
        for scenario_id in train_scenarios:
            train_indices.extend(scenario_to_samples[scenario_id])
        for scenario_id in val_scenarios:
            val_indices.extend(scenario_to_samples[scenario_id])
        
        print(f"\nFold {fold_idx + 1}: {len(train_indices):,} train samples, {len(val_indices):,} val samples")
        
        # Apply SMOTE if requested
        if use_smote:
            from torch.utils.data import TensorDataset
            
            # Apply SMOTE to training data
            resampled_data = apply_smote_to_fold(full_dataset, train_indices)
            
            # Create new dataset from resampled data
            spatial_list = [item[0] for item in resampled_data]
            rainfall_list = [item[1] for item in resampled_data]
            label_list = [item[2] for item in resampled_data]
            
            train_dataset = TensorDataset(
                torch.stack(spatial_list),
                torch.stack(rainfall_list),
                torch.stack(label_list)
            )
            
            val_subset = Subset(full_dataset, val_indices)
        else:
            # Use original data without SMOTE
            train_dataset = Subset(full_dataset, train_indices)
            val_subset = Subset(full_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        
        # Create fresh model for this fold
        model = model_factory().to(device)
        
        # Train this fold
        history, best_val_loss = train_one_fold(model, train_loader, val_loader, fold_idx, num_epochs, device)
        
        fold_histories.append(history)
        fold_best_losses.append(best_val_loss)
        
        # Save fold results
        save_dir = Path(f'./checkpoints/fold_{fold_idx + 1}')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(model.state_dict(), save_dir / 'best_model.pt')
        
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # Summary
    best_fold = np.argmin(fold_best_losses)
    
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*70)
    for i, loss in enumerate(fold_best_losses):
        marker = " ← BEST" if i == best_fold else ""
        print(f"Fold {i+1}: Best Val Loss = {loss:.4f}{marker}")
    
    avg_loss = np.mean(fold_best_losses)
    std_loss = np.std(fold_best_losses)
    print(f"\nAverage: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"Best Fold: {best_fold + 1}")
    print("="*70)
    
    return fold_histories, best_fold
