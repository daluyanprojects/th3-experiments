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

class CombinedLoss(nn.Module):    
    def __init__(self, class_weights):
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
        return 0.9 * ce_loss + 0.1 * dice_loss

def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.FloatTensor(weights)

def create_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05, betas=(0.9, 0.999))


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = 1e-6
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


class EarlyStopping:
    def __init__(self, patience: int):
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


def clip_gradients(model: nn.Module, max_norm: float):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def create_kfold_splits(scenario_ids: List[int], num_folds: int) -> List[Tuple[List[int], List[int]]]:
    np.random.seed(42)
    shuffled_ids = np.array(scenario_ids.copy())
    np.random.shuffle(shuffled_ids)
    
    folds = []
    fold_size = len(shuffled_ids) // num_folds
    
    for i in range(num_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < num_folds - 1 else len(shuffled_ids)
        val_scenarios = shuffled_ids[val_start:val_end].tolist()
        train_scenarios = np.concatenate([shuffled_ids[:val_start], shuffled_ids[val_end:]]).tolist()
        folds.append((train_scenarios, val_scenarios))
    
    return folds


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for spatial, rainfall, labels in loader:
        spatial = spatial.to(device)
        rainfall = rainfall.to(device)
        labels = labels.to(device)
        
        logits, _ = model(spatial, rainfall)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model, max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for spatial, rainfall, labels in loader:
            spatial = spatial.to(device)
            rainfall = rainfall.to(device)
            labels = labels.to(device)
            
            logits, _ = model(spatial, rainfall)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def train_one_fold(model, train_loader, val_loader, fold_idx: int, num_epochs: int, device: str):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}/3")
    print(f"{'='*70}")
    
    # Get class weights
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'dataset'):
        base_dataset = train_dataset.dataset
        indices = train_dataset.indices
        train_labels = np.array([base_dataset[i][2].item() for i in indices])
    else:
        train_labels = np.array([train_dataset[i][2].item() for i in range(len(train_dataset))])
    
    class_weights = calculate_class_weights(train_labels).to(device)
    
    print(f"\nClass distribution:")
    for i in range(5):
        count = (train_labels == i).sum()
        pct = 100 * count / len(train_labels)
        print(f"  Class {i}: {count:>6,} ({pct:>5.2f}%) weight={class_weights[i]:.3f}")
    
    # Setup
    criterion = CombinedLoss(class_weights=class_weights)
    optimizer = create_optimizer(model)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=num_epochs)
    early_stop = EarlyStopping(patience=20)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    # Training loop
    for epoch in range(num_epochs):
        lr = scheduler.step()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(lr)
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | LR:{lr:.6f} | "
              f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  → Best!")
        
        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    return history, best_val_loss


def train_kfold(model_factory, full_dataset, scenario_to_samples, batch_size=64, num_epochs=100, device='cuda', num_folds=3):
    scenario_ids = list(scenario_to_samples.keys())
    folds = create_kfold_splits(scenario_ids=scenario_ids, num_folds=num_folds)
    
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION")
    print("="*70)
    
    for i, (train_scen, val_scen) in enumerate(folds):
        print(f"\nFold {i+1}: Train={len(train_scen)} scenarios, Val={len(val_scen)}")
        print(f"  Train: {sorted(train_scen)}")
        print(f"  Val: {sorted(val_scen)}")
    
    fold_histories = []
    fold_best_losses = []
    
    for fold_idx, (train_scenarios, val_scenarios) in enumerate(folds):
        train_indices = []
        val_indices = []
        
        for sid in train_scenarios:
            train_indices.extend(scenario_to_samples[sid])
        for sid in val_scenarios:
            val_indices.extend(scenario_to_samples[sid])
        
        print(f"\nFold {fold_idx + 1}: {len(train_indices):,} train, {len(val_indices):,} val")
        
        train_dataset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        model = model_factory().to(device)
        
        history, best_val_loss = train_one_fold(model, train_loader, val_loader, fold_idx, num_epochs, device)
        
        fold_histories.append(history)
        fold_best_losses.append(best_val_loss)
        
        save_dir = Path(f'./checkpoints/fold_{fold_idx + 1}')
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / 'best_model.pt')
        
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    best_fold = np.argmin(fold_best_losses)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for i, loss in enumerate(fold_best_losses):
        marker = " ← BEST" if i == best_fold else ""
        print(f"Fold {i+1}: {loss:.4f}{marker}")
    
    print(f"\nAverage: {np.mean(fold_best_losses):.4f} ± {np.std(fold_best_losses):.4f}")
    print("="*70)
    
    return fold_histories, best_fold