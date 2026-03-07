# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from config import TrainConfig


class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, smooth: float = 1.0,
                 ce_weight: float = 0.9, dice_weight: float = 0.1):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss(weight=class_weights)
        self.smooth      = smooth
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred         = torch.softmax(pred, dim=1)
        target_oh    = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).float()
        intersection = (pred * target_oh).sum(dim=0)
        union        = pred.sum(dim=0) + target_oh.sum(dim=0)
        dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice_loss(pred, target)


def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.FloatTensor(weights)


def create_optimizer(model: nn.Module, cfg: TrainConfig) -> optim.Optimizer:
    return optim.AdamW(model.parameters(),
                       lr=cfg.lr,
                       weight_decay=cfg.weight_decay,
                       betas=cfg.betas)


def clip_gradients(model: nn.Module, max_norm: float):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class WarmupCosineScheduler:
    def __init__(self, optimizer: optim.Optimizer, cfg: TrainConfig):
        self.optimizer     = optimizer
        self.warmup_epochs = cfg.warmup_epochs
        self.total_epochs  = cfg.num_epochs
        self.base_lr       = cfg.lr
        self.min_lr        = cfg.min_lr
        self.current_epoch = 0

    def step(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / max(self.warmup_epochs, 1))
        else:
            progress = ((self.current_epoch - self.warmup_epochs) /
                        max(self.total_epochs - self.warmup_epochs, 1))
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.current_epoch += 1
        return lr


def create_kfold_splits(scenario_ids: List[int], num_folds: int) -> List[Tuple[List[int], List[int]]]:
    rng          = np.random.default_rng(42)
    shuffled_ids = rng.permutation(scenario_ids)
    fold_size    = len(shuffled_ids) // num_folds
    folds        = []

    for i in range(num_folds):
        val_start  = i * fold_size
        val_end    = val_start + fold_size if i < num_folds - 1 else len(shuffled_ids)
        val_scen   = shuffled_ids[val_start:val_end].tolist()
        train_scen = np.concatenate([shuffled_ids[:val_start], shuffled_ids[val_end:]]).tolist()
        folds.append((train_scen, val_scen))

    return folds


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
                cfg: TrainConfig, device: str) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for spatial, conditioning, labels in loader:
        spatial      = spatial.to(device)
        conditioning = conditioning.to(device)
        labels       = labels.to(device)

        logits, _  = model(spatial, conditioning)
        loss       = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model, cfg.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        pred        = logits.argmax(dim=1)
        correct    += (pred == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def validate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: str) -> Tuple[float, float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels      = [], []

    with torch.no_grad():
        for spatial, conditioning, labels in loader:
            spatial      = spatial.to(device)
            conditioning = conditioning.to(device)
            labels       = labels.to(device)

            logits, _  = model(spatial, conditioning)
            loss       = criterion(logits, labels)

            total_loss += loss.item()
            pred        = logits.argmax(dim=1)
            correct    += (pred == labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), correct / total, macro_f1


def train_one_fold(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   fold_idx: int, cfg: TrainConfig, device: str) -> Tuple[Dict, float]:
    print(f"\n{'='*70}\nFOLD {fold_idx + 1}\n{'='*70}")

    # Compute class weights from this fold's training labels
    train_dataset = train_loader.dataset
    if hasattr(train_dataset, 'dataset'):
        base_dataset = train_dataset.dataset
        train_labels = np.array([base_dataset[i][2].item() for i in train_dataset.indices])
    else:
        train_labels = np.array([train_dataset[i][2].item() for i in range(len(train_dataset))])

    class_weights = calculate_class_weights(train_labels).to(device)

    print("\nClass distribution:")
    for i in range(len(class_weights)):
        count = (train_labels == i).sum()
        print(f"  Class {i}: {count:>6,} ({100*count/len(train_labels):>5.2f}%)  weight={class_weights[i]:.3f}")

    criterion     = CombinedLoss(class_weights=class_weights)
    optimizer     = create_optimizer(model, cfg)
    scheduler     = WarmupCosineScheduler(optimizer, cfg)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
               'val_acc': [], 'val_macro_f1': [], 'lr': []}
    best_macro_f1    = -1.0
    best_model_state = None

    for epoch in range(cfg.num_epochs):
        lr = scheduler.step()

        train_loss, train_acc           = train_epoch(model, train_loader, criterion, optimizer, cfg, device)
        val_loss, val_acc, val_macro_f1 = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_macro_f1)
        history['lr'].append(lr)

        print(f"Epoch {epoch+1:3d}/{cfg.num_epochs} | LR:{lr:.6f} | "
              f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} | "
              f"Val F1:{val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1    = val_macro_f1
            best_model_state = model.state_dict().copy()
            print(f"  → Best Macro F1: {best_macro_f1:.4f}")

    model.load_state_dict(best_model_state)
    return history, best_macro_f1

def train_kfold(model, full_dataset, scenario_to_samples: Dict[int, List[int]],
                cfg: TrainConfig, device: str, checkpoint_dir: Path = None) -> Tuple[List[Dict], int]:
    
    checkpoint_dir = Path(checkpoint_dir)
    
    scenario_ids = list(scenario_to_samples.keys())
    folds        = create_kfold_splits(scenario_ids, cfg.num_folds)

    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION")
    print("="*70)
    print(f"Checkpoint directory: {checkpoint_dir}")
    for i, (train_scen, val_scen) in enumerate(folds):
        print(f"Fold {i+1}: {len(train_scen)} train scenarios, {len(val_scen)} val scenarios")

    fold_histories: List[Dict]  = []
    fold_best_f1s:  List[float] = []

    for fold_idx, (train_scenarios, val_scenarios) in enumerate(folds):
        train_indices = [idx for sid in train_scenarios for idx in scenario_to_samples[sid]]
        val_indices   = [idx for sid in val_scenarios   for idx in scenario_to_samples[sid]]

        print(f"\nFold {fold_idx+1}: {len(train_indices):,} train, {len(val_indices):,} val samples")

        train_loader = DataLoader(Subset(full_dataset, train_indices),
                                  batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(Subset(full_dataset, val_indices),
                                  batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        fold_model       = model().to(device)
        history, best_f1 = train_one_fold(fold_model, train_loader, val_loader, fold_idx, cfg, device)

        fold_histories.append(history)
        fold_best_f1s.append(best_f1)

        # Save checkpoint to specified directory
        save_dir = checkpoint_dir / f'fold_{fold_idx+1}'
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': fold_model.state_dict(),
                    'best_macro_f1':    best_f1,
                    'fold':             fold_idx + 1}, save_dir / 'best_model.pt')
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    best_fold = int(np.argmax(fold_best_f1s))

    print("\n" + "="*70)
    print("K-FOLD SUMMARY")
    print("="*70)
    for i, f1 in enumerate(fold_best_f1s):
        marker = " ← BEST" if i == best_fold else ""
        print(f"  Fold {i+1}: {f1:.4f}{marker}")
    print(f"\n  Average: {np.mean(fold_best_f1s):.4f} ± {np.std(fold_best_f1s):.4f}")
    print(f"\n  Checkpoints saved to: {checkpoint_dir}")
    print("="*70)

    return fold_histories, best_fold