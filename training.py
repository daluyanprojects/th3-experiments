import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import copy

class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']


class FloodPatchDataset(Dataset):
    def __init__(self, X_spatial: np.ndarray, y_labels: np.ndarray, rainfall: np.ndarray):
        self.N_patches   = X_spatial.shape[0]
        self.N_scenarios = rainfall.shape[0]

        # HWC → CHW for PyTorch
        self.X    = torch.tensor(X_spatial, dtype=torch.float32).permute(0, 3, 1, 2)
        # (N_scenarios, N_patches) → (N_patches, N_scenarios) for easy row indexing
        self.y    = torch.tensor(y_labels, dtype=torch.long).T
        self.rain = torch.tensor(rainfall, dtype=torch.float32)

        print(f"[FloodPatchDataset]")
        print(f"  Spatial tokens : {self.X.shape}  (N_patches, C, H, W)")
        print(f"  Labels         : {self.y.shape}  (N_patches, N_scenarios)")
        print(f"  Rainfall       : {self.rain.shape}  (N_scenarios, 26)")
        print(f"  Total samples  : {len(self):,}  (patches × scenarios)")

    def __len__(self):
        return self.N_patches * self.N_scenarios

    def __getitem__(self, idx):
        patch_idx    = idx // self.N_scenarios
        scenario_idx = idx  % self.N_scenarios

        spatial  = self.X[patch_idx]                   
        rainfall = self.rain[scenario_idx]            
        label    = self.y[patch_idx, scenario_idx]   

        return spatial, rainfall, label


def compute_class_weights(y_train: np.ndarray, num_classes: int = 5) -> torch.Tensor:

    flat = y_train.flatten()
    flat = flat[flat >= 0]  

    weights = np.zeros(num_classes, dtype=np.float32)
    total = len(flat)
    for c in range(num_classes):
        count = (flat == c).sum()
        weights[c] = total / (num_classes * count) if count > 0 else 0.0

    weights = weights / weights.sum() * num_classes  

    print("Class weights (inverse frequency):")
    for i, (name, w) in enumerate(zip(class_names, weights)):
        count = (flat == i).sum()
        pct   = count / total * 100
        print(f"  Class {i} ({name:10s}): {pct:5.2f}% of patches  →  weight {w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Optional[torch.optim.Optimizer], device: torch.device, is_train: bool = True) -> Tuple[float, float]:
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    total_batches = len(loader)
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (spatial, rainfall, labels) in enumerate(loader):
            spatial  = spatial.to(device)
            rainfall = rainfall.to(device)
            labels   = labels.to(device)

            logits, _ = model(spatial, rainfall)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += labels.size(0)

            if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == total_batches:
                phase = "TRAIN" if is_train else "VAL  "
                print(f"    [{phase}] batch {batch_idx+1:>5}/{total_batches}  "
                      f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    return total_loss / total, correct / total

def train_kfold(model_factory,full_dataset: FloodPatchDataset, y_train: np.ndarray, rain_normalized: np.ndarray, num_folds: int, 
                batch_size: int, num_epochs: int, lr: float, device: torch.device = torch.device('cpu'), num_classes: int = 5) -> Tuple[List[Dict], int]:

    N_patches   = full_dataset.N_patches
    N_scenarios = full_dataset.N_scenarios

    class_weights = compute_class_weights(y_train, num_classes).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    kf            = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    patch_indices = np.arange(N_patches)

    fold_histories = []
    best_val_acc   = -1.0
    best_fold_idx  = 0

    for fold, (train_patch_idx, val_patch_idx) in enumerate(kf.split(patch_indices)):

        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1} / {num_folds}")
        print(f"  Train patches  : {len(train_patch_idx):,}")
        print(f"  Val   patches  : {len(val_patch_idx):,}")
        print(f"  Train samples  : {len(train_patch_idx) * N_scenarios:,}  "
              f"(patches × {N_scenarios} scenarios)")
        print(f"  Val   samples  : {len(val_patch_idx)   * N_scenarios:,}")
        print(f"{'='*60}")

        train_flat = np.concatenate([np.arange(p * N_scenarios, (p + 1) * N_scenarios) for p in train_patch_idx])
        val_flat = np.concatenate([np.arange(p * N_scenarios, (p + 1) * N_scenarios) for p in val_patch_idx])

        train_loader = DataLoader(
            full_dataset, batch_size=batch_size,
            sampler=SubsetRandomSampler(train_flat),
            num_workers=2, pin_memory=False,
        )
        val_loader = DataLoader(
            full_dataset, batch_size=batch_size,
            sampler=SubsetRandomSampler(val_flat),
            num_workers=2, pin_memory=False,
        )

        model     = model_factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        history           = defaultdict(list)
        best_val_acc_fold = -1.0
        best_state        = None

        for epoch in range(num_epochs):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
            vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)
            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(vl_loss)
            history['val_acc'].append(vl_acc)

            print(f"  Epoch {epoch+1:2d}/{num_epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

            if vl_acc > best_val_acc_fold:
                best_val_acc_fold = vl_acc
                best_state        = copy.deepcopy(model.state_dict())

        history['best_val_acc'] = best_val_acc_fold
        history['best_state']   = best_state
        fold_histories.append(dict(history))

        print(f"\n  ✓ Fold {fold+1} best val acc: {best_val_acc_fold:.4f}")

        if best_val_acc_fold > best_val_acc:
            best_val_acc  = best_val_acc_fold
            best_fold_idx = fold

    print(f"\n{'='*60}")
    print(f"K-FOLD COMPLETE")
    print(f"  Best fold    : {best_fold_idx + 1}")
    print(f"  Best val acc : {best_val_acc:.4f}")
    print(f"{'='*60}")

    return fold_histories, best_fold_idx