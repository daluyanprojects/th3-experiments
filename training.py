import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score as sk_f1
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import copy
from config import TrainConfig

class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']

class FloodPatchDataset(Dataset):
    def __init__(self, X_spatial: np.ndarray, y_labels: np.ndarray, rainfall: np.ndarray):
        self.N_patches   = X_spatial.shape[0]
        self.N_scenarios = rainfall.shape[0]
        self.X    = torch.tensor(X_spatial, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y    = torch.tensor(y_labels,  dtype=torch.long).T
        self.rain = torch.tensor(rainfall,  dtype=torch.float32)

        print(f"[FloodPatchDataset]")
        print(f"  Spatial tokens : {self.X.shape}  (N_patches, C, H, W)")
        print(f"  Labels         : {self.y.shape}  (N_patches, N_scenarios)")
        print(f"  Rainfall       : {self.rain.shape}  (N_scenarios, timesteps)")
        print(f"  Total samples  : {len(self):,}  (patches × scenarios)")

    def __len__(self):
        return self.N_patches * self.N_scenarios

    def __getitem__(self, idx):
        patch_idx    = idx // self.N_scenarios
        scenario_idx = idx  % self.N_scenarios
        return self.X[patch_idx], self.rain[scenario_idx], self.y[patch_idx, scenario_idx]


def compute_class_weights(y_train: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    flat  = y_train.flatten()
    flat  = flat[flat >= 0]
    total = len(flat)

    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        count      = (flat == c).sum()
        weights[c] = total / (num_classes * count) if count > 0 else 0.0
    weights = weights / weights.sum() * num_classes

    print("Class weights (inverse frequency):")
    for i, (name, w) in enumerate(zip(class_names, weights)):
        count = (flat == i).sum()
        print(f"  Class {i} ({name:10s}): {count/total*100:5.2f}%  →  weight {w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    is_train:  bool = True,
) -> Tuple[float, float, Optional[float]]:

    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    total_batches              = len(loader)
    all_preds, all_labels      = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (spatial, rainfall, labels) in enumerate(loader):
            spatial  = spatial.to(device)
            rainfall = rainfall.to(device)
            labels   = labels.to(device)

            logits, _ = model(spatial, rainfall)
            loss      = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds       = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            if not is_train:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 1000 == 0 or (batch_idx + 1) == total_batches:
                phase = "TRAIN" if is_train else "VAL  "
                print(f"    [{phase}] batch {batch_idx+1:>5}/{total_batches}  "
                      f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    avg_loss = total_loss / total
    avg_acc  = correct / total
    macro_f1 = sk_f1(all_labels, all_preds, average='macro', zero_division=0) if not is_train else None

    return avg_loss, avg_acc, macro_f1


def train_kfold(
    model_factory,
    full_dataset: FloodPatchDataset,
    y_train:      np.ndarray,
    cfg:          TrainConfig,
    device:       torch.device = torch.device('cpu'),
) -> Tuple[List[Dict], int]:

    N_patches   = full_dataset.N_patches
    N_scenarios = full_dataset.N_scenarios

    class_weights = compute_class_weights(y_train, cfg.num_classes).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    kf            = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=42)
    patch_indices = np.arange(N_patches)

    fold_histories:  List[Dict] = []
    best_f1_overall: float      = -1.0
    best_fold_idx:   int        = 0

    for fold, (train_patch_idx, val_patch_idx) in enumerate(kf.split(patch_indices)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1} / {cfg.num_folds}")
        print(f"  Train samples : {len(train_patch_idx) * N_scenarios:,}")
        print(f"  Val   samples : {len(val_patch_idx)   * N_scenarios:,}")
        print(f"{'='*60}")

        train_flat = np.concatenate([
            np.arange(p * N_scenarios, (p+1) * N_scenarios) for p in train_patch_idx
        ])
        val_flat = np.concatenate([
            np.arange(p * N_scenarios, (p+1) * N_scenarios) for p in val_patch_idx
        ])

        train_loader = DataLoader(full_dataset, batch_size=cfg.batch_size,
                                  sampler=SubsetRandomSampler(train_flat),
                                  num_workers=2, pin_memory=False)
        val_loader   = DataLoader(full_dataset, batch_size=cfg.batch_size,
                                  sampler=SubsetRandomSampler(val_flat),
                                  num_workers=2, pin_memory=False)

        model     = model_factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

        history        = defaultdict(list)
        best_f1_fold   = -1.0
        best_state     = None

        for epoch in range(cfg.num_epochs):
            tr_loss, tr_acc, _      = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
            vl_loss, vl_acc, vl_f1  = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)
            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(vl_loss)
            history['val_acc'].append(vl_acc)
            history['val_macro_f1'].append(vl_f1)

            print(f"  Epoch {epoch+1:2d}/{cfg.num_epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  val_f1={vl_f1:.4f}")

            if vl_f1 > best_f1_fold:
                best_f1_fold = vl_f1
                best_state   = copy.deepcopy(model.state_dict())

        # Save checkpoint
        save_dir = cfg.output_dir / f'fold_{fold+1}'
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': best_state,
            'best_macro_f1':    best_f1_fold,
            'fold':             fold + 1,
        }, save_dir / 'best_model.pt')

        history['best_macro_f1'] = best_f1_fold
        history['best_state']    = best_state
        fold_histories.append(dict(history))

        print(f"\n  ✓ Fold {fold+1} best macro F1: {best_f1_fold:.4f}")
        if best_f1_fold > best_f1_overall:
            best_f1_overall = best_f1_fold
            best_fold_idx   = fold

    print(f"\n{'='*60}")
    print(f"K-FOLD COMPLETE  |  Best fold: {best_fold_idx+1}  |  Best macro F1: {best_f1_overall:.4f}")
    print(f"{'='*60}")
    return fold_histories, best_fold_idx