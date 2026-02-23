# training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                              recall_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import copy, json, time, sys
import torch.nn.functional as F

from vit import ViTFloodClassifier
from config import TrainConfig

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'

CLASS_NAMES = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce  = F.cross_entropy(inputs, targets, weight=self.weight,
                              label_smoothing=self.label_smoothing, reduction='none')
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()

class FloodPatchDataset(Dataset):
    def __init__(self, X_spatial: np.ndarray, y_labels: np.ndarray, rainfall: np.ndarray):
        assert X_spatial.shape[0] == rainfall.shape[0] == y_labels.shape[0]
        self.X    = torch.tensor(X_spatial, dtype=torch.float32)
        self.y    = torch.tensor(y_labels,  dtype=torch.long)
        self.rain = torch.tensor(rainfall,  dtype=torch.float32)
        print(f"[FloodPatchDataset]  {len(self):,} samples")
        print(f"  Spatial : {tuple(self.X.shape)}")
        print(f"  Rainfall: {tuple(self.rain.shape)}")
        print(f"  Labels  : {tuple(self.y.shape)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.rain[idx], self.y[idx]


def make_model(cfg: TrainConfig) -> nn.Module:
    return ViTFloodClassifier(
        spatial_channels   = cfg.spatial_channels,
        spatial_patch_size = cfg.patch_size,
        rainfall_timesteps = cfg.rainfall_timesteps,
        num_classes        = cfg.num_classes,
        embed_dim          = cfg.embed_dim,
        num_layers         = cfg.num_layers,
        num_heads          = cfg.num_heads,
        mlp_ratio          = cfg.mlp_ratio,
        dropout            = cfg.dropout,
        rainfall_method    = cfg.rainfall_method,
        spatial_method     = cfg.spatial_method,
        pooling_method     = cfg.pooling_method,
        learnable_pos_enc  = cfg.learnable_pos_enc,
    )

def compute_class_weights(y: np.ndarray, cfg: TrainConfig) -> torch.Tensor:
    flat   = y.flatten()
    flat   = flat[flat >= 0]
    total  = len(flat)
    weights = np.zeros(cfg.num_classes, dtype=np.float32)
    print(f"\nClass weights (power={cfg.weight_power}):")
    for c in range(cfg.num_classes):
        count      = (flat == c).sum()
        inv        = total / (cfg.num_classes * count) if count > 0 else 0.0
        weights[c] = inv ** cfg.weight_power
        print(f"  Class {c} ({CLASS_NAMES[c]:10s}): {count/total*100:5.2f}%  →  weight {weights[c]:.4f}")
    weights = weights / weights.sum() * cfg.num_classes
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(cfg: TrainConfig, class_weights: torch.Tensor) -> nn.Module:
    w = class_weights.to(DEVICE)
    if cfg.loss_type == 'focal':
        print(f"  Loss: FocalLoss(gamma={cfg.focal_gamma}, label_smoothing={cfg.label_smoothing})")
        return FocalLoss(weight=w, gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)
    else:
        print(f"  Loss: Weighted CrossEntropyLoss (weight_power={cfg.weight_power}, label_smoothing={cfg.label_smoothing})")
        return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    m = {
        'accuracy'   : accuracy_score(y_true, y_pred),
        'macro_f1'   : f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
    }
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, n in enumerate(CLASS_NAMES):
        m[f'precision_{n}'] = prec[i]
        m[f'recall_{n}']    = rec[i]
        m[f'f1_{n}']        = f1[i]
    crit = y_true >= 3
    if crit.sum() > 0:
        m['critical_recall'] = float(recall_score(y_true[crit] >= 3, y_pred[crit] >= 3))
    m['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    return m


def print_metrics(m: Dict, split: str = 'Validation'):
    print(f"\n{'='*60}\n{split} Metrics\n{'='*60}")
    print(f"  Accuracy     : {m['accuracy']:.4f}")
    print(f"  Macro F1     : {m['macro_f1']:.4f}")
    print(f"  Weighted F1  : {m['weighted_f1']:.4f}")
    if 'critical_recall' in m:
        print(f"  Critical Recall (Heavy+Extreme): {m['critical_recall']:.4f}")
    print(f"\n  {'Class':<12} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'-'*38}")
    for n in CLASS_NAMES:
        print(f"  {n:<12} {m[f'precision_{n}']:>8.4f} {m[f'recall_{n}']:>8.4f} {m[f'f1_{n}']:>8.4f}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: TrainConfig,
    optimizer: Optional[optim.Optimizer] = None,
    scaler=None,
    is_train: bool = True,
) -> Tuple[float, float, Optional[float]]:

    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, (spatial, rainfall, labels) in enumerate(loader):
            spatial  = spatial.to(DEVICE)
            rainfall = rainfall.to(DEVICE)
            labels   = labels.to(DEVICE)

            if is_train and scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(spatial, rainfall)
                    loss = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(spatial, rainfall)
                loss = criterion(logits, labels)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds_batch = logits.argmax(1)
            correct    += (preds_batch == labels).sum().item()
            total      += labels.size(0)
            if not is_train:
                all_preds.extend(preds_batch.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 500 == 0 or (i + 1) == n_batches:
                tag = "TRAIN" if is_train else "VAL  "
                print(f"    [{tag}] {i+1:>5}/{n_batches}  "
                      f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    avg_loss = total_loss / total
    avg_acc  = correct / total
    macro_f1 = None
    if not is_train:
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, avg_acc, macro_f1


def train_kfold(
    full_dataset: FloodPatchDataset,
    y_train: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[List[Dict], int]:

    ckpt_dir = cfg.output_dir / 'checkpoints'
    log_dir  = cfg.output_dir / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    class_weights = compute_class_weights(y_train, cfg)
    criterion     = build_criterion(cfg, class_weights)

    kf      = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=42)
    indices = np.arange(len(full_dataset))

    fold_histories: List[Dict] = []
    best_overall   = -1.0
    best_fold_idx  = 0
    _nw = 0 if sys.platform == 'win32' else 2

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1} / {cfg.num_folds}")
        print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}")
        print(f"{'='*60}")

        train_loader = DataLoader(
            full_dataset, batch_size=cfg.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=_nw, pin_memory=DEVICE.type == 'cuda',
            persistent_workers=(_nw > 0),
        )
        val_loader = DataLoader(
            full_dataset, batch_size=cfg.batch_size,
            sampler=SubsetRandomSampler(val_idx),
            num_workers=_nw, pin_memory=DEVICE.type == 'cuda',
            persistent_workers=(_nw > 0),
        )

        model     = make_model(cfg).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        scaler    = torch.amp.GradScaler('cuda') if USE_AMP else None

        history           = defaultdict(list)
        best_fold_monitor = -1.0
        best_state        = None

        for epoch in range(cfg.num_epochs):
            t0 = time.time()
            tr_loss, tr_acc, _      = run_epoch(model, train_loader, criterion, cfg, optimizer, scaler, is_train=True)
            vl_loss, vl_acc, vl_f1  = run_epoch(model, val_loader,  criterion, cfg, is_train=False)
            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(vl_loss)
            history['val_acc'].append(vl_acc)
            history['val_f1'].append(vl_f1)

            print(f"  Epoch {epoch+1:2d}/{cfg.num_epochs} ({time.time()-t0:.0f}s)  "
                  f"train loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
                  f"val loss={vl_loss:.4f}  acc={vl_acc:.4f}  macro_f1={vl_f1:.4f}")

            monitor = vl_f1 if cfg.checkpoint_metric == 'macro_f1' else vl_acc
            if monitor > best_fold_monitor:
                best_fold_monitor = monitor
                best_state        = copy.deepcopy(model.state_dict())

        torch.save(
            {'fold': fold + 1, 'model_state_dict': best_state,
             'best_monitor': best_fold_monitor, 'cfg': cfg},
            ckpt_dir / f'fold_{fold+1}_best.pth',
        )

        history['best_val_monitor'] = best_fold_monitor
        history['best_state']       = best_state
        fold_histories.append(dict(history))
        print(f"\n  ✓ Fold {fold+1} best {cfg.checkpoint_metric}: {best_fold_monitor:.4f}")

        if best_fold_monitor > best_overall:
            best_overall  = best_fold_monitor
            best_fold_idx = fold

    print(f"\n{'='*60}")
    print(f"K-FOLD COMPLETE  |  Best fold: {best_fold_idx+1}  |  Best {cfg.checkpoint_metric}: {best_overall:.4f}")
    print(f"{'='*60}")

    log = [{k: v for k, v in h.items() if k != 'best_state'} for h in fold_histories]
    with open(log_dir / 'fold_histories.json', 'w') as f:
        json.dump(log, f, indent=2)

    return fold_histories, best_fold_idx


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, cfg: TrainConfig, split_name: str = 'Test') -> Dict:
    log_dir = cfg.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    preds, trues, probs_list = [], [], []

    for spatial, rainfall, labels in test_loader:
        spatial, rainfall = spatial.to(DEVICE), rainfall.to(DEVICE)
        logits, _ = model(spatial, rainfall)
        probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(labels.numpy())

    y_true = np.array(trues)
    y_pred = np.array(preds)
    m = compute_metrics(y_true, y_pred)
    print_metrics(m, split=split_name)

    fig = _plot_confusion_matrix(
        m['confusion_matrix'],
        title=f'{split_name} Confusion Matrix',
        save_path=log_dir / f'{split_name.lower()}_cm.png',
    )
    plt.close(fig)

    results = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
               for k, v in m.items() if k != 'confusion_matrix'}
    results['confusion_matrix'] = m['confusion_matrix'].tolist()
    results['predictions']      = y_pred.tolist()
    results['labels']           = y_true.tolist()
    results['probabilities']    = np.vstack(probs_list).tolist()

    with open(log_dir / f'{split_name.lower()}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results → {log_dir}/{split_name.lower()}_results.json")
    return m



def _plot_confusion_matrix(cm: np.ndarray, title: str = 'Confusion Matrix',
                            save_path: Optional[Path] = None):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar_kws={'label': 'Proportion'})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_fold_histories(fold_histories: List[Dict], save_path: Optional[Path] = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, h in enumerate(fold_histories):
        axes[0].plot(h['train_loss'], label=f'Fold {i+1} train', linestyle='--')
        axes[0].plot(h['val_loss'],   label=f'Fold {i+1} val')
        axes[1].plot(h['train_acc'],  label=f'Fold {i+1} train', linestyle='--')
        axes[1].plot(h['val_acc'],    label=f'Fold {i+1} val')
    for ax, title, ylabel in zip(axes, ['Loss', 'Accuracy'], ['Loss', 'Accuracy']):
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig