import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import copy, json, time

from vit import ViTFloodClassifier, get_model_summary
from dataset import create_complete_dataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PATCH_SIZE  = 4
EMBED_DIM   = 128
NUM_HEADS   = 4
NUM_LAYERS  = 3
BATCH_SIZE  = 1024
NUM_EPOCHS  = 20
NUM_FOLDS   = 3
LR          = 3e-4

WEIGHT_DECAY   = 1e-4
GRAD_CLIP_NORM = 1.0
NUM_CLASSES    = 5
OUTPUT_DIR     = Path('./outputs')
CLASS_NAMES    = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class FloodPatchDataset(Dataset):
    def __init__(self, X_spatial: np.ndarray, y_labels: np.ndarray, rainfall: np.ndarray):
        assert X_spatial.shape[0] == rainfall.shape[0] == y_labels.shape[0],             f"Length mismatch: spatial={X_spatial.shape[0]}, rainfall={rainfall.shape[0]}, labels={y_labels.shape[0]}"

        self.X    = torch.tensor(X_spatial, dtype=torch.float32)  # (N, C, H, W)
        self.y    = torch.tensor(y_labels,  dtype=torch.long)     # (N,)
        self.rain = torch.tensor(rainfall,  dtype=torch.float32)  # (N, T)

        print(f"[FloodPatchDataset]  {len(self):,} samples")
        print(f"  Spatial : {tuple(self.X.shape)}")
        print(f"  Rainfall: {tuple(self.rain.shape)}")
        print(f"  Labels  : {tuple(self.y.shape)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.rain[idx], self.y[idx]



# ─────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────
def make_model() -> nn.Module:
    return ViTFloodClassifier(
        spatial_channels  = 3,
        spatial_patch_size= PATCH_SIZE,
        rainfall_timesteps= 13,
        num_classes       = NUM_CLASSES,
        embed_dim         = EMBED_DIM,
        num_layers        = NUM_LAYERS,
        num_heads         = NUM_HEADS,
        mlp_ratio         = 2.0,
        dropout           = 0.1,
        rainfall_method   = 'conv',
        spatial_method    = 'conv',
        pooling_method    = 'mean',
        learnable_pos_enc = True,
    )


# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────
def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    flat  = y.flatten()
    flat  = flat[flat >= 0]
    total = len(flat)
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    print("\nClass weights (inverse frequency):")
    for c in range(NUM_CLASSES):
        count      = (flat == c).sum()
        weights[c] = total / (NUM_CLASSES * count) if count > 0 else 0.0
        print(f"  Class {c} ({CLASS_NAMES[c]:10s}): {count/total*100:5.2f}%  →  weight {weights[c]:.4f}")
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# SINGLE EPOCH
# ─────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scaler,
    is_train: bool = True,
) -> Tuple[float, float]:
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)

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
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(spatial, rainfall)
                loss = criterion(logits, labels)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

            if (i + 1) % 500 == 0 or (i + 1) == n_batches:
                tag = "TRAIN" if is_train else "VAL  "
                print(f"    [{tag}] {i+1:>5}/{n_batches}  "
                      f"loss={total_loss/total:.4f}  acc={correct/total:.4f}")

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# K-FOLD TRAINING
# ─────────────────────────────────────────────
def train_kfold(
    full_dataset: FloodPatchDataset,
    y_train: np.ndarray,
) -> Tuple[List[Dict], int]:

    (OUTPUT_DIR / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'logs').mkdir(parents=True, exist_ok=True)

    class_weights = compute_class_weights(y_train).to(DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    kf      = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    indices = np.arange(len(full_dataset))

    fold_histories: List[Dict] = []
    best_val_acc  = -1.0
    best_fold_idx = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1} / {NUM_FOLDS}")
        print(f"  Train: {len(train_idx):,} samples")
        print(f"  Val  : {len(val_idx):,} samples")
        print(f"{'='*60}")

        train_loader = DataLoader(
            full_dataset, batch_size=BATCH_SIZE,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=2, pin_memory=DEVICE.type == 'cuda',
        )
        val_loader = DataLoader(
            full_dataset, batch_size=BATCH_SIZE,
            sampler=SubsetRandomSampler(val_idx),
            num_workers=2, pin_memory=DEVICE.type == 'cuda',
        )

        model     = make_model().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        scaler    = torch.amp.GradScaler('cuda') if USE_AMP else None

        history           = defaultdict(list)
        best_val_acc_fold = -1.0
        best_state        = None

        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, is_train=True)
            vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      None,   is_train=False)
            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(vl_loss)
            history['val_acc'].append(vl_acc)

            print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} ({time.time()-t0:.0f}s)  "
                  f"train loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
                  f"val loss={vl_loss:.4f}  acc={vl_acc:.4f}")

            if vl_acc > best_val_acc_fold:
                best_val_acc_fold = vl_acc
                best_state        = copy.deepcopy(model.state_dict())

        torch.save(
            {'fold': fold+1, 'model_state_dict': best_state, 'best_val_acc': best_val_acc_fold},
            OUTPUT_DIR / 'checkpoints' / f'fold_{fold+1}_best.pth',
        )

        history['best_val_acc'] = best_val_acc_fold
        history['best_state']   = best_state
        fold_histories.append(dict(history))

        print(f"\n  ✓ Fold {fold+1} best val acc: {best_val_acc_fold:.4f}")

        if best_val_acc_fold > best_val_acc:
            best_val_acc  = best_val_acc_fold
            best_fold_idx = fold

    print(f"\n{'='*60}")
    print(f"K-FOLD COMPLETE  |  Best fold: {best_fold_idx+1}  |  Best val acc: {best_val_acc:.4f}")
    print(f"{'='*60}")

    # Save fold histories (skip state dicts — too large for JSON)
    log = [{k: v for k, v in h.items() if k != 'best_state'} for h in fold_histories]
    with open(OUTPUT_DIR / 'logs' / 'fold_histories.json', 'w') as f:
        json.dump(log, f, indent=2)

    return fold_histories, best_fold_idx


# ─────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, split_name: str = 'Test') -> Dict:
    log_dir = OUTPUT_DIR / 'logs'; log_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    preds, trues, probs_list = [], [], []

    for spatial, rainfall, labels in test_loader:
        spatial, rainfall = spatial.to(DEVICE), rainfall.to(DEVICE)
        logits, _ = model(spatial, rainfall)
        probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(labels.numpy())

    y_true = np.array(trues);  y_pred = np.array(preds)
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


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def _plot_confusion_matrix(cm: np.ndarray, title: str = 'Confusion Matrix',
                            save_path: Optional[str] = None):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                cbar_kws={'label': 'Proportion'})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_fold_histories(fold_histories: List[Dict], save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, h in enumerate(fold_histories):
        axes[0].plot(h['train_loss'], label=f'Fold {i+1} train', linestyle='--')
        axes[0].plot(h['val_loss'],   label=f'Fold {i+1} val')
        axes[1].plot(h['train_acc'],  label=f'Fold {i+1} train', linestyle='--')
        axes[1].plot(h['val_acc'],    label=f'Fold {i+1} val')
    for ax, title, ylabel in zip(axes, ['Loss', 'Accuracy'], ['Loss', 'Accuracy']):
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig