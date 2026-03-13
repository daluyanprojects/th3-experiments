import torch
import torch.nn as nn
import torch.optim as optim
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


class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, smooth: float = 1.0,
                 ce_weight: float = 0.9, dice_weight: float = 0.1):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss(weight=class_weights)
        self.smooth      = smooth
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred      = torch.softmax(pred, dim=1)
        target_oh = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).float()
        intersection = (pred * target_oh).sum(dim=0)
        union        = pred.sum(dim=0) + target_oh.sum(dim=0)
        dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice_loss(pred, target)


class WarmupCosineScheduler:
    def __init__(self, optimizer, cfg: TrainConfig):
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
    def __init__(self, X_spatial: np.ndarray, y_labels: np.ndarray, 
                 rainfall: np.ndarray, conditioning: Optional[np.ndarray] = None):
        assert X_spatial.shape[0] == rainfall.shape[0] == y_labels.shape[0]
        
        self.X    = torch.tensor(X_spatial, dtype=torch.float32)
        self.y    = torch.tensor(y_labels,  dtype=torch.long)
        self.rain = torch.tensor(rainfall,  dtype=torch.float32)
        
        if conditioning is not None:
            assert conditioning.shape[0] == y_labels.shape[0], \
                f"Conditioning shape {conditioning.shape[0]} != labels {y_labels.shape[0]}"
            self.cond = torch.tensor(conditioning, dtype=torch.float32)
            has_cond_str = f"✓ with conditioning {tuple(self.cond.shape)}"
        else:
            self.cond = None
            has_cond_str = "⚠ without conditioning"
        
        print(f"[FloodPatchDataset]  {len(self):,} samples {has_cond_str}")
        print(f"  Spatial : {tuple(self.X.shape)}")
        print(f"  Rainfall: {tuple(self.rain.shape)}")
        if self.cond is not None:
            print(f"  Conditioning: {tuple(self.cond.shape)}")
        print(f"  Labels  : {tuple(self.y.shape)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.cond is not None:
            return self.X[idx], self.rain[idx], self.cond[idx], self.y[idx]  # 4-tuple
        else:
            return self.X[idx], self.rain[idx], self.y[idx]  # 3-tuple (backward compat)


def make_model(cfg: TrainConfig) -> nn.Module:
    return ViTFloodClassifier(
        spatial_channels    = cfg.spatial_channels,
        spatial_patch_size  = cfg.patch_size,
        rainfall_timesteps  = cfg.rainfall_timesteps,
        num_classes         = cfg.num_classes,
        embed_dim           = cfg.embed_dim,
        num_layers          = cfg.num_layers,
        num_heads           = cfg.num_heads,
        mlp_ratio           = cfg.mlp_ratio,
        dropout             = cfg.dropout,
        rainfall_method     = cfg.rainfall_method,
        rainfall_hidden     = cfg.rainfall_hidden,
        learnable_pos_enc   = cfg.learnable_pos_enc,
        use_conditioning    = cfg.use_conditioning,
        conditioning_dim    = cfg.conditioning_dim,
        conditioning_hidden = cfg.conditioning_hidden,
    )


def compute_class_weights(y: np.ndarray, cfg: TrainConfig) -> torch.Tensor:
    flat    = y.flatten()
    flat    = flat[flat >= 0]
    total   = len(flat)
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
        print(f"  Loss: CombinedLoss (ce={cfg.ce_weight}, dice={cfg.dice_weight})")
        return CombinedLoss(class_weights=w, ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = confusion_matrix(y_true, y_pred)

    # per-class IoU from confusion matrix
    iou_per_class = []
    for c in range(cm.shape[0]):
        tp    = cm[c, c]
        fp    = cm[:, c].sum() - tp
        fn    = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou_per_class.append(float(tp / denom) if denom > 0 else 0.0)

    m = {
        'accuracy'        : accuracy_score(y_true, y_pred),
        'macro_f1'        : f1_score(y_true, y_pred, average='macro',     zero_division=0),
        'weighted_f1'     : f1_score(y_true, y_pred, average='weighted',  zero_division=0),
        'macro_precision' : precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall'    : recall_score(y_true, y_pred, average='macro',    zero_division=0),
        'macro_iou'       : float(np.mean(iou_per_class)),
        'iou_per_class'   : iou_per_class,
    }

    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, n in enumerate(CLASS_NAMES):
        m[f'precision_{n}'] = float(prec[i])
        m[f'recall_{n}']    = float(rec[i])
        m[f'f1_{n}']        = float(f1[i])
        m[f'iou_{n}']       = iou_per_class[i]

    crit = y_true >= 3
    if crit.sum() > 0:
        m['critical_recall'] = float(recall_score(y_true[crit] >= 3, y_pred[crit] >= 3))

    m['confusion_matrix'] = cm
    return m


def print_metrics(m: Dict, split: str = 'Validation'):
    print(f"\n{'='*60}\n{split} Metrics\n{'='*60}")
    print(f"  Accuracy         : {m['accuracy']:.4f}")
    print(f"  Macro F1         : {m['macro_f1']:.4f}")
    print(f"  Weighted F1      : {m['weighted_f1']:.4f}")
    print(f"  Macro Precision  : {m['macro_precision']:.4f}")
    print(f"  Macro Recall     : {m['macro_recall']:.4f}")
    print(f"  Macro IoU        : {m['macro_iou']:.4f}")
    if 'critical_recall' in m:
        print(f"  Critical Recall (Heavy+Extreme): {m['critical_recall']:.4f}")
    print(f"\n  {'Class':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'IoU':>8}")
    print(f"  {'-'*48}")
    for n in CLASS_NAMES:
        print(f"  {n:<12} {m[f'precision_{n}']:>8.4f} {m[f'recall_{n}']:>8.4f} "
              f"{m[f'f1_{n}']:>8.4f} {m[f'iou_{n}']:>8.4f}")


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
        for i, batch in enumerate(loader):
            if len(batch) == 4:
                spatial, rainfall, conditioning, labels = batch
                spatial      = spatial.to(DEVICE)
                rainfall     = rainfall.to(DEVICE)
                conditioning = conditioning.to(DEVICE)
                labels       = labels.to(DEVICE)
            else:
                spatial, rainfall, labels = batch
                spatial      = spatial.to(DEVICE)
                rainfall     = rainfall.to(DEVICE)
                labels       = labels.to(DEVICE)
                conditioning = None

            if is_train and scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(spatial, rainfall, conditioning)
                    loss      = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(spatial, rainfall, conditioning)
                loss      = criterion(logits, labels)
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

    avg_loss        = total_loss / total
    avg_acc         = correct / total
    macro_f1        = None
    macro_precision = None
    macro_recall    = None
    macro_iou       = None

    if not is_train:
        macro_f1        = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
        macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_recall    = recall_score(all_labels, all_preds, average='macro',    zero_division=0)
        _cm = confusion_matrix(all_labels, all_preds)
        _ious = []
        for _c in range(_cm.shape[0]):
            _tp = _cm[_c, _c]; _fp = _cm[:, _c].sum() - _tp; _fn = _cm[_c, :].sum() - _tp
            _denom = _tp + _fp + _fn
            _ious.append(float(_tp / _denom) if _denom > 0 else 0.0)
        macro_iou = float(np.mean(_ious))

    return avg_loss, avg_acc, macro_f1, macro_precision, macro_recall, macro_iou


def train_kfold(
    full_dataset: FloodPatchDataset,
    y_train: np.ndarray,
    cfg: TrainConfig,
    ckpt_dir: str | Path,
    log_dir: str | Path
) -> Tuple[List[Dict], int]:
    ckpt_dir = Path(ckpt_dir)
    log_dir = Path(log_dir)

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
        scheduler = WarmupCosineScheduler(optimizer, cfg)
        scaler    = torch.amp.GradScaler('cuda') if USE_AMP else None

        history           = defaultdict(list)
        best_fold_monitor = -1.0
        best_state        = None

        for epoch in range(cfg.num_epochs):
            t0 = time.time()
            tr_loss, tr_acc, _, _, _, _                      = run_epoch(model, train_loader, criterion, cfg, optimizer, scaler, is_train=True)
            vl_loss, vl_acc, vl_f1, vl_prec, vl_rec, vl_iou  = run_epoch(model, val_loader,  criterion, cfg, is_train=False)
            lr = scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(vl_loss)
            history['val_acc'].append(vl_acc)
            history['val_f1'].append(vl_f1)
            history['val_precision'].append(vl_prec)
            history['val_recall'].append(vl_rec)
            history['val_iou'].append(vl_iou)

            print(f"  Epoch {epoch+1:2d}/{cfg.num_epochs} ({time.time()-t0:.0f}s)  "
                  f"lr={lr:.2e}  train loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
                  f"val loss={vl_loss:.4f}  acc={vl_acc:.4f}  f1={vl_f1:.4f}  "
                  f"prec={vl_prec:.4f}  rec={vl_rec:.4f}  iou={vl_iou:.4f}")


            monitor = vl_f1 if cfg.checkpoint_metric == 'macro_f1' else vl_acc
            if monitor > best_fold_monitor:
                best_fold_monitor = monitor
                best_state        = copy.deepcopy(model.state_dict())

        torch.save(
            {'fold': fold + 1, 'model_state_dict': best_state,
             'best_monitor': best_fold_monitor},
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
def evaluate(
    model: nn.Module, 
    test_loader: DataLoader, 
    cfg: TrainConfig, 
    split_name: str = 'Test', 
    log_dir: str | Path = None
) -> Dict:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    preds, trues, probs_list = [], [], []

    for batch in test_loader:
        if len(batch) == 4:
            spatial, rainfall, conditioning, labels = batch
            spatial, rainfall, conditioning = spatial.to(DEVICE), rainfall.to(DEVICE), conditioning.to(DEVICE)
        else:
            spatial, rainfall, labels = batch
            spatial, rainfall = spatial.to(DEVICE), rainfall.to(DEVICE)
            conditioning = None

        logits, _ = model(spatial, rainfall, conditioning)
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


def _plot_confusion_matrix(cm: np.ndarray, title: str = 'Confusion Matrix', save_path: Optional[Path] = None):
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