import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import copy, json, time, sys

from cnn import CNNFloodModel
from config import CNNTrainConfig
from dataset import FloodMapDataset

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'

CLASS_NAMES = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']


# ── Loss ───────────────────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    CE + Dice loss for segmentation.
    ignore_index=-1 excludes outside-mask pixels from both terms.
    """
    def __init__(
        self,
        class_weights : torch.Tensor,
        ignore_index  : int   = -1,
        smooth        : float = 1.0,
        ce_weight     : float = 0.9,
        dice_weight   : float = 0.1,
    ):
        super().__init__()
        self.ce          = nn.CrossEntropyLoss(
                               weight=class_weights,
                               ignore_index=ignore_index,
                           )
        self.ignore_index = ignore_index
        self.smooth       = smooth
        self.ce_weight    = ce_weight
        self.dice_weight  = dice_weight
        self.num_classes  = class_weights.shape[0]

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Mask outside pixels
        mask   = target != self.ignore_index               # (B, H, W) bool
        target_masked = target.clone()
        target_masked[~mask] = 0                           # safe for one_hot

        pred_soft = torch.softmax(pred, dim=1)             # (B, C, H, W)
        target_oh = torch.nn.functional.one_hot(
                        target_masked, num_classes=self.num_classes
                    ).permute(0, 3, 1, 2).float()          # (B, C, H, W)

        # Zero out outside pixels in both
        mask_4d = mask.unsqueeze(1).float()                # (B,1,H,W)
        pred_soft   = pred_soft   * mask_4d
        target_oh   = target_oh  * mask_4d

        intersection = (pred_soft * target_oh).sum(dim=(0, 2, 3))
        union        = pred_soft.sum(dim=(0, 2, 3)) + target_oh.sum(dim=(0, 2, 3))
        dice         = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.ce_weight  * self.ce(pred, target) +
                self.dice_weight * self.dice_loss(pred, target))


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.weight       = weight
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        ce  = F.cross_entropy(inputs, targets, weight=self.weight,
                              ignore_index=self.ignore_index, reduction='none')
        p_t = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma) * ce
        mask = targets != self.ignore_index
        return loss[mask].mean()


# ── LR Scheduler ──────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, cfg: CNNTrainConfig):
        self.optimizer     = optimizer
        self.warmup_epochs = cfg.warmup_epochs
        self.total_epochs  = cfg.num_epochs
        self.base_lr       = cfg.lr
        self.min_lr        = cfg.min_lr
        self.current_epoch = 0

    def step(self) -> float:
        e = self.current_epoch
        if e < self.warmup_epochs:
            lr = self.base_lr * (e / max(self.warmup_epochs, 1))
        else:
            progress = (e - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.current_epoch += 1
        return lr


# ── Model factory ─────────────────────────────────────────────────────────────
def make_model_cnn(cfg: CNNTrainConfig) -> CNNFloodModel:
    return CNNFloodModel(
        num_classes      = cfg.num_classes,
        in_channels      = cfg.spatial_channels,
        rainfall_dim     = cfg.rainfall_dim,
        conditioning_dim = cfg.conditioning_dim,
        context_dim      = cfg.context_dim,
        mlp_hidden       = cfg.mlp_hidden,
        encoder_weights  = cfg.encoder_weights,
    )


# ── Class weights ──────────────────────────────────────────────────────────────
def compute_class_weights(y: np.ndarray, cfg: CNNTrainConfig) -> torch.Tensor:
    """
    y : (S, H, W) label maps — -1 pixels are excluded.
    """
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


def build_criterion(cfg: CNNTrainConfig, class_weights: torch.Tensor) -> nn.Module:
    w = class_weights.to(DEVICE)
    if cfg.loss_type == 'focal':
        print(f"  Loss: FocalLoss(gamma={cfg.focal_gamma}, ignore_index={cfg.ignore_index})")
        return FocalLoss(weight=w, gamma=cfg.focal_gamma, ignore_index=cfg.ignore_index)
    else:
        print(f"  Loss: CombinedLoss (ce={cfg.ce_weight}, dice={cfg.dice_weight}, ignore_index={cfg.ignore_index})")
        return CombinedLoss(class_weights=w, ignore_index=cfg.ignore_index,
                            ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight)


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """y_true / y_pred are already flattened and masked (no -1 values)."""
    m = {
        'accuracy'   : accuracy_score(y_true, y_pred),
        'macro_f1'   : f1_score(y_true, y_pred, average='macro',    zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(5)))
    rec  = recall_score(y_true, y_pred,    average=None, zero_division=0, labels=list(range(5)))
    f1   = f1_score(y_true, y_pred,        average=None, zero_division=0, labels=list(range(5)))
    for i, n in enumerate(CLASS_NAMES):
        m[f'precision_{n}'] = float(prec[i])
        m[f'recall_{n}']    = float(rec[i])
        m[f'f1_{n}']        = float(f1[i])
    crit_mask = y_true >= 3
    if crit_mask.sum() > 0:
        m['critical_recall'] = float(recall_score(
            (y_true[crit_mask] >= 3).astype(int),
            (y_pred[crit_mask] >= 3).astype(int),
            zero_division=0,
        ))
    m['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=list(range(5)))
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


# ── Single epoch ───────────────────────────────────────────────────────────────
def run_epoch_cnn(
    model      : nn.Module,
    loader     : DataLoader,
    criterion  : nn.Module,
    cfg        : CNNTrainConfig,
    optimizer  : Optional[optim.Optimizer] = None,
    scaler     = None,
    is_train   : bool = True,
) -> Tuple[float, float, Optional[float]]:
    """
    One train or validation epoch over full spatial maps.

    Returns (avg_loss, avg_acc, macro_f1_or_None)
    macro_f1 computed only on validation pass.
    """
    model.train() if is_train else model.eval()
    total_loss, correct, total_valid = 0.0, 0, 0
    n_batches  = len(loader)
    all_preds, all_labels = [], []

    accum_steps = cfg.grad_accum_steps if is_train else 1
    if is_train:
        optimizer.zero_grad()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, (spatial, rainfall, conditioning, labels) in enumerate(loader):
            spatial      = spatial.to(DEVICE)           # (B, 3, H, W)
            rainfall     = rainfall.to(DEVICE)          # (B, 13)
            conditioning = conditioning.to(DEVICE)      # (B, 4)
            labels       = labels.to(DEVICE)            # (B, H, W)  int64

            if is_train and scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(spatial, rainfall, conditioning)   # (B,5,H,W)
                    loss      = criterion(logits, labels) / accum_steps
                scaler.scale(loss).backward()
                if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits, _ = model(spatial, rainfall, conditioning)
                loss      = criterion(logits, labels)
                if is_train:
                    (loss / accum_steps).backward()
                    if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                        optimizer.step()
                        optimizer.zero_grad()

            # ── Metrics (valid pixels only) ───────────────────────────────────
            with torch.no_grad():
                preds = logits.argmax(dim=1)             # (B, H, W)
                mask  = labels != cfg.ignore_index       # (B, H, W) bool

                valid_preds  = preds[mask].cpu().numpy()
                valid_labels = labels[mask].cpu().numpy()

                correct     += (valid_preds == valid_labels).sum()
                total_valid += mask.sum().item()
                total_loss  += loss.item() * (accum_steps if is_train else 1) * mask.sum().item()

                if not is_train:
                    all_preds.append(valid_preds)
                    all_labels.append(valid_labels)

            if (i + 1) % max(1, n_batches // 4) == 0 or (i + 1) == n_batches:
                tag = "TRAIN" if is_train else "VAL  "
                print(f"    [{tag}] {i+1:>4}/{n_batches}  "
                      f"loss={total_loss/max(total_valid,1):.4f}  "
                      f"acc={correct/max(total_valid,1):.4f}")

    avg_loss = total_loss / max(total_valid, 1)
    avg_acc  = correct    / max(total_valid, 1)
    macro_f1 = None
    if not is_train and all_preds:
        y_true   = np.concatenate(all_labels)
        y_pred   = np.concatenate(all_preds)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return avg_loss, avg_acc, macro_f1


# ── K-Fold training ────────────────────────────────────────────────────────────
def train_kfold_cnn(
    train_dataset : FloodMapDataset,
    y_train       : np.ndarray,        # (S, H, W) — for class weight computation
    cfg           : CNNTrainConfig,
) -> Tuple[List[Dict], int]:
    """
    K-fold cross-validation over scenario indices (not patch indices).

    With 35 training scenarios and num_folds=5:
        Each fold: 28 train scenarios / 7 val scenarios
    """
    ckpt_dir = cfg.output_dir / 'checkpoints'
    log_dir  = cfg.output_dir / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    class_weights = compute_class_weights(y_train, cfg)
    criterion     = build_criterion(cfg, class_weights)

    # KFold on scenario indices (35 scenarios total)
    kf      = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=42)
    indices = np.arange(len(train_dataset))   # [0 .. 34]

    _nw = 0 if sys.platform == 'win32' else 2

    fold_histories : List[Dict] = []
    best_overall   = -1.0
    best_fold_idx  = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1} / {cfg.num_folds}")
        print(f"  Train scenarios: {len(train_idx)}  |  Val scenarios: {len(val_idx)}")
        print(f"  Train ids: {train_idx.tolist()}")
        print(f"  Val   ids: {val_idx.tolist()}")
        print(f"{'='*60}")

        train_loader = DataLoader(
            train_dataset,
            batch_size  = cfg.batch_size,
            sampler     = SubsetRandomSampler(train_idx),
            num_workers = _nw,
            pin_memory  = DEVICE.type == 'cuda',
            persistent_workers = (_nw > 0),
        )
        val_loader = DataLoader(
            train_dataset,
            batch_size  = cfg.batch_size,
            sampler     = SubsetRandomSampler(val_idx),
            num_workers = _nw,
            pin_memory  = DEVICE.type == 'cuda',
            persistent_workers = (_nw > 0),
        )

        model     = make_model_cnn(cfg).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = WarmupCosineScheduler(optimizer, cfg)
        scaler    = torch.amp.GradScaler('cuda') if USE_AMP else None

        # Print parameter count once (first fold only)
        if fold == 0:
            pc = model.parameter_count()
            print(f"\n  Model parameters:")
            print(f"    Total      : {pc['total']:,}")
            print(f"    Backbone   : {pc['backbone']:,}")
            print(f"    FiLM layers: {pc['film_layers']:,}")
            print(f"    Context MLP: {pc['context_mlp']:,}")

        history           = defaultdict(list)
        best_fold_monitor = -1.0
        best_state        = None

        for epoch in range(cfg.num_epochs):
            t0 = time.time()
            tr_loss, tr_acc, _     = run_epoch_cnn(model, train_loader, criterion, cfg,
                                                    optimizer, scaler, is_train=True)
            vl_loss, vl_acc, vl_f1 = run_epoch_cnn(model, val_loader, criterion, cfg,
                                                    is_train=False)
            lr = scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_loss'].append(vl_loss)
            history['val_acc'].append(vl_acc)
            history['val_f1'].append(vl_f1 or 0.0)
            history['lr'].append(lr)

            print(f"  Epoch {epoch+1:2d}/{cfg.num_epochs} ({time.time()-t0:.0f}s)  "
                  f"lr={lr:.2e}  "
                  f"train loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
                  f"val loss={vl_loss:.4f}  acc={vl_acc:.4f}  "
                  f"macro_f1={vl_f1:.4f}")

            monitor = vl_f1 if cfg.checkpoint_metric == 'macro_f1' else vl_acc
            if monitor is not None and monitor > best_fold_monitor:
                best_fold_monitor = monitor
                best_state        = copy.deepcopy(model.state_dict())

        torch.save(
            {
                'fold'             : fold + 1,
                'model_state_dict' : best_state,
                'best_monitor'     : best_fold_monitor,
                'cfg'              : cfg.output_dir,   # for reference
            },
            ckpt_dir / f'fold_{fold+1}_best.pth',
        )

        history['best_val_monitor'] = best_fold_monitor
        fold_histories.append(dict(history))
        print(f"\n  ✓ Fold {fold+1} best {cfg.checkpoint_metric}: {best_fold_monitor:.4f}")

        if best_fold_monitor > best_overall:
            best_overall  = best_fold_monitor
            best_fold_idx = fold

    print(f"\n{'='*60}")
    print(f"K-FOLD COMPLETE  |  Best fold: {best_fold_idx+1}"
          f"  |  Best {cfg.checkpoint_metric}: {best_overall:.4f}")
    print(f"{'='*60}")

    # Save fold histories (exclude non-serialisable state dicts)
    log = [{k: v for k, v in h.items() if k != 'best_state'} for h in fold_histories]
    with open(log_dir / 'fold_histories.json', 'w') as f:
        json.dump(log, f, indent=2)

    return fold_histories, best_fold_idx


# ── Final test evaluation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_cnn(
    model      : nn.Module,
    test_loader: DataLoader,
    cfg        : CNNTrainConfig,
    split_name : str = 'Test',
) -> Dict:
    """
    Evaluate on the test set. Flattens (B,H,W) predictions,
    masks -1 pixels, then computes the same metrics as ViT branch.
    """
    log_dir = cfg.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds, all_trues = [], []

    for spatial, rainfall, conditioning, labels in test_loader:
        spatial      = spatial.to(DEVICE)
        rainfall     = rainfall.to(DEVICE)
        conditioning = conditioning.to(DEVICE)

        logits, _ = model(spatial, rainfall, conditioning)  # (B,5,H,W)
        preds     = logits.argmax(dim=1)                    # (B,H,W)

        mask = labels != cfg.ignore_index                   # (B,H,W) bool
        all_preds.append(preds[mask].cpu().numpy())
        all_trues.append(labels[mask].numpy())

    y_true = np.concatenate(all_trues)
    y_pred = np.concatenate(all_preds)

    m = compute_metrics(y_true, y_pred)
    print_metrics(m, split=split_name)

    fig = _plot_confusion_matrix(
        m['confusion_matrix'],
        title     = f'{split_name} Confusion Matrix',
        save_path = log_dir / f'{split_name.lower()}_cm.png',
    )
    plt.close(fig)

    results = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
               for k, v in m.items() if k != 'confusion_matrix'}
    results['confusion_matrix'] = m['confusion_matrix'].tolist()
    results['predictions']      = y_pred.tolist()
    results['labels']           = y_true.tolist()

    with open(log_dir / f'{split_name.lower()}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results → {log_dir}/{split_name.lower()}_results.json")
    return m


# ── Plotting ───────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(
    cm        : np.ndarray,
    title     : str           = 'Confusion Matrix',
    save_path : Optional[Path] = None,
):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
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


def plot_fold_histories(
    fold_histories : List[Dict],
    save_path      : Optional[Path] = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, h in enumerate(fold_histories):
        axes[0].plot(h['train_loss'], label=f'Fold {i+1} train', linestyle='--')
        axes[0].plot(h['val_loss'],   label=f'Fold {i+1} val')
        axes[1].plot(h['train_acc'],  label=f'Fold {i+1} train', linestyle='--')
        axes[1].plot(h['val_acc'],    label=f'Fold {i+1} val')
        axes[2].plot(h['val_f1'],     label=f'Fold {i+1}')

    titles  = ['Loss', 'Accuracy', 'Val Macro F1']
    ylabels = ['Loss', 'Accuracy', 'Macro F1']
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig