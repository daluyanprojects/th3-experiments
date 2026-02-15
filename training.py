import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F  


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


class WarmupCosineScheduler:
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # Create cosine scheduler for post-warmup
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
    
    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.min_lr + (self.base_lr - self.min_lr) * (
                self.current_epoch / self.warmup_epochs
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine annealing
            self.cosine_scheduler.step()
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def compute_class_weights(labels: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    # Count samples per class
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Compute weights: inverse frequency
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("\nClass Weights:")
    class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    for i, (name, count, weight) in enumerate(zip(class_names, class_counts, class_weights)):
        print(f"  Class {i} ({name:12s}): {count:6,} samples, weight: {weight:.4f}")
    
    return torch.FloatTensor(class_weights)

def setup_training(model: nn.Module, train_labels: np.ndarray, device: torch.device, config: Dict) -> Tuple[nn.Module, optim.Optimizer, object, object]:
    print("\n" + "="*70)
    print("PHASE 10: TRAINING SETUP")
    print("="*70)
    
    is_cuda = device.type == 'cuda'
    if not is_cuda and config.get('use_amp', False):
        print("\n⚠ WARNING: Mixed precision disabled (CPU detected)")
        config['use_amp'] = False
    
    print("\n[10.1] Loss Function")
    print("-" * 70)
    
    # Compute class weights
    class_weights = compute_class_weights(train_labels, config['num_classes'])
    class_weights = class_weights.to(device)
    
    if config['loss_type'] == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=config['focal_gamma'])
        print(f"  Loss: Focal Loss (gamma={config['focal_gamma']})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        print(f"  Loss: Weighted Cross-Entropy")
        if config.get('label_smoothing', 0.0) > 0:
            print(f"  Label smoothing: {config['label_smoothing']}")
    
    print("\n[10.2] Optimizer")
    print("-" * 70)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    
    print("\n[10.3] Learning Rate Scheduler")
    print("-" * 70)
    
    if config['scheduler_type'] == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=config['warmup_epochs'],
            total_epochs=config['num_epochs'],
            base_lr=config['learning_rate'],
            min_lr=config['min_lr']
        )
        print(f"  Scheduler: Warmup + Cosine Annealing")
        print(f"  Warmup epochs: {config['warmup_epochs']}")
        print(f"  Min LR: {config['min_lr']}")
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience'),
            min_lr=config['min_lr']
        )
        print(f"  Scheduler: ReduceLROnPlateau")
        print(f"  Factor: 0.5, Patience: {config.get('scheduler_patience')}")
    
    print("\n[10.4-10.5] Training Configuration")
    print("-" * 70)
    print(f"  Device: {device}")
    print(f"  Total epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient clipping: {config.get('grad_clip_norm')}")
    print(f"  Early stopping patience: {config.get('early_stop_patience')}")
    print(f"  Mixed precision (AMP): {config.get('use_amp')}")
    
    if config.get('use_amp', False) and is_cuda:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    print("\n" + "="*70)
    print("TRAINING SETUP COMPLETE")
    print("="*70 + "\n")
    
    return criterion, optimizer, scheduler, scaler


def get_default_config(device: Optional[torch.device] = None) -> Dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    is_cuda = device.type == 'cuda'
    
    return {
        # Loss
        'loss_type': 'ce',  # 'ce' or 'focal'
        'focal_gamma': 2.0,
        'label_smoothing': 0.0,
        'num_classes': 5,
        
        # Optimizer
        'learning_rate': 3e-4,
        'weight_decay': 0.05,
        
        # Scheduler
        'scheduler_type': 'warmup_cosine',  # 'warmup_cosine' or 'plateau'
        'warmup_epochs': 1,
        'min_lr': 1e-6,
        'scheduler_patience': 10,
        
        # Training
        'num_epochs': 10,
        'batch_size': 256, 
        'grad_clip_norm': 1.0,
        'early_stop_patience': 50,
        'use_amp': is_cuda, 
        
        # Checkpointing
        'save_every': 10,
        'save_best': True
    }



def compute_metrics(y_true: np.ndarray,y_pred: np.ndarray,num_classes: int = 5) -> Dict:
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    
    for i, name in enumerate(class_names):
        metrics[f'precision_{name}'] = precision[i]
        metrics[f'recall_{name}'] = recall[i]
        metrics[f'f1_{name}'] = f1[i]
    
    # Critical class metrics (Heavy + Extreme)
    critical_mask = (y_true >= 3)
    if critical_mask.sum() > 0:
        metrics['critical_recall'] = recall_score(
            y_true[critical_mask] >= 3,
            y_pred[critical_mask] >= 3
        )
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_metrics(metrics: Dict, split: str = 'Validation'):    
    print(f"\n{'='*70}")
    print(f"{split} Metrics")
    print(f"{'='*70}")
    
    print(f"\nOverall:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
    
    if 'critical_recall' in metrics:
        print(f"  Critical Class Recall (Heavy+Extreme): {metrics['critical_recall']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print(f"  {'-'*44}")
    
    class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    for name in class_names:
        prec = metrics[f'precision_{name}']
        rec = metrics[f'recall_{name}']
        f1 = metrics[f'f1_{name}']
        print(f"  {name:<12} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")


def plot_confusion_matrix(cm: np.ndarray, title: str = 'Confusion Matrix',figsize: Tuple[int, int] = (8, 6), save_path: Optional[str] = None):
    
    class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    
    plt.figure(figsize=figsize)
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Confusion matrix saved: {save_path}")
    
    return plt.gcf()


def plot_training_history(history: Dict, save_path: Optional[str] = None,figsize: Tuple[int, int] = (15, 5)):    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Learning rate
    axes[2].plot(history['lr'], linewidth=2, color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule', fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Training history saved: {save_path}")
    
    return fig