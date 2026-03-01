import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import KFold

from config import SVMConfig

CLASS_NAMES = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']


# ── Pipeline factory ───────────────────────────────────────────────────────────
def make_svm_pipeline(cfg: SVMConfig) -> Pipeline:
    svc = SVC(
        kernel                  = cfg.kernel,
        C                       = cfg.C,
        gamma                   = cfg.gamma,
        degree                  = cfg.degree,
        coef0                   = cfg.coef0,
        tol                     = cfg.tol,
        max_iter                = cfg.max_iter,
        decision_function_shape = cfg.decision_function_shape,
        class_weight            = cfg.class_weight if cfg.class_weight != 'None' else None,
        probability             = cfg.probability,
        random_state            = cfg.random_state,
        cache_size              = 2000,   # MB — increase if RAM allows
    )
    steps = []
    if cfg.scale_features:
        steps.append(('scaler', StandardScaler()))
    steps.append(('svc', svc))
    return Pipeline(steps)


# ── Metrics (identical API to XGBoost version) ─────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    m = {
        'accuracy'    : float(accuracy_score(y_true, y_pred)),
        'macro_f1'    : float(f1_score(y_true, y_pred, average='macro',    zero_division=0)),
        'weighted_f1' : float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    labels = list(range(5))
    prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    rec  = recall_score(   y_true, y_pred, average=None, zero_division=0, labels=labels)
    f1   = f1_score(       y_true, y_pred, average=None, zero_division=0, labels=labels)
    for i, name in enumerate(CLASS_NAMES):
        m[f'precision_{name}'] = float(prec[i])
        m[f'recall_{name}']    = float(rec[i])
        m[f'f1_{name}']        = float(f1[i])

    # Critical recall: Heavy + Extreme pixels
    crit_mask = y_true >= 3
    if crit_mask.sum() > 0:
        m['critical_recall'] = float(recall_score(
            (y_true[crit_mask] >= 3).astype(int),
            (y_pred[crit_mask] >= 3).astype(int),
            zero_division=0,
        ))
    else:
        m['critical_recall'] = 0.0

    m['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels)
    return m


def print_metrics(m: Dict, split: str = 'Validation'):
    print(f"\n{'='*60}")
    print(f"{split} Metrics")
    print(f"{'='*60}")
    print(f"  Accuracy     : {m['accuracy']:.4f}")
    print(f"  Macro F1     : {m['macro_f1']:.4f}")
    print(f"  Weighted F1  : {m['weighted_f1']:.4f}")
    if 'critical_recall' in m:
        print(f"  Critical Recall (Heavy+Extreme): {m['critical_recall']:.4f}")
    print(f"\n  {'Class':<12} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'─'*38}")
    for name in CLASS_NAMES:
        print(f"  {name:<12} "
              f"{m[f'precision_{name}']:>8.4f} "
              f"{m[f'recall_{name}']:>8.4f} "
              f"{m[f'f1_{name}']:>8.4f}")


# ── K-Fold training ────────────────────────────────────────────────────────────
def train_kfold_svm(
    X_train        : np.ndarray,   # (N, 35)
    y_train        : np.ndarray,   # (N,)
    cfg            : SVMConfig,
    feature_names  : Optional[List[str]] = None,
) -> Tuple[List[Dict], int]:
    
    ckpt_dir = cfg.output_dir / 'checkpoints'
    log_dir  = cfg.output_dir / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.random_state)

    fold_results  : List[Dict] = []
    best_overall  = -1.0
    best_fold_idx = 0

    print(f"\n{'='*60}")
    print(f"SVM {cfg.num_folds}-Fold Training  (patch-level split, mirrors ViT/XGB)")
    print(f"  Total patches  : {len(X_train):,}")
    print(f"  Features       : {X_train.shape[1]}")
    print(f"  Kernel         : {cfg.kernel}  C={cfg.C}  gamma={cfg.gamma}")
    print(f"  Scaler         : {'StandardScaler' if cfg.scale_features else 'None'}")
    print(f"  Class weight   : {cfg.class_weight}")
    print(f"  ⚠  Scenario leakage across folds — val F1 is optimistic")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"FOLD {fold+1} / {cfg.num_folds}")
        print(f"  Train patches : {len(train_idx):,}")
        print(f"  Val   patches : {len(val_idx):,}")
        print(f"{'─'*60}")

        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_vl, y_vl = X_train[val_idx],   y_train[val_idx]

        model = make_svm_pipeline(cfg)

        print(f"  Fitting SVM …  (this may take a few minutes)")
        model.fit(X_tr, y_tr)

        y_pred_vl   = model.predict(X_vl)
        val_metrics = compute_metrics(y_vl, y_pred_vl)
        val_f1      = val_metrics['macro_f1']
        print_metrics(val_metrics, split=f"Fold {fold+1} Validation")

        # Save checkpoint via pickle
        ckpt_path = ckpt_dir / f'fold_{fold+1}_best.pkl'
        with open(ckpt_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✓ Checkpoint → {ckpt_path}")

        elapsed = round(time.time() - t0, 1)
        result = {
            'fold'        : fold + 1,
            'model'       : model,
            'val_macro_f1': val_f1,
            'val_metrics' : val_metrics,
            'elapsed_s'   : elapsed,
        }
        fold_results.append(result)

        if val_f1 > best_overall:
            best_overall  = val_f1
            best_fold_idx = fold

        print(f"\n  Fold {fold+1} complete  ({elapsed}s)  val macro_f1={val_f1:.4f}")

    print(f"\n{'='*60}")
    print(f"K-FOLD COMPLETE  |  Best fold: {best_fold_idx+1}"
          f"  |  Best val macro_f1: {best_overall:.4f}")
    print(f"{'='*60}")

    # Save fold log (exclude non-serialisable model objects)
    log = []
    for r in fold_results:
        entry = {k: v for k, v in r.items() if k != 'model'}
        cm = entry['val_metrics'].get('confusion_matrix')
        if cm is not None:
            entry['val_metrics']['confusion_matrix'] = (
                cm.tolist() if hasattr(cm, 'tolist') else cm
            )
        log.append(entry)
    with open(log_dir / 'fold_results.json', 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  Fold log → {log_dir / 'fold_results.json'}")

    return fold_results, best_fold_idx


# ── Test evaluation ────────────────────────────────────────────────────────────
def evaluate_svm(
    model      : Pipeline,
    X_test     : np.ndarray,
    y_test     : np.ndarray,
    cfg        : SVMConfig,
    split_name : str = 'Test',
) -> Dict:
    log_dir = cfg.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {split_name} evaluation on {len(X_test):,} patches…")
    y_pred = model.predict(X_test)
    m      = compute_metrics(y_test, y_pred)
    print_metrics(m, split=split_name)

    # Confidence from predict_proba (requires cfg.probability=True)
    if cfg.probability:
        y_proba = model.predict_proba(X_test)   # (N, 5)
        conf    = y_proba.max(axis=1)
        print(f"\n  Confidence — mean={conf.mean():.4f}  median={np.median(conf):.4f}"
              f"  std={conf.std():.4f}  range=[{conf.min():.4f}, {conf.max():.4f}]")
        print(f"  Low(<0.5)={100*(conf<0.5).mean():.1f}%"
              f"  Mid={100*((conf>=0.5)&(conf<0.8)).mean():.1f}%"
              f"  High(>0.8)={100*(conf>0.8).mean():.1f}%")
        m['y_proba'] = y_proba
    else:
        y_proba = None

    # Persist (excluding large arrays)
    out = {k: (v.tolist() if hasattr(v, 'tolist') else v)
           for k, v in m.items() if k != 'y_proba'}
    with open(log_dir / f'{split_name.lower()}_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {log_dir}/{split_name.lower()}_results.json")

    _plot_confusion_matrix(
        m['confusion_matrix'],
        title     = f'Confusion Matrix — {split_name} (SVM)',
        save_path = log_dir / f'{split_name.lower()}_confusion_matrix.png',
    )
    return m


# ── Fold summary plot ──────────────────────────────────────────────────────────
def plot_fold_summary_svm(
    fold_results : List[Dict],
    cfg          : SVMConfig,
) -> plt.Figure:
    log_dir = cfg.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    folds  = [r['fold']         for r in fold_results]
    f1s    = [r['val_macro_f1'] for r in fold_results]
    best_f1 = max(f1s)
    colors  = ['#4CAF50' if f == best_f1 else '#2196F3' for f in f1s]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([f'Fold {i}' for i in folds], f1s, color=colors, edgecolor='white')
    ax.axhline(np.mean(f1s), color='red', linestyle='--',
               label=f'Mean {np.mean(f1s):.4f}')
    ax.set(ylabel='Val Macro F1',
           title='SVM Validation Macro F1 per Fold\n(⚠ optimistic — patch-level split)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for f1, bar in zip(f1s, bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{f1:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('SVM Cross-Validation Summary', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = log_dir / 'fold_summary.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Fold plot → {path}")
    return fig


# ── Confusion matrix ───────────────────────────────────────────────────────────
def _plot_confusion_matrix(
    cm        : np.ndarray,
    title     : str = 'Confusion Matrix',
    save_path : Optional[Path] = None,
) -> plt.Figure:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set(title=title, ylabel='True Label', xlabel='Predicted Label')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
