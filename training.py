import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import KFold

from config import XGBConfig

CLASS_NAMES = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']


# ── Class weights ──────────────────────────────────────────────────────────────
def compute_sample_weights(y: np.ndarray, cfg: XGBConfig) -> np.ndarray:
    N, C    = len(y), cfg.num_classes
    cw      = np.zeros(C, dtype=np.float64)
    print(f"\nSample weights (power={cfg.weight_power}):")
    for c in range(C):
        count = (y == c).sum()
        pct   = 100 * count / N
        cw[c] = (count / N) ** (-cfg.weight_power) if count > 0 else 0.0
        print(f"  Class {c} ({CLASS_NAMES[c]:<10}): {count:>10,}  ({pct:5.1f}%)"
              f"  →  raw={cw[c]:.4f}")
    cw     /= cw.mean()          # normalise: mean weight = 1
    print(f"  Normalised : {cw.round(4).tolist()}")
    sw      = np.zeros(N, dtype=np.float32)
    for c in range(C):
        sw[y == c] = cw[c]
    return sw


# ── Model factory ──────────────────────────────────────────────────────────────
def make_xgb_model(cfg: XGBConfig) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators          = cfg.n_estimators,
        max_depth             = cfg.max_depth,
        min_child_weight      = cfg.min_child_weight,
        subsample             = cfg.subsample,
        colsample_bytree      = cfg.colsample_bytree,
        colsample_bylevel     = cfg.colsample_bylevel,
        gamma                 = cfg.gamma,
        reg_alpha             = cfg.reg_alpha,
        reg_lambda            = cfg.reg_lambda,
        learning_rate         = cfg.learning_rate,
        objective             = cfg.objective,
        eval_metric           = cfg.eval_metric,
        num_class             = cfg.num_classes,
        tree_method           = cfg.tree_method,
        device                = cfg.device,
        n_jobs                = cfg.n_jobs,
        random_state          = cfg.random_state,
        early_stopping_rounds = cfg.early_stopping_rounds,
        use_label_encoder     = False,
        verbosity             = 1,
    )


# ── Metrics ────────────────────────────────────────────────────────────────────
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

    # Critical recall: any Heavy or Extreme pixel correctly flagged
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
def train_kfold_xgb(
    X_train       : np.ndarray,          # (N, 35)
    y_train       : np.ndarray,          # (N,)
    cfg           : XGBConfig,
    feature_names : Optional[List[str]] = None,
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
    print(f"XGBoost {cfg.num_folds}-Fold Training  (patch-level split, mirrors ViT)")
    print(f"  Total patches : {len(X_train):,}")
    print(f"  Features      : {X_train.shape[1]}")
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

        sw_tr = compute_sample_weights(y_tr, cfg) if cfg.use_class_weights else None

        model = make_xgb_model(cfg)
        model.fit(
            X_tr, y_tr,
            sample_weight = sw_tr,
            eval_set      = [(X_vl, y_vl)],
            verbose       = 50,
        )

        best_ntree = model.best_iteration + 1
        print(f"\n  Best round : {best_ntree}  (of {cfg.n_estimators} max)")

        y_pred_vl   = model.predict(X_vl)
        val_metrics = compute_metrics(y_vl, y_pred_vl)
        val_f1      = val_metrics['macro_f1']
        print_metrics(val_metrics, split=f"Fold {fold+1} Validation")

        # Feature importance top-5
        fi = model.feature_importances_
        if feature_names:
            top5 = np.argsort(fi)[::-1][:5]
            print(f"\n  Top-5 features by gain:")
            for rank, idx in enumerate(top5):
                print(f"    {rank+1}. {feature_names[idx]:<22}  {fi[idx]:.4f}")

        # Save checkpoint as portable JSON
        ckpt_path = ckpt_dir / f'fold_{fold+1}_best.json'
        model.save_model(str(ckpt_path))
        print(f"  ✓ Checkpoint → {ckpt_path}")

        result = {
            'fold'               : fold + 1,
            'model'              : model,
            'best_ntree'         : best_ntree,
            'val_macro_f1'       : val_f1,
            'val_metrics'        : val_metrics,
            'feature_importance' : fi.tolist(),
            'elapsed_s'          : round(time.time() - t0, 1),
        }
        fold_results.append(result)

        if val_f1 > best_overall:
            best_overall  = val_f1
            best_fold_idx = fold

        print(f"\n  Fold {fold+1} complete  ({result['elapsed_s']}s)"
              f"  val macro_f1={val_f1:.4f}")

    print(f"\n{'='*60}")
    print(f"K-FOLD COMPLETE  |  Best fold: {best_fold_idx+1}"
          f"  |  Best val macro_f1: {best_overall:.4f}")
    print(f"{'='*60}")

    # Persist fold log
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
def evaluate_xgb(
    model      : xgb.XGBClassifier,
    X_test     : np.ndarray,
    y_test     : np.ndarray,
    cfg        : XGBConfig,
    split_name : str = 'Test',
) -> Dict:
    log_dir = cfg.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {split_name} evaluation on {len(X_test):,} patches...")
    y_pred = model.predict(X_test)
    m      = compute_metrics(y_test, y_pred)
    print_metrics(m, split=split_name)

    # Confidence distribution from predict_proba
    y_proba = model.predict_proba(X_test)          # (N, 5)
    conf    = y_proba.max(axis=1)                  # (N,) max softmax per patch
    print(f"\n  Confidence — mean={conf.mean():.4f}  median={np.median(conf):.4f}"
          f"  std={conf.std():.4f}  range=[{conf.min():.4f}, {conf.max():.4f}]")
    print(f"  Low(<0.5)={100*(conf<0.5).mean():.1f}%"
          f"  Mid={100*((conf>=0.5)&(conf<0.8)).mean():.1f}%"
          f"  High(>0.8)={100*(conf>0.8).mean():.1f}%")

    # Save results
    out = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in m.items()}
    with open(log_dir / f'{split_name.lower()}_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {log_dir}/{split_name.lower()}_results.json")

    _plot_confusion_matrix(
        m['confusion_matrix'],
        title     = f'Confusion Matrix — {split_name} (XGBoost)',
        save_path = log_dir / f'{split_name.lower()}_confusion_matrix.png',
    )
    return m


# ── Feature importance plot ────────────────────────────────────────────────────
def plot_feature_importance(
    model          : xgb.XGBClassifier,
    feature_names  : List[str],
    cfg            : XGBConfig,
    top_n          : int = 20,
    importance_type: str = 'gain',
) -> plt.Figure:

    log_dir = cfg.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    booster  = model.get_booster()
    scores   = booster.get_score(importance_type=importance_type)
    fi       = np.array([scores.get(f'f{i}', 0.0) for i in range(len(feature_names))])
    top_idx  = np.argsort(fi)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.36)))
    ax.barh(
        [feature_names[i] for i in reversed(top_idx)],
        [fi[i]            for i in reversed(top_idx)],
        color='steelblue', edgecolor='white', linewidth=0.5,
    )
    ax.set(xlabel=f'Importance ({importance_type})',
           title=f'Top-{top_n} Feature Importances — XGBoost ({importance_type})')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = log_dir / f'feature_importance_{importance_type}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Feature importance plot → {path}")
    return fig


# ── Fold summary plot ──────────────────────────────────────────────────────────
def plot_fold_histories_xgb(
    fold_results : List[Dict],
    cfg          : XGBConfig,
) -> plt.Figure:
    log_dir = cfg.output_dir / 'logs'
    folds   = [r['fold']         for r in fold_results]
    f1s     = [r['val_macro_f1'] for r in fold_results]
    ntrees  = [r['best_ntree']   for r in fold_results]
    best_f1 = max(f1s)
    colors  = ['#4CAF50' if f == best_f1 else '#2196F3' for f in f1s]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Val F1
    bars = axes[0].bar([f'Fold {i}' for i in folds], f1s, color=colors, edgecolor='white')
    axes[0].axhline(np.mean(f1s), color='red', linestyle='--',
                    label=f'Mean {np.mean(f1s):.4f}')
    axes[0].set(ylabel='Val Macro F1', title='Validation Macro F1 per Fold\n(⚠ optimistic — patch-level split)')
    axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)
    for f1, bar in zip(f1s, bars):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f'{f1:.4f}', ha='center', va='bottom', fontsize=9)

    # Best n_estimators
    axes[1].bar([f'Fold {i}' for i in folds], ntrees, color='#FF9800', edgecolor='white')
    axes[1].axhline(cfg.n_estimators, color='red', linestyle='--',
                    label=f'Max ({cfg.n_estimators})', alpha=0.7)
    axes[1].set(ylabel='Best n_estimators', title='Boosting Rounds at Early Stop')
    axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('XGBoost 2-Fold Cross-Validation Summary', fontsize=12, fontweight='bold')
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