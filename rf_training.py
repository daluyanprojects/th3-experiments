import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import KFold

from rf_config import RFConfig

CLASS_NAMES = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']


# ── Class / sample weights ────────────────────────────────────────────────────

def compute_sample_weights(y: np.ndarray, cfg: RFConfig) -> np.ndarray:
    N, C = len(y), cfg.num_classes
    cw   = np.zeros(C, dtype=np.float64)
    print(f"\nSample weights  (power={cfg.weight_power}):")
    for c in range(C):
        count = (y == c).sum()
        pct   = 100 * count / N
        cw[c] = (count / N) ** (-cfg.weight_power) if count > 0 else 0.0
        print(f"  Class {c} ({CLASS_NAMES[c]:<10}): {count:>10,}  ({pct:5.1f}%)"
              f"  ->  raw={cw[c]:.4f}")
    cw /= cw.mean()
    print(f"  Normalised : {cw.round(4).tolist()}")
    sw = np.zeros(N, dtype=np.float32)
    for c in range(C):
        sw[y == c] = cw[c]
    return sw


# ── Model factory ─────────────────────────────────────────────────────────────

def make_rf_model(cfg: RFConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators          = cfg.n_estimators,
        max_depth             = cfg.max_depth,
        min_samples_split     = cfg.min_samples_split,
        min_samples_leaf      = cfg.min_samples_leaf,
        max_features          = cfg.max_features,
        max_samples           = cfg.max_samples,
        bootstrap             = cfg.bootstrap,
        oob_score             = cfg.oob_score,
        min_impurity_decrease = cfg.min_impurity_decrease,
        ccp_alpha             = cfg.ccp_alpha,
        class_weight          = None,     # we pass sample_weight to fit() instead
        n_jobs                = cfg.n_jobs,
        random_state          = cfg.random_state,
        verbose               = cfg.verbose,
    )


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

    # Critical recall: Heavy + Extreme patches correctly flagged
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
    if 'oob_score' in m:
        print(f"  OOB Score    : {m['oob_score']:.4f}")
    print(f"\n  {'Class':<12} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'─'*38}")
    for name in CLASS_NAMES:
        print(f"  {name:<12} "
              f"{m[f'precision_{name}']:>8.4f} "
              f"{m[f'recall_{name}']:>8.4f} "
              f"{m[f'f1_{name}']:>8.4f}")


# ── K-Fold training ───────────────────────────────────────────────────────────

def train_kfold_rf(
    X_train       : np.ndarray,           # (N, 35)
    y_train       : np.ndarray,           # (N,)
    cfg           : RFConfig,
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
    print(f"Random Forest {cfg.num_folds}-Fold Training  (patch-level split, mirrors XGB/ViT)")
    print(f"  Total patches : {len(X_train):,}")
    print(f"  Features      : {X_train.shape[1]}")
    print(f"  Trees         : {cfg.n_estimators}")
    print(f"  WARNING: Scenario leakage across folds — val F1 is optimistic")
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

        model = make_rf_model(cfg)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)

        # OOB score (available since bootstrap=True)
        oob = model.oob_score_ if cfg.oob_score else None
        if oob is not None:
            print(f"\n  OOB accuracy  : {oob:.4f}  (in-bag estimate)")

        y_pred_vl   = model.predict(X_vl)
        val_metrics = compute_metrics(y_vl, y_pred_vl)
        if oob is not None:
            val_metrics['oob_score'] = oob
        val_f1 = val_metrics['macro_f1']
        print_metrics(val_metrics, split=f"Fold {fold+1} Validation")

        # Feature importance top-5 (MDI / Gini impurity decrease)
        fi = model.feature_importances_
        if feature_names:
            top5 = np.argsort(fi)[::-1][:5]
            print(f"\n  Top-5 features (MDI importance):")
            for rank, idx in enumerate(top5):
                print(f"    {rank+1}. {feature_names[idx]:<22}  {fi[idx]:.4f}")

        # Save checkpoint as .pkl (RF has no native portable format like XGB JSON)
        ckpt_path = ckpt_dir / f'fold_{fold+1}_best.pkl'
        with open(ckpt_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Checkpoint -> {ckpt_path}")

        result = {
            'fold'               : fold + 1,
            'model'              : model,
            'oob_score'          : oob,
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

    # Persist fold log (JSON — strip non-serialisable objects)
    log = []
    for r in fold_results:
        entry = {k: v for k, v in r.items() if k not in ('model',)}
        cm = entry['val_metrics'].get('confusion_matrix')
        if cm is not None:
            entry['val_metrics']['confusion_matrix'] = (
                cm.tolist() if hasattr(cm, 'tolist') else cm
            )
        log.append(entry)
    with open(log_dir / 'fold_results.json', 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  Fold log -> {log_dir / 'fold_results.json'}")

    return fold_results, best_fold_idx


# ── Test evaluation ───────────────────────────────────────────────────────────

def evaluate_rf(
    model      : RandomForestClassifier,
    X_test     : np.ndarray,
    y_test     : np.ndarray,
    cfg        : RFConfig,
    split_name : str = 'Test',
) -> Dict:
    log_dir = cfg.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {split_name} evaluation on {len(X_test):,} patches...")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)   # (N, 5)  — RF outputs calibrated probs
    m       = compute_metrics(y_test, y_pred)
    print_metrics(m, split=split_name)

    # Confidence distribution
    conf = y_proba.max(axis=1)
    print(f"\n  Confidence — mean={conf.mean():.4f}  median={np.median(conf):.4f}"
          f"  std={conf.std():.4f}  range=[{conf.min():.4f}, {conf.max():.4f}]")
    print(f"  Low(<0.5)={100*(conf<0.5).mean():.1f}%"
          f"  Mid={100*((conf>=0.5)&(conf<0.8)).mean():.1f}%"
          f"  High(>0.8)={100*(conf>0.8).mean():.1f}%")

    # Save results
    out = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in m.items()}
    with open(log_dir / f'{split_name.lower()}_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Results -> {log_dir}/{split_name.lower()}_results.json")

    _plot_confusion_matrix(
        m['confusion_matrix'],
        title     = f'Confusion Matrix — {split_name} (Random Forest)',
        save_path = log_dir / f'{split_name.lower()}_confusion_matrix.png',
    )
    return m


# ── Feature importance plot ───────────────────────────────────────────────────

def plot_feature_importance(
    model         : RandomForestClassifier,
    feature_names : List[str],
    cfg           : RFConfig,
    top_n         : int = 20,
) -> plt.Figure:
    log_dir = cfg.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Per-tree importances -> mean + std
    fi_per_tree = np.array([t.feature_importances_ for t in model.estimators_])
    fi_mean = fi_per_tree.mean(axis=0)
    fi_std  = fi_per_tree.std(axis=0)

    top_idx = np.argsort(fi_mean)[::-1][:top_n]
    names   = [feature_names[i] for i in reversed(top_idx)]
    means   = [fi_mean[i]       for i in reversed(top_idx)]
    stds    = [fi_std[i]        for i in reversed(top_idx)]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.36)))
    ax.barh(names, means, xerr=stds, color='forestgreen',
            edgecolor='white', linewidth=0.5,
            error_kw={'ecolor': 'black', 'capsize': 3, 'linewidth': 1})
    ax.set(xlabel='MDI Importance (mean +/- std across trees)',
           title=f'Top-{top_n} Feature Importances — Random Forest (MDI)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = log_dir / 'feature_importance_mdi.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Feature importance plot -> {path}")
    return fig


# ── Fold summary plot ─────────────────────────────────────────────────────────

def plot_fold_summary_rf(
    fold_results : List[Dict],
    cfg          : RFConfig,
) -> plt.Figure:
    log_dir = cfg.output_dir / 'logs'
    folds   = [r['fold']         for r in fold_results]
    f1s     = [r['val_macro_f1'] for r in fold_results]
    oobs    = [r['oob_score'] if r['oob_score'] is not None else 0.0
               for r in fold_results]
    best_f1 = max(f1s)
    colors  = ['#4CAF50' if f == best_f1 else '#2196F3' for f in f1s]

    has_oob = any(r['oob_score'] is not None for r in fold_results)
    ncols   = 2 if has_oob else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    # Val macro F1
    bars = axes[0].bar([f'Fold {i}' for i in folds], f1s, color=colors, edgecolor='white')
    axes[0].axhline(np.mean(f1s), color='red', linestyle='--',
                    label=f'Mean {np.mean(f1s):.4f}')
    axes[0].set(ylabel='Val Macro F1',
                title='Validation Macro F1 per Fold\n(optimistic — patch-level split)')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    for f1, bar in zip(f1s, bars):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f'{f1:.4f}', ha='center', va='bottom', fontsize=9)

    # OOB accuracy (RF-specific)
    if has_oob:
        axes[1].bar([f'Fold {i}' for i in folds], oobs, color='#FF9800', edgecolor='white')
        axes[1].axhline(np.mean(oobs), color='red', linestyle='--',
                        label=f'Mean {np.mean(oobs):.4f}')
        axes[1].set(ylabel='OOB Accuracy',
                    title='Out-of-Bag Accuracy per Fold\n(unbiased in-training estimate)')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('Random Forest 2-Fold Cross-Validation Summary',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = log_dir / 'fold_summary.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Fold plot -> {path}")
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm        : np.ndarray,
    title     : str = 'Confusion Matrix',
    save_path : Optional[Path] = None,
) -> plt.Figure:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set(title=title, ylabel='True Label', xlabel='Predicted Label')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
