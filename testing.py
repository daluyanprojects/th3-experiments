import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from typing import Dict
from torch.utils.data import DataLoader, TensorDataset

FLOOD_CLASSES = {
    0: {'name': 'No Flood',  'color': '#FFFFFF', 'hex': 'FFFFFF'},
    1: {'name': 'Light',     'color': '#FFEB3B', 'hex': 'FFEB3B'},
    2: {'name': 'Moderate',  'color': '#FF9800', 'hex': 'FF9800'},
    3: {'name': 'Heavy',     'color': '#F44336', 'hex': 'F44336'},
    4: {'name': 'Extreme',   'color': '#9C27B0', 'hex': '9C27B0'},
}
CLASS_NAMES = [FLOOD_CLASSES[i]['name'] for i in range(5)]
FLOOD_CMAP  = ListedColormap([FLOOD_CLASSES[i]['color'] for i in range(5)])
FLOOD_NORM  = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], FLOOD_CMAP.N)

def compute_iou_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> np.ndarray:
    iou = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        denom = tp + fp + fn
        iou[c] = float(tp) / denom if denom > 0 else 0.0
    return iou

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> Dict:
    labels   = list(range(num_classes))
    acc      = accuracy_score(y_true, y_pred)
    prec_pc  = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec_pc   = recall_score(   y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_pc    = f1_score(       y_true, y_pred, average=None, labels=labels, zero_division=0)
    iou_pc   = compute_iou_per_class(y_true, y_pred, num_classes)

    return {
        'accuracy':          acc,
        'precision_macro':   float(prec_pc.mean()),
        'recall_macro':      float(rec_pc.mean()),
        'f1_macro':          float(f1_pc.mean()),
        'iou_macro':         float(iou_pc.mean()),
        'precision_class':   prec_pc,
        'recall_class':      rec_pc,
        'f1_class':          f1_pc,
        'iou_class':         iou_pc,
        'confusion_matrix':  confusion_matrix(y_true, y_pred, labels=labels),
    }

def prepare_ground_truth(y_test: np.ndarray, quadrant_indices: Dict, patch_metadata: Dict) -> Dict:
    print("\nPreparing ground truth maps for test scenarios...")
    N_scenarios, N_test = y_test.shape
    full_n_h = patch_metadata['num_patches_h']
    full_n_w = patch_metadata['num_patches_w']
    n_h = full_n_h - full_n_h // 2
    n_w = full_n_w // 2

    gt_maps = np.zeros((N_scenarios, n_h, n_w), dtype=np.int8)
    for s in range(N_scenarios):
        gt_maps[s] = y_test[s].reshape(n_h, n_w)

    print(f"  Q3 patch grid : {n_h} rows × {n_w} cols = {n_h*n_w} patches")
    print(f"  GT maps shape : {gt_maps.shape}  ✓")
    return {'flat': y_test, 'maps': gt_maps, 'n_h': n_h, 'n_w': n_w}

def predict_scenario(model, X_test, rain_normalized, scenario_idx, batch_size, device):
    model.eval()
    X_t    = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    rain_t = (torch.tensor(rain_normalized[scenario_idx], dtype=torch.float32)
              .unsqueeze(0).expand(len(X_test), -1))
    loader = DataLoader(TensorDataset(X_t, rain_t), batch_size=batch_size, shuffle=False)
    preds, confs = [], []
    with torch.no_grad():
        for sp, ra in loader:
            logits, _ = model(sp.to(device), ra.to(device))
            probs     = torch.softmax(logits, dim=1)
            preds.append(probs.argmax(1).cpu().numpy())
            confs.append(probs.max(1).values.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)

def evaluate_all_scenarios(model, X_test, y_test, conditioning_vectors, ground_truth, batch_size, device, num_classes) -> Dict:
    print("\nEvaluating all test scenarios...")
    N_scenarios = conditioning_vectors.shape[0]
    n_h, n_w    = ground_truth['n_h'], ground_truth['n_w']
    per_scenario, pred_maps = [], np.zeros((N_scenarios, n_h, n_w), dtype=np.int8)

    conf_maps = np.zeros((N_scenarios, n_h, n_w), dtype=np.float32)   # NEW

    for s in range(N_scenarios):
        preds, confs = predict_scenario(model, X_test, conditioning_vectors, s, batch_size, device)
        labels = ground_truth['flat'][s]
        m      = compute_metrics(labels, preds, num_classes)
        per_scenario.append({'scenario_id': s + 1, **m,
                             'preds_flat': preds, 'confs_flat': confs})   # confs added
        pred_maps[s] = preds.reshape(n_h, n_w)
        conf_maps[s] = confs.reshape(n_h, n_w)                           # NEW
        frac_low = (confs < 0.5).mean() * 100
        print(f"  RS{s+1:02d}  Acc={m['accuracy']:.4f}  Prec={m['precision_macro']:.4f}  "
              f"Rec={m['recall_macro']:.4f}  F1={m['f1_macro']:.4f}  IoU={m['iou_macro']:.4f}"
              f"  Conf={confs.mean():.3f}  Uncertain={frac_low:.1f}%")
        
    def agg(key):
        v = [r[key] for r in per_scenario]
        return float(np.mean(v)), float(np.std(v))

    aggregate = {
        'accuracy':         agg('accuracy'),
        'precision_macro':  agg('precision_macro'),
        'recall_macro':     agg('recall_macro'),
        'f1_macro':         agg('f1_macro'),
        'iou_macro':        agg('iou_macro'),
        'f1_class_avg':     np.mean([r['f1_class']  for r in per_scenario], axis=0),
        'iou_class_avg':    np.mean([r['iou_class'] for r in per_scenario], axis=0),
        'prec_class_avg':   np.mean([r['precision_class'] for r in per_scenario], axis=0),
        'rec_class_avg':    np.mean([r['recall_class']    for r in per_scenario], axis=0),
        'best_scenario':    int(np.argmax([r['f1_macro'] for r in per_scenario])) + 1,
        'worst_scenario':   int(np.argmin([r['f1_macro'] for r in per_scenario])) + 1,
    }

    print("\n" + "="*65)
    print("EVALUATION SUMMARY  —  Q3 Test Region (50 Scenarios)")
    print("="*65)
    for key, label in [('accuracy','Accuracy'), ('precision_macro','Precision (macro)'),
                       ('recall_macro','Recall (macro)'), ('f1_macro','F1 (macro)'),
                       ('iou_macro','IoU (macro)')]:
        m, s = aggregate[key]
        print(f"  {label:22s}: {m:.4f} ± {s:.4f}")
    print(f"  Best  scenario        : RS{aggregate['best_scenario']}")
    print(f"  Worst scenario        : RS{aggregate['worst_scenario']}")
    print("="*65)

    return {
        'per_scenario': per_scenario,
        'aggregate':    aggregate,
        'pred_maps':    pred_maps,
        'conf_maps':    conf_maps,          
        'gt_maps':      ground_truth['maps'],
        'n_h': n_h, 'n_w': n_w,
    }

def plot_metrics_summary(results: Dict) -> plt.Figure:
    agg    = results['aggregate']
    keys   = ['accuracy','precision_macro','recall_macro','f1_macro','iou_macro']
    labels = ['Accuracy','Precision\n(macro)','Recall\n(macro)','F1\n(macro)','IoU\n(macro)']
    colors = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0']
    means  = [agg[k][0] for k in keys]
    stds   = [agg[k][1] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85,
                  error_kw={'linewidth':2,'ecolor':'black'}, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance — Q3 Test Region  (50 Scenarios, Macro Average)",
                 fontsize=12, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.35)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.015,
                f'{m:.3f}', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_per_scenario_metrics(results: Dict) -> plt.Figure:
    per  = results['per_scenario']
    scen = [r['scenario_id'] for r in per]
    series = {
        'Accuracy':          [r['accuracy']        for r in per],
        'Precision (macro)': [r['precision_macro']  for r in per],
        'Recall (macro)':    [r['recall_macro']     for r in per],
        'F1 (macro)':        [r['f1_macro']         for r in per],
        'IoU (macro)':       [r['iou_macro']        for r in per],
    }
    colors  = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0']
    markers = ['o','s','^','D','v']

    fig, ax = plt.subplots(figsize=(15, 5))
    for (name, vals), color, marker in zip(series.items(), colors, markers):
        ax.plot(scen, vals, label=name, color=color,
                marker=marker, markersize=3, linewidth=1.5)

    best  = results['aggregate']['best_scenario']
    worst = results['aggregate']['worst_scenario']
    ax.axvline(best,  color='green', linestyle='--', alpha=0.5, label=f'Best (RS{best})')
    ax.axvline(worst, color='red',   linestyle='--', alpha=0.5, label=f'Worst (RS{worst})')
    ax.set_xlabel("Rainfall Scenario", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(scen[::5])
    ax.set_xticklabels([f'RS{s}' for s in scen[::5]], rotation=45, fontsize=8)
    ax.set_title("Per-Scenario Metrics — Q3 Test Region", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=4); ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig

def plot_per_class_metrics(results: Dict) -> plt.Figure:
    agg  = results['aggregate']
    prec = agg['prec_class_avg']
    rec  = agg['rec_class_avg']
    f1   = agg['f1_class_avg']
    iou  = agg['iou_class_avg']

    x, width = np.arange(5), 0.2
    fig, ax  = plt.subplots(figsize=(11, 5))
    ax.bar(x - 1.5*width, prec, width, label='Precision', color='#4CAF50', alpha=0.85)
    ax.bar(x - 0.5*width, rec,  width, label='Recall',    color='#FF9800', alpha=0.85)
    ax.bar(x + 0.5*width, f1,   width, label='F1',        color='#F44336', alpha=0.85)
    ax.bar(x + 1.5*width, iou,  width, label='IoU',       color='#9C27B0', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score (avg over 50 scenarios)", fontsize=11)
    ax.set_title("Per-Class Metrics — Q3 Test Region", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.25)
    tick_colors = ['#777777','#B8A000','#CC7700','#C62828','#6A1B9A']
    for tick, c in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(c); tick.set_fontweight('bold')
    plt.tight_layout()
    return fig

def plot_flood_on_dem(dem_q3: np.ndarray, flood_map: np.ndarray, scenario_id: int, metrics: Dict, mode: str, figsize: tuple, inset_center: tuple, inset_size: int) -> plt.Figure:
    H, W     = dem_q3.shape
    n_h, n_w = flood_map.shape

    # ── Upsample flood map to DEM resolution ─────────────────────────────────
    ph = H / n_h   
    pw = W / n_w  
    flood_full = np.zeros((H, W), dtype=np.float32)
    for i in range(n_h):
        for j in range(n_w):
            r0, r1 = int(i * ph), int((i+1) * ph)
            c0, c1 = int(j * pw), int((j+1) * pw)
            flood_full[r0:r1, c0:c1] = flood_map[i, j]

    # ── RGBA flood overlay ────────────────────────────────────────────────────
    hex_to_rgb = lambda h: tuple(int(h[i:i+2], 16)/255.0 for i in (1,3,5))
    flood_rgba = np.zeros((H, W, 4), dtype=np.float32)
    class_alphas = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    for cls, info in FLOOD_CLASSES.items():
        mask = flood_full == cls
        r, g, b = hex_to_rgb(info['color'])
        flood_rgba[mask, 0] = r
        flood_rgba[mask, 1] = g
        flood_rgba[mask, 2] = b
        flood_rgba[mask, 3] = class_alphas[cls]

    # ── Inset center: default to centroid of highest flood class ─────────────
    if inset_center is None:
        for cls in [4, 3, 2, 1]:
            hi = np.argwhere(flood_map == cls)
            if len(hi) > 0:
                pc = hi[len(hi)//2]
                inset_center = (int(pc[0] * ph + ph/2),
                                int(pc[1] * pw + pw/2))
                break
        if inset_center is None:
            inset_center = (H//2, W//2)

    ic_r, ic_c = inset_center
    iz = inset_size
    ir0 = max(0, ic_r - iz); ir1 = min(H, ic_r + iz)
    ic0 = max(0, ic_c - iz); ic1 = min(W, ic_c + iz)

    # ── DEM display range ─────────────────────────────────────────────────────
    dem_p2, dem_p98 = np.percentile(dem_q3, 2), np.percentile(dem_q3, 98)
    fig = plt.figure(figsize=figsize, facecolor='#1a1a1a')
    ax  = fig.add_axes([0.08, 0.08, 0.72, 0.84], facecolor='#1a1a1a')

    # Layer 1: DEM
    dem_im = ax.imshow(dem_q3, cmap='terrain', vmin=dem_p2, vmax=dem_p98, origin='upper', interpolation='bilinear', alpha=0.4)
    # Layer 2: Flood overlay — fully opaque colors, No Flood transparent
    ax.imshow(flood_rgba, origin='upper', interpolation='nearest')
    # Inset red box marker
    from matplotlib.patches import Rectangle
    rect = Rectangle((ic0, ir0), ic1-ic0, ir1-ir0, linewidth=2, edgecolor='red', facecolor='none', zorder=5)
    ax.add_patch(rect)

    # Axes styling — white text on dark bg
    ax.set_xlabel("Column (West  ->  East)", fontsize=9, color='white')
    ax.set_ylabel("Row (North  ->  South)",  fontsize=9, color='white')
    ax.tick_params(labelsize=8, colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    mode_label = "Predicted" if mode == 'pred' else "Ground Truth"
    title = f"Metro Manila Q3 -- {mode_label} Flood  (RS{scenario_id})"
    if metrics:
        m = metrics
        title += (f"\nAcc={m['accuracy']:.3f}  Prec={m['precision_macro']:.3f}  "
                  f"Rec={m['recall_macro']:.3f}  F1={m['f1_macro']:.3f}  "
                  f"IoU={m['iou_macro']:.3f}")
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='white')

    # ── DEM colorbar ─────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.82, 0.35, 0.025, 0.50])
    cbar = fig.colorbar(dem_im, cax=cbar_ax)
    cbar.set_label("Elevation (m)", fontsize=9, color='white')
    cbar.ax.tick_params(labelsize=8, colors='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # ── Flood class legend ────────────────────────────────────────────────────
    legend_ax = fig.add_axes([0.81, 0.08, 0.17, 0.22])
    legend_ax.set_facecolor('#1a1a1a')
    legend_ax.axis('off')
    legend_ax.set_title("Flood Class", fontsize=8, fontweight='bold',
                        pad=4, color='white')
    for i, (cls, info) in enumerate(reversed(list(FLOOD_CLASSES.items()))):
        color = info['color'] if cls > 0 else '#555555'   
        legend_ax.add_patch(
            plt.Rectangle((0.0, i * 0.19), 0.22, 0.15,
                          facecolor=color, edgecolor='#aaaaaa', linewidth=0.6,
                          transform=legend_ax.transAxes, clip_on=False)
        )
        legend_ax.text(0.28, i * 0.19 + 0.075, info['name'], transform=legend_ax.transAxes, va='center', fontsize=7.5, color='white') 

    # ── Inset zoom panel ─────────────────────────────────────────────────────
    inset_ax = fig.add_axes([0.44, 0.08, 0.28, 0.28], facecolor='#1a1a1a')
    inset_ax.imshow(dem_q3[ir0:ir1, ic0:ic1], cmap='terrain', vmin=dem_p2, vmax=dem_p98, origin='upper', interpolation='bilinear', alpha=0.4)
    inset_ax.imshow(flood_rgba[ir0:ir1, ic0:ic1], origin='upper', interpolation='nearest')
    inset_ax.set_xticks([]); inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor('red'); spine.set_linewidth(2)
    return fig

def plot_best_worst_dem(dem_q3:   np.ndarray, results:  Dict, figsize:  tuple = (18, 11) ) -> plt.Figure:
    best_idx  = results['aggregate']['best_scenario']  - 1
    worst_idx = results['aggregate']['worst_scenario'] - 1
    n_h, n_w  = results['n_h'], results['n_w']

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Best vs Worst Scenario — Q3 DEM + Flood Overlay", fontsize=14, fontweight='bold', y=1.01)

    H, W = dem_q3.shape
    ph, pw = H / n_h, W / n_w
    hex_to_rgb = lambda h: tuple(int(h[i:i+2], 16)/255.0 for i in (1,3,5))
    class_alphas = {0:0.0, 1:0.55, 2:0.65, 3:0.75, 4:0.88}
    dem_p2, dem_p98 = np.percentile(dem_q3, 2), np.percentile(dem_q3, 98)

    def make_rgba(fmap):
        flood_full = np.zeros((H, W), dtype=np.float32)
        for i in range(n_h):
            for j in range(n_w):
                r0,r1 = int(i*ph), int((i+1)*ph)
                c0,c1 = int(j*pw), int((j+1)*pw)
                flood_full[r0:r1, c0:c1] = fmap[i, j]
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        for cls, info in FLOOD_CLASSES.items():
            mask = flood_full == cls
            r,g,b = hex_to_rgb(info['color'])
            rgba[mask,0]=r; rgba[mask,1]=g; rgba[mask,2]=b
            rgba[mask,3] = class_alphas[cls]
        return rgba

    configs = [
        (0, 0, best_idx,  'gt',   'Best  GT'),
        (0, 1, best_idx,  'pred', 'Best  Predicted'),
        (1, 0, worst_idx, 'gt',   'Worst GT'),
        (1, 1, worst_idx, 'pred', 'Worst Predicted'),
    ]

    for row, col, s_idx, mode, label in configs:
        ax = axes[row, col]
        fmap = results['gt_maps'][s_idx] if mode == 'gt' else results['pred_maps'][s_idx]
        m    = results['per_scenario'][s_idx]
        rgba = make_rgba(fmap)

        im = ax.imshow(dem_q3, cmap='terrain', vmin=dem_p2, vmax=dem_p98, origin='upper', interpolation='bilinear')
        ax.imshow(rgba, origin='upper', interpolation='nearest')

        title = (f"RS{m['scenario_id']} — {label}\n"
                 f"Acc={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}  "
                 f"IoU={m['iou_macro']:.3f}")
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel("Column", fontsize=8)
        ax.set_ylabel("Row", fontsize=8)
        ax.tick_params(labelsize=7)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("Elevation (m)", fontsize=9)

    # Legend
    patches = [mpatches.Patch(color=FLOOD_CLASSES[c]['color'] if c>0 else '#DDD',
                               label=FLOOD_CLASSES[c]['name']) for c in range(5)]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    return fig

def plot_confidence_on_dem(
    dem_q3      : np.ndarray,
    conf_map    : np.ndarray,  
    scenario_id : int,
    config_label: str,
    confs_flat  : np.ndarray,   
    figsize     : tuple = (9, 11),
    inset_center: tuple = None,
    inset_size  : int   = 60,
) -> plt.Figure:
    H, W     = dem_q3.shape
    n_h, n_w = conf_map.shape

    # ── Upsample confidence map to DEM resolution ─────────────────────────────
    ph = H / n_h
    pw = W / n_w
    conf_full = np.zeros((H, W), dtype=np.float32)
    for i in range(n_h):
        for j in range(n_w):
            r0, r1 = int(i * ph), int((i+1) * ph)
            c0, c1 = int(j * pw), int((j+1) * pw)
            conf_full[r0:r1, c0:c1] = conf_map[i, j]

    # ── Inset center: default to region of lowest confidence ─────────────────
    if inset_center is None:
        low_patches = np.argwhere(conf_map < 0.5)
        if len(low_patches) > 0:
            pc = low_patches[len(low_patches) // 2]
            inset_center = (int(pc[0] * ph + ph/2), int(pc[1] * pw + pw/2))
        else:
            inset_center = (H//2, W//2)

    ic_r, ic_c = inset_center
    iz  = inset_size
    ir0 = max(0, ic_r - iz); ir1 = min(H, ic_r + iz)
    ic0 = max(0, ic_c - iz); ic1 = min(W, ic_c + iz)

    dem_p2, dem_p98 = np.percentile(dem_q3, 2), np.percentile(dem_q3, 98)
    frac_low        = (confs_flat < 0.5).mean() * 100

    fig = plt.figure(figsize=figsize, facecolor='#1a1a1a')
    ax  = fig.add_axes([0.08, 0.08, 0.72, 0.84], facecolor='#1a1a1a')

    # Layer 1: DEM
    dem_im = ax.imshow(dem_q3, cmap='terrain', vmin=dem_p2, vmax=dem_p98,
                       origin='upper', interpolation='bilinear', alpha=0.5)

    # Layer 2: Confidence heatmap
    conf_im = ax.imshow(conf_full, cmap='RdYlGn', vmin=0.0, vmax=1.0,
                        origin='upper', interpolation='nearest', alpha=0.65)

    # Layer 3: Uncertain patch overlay (conf < 0.5)
    uncertain = conf_full.copy()
    uncertain[conf_full >= 0.5] = np.nan
    uncertain[conf_full <  0.5] = 1.0
    ax.imshow(uncertain, cmap='cool', vmin=0, vmax=1, alpha=0.35,
              origin='upper', interpolation='nearest')

    # Inset marker
    from matplotlib.patches import Rectangle
    rect = Rectangle((ic0, ir0), ic1-ic0, ir1-ir0,
                     linewidth=2, edgecolor='red', facecolor='none', zorder=5)
    ax.add_patch(rect)

    # Axes styling
    ax.set_xlabel("Column (West  ->  East)", fontsize=9, color='white')
    ax.set_ylabel("Row (North  ->  South)",  fontsize=9, color='white')
    ax.tick_params(labelsize=8, colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    title = (f"Metro Manila Q3 — Model Confidence  (RS{scenario_id})\n"
             f"{config_label}\n"
             f"Mean={confs_flat.mean():.3f}  Std={confs_flat.std():.3f}  "
             f"Min={confs_flat.min():.3f}  Uncertain={frac_low:.1f}%")
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10, color='white')

    # ── DEM colorbar ──────────────────────────────────────────────────────────
    cbar_ax1 = fig.add_axes([0.82, 0.55, 0.025, 0.37])
    cbar1    = fig.colorbar(dem_im, cax=cbar_ax1)
    cbar1.set_label("Elevation (m)", fontsize=8, color='white')
    cbar1.ax.tick_params(labelsize=7, colors='white')
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color='white')

    # ── Confidence colorbar ───────────────────────────────────────────────────
    cbar_ax2 = fig.add_axes([0.82, 0.10, 0.025, 0.37])
    cbar2    = fig.colorbar(conf_im, cax=cbar_ax2)
    cbar2.set_label("Confidence", fontsize=8, color='white')
    cbar2.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar2.ax.tick_params(labelsize=7, colors='white')
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')
    cbar2.ax.axhline(y=0.5, color='white', linewidth=1.2, linestyle='--')
    cbar2.ax.text(2.5, 0.5, 'uncertain\nthreshold',
                  fontsize=6, va='center', color='white')

    # ── Inset zoom panel ──────────────────────────────────────────────────────
    inset_ax = fig.add_axes([0.44, 0.08, 0.28, 0.28], facecolor='#1a1a1a')
    inset_ax.imshow(dem_q3[ir0:ir1, ic0:ic1], cmap='terrain',
                    vmin=dem_p2, vmax=dem_p98, origin='upper',
                    interpolation='bilinear', alpha=0.5)
    inset_ax.imshow(conf_full[ir0:ir1, ic0:ic1], cmap='RdYlGn',
                    vmin=0.0, vmax=1.0, origin='upper',
                    interpolation='nearest', alpha=0.65)
    unc_inset = uncertain[ir0:ir1, ic0:ic1]
    inset_ax.imshow(unc_inset, cmap='cool', alpha=0.35,
                    vmin=0, vmax=1, origin='upper', interpolation='nearest')
    inset_ax.set_xticks([]); inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor('red')
        spine.set_linewidth(2)

    return fig