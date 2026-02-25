import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_recall_fscore_support)

FLOOD_CLASSES = {
    0: {'name': 'No Flood', 'color': '#FFFFFF', 'hex': 'FFFFFF'},
    1: {'name': 'Light',    'color': '#FFEB3B', 'hex': 'FFEB3B'},
    2: {'name': 'Moderate', 'color': '#FF9800', 'hex': 'FF9800'},
    3: {'name': 'Heavy',    'color': '#F44336', 'hex': 'F44336'},
    4: {'name': 'Extreme',  'color': '#9C27B0', 'hex': '9C27B0'},
}


# ─────────────────────────────────────────────────────────────────────────────
# Map reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_test_map_from_patches(
    patch_predictions: np.ndarray,
    original_shape   : Tuple[int, int],
    patch_size       : int,
) -> np.ndarray:
    H, W       = original_shape
    patches_h  = H // patch_size
    patches_w  = W // patch_size
    # Reshape from flat (e.g. 6400,) to grid (e.g. 80×80)
    patch_grid = patch_predictions.reshape(patches_h, patches_w)
    # Expand each patch to patch_size × patch_size pixels
    reconstructed_map = np.repeat(
        np.repeat(patch_grid, patch_size, axis=0), patch_size, axis=1
    )
    return reconstructed_map


# ─────────────────────────────────────────────────────────────────────────────
# Prediction — returns class labels AND per-patch confidence
# ─────────────────────────────────────────────────────────────────────────────

def predict_scenario_with_flags(
    model            : nn.Module,
    scenario_dataset,
    scenario_idx     : int,
    device           : str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference for a single scenario.

    Returns
    -------
    patch_predictions : np.ndarray, shape (patches_per_scenario,)
        Argmax class index for every patch.
    patch_confidences : np.ndarray, shape (patches_per_scenario,)
        Max softmax probability for every patch (0-1). Higher = more certain.
    """
    model.eval()
    patches_per_scenario = scenario_dataset.patches_per_scenario
    start_idx            = scenario_idx * patches_per_scenario
    end_idx              = start_idx + patches_per_scenario
    patch_predictions    = []
    patch_confidences    = []

    with torch.no_grad():
        for idx in range(start_idx, end_idx):
            spatial, conditioning, _ = scenario_dataset[idx]

            spatial      = spatial.unsqueeze(0).to(device)
            conditioning = conditioning.unsqueeze(0).to(device)

            logits, _  = model(spatial, conditioning)

            # softmax once → reuse for both argmax and max-prob confidence
            probs      = torch.softmax(logits, dim=1)          # (1, num_classes)
            pred       = probs.argmax(dim=1).item()
            confidence = probs.max(dim=1).values.item()        # max prob = confidence

            patch_predictions.append(pred)
            patch_confidences.append(confidence)

    return np.array(patch_predictions), np.array(patch_confidences, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_iou_per_class(
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    num_classes: int = 5,
) -> Dict[int, float]:
    iou_scores = {}
    for class_id in range(num_classes):
        true_mask    = (y_true == class_id)
        pred_mask    = (y_pred == class_id)
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union        = np.logical_or(true_mask, pred_mask).sum()
        iou_scores[class_id] = 0.0 if union == 0 else float(intersection / union)
    return iou_scores


def calculate_scenario_metrics(
    ground_truth_map: np.ndarray,
    predicted_map   : np.ndarray,
    confidence_map  : np.ndarray = None,   # optional — attached when available
    num_classes     : int        = 5,
) -> Dict:
    y_true = ground_truth_map.flatten()
    y_pred = predicted_map.flatten()

    accuracy              = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )
    iou_per_class = calculate_iou_per_class(y_true, y_pred, num_classes)
    mean_iou      = float(np.mean(list(iou_per_class.values())))
    cm            = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    metrics = {
        'accuracy'           : float(accuracy),
        'mean_iou'           : mean_iou,
        'iou_per_class'      : iou_per_class,
        'precision_per_class': precision.tolist(),
        'recall_per_class'   : recall.tolist(),
        'f1_per_class'       : f1.tolist(),
        'support_per_class'  : support.tolist(),
        'confusion_matrix'   : cm.tolist(),
    }

    # Attach confidence statistics when a confidence map is provided
    if confidence_map is not None:
        metrics['mean_confidence'] = float(confidence_map.mean())
        metrics['min_confidence']  = float(confidence_map.min())
        metrics['max_confidence']  = float(confidence_map.max())
        metrics['std_confidence']  = float(confidence_map.std())
        # Fraction of patches below the "uncertain" threshold
        metrics['frac_low_confidence'] = float((confidence_map < 0.5).mean())

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation configs
# ─────────────────────────────────────────────────────────────────────────────

EVAL_CONFIGS = [
    {'hasDrainage': True,  'hasSoil': True,  'label': 'Full (drainage + soil)'},
    {'hasDrainage': True,  'hasSoil': False, 'label': 'Drainage only'},
    {'hasDrainage': False, 'hasSoil': True,  'label': 'Soil only'},
    {'hasDrainage': False, 'hasSoil': False, 'label': 'Neither (minimal)'},
]


def evaluate_test_scenarios(
    model               : nn.Module,
    patch_size          : int,
    test_dataset_dict   : dict,
    FloodPatchDataset,
    test_scenario_ids   : list,
    ground_truth_maps   : dict,
    device              : str,
    save_dir            : str,
) -> Tuple[dict, dict, dict]:
    """
    Runs 4 evaluation passes — one per hasDrainage / hasSoil combination.

    Returns
    -------
    all_config_results      : { config_label: { scenario_id: metrics_dict } }
    all_config_predictions  : { config_label: { scenario_id: predicted_map  (H×W int) } }
    all_config_confidences  : { config_label: { scenario_id: confidence_map (H×W float32) } }
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_config_results      = {}
    all_config_predictions  = {}
    all_config_confidences  = {}

    for cfg_entry in EVAL_CONFIGS:
        has_drain = cfg_entry['hasDrainage']
        has_soil  = cfg_entry['hasSoil']
        label     = cfg_entry['label']

        print("\n" + "="*70)
        print(f"EVALUATING — {label}")
        print(f"  hasDrainage={has_drain}  |  hasSoil={has_soil}")
        print("="*70)

        eval_dataset = FloodPatchDataset(
            test_dataset_dict,
            training    = False,
            hasDrainage = has_drain,
            hasSoil     = has_soil,
        )

        scenario_results     = {}
        scenario_predictions = {}
        scenario_confidences = {}

        for scenario_idx, scenario_id in enumerate(test_scenario_ids):
            print(f"\n  [Scenario {scenario_id}] Predicting...")

            # ── Inference ────────────────────────────────────────────────────
            patch_predictions, patch_confidences = predict_scenario_with_flags(
                model, eval_dataset, scenario_idx, device
            )

            # ── Reconstruct spatial maps ──────────────────────────────────────
            predicted_map  = reconstruct_test_map_from_patches(
                patch_predictions, (320, 320), patch_size
            )
            confidence_map = reconstruct_test_map_from_patches(      # float map
                patch_confidences, (320, 320), patch_size
            )

            scenario_predictions[scenario_id] = predicted_map
            scenario_confidences[scenario_id] = confidence_map

            # ── Metrics (confidence stats included) ───────────────────────────
            ground_truth_map              = ground_truth_maps[scenario_id]
            metrics                       = calculate_scenario_metrics(
                ground_truth_map, predicted_map, confidence_map
            )
            scenario_results[scenario_id] = metrics

            print(f"  [Scenario {scenario_id}]  "
                  f"Acc={metrics['accuracy']:.4f}  "
                  f"mIoU={metrics['mean_iou']:.4f}  "
                  f"AvgConf={metrics['mean_confidence']:.4f}  "
                  f"LowConf={metrics['frac_low_confidence']*100:.1f}%")

        # ── Per-config summary ────────────────────────────────────────────────
        avg_acc  = np.mean([r['accuracy']              for r in scenario_results.values()])
        avg_miou = np.mean([r['mean_iou']              for r in scenario_results.values()])
        avg_f1   = np.mean([np.mean(r['f1_per_class']) for r in scenario_results.values()])
        avg_conf = np.mean([r['mean_confidence']       for r in scenario_results.values()])
        avg_low  = np.mean([r['frac_low_confidence']   for r in scenario_results.values()])

        print(f"\n  ── {label} Summary ──")
        print(f"  Avg Accuracy       : {avg_acc:.4f}")
        print(f"  Avg Mean IoU       : {avg_miou:.4f}")
        print(f"  Avg Macro F1       : {avg_f1:.4f}")
        print(f"  Avg Confidence     : {avg_conf:.4f}")
        print(f"  Avg Low-Conf Ratio : {avg_low*100:.1f}%")

        all_config_results[label]     = scenario_results
        all_config_predictions[label] = scenario_predictions
        all_config_confidences[label] = scenario_confidences

    # ── Cross-config comparison table ─────────────────────────────────────────
    print("\n" + "="*70)
    print("4-PASS EVALUATION SUMMARY")
    print("="*70)
    print(f"  {'Config':<30} {'Acc':>8} {'mIoU':>8} {'F1':>8} {'AvgConf':>9} {'LowConf%':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*10}")
    for cfg_entry in EVAL_CONFIGS:
        lbl      = cfg_entry['label']
        results  = all_config_results[lbl]
        avg_acc  = np.mean([r['accuracy']              for r in results.values()])
        avg_miou = np.mean([r['mean_iou']              for r in results.values()])
        avg_f1   = np.mean([np.mean(r['f1_per_class']) for r in results.values()])
        avg_conf = np.mean([r['mean_confidence']       for r in results.values()])
        avg_low  = np.mean([r['frac_low_confidence']   for r in results.values()])
        print(f"  {lbl:<30} {avg_acc:>8.4f} {avg_miou:>8.4f} {avg_f1:>8.4f}"
              f" {avg_conf:>9.4f} {avg_low*100:>9.1f}%")
    print("="*70)

    return all_config_results, all_config_predictions, all_config_confidences


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics: Dict, scenario_id: int, save_path: str = None):
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=range(5), yticklabels=range(5),
        cbar_kws={'label': 'Count'},
    )
    plt.title(f'Confusion Matrix - Scenario {scenario_id}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_confidence_map(
    confidence_map : np.ndarray,
    predicted_map  : np.ndarray,
    scenario_id    : int,
    config_label   : str,
    metrics        : Dict = None,
    save_path      : str  = None,
) -> plt.Figure:
    """
    Two-panel figure showing the predicted flood class map alongside
    a per-pixel confidence heatmap.

    Confidence = max softmax probability (range 0–1).
    Patches below 0.5 are highlighted with a cool-toned overlay on the
    right panel so uncertain regions are immediately visible.

    Parameters
    ----------
    confidence_map : H×W float32 array   (output of reconstruct_test_map_from_patches)
    predicted_map  : H×W int array
    scenario_id    : used in title
    config_label   : e.g. 'Full (drainage + soil)'
    metrics        : if provided, summary stats are printed in the title
    save_path      : if provided, figure is saved here
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1a1a1a')

    title_str = f"Scenario {scenario_id} — Prediction & Confidence  ({config_label})"
    if metrics:
        title_str += (f"\nAcc={metrics['accuracy']:.3f}  "
                      f"mIoU={metrics['mean_iou']:.3f}  "
                      f"AvgConf={metrics['mean_confidence']:.3f}  "
                      f"LowConf={metrics['frac_low_confidence']*100:.1f}%")
    fig.suptitle(title_str, fontsize=11, fontweight='bold', color='white', y=1.02)

    # ── Left panel: predicted flood class ─────────────────────────────────────
    ax_pred  = axes[0]
    cmap_flood = mcolors.ListedColormap([FLOOD_CLASSES[c]['color'] for c in range(5)])
    im_pred  = ax_pred.imshow(
        predicted_map, cmap=cmap_flood, vmin=0, vmax=4,
        origin='upper', interpolation='nearest',
    )
    ax_pred.set_title("Predicted Flood Class", color='white', fontsize=10, pad=8)
    ax_pred.set_facecolor('#1a1a1a')
    ax_pred.set_xlabel("Column (West → East)", fontsize=8, color='white')
    ax_pred.set_ylabel("Row (North → South)",  fontsize=8, color='white')
    ax_pred.tick_params(colors='white', labelsize=7)
    for spine in ax_pred.spines.values():
        spine.set_edgecolor('white')

    # Flood-class colour legend under the left panel
    legend_ax = fig.add_axes([0.07, 0.02, 0.35, 0.06])
    legend_ax.set_facecolor('#1a1a1a')
    legend_ax.axis('off')
    for i, (cls, info) in enumerate(FLOOD_CLASSES.items()):
        color = info['color'] if cls > 0 else '#555555'
        legend_ax.add_patch(
            plt.Rectangle(
                (i * 0.20, 0.0), 0.18, 0.9,
                facecolor=color, edgecolor='#aaaaaa', linewidth=0.6,
                transform=legend_ax.transAxes, clip_on=False,
            )
        )
        legend_ax.text(
            i * 0.20 + 0.09, -0.4, info['name'],
            transform=legend_ax.transAxes,
            ha='center', fontsize=6.5, color='white',
        )

    # ── Right panel: confidence heatmap ───────────────────────────────────────
    ax_conf = axes[1]
    im_conf = ax_conf.imshow(
        confidence_map, cmap='RdYlGn',
        vmin=0.0, vmax=1.0,
        origin='upper', interpolation='nearest',
    )

    # Highlight uncertain patches (confidence < 0.5) with a semi-transparent overlay
    low_conf_mask = np.where(confidence_map < 0.5, 1.0, np.nan)
    ax_conf.imshow(
        low_conf_mask,
        cmap='cool', alpha=0.40,
        origin='upper', interpolation='nearest',
        vmin=0, vmax=1,
    )

    conf_title = f"Model Confidence  (mean={confidence_map.mean():.3f}  std={confidence_map.std():.3f})"
    if metrics:
        conf_title += f"\nLow-conf patches (<0.5): {metrics['frac_low_confidence']*100:.1f}%"
    ax_conf.set_title(conf_title, color='white', fontsize=10, pad=8)
    ax_conf.set_facecolor('#1a1a1a')
    ax_conf.set_xlabel("Column (West → East)", fontsize=8, color='white')
    ax_conf.set_ylabel("Row (North → South)",  fontsize=8, color='white')
    ax_conf.tick_params(colors='white', labelsize=7)
    for spine in ax_conf.spines.values():
        spine.set_edgecolor('white')

    cbar = fig.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04)
    cbar.set_label("Confidence (max softmax prob)", fontsize=8, color='white')
    cbar.ax.tick_params(colors='white', labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Confidence map saved → {save_path}")
    return fig


def plot_flood_on_dem(
    dem          : np.ndarray,
    flood_map    : np.ndarray,
    scenario_id  : int,
    metrics      : Dict,
    mode         : str,
    figsize      : tuple,
    inset_center : tuple,
    inset_size   : int,
    confidence_map: np.ndarray = None,   # optional — adds a confidence contour layer
) -> plt.Figure:
    H, W     = dem.shape
    n_h, n_w = flood_map.shape

    # ── Upsample flood map to DEM resolution ──────────────────────────────────
    ph = H / n_h
    pw = W / n_w
    flood_full = np.zeros((H, W), dtype=np.float32)
    for i in range(n_h):
        for j in range(n_w):
            r0, r1 = int(i * ph), int((i+1) * ph)
            c0, c1 = int(j * pw), int((j+1) * pw)
            flood_full[r0:r1, c0:c1] = flood_map[i, j]

    # ── Upsample confidence map to DEM resolution (if provided) ───────────────
    conf_full = None
    if confidence_map is not None:
        conf_full = np.zeros((H, W), dtype=np.float32)
        ch, cw = confidence_map.shape
        cph = H / ch
        cpw = W / cw
        for i in range(ch):
            for j in range(cw):
                r0, r1 = int(i * cph), int((i+1) * cph)
                c0, c1 = int(j * cpw), int((j+1) * cpw)
                conf_full[r0:r1, c0:c1] = confidence_map[i, j]

    # ── RGBA flood overlay ─────────────────────────────────────────────────────
    hex_to_rgb   = lambda h: tuple(int(h[i:i+2], 16)/255.0 for i in (1, 3, 5))
    flood_rgba   = np.zeros((H, W, 4), dtype=np.float32)
    class_alphas = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    for cls, info in FLOOD_CLASSES.items():
        mask = flood_full == cls
        r, g, b = hex_to_rgb(info['color'])
        flood_rgba[mask, 0] = r
        flood_rgba[mask, 1] = g
        flood_rgba[mask, 2] = b
        flood_rgba[mask, 3] = class_alphas[cls]

    # ── Low-confidence RGBA overlay (semi-transparent hatching) ───────────────
    low_conf_rgba = None
    if conf_full is not None:
        low_conf_rgba        = np.zeros((H, W, 4), dtype=np.float32)
        uncertain            = conf_full < 0.5
        low_conf_rgba[uncertain, 0] = 0.0   # R
        low_conf_rgba[uncertain, 1] = 0.8   # G
        low_conf_rgba[uncertain, 2] = 1.0   # B  (cyan tint)
        low_conf_rgba[uncertain, 3] = 0.30  # alpha

    # ── Inset center ──────────────────────────────────────────────────────────
    if inset_center is None:
        for cls in [4, 3, 2, 1]:
            hi = np.argwhere(flood_map == cls)
            if len(hi) > 0:
                center_row   = int(np.mean(hi[:, 0]) * ph + ph / 2)
                center_col   = int(np.mean(hi[:, 1]) * pw + pw / 2)
                inset_center = (center_row, center_col)
                break
        if inset_center is None:
            inset_center = (H // 2, W // 2)
    ic_r, ic_c = inset_center
    iz  = inset_size
    ir0 = max(0, ic_r - iz);  ir1 = min(H, ic_r + iz)
    ic0 = max(0, ic_c - iz);  ic1 = min(W, ic_c + iz)

    # ── DEM display range ──────────────────────────────────────────────────────
    dem_p2, dem_p98 = np.percentile(dem, 2), np.percentile(dem, 98)

    fig    = plt.figure(figsize=figsize, facecolor='#1a1a1a')
    ax     = fig.add_axes([0.30, 0.08, 0.50, 0.84], facecolor='#1a1a1a')

    # Layer 1: DEM hillshade
    dem_im = ax.imshow(dem, cmap='terrain', vmin=dem_p2, vmax=dem_p98,
                       origin='upper', interpolation='bilinear', alpha=0.4)
    # Layer 2: Flood colour overlay
    ax.imshow(flood_rgba, origin='upper', interpolation='nearest')
    # Layer 3: Low-confidence highlight (optional)
    if low_conf_rgba is not None:
        ax.imshow(low_conf_rgba, origin='upper', interpolation='nearest')

    # Inset red box marker
    from matplotlib.patches import Rectangle
    rect = Rectangle((ic0, ir0), ic1-ic0, ir1-ir0,
                      linewidth=2, edgecolor='red', facecolor='none', zorder=5)
    ax.add_patch(rect)

    # Axes styling
    ax.set_xlabel("Column (West  →  East)", fontsize=9, color='white')
    ax.set_ylabel("Row (North  →  South)",  fontsize=9, color='white')
    ax.tick_params(labelsize=8, colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    mode_label = "Predicted" if mode == 'pred' else "Ground Truth"
    title      = f"Metro Manila Q3 — {mode_label} Flood  (RS{scenario_id})"
    if metrics:
        m          = metrics
        prec_macro = np.mean(m['precision_per_class'])
        rec_macro  = np.mean(m['recall_per_class'])
        f1_macro   = np.mean(m['f1_per_class'])
        title     += (f"\nAcc={m['accuracy']:.3f}  Prec={prec_macro:.3f}  "
                      f"Rec={rec_macro:.3f}  F1={f1_macro:.3f}  "
                      f"IoU={m['mean_iou']:.3f}")
        if 'mean_confidence' in m:
            title += (f"\nAvgConf={m['mean_confidence']:.3f}  "
                      f"LowConf={m['frac_low_confidence']*100:.1f}%")
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='white')

    # ── DEM colorbar ───────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.82, 0.35, 0.025, 0.50])
    cbar    = fig.colorbar(dem_im, cax=cbar_ax)
    cbar.set_label("Elevation (m)", fontsize=9, color='white')
    cbar.ax.tick_params(labelsize=8, colors='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # ── Flood class legend ─────────────────────────────────────────────────────
    legend_ax = fig.add_axes([0.81, 0.08, 0.17, 0.22])
    legend_ax.set_facecolor('#1a1a1a')
    legend_ax.axis('off')
    legend_ax.set_title("Flood Class", fontsize=8, fontweight='bold', pad=4, color='white')
    for i, (cls, info) in enumerate(reversed(list(FLOOD_CLASSES.items()))):
        color = info['color'] if cls > 0 else '#555555'
        legend_ax.add_patch(
            plt.Rectangle(
                (0.0, i * 0.19), 0.22, 0.15,
                facecolor=color, edgecolor='#aaaaaa', linewidth=0.6,
                transform=legend_ax.transAxes, clip_on=False,
            )
        )
        legend_ax.text(0.28, i * 0.19 + 0.075, info['name'],
                       transform=legend_ax.transAxes,
                       va='center', fontsize=7.5, color='white')

    # Confidence legend entry (only shown when overlay is active)
    if low_conf_rgba is not None:
        legend_ax.add_patch(
            plt.Rectangle(
                (0.0, 5 * 0.19), 0.22, 0.15,
                facecolor='#00CCFF', edgecolor='#aaaaaa', linewidth=0.6,
                alpha=0.5,
                transform=legend_ax.transAxes, clip_on=False,
            )
        )
        legend_ax.text(0.28, 5 * 0.19 + 0.075, 'Low conf.',
                       transform=legend_ax.transAxes,
                       va='center', fontsize=7.5, color='white')

    # ── Inset zoom panel ───────────────────────────────────────────────────────
    inset_ax = fig.add_axes([0.04, 0.35, 0.24, 0.35], facecolor='#1a1a1a')
    inset_ax.imshow(dem[ir0:ir1, ic0:ic1], cmap='terrain',
                    vmin=dem_p2, vmax=dem_p98,
                    origin='upper', interpolation='bilinear', alpha=0.4)
    inset_ax.imshow(flood_rgba[ir0:ir1, ic0:ic1],
                    origin='upper', interpolation='nearest')
    if low_conf_rgba is not None:
        inset_ax.imshow(low_conf_rgba[ir0:ir1, ic0:ic1],
                        origin='upper', interpolation='nearest')
    inset_ax.set_title("Zoom", fontsize=8, color='white', pad=3)
    inset_ax.set_xticks([]); inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor('red'); spine.set_linewidth(2)

    return fig