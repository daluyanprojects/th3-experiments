import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import zoom
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import SVMConfig

SPATIAL_SHAPE = (1152, 1152)
CLASS_NAMES   = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
CMAP_FLOOD    = ListedColormap(['#CCCCCC', '#FFFFFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B'])
FLOOD_CLASSES = {
    0: {'name': 'No Flood', 'color': '#FFFFFF'},
    1: {'name': 'Light',    'color': '#C6DBEF'},
    2: {'name': 'Moderate', 'color': '#6BAED6'},
    3: {'name': 'Heavy',    'color': '#2171B5'},
    4: {'name': 'Extreme',  'color': '#08306B'},
}


# ── Shared plot helpers ────────────────────────────────────────────────────────
def _upsample(fmap: np.ndarray, H: int, W: int) -> np.ndarray:
    n_h, n_w = fmap.shape
    return zoom(fmap.astype(np.float32), (H / n_h, W / n_w), order=0)


def _flood_rgba(flood: np.ndarray, alphas: Dict[int, float]) -> np.ndarray:
    H, W = flood.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    for cls, info in FLOOD_CLASSES.items():
        r, g, b = (int(info['color'][k:k+2], 16) / 255.0 for k in (1, 3, 5))
        m = flood == cls
        rgba[m, :3] = (r, g, b)
        rgba[m,  3] = alphas[cls]
    return rgba


def _style_ax(ax, xlabel='Column', ylabel='Row'):
    ax.set_xlabel(xlabel, fontsize=8, color='white')
    ax.set_ylabel(ylabel, fontsize=8, color='white')
    ax.tick_params(labelsize=7, colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('white')


def _flood_legend_patches():
    return [
        mpatches.Patch(
            color=FLOOD_CLASSES[c]['color'] if c > 0 else '#DDD',
            label=FLOOD_CLASSES[c]['name'],
        )
        for c in range(len(FLOOD_CLASSES))
    ]


# ── 1. Ground truth reconstruction ────────────────────────────────────────────
def reconstruct_ground_truth_maps_svm(
    test_scenario_meta : List[Dict],
    y_test             : np.ndarray,
    test_scenario_ids  : List[int],
    patch_size         : int = 4,
    spatial_shape      : Tuple[int, int] = SPATIAL_SHAPE,
) -> Dict[int, np.ndarray]:
    """
    Reassemble per-patch ground truth labels into (H, W) maps.
    Identical logic to the XGBoost version.
    """
    print("\n[Reconstructing Ground Truth Maps — SVM]")
    H, W    = spatial_shape
    gt_maps = {}

    for sid in test_scenario_ids:
        gt_map = np.full((H, W), -1, dtype=np.int8)
        patch_indices = [
            i for i, m in enumerate(test_scenario_meta)
            if m['scenario_id'] == sid
        ]
        for idx in patch_indices:
            pi, pj = test_scenario_meta[idx]['patch_coord']
            row, col = pi * patch_size, pj * patch_size
            gt_map[row:row + patch_size, col:col + patch_size] = int(y_test[idx])

        valid = (gt_map >= 0).sum()
        gt_maps[sid] = gt_map
        print(f"  Scenario {sid}: {valid:,} valid pixels")

    print(f"✓ Reconstructed {len(gt_maps)} ground truth maps")
    return gt_maps


# ── 2. Prediction map generation ──────────────────────────────────────────────
def generate_prediction_maps_svm(
    y_pred_flat        : np.ndarray,   # (N_test,)
    y_proba_flat       : Optional[np.ndarray],  # (N_test, 5) or None
    test_scenario_meta : List[Dict],
    test_scenario_ids  : List[int],
    cfg                : SVMConfig,
    patch_size         : int = 4,
    spatial_shape      : Tuple[int, int] = SPATIAL_SHAPE,
) -> Dict:
    """
    Reassemble flat SVM predictions into per-scenario (H, W) maps.

    If cfg.probability=False, y_proba_flat may be None; confidence maps
    will be skipped.
    """
    output_dir = Path(cfg.output_dir) / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    H, W = spatial_shape
    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE (SVM)")
    print("=" * 60)

    prediction_maps : Dict[int, Dict] = {}
    all_conf        : List[np.ndarray] = []

    for sid in tqdm(test_scenario_ids, desc="  Assembling maps"):
        patch_indices = [
            i for i, m in enumerate(test_scenario_meta)
            if m['scenario_id'] == sid
        ]

        pred_map = np.full((H, W), -1, dtype=np.int8)
        prob_map = np.zeros((H, W, 5), dtype=np.float32) if y_proba_flat is not None else None

        for idx in patch_indices:
            pi, pj   = test_scenario_meta[idx]['patch_coord']
            row, col = pi * patch_size, pj * patch_size
            sl = (slice(row, row + patch_size), slice(col, col + patch_size))
            pred_map[sl] = int(y_pred_flat[idx])
            if prob_map is not None:
                prob_map[sl] = y_proba_flat[idx]

        conf_map = prob_map.max(axis=2) if prob_map is not None else None

        valid_mask = pred_map >= 0
        if conf_map is not None:
            valid_conf = conf_map[valid_mask]
            if len(valid_conf):
                all_conf.append(valid_conf)

        prediction_maps[sid] = {
            'prediction_map' : pred_map,
            'probability_map': prob_map,
            'confidence_map' : conf_map,
        }
        print(f"  Scenario {sid}: {valid_mask.sum():,} valid pixels")

    # Per-scenario plots
    for sid, maps in prediction_maps.items():
        _plot_individual(maps['prediction_map'], maps['confidence_map'], sid, cfg, output_dir)

    # Grid overview
    _plot_prediction_grid(prediction_maps, cfg, output_dir)

    # Confidence histogram
    if all_conf:
        flat_conf = np.concatenate(all_conf)
        _plot_confidence_histogram(flat_conf, output_dir)

    return {'predictions': prediction_maps}


# ── 3. Scenario-level metrics ──────────────────────────────────────────────────
def compute_per_scenario_metrics(
    gt_maps       : Dict[int, np.ndarray],
    pred_maps_out : Dict,
    cfg           : SVMConfig,
) -> Dict:
    """
    Compute accuracy, macro-F1, and macro-IoU for each test scenario.
    Mirrors the XGBoost evaluate_per_scenario helper.
    """
    from sklearn.metrics import jaccard_score

    results : Dict = {'per_scenario': {}, 'aggregate': {}}
    f1s, ious, accs = [], [], []

    for sid, gt in gt_maps.items():
        pred_map = pred_maps_out['predictions'][sid]['prediction_map']
        mask     = gt >= 0

        y_true = gt[mask].astype(int)
        y_pred = pred_map[mask].astype(int)

        acc  = float((y_true == y_pred).mean())
        f1   = float(f1_score(y_true, y_pred, average='macro', zero_division=0, labels=list(range(5))))
        iou  = float(jaccard_score(y_true, y_pred, average='macro', zero_division=0, labels=list(range(5))))

        results['per_scenario'][sid] = {
            'scenario_id': sid,
            'accuracy'   : acc,
            'f1_macro'   : f1,
            'iou_macro'  : iou,
            'n_pixels'   : int(mask.sum()),
        }
        f1s.append(f1); ious.append(iou); accs.append(acc)
        print(f"  Scenario {sid}: acc={acc:.4f}  macro_f1={f1:.4f}  macro_iou={iou:.4f}"
              f"  ({mask.sum():,} px)")

    best_idx  = list(gt_maps.keys())[int(np.argmax(f1s))]
    worst_idx = list(gt_maps.keys())[int(np.argmin(f1s))]

    results['aggregate'] = {
        'mean_f1'   : float(np.mean(f1s)),
        'mean_iou'  : float(np.mean(ious)),
        'mean_acc'  : float(np.mean(accs)),
        'best_idx'  : best_idx,
        'worst_idx' : worst_idx,
    }
    print(f"\n  Aggregate — mean_f1={np.mean(f1s):.4f}  mean_iou={np.mean(ious):.4f}"
          f"  mean_acc={np.mean(accs):.4f}")
    print(f"  Best scenario: {best_idx}  |  Worst: {worst_idx}")
    return results


# ── Internal plot helpers ──────────────────────────────────────────────────────
def _plot_individual(pred_map, conf_map, sid, cfg, output_dir):
    # Prediction map
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(pred_map, cmap=CMAP_FLOOD, vmin=-1, vmax=cfg.num_classes - 1,
                   interpolation='nearest', aspect='equal')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=range(-1, cfg.num_classes))
    cbar.set_label('Flood Class', fontsize=12, fontweight='bold')
    cbar.ax.set_yticklabels(['NoData'] + CLASS_NAMES)
    ax.set_title(f'Predicted Flood Map — Scenario {sid} (SVM)',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(output_dir / f'prediction_scenario_{sid}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

    if conf_map is None:
        return

    # Confidence map
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig2.patch.set_facecolor('#1a1a1a')
    for ax2 in axes2:
        ax2.set_facecolor('#2a2a2a')

    valid_mask  = pred_map >= 0
    conf_masked = np.ma.masked_where(~valid_mask, conf_map)
    cmap_conf   = plt.cm.RdYlGn.copy()
    cmap_conf.set_bad(color='#2a2a2a')

    im1 = axes2[0].imshow(conf_masked, cmap=cmap_conf, vmin=0, vmax=1,
                          interpolation='nearest', aspect='equal')
    axes2[0].set_title(f'Confidence — Scenario {sid}',
                       fontsize=12, fontweight='bold', color='white')
    axes2[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes2[0], fraction=0.046)
    cbar1.set_label('Confidence', fontsize=10, color='white')
    cbar1.ax.tick_params(colors='white')

    low_mask    = valid_mask & (conf_map < 0.5)
    low_display = np.ma.masked_where(~valid_mask, low_mask.astype(float))
    cmap_low    = plt.cm.Reds.copy()
    cmap_low.set_bad(color='#2a2a2a')
    axes2[1].imshow(low_display, cmap=cmap_low, vmin=0, vmax=1,
                    interpolation='nearest', aspect='equal')
    axes2[1].set_title('Low Confidence Regions (<50%)',
                       fontsize=12, fontweight='bold', color='white')
    axes2[1].axis('off')
    n_valid = valid_mask.sum()
    n_low   = low_mask.sum()
    pct     = n_low / n_valid * 100 if n_valid > 0 else 0
    axes2[1].text(0.5, -0.04, f'{n_low:,} / {n_valid:,} valid pixels ({pct:.1f}%)',
                  transform=axes2[1].transAxes, ha='center', fontsize=10, color='white')
    fig2.savefig(output_dir / f'confidence_scenario_{sid}.png', dpi=400, bbox_inches='tight')
    plt.close(fig2)


def _plot_prediction_grid(prediction_maps, cfg, output_dir):
    nc = min(3, len(prediction_maps))
    nr = (len(prediction_maps) + nc - 1) // nc
    grid_fig, grid_axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    grid_axes = np.array(grid_axes).flatten()
    last_im   = None

    for idx, (sid, maps) in enumerate(prediction_maps.items()):
        last_im = grid_axes[idx].imshow(
            maps['prediction_map'], cmap=CMAP_FLOOD,
            vmin=-1, vmax=cfg.num_classes - 1,
            interpolation='nearest', aspect='equal',
        )
        grid_axes[idx].set_title(f'Scenario {sid}', fontsize=10, fontweight='bold')
        grid_axes[idx].axis('off')

    for idx in range(len(prediction_maps), len(grid_axes)):
        grid_axes[idx].axis('off')

    if last_im:
        cbar = plt.colorbar(last_im, ax=grid_axes, fraction=0.02, pad=0.04,
                            ticks=range(-1, cfg.num_classes))
        cbar.set_label('Flood Class', fontsize=11, fontweight='bold')
        cbar.ax.set_yticklabels(['NoData'] + CLASS_NAMES, fontsize=9)

    plt.suptitle('Predicted Flood Maps — All Test Scenarios (SVM)',
                 fontsize=13, fontweight='bold')
    grid_fig.savefig(output_dir / 'prediction_grid.png', dpi=400, bbox_inches='tight')
    plt.close(grid_fig)


def _plot_confidence_histogram(flat: np.ndarray, output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(flat, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(flat.mean(),     color='red',    linestyle='--', lw=2,
               label=f'Mean {flat.mean():.3f}')
    ax.axvline(np.median(flat), color='orange', linestyle='--', lw=2,
               label=f'Median {np.median(flat):.3f}')
    ax.set(xlabel='Confidence', ylabel='Frequency',
           title='Prediction Confidence Distribution (SVM)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / 'confidence_histogram.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
