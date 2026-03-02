import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from rf_config import RFConfig

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


def _upsample(fmap: np.ndarray, H: int, W: int) -> np.ndarray:
    n_h, n_w = fmap.shape
    return zoom(fmap.astype(np.float32), (H / n_h, W / n_w), order=0)


def _tight_bbox(dem: np.ndarray, flood_up: np.ndarray, pad: int) -> Tuple[int,int,int,int]:
    dem_valid  = ~np.isnan(dem)
    both_valid = dem_valid & (flood_up >= 0)
    dr, dc = np.where(dem_valid)
    vr, vc = np.where(both_valid)
    r0 = max(dr.min(), vr.min() - pad);  r1 = min(dr.max(), vr.max() + pad)
    c0 = max(dc.min(), vc.min() - pad);  c1 = min(dc.max(), vc.max() + pad)
    return int(r0), int(r1), int(c0), int(c1)


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

def reconstruct_ground_truth_maps_rf(
    test_scenario_meta : List[Dict],
    y_test             : np.ndarray,
    test_scenario_ids  : List[int],
    patch_size         : int = 4,
    spatial_shape      : Tuple[int,int] = SPATIAL_SHAPE,
) -> Dict[int, np.ndarray]:
    
    print("\n[Reconstructing Ground Truth Maps — Random Forest]")
    H, W   = spatial_shape
    gt_maps = {}

    for sid in test_scenario_ids:
        gt_map = np.full((H, W), -1, dtype=np.int8)
        patch_indices = [
            i for i, m in enumerate(test_scenario_meta)
            if m['scenario_id'] == sid
        ]
        for idx in patch_indices:
            pi, pj   = test_scenario_meta[idx]['patch_coord']
            row, col = pi * patch_size, pj * patch_size
            gt_map[row:row + patch_size, col:col + patch_size] = int(y_test[idx])

        valid = (gt_map >= 0).sum()
        gt_maps[sid] = gt_map
        print(f"  Scenario {sid}: {valid:,} valid pixels")

    print(f"  Reconstructed {len(gt_maps)} ground truth maps")
    return gt_maps


# ── 2. Prediction map generation ──────────────────────────────────────────────

def generate_prediction_maps_rf(
    model              : RandomForestClassifier,
    X_test             : np.ndarray,           # (N_test, 35)
    test_scenario_meta : List[Dict],
    test_scenario_ids  : List[int],
    cfg                : RFConfig,
    patch_size         : int = 4,
    spatial_shape      : Tuple[int,int] = SPATIAL_SHAPE,
) -> Dict:
    
    output_dir = Path(cfg.output_dir) / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE (Random Forest)")
    print("=" * 60)

    # Run inference once over all test patches
    print(f"  Running RF inference on {len(X_test):,} patches...")
    y_pred_flat  = model.predict(X_test)            # (N,)
    y_proba_flat = model.predict_proba(X_test)      # (N, 5)
    print(f"  Done.  pred shape={y_pred_flat.shape}  proba shape={y_proba_flat.shape}")

    H, W             = spatial_shape
    prediction_maps  = {}
    all_conf         = []

    for sid in tqdm(test_scenario_ids, desc="  Assembling maps"):
        patch_indices = [
            i for i, m in enumerate(test_scenario_meta)
            if m['scenario_id'] == sid
        ]

        pred_map = np.full((H, W),     -1, dtype=np.int8)
        prob_map = np.zeros((H, W, 5),     dtype=np.float32)

        for idx in patch_indices:
            pi, pj   = test_scenario_meta[idx]['patch_coord']
            row, col = pi * patch_size, pj * patch_size
            sl = (slice(row, row + patch_size), slice(col, col + patch_size))
            pred_map[sl] = int(y_pred_flat[idx])
            prob_map[sl] = y_proba_flat[idx]       # (5,) broadcast to (4, 4, 5)

        conf_map   = prob_map.max(axis=2)           # (H, W)
        valid_mask = pred_map >= 0
        valid_conf = conf_map[valid_mask]
        if len(valid_conf):
            all_conf.append(valid_conf)

        prediction_maps[sid] = {
            'prediction_map' : pred_map,
            'probability_map': prob_map,
            'confidence_map' : conf_map,
        }
        print(f"  Scenario {sid}: {valid_mask.sum():,} valid pixels")

    # ── Per-scenario plots ─────────────────────────────────────────────────
    print(f"\n  Saving per-scenario plots...")
    for sid, maps in prediction_maps.items():
        _plot_individual(maps['prediction_map'], maps['confidence_map'],
                         sid, cfg, output_dir)

    _plot_prediction_grid(prediction_maps, cfg, output_dir)

    if all_conf:
        flat_conf = np.concatenate(all_conf)
        _plot_confidence_histogram(flat_conf, output_dir)
        print(f"\n  Global confidence — mean={flat_conf.mean():.4f}"
              f"  median={np.median(flat_conf):.4f}"
              f"  std={flat_conf.std():.4f}")

    print(f"\n  All prediction outputs -> {output_dir}")
    return {'predictions': prediction_maps}


# ── 3. Per-scenario evaluation ────────────────────────────────────────────────

def evaluate_predictions_rf(
    prediction_maps    : Dict,
    gt_maps            : Dict,
    test_scenario_ids  : List[int],
    cfg                : RFConfig,
) -> Dict:
    log_dir = cfg.output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PER-SCENARIO EVALUATION (Random Forest)")
    print("=" * 60)

    per_scenario = {}
    f1s          = []

    for sid in test_scenario_ids:
        pred_map = prediction_maps[sid]['prediction_map']
        gt_map   = gt_maps[sid]

        # Only compare pixels inside mask
        valid = (gt_map >= 0) & (pred_map >= 0)
        if valid.sum() == 0:
            print(f"  Scenario {sid}: no valid overlap — skipped")
            continue

        y_t = gt_map[valid].astype(np.int32)
        y_p = pred_map[valid].astype(np.int32)

        acc     = float(accuracy_score(y_t, y_p))
        f1_mac  = float(f1_score(y_t, y_p, average='macro',    zero_division=0, labels=list(range(5))))
        iou_mac = float(jaccard_score(y_t, y_p, average='macro', zero_division=0, labels=list(range(5))))

        per_scenario[sid] = {
            'scenario_id' : sid,
            'accuracy'    : acc,
            'f1_macro'    : f1_mac,
            'iou_macro'   : iou_mac,
            'n_valid'     : int(valid.sum()),
        }
        f1s.append(f1_mac)
        print(f"  Scenario {sid:3d}: acc={acc:.4f}  f1={f1_mac:.4f}  iou={iou_mac:.4f}"
              f"  ({valid.sum():,} px)")

    f1_arr    = np.array(f1s)
    best_idx  = test_scenario_ids[int(np.argmax(f1_arr))]
    worst_idx = test_scenario_ids[int(np.argmin(f1_arr))]

    aggregate = {
        'mean_f1'   : float(f1_arr.mean()),
        'std_f1'    : float(f1_arr.std()),
        'min_f1'    : float(f1_arr.min()),
        'max_f1'    : float(f1_arr.max()),
        'best_idx'  : best_idx,
        'worst_idx' : worst_idx,
    }

    print(f"\n  Aggregate  —  mean F1={aggregate['mean_f1']:.4f}"
          f"  std={aggregate['std_f1']:.4f}"
          f"  best=RS{best_idx}  worst=RS{worst_idx}")

    results = {'per_scenario': per_scenario, 'aggregate': aggregate}

    with open(log_dir / 'per_scenario_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Per-scenario log -> {log_dir / 'per_scenario_results.json'}")

    return results


# ── 4. DEM overlay plots (unchanged from XGBoost) ────────────────────────────

def plot_prediction_vs_gt(
    dem        : np.ndarray,
    pred_map   : np.ndarray,
    gt_map     : np.ndarray,
    scenario_id: int,
    cfg        : RFConfig,
    figsize    : tuple = (14, 6),
    pad        : int = 5,
) -> plt.Figure:
    H_full, W_full = dem.shape
    pred_up = _upsample(pred_map, H_full, W_full)
    gt_up   = _upsample(gt_map,   H_full, W_full)
    r0, r1, c0, c1 = _tight_bbox(dem, gt_up, pad)

    dem_crop       = dem[r0:r1, c0:c1]
    pred_crop      = pred_up[r0:r1, c0:c1]
    gt_crop        = gt_up[r0:r1, c0:c1]
    dem_p2, dem_p98 = np.nanpercentile(dem_crop, [2, 98])
    alphas = {0: 0.0, 1: 0.55, 2: 0.65, 3: 0.75, 4: 0.88}

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes:
        ax.set_facecolor('#1a1a1a')

    for ax, fmap, label in zip(axes,
                                [gt_crop, pred_crop],
                                ['Ground Truth', 'RF Predicted']):
        ax.imshow(dem_crop, cmap='terrain', vmin=dem_p2, vmax=dem_p98,
                  origin='upper', interpolation='bilinear', aspect='equal', alpha=0.5)
        ax.imshow(_flood_rgba(fmap.astype(np.int32), alphas),
                  origin='upper', interpolation='nearest', aspect='equal')
        ax.set_title(f'RS{scenario_id} — {label}',
                     fontsize=12, fontweight='bold', color='white')
        _style_ax(ax)

    fig.legend(handles=_flood_legend_patches(), loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.04),
               facecolor='#2a2a2a', labelcolor='white', edgecolor='#555555')
    plt.tight_layout()
    return fig


def plot_best_worst_dem(
    dem    : np.ndarray,
    results: Dict,
    pred_maps: Dict,
    gt_maps  : Dict,
    figsize: tuple = (18, 11),
    pad    : int = 5,
) -> plt.Figure:
    H_full, W_full = dem.shape
    bi = results['aggregate']['best_idx']
    wi = results['aggregate']['worst_idx']

    keys = ['best_gt', 'best_pred', 'worst_gt', 'worst_pred']
    srcs = [gt_maps[bi], pred_maps[bi]['prediction_map'],
            gt_maps[wi], pred_maps[wi]['prediction_map']]
    maps_up = {k: _upsample(s, H_full, W_full) for k, s in zip(keys, srcs)}

    dem_valid   = ~np.isnan(dem)
    union_valid = np.zeros((H_full, W_full), bool)
    for fup in maps_up.values():
        union_valid |= (dem_valid & (fup >= 0))
    vr, vc = np.where(union_valid)
    if not len(vr):
        raise ValueError("No valid overlap between DEM and flood maps.")
    dr, dc  = np.where(dem_valid)
    r0 = max(dr.min(), vr.min()-pad);  r1 = min(dr.max(), vr.max()+pad)
    c0 = max(dc.min(), vc.min()-pad);  c1 = min(dc.max(), vc.max()+pad)

    dem_crop        = dem[r0:r1, c0:c1]
    maps_up         = {k: v[r0:r1, c0:c1] for k, v in maps_up.items()}
    dem_p2, dem_p98 = np.nanpercentile(dem_crop, [2, 98])
    alphas          = {0: 0.0, 1: 0.55, 2: 0.65, 3: 0.75, 4: 0.88}

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes.flat:
        ax.set_facecolor('#1a1a1a')
    fig.suptitle("Best vs Worst Scenario — Metro Manila DEM + Flood Overlay (Random Forest)",
                 fontsize=14, fontweight='bold', y=1.01, color='white')

    configs = [
        (0, 0, 'best_gt',    bi, 'Best GT'),
        (0, 1, 'best_pred',  bi, 'Best Predicted'),
        (1, 0, 'worst_gt',   wi, 'Worst GT'),
        (1, 1, 'worst_pred', wi, 'Worst Predicted'),
    ]
    for row, col, key, s_idx, label in configs:
        ax = axes[row, col]
        m  = results['per_scenario'][s_idx]
        ax.imshow(dem_crop, cmap='terrain', vmin=dem_p2, vmax=dem_p98,
                  origin='upper', interpolation='bilinear', aspect='equal', alpha=0.5)
        ax.imshow(_flood_rgba(maps_up[key].astype(np.int32), alphas),
                  origin='upper', interpolation='nearest', aspect='equal')
        ax.set_title(
            f"RS{m['scenario_id']} — {label}\n"
            f"Acc={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}  IoU={m['iou_macro']:.3f}",
            fontsize=10, fontweight='bold', color='white',
        )
        _style_ax(ax)

    fig.legend(handles=_flood_legend_patches(), loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.04),
               facecolor='#2a2a2a', labelcolor='white', edgecolor='#555555')
    fig.text(0.5, -0.02,
             f'Crop: rows [{r0}:{r1}], cols [{c0}:{c1}]  —  '
             f'{dem_crop.shape[0]}x{dem_crop.shape[1]} px',
             ha='center', color='#888888', fontsize=7, style='italic')
    plt.tight_layout()
    return fig


# ── Internal plot helpers ─────────────────────────────────────────────────────

def _plot_individual(
    pred_map  : np.ndarray,
    conf_map  : np.ndarray,
    sid       : int,
    cfg       : RFConfig,
    output_dir: Path,
):
    # Prediction map
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(pred_map, cmap=CMAP_FLOOD, vmin=-1, vmax=cfg.num_classes - 1,
                   interpolation='nearest', aspect='equal')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=range(-1, cfg.num_classes))
    cbar.set_label('Flood Class', fontsize=12, fontweight='bold')
    cbar.ax.set_yticklabels(['NoData'] + CLASS_NAMES)
    ax.set_title(f'Predicted Flood Map — Scenario {sid} (Random Forest)',
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(output_dir / f'prediction_scenario_{sid}.png', dpi=400, bbox_inches='tight')
    plt.close(fig)

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
    axes2[1].text(0.5, -0.04,
                  f'{n_low:,} / {n_valid:,} valid pixels ({pct:.1f}%)',
                  transform=axes2[1].transAxes, ha='center', fontsize=10, color='white')

    fig2.savefig(output_dir / f'confidence_scenario_{sid}.png', dpi=400, bbox_inches='tight')
    plt.close(fig2)


def _plot_prediction_grid(
    prediction_maps : Dict,
    cfg             : RFConfig,
    output_dir      : Path,
):
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

    plt.suptitle('Predicted Flood Maps — All Test Scenarios (Random Forest)',
                 fontsize=13, fontweight='bold')
    grid_fig.savefig(output_dir / 'prediction_grid.png', dpi=400, bbox_inches='tight')
    plt.close(grid_fig)


def _plot_confidence_histogram(flat: np.ndarray, output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(flat, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(flat.mean(),     color='red',    linestyle='--', lw=2,
               label=f'Mean {flat.mean():.3f}')
    ax.axvline(np.median(flat), color='orange', linestyle='--', lw=2,
               label=f'Median {np.median(flat):.3f}')
    ax.set(xlabel='Confidence', ylabel='Frequency',
           title='Prediction Confidence Distribution (Random Forest)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / 'confidence_histogram.png', dpi=400, bbox_inches='tight')
    plt.close(fig)
