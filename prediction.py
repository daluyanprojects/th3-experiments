import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy.ndimage import zoom
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import TrainConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATCH_SIZE    = 4
SPATIAL_SHAPE = (1152, 1152)
CLASS_NAMES   = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']

CMAP_FLOOD = ListedColormap(['#CCCCCC', '#FFFFFF', '#FDD835', '#FB8C00', '#E53935', '#6A1B9A'])
FLOOD_CLASSES = {
    0: {'name': 'No Flood', 'color': '#FFFFFF'},
    1: {'name': 'Light',    'color': '#FFEB3B'},
    2: {'name': 'Moderate', 'color': '#FF9800'},
    3: {'name': 'Heavy',    'color': '#F44336'},
    4: {'name': 'Extreme',  'color': '#9C27B0'},
}

def _upsample(fmap: np.ndarray, H: int, W: int) -> np.ndarray:
    n_h, n_w = fmap.shape
    return zoom(fmap.astype(np.float32), (H / n_h, W / n_w), order=0)


def _tight_bbox(dem: np.ndarray, flood_up: np.ndarray, pad: int) -> Tuple[int, int, int, int]:
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
        mpatches.Patch(color=FLOOD_CLASSES[c]['color'] if c > 0 else '#DDD',
                       label=FLOOD_CLASSES[c]['name'])
        for c in range(len(FLOOD_CLASSES))
    ]


def reconstruct_ground_truth_maps(
    test_metadata:  List[Dict],
    y_test:         np.ndarray,
    spatial_shape:  Tuple[int, int] = SPATIAL_SHAPE,
    patch_size:     int = PATCH_SIZE,
) -> Dict[int, np.ndarray]:
    print("\n[Reconstructing Ground Truth Maps]")
    scenarios: Dict = {}
    for idx, meta in enumerate(test_metadata):
        sid = meta['scenario_id']
        scenarios.setdefault(sid, {'labels': [], 'coords': []})
        scenarios[sid]['labels'].append(int(y_test[idx]))
        scenarios[sid]['coords'].append(meta['patch_coord'])

    gt_maps = {}
    for sid, data in scenarios.items():
        m = np.full(spatial_shape, -1, dtype=np.int8)
        for label, (i, j) in zip(data['labels'], data['coords']):
            m[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = label
        gt_maps[sid] = m
        print(f"  Scenario {sid}: {len(data['labels'])} patches")

    print(f"✓ Reconstructed {len(gt_maps)} ground truth maps")
    return gt_maps

@torch.no_grad()
def generate_prediction_maps(
    model:         torch.nn.Module,
    test_loader:   DataLoader,
    test_metadata: List[Dict],
    cfg:           TrainConfig,
    spatial_shape: Tuple[int, int] = SPATIAL_SHAPE,
    patch_size:    int = PATCH_SIZE,
) -> Dict:
    output_dir = Path(cfg.output_dir) / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)

    # ── Inference ─────────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_probs = [], []
    for batch in tqdm(test_loader, desc="  Predicting"):
        if len(batch) == 4:
            spatial, rainfall, conditioning, _ = batch
            conditioning = conditioning.to(DEVICE)
        else:
            spatial, rainfall, _ = batch
            conditioning = None
        logits, _ = model(spatial.to(DEVICE), rainfall.to(DEVICE), conditioning)
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ── Group & reconstruct ───────────────────────────────────────────────────
    scenarios: Dict = {}
    for idx, meta in enumerate(test_metadata):
        sid = meta['scenario_id']
        scenarios.setdefault(sid, {'preds': [], 'probs': [], 'coords': []})
        scenarios[sid]['preds'].append(all_preds[idx])
        scenarios[sid]['probs'].append(all_probs[idx])
        scenarios[sid]['coords'].append(meta['patch_coord'])

    prediction_maps, all_conf = {}, []
    for sid, data in scenarios.items():
        pred_map = np.full(spatial_shape, -1, dtype=np.int8)
        prob_map = np.zeros(spatial_shape + (cfg.num_classes,), dtype=np.float32)
        for pred, prob, (i, j) in zip(data['preds'], data['probs'], data['coords']):
            sl = (slice(i*patch_size, (i+1)*patch_size), slice(j*patch_size, (j+1)*patch_size))
            pred_map[sl] = pred
            prob_map[sl] = prob
        conf_map = prob_map.max(axis=2)
        prediction_maps[sid] = {'prediction_map': pred_map,
                                 'probability_map': prob_map,
                                 'confidence_map':  conf_map}
        valid = conf_map[conf_map > 0]
        if len(valid):
            all_conf.append(valid)
        print(f"  Scenario {sid}: {len(data['preds'])} patches")

    # ── Per-scenario plots ────────────────────────────────────────────────────
    nc = min(3, len(prediction_maps))
    nr = (len(prediction_maps) + nc - 1) // nc
    grid_fig, grid_axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    grid_axes = np.array(grid_axes).flatten()
    last_im   = None

    for idx, (sid, maps) in enumerate(prediction_maps.items()):
        pred, conf = maps['prediction_map'], maps['confidence_map']

        # ── Individual prediction map ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(pred, cmap=CMAP_FLOOD, vmin=-1, vmax=cfg.num_classes - 1,
                       interpolation='nearest', aspect='equal')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                            ticks=range(-1, cfg.num_classes))
        cbar.set_label('Flood Class', fontsize=12, fontweight='bold')
        cbar.ax.set_yticklabels(['NoData'] + CLASS_NAMES)
        ax.set_title(f'Predicted Flood Map — Scenario {sid}', fontsize=13, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(output_dir / f'prediction_scenario_{sid}.png', dpi=400, bbox_inches='tight')
        plt.close(fig)

        # ── Confidence map ─────────────────────────────────────────────────────
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        fig2.patch.set_facecolor('#1a1a1a')
        for ax2 in axes2:
            ax2.set_facecolor('#2a2a2a')

        valid_mask  = conf > 0
        conf_masked = np.ma.masked_where(~valid_mask, conf)
        cmap_conf   = plt.cm.RdYlGn.copy()
        cmap_conf.set_bad(color='#2a2a2a')

        im1 = axes2[0].imshow(conf_masked, cmap=cmap_conf, vmin=0, vmax=1,
                               interpolation='nearest', aspect='equal')
        axes2[0].set_title(f'Confidence — Scenario {sid}', fontsize=12,
                           fontweight='bold', color='white')
        axes2[0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes2[0], fraction=0.046)
        cbar1.set_label('Confidence', fontsize=10, color='white')
        cbar1.ax.tick_params(colors='white')
        plt.setp(cbar1.ax.yaxis.get_ticklabels(), color='white')

        low_mask    = valid_mask & (conf < 0.5)
        low_display = np.ma.masked_where(~valid_mask, low_mask.astype(float))
        cmap_low    = plt.cm.Reds.copy()
        cmap_low.set_bad(color='#2a2a2a')

        axes2[1].imshow(low_display, cmap=cmap_low, vmin=0, vmax=1,
                        interpolation='nearest', aspect='equal')
        axes2[1].set_title('Low Confidence Regions (<50%)', fontsize=12,
                           fontweight='bold', color='white')
        axes2[1].axis('off')

        n_valid = valid_mask.sum()
        n_low   = low_mask.sum()
        pct     = n_low / n_valid * 100 if n_valid > 0 else 0
        axes2[1].text(0.5, -0.04, f'{n_low:,} / {n_valid:,} valid pixels ({pct:.1f}%)',
                      transform=axes2[1].transAxes, ha='center',
                      fontsize=10, color='white')

        fig2.savefig(output_dir / f'confidence_scenario_{sid}.png', dpi=400, bbox_inches='tight')
        plt.close(fig2)

        # ── Grid thumbnail ─────────────────────────────────────────────────────
        last_im = grid_axes[idx].imshow(pred, cmap=CMAP_FLOOD, vmin=-1,
                                         vmax=cfg.num_classes - 1,
                                         interpolation='nearest', aspect='equal')
        grid_axes[idx].set_title(f'Scenario {sid}', fontsize=10, fontweight='bold')
        grid_axes[idx].axis('off')

    for idx in range(len(prediction_maps), len(grid_axes)):
        grid_axes[idx].axis('off')
    if last_im:
        cbar = plt.colorbar(last_im, ax=grid_axes, fraction=0.02, pad=0.04,
                            ticks=range(-1, cfg.num_classes))
        cbar.set_label('Flood Class', fontsize=11, fontweight='bold')
        cbar.ax.set_yticklabels(['NoData'] + CLASS_NAMES, fontsize=9)
    plt.suptitle('Predicted Flood Maps — All Test Scenarios', fontsize=13, fontweight='bold')
    grid_fig.savefig(output_dir / 'prediction_grid.png', dpi=400, bbox_inches='tight')
    plt.close(grid_fig)

    # ── Confidence stats + histogram ──────────────────────────────────────────
    if all_conf:
        flat = np.concatenate(all_conf)
        print(f"\n  Confidence — mean={flat.mean():.4f}  median={np.median(flat):.4f}"
              f"  std={flat.std():.4f}  range=[{flat.min():.4f}, {flat.max():.4f}]")
        low  = (flat < 0.5).mean() * 100
        mid  = ((flat >= 0.5) & (flat < 0.8)).mean() * 100
        high = (flat >= 0.8).mean() * 100
        print(f"  Low(<0.5)={low:.1f}%  Mid={mid:.1f}%  High(>0.8)={high:.1f}%")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(flat.mean(),     color='red',    linestyle='--', lw=2,
                   label=f'Mean {flat.mean():.3f}')
        ax.axvline(np.median(flat), color='orange', linestyle='--', lw=2,
                   label=f'Median {np.median(flat):.3f}')
        ax.set(xlabel='Confidence', ylabel='Frequency',
               title='Prediction Confidence Distribution')
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout()
        fig.savefig(output_dir / 'confidence_histogram.png', dpi=400, bbox_inches='tight')
        plt.close(fig)

    print(f"\n✓ Done — results in: {output_dir}")
    return {'predictions': prediction_maps}

def build_eval_results(
    gt_maps:       Dict[int, np.ndarray],
    pred_maps_dict: Dict,
    cfg:           TrainConfig,
) -> Dict:
    scenario_ids = sorted(gt_maps.keys())
    metrics, gt_list, pred_list = [], [], []

    for sid in scenario_ids:
        gt    = gt_maps[sid]
        pred  = pred_maps_dict[sid]['prediction_map']
        valid = (gt >= 0) & (pred >= 0)
        acc   = (gt[valid] == pred[valid]).mean() if valid.any() else 0.0

        ious, f1s, precs, recs = [], [], [], []
        for cls in range(cfg.num_classes):
            tp   = ((pred == cls) & (gt == cls) & valid).sum()
            fp   = ((pred == cls) & (gt != cls) & valid).sum()
            fn   = ((pred != cls) & (gt == cls) & valid).sum()
            iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            prec = tp / (tp + fp)      if (tp + fp) > 0      else 0.0
            rec  = tp / (tp + fn)      if (tp + fn) > 0      else 0.0
            f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            ious.append(iou); f1s.append(f1)
            precs.append(prec); recs.append(rec)

        metrics.append({'scenario_id':      sid,
                        'accuracy':         acc,
                        'iou_macro':        np.mean(ious),
                        'f1_macro':         np.mean(f1s),
                        'precision_macro':  np.mean(precs),
                        'recall_macro':     np.mean(recs)})
        gt_list.append(gt)
        pred_list.append(pred)

    f1s       = [m['f1_macro'] for m in metrics]
    best_idx  = int(np.argmax(f1s))
    worst_idx = int(np.argmin(f1s))

    print(f"\n✓ Eval: {len(scenario_ids)} scenarios  |  "
          f"Best RS{scenario_ids[best_idx]} F1={f1s[best_idx]:.3f}  |  "
          f"Worst RS{scenario_ids[worst_idx]} F1={f1s[worst_idx]:.3f}")

    return {'gt_maps':      gt_list,
            'pred_maps':    pred_list,
            'per_scenario': metrics,
            'scenario_ids': scenario_ids,
            'n_h':          gt_list[0].shape[0],
            'n_w':          gt_list[0].shape[1],
            'f1_scores':    f1s,
            'aggregate': {'best_scenario':  scenario_ids[best_idx],
                          'worst_scenario': scenario_ids[worst_idx],
                          'best_idx':       best_idx,
                          'worst_idx':      worst_idx}}


def plot_flood_on_dem(
    dem:           np.ndarray,
    flood_map:     np.ndarray,
    scenario_id:   int,
    metrics:       Dict,
    mode:          str,
    figsize:       tuple = (9, 11),
    inset_center:  Optional[Tuple[int, int]] = None,
    inset_size:    int = 120,
    pad:           int = 5,
) -> plt.Figure:
    H_full, W_full = dem.shape
    flood_up = _upsample(flood_map, H_full, W_full)
    r0, r1, c0, c1 = _tight_bbox(dem, flood_up, pad)

    dem_crop   = dem[r0:r1, c0:c1]
    flood_crop = flood_up[r0:r1, c0:c1]
    H, W       = dem_crop.shape

    rgba            = _flood_rgba(flood_crop, {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})
    dem_p2, dem_p98 = np.nanpercentile(dem_crop, [2, 98])

    if inset_center is None:
        for cls in [4, 3, 2, 1]:
            hi = np.argwhere(flood_crop == cls)
            if len(hi):
                pc = hi[len(hi) // 2]
                inset_center = (int(pc[0]), int(pc[1]))
                break
        inset_center = inset_center or (H // 2, W // 2)
    else:
        inset_center = (inset_center[0] - r0, inset_center[1] - c0)

    ic_r, ic_c = inset_center
    ir0 = max(0, ic_r - inset_size);  ir1 = min(H, ic_r + inset_size)
    ic0 = max(0, ic_c - inset_size);  ic1 = min(W, ic_c + inset_size)

    fig = plt.figure(figsize=figsize, facecolor='#1a1a1a')
    ax  = fig.add_axes([0.08, 0.08, 0.72, 0.84], facecolor='#1a1a1a')

    dem_im = ax.imshow(dem_crop, cmap='terrain', vmin=dem_p2, vmax=dem_p98,
                       origin='upper', interpolation='bilinear', alpha=0.4, aspect='equal')
    ax.imshow(rgba, origin='upper', interpolation='nearest', aspect='equal')
    ax.add_patch(Rectangle((ic0, ir0), ic1-ic0, ir1-ir0,
                            linewidth=2, edgecolor='red', facecolor='none', zorder=5))
    _style_ax(ax, 'Column (West → East)', 'Row (North → South)')

    mode_label = "Predicted" if mode == 'pred' else "Ground Truth"
    title = f"Metro Manila — {mode_label} Flood  (RS{scenario_id})"
    if metrics:
        m = metrics
        title += (f"\nAcc={m['accuracy']:.3f}  Prec={m['precision_macro']:.3f}  "
                  f"Rec={m['recall_macro']:.3f}  F1={m['f1_macro']:.3f}  IoU={m['iou_macro']:.3f}")
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='white')

    cbar_ax = fig.add_axes([0.82, 0.35, 0.025, 0.50])
    cbar = fig.colorbar(dem_im, cax=cbar_ax)
    cbar.set_label("Elevation (m)", fontsize=9, color='white')
    cbar.ax.tick_params(labelsize=8, colors='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    leg_ax = fig.add_axes([0.81, 0.08, 0.17, 0.22])
    leg_ax.set_facecolor('#1a1a1a'); leg_ax.axis('off')
    leg_ax.set_title("Flood Class", fontsize=8, fontweight='bold', pad=4, color='white')
    for i, (cls, info) in enumerate(reversed(list(FLOOD_CLASSES.items()))):
        leg_ax.add_patch(plt.Rectangle((0.0, i*0.19), 0.22, 0.15,
                                        facecolor=info['color'] if cls > 0 else '#555555',
                                        edgecolor='#aaaaaa', linewidth=0.6,
                                        transform=leg_ax.transAxes, clip_on=False))
        leg_ax.text(0.28, i*0.19+0.075, info['name'],
                    transform=leg_ax.transAxes, va='center', fontsize=7.5, color='white')

    asp      = (ir1-ir0) / max(ic1-ic0, 1)
    inset_fw = 0.28
    inset_fh = min(inset_fw * asp * figsize[0] / figsize[1], 0.32)
    ins = fig.add_axes([0.44, 0.06, inset_fw, inset_fh], facecolor='#1a1a1a')
    ins.imshow(dem_crop[ir0:ir1, ic0:ic1], cmap='terrain', vmin=dem_p2, vmax=dem_p98,
               origin='upper', interpolation='bilinear', alpha=0.4, aspect='equal')
    ins.imshow(rgba[ir0:ir1, ic0:ic1], origin='upper', interpolation='nearest', aspect='equal')
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_edgecolor('red'); sp.set_linewidth(2)

    fig.text(0.08, 0.01,
             f'Crop: rows [{r0}:{r1}], cols [{c0}:{c1}]  —  {H}×{W} px',
             color='#888888', fontsize=7, style='italic')
    return fig


def plot_best_worst_dem(
    dem:     np.ndarray,
    results: Dict,
    figsize: tuple = (18, 11),
    pad:     int = 5,
) -> plt.Figure:
    H_full, W_full = dem.shape
    bi, wi = results['aggregate']['best_idx'], results['aggregate']['worst_idx']

    keys = ['best_gt', 'best_pred', 'worst_gt', 'worst_pred']
    srcs = [results['gt_maps'][bi], results['pred_maps'][bi],
            results['gt_maps'][wi], results['pred_maps'][wi]]
    maps_up = {k: _upsample(s, H_full, W_full) for k, s in zip(keys, srcs)}

    dem_valid   = ~np.isnan(dem)
    union_valid = np.zeros((H_full, W_full), bool)
    for fup in maps_up.values():
        union_valid |= (dem_valid & (fup >= 0))

    vr, vc = np.where(union_valid)
    if not len(vr):
        raise ValueError("No valid overlap between DEM and flood maps.")
    dr, dc = np.where(dem_valid)
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
    fig.suptitle("Best vs Worst Scenario — Metro Manila DEM + Flood Overlay",
                 fontsize=14, fontweight='bold', y=1.01, color='white')

    configs = [(0, 0, 'best_gt',    bi, 'Best GT'),
               (0, 1, 'best_pred',  bi, 'Best Predicted'),
               (1, 0, 'worst_gt',   wi, 'Worst GT'),
               (1, 1, 'worst_pred', wi, 'Worst Predicted')]

    for row, col, key, s_idx, label in configs:
        ax = axes[row, col]
        m  = results['per_scenario'][s_idx]
        ax.imshow(dem_crop, cmap='terrain', vmin=dem_p2, vmax=dem_p98,
                  origin='upper', interpolation='bilinear', aspect='equal', alpha=0.5)
        ax.imshow(_flood_rgba(maps_up[key], alphas),
                  origin='upper', interpolation='nearest', aspect='equal')
        ax.set_title(f"RS{m['scenario_id']} — {label}\n"
                     f"Acc={m['accuracy']:.3f}  F1={m['f1_macro']:.3f}  IoU={m['iou_macro']:.3f}",
                     fontsize=10, fontweight='bold', color='white')
        _style_ax(ax)

    fig.legend(handles=_flood_legend_patches(), loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.04),
               facecolor='#2a2a2a', labelcolor='white', edgecolor='#555555')
    fig.text(0.5, -0.02,
             f'Crop: rows [{r0}:{r1}], cols [{c0}:{c1}]  —  '
             f'{dem_crop.shape[0]}×{dem_crop.shape[1]} px',
             ha='center', color='#888888', fontsize=7, style='italic')
    plt.tight_layout()
    return fig