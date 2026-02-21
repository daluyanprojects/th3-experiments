import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

from training import FloodPatchDataset, DEVICE, BATCH_SIZE, NUM_CLASSES, CLASS_NAMES
from torch.utils.data import DataLoader

OUTPUT_DIR  = Path('./outputs')
PATCH_SIZE  = 4                         # must match training config
SPATIAL_SHAPE = (1152, 1152)            # full spatial grid dimensions

CLASS_COLORS = ['#FFFFFF', '#FDD835', '#FB8C00', '#E53935', '#6A1B9A']
CMAP_FLOOD   = ListedColormap(['#CCCCCC'] + CLASS_COLORS)  # gray = NoData


def reconstruct_ground_truth_maps(test_metadata: List[Dict], y_test: np.ndarray, spatial_shape: Tuple[int, int] = SPATIAL_SHAPE, patch_size: int = PATCH_SIZE) -> Dict[int, np.ndarray]:
    print("\n[Reconstructing Ground Truth Maps]")
    print("-" * 60)

    scenarios: Dict = {}
    for idx, meta in enumerate(test_metadata):
        sid = meta['scenario_id']
        if sid not in scenarios:
            scenarios[sid] = {'labels': [], 'coords': []}
        scenarios[sid]['labels'].append(int(y_test[idx]))
        scenarios[sid]['coords'].append(meta['patch_coord'])

    ground_truth_maps = {}
    for sid, data in scenarios.items():
        gt_map = np.full(spatial_shape, -1, dtype=np.int8)
        for label, (i, j) in zip(data['labels'], data['coords']):
            r0, r1 = i * patch_size, (i + 1) * patch_size
            c0, c1 = j * patch_size, (j + 1) * patch_size
            gt_map[r0:r1, c0:c1] = label
        ground_truth_maps[sid] = gt_map
        print(f"  Scenario {sid}: {len(data['labels'])} patches")

    print(f"✓ Reconstructed {len(ground_truth_maps)} ground truth maps")
    return ground_truth_maps


@torch.no_grad()
def generate_prediction_maps(model: torch.nn.Module, test_loader: DataLoader, test_metadata: List[Dict], 
                             spatial_shape: Tuple[int, int] = SPATIAL_SHAPE, patch_size: int = PATCH_SIZE, 
                             output_dir: Path = OUTPUT_DIR / 'predictions') -> Dict:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("PREDICTION PIPELINE")
    print("="*60)

    # ── 1. Collect predictions ──────────────────
    print("\n[1] Running inference...")
    model.eval()
    all_preds, all_probs = [], []

    for spatial, rainfall, _ in tqdm(test_loader, desc="  Predicting"):
        spatial  = spatial.to(DEVICE)
        rainfall = rainfall.to(DEVICE)
        logits, _ = model(spatial, rainfall)
        probs     = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)           # (N,)
    all_probs = np.array(all_probs)           # (N, C)

    # ── 2. Group by scenario ────────────────────
    print("\n[2] Reconstructing spatial maps...")
    scenarios: Dict = {}
    for idx, meta in enumerate(test_metadata):
        sid = meta['scenario_id']
        if sid not in scenarios:
            scenarios[sid] = {'preds': [], 'probs': [], 'coords': []}
        scenarios[sid]['preds'].append(all_preds[idx])
        scenarios[sid]['probs'].append(all_probs[idx])
        scenarios[sid]['coords'].append(meta['patch_coord'])

    # ── 3. Reconstruct maps ─────────────────────
    prediction_maps = {}
    for sid, data in scenarios.items():
        pred_map = np.full(spatial_shape,           -1,  dtype=np.int8)
        prob_map = np.zeros(spatial_shape + (NUM_CLASSES,), dtype=np.float32)

        for pred, prob, (i, j) in zip(data['preds'], data['probs'], data['coords']):
            r0, r1 = i * patch_size, (i + 1) * patch_size
            c0, c1 = j * patch_size, (j + 1) * patch_size
            pred_map[r0:r1, c0:c1] = pred
            prob_map[r0:r1, c0:c1] = prob

        prediction_maps[sid] = {
            'prediction_map': pred_map,
            'probability_map': prob_map,
            'confidence_map':  prob_map.max(axis=2),
        }
        print(f"  Scenario {sid}: {len(data['preds'])} patches")

    # ── 4. Visualise ────────────────────────────
    print("\n[3] Generating visualisations...")
    _visualize_predictions(prediction_maps, output_dir)

    print("\n[4] Generating confidence maps...")
    confidence_maps = _generate_confidence_maps(prediction_maps, output_dir)

    print("\n" + "="*60)
    print(f"✅ Done  —  results in: {output_dir}")
    print("="*60)

    return {'predictions': prediction_maps, 'confidence': confidence_maps}


def compare_with_ground_truth(prediction_maps: Dict, ground_truth_maps: Dict, output_dir: Path = OUTPUT_DIR / 'predictions'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Comparison] Ground Truth vs Predictions")
    print("-" * 60)

    saved = 0
    for sid, maps in prediction_maps.items():
        if sid not in ground_truth_maps:
            print(f"  ⚠ Scenario {sid}: no ground truth, skipping")
            continue

        pred_map = maps['prediction_map']
        true_map = ground_truth_maps[sid]

        # Crop to bounding box of valid data
        valid = (true_map >= 0) & (pred_map >= 0)
        rows, cols = np.where(valid)
        if len(rows) == 0:
            print(f"  ⚠ Scenario {sid}: no valid overlap")
            continue

        pad = 20
        r0 = max(0, rows.min() - pad);  r1 = min(true_map.shape[0], rows.max() + pad)
        c0 = max(0, cols.min() - pad);  c1 = min(true_map.shape[1], cols.max() + pad)

        true_crop = true_map[r0:r1, c0:c1]
        pred_crop = pred_map[r0:r1, c0:c1]

        valid_crop = (true_crop >= 0) & (pred_crop >= 0)
        correct    = (pred_crop[valid_crop] == true_crop[valid_crop]).sum()
        accuracy   = correct / valid_crop.sum() * 100 if valid_crop.sum() > 0 else 0.0

        # Accuracy map: -1=NoData, 0=Wrong, 1=Correct
        acc_map = np.full_like(true_crop, -1, dtype=np.int8)
        acc_map[valid_crop] = (pred_crop[valid_crop] == true_crop[valid_crop]).astype(np.int8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        axes[0].imshow(true_crop, cmap=CMAP_FLOOD, vmin=-1, vmax=4, interpolation='nearest')
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        im_pred = axes[1].imshow(pred_crop, cmap=CMAP_FLOOD, vmin=-1, vmax=4, interpolation='nearest')
        axes[1].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(acc_map,
                       cmap=ListedColormap(['#CCCCCC', '#E53935', '#2E7D32']),
                       vmin=-1, vmax=1, interpolation='nearest')
        axes[2].set_title('Accuracy Map\n(Green=Correct, Red=Wrong)', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        cbar = plt.colorbar(im_pred, ax=axes[:2], fraction=0.02, pad=0.04, ticks=[-1,0,1,2,3,4])
        cbar.set_label('Flood Class', fontsize=11, fontweight='bold')
        cbar.ax.set_yticklabels(['NoData'] + CLASS_NAMES, fontsize=9)

        fig.text(0.5, 0.01,
                 f'Region [{r0}:{r1}, {c0}:{c1}]  —  {r1-r0}×{c1-c0} px',
                 ha='center', fontsize=9, style='italic')
        plt.suptitle(f'Scenario {sid}  —  Spatial Accuracy: {accuracy:.1f}%',
                     fontsize=14, fontweight='bold')

        fig.savefig(output_dir / f'comparison_scenario_{sid}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"  Scenario {sid}: {r1-r0}×{c1-c0} crop, acc={accuracy:.1f}%")
        saved += 1

    print(f"✓ Saved {saved} comparison plots")

def _visualize_predictions(prediction_maps: Dict, output_dir: Path):
    for sid, maps in prediction_maps.items():
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(maps['prediction_map'], cmap=CMAP_FLOOD, vmin=-1, vmax=4,
                       interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1,0,1,2,3,4])
        cbar.set_label('Flood Class', fontsize=12, fontweight='bold')
        cbar.ax.set_yticklabels(['NoData'] + CLASS_NAMES)
        ax.set_title(f'Predicted Flood Map — Scenario {sid}', fontsize=13, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(output_dir / f'prediction_scenario_{sid}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Summary grid
    n  = len(prediction_maps)
    nc = min(3, n)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
    axes = np.array(axes).flatten()

    last_im = None
    for idx, (sid, maps) in enumerate(prediction_maps.items()):
        last_im = axes[idx].imshow(maps['prediction_map'], cmap=CMAP_FLOOD,
                                   vmin=-1, vmax=4, interpolation='nearest')
        axes[idx].set_title(f'Scenario {sid}', fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    if last_im is not None:
        cbar = plt.colorbar(last_im, ax=axes, fraction=0.02, pad=0.04, ticks=[-1,0,1,2,3,4])
        cbar.set_label('Flood Class', fontsize=11, fontweight='bold')
        cbar.ax.set_yticklabels(['NoData'] + CLASS_NAMES, fontsize=9)

    plt.suptitle('Predicted Flood Maps — All Test Scenarios', fontsize=13, fontweight='bold')
    fig.savefig(output_dir / 'prediction_grid.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved {n} prediction maps + grid")


def _generate_confidence_maps(prediction_maps: Dict, output_dir: Path) -> Dict:
    confidence_maps = {}
    all_conf = []

    for sid, maps in prediction_maps.items():
        conf = maps['confidence_map']
        confidence_maps[sid] = conf
        valid = conf[conf > 0]
        if len(valid):
            all_conf.append(valid)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        im1 = axes[0].imshow(conf, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        axes[0].set_title(f'Confidence — Scenario {sid}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046).set_label('Confidence', fontsize=10)

        low_mask = (conf < 0.5) & (conf > 0)
        axes[1].imshow(low_mask, cmap='Reds', interpolation='nearest')
        axes[1].set_title('Low Confidence Regions (<50%)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        n_low  = low_mask.sum()
        n_all  = (conf > 0).sum()
        pct    = n_low / n_all * 100 if n_all > 0 else 0
        axes[1].text(0.5, -0.04, f'{n_low:,} pixels ({pct:.1f}%)',
                     transform=axes[1].transAxes, ha='center', fontsize=10)

        fig.savefig(output_dir / f'confidence_scenario_{sid}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Aggregate stats + histogram
    if all_conf:
        flat = np.concatenate(all_conf)
        print(f"\n  Confidence Statistics (all scenarios):")
        print(f"    Mean   : {flat.mean():.4f}")
        print(f"    Median : {np.median(flat):.4f}")
        print(f"    Std    : {flat.std():.4f}")
        print(f"    Min/Max: {flat.min():.4f} / {flat.max():.4f}")
        low = (flat < 0.5).mean() * 100
        mid = ((flat >= 0.5) & (flat < 0.8)).mean() * 100
        high= (flat >= 0.8).mean() * 100
        print(f"    Low (<0.5): {low:.1f}%  Mid (0.5-0.8): {mid:.1f}%  High (>0.8): {high:.1f}%")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(flat.mean(),       color='red',    linestyle='--', lw=2, label=f'Mean {flat.mean():.3f}')
        ax.axvline(np.median(flat),   color='orange', linestyle='--', lw=2, label=f'Median {np.median(flat):.3f}')
        ax.set(xlabel='Confidence', ylabel='Frequency', title='Prediction Confidence Distribution')
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout()
        fig.savefig(output_dir / 'confidence_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved confidence maps + histogram")

    return confidence_maps