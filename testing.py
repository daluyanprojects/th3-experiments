import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score,  precision_recall_fscore_support)

FLOOD_CLASSES = {
    0: {'name': 'No Flood',  'color': '#FFFFFF', 'hex': 'FFFFFF'},
    1: {'name': 'Light',     'color': '#FFEB3B', 'hex': 'FFEB3B'},
    2: {'name': 'Moderate',  'color': '#FF9800', 'hex': 'FF9800'},
    3: {'name': 'Heavy',     'color': '#F44336', 'hex': 'F44336'},
    4: {'name': 'Extreme',   'color': '#9C27B0', 'hex': '9C27B0'},
}

def reconstruct_test_map_from_patches(patch_predictions: np.ndarray, original_shape: Tuple[int, int], patch_size: int) -> np.ndarray:
    H, W = original_shape
    patches_h = H // patch_size  
    patches_w = W // patch_size 
    # Reshape from flat (6400,) to grid (80, 80)
    patch_grid = patch_predictions.reshape(patches_h, patches_w)
    # Expand each patch to 4×4 pixels
    reconstructed_map = np.repeat(np.repeat(patch_grid, patch_size, axis=0), patch_size, axis=1)
    return reconstructed_map


def predict_scenario(model: nn.Module, scenario_dataset, scenario_idx: int, device: str) -> np.ndarray:
    model.eval()
    
    patches_per_scenario = 6400
    start_idx = scenario_idx * patches_per_scenario
    end_idx = start_idx + patches_per_scenario
    patch_predictions = []
    
    with torch.no_grad():
        for idx in range(start_idx, end_idx):
            spatial, rainfall, _ = scenario_dataset[idx]
            
            # Add batch dimension
            spatial = spatial.unsqueeze(0).to(device)
            rainfall = rainfall.unsqueeze(0).to(device)
            
            # Forward pass
            logits, _ = model(spatial, rainfall)
            pred = logits.argmax(dim=1).item()
            
            patch_predictions.append(pred)
    
    return np.array(patch_predictions)


def calculate_iou_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> Dict[int, float]:
    iou_scores = {}
    
    for class_id in range(num_classes):
        # Binary masks for this class
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)
        
        # Intersection and union
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        # IoU
        if union == 0:
            iou_scores[class_id] = 0.0  # No samples of this class
        else:
            iou_scores[class_id] = intersection / union
    
    return iou_scores


def calculate_scenario_metrics(ground_truth_map: np.ndarray, predicted_map: np.ndarray, num_classes: int = 5) -> Dict:
    # Flatten 
    y_true = ground_truth_map.flatten()
    y_pred = predicted_map.flatten()
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes), zero_division=0)
    # IoU per class
    iou_per_class = calculate_iou_per_class(y_true, y_pred, num_classes)
    # Mean IoU (mIoU)
    mean_iou = np.mean(list(iou_per_class.values()))
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    metrics = {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'iou_per_class': iou_per_class,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'confusion_matrix': cm.tolist()
    }
    return metrics

def evaluate_test_scenarios(model: nn.Module, patch_size, test_dataset, test_scenario_ids: List[int], ground_truth_maps: Dict[int, np.ndarray], device: str, save_dir: str) -> Dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70) 
    print("EVALUATING TEST SCENARIOS")
    print("="*70)
    
    all_results = {}
    all_predictions = {}
    
    for scenario_idx, scenario_id in enumerate(test_scenario_ids):
        print(f"\n[Scenario {scenario_id}] Predicting...")
        # Predict all patches
        patch_predictions = predict_scenario(model, test_dataset, scenario_idx, device)
        # Reconstruct full map
        predicted_map = reconstruct_test_map_from_patches(patch_predictions, (320,320), patch_size)
        all_predictions[scenario_id] = predicted_map
        # Get ground truth
        ground_truth_map = ground_truth_maps[scenario_id]
        # Calculate metrics
        metrics = calculate_scenario_metrics(ground_truth_map, predicted_map)
        all_results[scenario_id] = metrics
        # Print summary
        print(f"[Scenario {scenario_id}] Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Mean IoU:  {metrics['mean_iou']:.4f}")
        print(f"  IoU per class:")
        for class_id, iou in metrics['iou_per_class'].items():
            print(f"    Class {class_id}: {iou:.4f}")
    
    # Calculate average metrics across scenarios
    avg_accuracy = np.mean([r['accuracy'] for r in all_results.values()])
    avg_mean_iou = np.mean([r['mean_iou'] for r in all_results.values()])
    
    print("\n" + "="*70)
    print("OVERALL TEST PERFORMANCE")
    print("="*70)
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Mean IoU: {avg_mean_iou:.4f}")
    
    # Save results
    results_summary = {
        'scenario_results': all_results,
        'predictions': {k: v.tolist() for k, v in all_predictions.items()},
        'average_accuracy': avg_accuracy,
        'average_mean_iou': avg_mean_iou
    }
    return all_results, all_predictions


def plot_confusion_matrix(metrics: Dict, scenario_id: int, save_path: str = None):
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5), cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Scenario {scenario_id}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_flood_on_dem(dem: np.ndarray, flood_map: np.ndarray, scenario_id: int, metrics: Dict, mode: str, figsize: tuple, inset_center: tuple, inset_size: int) -> plt.Figure:
    H, W     = dem.shape
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
        
    # ── Inset center ─────────────
    if inset_center is None:
        for cls in [4, 3, 2, 1]:
            hi = np.argwhere(flood_map == cls)
            if len(hi) > 0:
                center_row = int(np.mean(hi[:, 0]) * ph + ph / 2)
                center_col = int(np.mean(hi[:, 1]) * pw + pw / 2)
                inset_center = (center_row, center_col)
                break
        if inset_center is None:
            inset_center = (H // 2, W // 2)
    ic_r, ic_c = inset_center
    iz = inset_size
    ir0 = max(0, ic_r - iz); ir1 = min(H, ic_r + iz)
    ic0 = max(0, ic_c - iz); ic1 = min(W, ic_c + iz)

    # ── DEM display range ─────────────────────────────────────────────────────
    dem_p2, dem_p98 = np.percentile(dem, 2), np.percentile(dem, 98)
    fig = plt.figure(figsize=figsize, facecolor='#1a1a1a')
    ax  = fig.add_axes([0.30, 0.08, 0.50, 0.84], facecolor='#1a1a1a')  # shifted right

    # Layer 1: DEM
    dem_im = ax.imshow(dem, cmap='terrain', vmin=dem_p2, vmax=dem_p98, origin='upper', interpolation='bilinear', alpha=0.4)

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
        prec_macro = np.mean(m['precision_per_class'])
        rec_macro  = np.mean(m['recall_per_class'])
        f1_macro   = np.mean(m['f1_per_class'])
        title += (f"\nAcc={m['accuracy']:.3f}  Prec={prec_macro:.3f}  "
                f"Rec={rec_macro:.3f}  F1={f1_macro:.3f}  "
                f"IoU={m['mean_iou']:.3f}")
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

    # ── Inset zoom panel (left side, no overlap) ──────────────────────────────
    inset_ax = fig.add_axes([0.04, 0.35, 0.24, 0.35], facecolor='#1a1a1a')  # left side, vertically centered
    inset_ax.imshow(dem[ir0:ir1, ic0:ic1], cmap='terrain', vmin=dem_p2, vmax=dem_p98, origin='upper', interpolation='bilinear', alpha=0.4)
    inset_ax.imshow(flood_rgba[ir0:ir1, ic0:ic1], origin='upper', interpolation='nearest')
    inset_ax.set_title("Zoom", fontsize=8, color='white', pad=3)
    inset_ax.set_xticks([]); inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor('red'); spine.set_linewidth(2)

    return fig