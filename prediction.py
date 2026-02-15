import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm

def reconstruct_ground_truth_maps(
    test_loader,
    test_metadata,
    spatial_shape=(1152, 1152),
    patch_size=16
):
    """Reconstruct ground truth spatial maps from test patches"""
    
    print("\n[Reconstructing Ground Truth Maps]")
    print("-" * 70)
    
    all_labels = []
    
    # Collect all labels
    for _, _, labels in test_loader:
        all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    
    # Group by scenario
    scenarios = {}
    for idx, metadata in enumerate(test_metadata):
        scenario_id = metadata['scenario_id']
        
        if scenario_id not in scenarios:
            scenarios[scenario_id] = {
                'labels': [],
                'coords': []
            }
        
        scenarios[scenario_id]['labels'].append(all_labels[idx])
        scenarios[scenario_id]['coords'].append(metadata['patch_coord'])
    
    # Reconstruct maps
    ground_truth_maps = {}
    
    for scenario_id, data in scenarios.items():
        # Initialize empty map
        gt_map = np.full(spatial_shape, -1, dtype=np.int8)
        
        # Fill in ground truth
        for label, (i, j) in zip(data['labels'], data['coords']):
            row_start = i * patch_size
            row_end = (i + 1) * patch_size
            col_start = j * patch_size
            col_end = (j + 1) * patch_size
            
            gt_map[row_start:row_end, col_start:col_end] = label
        
        ground_truth_maps[scenario_id] = gt_map
        print(f"  Scenario {scenario_id}: {len(data['labels'])} patches")
    
    print(f"✓ Reconstructed {len(ground_truth_maps)} ground truth maps\n")
    
    return ground_truth_maps

def generate_prediction_maps(
    trainer,
    test_loader,
    test_metadata,
    spatial_shape=(1152, 1152),
    patch_size=16,
    output_dir='./predictions'
):
    """
    Generate full spatial prediction maps from patch predictions
    
    Args:
        trainer: Trained model
        test_loader: Test data loader
        test_metadata: Metadata with patch coordinates
        spatial_shape: Full spatial dimension
        patch_size: Patch size used
        output_dir: Directory to save results
    
    Returns:
        Dictionary with prediction maps for each scenario
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 13: PREDICTION PIPELINE")
    print("="*70)
    
    # =========================================================================
    # 13.1 RECONSTRUCT PREDICTION MAPS
    # =========================================================================
    print("\n[13.1] Reconstructing Prediction Maps")
    print("-" * 70)
    
    prediction_maps = reconstruct_spatial_predictions(
        trainer,
        test_loader,
        test_metadata,
        spatial_shape,
        patch_size
    )
    
    # =========================================================================
    # 13.2 VISUALIZATION
    # =========================================================================
    print("\n[13.2] Generating Visualizations")
    print("-" * 70)
    
    visualize_predictions(
        prediction_maps,
        output_path
    )
    
    # =========================================================================
    # 13.3 CONFIDENCE MAPPING
    # =========================================================================
    print("\n[13.3] Generating Confidence Maps")
    print("-" * 70)
    
    confidence_maps = generate_confidence_maps(
        prediction_maps,
        output_path
    )
    
    print("\n" + "="*70)
    print("✅ PREDICTION PIPELINE COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_path}")
    print("="*70 + "\n")
    
    return {
        'predictions': prediction_maps,
        'confidence': confidence_maps
    }

def reconstruct_spatial_predictions(
    trainer,
    test_loader,
    test_metadata,
    spatial_shape,
    patch_size
):
    """Reconstruct full spatial maps from patch predictions"""
    
    trainer.model.eval()
    
    # Get all predictions and probabilities
    all_predictions = []
    all_probs = []
    
    print("  Collecting predictions...")
    with torch.no_grad():
        for spatial, rainfall, labels in tqdm(test_loader, desc="  Predicting"):
            spatial = spatial.to(trainer.device)
            rainfall = rainfall.to(trainer.device)
            
            logits, _ = trainer.model(spatial, rainfall)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # Group by scenario
    scenarios = {}
    for idx, metadata in enumerate(test_metadata):
        scenario_id = metadata['scenario_id']
        
        if scenario_id not in scenarios:
            scenarios[scenario_id] = {
                'predictions': [],
                'probabilities': [],
                'coords': []
            }
        
        scenarios[scenario_id]['predictions'].append(all_predictions[idx])
        scenarios[scenario_id]['probabilities'].append(all_probs[idx])
        scenarios[scenario_id]['coords'].append(metadata['patch_coord'])
    
    # Reconstruct maps for each scenario
    prediction_maps = {}
    
    print(f"\n  Reconstructing {len(scenarios)} scenario maps...")
    for scenario_id, data in scenarios.items():
        # Initialize empty map
        pred_map = np.full(spatial_shape, -1, dtype=np.int8)
        prob_map = np.zeros(spatial_shape + (5,), dtype=np.float32)
        
        # Fill in predictions
        for pred, prob, (i, j) in zip(data['predictions'], data['probabilities'], data['coords']):
            row_start = i * patch_size
            row_end = (i + 1) * patch_size
            col_start = j * patch_size
            col_end = (j + 1) * patch_size
            
            pred_map[row_start:row_end, col_start:col_end] = pred
            prob_map[row_start:row_end, col_start:col_end] = prob
        
        prediction_maps[scenario_id] = {
            'prediction_map': pred_map,
            'probability_map': prob_map,
            'confidence_map': prob_map.max(axis=2)
        }
        
        print(f"    Scenario {scenario_id}: {len(data['predictions'])} patches")
    
    print(f"  ✓ Reconstructed {len(prediction_maps)} scenario maps")
    
    return prediction_maps

def visualize_predictions(prediction_maps, output_path):    
    # Define color scheme
    class_colors = ['#FFFFFF', '#FDD835', '#FB8C00', '#E53935', '#6A1B9A']  # White, Yellow, Orange, Red, Purple
    class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    cmap = ListedColormap(['#CCCCCC'] + class_colors)  # Gray for NoData
    
    # Visualize each scenario
    for scenario_id, maps in prediction_maps.items():
        pred_map = maps['prediction_map']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot prediction map
        im = ax.imshow(pred_map, cmap=cmap, vmin=-1, vmax=4, interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 0, 1, 2, 3, 4])
        cbar.set_label('Flood Class', fontsize=12, fontweight='bold')
        cbar.ax.set_yticklabels(['NoData'] + class_names)
        
        # Styling
        ax.set_title(f'Predicted Flood Map - Scenario {scenario_id}\n(Manila Core Region)', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / f'prediction_scenario_{scenario_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Saved {len(prediction_maps)} prediction maps")
    
    # Create summary grid
    create_prediction_grid(prediction_maps, output_path, class_colors)


def create_prediction_grid(prediction_maps, output_path, class_colors):    
    num_scenarios = len(prediction_maps)
    ncols = 3
    nrows = (num_scenarios + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if num_scenarios > 1 else [axes]
    
    cmap = ListedColormap(['#CCCCCC'] + class_colors)
    
    for idx, (scenario_id, maps) in enumerate(prediction_maps.items()):
        ax = axes[idx]
        pred_map = maps['prediction_map']
        
        im = ax.imshow(pred_map, cmap=cmap, vmin=-1, vmax=4, interpolation='nearest')
        ax.set_title(f'Scenario {scenario_id}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_scenarios, len(axes)):
        axes[idx].axis('off')
    
    # Add shared colorbar
    class_names = ['NoData', 'No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    cbar = plt.colorbar(im, ax=axes, fraction=0.02, pad=0.04, ticks=[-1, 0, 1, 2, 3, 4])
    cbar.set_label('Flood Class', fontsize=11, fontweight='bold')
    cbar.ax.set_yticklabels(class_names, fontsize=9)
    
    plt.suptitle('Predicted Flood Maps - All Test Scenarios (Manila Core)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_path / 'prediction_grid_all_scenarios.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved prediction grid")


def generate_confidence_maps(prediction_maps, output_path):    
    confidence_maps = {}
    
    for scenario_id, maps in prediction_maps.items():
        confidence_map = maps['confidence_map']
        confidence_maps[scenario_id] = confidence_map
        
        # Visualize confidence
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Confidence map
        ax1 = axes[0]
        im1 = ax1.imshow(confidence_map, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        ax1.set_title(f'Prediction Confidence - Scenario {scenario_id}', 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046)
        cbar1.set_label('Confidence', fontsize=11)
        
        # Low confidence regions (< 0.5)
        ax2 = axes[1]
        low_conf_mask = (confidence_map < 0.5) & (confidence_map > 0)
        im2 = ax2.imshow(low_conf_mask, cmap='Reds', interpolation='nearest')
        ax2.set_title(f'Low Confidence Regions (<50%) - Scenario {scenario_id}', 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        low_conf_count = low_conf_mask.sum()
        valid_pixels = (confidence_map > 0).sum()
        low_conf_pct = low_conf_count / valid_pixels * 100 if valid_pixels > 0 else 0
        
        ax2.text(0.5, -0.05, f'Low confidence: {low_conf_count:,} pixels ({low_conf_pct:.1f}%)',
                transform=ax2.transAxes, ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / f'confidence_scenario_{scenario_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Saved {len(confidence_maps)} confidence maps")
    
    # Aggregate confidence statistics
    analyze_confidence_statistics(confidence_maps, output_path)
    
    return confidence_maps


def analyze_confidence_statistics(confidence_maps, output_path):    
    all_confidences = []
    
    for scenario_id, conf_map in confidence_maps.items():
        valid_conf = conf_map[conf_map > 0]
        all_confidences.append(valid_conf)
    
    all_confidences = np.concatenate(all_confidences)
    
    print(f"\n  Confidence Statistics:")
    print(f"    Mean:   {all_confidences.mean():.4f}")
    print(f"    Median: {np.median(all_confidences):.4f}")
    print(f"    Std:    {all_confidences.std():.4f}")
    print(f"    Min:    {all_confidences.min():.4f}")
    print(f"    Max:    {all_confidences.max():.4f}")
    
    # Distribution
    low_conf = (all_confidences < 0.5).sum() / len(all_confidences) * 100
    mid_conf = ((all_confidences >= 0.5) & (all_confidences < 0.8)).sum() / len(all_confidences) * 100
    high_conf = (all_confidences >= 0.8).sum() / len(all_confidences) * 100
    
    print(f"\n  Confidence Distribution:")
    print(f"    Low (<0.5):      {low_conf:.1f}%")
    print(f"    Medium (0.5-0.8): {mid_conf:.1f}%")
    print(f"    High (>0.8):     {high_conf:.1f}%")
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_confidences, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(all_confidences.mean(), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {all_confidences.mean():.3f}')
    ax.axvline(np.median(all_confidences), color='orange', linestyle='--', linewidth=2,
              label=f'Median: {np.median(all_confidences):.3f}')
    
    ax.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence Distribution (All Test Scenarios)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved confidence histogram")

def compare_with_ground_truth_cropped(prediction_maps, ground_truth_maps, output_path):
    
    print("\n[Comparison] Ground Truth vs Predictions (Cropped View)")
    print("-" * 70)
    
    class_colors = ['#FFFFFF', '#FDD835', '#FB8C00', '#E53935', '#6A1B9A']
    cmap = ListedColormap(['#CCCCCC'] + class_colors)
    
    for scenario_id in prediction_maps.keys():
        if scenario_id not in ground_truth_maps:
            continue
        
        pred_map = prediction_maps[scenario_id]['prediction_map']
        true_map = ground_truth_maps[scenario_id]
        
        # Find bounding box of valid data
        valid_mask = (true_map >= 0) & (pred_map >= 0)
        rows, cols = np.where(valid_mask)
        
        if len(rows) == 0:
            print(f"  ⚠ Scenario {scenario_id}: No valid data")
            continue
        
        # Add padding
        padding = 20
        row_min = max(0, rows.min() - padding)
        row_max = min(true_map.shape[0], rows.max() + padding)
        col_min = max(0, cols.min() - padding)
        col_max = min(true_map.shape[1], cols.max() + padding)
        
        # Crop to valid region
        true_crop = true_map[row_min:row_max, col_min:col_max]
        pred_crop = pred_map[row_min:row_max, col_min:col_max]
        
        # FIX: Add constrained_layout=True
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        
        # Ground truth
        ax1 = axes[0]
        im1 = ax1.imshow(true_crop, cmap=cmap, vmin=-1, vmax=4, interpolation='nearest')
        ax1.set_title('Ground Truth\n(Manila Core)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Prediction
        ax2 = axes[1]
        im2 = ax2.imshow(pred_crop, cmap=cmap, vmin=-1, vmax=4, interpolation='nearest')
        ax2.set_title('Prediction\n(Model Output)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Difference
        valid_crop_mask = (true_crop >= 0) & (pred_crop >= 0)
        diff_crop = np.full_like(true_crop, -1, dtype=np.int8)
        diff_crop[valid_crop_mask] = (pred_crop[valid_crop_mask] == true_crop[valid_crop_mask]).astype(np.int8)
        
        ax3 = axes[2]
        im3 = ax3.imshow(diff_crop, cmap=ListedColormap(['#CCCCCC', '#E53935', '#2E7D32']), 
                        vmin=-1, vmax=1, interpolation='nearest')
        ax3.set_title('Accuracy Map\n(Green=Correct, Red=Wrong)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Calculate accuracy
        correct = (pred_crop[valid_crop_mask] == true_crop[valid_crop_mask]).sum()
        total = valid_crop_mask.sum()
        accuracy = correct / total * 100 if total > 0 else 0
        
        # Shared colorbar
        class_names = ['NoData', 'No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
        cbar = plt.colorbar(im1, ax=axes[:2], fraction=0.02, pad=0.04, 
                           ticks=[-1, 0, 1, 2, 3, 4])
        cbar.set_label('Flood Class', fontsize=11, fontweight='bold')
        cbar.ax.set_yticklabels(class_names, fontsize=9)
        
        # Add info
        info_text = f'Region: [{row_min}:{row_max}, {col_min}:{col_max}]\nSize: {row_max-row_min}×{col_max-col_min} pixels'
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=9, style='italic')
        
        plt.suptitle(f'Scenario {scenario_id} - Spatial Accuracy: {accuracy:.1f}%', 
                    fontsize=14, fontweight='bold')
        # REMOVE: plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # ← REMOVE THIS LINE
        plt.savefig(output_path / f'comparison_cropped_scenario_{scenario_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Scenario {scenario_id}: Cropped to {row_max-row_min}×{col_max-col_min}, Accuracy: {accuracy:.1f}%")
    
    print(f"  ✓ Saved {len(prediction_maps)} cropped comparison plots")