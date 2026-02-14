import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support,
)

def reconstruct_test_map_from_patches(
    patch_predictions: np.ndarray,
    original_shape: Tuple[int, int] = (320, 320),
    patch_size: int = 4
) -> np.ndarray:
    """
    Reconstruct full 320×320 map from 6400 patch predictions.
    """
    H, W = original_shape
    patches_h = H // patch_size  # 80
    patches_w = W // patch_size  # 80
    
    # Reshape from flat (6400,) to grid (80, 80)
    patch_grid = patch_predictions.reshape(patches_h, patches_w)
    
    # Expand each patch to 4×4 pixels
    reconstructed_map = np.repeat(np.repeat(patch_grid, patch_size, axis=0), patch_size, axis=1)
    
    return reconstructed_map


# =============================================================================
# 7.2 SCENARIO-LEVEL PREDICTION
# =============================================================================

def predict_scenario(
    model: nn.Module,
    scenario_dataset,
    scenario_idx: int,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Predict all patches for one scenario.
    """
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


# =============================================================================
# 7.3 EVALUATION METRICS
# =============================================================================

def calculate_iou_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5
) -> Dict[int, float]:
    """
    Calculate IoU (Intersection over Union) for each class.
    """
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


def calculate_scenario_metrics(
    ground_truth_map: np.ndarray,
    predicted_map: np.ndarray,
    num_classes: int = 5
) -> Dict:
    """
    Calculate comprehensive metrics for one scenario.
    """
    # Flatten for sklearn metrics
    y_true = ground_truth_map.flatten()
    y_pred = predicted_map.flatten()
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )
    
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


# =============================================================================
# 7.4 EVALUATE ALL TEST SCENARIOS
# =============================================================================

def evaluate_test_scenarios(
    model: nn.Module,
    test_dataset,
    test_scenario_ids: List[int],
    ground_truth_maps: Dict[int, np.ndarray],
    device: str = 'cuda',
    save_dir: str = './results'
) -> Dict:
    """
    Evaluate model on all test scenarios.
    """
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
        predicted_map = reconstruct_test_map_from_patches(patch_predictions)
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
    
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {save_dir / 'test_results.json'}")
    
    return all_results, all_predictions


# =============================================================================
# 7.5 VISUALIZATION: Confusion Matrix
# =============================================================================

def plot_confusion_matrix(
    metrics: Dict,
    scenario_id: int,
    save_path: str = None
):
    """Plot confusion matrix for one scenario."""
    cm = np.array(metrics['confusion_matrix'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=range(5),
        yticklabels=range(5),
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - Scenario {scenario_id}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


# =============================================================================
# 7.6 VISUALIZATION: Ground Truth vs Prediction
# =============================================================================

def plot_scenario_comparison(
    scenario_id: int,
    ground_truth_map: np.ndarray,
    predicted_map: np.ndarray,
    save_path: str = None
):
    """
    Side-by-side comparison of ground truth vs prediction.
    
    Args:
        scenario_id: Scenario ID
        ground_truth_map: (320, 320) ground truth
        predicted_map: (320, 320) predictions
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Ground truth
    im1 = axes[0].imshow(ground_truth_map, cmap='YlOrRd', vmin=0, vmax=4)
    axes[0].set_title(f'Ground Truth - Scenario {scenario_id}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='Flood Category')
    
    # Prediction
    im2 = axes[1].imshow(predicted_map, cmap='YlOrRd', vmin=0, vmax=4)
    axes[1].set_title(f'Prediction - Scenario {scenario_id}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], label='Flood Category')
    
    # Difference (error map)
    diff = np.abs(ground_truth_map - predicted_map)
    im3 = axes[2].imshow(diff, cmap='Reds', vmin=0, vmax=4)
    axes[2].set_title(f'Absolute Error - Scenario {scenario_id}')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], label='Error Magnitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


