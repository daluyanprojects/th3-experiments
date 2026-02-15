import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from training import print_metrics, plot_confusion_matrix

def evaluate_model(trainer, test_loader, train_loader=None, output_dir='./evaluation_results'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 12: COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # =========================================================================
    # 12.1 LOAD BEST MODEL
    # =========================================================================
    print("\n[12.1] Loading Best Model")
    print("-" * 70)
    
    best_checkpoint = trainer.checkpoint_dir / 'best_model.pth'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))
        print(f"  ✓ Loaded best model from epoch {trainer.current_epoch}")
    else:
        print(f"  ⚠ Best checkpoint not found, using current model state")
    
    trainer.model.eval()
    
    # =========================================================================
    # 12.2 & 12.3 TEST SET PREDICTION & METRICS
    # =========================================================================
    print("\n[12.2-12.3] Test Set Evaluation")
    print("-" * 70)
    
    test_metrics = trainer.test(test_loader, split_name='Test')
    
    # =========================================================================
    # 12.4 SPATIAL GENERALIZATION ANALYSIS
    # =========================================================================
    print("\n[12.4] Spatial Generalization Analysis")
    print("-" * 70)
    
    generalization_analysis = analyze_spatial_generalization(
        trainer,
        test_metrics,
        train_loader,
        test_loader,
        output_path
    )
    
    # =========================================================================
    # 12.5 ERROR ANALYSIS
    # =========================================================================
    print("\n[12.5] Error Analysis")
    print("-" * 70)
    
    error_analysis = perform_error_analysis(
        trainer,
        test_loader,
        output_path
    )
    
    # =========================================================================
    # GENERATE FINAL REPORT
    # =========================================================================
    generate_evaluation_report(
        test_metrics,
        generalization_analysis,
        error_analysis,
        output_path
    )
    
    print("\n" + "="*70)
    print("  EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_path}")
    print("="*70 + "\n")
    
    return {
        'test_metrics': test_metrics,
        'generalization': generalization_analysis,
        'errors': error_analysis
    }


def analyze_spatial_generalization( trainer, test_metrics, train_loader, test_loader, output_path):    
    analysis = {}
    
    # Get train metrics if available
    if train_loader is not None:
        print("\n  Computing training set metrics...")
        train_metrics = trainer.test(train_loader, split_name='Train')
        
        # Compare train vs test
        train_acc = train_metrics['accuracy']
        test_acc = test_metrics['accuracy']
        generalization_gap = train_acc - test_acc
        
        analysis['train_accuracy'] = train_acc
        analysis['test_accuracy'] = test_acc
        analysis['generalization_gap'] = generalization_gap
        
        print(f"\n  Generalization Gap:")
        print(f"    Train Accuracy (GMM):    {train_acc:.4f}")
        print(f"    Test Accuracy (Manila):  {test_acc:.4f}")
        print(f"    Gap:                     {generalization_gap:.4f}")
        
        if generalization_gap < 0.05:
            print(f"    → Excellent generalization! ✓")
        elif generalization_gap < 0.10:
            print(f"    → Good generalization")
        elif generalization_gap < 0.20:
            print(f"    → Moderate generalization")
        else:
            print(f"    → Poor generalization ⚠")
        
        # Per-class transfer analysis
        print(f"\n  Per-Class Transfer Performance:")
        print(f"  {'Class':<12} {'Train F1':>10} {'Test F1':>10} {'Transfer':>10}")
        print(f"  {'-'*44}")
        
        class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
        class_transfer = {}
        
        for i, name in enumerate(class_names):
            train_f1 = train_metrics[f'f1_{name}']
            test_f1 = test_metrics[f'f1_{name}']
            transfer = test_f1 / train_f1 if train_f1 > 0 else 0
            
            class_transfer[name] = {
                'train_f1': train_f1,
                'test_f1': test_f1,
                'transfer_ratio': transfer
            }
            
            status = "✓" if transfer > 0.8 else "⚠" if transfer > 0.5 else "✗"
            print(f"  {name:<12} {train_f1:>10.4f} {test_f1:>10.4f} {transfer:>9.2f}x {status}")
        
        analysis['class_transfer'] = class_transfer
        
        # Plot comparison
        plot_train_test_comparison(train_metrics, test_metrics, output_path)
    
    else:
        analysis['train_accuracy'] = None
        analysis['test_accuracy'] = test_metrics['accuracy']
        print("\n  ⚠ No training loader provided, skipping train/test comparison")
    
    return analysis


def perform_error_analysis(trainer, test_loader, output_path):    
    trainer.model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    all_spatial = []
    all_rainfall = []
    
    # Collect all predictions
    print("\n  Collecting predictions for error analysis...")
    with torch.no_grad():
        for spatial, rainfall, labels in test_loader:
            spatial = spatial.to(trainer.device)
            rainfall = rainfall.to(trainer.device)
            
            logits, _ = trainer.model(spatial, rainfall)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_spatial.append(spatial.cpu().numpy())
            all_rainfall.append(rainfall.cpu().numpy())
    
    # Convert to arrays
    y_pred = np.array(all_predictions)
    y_true = np.array(all_labels)
    y_probs = np.vstack(all_probs)
    X_spatial = np.vstack(all_spatial)
    X_rainfall = np.vstack(all_rainfall)
    
    # Find errors
    errors = y_pred != y_true
    error_indices = np.where(errors)[0]
    
    print(f"\n  Total errors: {errors.sum():,} / {len(y_true):,} ({errors.sum()/len(y_true)*100:.1f}%)")
    
    # Analyze error patterns
    error_analysis = {}
    
    # 1. Confusion patterns
    print(f"\n  Most Common Misclassifications:")
    confusion_pairs = {}
    for idx in error_indices:
        pair = (int(y_true[idx]), int(y_pred[idx]))
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    
    class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    for (true_cls, pred_cls), count in sorted_pairs[:10]:
        pct = count / errors.sum() * 100
        print(f"    {class_names[true_cls]:12s} → {class_names[pred_cls]:12s}: {count:4d} ({pct:5.1f}%)")
    
    error_analysis['confusion_pairs'] = confusion_pairs
    
    # 2. Confidence analysis
    correct_probs = y_probs[~errors].max(axis=1)
    error_probs = y_probs[errors].max(axis=1)
    
    print(f"\n  Confidence Analysis:")
    print(f"    Correct predictions - Mean confidence: {correct_probs.mean():.4f}")
    print(f"    Wrong predictions   - Mean confidence: {error_probs.mean():.4f}")
    
    # Find high-confidence errors 
    high_conf_errors = error_indices[error_probs > 0.8]
    if len(high_conf_errors) > 0:
        print(f"    ⚠ High-confidence errors: {len(high_conf_errors)} samples (>80% confidence but wrong)")
    
    error_analysis['correct_confidence'] = float(correct_probs.mean())
    error_analysis['error_confidence'] = float(error_probs.mean())
    error_analysis['high_conf_errors'] = len(high_conf_errors)
    
    # 3. Per-class error rates
    print(f"\n  Per-Class Error Rates:")
    print(f"  {'Class':<12} {'Total':>8} {'Errors':>8} {'Rate':>8}")
    print(f"  {'-'*36}")
    
    class_errors = {}
    for cls in range(5):
        cls_mask = y_true == cls
        if cls_mask.sum() > 0:
            cls_error_rate = (y_pred[cls_mask] != y_true[cls_mask]).sum() / cls_mask.sum()
            class_errors[class_names[cls]] = {
                'total': int(cls_mask.sum()),
                'errors': int((y_pred[cls_mask] != y_true[cls_mask]).sum()),
                'error_rate': float(cls_error_rate)
            }
            print(f"  {class_names[cls]:<12} {cls_mask.sum():>8} {(y_pred[cls_mask] != y_true[cls_mask]).sum():>8} {cls_error_rate:>7.1%}")
    
    error_analysis['class_errors'] = class_errors
    
    # 4. Visualize error examples
    plot_error_examples(X_spatial, X_rainfall, y_true, y_pred, y_probs, error_indices, class_names, output_path)
    
    # 5. Confidence distribution
    plot_confidence_distribution(correct_probs, error_probs, output_path)
    
    return error_analysis


def plot_train_test_comparison(train_metrics, test_metrics, output_path):    
    class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # F1 Score comparison
    ax1 = axes[0]
    train_f1 = [train_metrics[f'f1_{name}'] for name in class_names]
    test_f1 = [test_metrics[f'f1_{name}'] for name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax1.bar(x - width/2, train_f1, width, label='Train (GMM)', color='#2196F3')
    ax1.bar(x + width/2, test_f1, width, label='Test (Manila)', color='#FF9800')
    
    ax1.set_xlabel('Flood Class', fontweight='bold')
    ax1.set_ylabel('F1 Score', fontweight='bold')
    ax1.set_title('Per-Class F1: Train vs Test', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)
    
    ax2 = axes[1]
    train_recall = [train_metrics[f'recall_{name}'] for name in class_names]
    test_recall = [test_metrics[f'recall_{name}'] for name in class_names]
    
    ax2.bar(x - width/2, train_recall, width, label='Train (GMM)', color='#2196F3')
    ax2.bar(x + width/2, test_recall, width, label='Test (Manila)', color='#FF9800')
    
    ax2.set_xlabel('Flood Class', fontweight='bold')
    ax2.set_ylabel('Recall', fontweight='bold')
    ax2.set_title('Per-Class Recall: Train vs Test', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path / 'train_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: train_test_comparison.png")


def plot_error_examples(X_spatial, X_rainfall, y_true, y_pred, y_probs, error_indices, class_names, output_path, num_examples=6):
    
    if len(error_indices) == 0:
        print("    No errors to visualize!")
        return
    
    # Select diverse errors
    num_examples = min(num_examples, len(error_indices))
    example_indices = np.random.choice(error_indices, num_examples, replace=False)
    
    fig, axes = plt.subplots(2, num_examples, figsize=(3*num_examples, 6))
    if num_examples == 1:
        axes = axes.reshape(2, 1)
    
    for col, idx in enumerate(example_indices):
        # Show spatial patch (DEM channel)
        ax_spatial = axes[0, col]
        ax_spatial.imshow(X_spatial[idx, 0], cmap='terrain')
        ax_spatial.axis('off')
        
        true_cls = int(y_true[idx])
        pred_cls = int(y_pred[idx])
        conf = y_probs[idx, pred_cls]
        
        ax_spatial.set_title(
            f'True: {class_names[true_cls]}\n'
            f'Pred: {class_names[pred_cls]} ({conf:.2f})',
            fontsize=9, color='red'
        )
        
        # Show rainfall sequence
        ax_rain = axes[1, col]
        timesteps = np.arange(len(X_rainfall[idx]))
        ax_rain.plot(timesteps, X_rainfall[idx], 'b-o', linewidth=2, markersize=4)
        ax_rain.fill_between(timesteps, X_rainfall[idx], alpha=0.3)
        ax_rain.set_xlabel('Timestep', fontsize=8)
        ax_rain.set_ylabel('Rainfall', fontsize=8)
        ax_rain.grid(alpha=0.3)
        ax_rain.set_ylim(0, 1)
    
    plt.suptitle('Error Examples: Spatial Patches & Rainfall Sequences', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'error_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: error_examples.png")


def plot_confidence_distribution(correct_probs, error_probs, output_path):    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 21)
    
    ax.hist(correct_probs, bins=bins, alpha=0.6, label='Correct', color='green', edgecolor='black')
    ax.hist(error_probs, bins=bins, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    
    ax.set_xlabel('Prediction Confidence', fontweight='bold', fontsize=12)
    ax.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', 
                 fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: confidence_distribution.png")


def generate_evaluation_report(test_metrics, generalization, errors, output_path):
    report_path = output_path / 'evaluation_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f: 
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Test Performance
        f.write("TEST SET PERFORMANCE (Manila Core)\n")
        f.write("-"*70 + "\n")
        f.write(f"Overall Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1 Score:   {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1:      {test_metrics['weighted_f1']:.4f}\n\n")
        
        # Spatial Generalization
        if generalization['train_accuracy'] is not None:
            f.write("SPATIAL GENERALIZATION (GMM → Manila)\n")  # Arrow will work with UTF-8
            f.write("-"*70 + "\n")
            f.write(f"Train Accuracy (GMM):   {generalization['train_accuracy']:.4f}\n")
            f.write(f"Test Accuracy (Manila): {generalization['test_accuracy']:.4f}\n")
            f.write(f"Generalization Gap:     {generalization['generalization_gap']:.4f}\n\n")
        
        # Error Analysis
        f.write("ERROR ANALYSIS\n")
        f.write("-"*70 + "\n")
        f.write(f"Correct Confidence: {errors['correct_confidence']:.4f}\n")
        f.write(f"Error Confidence:   {errors['error_confidence']:.4f}\n")
        f.write(f"High-Conf Errors:   {errors['high_conf_errors']}\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"\n  ✓ Saved: evaluation_report.txt")

