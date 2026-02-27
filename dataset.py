import numpy as np
import json
from typing import Dict, Optional
from pathlib import Path


def create_complete_dataset(
    vit_results: Dict,
    output_dir: str = './processed_dataset'
) -> Dict:
    """
    Create complete dataset with conditioning vectors.
    
    NEW: Includes conditioning vectors for each sample
    """
    print("\n" + "="*70)
    print("PHASE 7: CREATE COMPLETE DATASET (WITH CONDITIONING)")
    print("="*70)
        
    print(f"\n[7.1-7.2] Dataset Structure")
    print("-" * 70)
    
    # Training data
    X_train_spatial = vit_results['train']['spatial_patches']
    X_train_rainfall = vit_results['train']['rainfall_sequences']
    X_train_conditioning = vit_results['train'].get('conditioning_vectors')  # NEW
    y_train = vit_results['train']['labels']
    train_metadata = vit_results['train']['metadata']
    
    # Test data
    X_test_spatial = vit_results['test']['spatial_patches']
    X_test_rainfall = vit_results['test']['rainfall_sequences']
    X_test_conditioning = vit_results['test'].get('conditioning_vectors')  # NEW
    y_test = vit_results['test']['labels']
    test_metadata = vit_results['test']['metadata']
    
    print(f"  Training Dataset:")
    print(f"    Spatial patches:    {X_train_spatial.shape}")
    print(f"    Rainfall sequences: {X_train_rainfall.shape}")
    if X_train_conditioning is not None:
        print(f"    Conditioning vecs:  {X_train_conditioning.shape}")  # NEW
    print(f"    Labels:             {y_train.shape}")
    print(f"    Total samples:      {len(y_train):,}")
    
    print(f"\n  Test Dataset:")
    print(f"    Spatial patches:    {X_test_spatial.shape}")
    print(f"    Rainfall sequences: {X_test_rainfall.shape}")
    if X_test_conditioning is not None:
        print(f"    Conditioning vecs:  {X_test_conditioning.shape}")  # NEW
    print(f"    Labels:             {y_test.shape}")
    print(f"    Total samples:      {len(y_test):,}")
    
    # -------------------------------------------------------------------------
    # 7.3 DATA VALIDATION (Updated)
    # -------------------------------------------------------------------------
    print(f"\n[7.3] Data Validation")
    print("-" * 70)
    
    validation = _validate_dataset(
        X_train_spatial, X_train_rainfall, X_train_conditioning, y_train,
        X_test_spatial, X_test_rainfall, X_test_conditioning, y_test
    )

    dataset = {
        'train': {
            'spatial': X_train_spatial,
            'rainfall': X_train_rainfall,
            'conditioning': X_train_conditioning,  # NEW
            'labels': y_train,
            'metadata': train_metadata,
            'scenario_ids': vit_results['train']['scenario_ids']
        },
        'test': {
            'spatial': X_test_spatial,
            'rainfall': X_test_rainfall,
            'conditioning': X_test_conditioning,  # NEW
            'labels': y_test,
            'metadata': test_metadata,
            'scenario_ids': vit_results['test']['scenario_ids']
        },
        'config': vit_results['config'],
        'validation': validation,
        'output_dir': output_dir,
        'encoding_info': vit_results.get('encoding_info')  # NEW
    }
    
    return dataset


def _validate_dataset(
    X_train_spatial, X_train_rainfall, X_train_conditioning, y_train,
    X_test_spatial, X_test_rainfall, X_test_conditioning, y_test
) -> Dict:
    """
    Validate dataset with conditioning vectors.
    """
    checks = {}
    
    # Shape consistency (updated to include conditioning)
    train_shapes_ok = (
        len(X_train_spatial) == len(X_train_rainfall) == len(y_train)
    )
    if X_train_conditioning is not None:
        train_shapes_ok = train_shapes_ok and (len(X_train_conditioning) == len(y_train))
    
    test_shapes_ok = (
        len(X_test_spatial) == len(X_test_rainfall) == len(y_test)
    )
    if X_test_conditioning is not None:
        test_shapes_ok = test_shapes_ok and (len(X_test_conditioning) == len(y_test))
    
    checks['shapes_consistent'] = train_shapes_ok and test_shapes_ok
    print(f"  Shape consistency: {'✓' if checks['shapes_consistent'] else '✗'}")
    
    # No NaN values
    train_no_nan = (
        not np.isnan(X_train_spatial).any() and
        not np.isnan(X_train_rainfall).any()
    )
    if X_train_conditioning is not None:
        train_no_nan = train_no_nan and (not np.isnan(X_train_conditioning).any())
    
    test_no_nan = (
        not np.isnan(X_test_spatial).any() and
        not np.isnan(X_test_rainfall).any()
    )
    if X_test_conditioning is not None:
        test_no_nan = test_no_nan and (not np.isnan(X_test_conditioning).any())
    
    checks['no_nans'] = train_no_nan and test_no_nan
    print(f"  No NaN values: {'✓' if checks['no_nans'] else '✗'}")
    
    # Label range [0-4]
    train_labels_valid = np.all((y_train >= 0) & (y_train <= 4))
    test_labels_valid = np.all((y_test >= 0) & (y_test <= 4))
    checks['labels_valid'] = train_labels_valid and test_labels_valid
    print(f"  Label range [0-4]: {'✓' if checks['labels_valid'] else '✗'}")
    
    # Data range [0-1] for normalized inputs
    train_range_ok = (
        X_train_spatial.min() >= 0 and X_train_spatial.max() <= 1 and
        X_train_rainfall.min() >= 0 and X_train_rainfall.max() <= 1
    )
    test_range_ok = (
        X_test_spatial.min() >= 0 and X_test_spatial.max() <= 1 and
        X_test_rainfall.min() >= 0 and X_test_rainfall.max() <= 1
    )
    checks['data_normalized'] = train_range_ok and test_range_ok
    print(f"  Data normalized [0-1]: {'✓' if checks['data_normalized'] else '✗'}")
    
    # Conditioning vector checks (NEW)
    if X_train_conditioning is not None and X_test_conditioning is not None:
        # Check dimensions
        cond_dim_ok = (
            X_train_conditioning.shape[1] == 4 and
            X_test_conditioning.shape[1] == 4
        )
        print(f"  Conditioning dims (4): {'✓' if cond_dim_ok else '✗'}")
        checks['conditioning_dims_ok'] = cond_dim_ok
        
        # Check pattern type range [0-3]
        pattern_ok = (
            X_train_conditioning[:, 0].min() >= 0 and
            X_train_conditioning[:, 0].max() <= 3 and
            X_test_conditioning[:, 0].min() >= 0 and
            X_test_conditioning[:, 0].max() <= 3
        )
        print(f"  Pattern types [0-3]: {'✓' if pattern_ok else '✗'}")
        checks['pattern_types_ok'] = pattern_ok
    else:
        print(f"  ⚠ Conditioning vectors not present")
        checks['conditioning_dims_ok'] = True  # Don't fail if not present
        checks['pattern_types_ok'] = True
    
    # Class distribution
    print(f"\n  Class Distribution:")
    train_dist, test_dist = _check_class_distribution(y_train, y_test)
    checks['train_distribution'] = train_dist
    checks['test_distribution'] = test_dist
    
    # Check for severe imbalance
    train_imbalance = max(train_dist.values()) / min(train_dist.values()) if train_dist else float('inf')
    test_imbalance = max(test_dist.values()) / min(test_dist.values()) if test_dist else float('inf')
    checks['imbalance_ratio'] = {'train': train_imbalance, 'test': test_imbalance}
    
    if train_imbalance > 100 or test_imbalance > 100:
        print(f"    ⚠ WARNING: Severe class imbalance detected (ratio > 100:1)")
    
    checks['all_passed'] = (
        checks['shapes_consistent'] and
        checks['no_nans'] and
        checks['labels_valid'] and
        checks['data_normalized'] and
        checks['conditioning_dims_ok'] and
        checks['pattern_types_ok']
    )
    
    return checks


def _check_class_distribution(y_train, y_test) -> tuple:
    """Unchanged from original"""
    train_dist = {}
    test_dist = {}
    
    print(f"    {'Class':<10} {'Train':<20} {'Test':<20}")
    print(f"    {'-'*50}")
    
    for cls in range(5):
        train_count = (y_train == cls).sum()
        test_count = (y_test == cls).sum()
        
        train_pct = train_count / len(y_train) * 100 if len(y_train) > 0 else 0
        test_pct = test_count / len(y_test) * 100 if len(y_test) > 0 else 0
        
        train_dist[cls] = int(train_count)
        test_dist[cls] = int(test_count)
        
        class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
        print(f"    {cls} ({class_names[cls]:<8}): "
              f"{train_count:>6,} ({train_pct:>5.1f}%)  "
              f"{test_count:>6,} ({test_pct:>5.1f}%)")
    
    return train_dist, test_dist