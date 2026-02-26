import numpy as np
from typing import Dict, Tuple


def extract_vit_patches(split_results: Dict, patch_size: int) -> Dict:
    print("\n" + "="*70)
    print("PHASE 6: PATCH EXTRACTION FOR ViT")
    print("="*70)
    
    print(f"\n[6.1] Patch Configuration")
    print("-" * 70)
    
    spatial_shape = (1152, 1152)
    patches_per_dim = spatial_shape[0] // patch_size
    total_possible_patches = patches_per_dim ** 2
    
    print(f"  Spatial shape: {spatial_shape}")
    print(f"  Patch size: {patch_size}×{patch_size}")
    print(f"  Patches per dimension: {patches_per_dim}×{patches_per_dim}")
    print(f"  Total possible patches: {total_possible_patches:,}")
    print(f"  Stride: {patch_size} (non-overlapping)")
    
    # -------------------------------------------------------------------------
    # 6.2 EXTRACT DATA
    # -------------------------------------------------------------------------
    print(f"\n[6.2] Data Already Extracted in Phase 5")
    print("-" * 70)
    
    train_data = split_results['train']
    test_data = split_results['test']
    
    print(f"  ✓ Spatial patches extracted and masked")
    print(f"  ✓ Rainfall sequences prepared")
    print(f"  ✓ Labels categorized")
    
    # -------------------------------------------------------------------------
    # 6.3 FORMAT FOR ViT
    # -------------------------------------------------------------------------
    print(f"\n[6.3] Format Verification")
    print("-" * 70)
    
    # Training data
    X_train_spatial = train_data['spatial_patches']      # (N_train, 3, 16, 16)
    X_train_rainfall = train_data['rainfall_sequences']   # (N_train, 13)
    X_train_conditioning = train_data.get('conditioning_vectors')
    y_train = train_data['labels']                        # (N_train,)
    
    # Test data
    X_test_spatial = test_data['spatial_patches']         # (N_test, 3, 16, 16)
    X_test_rainfall = test_data['rainfall_sequences']     # (N_test, 13)
    X_test_conditioning = test_data.get('conditioning_vectors') 
    y_test = test_data['labels']                          # (N_test,)
    
    print(f"  Training Set:")
    print(f"    Spatial patches: {X_train_spatial.shape} (N, C, H, W)")
    print(f"    Rainfall seqs:   {X_train_rainfall.shape} (N, T)")
    print(f"    Labels:          {y_train.shape} (N,)")
    
    print(f"\n  Test Set:")
    print(f"    Spatial patches: {X_test_spatial.shape} (N, C, H, W)")
    print(f"    Rainfall seqs:   {X_test_rainfall.shape} (N, T)")
    print(f"    Labels:          {y_test.shape} (N,)")
    
    # -------------------------------------------------------------------------
    # 6.4 QUALITY CHECKS
    # -------------------------------------------------------------------------
    print(f"\n[6.4] Quality Checks")
    print("-" * 70)
    
    checks = {
        'spatial_shape_correct': (
            X_train_spatial.shape[1:] == (3, patch_size, patch_size) and
            X_test_spatial.shape[1:] == (3, patch_size, patch_size)
        ),
        'rainfall_shape_correct': (
            X_train_rainfall.shape[1] == 13 and
            X_test_rainfall.shape[1] == 13
        ),
        'no_nans': (
            not np.isnan(X_train_spatial).any() and
            not np.isnan(X_test_spatial).any() and
            not np.isnan(X_train_rainfall).any() and
            not np.isnan(X_test_rainfall).any()
        ),
        'labels_valid': (
            np.all((y_train >= 0) & (y_train <= 4)) and
            np.all((y_test >= 0) & (y_test <= 4))
        ),
        'sample_counts_match': (
            len(X_train_spatial) == len(X_train_rainfall) == len(y_train) and
            len(X_test_spatial) == len(X_test_rainfall) == len(y_test)
        )
    }
    
    for check_name, passed in checks.items():
        print(f"  {check_name}: {'✓' if passed else '✗'}")
    
    all_passed = all(checks.values())
    
    # -------------------------------------------------------------------------
    # 6.5 SUMMARY STATISTICS
    # -------------------------------------------------------------------------
    print(f"\n[6.5] Dataset Summary")
    print("-" * 70)
    
    print(f"  Training:")
    print(f"    Samples: {len(y_train):,}")
    print(f"    Scenarios: {train_data['num_scenarios']}")
    print(f"    Region: GMM (outside Manila)")
    print(f"    Class balance:", end=' ')
    for cls in range(5):
        count = (y_train == cls).sum()
        print(f"{cls}:{count:,}", end=' ')
    print()
    
    print(f"\n  Test:")
    print(f"    Samples: {len(y_test):,}")
    print(f"    Scenarios: {test_data['num_scenarios']}")
    print(f"    Region: Manila core")
    print(f"    Class balance:", end=' ')
    for cls in range(5):
        count = (y_test == cls).sum()
        print(f"{cls}:{count:,}", end=' ')
    print()
    
    results = {
        'train': {
            'spatial_patches': X_train_spatial,
            'rainfall_sequences': X_train_rainfall,
            'conditioning_vectors': X_train_conditioning,
            'labels': y_train,
            'metadata': train_data['scenario_metadata'],
            'num_samples': len(y_train),
            'scenario_ids': train_data['scenario_ids']
        },
        'test': {
            'spatial_patches': X_test_spatial,
            'rainfall_sequences': X_test_rainfall,
            'conditioning_vectors': X_test_conditioning,
            'labels': y_test,
            'metadata': test_data['scenario_metadata'],
            'num_samples': len(y_test),
            'scenario_ids': test_data['scenario_ids']
        },
        'config': {
            'patch_size': patch_size,
            'spatial_shape': spatial_shape,
            'patches_per_dim': patches_per_dim,
            'num_classes': 5,
            'spatial_channels': 3,
            'rainfall_timesteps': 13
        },
        'encoding_info': split_results.get('encoding_info'),
        'quality_checks': checks
    }
    
    print(f"\n{'='*70}")
    print(f"ViT PATCH EXTRACTION COMPLETE")
    print(f"  Train: {len(y_train):,} samples ready")
    print(f"  Test: {len(y_test):,} samples ready")
    print(f"  All checks: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    print(f"{'='*70}\n")
    
    return results


def get_data_loader_ready(results: Dict) -> Tuple[Tuple, Tuple]:
    train_data = (
        results['train']['spatial_patches'],
        results['train']['rainfall_sequences'],
        results['train'].get('conditioning_vectors'),  
        results['train']['labels']
    )
    
    test_data = (
        results['test']['spatial_patches'],
        results['test']['rainfall_sequences'],
        results['test'].get('conditioning_vectors'),
        results['test']['labels']
    )
    
    return train_data, test_data