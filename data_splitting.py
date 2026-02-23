import numpy as np
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt


def split_spatial_data(
    categorization_results: Dict,
    preprocessed_spatial: Dict,
    rainfall_results: Dict,
    train_ratio: float,
    random_seed: int,
    patch_size: int
) -> Dict:
    
    print("\n" + "="*70)
    print("SPATIAL DATA SPLITTING")
    print("="*70)
    
    np.random.seed(random_seed)
    
    print(f"\n[5.1] Spatial Split Logic")
    print("-" * 70)
    
    print("  Data Structure:")
    print("    • Train region: Greater Metro Manila (GMM) - outside Manila box")
    print("    • Test region: Metro Manila core - inside Manila box")
    print("    • Spatially complementary (non-overlapping)")
    print()
    print("  Split Strategy:")
    print("    • All 50 scenarios used for BOTH regions")
    print("    • Training scenarios → GMM patches")
    print("    • Testing scenarios → Manila patches")
    
    # -------------------------------------------------------------------------
    # 5.2 IDENTIFY SPATIAL MASKS
    # -------------------------------------------------------------------------
    print(f"\n[5.2] Spatial Masks")
    print("-" * 70)
    
    # Get masks from preprocessed data
    manila_box_mask = preprocessed_spatial['mask']  # 1 = Manila, 0 = GMM
    
    # Create named masks
    gmm_mask = (1 - manila_box_mask).astype(bool)      # True in GMM region
    manila_mask = manila_box_mask.astype(bool)          # True in Manila region
    
    # Verify masks
    total_pixels = manila_box_mask.size
    gmm_pixels = gmm_mask.sum()
    manila_pixels = manila_mask.sum()
    
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  GMM region (train): {gmm_pixels:,} pixels ({gmm_pixels/total_pixels*100:.1f}%)")
    print(f"  Manila region (test): {manila_pixels:,} pixels ({manila_pixels/total_pixels*100:.1f}%)")
    print(f"  Coverage check: {gmm_pixels + manila_pixels:,} / {total_pixels:,} {'✓' if gmm_pixels + manila_pixels == total_pixels else '✗'}")
    
    # Verify alignment with categorization results
    train_patches = categorization_results['train']['num_valid_patches']
    test_patches = categorization_results['test']['num_valid_patches']
    
    print(f"\n  Patch Alignment:")
    print(f"    Train (GMM): {train_patches:,} patches")
    print(f"    Test (Manila): {test_patches:,} patches")
    
    # -------------------------------------------------------------------------
    # 5.3 RANDOM SCENARIO ASSIGNMENT
    # -------------------------------------------------------------------------
    print(f"\n[5.3] Random Scenario Assignment")
    print("-" * 70)
    
    num_scenarios = categorization_results['train']['num_scenarios']
    num_train_scenarios = int(num_scenarios * train_ratio)
    num_test_scenarios = num_scenarios - num_train_scenarios
    
    print(f"  Total scenarios: {num_scenarios}")
    print(f"  Train scenarios: {num_train_scenarios} ({train_ratio*100:.0f}%)")
    print(f"  Test scenarios: {num_test_scenarios} ({(1-train_ratio)*100:.0f}%)")
    print(f"  Random seed: {random_seed}")
    
    # Random scenario assignment
    all_scenario_ids = np.arange(1, num_scenarios + 1)
    np.random.shuffle(all_scenario_ids)
    
    train_scenario_ids = sorted(all_scenario_ids[:num_train_scenarios].tolist())
    test_scenario_ids = sorted(all_scenario_ids[num_train_scenarios:].tolist())
    
    print(f"\n  Train scenarios (GMM region):")
    print(f"    {train_scenario_ids}")
    print(f"\n  Test scenarios (Manila region):")
    print(f"    {test_scenario_ids}")
    
    # -------------------------------------------------------------------------
    # 5.4 EXTRACT DATA FOR EACH SPLIT
    # -------------------------------------------------------------------------
    print(f"\n[5.4] Extracting Split Data")
    print("-" * 70)
    
    # Extract training data (GMM patches from train scenarios)
    print("\n  Extracting Training Data (GMM region)...")
    train_data = _extract_split_data(
        scenario_ids=train_scenario_ids,
        spatial_inputs=preprocessed_spatial['train'],
        rainfall_sequences=rainfall_results['sequences'],
        categorized_maps=categorization_results['train']['categorized_maps'],
        patch_coords=categorization_results['train']['patch_coords'],
        split_name='train',
        patch_size=patch_size
    )
    
    # Extract testing data (Manila patches from test scenarios)
    print("\n  Extracting Testing Data (Manila region)...")
    test_data = _extract_split_data(
        scenario_ids=test_scenario_ids,
        spatial_inputs=preprocessed_spatial['test'],
        rainfall_sequences=rainfall_results['sequences'],
        categorized_maps=categorization_results['test']['categorized_maps'],
        patch_coords=categorization_results['test']['patch_coords'],
        split_name='test',
        patch_size=patch_size
    )
    
    # -------------------------------------------------------------------------
    # 5.5 VERIFY SPLIT
    # -------------------------------------------------------------------------
    print(f"\n[5.5] Split Verification")
    print("-" * 70)
    
    verification = _verify_split(train_data, test_data, train_scenario_ids, test_scenario_ids)
    

    results = {
        'train': train_data,
        'test': test_data,
        'metadata': {
            'random_seed': random_seed,
            'train_ratio': train_ratio,
            'num_scenarios': num_scenarios,
            'num_train_scenarios': num_train_scenarios,
            'num_test_scenarios': num_test_scenarios,
            'train_scenario_ids': train_scenario_ids,
            'test_scenario_ids': test_scenario_ids,
            'spatial_regions': {
                'train': 'Greater Metro Manila (GMM) - outside Manila box',
                'test': 'Metro Manila core - inside Manila box'
            },
            'gmm_pixels': int(gmm_pixels),
            'manila_pixels': int(manila_pixels),
            'train_patches': int(train_patches),
            'test_patches': int(test_patches),
            'split_strategy': 'Spatially complementary with random scenario assignment'
        },
        'masks': {
            'gmm_mask': gmm_mask,
            'manila_mask': manila_mask,
            'manila_box_mask': manila_box_mask
        },
        'verification': verification
    }
    
    print(f"\n{'='*70}")
    print(f"SPATIAL SPLITTING COMPLETE")
    print(f"  Train: {num_train_scenarios} scenarios × {train_patches:,} patches = {train_data['total_samples']:,} samples")
    print(f"  Test: {num_test_scenarios} scenarios × {test_patches:,} patches = {test_data['total_samples']:,} samples")
    print(f"  All verifications: {'✓ PASSED' if verification['all_passed'] else '✗ FAILED'}")
    print(f"{'='*70}\n")
    
    return results



def _extract_split_data( scenario_ids: List[int], spatial_inputs: Dict, rainfall_sequences: np.ndarray,
    categorized_maps: List[Dict], patch_coords: List[Tuple], split_name: str, patch_size : int
) -> Dict:
    
    dem = spatial_inputs['dem']
    infiltration = spatial_inputs['infiltration']
    landuse = spatial_inputs['landuse']

    num_patches = len(patch_coords)
    
    print(f"    Spatial inputs shape: {dem.shape}")
    print(f"    Number of patches: {num_patches:,}")
    print(f"    Number of scenarios: {len(scenario_ids)}")
    
    # Initialize storage
    all_spatial_patches = []
    all_rainfall_sequences = []
    all_labels = []
    all_scenario_metadata = []
    
    for scenario_id in scenario_ids:
        scenario_idx = scenario_id - 1  # Convert to 0-indexed
        
        # Get rainfall sequence for this scenario
        rainfall_seq = rainfall_sequences[scenario_idx]  # Shape: (13,)
        
        # Get categorized labels for this scenario
        cat_data = categorized_maps[scenario_idx]
        labels = cat_data['patch_labels']  # Shape: (num_valid_patches,)
        coords = cat_data['patch_coords']
        
        # Extract spatial patches
        for patch_idx, (i, j) in enumerate(coords):
            # Extract patch from each spatial input
            row_start = i * patch_size
            row_end = (i + 1) * patch_size
            col_start = j * patch_size
            col_end = (j + 1) * patch_size
            
            patch_dem = dem[row_start:row_end, col_start:col_end]
            patch_infilt = infiltration[row_start:row_end, col_start:col_end]
            patch_landuse = landuse[row_start:row_end, col_start:col_end]
            
            spatial_patch = np.stack([patch_dem, patch_infilt, patch_landuse], axis=0)
            
            all_spatial_patches.append(spatial_patch)
            all_rainfall_sequences.append(rainfall_seq)
            all_labels.append(labels[patch_idx])
            all_scenario_metadata.append({
                'scenario_id': scenario_id,
                'patch_coord': (i, j),
                'patch_idx': patch_idx
            })
    
    # Convert to arrays
    spatial_patches = np.array(all_spatial_patches)  # Shape: (N, 3, 16, 16)
    rainfall_seqs = np.array(all_rainfall_sequences)  # Shape: (N, 13)
    labels = np.array(all_labels, dtype=np.int64)     # Shape: (N,)
    
    total_samples = len(labels)
    
    print(f"    Total samples: {total_samples:,}")
    print(f"    Spatial patches: {spatial_patches.shape}")
    print(f"    Rainfall sequences: {rainfall_seqs.shape}")
    print(f"    Labels: {labels.shape}")
    
    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"    Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"      Class {cls}: {count:,} ({count/total_samples*100:.1f}%)")
    
    return {
        'spatial_patches': spatial_patches,
        'rainfall_sequences': rainfall_seqs,
        'labels': labels,
        'scenario_metadata': all_scenario_metadata,
        'total_samples': total_samples,
        'num_scenarios': len(scenario_ids),
        'scenario_ids': scenario_ids
    }


def _verify_split(train_data: Dict, test_data: Dict, train_scenario_ids: List[int], test_scenario_ids: List[int]) -> Dict:
    
    checks = {}
    
    # Check 1: No scenario overlap
    train_set = set(train_scenario_ids)
    test_set = set(test_scenario_ids)
    no_overlap = len(train_set & test_set) == 0
    print(f"  Scenario overlap: {len(train_set & test_set)} {'✓' if no_overlap else '✗'}")
    checks['no_scenario_overlap'] = no_overlap
    
    # Check 2: All scenarios accounted for
    all_scenarios = sorted(train_scenario_ids + test_scenario_ids)
    expected_scenarios = list(range(1, 51))
    all_accounted = all_scenarios == expected_scenarios
    print(f"  All scenarios accounted: {'✓' if all_accounted else '✗'}")
    checks['all_scenarios_accounted'] = all_accounted
    
    # Check 3: Data shapes valid
    train_shapes_ok = (
        train_data['spatial_patches'].ndim == 4 and
        train_data['rainfall_sequences'].ndim == 2 and
        train_data['labels'].ndim == 1
    )
    test_shapes_ok = (
        test_data['spatial_patches'].ndim == 4 and
        test_data['rainfall_sequences'].ndim == 2 and
        test_data['labels'].ndim == 1
    )
    shapes_ok = train_shapes_ok and test_shapes_ok
    print(f"  Data shapes valid: {'✓' if shapes_ok else '✗'}")
    checks['shapes_valid'] = shapes_ok
    
    # Check 4: Sample counts match
    train_match = (
        len(train_data['spatial_patches']) == 
        len(train_data['rainfall_sequences']) == 
        len(train_data['labels'])
    )
    test_match = (
        len(test_data['spatial_patches']) == 
        len(test_data['rainfall_sequences']) == 
        len(test_data['labels'])
    )
    counts_match = train_match and test_match
    print(f"  Sample counts match: {'✓' if counts_match else '✗'}")
    checks['sample_counts_match'] = counts_match
    
    # Check 5: No NaN values
    train_no_nan = (
        not np.isnan(train_data['spatial_patches']).any() and
        not np.isnan(train_data['rainfall_sequences']).any()
    )
    test_no_nan = (
        not np.isnan(test_data['spatial_patches']).any() and
        not np.isnan(test_data['rainfall_sequences']).any()
    )
    no_nans = train_no_nan and test_no_nan
    print(f"  No NaN values: {'✓' if no_nans else '✗'}")
    checks['no_nans'] = no_nans
    
    checks['all_passed'] = all(checks.values())
    
    return checks