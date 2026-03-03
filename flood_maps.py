import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt
import json
from scipy.ndimage import zoom
from matplotlib.colors import ListedColormap

def categorize_flood_maps(
    train_flood_maps: List[np.ndarray], test_flood_maps: List[np.ndarray], train_mask: np.ndarray, test_mask: np.ndarray,
    patch_size: int = 16, categorization_method: str = 'majority_vote', class_thresholds: Optional[List[float]] = None ) -> Dict:
    
    print("\n" + "="*70)
    print("PHASE 4: FLOOD MAP CATEGORIZATION")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 4.1 FLOOD SUSCEPTIBILITY CLASSES
    # -------------------------------------------------------------------------
    print(f"\n[4.1] Flood Susceptibility Classes")
    print("-" * 70)
    
    class_definitions = {
        0: {'name': 'No Flood', 'range': f'[0.00, {class_thresholds[0]:.2f})', 'color': '#2E7D32'},
        1: {'name': 'Light', 'range': f'[{class_thresholds[0]:.2f}, {class_thresholds[1]:.2f})', 'color': '#FDD835'},
        2: {'name': 'Moderate', 'range': f'[{class_thresholds[1]:.2f}, {class_thresholds[2]:.2f})', 'color': '#FB8C00'},
        3: {'name': 'Heavy', 'range': f'[{class_thresholds[2]:.2f}, {class_thresholds[3]:.2f})', 'color': '#E53935'},
        4: {'name': 'Extreme', 'range': f'[{class_thresholds[3]:.2f}, ∞)', 'color': '#6A1B9A'}
    }
    
    for cls, info in class_definitions.items():
        print(f"  Class {cls} - {info['name']:12s}: {info['range']} meters")
    
    # -------------------------------------------------------------------------
    # 4.2 PATCH-BASED CATEGORIZATION SETUP
    # -------------------------------------------------------------------------
    print(f"\n[4.2] Patch Configuration")
    print("-" * 70)
    
    spatial_shape = train_flood_maps[0].shape
    assert spatial_shape[0] % patch_size == 0 and spatial_shape[1] % patch_size == 0, \
        f"Shape {spatial_shape} not divisible by patch_size {patch_size}"
    
    num_patches_h = spatial_shape[0] // patch_size
    num_patches_w = spatial_shape[1] // patch_size
    total_patches = num_patches_h * num_patches_w
    
    print(f"  Spatial shape: {spatial_shape}")
    print(f"  Patch size: {patch_size}×{patch_size}")
    print(f"  Patches per dimension: {num_patches_h}×{num_patches_w}")
    print(f"  Total patches: {total_patches:,}")
    print(f"  Categorization method: {categorization_method}")
    
    # -------------------------------------------------------------------------
    # 4.3 CATEGORIZE ALL FLOOD MAPS
    # -------------------------------------------------------------------------
    print(f"\n[4.3] Categorizing Flood Maps")
    print("-" * 70)
    
    # Train flood maps
    print(f"\n  Train Set ({len(train_flood_maps)} scenarios):")
    train_categorized = []
    train_patch_coords = []
    
    for i, flood_map in enumerate(train_flood_maps, 1):
        cat_result = _categorize_single_map(
            flood_map=flood_map,
            mask=train_mask,
            patch_size=patch_size,
            class_thresholds=class_thresholds,
            method=categorization_method,
            scenario_id=i
        )
        train_categorized.append(cat_result)
        if i == 1:
            train_patch_coords = cat_result['patch_coords']
        
        print(f"    RS{i:02d}: {cat_result['num_valid_patches']:,} valid patches, "
              f"{cat_result['num_nodata_patches']:,} NoData")
    
    # Test flood maps
    print(f"\n  Test Set ({len(test_flood_maps)} scenarios):")
    test_categorized = []
    test_patch_coords = []
    
    for i, flood_map in enumerate(test_flood_maps, 1):
        cat_result = _categorize_single_map(
            flood_map=flood_map,
            mask=test_mask,
            patch_size=patch_size,
            class_thresholds=class_thresholds,
            method=categorization_method,
            scenario_id=i
        )
        test_categorized.append(cat_result)
        if i == 1:
            test_patch_coords = cat_result['patch_coords']
        
        print(f"    RS{i:02d}: {cat_result['num_valid_patches']:,} valid patches, "
              f"{cat_result['num_nodata_patches']:,} NoData")
    
    print(f"\n✓ Categorization complete")
    
    # -------------------------------------------------------------------------
    # 4.4 CLASS DISTRIBUTION ANALYSIS
    # -------------------------------------------------------------------------
    print(f"\n[4.4] Class Distribution Analysis")
    print("-" * 70)
    
    train_distribution = _analyze_class_distribution(train_categorized, 'Train')
    test_distribution = _analyze_class_distribution(test_categorized, 'Test')
    
    # Check for class imbalance
    train_class_counts = train_distribution['aggregated_counts']
    test_class_counts = test_distribution['aggregated_counts']
    
    print(f"\n  Class Imbalance Analysis:")
    for split_name, counts in [('Train', train_class_counts), ('Test', test_class_counts)]:
        total = sum(counts.values())
        if total > 0:
            max_pct = max(counts.values()) / total * 100
            min_pct = min(counts.values()) / total * 100 if len(counts) > 0 else 0
            imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
            print(f"    {split_name}: Max={max_pct:.1f}%, Min={min_pct:.1f}%, "
                  f"Ratio={imbalance_ratio:.1f}:1 {'⚠' if imbalance_ratio > 10 else '✓'}")
    
    results = {
        'train': {
            'categorized_maps': train_categorized,
            'patch_coords': train_patch_coords,
            'num_scenarios': len(train_flood_maps),
            'num_valid_patches': train_categorized[0]['num_valid_patches'],
            'distribution': train_distribution
        },
        'test': {
            'categorized_maps': test_categorized,
            'patch_coords': test_patch_coords,
            'num_scenarios': len(test_flood_maps),
            'num_valid_patches': test_categorized[0]['num_valid_patches'],
            'distribution': test_distribution
        },
        'config': {
            'patch_size': patch_size,
            'spatial_shape': spatial_shape,
            'num_patches_h': num_patches_h,
            'num_patches_w': num_patches_w,
            'total_patches': total_patches,
            'class_thresholds': class_thresholds,
            'class_definitions': class_definitions,
            'categorization_method': categorization_method
        }
    }
    
    print(f"\n{'='*70}")
    print(f"FLOOD CATEGORIZATION COMPLETE")
    print(f"  Train: {len(train_flood_maps)} scenarios, "
          f"{train_categorized[0]['num_valid_patches']:,} patches each")
    print(f"  Test: {len(test_flood_maps)} scenarios, "
          f"{test_categorized[0]['num_valid_patches']:,} patches each")
    print(f"{'='*70}\n")
    
    return results

def _categorize_single_map(flood_map: np.ndarray, mask: np.ndarray, patch_size: int,
    class_thresholds: List[float], method: str, scenario_id: int) -> Dict:
    
    H, W = flood_map.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Initialize outputs
    patch_labels = []
    patch_coords = []
    patch_stats = []
    
    valid_count = 0
    nodata_count = 0
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Extract patch
            row_start = i * patch_size
            row_end = (i + 1) * patch_size
            col_start = j * patch_size
            col_end = (j + 1) * patch_size
            
            patch_flood = flood_map[row_start:row_end, col_start:col_end]
            patch_mask = mask[row_start:row_end, col_start:col_end]
            
            # Check if patch is valid (has sufficient valid data)
            valid_pixels = patch_mask.sum()
            total_pixels = patch_size * patch_size
            valid_ratio = valid_pixels / total_pixels
            
            # STRICTER THRESHOLD: Require at least 90% valid pixels
            if valid_ratio < 0.9:  # Changed from 0.5 to 0.9
                nodata_count += 1
                continue  # Skip this patch entirely
            
            # Categorize based on method
            if method == 'majority_vote':
                label = _majority_vote(patch_flood, patch_mask, class_thresholds)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Skip if label is invalid (-1)
            if label == -1:
                nodata_count += 1
                continue
            
            patch_labels.append(label)
            patch_coords.append((i, j))
            
            # Store statistics (only from valid pixels)
            valid_depths = patch_flood[patch_mask > 0]
            if len(valid_depths) > 0:
                patch_stats.append({
                    'mean': float(valid_depths.mean()),
                    'max': float(valid_depths.max()),
                    'min': float(valid_depths.min())
                })
            else:
                patch_stats.append({'mean': 0.0, 'max': 0.0, 'min': 0.0})
            
            valid_count += 1
    
    return {
        'scenario_id': scenario_id,
        'patch_labels': np.array(patch_labels, dtype=np.int8),
        'patch_coords': patch_coords,
        'patch_stats': patch_stats,
        'num_valid_patches': valid_count,
        'num_nodata_patches': nodata_count
    }



def _majority_vote(patch_flood: np.ndarray, patch_mask: np.ndarray, thresholds: List[float]) -> int:
    valid_depths = patch_flood[patch_mask > 0]
    
    if len(valid_depths) == 0:
        return -1  # NoData
    
    # Classify each pixel
    pixel_classes = np.digitize(valid_depths, bins=thresholds)
    
    # Majority vote
    counts = Counter(pixel_classes)
    majority_class = counts.most_common(1)[0][0]
    
    return int(majority_class)

def _analyze_class_distribution(categorized_maps: List[Dict], split_name: str) -> Dict:    
    print(f"\n  {split_name} Set Class Distribution:")
    print(f"  {'Scenario':<10} {'Class 0':<10} {'Class 1':<10} {'Class 2':<10} {'Class 3':<10} {'Class 4':<10}")
    print(f"  {'-'*60}")
    
    scenario_distributions = []
    aggregated_counts = Counter()
    
    for result in categorized_maps:
        labels = result['patch_labels']
        counts = Counter(labels)
        
        # Print per scenario
        scenario_id = result['scenario_id']
        total = len(labels)
        
        row = f"  RS{scenario_id:02d}      "
        for cls in range(5):
            count = counts.get(cls, 0)
            pct = count / total * 100 if total > 0 else 0
            row += f"{count:4d} ({pct:4.1f}%)  "
        print(row)
        
        scenario_distributions.append(counts)
        aggregated_counts.update(counts)
    
    # Aggregated statistics
    total_patches = sum(aggregated_counts.values())
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<10}", end='')
    for cls in range(5):
        count = aggregated_counts.get(cls, 0)
        pct = count / total_patches * 100 if total_patches > 0 else 0
        print(f"{count:4d} ({pct:4.1f}%)  ", end='')
    print()
    
    return {
        'scenario_distributions': scenario_distributions,
        'aggregated_counts': dict(aggregated_counts),
        'total_patches': total_patches
    }


def save_categorization_params(results: Dict, output_path: str):
    params = {
        'config': results['config'],
        'train': {
            'num_scenarios': results['train']['num_scenarios'],
            'num_valid_patches': results['train']['num_valid_patches'],
            'class_distribution': results['train']['distribution']['aggregated_counts']
        },
        'test': {
            'num_scenarios': results['test']['num_scenarios'],
            'num_valid_patches': results['test']['num_valid_patches'],
            'class_distribution': results['test']['distribution']['aggregated_counts']
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\n✓ Categorization parameters saved to: {output_path}")

def visualize_categorized_maps(results: Dict, original_flood_maps: List[np.ndarray], split: str = 'train', num_scenarios: int = 4, figsize=(20, 10)):
    
    categorized_data = results[split]['categorized_maps']
    patch_size = results['config']['patch_size']
    spatial_shape = results['config']['spatial_shape']
    class_defs = results['config']['class_definitions']
    
    num_scenarios = min(num_scenarios, len(categorized_data))
    
    fig, axes = plt.subplots(num_scenarios, 2, figsize=figsize)
    if num_scenarios == 1:
        axes = axes.reshape(1, -1)
    
    # Create colormap with NoData as gray
    colors = ['#CCCCCC'] + [class_defs[i]['color'] for i in range(5)]  # Gray for NoData (-1)
    cmap_cat = ListedColormap(colors)
    
    for idx in range(num_scenarios):
        # Reconstruct categorized map
        cat_map = _reconstruct_map_from_patches(
            categorized_data[idx],
            patch_size,
            spatial_shape
        )
        
        # Original continuous map
        orig_map = original_flood_maps[idx].copy()
        
        # Mask NoData in original for better comparison
        if split == 'train':
            orig_map[results['config'].get('test_mask', np.zeros_like(orig_map)) > 0] = np.nan
        else:
            orig_map[results['config'].get('train_mask', np.zeros_like(orig_map)) > 0] = np.nan
        
        # Plot original
        ax1 = axes[idx, 0]
        im1 = ax1.imshow(orig_map, cmap='YlGnBu', vmin=0, vmax=1.5)
        ax1.set_title(f'RS{idx+1} - Original (Continuous)', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, label='Depth (m)')
        
        # Plot categorized (use masked array to show NoData as white)
        ax2 = axes[idx, 1]
        cat_map_masked = np.ma.masked_where(cat_map == -1, cat_map)
        im2 = ax2.imshow(cat_map_masked, cmap=cmap_cat, vmin=-1, vmax=4)
        ax2.set_title(f'RS{idx+1} - Categorized (5 Classes)', fontweight='bold')
        ax2.axis('off')
        
        # Custom colorbar
        cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, ticks=[-1, 0, 1, 2, 3, 4])
        cbar.set_label('Flood Class')
        cbar.ax.set_yticklabels(['NoData', 'No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme'])
    
    plt.suptitle(f'{split.capitalize()} Set: Original vs Categorized Flood Maps', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


def _reconstruct_map_from_patches(cat_result: Dict, patch_size: int, spatial_shape: Tuple) -> np.ndarray:
    reconstructed = np.full(spatial_shape, -1, dtype=np.int8)
    
    for label, (i, j) in zip(cat_result['patch_labels'], cat_result['patch_coords']):
        row_start = i * patch_size
        row_end = (i + 1) * patch_size
        col_start = j * patch_size
        col_end = (j + 1) * patch_size
        
        reconstructed[row_start:row_end, col_start:col_end] = label
    
    return reconstructed


def resize_flood_maps(flood_maps: List[np.ndarray], target_shape: Tuple[int, int] = (1152, 1152), split_name: str = 'Train') -> List[np.ndarray]:    
    print(f"\nResizing {split_name} Flood Maps")
    print("-" * 70)
    
    current_shape = flood_maps[0].shape
    print(f"  Current shape: {current_shape}")
    print(f"  Target shape: {target_shape}")
    
    if current_shape == target_shape:
        print(f"  ✓ Already at target shape, skipping resize")
        return flood_maps
    
    resized_maps = []
    zoom_factors = (target_shape[0] / current_shape[0], target_shape[1] / current_shape[1])
    
    print(f"  Zoom factors: {zoom_factors}")
    print(f"  Resizing {len(flood_maps)} flood maps...")
    
    for i, flood_map in enumerate(flood_maps, 1):
        # Use order=1 (bilinear) for continuous flood depth values
        resized = zoom(flood_map, zoom_factors, order=1)
        resized_maps.append(resized)
        
        if i % 5 == 0 or i == len(flood_maps):
            print(f"    Progress: {i}/{len(flood_maps)} maps resized")
    
    print(f"  ✓ All flood maps resized to {target_shape}")
    
    return resized_maps