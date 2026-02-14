import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
    
FLOOD_CLASSES = {
    0: {
        'name': 'No Flood',
        'range': (0.0, 0.15),
        'color': '#FFFFFF'  # White
    },
    1: {
        'name': 'Light',
        'range': (0.15, 0.24),
        'color': '#FFEB3B'  # Yellow
    },
    2: {
        'name': 'Moderate',
        'range': (0.24, 0.46),
        'color': '#FF9800'  # Orange
    },
    3: {
        'name': 'Heavy',
        'range': (0.46, 0.68),
        'color': '#F44336'  # Red
    },
    4: {
        'name': 'Extreme',
        'range': (0.68, float('inf')),
        'color': '#9C27B0'  # Purple
    }
}

def categorize_flood_map(flood_map: np.ndarray, class_ranges: Optional[Dict] = None) -> np.ndarray:
    if class_ranges is None:
        class_ranges = {k: v['range'] for k, v in FLOOD_CLASSES.items()}
    
    # Initialize output array with same shape
    categorized = np.zeros_like(flood_map, dtype=np.int8)
    
    # Handle NaN/NoData values - mark as -1
    nodata_mask = np.isnan(flood_map) 
    categorized[nodata_mask] = -1
    
    # Categorize each pixel based on flood depth ranges
    for class_id, (min_val, max_val) in class_ranges.items():
        # Create mask for this class
        if max_val == float('inf'):
            class_mask = (flood_map >= min_val) & ~nodata_mask
        else:
            class_mask = (flood_map >= min_val) & (flood_map < max_val) & ~nodata_mask
        
        categorized[class_mask] = class_id
    
    return categorized

def categorize_all_flood_maps(flood_maps: List[np.ndarray], class_ranges: Optional[Dict] = None, verbose: bool = True) -> List[np.ndarray]:
    if class_ranges is None:
        class_ranges = {k: v['range'] for k, v in FLOOD_CLASSES.items()}
    
    categorized_maps = []
    
    if verbose:
        print("\n" + "="*70)
        print("CATEGORIZING FLOOD MAPS")
        print("="*70)
        print(f"\nClass definitions:")
        for class_id, info in FLOOD_CLASSES.items():
            min_val, max_val = info['range']
            max_str = f"{max_val:.2f}" if max_val != float('inf') else "∞"
            print(f"  Class {class_id}: {info['name']:12s} [{min_val:.2f}, {max_str}) m")
        print()
    
    for idx, flood_map in enumerate(flood_maps):
        categorized = categorize_flood_map(flood_map, class_ranges)
        categorized_maps.append(categorized)
        
        if verbose:
            # Calculate class distribution
            total_valid = (categorized >= 0).sum()
            
            print(f"Scenario {idx+1:2d}:")
            for class_id, info in FLOOD_CLASSES.items():
                count = (categorized == class_id).sum()
                percentage = (count / total_valid * 100) if total_valid > 0 else 0
                print(f"  {info['name']:12s}: {count:8,} pixels ({percentage:5.2f}%)")
    
    if verbose:
        print(f"\n✓ Categorized {len(categorized_maps)} flood maps")
    
    return categorized_maps

def visualize_all_categorized_maps(categorized_maps: List[np.ndarray], scenario_ids: Optional[List[int]] = None,
                                    figsize: Tuple[int, int] = (25, 20), ncols: int = 5) -> plt.Figure:
    n_maps = len(categorized_maps)
    nrows = (n_maps + ncols - 1) // ncols
    
    if scenario_ids is None:
        scenario_ids = list(range(1, n_maps + 1))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Create custom colormap
    colors = [FLOOD_CLASSES[i]['color'] for i in range(5)]
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    for idx, (cat_map, scenario_id) in enumerate(zip(categorized_maps, scenario_ids)):
        ax = axes[idx]
        
        # Mask NoData
        plot_data = np.ma.masked_where(cat_map < 0, cat_map)
        
        # Plot
        im = ax.imshow(plot_data, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f'RS{scenario_id}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_maps, len(axes)):
        axes[idx].axis('off')
    
    # Add global legend
    patches = []
    for class_id, info in FLOOD_CLASSES.items():
        patch = mpatches.Patch(color=info['color'], 
                              label=f"{class_id}: {info['name']}")
        patches.append(patch)
    
    fig.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, 0.02),
              ncol=5, fontsize=12, frameon=True, fancybox=True)
    
    fig.suptitle('Categorized Flood Maps - All Scenarios', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    return fig

def extract_patches_from_map(map_data: np.ndarray, patch_size: int, stride: Optional[int] = None) -> np.ndarray:
    if stride is None:
        stride = patch_size
    
    h, w = map_data.shape
    patches = []
    
    # Calculate number of patches in each dimension
    num_patches_h = (h - patch_size) // stride + 1
    num_patches_w = (w - patch_size) // stride + 1
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            row_start = i * stride
            col_start = j * stride
            row_end = row_start + patch_size
            col_end = col_start + patch_size
            
            patch = map_data[row_start:row_end, col_start:col_end]
            patches.append(patch)
    
    return np.array(patches)

def categorize_patch_majority_vote(patch: np.ndarray) -> int:
    """
    Categorize a patch using majority voting
    
    Args:
        patch: 2D patch array with categorical values
    
    Returns:
        Most frequent class in the patch (excluding NoData=-1)
    """
    # Filter out NoData values (-1)
    valid_pixels = patch[patch >= 0]
    
    if len(valid_pixels) == 0:
        return -1  # All NoData
    
    # Find most frequent class
    unique, counts = np.unique(valid_pixels, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    
    return int(majority_class)

def categorize_flood_map_patch_based(flood_map: np.ndarray, patch_size: int, stride: Optional[int] = None,
                                     categorization_method: str = 'majority', class_ranges: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    if class_ranges is None:
        class_ranges = {k: v['range'] for k, v in FLOOD_CLASSES.items()}
    
    if stride is None:
        stride = patch_size
    
    # Categorize full map (pixel-wise)
    categorized_map = categorize_flood_map(flood_map, class_ranges)
    
    # Extract patches from categorized map
    patches = extract_patches_from_map(categorized_map, patch_size, stride)
    
    # Categorize each patch
    patch_labels = []
    
    if categorization_method == 'majority':
        for patch in patches:
            label = categorize_patch_majority_vote(patch)
            patch_labels.append(label)
    else:
        raise ValueError(f"Unknown categorization method: {categorization_method}")
    
    patch_labels = np.array(patch_labels, dtype=np.int8)
    
    # Metadata
    h, w = flood_map.shape
    num_patches_h = (h - patch_size) // stride + 1
    num_patches_w = (w - patch_size) // stride + 1
    
    metadata = {
        'patch_size': patch_size,
        'stride': stride,
        'num_patches': len(patch_labels),
        'num_patches_h': num_patches_h,
        'num_patches_w': num_patches_w,
        'original_shape': (h, w),
        'categorization_method': categorization_method
    }
    
    return patch_labels, metadata

def categorize_all_flood_maps_patch(flood_maps: List[np.ndarray], patch_size: int, stride: Optional[int] = None, categorization_method: str = 'majority',
                                    class_ranges: Optional[Dict] = None, verbose: bool = True) -> Tuple[List[np.ndarray], List[Dict]]:
    if class_ranges is None:
        class_ranges = {k: v['range'] for k, v in FLOOD_CLASSES.items()}
    
    if stride is None:
        stride = patch_size
    
    categorized_patches_list = []
    metadata_list = []
    
    if verbose:
        print("\n" + "="*70)
        print("PATCH-BASED FLOOD MAP CATEGORIZATION")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Patch size: {patch_size}×{patch_size}")
        print(f"  Stride: {stride} ({'non-overlapping' if stride == patch_size else 'overlapping'})")
        print(f"  Categorization method: {categorization_method}")
        print(f"\nClass definitions:")
        
        for class_id, info in FLOOD_CLASSES.items():
            min_val, max_val = info['range']
            max_str = f"{max_val:.2f}" if max_val != float('inf') else "∞"
            print(f"  Class {class_id}: {info['name']:12s} [{min_val:.2f}, {max_str}) m")
        print()
    
    for idx, flood_map in enumerate(flood_maps):
        patch_labels, metadata = categorize_flood_map_patch_based(
            flood_map,
            patch_size=patch_size,
            stride=stride,
            categorization_method=categorization_method,
            class_ranges=class_ranges
        )
        
        categorized_patches_list.append(patch_labels)
        metadata_list.append(metadata)
        
        if verbose:
            # Calculate class distribution for patches
            total_valid = (patch_labels >= 0).sum()
            
            print(f"Scenario {idx+1:2d} ({metadata['num_patches']} patches):")
            for class_id, info in FLOOD_CLASSES.items():
                count = (patch_labels == class_id).sum()
                percentage = (count / total_valid * 100) if total_valid > 0 else 0
                print(f"  {info['name']:12s}: {count:6,} patches ({percentage:5.2f}%)")
    
    if verbose:
        total_patches = sum(len(patches) for patches in categorized_patches_list)
        print(f"\n✓ Categorized {len(categorized_patches_list)} scenarios")
        print(f"✓ Total patches across all scenarios: {total_patches:,}")
        print(f"✓ Patches per scenario: {metadata_list[0]['num_patches']:,}")
    
    return categorized_patches_list, metadata_list

def reconstruct_map_from_patches(patch_labels: np.ndarray, metadata: Dict, method: str = 'nearest') -> np.ndarray:
    h, w = metadata['original_shape']
    patch_size = metadata['patch_size']
    stride = metadata['stride']
    num_patches_h = metadata['num_patches_h']
    num_patches_w = metadata['num_patches_w']
    
    # Initialize output map
    reconstructed = np.zeros((h, w), dtype=np.int8)
    
    if method == 'nearest':
        # Simple nearest neighbor: assign patch label to center of each patch
        patch_idx = 0
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                row_start = i * stride
                col_start = j * stride
                row_end = min(row_start + patch_size, h)
                col_end = min(col_start + patch_size, w)
                
                # Assign patch label to entire patch region
                reconstructed[row_start:row_end, col_start:col_end] = patch_labels[patch_idx]
                patch_idx += 1

    return reconstructed

def visualize_all_reconstructed_maps(reconstructed_maps: List[np.ndarray], scenario_ids: Optional[List[int]] = None, figsize: Tuple[int, int] = (25, 20), ncols: int = 5) -> plt.Figure:

    n_maps = len(reconstructed_maps)
    nrows = (n_maps + ncols - 1) // ncols
    
    if scenario_ids is None:
        scenario_ids = list(range(1, n_maps + 1))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Create custom colormap
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    for idx, (cat_map, scenario_id) in enumerate(zip(reconstructed_maps, scenario_ids)):
        ax = axes[idx]
        
        # Mask NoData
        plot_data = np.ma.masked_where(cat_map < 0, cat_map)
        
        # Plot
        im = ax.imshow(plot_data, cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f'RS{scenario_id}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_maps, len(axes)):
        axes[idx].axis('off')
    
    # Add global legend
    patches = []
    for class_id, info in FLOOD_CLASSES.items():
        patch = mpatches.Patch(color=info['color'], 
                              label=f"{class_id}: {info['name']}")
        patches.append(patch)
    
    fig.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, 0.02),
              ncol=5, fontsize=12, frameon=True, fancybox=True)
    
    fig.suptitle(f'Patch-Based Categorized Flood Maps ({PATCH_SIZE}×{PATCH_SIZE} patches)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    return fig