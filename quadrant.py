import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple, Dict, List, Optional


def stack_spatial_inputs(dem: np.ndarray, infiltration: np.ndarray, landuse: np.ndarray) -> np.ndarray:
    stacked = np.stack([dem, infiltration, landuse], axis=-1)
    print(f"[Stack] Spatial input shape: {stacked.shape}  |  Channels: DEM, Infiltration, Landuse")
    return stacked

def split_into_quadrants(spatial_stack: np.ndarray) -> Dict[str, np.ndarray]:
    H, W, C = spatial_stack.shape
    h, w = H // 2, W // 2

    quadrants = {
        'Q1': spatial_stack[:h,    :w,    :],   # top-left     → TRAIN
        'Q2': spatial_stack[:h,    w:w*2, :],   # top-right    → TRAIN
        'Q3': spatial_stack[h:h*2, :w,    :],   # bottom-left  → TEST
        'Q4': spatial_stack[h:h*2, w:w*2, :],   # bottom-right → TRAIN
    }

    for name, q in quadrants.items():
        role = "TEST" if name == 'Q3' else "TRAIN"
        print(f"  [{name} | {role}] shape: {q.shape}")

    return quadrants

def get_quadrant_patch_indices(H: int, W: int, patch_size: int, stride: Optional[int] = None,) -> Dict[str, np.ndarray]:
    if stride is None:
        stride = patch_size

    h_mid = H // 2
    w_mid = W // 2

    num_patches_h = (H - patch_size) // stride + 1
    num_patches_w = (W - patch_size) // stride + 1

    q_indices = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch_idx = i * num_patches_w + j
            row_center = i * stride + patch_size // 2
            col_center = j * stride + patch_size // 2

            if row_center < h_mid and col_center < w_mid:
                q_indices['Q1'].append(patch_idx)
            elif row_center < h_mid and col_center >= w_mid:
                q_indices['Q2'].append(patch_idx)
            elif row_center >= h_mid and col_center < w_mid:
                q_indices['Q3'].append(patch_idx)
            else:
                q_indices['Q4'].append(patch_idx)

    for name, idxs in q_indices.items():
        role = "TEST" if name == 'Q3' else "TRAIN"
        print(f"  [{name} | {role}] {len(idxs):,} patches")

    return {k: np.array(v) for k, v in q_indices.items()}


def extract_spatial_patches(spatial_stack: np.ndarray,patch_size: int, stride: Optional[int] = None,) -> np.ndarray:
    if stride is None:
        stride = patch_size

    H, W, C = spatial_stack.shape
    num_h = (H - patch_size) // stride + 1
    num_w = (W - patch_size) // stride + 1
    N = num_h * num_w

    patches = np.zeros((N, patch_size, patch_size, C), dtype=spatial_stack.dtype)
    idx = 0
    for i in range(num_h):
        for j in range(num_w):
            r, c = i * stride, j * stride
            patches[idx] = spatial_stack[r:r+patch_size, c:c+patch_size, :]
            idx += 1

    print(f"  Spatial patches extracted: {N:,}  shape: {patches.shape}")
    return patches

def build_train_test_split(spatial_patches: np.ndarray, flood_maps_categorized_patch: List[np.ndarray], quadrant_indices: Dict[str, np.ndarray], rainfall_scenarios: List = None) -> Dict:
    TRAIN_Q = ['Q1', 'Q2', 'Q4']
    TEST_Q  = ['Q3']

    train_idx = np.concatenate([quadrant_indices[q] for q in TRAIN_Q])
    test_idx  = np.concatenate([quadrant_indices[q] for q in TEST_Q])

    X_train = spatial_patches[train_idx]  
    X_test  = spatial_patches[test_idx]    

    S = len(flood_maps_categorized_patch)
    all_labels = np.stack(flood_maps_categorized_patch, axis=0) 
    y_train = all_labels[:, train_idx]   
    y_test  = all_labels[:, test_idx]   

    split = {
        'X_train':   X_train,
        'X_test':    X_test,
        'y_train':   y_train,
        'y_test':    y_test,
        'rainfall':  rainfall_scenarios,
        'indices': {
            'train': train_idx,
            'test':  test_idx,
        },
    }

    print("\n" + "="*60)
    print("QUADRANT SPLIT SUMMARY")
    print("="*60)
    print(f"  Train quadrants  : {TRAIN_Q}")
    print(f"  Test  quadrants  : {TEST_Q}")
    print(f"  X_train          : {X_train.shape}  (patches, h, w, C)")
    print(f"  X_test           : {X_test.shape}   (patches, h, w, C)")
    print(f"  y_train          : {y_train.shape}  (scenarios, patches)  — int8 class")
    print(f"  y_test           : {y_test.shape}   (scenarios, patches)  — int8 class")
    if rainfall_scenarios is not None:
        print(f"  Rainfall         : {len(rainfall_scenarios)} scenarios (NOT split spatially)")
    print("="*60)

    return split

def visualize_quadrant_split(dem: np.ndarray, figsize: Tuple[int, int] = (8, 9)) -> plt.Figure:
    H, W = dem.shape
    h, w = H // 2, W // 2

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dem, cmap='terrain', origin='upper')
    plt.colorbar(im, ax=ax, label='Elevation (m)', fraction=0.03)

    ax.axhline(h, color='white', linewidth=2.5, linestyle='--', alpha=0.9)
    ax.axvline(w, color='white', linewidth=2.5, linestyle='--', alpha=0.9)

    cfg = {
        'Q1': (h * 0.25, w * 0.25, 'TRAIN', '#388E3C'),
        'Q2': (h * 0.25, w * 1.25, 'TRAIN', '#388E3C'),
        'Q3': (h * 1.25, w * 0.25, 'TEST',  '#C62828'),
        'Q4': (h * 1.25, w * 1.25, 'TRAIN', '#388E3C'),
    }
    for qname, (yr, xr, role, color) in cfg.items():
        ax.text(xr, yr, f"{qname}\n({role})",
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.75))

    legend_handles = [
        mpatches.Patch(facecolor='#388E3C', alpha=0.75, label='Train (Q1, Q2, Q4)'),
        mpatches.Patch(facecolor='#C62828', alpha=0.75, label='Test  (Q3)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=11)
    ax.set_title("Quadrant-Based Spatial Split — Metro Manila", fontsize=14, pad=10)
    ax.set_xlabel("Column  (West → East)")
    ax.set_ylabel("Row  (North → South)")
    plt.tight_layout()
    return fig