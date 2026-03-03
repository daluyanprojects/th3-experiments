import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any


def _infer_map_shape_from_coords(
    scenario_metadata: list,
    patch_size: int,
) -> Tuple[int, int]:

    max_i = max(meta['patch_coord'][0] for meta in scenario_metadata)
    max_j = max(meta['patch_coord'][1] for meta in scenario_metadata)
    H = (max_i + 1) * patch_size
    W = (max_j + 1) * patch_size
    return H, W


def _reconstruct_one_split(
    spatial_patches:    np.ndarray,            # (N, C, ph, pw)
    rainfall:           np.ndarray,            # (N, T)
    labels:             np.ndarray,            # (N,)
    conditioning:       Optional[np.ndarray],  # (N, D) or None
    scenario_metadata:  List[Dict],            # one dict per patch
    scenario_ids:       List[int],             # ordered unique ids for this split
    patch_size:         int,
    map_h:              int,
    map_w:              int,
    nodata_label:       int,
    split_name:         str,
    verbose:            bool,
) -> Dict[str, Any]:

    N, C, ph, pw = spatial_patches.shape
    assert ph == patch_size and pw == patch_size, (
        f"Patch dims ({ph},{pw}) != patch_size={patch_size}"
    )
    assert N == len(scenario_metadata), (
        f"spatial_patches length {N} != scenario_metadata length {len(scenario_metadata)}"
    )

    n_scenarios = len(scenario_ids)
    T = rainfall.shape[1]

    # Map scenario_id → canvas index
    scenario_id_to_idx = {sid: idx for idx, sid in enumerate(scenario_ids)}

    # Allocate output canvases
    spatial_maps = np.zeros((n_scenarios, C, map_h, map_w), dtype=spatial_patches.dtype)
    label_maps   = np.full( (n_scenarios,    map_h, map_w),
                            fill_value=nodata_label, dtype=np.int64)
    rainfall_seq = np.zeros((n_scenarios, T), dtype=rainfall.dtype)
    D = conditioning.shape[1] if conditioning is not None else 0
    cond_out = np.zeros((n_scenarios, D), dtype=conditioning.dtype) \
               if conditioning is not None else None

    # Track which scenarios have had rainfall/conditioning assigned
    scene_filled = np.zeros(n_scenarios, dtype=bool)

    # ── Place every patch at its (i, j) coordinate ───────────────────────────
    for patch_idx, meta in enumerate(scenario_metadata):
        sid       = meta['scenario_id']
        i, j      = meta['patch_coord']
        scene_idx = scenario_id_to_idx[sid]

        row = i * patch_size
        col = j * patch_size

        if row + patch_size > map_h or col + patch_size > map_w:
            warnings.warn(
                f"[{split_name}] Patch ({i},{j}) for scenario {sid} falls outside "
                f"canvas ({map_h},{map_w}). Skipping.",
                stacklevel=2,
            )
            continue

        spatial_maps[scene_idx, :, row:row + patch_size, col:col + patch_size] = (
            spatial_patches[patch_idx]
        )
        label_maps[scene_idx, row:row + patch_size, col:col + patch_size] = (
            labels[patch_idx]
        )

        # Rainfall and conditioning are the same for every patch in a scenario
        if not scene_filled[scene_idx]:
            rainfall_seq[scene_idx] = rainfall[patch_idx]
            if cond_out is not None:
                cond_out[scene_idx] = conditioning[patch_idx]
            scene_filled[scene_idx] = True

    # Warn if any scenario ended up with no patches
    unfilled = np.where(~scene_filled)[0]
    if unfilled.size > 0:
        warnings.warn(
            f"[{split_name}] {unfilled.size} scenario(s) had zero patches: "
            f"indices {unfilled.tolist()}",
            stacklevel=2,
        )

    if verbose:
        n_valid  = int(np.sum(label_maps != nodata_label))
        n_total  = label_maps.size
        pct_v    = 100.0 * n_valid / n_total
        pct_nd   = 100.0 - pct_v
        print(f"[{split_name}] spatial_maps : {spatial_maps.shape}  dtype={spatial_maps.dtype}")
        print(f"[{split_name}] label_maps   : {label_maps.shape}  dtype={label_maps.dtype}")
        print(f"[{split_name}] rainfall_seq : {rainfall_seq.shape}  dtype={rainfall_seq.dtype}")
        if cond_out is not None:
            print(f"[{split_name}] conditioning : {cond_out.shape}  dtype={cond_out.dtype}")
        print(f"[{split_name}] Canvas fill  : {n_valid:,} valid px "
              f"({pct_v:.1f}%)  |  {n_total - n_valid:,} NoData ({pct_nd:.1f}%)")

    return {
        'spatial':      spatial_maps,   # (S, C, H, W)
        'labels':       label_maps,     # (S, H, W)   -1 where no patch
        'rainfall':     rainfall_seq,   # (S, T)
        'conditioning': cond_out,       # (S, D) or None
    }



def reconstruct_for_cnn(
    split_results:  Dict[str, Any],
    patch_size:     int  = 4,
    map_h:          Optional[int] = None,
    map_w:          Optional[int] = None,
    nodata_label:   int  = -1,
    verbose:        bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    output = {}

    for split_name in ('train', 'test'):
        split = split_results[split_name]

        spatial_patches = split['spatial_patches']
        rainfall        = split['rainfall_sequences']
        labels          = split['labels']
        conditioning    = split.get('conditioning_vectors', None)
        scenario_meta   = split['scenario_metadata']
        scenario_ids    = split['scenario_ids']

        # Resolve canvas dimensions
        H = map_h
        W = map_w
        if H is None or W is None:
            H_inf, W_inf = _infer_map_shape_from_coords(scenario_meta, patch_size)
            H = H if H is not None else H_inf
            W = W if W is not None else W_inf

        if verbose:
            patches_per_scene = len(scenario_meta) // len(scenario_ids)
            print(f"\n[{split_name}] {len(scenario_ids)} scenarios | "
                  f"canvas ({H}, {W}) | "
                  f"{patches_per_scene:,} patches/scenario")

        output[split_name] = _reconstruct_one_split(
            spatial_patches   = spatial_patches,
            rainfall          = rainfall,
            labels            = labels,
            conditioning      = conditioning,
            scenario_metadata = scenario_meta,
            scenario_ids      = scenario_ids,
            patch_size        = patch_size,
            map_h             = H,
            map_w             = W,
            nodata_label      = nodata_label,
            split_name        = split_name,
            verbose           = verbose,
        )

    return output['train'], output['test']

def validate_reconstruction(
    split_results:  Dict[str, Any],
    cnn_split:      Dict[str, Any],
    split_name:     str = 'train',
    scenario_index: int = 0,
    patch_size:     int = 4,
) -> bool:

    split        = split_results[split_name]
    patches_orig = split['spatial_patches']
    meta         = split['scenario_metadata']
    scenario_ids = split['scenario_ids']
    sid          = scenario_ids[scenario_index]

    patch_indices = [i for i, m in enumerate(meta) if m['scenario_id'] == sid]
    canvas        = cnn_split['spatial'][scenario_index]   # (C, H, W)
    n_ok          = 0

    for idx in patch_indices:
        i, j = meta[idx]['patch_coord']
        row  = i * patch_size
        col  = j * patch_size
        expected = patches_orig[idx]
        actual   = canvas[:, row:row + patch_size, col:col + patch_size]
        if not np.array_equal(expected, actual):
            print(f"  MISMATCH at patch ({i},{j}): "
                  f"expected {expected.ravel()[:4]} ... got {actual.ravel()[:4]} ...")
            return False
        n_ok += 1

    print(f"[{split_name}] Validation scenario {sid}: PASS ({n_ok:,} patches checked)")
    return True