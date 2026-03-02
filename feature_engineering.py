import numpy as np
from scipy import stats as scipy_stats
from typing import List, Optional, Tuple
from tqdm import tqdm

N_FEATURES = 35


# ── Feature names ──────────────────────────────────────────────────────────────
def get_feature_names() -> List[str]:
    names = [
        # DEM (7)
        'dem_mean', 'dem_std', 'dem_min', 'dem_max',
        'dem_slope_mean', 'dem_slope_std', 'dem_entropy',
        # Infiltration (5)
        'infilt_mean', 'infilt_std', 'infilt_min', 'infilt_max', 'infilt_entropy',
        # Land use (3)
        'landuse_mode', 'landuse_nunique', 'landuse_entropy',
        # Rainfall raw (13)
        *[f'rain_t{t:02d}' for t in range(13)],
        # Rainfall derived (3)
        'rain_total', 'rain_peak', 'rain_tpeak_idx',
        # Conditioning (4)
        'cond_pattern_type', 'cond_depth_norm', 'cond_tpeak', 'cond_has_tpeak',
    ]
    assert len(names) == N_FEATURES, f"Expected {N_FEATURES}, got {len(names)}"
    return names


# ── Slope map ──────────────────────────────────────────────────────────────────
def compute_slope_map(dem_map: np.ndarray) -> np.ndarray:
    dy, dx    = np.gradient(dem_map.astype(np.float64))
    slope_map = np.sqrt(dx**2 + dy**2).astype(np.float32)
    print(f"[compute_slope_map]  slope range: [{slope_map.min():.6f}, {slope_map.max():.6f}]"
          f"  mean={slope_map.mean():.6f}")
    return slope_map


# ── Entropy helper ─────────────────────────────────────────────────────────────
def _patch_entropy(values: np.ndarray, n_bins: int = 16) -> float:
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-7:          # catches both exact-constant and near-constant
        return 0.0
    safe_bins = min(n_bins, len(np.unique(values)))
    if safe_bins < 2:
        return 0.0
    counts, _ = np.histogram(values, bins=safe_bins, range=(vmin, vmax))
    counts     = counts[counts > 0].astype(np.float64)
    p          = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def _batch_entropy(flat_patches: np.ndarray, n_bins: int = 16) -> np.ndarray:
    N = flat_patches.shape[0]
    out = np.empty(N, dtype=np.float32)
    for i in range(N):
        out[i] = _patch_entropy(flat_patches[i], n_bins)
    return out


# ── Spatial features ───────────────────────────────────────────────────────────
def _extract_spatial_features(
    patches        : np.ndarray,   # (B, 3, 4, 4)
    slope_patches  : np.ndarray,   # (B, 16)  pre-extracted slope pixels
    entropy_bins   : int = 16,
) -> np.ndarray:
    B    = patches.shape[0]
    flat = patches.reshape(B, 3, -1)   # (B, 3, 16)
    feats = np.empty((B, 15), dtype=np.float32)

    # ── DEM (channel 0) ────────────────────────────────────────────────────────
    dem = flat[:, 0, :]                          # (B, 16)
    feats[:, 0] = dem.mean(axis=1)               # dem_mean
    feats[:, 1] = dem.std(axis=1)                # dem_std
    feats[:, 2] = dem.min(axis=1)                # dem_min
    feats[:, 3] = dem.max(axis=1)                # dem_max
    feats[:, 4] = slope_patches.mean(axis=1)     # dem_slope_mean
    feats[:, 5] = slope_patches.std(axis=1)      # dem_slope_std
    feats[:, 6] = _batch_entropy(dem, entropy_bins)  # dem_entropy

    # ── Infiltration (channel 1) ───────────────────────────────────────────────
    inf_ = flat[:, 1, :]                         # (B, 16)
    feats[:, 7]  = inf_.mean(axis=1)             # infilt_mean
    feats[:, 8]  = inf_.std(axis=1)              # infilt_std
    feats[:, 9]  = inf_.min(axis=1)              # infilt_min
    feats[:, 10] = inf_.max(axis=1)              # infilt_max
    feats[:, 11] = _batch_entropy(inf_, entropy_bins)  # infilt_entropy

    # ── Land use (channel 2) — categorical ────────────────────────────────────
    lu_raw = flat[:, 2, :]                       # (B, 16) float
    lu_int = np.round(lu_raw).astype(np.int32)
    feats[:, 12] = scipy_stats.mode(lu_int, axis=1, keepdims=False).mode.astype(np.float32)
    feats[:, 13] = np.array([len(np.unique(lu_int[i])) for i in range(B)], dtype=np.float32)
    feats[:, 14] = _batch_entropy(lu_raw, n_bins=11)  # 11 land use classes

    return feats


# ── Rainfall features ──────────────────────────────────────────────────────────
def _extract_rainfall_features(rainfall: np.ndarray) -> np.ndarray:
    B     = rainfall.shape[0]
    feats = np.empty((B, 16), dtype=np.float32)
    feats[:, :13] = rainfall.astype(np.float32)   # raw timesteps
    feats[:, 13]  = rainfall.sum(axis=1)           # rain_total
    feats[:, 14]  = rainfall.max(axis=1)           # rain_peak
    feats[:, 15]  = rainfall.argmax(axis=1).astype(np.float32) / 12.0  # rain_tpeak_idx [0,1]
    return feats


# ── Public API ─────────────────────────────────────────────────────────────────
def build_feature_matrix(
    spatial        : np.ndarray,            # (N, 3, 4, 4)
    rainfall       : np.ndarray,            # (N, 13)
    conditioning   : np.ndarray,            # (N, 4)
    full_dem_map   : Optional[np.ndarray] = None,  # (H, W) normalised DEM
    slope_map      : Optional[np.ndarray] = None,  # (H, W) precomputed slope
    patch_coords   : Optional[np.ndarray] = None,  # (N, 2) [row, col] of each patch
    patch_size     : int = 4,
    entropy_bins   : int = 16,
    chunk_size     : int = 250_000,
    verbose        : bool = True,
) -> Tuple[np.ndarray, List[str]]:
    
    N     = spatial.shape[0]
    names = get_feature_names()

    assert rainfall.shape    == (N, 13), f"Rainfall mismatch: {rainfall.shape}"
    assert conditioning.shape == (N, 4), f"Conditioning mismatch: {conditioning.shape}"

    # ── Resolve slope map ──────────────────────────────────────────────────────
    if slope_map is None and full_dem_map is not None:
        if verbose:
            print("[build_feature_matrix]  Computing slope map from full DEM...")
        slope_map = compute_slope_map(full_dem_map)

    if slope_map is None:
        print("  ⚠ WARNING: No slope_map or full_dem_map provided. "
              "dem_slope_mean and dem_slope_std will be 0.")

    # ── Pre-extract slope pixels per patch ────────────────────────────────────
    # slope_patches[i] = 16 slope values for patch i
    slope_patches_all = None
    if slope_map is not None and patch_coords is not None:
        if verbose:
            print(f"  Extracting slope patches for {N:,} patches...")
        slope_patches_all = np.empty((N, patch_size * patch_size), dtype=np.float32)
        for i in range(N):
            r, c = patch_coords[i]
            slope_patches_all[i] = slope_map[
                r:r + patch_size, c:c + patch_size
            ].flatten()
    elif slope_map is not None and patch_coords is None:
        # Fallback: extract slope from the DEM channel of spatial patches
        # This is less accurate (no neighbour context) but works without coords
        if verbose:
            print("  ⚠ patch_coords not provided — deriving slope from spatial DEM channel")
        slope_patches_all = None   # handled inside chunk loop below

    # ── Allocate output ────────────────────────────────────────────────────────
    X       = np.empty((N, N_FEATURES), dtype=np.float32)
    n_chunks = (N + chunk_size - 1) // chunk_size

    if verbose:
        print(f"\n[build_feature_matrix]  N={N:,}  F={N_FEATURES}  chunks={n_chunks}")

    chunks = tqdm(range(n_chunks), desc="  Features") if verbose else range(n_chunks)

    for c in chunks:
        lo = c * chunk_size
        hi = min(lo + chunk_size, N)
        B  = hi - lo

        sp  = spatial[lo:hi]       # (B, 3, 4, 4)
        ra  = rainfall[lo:hi]      # (B, 13)
        co  = conditioning[lo:hi]  # (B, 4)

        # Slope pixels for this chunk
        if slope_patches_all is not None:
            sl_px = slope_patches_all[lo:hi]    # (B, 16) pre-extracted
        elif slope_map is not None:
            # Fallback: use gradient of DEM channel of patch (less accurate)
            dem_ch = sp[:, 0, :, :].reshape(B, -1)  # (B, 16)
            dy, dx = np.gradient(dem_ch.reshape(B, patch_size, patch_size), axis=(1, 2))
            sl_px  = np.sqrt(dx**2 + dy**2).reshape(B, -1).astype(np.float32)
        else:
            sl_px = np.zeros((B, patch_size * patch_size), dtype=np.float32)

        spatial_feats = _extract_spatial_features(sp, sl_px, entropy_bins)  # (B, 15)
        rain_feats    = _extract_rainfall_features(ra)                        # (B, 16)
        cond_feats    = co.astype(np.float32)                                 # (B,  4)

        X[lo:hi] = np.concatenate([spatial_feats, rain_feats, cond_feats], axis=1)

    if verbose:
        print(f"  ✓ Done  shape={X.shape}  dtype={X.dtype}")
        _print_feature_summary(X, names)

    return X, names


def _print_feature_summary(X: np.ndarray, names: List[str]):
    print(f"\n  {'Feature':<22}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for i, name in enumerate(names):
        col = X[:, i]
        print(f"  {name:<22}  {col.mean():>8.4f}  {col.std():>8.4f}"
              f"  {col.min():>8.4f}  {col.max():>8.4f}")


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_feature_matrix(
    X     : np.ndarray,
    y     : np.ndarray,
    names : List[str],
    split : str = 'train',
) -> bool:
    print(f"\n[validate_feature_matrix]  split={split}")
    ok = True

    assert X.ndim == 2 and X.shape[1] == N_FEATURES, \
        f"Expected (N, {N_FEATURES}), got {X.shape}"
    print(f"  Shape      : {X.shape}  ✓")

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"  NaN        : {nan_count}  {'✓' if nan_count == 0 else '✗'}")
    print(f"  Inf        : {inf_count}  {'✓' if inf_count == 0 else '✗'}")
    if nan_count > 0 or inf_count > 0:
        ok = False

    unique_labels = np.unique(y)
    valid_labels  = np.all((unique_labels >= 0) & (unique_labels <= 4))
    print(f"  Labels     : {unique_labels.tolist()}  {'✓' if valid_labels else '✗'}")
    if not valid_labels:
        ok = False

    print(f"\n  Class distribution:")
    total = len(y)
    class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    for c, name in enumerate(class_names):
        count = (y == c).sum()
        print(f"    Class {c} ({name:<10}): {count:>10,}  ({count/total*100:5.1f}%)")

    # Slope sanity: slope features should be >= 0
    slope_ok = np.all(X[:, 4:6] >= 0)
    print(f"  Slope ≥ 0  : {'✓' if slope_ok else '✗'}")

    assert ok, "Validation failed"
    print(f"\n  ✓ All checks passed")
    return True