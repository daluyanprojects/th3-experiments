import numpy as np
from typing import Dict, Optional, Tuple
from scipy.stats import entropy as scipy_entropy

from config import SVMConfig


# ── Entropy helper ─────────────────────────────────────────────────────────────
def _patch_entropy(patch_flat: np.ndarray, bins: int) -> float:
    _, counts = np.unique(patch_flat, return_counts=True)
    return float(scipy_entropy(counts, base=2))


# ── Per-channel feature extractors ────────────────────────────────────────────
def _dem_features(patch: np.ndarray, slope_patch: Optional[np.ndarray], bins: int) -> np.ndarray:
    flat = patch.ravel()
    feats = [flat.mean(), flat.std(), flat.min(), flat.max()]
    if slope_patch is not None:
        sf = slope_patch.ravel()
        feats += [sf.mean(), sf.std()]
    else:
        feats += [0.0, 0.0]
    feats.append(_patch_entropy(flat, bins))
    return np.array(feats, dtype=np.float32)


def _infiltration_features(patch: np.ndarray, bins: int) -> np.ndarray:
    flat = patch.ravel()
    return np.array(
        [flat.mean(), flat.std(), flat.min(), flat.max(), _patch_entropy(flat, bins)],
        dtype=np.float32,
    )


def _landuse_features(patch: np.ndarray, bins: int) -> np.ndarray:
    flat   = patch.ravel().astype(np.float32)
    mode   = float(np.bincount(flat.astype(int).clip(0)).argmax())
    uniq   = float(np.unique(flat).size)
    ent    = _patch_entropy(flat, bins)
    return np.array([mode, uniq, ent], dtype=np.float32)


def _rainfall_features(seq: np.ndarray) -> np.ndarray:
    total     = float(seq.sum())
    peak      = float(seq.max())
    tpeak_idx = float(seq.argmax())
    return np.concatenate([seq.astype(np.float32),
                            np.array([total, peak, tpeak_idx], dtype=np.float32)])


# ── Main extractor ─────────────────────────────────────────────────────────────
def extract_svm_features(
    spatial_patches   : np.ndarray,     # (N, patch_h, patch_w, C)   C=3
    rainfall_sequences: np.ndarray,     # (N, T)                       T=13
    conditioning_vecs : Optional[np.ndarray],  # (N, 4) or None
    cfg               : SVMConfig,
    slope_patches     : Optional[np.ndarray] = None,  # (N, patch_h, patch_w)
) -> np.ndarray:
    
    N = len(spatial_patches)
    print(f"\n[SVM Feature Extraction]  N={N:,} patches")

    rows = []
    for i in range(N):
        sp = spatial_patches[i]          # (H_p, W_p, 3)
        dem_ch  = sp[..., 0]
        inf_ch  = sp[..., 1]
        lu_ch   = sp[..., 2]
        slope_ch = slope_patches[i] if slope_patches is not None else None

        dem_f  = _dem_features(dem_ch,  slope_ch,     cfg.entropy_bins)   # 7
        inf_f  = _infiltration_features(inf_ch,        cfg.entropy_bins)   # 5
        lu_f   = _landuse_features(lu_ch,              cfg.entropy_bins)   # 3
        rain_f = _rainfall_features(rainfall_sequences[i])                  # 16
        cond_f = (conditioning_vecs[i].astype(np.float32)
                  if conditioning_vecs is not None
                  else np.zeros(4, dtype=np.float32))                        # 4

        rows.append(np.concatenate([dem_f, inf_f, lu_f, rain_f, cond_f]))

    X = np.stack(rows, axis=0)   # (N, 35)
    assert X.shape[1] == cfg.n_features, (
        f"Feature dim mismatch: got {X.shape[1]}, expected {cfg.n_features}"
    )
    print(f"  Feature matrix shape : {X.shape}")
    print(f"  NaNs                 : {np.isnan(X).sum()}")
    print(f"  Range                : [{X.min():.4f}, {X.max():.4f}]")
    return X


def create_svm_dataset(vit_results: Dict, cfg: SVMConfig) -> Dict:
    print("\n" + "=" * 70)
    print("SVM DATASET PREPARATION")
    print("=" * 70)

    # ── Unpack ────────────────────────────────────────────────────────────────
    X_tr_sp   = vit_results['train']['spatial_patches']
    X_tr_rain = vit_results['train']['rainfall_sequences']
    X_tr_cond = vit_results['train'].get('conditioning_vectors')
    y_train   = vit_results['train']['labels']

    X_te_sp   = vit_results['test']['spatial_patches']
    X_te_rain = vit_results['test']['rainfall_sequences']
    X_te_cond = vit_results['test'].get('conditioning_vectors')
    y_test    = vit_results['test']['labels']

    # ── Channel-order normalisation ───────────────────────────────────────────
    def _to_channels_last(arr):
        if arr.ndim == 4 and arr.shape[1] < arr.shape[2]:
            return arr.transpose(0, 2, 3, 1)
        return arr

    X_tr_sp = _to_channels_last(X_tr_sp)
    X_te_sp = _to_channels_last(X_te_sp)

    # ── Metadata key normalisation ────────────────────────────────────────────
    def _get_meta(split_dict):
        return split_dict.get('metadata') or split_dict.get('scenario_metadata', [])

    tr_meta = _get_meta(vit_results['train'])
    te_meta = _get_meta(vit_results['test'])

    print(f"\n  Raw training  : {X_tr_sp.shape}  labels={y_train.shape}")
    print(f"  Raw test      : {X_te_sp.shape}  labels={y_test.shape}")

    # ── Feature extraction ────────────────────────────────────────────────────
    print("\n[Train features]")
    X_train = extract_svm_features(X_tr_sp, X_tr_rain, X_tr_cond, cfg)

    print("\n[Test features]")
    X_test  = extract_svm_features(X_te_sp, X_te_rain, X_te_cond, cfg)

    tr_sids = vit_results['train']['scenario_ids']
    te_sids = vit_results['test']['scenario_ids']

    # ── Subsampling for tractability ──────────────────────────────────────────
    if cfg.max_train_samples and len(X_train) > cfg.max_train_samples:
        rng  = np.random.default_rng(cfg.subsample_seed)
        idx  = rng.choice(len(X_train), cfg.max_train_samples, replace=False)
        idx  = np.sort(idx)
        X_train = X_train[idx]
        y_train = y_train[idx]

        # tr_meta / tr_sids may be per-scenario (e.g. 35 entries) or
        # per-patch (e.g. 2.6M entries).  Only index with patch indices
        # when the list is actually patch-length; otherwise pass through.
        n_patches_orig = len(X_tr_sp)
        if len(tr_meta) == n_patches_orig:
            train_meta_sub = [tr_meta[i] for i in idx]
        else:
            train_meta_sub = tr_meta          # per-scenario — keep as-is

        if len(tr_sids) == n_patches_orig:
            train_sids_sub = [tr_sids[i] for i in idx]
        else:
            train_sids_sub = tr_sids          # per-scenario — keep as-is

        print(f"\n  Subsampled training set: {len(X_train):,} / original (SVM tractability)")
    else:
        train_meta_sub = tr_meta
        train_sids_sub = tr_sids

    # ── Class distribution ────────────────────────────────────────────────────
    print(f"\n  Class Distribution (after subsampling):")
    class_names = ['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme']
    print(f"    {'Class':<12} {'Train':>12} {'Test':>12}")
    print(f"    {'─'*38}")
    for c in range(cfg.num_classes):
        tr_n = (y_train == c).sum()
        te_n = (y_test  == c).sum()
        print(f"    {c} {class_names[c]:<10}: {tr_n:>8,}  {te_n:>8,}")

    return {
        'train': {
            'X'           : X_train,
            'y'           : y_train,
            'metadata'    : train_meta_sub,
            'scenario_ids': train_sids_sub,
        },
        'test': {
            'X'           : X_test,
            'y'           : y_test,
            'metadata'    : te_meta,
            'scenario_ids': te_sids,
        },
        'feature_names': _build_feature_names(cfg),
        'config'        : cfg,
    }


def _build_feature_names(cfg: SVMConfig):
    names = []
    # DEM (7)
    names += ['dem_mean', 'dem_std', 'dem_min', 'dem_max',
              'dem_slope_mean', 'dem_slope_std', 'dem_entropy']
    # Infiltration (5)
    names += ['inf_mean', 'inf_std', 'inf_min', 'inf_max', 'inf_entropy']
    # Land use (3)
    names += ['lu_mode', 'lu_n_unique', 'lu_entropy']
    # Rainfall timesteps (13)
    names += [f'rain_t{t}' for t in range(cfg.rainfall_timesteps)]
    # Rainfall derived (3)
    names += ['rain_total', 'rain_peak', 'rain_tpeak_idx']
    # Conditioning (4)
    names += ['cond_pattern_type', 'cond_depth_norm', 'cond_tpeak', 'cond_has_tpeak']
    assert len(names) == cfg.n_features, f"name count {len(names)} != {cfg.n_features}"
    return names
