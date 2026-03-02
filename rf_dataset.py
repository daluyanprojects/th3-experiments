import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rf_config import RFConfig
from feature_engineering import (build_feature_matrix, validate_feature_matrix, compute_slope_map, get_feature_names)


def _extract_patch_coords(split_results: Dict, split: str) -> Optional[np.ndarray]:
    meta = split_results.get(split, {}).get('scenario_metadata', None)
    if meta is None:
        return None
    coords = np.array([m['patch_coord'] for m in meta], dtype=np.int32)  # (N, 2)
    return coords


def _extract_conditioning(rainfall_results: Dict, split_results: Dict, split: str) -> np.ndarray:
    cond = split_results[split].get('conditioning_vectors', None)
    if cond is not None:
        return np.asarray(cond, dtype=np.float32)
    
    cond_all = rainfall_results['conditioning_vectors']   # (S, 4)
    N        = split_results[split]['spatial_patches'].shape[0]
    return np.tile(cond_all[0], (N, 1)).astype(np.float32)


def _build_scenario_meta(split_results: Dict, split: str) -> List[Dict]:
    return split_results[split]['scenario_metadata']


def build_rf_datasets(
    split_results       : Dict,
    rainfall_results    : Dict,
    preprocessed_spatial: Dict,
    cfg                 : RFConfig,
    cache_dir           : Optional[Path] = None,
    verbose             : bool = True,
) -> Dict:
    
    if verbose:
        print("\n" + "=" * 60)
        print("RF DATASET BUILDER")
        print("=" * 60)

    # ── Try loading from cache ─────────────────────────────────────────────
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        result = _try_load_cache(cache_dir, verbose)
        if result is not None:
            return result

    # ── Slope map (computed once, shared across splits) ─────────────────────
    slope_map = None
    full_dem  = preprocessed_spatial.get('train', {}).get('dem', None)
    if cfg.compute_dem_slope and full_dem is not None:
        if verbose:
            print("\n  Computing slope map from full training DEM...")
        slope_map = compute_slope_map(full_dem)

    # ── Build features for each split ───────────────────────────────────────
    results = {}
    for split in ('train', 'test'):
        if verbose:
            print(f"\n  --- {split.upper()} SPLIT ---")

        spatial   = split_results[split]['spatial_patches']    # (N, 3, 4, 4)
        rainfall  = split_results[split]['rainfall_sequences'] # (N, 13)
        labels    = split_results[split]['labels']             # (N,)

        conditioning = _extract_conditioning(rainfall_results, split_results, split)
        patch_coords = _extract_patch_coords(split_results, split)

        if verbose:
            print(f"    spatial      : {spatial.shape}")
            print(f"    rainfall     : {rainfall.shape}")
            print(f"    conditioning : {conditioning.shape}")
            print(f"    labels       : {labels.shape}  unique={np.unique(labels).tolist()}")
            if patch_coords is not None:
                print(f"    patch_coords : {patch_coords.shape}")
            else:
                print(f"    patch_coords : None  (slope from DEM channel fallback)")

        X, feature_names = build_feature_matrix(
            spatial       = spatial,
            rainfall      = rainfall,
            conditioning  = conditioning,
            full_dem_map  = full_dem if split == 'train' else
                            preprocessed_spatial.get('test', {}).get('dem', None),
            slope_map     = slope_map,
            patch_coords  = patch_coords,
            patch_size    = cfg.patch_size,
            entropy_bins  = cfg.entropy_bins,
            verbose       = verbose,
        )

        validate_feature_matrix(X, labels, feature_names, split=split)

        results[f'X_{split}']   = X
        results[f'y_{split}']   = np.asarray(labels, dtype=np.int32)
        results[f'{split}_meta'] = _build_scenario_meta(split_results, split)

    results['feature_names'] = feature_names

    if verbose:
        print(f"\n  X_train : {results['X_train'].shape}")
        print(f"  X_test  : {results['X_test'].shape}")
        print(f"  y_train : {results['y_train'].shape}")
        print(f"  y_test  : {results['y_test'].shape}")

    # ── Optionally cache to disk ─────────────────────────────────────────────
    if cache_dir is not None:
        _save_cache(results, cache_dir, verbose)

    return results


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        'X_train'      : cache_dir / 'X_train.npy',
        'X_test'       : cache_dir / 'X_test.npy',
        'y_train'      : cache_dir / 'y_train.npy',
        'y_test'       : cache_dir / 'y_test.npy',
        'feature_names': cache_dir / 'feature_names.npy',
        'train_meta'   : cache_dir / 'train_meta.npy',
        'test_meta'    : cache_dir / 'test_meta.npy',
    }


def _try_load_cache(cache_dir: Path, verbose: bool) -> Optional[Dict]:
    paths = _cache_paths(cache_dir)
    if not all(p.exists() for p in paths.values()):
        return None
    if verbose:
        print(f"  Loading RF dataset from cache: {cache_dir}")
    return {
        'X_train'      : np.load(paths['X_train']),
        'X_test'       : np.load(paths['X_test']),
        'y_train'      : np.load(paths['y_train']),
        'y_test'       : np.load(paths['y_test']),
        'feature_names': np.load(paths['feature_names'], allow_pickle=True).tolist(),
        'train_meta'   : np.load(paths['train_meta'],    allow_pickle=True).tolist(),
        'test_meta'    : np.load(paths['test_meta'],     allow_pickle=True).tolist(),
    }


def _save_cache(results: Dict, cache_dir: Path, verbose: bool):
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(cache_dir)
    np.save(paths['X_train'],       results['X_train'])
    np.save(paths['X_test'],        results['X_test'])
    np.save(paths['y_train'],       results['y_train'])
    np.save(paths['y_test'],        results['y_test'])
    np.save(paths['feature_names'], np.array(results['feature_names'], dtype=object))
    np.save(paths['train_meta'],    np.array(results['train_meta'],    dtype=object))
    np.save(paths['test_meta'],     np.array(results['test_meta'],     dtype=object))
    if verbose:
        print(f"  RF dataset cached to: {cache_dir}")


def build_rf_datasets(
    split_results       : Dict,
    rainfall_results    : Dict,
    preprocessed_spatial: Dict,
    cfg                 : RFConfig,
    cache_dir           : Optional[Path] = None,
    force_rebuild       : bool = False,
    verbose             : bool = True,
) -> Dict:
    if force_rebuild and cache_dir is not None:
        cache_dir = Path(cache_dir)
        for p in _cache_paths(cache_dir).values():
            if p.exists():
                p.unlink()
        if verbose:
            print("  Cache cleared (force_rebuild=True)")

    return build_rf_datasets(
        split_results        = split_results,
        rainfall_results     = rainfall_results,
        preprocessed_spatial = preprocessed_spatial,
        cfg                  = cfg,
        cache_dir            = cache_dir,
        verbose              = verbose,
    )
