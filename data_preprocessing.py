import numpy as np
from scipy.ndimage import zoom, distance_transform_edt
from typing import Tuple, Dict
import json


def preprocess_spatial_data(
    train_dem, train_infilt, train_landuse, test_dem, test_infilt, test_landuse,
    mask, target_shape=(1152, 1152), nodata_method='interpolate', norm_method='minmax'
):
    print("\n" + "="*70)
    print("PHASE 2: SPATIAL INPUT PREPROCESSING")
    print("="*70)
    
    results = {}
    norm_params = {}
    
    print("\n[2.1] Handling NoData Values")
    print("-" * 70)
    
    # DEM
    train_dem_clean = _handle_nodata(train_dem, nodata_method, 'DEM (Train)')
    test_dem_clean = _handle_nodata(test_dem, nodata_method, 'DEM (Test)')
    
    # Infiltration (ensure non-negative)
    train_infilt_clean = _handle_nodata(train_infilt, nodata_method, 'Infiltration (Train)')
    train_infilt_clean = np.maximum(train_infilt_clean, 0)
    test_infilt_clean = _handle_nodata(test_infilt, nodata_method, 'Infiltration (Test)')
    test_infilt_clean = np.maximum(test_infilt_clean, 0)
    
    # Landuse (use nearest neighbor for categorical)
    train_landuse_clean = _handle_nodata(train_landuse, 'interpolate', 'Landuse (Train)')
    test_landuse_clean = _handle_nodata(test_landuse, 'interpolate', 'Landuse (Test)')
    
    print(f"\n✓ NoData handling complete")
    
    # -------------------------------------------------------------------------
    # RESIZE TO TARGET DIMENSION
    # -------------------------------------------------------------------------
    print(f"\n[2.2] Resizing to {target_shape}")
    print("-" * 70)
    
    current_shape = train_dem_clean.shape
    print(f"  Current: {current_shape}")
    print(f"  Target: {target_shape}")
    
    if current_shape != target_shape:
        # Resize all inputs
        train_dem_resized = _resize(train_dem_clean, target_shape, order=1, name='Train DEM')
        train_infilt_resized = _resize(train_infilt_clean, target_shape, order=1, name='Train Infiltration')
        train_landuse_resized = _resize(train_landuse_clean, target_shape, order=0, name='Train Landuse')
        
        test_dem_resized = _resize(test_dem_clean, target_shape, order=1, name='Test DEM')
        test_infilt_resized = _resize(test_infilt_clean, target_shape, order=1, name='Test Infiltration')
        test_landuse_resized = _resize(test_landuse_clean, target_shape, order=0, name='Test Landuse')
        
        mask_resized = _resize(mask, target_shape, order=0, name='Mask')
        
        print(f"\n✓ All inputs resized to {target_shape}")
    else:
        print(f"  ✓ Already at target shape, skipping resize")
        train_dem_resized = train_dem_clean
        train_infilt_resized = train_infilt_clean
        train_landuse_resized = train_landuse_clean
        test_dem_resized = test_dem_clean
        test_infilt_resized = test_infilt_clean
        test_landuse_resized = test_landuse_clean
        mask_resized = mask
    
    print(f"\n[2.3] Normalizing to [0, 1] (method: {norm_method})")
    print("-" * 70)
    
    # DEM
    train_dem_norm, test_dem_norm, dem_params = _normalize(
        train_dem_resized, test_dem_resized, norm_method, 'DEM'
    )
    norm_params['dem'] = dem_params
    
    # Infiltration
    train_infilt_norm, test_infilt_norm, infilt_params = _normalize(
        train_infilt_resized, test_infilt_resized, norm_method, 'Infiltration'
    )
    norm_params['infiltration'] = infilt_params
    
    # Landuse
    train_landuse_norm, test_landuse_norm, landuse_params = _normalize(
        train_landuse_resized, test_landuse_resized, norm_method, 'Landuse'
    )
    norm_params['landuse'] = landuse_params
    
    print(f"\n✓ Normalization complete")
    
    print(f"\n[2.4] Quality Checks")
    print("-" * 70)
    
    checks = _quality_check(
        train_dem_norm, train_infilt_norm, train_landuse_norm,
        test_dem_norm, test_infilt_norm, test_landuse_norm,
        target_shape
    )
    
    results = {
        'train': {
            'dem': train_dem_norm,
            'infiltration': train_infilt_norm,
            'landuse': train_landuse_norm
        },
        'test': {
            'dem': test_dem_norm,
            'infiltration': test_infilt_norm,
            'landuse': test_landuse_norm
        },
        'mask': mask_resized,
        'normalization_params': norm_params,
        'target_shape': target_shape,
        'quality_checks': checks
    }
    
    print(f"  All inputs: {target_shape}")
    print(f"  No NaN values: {checks['no_nans']}")
    
    return results


def _handle_nodata(data, method, name):
    """Handle NoData values in raster"""
    # Identify NoData
    nodata_mask = np.isnan(data) | (data == -9999) | (data < -1e10)
    nodata_count = nodata_mask.sum()
    
    if nodata_count == 0:
        print(f"  {name}: No NoData ✓")
        return data.copy()
    
    print(f"  {name}: {nodata_count:,} NoData pixels ({nodata_count/data.size*100:.2f}%)")
    
    data_filled = data.copy()
    
    if method == 'interpolate':
        if nodata_mask.any():
            indices = distance_transform_edt(nodata_mask, return_distances=False, return_indices=True)
            data_filled = data[tuple(indices)]
        print(f"    → Filled with nearest neighbor")

    return data_filled


def _resize(data, target_shape, order, name):
    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    resized = zoom(data, zoom_factors, order=order)
    method = 'bilinear' if order == 1 else 'nearest'
    print(f"  {name}: {data.shape} → {target_shape} ({method})")
    return resized

def _normalize(train_data, test_data, method, name):
    valid_train = train_data[~np.isnan(train_data)]
    valid_test = test_data[~np.isnan(test_data)]
    
    if method == 'minmax':
        # Use global min/max from BOTH regions
        data_min = min(valid_train.min(), valid_test.min())
        data_max = max(valid_train.max(), valid_test.max())
        
        train_norm = (train_data - data_min) / (data_max - data_min)
        test_norm = (test_data - data_min) / (data_max - data_min)
        
        params = {'method': 'minmax', 'min': float(data_min), 'max': float(data_max)}
        print(f"  {name}: Global range [{data_min:.4f}, {data_max:.4f}] → [0, 1]")
        print(f"    Train: [{valid_train.min():.4f}, {valid_train.max():.4f}]")
        print(f"    Test:  [{valid_test.min():.4f}, {valid_test.max():.4f}]")
        
    return train_norm, test_norm, params


def _quality_check(train_dem, train_infilt, train_landuse, test_dem, test_infilt, test_landuse, expected_shape):
    checks = {}
    
    # Shapes
    shapes_ok = all([
        train_dem.shape == expected_shape,
        test_dem.shape == expected_shape
    ])
    print(f"\n  Shape check: {expected_shape} {'✓' if shapes_ok else '✗'}")
    checks['shapes_ok'] = shapes_ok
    
    # No NaNs
    no_nans = not any([
        np.isnan(train_dem).any(),
        np.isnan(test_dem).any()
    ])
    print(f"  NaN check: {'✓ None found' if no_nans else '✗ NaNs detected'}")
    checks['no_nans'] = no_nans
    
    # Value ranges
    in_range = all([
        0 <= train_dem.min() <= train_dem.max() <= 1,
        0 <= test_dem.min() <= test_dem.max() <= 1
    ])
    print(f"  Range check [0,1]: Train [{train_dem.min():.3f}, {train_dem.max():.3f}], Test [{test_dem.min():.3f}, {test_dem.max():.3f}] {'✓' if in_range else '✗'}")
    checks['values_in_range'] = in_range
    
    checks['all_passed'] = all(checks.values())
    
    return checks


def save_preprocessing_params(results, output_path):
    params = {
        'target_shape': results['target_shape'],
        'normalization': results['normalization_params'],
        'quality_checks': results['quality_checks']
    }
    
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\n✓ Preprocessing parameters saved to: {output_path}")