import os
import numpy as np
import rasterio
from typing import Tuple, Dict, List
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List
import pandas as pd


"""
Normalization functions for input data
"""
def normalize_dem(dem_data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
    """
    Normalize DEM data
    """
    # Filter valid data (exclude NaN and extreme values)
    valid_mask = ~np.isnan(dem_data) & (dem_data != -9999) & (dem_data > -1e10)
    
    if method == 'minmax':
        dem_min = np.min(dem_data[valid_mask])
        dem_max = np.max(dem_data[valid_mask])
        
        dem_normalized = np.where(
            valid_mask,
            (dem_data - dem_min) / (dem_max - dem_min),
            np.nan
        )
        
        params = {
            'method': 'minmax',
            'min': float(dem_min),
            'max': float(dem_max)
        }
        
        print(f"DEM normalized (min-max): [{dem_min:.4f}, {dem_max:.4f}] -> [0.0, 1.0]")
        
    elif method == 'zscore':
        dem_mean = np.mean(dem_data[valid_mask])
        dem_std = np.std(dem_data[valid_mask])
        
        dem_normalized = np.where(
            valid_mask,
            (dem_data - dem_mean) / dem_std,
            np.nan
        )
        
        params = {
            'method': 'zscore',
            'mean': float(dem_mean),
            'std': float(dem_std)
        }
        
        print(f"DEM normalized (z-score): mean={dem_mean:.4f}, std={dem_std:.4f}")
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return dem_normalized, params


def normalize_infiltration(infilt_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
    # Filter valid data
    valid_mask = ~np.isnan(infilt_data) & (infilt_data != -9999) & (infilt_data > -1e10)
    
    infilt_min = np.min(infilt_data[valid_mask])
    infilt_max = np.max(infilt_data[valid_mask])
    
    infilt_normalized = np.where(
        valid_mask,
        (infilt_data - infilt_min) / (infilt_max - infilt_min),
        np.nan
    )
    
    params = {
        'method': 'minmax',
        'min': float(infilt_min),
        'max': float(infilt_max)
    }
    
    print(f"Infiltration normalized (min-max): [{infilt_min:.4f}, {infilt_max:.4f}] -> [0.0, 1.0]")
    
    return infilt_normalized, params


def normalize_landuse(landuse_data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
    # Filter valid data
    valid_mask = ~np.isnan(landuse_data) & (landuse_data != -9999) & (landuse_data > -1e10)
    
    if method == 'minmax':
        lu_min = np.min(landuse_data[valid_mask])
        lu_max = np.max(landuse_data[valid_mask])
        
        landuse_normalized = np.where(
            valid_mask,
            (landuse_data - lu_min) / (lu_max - lu_min),
            np.nan
        )
        
        params = {
            'method': 'minmax',
            'min': float(lu_min),
            'max': float(lu_max)
        }
        
        print(f"Landuse normalized (min-max): [{lu_min:.4f}, {lu_max:.4f}] -> [0.0, 1.0]")
        
    elif method == 'standard':
        # Standard scaling (useful for categorical treated as ordinal)
        lu_mean = np.mean(landuse_data[valid_mask])
        lu_std = np.std(landuse_data[valid_mask])
        
        landuse_normalized = np.where(
            valid_mask,
            (landuse_data - lu_mean) / lu_std,
            np.nan
        )
        
        params = {
            'method': 'standard',
            'mean': float(lu_mean),
            'std': float(lu_std)
        }
        
        print(f"Landuse normalized (standard): mean={lu_mean:.4f}, std={lu_std:.4f}")
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return landuse_normalized, params


def denormalize_dem(dem_normalized: np.ndarray, params: Dict) -> np.ndarray:
    """
    Denormalize DEM back to original scale
    """
    if params['method'] == 'minmax':
        return dem_normalized * (params['max'] - params['min']) + params['min']
    elif params['method'] == 'zscore':
        return dem_normalized * params['std'] + params['mean']
    else:
        raise ValueError(f"Unknown method: {params['method']}")


def denormalize_infiltration(infilt_normalized: np.ndarray, params: Dict) -> np.ndarray:
    """
    Denormalize infiltration back to original scale
    """
    return infilt_normalized * (params['max'] - params['min']) + params['min']


def denormalize_landuse(landuse_normalized: np.ndarray, params: Dict) -> np.ndarray:
    """
    Denormalize landuse back to original scale
    """
    if params['method'] == 'minmax':
        return landuse_normalized * (params['max'] - params['min']) + params['min']
    elif params['method'] == 'standard':
        return landuse_normalized * params['std'] + params['mean']
    else:
        raise ValueError(f"Unknown method: {params['method']}")
    


"""
NoData Handling function
"""
def handle_nodata(data: np.ndarray, nodata_value: Optional[float] = None, fill_method: str = 'mean', name: str = "Raster") -> np.ndarray:
    print(f"\nHandling NoData for {name}...")
    
    data_copy = data.copy().astype(np.float32)
    
    # Create mask for NoData values
    nodata_mask = np.zeros(data_copy.shape, dtype=bool)
    
    # Detect common NoData values
    nodata_mask |= np.isnan(data_copy)
    nodata_mask |= (data_copy == -9999)
    nodata_mask |= (data_copy < -1e10)
    
    # Add specific nodata value if provided
    if nodata_value is not None:
        nodata_mask |= (data_copy == nodata_value)
    
    # Count NoData
    nodata_count = nodata_mask.sum()
    total_pixels = data_copy.size
    nodata_percent = (nodata_count / total_pixels) * 100
    
    print(f"  NoData pixels: {nodata_count:,} / {total_pixels:,} ({nodata_percent:.2f}%)")
    
    if nodata_count == 0:
        print(f"  ✓ No NoData values found")
        return data_copy
    
    # Set all NoData to NaN first
    data_copy[nodata_mask] = np.nan
    
    # Fill based on method
    if fill_method == 'mean':
        fill_value = np.nanmean(data_copy)
        data_copy = np.nan_to_num(data_copy, nan=fill_value)
        print(f"  ✓ Filled with mean: {fill_value:.4f}")
        
    elif fill_method == 'median':
        fill_value = np.nanmedian(data_copy)
        data_copy = np.nan_to_num(data_copy, nan=fill_value)
        print(f"  ✓ Filled with median: {fill_value:.4f}")
        
    elif fill_method == 'zero':
        data_copy = np.nan_to_num(data_copy, nan=0.0)
        print(f"  ✓ Filled with zero")
        
    elif fill_method == 'interpolate':
        # Nearest neighbor interpolation using scipy
        try:
            from scipy.ndimage import distance_transform_edt
            
            # Find indices of nearest valid pixels
            invalid_mask = np.isnan(data_copy)
            if invalid_mask.any():
                indices = distance_transform_edt(
                    invalid_mask, 
                    return_distances=False, 
                    return_indices=True
                )
                data_copy = data_copy[tuple(indices)]
            print(f"  ✓ Filled with nearest neighbor interpolation")
        except ImportError:
            print(f"  ⚠ scipy not available, falling back to mean")
            fill_value = np.nanmean(data_copy)
            data_copy = np.nan_to_num(data_copy, nan=fill_value)
            print(f"  ✓ Filled with mean: {fill_value:.4f}")
    
    elif fill_method == 'remove':
        print(f"  ✓ NoData values kept as NaN (not filled)")
        # Keep as NaN
    
    else:
        raise ValueError(f"Unknown fill method: {fill_method}")
    
    return data_copy
