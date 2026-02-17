import os
import numpy as np
import rasterio
from typing import Tuple, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List

def load_dem(file_path: str) -> Tuple[np.ndarray, Dict]:
    with rasterio.open(file_path) as src:
        dem_data = src.read(1)  # Read first band
        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'nodata': src.nodata
        }
    
    print(f"DEM loaded: shape {dem_data.shape}, dtype {dem_data.dtype}")
    return dem_data, metadata


def load_infiltration_map(file_path: str) -> Tuple[np.ndarray, Dict]:
    with rasterio.open(file_path) as src:
        infilt_data = src.read(1)
        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'nodata': src.nodata
        }
    
    print(f"Infiltration map loaded: shape {infilt_data.shape}, dtype {infilt_data.dtype}")
    return infilt_data, metadata


def load_landuse_map(file_path: str) -> Tuple[np.ndarray, Dict]:
    with rasterio.open(file_path) as src:
        landuse_data = src.read(1)
        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'nodata': src.nodata
        }
    
    print(f"Landuse map loaded: shape {landuse_data.shape}, dtype {landuse_data.dtype}")
    return landuse_data, metadata


def load_flood_map(file_path: str, scenario_id: int = None) -> Tuple[np.ndarray, Dict]:
    with rasterio.open(file_path) as src:
        flood_data = src.read(1)
        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'nodata': src.nodata,
            'scenario_id': scenario_id
        }
    
    if scenario_id is not None:
        print(f"Flood map RS{scenario_id} loaded: shape {flood_data.shape}, dtype {flood_data.dtype}")
    else:
        print(f"Flood map loaded: shape {flood_data.shape}, dtype {flood_data.dtype}")
    
    return flood_data, metadata


def gmm_path_load_all_flood_maps(flood_dir: str, num_scenarios: int) -> Tuple[List[np.ndarray], List[Dict]]:
    flood_maps = []
    flood_metadata = []
    
    for i in range(1, num_scenarios + 1):
        file_path = os.path.join(flood_dir, f"RS{i}_GM30COP.tif")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        flood_data, metadata = load_flood_map(file_path, scenario_id=i)
        flood_maps.append(flood_data)
        flood_metadata.append(metadata)
    
    print(f"\nTotal flood maps loaded: {len(flood_maps)}")
    return flood_maps, flood_metadata

def gmm_no_manila_path_box_load_all_flood_maps(flood_dir: str, num_scenarios: int) -> Tuple[List[np.ndarray], List[Dict]]:
    flood_maps = []
    flood_metadata = []
    
    for i in range(1, num_scenarios + 1):
        file_path = os.path.join(flood_dir, f"RS{i}_GM30COP_noManila_box.tif")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        flood_data, metadata = load_flood_map(file_path, scenario_id=i)
        flood_maps.append(flood_data)
        flood_metadata.append(metadata)
    
    print(f"\nTotal flood maps loaded: {len(flood_maps)}")
    return flood_maps, flood_metadata

def gmm_no_manila_path_shape_load_all_flood_maps(flood_dir: str, num_scenarios: int) -> Tuple[List[np.ndarray], List[Dict]]:
    flood_maps = []
    flood_metadata = []
    
    for i in range(1, num_scenarios + 1):
        file_path = os.path.join(flood_dir, f"RS{i}_GM30COP_noManila.tif")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        flood_data, metadata = load_flood_map(file_path, scenario_id=i)
        flood_maps.append(flood_data)
        flood_metadata.append(metadata)
    
    print(f"\nTotal flood maps loaded: {len(flood_maps)}")
    return flood_maps, flood_metadata

def manila_path_shape_load_all_flood_maps(flood_dir: str, num_scenarios: int) -> Tuple[List[np.ndarray], List[Dict]]:
    flood_maps = []
    flood_metadata = []
    
    for i in range(1, num_scenarios + 1):
        file_path = os.path.join(flood_dir, f"RS{i}_GM30COP_Manila_box.tif")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        flood_data, metadata = load_flood_map(file_path, scenario_id=i)
        flood_maps.append(flood_data)
        flood_metadata.append(metadata)
    
    print(f"\nTotal flood maps loaded: {len(flood_maps)}")
    return flood_maps, flood_metadata

def get_raster_stats(data: np.ndarray, name: str = "Raster") -> Dict:
    valid_data = data[~np.isnan(data)]
    valid_data = valid_data[valid_data != -9999]
    valid_data = valid_data[valid_data > -1e10] 
    
    stats = {
        'name': name,
        'shape': data.shape,
        'dtype': data.dtype,
        'min': np.min(valid_data) if len(valid_data) > 0 else None,
        'max': np.max(valid_data) if len(valid_data) > 0 else None,
        'mean': np.mean(valid_data) if len(valid_data) > 0 else None,
        'std': np.std(valid_data) if len(valid_data) > 0 else None,
        'valid_pixels': len(valid_data),
        'total_pixels': data.size
    }
    
    return stats


def print_raster_stats(stats: Dict):
    print(f"\n{'='*60}")
    print(f"Statistics for: {stats['name']}")
    print(f"{'='*60}")
    print(f"Shape:         {stats['shape']}")
    print(f"Dtype:         {stats['dtype']}")
    print(f"Min value:     {stats['min']:.4f}" if stats['min'] is not None else "Min value:     None")
    print(f"Max value:     {stats['max']:.4f}" if stats['max'] is not None else "Max value:     None")
    print(f"Mean value:    {stats['mean']:.4f}" if stats['mean'] is not None else "Mean value:    None")
    print(f"Std dev:       {stats['std']:.4f}" if stats['std'] is not None else "Std dev:       None")
    print(f"Valid pixels:  {stats['valid_pixels']:,} / {stats['total_pixels']:,}")
    print(f"{'='*60}\n")


def load_rainfall_scenario(file_path: str, scenario_id: int = None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    if scenario_id is not None:
        print(f"Rainfall Scenario {scenario_id} loaded: {len(df)} time steps")
    else:
        print(f"Rainfall scenario loaded: {len(df)} time steps")
    return df

def load_all_flood_maps(flood_dir: str, num_scenarios: int) -> Tuple[List[np.ndarray], List[Dict]]:
    flood_maps = []
    flood_metadata = []
    
    for i in range(1, num_scenarios + 1):
        file_path = os.path.join(flood_dir, f"RS{i}_GM30COP_Manila_box.tif")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        flood_data, metadata = load_flood_map(file_path, scenario_id=i)
        flood_maps.append(flood_data)
        flood_metadata.append(metadata)
    
    print(f"\nTotal flood maps loaded: {len(flood_maps)}")
    return flood_maps, flood_metadata

def load_all_rainfall_scenarios(rainfall_dir: str) -> List[pd.DataFrame]:
    scenarios = []

    for file in os.listdir(rainfall_dir):
        if file.endswith(".csv") and "Rainfall_Scenario" in file:
            file_path = os.path.join(rainfall_dir, file)

            scenario_id = int(file.split("_")[2])
            df = load_rainfall_scenario(file_path, scenario_id=scenario_id)
            scenarios.append(df)

    print(f"\nTotal rainfall scenarios loaded: {len(scenarios)}")
    return scenarios



def get_rainfall_stats(scenario_df: pd.DataFrame, scenario_id: int = None) -> Dict:
    stats = {
        'scenario_id': scenario_id,
        'duration_hours': scenario_df['hour'].max(),
        'num_timesteps': len(scenario_df),
        'max_intensity': scenario_df['intensity_mmhr'].max(),
        'mean_intensity': scenario_df['intensity_mmhr'].mean(),
        'total_rainfall_mm': scenario_df['intensity_mmhr'].sum() * (scenario_df['hour'].iloc[1] - scenario_df['hour'].iloc[0]) if len(scenario_df) > 1 else 0,
        'peak_hour': scenario_df.loc[scenario_df['intensity_mmhr'].idxmax(), 'hour']
    }
    
    return stats


def print_rainfall_stats(stats: Dict):
    print(f"\n{'='*60}")
    if stats['scenario_id'] is not None:
        print(f"Rainfall Scenario {stats['scenario_id']} Statistics")
    else:
        print(f"Rainfall Scenario Statistics")
    print(f"{'='*60}")
    print(f"Duration:          {stats['duration_hours']:.2f} hours")
    print(f"Time steps:        {stats['num_timesteps']}")
    print(f"Max intensity:     {stats['max_intensity']:.4f} mm/hr")
    print(f"Mean intensity:    {stats['mean_intensity']:.4f} mm/hr")
    print(f"Total rainfall:    {stats['total_rainfall_mm']:.2f} mm")
    print(f"Peak at hour:      {stats['peak_hour']:.4f}")
    print(f"{'='*60}\n")

def visualize_raster(data: np.ndarray,  title: str = "Raster Data", cmap: str = 'terrain', figsize: Tuple[int, int] = (10, 8), 
                     vmin: Optional[float] = None, vmax: Optional[float] = None, show_colorbar: bool = True, auto_crop: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle nodata values
    plot_data = data.copy()
    plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)
    plot_data = np.ma.masked_where(plot_data == -9999, plot_data)
    plot_data = np.ma.masked_where(plot_data < -1e10, plot_data)
    
    if auto_crop:
        # Find the bounding box of valid (non-masked) data
        valid_mask = ~plot_data.mask if np.ma.is_masked(plot_data) else np.ones_like(data, dtype=bool)
        
        # Find rows and columns with valid data
        rows_with_data = np.any(valid_mask, axis=1)
        cols_with_data = np.any(valid_mask, axis=0)
        
        row_indices = np.where(rows_with_data)[0]
        col_indices = np.where(cols_with_data)[0]
        
        if len(row_indices) > 0 and len(col_indices) > 0:
            row_min, row_max = row_indices[0], row_indices[-1] + 1
            col_min, col_max = col_indices[0], col_indices[-1] + 1
            
            # Crop to valid extent
            plot_data = plot_data[row_min:row_max, col_min:col_max]
    
    # Plot
    im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    return fig

def visualize_flood_maps_grid(flood_maps: List[np.ndarray], scenario_ids: Optional[List[int]] = None, figsize: Tuple[int, int] = (25, 20),
                              ncols: int = 5, cmap: str = 'Blues', auto_crop: bool = False) -> plt.Figure:
    n_maps = len(flood_maps)
    nrows = (n_maps + ncols - 1) // ncols
    
    if scenario_ids is None:
        scenario_ids = list(range(1, n_maps + 1))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Find global vmin/vmax for consistent colorbar scaling
    all_valid_data = []
    for flood_map in flood_maps:
        valid = flood_map[~np.isnan(flood_map)]
        valid = valid[valid != -9999]
        if len(valid) > 0:
            all_valid_data.extend(valid)
    
    vmin = 0  
    vmax = np.max(all_valid_data) if len(all_valid_data) > 0 else 1
    
    for idx, (flood_map, scenario_id) in enumerate(zip(flood_maps, scenario_ids)):
        ax = axes[idx]
        
        # Handle nodata values
        plot_data = flood_map.copy()
        plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)
        plot_data = np.ma.masked_where(plot_data == -9999, plot_data)
        plot_data = np.ma.masked_where(plot_data < -1e10, plot_data)
        
        if auto_crop:
            valid_mask = ~plot_data.mask if np.ma.is_masked(plot_data) else np.ones_like(flood_map, dtype=bool)
            rows_with_data = np.any(valid_mask, axis=1)
            cols_with_data = np.any(valid_mask, axis=0)
            
            row_indices = np.where(rows_with_data)[0]
            col_indices = np.where(cols_with_data)[0]
            
            if len(row_indices) > 0 and len(col_indices) > 0:
                row_min, row_max = row_indices[0], row_indices[-1] + 1
                col_min, col_max = col_indices[0], col_indices[-1] + 1
                plot_data = plot_data[row_min:row_max, col_min:col_max]
        
        im = ax.imshow(plot_data, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'RS{scenario_id}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    for idx in range(n_maps, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Flood Maps for All Rainfall Scenarios', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig
