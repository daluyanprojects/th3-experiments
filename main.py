import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import Affine
from pathlib import Path
import geopandas as gpd
import pandas as pd
import sys

from config import TrainConfig


def setup_flood_visualization():
    """Setup flood color mapping and legend for visualization"""
    
    FLOOD_COLORS = {
        0: '#FFFFFF',  # No Flood
        1: '#C6DBEF',  # Light
        2: '#6BAED6',  # Moderate
        3: '#2171B5',  # Heavy
        4: '#08306B'   # Extreme
    }
    
    FLOOD_LABELS = {
        0: 'No Flood',
        1: 'Light',
        2: 'Moderate',
        3: 'Heavy',
        4: 'Extreme'
    }
    
    # Create colormap with transparent background
    colors_list = ['#FFFFFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B']
    flood_cmap = mcolors.ListedColormap(colors_list)
    flood_cmap.set_under(alpha=0)
    flood_cmap.set_bad(alpha=0)
    
    # Create legend patches
    legend_patches = [
        mpatches.Patch(color=FLOOD_COLORS[i], label=FLOOD_LABELS[i])
        for i in range(5)
    ]
    
    # Create boundary norm for discrete colors
    flood_norm = mcolors.BoundaryNorm(
        [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        6
    )
    
    return {
        'colors': FLOOD_COLORS,
        'labels': FLOOD_LABELS,
        'cmap': flood_cmap,
        'norm': flood_norm,
        'legend_patches': legend_patches
    }


# ============================================================================
# SECTION 2: GeoTIFF Setup
# ============================================================================

def setup_geotiff_transform():
    
    # Original DEM bounds
    src_left, src_top = 13457490.0, 1664610.0
    src_W, src_H = 1125, 1224
    
    # Target dimensions
    dst_W, dst_H = 1152, 1152
    src_right = 13491240.0
    src_bottom = 1627890.0
    
    # Calculate pixel size
    new_pixel_x = (src_right - src_left) / dst_W
    new_pixel_y = (src_bottom - src_top) / dst_H
    
    # Create affine transform (EPSG:3857 - Web Mercator)
    dem_transform = Affine(new_pixel_x, 0.0, src_left, 0.0, new_pixel_y, src_top)
    dem_crs = rasterio.crs.CRS.from_epsg(3857)
    
    return {
        'transform': dem_transform,
        'crs': dem_crs,
        'width': dst_W,
        'height': dst_H
    }


# ============================================================================
# SECTION 3: Barangay Lookup Table
# ============================================================================

# City mapping for administrative boundaries
_CITY_LABEL = {
    'PH1303901': 'Manila City',
    'PH1307401': 'Mandaluyong',
    'PH1307404': 'Marikina',
    'PH1307405': 'Pasig',
    'PH1307501': 'Las Piñas',
    'PH1307503': 'Parañaque',
    'PH1307602': 'San Juan',
    'PH1307605': 'Taguig'
}


def build_barangay_lookup(geojson_path: str) -> dict:
    """
    Build a lookup table from GeoJSON file with barangay information
    
    Args:
        geojson_path: Path to manila_barangay_geojson.geojson
        
    Returns:
        Dictionary mapping PSGC codes to barangay info
    """
    print(f"Loading barangay data from {geojson_path}...")
    
    # Read GeoJSON file
    gdf = gpd.read_file(geojson_path)
    gdf['adm3_pcode'] = gdf['adm3_pcode'].astype(str)
    
    # Filter and prepare data
    gdf_all = gdf.dropna(subset=['psgc_code']).copy()
    gdf_all['psgc_int'] = gdf_all['psgc_code'].astype(float).astype(int)
    
    # Build lookup dictionary
    lookup = {}
    for _, row in gdf_all.iterrows():
        psgc = int(row['psgc_int'])
        city = _CITY_LABEL.get(str(row['adm3_pcode']), str(row['adm3_pcode']))
        
        # Use barangay name or generate from PSGC
        name = row['adm4_en'] if pd.notna(row['adm4_en']) else f'Barangay (PSGC {psgc})'
        
        lookup[psgc] = {
            'name': name,
            'city': city,
            'adm4_pcode': row['adm4_pcode'] if pd.notna(row.get('adm4_pcode')) else '',
            'adm3_pcode': row['adm3_pcode'],
            'area_sqkm': float(row['shape_sqkm']) if pd.notna(row.get('shape_sqkm')) else 0.0,
        }
    
    print(f"✓ Barangay lookup ready: {len(lookup):,} entries")
    return lookup


# ============================================================================
# SECTION 4: Inference Setup
# ============================================================================

def run_flood_prediction(
    user_type: str,
    geojson_path: str,
    spatial_data_path: str,
    output_dir: str,
    storm_type: str,
    depth_mm: float,
    tpeak: float
):
    # Validate user_type
    if user_type.lower() not in ['pedestrian', 'vehicle']:
        raise ValueError(f"user_type must be 'pedestrian' or 'vehicle', got '{user_type}'")
    
    if user_type.lower() == 'pedestrian':
        from inference_ped import (
            InferenceEngine,
            _load_barangay_band_from_geojson,
            run_inference
        )
        output_dir_param = 'output_ped_dir'
        data_suffix = '_ped'
    else:  # vehicle
        from inference import (
            InferenceEngine,
            _load_barangay_band_from_geojson,
            run_inference
        )
        output_dir_param = 'output_dir'
        data_suffix = ''
    
    print("\n" + "="*60)
    print(f"Running Flood Prediction [{user_type.upper()}]")
    print("="*60)
    
    # Setup configuration and transforms
    cfg = TrainConfig()
    flood_viz = setup_flood_visualization()
    geotiff_setup = setup_geotiff_transform()
    
    # Load barangay band
    print("\nLoading barangay boundaries...")
    barangay_band = _load_barangay_band_from_geojson(
        geojson_path,
        geotiff_setup['transform'],
        geotiff_setup['crs'],
        geotiff_setup['height'],
        geotiff_setup['width']
    )
    
    # Initialize inference engine
    print("Initializing inference engine...")
    engine = InferenceEngine(patch_data_path=spatial_data_path)
    
    # Run inference
    print(f"\nPredicting: {storm_type.capitalize()}, {depth_mm} mm, tpeak={tpeak}")
    
    # Call run_inference with appropriate parameter name
    results = run_inference(
        engine=engine,
        storm_type=storm_type,
        depth_mm=depth_mm,
        tpeak=tpeak,
        dem_transform=geotiff_setup['transform'],
        dem_crs=geotiff_setup['crs'],
        barangay_band=barangay_band,
        **{output_dir_param: Path(output_dir)}
    )
    
    print(f"\n✓ File saved at: {results['tif_path']}")
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # Set user type directly (pedestrian or vehicle)
    user_type = 'vehicle'  # Change to 'pedestrian' as needed
    
    # Define file paths based on user type
    GEOJSON_PATH = 'manila_barangay_geojson.geojson'
    
    if user_type == 'pedestrian':
        SPATIAL_DATA_PATH = 'outputs_ped/spatial_data.npz'
        OUTPUT_DIR = './input_testing'
    else:  # vehicle
        SPATIAL_DATA_PATH = 'outputs/spatial_data.npz'
        OUTPUT_DIR = './input_testing'
    
    print(f"\n[{user_type.upper()}] Mode selected")
    print(f"  Spatial data: {SPATIAL_DATA_PATH}")
    print(f"  Output dir : {OUTPUT_DIR}")
    
    # Build barangay lookup table
    print("\nBuilding barangay lookup table...")
    barangay_lookup = build_barangay_lookup(GEOJSON_PATH)
    
    # Print summary
    print(f'\nBARANGAY_LOOKUP ready  : {len(barangay_lookup):,} entries')
    print(f'PSGC range             : {min(barangay_lookup):,} – {max(barangay_lookup):,}')
    cities = sorted(set(v["city"] for v in barangay_lookup.values()))
    print(f'Cities                 : {cities}')
    
    # Run flood prediction
    results = run_flood_prediction(
        user_type=user_type,
        geojson_path=GEOJSON_PATH,
        spatial_data_path=SPATIAL_DATA_PATH,
        output_dir=OUTPUT_DIR,
        storm_type='triangular',
        depth_mm=50.0,
        tpeak=0.5
    )
    
    print("\nDone! ✓")