import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import Affine
from pathlib import Path
import geopandas as gpd
import pandas as pd
import sys
import torch
from rasterio.features import rasterize
from shapely.geometry import box
import os
import json

from config import TrainConfig


def setup_flood_visualization():    
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


def load_barangay_band(geojson_path: str, dst_transform, dst_crs, H, W):
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.dropna(subset=['psgc_code']).copy()
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna() & gdf.geometry.is_valid]
    gdf['psgc_int'] = gdf['psgc_code'].astype(float).astype(int)
    
    # Clip to grid extent
    left = dst_transform.c
    top = dst_transform.f
    right = left + dst_transform.a * W
    bottom = top + dst_transform.e * H
    grid_box = box(left, min(top, bottom), right, max(top, bottom))
    gdf = gdf[gdf.geometry.intersects(grid_box)]
    
    print(f"  Rasterizing {len(gdf)} barangays...")
    
    shapes = [(geom, psgc) for geom, psgc in zip(gdf.geometry, gdf['psgc_int']) 
              if geom is not None and not geom.is_empty]
    
    return rasterize(shapes, out_shape=(H, W), transform=dst_transform, 
                     fill=0, dtype='int32', all_touched=True)


def select_best_checkpoint(checkpoint_dir: Path) -> tuple:
    fold_checkpoints = sorted(checkpoint_dir.glob('fold_*_best.pth'))
    
    if not fold_checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found in {checkpoint_dir}")
    
    fold_scores = {}
    
    # Load all checkpoints and extract F1 score metric
    print(f"  Evaluating {len(fold_checkpoints)} fold checkpoints by F1 score...")
    for ckpt_path in fold_checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Extract F1 score (higher is better)
            f1_score = ckpt.get('best_monitor', float('-inf'))
            fold_scores[ckpt_path] = {
                'value': f1_score,
                'epoch': ckpt.get('epoch', 'N/A'),
                'global_step': ckpt.get('global_step', 'N/A')
            }
        except Exception as e:
            print(f"    ⚠ Error loading {ckpt_path.name}: {e}")
            fold_scores[ckpt_path] = {'value': float('-inf')}
    
    # Select checkpoint with highest F1 score
    best_checkpoint_path = max(fold_scores.keys(), key=lambda k: fold_scores[k]['value'])
    
    return best_checkpoint_path, fold_scores


def run_complete_testing(
    user_type: str,
    geojson_path: str,
    spatial_data_path: str,
    output_dir: str,
    storm_type: str = 'triangular',
    depth_mm: float = 50.0,
    tpeak: float = 0.5,
    mask_tif_path: str = 'manila_box_shape.tif'
):
    is_pedestrian = user_type.lower() == 'pedestrian'
    
    # Import correct modules based on user type
    if is_pedestrian:
        from inference_ped import InferenceEngine as InferenceEngine_Mode, run_inference
    else:  # vehicle
        from inference import InferenceEngine as InferenceEngine_Mode, run_inference
    
    print("\n" + "="*70)
    print(f"COMPLETE TESTING PIPELINE - {user_type.upper()} MODE")
    print("="*70)
    
    # Setup transforms and visualization
    geotiff_setup = setup_geotiff_transform()
    dem_transform = geotiff_setup['transform']
    dem_crs = geotiff_setup['crs']
    
    # ✓ Load barangay boundaries
    print("\n[1] LOADING BARANGAY BOUNDARIES")
    print("-" * 70)
    print(f"GeoJSON file: {geojson_path}")
    _barangay_band = load_barangay_band(
        geojson_path,
        dem_transform,
        dem_crs,
        geotiff_setup['height'],
        geotiff_setup['width']
    )
    print(f"✓ Barangay grid shape: {_barangay_band.shape}")
    print(f"  Unique barangays: {len(np.unique(_barangay_band)) - 1}")
    
    # ✓ Load metadata
    print("\n[2] LOADING METADATA")
    print("-" * 70)
    if is_pedestrian:
        metadata_path = Path('outputs_ped/logs_tuned/metadata.json')
    else:
        metadata_path = Path('outputs/logs_tuned/metadata.json')
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        best_fold_idx = metadata.get('best_fold_idx', 0)
        print(f"✓ Loaded metadata from {metadata_path}")
        print(f"  best_fold_idx: {best_fold_idx}")
    else:
        print(f"⚠ Metadata not found at {metadata_path}")
        best_fold_idx = 0
    
    # ✓ Load config
    print("\n[3] LOADING CONFIG")
    print("-" * 70)
    if is_pedestrian:
        config_path = Path('outputs_ped/logs_tuned/config_tuned.json')
    else:
        config_path = Path('outputs/logs_tuned/config_tuned.json')
    
    cfg_tuned = TrainConfig.load(str(config_path))
    print(f"✓ Loaded {config_path}")
    print(f"  embed_dim: {cfg_tuned.embed_dim}")
    print(f"  num_layers: {cfg_tuned.num_layers}")
    print(f"  rainfall_hidden: {cfg_tuned.rainfall_hidden}")
    
    # ✓ Load checkpoint
    print("\n[4] LOADING CHECKPOINT")
    print("-" * 70)
    if is_pedestrian:
        checkpoint_dir = Path('outputs_ped/checkpoints_tuned')
    else:
        checkpoint_dir = Path('outputs/checkpoints_tuned')
    
    checkpoint_path, fold_scores = select_best_checkpoint(checkpoint_dir)
    print(f"✓ Selected BEST checkpoint: {checkpoint_path.name}")
    print(f"  Total folds evaluated: {len(fold_scores)}")
    
    # Print fold ranking
    sorted_folds = sorted(fold_scores.items(), 
                         key=lambda x: x[1]['value'], 
                         reverse=True)
    for rank, (path, score_info) in enumerate(sorted_folds, 1):
        is_best = "★ BEST" if path == checkpoint_path else "     "
        print(f"    {rank}. {is_best} {path.name:30s} | F1: {score_info['value']:.6f} | epoch: {score_info['epoch']}")
    
    # ✓ Initialize inference engine
    print("\n[5] INITIALIZING INFERENCE ENGINE")
    print("-" * 70)
    engine = InferenceEngine_Mode(
        checkpoint_path=str(checkpoint_path),
        patch_data_path=spatial_data_path,
        config=cfg_tuned
    )
    print(f"✓ Inference engine initialized")
    print(f"  Spatial data shape: {engine.test_spatial.shape}")
    print(f"  DEM shape: {engine.test_dem.shape}")
    
    # ✓ Run inference
    print("\n[6] RUNNING INFERENCE")
    print("-" * 70)
    print(f"Storm parameters:")
    print(f"  • Storm type        : {storm_type.capitalize()}")
    print(f"  • Rainfall depth    : {depth_mm} mm")
    print(f"  • Peak time (tpeak) : {tpeak}")
    
    # Get absolute path to mask file
    repo_root = os.getcwd()
    mask_path = os.path.join(repo_root, mask_tif_path)
    
    # Call run_inference with correct parameters based on user type
    if is_pedestrian:
        res = run_inference(
            engine=engine,
            storm_type=storm_type,
            depth_mm=depth_mm,
            tpeak=tpeak,
            output_ped_dir=Path(output_dir),
            dem_transform=dem_transform,
            dem_crs=dem_crs,
            barangay_band=_barangay_band,
            mask_tif_path=mask_path
        )
    else:  # vehicle
        res = run_inference(
            engine=engine,
            storm_type=storm_type,
            depth_mm=depth_mm,
            tpeak=tpeak,
            output_dir=Path(output_dir),
            dem_transform=dem_transform,
            dem_crs=dem_crs,
            barangay_band=_barangay_band,
            mask_tif_path=mask_path
        )
    
    # ✓ Print results summary
    print("\n[7] RESULTS SUMMARY")
    print("-" * 70)
    print(f"File saved at: {res['tif_path']}")
    print(f"\nPrediction Statistics:")
    print(f"  • Total patches     : {res['total_patches']:,}")
    print(f"  • Flooded patches   : {res['flooded_patches']:,} ({res['flooded_pct']:.1f}%)")
    print(f"  • Mean confidence   : {res['mean_confidence']:.4f}")
    
    print("\n" + "="*70)
    print("✓ TESTING COMPLETE")
    print("="*70 + "\n")
    
    return res


if __name__ == "__main__":
    
    # ============================================================================
    # CONFIGURATION - Edit these to change behavior
    # ============================================================================
    
    # Set user type: 'pedestrian' or 'vehicle'
    user_type = 'pedestrian'  # ← Change this to 'vehicle' to use vehicle model
    
    # Define file paths
    GEOJSON_PATH = 'manila_barangay_geojson.geojson'
    
    if user_type == 'pedestrian':
        SPATIAL_DATA_PATH = 'outputs_ped/spatial_data.npz'
        OUTPUT_DIR = './input_testing_ped'
    else:  # vehicle
        SPATIAL_DATA_PATH = 'outputs/spatial_data.npz'
        OUTPUT_DIR = './input_testing'
    
    # ============================================================================
    # MAIN EXECUTION
    # ============================================================================
    
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
    print(f'Cities                 : {", ".join(cities)}')
    
    # Run complete testing pipeline with user-specific logic
    print("\n\n")
    print("▼" * 70)
    print("PROCEEDING TO COMPLETE TESTING PIPELINE")
    print("▼" * 70)
    
    test_results = run_complete_testing(
        user_type=user_type,
        geojson_path=GEOJSON_PATH,
        spatial_data_path=SPATIAL_DATA_PATH,
        output_dir=OUTPUT_DIR,
        storm_type='triangular',
        depth_mm=50.0,
        tpeak=0.5,
        mask_tif_path='manila_box_shape.tif'
    )
    
    print("Done! ✓")
