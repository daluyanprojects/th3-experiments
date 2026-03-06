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
            fold_scores[ckpt_path] = {'value': f1_score}
        except Exception as e:
            print(f"    ⚠ Error loading {ckpt_path.name}: {e}")
            fold_scores[ckpt_path] = {'value': float('-inf')}
    
    # Select checkpoint with highest F1 score
    best_checkpoint_path = max(fold_scores.keys(), key=lambda k: fold_scores[k]['value'])
    return best_checkpoint_path, fold_scores


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
    
    is_pedestrian = user_type.lower() == 'pedestrian'
    
    # Import correct module based on user type
    if is_pedestrian:
        from inference_ped import InferenceEngine, run_inference
        output_dir_param = 'output_ped_dir'
        checkpoint_dir = Path('outputs_ped/checkpoints_tuned')
        config_path = Path('outputs_ped/logs_tuned/config_tuned.json')
        model_type = "PEDESTRIAN"
    else:  # vehicle
        from inference import InferenceEngine, run_inference
        output_dir_param = 'output_dir'
        checkpoint_dir = Path('outputs/checkpoints_tuned')
        config_path = Path('outputs/logs_tuned/config_tuned.json')
        model_type = "VEHICLE"
    
    print("\n" + "="*70)
    print(f"FLOOD PREDICTION MODEL - {model_type} MODE")
    print("="*70)
    
    # Setup transforms and visualization
    flood_viz = setup_flood_visualization()
    geotiff_setup = setup_geotiff_transform()
    
    # ✓ Load tuned configuration
    print("\n[1] LOADING TUNED CONFIGURATION")
    print("-" * 70)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    cfg_tuned = TrainConfig.load(str(config_path))
    print(f"Config file: {config_path}")
    print(f"✓ Config loaded successfully")
    print(f"\n  Model Architecture Parameters:")
    print(f"    • embed_dim          : {cfg_tuned.embed_dim}")
    print(f"    • num_layers         : {cfg_tuned.num_layers}")
    print(f"    • num_heads          : {getattr(cfg_tuned, 'num_heads', 'N/A')}")
    print(f"    • rainfall_hidden    : {cfg_tuned.rainfall_hidden}")
    print(f"    • mlp_ratio          : {getattr(cfg_tuned, 'mlp_ratio', 'N/A')}")
    print(f"    • dropout            : {getattr(cfg_tuned, 'dropout', 'N/A')}")
    print(f"\n  Training Parameters:")
    print(f"    • batch_size         : {cfg_tuned.batch_size}")
    print(f"    • learning_rate      : {getattr(cfg_tuned, 'learning_rate', 'N/A')}")
    print(f"    • weight_decay       : {getattr(cfg_tuned, 'weight_decay', 'N/A')}")
    print(f"    • num_epochs         : {getattr(cfg_tuned, 'num_epochs', 'N/A')}")
    print(f"    • checkpoint_metric  : {cfg_tuned.checkpoint_metric}")
    
    # ✓ Find BEST checkpoint from tuned checkpoints directory
    print("\n[2] LOADING CHECKPOINT")
    print("-" * 70)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Select best checkpoint across all folds
    checkpoint_path, fold_scores = select_best_checkpoint(checkpoint_dir)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"✓ Selected BEST checkpoint: {checkpoint_path.name}")
    print(f"  Total folds evaluated: {len(fold_scores)}")
    
    # Print comparison of all folds
    print(f"\n  Fold Performance Ranking (by F1 Score):")
    sorted_folds = sorted(fold_scores.items(), 
                         key=lambda x: x[1]['value'], 
                         reverse=True)
    for rank, (path, score_info) in enumerate(sorted_folds, 1):
        is_best = "★ BEST" if path == checkpoint_path else "     "
        print(f"    {rank}. {is_best} {path.name:30s} | F1: {score_info['value']:.6f}")
    
    # Load and display best checkpoint metadata
    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    f1_score = ckpt.get('best_monitor', 'N/A')
    print(f"\n  Best Checkpoint Details:")
    print(f"    • F1 Score           : {f1_score}")
    print(f"    • keys in checkpoint : {list(ckpt.keys())}")
    
    # Load barangay boundaries
    print("\n[3] LOADING BARANGAY BOUNDARIES")
    print("-" * 70)
    print(f"GeoJSON file: {geojson_path}")
    barangay_band = load_barangay_band(
        geojson_path,
        geotiff_setup['transform'],
        geotiff_setup['crs'],
        geotiff_setup['height'],
        geotiff_setup['width']
    )
    print(f"✓ Barangay grid shape: {barangay_band.shape}")
    print(f"  Unique barangays: {len(np.unique(barangay_band)) - 1}")  # -1 for fill value
    
    # Initialize inference engine with tuned config and checkpoint
    print("\n[4] INITIALIZING INFERENCE ENGINE")
    print("-" * 70)
    print(f"Initializing with:")
    print(f"  • checkpoint_path    : {checkpoint_path}")
    print(f"  • spatial_data_path  : {spatial_data_path}")
    print(f"  • config             : tuned (embed_dim={cfg_tuned.embed_dim})")
    
    engine = InferenceEngine(
        checkpoint_path=str(checkpoint_path),
        patch_data_path=spatial_data_path,
        config=cfg_tuned  # ✓ Pass tuned config!
    )
    
    # Print model information
    print("\n[5] MODEL INFORMATION")
    print("-" * 70)
    print(f"Model: {engine.model.__class__.__name__}")
    print(f"Device: {engine.model.device if hasattr(engine.model, 'device') else engine.device}")
    print(f"Mode: {'eval' if not engine.model.training else 'train'}")
    
    # Count parameters
    total_params = sum(p.numel() for p in engine.model.parameters())
    trainable_params = sum(p.numel() for p in engine.model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  • Total parameters  : {total_params:,}")
    print(f"  • Trainable params  : {trainable_params:,}")
    print(f"  • Model size        : {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Print spatial data info
    print(f"\nSpatial Data:")
    print(f"  • Patches loaded    : {engine.test_spatial.shape[0]:,}")
    print(f"  • Patch shape       : {engine.test_spatial.shape[1:]}")
    print(f"  • Map shape         : {engine.test_dem.shape}")
    print(f"  • Batch size        : {engine.batch_size}")
    
    # Run inference
    print("\n[6] RUNNING INFERENCE")
    print("-" * 70)
    print(f"Storm parameters:")
    print(f"  • Storm type        : {storm_type.capitalize()}")
    print(f"  • Rainfall depth    : {depth_mm} mm")
    print(f"  • Peak time (tpeak) : {tpeak}")
    print(f"  • Patches to process: {engine.test_spatial.shape[0]:,}")
    
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
    
    # Print results summary
    print("\n[7] RESULTS SUMMARY")
    print("-" * 70)
    print(f"Output file: {results['tif_path']}")
    print(f"\nPrediction Statistics:")
    print(f"  • Total patches     : {results['total_patches']:,}")
    print(f"  • Flooded patches   : {results['flooded_patches']:,} ({results['flooded_pct']:.1f}%)")
    print(f"  • Mean confidence   : {results['mean_confidence']:.4f}")
    print(f"  • Dominant class    : {results['dominant_class']} ({['No Flood', 'Light', 'Moderate', 'Heavy', 'Extreme'][results['dominant_class']]})")
    
    print(f"\nClass Distribution:")
    for cls, dist in results['class_distribution'].items():
        print(f"  • {dist['name']:12s}: {dist['count']:6,} patches ({dist['pct']:5.1f}%) - "
              f"mean conf: {dist['mean_confidence']:.4f}")
    
    print("\n" + "="*70)
    print("✓ INFERENCE COMPLETE")
    print("="*70 + "\n")
    
    return results


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
    
    # Run flood prediction with tuned model
    results = run_flood_prediction(
        user_type=user_type,
        geojson_path=GEOJSON_PATH,
        spatial_data_path=SPATIAL_DATA_PATH,
        output_dir=OUTPUT_DIR,
        storm_type='triangular',
        depth_mm=50.0,
        tpeak=0.5
    )
    
    print("\n" + "="*60)
    print("Done! ✓")
    print("="*60)