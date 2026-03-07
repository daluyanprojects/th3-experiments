import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from rasterio.transform import Affine
import geopandas as gpd
import rasterio
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import Affine
import warnings
from rasterio.warp import reproject, Resampling

from hyetograph import build_conditioning_vector
from vit import ViTFloodClassifier
from model_config import make_model
from config import DatasetConfig, ModelConfig


# ── Visualization Helpers ─────────────────────────────────────────────────────
FLOOD_COLORS = {
    0: '#FFFFFF', 
    1: '#C6DBEF',  
    2: '#6BAED6',  
    3: '#2171B5',
    4: '#08306B', 
}

FLOOD_LABELS = {0: 'No Flood', 1: 'Light', 2: 'Moderate', 3: 'Heavy', 4: 'Extreme'}
flood_cmap = mcolors.ListedColormap([FLOOD_COLORS[i] for i in range(5)])
flood_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)
legend_patches = [ mpatches.Patch(facecolor=FLOOD_COLORS[i], edgecolor='gray', linewidth=0.5, label=f'Class {i} — {FLOOD_LABELS[i]}') for i in range(5)]
warnings.filterwarnings("ignore", message="GeoSeries.notna", category=UserWarning)

def save_spatial_data(
    spatial_patches : np.ndarray,
    dem_transform   : Affine,
    dem_crs_epsg    : int,
    save_path       : str = 'checkpoints/spatial_data.npz',
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Store the 6 Affine coefficients: (a=pixel_w, b, c=left, d, e=pixel_h, f=top)
    transform_coeffs = np.array(
        [dem_transform.a, dem_transform.b, dem_transform.c,
         dem_transform.d, dem_transform.e, dem_transform.f],
        dtype=np.float64,
    )

    np.savez_compressed(
        save_path,
        spatial_patches  = spatial_patches,
        transform_coeffs = transform_coeffs,       # ← NEW: Affine coefficients
        crs_epsg         = np.array([dem_crs_epsg], dtype=np.int32),  # ← NEW
    )

    print(f"✓ Spatial data saved → {save_path}")
    print(f"  spatial_patches shape : {spatial_patches.shape}")
    print(f"  DEM transform         : {dem_transform}")
    print(f"  DEM CRS EPSG          : {dem_crs_epsg}")


# ── Inference Engine ──────────────────────────────────────────────────────────
class FloodInferenceEngine:
    def __init__(
        self,
        inference_config_path : str,
        spatial_data_path     : str,
        device                : Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # ── Load inference config ─────────────────────────────────────────────
        with open(inference_config_path) as f:
            cfg = json.load(f)

        self.rain_min         = cfg['rain_min']
        self.rain_max         = cfg['rain_max']
        self.patch_size       = cfg['patch_size']
        self.num_classes      = cfg['num_classes']
        self.input_channels   = cfg['input_channels']
        self.conditioning_dim = cfg['conditioning_dim']
        self.drain_channels   = cfg['drain_channels']
        self.soil_channels    = cfg['soil_channels']
        self.depth_min        = cfg['depth_min']
        self.depth_max        = cfg['depth_max']

        # ── Derive best model path from best_fold ─────────────────────────────
        best_fold  = cfg['best_fold']
        config_dir = Path(inference_config_path).parent
        model_path = config_dir / f'fold_{best_fold}' / 'best_model.pt'

        print(f"  Best fold   : {best_fold}")
        print(f"  Model path  : {model_path}")

        # ── Load spatial patches + DEM georeferencing ─────────────────────────
        data = np.load(spatial_data_path)
        self.spatial_patches = data['spatial_patches']   # (6400, 11, 4, 4)
        self.patches_per_map = self.spatial_patches.shape[0]

        if 'transform_coeffs' in data:
            tc = data['transform_coeffs']              
            self.dem_transform = Affine(
                tc[0], tc[1], tc[2],
                tc[3], tc[4], tc[5],
            )
            self.dem_crs_epsg = int(data['crs_epsg'][0])
            print(f"✓ DEM transform loaded from spatial_data.npz")
            print(f"  dem_transform : {self.dem_transform}")
            print(f"  dem_crs_epsg  : {self.dem_crs_epsg}")
        else:
            self.dem_transform = None
            self.dem_crs_epsg  = None
            print("⚠  WARNING: spatial_data.npz has no transform_coeffs.")
            print("   Bands 1 & 2 will NOT be geographically aligned.")
            print("   Re-run save_spatial_data() with dem_transform to fix.")

        print(f"✓ Spatial patches loaded: {self.spatial_patches.shape}")

        # ── Load model ────────────────────────────────────────────────────────
        self.model = self._load_model(model_path, cfg)
        self.model.eval()

        print(f"✓ Model loaded from: {model_path}")
        print(f"  Device : {self.device}")
        print(f"  Patches: {self.patches_per_map:,} per prediction")

    def _load_model(self, model_path: str, cfg: dict) -> ViTFloodClassifier:
        data_cfg  = DatasetConfig(
            patch_size        = cfg['patch_size'],
            num_classes       = cfg['num_classes'],
            input_channels    = cfg['input_channels'],
            conditioning_dim  = cfg['conditioning_dim'],
        )
        model_cfg = ModelConfig()

        model_cfg.embed_dim       = cfg.get('embed_dim')
        model_cfg.num_heads       = cfg.get('num_heads')
        model_cfg.num_layers      = cfg.get('num_layers')
        model_cfg.mlp_ratio       = cfg.get('mlp_ratio')
        model_cfg.dropout         = cfg.get('dropout')
        model_cfg.rainfall_hidden = cfg.get('rainfall_hidden')
        model_cfg.lr              = cfg.get('lr')
        model_cfg.weight_decay    = cfg.get('weight_decay')
        
        print(f"\n  Model hyperparameters from config:")
        print(f"    embed_dim       : {model_cfg.embed_dim}")
        print(f"    num_heads       : {model_cfg.num_heads}")
        print(f"    num_layers      : {model_cfg.num_layers}")
        print(f"    mlp_ratio       : {model_cfg.mlp_ratio}")
        print(f"    dropout         : {model_cfg.dropout}")
        print(f"    rainfall_hidden : {model_cfg.rainfall_hidden}")
        print(f"    lr              : {model_cfg.lr}")
        print(f"    weight_decay    : {model_cfg.weight_decay}")

        model = make_model(data_cfg, model_cfg)

        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device)

        print(f"  Best macro F1 (saved): {ckpt.get('best_macro_f1', 'N/A'):.4f}")
        return model

    def _apply_channel_masking(
        self,
        spatial_patch : np.ndarray,
        hasDrainage   : bool,
        hasSoil       : bool,
    ) -> np.ndarray:
        patch = spatial_patch.copy()
        if not hasDrainage:
            patch[self.drain_channels] = 0.0
        if not hasSoil:
            patch[self.soil_channels]  = 0.0
        return patch

    def _reconstruct_map(
        self,
        patch_predictions : np.ndarray,
        map_size          : Tuple[int, int] = (320, 320),
    ) -> np.ndarray:
        H, W       = map_size
        n_h        = H // self.patch_size
        n_w        = W // self.patch_size
        patch_grid = patch_predictions.reshape(n_h, n_w)
        flood_map  = np.repeat(
            np.repeat(patch_grid, self.patch_size, axis=0),
            self.patch_size, axis=1,
        )
        return flood_map

    @torch.no_grad()
    def predict(
        self,
        storm_type  : str,
        depth_mm    : float,
        hasDrainage : bool,
        hasSoil     : bool,
        tpeak       : Optional[float] = None,
        batch_size  : int = 512,
        map_size    : Tuple[int, int] = (320, 320),
    ) -> Dict:
        # ── 1. Build conditioning vector ─────────────────────────────────────
        conditioning_np, warnings = build_conditioning_vector(
            storm_type  = storm_type,
            depth_mm    = depth_mm,
            rain_min    = self.rain_min,
            rain_max    = self.rain_max,
            depth_min   = self.depth_min,
            depth_max   = self.depth_max,
            hasDrainage = hasDrainage,
            hasSoil     = hasSoil,
            tpeak       = tpeak,
        )

        if warnings:
            print("\n⚠ Input warnings:")
            for w in warnings:
                print(f"  {w}")

        # ── 2. Broadcast conditioning to all patches ──────────────────────────
        conditioning_all = np.tile(conditioning_np[np.newaxis], (self.patches_per_map, 1))

        # ── 3. Apply spatial channel masking to all patches ───────────────────
        spatial_all = np.stack([
            self._apply_channel_masking(self.spatial_patches[i], hasDrainage, hasSoil)
            for i in range(self.patches_per_map)
        ])                                                    # (6400, 11, 4, 4)

        # ── 4. Batch inference ────────────────────────────────────────────────
        patch_predictions = []

        for start in range(0, self.patches_per_map, batch_size):
            end = min(start + batch_size, self.patches_per_map)

            spatial_batch = torch.from_numpy(
                spatial_all[start:end]
            ).float().to(self.device)                        # (B, 11, 4, 4)

            cond_batch = torch.from_numpy(
                conditioning_all[start:end]
            ).float().to(self.device)                        # (B, conditioning_dim)

            logits, _ = self.model(spatial_batch, cond_batch)
            preds     = logits.argmax(dim=1).cpu().numpy()   # (B,)
            patch_predictions.append(preds)

        patch_predictions = np.concatenate(patch_predictions)  # (6400,)

        # ── 5. Reconstruct full map ───────────────────────────────────────────
        flood_map = self._reconstruct_map(patch_predictions, map_size)

        # ── 6. Summary ────────────────────────────────────────────────────────
        self._print_prediction_summary(
            flood_map, storm_type, depth_mm, tpeak, hasDrainage, hasSoil, warnings
        )

        return {
            'flood_map'  : flood_map,
            'patch_preds': patch_predictions,
            'warnings'   : warnings,
            'config'     : {
                'storm_type' : storm_type,
                'depth_mm'   : depth_mm,
                'tpeak'      : tpeak,
                'hasDrainage': hasDrainage,
                'hasSoil'    : hasSoil,
            },
        }

    def _print_prediction_summary(
        self,
        flood_map   : np.ndarray,
        storm_type  : str,
        depth_mm    : float,
        tpeak       : Optional[float],
        hasDrainage : bool,
        hasSoil     : bool,
        warnings    : list,
    ) -> None:
        total_pixels = flood_map.size
        print(f"\n{'='*60}")
        print(f"FLOOD PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Storm type  : {storm_type}")
        print(f"  Depth       : {depth_mm} mm")
        if tpeak is not None:
            print(f"  tpeak       : {tpeak}")
        print(f"  hasDrainage : {hasDrainage}")
        print(f"  hasSoil     : {hasSoil}")
        print(f"  Warnings    : {len(warnings)}")
        print(f"\n  Flood class distribution:")
        for cls_id, cls_name in FLOOD_LABELS.items():
            count = (flood_map == cls_id).sum()
            pct   = 100 * count / total_pixels
            print(f"    Class {cls_id} ({cls_name:<10}): "
                  f"{count:>7,} px  ({pct:>5.1f}%)")
        print(f"{'='*60}")




# ── Predict with Confidence ───────────────────────────────────────────────────
@torch.no_grad()
def predict_with_confidence(engine, storm_type, depth_mm, hasDrainage, hasSoil,
                             tpeak=None, batch_size=512):
    conditioning_np, warnings = build_conditioning_vector(
        storm_type  = storm_type,
        depth_mm    = depth_mm,
        rain_min    = engine.rain_min,
        rain_max    = engine.rain_max,
        depth_min   = engine.depth_min,   
        depth_max   = engine.depth_max,   
        hasDrainage = hasDrainage,
        hasSoil     = hasSoil,
        tpeak       = tpeak,
    )
    conditioning_all = np.tile(conditioning_np[np.newaxis], (engine.patches_per_map, 1))
    spatial_all = np.stack([
        engine._apply_channel_masking(engine.spatial_patches[i], hasDrainage, hasSoil)
        for i in range(engine.patches_per_map)
    ])
    patch_preds = []
    patch_confs = []
    for start in range(0, engine.patches_per_map, batch_size):
        end           = min(start + batch_size, engine.patches_per_map)
        spatial_batch = torch.from_numpy(spatial_all[start:end]).float().to(engine.device)
        cond_batch    = torch.from_numpy(conditioning_all[start:end]).float().to(engine.device)
        logits, _     = engine.model(spatial_batch, cond_batch)
        probs         = torch.softmax(logits, dim=1)
        patch_preds.append(probs.argmax(dim=1).cpu().numpy())
        patch_confs.append(probs.max(dim=1).values.cpu().numpy())
    patch_preds = np.concatenate(patch_preds)
    patch_confs = np.concatenate(patch_confs)
    return {
        'flood_map'  : engine._reconstruct_map(patch_preds),
        'conf_map'   : engine._reconstruct_map(patch_confs.astype(np.float32)),
        'patch_preds': patch_preds,
        'patch_confs': patch_confs,
        'warnings'   : warnings,
        'config'     : {
            'storm_type' : storm_type,
            'depth_mm'   : depth_mm,
            'tpeak'      : tpeak,
            'hasDrainage': hasDrainage,
            'hasSoil'    : hasSoil,
        },
    }

def visualize_result_tif(tif_path: str, save_path: str = None, figsize: tuple = (21, 6)) -> None:
    # ── EDITED: Class 0 should be transparent (alpha=0.0) ──
    flood_colors_rgba = [(1.0, 1.0, 1.0, 0.0)] + [mcolors.to_rgba(FLOOD_COLORS[i]) for i in range(1, 5)]
    flood_cmap = mcolors.ListedColormap(flood_colors_rgba)
    flood_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)
    conf_cmap  = 'RdYlGn'

    with rasterio.open(tif_path) as src:
        band1_raw = src.read(1).astype(np.int32)
        band2_raw = src.read(2).astype(np.float32) / 1000.0
        band3_raw = src.read(3) if src.count >= 3 else None
        nodata    = src.nodata
        extent    = [
            src.bounds.left, src.bounds.right,
            src.bounds.bottom, src.bounds.top,
        ]
        crs       = src.crs
        tags      = src.tags()
        b1_tags   = src.tags(1)
        b2_tags   = src.tags(2)
        b3_tags   = src.tags(3) if src.count >= 3 else {}
    
    valid_band1_2 = band1_raw != int(nodata) if nodata is not None else np.ones_like(band1_raw, dtype=bool)
    band1  = np.where(valid_band1_2, band1_raw.astype(float), np.nan)
    band2  = np.where(valid_band1_2, band2_raw, np.nan)
    
    if band3_raw is not None:
        valid_band3 = band3_raw != int(nodata) if nodata is not None else np.ones_like(band3_raw, dtype=bool)
        band3 = np.where(valid_band3, band3_raw.astype(float), np.nan)
    else:
        band3 = None
        valid_band3 = None

    # ── Print metadata ────────────────────────────────────────────────────────
    print(f"\nGeoTIFF: {tif_path}")
    print(f"  Band 1    : {b1_tags.get('description', 'Flood class')}")
    print(f"  Band 2    : {b2_tags.get('description', 'Confidence')}")
    if band3 is not None:
        print(f"  Band 3    : {b3_tags.get('description', 'Barangay PSGC Code')}")
    print(f"  Storm     : {tags.get('storm_type','?')}  {tags.get('depth_mm','?')}mm  tpeak={tags.get('tpeak','?')}")

    # ── Figure ────────────────────────────────────────────────────────────────
    n_cols = 3 if band3 is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # ── Band 1: Flood class ───────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.imshow(band1, cmap=flood_cmap, norm=flood_norm,
               extent=extent, interpolation='nearest', origin='upper')
    tpeak_str = tags.get('tpeak', 'None')
    title1 = (f"Band 1 — Flood Hazard Class\n"
              f"Storm: {tags.get('storm_type','?')}  |  Depth: {tags.get('depth_mm','?')} mm"
              + (f"  |  tpeak: {tpeak_str}" if tpeak_str not in ('None', '', 'none') else ''))
    ax1.set_title(title1, fontsize=9, fontweight='bold', pad=6)
    ax1.set_xlabel('Longitude', fontsize=8)
    ax1.set_ylabel('Latitude', fontsize=8)
    ax1.tick_params(labelsize=7)
    legend_patches = [
        mpatches.Patch(facecolor=FLOOD_COLORS[i], edgecolor='gray',
                       linewidth=0.5, label=f'Class {i} — {FLOOD_LABELS[i]}')
        for i in range(5)
    ]
    ax1.legend(handles=legend_patches, loc='lower left', fontsize=7,
               framealpha=0.85, title='Flood Class', title_fontsize=7)
    total = int(np.sum(valid_band1_2))
    if total > 0:
        dist = '\n'.join([
            f"C{i} {FLOOD_LABELS[i]}: {100*np.nansum(band1==i)/total:.1f}%"
            for i in range(5)
        ])
    else:
        dist = "No valid data"
    ax1.text(
        0.98, 0.98, dist,
        transform=ax1.transAxes, ha='right', va='top',
        fontsize=6.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='lightgray', alpha=0.88),
    )

    # ── Band 2: Confidence ────────────────────────────────────────────────────
    ax2  = axes[1]
    im2  = ax2.imshow(
        band2, cmap=conf_cmap, vmin=0.0, vmax=1.0,
        extent=extent, interpolation='nearest', origin='upper',
    )
    ax2.set_title(
        f"Band 2 — Model Confidence\n"
        f"Drainage: {'✓' if tags.get('hasDrainage')=='True' else '✗'}  |  "
        f"Soil: {'✓' if tags.get('hasSoil')=='True' else '✗'}",
        fontsize=9, fontweight='bold', pad=6,
    )
    ax2.set_xlabel('Longitude', fontsize=8)
    ax2.set_ylabel('Latitude',  fontsize=8)
    ax2.tick_params(labelsize=7)
    cbar = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04)
    cbar.set_label('Confidence', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.axhline(y=0.5, color='black', linewidth=1.2, linestyle='--')
    cbar.ax.text(2.3, 0.5, 'uncertain\nthreshold',
                 fontsize=6, va='center', color='black')
    uncertain_overlay = np.where(~np.isnan(band2) & (band2 < 0.5), 1.0, np.nan).astype(np.float32)
    ax2.imshow(uncertain_overlay, cmap='cool', alpha=0.30,
               vmin=0, vmax=1, extent=extent,
               interpolation='nearest', origin='upper')
    frac_low = (band2 < 0.5).mean() * 100
    stats    = (
        f"Mean : {band2.mean():.3f}\n"
        f"Std  : {band2.std():.3f}\n"
        f"Min  : {band2.min():.3f}\n"
        f"Max  : {band2.max():.3f}\n"
        f"Uncertain: {frac_low:.1f}%"
    )
    ax2.text(
        0.98, 0.98, stats,
        transform=ax2.transAxes, ha='right', va='top',
        fontsize=6.5,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='lightgray', alpha=0.88),
    )

    # ── Band 3: Barangay PSGC Code ────────────────────────────────────────────
    if band3 is not None:
        ax3 = axes[2]
        bar_masked   = np.ma.masked_invalid(np.where((band3 > 0) & ~np.isnan(band3), band3, np.nan))
        unique_codes = np.unique(band3_raw[(band3_raw > 0) & (band3_raw != int(nodata))])
        n_unique     = max(len(unique_codes), 1)
        bar_cmap     = plt.cm.get_cmap('tab20', n_unique)
        bar_cmap.set_bad(color='white')
        im3 = ax3.imshow(
            bar_masked, cmap=bar_cmap,
            extent=extent, interpolation='nearest', origin='upper',
        )
        ax3.set_title(
            f"Band 3 — Barangay Boundaries\nPSGC codes  |  {n_unique} barangay(s) in extent",
            fontsize=9, fontweight='bold', pad=6,
        )
        ax3.set_xlabel('Longitude', fontsize=8)
        ax3.set_ylabel('Latitude',  fontsize=8)
        ax3.tick_params(labelsize=7)
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.035, pad=0.04)
        cbar3.set_label('PSGC Code', fontsize=8)
        cbar3.ax.tick_params(labelsize=7)
        bar_stats = (
            f"Barangays : {n_unique}\n"
            f"Min code  : {int(unique_codes.min()) if n_unique else 'N/A'}\n"
            f"Max code  : {int(unique_codes.max()) if n_unique else 'N/A'}\n"
            f"Coverage  : {100*(band3_raw>0).mean():.1f}%"
        )
        ax3.text(
            0.98, 0.98, bar_stats,
            transform=ax3.transAxes, ha='right', va='top',
            fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.88),
        )

    fig.suptitle(
        f'EXP-DES-2 — GeoTIFF Result Visualization\n{tif_path}',
        fontsize=10, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Visualization saved → {save_path}")

    plt.show()

def _apply_mask_to_maps(
    flood_map: np.ndarray,
    conf_map: np.ndarray,
    barangay_map: np.ndarray,
    mask_tif_path: str | Path,
    dem_transform,
    dem_crs,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    map_shape = flood_map.shape
    
    # Load mask GeoTIFF
    with rasterio.open(mask_tif_path) as mask_src:
        mask_crs = mask_src.crs
        mask_transform = mask_src.transform
        mask_shape = mask_src.shape
        mask_raw = mask_src.read(1)
    
    # Check if reprojection/resampling needed
    if dem_crs != mask_crs or mask_shape != map_shape:
        if verbose:
            print(f"  ⚠ Mask alignment: CRS={dem_crs != mask_crs}, Shape={mask_shape != map_shape}")
            print(f"    → Reprojecting mask to match prediction...")
        
        # Reproject mask to prediction CRS and shape
        mask_aligned = np.zeros(map_shape, dtype=mask_raw.dtype)
        reproject(
            mask_raw,
            mask_aligned,
            src_transform=mask_transform,
            src_crs=mask_crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.nearest,
        )
    else:
        mask_aligned = mask_raw
        if verbose:
            print(f"  ✓ Mask perfectly aligned (same CRS and shape)")
    
    # Apply mask: pixels outside mask (value=0) → set to -1 (nodata)
    mask_binary = (mask_aligned > 0).astype(np.uint8)
    outside_mask = mask_binary == 0
    
    flood_map_masked = flood_map.copy()
    conf_map_masked = conf_map.copy()
    barangay_map_masked = barangay_map.copy()
    
    flood_map_masked[outside_mask] = -1
    conf_map_masked[outside_mask] = -1
    barangay_map_masked[outside_mask] = -1
    
    if verbose:
        n_inside = (mask_binary > 0).sum()
        n_total = mask_binary.size
        print(f"  Mask applied: {n_inside:,} / {n_total:,} pixels inside")
    
    return flood_map_masked, conf_map_masked, barangay_map_masked

def save_result_as_tif(result, save_path, transform, crs, map_size=320, barangay_band=None, mask_tif_path: Optional[str | Path] = None):
    flood_map = result['flood_map'].astype(np.int32)
    H, W      = flood_map.shape  
    conf_map  = result['conf_map'].astype(np.float32)
    bar_band = barangay_band if barangay_band is not None else np.zeros((H, W), dtype=np.int32)

    print(f"  Applying mask from {Path(mask_tif_path).name}...")
    flood_map, conf_map, bar_band = _apply_mask_to_maps(
        flood_map, conf_map, bar_band,
        mask_tif_path, transform, crs,
        verbose=True
    )

    b1 = flood_map.copy()
    b1[b1 == 0] = -1
    b2 = (conf_map * 1000).astype(np.int32)
    b3 = bar_band.astype(np.int32)

    with rasterio.open(
        save_path, 'w',
        driver    = 'GTiff',
        height    = H, width = W,
        count     = 3,
        dtype     = 'int32',
        crs       = crs,
        transform = transform,
        nodata    = -1,
        compress  = 'lzw',
    ) as dst:
        dst.write(b1, 1)   # flood class (0→-1 for transparency)
        dst.write(b2, 2)   # confidence ×1000
        dst.write(b3, 3)   # PSGC code
        
        cfg_meta = result.get('config', {})
        dst.update_tags(
            1,
            description='Flood class (0=Transparent 1=Light 2=Moderate 3=Heavy 4=Extreme)',
        )
        dst.update_tags(
            2,
            description='Confidence x1000 — divide by 1000 for 0.0-1.0',
        )
        dst.update_tags(
            3,
            description='Barangay PSGC Code (-1=outside Manila extent)',
        )
        dst.update_tags(
            model       = 'FloodInferenceEngine ViTFloodClassifier',
            storm_type  = str(cfg_meta.get('storm_type',  '')),
            depth_mm    = str(cfg_meta.get('depth_mm',    '')),
            tpeak       = str(cfg_meta.get('tpeak',       '')),
            hasDrainage = str(cfg_meta.get('hasDrainage', '')),
            hasSoil     = str(cfg_meta.get('hasSoil',     '')),
        )

    print(f"✓ Saved → {save_path}")
    
def build_subtitle(cfg: dict) -> str:
    parts = [f"Storm: {cfg['storm_type']}", f"Depth: {cfg['depth_mm']} mm"]
    if cfg['tpeak'] is not None:
        parts.append(f"tpeak: {cfg['tpeak']}")
    parts.append(f"Drainage: {'✓' if cfg['hasDrainage'] else '✗'}")
    parts.append(f"Soil: {'✓' if cfg['hasSoil'] else '✗'}")
    return '  |  '.join(parts)

def compute_class_pcts(flood_map):
    total = flood_map.size
    return {i: 100 * (flood_map == i).sum() / total for i in range(5)}

def add_warning_banner(ax, warnings):
    if warnings:
        ax.text(
            0.5, 0.02,
            f'⚠ {len(warnings)} warning(s) — inputs outside training range',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=7.5, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#B8860B',
                      edgecolor='none', alpha=0.88),
        )

def _load_barangay_band_from_geojson(geojson_path, dst_transform, dst_crs, H, W):
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(dst_crs)
    gdf = gdf.dropna(subset=['psgc_code']).copy()

    # Drop null / empty / invalid geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna() & gdf.geometry.is_valid]

    gdf['psgc_int'] = gdf['psgc_code'].astype(float).astype(int)

    # Clip to DEM grid extent
    left   = dst_transform.c
    top    = dst_transform.f
    right  = left + dst_transform.a * W
    bottom = top  + dst_transform.e * H
    grid_box = box(left, min(top, bottom), right, max(top, bottom))
    gdf = gdf[gdf.geometry.intersects(grid_box)]

    print(f"  Rasterizing {len(gdf)} barangays within DEM extent ...")

    shapes = [
        (geom, psgc)
        for geom, psgc in zip(gdf.geometry, gdf['psgc_int'])
        if geom is not None and not geom.is_empty
    ]

    band = rasterize(
        shapes,
        out_shape   = (H, W),
        transform   = dst_transform,
        fill        = 0,
        dtype       = 'int32',
        all_touched = True,
    )
    return band


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY — 4-pass per storm type
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    dict(label='Full (Drainage + Soil)',  hasDrainage=True,  hasSoil=True),
    dict(label='Drainage only',           hasDrainage=True,  hasSoil=False),
    dict(label='Soil only',               hasDrainage=False, hasSoil=True),
    dict(label='Neither (minimal)',       hasDrainage=False, hasSoil=False),
]

STORM_CONFIGS = [
    dict(storm_type='triangular',   depth_mm=9,  tpeak=0.5,  storm_label='Triangular | 9mm | tpeak=0.5'),
    dict(storm_type='front-loaded', depth_mm=34, tpeak=None, storm_label='Front-loaded | 34mm'),
    dict(storm_type='balanced',     depth_mm=31, tpeak=None, storm_label='Balanced | 31mm'),
    dict(storm_type='back-loaded',  depth_mm=35, tpeak=None, storm_label='Back-loaded | 35mm'),
]

def run_ablation_study(engine: 'FloodInferenceEngine') -> dict:
    all_results = {}
    for sc in STORM_CONFIGS:
        all_results[sc['storm_label']] = {}
        for ab in ABLATION_CONFIGS:
            all_results[sc['storm_label']][ab['label']] = predict_with_confidence(
                engine,
                storm_type  = sc['storm_type'],
                depth_mm    = sc['depth_mm'],
                tpeak       = sc['tpeak'],
                hasDrainage = ab['hasDrainage'],
                hasSoil     = ab['hasSoil'],
            )
    return all_results


def plot_4pass_flood(storm_label: str, all_results: dict, cfg, save_name: str) -> None:
    fig, axes = plt.subplots(
        4, 2, figsize=(13, 20),
        gridspec_kw={'width_ratios': [2.5, 1], 'hspace': 0.40, 'wspace': 0.25},
    )
    for row, ab in enumerate(ABLATION_CONFIGS):
        result   = all_results[storm_label][ab['label']]
        ax_flood = axes[row, 0]
        ax_bar   = axes[row, 1]
        pcts     = compute_class_pcts(result['flood_map'])

        ax_flood.imshow(result['flood_map'], cmap=flood_cmap, norm=flood_norm,
                        interpolation='nearest')
        ax_flood.set_title(
            f"{ab['label']}\n{build_subtitle(result['config'])}",
            fontsize=9, fontweight='bold', pad=5)
        ax_flood.axis('off')
        add_warning_banner(ax_flood, result['warnings'])

        bar_labels = [FLOOD_LABELS[i] for i in range(5)]
        values     = [pcts[i] for i in range(5)]
        colors     = [FLOOD_COLORS[i] for i in range(5)]
        bars       = ax_bar.barh(bar_labels, values, color=colors,
                                 edgecolor='gray', linewidth=0.5, height=0.6)
        for bar, val in zip(bars, values):
            if val > 1.5:
                ax_bar.text(bar.get_width() - 0.5,
                            bar.get_y() + bar.get_height() / 2,
                            f'{val:.1f}%', va='center', ha='right',
                            fontsize=8, color='white', fontweight='bold')
            elif val > 0.1:
                ax_bar.text(bar.get_width() + 0.5,
                            bar.get_y() + bar.get_height() / 2,
                            f'{val:.1f}%', va='center', ha='left',
                            fontsize=8, color='#333333')
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel('Coverage (%)', fontsize=8)
        ax_bar.set_title('Class\nDistribution', fontsize=8.5,
                         fontweight='bold', pad=5)
        ax_bar.tick_params(axis='both', labelsize=8)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.invert_yaxis()

    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=9,
               frameon=True, title='Flood Severity Classes', title_fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f'EXP-DES-2 — 4-Pass Ablation | {storm_label}\nManila Core',
        fontsize=12, fontweight='bold', y=1.01)
    path = str(cfg.train.output_dir / save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved → {path}')


def plot_4pass_confidence(storm_label: str, all_results: dict, cfg, save_name: str) -> None:
    fig, axes = plt.subplots(
        4, 2, figsize=(13, 20),
        gridspec_kw={'width_ratios': [2.5, 1], 'hspace': 0.40, 'wspace': 0.30},
    )
    conf_cmap = 'RdYlGn'
    for row, ab in enumerate(ABLATION_CONFIGS):
        result      = all_results[storm_label][ab['label']]
        ax_conf     = axes[row, 0]
        ax_hist     = axes[row, 1]
        conf_map    = result['conf_map']
        patch_confs = result['patch_confs']
        frac_low    = (patch_confs < 0.5).mean() * 100

        im = ax_conf.imshow(conf_map, cmap=conf_cmap, vmin=0.0, vmax=1.0,
                            interpolation='nearest')
        ax_conf.set_title(
            f"{ab['label']}\n{build_subtitle(result['config'])}\n"
            f"Model Confidence (softmax max-probability per patch)",
            fontsize=8.5, fontweight='bold', pad=5)
        ax_conf.axis('off')
        add_warning_banner(ax_conf, result['warnings'])

        cbar = plt.colorbar(im, ax=ax_conf, fraction=0.035, pad=0.03)
        cbar.set_label('Confidence', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.axhline(y=0.5, color='black', linewidth=1.2, linestyle='--')
        cbar.ax.text(2.3, 0.5, 'uncertain\nthreshold', fontsize=6,
                     va='center', color='black')

        uncertain_overlay = conf_map.copy().astype(float)
        uncertain_overlay[conf_map >= 0.5] = np.nan
        uncertain_overlay[conf_map <  0.5] = 1.0
        ax_conf.imshow(uncertain_overlay, cmap='cool', alpha=0.30,
                       vmin=0, vmax=1, interpolation='nearest')
        ax_conf.text(0.02, 0.98, f'{frac_low:.1f}% uncertain\n(conf < 0.5)',
                     transform=ax_conf.transAxes, ha='left', va='top',
                     fontsize=7.5,
                     color='red' if frac_low > 20 else 'gray',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                               edgecolor='lightgray', alpha=0.85))

        ax_hist.hist(patch_confs, bins=20, range=(0, 1), color='steelblue',
                     edgecolor='white', linewidth=0.4, alpha=0.85)
        ax_hist.axvline(x=0.5, color='red', linewidth=1.2, linestyle='--',
                        label='Uncertain (0.5)')
        ax_hist.axvline(x=patch_confs.mean(), color='orange', linewidth=1.2,
                        linestyle='-', label=f"Mean ({patch_confs.mean():.2f})")
        ax_hist.set_xlabel('Confidence', fontsize=8)
        ax_hist.set_ylabel('Patches', fontsize=8)
        ax_hist.set_title('Confidence\nDistribution', fontsize=8.5,
                          fontweight='bold', pad=5)
        ax_hist.tick_params(labelsize=7)
        ax_hist.legend(fontsize=6.5, loc='upper left')
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.text(0.97, 0.97, f'{frac_low:.1f}%\nuncertain',
                     transform=ax_hist.transAxes, ha='right', va='top',
                     fontsize=7.5,
                     color='red' if frac_low > 20 else 'gray',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                               edgecolor='lightgray', alpha=0.85))

    fig.suptitle(
        f'EXP-DES-2 — Confidence Maps | {storm_label}\nManila Core',
        fontsize=12, fontweight='bold', y=1.01)
    path = str(cfg.train.output_dir / save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved → {path}')


def run_inference(
    engine       : 'FloodInferenceEngine',
    storm_type   : str,
    depth_mm     : float,
    hasDrainage  : bool,
    hasSoil      : bool,
    output_dir   : str,
    tpeak        : Optional[float]  = None,
    geojson_path : Optional[str]    = None,
    stem         : Optional[str]    = None,
    figsize      : tuple            = (21, 6),
    mask_tif_path: Optional[str]    = None
) -> dict:
    
    import rasterio.crs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Auto-generate a readable filename stem ────────────────────────────────
    if stem is None:
        drain_tag = 'drain' if hasDrainage else 'nodrain'
        soil_tag  = 'soil'  if hasSoil     else 'nosoil'
        tpeak_tag = f'_tp{tpeak}'.replace('.', '') if tpeak is not None else ''
        stem = f'{storm_type}_{int(depth_mm)}mm{tpeak_tag}_{drain_tag}_{soil_tag}'

    tif_path = str(output_dir / f'{stem}.tif')
    viz_path = str(output_dir / f'{stem}_viz.png')

    # ── 1. Predict ────────────────────────────────────────────────────────────
    print(f'\n[run_inference] {stem}')
    result = predict_with_confidence(
        engine,
        storm_type  = storm_type,
        depth_mm    = depth_mm,
        hasDrainage = hasDrainage,
        hasSoil     = hasSoil,
        tpeak       = tpeak,
    )

    # ── 2. Build / load barangay band ─────────────────────────────────────────
    H, W = result['flood_map'].shape
    dem_crs = rasterio.crs.CRS.from_epsg(engine.dem_crs_epsg)

    if geojson_path is not None:
        barangay_band = _load_barangay_band_from_geojson(
            geojson_path, engine.dem_transform, dem_crs, H, W
        )
    else:
        barangay_band = np.zeros((H, W), dtype=np.int32)

    # ── 3. Save 3-band GeoTIFF ────────────────────────────────────────────────
    save_result_as_tif(
        result        = result,
        save_path     = tif_path,
        transform     = engine.dem_transform,
        crs           = dem_crs,
        barangay_band = barangay_band,
        mask_tif_path = mask_tif_path
    )

    # ── 4. Visualize ──────────────────────────────────────────────────────────
    visualize_result_tif(
        tif_path  = tif_path,
        save_path = viz_path,
        figsize   = figsize,
    )

    result['tif_path'] = tif_path
    result['viz_path'] = viz_path
    print(f'✓ Done  →  {tif_path}')
    print(f'          {viz_path}')
    return result
