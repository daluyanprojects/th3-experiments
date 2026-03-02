import warnings as _warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from typing import Optional
import matplotlib.patches as mpatches, matplotlib.colors as mcolors
import rasterio
import numpy as np, torch, matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from shapely.geometry import box
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from typing import Optional, Dict, List, Tuple

from hyetograph import build_inference_inputs
from config import TrainConfig
from training import make_model

warnings.filterwarnings("ignore", message="GeoSeries.notna", category=UserWarning)


# ── Flood class metadata ───────────────────────────────────────────────────────
FLOOD_CLASSES = {
    0: 'No Flood',
    1: 'Light',
    2: 'Moderate',
    3: 'Heavy',
    4: 'Extreme',
}

PATCH_SIZE  = 4
MAP_SHAPE   = (1152, 1152)

# ── Engine dataclass ───────────────────────────────────────────────────────────
@dataclass
class InferenceEngine:
    checkpoint_path : Optional[str | Path] = None
    patch_data_path : Optional[str | Path] = None
    device_str      : Optional[str]        = None

    # Populated during __post_init__
    model           : torch.nn.Module      = field(init=False, repr=False)
    test_spatial    : np.ndarray           = field(init=False, repr=False)
    patch_indices   : np.ndarray           = field(init=False, repr=False)
    test_dem        : np.ndarray           = field(init=False, repr=False)
    cfg             : TrainConfig          = field(init=False, repr=False)
    device          : torch.device         = field(init=False, repr=False)
    batch_size      : int                  = field(init=False)

    def __post_init__(self):
        self.cfg    = TrainConfig()
        self.cfg.use_conditioning = True
        self.device = torch.device(
            self.device_str if self.device_str
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        print(f"[InferenceEngine] device = {self.device}")
        self._load_patches()
        self._load_model()
        self.batch_size = self.cfg.batch_size

    # ── Patch loading ──────────────────────────────────────────────────────────
    def _load_patches(self):
        path = Path(self.patch_data_path) if self.patch_data_path \
               else self.cfg.output_dir / 'test_patches.npz'

        if not path.exists():
            raise FileNotFoundError(
                f"Test patch file not found: {path}\n"
                "  Run save_test_patches.py after training to generate it."
            )

        data = np.load(path)
        self.test_spatial  = data['spatial'].astype(np.float32)   # (N, 3, 4, 4)
        self.patch_indices = data['patch_indices'].astype(np.int32) # (N, 2)
        self.test_dem      = data['dem'].astype(np.float32)         # (1152, 1152)

        N = self.test_spatial.shape[0]
        print(f"[InferenceEngine] Loaded {N:,} Manila patches  "
              f"| spatial {self.test_spatial.shape}  "
              f"| dem {self.test_dem.shape}")

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self):
        self.model = make_model(self.cfg).to(self.device)

        # Resolve checkpoint path
        if self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)
        else:
            ckpt_dir   = self.cfg.output_dir / 'checkpoints'
            ckpt_files = sorted(ckpt_dir.glob('fold_*_best.pth'))
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

            best_score, ckpt_path = -np.inf, ckpt_files[0]
            for f in ckpt_files:
                ckpt = torch.load(f, map_location='cpu', weights_only=False)
                if ckpt.get('best_monitor', -np.inf) > best_score:
                    best_score, ckpt_path = ckpt['best_monitor'], f

            print(f"[InferenceEngine] Auto-selected: {ckpt_path.name}  "
                  f"({self.cfg.checkpoint_metric}={best_score:.4f})")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f"[InferenceEngine] Model loaded  ← {ckpt_path.name}")


# ── Core inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def _run_forward(
    engine      : InferenceEngine,
    rainfall_seq: np.ndarray,        # (13,)
    conditioning: np.ndarray,        # (4,)
) -> tuple[np.ndarray, np.ndarray]:
    
    N = engine.test_spatial.shape[0]

    # Tile scalar inputs across all patches
    rainfall_tiled     = np.tile(rainfall_seq,  (N, 1)).astype(np.float32)  # (N,13)
    conditioning_tiled = np.tile(conditioning,  (N, 1)).astype(np.float32)  # (N,4)

    ds = TensorDataset(
        torch.from_numpy(engine.test_spatial),
        torch.from_numpy(rainfall_tiled),
        torch.from_numpy(conditioning_tiled),
    )
    dl = DataLoader(
        ds,
        batch_size = engine.batch_size,
        shuffle    = False,
        num_workers= 2,
        pin_memory = engine.device.type == 'cuda',
    )

    all_preds, all_probs = [], []
    for sp_b, rf_b, cd_b in dl:
        logits, _ = engine.model(
            sp_b.to(engine.device),
            rf_b.to(engine.device),
            cd_b.to(engine.device),
        )    
        probs  = F.softmax(logits, dim=1).cpu().numpy()              # (B, 5)
        preds  = probs.argmax(axis=1)                   # (B,)
        all_probs.append(probs)
        all_preds.append(preds)

    predictions   = np.concatenate(all_preds)           # (N,)
    probabilities = np.concatenate(all_probs, axis=0)   # (N, 5)
    return predictions, probabilities


# ── Spatial reconstruction ─────────────────────────────────────────────────────
def _reconstruct_map(
    predictions   : np.ndarray,   # (N,)
    patch_indices : np.ndarray,   # (N, 2)
    map_shape     : tuple = MAP_SHAPE,
    patch_size    : int   = PATCH_SIZE,
) -> np.ndarray:
    """Fill a (H, W) map with predicted class per patch. -1 = outside Manila mask."""
    flood_map = np.full(map_shape, fill_value=-1, dtype=np.int8)
    for pred, (r, c) in zip(predictions, patch_indices):
        flood_map[r:r + patch_size, c:c + patch_size] = pred
    return flood_map


# ── Summary stats ──────────────────────────────────────────────────────────────
def _build_summary(
    predictions  : np.ndarray,    # (N,)
    probabilities: np.ndarray,    # (N, 5)
) -> dict:
    total = len(predictions)
    class_dist = {}
    for cls, name in FLOOD_CLASSES.items():
        mask  = predictions == cls
        count = int(mask.sum())
        class_dist[cls] = {
            'name'           : name,
            'count'          : count,
            'pct'            : round(100.0 * count / total, 2) if total > 0 else 0.0,
            'mean_confidence': round(float(probabilities[mask, cls].mean()), 4)
                               if count > 0 else 0.0,
        }
    flooded_patches = int((predictions > 0).sum())
    return {
        'total_patches'    : total,
        'flooded_patches'  : flooded_patches,
        'flooded_pct'      : round(100.0 * flooded_patches / total, 2),
        'dominant_class'   : int(np.bincount(predictions).argmax()),
        'mean_confidence'  : round(float(probabilities.max(axis=1).mean()), 4),
        'class_distribution': class_dist,
    }


def _storm_label(storm_type: str, depth_mm: float, tpeak: Optional[float]) -> str:
    label = f"{storm_type.title()}, {depth_mm} mm"
    if tpeak is not None and storm_type == 'triangular':
        label += f", tpeak={tpeak}"
    return label


def predict_with_confidence(
    engine    : InferenceEngine,
    storm_type: str,
    depth_mm  : float,
    tpeak     : Optional[float] = None,
    verbose   : bool            = True,
) -> dict:
    
    # 1. Build hyetograph + conditioning vector
    rainfall_seq, conditioning, warn_list = build_inference_inputs(
        storm_type, depth_mm, tpeak
    )

    # Surface OOD warnings
    for w in warn_list:
        _warnings.warn(f"[OOD] {w}", stacklevel=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Predicting: {_storm_label(storm_type, depth_mm, tpeak)}")
        print(f"{'='*60}")
        print(f"  Conditioning: pattern_type={conditioning[0]:.0f}  "
              f"depth_norm={conditioning[1]:.3f}  "
              f"tpeak={conditioning[2]:.2f}  "
              f"has_tpeak={conditioning[3]:.0f}")
        if warn_list:
            for w in warn_list:
                print(f"  ⚠  {w}")

    # 2. Forward pass
    predictions, probabilities = _run_forward(engine, rainfall_seq, conditioning)
    confidence = probabilities.max(axis=1)   # (N,)

    # 3. Reconstruct spatial map
    flood_map = _reconstruct_map(predictions, engine.patch_indices)

    # 4. Summary
    summary = _build_summary(predictions, probabilities)

    if verbose:
        print(f"\n  Flooded area : {summary['flooded_pct']:.1f}%  "
              f"({summary['flooded_patches']:,} / {summary['total_patches']:,} patches)")
        print(f"  Mean confidence : {summary['mean_confidence']:.3f}")
        print(f"\n  {'Class':<12} {'Count':>8}  {'%':>6}  {'Avg conf':>9}")
        print(f"  {'-'*42}")
        for cls_info in summary['class_distribution'].values():
            print(f"  {cls_info['name']:<12} {cls_info['count']:>8,}  "
                  f"{cls_info['pct']:>5.1f}%  {cls_info['mean_confidence']:>9.3f}")

    return {
        'flood_map'         : flood_map,
        'predictions'       : predictions,
        'probabilities'     : probabilities,
        'confidence'        : confidence,
        'rainfall_sequence' : rainfall_seq,
        'conditioning'      : conditioning,
        'summary'           : summary,
        'warnings'          : warn_list,
        'storm_label'       : _storm_label(storm_type, depth_mm, tpeak),
    }

# ── Barangay band: reprojected ONCE onto dem_transform/dem_crs grid ──────────
def compute_class_pcts(flood_map):
    valid = flood_map[flood_map >= 0]
    total = len(valid)
    return {i: 100 * (valid == i).sum() / total if total > 0 else 0.0
            for i in range(5)}

def _load_barangay_band_from_geojson(geojson_path, dst_transform, dst_crs, H, W):
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(dst_crs)
    gdf = gdf.dropna(subset=['psgc_code']).copy()

    # ── Drop null/empty/invalid geometries ───────────────────────────────────
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna() & gdf.geometry.is_valid]

    gdf['psgc_int'] = gdf['psgc_code'].astype(float).astype(int)

    # ── Clip to DEM grid extent ───────────────────────────────────────────────
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


def save_prediction_tiff(result, engine, dem_crs, dem_transform, _barangay_band, save_path: str):
    flood_map  = result['flood_map'].astype(np.float32)
    confidence = result['confidence']

    conf_map = np.full(flood_map.shape, fill_value=-1.0, dtype=np.float32)
    for conf_val, (r, c) in zip(confidence, engine.patch_indices):
        conf_map[r:r + 4, c:c + 4] = float(conf_val)

    H, W = flood_map.shape

    # Single 3-band int32 file — all bands spatially aligned
    with rasterio.open(
        save_path, 'w',
        driver    = 'GTiff',
        height    = H, width = W,
        count     = 3,
        dtype     = 'int32',
        crs       = dem_crs,
        transform = dem_transform,
        nodata    = -1,
    ) as dst:
        dst.write(flood_map.astype(np.int32),          1)  
        dst.write((conf_map * 1000).astype(np.int32),  2)   
        dst.write(_barangay_band.astype(np.int32),     3)   
        dst.update_tags(
            BAND_1      = 'Flood class (0=No Flood 1=Light 2=Moderate 3=Heavy 4=Extreme)',
            BAND_2      = 'Confidence x1000 (divide by 1000 to get 0.0-1.0)',
            BAND_3      = 'Barangay PSGC Code',
            STORM_LABEL = result.get('storm_label', ''),
        )

    print(f"✓ Saved → {save_path}")
    
def visualize_prediction_tiff(tiff_path: str, title: str = None, figsize=(15, 4)):
    with rasterio.open(tiff_path) as src:
        flood_map = src.read(1).astype(np.float32)
        conf_map  = src.read(2).astype(np.float32) / 1000.0
        bar_map   = src.read(3) if src.count >= 3 else None
        transform = src.transform
        bounds    = src.bounds

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    nodata_mask = flood_map < 0

    # Zoom to valid flood pixels
    valid_rows, valid_cols = np.where(~nodata_mask)
    if len(valid_rows):
        r0, r1 = max(valid_rows.min(), 0), min(valid_rows.max(), flood_map.shape[0]-1)
        c0, c1 = max(valid_cols.min(), 0), min(valid_cols.max(), flood_map.shape[1]-1)
        xlim = (transform.c + c0*transform.a, transform.c + c1*transform.a)
        ylim = (transform.f + r1*transform.e, transform.f + r0*transform.e)
    else:
        xlim = (bounds.left, bounds.right)
        ylim = (bounds.bottom, bounds.top)

    flood_masked = np.ma.masked_where(nodata_mask, flood_map)
    conf_masked  = np.ma.masked_where(nodata_mask, conf_map)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title or tiff_path, fontsize=11, fontweight='bold', y=1.02)

    # ── Band 1: Flood Class ───────────────────────────────────────────────────
    cmap_flood = mcolors.ListedColormap(['#FFFFFF','#C6DBEF','#6BAED6','#2171B5','#08306B'])
    cmap_flood.set_bad(alpha=0)
    norm_flood = mcolors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5], 5)

    axes[0].imshow(flood_masked, cmap=cmap_flood, norm=norm_flood,
                   extent=extent, interpolation='nearest', aspect='equal')
    axes[0].set_xlim(*xlim); axes[0].set_ylim(*ylim)
    axes[0].set_title('Flood Class', fontsize=9, fontweight='bold')
    axes[0].tick_params(labelsize=6)

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(
        ['#FFFFFF','#C6DBEF','#6BAED6','#2171B5','#08306B'],
        ['No Flood','Light','Moderate','Heavy','Extreme'])]
    axes[0].legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.8)

    # ── Band 2: Confidence ────────────────────────────────────────────────────
    cmap_conf = plt.cm.RdYlGn.copy(); cmap_conf.set_bad(alpha=0)
    im2 = axes[1].imshow(conf_masked, cmap=cmap_conf, vmin=0, vmax=1,
                          extent=extent, interpolation='nearest', aspect='equal')
    axes[1].set_xlim(*xlim); axes[1].set_ylim(*ylim)
    axes[1].set_title('Confidence', fontsize=9, fontweight='bold')
    axes[1].tick_params(labelsize=6)

    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046)
    cb2.ax.tick_params(labelsize=6)
    cb2.ax.axhline(y=0.5, color='black', linewidth=1, linestyle='--')

    # Uncertain overlay
    unc = np.where(~nodata_mask & (conf_map < 0.5), 1.0, np.nan)
    axes[1].imshow(unc, cmap='cool', alpha=0.3, vmin=0, vmax=1,
                   extent=extent, interpolation='nearest', aspect='equal')

    valid_conf = conf_map[~nodata_mask]
    axes[1].text(0.02, 0.02,
                 f"μ={valid_conf.mean():.2f}  σ={valid_conf.std():.2f}\n"
                 f"Uncertain: {(valid_conf<0.5).mean()*100:.1f}%",
                 transform=axes[1].transAxes, fontsize=6, va='bottom',
                 bbox=dict(boxstyle='round', fc='white', ec='lightgray', alpha=0.85))

    # ── Band 3: Barangay ─────────────────────────────────────────────────────
    if bar_map is not None:
        bar_masked   = np.ma.masked_equal(bar_map, 0)
        unique_codes = np.unique(bar_map[bar_map > 0])
        cmap_bar     = plt.cm.get_cmap('tab20', max(len(unique_codes), 1))
        cmap_bar.set_bad(color='white')

        axes[2].imshow(bar_masked, cmap=cmap_bar,
                       extent=extent, interpolation='nearest', aspect='equal')

        bar_rows, bar_cols = np.where(bar_map > 0)
        if len(bar_rows):
            bpad = 20
            br0, br1 = max(bar_rows.min()-bpad,0), min(bar_rows.max()+bpad, bar_map.shape[0]-1)
            bc0, bc1 = max(bar_cols.min()-bpad,0), min(bar_cols.max()+bpad, bar_map.shape[1]-1)
            axes[2].set_xlim(transform.c + bc0*transform.a, transform.c + bc1*transform.a)
            axes[2].set_ylim(transform.f + br1*transform.e, transform.f + br0*transform.e)
        else:
            axes[2].set_xlim(*xlim); axes[2].set_ylim(*ylim)

        axes[2].set_title(f'Barangays ({len(unique_codes)})', fontsize=9, fontweight='bold')
        axes[2].tick_params(labelsize=6)
        axes[2].text(0.02, 0.02, f"{len(unique_codes)} barangay(s)",
                     transform=axes[2].transAxes, fontsize=6, va='bottom',
                     bbox=dict(boxstyle='round', fc='white', ec='lightgray', alpha=0.85))
    else:
        axes[2].axis('off')
        axes[2].set_title('Barangay (N/A)', fontsize=9)

    plt.tight_layout()
    return fig

def run_inference(
    engine: InferenceEngine,
    storm_type: str,
    depth_mm: float,
    output_dir: str | Path,
    dem_transform: rasterio.Affine,
    dem_crs: rasterio.crs.CRS,
    tpeak: Optional[float] = None,
    barangay_band: Optional[np.ndarray] = None,
    stem: Optional[str] = None,
    visualize: bool = True,
    figsize: tuple = (15, 4),
) -> Dict:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate filename stem if not provided
    if stem is None:
        tp_tag = f"_tp{str(tpeak).replace('.', '')}" if tpeak is not None else ""
        stem = f"{storm_type.replace('-', '_')}_{int(depth_mm)}mm{tp_tag}"

    tif_path = str(output_dir / f"{stem}.tif")

    # 2. Predict
    result = predict_with_confidence(
        engine=engine,
        storm_type=storm_type,
        depth_mm=depth_mm,
        tpeak=tpeak,
        verbose=True
    )

    # 3. Save GeoTIFF
    # Note: Using your existing save_prediction_tiff logic
    save_prediction_tiff(
        result=result,
        engine=engine,
        dem_crs=dem_crs,
        dem_transform=dem_transform,
        _barangay_band=barangay_band if barangay_band is not None else np.zeros((1152, 1152), dtype='int32'),
        save_path=tif_path
    )

    # 4. Optional Visualization
    if visualize:
        title = f"{storm_type.title()} | {depth_mm}mm" + (f" | tpeak={tpeak}" if tpeak else "")
        visualize_prediction_tiff(tif_path, title=title, figsize=figsize)
        plt.show()

    result["tif_path"] = tif_path
    return result
