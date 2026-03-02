import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
import matplotlib.patches as mpatches, matplotlib.colors as mcolors
import rasterio
import numpy as np, torch, matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from shapely.geometry import box
import matplotlib.pyplot as plt
from pathlib import Path
from hyetograph import build_conditioning_vector as hyeto_build_conditioning

# ── Visual setup ──────────────────────────────────────────────────────────────
FLOOD_COLORS = {
     0: '#FFFFFF',
     1: '#C6DBEF',
     2: '#6BAED6',
     3: '#2171B5',
     4: '#08306B',
}
FLOOD_LABELS = {0: 'No Flood', 1: 'Light', 2: 'Moderate', 3: 'Heavy', 4: 'Extreme'}


FLOOD_CLASS_NAMES = {
    0: 'No Flood',
    1: 'Light',
    2: 'Moderate',
    3: 'Heavy',
    4: 'Extreme',
}

VALID_RANGES = {
    'front-loaded': {'depth_mm': (6, 78),   'tpeak': None},
    'balanced':     {'depth_mm': (19, 78),  'tpeak': None},
    'back-loaded':  {'depth_mm': (7, 75),   'tpeak': None},
    'triangular':   {'depth_mm': (5, 77),   'tpeak': (0.1, 0.9)},
}


def validate_storm_input(storm_type: str, depth_mm: float, tpeak: Optional[float] = None, strict: bool = False) -> Tuple[bool, List[str]]:
    if storm_type not in VALID_RANGES:
        return False, [
            f"Unknown storm type: '{storm_type}'. "
            f"Must be one of {list(VALID_RANGES.keys())}"
        ]

    warnings = []
    ranges   = VALID_RANGES[storm_type]
    d_min, d_max = ranges['depth_mm']

    if not (d_min <= depth_mm <= d_max):
        return False, [
            f"Depth {depth_mm} mm out of range [{d_min}, {d_max}] mm "
            f"for '{storm_type}'"
        ]

    if storm_type == 'triangular':
        if tpeak is None:
            return False, ["tpeak is required for triangular storms."]
        tp_min, tp_max = ranges['tpeak']
        if not (tp_min <= tpeak <= tp_max):
            return False, [
                f"tpeak {tpeak} out of range [{tp_min}, {tp_max}] "
                f"for triangular storms"
            ]
    else:
        if tpeak is not None:
            msg = (f"tpeak is not applicable for '{storm_type}' storms "
                   f"and will be ignored.")
            if strict:
                return False, [msg]
            warnings.append(msg)

    return True, warnings

# ── Inference Engine ──────────────────────────────────────────────────────────
class FloodInference:
    def __init__(
        self,
        model:                torch.nn.Module,
        device:               torch.device,
        rain_min:             float,
        rain_max:             float,
        X_spatial:            np.ndarray,
        num_classes:          int,
        batch_size:           int,
        manila_patch_indices: np.ndarray,
    ):
        self.model       = model
        self.device      = device
        self.rain_min    = rain_min
        self.rain_max    = rain_max
        self.X_spatial   = X_spatial
        self.num_classes = num_classes
        self.batch_size  = batch_size

        # Manila mask support
        self.manila_patch_indices = manila_patch_indices
        self.masked_mode = manila_patch_indices is not None

        # Pre-convert to CHW tensor for PyTorch
        self.X_spatial_tensor = (
            torch.from_numpy(X_spatial).float().permute(0, 3, 1, 2)
        )
        self.n_patches = X_spatial.shape[0]   # N_MANILA or N_H*N_W

        print(f"\n{'='*60}")
        print(f"FLOOD INFERENCE ENGINE READY")
        print(f"{'='*60}")
        print(f"  Device         : {device}")
        print(f"  Model          : {type(model).__name__}")
        print(f"  Spatial patches: {self.n_patches:,} "
              f"{'(Manila mask)' if self.masked_mode else '(full grid)'}")
        print(f"  Patch shape    : {X_spatial.shape[1:]}")
        print(f"  Rain range     : [{rain_min:.4f}, {rain_max:.4f}]")
        print(f"  Num classes    : {num_classes}")
        print(f"  Batch size     : {batch_size}")
        if self.masked_mode:
            print(f"  Manila indices : {len(manila_patch_indices):,} patches indexed")
        print(f"{'='*60}\n")

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        storm_type: str,
        depth_mm:   float,
        tpeak:      Optional[float] = None,
        n_h:        Optional[int]   = None,
        n_w:        Optional[int]   = None,
        batch_size: Optional[int]   = None,
        verbose:    bool = True,
        strict:     bool = False,
    ) -> Dict:
        bs = batch_size if batch_size is not None else self.batch_size

        # 1. Validate ──────────────────────────────────────────────────────────
        is_valid, val_warnings = validate_storm_input(
            storm_type, depth_mm, tpeak, strict=strict
        )
        if not is_valid:
            raise ValueError(f"Invalid storm input: {val_warnings[0]}")

        # 2. Build conditioning vector ─────────────────────────────────────────
        cond_vec, hyeto_warnings = hyeto_build_conditioning(
            storm_type = storm_type,
            depth_mm   = depth_mm,
            rain_min   = self.rain_min,
            rain_max   = self.rain_max,
            tpeak      = tpeak,
        )
        all_warnings = val_warnings + hyeto_warnings

        # 3. Broadcast conditioning to all Manila patches ──────────────────────
        cond_all    = np.tile(cond_vec[np.newaxis], (self.n_patches, 1))
        cond_tensor = torch.from_numpy(cond_all).float()

        # 4. Batch inference ───────────────────────────────────────────────────
        all_preds = []
        all_confs = []

        if verbose:
            print(f"Predicting {self.n_patches:,} patches (batch_size={bs})...")

        for start in range(0, self.n_patches, bs):
            end = min(start + bs, self.n_patches)

            X_batch    = self.X_spatial_tensor[start:end].to(self.device)
            cond_batch = cond_tensor[start:end].to(self.device)

            logits, _ = self.model(X_batch, cond_batch)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()

            all_preds.append(preds)
            all_confs.append(confs)

            if verbose and (end % (bs * 5) == 0 or end == self.n_patches):
                print(f"  checkmark {end:,}/{self.n_patches:,} patches")

        # Manila-level flat arrays
        patch_predictions = np.concatenate(all_preds)   # (N_manila,)
        patch_confidences = np.concatenate(all_confs)   # (N_manila,)

        # 5. Reconstruct 2D maps ───────────────────────────────────────────────
        flood_map = None
        conf_map  = None

        if n_h is not None and n_w is not None:
            n_total = n_h * n_w

            if self.masked_mode:
                # Sparse -> dense:
                # Non-Manila slots stay 0 (No Flood) / 0.0 (no confidence)
                pred_grid = np.zeros(n_total, dtype=np.int8)
                conf_grid = np.zeros(n_total, dtype=np.float32)
                pred_grid[self.manila_patch_indices] = patch_predictions
                conf_grid[self.manila_patch_indices] = patch_confidences
            else:
                # Full grid mode: n_h * n_w must match exactly
                if n_total != self.n_patches:
                    raise ValueError(
                        f"Grid ({n_h} x {n_w} = {n_total}) does not match "
                        f"n_patches = {self.n_patches}. "
                        f"If using Manila mask, pass manila_patch_indices "
                        f"to FloodInference.__init__."
                    )
                pred_grid = patch_predictions.astype(np.int8)
                conf_grid = patch_confidences.astype(np.float32)

            flood_map = pred_grid.reshape(n_h, n_w).astype(np.float32)
            conf_map  = conf_grid.reshape(n_h, n_w).astype(np.float32)

        # 6. Statistics — Manila patches only (excludes non-Manila zeros) ──────
        stats = self._compute_statistics(patch_predictions)

        # 7. Summary ───────────────────────────────────────────────────────────
        if verbose:
            self._print_summary(
                storm_type, depth_mm, tpeak,
                all_warnings, stats, patch_confidences
            )

        return {
            'patch_predictions':   patch_predictions,
            'patch_confidences':   patch_confidences,
            'flood_map':           flood_map,
            'conf_map':            conf_map,
            'statistics':          stats,
            'warnings':            all_warnings,
            'config': {
                'storm_type': storm_type,
                'depth_mm':   depth_mm,
                'tpeak':      tpeak,
            },
            'conditioning_vector': cond_vec,
        }

    def predict_batch(
        self,
        test_configs: List[Dict],
        n_h:          Optional[int] = None,
        n_w:          Optional[int] = None,
        batch_size:   Optional[int] = None,
        verbose:      bool = True,
        strict:       bool = False,
    ) -> List[Dict]:
        results = []

        for i, config in enumerate(test_configs):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Test {i+1}/{len(test_configs)}")
                print(f"{'='*60}")

            try:
                result = self.predict(
                    storm_type = config['storm_type'],
                    depth_mm   = config['depth_mm'],
                    tpeak      = config.get('tpeak', None),
                    n_h        = n_h,
                    n_w        = n_w,
                    batch_size = batch_size,
                    verbose    = verbose,
                    strict     = strict,
                )
                results.append(result)

            except Exception as e:
                print(f"  Error: {e}")
                results.append({'error': str(e), 'config': config})

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_statistics(self, patch_predictions: np.ndarray) -> Dict:
        total = patch_predictions.size
        stats = {}
        for cls_id, cls_name in FLOOD_CLASS_NAMES.items():
            count = int((patch_predictions == cls_id).sum())
            stats[cls_name] = {
                'count':      count,
                'percentage': float(100 * count / total) if total > 0 else 0.0,
            }
        return stats

    def _print_summary(
        self,
        storm_type:        str,
        depth_mm:          float,
        tpeak:             Optional[float],
        warnings:          List[str],
        stats:             Dict,
        patch_confidences: Optional[np.ndarray] = None,
    ) -> None:
        """Pretty-print prediction summary."""
        print(f"\n{'='*60}")
        print(f"PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Storm type     : {storm_type}")
        print(f"  Depth          : {depth_mm} mm")
        if tpeak is not None:
            print(f"  tpeak          : {tpeak}")

        if warnings:
            print(f"  Warnings       : {len(warnings)}")
            for w in warnings:
                print(f"    - {w}")
        else:
            print(f"  No warnings    : inputs within training range")

        if patch_confidences is not None:
            frac_low = float((patch_confidences < 0.5).mean() * 100)
            print(f"\n  Confidence stats (Manila patches):")
            print(f"    Mean      : {patch_confidences.mean():.4f}")
            print(f"    Std       : {patch_confidences.std():.4f}")
            print(f"    Min       : {patch_confidences.min():.4f}")
            print(f"    Max       : {patch_confidences.max():.4f}")
            print(f"    Uncertain : {frac_low:.1f}%  (conf < 0.5)")

        print(f"\n  Flood class distribution (Manila patches):")
        for cls_name, data in stats.items():
            pct_str   = f"{data['percentage']:>5.1f}%"
            count_str = f"{data['count']:>7,}"
            bar       = '#' * int(data['percentage'] / 2)
            print(f"    {cls_name:<12}: {count_str} patches  {pct_str}  {bar}")
        print(f"{'='*60}\n")

# ── Helper functions ──────────────────────────────────────────────────────────
def build_subtitle(cfg_dict):
    parts = [f"Storm: {cfg_dict['storm_type']}", f"Depth: {cfg_dict['depth_mm']} mm"]
    if cfg_dict.get('tpeak') is not None:
        parts.append(f"tpeak: {cfg_dict['tpeak']}")
    return '  |  '.join(parts)

def compute_class_pcts(flood_map):
    total = flood_map.size
    return {i: 100 * (flood_map == i).sum() / total for i in range(5)}

def save_result_as_tif(flood_map, conf_map, config, save_path, transform=None, crs=None, barangay_band=None, mask_2d=None):
    H, W    = flood_map.shape
    n_bands = 3 if barangay_band is not None else 2

    # Convert to int32 first, then stamp nodata on outside-Manila pixels
    b1 = flood_map.astype(np.int32)
    b2 = (conf_map * 1000).astype(np.int32)
    outside = ~mask_2d
    b1[outside] = -1
    b2[outside] = -1

    with rasterio.open(
        save_path, mode='w', driver='GTiff',
        height=H, width=W, count=n_bands, dtype='int32',
        crs=crs, transform=transform, compress='lzw',
        nodata=-1,
    ) as dst:
        dst.write(b1, 1)
        dst.write(b2, 2)
        if barangay_band is not None:
            b3 = barangay_band.astype(np.int32)
            if mask_2d is not None:
                b3[outside] = -1
            dst.write(b3, 3)

        dst.update_tags(1, description='Flood Hazard Class (0=None 1=Light 2=Moderate 3=Heavy 4=Extreme)')
        dst.update_tags(2, description='Confidence x1000 — divide by 1000 for 0.0-1.0')
        if barangay_band is not None:
            dst.update_tags(3, description='Barangay PSGC Code (-1=outside Manila extent)')

        dst.update_tags(
            model      = 'EXP-DES-3 ViTFloodClassifier',
            storm_type = config['storm_type'],
            depth_mm   = str(config['depth_mm']),
            tpeak      = str(config.get('tpeak')),
        )
    print(f"  ✓ GeoTIFF saved → {save_path}")

def visualize_result_tif(tif_path, save_path=None, figsize=(21, 6)):
    _flood_cmap = mcolors.ListedColormap([FLOOD_COLORS[i] for i in range(5)])
    _flood_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)
    _conf_cmap  = 'RdYlGn'

    with rasterio.open(tif_path) as src:
        band1_raw = src.read(1).astype(np.int32)
        band2_raw = src.read(2).astype(np.float32) / 1000.0
        band3_raw = src.read(3) 
        nodata    = src.nodata  
        tags      = src.tags()
        b1_tag    = src.tags(1)
        b2_tag    = src.tags(2)
        b3_tag    = src.tags(3) 

    valid  = band1_raw != int(nodata) if nodata is not None else np.ones_like(band1_raw, dtype=bool)
    band1  = np.where(valid, band1_raw.astype(float), np.nan)
    band2  = np.where(valid, band2_raw,                np.nan)
    band3  = None
    band3 = np.where(valid, band3_raw.astype(float), np.nan)

    # Crop all bands to the bounding box of valid pixels
    valid_rows, valid_cols = np.where(valid)
    pad = 0
    vr0 = max(int(valid_rows.min()) - pad, 0)
    vr1 = min(int(valid_rows.max()) + pad, band1.shape[0] - 1)
    vc0 = max(int(valid_cols.min()) - pad, 0)
    vc1 = min(int(valid_cols.max()) + pad, band1.shape[1] - 1)
    band1    = band1   [vr0:vr1+1, vc0:vc1+1]
    band2    = band2   [vr0:vr1+1, vc0:vc1+1]
    band3    = band3   [vr0:vr1+1, vc0:vc1+1] 
    band3_raw = band3_raw[vr0:vr1+1, vc0:vc1+1] 
    valid    = valid   [vr0:vr1+1, vc0:vc1+1]

    print(f"\nGeoTIFF: {tif_path}")
    print(f"  Band 1 : {b1_tag.get('description', 'Flood class')}")
    print(f"  Band 2 : {b2_tag.get('description', 'Confidence')}")
    if band3 is not None:
        print(f"  Band 3 : {b3_tag.get('description', 'Barangay PSGC Code')}")
    print(f"  Storm  : {tags.get('storm_type','?')}  {tags.get('depth_mm','?')}mm  tpeak={tags.get('tpeak','?')}")

    n_cols = 3 if band3 is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # ── Band 1: Flood class ───────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.imshow(band1, cmap=_flood_cmap, norm=_flood_norm, interpolation='nearest', origin='upper')
    tpeak_str = tags.get('tpeak', 'None')
    title1 = (f"Band 1 — Flood Hazard Class\n"
               f"Storm: {tags.get('storm_type','?')}  |  Depth: {tags.get('depth_mm','?')} mm"
               + (f"  |  tpeak: {tpeak_str}" if tpeak_str not in ('None', '', 'none') else ''))
    ax1.set_title(title1, fontsize=9, fontweight='bold', pad=6)
    ax1.set_xlabel('Column', fontsize=8)
    ax1.set_ylabel('Row',    fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.legend(
        handles=[mpatches.Patch(facecolor=FLOOD_COLORS[i], edgecolor='gray',
                                linewidth=0.5, label=f'Class {i} — {FLOOD_LABELS[i]}')
                 for i in range(5)],
        loc='lower left', fontsize=7, framealpha=0.85,
        title='Flood Class', title_fontsize=7)
    total = int(np.sum(valid))
    ax1.text(0.98, 0.98,
             '\n'.join([f"C{i} {FLOOD_LABELS[i]}: {100*np.nansum(band1==i)/total:.1f}%" for i in range(5)]),
             transform=ax1.transAxes, ha='right', va='top', fontsize=6.5,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=0.88))

    # ── Band 2: Confidence ────────────────────────────────────────────────────
    ax2 = axes[1]
    im2 = ax2.imshow(band2, cmap=_conf_cmap, vmin=0.0, vmax=1.0,
                     interpolation='nearest', origin='upper')
    ax2.set_title('Band 2 — Model Confidence', fontsize=9, fontweight='bold', pad=6)
    ax2.set_xlabel('Column', fontsize=8)
    ax2.set_ylabel('Row',    fontsize=8)
    ax2.tick_params(labelsize=7)
    cbar = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04)
    cbar.set_label('Confidence', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.axhline(y=0.5, color='black', linewidth=1.2, linestyle='--')
    cbar.ax.text(2.3, 0.5, 'uncertain\nthreshold', fontsize=6, va='center')
    uncertain_overlay = np.where(~np.isnan(band2) & (band2 < 0.5), 1.0, np.nan).astype(np.float32)
    ax2.imshow(uncertain_overlay, cmap='cool', alpha=0.30, vmin=0, vmax=1, interpolation='nearest', origin='upper')
    frac_low = (band2 < 0.5).mean() * 100
    ax2.text(0.98, 0.98,
             f"Mean : {band2.mean():.3f}\nStd  : {band2.std():.3f}\n"
             f"Min  : {band2.min():.3f}\nMax  : {band2.max():.3f}\nUncertain: {frac_low:.1f}%",
             transform=ax2.transAxes, ha='right', va='top', fontsize=6.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.88))

    # ── Band 3: Barangay PSGC Code ────────────────────────────────────────────
    ax3 = axes[2]
    bar_masked   = np.ma.masked_invalid(np.where((band3 > 0) & ~np.isnan(band3), band3, np.nan))
    unique_codes = np.unique(band3_raw[(band3_raw > 0) & (band3_raw != int(nodata))])
    n_unique     = max(len(unique_codes), 1)
    bar_cmap     = plt.cm.get_cmap('tab20', n_unique)
    bar_cmap.set_bad(color='white')
    im3 = ax3.imshow(bar_masked, cmap=bar_cmap,
                        interpolation='nearest', origin='upper')
    ax3.set_title(
        f"Band 3 — Barangay Boundaries\nPSGC codes  |  {n_unique} barangay(s) in extent",
        fontsize=9, fontweight='bold', pad=6)
    ax3.set_xlabel('Column', fontsize=8)
    ax3.set_ylabel('Row',    fontsize=8)
    ax3.tick_params(labelsize=7)
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.035, pad=0.04)
    cbar3.set_label('PSGC Code', fontsize=8)
    cbar3.ax.tick_params(labelsize=7)
    ax3.text(0.98, 0.98,
                f"Barangays : {n_unique}\n"
                f"Min code  : {int(unique_codes.min()) if n_unique else 'N/A'}\n"
                f"Max code  : {int(unique_codes.max()) if n_unique else 'N/A'}\n"
                f"Coverage  : {100*(band3_raw>0).mean():.1f}%",
                transform=ax3.transAxes, ha='right', va='top', fontsize=6.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.88))

    fig.suptitle(
        f'EXP-DES-3 — GeoTIFF Result Visualization\n{tif_path}',
        fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Visualization saved → {save_path}")
    plt.show()

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

def _plot_flood_map(ax_flood, ax_bar, flood_disp, pcts, cfg_label, subtitle, N_H, N_W,
    r0, r1, c0, c1, flood_cmap, flood_norm, legend_patches, fig) -> None:

    fig.suptitle(
        f'EXP-DES-3 — Flood Prediction Map ({N_H}×{N_W})\n{cfg_label}  |  {subtitle}',
        fontsize=12, fontweight='bold', y=1.03,
    )
    ax_flood.imshow(flood_disp, cmap=flood_cmap, norm=flood_norm,
                    interpolation='nearest', origin='upper')
    ax_flood.set_title('Flood Hazard Classification', fontsize=9, fontweight='bold', pad=8)
    ax_flood.set_xlabel('Column (West → East)', fontsize=8, labelpad=6)
    ax_flood.set_ylabel('Row (North → South)',  fontsize=8, labelpad=6)
    ax_flood.tick_params(labelsize=7)
    ax_flood.set_xlim(c0, c1);  ax_flood.set_ylim(r1, r0)

    labels = [FLOOD_LABELS[j] for j in range(5)]
    values = [pcts[j]         for j in range(5)]
    colors = [FLOOD_COLORS[j] for j in range(5)]
    bars   = ax_bar.barh(labels, values, color=colors, edgecolor='gray', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, values):
        if val > 1.5:
            ax_bar.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', ha='right', fontsize=8,
                        color='white', fontweight='bold')
        elif val > 0.1:
            ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', ha='left', fontsize=8, color='#333333')
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel('Coverage (%)', fontsize=8, labelpad=6)
    ax_bar.set_title('Class\nDistribution', fontsize=8.5, fontweight='bold', pad=8)
    ax_bar.tick_params(axis='both', labelsize=8)
    ax_bar.spines['top'].set_visible(False);  ax_bar.spines['right'].set_visible(False)
    ax_bar.invert_yaxis()
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=8.5,
               frameon=True, title='Flood Severity Classes', title_fontsize=8.5,
               bbox_to_anchor=(0.5, -0.06))

def _plot_confidence_map(ax_conf, ax_hist, conf_disp, conf_map, patch_confs, frac_low, cfg_label, subtitle,
    N_H, N_W, r0, r1, c0, c1, manila_mask_2d, fig) -> None:

    fig.suptitle(
        f'EXP-DES-3 — Confidence Map ({N_H}×{N_W})\n{cfg_label}  |  {subtitle}',
        fontsize=12, fontweight='bold', y=1.03,
    )
    im = ax_conf.imshow(conf_disp, cmap='RdYlGn', vmin=0.0, vmax=1.0,
                        interpolation='nearest', origin='upper')
    ax_conf.set_title('Model Confidence', fontsize=9, fontweight='bold', pad=8)
    ax_conf.set_xlabel('Column (West → East)', fontsize=8, labelpad=6)
    ax_conf.set_ylabel('Row (North → South)',  fontsize=8, labelpad=6)
    ax_conf.tick_params(labelsize=7)
    ax_conf.set_xlim(c0, c1);  ax_conf.set_ylim(r1, r0)
    cbar = fig.colorbar(im, ax=ax_conf, fraction=0.035, pad=0.04)
    cbar.set_label('Confidence', fontsize=8, labelpad=6)
    cbar.ax.tick_params(labelsize=7);  cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.axhline(y=0.5, color='black', linewidth=1.2, linestyle='--')
    cbar.ax.text(2.5, 0.5, 'uncertain\nthreshold', fontsize=6, va='center')
    uncertain_overlay = np.where(manila_mask_2d & (conf_map < 0.5), 1.0, np.nan)
    ax_conf.imshow(uncertain_overlay, cmap='cool', alpha=0.30,
                   vmin=0, vmax=1, interpolation='nearest', origin='upper')
    ax_conf.text(0.02, 0.98, f'{frac_low:.1f}% uncertain\n(conf < 0.5)',
                 transform=ax_conf.transAxes, ha='left', va='top', fontsize=7.5,
                 color='red' if frac_low > 20 else 'gray',
                 bbox=dict(boxstyle='round,pad=0.30', facecolor='white',
                           edgecolor='lightgray', alpha=0.85))
    ax_hist.hist(patch_confs, bins=20, range=(0, 1),
                 color='steelblue', edgecolor='white', linewidth=0.4, alpha=0.85)
    ax_hist.axvline(x=0.5,               color='red',    linewidth=1.2, linestyle='--', label='Uncertain (0.5)')
    ax_hist.axvline(x=patch_confs.mean(), color='orange', linewidth=1.2,
                    label=f"Mean ({patch_confs.mean():.2f})")
    ax_hist.set_xlabel('Confidence', fontsize=8, labelpad=6)
    ax_hist.set_ylabel('Patches',    fontsize=8, labelpad=6)
    ax_hist.set_title('Confidence\nDistribution', fontsize=8.5, fontweight='bold', pad=8)
    ax_hist.tick_params(labelsize=7);  ax_hist.legend(fontsize=6.5, loc='upper left')
    ax_hist.spines['top'].set_visible(False);  ax_hist.spines['right'].set_visible(False)

def run_and_export(
    results:          List[Dict],
    test_configs:     List[Dict],
    output_dir,                       
    N_H:              int,
    N_W:              int,
    manila_patch_indices: np.ndarray,
    dem_transform,
    dem_crs,
    flood_cmap,
    flood_norm,
    legend_patches:   list,
    barangay_band:    Optional[np.ndarray] = None,
    prefix:           str = 'web',
    dpi:              int = 150,
) -> None:


    output_dir = Path(output_dir)

    # ── Build Manila mask once ────────────────────────────────────────────────
    manila_mask      = np.zeros(N_H * N_W, dtype=bool)
    manila_mask[manila_patch_indices] = True
    manila_mask_2d   = manila_mask.reshape(N_H, N_W)

    rows, cols = np.where(manila_mask_2d)
    r0 = int(rows.min());  r1 = int(rows.max())
    c0 = int(cols.min());  c1 = int(cols.max())

    # ── Per-config loop ───────────────────────────────────────────────────────
    for i, (result, config) in enumerate(zip(results, test_configs), start=1):
        cfg_label = f"Config {i}: {config['storm_type'].title()} | {config['depth_mm']} mm"

        if 'error' in result:
            print(f"\n{cfg_label}: ERROR — {result['error']}")
            continue

        flood_map   = result['flood_map'].astype(np.float32)
        conf_map    = result['conf_map'].astype(np.float32)
        patch_confs = result['patch_confidences']
        pcts        = compute_class_pcts(flood_map)
        frac_low    = float((patch_confs < 0.5).mean() * 100)
        subtitle    = build_subtitle(config)
        flood_disp  = np.where(manila_mask_2d, flood_map, np.nan)
        conf_disp   = np.where(manila_mask_2d, conf_map,  np.nan)

        # Figure 1 — Flood Map ────────────────────────────────────────────────
        fig1, (ax_flood, ax_bar) = plt.subplots(
            1, 2, figsize=(15, 7),
            gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.30},
        )
        _plot_flood_map(ax_flood, ax_bar, flood_disp, pcts,
                        cfg_label, subtitle, N_H, N_W,
                        r0, r1, c0, c1,
                        flood_cmap, flood_norm, legend_patches, fig1)
        plt.savefig(output_dir / f'{prefix}_config{i}_flood_map.png',
                    dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.show();  plt.close(fig1)

        # Figure 2 — Confidence Map ───────────────────────────────────────────
        fig2, (ax_conf, ax_hist) = plt.subplots(
            1, 2, figsize=(15, 7),
            gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.35},
        )
        _plot_confidence_map(ax_conf, ax_hist, conf_disp, conf_map,
                             patch_confs, frac_low,
                             cfg_label, subtitle, N_H, N_W,
                             r0, r1, c0, c1, manila_mask_2d, fig2)
        plt.savefig(output_dir / f'{prefix}_config{i}_confidence_map.png',
                    dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.show();  plt.close(fig2)

        # GeoTIFF export ──────────────────────────────────────────────────────
        tif_out = str(output_dir / f'{prefix}_config{i}_result.tif')
        save_result_as_tif(
            flood_map=flood_map, conf_map=conf_map,
            barangay_band=barangay_band, config=config,
            save_path=tif_out, transform=dem_transform,
            crs=dem_crs, mask_2d=manila_mask_2d,
        )

        # TIF band visualization ───────────────────────────────────────────────
        visualize_result_tif(
            tif_path  = tif_out,
            save_path = str(output_dir / f'{prefix}_config{i}_tif_bands.png'),
        )

    print("\n✓ All configurations processed.")