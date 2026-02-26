import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ── Flood class styling ────────────────────────────────────────────────────────
CLASS_COLORS = {
     0: '#FFFFFF',   # No Flood
     1: '#FFEB3B',   # Light
     2: '#FF9800',   # Moderate
     3: '#F44336',   # Heavy
     4: '#9C27B0',   # Extreme
}
CLASS_LABELS = {
    -1: 'Outside', 0: 'No Flood', 1: 'Light',
     2: 'Moderate', 3: 'Heavy',   4: 'Extreme'
}

FLOOD_CMAP  = mcolors.ListedColormap([CLASS_COLORS[k] for k in range(5)])
FLOOD_NORM  = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], FLOOD_CMAP.N)
FLOOD_TICKS = [0, 1, 2, 3, 4]


def _to_masked(flood_map):
    arr = flood_map.astype(np.float32)
    arr[arr == -1] = np.nan
    return arr


def _add_flood_colorbar(fig, ax, im):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=FLOOD_TICKS)
    cbar.set_ticklabels([CLASS_LABELS[t] for t in FLOOD_TICKS])
    cbar.ax.tick_params(labelsize=8)
    return cbar


def _storm_title(result):
    return result['storm_label']


def _flood_imshow(ax, flood_map, **kwargs):
    masked = _to_masked(flood_map)
    cmap = FLOOD_CMAP
    cmap.set_bad(color='none')   # NaN → transparent
    return ax.imshow(masked, cmap=cmap, norm=FLOOD_NORM,
                     interpolation='none', **kwargs)


# ── 1. Single map: DEM + flood side by side ────────────────────────────────────
def plot_flood_map(result, engine, save_path=None):
    """DEM and predicted flood map side by side. Outside mask is transparent."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle(f"Flood Prediction  —  {_storm_title(result)}",
                 fontsize=13, fontweight='bold')

    # DEM
    im0 = axes[0].imshow(engine.test_dem, cmap='terrain', interpolation='none')
    axes[0].set_title('Digital Elevation Model', fontsize=11)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Elevation (m)')
    axes[0].axis('off')

    # Flood map — DEM as background, flood overlay on top
    im1 = _flood_imshow(axes[1], result['flood_map'])
    axes[1].set_title('Predicted Flood Category', fontsize=11)
    _add_flood_colorbar(fig, axes[1], im1)
    axes[1].axis('off')

    # Stats box
    s = result['summary']
    stats_text = (f"Flooded: {s['flooded_pct']:.1f}%\n"
                  f"Mean conf: {s['mean_confidence']:.3f}\n"
                  f"Dominant: {CLASS_LABELS[s['dominant_class']]}")
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                 fontsize=8, va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    plt.show()


# ── 2. Confidence map ──────────────────────────────────────────────────────────
def plot_confidence(result, engine, save_path=None):
    """Flood prediction overlaid with per-patch confidence. Outside is transparent."""
    # Build confidence spatial map — NaN outside mask
    conf_map = np.full((1152, 1152), fill_value=np.nan)
    patch_size = 4
    for conf, (r, c) in zip(result['confidence'], engine.patch_indices):
        conf_map[r:r + patch_size, c:c + patch_size] = conf

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle(f"Confidence  —  {_storm_title(result)}",
                 fontsize=13, fontweight='bold')

    # Flood map
    im0 = _flood_imshow(axes[0], result['flood_map'])
    axes[0].set_title('Predicted Flood Category', fontsize=11)
    _add_flood_colorbar(fig, axes[0], im0)
    axes[0].axis('off')

    # Confidence map — NaN (outside) renders transparent over DEM background
    conf_cmap = plt.cm.RdYlGn.copy()
    conf_cmap.set_bad(color='none')
    im1 = axes[1].imshow(conf_map, cmap=conf_cmap, vmin=0.2, vmax=1.0,
                          interpolation='none')
    axes[1].set_title('Prediction Confidence (softmax max)', fontsize=11)
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Confidence', fontsize=9)
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    plt.show()


# ── 3. Multi-scenario comparison ───────────────────────────────────────────────
def plot_comparison(results, engine, save_path=None):
    """Side-by-side flood maps. Outside pixels transparent over DEM background."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    fig.patch.set_facecolor('white')
    if n == 1:
        axes = [axes]

    fig.suptitle('Flood Prediction Comparison', fontsize=13, fontweight='bold')

    for ax, result in zip(axes, results):
        im = _flood_imshow(ax, result['flood_map'])
        ax.set_title(_storm_title(result), fontsize=9, fontweight='bold')
        _add_flood_colorbar(fig, ax, im)
        ax.axis('off')
        ax.set_facecolor('white')

        s = result['summary']
        stats = (f"Flooded: {s['flooded_pct']:.1f}%\n"
                 f"Conf: {s['mean_confidence']:.3f}")
        if result['warnings']:
            stats += "\n⚠ OOD"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    plt.show()


# ── 4. Class distribution bar chart ───────────────────────────────────────────
def plot_class_distribution(results, labels=None, save_path=None):
    """Grouped bar chart of class distribution across multiple results."""
    n = len(results)
    labels = labels or [r['storm_label'] for r in results]
    classes = [0, 1, 2, 3, 4]
    class_names = [CLASS_LABELS[c] for c in classes]

    x = np.arange(len(classes))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (result, label) in enumerate(zip(results, labels)):
        pcts = [result['summary']['class_distribution'][c]['pct'] for c in classes]
        ax.bar(x + i * width - (n - 1) * width / 2, pcts,
               width, label=label,
               color=[CLASS_COLORS[c] for c in classes],
               edgecolor='grey', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel('% of Manila patches')
    ax.set_title('Class Distribution by Storm Scenario')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    plt.show()


# ── 5. Hyetograph panel ────────────────────────────────────────────────────────
def plot_hyetograph(results, labels=None, save_path=None):
    """Plot the rainfall sequences back-converted to mm/hr."""
    from hyetograph import RAIN_MAX
    labels = labels or [r['storm_label'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10.colors

    for i, (result, label) in enumerate(zip(results, labels)):
        seq_raw = result['rainfall_sequence'] * RAIN_MAX
        timesteps = np.arange(1, len(seq_raw) + 1)
        ax.step(timesteps, seq_raw, where='mid',
                color=colors[i % 10], label=label, lw=2)

    ax.set_xlabel('Timestep (5-min blocks)')
    ax.set_ylabel('Intensity (mm/hr)')
    ax.set_title('Rainfall Hyetographs')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, len(seq_raw) + 0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved → {save_path}")
    plt.show()