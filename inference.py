import warnings as _warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset

from hyetograph import build_inference_inputs
from config import TrainConfig
from training import make_model


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