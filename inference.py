import numpy as np
import torch
from typing import Optional, Dict, List, Tuple

from hyetograph import build_conditioning_vector as hyeto_build_conditioning

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


def validate_storm_input(
    storm_type: str,
    depth_mm:   float,
    tpeak:      Optional[float] = None,
    strict:     bool = False,
) -> Tuple[bool, List[str]]:
    """Validate user storm input against training ranges."""
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
