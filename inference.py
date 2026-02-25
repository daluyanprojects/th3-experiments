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
    depth_mm: float,
    tpeak: Optional[float] = None,
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate user storm input against training ranges."""
    if storm_type not in VALID_RANGES:
        return False, [f"Unknown storm type: {storm_type}. Must be one of {list(VALID_RANGES.keys())}"]
    
    warnings = []
    ranges = VALID_RANGES[storm_type]
    d_min, d_max = ranges['depth_mm']
    
    # Depth validation
    if not (d_min <= depth_mm <= d_max):
        return False, [f"Depth {depth_mm}mm out of range [{d_min}, {d_max}]mm for {storm_type}"]
    
    # tpeak validation
    if storm_type == 'triangular':
        if tpeak is None:
            return False, ["tpeak required for triangular storms"]
        tp_min, tp_max = ranges['tpeak']
        if not (tp_min <= tpeak <= tp_max):
            return False, [f"tpeak {tpeak} out of range [{tp_min}, {tp_max}] for triangular"]
    else:
        if tpeak is not None:
            msg = f"tpeak not applicable for {storm_type} storms (will be ignored)"
            if strict:
                return False, [msg]
            else:
                warnings.append(msg)
    
    return True, warnings


# ── Inference Engine ──────────────────────────────────────────────────────────
class FloodInference:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        rain_min: float,
        rain_max: float,
        X_spatial: np.ndarray,
        num_classes: int = 5,
        batch_size: int = 512,
    ):
        self.model = model
        self.device = device
        self.rain_min = rain_min
        self.rain_max = rain_max
        self.X_spatial = X_spatial
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        self.X_spatial_tensor = torch.from_numpy(X_spatial).float().permute(0, 3, 1, 2)
        self.n_patches = X_spatial.shape[0]
        
        print(f"\n{'='*60}")
        print(f"FLOOD INFERENCE ENGINE READY")
        print(f"{'='*60}")
        print(f"  Device         : {device}")
        print(f"  Model          : {type(model).__name__}")
        print(f"  Spatial patches: {self.n_patches:,}")
        print(f"  Patch shape    : {X_spatial.shape[1:]}")
        print(f"  Rain range     : [{rain_min:.4f}, {rain_max:.4f}]")
        print(f"  Num classes    : {num_classes}")
        print(f"  Batch size     : {batch_size}")
        print(f"{'='*60}\n")

    @torch.no_grad()
    def predict(
        self,
        storm_type: str,
        depth_mm: float,
        tpeak: Optional[float] = None,
        n_h: Optional[int] = None,
        n_w: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        strict: bool = False,
    ) -> Dict:
        """
        Predict flood map for a single storm scenario.
        
        Args:
            storm_type: one of ['front-loaded', 'balanced', 'back-loaded', 'triangular']
            depth_mm: total storm depth in mm
            tpeak: peak time fraction (only for triangular)
            n_h, n_w: Grid dimensions for map reconstruction (optional)
                      If provided: reconstructs 2D maps
                      If None: returns flat patch predictions only
            batch_size: inference batch size (uses default if None)
            verbose: print progress
            strict: validation strictness
            
        Returns:
            Dict with keys:
            - 'patch_predictions': (N_patches,) class predictions
            - 'patch_confidences': (N_patches,) softmax max-probs
            - 'flood_map': (n_h, n_w) reconstructed map (if n_h/n_w provided)
            - 'conf_map': (n_h, n_w) confidence map (if n_h/n_w provided)
            - 'statistics': class distribution
            - 'warnings': validation warnings
            - 'config': input parameters
            - 'conditioning_vector': (19,) built vector
        """
        if batch_size is None:
            batch_size = self.batch_size

        # ── 1. Validate input ─────────────────────────────────────────────────
        is_valid, validation_warnings = validate_storm_input(
            storm_type, depth_mm, tpeak, strict=strict
        )
        if not is_valid:
            raise ValueError(f"Invalid input: {validation_warnings[0]}")

        # ── 2. Build conditioning vector ──────────────────────────────────────
        cond_vec, hyeto_warnings = hyeto_build_conditioning(
            storm_type=storm_type,
            depth_mm=depth_mm,
            rain_min=self.rain_min,
            rain_max=self.rain_max,
            tpeak=tpeak,
        )
        all_warnings = validation_warnings + hyeto_warnings

        # ── 3. Broadcast conditioning to all patches ──────────────────────────
        cond_all = np.tile(cond_vec[np.newaxis], (self.n_patches, 1))
        cond_tensor = torch.from_numpy(cond_all).float()

        # ── 4. Batch inference ────────────────────────────────────────────────
        patch_predictions = []
        patch_confidences = []

        if verbose:
            print(f"Predicting {self.n_patches:,} patches (batch_size={batch_size})...")

        for start_idx in range(0, self.n_patches, batch_size):
            end_idx = min(start_idx + batch_size, self.n_patches)

            X_batch = self.X_spatial_tensor[start_idx:end_idx].to(self.device)
            cond_batch = cond_tensor[start_idx:end_idx].to(self.device)

            logits, _ = self.model(X_batch, cond_batch)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()

            patch_predictions.append(preds)
            patch_confidences.append(confs)

            if verbose and (end_idx % (batch_size * 5) == 0 or end_idx == self.n_patches):
                print(f"  ✓ {end_idx:,}/{self.n_patches:,} patches")

        patch_predictions = np.concatenate(patch_predictions)
        patch_confidences = np.concatenate(patch_confidences)

        # ── 5. Optionally reconstruct 2D maps ─────────────────────────────────
        flood_map = None
        conf_map = None
        
        if n_h is not None and n_w is not None:
            # Validate grid dimensions
            if n_h * n_w != self.n_patches:
                raise ValueError(
                    f"Grid dimensions (n_h={n_h}, n_w={n_w}) → {n_h*n_w} patches "
                    f"but got {self.n_patches} patches. Mismatch!"
                )
            
            # Reconstruct as 2D maps
            flood_map = patch_predictions.reshape(n_h, n_w).astype(np.float32)
            conf_map = patch_confidences.reshape(n_h, n_w).astype(np.float32)

        # ── 6. Compute statistics ─────────────────────────────────────────────
        stats = self._compute_statistics(patch_predictions)

        # ── 7. Summary printout ───────────────────────────────────────────────
        if verbose:
            self._print_summary(
                storm_type, depth_mm, tpeak, all_warnings, stats, patch_confidences
            )

        return {
            'patch_predictions': patch_predictions,
            'patch_confidences': patch_confidences,
            'flood_map': flood_map,
            'conf_map': conf_map,
            'statistics': stats,
            'warnings': all_warnings,
            'config': {
                'storm_type': storm_type,
                'depth_mm': depth_mm,
                'tpeak': tpeak,
            },
            'conditioning_vector': cond_vec,
        }

    def predict_batch(
        self,
        test_configs: List[Dict],
        n_h: Optional[int] = None,
        n_w: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        strict: bool = False,
    ) -> List[Dict]:
        """
        Run predictions for multiple storm scenarios.
        
        Args:
            test_configs: List of dicts with keys:
                - 'storm_type': str
                - 'depth_mm': float
                - 'tpeak': Optional[float]
            n_h, n_w: Grid dimensions (passed to all predictions)
            batch_size: inference batch size (uses default if None)
            verbose: print progress
            strict: validation strictness
            
        Returns:
            List of prediction results (one per config)
        """
        if batch_size is None:
            batch_size = self.batch_size

        results = []
        
        for i, config in enumerate(test_configs):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Test {i+1}/{len(test_configs)}")
                print(f"{'='*60}")
            
            try:
                result = self.predict(
                    storm_type=config['storm_type'],
                    depth_mm=config['depth_mm'],
                    tpeak=config.get('tpeak', None),
                    n_h=n_h,
                    n_w=n_w,
                    batch_size=batch_size,
                    verbose=verbose,
                    strict=strict,
                )
                results.append(result)
            except ValueError as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'error': str(e),
                    'config': config,
                })
        
        return results

    def _compute_statistics(self, patch_predictions: np.ndarray) -> Dict:
        """Compute flood class distribution statistics."""
        stats = {}
        total = patch_predictions.size
        
        for cls_id, cls_name in FLOOD_CLASS_NAMES.items():
            count = (patch_predictions == cls_id).sum()
            pct = 100 * count / total if total > 0 else 0
            stats[cls_name] = {
                'count': int(count),
                'percentage': float(pct),
            }
        
        return stats

    def _print_summary(
        self,
        storm_type: str,
        depth_mm: float,
        tpeak: Optional[float],
        warnings: List[str],
        stats: Dict,
        patch_confidences: Optional[np.ndarray] = None,
    ) -> None:
        """Pretty-print prediction summary with confidence stats."""
        print(f"\n{'='*60}")
        print(f"PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Storm type     : {storm_type}")
        print(f"  Depth          : {depth_mm} mm")
        if tpeak is not None:
            print(f"  tpeak          : {tpeak}")
        
        if warnings:
            print(f"  ⚠ Warnings     : {len(warnings)}")
            for w in warnings:
                print(f"    - {w}")
        else:
            print(f"  ✓ No warnings")
        
        # Confidence statistics
        if patch_confidences is not None:
            frac_low = (patch_confidences < 0.5).mean() * 100
            print(f"\n  Confidence stats:")
            print(f"    Mean     : {patch_confidences.mean():.4f}")
            print(f"    Std      : {patch_confidences.std():.4f}")
            print(f"    Min      : {patch_confidences.min():.4f}")
            print(f"    Max      : {patch_confidences.max():.4f}")
            print(f"    Uncertain: {frac_low:.1f}%  (conf < 0.5)")
        
        print(f"\n  Flood class distribution:")
        for cls_name, data in stats.items():
            pct_str = f"{data['percentage']:>5.1f}%"
            count_str = f"{data['count']:>7,}"
            bar_width = int(data['percentage'] / 2)
            bar = '█' * bar_width
            print(f"    {cls_name:<12}: {count_str} px  {pct_str}  {bar}")
        print(f"{'='*60}\n")