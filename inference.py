"""
Web inference entry point for EXP-DES-2.
Loads a trained ViTFloodClassifier and runs flood prediction from user-defined storm parameters and spatial toggles.
"""
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from hyetograph import build_conditioning_vector
from vit import ViTFloodClassifier
from model_config import make_model
from config import DatasetConfig, ModelConfig


FLOOD_CLASS_NAMES = {
    0: 'No Flood',
    1: 'Light',
    2: 'Moderate',
    3: 'Heavy',
    4: 'Extreme',
}


def save_spatial_data(spatial_patches : np.ndarray, save_path: str = 'checkpoints/spatial_data.npz') -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(save_path, spatial_patches=spatial_patches)
    print(f"✓ Spatial data saved → {save_path}")
    print(f"  spatial_patches shape: {spatial_patches.shape}")


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

        # ── Load spatial patches ──────────────────────────────────────────────
        data = np.load(spatial_data_path)
        self.spatial_patches = data['spatial_patches']   # (6400, 11, 4, 4)
        self.patches_per_map = self.spatial_patches.shape[0]

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

        model = make_model(data_cfg, model_cfg)

        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device)

        print(f"  Best macro F1 (saved): {ckpt.get('best_macro_f1', 'N/A'):.4f}")
        return model

    def _apply_channel_masking(self, spatial_patch : np.ndarray, hasDrainage: bool, hasSoil: bool,) -> np.ndarray:
        patch = spatial_patch.copy()
        if not hasDrainage:
            patch[self.drain_channels] = 0.0
        if not hasSoil:
            patch[self.soil_channels]  = 0.0
        return patch

    def _reconstruct_map(self, patch_predictions : np.ndarray, map_size: Tuple[int, int] = (320, 320)) -> np.ndarray:
        H, W       = map_size
        n_h        = H // self.patch_size
        n_w        = W // self.patch_size
        patch_grid = patch_predictions.reshape(n_h, n_w)
        flood_map  = np.repeat(
            np.repeat(patch_grid, self.patch_size, axis=0),
            self.patch_size, axis=1
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

        # Print any warnings
        if warnings:
            print("\n⚠ Input warnings:")
            for w in warnings:
                print(f"  {w}")

        # ── 2. Broadcast conditioning to all patches ──────────────────────────
        # Shape: (6400, 19) — same vector for every patch in this scenario
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
            ).float().to(self.device)                        # (B, 19)

            logits, _ = self.model(spatial_batch, cond_batch)
            preds     = logits.argmax(dim=1).cpu().numpy()   # (B,)
            patch_predictions.append(preds)

        patch_predictions = np.concatenate(patch_predictions)  # (6400,)

        # ── 5. Reconstruct full map ───────────────────────────────────────────
        flood_map = self._reconstruct_map(patch_predictions, map_size)

        # ── 6. Summary ────────────────────────────────────────────────────────
        self._print_prediction_summary(flood_map, storm_type, depth_mm,
                                       tpeak, hasDrainage, hasSoil, warnings)

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
        for cls_id, cls_name in FLOOD_CLASS_NAMES.items():
            count = (flood_map == cls_id).sum()
            pct   = 100 * count / total_pixels
            print(f"    Class {cls_id} ({cls_name:<10}): "
                  f"{count:>7,} px  ({pct:>5.1f}%)")
        print(f"{'='*60}")