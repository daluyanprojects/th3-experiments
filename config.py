# config_cnn.py
"""
CNNTrainConfig
==============
Mirrors ViT TrainConfig but tuned for the CNN segmentation branch.

Key differences from ViT config
--------------------------------
- batch_size     : 2  (full 1152×1152 maps are large; ViT used 1024 patches)
- grad_accum     : 8  (effective batch = 16, simulating ViT's large batch)
- num_epochs     : 50 (ViT only ran 1; CNN needs more with only 35 scenarios)
- num_folds      : 5  (ViT used 2; more folds needed with only 35 scenarios)
- encoder        : 'efficientnet-b3' with ImageNet weights
- context_dim    : 256 (FiLM context vector size)
- mlp_hidden     : 128 (rainfall+conditioning MLP hidden size)
- loss           : same CombinedLoss (ce=0.9, dice=0.1)
- ignore_index   : -1  (outside-mask pixels excluded from loss + metrics)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class CNNTrainConfig:
    # ── Encoder / decoder ─────────────────────────────────────────────────────
    encoder_name    : str   = 'efficientnet-b3'
    encoder_weights : str   = 'imagenet'          # None to train from scratch
    decoder_channels: tuple = (256, 128, 64, 32, 16)

    # ── Rainfall + conditioning injection ────────────────────────────────────
    use_conditioning    : bool = True
    rainfall_dim        : int  = 13
    conditioning_dim    : int  = 4
    context_dim         : int  = 256   # FiLM context vector size
    mlp_hidden          : int  = 128   # rainfall+conditioning MLP hidden

    # ── Data ──────────────────────────────────────────────────────────────────
    num_classes      : int = 5
    spatial_channels : int = 3
    map_h            : int = 1152
    map_w            : int = 1152
    ignore_index     : int = -1       # outside-mask pixels — excluded from loss

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size       : int   = 2      # per-GPU; full maps are memory-heavy
    grad_accum_steps : int   = 8      # effective batch = batch_size × grad_accum
    num_epochs       : int   = 20
    num_folds        : int   = 2     # more folds needed with only 35 scenarios
    lr               : float = 1e-4
    weight_decay     : float = 1e-4
    grad_clip_norm   : float = 1.0
    warmup_epochs    : int   = 5
    min_lr           : float = 1e-6

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_type        : str   = 'combined'   # 'combined' or 'focal'
    ce_weight        : float = 0.9
    dice_weight      : float = 0.1
    focal_gamma      : float = 2.0
    label_smoothing  : float = 0.0
    weight_power     : float = 0.5          # class weight smoothing exponent

    # ── Evaluation ────────────────────────────────────────────────────────────
    checkpoint_metric : str = 'macro_f1'   # 'macro_f1' or 'accuracy'

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir : Path = Path('./outputs_cnn')

    # ── Derived (read-only helpers) ───────────────────────────────────────────
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    def save(self, path: Path):
        d = asdict(self)
        d['output_dir']      = str(d['output_dir'])
        d['decoder_channels']= list(d['decoder_channels'])
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'CNNTrainConfig':
        with open(path) as f:
            d = json.load(f)
        d['output_dir']       = Path(d['output_dir'])
        d['decoder_channels'] = tuple(d['decoder_channels'])
        return cls(**d)

    def summary(self):
        print("\nCNNTrainConfig")
        print("=" * 40)
        print(f"  Encoder        : {self.encoder_name} ({self.encoder_weights})")
        print(f"  Decoder ch     : {self.decoder_channels}")
        print(f"  Conditioning   : {'✓' if self.use_conditioning else '✗'}"
              f"  context_dim={self.context_dim}")
        print(f"  Batch size     : {self.batch_size} × {self.grad_accum_steps} accum"
              f" = {self.effective_batch_size} effective")
        print(f"  Epochs / Folds : {self.num_epochs} / {self.num_folds}")
        print(f"  LR             : {self.lr}  (warmup={self.warmup_epochs}, min={self.min_lr})")
        print(f"  Loss           : {self.loss_type}  ce={self.ce_weight} dice={self.dice_weight}")
        print(f"  ignore_index   : {self.ignore_index}  (outside-mask pixels)")
        print(f"  Output dir     : {self.output_dir}")
        print("=" * 40)