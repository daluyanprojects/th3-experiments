# config.py
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

@dataclass
class TrainConfig:
    patch_size:         int = 4
    embed_dim:          int = 256
    num_heads:          int = 8
    num_layers:         int = 4
    mlp_ratio:          float = 2.0
    dropout:            float = 0.1
    rainfall_method:    str = 'conv'
    spatial_method:     str = 'conv'
    pooling_method:     str = 'mean'
    learnable_pos_enc:  bool = True
    batch_size:         int = 1024
    num_epochs:         int = 50
    num_folds:          int = 5
    lr:                 float = 3e-4
    weight_decay:       float = 1e-4
    grad_clip_norm:     float = 1.0
    loss_type:          str = 'ce'          # 'ce' or 'focal'
    focal_gamma:        float = 2.0
    label_smoothing:    float = 0.0
    weight_power:       float = 0.5
    checkpoint_metric:  str = 'macro_f1'
    num_classes:        int = 5
    spatial_channels:   int = 3
    rainfall_timesteps: int = 13
    output_dir:         Path = Path('./outputs')

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({k: str(v) if isinstance(v, Path) else v 
                       for k, v in asdict(self).items()}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        d['output_dir'] = Path(d['output_dir'])
        return cls(**d)