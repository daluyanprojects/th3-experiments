from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

@dataclass
class TrainConfig:
    # Architecture
    patch_size:         int = 4
    embed_dim:          int = 256
    num_heads:          int = 8
    num_layers:         int = 4
    mlp_ratio:          float = 2.0
    dropout:            float = 0.1
    
    # Rainfall embedding
    rainfall_method:    str = 'conv'
    rainfall_hidden:    int = 64
    
    # NEW: Conditioning embedding
    use_conditioning:   bool = True        # Whether to use conditioning vectors
    conditioning_dim:   int = 4            # Dimension of conditioning vector
    conditioning_hidden: int = 64          # Hidden size for conditioning MLP
    
    # Positional encoding
    learnable_pos_enc:  bool = True
    
    # Training
    batch_size:         int = 1024
    num_epochs:         int = 20
    num_folds:          int = 2
    lr:                 float = 3e-4
    weight_decay:       float = 1e-4
    grad_clip_norm:     float = 1.0
    warmup_epochs:      int   = 10
    min_lr:              float = 1e-6
    
    # Loss
    loss_type:          str  = 'combined'              # 'or 'focal'
    focal_gamma:        float = 2.0
    label_smoothing:    float = 0.0
    weight_power:       float = 0.5        # For class weight computation
    ce_weight:          float = 0.9
    dice_weight:        float = 0.1
    
    # Evaluation
    checkpoint_metric:  str = 'macro_f1'   # 'macro_f1' or 'accuracy'
    
    # Data
    num_classes:        int = 5
    spatial_channels:   int = 3
    rainfall_timesteps: int = 13
    
    # Output
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