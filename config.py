from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

@dataclass
class ModelConfig:
    in_channels:       int   = 3
    patch_size:        int   = 4
    num_classes:       int   = 5
    embed_dim:         int   = 256
    num_layers:        int   = 4
    num_heads:         int   = 8
    mlp_ratio:         float = 2.0
    dropout:           float = 0.1
    learnable_pos_enc: bool  = True
    rainfall_method:   str   = 'conv'
    num_timesteps:     int   = 26
    rainfall_hidden:   int   = 64

@dataclass
class TrainConfig:
    batch_size:       int   = 1024
    num_epochs:       int   = 10
    num_folds:        int   = 2
    lr:               float = 3e-4
    weight_decay:     float = 1e-4
    num_classes:      int   = 5
    betas:            tuple = (0.9, 0.999)
    output_dir:       Path  = Path('./checkpoints')

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            d = asdict(self)
            d['output_dir'] = str(self.output_dir)
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path):
        with open(path) as f:
            d = json.load(f)
        d['output_dir'] = Path(d['output_dir'])
        return cls(**d)

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            d = {
                'model': asdict(self.model),
                'train': {**asdict(self.train), 'output_dir': str(self.train.output_dir)},
            }
            json.dump(d, f, indent=2)