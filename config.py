from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional
import json

# ── Dataset info ──────────────────────
@dataclass
class DatasetConfig:
    patch_size:           int  = 4
    num_classes:          int  = 5
    rainfall_timesteps:   int  = 13
    input_channels:       int  = None 
    training_samples:     int  = None
    test_samples:         int  = None
    num_train_scenarios:  int  = None
    num_test_scenarios:   int  = None
    patches_per_scenario: int  = None
    channel_breakdown: Dict   = field(default_factory=dict) 

    def populate(self, spatial_patches, train_dataset, test_dataset,
                 train_data, drainage_resized, soil_resized):
        self.patch_size           = spatial_patches.shape[-1]
        self.input_channels       = spatial_patches.shape[1]
        self.rainfall_timesteps   = train_data['rainfall_sequences'][0].shape[0]
        self.training_samples     = train_dataset['total_samples']
        self.test_samples         = test_dataset['total_samples']
        self.num_train_scenarios  = train_dataset['num_scenarios']
        self.num_test_scenarios   = test_dataset['num_scenarios']
        self.patches_per_scenario = train_dataset['patches_per_scenario']
        self.channel_breakdown    = {
            'dem':          1,
            'infiltration': 1,
            'landuse':      1,
            'drainage':     len(drainage_resized),
            'soil':         len(soil_resized),
        }
        self._print()

    def _print(self):
        print("\nDatasetConfig populated:")
        print("-" * 70)
        for k, v in asdict(self).items():
            print(f"  {k:<25}: {v}")
        print(f"\n✓ {self.input_channels} channels, "
              f"patch_size={self.patch_size}, "
              f"timesteps={self.rainfall_timesteps}")


# ── Model architecture ────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    embed_dim:        int   = 256
    num_layers:       int   = 4
    num_heads:        int   = 8
    mlp_ratio:        float = 2.0
    dropout:          float = 0.1
    rainfall_method:  str   = 'conv'
    learnable_pos_enc: bool = True
    use_cls_token:    bool  = False


# ── Training ──────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    batch_size:      int   = 1024
    num_epochs:      int   = 50
    num_folds:       int   = 10
    lr:              float = 1e-4
    weight_decay:    float = 0.05
    betas:           tuple = (0.9, 0.999)
    warmup_epochs:   int   = 10
    min_lr:          float = 1e-6
    grad_clip_norm:  float = 1.0
    early_stop_patience: int = 10
    checkpoint_metric:   str = 'macro_f1'
    output_dir:      Path = Path('./checkpoints')

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
        d['betas'] = tuple(d['betas'])
        return cls(**d)


@dataclass
class Config:
    data:  DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig   = field(default_factory=ModelConfig)
    train: TrainConfig   = field(default_factory=TrainConfig)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            'data':  asdict(self.data),
            'model': asdict(self.model),
            'train': {**asdict(self.train), 'output_dir': str(self.train.output_dir)},
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)