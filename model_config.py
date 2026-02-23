from vit_architecture import ViTFloodClassifier, count_parameters
from config import Config, ModelConfig, DatasetConfig

def make_model(data_cfg: DatasetConfig, model_cfg: ModelConfig) -> ViTFloodClassifier:
    return ViTFloodClassifier(
        in_channels       = data_cfg.input_channels,
        patch_size        = data_cfg.patch_size,
        num_classes       = data_cfg.num_classes,
        num_timesteps     = data_cfg.rainfall_timesteps,
        embed_dim         = model_cfg.embed_dim,
        num_layers        = model_cfg.num_layers,
        num_heads         = model_cfg.num_heads,
        mlp_ratio         = model_cfg.mlp_ratio,
        dropout           = model_cfg.dropout,
        rainfall_method   = model_cfg.rainfall_method,
        learnable_pos_enc = model_cfg.learnable_pos_enc,
        use_cls_token     = model_cfg.use_cls_token,
    )


def model_factory(data_cfg: DatasetConfig, model_cfg: ModelConfig):
    return lambda: make_model(data_cfg, model_cfg)


def print_model_details(model: ViTFloodClassifier, data_cfg: DatasetConfig,
                         channel_names: list):
    total_params, trainable_params = count_parameters(model)
    ch = data_cfg.channel_breakdown

    print("\n" + "="*70)
    print("DETAILED MODEL ARCHITECTURE")
    print("="*70)
    print(f"Total parameters   : {total_params:,}")
    print(f"Trainable          : {trainable_params:,}")

    print(f"\nSpatial Input — {data_cfg.input_channels} channels:")
    idx = 0
    groups = [
        ('DEM',          ch['dem']),
        ('Infiltration', ch['infiltration']),
        ('Landuse',      ch['landuse']),
        ('Drainage',     ch['drainage']),
        ('Soil',         ch['soil']),
    ]
    for group_name, n in groups:
        names = channel_names[idx: idx + n]
        if n == 1:
            print(f"  [{idx:2d}]      {group_name}")
        else:
            print(f"  [{idx:2d}-{idx+n-1:2d}]   {group_name} × {n}  {names}")
        idx += n

    print(f"\nTransformer: {len(model.transformer.layers)} layers  |  "
          f"embed={model.embed_dim}  |  "
          f"heads={model.transformer.layers[0].attention.num_heads}")
    print("="*70)