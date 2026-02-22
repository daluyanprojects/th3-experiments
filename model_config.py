import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from vit_architecture import ViTFloodClassifier, count_parameters


DATASET_INFO: Dict[str, Any] = {
    'training_samples':     None,
    'test_samples':         None,
    'num_train_scenarios':  None,
    'num_test_scenarios':   None,
    'patches_per_scenario': None,
    'patch_size':           None,
    'channels':             None,   
    'input_channels':       None,   
    'num_classes':          5,     
    'rainfall_timesteps':   None,
}

CHANNEL_NAMES = []  


def update_dataset_info(spatial_patches, train_dataset, test_dataset, train_data, drainage_resized, soil_resized, channel_names: Optional[list] = None):
    global DATASET_INFO, CHANNEL_NAMES
    DATASET_INFO.update({
        'training_samples':     train_dataset['total_samples'],
        'test_samples':         test_dataset['total_samples'],
        'num_train_scenarios':  train_dataset['num_scenarios'],
        'num_test_scenarios':   test_dataset['num_scenarios'],
        'patches_per_scenario': train_dataset['patches_per_scenario'],
        'patch_size':           spatial_patches.shape[-1],
        'input_channels':       spatial_patches.shape[1],
        'rainfall_timesteps':   train_data['rainfall_sequences'][0].shape[0],
        'channels': {
            'dem':          1,
            'infiltration': 1,
            'landuse':      1,
            'drainage':     len(drainage_resized),
            'soil':         len(soil_resized),
        },
    })

    print("\nDATASET_INFO updated dynamically:")
    print("-" * 70)
    for key, val in DATASET_INFO.items():
        print(f"  {key:<25}: {val}")

    print(f"\n✓ DATASET_INFO ready — {DATASET_INFO['input_channels']} channels, "
          f"patch_size={DATASET_INFO['patch_size']}, "
          f"timesteps={DATASET_INFO['rainfall_timesteps']}")


CONFIG_BASE = {
    'name':            'Base ViT',
    'embed_dim':       256,
    'num_layers':      4,
    'num_heads':       8,
    'mlp_ratio':       2.0,
    'dropout':         0.1,
    'rainfall_method': 'conv',
}


def create_model_from_config(config: Dict[str, Any], **override_kwargs) -> ViTFloodClassifier:
    model_params = {
        'patch_size':      DATASET_INFO['patch_size'],
        'embed_dim':       config['embed_dim'],
        'num_layers':      config['num_layers'],
        'num_heads':       config['num_heads'],
        'mlp_ratio':       config['mlp_ratio'],
        'dropout':         config['dropout'],
        'num_classes':     DATASET_INFO['num_classes'],
        'in_channels':     DATASET_INFO['input_channels'],
        'rainfall_method': config.get('rainfall_method', 'conv'),
        'num_timesteps':   DATASET_INFO['rainfall_timesteps'],
    }
    model_params.update(override_kwargs)
    return ViTFloodClassifier(**model_params)

def create_base_model(**kwargs)              -> ViTFloodClassifier:
    return create_model_from_config(CONFIG_BASE, **kwargs)

def compare_all_configurations():
    ch = DATASET_INFO['channels']
    print("\n" + "="*95)
    print("MODEL CONFIGURATION COMPARISON")
    print("="*95)
    print(f"  Input channels : {DATASET_INFO['input_channels']}  "
          f"(DEM {ch['dem']} + Infiltration {ch['infiltration']} + "
          f"Landuse {ch['landuse']} + Drainage {ch['drainage']} + Soil {ch['soil']})")
    print(f"  Patch size     : {DATASET_INFO['patch_size']}×{DATASET_INFO['patch_size']}")
    print(f"  Timesteps      : {DATASET_INFO['rainfall_timesteps']}")
    print(f"  Classes        : {DATASET_INFO['num_classes']}  "
          f"(No Flood / Light / Moderate / Heavy / Extreme)")
    print(f"  Train samples  : {DATASET_INFO['training_samples']:,}")
    print(f"  Test samples   : {DATASET_INFO['test_samples']:,}")
    print("="*95)

    configs = [CONFIG_BASE]

    print(f"\n{'Config':<30} {'Params':>12}  {'Embed':<8} "
          f"{'Layers':<8} {'Heads':<8} {'Rain':<10} {'Dropout'}")
    print("-"*95)

    for cfg in configs:
        model = create_model_from_config(cfg)
        total_params, _ = count_parameters(model)
        print(f"{cfg['name']:<30} {total_params:>12,}  "
              f"{cfg['embed_dim']:<8} {cfg['num_layers']:<8} "
              f"{cfg['num_heads']:<8} {cfg['rainfall_method']:<10} "
              f"{cfg['dropout']:.2f}")

    print("="*95)


def print_model_details(model: ViTFloodClassifier):
    total_params, trainable_params = count_parameters(model)
    ch = DATASET_INFO['channels']
    actual_in = model.patch_embedding.projection.in_channels

    print("\n" + "="*70)
    print("DETAILED MODEL ARCHITECTURE")
    print("="*70)
    print(f"\nModel              : {model.__class__.__name__}")
    print(f"Total parameters   : {total_params:,}")
    print(f"Trainable          : {trainable_params:,}")

    print(f"\nSpatial Input — {actual_in} channels:")
    idx = 0
    groups = [
        ('DEM',          ch['dem']),
        ('Infiltration', ch['infiltration']),
        ('Landuse',      ch['landuse']),
        ('Drainage',     ch['drainage']),
        ('Soil',         ch['soil']),
    ]
    for group_name, n in groups:
        names = CHANNEL_NAMES[idx: idx + n]
        if n == 1:
            print(f"  [{idx:2d}]      {group_name}")
        else:
            print(f"  [{idx:2d}-{idx+n-1:2d}]   {group_name} × {n}  {names}")
        idx += n

    print(f"\nPatch Embedding:")
    print(f"  Conv2d({actual_in}, {model.embed_dim}, "
          f"kernel={model.patch_size}, stride={model.patch_size})")
    print(f"  → (batch, 1, {model.embed_dim})")

    print(f"\nRainfall Embedding:")
    print(f"  Input  : (batch, {model.num_timesteps})")
    print(f"  Method : {model.rainfall_embedding.method}")
    print(f"  → (batch, 1, {model.embed_dim})")

    print(f"\nTransformer:")
    print(f"  Tokens  : 2  [rainfall | patch]")
    print(f"  Layers  : {len(model.transformer.layers)}")
    print(f"  Embed   : {model.embed_dim}")
    print(f"  Heads   : {model.transformer.layers[0].attention.num_heads}")
    print(f"  MLP     : {model.transformer.layers[0].mlp[0].out_features // model.embed_dim}×")

    print(f"\nClassifier:")
    print(f"  Mean pool → Linear({model.embed_dim}, {model.num_classes})")
    print(f"  Classes: No Flood | Light | Moderate | Heavy | Extreme")
    print("="*70)