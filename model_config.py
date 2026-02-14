"""
Model Configuration for ViT with Temporal Rainfall Sequences
=============================================================

UPDATED: Now supports 13-timestep rainfall sequences instead of scalar values
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from vit_architecture import ViTFloodClassifier, count_parameters

DATASET_INFO = {
    'training_samples': 96_000,
    'test_samples': 32_000,
    'num_train_scenarios': 15,
    'num_test_scenarios': 5,
    'patches_per_scenario': 6_400,
    'patch_size': 4,  # 4x4 patches
    'input_channels': 3,  # DEM, Infiltration, Landuse
    'num_classes': 5,  # Flood categories: 0-4
    'rainfall_timesteps': 13  # NEW: Temporal rainfall sequence length
}

# Configuration 1: TINY (For quick experiments and debugging)
CONFIG_TINY = {
    'name': 'Tiny ViT (Temporal)',
    'patch_size': 4,
    'embed_dim': 128,
    'num_layers': 4,
    'num_heads': 4,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    'rainfall_method': 'conv',  # NEW: 'conv', 'mlp', or 'attention'
    'description': 'Fast training, good for debugging',
    'expected_params': '~500K',
    'use_case': 'Quick experiments, debugging, baseline'
}

# Configuration 2: SMALL (RECOMMENDED STARTING POINT)
CONFIG_SMALL = {
    'name': 'Small ViT (Temporal)',
    'patch_size': 4,
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    'rainfall_method': 'conv',  # NEW: Recommended for temporal patterns
    'description': 'Good balance for 96K samples with temporal rainfall',
    'expected_params': '~3M',
    'use_case': 'Main experiments, good starting point'
}

# Configuration 3: MEDIUM (If small underfits)
CONFIG_MEDIUM = {
    'name': 'Medium ViT (Temporal)',
    'patch_size': 4,
    'embed_dim': 384,
    'num_layers': 8,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'dropout': 0.15,
    'rainfall_method': 'conv',
    'description': 'More capacity if needed',
    'expected_params': '~8M',
    'use_case': 'If small model underfits'
}

# Configuration 4: BASE (If you need more capacity)
CONFIG_BASE = {
    'name': 'Base ViT (Temporal)',
    'patch_size': 4,
    'embed_dim': 512,
    'num_layers': 12,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'dropout': 0.2,
    'rainfall_method': 'conv',
    'description': 'High capacity, may need regularization',
    'expected_params': '~20M',
    'use_case': 'If medium model underfits, use with augmentation'
}

# Configuration 5: REGULARIZED SMALL (If overfitting occurs)
CONFIG_SMALL_REGULARIZED = {
    'name': 'Small ViT (Reg, Temp)',
    'patch_size': 4,
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'dropout': 0.25,
    'rainfall_method': 'conv',
    'description': 'Same as small but with more dropout',
    'expected_params': '~3M',
    'use_case': 'If small model overfits'
}

# Configuration 6: COMPACT (If you want faster training)
CONFIG_COMPACT = {
    'name': 'Compact ViT (Temporal)',
    'patch_size': 4,
    'embed_dim': 192,
    'num_layers': 6,
    'num_heads': 6,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    'rainfall_method': 'conv',
    'description': 'Faster than small, still reasonable capacity',
    'expected_params': '~2M',
    'use_case': 'If training time is a concern'
}


def create_model_from_config(config: Dict[str, Any], **override_kwargs) -> ViTFloodClassifier:
    model_params = {
        'patch_size': config['patch_size'],
        'embed_dim': config['embed_dim'],
        'num_layers': config['num_layers'],
        'num_heads': config['num_heads'],
        'mlp_ratio': config['mlp_ratio'],
        'dropout': config['dropout'],
        'num_classes': DATASET_INFO['num_classes'],
        'in_channels': DATASET_INFO['input_channels'],
        'rainfall_method': config.get('rainfall_method', 'conv'), 
        'num_timesteps': DATASET_INFO['rainfall_timesteps'] 
    }
    
    # Override with any custom parameters
    model_params.update(override_kwargs)
    
    return ViTFloodClassifier(**model_params)


def create_tiny_model(**kwargs) -> ViTFloodClassifier:
    return create_model_from_config(CONFIG_TINY, **kwargs)


def create_small_model(**kwargs) -> ViTFloodClassifier:
    return create_model_from_config(CONFIG_SMALL, **kwargs)


def create_medium_model(**kwargs) -> ViTFloodClassifier:
    return create_model_from_config(CONFIG_MEDIUM, **kwargs)


def create_base_model(**kwargs) -> ViTFloodClassifier:
    return create_model_from_config(CONFIG_BASE, **kwargs)


def create_compact_model(**kwargs) -> ViTFloodClassifier:
    return create_model_from_config(CONFIG_COMPACT, **kwargs)


def create_small_regularized_model(**kwargs) -> ViTFloodClassifier:
    return create_model_from_config(CONFIG_SMALL_REGULARIZED, **kwargs)


def compare_all_configurations():
    print("\n" + "="*90)
    print("MODEL CONFIGURATION COMPARISON (WITH TEMPORAL RAINFALL)")
    print("="*90)
    print(f"Dataset: {DATASET_INFO['training_samples']:,} training samples")
    print(f"Patch size: {DATASET_INFO['patch_size']}x{DATASET_INFO['patch_size']}")
    print(f"Input channels: {DATASET_INFO['input_channels']} (DEM, Infiltration, Landuse)")
    print(f"Rainfall timesteps: {DATASET_INFO['rainfall_timesteps']} (temporal sequence)")
    print(f"Output classes: {DATASET_INFO['num_classes']}")
    print("="*90)
    
    configs = [
        CONFIG_TINY, 
        CONFIG_COMPACT, 
        CONFIG_SMALL, 
        CONFIG_SMALL_REGULARIZED,
        CONFIG_MEDIUM, 
        CONFIG_BASE
    ]
    
    print(f"\n{'Config':<30} {'Params':<12} {'Embed':<8} {'Layers':<8} {'Heads':<8} {'Rain Method':<12} {'Dropout':<10}")
    print("-"*90)
    
    for config in configs:
        model = create_model_from_config(config)
        total_params, _ = count_parameters(model)
        
        rain_method = config.get('rainfall_method', 'conv')
                
        print(f"{config['name']:<30} {total_params:>10,}  "
              f"{config['embed_dim']:<8} {config['num_layers']:<8} "
              f"{config['num_heads']:<8} {rain_method:<12} {config['dropout']:<10.2f}")
    


def print_model_details(model: ViTFloodClassifier):
    print("\n" + "="*70)
    print("DETAILED MODEL ARCHITECTURE")
    print("="*70)
    
    total_params, trainable_params = count_parameters(model)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(f"\nInput Processing:")
    print(f"  Spatial patches: ({model.patch_embedding.projection.in_channels}, "
          f"{model.patch_size}, {model.patch_size}) → ({model.embed_dim},)")
    print(f"  Rainfall sequence: ({model.num_timesteps},) → ({model.embed_dim},)")
    print(f"  Rainfall method: {model.rainfall_embedding.method}")
    
    print(f"\nTransformer:")
    print(f"  Layers: {len(model.transformer.layers)}")
    print(f"  Embedding dimension: {model.embed_dim}")
    print(f"  Attention heads: {model.transformer.layers[0].attention.num_heads}")
    print(f"  MLP ratio: {model.transformer.layers[0].mlp[0].out_features / model.embed_dim:.1f}x")
    
    print(f"\nClassifier:")
    print(f"  Input: ({model.embed_dim},)")
    print(f"  Output: ({model.num_classes},) classes")
    
    print("="*70)