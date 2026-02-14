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
    'num_classes': 5  # Flood categories: 0-4
}


# =============================================================================
# RECOMMENDED MODEL CONFIGURATIONS
# =============================================================================

"""
For 96K training samples, we recommend starting with SMALL to MEDIUM models.
Large models (>20M params) may overfit unless you use heavy regularization.
"""

# Configuration 1: TINY (For quick experiments and debugging)
CONFIG_TINY = {
    'name': 'Tiny ViT',
    'patch_size': 4,
    'embed_dim': 128,
    'num_layers': 4,
    'num_heads': 4,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    'description': 'Fast training, good for debugging',
    'expected_params': '~500K',
    'use_case': 'Quick experiments, debugging, baseline'
}

# Configuration 2: SMALL (RECOMMENDED STARTING POINT)
CONFIG_SMALL = {
    'name': 'Small ViT',
    'patch_size': 4,
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    'description': 'Good balance for 96K samples',
    'expected_params': '~3M',
    'use_case': 'Main experiments, good starting point'
}

# Configuration 3: MEDIUM (If small underfits)
CONFIG_MEDIUM = {
    'name': 'Medium ViT',
    'patch_size': 4,
    'embed_dim': 384,
    'num_layers': 8,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'dropout': 0.15,  # Slightly higher dropout
    'description': 'More capacity if needed',
    'expected_params': '~8M',
    'use_case': 'If small model underfits'
}

# Configuration 4: BASE (If you need more capacity)
CONFIG_BASE = {
    'name': 'Base ViT',
    'patch_size': 4,
    'embed_dim': 512,
    'num_layers': 12,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'dropout': 0.2,  # Higher dropout to prevent overfitting
    'description': 'High capacity, may need regularization',
    'expected_params': '~20M',
    'use_case': 'If medium model underfits, use with augmentation'
}

# Configuration 5: REGULARIZED SMALL (If overfitting occurs)
CONFIG_SMALL_REGULARIZED = {
    'name': 'Small ViT (Regularized)',
    'patch_size': 4,
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'dropout': 0.25,  # Higher dropout
    'description': 'Same as small but with more dropout',
    'expected_params': '~3M',
    'use_case': 'If small model overfits'
}

# Configuration 6: COMPACT (If you want faster training)
CONFIG_COMPACT = {
    'name': 'Compact ViT',
    'patch_size': 4,
    'embed_dim': 192,
    'num_layers': 6,
    'num_heads': 6,
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    'description': 'Faster than small, still reasonable capacity',
    'expected_params': '~2M',
    'use_case': 'If training time is a concern'
}


# =============================================================================
# MODEL FACTORY FUNCTIONS FOR EACH CONFIGURATION
# =============================================================================

def create_model_from_config(config: Dict[str, Any], **override_kwargs) -> ViTFloodClassifier:
    """
    Create a ViT model from a configuration dictionary.
    """
    model_params = {
        'patch_size': config['patch_size'],
        'embed_dim': config['embed_dim'],
        'num_layers': config['num_layers'],
        'num_heads': config['num_heads'],
        'mlp_ratio': config['mlp_ratio'],
        'dropout': config['dropout'],
        'num_classes': DATASET_INFO['num_classes'],
        'in_channels': DATASET_INFO['input_channels']
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
    print("\n" + "="*80)
    print("MODEL CONFIGURATION COMPARISON")
    print("="*80)
    print(f"Dataset: {DATASET_INFO['training_samples']:,} training samples")
    print(f"Patch size: {DATASET_INFO['patch_size']}x{DATASET_INFO['patch_size']}")
    print(f"Input channels: {DATASET_INFO['input_channels']}")
    print(f"Output classes: {DATASET_INFO['num_classes']}")
    print("="*80)
    
    configs = [CONFIG_TINY, CONFIG_COMPACT, CONFIG_SMALL, CONFIG_SMALL_REGULARIZED, CONFIG_MEDIUM, CONFIG_BASE]
    
    print(f"\n{'Config':<25} {'Params':<12} {'Embed':<8} {'Layers':<8} {'Heads':<8} {'Dropout':<10}")
    print("-"*80)
    
    for config in configs:
        model = create_model_from_config(config)
        total_params, _ = count_parameters(model)
                
        print(f"{config['name']:<25} {total_params:>10,}  "
              f"{config['embed_dim']:<8} {config['num_layers']:<8} "
              f"{config['num_heads']:<8} {config['dropout']:<10.2f}")
