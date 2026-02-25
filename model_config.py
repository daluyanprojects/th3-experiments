from vit import ViT
from config import ModelConfig

def make_model(model_cfg: ModelConfig) -> ViT:
    return ViT(
        in_channels       = model_cfg.in_channels,
        patch_size        = model_cfg.patch_size,
        num_classes       = model_cfg.num_classes,
        embed_dim         = model_cfg.embed_dim,
        num_layers        = model_cfg.num_layers,
        num_heads         = model_cfg.num_heads,
        mlp_ratio         = model_cfg.mlp_ratio,
        dropout           = model_cfg.dropout,
        learnable_pos_enc = model_cfg.learnable_pos_enc,
        rainfall_method   = model_cfg.rainfall_method,
        num_timesteps     = model_cfg.num_timesteps,
        conditioning_dim  = model_cfg.conditioning_dim,
        rainfall_hidden   = model_cfg.rainfall_hidden,
    )

def model_factory(model_cfg: ModelConfig):
    return lambda: make_model(model_cfg)