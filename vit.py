import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class RainfallEncoder(nn.Module):
    def __init__(self, num_timesteps: int, embed_dim: int, hidden_dim: int,
                 method: str, dropout: float):
        super().__init__()
        self.method        = method
        self.num_timesteps = num_timesteps
        self.embed_dim     = embed_dim

        if method == 'conv':
            self.encoder = nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.projection = nn.Linear(hidden_dim, embed_dim)
        else:
            raise ValueError(f"Unknown rainfall method: {method}. Only 'conv' is supported.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.unsqueeze(1)        
        out = self.encoder(x)        
        out = self.projection(out)    
        out = self.dropout(out)
        return out.unsqueeze(1)       


# ── Patch Embedding ───────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=in_channels, out_channels=embed_dim,
            kernel_size=patch_size,  stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)   # (B, embed_dim, 1, 1)
        x = x.flatten(2)         # (B, embed_dim, 1)
        x = x.transpose(1, 2)   # (B, 1, embed_dim)
        return x


# ── Positional Encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, num_positions: int, embed_dim: int, learnable: bool = True):
        super().__init__()
        if learnable:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, embed_dim) * 0.02)
        else:
            self.register_buffer('pos_embedding', self._sinusoidal(num_positions, embed_dim))

    def _sinusoidal(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, num_positions, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


# ── Transformer Block ─────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1     = nn.LayerNorm(embed_dim)
        self.norm2     = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_norm   = self.norm1(x)
        attn_out, attn_weights = self.attention(
            x_norm, x_norm, x_norm, need_weights=return_attention
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights if return_attention else None


# ── Transformer Encoder ───────────────────────────────────────────────────────
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int,
                 mlp_ratio: float, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[List]]:
        attn_maps = [] if return_attention else None
        for layer in self.layers:
            x, attn = layer(x, return_attention=return_attention)
            if return_attention and attn is not None:
                attn_maps.append(attn)
        return self.norm(x), attn_maps


# ── Classification Head ───────────────────────────────────────────────────────
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x[:, 1, :])   # always reads patch token at index 1


# ── Full Model ────────────────────────────────────────────────────────────────
class ViTFloodClassifier(nn.Module):
    def __init__(
        self,
        spatial_channels:   int,
        spatial_patch_size: int,
        rainfall_timesteps: int,
        num_classes:        int,
        embed_dim:          int,
        num_layers:         int,
        num_heads:          int,
        mlp_ratio:          float,
        dropout:            float,
        rainfall_method:    str,
        learnable_pos_enc:  bool,
        rainfall_hidden:    int,
    ):
        super().__init__()
        self.patch_size    = spatial_patch_size
        self.embed_dim     = embed_dim
        self.num_classes   = num_classes
        self.num_timesteps = rainfall_timesteps

        self.patch_embedding  = PatchEmbedding(spatial_channels, spatial_patch_size, embed_dim)
        self.rainfall_encoder = RainfallEncoder(rainfall_timesteps, embed_dim,
                                                rainfall_hidden, rainfall_method, dropout)
        self.pos_encoding     = PositionalEncoding(2, embed_dim, learnable_pos_enc)
        self.transformer      = TransformerEncoder(num_layers, embed_dim, num_heads, mlp_ratio, dropout)
        self.classifier       = ClassificationHead(embed_dim, num_classes, dropout)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, spatial_patch: torch.Tensor, rainfall_sequence: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[List]]:
        patch_token = self.patch_embedding(spatial_patch)      
        rain_token  = self.rainfall_encoder(rainfall_sequence)
        tokens      = torch.cat([rain_token, patch_token], dim=1)  
        tokens      = self.pos_encoding(tokens)
        encoded, attn_maps = self.transformer(tokens, return_attention=return_attention)
        logits      = self.classifier(encoded)
        return logits, attn_maps if return_attention else None


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_summary(model: ViTFloodClassifier) -> str:
    total, trainable = count_parameters(model)
    lines = [
        "=" * 70,
        "MODEL SUMMARY",
        "=" * 70,
        f"Model             : {model.__class__.__name__}",
        f"Total parameters  : {total:,}",
        f"Trainable         : {trainable:,}",
        f"Non-trainable     : {total - trainable:,}",
        f"Rainfall timesteps: {model.num_timesteps}",
        f"Rainfall method   : {model.rainfall_encoder.method}",
        "=" * 70,
    ]
    return "\n".join(lines)