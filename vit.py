import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math


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
            raise ValueError(f"Unknown rainfall method: {method}.")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.unsqueeze(1)
        out = self.encoder(x)
        out = self.projection(out)
        out = self.dropout(out)
        return out.unsqueeze(1)


class ConditioningEncoder(nn.Module):
    def __init__(self, conditioning_dim: int, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conditioning_dim = conditioning_dim
        self.embed_dim = embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)       # (batch, embed_dim)
        return out.unsqueeze(1)  # (batch, 1, embed_dim)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=in_channels, out_channels=embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, max_positions: int, embed_dim: int, learnable: bool = True):
        super().__init__()
        self.max_positions = max_positions
        if learnable:
            self.pos_embedding = nn.Parameter(torch.randn(1, max_positions, embed_dim) * 0.02)
        else:
            self.register_buffer('pos_embedding', self._sinusoidal(max_positions, embed_dim))

    def _sinusoidal(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, num_positions, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.size(1)
        return x + self.pos_embedding[:, :num_tokens, :]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1     = nn.LayerNorm(embed_dim)
        self.norm2     = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                               dropout=dropout, batch_first=True)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_norm   = self.norm1(x)
        attn_out, attn_weights = self.attention(x_norm, x_norm, x_norm,
                                                need_weights=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights if return_attention else None


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


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float, pooling: str = 'first'):
        super().__init__()
        self.pooling = pooling
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'first':
            pooled = x[:, 0, :]
        elif self.pooling == 'mean':
            pooled = x.mean(dim=1)
        elif self.pooling == 'last':
            pooled = x[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.mlp(pooled)


class ViTFloodClassifier(nn.Module):
    def __init__(
        self,
        spatial_channels:    int,
        spatial_patch_size:  int,
        rainfall_timesteps:  int,
        num_classes:         int,
        embed_dim:           int,
        num_layers:          int,
        num_heads:           int,
        mlp_ratio:           float,
        dropout:             float,
        rainfall_method:     str,
        learnable_pos_enc:   bool,
        rainfall_hidden:     int,
        use_conditioning:    bool,
        conditioning_dim:    int,
        conditioning_hidden: int,
    ):
        super().__init__()
        self.patch_size           = spatial_patch_size
        self.embed_dim            = embed_dim
        self.num_classes          = num_classes
        self.num_timesteps        = rainfall_timesteps
        self.use_conditioning     = use_conditioning
        self.conditioning_dim     = conditioning_dim  # stored for zero-fill fallback

        self.patch_embedding  = PatchEmbedding(spatial_channels, spatial_patch_size, embed_dim)
        self.rainfall_encoder = RainfallEncoder(rainfall_timesteps, embed_dim,
                                                rainfall_hidden, rainfall_method, dropout)

        if use_conditioning:
            self.conditioning_encoder = ConditioningEncoder(
                conditioning_dim, embed_dim, conditioning_hidden, dropout
            )

        max_tokens       = 3 if use_conditioning else 2
        self.pos_encoding = PositionalEncoding(max_tokens, embed_dim, learnable_pos_enc)
        self.transformer  = TransformerEncoder(num_layers, embed_dim, num_heads, mlp_ratio, dropout)
        self.classifier   = ClassificationHead(embed_dim, num_classes, dropout, pooling='first')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        spatial_patch:     torch.Tensor,
        rainfall_sequence: torch.Tensor,
        conditioning:      Optional[torch.Tensor] = None,
        return_attention:  bool = False,
    ) -> Tuple[torch.Tensor, Optional[List]]:

        patch_token = self.patch_embedding(spatial_patch)    # (batch, 1, embed_dim)
        rain_token  = self.rainfall_encoder(rainfall_sequence)  # (batch, 1, embed_dim)

        if self.use_conditioning:
            # Zero-fill if conditioning is missing (e.g. inference without conditioning data)
            if conditioning is None:
                conditioning = torch.zeros(
                    spatial_patch.size(0), self.conditioning_dim,
                    device=spatial_patch.device, dtype=spatial_patch.dtype,
                )
            cond_token = self.conditioning_encoder(conditioning)  # (batch, 1, embed_dim)
            tokens = torch.cat([patch_token, rain_token, cond_token], dim=1)  # (batch, 3, embed_dim)

        tokens  = self.pos_encoding(tokens)
        encoded, attn_maps = self.transformer(tokens, return_attention=return_attention)
        logits  = self.classifier(encoded)

        return logits, attn_maps if return_attention else None


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_summary(model: ViTFloodClassifier) -> str:
    total, trainable = count_parameters(model)
    lines = [
        "=" * 70,
        "VITFLOODCLASSIFIER — MODEL SUMMARY",
        "=" * 70,
        f"  Spatial channels   : {model.patch_embedding.projection.in_channels}",
        f"  Patch size         : {model.patch_size}×{model.patch_size}",
        f"  Num classes        : {model.num_classes}",
        f"  Embed dim          : {model.embed_dim}",
        f"  Transformer layers : {len(model.transformer.layers)}",
        f"  Attention heads    : {model.transformer.layers[0].attention.num_heads}",
        f"  Rainfall timesteps : {model.num_timesteps}",
        f"  Rainfall method    : {model.rainfall_encoder.method}",
    ]

    if model.use_conditioning:
        lines.extend([
            "-" * 70,
            f"  Conditioning       : ✓ ENABLED (zero-fill if absent at inference)",
            f"  Conditioning dim   : {model.conditioning_encoder.conditioning_dim}",
            f"  Num tokens         : 3 (rainfall + patch + conditioning)",
        ])
    else:
        lines.extend([
            "-" * 70,
            f"  Conditioning       : ✗ DISABLED",
            f"  Num tokens         : 2 (rainfall + patch)",
        ])

    lines.extend([
        "-" * 70,
        f"  Total parameters   : {total:,}",
        f"  Trainable params   : {trainable:,}",
        "=" * 70,
    ])
    return "\n".join(lines)