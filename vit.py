import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


# ─────────────────────────────────────────────
# RAINFALL ENCODER
# Accepts (batch, 13) normalized intensity sequence
# Outputs (batch, 1, embed_dim) — a single conditioning token
# ─────────────────────────────────────────────

class RainfallEncoder(nn.Module):
    def __init__(
        self,
        num_timesteps: int,
        embed_dim: int,
        hidden_dim: int,
        method: str,
        dropout: float,
    ):
        super().__init__()
        self.method = method
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim

        if method == 'conv':
            self.encoder = nn.Sequential(
                # (batch, 1, 13) → (batch, hidden_dim, 13)
                nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                # (batch, hidden_dim, 13) → (batch, hidden_dim, 13)
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                # (batch, hidden_dim, 13) → (batch, hidden_dim, 1)
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),                          # (batch, hidden_dim)
            )
            self.projection = nn.Linear(hidden_dim, embed_dim)

        else:
            raise ValueError(f"Unknown rainfall method: {method}. Choose 'mlp', 'conv', or 'attn'.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 13) normalized rainfall intensities

        Returns:
            token: (batch, 1, embed_dim)
        """
        if self.method == 'conv':
            x = x.unsqueeze(1)                            # (batch, 1, 13)
            out = self.encoder(x)                         # (batch, hidden_dim)
            out = self.projection(out)                    # (batch, embed_dim

        out = self.dropout(out)
        return out.unsqueeze(1)                           # (batch, 1, embed_dim)


# ─────────────────────────────────────────────
# PATCH EMBEDDING
# Accepts one (4, 4, 3) patch per batch item
# Conv2d with kernel=patch_size collapses it to a single token
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Projects a single (C, patch_size, patch_size) spatial patch
    into a 1-token sequence of dimension embed_dim.

    Input:  (batch, C, patch_size, patch_size)
    Output: (batch, 1, embed_dim)
    """

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)    # (batch, embed_dim, 1, 1)
        x = x.flatten(2)         # (batch, embed_dim, 1)
        x = x.transpose(1, 2)    # (batch, 1, embed_dim)
        return x


# ─────────────────────────────────────────────
# POSITIONAL ENCODING
# Fixed at 2 positions: [0]=rainfall token, [1]=patch token
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Learnable or sinusoidal positional encoding.
    num_positions=2: position 0 = rainfall token, position 1 = spatial patch token.
    """

    def __init__(self, num_positions: int, embed_dim: int, learnable: bool = True):
        super().__init__()
        if learnable:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, embed_dim) * 0.02)
        else:
            pos_embedding = self._sinusoidal(num_positions, embed_dim)
            self.register_buffer('pos_embedding', pos_embedding)

    def _sinusoidal(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, num_positions, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


# ─────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention(
            x_norm, x_norm, x_norm, need_weights=return_attention
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights if return_attention else None


# ─────────────────────────────────────────────
# TRANSFORMER ENCODER
# ─────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(
        self, num_layers: int, embed_dim: int, num_heads: int,
        mlp_ratio: float, dropout: float
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List]]:
        attn_maps = [] if return_attention else None
        for layer in self.layers:
            x, attn = layer(x, return_attention=return_attention)
            if return_attention and attn is not None:
                attn_maps.append(attn)
        return self.norm(x), attn_maps


# ─────────────────────────────────────────────
# CLASSIFICATION HEAD
# Operates on the patch token (position 1) — NOT the rainfall token
# rainfall token is a conditioning signal, not a prediction target
# ─────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Reads the patch token (index 1) from the encoded sequence and
    maps it to class logits.

    Deliberately uses index 1 (patch token) rather than mean pooling
    because index 0 is the rainfall conditioning token — averaging
    them would mix conditioning with spatial features.
    """

    def __init__(self, embed_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2, embed_dim)
        # Index 1 = patch token (index 0 is rainfall conditioning token)
        patch_token = x[:, 1, :]        # (batch, embed_dim)
        return self.mlp(patch_token)    # (batch, num_classes)



class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        num_layers: int ,
        num_heads: int ,
        mlp_ratio: float ,
        dropout: float ,
        learnable_pos_enc: bool,
        rainfall_method: str ,
        num_timesteps: int ,
        rainfall_hidden: int ,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # 1. Spatial patch → token
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim)

        # 2. Rainfall sequence → conditioning token
        self.rainfall_encoder = RainfallEncoder(num_timesteps=num_timesteps, embed_dim=embed_dim, hidden_dim=rainfall_hidden,
                                                method=rainfall_method, dropout=dropout)

        # 3. Positional encoding over 2 tokens: [rainfall, patch]
        self.pos_encoding = PositionalEncoding(num_positions=2, embed_dim=embed_dim, learnable=learnable_pos_enc)

        # 4. Transformer encoder
        self.transformer = TransformerEncoder(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)

        # 5. Classification head (reads patch token at index 1)
        self.classifier = ClassificationHead(embed_dim=embed_dim, num_classes=num_classes, dropout=dropout)

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

    def forward(self, spatial_patch: torch.Tensor, rainfall_sequence: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[List]]:
        patch_token = self.patch_embedding(spatial_patch)
        rain_token = self.rainfall_encoder(rainfall_sequence)
        tokens = torch.cat([rain_token, patch_token], dim=1)
        tokens = self.pos_encoding(tokens)
        encoded, attn_maps = self.transformer(tokens, return_attention=return_attention)
        logits = self.classifier(encoded)
        return logits, attn_maps if return_attention else None


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_summary(model: ViT) -> str:
    total, trainable = count_parameters(model)
    lines = [
        "=" * 70,
        "VIT — MODEL SUMMARY",
        "=" * 70,
        f"  Spatial channels   : {model.patch_embedding.projection.in_channels}",
        f"  Patch size         : {model.patch_size}×{model.patch_size}",
        f"  Num classes        : {model.num_classes}",
        f"  Embed dim (D)      : {model.embed_dim}",
        f"  Transformer layers : {len(model.transformer.layers)}",
        f"  Attention heads    : {model.transformer.layers[0].attention.num_heads}",
        f"  Rainfall timesteps : {model.num_timesteps}",
        f"  Rainfall method    : {model.rainfall_encoder.method}",
        f"  Token sequence     : [rainfall_token | patch_token]  (length 2)",
        f"  Classification on  : patch token (index 1)",
        "-" * 70,
        f"  Total parameters   : {total:,}",
        f"  Trainable params   : {trainable:,}",
        "=" * 70,
    ]
    return "\n".join(lines)
