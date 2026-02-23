import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RainfallSequenceEmbedding(nn.Module):
    """Embed temporal rainfall sequence into a single token"""
    
    def __init__(
        self, 
        num_timesteps: int,
        embed_dim: int , 
        hidden_dim: int,
        method: str 
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim
        self.method = method
        
        if method == 'conv':
            # Recommended: 1D Convolution
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            self.projection = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == 'conv':
            # (batch, 13) -> (batch, 1, 13)
            x = x.unsqueeze(1)
            # Conv encoding
            x = self.temporal_encoder(x)  # (batch, hidden_dim)
            x = self.projection(x)  # (batch, embed_dim)
            
        elif self.method == 'mlp':
            # Direct MLP
            x = self.mlp(x)  # (batch, embed_dim)
            
        elif self.method == 'attention':
            # (batch, 13) -> (batch, 13, 1)
            x = x.unsqueeze(-1)
            x = self.timestep_embed(x)  # (batch, 13, hidden_dim)
            x, _ = self.temporal_attention(x, x, x)
            x = x.mean(dim=1)  # Pool over timesteps
            x = self.projection(x)  # (batch, embed_dim)
        
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        
        return x

class SpatialPatchEmbedding(nn.Module):
    """Embed spatial patch into a single token"""
    
    def __init__(
        self,
        in_channels: int ,
        patch_size: int ,
        embed_dim: int ,
        method: str      # 'conv' or 'flatten'
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.method = method
        
        if method == 'conv':
            # CORRECTED: kernel_size=patch_size to collapse entire patch
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,  # 16×16 kernel for 16×16 patch
                stride=patch_size
            )
        elif method == 'flatten':
            # Alternative: flatten + linear
            self.projection = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.method == 'conv':
            # (batch, 3, 16, 16) -> (batch, embed_dim, 1, 1)
            x = self.projection(x)
            # Flatten spatial dims: (batch, embed_dim, 1, 1) -> (batch, embed_dim, 1)
            x = x.flatten(2)
            # Transpose: (batch, embed_dim, 1) -> (batch, 1, embed_dim)
            x = x.transpose(1, 2)
            
        elif self.method == 'flatten':
            # (batch, 3, 16, 16) -> (batch, 768)
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            # Linear: (batch, 768) -> (batch, embed_dim)
            x = self.projection(x)
            # Add sequence dim: (batch, embed_dim) -> (batch, 1, embed_dim)
            x = x.unsqueeze(1)
        
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float ,
        dropout: float 
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Pre-norm architecture
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attention(
            x_norm, x_norm, x_norm,
            need_weights=return_attention
        )
        x = x + attn_output  # Residual
        
        # MLP block
        x = x + self.mlp(self.norm2(x))  # Residual
        
        return (x, attn_weights) if return_attention else (x, None)


class ViTFloodClassifier(nn.Module):
    def __init__(
        self,
        # Input configuration
        spatial_channels: int,
        spatial_patch_size: int,
        rainfall_timesteps: int,
        num_classes: int,
        
        # Model architecture
        embed_dim: int,
        num_layers: int ,
        num_heads: int ,
        mlp_ratio: float ,
        dropout: float ,
        
        # Embedding methods
        rainfall_method: str ,  
        spatial_method: str , 
        
        # Classification
        pooling_method: str , 
        learnable_pos_enc: bool 
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.pooling_method = pooling_method
        self.rainfall_timesteps = rainfall_timesteps
        
        
        self.spatial_embedding = SpatialPatchEmbedding(
            in_channels=spatial_channels,
            patch_size=spatial_patch_size,
            embed_dim=embed_dim,
            method=spatial_method
        )
        
        self.rainfall_embedding = RainfallSequenceEmbedding(
            num_timesteps=rainfall_timesteps,
            embed_dim=embed_dim,
            hidden_dim=128,
            method=rainfall_method
        )
        
        if learnable_pos_enc:
            self.pos_embedding = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)
        else:
            self.register_buffer('pos_embedding', self._sinusoidal_embedding(2, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        self._init_weights()
    
    def _sinusoidal_embedding(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pos_embedding = torch.zeros(1, num_positions, embed_dim)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_embedding
    
    def _init_weights(self):
        for m in self.modules():
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
    
    def forward(self, spatial_patch: torch.Tensor, rainfall_sequence: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        # 1. Embed spatial patch: (batch, 3, 16, 16) -> (batch, 1, embed_dim)
        patch_token = self.spatial_embedding(spatial_patch)
        
        # 2. Embed rainfall sequence: (batch, 13) -> (batch, 1, embed_dim)
        rainfall_token = self.rainfall_embedding(rainfall_sequence)
        
        # 3. C. CONCATENATE TOKENS: [rainfall_token, patch_token]
        tokens = torch.cat([rainfall_token, patch_token], dim=1)  # (batch, 2, embed_dim)
        
        # 4. D. ADD POSITIONAL ENCODING
        tokens = tokens + self.pos_embedding  # (batch, 2, embed_dim)
        
        # 5. E. TRANSFORMER ENCODER
        attention_maps = [] if return_attention else None
        
        for block in self.transformer_blocks:
            tokens, attn_weights = block(tokens, return_attention=return_attention)
            if return_attention and attn_weights is not None:
                attention_maps.append(attn_weights)
        
        tokens = self.norm(tokens)
        
        # 6. F. CLASSIFICATION
        if self.pooling_method == 'mean':
            pooled = tokens.mean(dim=1)  # Mean pool over 2 tokens
        elif self.pooling_method == 'first':
            pooled = tokens[:, 0]  # Use rainfall token
        else:
            raise ValueError(f"Unknown pooling_method: {self.pooling_method}")
        
        logits = self.classifier(pooled)  # (batch, num_classes)
        
        return logits, attention_maps


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_summary(model: ViTFloodClassifier) -> str:
    total, trainable = count_parameters(model)
    
    summary = [
        "=" * 70,
        "VISION TRANSFORMER FOR FLOOD CLASSIFICATION",
        "=" * 70,
        f"Total parameters: {total:,}",
        f"Trainable parameters: {trainable:,}",
        "",
        "Architecture:",
        f"  Embedding dimension: {model.embed_dim}",
        f"  Number of transformer layers: {len(model.transformer_blocks)}",
        f"  Number of classes: {model.num_classes}",
        "",
        "Input configuration:",
        f"  Rainfall timesteps: {model.rainfall_timesteps}",
        f"  Pooling method: {model.pooling_method}",
        "=" * 70
    ]
    
    return "\n".join(summary)