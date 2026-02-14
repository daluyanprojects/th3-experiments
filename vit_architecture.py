"""
ViT Architecture for Flood Prediction
======================================

This module contains the Vision Transformer (ViT) architecture components
for flood prediction from spatial patches and rainfall data.

Architecture Overview:
    Input: Spatial patches (DEM, Infiltration, Landuse) + Rainfall value
    Output: Flood category classification (0-4) per patch

Components:
    - PatchEmbedding: Converts image patches to token embeddings
    - RainfallEmbedding: Embeds scalar rainfall values
    - TransformerEncoder: Stack of transformer blocks
    - ClassificationHead: Outputs class probabilities
    - ViTFloodClassifier: Complete end-to-end model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# =============================================================================
# 4.1 INPUT PROCESSING COMPONENTS
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Converts image patches into token embeddings using a convolutional layer.
    
    This is equivalent to splitting the image into non-overlapping patches
    and linearly projecting each patch into the embedding space.
    
    Args:
        in_channels (int): Number of input channels (3 for DEM, Infiltration, Landuse)
        patch_size (int): Size of each patch (e.g., 4x4)
        embed_dim (int): Dimension of the embedding space (e.g., 128, 256, 512)
    
    Input Shape:
        (batch_size, in_channels, patch_size, patch_size)
    
    Output Shape:
        (batch_size, 1, embed_dim)  # Single patch becomes one token
    """
    
    def __init__(self, in_channels: int = 3, patch_size: int = 4, embed_dim: int = 256):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Convolutional projection: kernel=patch_size, stride=patch_size
        # This effectively splits the patch and projects it
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, patch_size, patch_size)
        
        Returns:
            (batch_size, 1, embed_dim)
        """
        # Apply convolution: (batch, in_channels, H, W) -> (batch, embed_dim, 1, 1)
        x = self.projection(x)
        
        # Flatten spatial dimensions: (batch, embed_dim, 1, 1) -> (batch, embed_dim, 1)
        x = x.flatten(2)
        
        # Transpose: (batch, embed_dim, 1) -> (batch, 1, embed_dim)
        x = x.transpose(1, 2)
        
        return x

class RainfallEmbedding(nn.Module):
    """
    Embeds scalar rainfall values into the same embedding space as patches.
    
    Uses a 2-layer MLP to project the rainfall value into a rich embedding
    that can be used as an additional token in the transformer.
    
    Args:
        embed_dim (int): Dimension of the embedding space
        hidden_dim (int): Dimension of the hidden layer (default: 128)
    
    Input Shape:
        (batch_size, 1) - normalized rainfall values
    
    Output Shape:
        (batch_size, 1, embed_dim) - rainfall token embedding
    """
    
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1) - rainfall values
        
        Returns:
            (batch_size, 1, embed_dim) - embedded rainfall token
        """
        # x shape: (batch, 1)
        x = self.mlp(x)  # (batch, embed_dim)
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        return x
    
# =============================================================================
# 4.2 POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Adds learnable positional embeddings to patch tokens.
    
    Since transformers don't have inherent notion of position, we add
    positional information to help the model understand spatial relationships.
    
    Args:
        num_positions (int): Number of positions (typically num_patches + 1 for rainfall token)
        embed_dim (int): Dimension of embeddings
        learnable (bool): If True, use learnable embeddings; if False, use fixed sinusoidal
    """
    
    def __init__(self, num_positions: int, embed_dim: int, learnable: bool = True):
        super().__init__()
        
        self.num_positions = num_positions
        self.embed_dim = embed_dim
        
        if learnable:
            # Learnable positional embeddings
            self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, embed_dim))
        else:
            # Fixed sinusoidal positional encoding
            pos_embedding = self._create_sinusoidal_embedding(num_positions, embed_dim)
            self.register_buffer('pos_embedding', pos_embedding)
    
    def _create_sinusoidal_embedding(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        """
        Creates fixed sinusoidal positional embeddings.
        """
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pos_embedding = torch.zeros(1, num_positions, embed_dim)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_embedding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_positions, embed_dim)
        
        Returns:
            (batch_size, num_positions, embed_dim) - with positional encoding added
        """
        return x + self.pos_embedding

# =============================================================================
# 4.3 TRANSFORMER ENCODER
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Single Transformer Encoder Block.
    
    Architecture:
        x -> LayerNorm -> Multi-Head Attention -> Residual
          -> LayerNorm -> Feed-Forward Network -> Residual
    
    Args:
        embed_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input format: (batch, seq, feature)
        )
        
        # Feed-forward network (MLP)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch_size, num_tokens, embed_dim)
            return_attention: If True, return attention weights
        
        Returns:
            x: (batch_size, num_tokens, embed_dim)
            attn_weights: (batch_size, num_heads, num_tokens, num_tokens) or None
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attention(x_norm, x_norm, x_norm, need_weights=return_attention)
        x = x + attn_output
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        else:
            return x, None


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer Encoder Blocks.
    
    Args:
        num_layers (int): Number of transformer blocks
        embed_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            x: (batch_size, num_tokens, embed_dim)
            return_attention: If True, return attention weights from all layers
        
        Returns:
            x: (batch_size, num_tokens, embed_dim)
            attention_maps: List of attention weights from each layer or None
        """
        attention_maps = [] if return_attention else None
        
        for layer in self.layers:
            x, attn_weights = layer(x, return_attention=return_attention)
            if return_attention and attn_weights is not None:
                attention_maps.append(attn_weights)
        
        x = self.norm(x)
        
        return x, attention_maps


# =============================================================================
# 4.4 CLASSIFICATION HEAD
# =============================================================================

class ClassificationHead(nn.Module):
    """
    Classification head for per-patch flood category prediction.
    
    Takes the aggregated token representation and outputs class probabilities.
    
    Args:
        embed_dim (int): Dimension of input embeddings
        num_classes (int): Number of output classes (5 for flood categories 0-4)
        dropout (float): Dropout rate
        use_cls_token (bool): If True, use [CLS] token; else use mean pooling
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.1,
        use_cls_token: bool = False
    ):
        super().__init__()
        
        self.use_cls_token = use_cls_token
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_tokens, embed_dim)
        
        Returns:
            (batch_size, num_classes) - class logits
        """
        if self.use_cls_token:
            # Use the first token ([CLS] token)
            x = x[:, 0]  # (batch, embed_dim)
        else:
            # Global average pooling over all tokens
            x = x.mean(dim=1)  # (batch, embed_dim)
        
        # Apply classification MLP
        x = self.mlp(x)  # (batch, num_classes)
        
        return x


# =============================================================================
# 4.5 COMPLETE VIT MODEL
# =============================================================================

class ViTFloodClassifier(nn.Module):
    """
    Complete Vision Transformer for Flood Prediction.
    
    Architecture:
        1. Patch Embedding: Convert spatial patch to token
        2. Rainfall Embedding: Convert rainfall value to token
        3. Concatenate tokens: [rainfall_token, patch_token]
        4. Add positional encoding
        5. Transformer encoder
        6. Classification head
    
    Args:
        in_channels (int): Number of input channels (default: 3 for DEM, Infiltration, Landuse)
        patch_size (int): Size of input patch (default: 4)
        num_classes (int): Number of output classes (default: 5)
        embed_dim (int): Embedding dimension (default: 256)
        num_layers (int): Number of transformer layers (default: 6)
        num_heads (int): Number of attention heads (default: 8)
        mlp_ratio (float): MLP hidden dim ratio (default: 4.0)
        dropout (float): Dropout rate (default: 0.1)
        use_cls_token (bool): Use CLS token for classification (default: False, uses mean pooling)
        learnable_pos_enc (bool): Use learnable positional encoding (default: True)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 4,
        num_classes: int = 5,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cls_token: bool = False,
        learnable_pos_enc: bool = True
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        
        # 4.1: Input processing
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        self.rainfall_embedding = RainfallEmbedding(
            embed_dim=embed_dim,
            hidden_dim=128
        )
        
        # 4.2: Positional encoding
        # num_positions = 2 (rainfall token + 1 patch token)
        num_positions = 2
        self.pos_encoding = PositionalEncoding(
            num_positions=num_positions,
            embed_dim=embed_dim,
            learnable=learnable_pos_enc
        )
        
        # 4.3: Transformer encoder
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # 4.4: Classification head
        self.classifier = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_cls_token=use_cls_token
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights using Xavier/He initialization."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        spatial_patch: torch.Tensor,
        rainfall_value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through the ViT model.
        
        Args:
            spatial_patch: (batch_size, in_channels, patch_size, patch_size)
            rainfall_value: (batch_size, 1)
            return_attention: If True, return attention maps from all layers
        
        Returns:
            logits: (batch_size, num_classes) - class logits
            attention_maps: List of attention weights from each layer or None
        """
        batch_size = spatial_patch.shape[0]
        
        # 1. Embed spatial patch
        patch_tokens = self.patch_embedding(spatial_patch)  # (batch, 1, embed_dim)
        
        # 2. Embed rainfall value
        rainfall_tokens = self.rainfall_embedding(rainfall_value)  # (batch, 1, embed_dim)
        
        # 3. Concatenate tokens: [rainfall_token, patch_token]
        # This allows the model to condition patch understanding on rainfall
        tokens = torch.cat([rainfall_tokens, patch_tokens], dim=1)  # (batch, 2, embed_dim)
        
        # 4. Add positional encoding
        tokens = self.pos_encoding(tokens)  # (batch, 2, embed_dim)
        
        # 5. Pass through transformer encoder
        encoded, attention_maps = self.transformer(tokens, return_attention=return_attention)  # (batch, 2, embed_dim)
        
        # 6. Classification head
        logits = self.classifier(encoded)  # (batch, num_classes)
        
        if return_attention:
            return logits, attention_maps
        else:
            return logits, None
    
    
# =============================================================================
# MODEL 
# =============================================================================
def create_vit_base(num_classes: int = 5, **kwargs) -> ViTFloodClassifier:
    """
    Create a base ViT model.
    
    Config: 12 layers, 512 dim, 8 heads
    Params: ~20M
    """
    return ViTFloodClassifier(
        embed_dim=512,
        num_layers=12,
        num_heads=8,
        num_classes=num_classes,
        **kwargs
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_summary(model: nn.Module) -> str:
    """
    Get a summary string of the model architecture and parameters.
    """
    total, trainable = count_parameters(model)
    
    summary = []
    summary.append("=" * 70)
    summary.append("MODEL SUMMARY")
    summary.append("=" * 70)
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Total parameters: {total:,}")
    summary.append(f"Trainable parameters: {trainable:,}")
    summary.append(f"Non-trainable parameters: {total - trainable:,}")
    summary.append("=" * 70)
    
    return "\n".join(summary)

