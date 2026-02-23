import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RainfallSequenceEmbedding(nn.Module):
    def __init__(self, num_timesteps: int, embed_dim: int, hidden_dim: int, method: str):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim
        self.method = method
        
        if method == 'conv':
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
        batch_size = x.shape[0]
        
        if self.method == 'conv':
            x = x.unsqueeze(1)                  
            x = self.temporal_encoder(x)          
            x = self.projection(x)             
            
            x = x.unsqueeze(1)                   
            
        elif self.method == 'mlp':
            x = self.mlp(x)
            x = x.unsqueeze(1)
            
        elif self.method == 'attention':
            x = x.unsqueeze(-1)
            x = self.timestep_embed(x)
            x, _ = self.temporal_attention(x, x, x)
            x = x.mean(dim=1)
            x = self.projection(x)
            x = x.unsqueeze(1)
        
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.projection = nn.Conv2d(
            in_channels=in_channels,  
            out_channels=embed_dim,    
            kernel_size=patch_size,   
            stride=patch_size       
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)   
        x = x.flatten(2)                 
        x = x.transpose(1, 2)   
        return x              


class PositionalEncoding(nn.Module):
    def __init__(self, num_positions: int, embed_dim: int, learnable: bool = True):
        super().__init__()
        self.num_positions = num_positions
        self.embed_dim = embed_dim
        
        if learnable:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, embed_dim))
        else:
            pos_embedding = self._create_sinusoidal_embedding(num_positions, embed_dim)
            self.register_buffer('pos_embedding', pos_embedding)
    
    def _create_sinusoidal_embedding(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pos_embedding = torch.zeros(1, num_positions, embed_dim)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_embedding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
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
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attention(
            x_norm, x_norm, x_norm,          
            need_weights=return_attention     
                                             
        )
        x = x + attn_output  
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights  
        else:
            return x, None


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
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
        
        # Final layer norm: applied once after all transformer blocks
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        attention_maps = [] if return_attention else None
        
        for layer in self.layers:
            x, attn_weights = layer(x, return_attention=return_attention)
            if return_attention and attn_weights is not None:
                attention_maps.append(attn_weights)
        
        x = self.norm(x)   # final normalization before classification
        return x, attention_maps


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float, use_cls_token: bool):
        super().__init__()
        self.use_cls_token = use_cls_token
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),  
            nn.GELU(),
            nn.Dropout(dropout),               
            nn.Linear(embed_dim, num_classes)  
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            x = x[:, 0]        
        else:
            x = x.mean(dim=1)  
                               
        
        x = self.mlp(x)        
        return x


class ViTFloodClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,           
        patch_size: int,           
        num_classes: int,          
        embed_dim: int,             
        num_layers: int,            
        num_heads: int,            
        mlp_ratio: float,           
        dropout: float,             
        use_cls_token: bool = False,       
        learnable_pos_enc: bool = True,    
        rainfall_method: str = 'conv',     
        num_timesteps: int = 13          
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.num_timesteps = num_timesteps
        
        # --- COMPONENT 1: Spatial Patch Encoder ---
        # Learns a compressed representation of the terrain at this patch location.
        # The 11 input channels encode physical properties that govern flood behavior:
        # elevation (DEM) determines flow direction and ponding risk; infiltration and
        # soil type determine how much rain is absorbed vs. runs off; landuse determines
        # surface permeability; drainage networks determine water removal capacity.
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # --- COMPONENT 2: Temporal Rainfall Encoder ---
        # Learns a compressed representation of the rainfall event's temporal profile.
        # hidden_dim=128 provides sufficient capacity to capture temporal patterns
        # while remaining much smaller than embed_dim=256, keeping the rainfall encoder
        # lightweight relative to the spatial encoder.
        self.rainfall_embedding = RainfallSequenceEmbedding(
            num_timesteps=num_timesteps,
            embed_dim=embed_dim,
            hidden_dim=128,
            method=rainfall_method
        )
        
        # --- COMPONENT 3: Positional Encoding ---
        # Only 2 positions needed: [0] = rainfall token, [1] = spatial token.
        # The ordering (rainfall first) is arbitrary but must be consistent —
        # the model learns position-specific roles through training.
        self.pos_encoding = PositionalEncoding(
            num_positions=2,
            embed_dim=embed_dim,
            learnable=learnable_pos_enc
        )
        
        # --- COMPONENT 4: Transformer Encoder ---
        # The core of the model. With num_layers=4 and 2 tokens, this is a relatively
        # lightweight transformer focused entirely on modeling the rainfall-terrain
        # interaction rather than spatial relationships between many patches.
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # --- COMPONENT 5: Classification Head ---
        # Converts the fused rainfall-terrain representation to flood class probabilities.
        self.classifier = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_cls_token=use_cls_token
        )
        
        # --- WEIGHT INITIALIZATION ---
        # self.apply() recursively calls _init_weights on every submodule in the model.
        # This happens ONCE at construction time, before any training.
        # Proper initialization is critical: bad initialization can cause:
        #   - Vanishing gradients (activations → 0 from the first forward pass)
        #   - Exploding gradients (activations → ∞, NaN loss immediately)
        #   - Symmetry breaking failure (all neurons learn identical features)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        spatial_patch: torch.Tensor,       
        rainfall_sequence: torch.Tensor,   
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        
        # Step 1: Encode the spatial patch into a single terrain token
        # Each of the 11 channels contributes to a holistic terrain representation
        patch_tokens = self.patch_embedding(spatial_patch)          # → (batch, 1, embed_dim)
        
        # Step 2: Encode the temporal rainfall sequence into a single rainfall token
        # The Conv1d encoder extracts temporal patterns before projecting to embed_dim
        rainfall_tokens = self.rainfall_embedding(rainfall_sequence) # → (batch, 1, embed_dim)
        
        # Step 3: Concatenate into a 2-token multimodal sequence
        # Ordering: [rainfall | patch] — rainfall first, patch second.
        # This ordering is preserved by positional encoding so the model always
        # knows which token represents which modality.
        tokens = torch.cat([rainfall_tokens, patch_tokens], dim=1)  # → (batch, 2, embed_dim)
        
        # Step 4: Add positional encoding to distinguish token roles
        # Without this, attention is blind to which token is rainfall and which is terrain
        tokens = self.pos_encoding(tokens)                          # → (batch, 2, embed_dim)
        
        # Step 5: Cross-modal fusion via stacked self-attention layers
        # Each layer refines the token representations by attending to the other token,
        # progressively building a joint rainfall-terrain feature representation
        encoded, attention_maps = self.transformer(
            tokens, return_attention=return_attention
        )                                                           # → (batch, 2, embed_dim)
        
        # Step 6: Pool across tokens and classify into flood severity
        # Mean pooling ensures both rainfall and terrain representations contribute
        # equally to the final flood class prediction
        logits = self.classifier(encoded)                          # → (batch, 5)
        
        if return_attention:
            return logits, attention_maps  # attention_maps: list of (batch, heads, 2, 2) per layer
        else:
            return logits, None


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_summary(model: nn.Module) -> str:
    total, trainable = count_parameters(model)
    
    summary = []
    summary.append("=" * 70)
    summary.append("MODEL SUMMARY (WITH TEMPORAL RAINFALL)")
    summary.append("=" * 70)
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Total parameters: {total:,}")
    summary.append(f"Trainable parameters: {trainable:,}")
    summary.append(f"Non-trainable parameters: {total - trainable:,}")
    summary.append(f"Rainfall timesteps: {model.num_timesteps}")
    summary.append(f"Rainfall embedding method: {model.rainfall_embedding.method}")
    summary.append("=" * 70)
    
    return "\n".join(summary)
