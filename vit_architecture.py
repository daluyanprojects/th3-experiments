import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RainfallSequenceEmbedding(nn.Module):
    """
    Encodes a 1D rainfall time series (13 timesteps of mm/hr intensity) into a single
    dense embedding vector that participates in transformer attention alongside the
    spatial patch token.

    WHY A SEPARATE RAINFALL ENCODER?
    ---------------------------------
    The naive approach would be to concatenate the raw 13-value rainfall vector directly
    to the flattened spatial patch features before classification. This is problematic
    because:
      1. It treats all 13 timesteps as independent, unordered features — losing temporal
         structure like ramp-up, peak intensity, and recession patterns that are physically
         meaningful for flood prediction.
      2. The scale mismatch between raw rainfall values (mm/hr) and normalized spatial
         features could bias learning toward whichever modality has larger magnitude.
      3. It prevents the model from learning cross-modal interactions via attention —
         instead forcing a rigid early fusion.

    By encoding rainfall into the SAME embedding space as the spatial patch token,
    we enable the transformer's attention mechanism to learn dynamic, context-dependent
    relationships between "how much rain fell and when" versus "what kind of terrain
    receives that rain."

    WHY CONV1D SPECIFICALLY?
    ------------------------
    The 'conv' method uses two 1D convolutional layers, which is well-suited here because:
      - Rainfall sequences have LOCAL temporal structure: a spike at timestep 7 is more
        related to timesteps 6 and 8 than to timestep 1. Conv1d with kernel_size=3
        explicitly models this local dependency via sliding windows.
      - It is parameter-efficient compared to attention-based methods, important given
        the rainfall sequence is only 13 timesteps long — attention would be overkill.
      - Conv1d is translation-equivariant, so similar rainfall patterns at different
        positions in the sequence produce similar feature responses, which is desirable
        (a 2-hour intense burst should be recognized regardless of when it occurs).
    """
    def __init__(self, num_timesteps: int, embed_dim: int, hidden_dim: int, method: str):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim
        self.method = method
        
        if method == 'conv':
            # First Conv1d: expands from 1 input channel to hidden_dim channels.
            # This is analogous to learning hidden_dim different "filters" over the
            # rainfall sequence — each filter may respond to different patterns like
            # sustained intensity, sharp peaks, or gradual build-up.
            #
            # kernel_size=3, padding=1: ensures output sequence length equals input
            # length (13 → 13), preserving temporal resolution through both conv layers
            # before pooling.
            #
            # Second Conv1d: refines the hidden_dim feature maps with another round of
            # local temporal context, effectively giving a receptive field of 5 timesteps
            # (3 + 3 - 1) without explicitly widening the kernel.
            #
            # GELU activation: smoother than ReLU (no hard zero cutoff), which helps
            # gradient flow for small but informative rainfall values near zero.
            #
            # AdaptiveAvgPool1d(1): collapses the temporal dimension entirely, producing
            # a single hidden_dim-dimensional vector. Average pooling is preferred over
            # max pooling here because cumulative/average rainfall intensity is physically
            # more relevant to flood depth than just the single peak value.
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            # Linear projection to embed_dim: aligns the rainfall embedding dimensionality
            # with the spatial patch embedding so both tokens are in the same vector space
            # when fed to the transformer. Without this alignment, concatenation would be
            # geometrically meaningless — attention scores between mismatched spaces
            # would be uninterpretable.
            self.projection = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        if self.method == 'conv':
            # Conv1d requires (batch, in_channels, length) format.
            # We treat the single rainfall intensity dimension as 1 input channel,
            # analogous to a grayscale image having 1 color channel.
            x = x.unsqueeze(1)                   # (batch, 13) → (batch, 1, 13)
            x = self.temporal_encoder(x)          # (batch, 1, 13) → (batch, hidden_dim)
            x = self.projection(x)                # (batch, hidden_dim) → (batch, embed_dim)
            
            # Unsqueeze to add the "sequence" dimension required by the transformer.
            # The transformer expects (batch, num_tokens, embed_dim), and this rainfall
            # embedding becomes token #0 in the 2-token sequence.
            x = x.unsqueeze(1)                   # (batch, embed_dim) → (batch, 1, embed_dim)
            
        elif self.method == 'mlp':
            x = self.mlp(x)
            x = x.unsqueeze(1)
            
        elif self.method == 'attention':
            # Each timestep becomes its own token → self-attention captures global
            # temporal dependencies across the full sequence. More expressive than conv
            # but heavier; may overfit on short sequences like 13 timesteps.
            x = x.unsqueeze(-1)
            x = self.timestep_embed(x)
            x, _ = self.temporal_attention(x, x, x)
            x = x.mean(dim=1)
            x = self.projection(x)
            x = x.unsqueeze(1)
        
        return x


class PatchEmbedding(nn.Module):
    """
    Projects a single multi-channel spatial patch (11 channels × 4×4 pixels) into
    a single dense embedding vector of dimension embed_dim.

    WHY TREAT EACH PATCH INDEPENDENTLY?
    ------------------------------------
    In standard ViT applied to large images, the image is divided into many non-overlapping
    patches and all patches are processed together, allowing global spatial attention.
    Here, we go further — each 4×4 patch is processed completely independently as a
    single sample. This is a deliberate design choice for this flood prediction task:

      1. SCALABILITY: Metro Manila's 320×320 grid produces 6,400 patches per scenario
         and 50 scenarios = 320,000 training samples. Processing all patches jointly
         with full spatial attention would be computationally prohibitive.

      2. SPATIAL LOCALITY OF FLOOD: Flood depth at a given location is primarily
         determined by local terrain properties (elevation, soil, drainage capacity)
         plus rainfall input. While spatial context does matter at larger scales
         (upstream catchment), the patch-level features capture the most direct
         physical drivers.

      3. DATA AUGMENTATION EFFECT: Treating patches independently means the model
         sees each patch under 35 different rainfall scenarios during training,
         learning which terrain configurations flood under which rainfall conditions.

    WHY CONV2D INSTEAD OF LINEAR PROJECTION?
    -----------------------------------------
    Both are mathematically equivalent for non-overlapping patches (kernel=stride=patch_size),
    but Conv2d is preferred because:
      - It naturally handles 2D spatial structure within the patch (rows and columns
        of pixels are spatially meaningful, not arbitrary).
      - The weight sharing across spatial positions within the kernel is appropriate —
        a high-elevation pixel in the top-left of a patch should be treated similarly
        to one in the bottom-right.
      - It integrates seamlessly with PyTorch's spatial data conventions (NCHW format).
    """
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # kernel_size=patch_size and stride=patch_size means the convolution window
        # covers exactly ONE patch with no overlap — the entire 4×4 spatial extent
        # is compressed into a single embed_dim-dimensional output vector.
        # This is the standard ViT patch projection trick from Dosovitskiy et al. (2020).
        self.projection = nn.Conv2d(
            in_channels=in_channels,   # 11 geographic/terrain channels
            out_channels=embed_dim,    # maps to transformer's working dimension
            kernel_size=patch_size,    # 4 — covers the full patch in one step
            stride=patch_size          # 4 — no overlap between patch windows
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: one 4×4 patch with 11 terrain channels per sample
        x = self.projection(x)    # (batch, 11, 4, 4) → (batch, embed_dim, 1, 1)
        
        # Collapse the two spatial dimensions (both size 1 after projection)
        # into a single sequence dimension for the transformer
        x = x.flatten(2)          # (batch, embed_dim, 1, 1) → (batch, embed_dim, 1)
        
        # Transformer expects (batch, seq_len, features) — swap last two dims
        x = x.transpose(1, 2)    # (batch, embed_dim, 1) → (batch, 1, embed_dim)
        return x                  # 1 token fully representing the spatial patch


class PositionalEncoding(nn.Module):
    """
    Injects positional information into token embeddings so the transformer can
    distinguish between the rainfall token (position 0) and the spatial patch token
    (position 1).

    WHY IS POSITIONAL ENCODING NECESSARY?
    --------------------------------------
    The transformer's self-attention operation is fundamentally permutation-invariant:
    it computes attention scores based purely on token content (via dot products), with
    no inherent notion of order or position. Without positional encoding:
      - [rainfall_token, patch_token] and [patch_token, rainfall_token] would produce
        IDENTICAL attention outputs, making the model unable to distinguish modalities
        by position.
      - The classification head would receive identically-ordered information regardless
        of which token came first, preventing learning of position-specific roles.

    Even though we only have 2 tokens (so "order" is simple), positional encoding is
    still important because it gives each token a unique positional identity that the
    attention mechanism can use to specialize: "I am the rainfall context" vs
    "I am the spatial terrain context."

    WHY LEARNABLE OVER SINUSOIDAL FOR 2 TOKENS?
    ---------------------------------------------
    Sinusoidal encodings were designed for variable-length, potentially long sequences
    where the encoding must generalize to unseen lengths. For our fixed 2-token sequence:
      - Learnable embeddings have only 2 × embed_dim parameters — negligible overhead.
      - They adapt during training to optimally separate the two token roles in the
        embedding space, rather than being constrained to a fixed mathematical pattern.
      - The model can learn that position 0 (rainfall) should be encoded differently
        from position 1 (terrain) in a task-specific way that sinusoids cannot capture.
    """
    def __init__(self, num_positions: int, embed_dim: int, learnable: bool = True):
        super().__init__()
        self.num_positions = num_positions
        self.embed_dim = embed_dim
        
        if learnable:
            # nn.Parameter registers this tensor as a model parameter,
            # so it's included in optimizer updates and model.state_dict().
            # Initialized with randn — small random values that the model refines
            # during training to optimally distinguish position 0 from position 1.
            self.pos_embedding = nn.Parameter(torch.randn(1, num_positions, embed_dim))
        else:
            # register_buffer: stored in model state but NOT a learnable parameter.
            # Moves with the model (CPU↔GPU) and is saved/loaded with checkpoints,
            # but receives no gradient updates.
            pos_embedding = self._create_sinusoidal_embedding(num_positions, embed_dim)
            self.register_buffer('pos_embedding', pos_embedding)
    
    def _create_sinusoidal_embedding(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        # Original formulation from "Attention Is All You Need" (Vaswani et al., 2017).
        # Uses sine for even dimensions and cosine for odd dimensions at geometrically
        # spaced frequencies. This ensures that:
        #   1. Each position has a unique encoding vector.
        #   2. The distance between encodings is consistent across the sequence.
        #   3. The model can generalize to longer sequences than seen during training.
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pos_embedding = torch.zeros(1, num_positions, embed_dim)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_embedding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Element-wise addition: positional info is additively fused into the token
        # content, not concatenated. This preserves embed_dim while allowing the model
        # to learn to separate positional from semantic information internally.
        return x + self.pos_embedding


class TransformerBlock(nn.Module):
    """
    A single transformer layer: the fundamental building block of the encoder.
    Consists of Multi-Head Self-Attention (MHSA) followed by a position-wise MLP,
    each wrapped with residual connections and layer normalization.

    WHAT DOES ATTENTION ACCOMPLISH FOR FLOOD PREDICTION?
    -----------------------------------------------------
    With 2 tokens [rainfall, patch], self-attention computes 4 pairwise interactions:
      1. rainfall → rainfall: How does the rainfall context relate to itself?
         (less meaningful with a single aggregated token, but still valid)
      2. rainfall → patch: How should rainfall information be interpreted given
         this specific terrain? (e.g. high intensity matters more for impermeable surfaces)
      3. patch → rainfall: How should terrain features be reweighted given the
         rainfall pattern? (e.g. drainage capacity is irrelevant under light rain)
      4. patch → patch: How does the spatial context relate to itself?

    Interactions 2 and 3 are the most important — they capture the physical intuition
    that flood severity is a JOINT function of terrain AND rainfall, not just either
    alone. A flat, clay-heavy patch flooded by heavy rain is very different from the
    same patch under light drizzle.

    WHY PRE-NORM (LayerNorm BEFORE attention/MLP)?
    -----------------------------------------------
    The original transformer used POST-norm (LayerNorm after residual addition), but
    research has shown Pre-norm is significantly more stable for training deeper models:
      - In post-norm, gradients must flow through LayerNorm at every layer during
        backprop, which can cause gradient vanishing in deeper networks.
      - In pre-norm, the residual path is "clean" — gradients flow directly through
        the skip connections without passing through normalization, ensuring stable
        gradient magnitudes even in deep stacks.
      - Pre-norm allows larger learning rates and faster convergence.

    WHY MULTI-HEAD ATTENTION OVER SINGLE-HEAD?
    -------------------------------------------
    Multiple heads allow the model to attend to different "aspects" simultaneously
    in parallel subspaces of the embedding:
      - One head might specialize in matching rainfall peak intensity to terrain permeability.
      - Another might focus on cumulative rainfall vs. drainage capacity.
      - Yet another might capture elevation vs. rainfall duration interactions.
    Each head operates on embed_dim/num_heads dimensions, so total computation is
    similar to single-head attention but with richer representational diversity.

    WHY GELU OVER RELU IN THE MLP?
    --------------------------------
    ReLU hard-zeros all negative inputs, which can cause "dead neurons" — units that
    never activate and stop receiving gradient updates. GELU (Gaussian Error Linear Unit)
    has a smooth, non-zero gradient for slightly negative inputs, which:
      - Prevents dead neurons even for small rainfall values near zero.
      - Produces smoother loss landscapes, which aids optimization.
      - Is empirically preferred in all modern transformer architectures (BERT, GPT, ViT).
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        
        # Two separate LayerNorms: one for the attention sublayer, one for the MLP sublayer.
        # LayerNorm normalizes across the embed_dim dimension for each token independently,
        # stabilizing the distribution of activations entering each sublayer and making
        # training less sensitive to initialization and learning rate.
        self.norm1 = nn.LayerNorm(embed_dim)   # applied before attention
        self.norm2 = nn.LayerNorm(embed_dim)   # applied before MLP
        
        # batch_first=True: expects (batch, seq, embed) rather than PyTorch's historical
        # default of (seq, batch, embed). This is more intuitive and avoids unnecessary
        # tensor transposes throughout the forward pass.
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,   # embed_dim must be divisible by num_heads
            dropout=dropout,       # randomly zeroes attention weights during training
                                   # to prevent over-reliance on specific token interactions
            batch_first=True
        )
        
        # MLP (Feed-Forward Network): applies the same 2-layer network independently
        # to each token position. mlp_ratio controls the expansion — a ratio of 2.0
        # with embed_dim=256 gives hidden_dim=512, providing enough capacity to learn
        # non-linear combinations of the attended features.
        #
        # The role of the MLP is to process the attended information within each token:
        # after attention has GATHERED relevant information across tokens, the MLP
        # PROCESSES that information within each token independently.
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # PRE-NORM PATTERN:
        # normalize → transform → add residual
        # The residual connection (x + output) is critical for two reasons:
        #   1. GRADIENT FLOW: gradients can bypass the attention/MLP sublayers entirely
        #      via the skip connection, preventing vanishing gradients in deeper stacks.
        #   2. INCREMENTAL REFINEMENT: each layer makes small adjustments to the token
        #      representations rather than wholesale transformations, making optimization
        #      more stable and interpretable.

        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attention(
            x_norm, x_norm, x_norm,          # query, key, value are all the same (self-attention)
            need_weights=return_attention     # only compute attention weights when needed
                                             # (saves computation during training)
        )
        x = x + attn_output   # residual: preserve original token info + add attended context

        # MLP sublayer with its own pre-norm and residual
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights   # attn_weights: (batch, num_heads, 2, 2) — useful for
                                     # visualizing which token attends to which
        else:
            return x, None


class TransformerEncoder(nn.Module):
    """
    Stack of num_layers TransformerBlocks followed by a final LayerNorm.

    WHY STACK MULTIPLE LAYERS?
    --------------------------
    Each transformer block refines the token representations by one round of
    attention + MLP. Stacking multiple layers allows hierarchical feature extraction:

      - Layer 1: Learns basic cross-modal associations — e.g., "high rainfall + low
        elevation → attend more to drainage features."
      - Layer 2: Refines these associations — e.g., "even given high rainfall, clay soil
        (low infiltration) dominates over drainage network density."
      - Layers 3-4: Captures higher-order interactions — e.g., "the combination of
        impermeable landuse, flat DEM, poor drainage, AND sustained heavy rainfall
        predicts extreme flooding."

    Each layer builds on the representations refined by all previous layers, enabling
    progressively more abstract and task-relevant feature combinations.

    WHY FINAL LAYERNORM?
    --------------------
    After all transformer blocks, a final LayerNorm standardizes the output token
    representations before they're passed to the classification head. This is standard
    ViT practice (Dosovitskiy et al., 2020) and ensures:
      - The classification head receives consistently scaled inputs regardless of
        network depth or training stage.
      - The scale of pre-classification features doesn't grow unboundedly with depth.
    """
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        
        # nn.ModuleList properly registers each TransformerBlock as a submodule,
        # ensuring their parameters appear in model.parameters(), model.state_dict(),
        # and are correctly moved when calling model.to(device).
        # A plain Python list would NOT register submodules properly.
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
                # Collecting per-layer attention maps enables post-hoc interpretability:
                # you can visualize how attention between rainfall and patch tokens
                # evolves across layers — useful for understanding what the model learned
                # and for debugging unexpected predictions.
                attention_maps.append(attn_weights)
        
        x = self.norm(x)   # final normalization before classification
        return x, attention_maps


class ClassificationHead(nn.Module):
    """
    Maps the transformer's output token representations to flood severity class logits.

    WHY MEAN POOLING OVER CLS TOKEN?
    ---------------------------------
    Two common strategies exist for aggregating transformer outputs for classification:

    1. CLS TOKEN (use_cls_token=True): A special learnable [CLS] token is prepended
       to the sequence. Its output representation after the transformer is used for
       classification. The CLS token has no content of its own — it learns to aggregate
       information from all other tokens via attention.

    2. MEAN POOLING (use_cls_token=False, our default): The output representations of
       ALL tokens are averaged and used for classification.

    For our 2-token setup, mean pooling is preferred because:
      - With only 2 tokens, adding a 3rd [CLS] token increases sequence length by 50%
        and adds complexity without clear benefit.
      - Mean pooling explicitly ensures BOTH the rainfall token AND the spatial patch
        token contribute equally to the final prediction — reflecting the physical
        reality that flood class is jointly determined by both inputs.
      - CLS token aggregation works best when there are many tokens and the CLS token
        needs to selectively attend to the most relevant ones. With 2 tokens,
        this selective mechanism is unnecessary.

    WHY A 2-LAYER MLP HEAD?
    ------------------------
    A single linear layer (embed_dim → num_classes) would be the simplest option, but
    a 2-layer MLP with GELU provides:
      - A non-linear transformation that can learn complex decision boundaries in the
        embed_dim-dimensional space before projecting to class logits.
      - Additional capacity specifically allocated to the classification task, separate
        from the general-purpose transformer encoder.
      - Dropout for regularization specifically at the classification stage,
        complementing the dropout already applied within transformer blocks.

    WHY NO SOFTMAX IN FORWARD?
    ---------------------------
    Raw logits (unnormalized scores) are returned because:
      - nn.CrossEntropyLoss internally applies log-softmax + NLL loss, which is
        numerically more stable than computing softmax separately then applying log.
      - During inference, argmax of logits gives the same result as argmax of softmax
        probabilities (softmax is monotonic), so softmax is unnecessary for prediction.
      - If probability scores are needed (e.g. for calibration), softmax can be applied
        externally: torch.softmax(logits, dim=1).
    """
    def __init__(self, embed_dim: int, num_classes: int, dropout: float, use_cls_token: bool):
        super().__init__()
        self.use_cls_token = use_cls_token
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),   # non-linear feature transformation
            nn.GELU(),
            nn.Dropout(dropout),               # regularization before final projection
            nn.Linear(embed_dim, num_classes)  # project to class logits — no softmax
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            x = x[:, 0]        # extract only the CLS token representation
        else:
            x = x.mean(dim=1)  # (batch, 2, embed_dim) → (batch, embed_dim)
                               # equal contribution from rainfall and spatial tokens
        
        x = self.mlp(x)        # (batch, embed_dim) → (batch, num_classes)
        return x


class ViTFloodClassifier(nn.Module):
    """
    A Vision Transformer adapted for patch-level flood severity classification,
    fusing multi-channel terrain data with temporal rainfall sequences.

    OVERALL ARCHITECTURE PHILOSOPHY
    --------------------------------
    Standard ViT (Dosovitskiy et al., 2020) treats an image as a sequence of
    non-overlapping patches and applies global self-attention across all of them.
    This model makes three key adaptations for the flood prediction domain:

    ADAPTATION 1 — PATCH-LEVEL INDEPENDENCE:
        Rather than processing all 6,400 patches of a scenario jointly, each patch
        is processed independently as a single sample. This transforms a spatial
        prediction problem into a tabular-style classification problem, enabling:
          - Massive parallelization: 6,400 patches × 35 scenarios = 224,000 independent
            training samples processed in parallel via batching.
          - Memory efficiency: no need to hold the full 320×320 feature map in memory.
          - Implicit data augmentation: each terrain patch is seen under 35 different
            rainfall scenarios, teaching the model the rainfall-sensitivity of each
            terrain configuration.

    ADAPTATION 2 — CROSS-MODAL TOKEN FUSION:
        Instead of having many spatial tokens (one per sub-patch), we have exactly
        2 tokens: one for the entire spatial patch, and one for the entire rainfall
        sequence. This enables the transformer's attention to focus entirely on
        CROSS-MODAL interactions — learning how rainfall and terrain jointly determine
        flood class — rather than spatial relationships between patches.

    ADAPTATION 3 — TEMPORAL RAINFALL ENCODING:
        Rainfall is not a static spatial feature but a time series with meaningful
        temporal structure (onset, peak, duration, recession). Encoding it through
        a Conv1d temporal encoder before fusion preserves this structure and allows
        the model to differentiate scenarios with similar total rainfall but different
        temporal profiles (e.g. one intense hour vs. 13 hours of drizzle).

    DATA FLOW SUMMARY:
        Spatial patch (11ch, 4×4) → PatchEmbedding → (batch, 1, embed_dim)  ─────┐
                                                                                 ├→ Concat → (batch, 2, embed_dim)
        Rainfall (13,) → RainfallEmbedding (Conv1D) → (batch, 1, embed_dim) ─────┘
              ↓
        PositionalEncoding → (batch, 2, embed_dim)
              ↓
        TransformerEncoder (4 layers) → (batch, 2, embed_dim)
              ↓
        ClassificationHead (mean pool + MLP) → (batch, 5)  [flood class logits]
    """
    def __init__(
        self,
        in_channels: int,           # 11: DEM, infiltration, landuse, drainage×4, soil×4
        patch_size: int,            # 4: each patch is 4×4 pixels
        num_classes: int,           # 5: No Flood, Light, Moderate, Heavy, Extreme
        embed_dim: int,             # transformer working dimension (256 in base config)
        num_layers: int,            # number of transformer blocks (4 in base config)
        num_heads: int,             # attention heads — embed_dim must be divisible by this
        mlp_ratio: float,           # MLP hidden size = embed_dim × mlp_ratio
        dropout: float,             # regularization rate (0.1 = 10% of neurons dropped)
        use_cls_token: bool = False,        # False: use mean pooling for classification
        learnable_pos_enc: bool = True,     # True: position embeddings are learned
        rainfall_method: str = 'conv',      # temporal encoding strategy
        num_timesteps: int = 13             # rainfall sequence length
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
            # Truncated Normal (std=0.02): the standard ViT weight initialization
            # from Dosovitskiy et al. (2020). Similar to Xavier Glorot initialization
            # in that it keeps the variance of activations consistent across layers,
            # but truncated at ±2σ to prevent occasional extreme values that could
            # destabilize early training. std=0.02 is empirically optimal for transformers.
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # Zero bias initialization: biases start neutral and learn offsets
                # only as needed. Non-zero initial biases can introduce systematic
                # errors in early training before the model has converged.
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            # He (Kaiming) initialization: designed for layers followed by ReLU-family
            # activations (including GELU). It scales weights by sqrt(2/fan_in) to
            # ensure that the variance of activations remains approximately 1.0 through
            # the network, preventing both vanishing and exploding activations.
            #
            # mode='fan_out': scales based on the number of OUTPUT connections rather
            # than input connections. Preferred when the focus is on preserving variance
            # in the FORWARD pass (as opposed to 'fan_in' which focuses on backward pass).
            # For conv layers preceding pooling or projection, fan_out is standard.
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            # LayerNorm has two learnable parameters: weight (γ) and bias (β).
            # Initializing weight=1.0 and bias=0.0 means LayerNorm starts as an
            # identity transformation (output = normalized input × 1 + 0 = normalized input).
            # This is crucial: if LayerNorm started with random weights, it would
            # distort the carefully initialized activations from the layers above,
            # undermining the purpose of weight initialization entirely.
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        spatial_patch: torch.Tensor,        # (batch, 11, 4, 4) — terrain channels
        rainfall_sequence: torch.Tensor,    # (batch, 13) — mm/hr per timestep
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