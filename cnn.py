import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Tuple


# ── FiLM layer ─────────────────────────────────────────────────────────────────
class FiLM(nn.Module):
    def __init__(self, num_channels: int, context_dim: int):
        super().__init__()
        self.proj = nn.Linear(context_dim, num_channels * 2)  # → γ and β
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias[:num_channels])   # γ init = 1
        nn.init.zeros_(self.proj.bias[num_channels:])  # β init = 0

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, C, H, W)
        context : (B, context_dim)
        """
        params   = self.proj(context)                        # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)                 # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)             # (B, C, 1, 1)
        return gamma * x + beta


# ── Rainfall + conditioning context MLP ───────────────────────────────────────
class RainfallConditioningMLP(nn.Module):
    def __init__(
        self,
        rainfall_dim    : int = 13,
        conditioning_dim: int = 4,
        hidden_dim      : int = 128,
        context_dim     : int = 256,
    ):
        super().__init__()
        in_dim = rainfall_dim + conditioning_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, rainfall: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rainfall, conditioning], dim=1)  # (B, 17)
        return self.mlp(x)                              # (B, context_dim)


# ── Main model ─────────────────────────────────────────────────────────────────
class CNNFloodModel(nn.Module):
    # EfficientNet-B3 decoder channel sizes (smp default)
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    def __init__(
        self,
        num_classes      : int  = 5,
        in_channels      : int  = 3,
        rainfall_dim     : int  = 13,
        conditioning_dim : int  = 4,
        context_dim      : int  = 256,
        mlp_hidden       : int  = 128,
        encoder_weights  : str  = 'imagenet',
    ):
        super().__init__()

        # ── Spatial backbone (encoder + decoder, no final activation) ──────────
        self.backbone = smp.Unet(
            encoder_name    = 'efficientnet-b3',
            encoder_weights = encoder_weights,
            in_channels     = in_channels,
            classes         = num_classes,
            activation      = None,          # raw logits
            decoder_channels= self.DECODER_CHANNELS,
        )

        # ── Rainfall + conditioning context encoder ────────────────────────────
        self.context_mlp = RainfallConditioningMLP(
            rainfall_dim     = rainfall_dim,
            conditioning_dim = conditioning_dim,
            hidden_dim       = mlp_hidden,
            context_dim      = context_dim,
        )

        # ── FiLM layers — one per decoder stage ───────────────────────────────
        # DECODER_CHANNELS matches smp U-Net decoder output channels per stage
        self.film_layers = nn.ModuleList([
            FiLM(num_channels=ch, context_dim=context_dim)
            for ch in self.DECODER_CHANNELS
        ])

        # ── Register forward hooks to apply FiLM inside the decoder ───────────
        self._hook_handles   = []
        self._context_vector = None   # set before each forward pass
        self._hook_stage     = 0      # tracks which FiLM layer to apply

        self._register_decoder_hooks()

    # ── Hook registration ──────────────────────────────────────────────────────
    def _register_decoder_hooks(self):
        decoder_blocks = list(self.backbone.decoder.blocks)
        assert len(decoder_blocks) == len(self.DECODER_CHANNELS), (
            f"Expected {len(self.DECODER_CHANNELS)} decoder blocks, "
            f"got {len(decoder_blocks)}"
        )
        for stage_idx, block in enumerate(decoder_blocks):
            handle = block.register_forward_hook(
                self._make_film_hook(stage_idx)
            )
            self._hook_handles.append(handle)

    def _make_film_hook(self, stage_idx: int):
        def hook(module, input, output):
            if self._context_vector is not None:
                return self.film_layers[stage_idx](output, self._context_vector)
        return hook

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        spatial     : torch.Tensor,            # (B, 3, H, W)
        rainfall    : torch.Tensor,            # (B, 13)
        conditioning: Optional[torch.Tensor],  # (B, 4)
    ) -> Tuple[torch.Tensor, None]:
        # Build context vector from rainfall + conditioning
        if conditioning is None:
            conditioning = torch.zeros(
                spatial.size(0), 4, dtype=spatial.dtype, device=spatial.device
            )
        self._context_vector = self.context_mlp(rainfall, conditioning)  # (B, context_dim)

        # Forward through backbone — FiLM hooks fire inside decoder
        logits = self.backbone(spatial)   # (B, 5, H, W)

        # Clear context so hooks are no-ops if called outside forward
        self._context_vector = None

        return logits, None

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def parameter_count(self) -> dict:
        total    = sum(p.numel() for p in self.parameters())
        backbone = sum(p.numel() for p in self.backbone.parameters())
        film     = sum(p.numel() for p in self.film_layers.parameters())
        ctx_mlp  = sum(p.numel() for p in self.context_mlp.parameters())
        return {
            'total'      : total,
            'backbone'   : backbone,
            'film_layers': film,
            'context_mlp': ctx_mlp,
        }