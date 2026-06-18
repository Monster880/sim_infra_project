"""Model loader with optimization support for VLA models."""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading and optimization."""
    model_name_or_path: str = "lerobot/smolvla_base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"  # float32, float16, bfloat16, int8
    max_layer_skip: int = 0  # Number of layers to skip (0 = auto)
    enable_kv_cache: bool = True
    compile_model: bool = False  # torch.compile
    num_visual_tokens: int = 64
    action_chunk_size: int = 50
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class ModelLoader:
    """Loads and prepares VLA models with various optimizations applied."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.load_time = 0.0
        self.optimization_stats = {}

    def load(self) -> nn.Module:
        """Load model with specified optimizations."""
        start_time = time.time()
        logger.info(f"Loading model: {self.config.model_name_or_path}")

        # Simulate model loading (in production, use actual LeRobot/HF loading)
        self.model = self._create_smolvla_model()

        # Apply layer skipping BEFORE quantization
        # (quantization changes layer structure, so skip must happen first)
        if self.config.max_layer_skip > 0:
            self.model = self._apply_layer_skip(self.model)

        # Apply dtype optimization / quantization
        self.model = self._apply_dtype(self.model)

        # Apply torch.compile if requested
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.model = self._apply_compile(self.model)

        # Move to device
        self.model = self.model.to(self.config.device)

        self.load_time = time.time() - start_time
        logger.info(f"Model loaded in {self.load_time:.2f}s")
        return self.model

    def _create_smolvla_model(self) -> nn.Module:
        """Create a SmolVLA-like model architecture for inference."""
        return SmolVLAInferenceModel(
            num_visual_tokens=self.config.num_visual_tokens,
            action_chunk_size=self.config.action_chunk_size,
            enable_kv_cache=self.config.enable_kv_cache,
        )

    def _apply_dtype(self, model: nn.Module) -> nn.Module:
        """Apply precision optimization."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if self.config.dtype == "int8":
            model = self._quantize_int8(model)
            self.optimization_stats["quantization"] = "INT8"
        elif self.config.dtype in dtype_map:
            # FP16 on CPU can cause issues with some ops; keep model in fp32
            # but track the intended dtype for memory reporting
            if self.config.dtype == "float16" and self.config.device == "cpu":
                self.optimization_stats["dtype"] = "float16 (simulated on CPU)"
            else:
                model = model.to(dtype=dtype_map[self.config.dtype])
                self.optimization_stats["dtype"] = self.config.dtype

        return model

    def _quantize_int8(self, model: nn.Module) -> nn.Module:
        """Apply INT8 dynamic quantization."""
        quantized = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        logger.info("Applied INT8 dynamic quantization")
        return quantized

    def _apply_layer_skip(self, model: nn.Module) -> nn.Module:
        """Apply layer skipping optimization."""
        if hasattr(model, 'set_layer_skip'):
            model.set_layer_skip(self.config.max_layer_skip)
            self.optimization_stats["layer_skip"] = self.config.max_layer_skip
            logger.info(f"Applied layer skipping: {self.config.max_layer_skip} layers")
        return model

    def _apply_compile(self, model: nn.Module) -> nn.Module:
        """Apply torch.compile for graph optimization."""
        try:
            model = torch.compile(model, mode="reduce-overhead")
            self.optimization_stats["compiled"] = True
            logger.info("Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
        return model

    def get_stats(self) -> Dict[str, Any]:
        """Get model loading and optimization statistics."""
        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        memory_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024) if self.model else 0

        return {
            "model_name": self.config.model_name_or_path,
            "device": self.config.device,
            "total_params": total_params,
            "memory_mb": round(memory_mb, 2),
            "load_time_s": round(self.load_time, 3),
            "optimizations": self.optimization_stats,
        }


class SmolVLAInferenceModel(nn.Module):
    """
    Simplified SmolVLA model architecture for inference benchmarking.
    
    Architecture mirrors the real SmolVLA:
    - Vision Encoder (SigLIP-like): processes RGB images into visual tokens
    - Language Model (SmolLM2-like): processes instruction + visual tokens
    - Action Expert (Flow Matching Transformer): generates action chunks
    """

    def __init__(
        self,
        vision_dim: int = 384,
        language_dim: int = 576,
        action_dim: int = 7,  # 6-DOF + gripper
        num_visual_tokens: int = 64,
        action_chunk_size: int = 50,
        num_vision_layers: int = 12,
        num_language_layers: int = 24,
        num_action_layers: int = 8,
        enable_kv_cache: bool = True,
    ):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.action_chunk_size = action_chunk_size
        self.enable_kv_cache = enable_kv_cache
        self.num_language_layers = num_language_layers
        self.active_language_layers = num_language_layers  # Can be reduced by layer skip

        # Vision Encoder (SigLIP-like)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, vision_dim // 4, kernel_size=16, stride=16),
            nn.GELU(),
            nn.Conv2d(vision_dim // 4, vision_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(vision_dim // 2, vision_dim, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.vision_proj = nn.Linear(vision_dim, language_dim)

        # Pixel Shuffle for token reduction
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.token_reduction = nn.Linear(vision_dim * 4, language_dim)

        # Language Model layers (SmolLM2-like transformer blocks)
        self.language_layers = nn.ModuleList([
            TransformerBlock(language_dim, num_heads=8)
            for _ in range(num_language_layers)
        ])

        # State projection
        self.state_proj = nn.Linear(action_dim, language_dim)

        # Action Expert (Flow Matching Transformer)
        self.action_expert = ActionExpert(
            input_dim=language_dim,
            action_dim=action_dim,
            num_layers=num_action_layers,
            chunk_size=action_chunk_size,
        )

        # KV Cache storage
        self._kv_cache = None

    def set_layer_skip(self, num_skip: int):
        """Set number of language model layers to skip."""
        self.active_language_layers = max(1, self.num_language_layers - num_skip)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        instruction_tokens: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            images: [B, C, H, W] RGB images
            state: [B, action_dim] current robot state
            instruction_tokens: [B, seq_len, dim] pre-encoded instruction
            use_cache: whether to use KV cache
            
        Returns:
            actions: [B, chunk_size, action_dim] predicted action chunk
        """
        batch_size = images.shape[0]

        # 1. Vision encoding with token reduction
        visual_features = self.vision_encoder(images)  # [B, D, H', W']
        B, D, H, W = visual_features.shape
        visual_tokens = visual_features.reshape(B, D, -1).permute(0, 2, 1)  # [B, N, D]
        visual_tokens = self.vision_proj(visual_tokens[:, :self.num_visual_tokens, :])

        # 2. State projection
        state_token = self.state_proj(state).unsqueeze(1)  # [B, 1, D]

        # 3. Concatenate all tokens
        if instruction_tokens is not None:
            tokens = torch.cat([visual_tokens, instruction_tokens, state_token], dim=1)
        else:
            # Use learnable instruction placeholder
            tokens = torch.cat([visual_tokens, state_token], dim=1)

        # 4. Language model processing (with layer skipping)
        for i in range(self.active_language_layers):
            tokens = self.language_layers[i](tokens)

        # 5. Action expert generates action chunk
        actions = self.action_expert(tokens, state)

        return actions

    def clear_cache(self):
        """Clear KV cache."""
        self._kv_cache = None


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention and FFN."""

    def __init__(self, dim: int, num_heads: int = 8, ff_mult: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ff_mult)),
            nn.GELU(),
            nn.Linear(int(dim * ff_mult), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class ActionExpert(nn.Module):
    """
    Flow Matching Transformer for action chunk generation.
    Uses interleaved cross-attention and self-attention.
    """

    def __init__(
        self,
        input_dim: int = 576,
        action_dim: int = 7,
        num_layers: int = 8,
        chunk_size: int = 50,
        num_heads: int = 8,
        num_denoise_steps: int = 10,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.num_denoise_steps = num_denoise_steps
        hidden_dim = int(input_dim * 0.75)  # 75% of VLM dim as per SmolVLA

        # Action token embedding
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Time embedding for flow matching
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Interleaved cross-attention and self-attention layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                # Cross-attention: action tokens attend to VLM features
                self.layers.append(CrossAttentionBlock(hidden_dim, input_dim, num_heads))
            else:
                # Self-attention: action tokens attend to each other
                self.layers.append(SelfAttentionBlock(hidden_dim, num_heads))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # Context projection
        self.context_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, context: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Generate action chunk using flow matching.
        
        Args:
            context: [B, seq_len, input_dim] VLM output features
            state: [B, action_dim] current state
            
        Returns:
            actions: [B, chunk_size, action_dim]
        """
        batch_size = context.shape[0]
        device = context.device

        # Initialize with noise (flow matching starting point)
        x = torch.randn(batch_size, self.chunk_size, self.action_dim, device=device)

        # Project context for cross-attention
        context_proj = self.context_proj(context)

        # Iterative denoising (flow matching)
        for step in range(self.num_denoise_steps):
            t = torch.full((batch_size, 1), step / self.num_denoise_steps, device=device)
            time_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, D]

            # Embed current noisy actions
            h = self.action_embed(x) + time_emb

            # Process through interleaved attention layers
            for layer in self.layers:
                if isinstance(layer, CrossAttentionBlock):
                    h = layer(h, context_proj)
                else:
                    h = layer(h)

            # Predict velocity field
            velocity = self.output_proj(h)

            # Euler step
            dt = 1.0 / self.num_denoise_steps
            x = x + velocity * dt

        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block where queries attend to context."""

    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.context_proj = nn.Linear(context_dim, dim) if context_dim != dim else nn.Identity()
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross attention
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        x = x + self.cross_attn(q, kv, kv)[0]
        x = x + self.ffn(self.norm_ff(x))
        return x


class SelfAttentionBlock(nn.Module):
    """Causal self-attention block."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        # Causal mask
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = x + self.attn(normed, normed, normed, attn_mask=mask)[0]
        x = x + self.ffn(self.norm_ff(x))
        return x
