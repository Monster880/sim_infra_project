"""Model quantization engine for inference acceleration."""

import time
import logging
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    method: str = "dynamic"  # dynamic, static, weight_only
    dtype: str = "float16"  # float16, bfloat16, int8, int4
    calibration_samples: int = 100
    per_channel: bool = True
    symmetric: bool = True
    exclude_layers: Set[str] = None  # Layers to exclude from quantization
    
    def __post_init__(self):
        if self.exclude_layers is None:
            self.exclude_layers = {"output_proj", "state_proj"}


class Quantizer:
    """
    Model quantization engine supporting multiple strategies.
    
    Strategies:
    - FP16: Half-precision floating point (best balance of speed/accuracy)
    - BF16: Brain floating point (better numerical stability)
    - INT8 Dynamic: Dynamic quantization of linear layers
    - INT8 Static: Static quantization with calibration
    - Weight-only INT4: 4-bit weight quantization
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._original_size_mb = 0
        self._quantized_size_mb = 0
        self._quantization_time = 0
        self._accuracy_delta = 0

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply quantization to the model."""
        start_time = time.time()
        
        # Record original size
        self._original_size_mb = self._get_model_size_mb(model)
        
        if self.config.dtype == "float16":
            model = self._quantize_fp16(model)
        elif self.config.dtype == "bfloat16":
            model = self._quantize_bf16(model)
        elif self.config.dtype == "int8":
            if self.config.method == "dynamic":
                model = self._quantize_int8_dynamic(model)
            else:
                model = self._quantize_int8_static(model)
        elif self.config.dtype == "int4":
            model = self._quantize_int4_weight_only(model)
        
        # Record quantized size
        self._quantized_size_mb = self._get_model_size_mb(model)
        self._quantization_time = time.time() - start_time
        
        logger.info(
            f"Quantization complete: {self._original_size_mb:.1f}MB -> "
            f"{self._quantized_size_mb:.1f}MB "
            f"({self._get_compression_ratio():.1f}x compression)"
        )
        
        return model

    def _quantize_fp16(self, model: nn.Module) -> nn.Module:
        """Convert model to FP16."""
        return model.half()

    def _quantize_bf16(self, model: nn.Module) -> nn.Module:
        """Convert model to BF16."""
        return model.to(dtype=torch.bfloat16)

    def _quantize_int8_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply INT8 dynamic quantization to linear layers."""
        # Identify layers to quantize
        layers_to_quantize = {nn.Linear}
        
        quantized = torch.ao.quantization.quantize_dynamic(
            model,
            layers_to_quantize,
            dtype=torch.qint8
        )
        
        return quantized

    def _quantize_int8_static(self, model: nn.Module) -> nn.Module:
        """Apply INT8 static quantization with calibration."""
        # For static quantization, we need calibration data
        # This is a simplified version
        model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        model_prepared = torch.ao.quantization.prepare(model)
        
        # Run calibration (in production, use real data)
        self._run_calibration(model_prepared)
        
        model_quantized = torch.ao.quantization.convert(model_prepared)
        return model_quantized

    def _quantize_int4_weight_only(self, model: nn.Module) -> nn.Module:
        """Apply 4-bit weight-only quantization."""
        # Simulate INT4 by quantizing weights
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name not in self.config.exclude_layers:
                weight = module.weight.data
                # Simulate 4-bit quantization
                scale = weight.abs().max() / 7.0  # 4-bit range: -8 to 7
                quantized_weight = torch.clamp(
                    torch.round(weight / scale), -8, 7
                ) * scale
                module.weight.data = quantized_weight
        
        return model

    def _run_calibration(self, model: nn.Module):
        """Run calibration pass with dummy data."""
        model.eval()
        with torch.no_grad():
            for _ in range(self.config.calibration_samples):
                dummy_input = torch.randn(1, 3, 256, 256)
                dummy_state = torch.randn(1, 7)
                try:
                    model(dummy_input, dummy_state)
                except Exception:
                    pass

    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_bytes += buffer.numel() * buffer.element_size()
        return total_bytes / (1024 * 1024)

    def _get_compression_ratio(self) -> float:
        """Get compression ratio."""
        if self._quantized_size_mb == 0:
            return 1.0
        return self._original_size_mb / self._quantized_size_mb

    def get_stats(self) -> Dict[str, Any]:
        """Get quantization statistics."""
        return {
            "method": self.config.method,
            "dtype": self.config.dtype,
            "original_size_mb": round(self._original_size_mb, 2),
            "quantized_size_mb": round(self._quantized_size_mb, 2),
            "compression_ratio": round(self._get_compression_ratio(), 2),
            "quantization_time_s": round(self._quantization_time, 3),
            "memory_saved_mb": round(self._original_size_mb - self._quantized_size_mb, 2),
        }
