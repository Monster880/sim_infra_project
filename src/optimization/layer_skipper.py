"""Adaptive layer skipping for dynamic inference acceleration."""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayerSkipConfig:
    """Configuration for adaptive layer skipping."""
    total_layers: int = 24  # Total VLM layers
    min_active_layers: int = 6  # Minimum layers to keep active
    max_skip_ratio: float = 0.5  # Maximum fraction of layers to skip
    adaptive: bool = True  # Enable adaptive skipping
    confidence_threshold: float = 0.8  # Confidence threshold for early exit
    warmup_steps: int = 10  # Steps before enabling adaptive mode
    
    # Task complexity estimation
    complexity_bins: int = 3  # low, medium, high
    complexity_history_size: int = 50


class AdaptiveLayerSkipper:
    """
    Adaptive layer skipping strategy for VLA inference.
    
    Key insight: Not all inference steps require the same computational depth.
    Simple actions (e.g., moving in a straight line) can use fewer layers,
    while complex actions (e.g., grasping with precision) need more layers.
    
    Strategies:
    1. Static Skip: Always skip a fixed number of upper layers (SmolVLA default)
    2. Confidence-based Early Exit: Exit when intermediate features are confident
    3. Complexity-adaptive: Estimate task complexity and adjust skip ratio
    4. Temporal-adaptive: Use more layers for first chunk, fewer for continuations
    """

    def __init__(self, config: LayerSkipConfig):
        self.config = config
        self._current_active_layers = config.total_layers
        self._step_count = 0
        self._complexity_history: List[float] = []
        self._latency_savings: List[float] = []
        self._skip_decisions: List[int] = []

    def compute_skip_layers(
        self,
        features: Optional[torch.Tensor] = None,
        is_first_chunk: bool = False,
        task_complexity: Optional[float] = None,
    ) -> int:
        """
        Determine how many layers to skip for current inference step.
        
        Args:
            features: Intermediate features for confidence estimation
            is_first_chunk: Whether this is the first chunk in a sequence
            task_complexity: External complexity estimate [0, 1]
            
        Returns:
            Number of layers to skip
        """
        self._step_count += 1
        
        if not self.config.adaptive or self._step_count < self.config.warmup_steps:
            # Static skip during warmup
            skip = int(self.config.total_layers * self.config.max_skip_ratio)
            self._skip_decisions.append(skip)
            return skip
        
        # Adaptive strategies
        skip = 0
        
        if is_first_chunk:
            # Use more layers for first chunk (need full context understanding)
            skip = int(self.config.total_layers * self.config.max_skip_ratio * 0.3)
        elif task_complexity is not None:
            # Complexity-adaptive skip
            skip = self._complexity_adaptive_skip(task_complexity)
        elif features is not None:
            # Confidence-based early exit
            skip = self._confidence_based_skip(features)
        else:
            # Temporal-adaptive: gradually increase skip for continuations
            skip = self._temporal_adaptive_skip()
        
        # Clamp to valid range
        max_skip = self.config.total_layers - self.config.min_active_layers
        skip = min(skip, max_skip)
        skip = max(skip, 0)
        
        self._skip_decisions.append(skip)
        self._current_active_layers = self.config.total_layers - skip
        
        return skip

    def _complexity_adaptive_skip(self, complexity: float) -> int:
        """
        Adjust skip based on task complexity.
        
        Low complexity (straight moves) -> skip more layers
        High complexity (precise grasps) -> skip fewer layers
        """
        self._complexity_history.append(complexity)
        if len(self._complexity_history) > self.config.complexity_history_size:
            self._complexity_history.pop(0)
        
        # Inverse relationship: higher complexity -> fewer skips
        max_skip = self.config.total_layers - self.config.min_active_layers
        skip_ratio = (1 - complexity) * self.config.max_skip_ratio
        
        return int(max_skip * skip_ratio)

    def _confidence_based_skip(self, features: torch.Tensor) -> int:
        """
        Skip remaining layers if features are already confident.
        
        Uses feature norm as a proxy for confidence:
        higher norm = more decisive features = can skip more.
        """
        # Compute feature confidence (simplified)
        feature_norm = features.norm(dim=-1).mean().item()
        
        # Normalize to [0, 1] range (empirical bounds)
        confidence = min(1.0, feature_norm / 10.0)
        
        if confidence >= self.config.confidence_threshold:
            # High confidence: skip more layers
            max_skip = self.config.total_layers - self.config.min_active_layers
            return int(max_skip * 0.7)
        else:
            # Low confidence: skip fewer layers
            return int((self.config.total_layers - self.config.min_active_layers) * 0.3)

    def _temporal_adaptive_skip(self) -> int:
        """
        Temporal adaptation: use fewer layers for continuation chunks.
        
        Rationale: After the first chunk establishes context, subsequent
        chunks can rely on cached representations and skip more layers.
        """
        # Increase skip ratio over time (up to max)
        chunk_index = self._step_count - self.config.warmup_steps
        adaptation_factor = min(1.0, chunk_index / 20.0)  # Saturate after 20 chunks
        
        max_skip = self.config.total_layers - self.config.min_active_layers
        skip = int(max_skip * self.config.max_skip_ratio * (0.5 + 0.5 * adaptation_factor))
        
        return skip

    def estimate_task_complexity(
        self,
        action_variance: float,
        state_change_rate: float,
        visual_entropy: float,
    ) -> float:
        """
        Estimate task complexity from observable signals.
        
        Args:
            action_variance: Variance of recent actions (high = complex)
            state_change_rate: Rate of state changes (high = dynamic)
            visual_entropy: Entropy of visual features (high = complex scene)
            
        Returns:
            Complexity score in [0, 1]
        """
        # Weighted combination of signals
        complexity = (
            0.4 * min(1.0, action_variance / 0.1) +
            0.3 * min(1.0, state_change_rate / 0.5) +
            0.3 * min(1.0, visual_entropy / 5.0)
        )
        
        return np.clip(complexity, 0.0, 1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get layer skipping statistics."""
        if not self._skip_decisions:
            return {"status": "no_data"}
        
        decisions = np.array(self._skip_decisions)
        
        return {
            "total_steps": self._step_count,
            "current_active_layers": self._current_active_layers,
            "skip_stats": {
                "mean_skip": float(np.mean(decisions)),
                "max_skip": int(np.max(decisions)),
                "min_skip": int(np.min(decisions)),
                "std_skip": float(np.std(decisions)),
            },
            "efficiency": {
                "avg_compute_ratio": 1.0 - float(np.mean(decisions)) / self.config.total_layers,
                "theoretical_speedup": self.config.total_layers / (self.config.total_layers - float(np.mean(decisions))),
            },
            "complexity_history": (
                self._complexity_history[-10:] if self._complexity_history else []
            ),
        }
