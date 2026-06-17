"""Pipeline optimizer for parallel and pipelined inference execution."""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline optimization."""
    enable_vision_pipeline: bool = True  # Pipeline vision encoding
    enable_prefetch: bool = True  # Prefetch next observation
    enable_overlap: bool = True  # Overlap compute and communication
    num_workers: int = 2  # Number of pipeline workers
    vision_batch_size: int = 2  # Batch multiple camera views
    max_queue_depth: int = 4  # Maximum pipeline queue depth


class PipelineOptimizer:
    """
    Pipeline optimizer for VLA inference.
    
    Implements three key optimizations:
    
    1. Vision-Action Pipeline Parallelism:
       While the action expert processes chunk N, the vision encoder
       processes observations for chunk N+1.
       
    2. Observation Prefetch:
       Pre-encode the next observation while current actions execute,
       reducing the critical path latency.
       
    3. Multi-Camera Batching:
       Batch multiple camera views through the vision encoder in a
       single forward pass instead of sequential processing.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self._vision_queue: Queue = Queue(maxsize=config.max_queue_depth)
        self._action_queue: Queue = Queue(maxsize=config.max_queue_depth)
        
        # Pipeline state
        self._prefetched_features: Optional[torch.Tensor] = None
        self._is_running = False
        
        # Statistics
        self._pipeline_utilization: List[float] = []
        self._overlap_savings_ms: List[float] = []
        self._total_batched_views = 0

    def start(self):
        """Start the pipeline."""
        self._is_running = True
        logger.info("Pipeline optimizer started")

    def stop(self):
        """Stop the pipeline."""
        self._is_running = False
        self._executor.shutdown(wait=True)
        logger.info("Pipeline optimizer stopped")

    def process_observation(
        self,
        images: List[torch.Tensor],
        vision_encoder: nn.Module,
    ) -> torch.Tensor:
        """
        Process observation with pipeline optimization.
        
        Batches multiple camera views and pipelines with action generation.
        """
        start_time = time.perf_counter()
        
        # Check for prefetched features
        if self._prefetched_features is not None:
            features = self._prefetched_features
            self._prefetched_features = None
            
            elapsed = (time.perf_counter() - start_time) * 1000
            self._overlap_savings_ms.append(elapsed)
            return features
        
        # Batch multiple camera views
        if self.config.enable_vision_pipeline and len(images) > 1:
            features = self._batch_vision_encode(images, vision_encoder)
        else:
            features = self._single_vision_encode(images[0], vision_encoder)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        return features

    def prefetch_observation(
        self,
        images: List[torch.Tensor],
        vision_encoder: nn.Module,
    ):
        """
        Prefetch next observation encoding in background.
        Called while current actions are being executed.
        """
        if not self.config.enable_prefetch:
            return
        
        def _encode():
            with torch.no_grad():
                if len(images) > 1:
                    features = self._batch_vision_encode(images, vision_encoder)
                else:
                    features = self._single_vision_encode(images[0], vision_encoder)
                self._prefetched_features = features
        
        self._executor.submit(_encode)

    def _batch_vision_encode(
        self,
        images: List[torch.Tensor],
        encoder: nn.Module,
    ) -> torch.Tensor:
        """Batch multiple camera views through vision encoder."""
        # Stack images into a batch
        batch = torch.stack(images, dim=0)  # [num_cameras, C, H, W]
        self._total_batched_views += len(images)
        
        with torch.no_grad():
            features = encoder(batch)  # [num_cameras, ...]
        
        # Concatenate features from all views
        if features.dim() == 3:
            # [num_cameras, num_tokens, dim] -> [1, num_cameras*num_tokens, dim]
            features = features.reshape(1, -1, features.shape[-1])
        
        return features

    def _single_vision_encode(
        self,
        image: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        """Encode single camera view."""
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            features = encoder(image)
        return features

    def compute_pipeline_schedule(
        self,
        vision_time_ms: float,
        action_time_ms: float,
        communication_time_ms: float,
    ) -> Dict[str, Any]:
        """
        Compute optimal pipeline schedule.
        
        Returns timing information for pipeline stages.
        """
        # Without pipeline: sequential
        sequential_time = vision_time_ms + action_time_ms + communication_time_ms
        
        # With pipeline: overlap vision(N+1) with action(N)
        pipeline_time = max(vision_time_ms, action_time_ms) + communication_time_ms
        
        # Speedup
        speedup = sequential_time / pipeline_time if pipeline_time > 0 else 1.0
        
        # Utilization
        total_compute = vision_time_ms + action_time_ms
        utilization = total_compute / (pipeline_time * self.config.num_workers)
        self._pipeline_utilization.append(utilization)
        
        return {
            "sequential_time_ms": round(sequential_time, 2),
            "pipeline_time_ms": round(pipeline_time, 2),
            "speedup": round(speedup, 2),
            "utilization": round(utilization, 4),
            "bottleneck": "vision" if vision_time_ms > action_time_ms else "action",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "is_running": self._is_running,
            "total_batched_views": self._total_batched_views,
            "avg_utilization": (
                float(np.mean(self._pipeline_utilization))
                if self._pipeline_utilization else 0.0
            ),
            "avg_overlap_savings_ms": (
                float(np.mean(self._overlap_savings_ms))
                if self._overlap_savings_ms else 0.0
            ),
            "prefetch_active": self._prefetched_features is not None,
        }
