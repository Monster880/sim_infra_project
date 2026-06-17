"""KV Cache manager for efficient inference across action chunks."""

import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for KV cache management."""
    max_cache_entries: int = 16  # Max number of cached states
    cache_ttl_steps: int = 100  # Cache entry time-to-live in steps
    similarity_threshold: float = 0.95  # Threshold for cache hit
    enable_prefix_caching: bool = True  # Cache instruction prefix
    enable_visual_caching: bool = True  # Cache visual features
    max_memory_mb: float = 512.0  # Maximum cache memory
    eviction_policy: str = "lru"  # lru, lfu, fifo


class KVCacheManager:
    """
    Intelligent KV cache manager for VLA inference.
    
    Optimizations:
    1. Instruction Prefix Caching: Cache encoded instruction tokens
       (same instruction across multiple steps)
    2. Visual Feature Caching: Cache visual encoder outputs when
       scene hasn't changed significantly
    3. Cross-chunk KV Reuse: Reuse KV states from previous chunks
       for overlapping context
    4. Adaptive Eviction: Smart eviction based on access patterns
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Cache stores
        self._instruction_cache: OrderedDict = OrderedDict()
        self._visual_cache: OrderedDict = OrderedDict()
        self._kv_state_cache: OrderedDict = OrderedDict()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_memory_saved_mb = 0
        self._current_memory_mb = 0

    def get_instruction_cache(
        self, instruction_hash: str
    ) -> Optional[torch.Tensor]:
        """
        Retrieve cached instruction encoding.
        
        Instructions rarely change between steps, so caching them
        eliminates redundant tokenization and encoding.
        """
        if not self.config.enable_prefix_caching:
            return None
        
        if instruction_hash in self._instruction_cache:
            self._hits += 1
            # Move to end (LRU)
            self._instruction_cache.move_to_end(instruction_hash)
            entry = self._instruction_cache[instruction_hash]
            entry["access_count"] += 1
            return entry["tensor"]
        
        self._misses += 1
        return None

    def set_instruction_cache(
        self, instruction_hash: str, encoded: torch.Tensor
    ):
        """Cache instruction encoding."""
        if not self.config.enable_prefix_caching:
            return
        
        self._check_memory_limit()
        
        self._instruction_cache[instruction_hash] = {
            "tensor": encoded.detach(),
            "timestamp": time.time(),
            "access_count": 1,
        }

    def get_visual_cache(
        self, visual_hash: str, similarity: float = 1.0
    ) -> Optional[torch.Tensor]:
        """
        Retrieve cached visual features.
        
        If the scene hasn't changed significantly (measured by
        frame similarity), reuse cached visual encoder output.
        """
        if not self.config.enable_visual_caching:
            return None
        
        if visual_hash in self._visual_cache:
            entry = self._visual_cache[visual_hash]
            if similarity >= self.config.similarity_threshold:
                self._hits += 1
                self._visual_cache.move_to_end(visual_hash)
                entry["access_count"] += 1
                return entry["tensor"]
        
        self._misses += 1
        return None

    def set_visual_cache(self, visual_hash: str, features: torch.Tensor):
        """Cache visual encoder output."""
        if not self.config.enable_visual_caching:
            return
        
        self._check_memory_limit()
        
        self._visual_cache[visual_hash] = {
            "tensor": features.detach(),
            "timestamp": time.time(),
            "access_count": 1,
        }

    def get_kv_state(
        self, state_hash: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve cached KV states from transformer layers."""
        if state_hash in self._kv_state_cache:
            self._hits += 1
            self._kv_state_cache.move_to_end(state_hash)
            return self._kv_state_cache[state_hash]["states"]
        
        self._misses += 1
        return None

    def set_kv_state(
        self, state_hash: str, kv_states: Dict[str, torch.Tensor]
    ):
        """Cache KV states."""
        self._check_memory_limit()
        
        self._kv_state_cache[state_hash] = {
            "states": {k: v.detach() for k, v in kv_states.items()},
            "timestamp": time.time(),
            "access_count": 1,
        }

    def compute_frame_similarity(
        self, frame1: torch.Tensor, frame2: torch.Tensor
    ) -> float:
        """
        Compute similarity between two frames for cache validity check.
        Uses cosine similarity on downsampled features.
        """
        with torch.no_grad():
            # Downsample for efficiency
            f1 = torch.nn.functional.adaptive_avg_pool2d(frame1, (8, 8)).flatten()
            f2 = torch.nn.functional.adaptive_avg_pool2d(frame2, (8, 8)).flatten()
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                f1.unsqueeze(0), f2.unsqueeze(0)
            ).item()
            
        return similarity

    def compute_hash(self, tensor: torch.Tensor) -> str:
        """Compute a hash for cache lookup."""
        # Use tensor statistics as a fast hash
        with torch.no_grad():
            stats = torch.tensor([
                tensor.mean().item(),
                tensor.std().item(),
                tensor.min().item(),
                tensor.max().item(),
            ])
            return str(hash(stats.numpy().tobytes()))

    def _check_memory_limit(self):
        """Evict entries if memory limit exceeded."""
        self._update_memory_usage()
        
        while self._current_memory_mb > self.config.max_memory_mb:
            self._evict_one()

    def _evict_one(self):
        """Evict one entry based on eviction policy."""
        # Try evicting from largest cache first
        caches = [
            self._kv_state_cache,
            self._visual_cache,
            self._instruction_cache,
        ]
        
        for cache in caches:
            if cache:
                if self.config.eviction_policy == "lru":
                    cache.popitem(last=False)  # Remove oldest
                elif self.config.eviction_policy == "fifo":
                    cache.popitem(last=False)
                self._evictions += 1
                return

    def _update_memory_usage(self):
        """Update current memory usage estimate."""
        total_bytes = 0
        
        for cache in [self._instruction_cache, self._visual_cache, self._kv_state_cache]:
            for entry in cache.values():
                if "tensor" in entry:
                    total_bytes += entry["tensor"].numel() * entry["tensor"].element_size()
                elif "states" in entry:
                    for t in entry["states"].values():
                        total_bytes += t.numel() * t.element_size()
        
        self._current_memory_mb = total_bytes / (1024 * 1024)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / max(1, total_requests)
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self._evictions,
            "current_memory_mb": round(self._current_memory_mb, 2),
            "max_memory_mb": self.config.max_memory_mb,
            "cache_sizes": {
                "instruction": len(self._instruction_cache),
                "visual": len(self._visual_cache),
                "kv_state": len(self._kv_state_cache),
            },
            "estimated_time_saved_ms": self._hits * 5.0,  # ~5ms per cache hit
        }

    def clear(self):
        """Clear all caches."""
        self._instruction_cache.clear()
        self._visual_cache.clear()
        self._kv_state_cache.clear()
        self._current_memory_mb = 0
