"""Action queue with intelligent chunk management and blending."""

import time
import threading
import logging
from typing import Optional, Callable, List
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ActionQueueConfig:
    """Configuration for the action queue."""
    max_size: int = 200
    chunk_size: int = 50
    action_dim: int = 7
    blend_overlap: int = 10
    blend_method: str = "weighted_average"  # weighted_average, exponential, linear
    refill_threshold: float = 0.3  # Trigger refill when queue < 30% full
    control_frequency_hz: float = 50.0  # Robot control frequency


class ActionQueue:
    """
    Thread-safe action queue with intelligent chunk blending.
    
    Manages the buffer between inference (slow, chunked) and
    execution (fast, single-action). Implements smooth blending
    between consecutive action chunks to avoid discontinuities.
    """

    def __init__(self, config: ActionQueueConfig):
        self.config = config
        self._queue: deque = deque(maxlen=config.max_size)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        
        # Statistics
        self._total_actions_consumed = 0
        self._total_chunks_received = 0
        self._underflow_count = 0
        self._blend_count = 0
        self._queue_size_history: List[int] = []
        
        # Last chunk for blending
        self._last_chunk: Optional[np.ndarray] = None
        self._last_chunk_remaining: int = 0

    def push_chunk(self, actions: np.ndarray):
        """
        Push a new action chunk into the queue with blending.
        
        Args:
            actions: [chunk_size, action_dim] array of actions
        """
        with self._lock:
            self._total_chunks_received += 1
            
            if len(self._queue) > 0 and self._last_chunk is not None:
                # Blend overlapping region
                actions = self._blend_chunks(actions)
                self._blend_count += 1
            
            for action in actions:
                self._queue.append(action)
            
            self._last_chunk = actions.copy()
            self._last_chunk_remaining = len(actions)
            self._not_empty.notify_all()

    def pop_action(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Pop the next action from the queue.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Single action array or None if queue is empty
        """
        with self._not_empty:
            if len(self._queue) == 0:
                self._not_empty.wait(timeout=timeout)
            
            if len(self._queue) > 0:
                action = self._queue.popleft()
                self._total_actions_consumed += 1
                if self._last_chunk_remaining > 0:
                    self._last_chunk_remaining -= 1
                return action
            else:
                self._underflow_count += 1
                return None

    def needs_refill(self) -> bool:
        """Check if queue needs new actions."""
        with self._lock:
            fill_ratio = len(self._queue) / self.config.max_size
            return fill_ratio <= self.config.refill_threshold

    def _blend_chunks(self, new_chunk: np.ndarray) -> np.ndarray:
        """
        Blend new chunk with remaining actions in queue.
        
        Uses configurable blending method to ensure smooth transitions
        between consecutive action chunks.
        """
        overlap = min(self.config.blend_overlap, len(self._queue), len(new_chunk))
        
        if overlap == 0:
            return new_chunk
        
        blended = new_chunk.copy()
        
        # Get overlapping actions from queue
        queue_tail = list(self._queue)[-overlap:]
        
        for i in range(overlap):
            if self.config.blend_method == "weighted_average":
                # Linear weight transition
                weight = (i + 1) / (overlap + 1)
                blended[i] = (1 - weight) * queue_tail[i] + weight * new_chunk[i]
            
            elif self.config.blend_method == "exponential":
                # Exponential decay blending
                weight = 1 - np.exp(-3 * (i + 1) / overlap)
                blended[i] = (1 - weight) * queue_tail[i] + weight * new_chunk[i]
            
            elif self.config.blend_method == "linear":
                # Simple linear interpolation
                weight = i / overlap
                blended[i] = (1 - weight) * queue_tail[i] + weight * new_chunk[i]
        
        return blended

    def get_stats(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            return {
                "current_size": len(self._queue),
                "max_size": self.config.max_size,
                "fill_ratio": len(self._queue) / self.config.max_size,
                "total_actions_consumed": self._total_actions_consumed,
                "total_chunks_received": self._total_chunks_received,
                "underflow_count": self._underflow_count,
                "blend_count": self._blend_count,
                "effective_control_hz": (
                    self._total_actions_consumed / 
                    max(1, self._total_chunks_received) * 
                    self.config.control_frequency_hz
                ),
            }

    def clear(self):
        """Clear the queue."""
        with self._lock:
            self._queue.clear()
            self._last_chunk = None
            self._last_chunk_remaining = 0

    @property
    def size(self) -> int:
        """Current queue size."""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """Whether queue is empty."""
        return len(self._queue) == 0
