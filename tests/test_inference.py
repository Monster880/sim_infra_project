"""Tests for the inference engine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.inference.inference_engine import InferenceEngine, InferenceConfig
from src.inference.model_loader import ModelLoader, ModelConfig
from src.inference.action_queue import ActionQueue, ActionQueueConfig
from src.optimization.quantizer import Quantizer, QuantizationConfig
from src.optimization.layer_skipper import AdaptiveLayerSkipper, LayerSkipConfig


def test_model_loader():
    """Test model loading with various configurations."""
    print("Testing model loader...")
    
    config = ModelConfig(device="cpu", dtype="float32")
    loader = ModelLoader(config)
    model = loader.load()
    
    assert model is not None
    stats = loader.get_stats()
    assert stats["total_params"] > 0
    print(f"  Model loaded: {stats['total_params']:,} params, {stats['memory_mb']:.1f} MB")
    print("  PASSED")


def test_inference_engine():
    """Test basic inference."""
    print("Testing inference engine...")
    
    config = InferenceConfig(
        model_config=ModelConfig(device="cpu", dtype="float32"),
        enable_quantization=False,
        enable_layer_skip=False,
        enable_kv_cache=False,
        enable_async=False,
    )
    
    engine = InferenceEngine(config)
    engine.initialize()
    
    # Run inference
    images = torch.randn(1, 3, 256, 256)
    state = torch.randn(1, 7)
    
    actions, metrics = engine.predict(images, state)
    
    assert actions.shape == (1, 50, 7), f"Expected (1, 50, 7), got {actions.shape}"
    assert metrics["inference_time_ms"] > 0
    print(f"  Inference: {metrics['inference_time_ms']:.2f}ms, shape={actions.shape}")
    
    engine.shutdown()
    print("  PASSED")


def test_quantization():
    """Test model quantization."""
    print("Testing quantization...")
    
    # Load model
    config = ModelConfig(device="cpu", dtype="float32")
    loader = ModelLoader(config)
    model = loader.load()
    
    # Quantize
    quant_config = QuantizationConfig(method="dynamic", dtype="int8")
    quantizer = Quantizer(quant_config)
    quantized_model = quantizer.quantize(model)
    
    stats = quantizer.get_stats()
    assert stats["compression_ratio"] > 1.0
    print(f"  Compression: {stats['compression_ratio']:.2f}x")
    print(f"  Size: {stats['original_size_mb']:.1f}MB -> {stats['quantized_size_mb']:.1f}MB")
    print("  PASSED")


def test_layer_skipper():
    """Test adaptive layer skipping."""
    print("Testing layer skipper...")
    
    config = LayerSkipConfig(total_layers=24, adaptive=True)
    skipper = AdaptiveLayerSkipper(config)
    
    # Test static skip (warmup)
    for i in range(10):
        skip = skipper.compute_skip_layers()
    
    # Test adaptive skip
    skip = skipper.compute_skip_layers(task_complexity=0.2)  # Low complexity
    assert skip > 0
    print(f"  Low complexity skip: {skip} layers")
    
    skip = skipper.compute_skip_layers(task_complexity=0.9)  # High complexity
    print(f"  High complexity skip: {skip} layers")
    
    stats = skipper.get_stats()
    assert stats["total_steps"] > 0
    print(f"  Theoretical speedup: {stats['efficiency']['theoretical_speedup']:.2f}x")
    print("  PASSED")


def test_action_queue():
    """Test action queue with blending."""
    print("Testing action queue...")
    
    config = ActionQueueConfig(
        max_size=200,
        chunk_size=50,
        action_dim=7,
        blend_overlap=10,
    )
    queue = ActionQueue(config)
    
    # Push first chunk
    chunk1 = np.random.randn(50, 7).astype(np.float32)
    queue.push_chunk(chunk1)
    assert queue.size == 50
    
    # Push second chunk (should blend)
    chunk2 = np.random.randn(50, 7).astype(np.float32)
    queue.push_chunk(chunk2)
    
    # Pop actions
    for _ in range(10):
        action = queue.pop_action()
        assert action is not None
        assert len(action) == 7
    
    stats = queue.get_stats()
    print(f"  Queue size: {stats['current_size']}")
    print(f"  Actions consumed: {stats['total_actions_consumed']}")
    print(f"  Chunks received: {stats['total_chunks_received']}")
    print("  PASSED")


def test_optimized_inference():
    """Test inference with all optimizations enabled."""
    print("Testing optimized inference (all optimizations)...")
    
    config = InferenceConfig(
        model_config=ModelConfig(device="cpu", dtype="float32"),
        enable_quantization=False,  # Skip INT8 on CPU for speed
        enable_layer_skip=True,
        layer_skip_ratio=0.5,
        enable_kv_cache=True,
        enable_async=True,
    )
    
    engine = InferenceEngine(config)
    engine.initialize()
    
    images = torch.randn(1, 3, 256, 256)
    state = torch.randn(1, 7)
    
    # Run multiple inferences
    latencies = []
    for _ in range(20):
        actions, metrics = engine.predict(images, state)
        latencies.append(metrics["inference_time_ms"])
    
    avg_latency = np.mean(latencies)
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Throughput: {50 * 1000 / avg_latency:.0f} actions/sec")
    
    perf_stats = engine.get_performance_stats()
    print(f"  P95 latency: {perf_stats['latency_ms']['p95']:.2f}ms")
    
    engine.shutdown()
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("EMBODIED AI INFERENCE INFRASTRUCTURE - TEST SUITE")
    print("=" * 60)
    
    test_model_loader()
    test_inference_engine()
    test_quantization()
    test_layer_skipper()
    test_action_queue()
    test_optimized_inference()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
