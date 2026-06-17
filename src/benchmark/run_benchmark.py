"""Comprehensive benchmark suite for inference performance evaluation."""

import time
import json
import logging
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.inference_engine import InferenceEngine, InferenceConfig
from src.inference.model_loader import ModelConfig
from src.optimization.quantizer import Quantizer, QuantizationConfig
from src.optimization.layer_skipper import AdaptiveLayerSkipper, LayerSkipConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    num_warmup: int = 10
    num_iterations: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])
    image_sizes: List[int] = field(default_factory=lambda: [256, 512])
    action_chunk_sizes: List[int] = field(default_factory=lambda: [10, 25, 50])
    device: str = "cpu"
    output_dir: str = "./benchmark_results"
    
    # Optimization configurations to benchmark
    test_baseline: bool = True
    test_fp16: bool = True
    test_int8: bool = True
    test_layer_skip: bool = True
    test_kv_cache: bool = True
    test_async: bool = True
    test_combined: bool = True


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    config: Dict[str, Any]
    latency_ms: Dict[str, float]  # mean, median, p95, p99, min, max
    throughput: Dict[str, float]  # actions/sec, inferences/sec
    memory_mb: float
    model_size_mb: float
    optimization_details: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for VLA inference.
    
    Benchmarks multiple optimization configurations and produces
    comparative results for latency, throughput, and memory usage.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_all(self) -> List[BenchmarkResult]:
        """Run all configured benchmarks."""
        logger.info("Starting comprehensive benchmark suite")
        
        configurations = self._build_configurations()
        
        for name, inference_config in configurations.items():
            logger.info(f"Running benchmark: {name}")
            try:
                result = self._run_single_benchmark(name, inference_config)
                self.results.append(result)
                logger.info(
                    f"  {name}: {result.latency_ms['mean']:.2f}ms mean latency, "
                    f"{result.throughput['actions_per_sec']:.1f} actions/sec"
                )
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
        
        # Save results
        self._save_results()
        
        return self.results

    def _build_configurations(self) -> Dict[str, InferenceConfig]:
        """Build all benchmark configurations."""
        configs = {}
        
        if self.config.test_baseline:
            configs["baseline_fp32"] = InferenceConfig(
                model_config=ModelConfig(device=self.config.device, dtype="float32"),
                enable_quantization=False,
                enable_layer_skip=False,
                enable_kv_cache=False,
                enable_async=False,
            )
        
        if self.config.test_fp16:
            configs["optimized_fp16"] = InferenceConfig(
                model_config=ModelConfig(device=self.config.device, dtype="float16"),
                enable_quantization=True,
                quantization_dtype="float16",
                enable_layer_skip=False,
                enable_kv_cache=False,
                enable_async=False,
            )
        
        if self.config.test_int8:
            configs["optimized_int8"] = InferenceConfig(
                model_config=ModelConfig(device=self.config.device, dtype="int8"),
                enable_quantization=True,
                quantization_dtype="int8",
                enable_layer_skip=False,
                enable_kv_cache=False,
                enable_async=False,
            )
        
        if self.config.test_layer_skip:
            configs["layer_skip_50pct"] = InferenceConfig(
                model_config=ModelConfig(device=self.config.device, dtype="float32"),
                enable_quantization=False,
                enable_layer_skip=True,
                layer_skip_ratio=0.5,
                enable_kv_cache=False,
                enable_async=False,
            )
        
        if self.config.test_kv_cache:
            configs["kv_cache_enabled"] = InferenceConfig(
                model_config=ModelConfig(device=self.config.device, dtype="float32"),
                enable_quantization=False,
                enable_layer_skip=False,
                enable_kv_cache=True,
                enable_async=False,
            )
        
        if self.config.test_async:
            configs["async_inference"] = InferenceConfig(
                model_config=ModelConfig(device=self.config.device, dtype="float32"),
                enable_quantization=False,
                enable_layer_skip=False,
                enable_kv_cache=False,
                enable_async=True,
            )
        
        if self.config.test_combined:
            configs["combined_all_optimizations"] = InferenceConfig(
                model_config=ModelConfig(device=self.config.device, dtype="float16"),
                enable_quantization=True,
                quantization_dtype="float16",
                enable_layer_skip=True,
                layer_skip_ratio=0.5,
                enable_kv_cache=True,
                enable_async=True,
            )
        
        return configs

    def _run_single_benchmark(
        self, name: str, config: InferenceConfig
    ) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        # Initialize engine
        engine = InferenceEngine(config)
        init_stats = engine.initialize()
        
        # Prepare test data
        batch_size = 1
        image_size = 256
        images = torch.randn(batch_size, 3, image_size, image_size)
        state = torch.randn(batch_size, config.action_dim)
        
        # Warmup
        for _ in range(self.config.num_warmup):
            engine.predict(images, state)
        
        # Benchmark
        latencies = []
        for _ in range(self.config.num_iterations):
            start = time.perf_counter()
            actions, metrics = engine.predict(images, state)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        
        # Compute statistics
        latencies = np.array(latencies)
        
        # Memory measurement
        memory_mb = self._measure_memory(engine)
        
        result = BenchmarkResult(
            name=name,
            config={
                "device": config.model_config.device,
                "dtype": config.model_config.dtype,
                "layer_skip": config.enable_layer_skip,
                "layer_skip_ratio": config.layer_skip_ratio if config.enable_layer_skip else 0,
                "kv_cache": config.enable_kv_cache,
                "async": config.enable_async,
                "quantization": config.quantization_dtype if config.enable_quantization else "none",
            },
            latency_ms={
                "mean": float(np.mean(latencies)),
                "median": float(np.median(latencies)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "std": float(np.std(latencies)),
            },
            throughput={
                "actions_per_sec": config.action_chunk_size * 1000.0 / float(np.mean(latencies)),
                "inferences_per_sec": 1000.0 / float(np.mean(latencies)),
                "chunks_per_sec": 1000.0 / float(np.mean(latencies)),
            },
            memory_mb=memory_mb,
            model_size_mb=init_stats.get("memory_mb", 0),
            optimization_details=init_stats.get("optimizations", {}),
        )
        
        # Cleanup
        engine.shutdown()
        
        return result

    def _measure_memory(self, engine: InferenceEngine) -> float:
        """Measure current memory usage."""
        if torch.cuda.is_available() and engine.config.model_config.device == "cuda":
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)

    def _save_results(self):
        """Save benchmark results to JSON."""
        output_path = Path(self.config.output_dir) / "benchmark_results.json"
        
        results_dict = []
        for r in self.results:
            results_dict.append({
                "name": r.name,
                "config": r.config,
                "latency_ms": r.latency_ms,
                "throughput": r.throughput,
                "memory_mb": r.memory_mb,
                "model_size_mb": r.model_size_mb,
                "optimization_details": r.optimization_details,
            })
        
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

    def generate_comparison_table(self) -> str:
        """Generate a markdown comparison table."""
        if not self.results:
            return "No results available."
        
        # Find baseline for speedup calculation
        baseline_latency = None
        for r in self.results:
            if "baseline" in r.name:
                baseline_latency = r.latency_ms["mean"]
                break
        
        if baseline_latency is None:
            baseline_latency = self.results[0].latency_ms["mean"]
        
        lines = [
            "| Configuration | Latency (ms) | Speedup | Actions/sec | Memory (MB) |",
            "|---|---|---|---|---|",
        ]
        
        for r in self.results:
            speedup = baseline_latency / r.latency_ms["mean"] if r.latency_ms["mean"] > 0 else 0
            lines.append(
                f"| {r.name} | {r.latency_ms['mean']:.2f} | "
                f"{speedup:.2f}x | {r.throughput['actions_per_sec']:.0f} | "
                f"{r.memory_mb:.0f} |"
            )
        
        return "\n".join(lines)


def main():
    """Run the benchmark suite."""
    logging.basicConfig(level=logging.INFO)
    
    config = BenchmarkConfig(
        num_warmup=5,
        num_iterations=50,
        device="cpu",
        output_dir="/home/ubuntu/embodied-infra/benchmark_results",
    )
    
    runner = BenchmarkRunner(config)
    results = runner.run_all()
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(runner.generate_comparison_table())
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
