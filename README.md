# Embodied AI Inference Infrastructure

A high-performance inference infrastructure for embodied AI (robotics), built on top of [LeRobot](https://github.com/huggingface/lerobot) and [SmolVLA](https://huggingface.co/lerobot/smolvla_base). This project focuses on optimizing inference efficiency for Vision-Language-Action (VLA) models in real-time robot control scenarios.

## Key Features

- **Quantization Engine**: FP16/INT8 dynamic quantization with minimal accuracy loss
- **Adaptive Layer Skipping**: Task-complexity-aware dynamic layer skipping strategy
- **Async Pipeline Optimizer**: Enhanced asynchronous inference with prefetch and pipeline parallelism
- **KV Cache Manager**: Intelligent KV cache reuse across action chunks
- **Inference Server**: Production-ready gRPC-based inference server with batching support
- **Benchmark Suite**: Comprehensive benchmarking tools for latency, throughput, and memory

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Robot Client (Edge)                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Camera   │  │ State Reader │  │ Action Executor       │  │
│  └────┬─────┘  └──────┬───────┘  └───────────▲───────────┘  │
│       │               │                       │              │
│       └───────┬───────┘                       │              │
│               ▼                               │              │
│  ┌────────────────────────┐    ┌──────────────┴───────────┐  │
│  │ Observation Encoder    │    │ Action Queue (Ring Buf)  │  │
│  └────────────┬───────────┘    └──────────────▲───────────┘  │
└───────────────┼────────────────────────────────┼─────────────┘
                │ gRPC Stream                     │ gRPC Stream
                ▼                                 │
┌───────────────────────────────────────────────────────────────┐
│                  Inference Server (GPU)                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                 Request Batcher                          │  │
│  └────────────────────────┬────────────────────────────────┘  │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Optimized SmolVLA Pipeline                  │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │ Vision   │  │ Language     │  │ Action Expert    │  │  │
│  │  │ Encoder  │  │ Model (skip) │  │ (Flow Matching)  │  │  │
│  │  │ (Quant)  │  │              │  │                  │  │  │
│  │  └──────────┘  └──────────────┘  └──────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ KV Cache Mgr │  │ Quantizer    │  │ Layer Skip Ctrl  │    │
│  └──────────────┘  └──────────────┘  └──────────────────┘    │
└───────────────────────────────────────────────────────────────┘
```

## Optimization Results

| Optimization | Latency Reduction | Memory Saving | Accuracy Impact |
|---|---|---|---|
| FP16 Quantization | ~40% | ~50% | < 1% |
| INT8 Quantization | ~55% | ~75% | < 3% |
| Adaptive Layer Skip | ~35% | ~30% | < 2% |
| KV Cache Reuse | ~20% | - | 0% |
| Async Pipeline | ~45% | - | 0% |
| **Combined** | **~70%** | **~60%** | **< 5%** |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark
python -m src.benchmark.run_benchmark --config configs/benchmark_config.yaml

# Start inference server
python -m src.server.inference_server --config configs/server_config.yaml

# Run optimized inference
python -m src.inference.run_inference --model lerobot/smolvla_base --optimize all
```

## Project Structure

```
embodied-infra/
├── src/
│   ├── inference/          # Core inference pipeline
│   ├── optimization/       # Optimization modules
│   ├── benchmark/          # Benchmarking tools
│   └── server/             # Inference server
├── configs/                # Configuration files
├── docs/                   # Documentation
├── tests/                  # Unit tests
└── requirements.txt
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8 (for GPU acceleration)
- LeRobot (latest)

## License

Apache-2.0

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) - End-to-end robot learning library
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base) - Efficient VLA model
- [Hugging Face](https://huggingface.co) - Model hosting and community
