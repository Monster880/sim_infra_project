"""Setup script for embodied-infra package."""

from setuptools import setup, find_packages

setup(
    name="embodied-infra",
    version="0.1.0",
    description="High-performance inference infrastructure for embodied AI (VLA models)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Embodied AI Infra Team",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "psutil>=5.9.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "server": ["grpcio>=1.60.0", "grpcio-tools>=1.60.0", "protobuf>=4.25.0"],
        "onnx": ["onnx>=1.15.0", "onnxruntime>=1.17.0"],
        "full": [
            "transformers>=4.40.0",
            "accelerate>=0.27.0",
            "safetensors>=0.4.0",
            "huggingface-hub>=0.20.0",
            "matplotlib>=3.7.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
