# FlashAttention Implementation - Progress Report

**Author:** Pablo Rodriguez
**Date:** November 15, 2025 (Updated: November 19, 2025)
**For:** Shuang Ma, Pei Yu Lin (Team Members)

---

## Summary

I attempted to implement the FlashAttention algorithm from the paper in PyTorch. The goal was to understand the memory-efficient attention mechanism described in the paper. The implementation shows memory savings compared to standard attention and produces numerically similar results.

---

## What I Implemented

### Core Components

Based on the FlashAttention paper, I implemented:

- **Standard Attention** (`src/attention/standard_attention.py`)
  The baseline O(L²) attention that materializes the full attention matrix

- **Online Softmax** (`src/utils/online_softmax.py`)
  The 2-pass online softmax algorithm from Section 3 of the paper

- **Single-Row FlashAttention** (`src/attention/flash_attention_single.py`)
  Implementation of the core recurrence relation (equation 14) for a single query

- **Tiled FlashAttention** (`src/attention/flash_attention_tiled.py`)
  Full tiled implementation that processes K and V in blocks

- **FlashAttention-2** (`src/attention/flash_attention_v2.py`)
  Variant that reduces FLOPs by normalizing at the end instead of in the inner loop

- **Triton Kernel** (`src/attention/flash_attention_triton.py`)
  Basic GPU kernel using Triton (educational version)

### Testing & Benchmarking

- **Unit Tests** (`tests/test_correctness.py`)
  23 tests comparing implementations against PyTorch's standard attention - [✓] 23/23 passing

- **CPU Benchmarks** (`benchmarks/benchmark_cpu.py`)
  Memory scaling analysis for different sequence lengths

- **GPU Benchmarks** (`benchmarks/benchmark_gpu.py`)
  Actual GPU memory measurements on RTX 3070 (limited to 8K sequence length due to GPU memory)

- **Visualization** (`scripts/visualize_results.py`)
  Generates plots for memory scaling, time comparison, and numerical accuracy

---

## Benchmark Results

### GPU Measurements (RTX 3070)

| Seq Length | Std Time (ms) | Flash Time (ms) | Std Mem (MB) | Flash Mem (MB) | Savings |
|------------|---------------|-----------------|--------------|----------------|---------|
| 512        | 0.59          | 2.12            | 29.12        | 23.22          | 20.3%   |
| 1024       | 1.38          | 4.05            | 82.12        | 38.31          | 53.3%   |
| 2048       | 4.33          | 8.59            | 284.12       | 68.50          | 75.9%   |
| 4096       | 14.28         | 29.01           | 1072.12      | 128.88         | 88.0%   |
| 8192       | 58.72         | 105.54          | 4184.12      | 249.62         | 94.0%   |

**Note:** GPU can't handle sequences longer than 8K due to memory constraints. The pure PyTorch implementation is slower than standard attention due to Python loops - real FlashAttention uses fused CUDA kernels for speed.

### Numerical Accuracy

- Maximum difference from standard attention: < 5×10⁻⁷
- All tests pass correctness checks (23/23)
- No NaN or Inf values with extreme inputs

Standard attention uses O(L²) memory (quadratic scaling), while FlashAttention aims for O(L) memory by processing in blocks.

---

## Project Structure

The project was organized as a Python package (just for practice/learning - not necessary for the class):

```
ecs271-project/
├── src/                           # Core implementations
│   ├── attention/
│   │   ├── standard_attention.py
│   │   ├── flash_attention_single.py
│   │   ├── flash_attention_tiled.py
│   │   ├── flash_attention_v2.py
│   │   └── flash_attention_triton.py
│   └── utils/
│       └── online_softmax.py
│
├── tests/                         # Unit tests
│   └── test_correctness.py
│
├── benchmarks/                    # Benchmarking scripts
│   ├── benchmark_cpu.py
│   └── benchmark_gpu.py
│
├── scripts/                       # Utilities
│   └── visualize_results.py
│
├── results/                       # Generated benchmark data
├── plots/                         # Generated visualizations
├── docs/                          # Documentation
└── setup.py                       # Package setup (for learning purposes)
```

The `__init__.py` files and `setup.py` were added to make it a proper Python package - mostly just for learning/practice.

---

## How to Run

### Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch matplotlib # though needs specific torch url-index for CUDA
```

### Run Tests

```bash
python tests/test_correctness.py
```

### Run Benchmarks

```bash
# CPU benchmarks
python benchmarks/benchmark_cpu.py

# GPU benchmarks (requires CUDA)
python benchmarks/benchmark_gpu.py
```

### Generate Visualizations

```bash
python scripts/visualize_results.py
```

### Run Individual Implementations

You can also run each implementation file directly to see demos:

```bash
python src/attention/standard_attention.py
python src/utils/online_softmax.py
python src/attention/flash_attention_single.py
```

---

## Things I'm Unsure About

- Whether the tiling strategy exactly matches the paper's approach
- If the block size choices are optimal
- The Triton kernel implementation might be oversimplified compared to the real FlashAttention

---

## Status

[✓] Core algorithm implemented
[✓] Tests passing (23/23)
[✓] Memory efficiency demonstrated (up to 94% savings at 8K tokens)
[✓] Numerical correctness verified
[✓] Basic documentation written
