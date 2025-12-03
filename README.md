# FlashAttention Implementation

**ECS271 Machine Learning Project**
**Team:** Shuang Ma, Pablo Rodriguez, Pei Yu Lin
**Fall 2025**

An educational implementation of the FlashAttention algorithm attempting to replicate the memory-efficient attention mechanism described in the paper.

---

## Project Overview

This project implements FlashAttention, a memory-efficient approach to computing self-attention in Transformer models. We demonstrate how the algorithm avoids materializing the full O(L²) attention matrix by using tiling and online softmax computation.

**Key Results:**
- **93.5% memory savings** at 8K sequence length (4208MB → 274MB) on H200 GPU
- **4.1x speedup** with Triton kernel implementation on RTX 3070 GPU
- Numerical equivalence maintained (max difference < 10⁻⁶)
- Tested on both enterprise (H200 150GB) and consumer (RTX 3070 8GB) hardware
- 23 comprehensive unit tests validating correctness

Standard attention requires O(L²) memory for the attention matrix, which becomes prohibitive for long sequences. FlashAttention processes attention in blocks to reduce memory usage while maintaining numerical correctness.

---

## Project Structure

```
ecs271-project/
├── README.md
├── requirements.txt
├── .gitignore
├── setup.py                       # For package structure (learning purposes)
│
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
└── docs/                          # Documentation
    ├── flashattn.pdf
    ├── project-document.md
    ├── project_progress_report.md
    └── understanding_flashattention.md
```

**Note:** The project is organized as a Python package with `setup.py` and `__init__.py` files - this was mostly for learning/practice and isn't necessary for the class project.
(Slightly regret it because the import lines are now much more confusing)

---

## Getting Started

### Setup

1. Clone the repository and navigate to the project directory

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch matplotlib
   ```

### Running the Code

**Run tests:**
```bash
python tests/test_correctness.py
```

**Run benchmarks:**
```bash
# CPU benchmarks
python benchmarks/benchmark_cpu.py

# GPU benchmarks (requires CUDA)
python benchmarks/benchmark_gpu.py
```

**Generate plots:**
```bash
python scripts/visualize_results.py
```

**Run individual implementations:**
```bash
python src/attention/standard_attention.py
python src/utils/online_softmax.py
python src/attention/flash_attention_single.py
python src/attention/flash_attention_tiled.py
```

---

## Implementation Details

The implementation follows the FlashAttention paper:

1. **Standard Attention** - Baseline O(L²) implementation
2. **Online Softmax** - Incremental softmax algorithm with rescaling
3. **Single-Row FlashAttention** - Core recurrence relation (Equation 14)
4. **Tiled FlashAttention-1** - Full tiled implementation with blocks (PyTorch)
5. **FlashAttention-2** - Optimized variant with unnormalized outputs and fewer FLOPs (PyTorch)
6. **Triton Kernels** - Fused GPU implementations for both FA-1 and FA-2 achieving real speedups

All implementations are compared against PyTorch's standard attention for correctness with 23 comprehensive unit tests.

---

## Notes

- **PyTorch implementations** use Python loops and are slower than standard attention (3-22x slower) due to loop overhead
- **Triton kernels** achieve both memory efficiency AND speed gains (4.1x faster at 8K tokens) through kernel fusion
- Tested on NVIDIA H200 (150GB VRAM) for main experiments and RTX 3070 (8GB VRAM) for Triton kernels
- GPU benchmarks tested up to 8K sequence length
- Numerical accuracy: max difference < 5×10⁻⁷ from standard attention for PyTorch, < 2×10⁻³ for Triton

---

## Future Work

While this project successfully implements FlashAttention and demonstrates its benefits, there are several interesting directions for extending this work:

### 1. Domain-Specific Optimizations
- **Long-form documents** - Optimize for 100K+ token sequences (legal documents, books, scientific papers)
- **Genomics applications** - DNA/protein sequences are extremely long and benefit from efficient attention
- **Time series analysis** - Financial data, sensor streams, and temporal data with long-range dependencies

### 2. Hardware Portability
- **AMD GPU support** - FlashAttention is primarily NVIDIA-focused; ROCm implementation would broaden accessibility
- **Apple Silicon** - Metal Performance Shaders implementation for M-series chips
- **Edge deployment** - Optimize for mobile/embedded GPUs (Jetson, mobile devices)

### 3. Algorithmic Improvements
- **Sparse + Flash hybrid** - Combine FlashAttention with learned sparsity patterns for further gains
- **Dynamic block sizing** - Adaptively adjust block size based on attention entropy or sequence characteristics
- **Mixed precision** - INT8/FP16 quantization while maintaining numerical accuracy

### 4. Practical Tooling
- **Auto-tuning library** - Automatically discover optimal block sizes and configurations for any GPU architecture
- **Drop-in replacement** - Seamless integration to replace `torch.nn.MultiheadAttention` with FlashAttention
- **Profiling and visualization** - Tools to analyze memory/time bottlenecks in attention operations

### 5. Research Directions
- **Generalized IO-aware tiling** - Apply the tiling approach to other quadratic operations (cross-attention, graph attention, bilinear layers)
- **Backward pass optimization** - Simplify the complex backward pass while maintaining numerical stability
- **Theoretical analysis** - Prove tighter bounds on numerical error and analyze stability under various conditions

---

## Citation

This is an educational reimplementation. The original FlashAttention algorithm:

```bibtex
@inproceedings{dao2022flashattention,
  title={Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
