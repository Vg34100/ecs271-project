# FlashAttention Implementation

**ECS271 Machine Learning Project**
**Team:** Shuang Ma, Pablo Rodriguez, Pei Yu Lin
**Fall 2025**

An educational implementation of the FlashAttention algorithm attempting to replicate the memory-efficient attention mechanism described in the paper.

---

## Project Overview

This project attempts to implement FlashAttention, a memory-efficient approach to computing self-attention in Transformer models. The goal is to understand how the algorithm avoids materializing the full O(L²) attention matrix by using tiling and online softmax computation.

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
2. **Online Softmax** - 2-pass algorithm from Section 3 of the paper
3. **Single-Row FlashAttention** - Core recurrence relation (Equation 14)
4. **Tiled FlashAttention** - Full tiled implementation with blocks
5. **FlashAttention-2** - Variant with fewer FLOPs
6. **Triton Kernel** - Basic GPU kernel (educational version)

All implementations are compared against PyTorch's standard attention for correctness.

---

## Notes

- The PyTorch implementation uses Python loops and is slower than standard attention - real FlashAttention uses optimized CUDA kernels
- Focus is on demonstrating the algorithm and memory efficiency, not speed
- GPU benchmarks limited to 8K sequence length due to hardware constraints
- Numerical accuracy: max difference < 5×10⁻⁷ from standard attention

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
