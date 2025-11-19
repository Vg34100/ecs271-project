# Understanding FlashAttention: A Beginner's Guide

## What Problem Does FlashAttention Solve?

### The Attention Mechanism (Quick Recap)
In Transformers (the architecture behind GPT, BERT, etc.), **attention** lets the model look at all parts of the input when processing each word. The formula is:

```
Output = softmax(Q @ K^T) @ V
```

Where:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What information do I have?"
- **V** (Value): "What's the actual content?"

### The Problem: Memory Explosion

When you compute `Q @ K^T`, you create a matrix of size **L × L** where L is the sequence length (number of tokens).

**Example:**
- 1,000 tokens → 1,000 × 1,000 = 1 million numbers
- 10,000 tokens → 10,000 × 10,000 = 100 million numbers
- 32,000 tokens → 32,000 × 32,000 = 1 BILLION numbers

Each number takes 4 bytes (float32), so:
- 1K tokens: 4 MB
- 10K tokens: 400 MB
- 32K tokens: 4 GB (just for ONE attention layer!)

This is called **quadratic memory complexity** - memory grows with the SQUARE of sequence length.

### GPU Memory Hierarchy

To understand FlashAttention, you need to know how GPU memory works (Shuang Ma showed me a diagram similar to this): 

```
┌─────────────────────────────┐
│   HBM (High Bandwidth Memory)  │  <- LARGE but SLOW (40-80 GB)
│   "Main GPU Memory"            │     Reading/writing here is expensive
└─────────────────────────────┘
              ↑↓
┌─────────────────────────────┐
│   SRAM (Shared Memory/Cache)  │  <- SMALL but FAST (20-200 KB)
│   "On-chip Memory"             │     Operations here are cheap
└─────────────────────────────┘
              ↑↓
┌─────────────────────────────┐
│   GPU Registers                │  <- TINY but FASTEST
└─────────────────────────────┘
```

**The Bottleneck:** Moving data between HBM and SRAM is SLOW. Standard attention:
1. Writes Q@K^T (L×L matrix) to HBM
2. Reads it back to compute softmax
3. Writes softmax result to HBM
4. Reads it back to multiply by V

This is **memory-bound** - the GPU spends more time moving data than computing

## How FlashAttention Solves This

### Key Insight: Never Materialize the Full L×L Matrix

Instead of computing the entire attention matrix at once:
1. **Tile the computation** - process small blocks at a time
2. **Keep intermediate results in SRAM** - never write L×L to HBM
3. **Use online algorithms** - update results incrementally

### The Online Softmax Trick

Normal softmax needs to see ALL values first:
```python
# Need to know max of ALL values for numerical stability
max_val = max(x1, x2, ..., xN)
softmax = exp(xi - max_val) / sum(exp(xj - max_val))
```

**Problem:** We can't compute softmax until we've seen everything

**Solution - Online Softmax:** Update the softmax incrementally:
```python
# Process one element at a time
m = -infinity  # running maximum
d = 0          # running denominator

for xi in values:
    m_old = m
    m = max(m, xi)
    # RESCALE previous results because max changed!
    d = d * exp(m_old - m) + exp(xi - m)
```

The key is **rescaling** when the maximum changes. This is mathematically equivalent to regular softmax but processes data incrementally.

### The Rescaling Formula (I get a bit lost at this part)

When we see a new block of data and the maximum changes from `m_old` to `m_new`:

1. **Rescale the denominator:**
```python
d_new = d_old * exp(m_old - m_new) + sum(exp(new_values - m_new))
```

2. **Rescale the output:**
```python
o_new = o_old * (d_old/d_new) * exp(m_old - m_new) + sum(exp(new_values - m_new)/d_new * V_new)
```

This is equation (14) from the FlashAttention paper.

## Why This Matters for Your GPU

### Memory Savings

| Sequence Length | Standard Attention | FlashAttention | Savings |
|----------------|-------------------|----------------|---------|
| 1K tokens      | 4 MB              | 0.3 MB         | 13x     |
| 4K tokens      | 64 MB             | 1.2 MB         | 53x     |
| 16K tokens     | 1 GB              | 5 MB           | 200x    |

### What This Enables

1. **Longer Context Windows:** GPT-4 can handle 128K tokens because of techniques like this
2. **Larger Batch Sizes:** Fit more samples in memory = faster training
3. **Cheaper Inference:** Less memory = can use smaller/cheaper GPUs

## FlashAttention Variants

### FlashAttention-1 (Original)
- Basic tiling and online softmax
- What we implemented
- ~2-4x speedup over standard attention

### FlashAttention-2
- Better parallelization (across sequence length, not just batch)
- Reduced non-matmul FLOPs
- Better GPU occupancy
- ~2x faster than FlashAttention-1

### FlashAttention-3 (Latest)
- Specific to Hopper GPUs (H100)
- Uses asynchronous operations
- Even better hardware utilization

# AI Summary to Help Understanding:
## Connecting to Your Project

Your proposal mentions:
- **Variable sequence lengths (1K to 32K)** - This is to show the quadratic scaling problem
- **Model dimensions (1024-8192)** - Hidden size affects memory too, but less than sequence length
- **Wall-clock time** - Actual speed measurement
- **GPU memory footprint** - Peak memory usage during computation

## What Makes a Good Implementation?

1. **Correctness** - Results match standard attention (within floating-point error)
2. **Memory Efficiency** - Peak memory is O(L) not O(L²)
3. **Speed** - Should be faster (but this requires CUDA kernels)
4. **Numerical Stability** - No overflow/underflow even with large values

## Common Confusions

**Q: Why is our pure PyTorch FlashAttention slower than standard attention?**
A: Because Python loops are slow! The real speedup comes from:
- Fused CUDA kernels (no kernel launch overhead)
- Actual SRAM management on GPU
- Hardware-specific optimizations

**Q: If it's slower in PyTorch, what's the point?**
A: We're demonstrating the ALGORITHM. The memory savings are real! Speed requires CUDA kernels.

**Q: What's the difference between our implementation and the real one?**
A: Real FlashAttention:
- Written in CUDA/Triton (not Python)
- Actually manages GPU SRAM
- Fuses all operations into one kernel
- Uses hardware-specific optimizations

**Q: Why do we need to "rescale" when the maximum changes?**
A: Numerical stability. `exp(1000)` overflows, but `exp(1000 - 1000) = exp(0) = 1` is fine. When we see a larger maximum, all previous exponentials need to be adjusted relative to the new max.

## Resources

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - Original paper
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691) - Improved version
- [Online Softmax Paper](https://arxiv.org/abs/1805.02867) - The mathematical foundation
- [GPU Memory Hierarchy](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) - NVIDIA's explanation
