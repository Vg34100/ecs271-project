"""
FlashAttention Benchmarking (Step 6)
=====================================
Benchmark standard attention vs tiled FlashAttention.

Measures:
- Peak memory usage
- Forward pass time
- Numerical difference from standard attention
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import time
from typing import Dict, List, Tuple

import torch

from src.attention.flash_attention_tiled import flash_attention_tiled_vectorized
from src.attention.standard_attention import standard_attention_simple


def measure_peak_memory_cpu(func, *args, **kwargs):
    """
    Measure approximate peak memory usage for a function (CPU).

    Note: For accurate GPU memory measurement, use torch.cuda.max_memory_allocated()
    This is a simple approximation for CPU.
    """
    gc.collect()

    # Get initial memory (if tracemalloc is available)
    try:
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, peak / (1024 * 1024)  # Convert to MB
    except ImportError:
        # Fallback: just run the function
        result = func(*args, **kwargs)
        return result, 0.0


def estimate_memory_standard_attention(
    batch_size: int, num_heads: int, seq_len: int, head_dim: int
) -> float:
    """
    Estimate peak memory usage for standard attention (in MB).

    Peak memory includes:
    - Q, K, V tensors: 3 * batch * heads * seq * dim * 4 bytes
    - Attention scores: batch * heads * seq * seq * 4 bytes
    - Attention weights: batch * heads * seq * seq * 4 bytes
    - Output: batch * heads * seq * dim * 4 bytes
    """
    qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim * 4
    scores_memory = batch_size * num_heads * seq_len * seq_len * 4
    weights_memory = batch_size * num_heads * seq_len * seq_len * 4
    output_memory = batch_size * num_heads * seq_len * head_dim * 4

    # Peak is when we have Q, K, V, scores, weights, and output all in memory
    peak_memory = qkv_memory + scores_memory + weights_memory + output_memory
    return peak_memory / (1024 * 1024)  # MB


def estimate_memory_flash_attention(
    batch_size: int, num_heads: int, seq_len: int, head_dim: int, block_size: int
) -> float:
    """
    Estimate peak memory usage for FlashAttention (in MB).

    Peak memory includes:
    - Q, K, V tensors: 3 * batch * heads * seq * dim * 4 bytes
    - State tensors (m, d, o): batch * heads * seq * (1 + 1 + dim) * 4 bytes
    - One block of scores: batch * heads * seq * block_size * 4 bytes
    - Output: batch * heads * seq * dim * 4 bytes

    Key difference: We never materialize the full seq x seq attention matrix!
    """
    qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim * 4
    state_memory = batch_size * num_heads * seq_len * (1 + 1 + head_dim) * 4
    block_scores_memory = batch_size * num_heads * seq_len * block_size * 4
    output_memory = batch_size * num_heads * seq_len * head_dim * 4

    peak_memory = qkv_memory + state_memory + block_scores_memory + output_memory
    return peak_memory / (1024 * 1024)  # MB


def benchmark_single_config(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    block_size: int = 64,
    num_runs: int = 10,
) -> Dict:
    """
    Benchmark standard vs FlashAttention for a single configuration.
    """
    # Create random tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Warm up
    _ = standard_attention_simple(Q, K, V)
    _ = flash_attention_tiled_vectorized(Q, K, V, block_size=block_size)

    # Measure standard attention time
    standard_times = []
    for _ in range(num_runs):
        gc.collect()
        start = time.perf_counter()
        standard_output = standard_attention_simple(Q, K, V)
        standard_times.append(time.perf_counter() - start)

    # Measure FlashAttention time
    flash_times = []
    for _ in range(num_runs):
        gc.collect()
        start = time.perf_counter()
        flash_output = flash_attention_tiled_vectorized(Q, K, V, block_size=block_size)
        flash_times.append(time.perf_counter() - start)

    # Compute numerical difference
    max_diff = torch.abs(flash_output - standard_output).max().item()
    mean_diff = torch.abs(flash_output - standard_output).mean().item()

    # Estimate memory usage
    standard_memory = estimate_memory_standard_attention(
        batch_size, num_heads, seq_len, head_dim
    )
    flash_memory = estimate_memory_flash_attention(
        batch_size, num_heads, seq_len, head_dim, block_size
    )

    return {
        "seq_len": seq_len,
        "standard_time_ms": sum(standard_times) / len(standard_times) * 1000,
        "flash_time_ms": sum(flash_times) / len(flash_times) * 1000,
        "standard_memory_mb": standard_memory,
        "flash_memory_mb": flash_memory,
        "memory_savings_pct": (1 - flash_memory / standard_memory) * 100,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "speedup": sum(standard_times) / sum(flash_times),
    }


def run_benchmarks():
    """
    Run comprehensive benchmarks and create a comparison table.
    """
    print("=" * 80)
    print("FlashAttention Benchmarking")
    print("=" * 80)

    # Configuration
    batch_size = 1
    num_heads = 1
    head_dim = 64
    block_size = 64

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Block size: {block_size}")
    print()

    # Test different sequence lengths
    seq_lengths = [512, 1024, 2048, 4096]
    results = []

    print("Running benchmarks...")
    for seq_len in seq_lengths:
        print(f"  Testing seq_len={seq_len}...", end=" ", flush=True)
        try:
            result = benchmark_single_config(
                batch_size, num_heads, seq_len, head_dim, block_size
            )
            results.append(result)
            print(f"Done (max_diff={result['max_diff']:.2e})")
        except Exception as e:
            print(f"Failed: {e}")

    # Print results table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(
        f"{'Seq Len':>8} | {'Std Time':>10} | {'Flash Time':>11} | "
        f"{'Std Mem':>10} | {'Flash Mem':>11} | {'Mem Save':>9} | {'Max Diff':>10}"
    )
    print(
        f"{'':>8} | {'(ms)':>10} | {'(ms)':>11} | "
        f"{'(MB)':>10} | {'(MB)':>11} | {'(%)':>9} | {'':>10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r['seq_len']:>8} | {r['standard_time_ms']:>10.2f} | {r['flash_time_ms']:>11.2f} | "
            f"{r['standard_memory_mb']:>10.2f} | {r['flash_memory_mb']:>11.2f} | "
            f"{r['memory_savings_pct']:>8.1f}% | {r['max_diff']:>10.2e}"
        )

    print("-" * 80)

    # Summary
    print("\nKEY INSIGHTS:")
    print("-" * 80)

    if len(results) >= 2:
        mem_ratio_first = (
            results[0]["standard_memory_mb"] / results[0]["flash_memory_mb"]
        )
        mem_ratio_last = (
            results[-1]["standard_memory_mb"] / results[-1]["flash_memory_mb"]
        )

        print(f"1. Memory Efficiency:")
        print(
            f"   - At seq_len={results[0]['seq_len']}: Standard uses {mem_ratio_first:.1f}x more memory"
        )
        print(
            f"   - At seq_len={results[-1]['seq_len']}: Standard uses {mem_ratio_last:.1f}x more memory"
        )
        print(f"   - Memory savings grow with sequence length!")
        print()

        print(f"2. Memory Scaling:")
        print(f"   - Standard attention: O(L^2) - grows quadratically")
        print(
            f"   - FlashAttention: O(L * block_size) - grows linearly with block_size constant"
        )
        print()

        print(f"3. Numerical Accuracy:")
        max_diff_all = max(r["max_diff"] for r in results)
        print(f"   - Maximum difference across all tests: {max_diff_all:.2e}")
        print(
            f"   - All differences < 1e-4: {all(r['max_diff'] < 1e-4 for r in results)}"
        )
        print()

        print(f"4. Performance Note:")
        print(f"   - FlashAttention in pure PyTorch is SLOWER due to Python loops")
        print(f"   - Real benefit requires CUDA kernels (fused operations)")
        print(f"   - This implementation demonstrates the ALGORITHM, not the speed")

    print("\n" + "=" * 80)

    return results


def memory_scaling_analysis():
    """
    Analyze how memory scales with sequence length.
    """
    print("\n" + "=" * 80)
    print("MEMORY SCALING ANALYSIS")
    print("=" * 80)

    batch_size = 1
    num_heads = 8
    head_dim = 64
    block_size = 64

    print(
        f"\nConfiguration: batch={batch_size}, heads={num_heads}, dim={head_dim}, block={block_size}"
    )
    print()

    print(
        f"{'Seq Len':>10} | {'Standard Mem (MB)':>18} | {'Flash Mem (MB)':>16} | {'Ratio':>8}"
    )
    print("-" * 60)

    for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        std_mem = estimate_memory_standard_attention(
            batch_size, num_heads, seq_len, head_dim
        )
        flash_mem = estimate_memory_flash_attention(
            batch_size, num_heads, seq_len, head_dim, block_size
        )
        ratio = std_mem / flash_mem

        print(f"{seq_len:>10} | {std_mem:>18.2f} | {flash_mem:>16.2f} | {ratio:>7.2f}x")

    print("-" * 60)
    print("\nObservation: As sequence length increases, the memory advantage grows!")
    print(
        "This is because standard attention stores L x L matrix, while FlashAttention"
    )
    print("only stores L x block_size intermediate results.")


def block_size_analysis():
    """
    Analyze effect of different block sizes.
    """
    print("\n" + "=" * 80)
    print("BLOCK SIZE ANALYSIS")
    print("=" * 80)

    batch_size = 1
    num_heads = 1
    seq_len = 1024
    head_dim = 64

    print(
        f"\nConfiguration: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}"
    )
    print()

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    standard_output = standard_attention_simple(Q, K, V)

    print(
        f"{'Block Size':>12} | {'Flash Mem (MB)':>16} | {'Time (ms)':>12} | {'Max Diff':>12}"
    )
    print("-" * 60)

    for block_size in [16, 32, 64, 128, 256, 512]:
        flash_mem = estimate_memory_flash_attention(
            batch_size, num_heads, seq_len, head_dim, block_size
        )

        # Measure time
        times = []
        for _ in range(5):
            start = time.perf_counter()
            flash_output = flash_attention_tiled_vectorized(
                Q, K, V, block_size=block_size
            )
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times) * 1000
        max_diff = torch.abs(flash_output - standard_output).max().item()

        print(
            f"{block_size:>12} | {flash_mem:>16.2f} | {avg_time:>12.2f} | {max_diff:>12.2e}"
        )

    print("-" * 60)
    print(
        "\nObservation: Larger block sizes use more memory but have fewer iterations."
    )
    print(
        "\nObservation: Larger block sizes use more memory but have fewer iterations."
    )
    print("\nObservation: Larger block sizes use more memory but have fewer iterations.")
    print("In a real CUDA implementation, block size is tuned for GPU architecture.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run main benchmarks
    results = run_benchmarks()

    # Additional analyses
    memory_scaling_analysis()
    block_size_analysis()
