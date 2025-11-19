"""
FlashAttention with Triton Kernel
===================================
A simplified Triton implementation of FlashAttention.

Triton is a language and compiler for writing GPU kernels in Python.
This is how the actual FlashAttention is implemented (not raw CUDA).

Note: This is a simplified educational version. The real FlashAttention
Triton kernel is more optimized with better memory access patterns.
"""

import sys
from pathlib import Path
# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import os
# Set Triton cache to project directory (avoids /tmp clutter)
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["TRITON_CACHE_DIR"] = os.path.join(_script_dir, ".triton_cache")

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attention_forward_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_ok,
    seq_len,
    scale,
    BLOCK_M: tl.constexpr,  # Block size for queries
    BLOCK_N: tl.constexpr,  # Block size for keys/values
    BLOCK_K: tl.constexpr,  # Head dimension
):
    """
    Triton kernel for FlashAttention forward pass.

    Each program instance computes one block of output (BLOCK_M rows).
    """
    # Get program ID
    pid_m = tl.program_id(0)  # Which query block
    pid_bh = tl.program_id(1)  # Which batch and head

    # Calculate batch and head indices
    num_heads = stride_qb // stride_qh
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    # Offset pointers to current batch and head
    Q_ptr = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    K_ptr = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    V_ptr = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    O_ptr = O_ptr + batch_idx * stride_ob + head_idx * stride_oh

    # Compute row indices for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulators
    # m_i: running maximum (for numerical stability)
    # l_i: running sum of exp(scores - m)
    # acc: unnormalized output accumulator
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Load Q block (stays in SRAM for all K/V blocks)
    # Q shape: (BLOCK_M, BLOCK_K)
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q_mask = offs_m[:, None] < seq_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Loop over K/V blocks
    num_blocks_n = tl.cdiv(seq_len, BLOCK_N)
    for block_n_idx in range(num_blocks_n):
        # Load K block - need to load as (BLOCK_N, BLOCK_K) then transpose
        k_offs_n = block_n_idx * BLOCK_N + offs_n
        # K is stored as (seq_len, head_dim), load block and transpose for matmul
        k_ptrs = K_ptr + k_offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k_mask = k_offs_n[:, None] < seq_len
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)  # (BLOCK_N, BLOCK_K)

        # Transpose K for Q @ K^T computation
        k = tl.trans(k_block)  # (BLOCK_K, BLOCK_N)

        # Compute attention scores: Q @ K^T
        # q: (BLOCK_M, BLOCK_K), k: (BLOCK_K, BLOCK_N)
        # scores: (BLOCK_M, BLOCK_N)
        scores = tl.dot(q, k) * scale

        # Mask out invalid positions
        scores = tl.where(k_offs_n[None, :] < seq_len, scores, float("-inf"))

        # Online softmax: compute new maximum
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))

        # Rescale old values
        alpha = tl.exp(m_i - m_i_new)

        # Compute exp(scores - m_i_new)
        p = tl.exp(scores - m_i_new[:, None])

        # Update running sum
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load V block
        v_ptrs = V_ptr + k_offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v_mask = k_offs_n[:, None] < seq_len
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Update accumulator
        # acc: (BLOCK_M, BLOCK_K), p: (BLOCK_M, BLOCK_N), v: (BLOCK_N, BLOCK_K)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        # Update maximum
        m_i = m_i_new

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output
    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    o_mask = offs_m[:, None] < seq_len
    tl.store(o_ptrs, acc, mask=o_mask)


def flash_attention_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """
    FlashAttention using Triton kernel.

    Args:
        Q: Query tensor (batch, heads, seq_len, head_dim)
        K: Key tensor (batch, heads, seq_len, head_dim)
        V: Value tensor (batch, heads, seq_len, head_dim)
        block_m: Block size for queries
        block_n: Block size for keys/values

    Returns:
        Output tensor (batch, heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape

    # Ensure tensors are contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Allocate output
    O = torch.empty_like(Q)

    # Scale factor
    scale = 1.0 / math.sqrt(head_dim)

    # Grid: (num_query_blocks, batch * heads)
    grid = (triton.cdiv(seq_len, block_m), batch_size * num_heads)

    # Launch kernel
    _flash_attention_forward_kernel[grid](
        Q,
        K,
        V,
        O,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        O.stride(3),
        seq_len,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=head_dim,
    )

    return O


def verify_triton_kernel():
    """Verify Triton kernel produces correct results."""
    print("=" * 70)
    print("Verifying Triton FlashAttention Kernel")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping Triton test")
        return False

    torch.manual_seed(42)
    device = "cuda"

    # Test configuration
    batch_size = 2
    num_heads = 4
    seq_len = 256
    head_dim = 64

    print(f"\nConfiguration:")
    print(f"  Batch: {batch_size}, Heads: {num_heads}")
    print(f"  Seq len: {seq_len}, Head dim: {head_dim}")
    print()

    # Create tensors
    Q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32
    )
    K = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32
    )
    V = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32
    )

    # Standard attention (ground truth)
    from standard_attention import standard_attention_simple

    standard_output = standard_attention_simple(Q, K, V)

    # Triton kernel
    print("Running Triton kernel...")
    try:
        triton_output = flash_attention_triton(Q, K, V, block_m=64, block_n=64)

        # Compare
        max_diff = torch.abs(triton_output - standard_output).max().item()
        mean_diff = torch.abs(triton_output - standard_output).mean().item()

        print(f"Max difference: {max_diff:.2e}")
        print(f"Mean difference: {mean_diff:.2e}")

        if max_diff < 2e-3:  # Allow small GPU numerical differences
            print("PASS: Triton kernel produces correct results!")
            print(
                "(Note: Small differences expected due to GPU floating-point optimizations)"
            )
            return True
        else:
            print(f"WARNING: Large difference detected: {max_diff}")
            return False

    except Exception as e:
        print(f"Error running Triton kernel: {e}")
        import traceback

        traceback.print_exc()
        return False


def benchmark_triton_vs_pytorch():
    """Benchmark Triton kernel vs PyTorch implementations."""
    print("\n" + "=" * 70)
    print("Benchmarking: Standard vs PyTorch FlashAttention vs Triton")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    from src.attention.standard_attention import standard_attention_simple
    from src.attention.flash_attention_tiled import flash_attention_tiled_vectorized
    import time

    torch.manual_seed(42)
    device = "cuda"

    batch_size = 1
    num_heads = 8
    head_dim = 64

    print(f"\nConfig: batch={batch_size}, heads={num_heads}, dim={head_dim}")
    print()

    results = []

    for seq_len in [512, 1024, 2048, 4096, 8192]:
        print(f"Sequence length: {seq_len}")

        Q = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32
        )
        K = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32
        )
        V = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32
        )

        # Warmup
        _ = standard_attention_simple(Q, K, V)
        torch.cuda.synchronize()

        # Standard attention
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out_std = standard_attention_simple(Q, K, V)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        std_time = sum(times) / len(times) * 1000

        # PyTorch FlashAttention
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out_pytorch = flash_attention_tiled_vectorized(Q, K, V, block_size=64)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        pytorch_time = sum(times) / len(times) * 1000

        # Triton FlashAttention
        try:
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_triton = flash_attention_triton(Q, K, V, block_m=64, block_n=64)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            triton_time = sum(times) / len(times) * 1000

            diff = torch.abs(out_triton - out_std).max().item()

            print(f"  Standard:   {std_time:7.2f} ms")
            print(
                f"  PyTorch FA: {pytorch_time:7.2f} ms (vs std: {pytorch_time / std_time:.2f}x)"
            )
            print(
                f"  Triton FA:  {triton_time:7.2f} ms (vs std: {triton_time / std_time:.2f}x)"
            )
            print(f"  Triton max diff: {diff:.2e}")
            print()

            results.append(
                {
                    "seq_len": seq_len,
                    "std_ms": std_time,
                    "pytorch_fa_ms": pytorch_time,
                    "triton_fa_ms": triton_time,
                    "triton_diff": diff,
                }
            )

        except Exception as e:
            print(f"  Triton failed: {e}")
            print()

    return results


if __name__ == "__main__":
    success = verify_triton_kernel()

    if success:
        benchmark_triton_vs_pytorch()
    else:
        print("\nTriton kernel verification failed. Please check the implementation.")
