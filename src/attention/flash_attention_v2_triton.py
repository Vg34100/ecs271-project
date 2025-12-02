# flash_attention_v2_triton.py



import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import torch
import triton
import triton.language as tl
from src.attention.standard_attention import standard_attention_simple
from src.attention.flash_attention_v2 import flash_attention_v2_vectorized

@triton.jit
def flash_attn_v2_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_q_bh, stride_q_m, stride_q_d,
    stride_k_bh, stride_k_n, stride_k_d,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_o_bh, stride_o_m, stride_o_d,
    n_ctx, head_dim,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Q, K, V, O: [B*H, L, D] contiguous-like tensors

    每個 Triton program 負責：某個 (bh, m) 的一整 row attention。
    內部在 N 維度上分 block loop (FlashAttention-2 style).
    """

    pid = tl.program_id(0)  # 0..(B*H*L-1)
    bh = pid // n_ctx       # batch*head index
    m  = pid % n_ctx        # query index

    # head_dim / D's offsets
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # pointed row's Q
    q_ptrs = Q_ptr + bh * stride_q_bh + m * stride_q_m + offs_d * stride_q_d
    q = tl.where(d_mask, tl.load(q_ptrs, mask=d_mask, other=0.0), 0.0)

    # initialize FA-2 state：m = -inf, d = 0, acc = 0
    m_i = tl.full((), -float("inf"), tl.float32)  # scalar
    d_i = tl.full((), 0.0, tl.float32)           # scalar
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32) # vector for output

    # 迴圈掃過所有 K/V blocks
    # suppose n_ctx is BLOCK_N's mutiple
    for start_n in range(0, n_ctx, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < n_ctx

        # load K_block: shape (BLOCK_N, head_dim)
        k_ptrs = (
            K_ptr
            + bh * stride_k_bh
            + offs_n[:, None] * stride_k_n
            + offs_d[None, :] * stride_k_d
        )
        k_block = tl.where(n_mask[:, None] & d_mask[None, :],
                           tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0),
                           0.0)

        # scores_block = q · k_j * scale  => shape (BLOCK_N,)
        # (BLOCK_N, D) * (D,) => (BLOCK_N,)
        scores_block = tl.sum(k_block * q[None, :], axis=1) * scale

        # block max value
        m_local = tl.max(scores_block, axis=0)
        # update global running max
        m_new = tl.maximum(m_i, m_local)

        # compute scaling factors
        exp_old = tl.exp(m_i - m_new)    # scalar
        # exp_scores = exp(scores_block - m_new)
        exp_scores = tl.exp(scores_block - m_new)

        # update denominator d_i
        d_new = d_i * exp_old + tl.sum(exp_scores, axis=0)

        # update unnormalized accumulator acc（FA-2 核心）
        # acc = acc * exp_old + Σ exp_scores_j * V_j
        v_ptrs = (
            V_ptr
            + bh * stride_v_bh
            + offs_n[:, None] * stride_v_n
            + offs_d[None, :] * stride_v_d
        )
        v_block = tl.where(n_mask[:, None] & d_mask[None, :],
                           tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0),
                           0.0)

        # sum_j exp_scores[j] * v_j  => shape (BLOCK_D,)
        contrib = tl.sum(v_block * exp_scores[:, None], axis=0)

        acc = acc * exp_old + contrib

        # 寫回狀態
        m_i = m_new
        d_i = d_new

    # last normalize：o = acc / d_i
    o = acc / d_i

    # write back O_ptr
    o_ptrs = O_ptr + bh * stride_o_bh + m * stride_o_m + offs_d * stride_o_d
    tl.store(o_ptrs, o, mask=d_mask)


def flash_attention_v2_triton(Q, K, V, block_n: int = 64, block_d: int = 64):
    """
    Python wrapper for the Triton kernel.

    Args:
        Q, K, V: [batch, n_heads, seq_len, head_dim]
    Returns:
        O: same shape as Q
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Use CUDA tensors for Triton"
    batch, n_heads, n_ctx, head_dim = Q.shape
    scale = head_dim ** -0.5

    # flatten batch and heads -> [B*H, L, D]
    BH = batch * n_heads
    Q_ = Q.contiguous().view(BH, n_ctx, head_dim)
    K_ = K.contiguous().view(BH, n_ctx, head_dim)
    V_ = V.contiguous().view(BH, n_ctx, head_dim)
    O_ = torch.empty_like(Q_)

    # strides (row-major: [bh, m, d])
    stride_q_bh, stride_q_m, stride_q_d = Q_.stride()
    stride_k_bh, stride_k_n, stride_k_d = K_.stride()
    stride_v_bh, stride_v_n, stride_v_d = V_.stride()
    stride_o_bh, stride_o_m, stride_o_d = O_.stride()

    # grid: one program per (bh, m)
    grid = (BH * n_ctx,)

    flash_attn_v2_fwd_kernel[grid](
        Q_, K_, V_, O_,
        stride_q_bh, stride_q_m, stride_q_d,
        stride_k_bh, stride_k_n, stride_k_d,
        stride_v_bh, stride_v_n, stride_v_d,
        stride_o_bh, stride_o_m, stride_o_d,
        n_ctx, head_dim,
        scale,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )

    return O_.view(batch, n_heads, n_ctx, head_dim)


def verify_flashattention_v2_triton():
    torch.manual_seed(0)
    device = "cuda"

    batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    

    out_ref = standard_attention_simple(Q, K, V)              # ground truth
    out_v2 = flash_attention_v2_vectorized(Q, K, V, 64)       # your PyTorch FA-2
    out_tri = flash_attention_v2_triton(Q, K, V, 64, 64)      # Triton kernel

    print("max diff Triton vs standard:",
          (out_tri - out_ref).abs().max().item())
    print("max diff Triton vs PyTorch FA2:",
          (out_tri - out_v2).abs().max().item())
    
def compare_fa2_vs_fa2_triton():
    """
    Compare FlashAttention-2 (PyTorch version) vs FlashAttention-2 (Triton kernel).
    Matching the style of compare_v1_vs_v2() in flash_attention_v2.py.
    """
    print("=" * 70)
    print("FlashAttention-2 PyTorch vs FlashAttention-2 Triton Comparison")
    print("=" * 70)

    import time
    torch.manual_seed(42)
    device = "cuda"

    batch_size, num_heads, seq_len, head_dim = 2, 4, 512, 64
    block_size = 64

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Block size: {block_size}")
    print()

    # Ground truth (standard attention)
    standard_output = standard_attention_simple(Q, K, V)

    # FlashAttention-2 PyTorch version
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fa2_output_torch = flash_attention_v2_vectorized(Q, K, V, block_size)
    torch.cuda.synchronize()
    fa2_torch_time = time.perf_counter() - t0

    # FlashAttention-2 Triton version
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fa2_output_triton = flash_attention_v2_triton(Q, K, V, block_n=block_size, block_d=head_dim)
    torch.cuda.synchronize()
    fa2_triton_time = time.perf_counter() - t0

    # Differences
    diff_torch = torch.abs(fa2_output_torch - standard_output).max().item()
    diff_triton = torch.abs(fa2_output_triton - standard_output).max().item()
    diff_between = torch.abs(fa2_output_torch - fa2_output_triton).max().item()

    # Print table
    print("Results:")
    print(f"  Standard Attention time:       N/A (baseline)")
    print(f"  FlashAttention-2 PyTorch time: {fa2_torch_time * 1000:.2f} ms")
    print(f"  FlashAttention-2 Triton time:  {fa2_triton_time * 1000:.2f} ms")
    print(f"  Speedup (Triton / PyTorch):    {fa2_torch_time / fa2_triton_time:.2f}x")
    print()
    print(f"  FA-2 PyTorch vs Standard max diff: {diff_torch:.3e}")
    print(f"  FA-2 Triton vs Standard max diff:  {diff_triton:.3e}")
    print(f"  PyTorch vs Triton max diff:        {diff_between:.3e}")
    print()

if __name__ == "__main__":          
   verify_flashattention_v2_triton()
   print("\n")
   compare_fa2_vs_fa2_triton()