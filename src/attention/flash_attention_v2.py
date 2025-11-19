"""
FlashAttention-2 Implementation
================================
Implements key improvements from FlashAttention-2 paper:
1. Reduced non-matmul FLOPs by not rescaling output in inner loop
2. Better parallelization (can parallelize over sequence length)
3. Optimized memory access patterns

Key difference from FlashAttention-1:
- FlashAttention-1: Rescale output O after each block
- FlashAttention-2: Keep unnormalized output, normalize at the end

This reduces the number of operations and improves numerical stability.
"""

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import torch
from src.attention.standard_attention import standard_attention_simple


def flash_attention_v2_single_row(
    q_row: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 64
) -> torch.Tensor:
    """
    FlashAttention-2 style: Keep unnormalized accumulator, normalize at end.

    Instead of:
        o_i' = o_{i-1}' * (d_{i-1}'/d_i') * exp(m_{i-1} - m_i) + exp(x_i - m_i)/d_i' * V[i]

    We do:
        o_i = o_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i) * V[i]
        (unnormalized accumulator)

    Then at the end:
        output = o_N / d_N

    This saves divisions in the inner loop.
    """
    seq_len, head_dim = K.shape
    scale = head_dim**-0.5
    num_blocks = (seq_len + block_size - 1) // block_size

    # Initialize states
    m = torch.tensor(float("-inf"), dtype=q_row.dtype, device=q_row.device)
    d = torch.tensor(0.0, dtype=q_row.dtype, device=q_row.device)
    # Keep UNNORMALIZED output (key difference from v1)
    o_unnorm = torch.zeros(head_dim, dtype=q_row.dtype, device=q_row.device)

    # Process each block
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)

        K_block = K[start:end]
        V_block = V[start:end]

        # Compute attention scores
        x_block = (q_row @ K_block.T) * scale

        # Local maximum
        m_local = x_block.max()

        # Save old states
        m_old = m.clone()

        # Update maximum
        m = torch.maximum(m, m_local)

        # Rescale factors
        exp_old = torch.exp(m_old - m)
        exp_scores = torch.exp(x_block - m)

        # Update denominator
        d = d * exp_old + exp_scores.sum()

        # Update UNNORMALIZED output (no division by d!)
        # This is the key optimization in FlashAttention-2
        o_unnorm = o_unnorm * exp_old + (exp_scores.unsqueeze(-1) * V_block).sum(dim=0)

    # Normalize at the end (only one division!)
    o = o_unnorm / d

    return o


def flash_attention_v2_vectorized(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 64
) -> torch.Tensor:
    """
    Vectorized FlashAttention-2 for full batches.

    Key optimizations over FlashAttention-1:
    1. Keeps unnormalized output accumulator
    2. Single normalization at the end
    3. Fewer FLOPs in inner loop

    Args:
        Q: (batch_size, num_heads, seq_len, head_dim)
        K: (batch_size, num_heads, seq_len, head_dim)
        V: (batch_size, num_heads, seq_len, head_dim)
        block_size: Tile size for K/V blocks

    Returns:
        Output tensor of same shape as Q
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = head_dim**-0.5
    num_blocks = (seq_len + block_size - 1) // block_size

    # Initialize states
    m = torch.full(
        (batch_size, num_heads, seq_len), float("-inf"), dtype=Q.dtype, device=Q.device
    )
    d = torch.zeros((batch_size, num_heads, seq_len), dtype=Q.dtype, device=Q.device)
    # UNNORMALIZED output accumulator
    o_unnorm = torch.zeros_like(Q)

    # Process K/V blocks
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)

        K_block = K[:, :, start:end, :]
        V_block = V[:, :, start:end, :]

        # Compute scores: (batch, heads, seq_len, block_size)
        scores_block = torch.matmul(Q, K_block.transpose(-2, -1)) * scale

        # Local maximum
        m_local = scores_block.max(dim=-1).values

        # Save old maximum
        m_old = m.clone()

        # Update maximum
        m = torch.maximum(m, m_local)

        # Compute exponentials
        exp_old = torch.exp(m_old - m)
        exp_scores = torch.exp(scores_block - m.unsqueeze(-1))

        # Update denominator
        d = d * exp_old + exp_scores.sum(dim=-1)

        # Update UNNORMALIZED output (key difference!)
        # No division by d in the loop
        o_unnorm = o_unnorm * exp_old.unsqueeze(-1) + torch.matmul(exp_scores, V_block)

    # Single normalization at the end
    o = o_unnorm / d.unsqueeze(-1)

    return o


def flash_attention_v2_with_causal_mask(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 64
) -> torch.Tensor:
    """
    FlashAttention-2 with causal masking (for autoregressive models).

    In causal attention, position i can only attend to positions <= i.
    This is crucial for language models like GPT.

    The mask is applied by setting future positions to -inf before softmax.
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = head_dim**-0.5
    num_blocks = (seq_len + block_size - 1) // block_size

    # Initialize states
    m = torch.full(
        (batch_size, num_heads, seq_len), float("-inf"), dtype=Q.dtype, device=Q.device
    )
    d = torch.zeros((batch_size, num_heads, seq_len), dtype=Q.dtype, device=Q.device)
    o_unnorm = torch.zeros_like(Q)

    # Process K/V blocks
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)

        K_block = K[:, :, start:end, :]
        V_block = V[:, :, start:end, :]

        # Compute scores
        scores_block = torch.matmul(Q, K_block.transpose(-2, -1)) * scale

        # Apply causal mask: position i cannot attend to position j > i
        # Create mask for this block
        # Query positions: 0 to seq_len-1
        # Key positions in this block: start to end-1
        query_pos = torch.arange(seq_len, device=Q.device).unsqueeze(-1)
        key_pos = torch.arange(start, end, device=Q.device).unsqueeze(0)

        # Mask where key position > query position (future positions)
        causal_mask = key_pos > query_pos  # (seq_len, block_size)

        # Apply mask (set future positions to -inf)
        scores_block = scores_block.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Rest is same as non-causal version
        # Handle all -inf case (when entire block is masked)
        m_local = scores_block.max(dim=-1).values
        m_local = torch.where(torch.isinf(m_local), m, m_local)

        m_old = m.clone()
        m = torch.maximum(m, m_local)

        exp_old = torch.exp(m_old - m)
        exp_scores = torch.exp(scores_block - m.unsqueeze(-1))
        # Masked positions have exp(-inf) = 0, so they don't contribute

        d = d * exp_old + exp_scores.sum(dim=-1)
        o_unnorm = o_unnorm * exp_old.unsqueeze(-1) + torch.matmul(exp_scores, V_block)

    # Normalize (handle case where d=0 for all-masked rows)
    o = o_unnorm / d.unsqueeze(-1).clamp(min=1e-9)

    return o


def compare_v1_vs_v2():
    """
    Compare FlashAttention-1 vs FlashAttention-2 implementations.
    """
    from src.attention.flash_attention_tiled import flash_attention_tiled_vectorized

    print("=" * 70)
    print("FlashAttention-1 vs FlashAttention-2 Comparison")
    print("=" * 70)

    torch.manual_seed(42)

    batch_size, num_heads, seq_len, head_dim = 2, 4, 512, 64
    block_size = 64

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Block size: {block_size}")
    print()

    # Standard attention (ground truth)
    standard_output = standard_attention_simple(Q, K, V)

    # FlashAttention-1
    import time

    start = time.perf_counter()
    v1_output = flash_attention_tiled_vectorized(Q, K, V, block_size=block_size)
    v1_time = time.perf_counter() - start

    # FlashAttention-2
    start = time.perf_counter()
    v2_output = flash_attention_v2_vectorized(Q, K, V, block_size=block_size)
    v2_time = time.perf_counter() - start

    # Compare results
    v1_diff = torch.abs(v1_output - standard_output).max().item()
    v2_diff = torch.abs(v2_output - standard_output).max().item()
    v1_v2_diff = torch.abs(v1_output - v2_output).max().item()

    print("Results:")
    print(f"  FlashAttention-1 time: {v1_time * 1000:.2f} ms")
    print(f"  FlashAttention-2 time: {v2_time * 1000:.2f} ms")
    print(f"  Speedup: {v1_time / v2_time:.2f}x")
    print()
    print(f"  V1 vs Standard max diff: {v1_diff:.2e}")
    print(f"  V2 vs Standard max diff: {v2_diff:.2e}")
    print(f"  V1 vs V2 max diff: {v1_v2_diff:.2e}")
    print()

    # Key algorithmic difference
    print("Key Differences:")
    print("  FlashAttention-1:")
    print("    - Normalizes output after each block")
    print("    - More divisions in inner loop")
    print("    - O_i = O_{i-1} * rescale_factor + new_contribution / d_i")
    print()
    print("  FlashAttention-2:")
    print("    - Keeps unnormalized accumulator")
    print("    - Single normalization at the end")
    print("    - O_unnorm_i = O_unnorm_{i-1} * exp_factor + sum(exp * V)")
    print("    - O_final = O_unnorm / d")
    print()
    print("  Result: Fewer FLOPs, potentially better numerical stability")


def verify_flashattention_v2():
    """
    Verify FlashAttention-2 correctness.
    """
    print("=" * 70)
    print("Verifying FlashAttention-2 Correctness")
    print("=" * 70)

    torch.manual_seed(42)

    # Test 1: Basic correctness
    print("\nTest 1: Basic correctness (batch=2, heads=4, seq=256, dim=64)")
    batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 64

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    standard_output = standard_attention_simple(Q, K, V)
    v2_output = flash_attention_v2_vectorized(Q, K, V, block_size=64)

    diff = torch.abs(v2_output - standard_output).max().item()
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-4, f"FlashAttention-2 failed! Diff: {diff}"
    print("PASS!")

    # Test 2: Different block sizes
    print("\nTest 2: Different block sizes")
    for block_size in [16, 32, 64, 128]:
        v2_output = flash_attention_v2_vectorized(Q, K, V, block_size=block_size)
        diff = torch.abs(v2_output - standard_output).max().item()
        print(f"  Block size {block_size}: Max diff = {diff:.2e}")
        assert diff < 1e-4
    print("PASS!")

    # Test 3: Causal masking
    print("\nTest 3: Causal masking (autoregressive)")
    batch_size, num_heads, seq_len, head_dim = 1, 1, 128, 32

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Standard causal attention
    scale = head_dim**-0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)
    standard_causal = torch.matmul(attn_weights, V)

    # Our causal FlashAttention-2
    v2_causal = flash_attention_v2_with_causal_mask(Q, K, V, block_size=32)

    diff = torch.abs(v2_causal - standard_causal).max().item()
    print(f"Max difference (causal): {diff:.2e}")
    assert diff < 1e-4, f"Causal FlashAttention-2 failed! Diff: {diff}"
    print("PASS!")

    print("\n" + "=" * 70)
    print("All FlashAttention-2 Tests PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    verify_flashattention_v2()
    print("\n")
    compare_v1_vs_v2()
