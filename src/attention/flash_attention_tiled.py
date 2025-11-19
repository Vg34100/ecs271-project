"""
Tiled FlashAttention (Steps 4-5)
=================================
Implements the full tiled FlashAttention algorithm from Section 4 of the FlashAttention paper.

This extends single-row FlashAttention to:
1. Process K and V in blocks (tiles) instead of one element at a time
2. Handle all query rows (not just one)
3. Support batches and multiple attention heads

Key insight: By processing in blocks, we can leverage vectorized operations
while still maintaining the memory efficiency of the online algorithm.
"""

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import torch

from src.attention.standard_attention import standard_attention_simple


def flash_attention_tiled_single_row(
    q_row: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 64
) -> torch.Tensor:
    """
    Compute attention for a single query row using tiled FlashAttention.

    This is the tiled algorithm from Section 4 (page 5-6 of the PDF):
    - Divide K and V into blocks of size block_size
    - For each block, compute local attention scores
    - Update running max, denominator, and output with rescaling

    Args:
        q_row: Single query vector of shape (head_dim,)
        K: Key matrix of shape (seq_len, head_dim)
        V: Value matrix of shape (seq_len, head_dim)
        block_size: Size of each tile (b in the paper)

    Returns:
        Output vector of shape (head_dim,)
    """
    seq_len, head_dim = K.shape
    scale = head_dim**-0.5
    num_blocks = (seq_len + block_size - 1) // block_size  # Ceiling division

    # Initialize states
    m = torch.tensor(float("-inf"), dtype=q_row.dtype, device=q_row.device)
    d = torch.tensor(0.0, dtype=q_row.dtype, device=q_row.device)
    o = torch.zeros(head_dim, dtype=q_row.dtype, device=q_row.device)

    # Process each block (tile)
    for block_idx in range(num_blocks):
        # Get block boundaries
        start = block_idx * block_size
        end = min(start + block_size, seq_len)

        # Extract current block of K and V
        K_block = K[start:end]  # (block_size, head_dim)
        V_block = V[start:end]  # (block_size, head_dim)

        # Compute attention scores for this block: x_block = q @ K_block.T
        # Shape: (block_size,)
        x_block = (q_row @ K_block.T) * scale

        # Compute local maximum within the block
        m_local = x_block.max()

        # Save old states
        m_old = m.clone()
        d_old = d.clone()

        # Update global maximum
        m = torch.maximum(m, m_local)

        # Compute exp(x_i - m) for all elements in block
        exp_scores = torch.exp(x_block - m)  # (block_size,)

        # Update denominator with rescaling
        # d_i' = d_{i-1}' * exp(m_{i-1} - m_i) + sum(exp(x_j - m_i))
        d = d_old * torch.exp(m_old - m) + exp_scores.sum()

        # Update output with rescaling
        # o_i' = o_{i-1}' * (d_{i-1}'/d_i') * exp(m_{i-1} - m_i) + sum(exp(x_j - m_i)/d_i' * V[j])
        rescale_factor = (d_old / d) * torch.exp(m_old - m)
        weighted_values = (
            exp_scores.unsqueeze(-1) / d
        ) * V_block  # (block_size, head_dim)
        o = o * rescale_factor + weighted_values.sum(dim=0)

    return o


def flash_attention_tiled(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 64
) -> torch.Tensor:
    """
    Full tiled FlashAttention for all query rows, batches, and heads.

    This is the complete implementation that:
    - Processes all query rows
    - Handles batch dimension
    - Handles multiple attention heads
    - Uses tiling for memory efficiency

    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        block_size: Size of each tile for K/V

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = head_dim**-0.5
    num_blocks = (seq_len + block_size - 1) // block_size

    # Initialize output tensor
    O = torch.zeros_like(Q)

    # Process each batch and head (these are fully parallel in real impl)
    for b in range(batch_size):
        for h in range(num_heads):
            # Extract Q, K, V for this batch and head
            Q_bh = Q[b, h]  # (seq_len, head_dim)
            K_bh = K[b, h]  # (seq_len, head_dim)
            V_bh = V[b, h]  # (seq_len, head_dim)

            # Process each query row
            for q_idx in range(seq_len):
                q_row = Q_bh[q_idx]  # (head_dim,)

                # Initialize states for this query row
                m = torch.tensor(float("-inf"), dtype=Q.dtype, device=Q.device)
                d = torch.tensor(0.0, dtype=Q.dtype, device=Q.device)
                o = torch.zeros(head_dim, dtype=Q.dtype, device=Q.device)

                # Process K/V in blocks
                for block_idx in range(num_blocks):
                    start = block_idx * block_size
                    end = min(start + block_size, seq_len)

                    K_block = K_bh[start:end]
                    V_block = V_bh[start:end]

                    # Compute attention scores for block
                    x_block = (q_row @ K_block.T) * scale

                    # Local maximum
                    m_local = x_block.max()

                    # Save and update states
                    m_old = m.clone()
                    d_old = d.clone()
                    m = torch.maximum(m, m_local)

                    # Rescaling
                    exp_scores = torch.exp(x_block - m)
                    d = d_old * torch.exp(m_old - m) + exp_scores.sum()

                    rescale_factor = (d_old / d) * torch.exp(m_old - m)
                    weighted_values = (exp_scores.unsqueeze(-1) / d) * V_block
                    o = o * rescale_factor + weighted_values.sum(dim=0)

                O[b, h, q_idx] = o

    return O


def flash_attention_tiled_vectorized(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 64
) -> torch.Tensor:
    """
    Vectorized tiled FlashAttention - processes all query rows in parallel.

    This version vectorizes over query rows (still sequential over K/V blocks),
    which is more efficient than the triple-nested loop version.

    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        block_size: Size of each tile for K/V

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    scale = head_dim**-0.5
    num_blocks = (seq_len + block_size - 1) // block_size

    # Initialize states for all queries
    # Shape: (batch_size, num_heads, seq_len)
    m = torch.full(
        (batch_size, num_heads, seq_len), float("-inf"), dtype=Q.dtype, device=Q.device
    )
    d = torch.zeros((batch_size, num_heads, seq_len), dtype=Q.dtype, device=Q.device)
    # Shape: (batch_size, num_heads, seq_len, head_dim)
    o = torch.zeros_like(Q)

    # Process K/V blocks sequentially (this is the tiling dimension)
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, seq_len)
        actual_block_size = end - start

        # Extract K/V block
        # Shape: (batch_size, num_heads, block_size, head_dim)
        K_block = K[:, :, start:end, :]
        V_block = V[:, :, start:end, :]

        # Compute attention scores for all queries with this K block
        # Q: (batch, heads, seq_len, head_dim)
        # K_block: (batch, heads, block_size, head_dim)
        # Result: (batch, heads, seq_len, block_size)
        scores_block = torch.matmul(Q, K_block.transpose(-2, -1)) * scale

        # Local maximum for each query
        # Shape: (batch_size, num_heads, seq_len)
        m_local = scores_block.max(dim=-1).values

        # Save old states
        m_old = m.clone()
        d_old = d.clone()

        # Update maximum
        m = torch.maximum(m, m_local)

        # Compute exp(scores - m) for numerical stability
        # Shape: (batch, heads, seq_len, block_size)
        exp_scores = torch.exp(scores_block - m.unsqueeze(-1))

        # Update denominator
        # Shape: (batch, heads, seq_len)
        d = d_old * torch.exp(m_old - m) + exp_scores.sum(dim=-1)

        # Update output with rescaling
        # rescale_factor: (batch, heads, seq_len)
        rescale_factor = (d_old / d) * torch.exp(m_old - m)

        # weighted_values: (batch, heads, seq_len, head_dim)
        # exp_scores: (batch, heads, seq_len, block_size)
        # V_block: (batch, heads, block_size, head_dim)
        weighted_values = torch.matmul(exp_scores, V_block) / d.unsqueeze(-1)

        o = o * rescale_factor.unsqueeze(-1) + weighted_values

    return o


def verify_tiled_flash_attention():
    """
    Verify that tiled FlashAttention produces correct results.
    """
    print("=" * 60)
    print("Verifying Tiled FlashAttention Correctness")
    print("=" * 60)

    torch.manual_seed(42)

    # Test 1: Single row tiled version
    print("\nTest 1: Single row tiled version (seq_len=128, head_dim=32)")
    seq_len = 128
    head_dim = 32
    block_size = 32

    q_row = torch.randn(head_dim)
    K = torch.randn(seq_len, head_dim)
    V = torch.randn(seq_len, head_dim)

    tiled_output = flash_attention_tiled_single_row(q_row, K, V, block_size=block_size)

    # Standard attention for single row
    scale = head_dim**-0.5
    scores = (q_row @ K.T) * scale
    attention_weights = torch.softmax(scores, dim=-1)
    standard_output = attention_weights @ V

    diff = torch.abs(tiled_output - standard_output).max()
    print(f"Block size: {block_size}")
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-5, f"Single row tiled mismatch! Diff: {diff}"
    print("PASS: Single row tiled test passed!")

    # Test 2: Full batched tiled version (non-vectorized)
    print("\n" + "-" * 60)
    print("Test 2: Full batched tiled (batch=1, heads=1, seq=64, dim=32)")
    batch_size = 1
    num_heads = 1
    seq_len = 64
    head_dim = 32
    block_size = 16

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    tiled_output = flash_attention_tiled(Q, K, V, block_size=block_size)
    standard_output = standard_attention_simple(Q, K, V)

    diff = torch.abs(tiled_output - standard_output).max()
    print(f"Block size: {block_size}")
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-5, f"Batched tiled mismatch! Diff: {diff}"
    print("PASS: Batched tiled test passed!")

    # Test 3: Vectorized tiled version
    print("\n" + "-" * 60)
    print("Test 3: Vectorized tiled (batch=2, heads=4, seq=128, dim=64)")
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64
    block_size = 32

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    vectorized_output = flash_attention_tiled_vectorized(Q, K, V, block_size=block_size)
    standard_output = standard_attention_simple(Q, K, V)

    diff = torch.abs(vectorized_output - standard_output).max()
    print(f"Block size: {block_size}")
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-5, f"Vectorized tiled mismatch! Diff: {diff}"
    print("PASS: Vectorized tiled test passed!")

    # Test 4: Different block sizes
    print("\n" + "-" * 60)
    print("Test 4: Different block sizes")
    batch_size = 1
    num_heads = 2
    seq_len = 256
    head_dim = 64

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    standard_output = standard_attention_simple(Q, K, V)

    for block_size in [16, 32, 64, 128, 256]:
        vectorized_output = flash_attention_tiled_vectorized(
            Q, K, V, block_size=block_size
        )
        diff = torch.abs(vectorized_output - standard_output).max()
        print(f"  Block size {block_size:3d}: Max diff = {diff:.2e}")
        assert diff < 1e-4, f"Block size {block_size} failed! Diff: {diff}"
    print("PASS: All block sizes passed!")

    # Test 5: Numerical stability
    print("\n" + "-" * 60)
    print("Test 5: Numerical stability (extreme values)")
    batch_size = 1
    num_heads = 1
    seq_len = 64
    head_dim = 32

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim) * 10
    K = torch.randn(batch_size, num_heads, seq_len, head_dim) * 10
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    vectorized_output = flash_attention_tiled_vectorized(Q, K, V, block_size=16)
    standard_output = standard_attention_simple(Q, K, V)

    diff = torch.abs(vectorized_output - standard_output).max()
    print(f"Max difference: {diff:.2e}")
    print(f"Has NaN: {torch.isnan(vectorized_output).any()}")
    print(f"Has Inf: {torch.isinf(vectorized_output).any()}")
    assert diff < 1e-4, f"Numerical stability test failed! Diff: {diff}"
    assert not torch.isnan(vectorized_output).any(), "NaN in output!"
    assert not torch.isinf(vectorized_output).any(), "Inf in output!"
    print("PASS: Numerical stability test passed!")

    print("\n" + "=" * 60)
    print("All Tiled FlashAttention Tests PASSED!")
    print("=" * 60)


def compare_implementations():
    """
    Compare different FlashAttention implementations.
    """
    print("=" * 60)
    print("Comparing FlashAttention Implementations")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 1
    num_heads = 1
    seq_len = 128
    head_dim = 64
    block_size = 32

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

    # Standard attention
    import time

    start = time.time()
    standard_output = standard_attention_simple(Q, K, V)
    standard_time = time.time() - start
    print(f"Standard attention: {standard_time * 1000:.2f} ms")

    # Tiled (non-vectorized) - slow due to Python loops
    start = time.time()
    tiled_output = flash_attention_tiled(Q, K, V, block_size=block_size)
    tiled_time = time.time() - start
    diff_tiled = torch.abs(tiled_output - standard_output).max()
    print(f"Tiled (loop): {tiled_time * 1000:.2f} ms, Max diff: {diff_tiled:.2e}")

    # Vectorized tiled
    start = time.time()
    vectorized_output = flash_attention_tiled_vectorized(Q, K, V, block_size=block_size)
    vectorized_time = time.time() - start
    diff_vectorized = torch.abs(vectorized_output - standard_output).max()
    print(
        f"Tiled (vectorized): {vectorized_time * 1000:.2f} ms, Max diff: {diff_vectorized:.2e}"
    )
    print(
        f"Tiled (vectorized): {vectorized_time * 1000:.2f} ms, Max diff: {diff_vectorized:.2e}"
    )

    print(f"\nNote: Python loops are slow. Real FlashAttention uses CUDA kernels.")
    print(f"The key benefit is MEMORY, not speed (in this pure PyTorch impl).")


if __name__ == "__main__":
    verify_tiled_flash_attention()
    print("\n")
    compare_implementations()
