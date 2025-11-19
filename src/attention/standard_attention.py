"""
Standard Attention Implementation (Baseline)
============================================
This implements the naive attention mechanism that explicitly materializes
the full attention matrix in memory.

Formula: O = softmax(Q @ K.T) @ V

This serves as our baseline for correctness testing and memory comparison.
"""

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F


def standard_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Compute standard scaled dot-product attention.

    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    # Get dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape

    # Scale factor for numerical stability (standard in transformers)
    scale = head_dim**-0.5

    # Step 1: Compute attention scores (QK^T)
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    # This materializes the full L x L attention matrix!
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    print(f"  Attention scores shape: {scores.shape}")
    print(
        f"  Memory for scores: {scores.element_size() * scores.nelement() / 1024 / 1024:.2f} MB"
    )

    # Step 2: Apply softmax to get attention weights
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    attention_weights = F.softmax(scores, dim=-1)
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(
        f"  Memory for weights: {attention_weights.element_size() * attention_weights.nelement() / 1024 / 1024:.2f} MB"
    )

    # Step 3: Apply attention weights to values
    # Shape: (batch_size, num_heads, seq_len, head_dim)
    output = torch.matmul(attention_weights, V)

    return output


def standard_attention_simple(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Simplified version without print statements for benchmarking.
    """
    scale = Q.shape[-1] ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output


def measure_memory_usage():
    """
    Demonstrate memory usage of standard attention at different sequence lengths.
    """
    print("=" * 60)
    print("Standard Attention Memory Usage Analysis")
    print("=" * 60)

    # Test with single batch, single head for clarity
    batch_size = 1
    num_heads = 1
    head_dim = 64

    for seq_len in [128, 256, 512, 1024, 2048]:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        # Create random Q, K, V tensors
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Calculate theoretical memory for attention matrix
        attention_matrix_size = seq_len * seq_len * 4  # float32 = 4 bytes
        print(
            f"  Theoretical attention matrix size: {attention_matrix_size / 1024 / 1024:.2f} MB"
        )
        print(f"  Theoretical attention matrix size: {attention_matrix_size / 1024 / 1024:.2f} MB")

        # Compute attention
        output = standard_attention(Q, K, V)
        print(f"  Output shape: {output.shape}")

        # Verify output shape
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    print("\n" + "=" * 60)
    print("Key Insight: Memory grows quadratically with sequence length!")
    print("  - seq_len=1024: ~4 MB for attention matrix")
    print("  - seq_len=2048: ~16 MB for attention matrix")
    print("  - seq_len=4096: ~64 MB for attention matrix")
    print("  - seq_len=8192: ~256 MB for attention matrix")
    print("=" * 60)


if __name__ == "__main__":
    measure_memory_usage()

    # Quick correctness check
    print("\nQuick Correctness Check:")
    Q = torch.randn(2, 4, 128, 64)  # batch=2, heads=4, seq=128, dim=64
    K = torch.randn(2, 4, 128, 64)
    V = torch.randn(2, 4, 128, 64)

    output = standard_attention_simple(Q, K, V)
    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output contains NaN: {torch.isnan(output).any()}")
    print(f"Output contains Inf: {torch.isinf(output).any()}")
