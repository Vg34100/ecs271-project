"""
Single-Row FlashAttention (Step 3)
===================================
Implements FlashAttention for a single query row without tiling.

This is the core algorithm from Section 4 of the FlashAttention paper.
For a single query row, we compute attention with V without materializing
the full attention scores.

Key recurrence relation (equation 14):
    o_i' = o_{i-1}' * (d_{i-1}' / d_i') * exp(m_{i-1} - m_i) + exp(x_i - m_i) / d_i' * V[i]

States tracked:
- m_i: running maximum of attention scores
- d_i': running denominator (rescaled)
- o_i': partial output (rescaled)
"""

import sys
from pathlib import Path


# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import torch

from src.attention.standard_attention import standard_attention_simple


def flash_attention_single_row(
    q_row: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Compute attention output for a single query row using FlashAttention algorithm.

    This implements the 1-pass algorithm from Section 4:
    - Process one K/V pair at a time
    - Track m, d', and o' states
    - Never materialize full attention scores

    Args:
        q_row: Single query vector of shape (head_dim,)
        K: Key matrix of shape (seq_len, head_dim)
        V: Value matrix of shape (seq_len, head_dim)

    Returns:
        Output vector of shape (head_dim,)
    """
    seq_len, head_dim = K.shape
    scale = head_dim**-0.5

    # Initialize states (equation 14 initial conditions)
    # m_0 = -inf, d_0' = 0, o_0' = zeros
    m = torch.tensor(float("-inf"), dtype=q_row.dtype, device=q_row.device)
    d = torch.tensor(0.0, dtype=q_row.dtype, device=q_row.device)
    o = torch.zeros(head_dim, dtype=q_row.dtype, device=q_row.device)

    # Process each K/V pair sequentially
    for i in range(seq_len):
        # Compute attention score for this position: x_i = q . k_i
        x_i = torch.dot(q_row, K[i]) * scale

        # Save old states
        m_old = m.clone()
        d_old = d.clone()

        # Update maximum: m_i = max(m_{i-1}, x_i)
        m = torch.maximum(m, x_i)

        # Update denominator with rescaling (equation 10):
        # d_i' = d_{i-1}' * exp(m_{i-1} - m_i) + exp(x_i - m_i)
        d = d_old * torch.exp(m_old - m) + torch.exp(x_i - m)

        # Update output with rescaling (equation 14):
        # o_i' = o_{i-1}' * (d_{i-1}'/d_i') * exp(m_{i-1} - m_i) + exp(x_i - m_i)/d_i' * V[i]
        o = o * (d_old / d) * torch.exp(m_old - m) + (torch.exp(x_i - m) / d) * V[i]

    return o


def flash_attention_single_row_verbose(
    q_row: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> torch.Tensor:
    """
    Same as flash_attention_single_row but with detailed print statements.
    """
    seq_len, head_dim = K.shape
    scale = head_dim**-0.5

    print(f"FlashAttention Single Row Algorithm")
    print(f"Query shape: {q_row.shape}")
    print(f"K shape: {K.shape}, V shape: {V.shape}")
    print(f"Scale factor: {scale:.4f}")
    print("-" * 50)

    # Initialize states
    m = torch.tensor(float("-inf"), dtype=q_row.dtype, device=q_row.device)
    d = torch.tensor(0.0, dtype=q_row.dtype, device=q_row.device)
    o = torch.zeros(head_dim, dtype=q_row.dtype, device=q_row.device)

    print(f"Initial: m=-inf, d'=0, o'=zeros({head_dim})")
    print()

    for i in range(seq_len):
        x_i = torch.dot(q_row, K[i]) * scale

        m_old = m.clone()
        d_old = d.clone()

        m = torch.maximum(m, x_i)

        rescale = torch.exp(m_old - m)
        new_exp = torch.exp(x_i - m)

        d = d_old * rescale + new_exp

        # The key rescaling step for output
        o = o * (d_old / d) * rescale + (new_exp / d) * V[i]

        if i < 3 or i >= seq_len - 2:
            print(
                f"i={i}: x_i={x_i.item():.4f}, m={m.item():.4f}, "
                f"d'={d.item():.4f}, |o'|={torch.norm(o).item():.4f}"
            )
        elif i == 3:
            print("... (skipping middle iterations)")

    print(f"\nFinal output norm: {torch.norm(o).item():.4f}")
    return o


def verify_single_row_flash_attention():
    """
    Verify that single-row FlashAttention produces correct results.
    """
    print("=" * 60)
    print("Verifying Single-Row FlashAttention Correctness")
    print("=" * 60)

    torch.manual_seed(42)

    # Test 1: Small case
    print("\nTest 1: Small sequence (seq_len=8, head_dim=4)")
    seq_len = 8
    head_dim = 4

    q_row = torch.randn(head_dim)
    K = torch.randn(seq_len, head_dim)
    V = torch.randn(seq_len, head_dim)

    # Compute using FlashAttention
    flash_output = flash_attention_single_row(q_row, K, V)

    # Compute using standard attention (ground truth)
    # Standard attention: o = softmax(q @ K.T) @ V
    scale = head_dim**-0.5
    scores = (q_row @ K.T) * scale  # (seq_len,)
    attention_weights = torch.softmax(scores, dim=-1)  # (seq_len,)
    standard_output = attention_weights @ V  # (head_dim,)

    diff = torch.abs(flash_output - standard_output).max()
    print(f"Flash output:    {flash_output}")
    print(f"Standard output: {standard_output}")
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-5, f"Single-row FlashAttention mismatch! Diff: {diff}"
    print("PASS: Small sequence test passed!")

    # Test 2: Larger sequence
    print("\n" + "-" * 60)
    print("Test 2: Larger sequence (seq_len=256, head_dim=64)")
    seq_len = 256
    head_dim = 64

    q_row = torch.randn(head_dim)
    K = torch.randn(seq_len, head_dim)
    V = torch.randn(seq_len, head_dim)

    flash_output = flash_attention_single_row(q_row, K, V)

    scale = head_dim**-0.5
    scores = (q_row @ K.T) * scale
    attention_weights = torch.softmax(scores, dim=-1)
    standard_output = attention_weights @ V

    diff = torch.abs(flash_output - standard_output).max()
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-4, f"Large sequence mismatch! Diff: {diff}"
    print("PASS: Large sequence test passed!")

    # Test 3: Numerical stability
    print("\n" + "-" * 60)
    print("Test 3: Numerical stability (extreme values)")
    seq_len = 32
    head_dim = 16

    # Create Q and K that produce very large attention scores
    q_row = torch.randn(head_dim) * 10
    K = torch.randn(seq_len, head_dim) * 10
    V = torch.randn(seq_len, head_dim)

    flash_output = flash_attention_single_row(q_row, K, V)

    scale = head_dim**-0.5
    scores = (q_row @ K.T) * scale
    attention_weights = torch.softmax(scores, dim=-1)
    standard_output = attention_weights @ V

    diff = torch.abs(flash_output - standard_output).max()
    print(f"Max difference: {diff:.2e}")
    print(f"Flash output has NaN: {torch.isnan(flash_output).any()}")
    print(f"Flash output has Inf: {torch.isinf(flash_output).any()}")
    assert diff < 1e-4, f"Numerical stability test failed! Diff: {diff}"
    assert not torch.isnan(flash_output).any(), "NaN in FlashAttention output!"
    assert not torch.isinf(flash_output).any(), "Inf in FlashAttention output!"
    print("PASS: Numerical stability test passed!")

    print("\n" + "=" * 60)
    print("All Single-Row FlashAttention Tests PASSED!")
    print("=" * 60)


def demo_single_row():
    """
    Demonstrate single-row FlashAttention with verbose output.
    """
    print("=" * 60)
    print("Single-Row FlashAttention Demonstration")
    print("=" * 60)

    torch.manual_seed(123)

    seq_len = 6
    head_dim = 4

    q_row = torch.randn(head_dim)
    K = torch.randn(seq_len, head_dim)
    V = torch.randn(seq_len, head_dim)

    print(f"\nQuery: {q_row}")
    print()

    flash_output = flash_attention_single_row_verbose(q_row, K, V)

    # Compare with standard
    scale = head_dim**-0.5
    scores = (q_row @ K.T) * scale
    attention_weights = torch.softmax(scores, dim=-1)
    standard_output = attention_weights @ V

    print(f"\nFlash output:    {flash_output}")
    print(f"Standard output: {standard_output}")
    print(f"Difference: {torch.abs(flash_output - standard_output).max():.2e}")

    print(f"\nAttention weights (for reference): {attention_weights}")
    print(f"Sum of weights: {attention_weights.sum():.6f}")


if __name__ == "__main__":
    verify_single_row_flash_attention()
    print("\n")
    demo_single_row()
