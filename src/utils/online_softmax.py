"""
Online Softmax Implementation (Step 2)
=======================================
Implements the 2-pass online softmax algorithm from Section 3 of the FlashAttention paper.

The key insight is to track:
- m_i: running maximum (for numerical stability)
- d_i': running denominator (rescaled as max changes)

This reduces memory access from 3 passes to 2 passes.
"""

import sys
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

import torch


def safe_softmax_3pass(x: torch.Tensor) -> torch.Tensor:
    """
    3-pass safe softmax (Algorithm from Section 2).

    Pass 1: Find maximum
    Pass 2: Compute denominator
    Pass 3: Compute final values

    Args:
        x: Input tensor of shape (..., N) - softmax along last dimension

    Returns:
        Softmax output of same shape
    """
    # Pass 1: Find maximum for numerical stability
    m = x.max(dim=-1, keepdim=True).values

    # Pass 2: Compute denominator
    d = torch.sum(torch.exp(x - m), dim=-1, keepdim=True)

    # Pass 3: Compute softmax values
    a = torch.exp(x - m) / d

    return a


def online_softmax_2pass(x: torch.Tensor) -> torch.Tensor:
    """
    2-pass online softmax (Algorithm from Section 3).

    Pass 1: Compute m_i and d_i' together using recurrence
    Pass 2: Compute final softmax values

    Uses the recurrence relation (equation 10):
        d_i' = d_{i-1}' * exp(m_{i-1} - m_i) + exp(x_i - m_i)

    Args:
        x: Input tensor of shape (..., N) - softmax along last dimension

    Returns:
        Softmax output of same shape
    """
    # Get the sequence length (last dimension)
    N = x.shape[-1]

    # Initialize states
    # m_0 = -inf, d_0' = 0
    m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype, device=x.device)
    d = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    # Pass 1: Compute m_N and d_N' using online algorithm
    for i in range(N):
        x_i = x[..., i]  # Current element

        # Update maximum: m_i = max(m_{i-1}, x_i)
        m_old = m.clone()
        m = torch.maximum(m, x_i)

        # Update denominator with rescaling: equation (10)
        # d_i' = d_{i-1}' * exp(m_{i-1} - m_i) + exp(x_i - m_i)
        d = d * torch.exp(m_old - m) + torch.exp(x_i - m)

    # Pass 2: Compute final softmax values
    # a_i = exp(x_i - m_N) / d_N'
    a = torch.exp(x - m.unsqueeze(-1)) / d.unsqueeze(-1)

    return a


def online_softmax_2pass_verbose(x: torch.Tensor) -> torch.Tensor:
    """
    Same as online_softmax_2pass but with detailed print statements for debugging.
    """
    N = x.shape[-1]

    print(f"Online Softmax 2-Pass Algorithm")
    print(f"Input shape: {x.shape}, N={N}")
    print("-" * 50)

    # Initialize states
    m = torch.full(x.shape[:-1], float("-inf"), dtype=x.dtype, device=x.device)
    d = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    print(f"Initial state: m_0 = -inf, d_0' = 0")
    print()

    # Pass 1: Online computation of m and d'
    print("Pass 1: Computing m_N and d_N'")
    for i in range(N):
        x_i = x[..., i]
        m_old = m.clone()

        # Update maximum
        m = torch.maximum(m, x_i)

        # Update denominator with rescaling
        rescale_factor = torch.exp(m_old - m)
        new_contribution = torch.exp(x_i - m)
        d = d * rescale_factor + new_contribution

        if i < 5 or i >= N - 2:  # Print first few and last few iterations
            print(
                f"  i={i}: x_i={x_i.item():.4f}, m={m.item():.4f}, "
                f"rescale={rescale_factor.item():.4f}, d'={d.item():.4f}"
            )
        elif i == 5:
            print(f"  ... (skipping middle iterations)")

    print(f"\nFinal: m_N = {m.item():.4f}, d_N' = {d.item():.4f}")
    print()

    # Pass 2: Compute softmax values
    print("Pass 2: Computing softmax values")
    a = torch.exp(x - m.unsqueeze(-1)) / d.unsqueeze(-1)
    print(f"  a = exp(x - m_N) / d_N'")
    print(f"  Sum of softmax values: {a.sum().item():.6f} (should be 1.0)")

    return a


def verify_online_softmax():
    """
    Verify that online softmax produces identical results to PyTorch's softmax.
    """
    print("=" * 60)
    print("Verifying Online Softmax Correctness")
    print("=" * 60)

    # Test 1: Simple 1D case
    print("\nTest 1: Simple 1D tensor")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    pytorch_result = torch.softmax(x, dim=-1)
    safe_3pass_result = safe_softmax_3pass(x)
    online_2pass_result = online_softmax_2pass(x)

    print(f"Input: {x}")
    print(f"PyTorch softmax:     {pytorch_result}")
    print(f"Safe 3-pass softmax: {safe_3pass_result}")
    print(f"Online 2-pass:       {online_2pass_result}")

    # Check numerical equivalence
    diff_3pass = torch.abs(pytorch_result - safe_3pass_result).max()
    diff_2pass = torch.abs(pytorch_result - online_2pass_result).max()
    print(f"\nMax diff (3-pass vs PyTorch): {diff_3pass:.2e}")
    print(f"Max diff (2-pass vs PyTorch): {diff_2pass:.2e}")

    assert diff_3pass < 1e-6, "3-pass softmax mismatch!"
    assert diff_2pass < 1e-6, "2-pass softmax mismatch!"
    print("PASS: All results match PyTorch softmax!")

    # Test 2: Larger random tensor
    print("\n" + "-" * 60)
    print("Test 2: Random tensor (100 elements)")
    x = torch.randn(100)

    pytorch_result = torch.softmax(x, dim=-1)
    online_result = online_softmax_2pass(x)

    diff = torch.abs(pytorch_result - online_result).max()
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-5, "Online softmax mismatch on random data!"
    print("PASS: Random tensor test passed!")

    # Test 3: Numerical stability with large values
    print("\n" + "-" * 60)
    print("Test 3: Numerical stability (large values)")
    x = torch.tensor([1000.0, 1001.0, 1002.0])  # Would overflow without safe softmax

    pytorch_result = torch.softmax(x, dim=-1)
    online_result = online_softmax_2pass(x)

    print(f"Input (large values): {x}")
    print(f"PyTorch softmax: {pytorch_result}")
    print(f"Online softmax:  {online_result}")

    diff = torch.abs(pytorch_result - online_result).max()
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-5, "Online softmax failed on large values!"
    assert not torch.isnan(online_result).any(), "NaN in result!"
    assert not torch.isinf(online_result).any(), "Inf in result!"
    print("PASS: Numerical stability test passed!")

    # Test 4: Batch processing
    print("\n" + "-" * 60)
    print("Test 4: Batch processing (10 x 50)")
    x = torch.randn(10, 50)

    pytorch_result = torch.softmax(x, dim=-1)
    online_result = online_softmax_2pass(x)

    diff = torch.abs(pytorch_result - online_result).max()
    print(f"Max difference: {diff:.2e}")
    assert diff < 1e-5, "Online softmax mismatch on batch!"
    print("PASS: Batch processing test passed!")

    print("\n" + "=" * 60)
    print("All Online Softmax Tests PASSED!")
    print("=" * 60)


def demo_online_algorithm():
    """
    Step-by-step demonstration of the online softmax algorithm.
    """
    print("=" * 60)
    print("Step-by-Step Online Softmax Demonstration")
    print("=" * 60)

    x = torch.tensor([2.0, 1.0, 4.0, 3.0])
    print(f"\nInput: x = {x.tolist()}")
    print(f"Expected softmax: {torch.softmax(x, dim=-1).tolist()}")
    print()

    result = online_softmax_2pass_verbose(x.unsqueeze(0))
    print(f"\nResult: {result.squeeze().tolist()}")


if __name__ == "__main__":
    # Run verification tests
    verify_online_softmax()

    print("\n")

    # Run demonstration
    demo_online_algorithm()
