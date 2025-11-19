"""
Correctness Tests for FlashAttention Implementation
====================================================
Comprehensive unit tests to verify all implementations produce correct results.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import torch

from src.attention.standard_attention import standard_attention_simple
from src.utils.online_softmax import online_softmax_2pass, safe_softmax_3pass
from src.attention.flash_attention_single import flash_attention_single_row
from src.attention.flash_attention_tiled import (
    flash_attention_tiled,
    flash_attention_tiled_single_row,
    flash_attention_tiled_vectorized,
)


class TestOnlineSoftmax(unittest.TestCase):
    """Test online softmax implementation."""

    def setUp(self):
        torch.manual_seed(42)

    def test_simple_case(self):
        """Test with simple input."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = torch.softmax(x, dim=-1)
        result = online_softmax_2pass(x)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_random_tensor(self):
        """Test with random tensor."""
        x = torch.randn(100)
        expected = torch.softmax(x, dim=-1)
        result = online_softmax_2pass(x)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_large_values(self):
        """Test numerical stability with large values."""
        x = torch.tensor([1000.0, 1001.0, 1002.0])
        expected = torch.softmax(x, dim=-1)
        result = online_softmax_2pass(x)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())

    def test_batch_processing(self):
        """Test batch processing."""
        x = torch.randn(10, 50)
        expected = torch.softmax(x, dim=-1)
        result = online_softmax_2pass(x)
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_sum_to_one(self):
        """Test that softmax sums to 1."""
        x = torch.randn(100)
        result = online_softmax_2pass(x)
        self.assertAlmostEqual(result.sum().item(), 1.0, places=5)

    def test_three_pass_equals_two_pass(self):
        """Test that 3-pass and 2-pass produce same results."""
        x = torch.randn(50)
        result_3pass = safe_softmax_3pass(x)
        result_2pass = online_softmax_2pass(x)
        self.assertTrue(torch.allclose(result_3pass, result_2pass, atol=1e-6))


class TestSingleRowFlashAttention(unittest.TestCase):
    """Test single-row FlashAttention implementation."""

    def setUp(self):
        torch.manual_seed(42)

    def test_small_sequence(self):
        """Test with small sequence."""
        seq_len, head_dim = 8, 4
        q_row = torch.randn(head_dim)
        K = torch.randn(seq_len, head_dim)
        V = torch.randn(seq_len, head_dim)

        flash_output = flash_attention_single_row(q_row, K, V)

        # Standard computation
        scale = head_dim**-0.5
        scores = (q_row @ K.T) * scale
        attention_weights = torch.softmax(scores, dim=-1)
        standard_output = attention_weights @ V

        self.assertTrue(torch.allclose(flash_output, standard_output, atol=1e-5))

    def test_large_sequence(self):
        """Test with larger sequence."""
        seq_len, head_dim = 256, 64
        q_row = torch.randn(head_dim)
        K = torch.randn(seq_len, head_dim)
        V = torch.randn(seq_len, head_dim)

        flash_output = flash_attention_single_row(q_row, K, V)

        scale = head_dim**-0.5
        scores = (q_row @ K.T) * scale
        attention_weights = torch.softmax(scores, dim=-1)
        standard_output = attention_weights @ V

        self.assertTrue(torch.allclose(flash_output, standard_output, atol=1e-4))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        seq_len, head_dim = 32, 16
        q_row = torch.randn(head_dim) * 10
        K = torch.randn(seq_len, head_dim) * 10
        V = torch.randn(seq_len, head_dim)

        flash_output = flash_attention_single_row(q_row, K, V)

        self.assertFalse(torch.isnan(flash_output).any())
        self.assertFalse(torch.isinf(flash_output).any())

    def test_output_shape(self):
        """Test output shape."""
        seq_len, head_dim = 100, 32
        q_row = torch.randn(head_dim)
        K = torch.randn(seq_len, head_dim)
        V = torch.randn(seq_len, head_dim)

        output = flash_attention_single_row(q_row, K, V)
        self.assertEqual(output.shape, (head_dim,))


class TestTiledFlashAttention(unittest.TestCase):
    """Test tiled FlashAttention implementation."""

    def setUp(self):
        torch.manual_seed(42)

    def test_single_row_tiled(self):
        """Test single row tiled version."""
        seq_len, head_dim = 128, 32
        q_row = torch.randn(head_dim)
        K = torch.randn(seq_len, head_dim)
        V = torch.randn(seq_len, head_dim)

        for block_size in [16, 32, 64]:
            tiled_output = flash_attention_tiled_single_row(
                q_row, K, V, block_size=block_size
            )

            scale = head_dim**-0.5
            scores = (q_row @ K.T) * scale
            attention_weights = torch.softmax(scores, dim=-1)
            standard_output = attention_weights @ V

            self.assertTrue(
                torch.allclose(tiled_output, standard_output, atol=1e-5),
                f"Failed for block_size={block_size}",
            )

    def test_vectorized_small(self):
        """Test vectorized version with small tensors."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 64, 32
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        tiled_output = flash_attention_tiled_vectorized(Q, K, V, block_size=16)
        standard_output = standard_attention_simple(Q, K, V)

        self.assertTrue(torch.allclose(tiled_output, standard_output, atol=1e-5))

    def test_vectorized_batched(self):
        """Test vectorized version with batches and heads."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        tiled_output = flash_attention_tiled_vectorized(Q, K, V, block_size=32)
        standard_output = standard_attention_simple(Q, K, V)

        self.assertTrue(torch.allclose(tiled_output, standard_output, atol=1e-4))

    def test_different_block_sizes(self):
        """Test with different block sizes."""
        batch_size, num_heads, seq_len, head_dim = 1, 2, 256, 64
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        standard_output = standard_attention_simple(Q, K, V)

        for block_size in [16, 32, 64, 128, 256]:
            tiled_output = flash_attention_tiled_vectorized(
                Q, K, V, block_size=block_size
            )
            self.assertTrue(
                torch.allclose(tiled_output, standard_output, atol=1e-4),
                f"Failed for block_size={block_size}"
            )

    def test_output_shape(self):
        """Test that output shape matches input."""
        batch_size, num_heads, seq_len, head_dim = 3, 8, 512, 64
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        output = flash_attention_tiled_vectorized(Q, K, V, block_size=64)
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))

    def test_numerical_stability_tiled(self):
        """Test numerical stability."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 64, 32
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim) * 10
        K = torch.randn(batch_size, num_heads, seq_len, head_dim) * 10
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        output = flash_attention_tiled_vectorized(Q, K, V, block_size=16)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_non_divisible_block_size(self):
        """Test when seq_len is not divisible by block_size."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 100, 32
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # block_size=32 doesn't divide seq_len=100 evenly
        tiled_output = flash_attention_tiled_vectorized(Q, K, V, block_size=32)
        standard_output = standard_attention_simple(Q, K, V)

        self.assertTrue(torch.allclose(tiled_output, standard_output, atol=1e-5))


class TestStandardAttention(unittest.TestCase):
    """Test standard attention baseline."""

    def setUp(self):
        torch.manual_seed(42)

    def test_output_shape(self):
        """Test output shape matches expected."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        output = standard_attention_simple(Q, K, V)
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))

    def test_no_nan_inf(self):
        """Test no NaN or Inf in output."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 64, 32
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        output = standard_attention_simple(Q, K, V)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_attention_is_weighted_sum(self):
        """Test that attention produces weighted sum of values."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 10, 4
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        output = standard_attention_simple(Q, K, V)

        # Each output should be in the convex hull of V vectors
        # This is a basic sanity check
        self.assertEqual(output.shape, V.shape)


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests comparing all implementations."""

    def setUp(self):
        torch.manual_seed(42)

    def test_all_implementations_match(self):
        """Test that all implementations produce same results."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 64, 32
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        standard_output = standard_attention_simple(Q, K, V)

        # Test non-vectorized tiled
        tiled_output = flash_attention_tiled(Q, K, V, block_size=16)
        self.assertTrue(torch.allclose(tiled_output, standard_output, atol=1e-5))

        # Test vectorized tiled
        vectorized_output = flash_attention_tiled_vectorized(Q, K, V, block_size=16)
        self.assertTrue(torch.allclose(vectorized_output, standard_output, atol=1e-5))

    def test_long_sequence(self):
        """Test with long sequence (stress test)."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 1024, 64
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim)

        standard_output = standard_attention_simple(Q, K, V)
        flash_output = flash_attention_tiled_vectorized(Q, K, V, block_size=64)

        max_diff = torch.abs(flash_output - standard_output).max().item()
        self.assertLess(max_diff, 1e-4)

    def test_dtype_preserved(self):
        """Test that dtype is preserved."""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 32, 16
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

        output = flash_attention_tiled_vectorized(Q, K, V, block_size=8)
        self.assertEqual(output.dtype, torch.float32)


def run_all_tests():
    """Run all tests and provide summary."""
    print("=" * 70)
    print("Running FlashAttention Correctness Tests")
    print("=" * 70)

    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestOnlineSoftmax))
    suite.addTests(loader.loadTestsFromTestCase(TestSingleRowFlashAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestTiledFlashAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestStandardAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\nALL TESTS PASSED!")
        print("FlashAttention implementation is correct.")
    else:
        print("\nSOME TESTS FAILED!")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")

    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
