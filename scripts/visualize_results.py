"""
Visualization for FlashAttention Benchmarks
============================================
Creates publication-quality plots for the technical report.
"""

import json
import os

# Try to import matplotlib, install if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Installing matplotlib...")
    os.system("pip install matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11


def load_results(filename='benchmark_results.json'):
    """Load benchmark results from JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: {filename} not found. Using sample data.")
        return generate_sample_data()


def generate_sample_data():
    """Generate sample data for testing visualization."""
    return {
        'sequence_length': [
            {'seq_len': 512, 'std_time_ms': 1.5, 'flash_time_ms': 3.0,
             'std_memory_mb': 20, 'flash_memory_mb': 6, 'memory_savings_pct': 70, 'max_diff': 2e-7},
            {'seq_len': 1024, 'std_time_ms': 3.8, 'flash_time_ms': 6.3,
             'std_memory_mb': 72, 'flash_memory_mb': 12, 'memory_savings_pct': 83, 'max_diff': 1.7e-7},
            {'seq_len': 2048, 'std_time_ms': 14.3, 'flash_time_ms': 14.5,
             'std_memory_mb': 272, 'flash_memory_mb': 24, 'memory_savings_pct': 91, 'max_diff': 1.8e-7},
            {'seq_len': 4096, 'std_time_ms': 51.3, 'flash_time_ms': 69.6,
             'std_memory_mb': 1056, 'flash_memory_mb': 48, 'memory_savings_pct': 95, 'max_diff': 1e-7},
            {'seq_len': 8192, 'std_time_ms': 205, 'flash_time_ms': 280,
             'std_memory_mb': 4160, 'flash_memory_mb': 96, 'memory_savings_pct': 97.7, 'max_diff': 1.5e-7},
        ],
        'model_dimension': [
            {'head_dim': 64, 'hidden_size': 512, 'std_time_ms': 14, 'flash_time_ms': 15,
             'std_memory_mb': 272, 'flash_memory_mb': 24},
            {'head_dim': 96, 'hidden_size': 768, 'std_time_ms': 16, 'flash_time_ms': 18,
             'std_memory_mb': 288, 'flash_memory_mb': 32},
            {'head_dim': 128, 'hidden_size': 1024, 'std_time_ms': 20, 'flash_time_ms': 22,
             'std_memory_mb': 304, 'flash_memory_mb': 40},
        ],
        'block_size': [
            {'block_size': 32, 'time_ms': 100, 'peak_memory_mb': 50, 'max_diff': 2e-7},
            {'block_size': 64, 'time_ms': 70, 'peak_memory_mb': 52, 'max_diff': 1.5e-7},
            {'block_size': 128, 'time_ms': 60, 'peak_memory_mb': 56, 'max_diff': 1.5e-7},
            {'block_size': 256, 'time_ms': 65, 'peak_memory_mb': 64, 'max_diff': 2e-7},
        ]
    }


def plot_memory_scaling(results, save_path='plots/memory_scaling.png'):
    """Plot memory usage vs sequence length for standard vs FlashAttention."""
    seq_data = results.get('sequence_length', [])
    if not seq_data:
        print("No sequence length data available")
        return

    seq_lens = [r['seq_len'] for r in seq_data]
    std_mem = [r['std_memory_mb'] for r in seq_data]
    flash_mem = [r['flash_memory_mb'] for r in seq_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both lines
    ax.plot(seq_lens, std_mem, 'o-', label='Standard Attention', linewidth=2, markersize=10, color='#FF6B6B')
    ax.plot(seq_lens, flash_mem, 's-', label='FlashAttention', linewidth=2, markersize=10, color='#4ECDC4')

    # Add memory savings annotations
    for i, (sl, sm, fm) in enumerate(zip(seq_lens, std_mem, flash_mem)):
        savings = (1 - fm/sm) * 100
        ax.annotate(f'{savings:.0f}% saved',
                   xy=(sl, fm), xytext=(sl, fm * 1.5),
                   ha='center', fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Peak Memory Usage (MB)')
    ax.set_title('Memory Usage: Standard Attention vs FlashAttention')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add text annotation about O(L²) vs O(L)
    ax.text(0.95, 0.05, 'Standard: O(L²)\nFlashAttention: O(L)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_time_comparison(results, save_path='plots/time_comparison.png'):
    """Plot execution time comparison."""
    seq_data = results.get('sequence_length', [])
    if not seq_data:
        print("No sequence length data available")
        return

    seq_lens = [r['seq_len'] for r in seq_data]
    std_time = [r['std_time_ms'] for r in seq_data]
    flash_time = [r['flash_time_ms'] for r in seq_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(seq_lens))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], std_time, width, label='Standard Attention', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], flash_time, width, label='FlashAttention', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Execution Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_memory_savings_percentage(results, save_path='plots/memory_savings.png'):
    """Plot memory savings percentage vs sequence length."""
    seq_data = results.get('sequence_length', [])
    if not seq_data:
        print("No sequence length data available")
        return

    seq_lens = [r['seq_len'] for r in seq_data]
    savings = [r['memory_savings_pct'] for r in seq_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(seq_lens, savings, 'o-', linewidth=2, markersize=12, color='#45B7D1')
    ax.fill_between(seq_lens, savings, alpha=0.3, color='#45B7D1')

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Memory Savings (%)')
    ax.set_title('FlashAttention Memory Savings vs Sequence Length')
    ax.set_xscale('log', base=2)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)

    # Add percentage labels
    for sl, sv in zip(seq_lens, savings):
        ax.annotate(f'{sv:.1f}%', xy=(sl, sv), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

    # Add annotation about increasing savings
    ax.text(0.5, 0.2, 'Memory savings increase with sequence length\nbecause standard attention scales O(L²)',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_numerical_accuracy(results, save_path='plots/numerical_accuracy.png'):
    """Plot numerical accuracy (max difference from standard attention)."""
    seq_data = results.get('sequence_length', [])
    if not seq_data:
        print("No sequence length data available")
        return

    seq_lens = [r['seq_len'] for r in seq_data]
    max_diffs = [r['max_diff'] for r in seq_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(seq_lens, max_diffs, 'o-', linewidth=2, markersize=10, color='#96CEB4')

    # Add threshold line
    ax.axhline(y=1e-4, color='red', linestyle='--', linewidth=2, label='Acceptable threshold (1e-4)')

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Maximum Absolute Difference')
    ax.set_title('Numerical Accuracy: FlashAttention vs Standard Attention')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0.5, 0.9, 'All results below threshold = numerically equivalent',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_block_size_effect(results, save_path='plots/block_size_effect.png'):
    """Plot effect of block size on performance."""
    block_data = results.get('block_size', [])
    if not block_data:
        print("No block size data available")
        return

    block_sizes = [r['block_size'] for r in block_data]
    times = [r['time_ms'] for r in block_data]
    memory = [r['peak_memory_mb'] for r in block_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time plot
    ax1.plot(block_sizes, times, 'o-', linewidth=2, markersize=10, color='#FF6B6B')
    ax1.set_xlabel('Block Size')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Block Size vs Execution Time')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)

    # Memory plot
    ax2.plot(block_sizes, memory, 's-', linewidth=2, markersize=10, color='#4ECDC4')
    ax2.set_xlabel('Block Size')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('Block Size vs Peak Memory')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Effect of Block Size on FlashAttention Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_algorithm_comparison(save_path='plots/algorithm_comparison.png'):
    """Create a visual comparison of algorithm complexity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Theoretical curves
    L = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    std_mem = [l * l for l in L]  # O(L²)
    flash_mem = [l * 128 for l in L]  # O(L * block_size)

    # Normalize to show relative scaling
    std_mem_norm = [m / std_mem[0] for m in std_mem]
    flash_mem_norm = [m / flash_mem[0] for m in flash_mem]

    ax.plot(L, std_mem_norm, 'o-', label='Standard Attention O(L²)', linewidth=2, markersize=8, color='#FF6B6B')
    ax.plot(L, flash_mem_norm, 's-', label='FlashAttention O(L)', linewidth=2, markersize=8, color='#4ECDC4')

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Relative Memory Usage (normalized)')
    ax.set_title('Theoretical Memory Complexity Comparison')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0.05, 0.95, 'Quadratic vs Linear Scaling',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=14, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_triton_comparison(results, save_path='plots/triton_comparison.png'):
    """Plot execution time comparison: Standard vs PyTorch FA vs Triton FA."""
    triton_data = results.get('triton_benchmark', [])
    if not triton_data:
        print("No Triton benchmark data available")
        return

    seq_lens = [r['seq_len'] for r in triton_data]
    std_time = [r['std_time_ms'] for r in triton_data]
    pytorch_time = [r['pytorch_fa_time_ms'] for r in triton_data]
    triton_time = [r['triton_fa_time_ms'] for r in triton_data]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = range(len(seq_lens))
    width = 0.25

    bars1 = ax.bar([i - width for i in x], std_time, width, label='Standard Attention', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar([i for i in x], pytorch_time, width, label='PyTorch FlashAttention', color='#FFE66D', alpha=0.8)
    bars3 = ax.bar([i + width for i in x], triton_time, width, label='Triton FlashAttention', color='#4ECDC4', alpha=0.8)

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Execution Time (ms)')
    ax.set_title('Execution Time: Standard vs PyTorch FlashAttention vs Triton Kernel')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Add speedup annotations for Triton
    for i, (st, tt) in enumerate(zip(std_time, triton_time)):
        speedup = st / tt
        ax.annotate(f'{speedup:.1f}x faster',
                   xy=(i + width, tt), xytext=(i + width, tt * 0.5),
                   ha='center', fontsize=9, fontweight='bold', color='#2d6a4f',
                   arrowprops=dict(arrowstyle='->', color='#2d6a4f', alpha=0.7))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_triton_speedup(results, save_path='plots/triton_speedup.png'):
    """Plot Triton speedup factor vs sequence length."""
    triton_data = results.get('triton_benchmark', [])
    if not triton_data:
        print("No Triton benchmark data available")
        return

    seq_lens = [r['seq_len'] for r in triton_data]
    speedups = [r['triton_speedup'] for r in triton_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(seq_lens, speedups, 'o-', linewidth=3, markersize=12, color='#2d6a4f')
    ax.fill_between(seq_lens, speedups, alpha=0.3, color='#2d6a4f')

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Speedup (x times faster than Standard)')
    ax.set_title('Triton FlashAttention Speedup vs Standard Attention')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')

    # Add labels
    for sl, sp in zip(seq_lens, speedups):
        ax.annotate(f'{sp:.1f}x', xy=(sl, sp), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=12, fontweight='bold')

    ax.legend()
    ax.set_ylim([0, max(speedups) * 1.3])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_all_plots(results=None):
    """Generate all plots for the technical report."""
    print("=" * 60)
    print("Generating Visualization Plots")
    print("=" * 60)

    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    if results is None:
        results = load_results()

    print("\nCreating plots...")

    # Generate all plots
    plot_memory_scaling(results, 'plots/memory_scaling.png')
    plot_time_comparison(results, 'plots/time_comparison.png')
    plot_memory_savings_percentage(results, 'plots/memory_savings.png')
    plot_numerical_accuracy(results, 'plots/numerical_accuracy.png')
    plot_block_size_effect(results, 'plots/block_size_effect.png')
    plot_algorithm_comparison('plots/algorithm_comparison.png')
    plot_triton_comparison(results, 'plots/triton_comparison.png')
    plot_triton_speedup(results, 'plots/triton_speedup.png')

    print("\nAll plots generated successfully!")
    print("Plots saved in 'plots/' directory")
    print("\nGenerated plots:")
    for f in os.listdir('plots'):
        if f.endswith('.png'):
            print(f"  - plots/{f}")


if __name__ == "__main__":
    create_all_plots()
