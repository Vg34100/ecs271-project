/fsx/ubuntu/miniconda3/envs/shuangma_env/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
================================================================================
FlashAttention GPU Benchmarking Suite
================================================================================
GPU Available: NVIDIA H200
GPU Memory: 150.12 GB

Running sequence length benchmark...
================================================================================
SEQUENCE LENGTH BENCHMARK
================================================================================

Configuration:
  Batch size: 1
  Number of heads: 8
  Head dimension: 64
  Block size: 128
  Device: cuda

 Seq Len |   Std Time |  Flash Time |    Std Mem |   Flash Mem |  Mem Save |   Max Diff
         |       (ms) |        (ms) |       (MB) |        (MB) |       (%) |           
-------------------------------------------------------------------------------------
     512 |       0.14 |        0.92 |      53.00 |       47.09 |     11.1% |   5.07e-07
    1024 |       0.20 |        1.63 |     106.00 |       62.19 |     41.3% |   5.22e-07
    2048 |       0.47 |        3.20 |     308.00 |       92.38 |     70.0% |   4.92e-07
    4096 |       1.65 |        6.19 |    1096.00 |      152.75 |     86.1% |   4.40e-07
    8192 |       6.85 |       17.92 |    4208.00 |      273.50 |     93.5% |   4.40e-07
-------------------------------------------------------------------------------------

Running model dimension benchmark...

================================================================================
MODEL DIMENSION BENCHMARK
================================================================================

Configuration:
  Batch size: 1
  Number of heads: 8
  Sequence length: 2048
  Block size: 64
  Device: cuda

  Head Dim |  Hidden Size |   Std Time |  Flash Time |    Std Mem |   Flash Mem |  Mem Save |   Max Diff
           |   (head*dim) |       (ms) |        (ms) |       (MB) |        (MB) |       (%) |           
---------------------------------------------------------------------------------------------------------
        64 |          512 |       0.46 |        6.07 |     308.00 |       76.38 |     75.2% |   5.66e-07
       128 |         1024 |       0.62 |        6.15 |     328.00 |      112.38 |     65.7% |   4.62e-07
       256 |         2048 |       0.95 |        6.14 |     368.00 |      184.38 |     49.9% |   4.02e-07
       512 |         4096 |       1.62 |        7.30 |     448.00 |      328.38 |     26.7% |   5.51e-07
      1024 |         8192 |       2.95 |       11.67 |     608.00 |      616.38 |     -1.4% |   5.96e-07
---------------------------------------------------------------------------------------------------------

Running block size benchmark...

================================================================================
BLOCK SIZE BENCHMARK
================================================================================

Configuration:
  Batch size: 1
  Number of heads: 8
  Sequence length: 4096
  Head dimension: 64
  Device: cuda

Standard Attention:
  Time: 1.65 ms
  Peak Memory: 1096.00 MB

  Block Size |    Time (ms) |  Peak Mem (MB) |     Max Diff
-------------------------------------------------------
          32 |        24.05 |         112.75 |     5.22e-07
          64 |        12.13 |         128.75 |     6.26e-07
         128 |         6.16 |         160.75 |     6.26e-07
         256 |         3.88 |         224.75 |     5.96e-07
         512 |         3.19 |         352.75 |     5.81e-07
        1024 |         2.69 |         608.75 |     6.26e-07
-------------------------------------------------------

Results saved to benchmark_results.json

================================================================================
BENCHMARK SUMMARY
================================================================================

Sequence Length Tests:
  - Tested up to 8192 tokens
  - Maximum memory savings: 93.5%
  - All results numerically correct (max diff < 1e-4)

Model Dimension Tests:
  - Tested hidden sizes up to 8192

Block Size Tests:
  - Optimal block size: 1024 (fastest)

================================================================================