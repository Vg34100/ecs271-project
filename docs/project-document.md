# Machine Learning Project Proposal

**Team:** Shuang Ma, Pablo Rodriguez, Pei Yu Lin
**Date:** November 3, 2025
**Course:** ECS271

---

## Problem

Accelerate the self-attention mechanism in Transformer models through memory-efficient algorithms.

## Motivation

Transformers serve as the foundational architecture for modern LLMs. The autoregressive nature of LLM inference necessitates repeated attention computations over growing sequences, where inefficient memory access patterns dominate runtime.

As sequence length increases, the quadratic complexity of attention not only escalates computational costs but also intensifies memory bandwidth pressure, as large intermediate matrices (e.g., QK^T) must be written to and read from GPU high-bandwidth memory (HBM). Optimizing attention is essential for enabling longer context windows, higher throughput, and more scalable LLM serving.

## Dataset

Since this project focuses on computational efficiency rather than task-specific performance, we will use synthetic benchmark datasets with:
- Variable sequence lengths: 1K to 32K tokens
- Variable model dimensions: hidden sizes 1024–8192

**Evaluation metrics:**
- Wall-clock time per attention forward pass
- GPU memory footprint (including intermediate activations)

## Methods

We will implement a memory-efficient attention algorithm based on FlashAttention, which leverages kernel fusion and tiling to minimize HBM traffic:

- **Kernel Fusion:** Combining matrix multiplication, softmax, and masking into a single GPU kernel to avoid writing intermediate results to HBM
- **Tiling and On-Chip Memory Management:** Decomposing large Q, K, V matrices into blocks that fit into SRAM/L1 cache, performing softmax rescaling online to maintain numerical stability

## Deliverables

A technical report comparing the performance and memory access patterns of standard attention and FlashAttention variants, with open-source code for our implementation.
