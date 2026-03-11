# Technical Plan

## 1) System Architecture
- Framework: PyTorch 2.x
- Distributed backend: torch.distributed (NCCL), optional DeepSpeed integration
- Parallelism mix for 3 GPUs:
  - Data parallel: degree 1-3 depending on TP/PP choice
  - Tensor parallel: degree 1 or 3 (3 preferred for larger layers)
  - Pipeline parallel: degree 1-3 (2-stage practical, 3-stage for memory pressure)
  - Sequence/context parallel: enabled when TP > 1
  - Expert parallel (MoE): optional, EP degree aligned with TP world

## 2) Model Design (GPT-OSS-style)
- Decoder-only transformer
- RoPE positional encoding
- RMSNorm
- SwiGLU FFN
- GQA/MQA attention for KV efficiency
- FlashAttention2 path when available
- bf16 mixed precision default

## 3) Training Stack
- FSDP/ZeRO-style state sharding
- Activation checkpointing
- Gradient accumulation for effective batch scaling
- Fused optimizers where available
- Compiled/fused kernels guard-railed by capability detection

## 4) Data Pipeline
- Streaming dataset support
- Sharded tokenized binary format
- Deterministic resume from checkpoints
- Dynamic packing for high token utilization

## 5) Eval + Observability
- Throughput: tokens/s/GPU
- Memory: peak allocated/reserved
- Stability: grad norm, loss spikes, NaN/Inf detectors
- Quality: perplexity on held-out sets

## 6) 3x A100 Target Modes
- Mode A (dense debug): TP=1, PP=1, DP=3
- Mode B (memory optimized): TP=3, PP=1, DP=1 + seq parallel
- Mode C (pipeline): TP=1, PP=3, DP=1
- Mode D (MoE hybrid): TP=3, EP=3, PP=1, DP=1

## 7) Delivery Strategy
- Phase 1: infra + model skeleton + unit tests
- Phase 2: distributed correctness + checkpointing
- Phase 3: performance tuning + profiling
- Phase 4: scaled runs + docs + reproducibility pack
