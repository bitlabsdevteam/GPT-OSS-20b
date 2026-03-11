# 3x A100 Feasibility / Tradeoffs (Living Document)

## Defensible defaults now
- Start with DP-only or light TP to keep debugging tractable
- Prioritize deterministic correctness + checkpoint reliability before exotic kernels
- Keep model architecture modern, but gate optional fast paths behind capability checks

## Why this sequence
- 3 GPUs leave little room for runaway complexity
- TP+PP+SP+EP all-at-once is high risk and hard to debug
- Establishing a stable baseline allows profiling-guided complexity additions

## Current practical target
- Build a robust stack that can run:
  1) correctness tests and smoke training at small scale
  2) medium-scale experiments (1B-7B)
  3) 20B configuration dry runs / limited token runs

## Deferred until baseline stable
- Full MoE routing optimization
- Full FlashAttention/TransformerEngine integration
- Aggressive CUDA kernel specialization
