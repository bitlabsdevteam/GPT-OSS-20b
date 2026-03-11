# Feasibility Assessment for GPT-OSS-style 20B on 3x A100

## TL;DR
- **Full scratch pretraining of dense 20B to frontier quality on 3x A100 is infeasible** in practical time/cost.
- **Engineering the full codebase is feasible** and should be done now.
- **Validation path is feasible** via small-scale training, ablations, synthetic tests, and selective continued pretraining.

## Hardware Assumption
- 3x A100, likely NVLink on a single node
- Memory scenarios:
  - A100 80GB -> 240GB total HBM
  - A100 40GB -> 120GB total HBM (much tighter)

## Memory Reality Check (Dense 20B)
Approximate params memory:
- FP16 params: 20B * 2 bytes ~ 40 GB
- Gradients FP16: ~40 GB
- Adam states FP32 m,v: ~160 GB
- Total raw optimizer+model state (no sharding): ~240 GB before activations/buffers

With ZeRO/FSDP sharding across 3 GPUs, state fits better, but:
- Activations + temp buffers + fragmentation still heavy
- Sequence length and microbatch become severe bottlenecks
- Throughput remains low for true pretraining token budgets

## Compute Reality Check
20B-quality pretraining token budgets are usually hundreds of billions to trillions of tokens. On 3x A100, wall-clock to competitive quality is prohibitive.

## What *is* Feasible on 3x A100
1. Implement robust distributed stack with:
   - Tensor parallel (limited degree)
   - Sequence/context parallel
   - Data parallel/FSDP
   - Pipeline parallel (2-3 stages)
   - Optional MoE experts for capacity scaling
2. Train and harden smaller dense configs (1B/3B/7B)
3. Perform short-run smoke pretraining for 20B config
4. Do continued pretraining / instruction tuning on existing large checkpoints
5. Optimize inference kernels + serving stack for 20B inference/fine-tune workflows

## Recommendation
Treat this project as:
- **Primary**: production-grade training/inference framework
- **Secondary**: scaled experiments proving correctness and efficiency
- **Not primary**: full dense 20B frontier pretraining from zero
