# Immediate Coding Milestones

## M0 (today)
- [x] Initialize repository structure
- [x] Draft feasibility, technical plan, blockers
- [x] Create config skeletons for 3x A100 modes
- [x] Create distributed training entrypoint scaffold

## M1 (next 24h)
- [ ] Implement executable tiny GPT forward/backward with DDP/FSDP toggle
- [ ] Add activation checkpointing + bf16 autocast
- [ ] Add synthetic data loader for deterministic perf tests
- [ ] Add checkpoint save/load and resume tests

## M2
- [ ] Add TP primitives and sequence parallel path
- [ ] Add pipeline parallel stage partitioning
- [ ] Integrate flash attention path + fallback kernels

## M3
- [ ] Add MoE block + expert parallel routing prototype
- [ ] Benchmark mode matrix on 3x A100
- [ ] Publish reproducible perf report
