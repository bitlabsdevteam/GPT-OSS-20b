# GPT-OSS-20B

Open, from-scratch training/inference codebase for a GPT-OSS-style ~20B decoder-only transformer, designed explicitly for **3x A100** constraints.

## Mission Constraints
- Hardware: exactly **3x NVIDIA A100** (assume 80GB unless stated otherwise)
- Training design must be explicit about feasibility and compromises
- Parallelism targets: tensor, context/sequence, data, pipeline, expert (where feasible)
- Practicality > hype: stable, measurable, reproducible

## Current Status
- ✅ Initial feasibility assessment drafted (`docs/FEASIBILITY_3xA100.md`)
- ✅ Technical plan drafted (`docs/TECHNICAL_PLAN.md`)
- ✅ Milestones and blockers tracked (`docs/MILESTONES.md`, `docs/BLOCKERS_ASSUMPTIONS.md`)
- ✅ Bootstrap training stack scaffolded (`src/gpt_oss_20b/`)
- 🔄 Next: implement executable minimal training loop and distributed launch configs

## Repo Layout
- `docs/` — feasibility, plan, milestones, blockers
- `configs/` — model/training/system configs
- `src/gpt_oss_20b/` — source code
- `scripts/` — launcher and utility scripts

## Quick Start (WIP)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# single-node, 3 GPUs (WIP)
torchrun --nproc_per_node=3 -m gpt_oss_20b.train --config configs/train_3xa100.yaml
```

## Feasibility Summary (Short)
A dense 20B model is **not realistically trainable from scratch to production quality on only 3x A100** within sane time/budget. What is feasible:
1. Build full production-grade codebase and training system
2. Run correctness/scale tests + short curriculum runs
3. Train smaller dense baselines (e.g., 1B–7B) to validate stack
4. Use PEFT/continued pretraining for larger checkpoints
5. Use MoE to emulate higher capacity under fixed FLOPs

See `docs/FEASIBILITY_3xA100.md` for detailed math and risk analysis.
