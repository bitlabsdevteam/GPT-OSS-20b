#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2
torchrun --nproc_per_node=3 -m gpt_oss_20b.train --config configs/train_3xa100.yaml
