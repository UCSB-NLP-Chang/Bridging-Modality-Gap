#!/usr/bin/env bash
set -euo pipefail

# Schedule the models listed in models_gpu1.txt on GPUs 3..4 (one job per GPU)
python run_models_two_modes_parallel_split.py \
    --model-list model_list.txt \
    --base-gpu 0 \
    --num-gpus 8 \
    --time-generate
