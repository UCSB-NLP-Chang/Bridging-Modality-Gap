#!/usr/bin/env bash
set -euo pipefail

python run_mathverse_parallel.py \
    --model-list models_list.txt \
    --base-gpu 0 \
    --num-gpus 4 \
    --outdir last_results \
    --tries 4 \
    --batch-size 16 \
    --seed 42 \
    --temperature 0.6
