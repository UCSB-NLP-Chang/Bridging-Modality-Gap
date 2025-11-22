# Bridging the Modality Gap in Multimodal LLMs

This repository contains the code for our paper: [**Rethinking the Text-Vision Reasoning Imbalance in MLLMs through the Lens of Training Recipes**](https://arxiv.org/abs/2510.22836).


- Training is built on the VERL RL library under `verl/` and uses Qwen2.5‑VL models as the backbone.
- Evaluation includes PGPS9K and MathVerse.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Data: PGPS9K](#data-pgps9k)
  - [Prepare training parquet files](#prepare-training-parquet-files)
- [Training (VERL)](#training-verl)
- [Evaluation](#evaluation)
  - [PGPS9K (vLLM)](#pgps9k-vllm)
  - [MathVerse (vLLM)](#mathverse-vllm)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Repository Structure

```text
eval/
  pgps9k/                  # PGPS9K inference + evaluation with vLLM
    run.sh                 # Parallel runner
    run_models_two_modes_parallel_split.py
    test-qwen-2d5-vl.py                    # base prompt
    test-qwen-2d5-vl-fullcond-prompt.py   # full-condition prompt
    eval.py                 # grade answers.
    model_list.txt          # model aliases or paths to evaluate
    PGPS9K/                 # copy of the dataset for evaluation (see below)
  mathverse/               # MathVerse inference + summarization scripts
    run_mathverse_parallel.py
    inference.py
    eval-results.py
    model_list.txt

verl/                      # VERL training library and recipes
  data/                    # data prep scripts and parquet outputs
    geo9k.py               # PGPS9K → parquet (text+image or image-only)
    geo9k-contrastive.py   # PGPS9K → parquet (contrastive variant)
    PGPS9K/                # copy of the dataset for training (see below)
  dapo_full-text.sh        # Stage 1 training (text+image)
  dapo_no-text.sh          # Stage 2 training (image-only)
  dapo_kl.sh               # contrastive-KL training
  customized_reward_geo9k.py
```

## Setup
This project builds on the Verl reinforcement learning framework. Please follow the official Verl installation guide to set up your environment. Ensure that you have all dependencies installed (including PyTorch, Verl, and any model-specific requirements). The training scripts assume access to the Qwen-2.5-VL-7B model (from ModelScope or HuggingFace) as the base multimodal LLM.

## Data: PGPS9K
Request PGPS9K from the CASIA website: https://nlpr.ia.ac.cn/databases/CASIA-PGPS9K/index.html

Place a copy of the dataset in two locations:

- Training: `verl/data/PGPS9K/`
- Evaluation: `eval/pgps9k/PGPS9K/`

Expected folder layout at each location:

```text
PGPS9K/
├── train.json        # training split
├── test.json         # test split (evaluation)
└── Diagram_Visual/   # diagram images
    ├── <image_1>.png (or .jpg)
    ├── <image_2>.png
    └── ...
```

### Prepare training parquet files

From the repository root:

```bash
# Text + image (Stage 1)
python verl/data/geo9k.py --local_dir geo9k-full-text-condition

# Image-only (Stage 2)
python verl/data/geo9k.py --local_dir geo9k-no-text-condition --no-text

# Contrastive variant (for KL recipe)
python verl/data/geo9k-contrastive.py --local_dir geo9k-contrastive-text-condition
```

These commands create the following directories under `verl/data/`:

- `geo9k-full-text-condition/{train.parquet,test.parquet}`
- `geo9k-no-text-condition/{train.parquet,test.parquet}`
- `geo9k-contrastive-text-condition/{train.parquet,test.parquet}`

## Training (VERL)

### full text condition
```bash
bash dapo_full-text.sh
```

### without text condition
```bash
bash dapo_no-text.sh
```

### Mix data training
```bash
bash dapo-mix-data.sh
```

### Currirulum training
```bash
bash dapo-curriculum.sh
```

### KL training
```bash
bash dapo_kl.sh
```

### Curriculum after KL training
```bash
bash dapo-kl-curriculum.sh
```
Tips: You may need to convert the FSDP checkpoints to HuggingFace format for curriculum training and evaluation. See Verl documentation for details.

## Evaluation

### PGPS9K (vLLM)
1) Launch parallel inference (both base and full-condition prompts):

```bash
cd eval/pgps9k
bash run.sh
```

Outputs are saved under `eval/pgps9k/results/...` and logs under `eval/pgps9k/results/_parallel_logs_split/...`.

3) Compute accuracy against ground truth:

```bash
# Evaluate one result directory
python eval.py --answer-dir results/<your_result_dir> --gt-file PGPS9K/test.json

# Or scan all result subfolders and summarize
python eval.py --scan-all-results --results-root results --scan-output summary.txt
```


### MathVerse (vLLM)

- run a single model

```bash
cd eval/mathverse
python inference.py --model qwen2_5_vl_7b --outdir results_mathverse --tries 4 --temperature 0.8 --batch-size 16
```

- run multiple models in parallel

```bash
# Edit eval/mathverse/model_list.txt first
python run_mathverse_parallel.py --model-list model_list.txt --base-gpu 0 --num-gpus 8 --outdir results_mathverse
```

Evaluate the results:

- Score every jsonl file under the output directory and write scored files as `*_scored_qwen.jsonl`:

```bash
python eval-qwen.py --scan-all-results --results-root results_mathverse --scan-output scored_qwen_summary.txt
```

- Or score specific file(s):

```bash
python eval-qwen.py results_mathverse/<your_predictions>.jsonl
```

Summarize evaluation (overall/text/vision splits and per-type):

```bash
python eval-results.py --results-root results_mathverse --full
```

## Acknowledgement

Our training script is mainly based on [verl](https://github.com/volcengine/verl).

## Citation

Please cite this project if it helps your research.

```
@article{yao2025rethinking,
  title   = {Rethinking the Text-Vision Reasoning Imbalance in MLLMs through the Lens of Training Recipes},
  author  = {Yao, Guanyu and Wu, Qiucheng and Zhang, Yang and Wang, Zhaowen and Zhao, Handong and Chang, Shiyu},
  journal = {arXiv preprint arXiv:2510.22836},
  year    = {2025}
}
```
