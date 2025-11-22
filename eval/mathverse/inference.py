#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import random
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset
from vllm import LLM, SamplingParams, EngineArgs
from tqdm import tqdm

MODEL_MAP = {
    "qwen2_5_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_5_vl_7b_kl_curriculum": "yaogy/qwen2_5_vl_7b_kl_curriculum",
    "qwen2_5_vl_3b_kl_curriculum": "yaogy/qwen2_5_vl_3b_kl_curriculum",
}

PLACEHOLDER = "<|image_pad|>"
def build_prompt(q: str) -> str:
    return ("<|im_start|>user\n"
            f"<|vision_start|>{PLACEHOLDER}<|vision_end|>{q}\n"
            "You FIRST think step-by-step inside <think></think>, "
            "then wrap the final answer in \\boxed{}."
            "<|im_end|>\n<|im_start|>assistant\n<think>")

def load_mathverse():
    return load_dataset("AI4Math/MathVerse", "testmini",
                        split="testmini", trust_remote_code=True)

# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_MAP, default="qwen2.5")
    parser.add_argument("--outdir", default="results_mathverse")
    parser.add_argument("--tries", type=int, default=4)          
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--problem-versions", default=None)
    parser.add_argument("--batch-size", type=int, default=16,    
                        help="How many samples per vLLM call")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    all_versions = {
        "vision": ("Vision Intensive", "Vision Dominant", "Vision Only"),
        "text":   ("Text Dominant", "Text Lite")
    }
    allowed_versions = None
    if args.problem_versions:
        allowed_versions = all_versions[args.problem_versions]

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fname = (f"{args.model}_{args.problem_versions or 'full'}_tries_{args.tries}_generations.jsonl")
    results_path = outdir / fname


    engine_args = EngineArgs(
        model=MODEL_MAP[args.model],
        max_num_seqs=args.batch_size,
        mm_processor_kwargs={
            'min_pixels': 256 * 28 * 28,
            'max_pixels': 1280 * 28 * 28,
        },
        trust_remote_code=True,
        tensor_parallel_size=1,
        seed=args.seed, 
    )
    init_args = engine_args.__dict__.copy()

    llm = LLM(**init_args)

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=4096,
        n=args.tries,
        seed=args.seed,
    )

    ds = load_mathverse()
    print(f"Loaded {len(ds):,} samples.")

    requests, metas = [], [] 
    for row in tqdm(ds, desc="Preparing requests"):
        if allowed_versions and row["problem_version"] not in allowed_versions:
            continue
        prompt = build_prompt(row["question_for_eval"])
        img = row["image"].convert("RGB")
        requests.append({"prompt": prompt, "multi_modal_data": {"image": img}})
        metas.append({                 
            "sample_index": row["sample_index"],
            "problem_index": row["problem_index"],
            "problem_version": row["problem_version"],
            "question": row["question_for_eval"],
            "question_type": row["question_type"],
            "answer": row["answer"],
        })

    with results_path.open("w", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(requests), args.batch_size),
                          desc="Running vLLM inference"):
            end = start + args.batch_size
            batch_requests = requests[start:end]
            batch_metas = metas[start:end]

            outputs = llm.generate(batch_requests, sampling, use_tqdm=False)
            # Modified to handle multiple outputs (n=4) per input
            for meta, output in zip(batch_metas, outputs):
                for i, generated in enumerate(output.outputs):
                    record: Dict[str, Any] = {
                        **meta, 
                        "prediction": generated.text,
                        "sample_id": i  # Add sample ID to distinguish between the 4 samples
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Results written to {results_path}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
