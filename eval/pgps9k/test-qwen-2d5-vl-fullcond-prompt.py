# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
Offline VLM inference with vLLM (image QA demo)
- Works with Qwen/Qwen2.5-VL, InternVL, MiniCPM-V families
- Uses chat templates to insert the correct image placeholders
- Ensures #image placeholders == #provided images to avoid "prompt updates" error
"""
from tqdm import tqdm

import os
import json
from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple, Optional, List, Tuple, Dict

from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser

import random



# ============ Data structures ============
class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: List[str]
    stop_token_ids: Optional[List[int]] = None
    lora_requests: Optional[List[LoRARequest]] = None


# ============ Model name resolver ============
def resolve_model_name(input_model_name: str) -> str:
    """Map short aliases / local ckpt tags to HF repo or local dir."""
    m = input_model_name
    local_aliases = {
    "qwen2_5_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_5_vl_7b_kl_curriculum": "yaogy/qwen2_5_vl_7b_kl_curriculum",
    "qwen2_5_vl_3b_kl_curriculum": "yaogy/qwen2_5_vl_3b_kl_curriculum",
    }
    if m in local_aliases:
        return local_aliases[m]

    # Fallback: assume it's already a HF repo id or local dir
    return m


def render_prompt(
    question: str,
) -> str:

    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"
    
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"

    return prompt


# ============ Helpers for your dataset-specific pre-processing ============
def process_condition(message: str) -> str:
    return message


def process_image_info(condition: str) -> str:
    tokens = condition.strip().split()
    if not tokens:
        return ""

    # Rule 5: "line d lieson A C E F" — A,C,E,F lie on line d
    if tokens[0] == "line" and len(tokens) >= 4 and tokens[2] == "lieson":
        line_name = tokens[1]
        points = ", ".join(tokens[3:])
        return f"{points} lie on line {line_name}"

    # Rule 1: "line U A" — return nothing
    if tokens[0] == "line" and len(tokens) == 3:
        return ""

    # Rule 2: "line R A P" — A is on line RP
    if tokens[0] == "line" and len(tokens) == 4:
        return f"{tokens[2]} is on line {tokens[1]}{tokens[3]}"

    # Rule 3: "line D E I K" — E, I are on line DK
    if tokens[0] == "line" and len(tokens) >= 5:
        base = tokens[1:-2]
        line_ends = tokens[-2:]
        base_str = ", ".join(base)
        return f"{base_str} are on line {line_ends[0]}{line_ends[1]}"

    # Rule 4: "\\odot A lieson S T U P Q R"
    if tokens[0] == "\\odot" and len(tokens) >= 4 and tokens[2] == "lieson":
        circle_name = tokens[1]
        points = ", ".join(tokens[3:])
        return f"In \\odot {circle_name}, point {points} lie on \\odot {circle_name}"

    return ""


def build_question_text(item: Dict) -> str:
    imageinfo_str = ""
    for s in item.get("parsing_stru_seqs", []):
        cur = process_image_info(s)
        if cur:
            imageinfo_str += cur + ", "
    condition_str = ""
    for c in item.get("parsing_sem_seqs", []):
        condition_str += process_condition(c) + ", "
    if condition_str.endswith(", "):
        condition_str = condition_str[:-2]

    return (
        "In this problem, "
        + imageinfo_str
        + condition_str
        + ".\nBased on these conditions, answer the question: "
        + item["text"]
        + "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        + "The reasoning process MUST BE enclosed within <think> </think> tags. "
        + "The final answer MUST BE put in \\boxed{}"
    )


# ============ Timer ============
@contextmanager
def time_counter(enable: bool):
    if enable:
        import time
        st = time.time()
        yield
        el = time.time() - st
        print("-" * 50)
        print(f"-- generate time = {el:.3f}s")
        print("-" * 50)
    else:
        yield


# ============ Arg parser ============
def parse_args():
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with vision-language models for text generation'
    )
    parser.add_argument('--ip-model-name', required=True, type=str,
                        help='Model alias or HF repo/local path (see resolve_model_name).')
    parser.add_argument('--data-json', type=str, default="PGPS9K/test.json",
                        help='Path to dataset JSON.')
    parser.add_argument('--image-root', type=str, default="PGPS9K/Diagram_Visual",
                        help='Folder containing images referenced by dataset.')
    parser.add_argument('--num-prompts', type=int, default=4, help='Candidates per question (n).')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=16)  # maps to max_num_seqs
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable-mm-preprocessor-cache', action='store_true')
    parser.add_argument('--time-generate', action='store_true')
    parser.add_argument('--out-dir', type=str, default=None)
    return parser.parse_args()


# ============ Main ============
def main():
    args = parse_args()

    model_name = resolve_model_name(args.ip_model_name)
    random.seed(args.seed)
    # Engine args
    engine_args = EngineArgs(
        model=model_name,
        tensor_parallel_size=1,
        max_num_seqs=args.batch_size,
        mm_processor_kwargs={
            'min_pixels': 256 * 28 * 28,
            'max_pixels': 1280 * 28 * 28,
        },
        trust_remote_code=True,
        seed=args.seed,
    )

    llm = LLM(**asdict(engine_args))

    # Load data
    with open(args.data_json, "r") as f:
        test_items = json.load(f)
    question_keys = sorted(test_items.keys())

    # Build inputs
    inputs: List[dict] = []

    for qk in tqdm(question_keys):
        item = test_items[qk]
        question_text = build_question_text(item)
        # Render prompt & stop ids with the model's chat template
        prompt = render_prompt(question_text)

        # Load image
        img_path = os.path.join(args.image_root, item['diagram'])
        img: Image.Image = Image.open(img_path).convert("RGB")

        # vLLM offline: supply prompt + PIL.Image via multi_modal_data
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        })
        

    # Sampling params
    sampling_kwargs = {
        "temperature": args.temperature,
        "n": args.num_prompts,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }

    sampling_params = SamplingParams(**sampling_kwargs)
    
    # Generate
    with time_counter(args.time_generate):
        outputs = llm.generate(inputs, sampling_params=sampling_params)

    # Save outputs
    out_dir = args.out_dir or f"results/new_9k_{args.ip_model_name}_full_cond_max{args.max_tokens}_n{args.num_prompts}_tmp_{args.temperature}_bs{args.batch_size}"
    os.makedirs(out_dir, exist_ok=True)

    for i, o in enumerate(outputs):
        qkey = question_keys[i]
        for j, cand in enumerate(o.outputs):
            text = cand.text
            with open(os.path.join(out_dir, f"q{qkey}_attempt{j}.txt"), "w") as f:
                f.write(text)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
