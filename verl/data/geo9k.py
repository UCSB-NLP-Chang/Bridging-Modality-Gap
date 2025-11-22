# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prepare the PGPS9K dataset for supervised fine-tuning.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from tqdm import tqdm

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "PGPS9K"
TRAIN_JSON_PATH = DATA_DIR / "train.json"
TEST_JSON_PATH = DATA_DIR / "test.json"
DIAGRAM_DIR = DATA_DIR / "Diagram_Visual"
DEFAULT_OUTPUT_SUBDIR = "geo9k"
DEFAULT_INSTRUCTION = (
    r"You should FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
)


def load_data(json_path: Path) -> Dict[str, dict]:
    with json_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    logger.info("Loaded %d samples from %s", len(dataset), json_path)
    return dataset


def encode_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        try:
            return str(path.relative_to(SCRIPT_DIR))
        except ValueError:
            return path.name


def construct_full_condition(data_dict: dict, no_text=False) -> str:
    image_parts = [process_image_info(info) for info in data_dict["parsing_stru_seqs"]]
    image_parts = [part for part in image_parts if part]

    condition_parts = [process_condition(condition) for condition in data_dict["parsing_sem_seqs"]]
    condition_parts = [part for part in condition_parts if part]

    segments: List[str] = []
    if image_parts:
        segments.append(", ".join(image_parts))
    if condition_parts:
        segments.append(", ".join(condition_parts))

    if not no_text:
        condition_str = ", ".join(segments)
        return (
            f"In this problem, {condition_str}.\n"
            f"Based on these conditions, answer the question: {data_dict['text']}"
        )
    return f"Based on the information in the image, answer the question: {data_dict['text']}"


def process_condition(message: str) -> str:
    return message


def process_image_info(condition: str) -> str:
    tokens = condition.strip().split()

    if not tokens:
        return ""

    if tokens[0] == "line" and len(tokens) >= 4 and tokens[2] == "lieson":
        line_name = tokens[1]
        points = ", ".join(tokens[3:])
        return f"{points} lie on line {line_name}"

    if tokens[0] == "line" and len(tokens) == 3:
        return ""

    if tokens[0] == "line" and len(tokens) == 4:
        return f"{tokens[2]} is on line {tokens[1]}{tokens[3]}"

    if tokens[0] == "line" and len(tokens) >= 5:
        base = tokens[1:-2]
        line_ends = tokens[-2:]
        base_str = ", ".join(base)
        return f"{base_str} are on line {line_ends[0]}{line_ends[1]}"

    if tokens[0] == "\\odot" and len(tokens) >= 4 and tokens[2] == "lieson":
        circle_name = tokens[1]
        points = ", ".join(tokens[3:])
        return f"In \\odot {circle_name}, point {points} lie on \\odot {circle_name}"

    return ""


def prepare_split(split_name: str, split_data: Dict[str, dict], diagram_dir: Path, no_text=False) -> List[dict]:
    prompts: List[str] = []
    answers: List[str] = []
    image_paths: List[Path] = []

    for sample in split_data.values():
        prompts.append(construct_full_condition(sample,no_text))
        answers.append(sample["answer"])
        image_paths.append(diagram_dir / sample["diagram"])

    encoded_images = []
    for image_path in tqdm(image_paths, desc=f"Encoding {split_name} images"):
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with image_path.open("rb") as image_file:
            image_bytes = image_file.read()

        encoded_images.append([{"bytes": image_bytes, "path": encode_relative_path(image_path)}])

    raw_examples = []
    for prompt, answer, image in zip(prompts, answers, encoded_images):
        raw_examples.append(
            {
                "problem": prompt,
                "solution": answer,
                "answer": answer,
                "input_image": image,
            }
        )

    return raw_examples


def make_map_fn(split: str):
    def process_fn(example: dict, idx: int) -> dict:
        question_raw = example.pop("problem")
        answer_raw = example.pop("solution")
        solution = example.pop("answer")
        image = example.pop("input_image")

        prompt = "<image>" + question_raw + "\n" + DEFAULT_INSTRUCTION

        return {
            "data_source": "9k",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "images": image,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--no-text", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.local_dir)
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_data(TRAIN_JSON_PATH)
    test_data = load_data(TEST_JSON_PATH)

    train_examples = prepare_split("train", train_data, DIAGRAM_DIR, no_text=args.no_text)
    test_examples = prepare_split("test", test_data, DIAGRAM_DIR, no_text=args.no_text)

    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(str(output_dir / "train.parquet"))
    test_dataset.to_parquet(str(output_dir / "test.parquet"))

    logger.info("Saved train and test datasets to %s", output_dir)


if __name__ == "__main__":
    main()
