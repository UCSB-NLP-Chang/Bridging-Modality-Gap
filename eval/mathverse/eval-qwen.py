"""Score MathVerse predictions using Qwen-3 8B as a judge."""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List
import pdb
from mathruler.grader import extract_boxed_content
from tqdm import tqdm
from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: """.strip()


MODEL_DIR = "Qwen/Qwen3-8B"
SAMPLING = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2)
CHOICE_SET = {"a", "b", "c", "d", "e"}
ALL_VERSIONS = (
    "Vision Intensive",
    "Vision Dominant",
    "Vision Only",
    "Text Dominant",
    "Text Lite",
)

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = HERE / "last_results"

_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM(model=MODEL_DIR, dtype="bfloat16", trust_remote_code=True)
    return _LLM


def extract_numbers(text: str) -> List[float]:
    if not isinstance(text, str):
        return []
    matches = re.findall(r"-?\d+\.\d+(?:e[+-]?\d+)?|-?\d+", text)
    return [float(m) for m in matches]


def numbers_match_with_tolerance(gt_values: List[float], pred_values: List[float], tolerance: float = 0.05) -> bool:
    if not gt_values or not pred_values:
        return False

    if len(gt_values) == len(pred_values):
        return all(abs(g - p) / abs(g) <= tolerance if g != 0 else abs(p) <= tolerance for g, p in zip(gt_values, pred_values))

    if len(gt_values) == 1:
        g = gt_values[0]
        if g == 0:
            return any(abs(p) <= tolerance for p in pred_values)
        return any(abs(g - p) / abs(g) <= tolerance for p in pred_values)

    if len(pred_values) == 1:
        p = pred_values[0]
        if p == 0:
            return any(abs(g) <= tolerance for g in gt_values)
        return any(abs(g - p) / abs(g) <= tolerance if g != 0 else abs(p) <= tolerance for g in gt_values)

    return False


def recover_from_none(raw_prediction: str) -> str:
    try:
        tail = raw_prediction.split("</think>", 1)[1]
    except Exception:
        return raw_prediction
    match = re.search(r"boxed{(.*?)}", tail)
    return match.group(1) if match else tail


def judge_consistency_qwen(question: str, answer: str, prediction: str) -> int:
    prompt = PROMPT_TEMPLATE.format(question=question, gt=answer, extraction=prediction[-10:])
    try:
        outputs = get_llm().generate(prompt, SAMPLING, use_tqdm=False)
    except Exception as exc:  # pragma: no cover - robustness guard
        warnings.warn(f"Qwen judge failed: {exc}")
        return 0
    text = outputs[0].outputs[0].text.strip()
    match = re.search(r"[01]", text)
    return int(match.group()) if match else 0


def score_prediction(record: Dict[str, str]) -> int:
    question = record["question"]
    gt = record["answer"]
    extracted = extract_boxed_content(record["prediction"]) or ""
    extracted = extracted.replace("°", "").strip()

    question_type = record.get("question_type")
    if question_type == "multi-choice":
        choice = extracted.lower().strip()
        if choice in CHOICE_SET:
            if gt.lower().strip() == choice:
                return 1
            return 0
    prediction = extracted
    if extracted.lower() == "none":
        prediction = recover_from_none(record["prediction"]).strip()

    gt_numbers = extract_numbers(gt)
    pred_numbers = extract_numbers(prediction)
    if gt_numbers and pred_numbers:
        if numbers_match_with_tolerance(gt_numbers, pred_numbers, tolerance=0.05):
            return 1
        return 0
    
    return judge_consistency_qwen(question, gt, prediction)


def evaluate_files(paths: Iterable[Path], emit: Callable[[str], None] = print) -> None:
    total_correct = 0
    total_samples = 0
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for path in paths:
        out_path = path.with_name(path.stem + "_scored_qwen.jsonl")
        if out_path.exists():
            emit(f"[skip] Already scored → {out_path}")
            with out_path.open("r", encoding="utf-8") as handle:
                scored = [json.loads(line) for line in handle]
            for sample in scored:
                flag = int(sample.get("correct", 0))
                total_correct += flag
                total_samples += 1
                version = sample.get("problem_version", "Unknown")
                stats[version]["correct"] += flag
                stats[version]["total"] += 1
            continue

        with path.open("r", encoding="utf-8") as handle:
            data = [json.loads(line) for line in handle]

        for sample in tqdm(data, desc=path.name):
            flag = score_prediction(sample)
            sample["correct"] = flag

            total_correct += flag
            total_samples += 1

            version = sample.get("problem_version", "Unknown")
            stats[version]["correct"] += flag
            stats[version]["total"] += 1

        with out_path.open("w", encoding="utf-8") as fout:
            for row in data:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        emit(f"[✓] Scored file written → {out_path}")

    emit("\n===== SUMMARY (Qwen) =====")
    if total_samples:
        emit(f"Overall ACC: {total_correct}/{total_samples} = {total_correct/total_samples:.2%}")
    else:
        emit("No samples scored.")

    emit("\nAccuracy by problem_version:")
    for version in ALL_VERSIONS:
        if stats[version]["total"] == 0:
            continue
        c = stats[version]["correct"]
        t = stats[version]["total"]
        emit(f"  {version:<15}: {c}/{t} = {c/t:.2%}")

    for version, summary in stats.items():
        if version in ALL_VERSIONS:
            continue
        c = summary["correct"]
        t = summary["total"]
        emit(f"  {version:<15}: {c}/{t} = {c/t:.2%}")


def _collect_jsonl_groups(root: Path) -> Dict[Path, List[Path]]:
    groups: Dict[Path, List[Path]] = defaultdict(list)
    for path in root.rglob("*.jsonl"):
        if path.name.endswith("_scored_qwen.jsonl"):
            continue
        groups[path.parent].append(path)
    return {directory: sorted(files) for directory, files in groups.items()}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MathVerse predictions with Qwen-3 8B judge")
    parser.add_argument("paths", nargs="*", help="One or more prediction jsonl files")
    parser.add_argument("--scan-all-results", action="store_true", help="Score every jsonl file under --results-root")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Directory to scan when --scan-all-results is set (default: results_mathverse)",
    )
    parser.add_argument(
        "--scan-output",
        default=None,
        help="Optional file to mirror --scan-all-results output",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    if args.scan_all_results:
        root = Path(args.results_root)
        if not root.is_absolute():
            root = HERE / args.results_root
        if not root.exists():
            raise FileNotFoundError(f"Results root not found: {root}")
        root = root.resolve()

        groups = _collect_jsonl_groups(root)
        if not groups:
            print(f"No prediction files found under {root}.")
            return 0

        output_handle = None
        if args.scan_output:
            output_path = Path(args.scan_output)
            if not output_path.is_absolute():
                output_path = root / output_path
            output_dir = output_path.parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", encoding="utf-8")

            def emit(line: str = "") -> None:
                print(line)
                output_handle.write(line + "\n")
        else:

            def emit(line: str = "") -> None:
                print(line)

        try:
            directories = sorted(groups.items(), key=lambda item: str(item[0]))
            for idx, (directory, files) in enumerate(directories):
                label = directory.relative_to(root) if directory != root else Path(".")
                emit(f"Directory: {label}")
                evaluate_files(files, emit=emit)
                if idx != len(directories) - 1:
                    emit()
        finally:
            if output_handle is not None:
                output_handle.close()
        return 0

    if not args.paths:
        raise SystemExit("Please provide one or more prediction files or use --scan-all-results.")

    files = [Path(p) for p in args.paths]
    evaluate_files(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
