import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = HERE / "results_mathverse"
VISION_VERSIONS = {"Vision Intensive", "Vision Dominant", "Vision Only"}
TEXT_VERSIONS = {"Text Dominant", "Text Lite"}
QUESTION_MULTI = "multi-choice"
QUESTION_FREE = "free-form"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MathVerse evaluation results")
    parser.add_argument(
        "files",
        nargs="*",
        help="Scored jsonl result files (defaults to every *_scored_qwen.jsonl under --results-root)",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Directory to scan when no files are provided",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also print accuracy per problem_version",
    )
    return parser.parse_args()


def _collect_files(paths: List[str], results_root: str) -> List[Path]:
    if paths:
        return [Path(p).expanduser().resolve() for p in paths]

    root = Path(results_root)
    if not root.is_absolute():
        root = HERE / results_root
    if not root.exists():
        raise FileNotFoundError(f"Results root not found: {root}")
    root = root.resolve()

    candidates = sorted(root.glob("*_scored_qwen.jsonl"))
    if not candidates:
        candidates = sorted(root.glob("*.jsonl"))
    if not candidates:
        raise SystemExit(f"No scored jsonl files found under {root}.")
    return candidates


def _bucket() -> Dict[str, int]:
    return {"correct": 0, "total": 0}


def evaluate(paths: Iterable[Path]) -> Dict[str, Dict[str, float]]:
    total_correct = total_samples = 0
    vision_correct = vision_total = 0
    text_correct = text_total = 0
    version_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    type_stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: {
            "overall": _bucket(),
            "vision": _bucket(),
            "text": _bucket(),
            "versions": defaultdict(_bucket),
        }
    )

    for path in paths:
        print(f"Processing: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        for sample in tqdm(data, desc=path.name):
            prediction = sample.get("prediction", "")
            if prediction and "[gemini error]" not in prediction:
                flag = int(sample.get("correct", 0))
                version = sample.get("problem_version", "Unknown")
                q_type = sample.get("question_type", QUESTION_FREE)
                if q_type != QUESTION_MULTI:
                    q_type = QUESTION_FREE

                total_correct += flag
                total_samples += 1

                version_stats[version]["correct"] += flag
                version_stats[version]["total"] += 1

                type_stats[q_type]["overall"]["correct"] += flag
                type_stats[q_type]["overall"]["total"] += 1
                type_stats[q_type]["versions"][version]["correct"] += flag
                type_stats[q_type]["versions"][version]["total"] += 1

                if version in VISION_VERSIONS:
                    vision_correct += flag
                    vision_total += 1
                    type_stats[q_type]["vision"]["correct"] += flag
                    type_stats[q_type]["vision"]["total"] += 1
                elif version in TEXT_VERSIONS:
                    text_correct += flag
                    text_total += 1
                    type_stats[q_type]["text"]["correct"] += flag
                    type_stats[q_type]["text"]["total"] += 1

    return {
        "overall": {"correct": total_correct, "total": total_samples},
        "vision": {"correct": vision_correct, "total": vision_total},
        "text": {"correct": text_correct, "total": text_total},
        "versions": version_stats,
        "type": type_stats,
    }


def _fmt(correct: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{correct}/{total} = {correct/total:.2%}"


def main() -> None:
    args = parse_args()
    files = _collect_files(args.files, args.results_root)
    stats = evaluate(files)

    overall = stats["overall"]
    vision = stats["vision"]
    text = stats["text"]

    print("\n========== SUMMARY ==========")
    print(f"Overall ACC: {_fmt(overall['correct'], overall['total'])}")
    print(f"Vision  ACC: {_fmt(vision['correct'], vision['total'])}")
    print(f"Text    ACC: {_fmt(text['correct'], text['total'])}")

    type_stats = stats["type"]

    print("\n========== SUMMARY BY QUESTION TYPE ==========")
    for q_type in sorted(type_stats.keys()):
        q_label = q_type.upper()
        overall_bucket = type_stats[q_type]["overall"]
        print(f"\n[{q_label}] ACC: {_fmt(overall_bucket['correct'], overall_bucket['total'])}")

        vision_bucket = type_stats[q_type]["vision"]
        if vision_bucket["total"]:
            print(f"Vision  ACC: {_fmt(vision_bucket['correct'], vision_bucket['total'])}")
        text_bucket = type_stats[q_type]["text"]
        if text_bucket["total"]:
            print(f"Text    ACC: {_fmt(text_bucket['correct'], text_bucket['total'])}")

        if args.full:
            print(f"Accuracy by problem_version ({q_label}):")
            for version, summary in sorted(type_stats[q_type]["versions"].items()):
                print(f"  {version:<15}: {_fmt(summary['correct'], summary['total'])}")

    if args.full:
        print("\n========== ACCURACY BY PROBLEM VERSION ==========")
        for version, summary in sorted(stats["versions"].items()):
            print(f"  {version:<15}: {_fmt(summary['correct'], summary['total'])}")


if __name__ == "__main__":
    main()
