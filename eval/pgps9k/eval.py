import argparse
import json
import math
import os
import re
import warnings

import numpy as np
from mathruler.grader import extract_boxed_content, grade_answer

# -------------------------------------------------
# 1.  Config
# -------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_ROOT = os.path.join(HERE, "results")
DEFAULT_GT_FILE = os.path.join(HERE, "PGPS9K", "test.json")


# -------------------------------------------------
# 2.  CLI
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PGPS9K results")
    parser.add_argument(
        "--answer-dir",
        help="Directory containing answer files (defaults to latest run)",
        required=True,
    )
    parser.add_argument(
        "--gt-file",
        default=DEFAULT_GT_FILE,
        help="Ground-truth JSON file path",
    )
    parser.add_argument(
        "--scan-all-results",
        action="store_true",
        help="Evaluate each subdirectory under the results root",
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing result subdirectories when scanning",
    )
    parser.add_argument(
        "--scan-output",
        default=None,
        help="Optional text file to capture --scan-all-results output",
    )
    return parser.parse_args()


# -------------------------------------------------
# 3.  Helpers
# -------------------------------------------------
def is_close_enough(t_input, b_input):
    if t_input == "None" or b_input == "None":
        return False
    try:
        if isinstance(t_input, str):
            expr = re.sub(r'\\sqrt{([^}]+)}', r'math.sqrt(\1)', t_input)
            expr = re.sub(r'(\d)(math\.sqrt)', r'\1*\2', expr)
            expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
            expr = re.sub(r'\)(\s*\d)', r')*\1', expr)
            expr = re.sub(r'\)(\s*math\.sqrt)', r')*\1', expr)
            with warnings.catch_warnings():
                warnings.simplefilter("error", SyntaxWarning)
                t_val = eval(expr, {"__builtins__": {}}, {"math": math})
        else:
            t_val = float(t_input)
        b_val = float(b_input)
    except Exception:
        return False

    if t_val == b_val == 0:
        return True
    if b_val == 0:
        return False
    try:
        t_val - b_val
    except Exception:
        return False

    return abs(t_val - b_val) / abs(b_val) <= 0.01


def pass_at_1(total, correct):
    if total == 0:
        return float("nan")
    return 1 - np.prod(1 - 1 / np.arange(total - correct + 1, total + 1))


def show(tag, T, C, N, total_len=0, emit=print):
    emit(f"=== {tag} ===")
    if T:
        emit(f"Accuracy: {C / T:.4f}")
        emit(f"Total   : {T}")
        emit(f"Correct : {C}")
        emit(f"None    : {N}")
        emit(f"avg len: {total_len / T}")
        emit(f"Pass@1  : {pass_at_1(T, C):.4f}\n")
    else:
        emit("No data.\n")


# -------------------------------------------------
# 4.  Evaluation
# -------------------------------------------------
def evaluate_directory(answer_dir, gt_dict, label=None, emit=print):
    if label is not None:
        emit(f"Directory: {label}")

    tot = crr = none = 0
    total_len = 0

    for fname in sorted(os.listdir(answer_dir)):
        fpath = os.path.join(answer_dir, fname)
        if not os.path.isfile(fpath):
            continue

        parts = fname.split("_")
        if len(parts) < 2:
            continue

        qname = "prob_" + parts[1]

        with open(fpath) as f:
            answer = f.read()
            total_len += len(answer)
            ans = extract_boxed_content(answer).replace("°", "")

        gt = gt_dict.get(qname)
        if gt is None:
            continue

        ok = grade_answer(ans, gt["answer"]) or is_close_enough(ans, gt["answer"])

        tot += 1
        if ok:
            crr += 1
        if ans == "None":
            none += 1

    show("Overall", tot, crr, none, total_len, emit=emit)


# -------------------------------------------------
# 5.  Entrypoint
# -------------------------------------------------
def main():
    args = parse_args()

    answer_dir = args.answer_dir
    gt_file_path = args.gt_file
    results_root = args.results_root

    if not os.path.isabs(gt_file_path):
        gt_file_path = os.path.join(HERE, gt_file_path)

    if not os.path.exists(gt_file_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_file_path}")

    with open(gt_file_path) as f:
        gt_dict = json.load(f)

    if args.scan_all_results:
        if not os.path.isabs(results_root):
            results_root = os.path.join(HERE, results_root)
        if not os.path.isdir(results_root):
            raise NotADirectoryError(f"Results root not found: {results_root}")

        subdirs = [d for d in sorted(os.listdir(results_root)) if os.path.isdir(os.path.join(results_root, d))]
        if not subdirs:
            print(f"No subdirectories found under {results_root}.")
            return

        output_handle = None
        if args.scan_output:
            output_path = args.scan_output
            if not os.path.isabs(output_path):
                output_path = os.path.join(results_root, output_path)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_handle = open(output_path, "w", encoding="utf-8")

            def emit(line=""):
                print(line)
                output_handle.write(line + "\n")
        else:
            def emit(line=""):
                print(line)

        try:
            for idx, subdir in enumerate(subdirs):
                full_path = os.path.join(results_root, subdir)
                evaluate_directory(full_path, gt_dict, label=subdir, emit=emit)
                if idx != len(subdirs) - 1:
                    emit()
        finally:
            if output_handle is not None:
                output_handle.close()
    else:
        if not os.path.isabs(answer_dir):
            answer_dir = os.path.join(HERE, answer_dir)
        if not os.path.isdir(answer_dir):
            raise NotADirectoryError(f"Answer directory not found: {answer_dir}")
        evaluate_directory(answer_dir, gt_dict)


if __name__ == "__main__":
    main()
