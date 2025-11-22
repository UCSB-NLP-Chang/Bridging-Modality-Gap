#!/usr/bin/env python3
"""Parallel runner scheduling (full_cond, base) using dedicated scripts."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import List

HERE = Path(__file__).resolve().parent
DEFAULT_BASE_RUNNER = HERE / "test-qwen-2d5-vl.py"
DEFAULT_FULL_RUNNER = HERE / "test-qwen-2d5-vl-fullcond-prompt.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run (full_cond, base) inference for multiple models in parallel")
    parser.add_argument("--model", dest="models", action="append", help="Model alias or path. Repeatable.")
    parser.add_argument("--model-list", type=str, help="Path to text file with one model per line (blank/# lines ignored)")
    parser.add_argument("--base-gpu", type=int, default=0, help="First GPU id to use")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs (workers) to spawn")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child processes")
    parser.add_argument("--base-runner", default=str(DEFAULT_BASE_RUNNER), help="Inference script for base mode")
    parser.add_argument("--full-runner", default=str(DEFAULT_FULL_RUNNER), help="Inference script for full_cond mode")
    parser.add_argument("--num-prompts", type=int, help="Forwarded to both runners if set")
    parser.add_argument("--batch-size", type=int, help="Forwarded to full_cond runner if set")
    parser.add_argument("--max-tokens", type=int, help="Forwarded to both runners if set")
    parser.add_argument("--temperature", type=float, help="Forwarded to both runners if set")
    parser.add_argument("--seed", type=int, default=42, help="Forwarded to both runners")
    parser.add_argument("--modality", choices=["image", "video"], help="Forwarded to base runner if set")
    parser.add_argument("--num-frames", type=int, help="Forwarded to base runner if set")
    parser.add_argument("--image-repeat-prob", type=float, help="Forwarded to base runner if set")
    parser.add_argument("--use-different-prompt-per-request", action="store_true", help="Forwarded flag for base runner")
    parser.add_argument("--data-json", type=str, help="Forwarded to full_cond runner if set")
    parser.add_argument("--image-root", type=str, help="Forwarded to full_cond runner if set")
    parser.add_argument("--out-dir", type=str, help="Forwarded to full_cond runner if set")
    parser.add_argument("--time-generate", action="store_true", help="Forwarded flag for both runners")
    parser.add_argument("--disable-mm-preprocessor-cache", action="store_true", help="Forwarded flag for both runners")
    parser.add_argument("--dry-run", action="store_true", help="Print job plan without executing")
    return parser.parse_args()


def _load_models(args: argparse.Namespace) -> List[str]:
    models: List[str] = []
    if args.models:
        models.extend(m.strip() for m in args.models if m and m.strip())
    if args.model_list:
        path = Path(args.model_list)
        if not path.exists():
            raise FileNotFoundError(f"--model-list file not found: {path}")
        for line in path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            models.append(s)
    if not models:
        raise SystemExit("Please provide at least one model via --model (repeatable) or --model-list")
    seen = set()
    ordered: List[str] = []
    for m in models:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered


def _add_if(flags: List[str], option: str, value) -> None:
    if value is not None:
        flags.extend([option, str(value)])


def _base_flags(args: argparse.Namespace) -> List[str]:
    flags: List[str] = []
    _add_if(flags, "--num-prompts", args.num_prompts)
    _add_if(flags, "--modality", args.modality)
    _add_if(flags, "--num-frames", args.num_frames)
    _add_if(flags, "--seed", args.seed)
    _add_if(flags, "--temperature", args.temperature)
    _add_if(flags, "--image-repeat-prob", args.image_repeat_prob)
    _add_if(flags, "--max-tokens", args.max_tokens)
    if args.time_generate:
        flags.append("--time-generate")
    if args.disable_mm_preprocessor_cache:
        flags.append("--disable-mm-preprocessor-cache")
    if args.use_different_prompt_per_request:
        flags.append("--use-different-prompt-per-request")
    return flags


def _full_flags(args: argparse.Namespace) -> List[str]:
    flags: List[str] = []
    _add_if(flags, "--num-prompts", args.num_prompts)
    _add_if(flags, "--batch-size", args.batch_size)
    _add_if(flags, "--max-tokens", args.max_tokens)
    _add_if(flags, "--temperature", args.temperature)
    _add_if(flags, "--seed", args.seed)
    _add_if(flags, "--data-json", args.data_json)
    _add_if(flags, "--image-root", args.image_root)
    _add_if(flags, "--out-dir", args.out_dir)
    if args.time_generate:
        flags.append("--time-generate")
    if args.disable_mm_preprocessor_cache:
        flags.append("--disable-mm-preprocessor-cache")
    return flags


def build_jobs(args: argparse.Namespace) -> List[dict]:
    models = _load_models(args)
    base_runner = Path(args.base_runner).resolve()
    full_runner = Path(args.full_runner).resolve()
    if not base_runner.exists():
        raise FileNotFoundError(f"Base runner script not found: {base_runner}")
    if not full_runner.exists():
        raise FileNotFoundError(f"Full_cond runner script not found: {full_runner}")

    base_args = _base_flags(args)
    full_args = _full_flags(args)

    modes = [
        ("full_cond", full_runner, full_args),
        ("base", base_runner, base_args),
    ]

    jobs: List[dict] = []
    jid = 0
    for model in models:
        for mode_name, runner_path, mode_args in modes:
            cmd = [args.python, str(runner_path), "--ip-model-name", model, *mode_args]
            jobs.append({
                "jid": jid,
                "model": model,
                "mode": mode_name,
                "runner": runner_path,
                "cmd": cmd,
            })
            jid += 1
    return jobs


def stream_pipe(pipe, log_f, prefix: str, to_stderr: bool) -> None:
    target = sys.stderr if to_stderr else sys.stdout
    try:
        for line in pipe:
            log_f.write(line)
            target.write(f"{prefix}{line}")
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)

    ts = time.strftime("%Y%m%d-%H%M%S")
    logs_dir = HERE / "results" / "_parallel_logs_split" / ts
    logs_dir.mkdir(parents=True, exist_ok=True)

    queue: Queue = Queue()
    for job in jobs:
        queue.put(job)

    if args.dry_run:
        last_gpu = args.base_gpu + max(0, args.num_gpus - 1)
        print(f"[dry-run] {queue.qsize()} jobs across GPUs {args.base_gpu}..{last_gpu}")
        for job in list(queue.queue):
            print(f"  - job{job['jid']}: {job['model']}/{job['mode']} -> {' '.join(job['cmd'])}")
        print(f"[dry-run] Logs will be under: {logs_dir}")
        return 0

    active_procs: List[subprocess.Popen] = []
    active_lock = threading.Lock()
    any_failure = {"rc": 0}

    def worker_loop(gpu_slot: int) -> None:
        gpu_id = args.base_gpu + gpu_slot
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                break

            jid = job["jid"]
            model = job["model"]
            mode_name = job["mode"]
            cmd = job["cmd"]

            out_log = logs_dir / f"gpu{gpu_id}-job{jid}-{model}-{mode_name}.out.log"
            err_log = logs_dir / f"gpu{gpu_id}-job{jid}-{model}-{mode_name}.err.log"

            print(f"[launch][GPU {gpu_id}] job{jid}:{model}/{mode_name} -> {' '.join(cmd)}")
            print(f"         logs: {out_log} | {err_log}")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(HERE),
            )

            with active_lock:
                active_procs.append(proc)

            stdout_f = open(out_log, "w", buffering=1)
            stderr_f = open(err_log, "w", buffering=1)

            prefix = f"[gpu{gpu_id} job{jid}:{model}/{mode_name}] "
            t_out = threading.Thread(target=stream_pipe, args=(proc.stdout, stdout_f, prefix, False), daemon=True)
            t_err = threading.Thread(target=stream_pipe, args=(proc.stderr, stderr_f, prefix, True), daemon=True)
            t_out.start()
            t_err.start()

            code = 0
            try:
                code = proc.wait()
            finally:
                try:
                    t_out.join()
                    t_err.join()
                except Exception:
                    pass
                try:
                    stdout_f.flush()
                    stderr_f.flush()
                except Exception:
                    pass
                stdout_f.close()
                stderr_f.close()
                with active_lock:
                    try:
                        active_procs.remove(proc)
                    except ValueError:
                        pass

            if code != 0:
                any_failure["rc"] = code
                print(f"[error][GPU {gpu_id}] job{jid}:{model}/{mode_name} exited with code {code}")

            queue.task_done()

    workers = []
    for slot in range(max(1, args.num_gpus)):
        worker = threading.Thread(target=worker_loop, args=(slot,), daemon=True)
        worker.start()
        workers.append(worker)

    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        print("\n[interrupt] Stopping all jobs ...")
        with active_lock:
            for proc in list(active_procs):
                try:
                    proc.terminate()
                except Exception:
                    pass
        time.sleep(2)
        with active_lock:
            for proc in list(active_procs):
                try:
                    proc.kill()
                except Exception:
                    pass
        return 130

    print(f"All done. Logs at: {logs_dir}")
    return any_failure["rc"]


if __name__ == "__main__":
    raise SystemExit(main())
