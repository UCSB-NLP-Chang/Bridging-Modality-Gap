#!/usr/bin/env python3
"""Schedule MathVerse inference.py jobs across multiple GPUs in parallel."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
DEFAULT_RUNNER = HERE / "inference.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel runner for MathVerse inference")
    parser.add_argument("--model", dest="models", action="append", help="Model alias accepted by inference.py (repeatable)")
    parser.add_argument("--model-list", help="Text file listing models to evaluate (blank/# lines ignored)")
    parser.add_argument("--base-gpu", type=int, default=0, help="First CUDA device index to use")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPU workers to spawn")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child processes")
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER), help="Path to inference.py (or compatible) script")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without executing")
    parser.add_argument("--log-root", default=None, help="Where to store stdout/stderr logs (default: results/_parallel_logs)")

    parser.add_argument("--outdir", default="results_mathverse", help="Forwarded to inference.py")
    parser.add_argument("--tries", type=int, help="Forwarded to inference.py")
    parser.add_argument("--temperature", type=float, help="Forwarded to inference.py")
    parser.add_argument("--problem-versions", help="Forwarded to inference.py")
    parser.add_argument("--batch-size", type=int, help="Forwarded to inference.py")
    parser.add_argument("--seed", type=int, default=42, help="Forwarded to inference.py")
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
        raise SystemExit("Please provide at least one model via --model or --model-list")
    seen = set()
    ordered: List[str] = []
    for model in models:
        if model not in seen:
            ordered.append(model)
            seen.add(model)
    return ordered


def _forward_flags(args: argparse.Namespace) -> List[str]:
    forwarded: List[str] = []

    def add(flag: str, value: Any) -> None:
        forwarded.extend([flag, str(value)])

    add("--outdir", args.outdir)
    if args.tries is not None:
        add("--tries", args.tries)
    if args.temperature is not None:
        add("--temperature", args.temperature)
    if args.problem_versions:
        add("--problem-versions", args.problem_versions)
    if args.batch_size is not None:
        add("--batch-size", args.batch_size)
    if args.seed is not None:
        add("--seed", args.seed)
    return forwarded


def sanitize(text: str) -> str:
    safe = text.replace("/", "_").replace("\\", "_").replace(":", "_")
    return safe.replace(" ", "_")


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
    models = _load_models(args)
    runner = Path(args.runner).resolve()
    if not runner.exists():
        raise FileNotFoundError(f"Runner script not found: {runner}")

    forwarded = _forward_flags(args)
    log_root = Path(args.log_root) if args.log_root else HERE / "results" / "_parallel_logs"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logs_dir = log_root / timestamp
    logs_dir.mkdir(parents=True, exist_ok=True)

    queue: Queue[Dict[str, Any]] = Queue()
    jobs: List[Dict[str, Any]] = []
    for jid, model in enumerate(models):
        cmd = [args.python, str(runner), "--model", model, *forwarded]
        job = {"jid": jid, "model": model, "cmd": cmd}
        jobs.append(job)
        queue.put(job)

    if args.dry_run:
        last_gpu = args.base_gpu + max(0, args.num_gpus - 1)
        print(f"[dry-run] {queue.qsize()} jobs across GPUs {args.base_gpu}..{last_gpu}")
        for job in jobs:
            print(f"  - job{job['jid']}: {job['model']} -> {' '.join(job['cmd'])}")
        print(f"[dry-run] Logs will be under: {logs_dir}")
        return 0

    active: List[subprocess.Popen] = []
    lock = threading.Lock()
    status = {"rc": 0}

    def worker(slot: int) -> None:
        gpu_id = args.base_gpu + slot
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                break

            jid = job["jid"]
            model = job["model"]
            cmd = job["cmd"]
            tag = sanitize(model)
            out_log = logs_dir / f"gpu{gpu_id}-job{jid}-{tag}.out.log"
            err_log = logs_dir / f"gpu{gpu_id}-job{jid}-{tag}.err.log"

            print(f"[launch][GPU {gpu_id}] job{jid}:{model} -> {' '.join(cmd)}")
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
                cwd=str(runner.parent),
            )

            with lock:
                active.append(proc)

            stdout_f = open(out_log, "w", buffering=1)
            stderr_f = open(err_log, "w", buffering=1)

            prefix = f"[gpu{gpu_id} job{jid}:{model}] "
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
                    stdout_f.flush(); stderr_f.flush()
                except Exception:
                    pass
                stdout_f.close(); stderr_f.close()
                with lock:
                    try:
                        active.remove(proc)
                    except ValueError:
                        pass

            if code != 0:
                status["rc"] = code
                print(f"[error][GPU {gpu_id}] job{jid}:{model} exited with code {code}")

            queue.task_done()

    workers: List[threading.Thread] = []
    for slot in range(max(1, args.num_gpus)):
        thread = threading.Thread(target=worker, args=(slot,), daemon=True)
        thread.start()
        workers.append(thread)

    try:
        for thread in workers:
            thread.join()
    except KeyboardInterrupt:
        print("\n[interrupt] Terminating all jobs...")
        with lock:
            for proc in list(active):
                try:
                    proc.terminate()
                except Exception:
                    pass
        time.sleep(2)
        with lock:
            for proc in list(active):
                try:
                    proc.kill()
                except Exception:
                    pass
        return 130

    print(f"All done. Logs at: {logs_dir}")
    return status["rc"]


if __name__ == "__main__":
    raise SystemExit(main())
