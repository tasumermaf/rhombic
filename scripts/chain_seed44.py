"""Watcher: wait for Seed-43 to complete, run lm-eval, then launch Seed-44.

Run with: nohup python scripts/chain_seed44.py >> results/chain_seed44.log 2>&1 &
"""
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results/Seed-43")
RESULTS_JSON = RESULTS_DIR / "results.json"
TARGET_STEP = 10000
POLL_INTERVAL = 300  # 5 minutes


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat()}] {msg}", flush=True)


def seed43_complete() -> bool:
    if not RESULTS_JSON.exists():
        return False
    try:
        d = json.loads(RESULTS_JSON.read_text())
        last = d["checkpoints"][-1]
        return last["step"] >= TARGET_STEP
    except (json.JSONDecodeError, KeyError, IndexError):
        return False


def run_lm_eval() -> None:
    """Run lm-eval on Seed-43 merged model (or merge first if needed)."""
    merged_dir = RESULTS_DIR / "merged_model"
    adapter_state = RESULTS_DIR / "adapter_state.pt"

    # If --save-merged didn't produce merged_model, merge manually
    if not merged_dir.exists() and adapter_state.exists():
        log("Merged model not found — running merge_adapter.py")
        merge_cmd = [
            "python", "scripts/merge_adapter.py",
            "--adapter", str(adapter_state),
            "--output", str(merged_dir),
        ]
        subprocess.run(merge_cmd, check=True)
        log(f"Merge complete: {merged_dir}")

    if not merged_dir.exists():
        log("WARNING: No merged model available for lm-eval, skipping")
        return

    # Run lm-eval with same tasks as previous baseline
    eval_cmd = [
        "python", "scripts/eval_language_benchmarks.py",
        "--model", str(merged_dir),
        "--tasks", "hellaswag", "arc_easy", "piqa", "winogrande", "openbookqa", "boolq",
        "--num-fewshot", "0",
        "--batch-size", "8",
        "--output", "results/lm-eval/seed43-adapted/results.json",
    ]
    log(f"Running lm-eval: {' '.join(eval_cmd)}")
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log("lm-eval complete for Seed-43")
        log(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    else:
        log(f"lm-eval FAILED (rc={result.returncode}): {result.stderr[-300:]}")


def main() -> None:
    log("Chain watcher started — waiting for Seed-43 to hit 10K steps")

    while not seed43_complete():
        time.sleep(POLL_INTERVAL)

    log("Seed-43 complete!")

    # Phase 1: lm-eval benchmarks (~10 min)
    log("Phase 1: Running lm-eval on Seed-43 merged model")
    run_lm_eval()

    # Phase 2: Launch Seed-44
    log("Phase 2: Launching Seed-44")
    cmd = [
        "python", "scripts/train_cybernetic.py",
        "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--seed", "44",
        "--output", "results/Seed-44",
        "--save-merged",
        "--max-steps", "10000",
        "--feedback-interval", "100",
        "--lr", "2e-4",
        "--batch-size", "2",
        "--gradient-accumulation", "8",
        "--warmup-steps", "200",
        "--bridge-mode", "identity",
        "--n-channels", "6",
        "--steersman-preset", "default",
    ]

    log(f"Command: {' '.join(cmd)}")

    with open("results/Seed-44.log", "w") as logfile:
        proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT)
        log(f"Seed-44 launched as PID {proc.pid}")
        proc.wait()
        log(f"Seed-44 finished with return code {proc.returncode}")


if __name__ == "__main__":
    main()
