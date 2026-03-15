"""Local auto-chain: watches T-001r2 completion, then runs lm-eval benchmarks.

Polls results.json until training reaches max_steps, then sequentially runs:
  1. Baseline lm-eval (TinyLlama base)
  2. Adapted lm-eval (merged model from T-001r2)

Works on Windows (no kill -0 dependency).
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(r"C:\Falco\rhombic\results")
T001R2_DIR = RESULTS_DIR / "T-001-full-r2"
LOG = RESULTS_DIR / "local_auto_chain.log"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def get_training_status():
    """Read results.json and return (last_step, max_steps)."""
    results_file = T001R2_DIR / "results.json"
    if not results_file.exists():
        return 0, 10000
    with open(results_file) as f:
        data = json.load(f)
    fb = data.get("feedback_log", [])
    if not fb:
        return 0, data.get("config", {}).get("max_steps", 10000)
    max_steps = data.get("config", {}).get("max_steps", 10000)
    return fb[-1]["step"], max_steps

def run_lm_eval(model_path, output_name):
    """Run lm-eval harness on a model."""
    output_path = RESULTS_DIR / "benchmarks" / output_name
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", "hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq",
        "--batch_size", "auto",
        "--output_path", str(output_path),
        "--device", "cuda:0",
    ]
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        log(f"lm-eval complete for {output_name}")
        # Print last 20 lines of stdout (contains results table)
        for line in result.stdout.strip().split("\n")[-20:]:
            log(f"  {line}")
    else:
        log(f"lm-eval FAILED for {output_name}")
        for line in result.stderr.strip().split("\n")[-10:]:
            log(f"  ERR: {line}")
    return result.returncode

def main():
    log("Local auto-chain started. Watching T-001r2...")

    # Poll until training completes
    while True:
        step, max_steps = get_training_status()
        if step >= max_steps:
            log(f"T-001r2 COMPLETE at step {step}/{max_steps}")
            break
        log(f"T-001r2 at step {step}/{max_steps} — sleeping 5 min")
        time.sleep(300)  # Check every 5 minutes

    # Step 1: Baseline benchmarks
    log("=== Phase 1: Baseline benchmarks ===")
    run_lm_eval("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama_baseline")

    # Step 2: Adapted model benchmarks
    merged_dir = T001R2_DIR / "merged_model"
    if merged_dir.exists():
        log("=== Phase 2: Adapted model benchmarks ===")
        run_lm_eval(str(merged_dir), "tinyllama_t001r2_adapted")
    else:
        log(f"WARNING: No merged model at {merged_dir}")
        log("Run manually after merge completes.")

    log("=== Local auto-chain complete! ===")

if __name__ == "__main__":
    main()
