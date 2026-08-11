"""S2 timing-pilot queue runner — sequential by construction.

Runs the registered S2 queue (4 families x 3 timing runs) one at a time by
invoking s2_timing_pilot.py synchronously, skipping runs whose TIMING.md
already exists. If a run is already in progress when this starts (detached
launch), it first waits for that run's TIMING.md. Each family is attempted
even if a prior family failed (gating failures fail fast and are recorded
by the pilot script); a (family, run) is attempted at most once.

Launch detached:
  Start-Process C:\miniconda3\envs\falco\python.exe -ArgumentList
    "scripts\s2_queue_runner.py" -RedirectStandardOutput
    "results\s2-timing-pilots\logs\queue_runner.log" -WindowStyle Hidden
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import gpu_guard  # gpu_guard retrofit 2026-08-11 (runner predates the arbiter)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "s2-timing-pilots"
PY = r"C:\miniconda3\envs\falco\python.exe"
FAMILIES = ["gemma2-2b", "qwen2.5-3b", "qwen2.5-7b", "llama3.1-8b"]
RUNS = [1, 2, 3]

os.environ.setdefault("HF_HUB_CACHE", r"C:\falco\hf-cache\hub")


def done(fam, k):
    return (OUT / fam / f"run_{k}" / "TIMING.md").exists()


def in_progress(fam, k):
    d = OUT / fam / f"run_{k}"
    return d.exists() and not done(fam, k)


def main():
    # Wait out any run already training from a prior detached launch.
    for fam in FAMILIES:
        for k in RUNS:
            if in_progress(fam, k):
                print(f"[queue] waiting on in-progress {fam} run {k}",
                      flush=True)
                while in_progress(fam, k):
                    time.sleep(120)
                print(f"[queue] {fam} run {k} finished", flush=True)

    for fam in FAMILIES:
        for k in RUNS:
            if done(fam, k):
                print(f"[queue] skip {fam} run {k} (TIMING.md exists)",
                      flush=True)
                continue
            # Between EVERY run, in addition to the guard inside the pilot
            # (house pattern, granularity_queue.py) — worst-case wait for
            # another GPU user is one run.
            gpu_guard.pause_gate()
            print(f"[queue] launching {fam} run {k}", flush=True)
            r = subprocess.run(
                [PY, str(ROOT / "scripts" / "s2_timing_pilot.py"),
                 "--family", fam, "--run", str(k)],
                cwd=str(ROOT), capture_output=True, text=True)
            tail = (r.stdout or "")[-400:] + (r.stderr or "")[-400:]
            print(f"[queue] {fam} run {k} exit {r.returncode}\n{tail}",
                  flush=True)
            if r.returncode != 0 and not done(fam, k):
                print(f"[queue] {fam} run {k} FAILED; continuing to next "
                      f"family (gating or OOM is a finding, not a retry)",
                      flush=True)
                break  # skip remaining runs of a failing family
    print("[queue] complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
