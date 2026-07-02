#!/usr/bin/env python3
"""
Training Dashboard — Three-Machine Status at a Glance

Reads local results + SSHs to Hermes to show all active experiments.
Run: python scripts/dashboard.py

Requires: SSH key-based auth to hermes (ssh -p 2222 timm156@hermes)
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ---------------------------------------------------------------------------
# Local experiments
# ---------------------------------------------------------------------------
def read_local_experiments() -> list[dict]:
    """Read results.json from all local experiment directories."""
    experiments = []

    # Corpus baselines (L3-L6)
    baselines_dir = RESULTS_DIR / "corpus-baselines"
    if baselines_dir.exists():
        for d in sorted(baselines_dir.iterdir()):
            rj = d / "results.json"
            if rj.exists():
                try:
                    with open(rj) as f:
                        data = json.load(f)
                    cps = data.get("checkpoints", [])
                    if cps:
                        last = cps[-1]
                        experiments.append({
                            "machine": "Meridian",
                            "name": d.name,
                            "step": last.get("step", "?"),
                            "total_steps": 10000,
                            "val_loss": last.get("val_loss", None),
                            "fiedler": last.get("fiedler_mean", last.get("fiedler", None)),
                            "deviation": last.get("deviation_mean", last.get("deviation", None)),
                            "n_checkpoints": len(cps),
                        })
                except Exception as e:
                    experiments.append({
                        "machine": "Meridian",
                        "name": d.name,
                        "error": str(e),
                    })

    # Channel ablation (H1-H5 local copies)
    ablation_dir = RESULTS_DIR / "channel-ablation"
    if ablation_dir.exists():
        for fname in sorted(ablation_dir.glob("H-*-results.json")):
            try:
                with open(fname) as f:
                    data = json.load(f)
                cps = data.get("checkpoints", data if isinstance(data, list) else [])
                if cps:
                    last = cps[-1]
                    experiments.append({
                        "machine": "Hermes (cached)",
                        "name": fname.stem.replace("-results", ""),
                        "step": last.get("step", "?"),
                        "total_steps": 10000,
                        "val_loss": last.get("val_loss", last.get("lm_loss", None)),
                        "fiedler": last.get("fiedler_mean", last.get("fiedler", None)),
                        "n_checkpoints": len(cps),
                    })
            except Exception:
                pass

    return experiments


# ---------------------------------------------------------------------------
# Hermes experiments (via SSH)
# ---------------------------------------------------------------------------
def read_hermes_status() -> list[dict]:
    """SSH to Hermes and read latest training log lines."""
    experiments = []

    try:
        result = subprocess.run(
            ["ssh", "-p", "2222", "-o", "ConnectTimeout=5", "timm156@hermes",
             "for d in /home/timm156/rhombic/results/channel-ablation/H-ch*/; do "
             "  name=$(basename $d); "
             "  if [ -f $d/train.log ]; then "
             "    last=$(tail -1 $d/train.log); "
             "    echo \"$name|$last\"; "
             "  fi; "
             "done; "
             "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null"],
            capture_output=True, text=True, timeout=10
        )

        lines = result.stdout.strip().split("\n")
        gpu_line = None

        for line in lines:
            if "|" in line and line.startswith("H-ch"):
                parts = line.split("|", 1)
                name = parts[0].strip()
                log_line = parts[1].strip()

                # Parse: Step  3700/10000 | LM: 0.4260 | ...
                step = None
                loss = None
                import re
                step_match = re.search(r'Step\s+(\d+)/(\d+)', log_line)
                loss_match = re.search(r'LM:\s+([\d.]+)', log_line)
                eta_match = re.search(r'ETA:\s+(\d+)m', log_line)

                if step_match:
                    step = int(step_match.group(1))
                    total = int(step_match.group(2))
                if loss_match:
                    loss = float(loss_match.group(1))

                exp = {
                    "machine": "Hermes",
                    "name": name,
                    "step": step,
                    "total_steps": total if step_match else 10000,
                    "val_loss": loss,
                }
                if eta_match:
                    exp["eta_minutes"] = int(eta_match.group(1))
                experiments.append(exp)

            elif "%" in line and "MiB" in line:
                gpu_line = line.strip()

        if gpu_line:
            experiments.append({
                "machine": "Hermes",
                "name": "GPU",
                "info": gpu_line,
            })

    except subprocess.TimeoutExpired:
        experiments.append({"machine": "Hermes", "name": "CONNECTION", "error": "SSH timeout"})
    except Exception as e:
        experiments.append({"machine": "Hermes", "name": "CONNECTION", "error": str(e)})

    return experiments


# ---------------------------------------------------------------------------
# Local GPU status
# ---------------------------------------------------------------------------
def read_local_gpu() -> dict:
    """Read local GPU status via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return {"info": result.stdout.strip()}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def print_dashboard(local_exps: list, hermes_exps: list, gpu: dict):
    """Print formatted dashboard."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("=" * 72)
    print(f"  TRAINING DASHBOARD — {now}")
    print("=" * 72)
    print()

    # Local GPU
    print(f"  Meridian GPU: {gpu.get('info', gpu.get('error', 'unknown'))}")

    # Hermes GPU
    for e in hermes_exps:
        if e.get("name") == "GPU":
            print(f"  Hermes GPU:   {e.get('info', 'unknown')}")
            break
    print()

    # Table
    print(f"  {'Machine':<15} {'Experiment':<30} {'Step':<12} {'Val Loss':<10} {'Fiedler':<10} {'ETA':<8}")
    print(f"  {'-'*15} {'-'*30} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")

    all_exps = local_exps + [e for e in hermes_exps if e.get("name") != "GPU"]

    for e in all_exps:
        if "error" in e:
            print(f"  {e['machine']:<15} {e['name']:<30} ERROR: {e['error']}")
            continue

        machine = e.get("machine", "?")
        name = e.get("name", "?")
        step = e.get("step", "?")
        total = e.get("total_steps", "?")
        val_loss = e.get("val_loss")
        fiedler = e.get("fiedler")
        eta = e.get("eta_minutes")

        step_str = f"{step}/{total}" if step != "?" else "?"
        loss_str = f"{val_loss:.4f}" if val_loss is not None else "—"
        fiedler_str = f"{fiedler:.5f}" if fiedler is not None else "—"
        eta_str = f"{eta}m" if eta is not None else "—"

        print(f"  {machine:<15} {name:<30} {step_str:<12} {loss_str:<10} {fiedler_str:<10} {eta_str:<8}")

    print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    local_exps = read_local_experiments()
    hermes_exps = read_hermes_status()
    gpu = read_local_gpu()

    print_dashboard(local_exps, hermes_exps, gpu)


if __name__ == "__main__":
    main()
