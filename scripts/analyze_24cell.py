#!/usr/bin/env python3
"""Analyze 24C-001 checkpoint data for 12+12 BD emergence.

Reads results.json from the 24-cell experiment and reports:
  - Eigenvalue spectrum analysis at each checkpoint
  - 12|12 split detection (gap between eigenvalue 12 and 13)
  - Co/cross ratio evolution (when measurable)
  - BD emergence timeline comparison with other polytopes

Can run locally on results.json copied from Hermes, or with --remote
to fetch via SSH.

Usage:
  python scripts/analyze_24cell.py results/channel-ablation/24C-001/results.json
  python scripts/analyze_24cell.py --remote  # SSH to hermes
  python scripts/analyze_24cell.py --watch --remote  # Continuous monitoring

TASUMER MAF — The geometry IS the argument.
"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

import numpy as np

# 24-cell pair specification (D4 root polytope)
# 24 vertices as permutations of (±1,±1,0,0) in R^4
# 12 antipodal pairs, grouped by coordinate planes:
# wx: (1,1,0,0)↔(-1,-1,0,0), (1,-1,0,0)↔(-1,1,0,0)
# wy: (1,0,1,0)↔(-1,0,-1,0), (1,0,-1,0)↔(-1,0,1,0)
# wz: (1,0,0,1)↔(-1,0,0,-1), (1,0,0,-1)↔(-1,0,1,0)  [corrected: (-1,0,0,1)]
# xy: (0,1,1,0)↔(0,-1,-1,0), (0,1,-1,0)↔(0,-1,1,0)
# xz: (0,1,0,1)↔(0,-1,0,-1), (0,1,0,-1)↔(0,-1,0,1)
# yz: (0,0,1,1)↔(0,0,-1,-1), (0,0,1,-1)↔(0,0,-1,1)
N_COAXIAL = 12   # pairs
N_CROSS = 264    # cross-axial pairs (C(24,2) - 12 = 276 - 12 = 264)
SIGNAL_DENSITY = N_COAXIAL / (N_COAXIAL + N_CROSS)  # 0.0435

# Reference data from other polytopes for comparison
REFERENCE = {
    "octahedron": {"n": 4, "k": 2, "step_first_bd": 500, "final_cocross": 473622},
    "rd":         {"n": 6, "k": 3, "step_first_bd": 200, "final_cocross": 70404},
    "tesseract":  {"n": 8, "k": 4, "step_first_bd": 400, "final_cocross": 41564},
}


def load_results(path: str | Path) -> dict:
    """Load results.json from local path."""
    with open(path) as f:
        return json.load(f)


def load_remote() -> dict:
    """Load results.json from Hermes via SSH."""
    remote_path = "/home/timm156/rhombic/results/channel-ablation/24C-001/results.json"
    result = subprocess.run(
        ["ssh", "hermes", f"cat {remote_path}"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        print(f"SSH error: {result.stderr}")
        sys.exit(1)
    return json.loads(result.stdout)


def analyze_eigenvalues(eig_pattern: list) -> dict:
    """Analyze an eigenvalue pattern for 12+12 BD structure.

    Args:
        eig_pattern: list of [value, multiplicity] pairs from results.json

    Returns:
        dict with analysis metrics
    """
    # Flatten eigenvalues
    eigs = []
    for val, mult in eig_pattern:
        eigs.extend([val] * int(mult))
    eigs = np.array(sorted(eigs))
    n = len(eigs)

    if n != 24:
        return {"error": f"Expected 24 eigenvalues, got {n}"}

    # Basic stats
    analysis = {
        "n_eigenvalues": n,
        "min": float(eigs[0]),
        "max": float(eigs[-1]),
        "mean": float(eigs.mean()),
        "fiedler": float(eigs[1]) if n > 1 else 0.0,
    }

    # 12|12 split analysis
    gap_12_13 = float(eigs[12] - eigs[11])  # gap between 12th and 13th eigenvalue
    inter_gaps = np.diff(eigs)
    max_gap = float(inter_gaps.max())
    max_gap_pos = int(inter_gaps.argmax())

    analysis["gap_12_13"] = gap_12_13
    analysis["max_gap"] = max_gap
    analysis["max_gap_position"] = max_gap_pos

    # BD detection: is the 12|13 gap the dominant gap?
    # For perfect BD: gap at position 11 (between eig[11] and eig[12]) should be
    # the largest gap by far, with eigs[0:12] near zero and eigs[12:24] large
    if max_gap > 0:
        gap_dominance = gap_12_13 / max_gap
    else:
        gap_dominance = 0.0
    analysis["gap_dominance"] = gap_dominance

    # Is the max gap at position 11? (between eigenvalues 12 and 13)
    analysis["split_at_12"] = (max_gap_pos == 11)

    # Block statistics (assuming 12+12 split)
    block_low = eigs[:12]
    block_high = eigs[12:]
    analysis["block_low_mean"] = float(block_low.mean())
    analysis["block_low_max"] = float(block_low.max())
    analysis["block_high_mean"] = float(block_high.mean())
    analysis["block_high_min"] = float(block_high.min())

    # Block separation ratio
    if block_low.max() > 0:
        analysis["block_separation"] = float(block_high.min() / block_low.max())
    else:
        analysis["block_separation"] = float("inf")

    # Smoothness: is the spectrum smooth (no split) or gapped?
    # Coefficient of variation of inter-eigenvalue gaps
    if inter_gaps.mean() > 0:
        analysis["gap_cv"] = float(inter_gaps.std() / inter_gaps.mean())
    else:
        analysis["gap_cv"] = 0.0

    # BD emergence score (0 = smooth/no BD, 1 = perfect 12+12)
    # Based on: gap dominance at position 11, block separation, near-zero low block
    if analysis["split_at_12"] and gap_12_13 > 0.01:
        bd_score = min(1.0, gap_dominance * 0.5 + min(1.0, analysis["block_separation"] / 100) * 0.5)
    else:
        bd_score = gap_dominance * 0.1 if max_gap_pos == 11 else 0.0
    analysis["bd_score"] = bd_score

    return analysis


def analyze_checkpoint(checkpoint: dict, step: int) -> dict:
    """Full analysis of a single checkpoint."""
    result = {
        "step": step,
        "fiedler_mean": checkpoint.get("fiedler_mean", 0),
        "co_cross_ratio": checkpoint.get("co_cross_ratio"),
        "val_loss": checkpoint.get("val_loss"),
        "contrastive_loss": checkpoint.get("contrastive_loss"),
        "spectral_loss": checkpoint.get("spectral_loss"),
        "wall_time": checkpoint.get("wall_time", 0),
    }
    return result


def print_report(data: dict):
    """Print comprehensive analysis report."""
    checkpoints = data.get("checkpoints", [])
    feedback_log = data.get("feedback_log", [])
    config = data.get("config", {})

    print("=" * 72)
    print("24C-001: 24-Cell D4 Root Polytope — Analysis Report")
    print("=" * 72)

    print(f"\nConfiguration:")
    print(f"  Model:    {config.get('model_name', 'unknown')}")
    print(f"  Channels: {config.get('n_channels', '?')}")
    print(f"  Steps:    {config.get('max_steps', '?')}")
    print(f"  Rank:     {config.get('rank', '?')}")

    print(f"\n  Bridge params:   {data.get('bridge_params', '?')}")
    print(f"  Trainable params: {data.get('trainable_params', '?')}")
    print(f"  Signal density:  {SIGNAL_DENSITY:.4f} (12 co / 276 total)")

    # Checkpoint timeline
    print(f"\n{'-' * 72}")
    print("Checkpoint Timeline")
    print(f"{'-' * 72}")

    if not checkpoints:
        print("  No checkpoints yet.")
        return

    for chk in checkpoints:
        step = chk["step"]
        fiedler = chk.get("fiedler_mean", 0)
        co_cross = chk.get("co_cross_ratio")
        val_loss = chk.get("val_loss", 0)
        wall_time = chk.get("wall_time", 0)

        co_str = f"{co_cross:,.0f}:1" if co_cross and co_cross > 1 else str(co_cross)
        print(f"\n  Step {step:>5d} ({wall_time/60:.1f} min):")
        print(f"    Fiedler: {fiedler:.6f}  |  Co/Cross: {co_str}  |  Val loss: {val_loss:.4f}")

    # Eigenvalue analysis at each feedback step
    print(f"\n{'-' * 72}")
    print("Eigenvalue Spectrum Analysis")
    print(f"{'-' * 72}")

    for entry in feedback_log:
        step = entry.get("step", 0)
        eig_pattern = entry.get("eigenvalue_pattern", [])
        if not eig_pattern or step == 0:
            continue

        analysis = analyze_eigenvalues(eig_pattern)
        if "error" in analysis:
            print(f"\n  Step {step}: {analysis['error']}")
            continue

        bd_indicator = "[BD]" if analysis["bd_score"] > 0.5 else "[~BD]" if analysis["bd_score"] > 0.1 else "[--]"
        split_str = "YES" if analysis["split_at_12"] else "no"

        print(f"\n  Step {step}: {bd_indicator}")
        print(f"    Eigenvalue range: [{analysis['min']:.6f}, {analysis['max']:.6f}]")
        print(f"    12|12 gap: {analysis['gap_12_13']:.6f}  (max gap: {analysis['max_gap']:.6f} at pos {analysis['max_gap_position']})")
        print(f"    Split at 12: {split_str}  |  Gap dominance: {analysis['gap_dominance']:.3f}")
        print(f"    Low block [0:12]:  mean={analysis['block_low_mean']:.6f}, max={analysis['block_low_max']:.6f}")
        print(f"    High block [12:24]: mean={analysis['block_high_mean']:.6f}, min={analysis['block_high_min']:.6f}")
        print(f"    Block separation: {analysis['block_separation']:.1f}x  |  BD score: {analysis['bd_score']:.3f}")

    # Comparison with other polytopes
    print(f"\n{'-' * 72}")
    print("Comparison with Other Polytopes")
    print(f"{'-' * 72}")
    print(f"\n  {'Polytope':<12} {'n':>3} {'k':>3} {'Signal':>8} {'First BD':>10} {'Final Co/Cross':>15}")
    print(f"  {'-'*12} {'-'*3} {'-'*3} {'-'*8} {'-'*10} {'-'*15}")
    for name, ref in REFERENCE.items():
        sig = ref["k"] / (ref["n"] * (ref["n"] - 1) / 2)
        print(f"  {name:<12} {ref['n']:>3} {ref['k']:>3} {sig:>8.4f} {ref['step_first_bd']:>10} {ref['final_cocross']:>14,}:1")
    print(f"  {'24-cell':<12} {24:>3} {12:>3} {SIGNAL_DENSITY:>8.4f} {'TBD':>10} {'TBD':>15}")

    print(f"\n  Signal density = co-axial pairs / total pairs")
    print(f"  24-cell signal density (0.044) is 3.7× lower than tesseract (0.167)")
    print(f"  Prediction: BD emergence may require more steps or be weaker")

    # Prediction
    latest = feedback_log[-1] if feedback_log else {}
    latest_step = latest.get("step", 0)
    latest_eigs = latest.get("eigenvalue_pattern", [])

    if latest_eigs and latest_step > 0:
        analysis = analyze_eigenvalues(latest_eigs)
        if "error" not in analysis:
            print(f"\n{'-' * 72}")
            print("Current Assessment")
            print(f"{'-' * 72}")
            if analysis["bd_score"] > 0.5:
                print(f"  [BD] BD EMERGING — 12+12 split detected at step {latest_step}")
                print(f"    Block separation: {analysis['block_separation']:.1f}x")
            elif analysis["bd_score"] > 0.1:
                print(f"  [~BD] EARLY BD SIGNAL — gap developing at position 12")
                print(f"    Gap dominance: {analysis['gap_dominance']:.3f}")
            elif analysis["split_at_12"]:
                print(f"  [--] GAP AT POSITION 12 — but not yet dominant")
                print(f"    Gap: {analysis['gap_12_13']:.6f} vs max: {analysis['max_gap']:.6f}")
            else:
                print(f"  [--] NO BD YET — smooth spectrum, max gap at position {analysis['max_gap_position']}")
                if latest_step < 500:
                    print(f"    Expected — tesseract showed first BD signal at step ~400")
                    print(f"    24-cell may need more steps due to lower signal density")

    print(f"\n{'=' * 72}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze 24C-001 checkpoint data for 12+12 BD emergence"
    )
    parser.add_argument(
        "results_file", nargs="?", default=None,
        help="Path to results.json (default: fetch from Hermes)"
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="Fetch results.json from Hermes via SSH"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of human-readable report"
    )
    args = parser.parse_args()

    if args.results_file:
        data = load_results(args.results_file)
    elif args.remote:
        data = load_remote()
    else:
        # Try local path first, then remote
        local_path = Path("results/channel-ablation/24C-001/results.json")
        if local_path.exists():
            data = load_results(local_path)
        else:
            print("No local results.json found, fetching from Hermes...")
            data = load_remote()

    if args.json:
        # Add eigenvalue analysis to each feedback entry
        for entry in data.get("feedback_log", []):
            eig_pattern = entry.get("eigenvalue_pattern", [])
            if eig_pattern:
                entry["bd_analysis"] = analyze_eigenvalues(eig_pattern)
        json.dump(data, sys.stdout, indent=2)
    else:
        print_report(data)


if __name__ == "__main__":
    main()
