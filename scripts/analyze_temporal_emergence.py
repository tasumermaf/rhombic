#!/usr/bin/env python3
"""Analyze temporal emergence of block-diagonal structure across checkpoint steps.

Works on flat-file checkpoint bridges (bridge_stepN_*.npy pattern).
"""
import os
import sys
import numpy as np
import json
import re

CO_PLANAR = [(0, 1), (2, 3), (4, 5)]
CO_PLANAR_SET = set(CO_PLANAR)


def analyze_bridge(B):
    """Analyze a single 6x6 bridge matrix."""
    if B.shape[0] != 6 or B.shape[1] != 6:
        return None

    co_mags = []
    cross_mags = []
    off_diag = []

    for i in range(6):
        for j in range(i + 1, 6):
            mag = abs(B[i, j])
            off_diag.append((mag, (i, j)))
            if (i, j) in CO_PLANAR_SET:
                co_mags.append(mag)
            else:
                cross_mags.append(mag)

    co_mean = np.mean(co_mags)
    cross_mean = np.mean(cross_mags)

    # Block-diagonal: top 3 off-diagonal entries are co-planar
    off_diag.sort(reverse=True)
    top3 = set(p for _, p in off_diag[:3])
    is_block_diag = top3 == CO_PLANAR_SET

    return co_mean, cross_mean, is_block_diag


def analyze_experiment(base_dir, pattern_type="flat"):
    """Analyze all checkpoint steps in an experiment directory.

    pattern_type:
      'flat' = bridge_stepN_*.npy in single directory
      'subdir' = checkpoint_step_N/bridge_*.npy subdirectories
    """
    if pattern_type == "flat":
        files = os.listdir(base_dir)
        steps = set()
        for f in files:
            m = re.match(r'bridge_step(\d+)_', f)
            if m:
                steps.add(int(m.group(1)))
        steps = sorted(steps)

        print(f"Found {len(steps)} checkpoint steps ({len(steps) * 112} est. bridges)")

        results = []
        for si, step in enumerate(steps):
            prefix = f"bridge_step{step}_"
            bridge_files = [f for f in files if f.startswith(prefix)]

            co_vals = []
            cross_vals = []
            bd_count = 0
            total = 0

            for bf in bridge_files:
                B = np.load(os.path.join(base_dir, bf))
                result = analyze_bridge(B)
                if result is None:
                    continue
                co_mean, cross_mean, is_bd = result
                co_vals.append(co_mean)
                cross_vals.append(cross_mean)
                if is_bd:
                    bd_count += 1
                total += 1

            if total == 0:
                continue

            overall_co = float(np.mean(co_vals))
            overall_cross = float(np.mean(cross_vals))
            ratio = overall_co / max(overall_cross, 1e-10)
            bd_pct = 100.0 * bd_count / total

            results.append({
                "step": step,
                "bridges": total,
                "co_mean": overall_co,
                "cross_mean": overall_cross,
                "ratio": float(ratio),
                "block_diag_pct": float(bd_pct),
            })

            if si % 20 == 0:
                print(f"  Step {step}: ratio={ratio:.1f}:1, BD={bd_pct:.0f}%")

        return results

    elif pattern_type == "subdir":
        checkpoint_dirs = []
        for d in os.listdir(base_dir):
            m = re.match(r'checkpoint_step_(\d+)', d)
            if m:
                checkpoint_dirs.append((int(m.group(1)), os.path.join(base_dir, d)))
        checkpoint_dirs.sort()

        print(f"Found {len(checkpoint_dirs)} checkpoint directories")

        results = []
        for si, (step, cdir) in enumerate(checkpoint_dirs):
            bridge_files = [f for f in os.listdir(cdir) if f.startswith("bridge_") and f.endswith(".npy")]
            co_vals = []
            cross_vals = []
            bd_count = 0
            total = 0

            for bf in bridge_files:
                B = np.load(os.path.join(cdir, bf))
                result = analyze_bridge(B)
                if result is None:
                    continue
                co_mean, cross_mean, is_bd = result
                co_vals.append(co_mean)
                cross_vals.append(cross_mean)
                if is_bd:
                    bd_count += 1
                total += 1

            if total == 0:
                continue

            overall_co = float(np.mean(co_vals))
            overall_cross = float(np.mean(cross_vals))
            ratio = overall_co / max(overall_cross, 1e-10)
            bd_pct = 100.0 * bd_count / total

            results.append({
                "step": step,
                "bridges": total,
                "co_mean": overall_co,
                "cross_mean": overall_cross,
                "ratio": float(ratio),
                "block_diag_pct": float(bd_pct),
            })

            if si % 10 == 0:
                print(f"  Step {step}: ratio={ratio:.1f}:1, BD={bd_pct:.0f}%")

        return results


def print_summary(results, name):
    """Print key emergence milestones."""
    print(f"\n{'='*60}")
    print(f"  {name}: {len(results)} steps analyzed")
    print(f"{'='*60}")
    print(f"{'Step':>6} | {'Bridges':>7} | {'Co-planar':>10} | {'Cross-plan':>10} | {'Ratio':>10} | {'BD%':>5}")
    print(f"{'------':>6}-+-{'-------':>7}-+-{'----------':>10}-+-{'----------':>10}-+-{'----------':>10}-+-{'-----':>5}")

    milestones = [0, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 12000, 12900]
    for target in milestones:
        matches = [r for r in results if r["step"] == target]
        if matches:
            r = matches[0]
            print(f"{r['step']:6d} | {r['bridges']:7d} | {r['co_mean']:10.6f} | {r['cross_mean']:10.6f} | {r['ratio']:8.1f}:1 | {r['block_diag_pct']:5.0f}%")


if __name__ == "__main__":
    experiments = {
        "exp3": {
            "dir": "results/exp3",
            "type": "flat",
            "model": "Qwen 7B",
            "label": "Qwen 7B Cybernetic",
        },
        "exp3_tinyllama": {
            "dir": "results/exp3_tinyllama",
            "type": "flat",
            "model": "TinyLlama 1.1B",
            "label": "TinyLlama 1.1B Cybernetic",
        },
        "C-001": {
            "dir": "results/corpus-baselines/C-001-identity-default",
            "type": "flat",
            "model": "TinyLlama 1.1B",
            "label": "C-001 Identity Init Cybernetic",
        },
        "C-002": {
            "dir": "results/corpus-baselines/C-002-geometric-default",
            "type": "flat",
            "model": "TinyLlama 1.1B",
            "label": "C-002 Geometric Init Cybernetic",
        },
        "C-003": {
            "dir": "results/corpus-baselines/C-003-corpus-coupled-default",
            "type": "flat",
            "model": "TinyLlama 1.1B",
            "label": "C-003 Corpus-Coupled Init Cybernetic",
        },
    }

    # Allow filtering by command line
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(experiments.keys())

    all_results = {}
    for name in targets:
        if name not in experiments:
            print(f"Unknown experiment: {name}")
            continue
        exp = experiments[name]
        print(f"\n--- Analyzing {exp['label']} ---")
        if not os.path.exists(exp["dir"]):
            print(f"  Directory not found: {exp['dir']}")
            continue
        results = analyze_experiment(exp["dir"], exp["type"])
        all_results[name] = results
        print_summary(results, exp["label"])

        # Save individual results
        outfile = f"results/{name}_temporal_emergence.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {outfile}")

    print("\n\nDone.")
