"""PC-001: Post-hoc co/cross recovery for 24C-001.

Computes co-axial vs cross-axial coupling ratio from saved bridge .npy
checkpoint files, filling the CL2 blind spot (co_cross was None for n=24
during training due to n_channels==6 gate in the logging code).
"""

import numpy as np
import json
import os
import glob

RESULTS_DIR = "/home/timm156/rhombic/results/channel-ablation/24C-001"

# 24-cell co-axial pairs (from train_contrastive_bridge.py)
COAXIAL = [
    (0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10),
    (12, 15), (13, 14), (16, 19), (17, 18), (20, 23), (21, 22),
]
COAXIAL_SET = set(COAXIAL)
CROSSAXIAL = [
    (i, j) for i in range(24) for j in range(i + 1, 24)
    if (i, j) not in COAXIAL_SET
]


def compute_co_cross(bridge):
    co_vals = [abs(bridge[i, j]) for i, j in COAXIAL]
    cross_vals = [abs(bridge[i, j]) for i, j in CROSSAXIAL]
    mean_co = float(np.mean(co_vals))
    mean_cross = float(np.mean(cross_vals))
    ratio = mean_co / mean_cross if mean_cross > 1e-12 else float("inf")
    return {"mean_co": mean_co, "mean_cross": mean_cross, "ratio": ratio}


def get_steps():
    pattern = os.path.join(RESULTS_DIR, "bridge_step*_model_layers_0_self_attn_k_proj.npy")
    files = glob.glob(pattern)
    steps = set()
    for f in files:
        name = os.path.basename(f)
        step_str = name.split("_")[1].replace("step", "")
        steps.add(int(step_str))
    final = os.path.join(RESULTS_DIR, "bridge_final_model_layers_0_self_attn_k_proj.npy")
    if os.path.exists(final):
        steps.add(10000)
    return sorted(steps)


def load_bridges(step):
    prefix = "bridge_final" if step == 10000 else "bridge_step{}".format(step)
    bridges = []
    for layer in range(22):
        for proj in ["k_proj", "o_proj", "q_proj", "v_proj"]:
            path = os.path.join(
                RESULTS_DIR,
                "{}_model_layers_{}_self_attn_{}.npy".format(prefix, layer, proj)
            )
            if os.path.exists(path):
                bridges.append(np.load(path))
    return bridges


def main():
    steps = get_steps()
    print("Found {} checkpoint steps: {}..{}".format(len(steps), steps[0], steps[-1]))
    print("Co-axial: {}, Cross-axial: {}".format(len(COAXIAL), len(CROSSAXIAL)))
    print()
    print("{:>6} | {:>12} | {:>10} | {:>12} | {:>9}".format(
        "step", "co_cross", "mean_co", "mean_cross", "n_bridges"))
    print("-" * 65)

    results = []
    for step in steps:
        bridges = load_bridges(step)
        if not bridges:
            print("{:>6} | NO_DATA".format(step))
            continue

        ratios = []
        cos = []
        crosses = []
        for b in bridges:
            r = compute_co_cross(b)
            ratios.append(r["ratio"])
            cos.append(r["mean_co"])
            crosses.append(r["mean_cross"])

        avg_ratio = float(np.mean(ratios))
        avg_co = float(np.mean(cos))
        avg_cross = float(np.mean(crosses))

        print("{:>6} | {:>12.1f} | {:>10.6f} | {:>12.8f} | {:>9}".format(
            step, avg_ratio, avg_co, avg_cross, len(bridges)))

        results.append({
            "step": step,
            "co_cross_ratio": avg_ratio,
            "mean_co": avg_co,
            "mean_cross": avg_cross,
            "n_bridges": len(bridges),
        })

    out_path = os.path.join(RESULTS_DIR, "pc_001_co_cross_recovery.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "PC-001",
            "description": "Post-hoc co/cross recovery for 24C-001",
            "n_coaxial": len(COAXIAL),
            "n_crossaxial": len(CROSSAXIAL),
            "checkpoints": results,
        }, f, indent=2)
    print("\nSaved to", out_path)

    if results:
        peak = max(results, key=lambda r: r["co_cross_ratio"])
        final = results[-1]
        print("\n=== SUMMARY ===")
        print("Peak co/cross: {:.1f}:1 at step {}".format(peak["co_cross_ratio"], peak["step"]))
        print("Final co/cross: {:.1f}:1 at step {}".format(final["co_cross_ratio"], final["step"]))
        over_1k = [r for r in results if r["co_cross_ratio"] > 1000]
        if over_1k:
            print("First >1000:1 at step {}".format(over_1k[0]["step"]))
        over_5k = [r for r in results if r["co_cross_ratio"] > 5000]
        if over_5k:
            print("First >5000:1 at step {}".format(over_5k[0]["step"]))


if __name__ == "__main__":
    main()
