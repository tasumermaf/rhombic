#!/usr/bin/env python3
"""Per-layer and per-module bridge structure analysis.

Answers: Do different transformer layers develop different coupling strengths?
Do attention modules (q/k/v/o) differ in their block-diagonal structure?
"""
import os
import re
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict

CO_PLANAR = [(0, 1), (2, 3), (4, 5)]
CO_SET = set(CO_PLANAR)

QWEN_COLOR = '#B34444'
TINY_COLOR = '#3D3D6B'


def analyze_final_bridges(base_dir, label=""):
    """Load all final bridges, group by layer and module."""
    files = [f for f in os.listdir(base_dir) if f.startswith("bridge_final_")]
    by_layer = defaultdict(dict)

    for f in files:
        m = re.match(r'bridge_final_model_layers_(\d+)_self_attn_(\w+)\.npy', f)
        if not m:
            continue
        layer = int(m.group(1))
        module = m.group(2)
        B = np.load(os.path.join(base_dir, f))

        if B.shape[0] < 6:
            continue

        co_mags = []
        cross_mags = []
        for i in range(6):
            for j in range(i + 1, 6):
                mag = abs(B[i, j])
                if (i, j) in CO_SET:
                    co_mags.append(mag)
                else:
                    cross_mags.append(mag)

        by_layer[layer][module] = {
            "co_mean": float(np.mean(co_mags)),
            "cross_mean": float(np.mean(cross_mags)),
            "ratio": float(np.mean(co_mags) / max(np.mean(cross_mags), 1e-10)),
            "diag_mean": float(np.mean(np.diag(B))),
        }

    return by_layer


def print_table(by_layer, label):
    """Print per-layer table."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    header = f"{'Layer':>5} | {'q_proj co':>10} | {'k_proj co':>10} | {'v_proj co':>10} | {'o_proj co':>10} | {'Layer avg':>10}"
    print(header)
    print("-" * len(header))

    module_ratios = defaultdict(list)
    module_co = defaultdict(list)
    layer_data = []

    for layer in sorted(by_layer.keys()):
        modules = by_layer[layer]
        parts = []
        vals = []
        for mod in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if mod in modules:
                co = modules[mod]["co_mean"]
                vals.append(co)
                module_ratios[mod].append(modules[mod]["ratio"])
                module_co[mod].append(co)
                parts.append(f"{co:10.4f}")
            else:
                parts.append(f"{'--':>10}")
        avg = np.mean(vals) if vals else 0
        layer_data.append((layer, avg))
        print(f"{layer:5d} | {' | '.join(parts)} | {avg:10.4f}")

    print()
    print("Module averages:")
    for mod in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if mod in module_co:
            co_arr = module_co[mod]
            ratio_arr = module_ratios[mod]
            print(f"  {mod}: co-planar = {np.mean(co_arr):.4f} +/- {np.std(co_arr):.4f}, "
                  f"ratio = {np.mean(ratio_arr):,.0f}:1 +/- {np.std(ratio_arr):,.0f}")

    return layer_data, module_co


def plot_per_layer(all_data, output_prefix):
    """Plot per-layer coupling strength comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        "Qwen 7B (exp3)": QWEN_COLOR,
        "TinyLlama 1.1B (exp3_tinyllama)": TINY_COLOR,
    }

    for label, layer_data in all_data.items():
        layers = [l for l, _ in layer_data]
        avgs = [a for _, a in layer_data]
        color = colors.get(label, "#666666")
        ax.plot(layers, avgs, color=color, linewidth=1.5, marker="o", markersize=3, label=label)

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Co-planar Coupling Magnitude")
    ax.set_title("Per-Layer Block-Diagonal Coupling Strength")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/{output_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: results/{output_prefix}.png")


def plot_heatmap(base_dir, label, layer_idx, output_prefix):
    """Plot a sample 6x6 bridge matrix as heatmap showing block-diagonal structure."""
    # Find a bridge from a middle layer
    target = f"bridge_final_model_layers_{layer_idx}_self_attn_q_proj.npy"
    fpath = os.path.join(base_dir, target)
    if not os.path.exists(fpath):
        print(f"  Bridge not found: {fpath}")
        return

    B = np.load(fpath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: raw bridge matrix
    im1 = ax1.imshow(B, cmap="RdBu_r", vmin=-np.max(np.abs(B)), vmax=np.max(np.abs(B)))
    ax1.set_title(f"{label}\nLayer {layer_idx} q_proj (raw)")
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Channel")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Annotate co-planar blocks
    for pair in CO_PLANAR:
        i, j = pair
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="green", linewidth=2)
        ax1.add_patch(rect)
        rect2 = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="green", linewidth=2)
        ax1.add_patch(rect2)

    # Right: absolute off-diagonal only
    off_diag = np.abs(B.copy())
    np.fill_diagonal(off_diag, 0)
    im2 = ax2.imshow(off_diag, cmap="hot_r")
    ax2.set_title(f"Off-diagonal magnitudes\n(green = co-planar pairs)")
    ax2.set_xlabel("Channel")
    ax2.set_ylabel("Channel")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    for pair in CO_PLANAR:
        i, j = pair
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="lime", linewidth=2)
        ax2.add_patch(rect)
        rect2 = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="lime", linewidth=2)
        ax2.add_patch(rect2)

    plt.tight_layout()
    plt.savefig(f"results/{output_prefix}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: results/{output_prefix}.png")


if __name__ == "__main__":
    experiments = {
        "Qwen 7B (exp3)": "results/exp3",
        "TinyLlama 1.1B (exp3_tinyllama)": "results/exp3_tinyllama",
    }

    all_layer_data = {}
    for label, base_dir in experiments.items():
        if not os.path.exists(base_dir):
            print(f"Skipping {label}: directory not found")
            continue
        by_layer = analyze_final_bridges(base_dir, label)
        layer_data, module_co = print_table(by_layer, label)
        all_layer_data[label] = layer_data

    # Per-layer comparison plot
    if all_layer_data:
        plot_per_layer(all_layer_data, "fig_per_layer_coupling")

    # Heatmaps — one cybernetic, one non-cybernetic for comparison
    print("\n--- Bridge Heatmaps ---")
    plot_heatmap("results/exp3", "Qwen 7B Cybernetic", 14, "fig_heatmap_cybernetic")

    # Non-cybernetic comparison
    if os.path.exists("results/exp2"):
        plot_heatmap("results/exp2", "Qwen 7B Non-Cybernetic (FCC)", 14, "fig_heatmap_non_cybernetic")
