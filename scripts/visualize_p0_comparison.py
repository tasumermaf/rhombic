#!/usr/bin/env python
"""P0 Definitive Comparison: Standard LoRA vs TeLoRA

Generates a multi-panel figure showing:
  1. Loss comparison (LoRA vs TeLoRA over training steps)
  2. Bridge heatmap evolution (identity -> BD emergence, TeLoRA only)
  3. Fiedler connectivity emergence (TeLoRA only)
  4. Bridge deviation from identity (TeLoRA only)

This is the "blow minds" visualization: watch 4D geometry form
inside a neural network in real-time.

Usage:
  python scripts/visualize_p0_comparison.py --results-dir results/p0-proof
  python scripts/visualize_p0_comparison.py --results-dir results/p0-proof --output figures/p0_proof.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ── Palette (from rhombic project identity) ──────────────────────
CUBIC = '#3D3D6B'   # Standard LoRA color
FCC   = '#B34444'   # TeLoRA color
GOLD  = '#C49A2A'   # Accent
TEAL  = '#2A7A7A'   # Accent
BG    = '#FAFAF7'   # Background


def load_results(results_dir: Path) -> tuple[dict, dict]:
    """Load both experiment results from the P0 proof directory."""
    lora_path = results_dir / "standard_lora_r24" / "results.json"
    telora_path = results_dir / "rhombi_learnable_r24" / "results.json"

    if not lora_path.exists():
        print(f"Missing: {lora_path}")
        sys.exit(1)
    if not telora_path.exists():
        print(f"Missing: {telora_path}")
        sys.exit(1)

    with open(lora_path) as f:
        lora = json.load(f)
    with open(telora_path) as f:
        telora = json.load(f)

    return lora, telora


def extract_metrics(data: dict) -> dict:
    """Extract plotting metrics from results.json checkpoints."""
    cps = data["checkpoints"]
    metrics = {
        "steps": [c["step"] for c in cps],
        "loss": [c["train_loss"] for c in cps],
        "wall_time": [c["wall_time"] for c in cps],
    }

    # Bridge metrics (only present for TeLoRA with trainable bridge)
    if cps and cps[0].get("bridge_fiedler"):
        first_key = list(cps[0]["bridge_fiedler"].keys())[0]
        metrics["fiedler"] = [c["bridge_fiedler"][first_key] for c in cps]
        metrics["deviation"] = [c["bridge_deviation"][first_key] for c in cps]

        # Extract bridge matrices for heatmap evolution
        metrics["bridges"] = []
        for c in cps:
            if c.get("bridge_matrices") and first_key in c["bridge_matrices"]:
                metrics["bridges"].append(np.array(c["bridge_matrices"][first_key]))

        # Compute co/cross ratio from bridge matrices
        co_cross = []
        for B in metrics["bridges"]:
            n = B.shape[0]
            half = n // 2
            # Block-diagonal: within blocks (0:half, 0:half) and (half:, half:)
            co_vals = np.abs(np.concatenate([
                B[:half, :half][np.triu_indices(half, 1)],
                B[half:, half:][np.triu_indices(half - (n - 2*half), 1)]
            ]))
            # Cross-block: between blocks
            cross_vals = np.abs(B[:half, half:].flatten())
            co_mean = co_vals.mean() if len(co_vals) > 0 else 0
            cross_mean = cross_vals.mean() if len(cross_vals) > 0 else 1e-10
            co_cross.append(co_mean / max(cross_mean, 1e-10))
        metrics["co_cross"] = co_cross

    return metrics


def plot_comparison(lora_metrics: dict, telora_metrics: dict,
                    output_path: Path | None = None):
    """Generate the definitive comparison figure."""
    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.08, right=0.95, top=0.93, bottom=0.05)

    # ── Panel 1: Loss Comparison (top left, spans 2 cols) ──
    ax_loss = fig.add_subplot(gs[0, :2])
    ax_loss.set_facecolor(BG)
    ax_loss.plot(lora_metrics["steps"], lora_metrics["loss"],
                 color=CUBIC, linewidth=2.5, label="Standard LoRA (r=24)",
                 alpha=0.9)
    ax_loss.plot(telora_metrics["steps"], telora_metrics["loss"],
                 color=FCC, linewidth=2.5, label="TeLoRA (r=24, n=6)",
                 alpha=0.9)
    ax_loss.set_xlabel("Training Step", fontsize=11)
    ax_loss.set_ylabel("Training Loss", fontsize=11)
    ax_loss.set_title("Loss Convergence", fontsize=13, fontweight='bold')
    ax_loss.legend(fontsize=10, framealpha=0.9)
    ax_loss.grid(True, alpha=0.3)

    # Final loss annotation
    lora_final = lora_metrics["loss"][-1]
    telora_final = telora_metrics["loss"][-1]
    improvement = (lora_final - telora_final) / lora_final * 100
    ax_loss.annotate(
        f"TeLoRA: {telora_final:.4f}\nLoRA: {lora_final:.4f}\n"
        f"{'%.1f' % abs(improvement)}% {'better' if improvement > 0 else 'worse'}",
        xy=(0.98, 0.95), xycoords='axes fraction',
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )

    # ── Panel 2: Wall-clock comparison (top right) ──
    ax_time = fig.add_subplot(gs[0, 2])
    ax_time.set_facecolor(BG)
    lora_time = lora_metrics["wall_time"][-1]
    telora_time = telora_metrics["wall_time"][-1]
    bars = ax_time.bar(["LoRA", "TeLoRA"],
                       [lora_time / 60, telora_time / 60],
                       color=[CUBIC, FCC], width=0.5, alpha=0.85)
    ax_time.set_ylabel("Wall Time (minutes)", fontsize=11)
    ax_time.set_title("Training Time", fontsize=13, fontweight='bold')
    for bar, t in zip(bars, [lora_time, telora_time]):
        ax_time.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{t/60:.1f}m", ha='center', va='bottom', fontsize=10)

    # ── Panel 3: Bridge Heatmap Evolution (middle row, 3 snapshots) ──
    if "bridges" in telora_metrics and len(telora_metrics["bridges"]) >= 3:
        bridges = telora_metrics["bridges"]
        steps = telora_metrics["steps"]
        # Pick 3 snapshots: early, middle, final
        n_cp = len(bridges)
        indices = [0, n_cp // 2, n_cp - 1]

        for i, idx in enumerate(indices):
            ax_heat = fig.add_subplot(gs[1, i])
            B = bridges[idx]
            vmax = max(abs(B.min()), abs(B.max()), 0.01)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax_heat.imshow(B, cmap='RdBu_r', norm=norm, aspect='equal')
            ax_heat.set_title(f"Step {steps[idx]}", fontsize=11, fontweight='bold')

            # Mark block-diagonal structure
            n = B.shape[0]
            half = n // 2
            ax_heat.axhline(half - 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
            ax_heat.axvline(half - 0.5, color='black', linewidth=0.5, linestyle='--', alpha=0.5)

            if i == 0:
                ax_heat.set_ylabel("Bridge Channel", fontsize=10)
            ax_heat.set_xlabel("Bridge Channel", fontsize=10)

        # Shared colorbar
        fig.colorbar(im, ax=[fig.axes[j] for j in range(2, 5)],
                     shrink=0.8, label="Bridge Weight")

    # ── Panel 4: Fiedler Convergence (bottom left) ──
    if "fiedler" in telora_metrics:
        ax_fiedler = fig.add_subplot(gs[2, 0])
        ax_fiedler.set_facecolor(BG)
        ax_fiedler.plot(telora_metrics["steps"], telora_metrics["fiedler"],
                        color=FCC, linewidth=2, marker='o', markersize=3)
        ax_fiedler.set_xlabel("Training Step", fontsize=11)
        ax_fiedler.set_ylabel("Fiedler Value (lambda_2)", fontsize=11)
        ax_fiedler.set_title("Algebraic Connectivity", fontsize=13, fontweight='bold')
        ax_fiedler.grid(True, alpha=0.3)
        # Log scale if values span orders of magnitude
        fiedler_range = max(telora_metrics["fiedler"]) / max(min(telora_metrics["fiedler"]), 1e-10)
        if fiedler_range > 100:
            ax_fiedler.set_yscale('log')

    # ── Panel 5: Co/Cross Ratio (bottom middle) ──
    if "co_cross" in telora_metrics:
        ax_cc = fig.add_subplot(gs[2, 1])
        ax_cc.set_facecolor(BG)
        ax_cc.plot(telora_metrics["steps"], telora_metrics["co_cross"],
                   color=GOLD, linewidth=2, marker='s', markersize=3)
        ax_cc.set_xlabel("Training Step", fontsize=11)
        ax_cc.set_ylabel("Co-planar / Cross-planar Ratio", fontsize=11)
        ax_cc.set_title("Block-Diagonal Emergence", fontsize=13, fontweight='bold')
        ax_cc.grid(True, alpha=0.3)
        ax_cc.set_yscale('log')
        # Annotate final ratio
        final_cc = telora_metrics["co_cross"][-1]
        ax_cc.annotate(f"Final: {final_cc:.0f}:1",
                       xy=(0.95, 0.95), xycoords='axes fraction',
                       ha='right', va='top', fontsize=11, fontweight='bold',
                       color=GOLD)

    # ── Panel 6: Deviation from Identity (bottom right) ──
    if "deviation" in telora_metrics:
        ax_dev = fig.add_subplot(gs[2, 2])
        ax_dev.set_facecolor(BG)
        ax_dev.plot(telora_metrics["steps"], telora_metrics["deviation"],
                    color=TEAL, linewidth=2, marker='^', markersize=3)
        ax_dev.set_xlabel("Training Step", fontsize=11)
        ax_dev.set_ylabel("Frobenius Deviation", fontsize=11)
        ax_dev.set_title("Deviation from Identity", fontsize=13, fontweight='bold')
        ax_dev.grid(True, alpha=0.3)

    # ── Title ──
    fig.suptitle(
        "P0 Proof: Standard LoRA vs TeLoRA (Topology-Enhanced LoRA)\n"
        "Qwen2.5-1.5B-Instruct | Alpaca-cleaned | rank=24 | seed=42",
        fontsize=15, fontweight='bold', y=0.98
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"Saved: {output_path}")

        # Also save as PDF for paper
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor=BG)
        print(f"Saved: {pdf_path}")
    else:
        plt.show()

    plt.close(fig)


def print_summary(lora_metrics: dict, telora_metrics: dict):
    """Print a text summary of the comparison."""
    print("\n" + "=" * 60)
    print("P0 PROOF: STANDARD LoRA vs TeLoRA — SUMMARY")
    print("=" * 60)

    lora_final = lora_metrics["loss"][-1]
    telora_final = telora_metrics["loss"][-1]
    improvement = (lora_final - telora_final) / lora_final * 100

    print(f"\nFinal Training Loss:")
    print(f"  Standard LoRA (r=24):  {lora_final:.6f}")
    print(f"  TeLoRA (r=24, n=6):    {telora_final:.6f}")
    print(f"  Difference:            {abs(improvement):.2f}% {'better' if improvement > 0 else 'worse'}")

    lora_time = lora_metrics["wall_time"][-1]
    telora_time = telora_metrics["wall_time"][-1]
    time_ratio = telora_time / lora_time

    print(f"\nTraining Time:")
    print(f"  Standard LoRA:  {lora_time/60:.1f} minutes")
    print(f"  TeLoRA:         {telora_time/60:.1f} minutes")
    print(f"  Overhead:       {(time_ratio - 1)*100:.1f}%")

    if "fiedler" in telora_metrics:
        print(f"\nTeLoRA Bridge Metrics (final):")
        print(f"  Fiedler (connectivity):  {telora_metrics['fiedler'][-1]:.6f}")
        print(f"  Deviation from identity: {telora_metrics['deviation'][-1]:.6f}")
        if "co_cross" in telora_metrics:
            print(f"  Co/Cross ratio:          {telora_metrics['co_cross'][-1]:.1f}:1")

    # Parameter counts
    lora_params = lora_metrics.get("trainable_params", "?")
    telora_params = telora_metrics.get("trainable_params", "?")
    print(f"\nTrainable Parameters:")
    print(f"  Standard LoRA:  {lora_params:,}" if isinstance(lora_params, int) else f"  Standard LoRA:  {lora_params}")
    print(f"  TeLoRA:         {telora_params:,}" if isinstance(telora_params, int) else f"  TeLoRA:         {telora_params}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="P0 Proof: LoRA vs TeLoRA comparison visualization"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/p0-proof",
        help="Directory containing standard_lora_r24/ and rhombi_learnable_r24/",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output image path. Default: results-dir/p0_proof_comparison.png",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    lora_data, telora_data = load_results(results_dir)

    lora_metrics = extract_metrics(lora_data)
    telora_metrics = extract_metrics(telora_data)

    # Add param counts
    lora_metrics["trainable_params"] = lora_data.get("trainable_params")
    telora_metrics["trainable_params"] = telora_data.get("trainable_params")

    print_summary(lora_metrics, telora_metrics)

    output_path = Path(args.output) if args.output else results_dir / "p0_proof_comparison.png"
    plot_comparison(lora_metrics, telora_metrics, output_path)


if __name__ == "__main__":
    main()
