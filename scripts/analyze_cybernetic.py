"""Post-hoc analysis of Experiment 3 cybernetic training results.

Reads results.json from a cybernetic training run and produces:
1. Control trajectory table (text)
2. Matplotlib plots if available:
   - Fiedler value + spectral weight over time
   - Co/cross ratio + contrastive weight over time
   - Deviation + bridge LR over time
   - Loss components over time
3. Phase detection: identifies attractor convergence points

Usage:
  python scripts/analyze_cybernetic.py results/exp3
  python scripts/analyze_cybernetic.py results/exp3 --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_results(path: Path) -> dict:
    with open(path / "results.json") as f:
        return json.load(f)


def print_trajectory(results: dict):
    """Print the full control trajectory table."""
    feedback = results.get("feedback_log", [])
    checkpoints = results.get("checkpoints", [])

    print(f"\n{'='*80}")
    print(f"CYBERNETIC TRAINING ANALYSIS — {results.get('experiment', '?')}")
    print(f"{'='*80}")

    config = results.get("config", {})
    print(f"Model:      {config.get('model_name', '?')}")
    print(f"Steps:      {config.get('max_steps', '?')}")
    print(f"Adapters:   {results.get('trainable_params', '?'):,} trainable")
    print(f"Feedback cycles: {len(feedback)}")

    if not feedback:
        print("No feedback data found.")
        return

    # Trajectory table
    print(f"\n{'Step':>6} | {'Fiedler':>8} | {'F_trend':>8} | "
          f"{'Co/Cross':>8} | {'CC_trend':>8} | "
          f"{'Dev':>8} | {'D_trend':>8} | "
          f"{'C_wt':>6} | {'S_wt':>6} | {'S_tgt':>6} | {'B_lr':>5}")
    print(f"{'-'*100}")

    for entry in feedback:
        co = f"{entry['co_cross_ratio']:.3f}" if entry.get('co_cross_ratio') else "  N/A "
        print(
            f"{entry['step']:>6} | "
            f"{entry['fiedler_mean']:>8.5f} | "
            f"{entry['fiedler_trend']:>+8.5f} | "
            f"{co:>8} | "
            f"{entry['co_cross_trend']:>+8.5f} | "
            f"{entry['deviation_mean']:>8.5f} | "
            f"{entry['deviation_trend']:>+8.5f} | "
            f"{entry['contrastive_weight']:>6.3f} | "
            f"{entry['spectral_weight']:>6.3f} | "
            f"{entry['spectral_target']:>6.3f} | "
            f"{entry['bridge_lr_scale']:>5.2f}"
        )

    # Control signal history
    print(f"\n{'='*80}")
    print(f"CONTROL SIGNAL LOG")
    print(f"{'='*80}")
    for entry in feedback:
        signals = entry.get("control_signals", {})
        if signals:
            print(f"\n  Step {entry['step']}:")
            for law, msg in signals.items():
                print(f"    {law}: {msg}")

    # Phase detection: find where metrics stabilize
    if len(feedback) > 5:
        fiedlers = [e["fiedler_mean"] for e in feedback]
        # Moving window variance
        window = 5
        variances = []
        for i in range(window, len(fiedlers)):
            chunk = fiedlers[i-window:i]
            variances.append(np.var(chunk))

        if variances:
            min_var_idx = np.argmin(variances) + window
            print(f"\n{'='*80}")
            print(f"ATTRACTOR DETECTION")
            print(f"{'='*80}")
            print(f"Minimum Fiedler variance at step {feedback[min_var_idx]['step']}")
            print(f"  Fiedler: {feedback[min_var_idx]['fiedler_mean']:.5f}")
            co = feedback[min_var_idx].get('co_cross_ratio')
            if co:
                print(f"  Co/Cross: {co:.3f}")
            print(f"  Deviation: {feedback[min_var_idx]['deviation_mean']:.5f}")

    # Loss trajectory
    if checkpoints:
        print(f"\n{'='*80}")
        print(f"LOSS TRAJECTORY")
        print(f"{'='*80}")
        print(f"{'Step':>6} | {'LM_train':>8} | {'LM_val':>8} | "
              f"{'C_loss':>8} | {'S_loss':>8}")
        print(f"{'-'*50}")
        for cp in checkpoints:
            print(
                f"{cp['step']:>6} | "
                f"{cp['train_loss']:>8.4f} | "
                f"{cp.get('val_loss', 0):>8.4f} | "
                f"{cp.get('contrastive_loss', 0):>8.5f} | "
                f"{cp.get('spectral_loss', 0):>8.5f}"
            )


def plot_trajectory(results: dict, output_dir: Path):
    """Generate matplotlib plots of the control trajectory."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    feedback = results.get("feedback_log", [])
    checkpoints = results.get("checkpoints", [])

    if not feedback:
        return

    steps = [e["step"] for e in feedback]
    fiedlers = [e["fiedler_mean"] for e in feedback]
    deviations = [e["deviation_mean"] for e in feedback]
    c_weights = [e["contrastive_weight"] for e in feedback]
    s_weights = [e["spectral_weight"] for e in feedback]
    s_targets = [e["spectral_target"] for e in feedback]
    b_lrs = [e["bridge_lr_scale"] for e in feedback]

    co_cross_steps = []
    co_cross_vals = []
    for e in feedback:
        if e.get("co_cross_ratio") is not None:
            co_cross_steps.append(e["step"])
            co_cross_vals.append(e["co_cross_ratio"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cybernetic Bridge Training — Control Trajectory", fontsize=14)

    # Fiedler + spectral weight
    ax1 = axes[0, 0]
    color1, color2 = "#B34444", "#3D3D6B"
    ax1.plot(steps, fiedlers, color=color1, linewidth=2, label="Fiedler (connectivity)")
    ax1.set_ylabel("Fiedler value", color=color1)
    ax1_r = ax1.twinx()
    ax1_r.plot(steps, s_weights, color=color2, linewidth=1, alpha=0.7, label="Spectral weight")
    ax1_r.plot(steps, s_targets, color=color2, linewidth=1, linestyle="--", alpha=0.5, label="Spectral target")
    ax1_r.set_ylabel("Weight / Target", color=color2)
    ax1.set_title("Control Law 1: CONNECTIVITY")
    ax1.legend(loc="upper left")
    ax1_r.legend(loc="upper right")

    # Co/cross + contrastive weight
    ax2 = axes[0, 1]
    if co_cross_vals:
        ax2.plot(co_cross_steps, co_cross_vals, color=color1, linewidth=2, label="Co/Cross ratio")
        ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Co/Cross ratio", color=color1)
    ax2_r = ax2.twinx()
    ax2_r.plot(steps, c_weights, color=color2, linewidth=1, alpha=0.7, label="Contrastive weight")
    ax2_r.set_ylabel("Weight", color=color2)
    ax2.set_title("Control Law 2: DIRECTIONALITY")
    ax2.legend(loc="upper left")
    ax2_r.legend(loc="upper right")

    # Deviation + bridge LR
    ax3 = axes[1, 0]
    ax3.plot(steps, deviations, color=color1, linewidth=2, label="Bridge deviation")
    ax3.set_ylabel("||B - I||", color=color1)
    ax3_r = ax3.twinx()
    ax3_r.plot(steps, b_lrs, color=color2, linewidth=1, alpha=0.7, label="Bridge LR scale")
    ax3_r.set_ylabel("LR scale", color=color2)
    ax3.set_title("Control Law 3: STABILITY")
    ax3.set_xlabel("Step")
    ax3.legend(loc="upper left")
    ax3_r.legend(loc="upper right")

    # Loss components
    ax4 = axes[1, 1]
    if checkpoints:
        cp_steps = [c["step"] for c in checkpoints]
        ax4.plot(cp_steps, [c["train_loss"] for c in checkpoints],
                 color=color1, linewidth=2, label="Train LM")
        ax4.plot(cp_steps, [c.get("val_loss", 0) for c in checkpoints],
                 color=color1, linewidth=1, linestyle="--", label="Val LM")
        ax4.plot(cp_steps, [c.get("contrastive_loss", 0) for c in checkpoints],
                 color="#448844", linewidth=1, label="Contrastive")
        ax4.plot(cp_steps, [c.get("spectral_loss", 0) for c in checkpoints],
                 color=color2, linewidth=1, label="Spectral")
    ax4.set_title("Loss Components")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Loss")
    ax4.legend()

    plt.tight_layout()
    plot_path = output_dir / "cybernetic_trajectory.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cybernetic training results"
    )
    parser.add_argument("path", type=str, help="Path to results directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    path = Path(args.path)
    results = load_results(path)
    print_trajectory(results)

    if args.plot:
        plot_trajectory(results, path)


if __name__ == "__main__":
    main()
