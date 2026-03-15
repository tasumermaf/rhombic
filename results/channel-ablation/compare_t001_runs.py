"""Compare T-001 run 1 (partial, 2700 steps) vs T-001r2 (in progress).

Plots overlaid trajectories for Fiedler, co/cross ratio, and loss.
Used for reproducibility verification and to project r2 final metrics.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
r1_path = Path(r"C:\falco\rhombic\results\T-001-full\results.json")
r2_path = Path(r"C:\falco\rhombic\results\T-001-full-r2\results.json")

with open(r1_path) as f:
    r1 = json.load(f)
with open(r2_path) as f:
    r2 = json.load(f)

r1_fb = r1["feedback_log"]
r2_fb = r2["feedback_log"]

r1_steps = [e["step"] for e in r1_fb]
r2_steps = [e["step"] for e in r2_fb]

r1_fiedler = [e.get("fiedler_mean", 0) for e in r1_fb]
r2_fiedler = [e.get("fiedler_mean", 0) for e in r2_fb]

r1_cocross = [e.get("co_cross_ratio", None) for e in r1_fb]
r2_cocross = [e.get("co_cross_ratio", None) for e in r2_fb]

# Filter None values for co/cross
r1_cc_valid = [(s, c) for s, c in zip(r1_steps, r1_cocross) if c and c > 0]
r2_cc_valid = [(s, c) for s, c in zip(r2_steps, r2_cocross) if c and c > 0]

# Colors
C1 = '#E91E63'  # run 1 (pink)
C2 = '#2196F3'  # run 2 (blue)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Fiedler
ax = axes[0]
ax.plot(r1_steps, r1_fiedler, color=C1, lw=2, label=f'T-001r1 (seed 42, {len(r1_fb)} entries)', marker='o', ms=3)
ax.plot(r2_steps, r2_fiedler, color=C2, lw=2, label=f'T-001r2 (seed 42, {len(r2_fb)} entries)', marker='s', ms=3)
ax.set_yscale('log')
ax.set_xlabel('Step')
ax.set_ylabel('Fiedler Eigenvalue (log)')
ax.set_title('Fiedler Convergence: r1 vs r2', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, which='both')

# Panel 2: Co/Cross ratio
ax = axes[1]
if r1_cc_valid:
    s1, c1 = zip(*r1_cc_valid)
    ax.plot(s1, c1, color=C1, lw=2, label='T-001r1', marker='o', ms=3)
if r2_cc_valid:
    s2, c2 = zip(*r2_cc_valid)
    ax.plot(s2, c2, color=C2, lw=2, label='T-001r2', marker='s', ms=3)
ax.set_yscale('log')
ax.set_xlabel('Step')
ax.set_ylabel('Co/Cross Ratio')
ax.set_title('Co/Cross Ratio: r1 vs r2', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, which='both')

# Panel 3: Step-by-step comparison at matching steps
ax = axes[2]
# Find matching steps
common_steps = sorted(set(r1_steps) & set(r2_steps))
if common_steps:
    r1_dict = {e["step"]: e for e in r1_fb}
    r2_dict = {e["step"]: e for e in r2_fb}

    r1_vals = [r1_dict[s].get("fiedler_mean", 0) for s in common_steps]
    r2_vals = [r2_dict[s].get("fiedler_mean", 0) for s in common_steps]

    # Scatter: r1 vs r2 at same step
    ax.scatter(r1_vals, r2_vals, s=80, c=[s for s in common_steps],
              cmap='viridis', edgecolors='black', linewidths=0.5, zorder=5)

    # Perfect agreement line
    lim = [min(min(r1_vals), min(r2_vals)) * 0.5, max(max(r1_vals), max(r2_vals)) * 2]
    ax.plot(lim, lim, 'k--', alpha=0.3, label='Perfect agreement')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('T-001r1 Fiedler')
    ax.set_ylabel('T-001r2 Fiedler')
    ax.set_title('Reproducibility: Same-Step Comparison', fontweight='bold')
    ax.legend(fontsize=9)

    # Compute correlation at matching steps
    if len(common_steps) > 1:
        corr = np.corrcoef(r1_vals, r2_vals)[0, 1]
        max_dev = max(abs(a - b) / max(a, 1e-10) for a, b in zip(r1_vals, r2_vals))
        ax.text(0.05, 0.95, f'r = {corr:.4f}\nmax dev = {max_dev:.1%}\nn = {len(common_steps)} steps',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
else:
    ax.text(0.5, 0.5, 'No matching steps yet', transform=ax.transAxes,
            ha='center', va='center', fontsize=14)

ax.grid(True, alpha=0.2, which='both')

fig.tight_layout(pad=2)
out = Path(r"C:\falco\rhombic\results\channel-ablation\fig_t001_reproducibility.png")
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Saved: {out}")
print(f"\nT-001r1: {len(r1_fb)} entries, steps 0-{r1_steps[-1]}")
print(f"T-001r2: {len(r2_fb)} entries, steps 0-{r2_steps[-1]}")
print(f"Common steps: {common_steps}")
if r1_cc_valid:
    print(f"r1 final co/cross: {r1_cc_valid[-1][1]:,.0f}:1")
if r2_cc_valid:
    print(f"r2 latest co/cross: {r2_cc_valid[-1][1]:,.0f}:1")
