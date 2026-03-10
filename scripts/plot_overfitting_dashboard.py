#!/usr/bin/env python
"""
Paper 3, Graphic 2: Overfitting Detection Dashboard
Two-panel figure:
  Left:  Scatter — deviation vs. generalization gap (r=0.888)
  Right: Time series — phase transition visualization
Data source: results/exp3a-overfit/results.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────
CUBIC  = '#3D3D6B'
FCC    = '#B34444'
GOLD   = '#C49A2A'
TEAL   = '#2A7A7A'
GREY   = '#666666'

# ── Load data ────────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent
data = json.load(open(root / 'results' / 'exp3a-overfit' / 'results.json'))

ckpts = data['checkpoints']

# Extract arrays, skipping step 0 (null train_loss)
steps     = [c['step'] for c in ckpts if c['step'] > 0]
train_l   = [c['train_loss'] for c in ckpts if c['step'] > 0]
val_l     = [c['val_loss'] for c in ckpts if c['step'] > 0]
gap       = [c['train_val_gap'] for c in ckpts if c['step'] > 0]
deviation = [c['deviation_mean'] for c in ckpts if c['step'] > 0]
fiedler   = [c['fiedler_mean'] for c in ckpts if c['step'] > 0]

# ── Compute correlation ─────────────────────────────────────────────────
dev_arr = np.array(deviation)
gap_arr = np.array(gap)
r, p = stats.pearsonr(dev_arr, gap_arr)
slope, intercept = np.polyfit(dev_arr, gap_arr, 1)
print(f'Pearson r = {r:.3f}, p = {p:.2e}')

# ── Identify phase transition ───────────────────────────────────────────
# The phase transition is where val_loss begins rising while train_loss
# keeps falling. Find the first step where val_loss exceeds its minimum
# by a significant margin AND the gap is positive and growing.
val_arr = np.array(val_l)
train_arr = np.array(train_l)
val_min_idx = np.argmin(val_arr)
# Look for where val_loss first exceeds 1.5x its minimum
for ti in range(val_min_idx, len(val_arr)):
    if val_arr[ti] > val_arr[val_min_idx] * 1.5 and gap_arr[ti] > 0.2:
        transition_idx = ti
        break
else:
    transition_idx = val_min_idx + 3  # fallback
transition_step = steps[transition_idx]
print(f'Phase transition at step {transition_step}')

# ── Figure ───────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
fig.suptitle('Overfitting Detection via Lattice Deviation — Exp 3a',
             fontsize=14, fontweight='bold', y=0.97)

# ── Panel 1: Scatter — deviation vs generalization gap ───────────────────
scatter = ax1.scatter(dev_arr, gap_arr, c=steps, cmap='coolwarm',
                      s=40, alpha=0.8, edgecolors='white', linewidths=0.5,
                      zorder=3)
cbar = plt.colorbar(scatter, ax=ax1, label='Training Step', pad=0.02)
cbar.ax.tick_params(labelsize=8)

# Regression line
x_fit = np.linspace(dev_arr.min(), dev_arr.max(), 100)
y_fit = slope * x_fit + intercept
ax1.plot(x_fit, y_fit, color=FCC, linewidth=2, linestyle='--',
         label=f'r = {r:.3f} (p < {p:.0e})', zorder=2)

# Mark the phase transition point
ax1.scatter([dev_arr[transition_idx]], [gap_arr[transition_idx]],
            s=200, facecolors='none', edgecolors=GOLD, linewidths=2.5,
            zorder=5, label=f'Phase transition (step {transition_step})')

ax1.set_xlabel('Lattice Deviation (mean)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Generalization Gap (val - train)', fontsize=11, fontweight='bold')
ax1.set_title('Deviation Predicts Overfitting', fontweight='bold')
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Annotate the three phases
ax1.annotate('Healthy\ntraining',
             xy=(dev_arr[2], gap_arr[2]),
             xytext=(dev_arr[2] + 0.01, gap_arr[2] + 0.3),
             fontsize=8, fontstyle='italic', color=TEAL,
             arrowprops=dict(arrowstyle='->', color=TEAL, lw=1),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor=TEAL, alpha=0.8))

ax1.annotate('Saturated\noverfitting',
             xy=(dev_arr[-5], gap_arr[-5]),
             xytext=(dev_arr[-5] - 0.04, gap_arr[-5] - 0.25),
             fontsize=8, fontstyle='italic', color=FCC,
             arrowprops=dict(arrowstyle='->', color=FCC, lw=1),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor=FCC, alpha=0.8))

# ── Panel 2: Time series — phase transition ─────────────────────────────
ax2.plot(steps, train_l, color=CUBIC, linewidth=2, label='Train loss')
ax2.plot(steps, val_l, color=FCC, linewidth=2, label='Val loss')
ax2.fill_between(steps, train_l, val_l, alpha=0.12, color=FCC,
                 label='Generalization gap')

ax2_dev = ax2.twinx()
ax2_dev.plot(steps, deviation, color=GOLD, linewidth=2, linestyle='--',
             label='Lattice deviation')
ax2_dev.set_ylabel('Lattice Deviation', color=GOLD, fontweight='bold',
                   fontsize=10)
ax2_dev.tick_params(axis='y', labelcolor=GOLD)

# Mark the phase transition
ax2.axvline(x=transition_step, color=GREY, linewidth=1.5, linestyle=':',
            alpha=0.7)
ax2.annotate(f'Phase transition\nstep {transition_step}',
             xy=(transition_step, val_l[transition_idx]),
             xytext=(transition_step + 800, val_l[transition_idx] * 0.6),
             fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                       edgecolor=GOLD, alpha=0.9))

# Mark the three regimes based on the detected transition
regime1_end = max(200, transition_step - 200)
regime2_end = transition_step + 300
ax2.axvspan(0, regime1_end, alpha=0.05, color=TEAL, zorder=0)
ax2.axvspan(regime1_end, regime2_end, alpha=0.05, color=GOLD, zorder=0)
ax2.axvspan(regime2_end, steps[-1], alpha=0.05, color=FCC, zorder=0)

ylim_top = ax2.get_ylim()[1]
ax2.text(regime1_end / 2, ylim_top * 0.92, 'Learning', fontsize=8,
         ha='center', fontstyle='italic', color=TEAL)
ax2.text((regime1_end + regime2_end) / 2, ylim_top * 0.92, 'Transition',
         fontsize=8, ha='center', fontstyle='italic', color=GOLD)
ax2.text((regime2_end + steps[-1]) / 2, ylim_top * 0.92, 'Overfitting',
         fontsize=8, ha='center', fontstyle='italic', color=FCC)

ax2.set_xlabel('Training Step', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Phase Transition in Overfitting', fontweight='bold')

# Combined legend
lines2, labels2 = ax2.get_legend_handles_labels()
lines2b, labels2b = ax2_dev.get_legend_handles_labels()
ax2.legend(lines2 + lines2b, labels2 + labels2b, fontsize=8, loc='center right',
           framealpha=0.9)
ax2.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = root / 'assets' / 'paper3' / 'overfitting_dashboard.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
