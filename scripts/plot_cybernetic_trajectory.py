#!/usr/bin/env python
"""
Paper 3, Graphic 1: Cybernetic Control Trajectory
4-panel figure showing the closed-loop steersman's behavior over training steps.
Data source: results/exp3_test/results.json (feedback_log)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────
CUBIC  = '#3D3D6B'
FCC    = '#B34444'
GOLD   = '#C49A2A'
TEAL   = '#2A7A7A'
GREY   = '#666666'

# ── Load data ────────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent
data = json.load(open(root / 'results' / 'exp3_test' / 'results.json'))

feedback = data['feedback_log']
checkpoints = data['checkpoints']

steps     = [e['step'] for e in feedback]
fiedler   = [e['fiedler_mean'] for e in feedback]
spec_wt   = [e['spectral_weight'] for e in feedback]
co_cross  = [e['co_cross_ratio'] for e in feedback]
contr_wt  = [e['contrastive_weight'] for e in feedback]
deviation = [e['deviation_mean'] for e in feedback]
bridge_lr = [e['bridge_lr_scale'] for e in feedback]

# Loss components from checkpoints (steps 25, 50)
ck_steps  = [c['step'] for c in checkpoints]
train_l   = [c['train_loss'] for c in checkpoints]
val_l     = [c['val_loss'] for c in checkpoints]
contr_l   = [c['contrastive_loss'] for c in checkpoints]
spec_l    = [c['spectral_loss'] for c in checkpoints]

# Replace NaN/Inf for plotting
co_cross_clean = []
for v in co_cross:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        co_cross_clean.append(np.nan)
    else:
        co_cross_clean.append(v)

# ── Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
fig.suptitle('Cybernetic Control Trajectory — Exp 3 (50 steps, 96 adapters)',
             fontsize=14, fontweight='bold', y=0.97)

# Panel 1: Fiedler value + spectral weight
ax1 = axes[0, 0]
ln1 = ax1.plot(steps, fiedler, color=FCC, linewidth=2, marker='o',
               markersize=5, label='Fiedler value (connectivity)')
ax1.fill_between(steps, fiedler, alpha=0.15, color=FCC)
ax1.set_ylabel('Fiedler Value', color=FCC, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=FCC)
ax1.set_xlabel('Training Step')
ax1.set_title('Connectivity Sensor + Spectral Weight', fontweight='bold')

ax1b = ax1.twinx()
ln2 = ax1b.plot(steps, spec_wt, color=CUBIC, linewidth=2, linestyle='--',
                marker='s', markersize=4, label='Spectral weight (control)')
ax1b.set_ylabel('Spectral Weight', color=CUBIC, fontweight='bold')
ax1b.tick_params(axis='y', labelcolor=CUBIC)

lines = ln1 + ln2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Panel 2: Co/Cross ratio + contrastive weight
ax2 = axes[0, 1]
# Filter NaN for plotting
valid_mask = [not np.isnan(v) if isinstance(v, float) else True for v in co_cross_clean]
valid_steps = [s for s, m in zip(steps, valid_mask) if m]
valid_cc = [v for v, m in zip(co_cross_clean, valid_mask) if m]

ln3 = ax2.plot(valid_steps, valid_cc, color=FCC, linewidth=2, marker='o',
               markersize=5, label='Co/Cross ratio (directionality)')
ax2.fill_between(valid_steps, valid_cc, alpha=0.15, color=FCC)
ax2.set_ylabel('Co/Cross Ratio', color=FCC, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=FCC)
ax2.set_xlabel('Training Step')
ax2.set_title('Directionality Sensor + Contrastive Weight', fontweight='bold')

ax2b = ax2.twinx()
ln4 = ax2b.plot(steps, contr_wt, color=CUBIC, linewidth=2, linestyle='--',
                marker='s', markersize=4, label='Contrastive weight (control)')
ax2b.set_ylabel('Contrastive Weight', color=CUBIC, fontweight='bold')
ax2b.tick_params(axis='y', labelcolor=CUBIC)

lines2 = ln3 + ln4
labels2 = [l.get_label() for l in lines2]
ax2.legend(lines2, labels2, loc='center right', fontsize=8, framealpha=0.9)
ax2.grid(True, alpha=0.3)

# Panel 3: Deviation + bridge LR
ax3 = axes[1, 0]
ln5 = ax3.plot(steps, deviation, color=FCC, linewidth=2, marker='o',
               markersize=5, label='Deviation (stability)')
ax3.fill_between(steps, deviation, alpha=0.15, color=FCC)
ax3.set_ylabel('Deviation', color=FCC, fontweight='bold')
ax3.tick_params(axis='y', labelcolor=FCC)
ax3.set_xlabel('Training Step')
ax3.set_title('Stability Sensor + Bridge LR Scale', fontweight='bold')

ax3b = ax3.twinx()
ln6 = ax3b.plot(steps, bridge_lr, color=CUBIC, linewidth=2, linestyle='--',
                marker='s', markersize=4, label='Bridge LR scale (control)')
ax3b.set_ylabel('Bridge LR Scale', color=CUBIC, fontweight='bold')
ax3b.tick_params(axis='y', labelcolor=CUBIC)

lines3 = ln5 + ln6
labels3 = [l.get_label() for l in lines3]
ax3.legend(lines3, labels3, loc='center right', fontsize=8, framealpha=0.9)
ax3.grid(True, alpha=0.3)

# Panel 4: Loss components
ax4 = axes[1, 1]
ax4.plot(ck_steps, train_l, color=FCC, linewidth=2.5, marker='o',
         markersize=6, label='Train loss')
ax4.plot(ck_steps, val_l, color=CUBIC, linewidth=2.5, marker='s',
         markersize=6, label='Val loss')
ax4.plot(ck_steps, [abs(c) for c in contr_l], color=TEAL, linewidth=2,
         marker='^', markersize=5, label='|Contrastive loss|')
ax4.plot(ck_steps, spec_l, color=GOLD, linewidth=2, marker='D',
         markersize=5, label='Spectral loss')
ax4.set_xlabel('Training Step')
ax4.set_ylabel('Loss', fontweight='bold')
ax4.set_title('Loss Components', fontweight='bold')
ax4.set_yscale('log')
ax4.legend(fontsize=8, framealpha=0.9)
ax4.grid(True, alpha=0.3, which='both')

# ── Annotations ──────────────────────────────────────────────────────────
# Mark the control feedback direction in panels 1-3
for ax in [ax1, ax2, ax3]:
    ax.annotate('', xy=(45, ax.get_ylim()[0]),
                xytext=(5, ax.get_ylim()[0]),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))

# Add text annotation for the key finding
ax4.annotate('Train loss: 5.35 → 0.48\n(91% reduction in 50 steps)',
             xy=(37, 1.0), fontsize=8, fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=GREY, alpha=0.8))

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = root / 'assets' / 'paper3' / 'cybernetic_trajectory.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
