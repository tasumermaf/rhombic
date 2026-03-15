"""
Generate the definitive channel ablation comparison figure for Paper 3.

Three panels (1x3): Val Loss, Bridge Fiedler, Co-Planar/Cross-Planar Ratio.
Data sources:
  n=3: H-ch3/results.json
  n=4: H-ch4/results_hermes.json
  n=6: H-ch6/metrics_hermes.csv
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
from pathlib import Path

BASE = Path(r"C:\falco\rhombic\results\channel-ablation")

# -- Load n=3 --
with open(BASE / "H-ch3" / "results.json") as f:
    ch3_data = json.load(f)
ch3_steps = [c["step"] for c in ch3_data["checkpoints"]]
ch3_val = [c["val_loss"] for c in ch3_data["checkpoints"]]
ch3_fiedler = [c["fiedler_mean"] for c in ch3_data["checkpoints"]]

# -- Load n=4 --
with open(BASE / "H-ch4" / "results_hermes.json") as f:
    ch4_data = json.load(f)
ch4_steps = [c["step"] for c in ch4_data["checkpoints"]]
ch4_val = [c["val_loss"] for c in ch4_data["checkpoints"]]
ch4_fiedler = [c["fiedler_mean"] for c in ch4_data["checkpoints"]]

# -- Load n=6 --
ch6_steps, ch6_val, ch6_fiedler, ch6_ratio = [], [], [], []
with open(BASE / "H-ch6" / "metrics_hermes.csv") as f:
    lines = f.readlines()
# First line is a count header, skip it; data starts at line 2
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split(",")
    if len(parts) < 7:
        continue
    try:
        step = int(parts[0])
        val_loss = float(parts[1])
        fiedler = float(parts[2])
        ratio = float(parts[3])
        ch6_steps.append(step)
        ch6_val.append(val_loss)
        ch6_fiedler.append(fiedler)
        ch6_ratio.append(ratio)
    except (ValueError, IndexError):
        continue

# -- Colors --
C3 = '#2196F3'   # blue
C4 = '#FF9800'   # orange
C6 = '#B34444'   # FCC red

LW = 1.8
LABEL_SIZE = 14
TITLE_SIZE = 14.5
TICK_SIZE = 11

# -- Figure --
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# ===== Panel 1: Val Loss =====
ax1.plot(ch3_steps, ch3_val, color=C3, lw=LW, label='n = 3', zorder=3)
ax1.plot(ch4_steps, ch4_val, color=C4, lw=LW, label='n = 4', zorder=3)
ax1.plot(ch6_steps, ch6_val, color=C6, lw=LW, label='n = 6', zorder=3)
ax1.set_xlabel('Step', fontsize=LABEL_SIZE)
ax1.set_ylabel('Val Loss', fontsize=LABEL_SIZE)
ax1.set_title('Val Loss Independent of Channel Count',
              fontsize=TITLE_SIZE, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
ax1.tick_params(labelsize=TICK_SIZE)
ax1.set_xlim(0, 10200)

# Convergence reference
final_val = np.mean([ch3_val[-1], ch4_val[-1], ch6_val[-1]])
ax1.axhline(y=final_val, color='gray', ls='--', lw=0.8, alpha=0.5)
ax1.annotate(f'~{final_val:.3f}', xy=(8500, final_val),
             xytext=(0, -16), textcoords='offset points',
             fontsize=10, color='gray', ha='center')
ax1.grid(True, alpha=0.2)
ax1.text(0.02, 0.02, '(a)', transform=ax1.transAxes, fontsize=13,
         fontweight='bold', va='bottom')

# ===== Panel 2: Bridge Fiedler (log scale) =====
ax2.plot(ch3_steps, ch3_fiedler, color=C3, lw=LW, label='n = 3', zorder=3)
ax2.plot(ch4_steps, ch4_fiedler, color=C4, lw=LW, label='n = 4', zorder=3)
ax2.plot(ch6_steps, ch6_fiedler, color=C6, lw=LW, label='n = 6', zorder=3)
ax2.set_yscale('log')
ax2.set_xlabel('Step', fontsize=LABEL_SIZE)
ax2.set_ylabel('Bridge Fiedler (log)', fontsize=LABEL_SIZE)
ax2.set_title('Bridge Fiedler: Structure vs Connectivity',
              fontsize=TITLE_SIZE, fontweight='bold')
ax2.legend(fontsize=11, loc='right', framealpha=0.9)
ax2.tick_params(labelsize=TICK_SIZE)
ax2.set_xlim(0, 10200)
ax2.grid(True, alpha=0.2, which='both')

# Horizontal reference lines with inline labels
ch6_final_fiedler = ch6_fiedler[-1]
ch3_final_fiedler = ch3_fiedler[-1]
ax2.axhline(y=ch6_final_fiedler, color=C6, ls=':', lw=1.0, alpha=0.5)
ax2.axhline(y=ch3_final_fiedler, color=C3, ls=':', lw=1.0, alpha=0.5)

# Regime labels inside the plot area
ax2.text(9800, ch3_final_fiedler * 1.6, 'Connected\nregime',
         fontsize=8.5, color=C3, ha='right', va='bottom',
         fontstyle='italic', alpha=0.8)
ax2.text(9800, ch6_final_fiedler * 0.35, 'Block-diagonal\nregime',
         fontsize=8.5, color=C6, ha='right', va='top',
         fontstyle='italic', alpha=0.8)

# Annotations for contrastive vs spectral -- placed carefully
# "Spectral only" near n=3/4 lines at ~step 4000
ax2.annotate('Spectral only', xy=(4500, ch3_fiedler[44]),
             fontsize=9, color=C3, fontstyle='italic',
             xytext=(-50, 25), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color=C3, lw=0.8))

# "Contrastive ACTIVE" near n=6 line
ax2.annotate('Contrastive ACTIVE', xy=(3500, ch6_fiedler[34]),
             fontsize=9, color=C6, fontweight='bold',
             xytext=(30, 25), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color=C6, lw=0.8))

# Text box -- lower left
textstr = 'Contrastive loss encodes RD\nface-pair geometry.\nActive only for n = 6.'
props = dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', edgecolor=C6, alpha=0.9)
ax2.text(0.03, 0.35, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)
ax2.text(0.02, 0.02, '(b)', transform=ax2.transAxes, fontsize=13,
         fontweight='bold', va='bottom')

# Fill between to shade the divergence
ax2.fill_between([0, 10200], ch6_final_fiedler * 0.5, ch3_final_fiedler * 2,
                 alpha=0.03, color='gray')

# ===== Panel 3: Co-planar/Cross-planar ratio (n=6 only) =====
ax3.plot(ch6_steps, ch6_ratio, color=C6, lw=LW + 0.3, zorder=3)
ax3.set_yscale('log')
ax3.set_xlabel('Step', fontsize=LABEL_SIZE)
ax3.set_ylabel('Co-Planar / Cross-Planar Ratio', fontsize=LABEL_SIZE)
ax3.set_title('Contrastive Loss Drives 76,000:1 Separation',
              fontsize=TITLE_SIZE, fontweight='bold')
ax3.tick_params(labelsize=TICK_SIZE)
ax3.set_xlim(0, 10200)
ax3.grid(True, alpha=0.2, which='both')

# Mark the peak
peak_idx = int(np.argmax(ch6_ratio))
peak_step = ch6_steps[peak_idx]
peak_val = ch6_ratio[peak_idx]
ax3.plot(peak_step, peak_val, 'o', color=C6, markersize=7,
         markeredgecolor='white', markeredgewidth=1.5, zorder=5)
ax3.annotate(f'Peak: {peak_val:,.0f}:1\n(step {peak_step})',
             xy=(peak_step, peak_val),
             xytext=(-120, -25), textcoords='offset points',
             fontsize=10, fontweight='bold', color=C6,
             arrowprops=dict(arrowstyle='->', color=C6, lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=C6, alpha=0.9))

# Starting value annotation
ax3.annotate(f'{ch6_ratio[0]:.0f}:1',
             xy=(ch6_steps[0], ch6_ratio[0]),
             xytext=(25, 10), textcoords='offset points',
             fontsize=9, color='gray')

# Three orders of magnitude arrow
ax3.annotate('', xy=(500, peak_val * 0.7), xytext=(500, ch6_ratio[0] * 1.4),
             arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.2))
ax3.text(700, np.sqrt(ch6_ratio[0] * peak_val), '~1000x',
         fontsize=10, color='#555555', fontweight='bold', va='center')

ax3.text(0.02, 0.02, '(c)', transform=ax3.transAxes, fontsize=13,
         fontweight='bold', va='bottom')

# -- Final layout --
fig.tight_layout(pad=2.0, w_pad=3.5)

out_png = BASE / "fig_channel_ablation_definitive.png"
out_pdf = BASE / "fig_channel_ablation_definitive.pdf"
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
plt.close(fig)

# Summary stats
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print(f"\n-- Data Summary --")
print(f"n=3: {len(ch3_steps)} pts, steps {ch3_steps[0]}-{ch3_steps[-1]}")
print(f"  Val loss: {ch3_val[0]:.4f} -> {ch3_val[-1]:.4f}")
print(f"  Fiedler:  {ch3_fiedler[0]:.6f} -> {ch3_fiedler[-1]:.6f}")
print(f"n=4: {len(ch4_steps)} pts, steps {ch4_steps[0]}-{ch4_steps[-1]}")
print(f"  Val loss: {ch4_val[0]:.4f} -> {ch4_val[-1]:.4f}")
print(f"  Fiedler:  {ch4_fiedler[0]:.6f} -> {ch4_fiedler[-1]:.6f}")
print(f"n=6: {len(ch6_steps)} pts, steps {ch6_steps[0]}-{ch6_steps[-1]}")
print(f"  Val loss: {ch6_val[0]:.4f} -> {ch6_val[-1]:.4f}")
print(f"  Fiedler:  {ch6_fiedler[0]:.6f} -> {ch6_fiedler[-1]:.6f}")
print(f"  Ratio:    {ch6_ratio[0]:.0f}:1 -> peak {max(ch6_ratio):,.0f}:1 (step {ch6_steps[peak_idx]})")
