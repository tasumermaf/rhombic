#!/usr/bin/env python3
"""
Generate publication-quality figure: Initialization Convergence Proof.

Shows that identity, geometric, and corpus-coupled initializations
all converge to the same final state at 10K steps.
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path("C:/falco/rhombic/results")
C001_PATH = ROOT / "C-001_temporal_emergence.json"
C002_PATH = ROOT / "corpus-baselines" / "C-002-geometric-default" / "results.json"
C003_PATH = ROOT / "corpus-baselines" / "C-003-corpus-coupled-default" / "results.json"
H3_PATH   = ROOT / "channel-ablation" / "H-ch6" / "metrics_hermes.csv"
OUT_PNG = ROOT / "fig_init_convergence_final.png"
OUT_PDF = ROOT / "fig_init_convergence_final.pdf"

# ── Colors ─────────────────────────────────────────────────────────────
C001_COLOR = '#2196F3'   # blue
C002_COLOR = '#FF9800'   # orange
C003_COLOR = '#4CAF50'   # green
H3_COLOR   = '#B34444'   # red

# ── Load Data ──────────────────────────────────────────────────────────

# C-001: list of dicts with 'step', 'ratio' (no val_loss, no fiedler_mean)
with open(C001_PATH) as f:
    c001_raw = json.load(f)
c001_steps = np.array([d['step'] for d in c001_raw if d['step'] > 0])
c001_ratio = np.array([d['ratio'] for d in c001_raw if d['step'] > 0])

# C-002: dict with 'checkpoints', each has step/val_loss/co_cross_ratio/fiedler_mean
with open(C002_PATH) as f:
    c002_raw = json.load(f)['checkpoints']
c002_steps     = np.array([d['step'] for d in c002_raw])
c002_ratio     = np.array([d['co_cross_ratio'] for d in c002_raw])
c002_val_loss  = np.array([d['val_loss'] for d in c002_raw])
c002_fiedler   = np.array([d['fiedler_mean'] for d in c002_raw])

# C-003: same format as C-002
with open(C003_PATH) as f:
    c003_raw = json.load(f)['checkpoints']
c003_steps     = np.array([d['step'] for d in c003_raw])
c003_ratio     = np.array([d['co_cross_ratio'] for d in c003_raw])
c003_val_loss  = np.array([d['val_loss'] for d in c003_raw])
c003_fiedler   = np.array([d['fiedler_mean'] for d in c003_raw])

# H3: CSV — first line is count, then data
# Columns: step,val_loss,fiedler_mean,co_cross_ratio,contrastive_loss,deviation_mean,spectral_gap
with open(H3_PATH) as f:
    lines = f.readlines()
h3_steps, h3_val_loss, h3_fiedler, h3_ratio = [], [], [], []
for line in lines[1:]:  # skip first line (count)
    parts = line.strip().split(',')
    if len(parts) >= 4:
        h3_steps.append(int(parts[0]))
        h3_val_loss.append(float(parts[1]))
        h3_fiedler.append(float(parts[2]))
        h3_ratio.append(float(parts[3]))
h3_steps    = np.array(h3_steps)
h3_val_loss = np.array(h3_val_loss)
h3_fiedler  = np.array(h3_fiedler)
h3_ratio    = np.array(h3_ratio)

# ── Figure Setup ───────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Initialization Strategy Convergence Proof',
             fontsize=15, fontweight='bold', y=0.97)
fig.text(0.5, 0.935,
         'All initializations converge to the same final state — init is cosmetic',
         ha='center', fontsize=10, fontstyle='italic', color='#555555')

LW = 1.6
ALPHA = 0.9
LABEL_C001 = 'Identity (C-001)'
LABEL_C002 = 'Geometric (C-002)'
LABEL_C003 = 'Corpus-coupled (C-003)'
LABEL_H3   = 'Identity (H3, ablation)'

# ── Panel 1: Co/Cross Ratio (log scale) ───────────────────────────────
ax1 = axes[0, 0]
ax1.set_title('(a)  Co/Cross Ratio', fontweight='bold', loc='left')

ax1.semilogy(c001_steps, c001_ratio, color=C001_COLOR, lw=LW, alpha=ALPHA, label=LABEL_C001)
ax1.semilogy(c002_steps, c002_ratio, color=C002_COLOR, lw=LW, alpha=ALPHA, label=LABEL_C002)
ax1.semilogy(c003_steps, c003_ratio, color=C003_COLOR, lw=LW, alpha=ALPHA, label=LABEL_C003)
ax1.semilogy(h3_steps,   h3_ratio,   color=H3_COLOR,   lw=LW, alpha=ALPHA, label=LABEL_H3)

# Mark C-003 peak at step 9000
c003_peak_idx = np.argmax(c003_ratio)
c003_peak_step = c003_steps[c003_peak_idx]
c003_peak_val  = c003_ratio[c003_peak_idx]
ax1.annotate(f'Peak: {c003_peak_val:,.0f}:1\n(step {c003_peak_step})',
             xy=(c003_peak_step, c003_peak_val),
             xytext=(c003_peak_step - 2500, c003_peak_val * 1.8),
             arrowprops=dict(arrowstyle='->', color=C003_COLOR, lw=1.2),
             fontsize=8, color=C003_COLOR, fontweight='bold')

# Convergence band annotation
ax1.axhspan(60000, 80000, alpha=0.06, color='gray')
ax1.text(10200, 69000, '64K–71K band', fontsize=7.5, color='#666666', va='center')

ax1.set_xlabel('Training Step')
ax1.set_ylabel('Co/Cross Ratio')
ax1.set_ylim(1, 200000)
ax1.set_xlim(0, 10500)
ax1.legend(loc='lower right', framealpha=0.9)

# ── Panel 2: Val Loss ─────────────────────────────────────────────────
ax2 = axes[0, 1]
ax2.set_title('(b)  Validation Loss', fontweight='bold', loc='left')

# C-001 has no val_loss — only plot runs that have it
ax2.plot(c002_steps, c002_val_loss, color=C002_COLOR, lw=LW, alpha=ALPHA, label=LABEL_C002)
ax2.plot(c003_steps, c003_val_loss, color=C003_COLOR, lw=LW, alpha=ALPHA, label=LABEL_C003)
ax2.plot(h3_steps,   h3_val_loss,   color=H3_COLOR,   lw=LW, alpha=ALPHA, label=LABEL_H3)

# Annotate final values
final_pairs = [
    (c002_steps[-1], c002_val_loss[-1], C002_COLOR, 'C-002'),
    (c003_steps[-1], c003_val_loss[-1], C003_COLOR, 'C-003'),
    (h3_steps[-1],   h3_val_loss[-1],   H3_COLOR,   'H3'),
]
y_offsets = [0.0012, 0.0000, -0.0012]
for (s, v, c, lab), yoff in zip(final_pairs, y_offsets):
    ax2.annotate(f'{v:.4f}',
                 xy=(s, v), xytext=(s - 2000, v + yoff),
                 arrowprops=dict(arrowstyle='->', color=c, lw=0.8),
                 fontsize=8, color=c, fontweight='bold')

# Convergence band
ax2.axhspan(0.4010, 0.4016, alpha=0.08, color='gray')
ax2.text(500, 0.4013, '0.4010–0.4015', fontsize=7.5, color='#666666', va='center')

ax2.set_xlabel('Training Step')
ax2.set_ylabel('Validation Loss')
ax2.set_ylim(0.395, 0.48)
ax2.set_xlim(0, 10500)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.text(500, 0.475, 'C-001: no val_loss data', fontsize=7.5,
         color=C001_COLOR, fontstyle='italic', alpha=0.7)

# ── Panel 3: Ratio Delta (% difference from mean) ─────────────────────
ax3 = axes[1, 0]
ax3.set_title('(c)  Ratio Delta (% difference from mean)', fontweight='bold', loc='left')

# Find common steps for runs that have ratio data
# C-002, C-003, H3 all have steps at 100-step intervals up to 10000
# C-001 goes only to 4000
# Use C-002, C-003, H3 for the full range
common_steps_full = np.intersect1d(c002_steps, np.intersect1d(c003_steps, h3_steps))

# Get ratio values at common steps for C-002, C-003, H3
def get_at_steps(steps_arr, val_arr, target_steps):
    """Get values at specific steps"""
    result = []
    step_to_val = dict(zip(steps_arr, val_arr))
    for s in target_steps:
        if s in step_to_val:
            result.append(step_to_val[s])
        else:
            result.append(np.nan)
    return np.array(result)

c002_at_common = get_at_steps(c002_steps, c002_ratio, common_steps_full)
c003_at_common = get_at_steps(c003_steps, c003_ratio, common_steps_full)
h3_at_common   = get_at_steps(h3_steps,   h3_ratio,   common_steps_full)

# Also get C-001 at common steps (limited range)
c001_at_common_full = get_at_steps(c001_steps, c001_ratio, common_steps_full)

# Compute mean at each step (use available runs)
mean_ratio = np.zeros(len(common_steps_full))
for i in range(len(common_steps_full)):
    vals = [v for v in [c002_at_common[i], c003_at_common[i], h3_at_common[i]]
            if not np.isnan(v)]
    if not np.isnan(c001_at_common_full[i]):
        vals.append(c001_at_common_full[i])
    mean_ratio[i] = np.mean(vals) if vals else np.nan

# Compute % delta for each run
def pct_delta(vals, mean):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(vals - mean) / mean * 100

c002_delta = pct_delta(c002_at_common, mean_ratio)
c003_delta = pct_delta(c003_at_common, mean_ratio)
h3_delta   = pct_delta(h3_at_common, mean_ratio)
c001_delta = pct_delta(c001_at_common_full, mean_ratio)

# Also compute max pairwise delta at each step
max_pairwise = np.zeros(len(common_steps_full))
for i in range(len(common_steps_full)):
    vals = []
    if not np.isnan(c002_at_common[i]): vals.append(c002_at_common[i])
    if not np.isnan(c003_at_common[i]): vals.append(c003_at_common[i])
    if not np.isnan(h3_at_common[i]):   vals.append(h3_at_common[i])
    if not np.isnan(c001_at_common_full[i]): vals.append(c001_at_common_full[i])
    if len(vals) >= 2:
        max_pairwise[i] = (max(vals) - min(vals)) / np.mean(vals) * 100
    else:
        max_pairwise[i] = np.nan

# Plot max pairwise delta as a filled area
valid_mask = ~np.isnan(max_pairwise)
ax3.fill_between(common_steps_full[valid_mask], 0, max_pairwise[valid_mask],
                 alpha=0.15, color='#888888', label='Max pairwise spread')
ax3.plot(common_steps_full[valid_mask], max_pairwise[valid_mask],
         color='#555555', lw=1.8, alpha=0.8)

# Plot individual run deltas
c001_valid = ~np.isnan(c001_delta)
ax3.plot(common_steps_full[c001_valid], c001_delta[c001_valid],
         color=C001_COLOR, lw=1.2, alpha=0.7, linestyle='--', label=LABEL_C001)
ax3.plot(common_steps_full, c002_delta, color=C002_COLOR, lw=1.2, alpha=0.7, linestyle='--', label=LABEL_C002)
ax3.plot(common_steps_full, c003_delta, color=C003_COLOR, lw=1.2, alpha=0.7, linestyle='--', label=LABEL_C003)
ax3.plot(common_steps_full, h3_delta,   color=H3_COLOR,   lw=1.2, alpha=0.7, linestyle='--', label=LABEL_H3)

# 10% threshold line
ax3.axhline(10, color='red', lw=0.8, linestyle=':', alpha=0.5)
ax3.text(10200, 10, '10%', fontsize=7.5, color='red', alpha=0.6, va='center')

# Annotate final spread
final_spread = max_pairwise[valid_mask][-1]
ax3.annotate(f'Final spread: {final_spread:.1f}%',
             xy=(common_steps_full[valid_mask][-1], final_spread),
             xytext=(7000, max(30, final_spread + 15)),
             arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
             fontsize=8.5, color='#555555', fontweight='bold')

ax3.set_xlabel('Training Step')
ax3.set_ylabel('Ratio Deviation from Mean (%)')
ax3.set_xlim(0, 10500)
ax3.set_ylim(0, max(100, np.nanmax(max_pairwise[valid_mask]) * 1.1))
ax3.legend(loc='upper right', framealpha=0.9, fontsize=7.5)

# ── Panel 4: Bridge Fiedler ───────────────────────────────────────────
ax4 = axes[1, 1]
ax4.set_title('(d)  Bridge Fiedler Value', fontweight='bold', loc='left')

# C-001 has no fiedler_mean — only plot runs that have it
ax4.plot(c002_steps, c002_fiedler, color=C002_COLOR, lw=LW, alpha=ALPHA, label=LABEL_C002)
ax4.plot(c003_steps, c003_fiedler, color=C003_COLOR, lw=LW, alpha=ALPHA, label=LABEL_C003)
ax4.plot(h3_steps,   h3_fiedler,   color=H3_COLOR,   lw=LW, alpha=ALPHA, label=LABEL_H3)

# Convergence band
ax4.axhspan(0.00005, 0.00012, alpha=0.08, color='gray')

# Annotate final values
final_fiedler = [
    (c002_steps[-1], c002_fiedler[-1], C002_COLOR, 'C-002'),
    (c003_steps[-1], c003_fiedler[-1], C003_COLOR, 'C-003'),
    (h3_steps[-1],   h3_fiedler[-1],   H3_COLOR,   'H3'),
]
y_offsets_f = [0.0008, 0.0005, 0.0002]
for (s, v, c, lab), yoff in zip(final_fiedler, y_offsets_f):
    ax4.annotate(f'{v:.2e}',
                 xy=(s, v), xytext=(s - 3500, v + yoff),
                 arrowprops=dict(arrowstyle='->', color=c, lw=0.8),
                 fontsize=8, color=c, fontweight='bold')

ax4.set_xlabel('Training Step')
ax4.set_ylabel('Fiedler Mean')
ax4.set_xlim(0, 10500)
ax4.set_ylim(-0.0002, 0.012)
ax4.legend(loc='upper right', framealpha=0.9)
ax4.text(500, 0.011, 'C-001: no Fiedler data', fontsize=7.5,
         color=C001_COLOR, fontstyle='italic', alpha=0.7)

# ── Final layout ──────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.02, 1, 0.92])

# Footer
fig.text(0.5, 0.008,
         'C-001: identity init (4K steps) | C-002: geometric init (10K steps) | '
         'C-003: corpus-coupled init (10K steps) | H3: identity init, 6-channel ablation (10K steps)',
         ha='center', fontsize=7.5, color='#888888')

plt.savefig(OUT_PNG, bbox_inches='tight', facecolor='white')
plt.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")

# ── Summary statistics ────────────────────────────────────────────────
print("\n=== Final Values at Step 10,000 ===")
print(f"  C-002  ratio={c002_ratio[-1]:,.0f}  val_loss={c002_val_loss[-1]:.4f}  fiedler={c002_fiedler[-1]:.2e}")
print(f"  C-003  ratio={c003_ratio[-1]:,.0f}  val_loss={c003_val_loss[-1]:.4f}  fiedler={c003_fiedler[-1]:.2e}")
print(f"  H3     ratio={h3_ratio[-1]:,.0f}  val_loss={h3_val_loss[-1]:.4f}  fiedler={h3_fiedler[-1]:.2e}")
print(f"  C-001  ratio={c001_ratio[-1]:,.0f}  (at step {c001_steps[-1]}, no val_loss/fiedler)")
print(f"\n  C-003 peak: {c003_peak_val:,.0f}:1 at step {c003_peak_step}")
print(f"  Final max pairwise spread: {final_spread:.1f}%")
