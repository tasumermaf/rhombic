#!/usr/bin/env python
"""
Generate publication-quality bridge matrix evolution figure for C-002.

Shows how the 6x6 TeLoRA bridge develops block-diagonal structure
over training, with 4 heatmap panels at key steps plus 2 line plots
showing the quantitative evolution of co-planar vs cross-planar coupling.

Layout: top row = 4 heatmaps + colorbar, bottom row = 2 line metrics

Output:
  results/fig_bridge_evolution_c002.png  (300 DPI)
  results/fig_bridge_evolution_c002.pdf

Usage:
  cd C:/falco/rhombic
  python scripts/fig_bridge_evolution_c002.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'results' / 'corpus-baselines' / 'C-002-geometric-default'
OUT_PNG = ROOT / 'results' / 'fig_bridge_evolution_c002.png'
OUT_PDF = ROOT / 'results' / 'fig_bridge_evolution_c002.pdf'

# ── Palette (from rhombic/visualize.py) ──────────────────────────────────
FCC_RED = '#B34444'        # Geometric Essence (67)
CUBIC_BLUE = '#3D3D6B'    # Fall of Neutral Events (11)
ARROW_AZURE = '#4A90E2'   # Arrow of Complexity (17)
SYNC_ORANGE = '#FF8C42'   # Synchronicity (89)
JUSTICE_GREEN = '#52A352'  # Sole Atom (19)
KAOS_GOLD = '#FFD700'      # Kaos (23)
GRID_SILVER = '#C0C0C0'   # Time Matrix (29)
CYAN = '#00CED1'           # Co-planar highlight
DIAG_BG = '#E8E8E8'
DIAG_FG = '#888888'

# ── Style ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
})

# ── Co-planar / cross-planar definitions ─────────────────────────────────
CO_PAIRS = [(0, 1), (2, 3), (4, 5)]
CO_CELLS = [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)]
CROSS_CELLS = [(i, j) for i in range(6) for j in range(6)
               if i != j and (i, j) not in CO_CELLS]


def co_cross_stats(B):
    """Compute co-planar mean, cross-planar mean, and ratio."""
    co = np.mean([abs(B[i, j]) for i, j in CO_CELLS])
    cr = np.mean([abs(B[i, j]) for i, j in CROSS_CELLS])
    return co, cr, co / max(cr, 1e-10)


def offdiag(B):
    mask = np.ones_like(B, dtype=bool)
    np.fill_diagonal(mask, False)
    return B[mask]


# ── Load checkpoint timeseries ───────────────────────────────────────────
with open(BASE / 'results.json') as f:
    data = json.load(f)

# ── Compute per-step statistics from actual bridge files ─────────────────
LAYER = 10
PROJ = 'q'

snapshot_steps = list(range(0, 10001, 100))
co_means = []
cross_means = []
ratios = []
valid_steps = []

for s in snapshot_steps:
    tag = f'step{s}'
    path = BASE / f'bridge_{tag}_model_layers_{LAYER}_self_attn_{PROJ}_proj.npy'
    if path.exists():
        B = np.load(str(path))
        co, cr, ratio = co_cross_stats(B)
        co_means.append(co)
        cross_means.append(cr)
        ratios.append(ratio)
        valid_steps.append(s)

co_means = np.array(co_means)
cross_means = np.array(cross_means)
ratios = np.array(ratios)
valid_steps = np.array(valid_steps)

# ── Select 4 representative steps for heatmaps ──────────────────────────
HEATMAP_STEPS = [0, 500, 2000, 10000]
HEATMAP_LABELS = [
    'Step 0\n(Initialization)',
    'Step 500',
    'Step 2,000',
    'Step 10,000\n(Final)',
]

bridges = {}
for s in HEATMAP_STEPS:
    path = BASE / f'bridge_step{s}_model_layers_{LAYER}_self_attn_{PROJ}_proj.npy'
    bridges[s] = np.load(str(path))

# ── Color normalization ──────────────────────────────────────────────────
# Use the final step's range for consistent scaling across all panels.
all_offdiag = np.concatenate([offdiag(bridges[s]) for s in HEATMAP_STEPS])
vmax = float(np.percentile(np.abs(all_offdiag), 99))
vmax = max(vmax, 0.01)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

# ── Build figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4.5))

# Outer gridspec: heatmap row and line-plot row
outer = gridspec.GridSpec(
    1, 2, figure=fig,
    width_ratios=[4.5, 2.2],
    left=0.04, right=0.97, top=0.82, bottom=0.12,
    wspace=0.35,
)

# Left half: 4 heatmaps + colorbar
gs_heat = gridspec.GridSpecFromSubplotSpec(
    1, 5, subplot_spec=outer[0],
    width_ratios=[1, 1, 1, 1, 0.06],
    wspace=0.40,
)

# Right half: 2 stacked line plots
gs_lines = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[1],
    wspace=0.45,
)

# ── Panels A-D: Heatmaps ────────────────────────────────────────────────
for col, (step, label) in enumerate(zip(HEATMAP_STEPS, HEATMAP_LABELS)):
    ax = fig.add_subplot(gs_heat[0, col])
    B = bridges[step].copy()
    co, cr, ratio = co_cross_stats(B)

    B_disp = B.copy()
    diag_vals = np.diag(B).copy()
    np.fill_diagonal(B_disp, np.nan)

    ax.imshow(B_disp, cmap='RdBu_r', norm=norm, aspect='equal',
              interpolation='nearest')

    # Diagonal: gray background + value
    for i in range(6):
        ax.add_patch(patches.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            linewidth=0, facecolor=DIAG_BG, zorder=2
        ))
        ax.text(i, i, f'{diag_vals[i]:.2f}', ha='center', va='center',
                fontsize=5.5, color=DIAG_FG, zorder=3)

    # Off-diagonal annotations
    for r in range(6):
        for c_idx in range(6):
            if r != c_idx:
                val = B[r, c_idx]
                txt_color = 'white' if abs(val) > vmax * 0.55 else 'black'
                if abs(val) < 0.001 and val != 0:
                    fmt = f'{val:.0e}'
                else:
                    fmt = f'{val:.3f}'
                ax.text(c_idx, r, fmt, ha='center', va='center',
                        fontsize=5, color=txt_color, zorder=3)

    # Cyan rectangles highlighting co-planar pairs
    for (pi, pj) in CO_PAIRS:
        for (ri, ci) in [(pi, pj), (pj, pi)]:
            rect = patches.Rectangle(
                (ci - 0.48, ri - 0.48), 0.96, 0.96,
                linewidth=2.0, edgecolor=CYAN, facecolor='none',
                zorder=4, linestyle='-'
            )
            ax.add_patch(rect)

    # Title
    ratio_str = f'{ratio:.1f}x' if ratio < 1e6 else f'{ratio:.0e}x'
    ax.set_title(f'{label}\nco/cross = {ratio_str}', fontsize=8.5,
                 fontweight='bold' if step == 10000 else 'normal',
                 pad=6)

    # Ticks
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels([f'ch{i}' for i in range(6)], fontsize=6.5)
    ax.set_yticklabels([f'ch{i}' for i in range(6)], fontsize=6.5)

    # Panel letter
    letter = chr(65 + col)
    ax.text(-0.22, 1.22, letter, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

# ── Colorbar ─────────────────────────────────────────────────────────────
cbar_ax = fig.add_subplot(gs_heat[0, 4])
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Off-diag\nweight', fontsize=7.5)
cbar.ax.tick_params(labelsize=6.5)

# ── Panel E: Co-planar and cross-planar magnitude ────────────────────────
ax_e = fig.add_subplot(gs_lines[0, 0])
ax_e.plot(valid_steps, co_means, '-', color=FCC_RED, linewidth=1.8,
          label='Co-planar', zorder=3)
ax_e.plot(valid_steps, cross_means, '-', color=CUBIC_BLUE, linewidth=1.8,
          label='Cross-planar', zorder=3)
ax_e.set_xlabel('Training Step', fontsize=8.5)
ax_e.set_ylabel('Mean |weight|', fontsize=8.5)
ax_e.legend(fontsize=7, loc='center left', framealpha=0.9)
ax_e.tick_params(labelsize=7)
ax_e.grid(True, alpha=0.25)
ax_e.set_xlim(0, 10000)
# Mark heatmap snapshot positions
for s in HEATMAP_STEPS:
    ax_e.axvline(s, color=GRID_SILVER, linestyle=':', linewidth=0.7, alpha=0.7)
ax_e.text(-0.18, 1.12, 'E', transform=ax_e.transAxes,
          fontsize=12, fontweight='bold', va='top')

# ── Panel F: Co/Cross ratio ──────────────────────────────────────────────
ax_f = fig.add_subplot(gs_lines[0, 1])
ax_f.plot(valid_steps, ratios, '-', color=SYNC_ORANGE, linewidth=1.8, zorder=3)
ax_f.set_xlabel('Training Step', fontsize=8.5)
ax_f.set_ylabel('Co/Cross Ratio', fontsize=8.5)
ax_f.tick_params(labelsize=7)
ax_f.grid(True, alpha=0.25)
ax_f.set_xlim(0, 10000)
for s in HEATMAP_STEPS:
    ax_f.axvline(s, color=GRID_SILVER, linestyle=':', linewidth=0.7, alpha=0.7)
ax_f.text(-0.18, 1.12, 'F', transform=ax_f.transAxes,
          fontsize=12, fontweight='bold', va='top')

# ── Suptitle ─────────────────────────────────────────────────────────────
fig.suptitle(
    'Bridge Matrix Evolution During Training  (C-002, Layer 10 q_proj)',
    fontsize=12, fontweight='bold', y=0.96
)

# ── Legend for heatmaps ──────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=CYAN, linewidth=2.0, linestyle='-',
           label='Co-planar RD pairs: (0,1), (2,3), (4,5)'),
    patches.Patch(facecolor=DIAG_BG, edgecolor='#CCCCCC',
                  label='Diagonal (self-coupling)'),
]
fig.legend(handles=legend_handles, loc='lower left', ncol=2,
           fontsize=7, frameon=True, bbox_to_anchor=(0.04, 0.0))

# ── Save ─────────────────────────────────────────────────────────────────
fig.savefig(str(OUT_PNG), dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(str(OUT_PDF), bbox_inches='tight', facecolor='white')
plt.close()

print(f'Saved: {OUT_PNG}')
print(f'Saved: {OUT_PDF}')

# ── Summary ──────────────────────────────────────────────────────────────
print()
print('=== Bridge Evolution Summary (Layer 10, q_proj) ===')
print(f'{"Step":>8}  {"Co-planar":>12}  {"Cross-planar":>14}  {"Ratio":>10}')
print('-' * 50)
for s in HEATMAP_STEPS:
    idx = list(valid_steps).index(s)
    print(f'{s:>8}  {co_means[idx]:>12.6f}  {cross_means[idx]:>14.6f}  {ratios[idx]:>9.1f}x')
