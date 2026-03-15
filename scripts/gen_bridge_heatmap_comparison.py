#!/usr/bin/env python
"""
Generate publication-quality bridge heatmap comparison figure.

Compares CYBERNETIC (C-002 geometric init, TinyLlama 1.1B) vs
NON-CYBERNETIC (exp2 Qwen 7B) bridge matrices across 4 attention modules.

Layout: 2 rows x 4 columns
  Top row    = Cybernetic (C-002, TinyLlama 1.1B)
  Bottom row = Non-Cybernetic (exp2, Qwen 7B)
  Columns    = q_proj, k_proj, v_proj, o_proj (layer 10)

Output:
  results/fig_bridge_heatmap_c002_vs_exp2.png  (300 DPI)
  results/fig_bridge_heatmap_c002_vs_exp2.pdf

Usage:
  cd C:/falco/rhombic
  python scripts/gen_bridge_heatmap_comparison.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
C002_TPL = str(ROOT / 'results/corpus-baselines/C-002-geometric-default'
               '/bridge_final_model_layers_10_self_attn_{}_proj.npy')
EXP2_TPL = str(ROOT / 'results/exp2/rhombi_fcc_r24'
               '/bridge_final_model_layers_10_self_attn_{}_proj.npy')
OUT_PNG = ROOT / 'results/fig_bridge_heatmap_c002_vs_exp2.png'
OUT_PDF = ROOT / 'results/fig_bridge_heatmap_c002_vs_exp2.pdf'

# ── Load data ──────────────────────────────────────────────────────────
MODULES = ['q', 'k', 'v', 'o']
MODULE_LABELS = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

c002 = {m: np.load(C002_TPL.format(m)) for m in MODULES}
exp2 = {m: np.load(EXP2_TPL.format(m)) for m in MODULES}

# ── Co-planar / cross-planar ratio ────────────────────────────────────
CO_PAIRS = [(0, 1), (2, 3), (4, 5)]
CROSS_PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)
               if (i, j) not in CO_PAIRS]


def compute_ratio(B):
    co_mean = np.mean([abs(B[i, j]) for i, j in CO_PAIRS])
    cross_mean = np.mean([abs(B[i, j]) for i, j in CROSS_PAIRS])
    ratio = co_mean / max(cross_mean, 1e-10)
    return ratio, co_mean, cross_mean


# ── Global off-diagonal range for consistent colormapping ─────────────
def offdiag(B):
    mask = np.ones_like(B, dtype=bool)
    np.fill_diagonal(mask, False)
    return B[mask]

all_offdiag = np.concatenate(
    [offdiag(c002[m]) for m in MODULES] +
    [offdiag(exp2[m]) for m in MODULES]
)
vmax = float(np.percentile(np.abs(all_offdiag), 99))

# ── Palette ────────────────────────────────────────────────────────────
FCC_RED = '#B34444'
CYAN = '#00CED1'
DIAG_BG = '#E8E8E8'
DIAG_FG = '#888888'

# ── Build figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

datasets = [
    (c002, 'Cybernetic (C-002, TinyLlama 1.1B)'),
    (exp2, 'Non-Cybernetic (exp2, Qwen 7B)'),
]

for col, m in enumerate(MODULES):
    for row, (data, row_label) in enumerate(datasets):
        ax = axes[row, col]
        B = data[m].copy()
        ratio, co_mean, cross_mean = compute_ratio(B)

        # Display matrix with diagonal masked
        B_disp = B.copy()
        diag_vals = np.diag(B).copy()
        np.fill_diagonal(B_disp, np.nan)

        ax.imshow(B_disp, cmap='RdBu_r', norm=norm, aspect='equal')

        # Diagonal: light gray background + value text
        for i in range(6):
            ax.add_patch(patches.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                linewidth=0, facecolor=DIAG_BG, zorder=2
            ))
            ax.text(i, i, f'{diag_vals[i]:.2f}', ha='center', va='center',
                    fontsize=7, color=DIAG_FG, zorder=3)

        # Off-diagonal annotations
        for r in range(6):
            for c_idx in range(6):
                if r != c_idx:
                    val = B[r, c_idx]
                    txt_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(c_idx, r, f'{val:.3f}', ha='center', va='center',
                            fontsize=6.5, color=txt_color, zorder=3)

        # Cyan rectangles for co-planar pairs (cybernetic row only)
        if row == 0:
            for (pi, pj) in CO_PAIRS:
                for (ri, ci) in [(pi, pj), (pj, pi)]:
                    rect = patches.Rectangle(
                        (ci - 0.48, ri - 0.48), 0.96, 0.96,
                        linewidth=2.2, edgecolor=CYAN, facecolor='none',
                        zorder=4, linestyle='-'
                    )
                    ax.add_patch(rect)

        # Subplot title
        ax.set_title(f'{MODULE_LABELS[col]}\nco/cross = {ratio:.1f}x',
                      fontsize=11,
                      fontweight='bold' if row == 0 else 'normal')

        # Tick labels
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels([f'ch{i}' for i in range(6)], fontsize=8)
        ax.set_yticklabels([f'ch{i}' for i in range(6)], fontsize=8)

        # Row label on leftmost column
        if col == 0:
            ax.set_ylabel(row_label, fontsize=11, fontweight='bold',
                          color=FCC_RED if row == 0 else '#444444')

# ── Colorbar ──────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Off-diagonal bridge weight', fontsize=10)

# ── Suptitle ──────────────────────────────────────────────────────────
fig.suptitle(
    'Bridge Matrix Comparison: Cybernetic vs Non-Cybernetic Initialization\n'
    'Layer 10 Attention Modules  (6\u00d76 TeLoRA bridge)',
    fontsize=14, fontweight='bold', y=0.98
)

# ── Legend ─────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color=CYAN, linewidth=2.2, linestyle='-',
           label='Co-planar RD pairs: (0,1), (2,3), (4,5)'),
    patches.Patch(facecolor=DIAG_BG, edgecolor='#CCCCCC',
                  label='Diagonal (masked)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2,
           fontsize=9, frameon=True, bbox_to_anchor=(0.45, 0.01))

plt.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.08,
                    wspace=0.35, hspace=0.35)

# ── Save ──────────────────────────────────────────────────────────────
fig.savefig(str(OUT_PNG), dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(str(OUT_PDF), bbox_inches='tight', facecolor='white')
plt.close()

print(f'Saved: {OUT_PNG}')
print(f'Saved: {OUT_PDF}')

# ── Summary table ─────────────────────────────────────────────────────
print()
print('=== Co-planar / Cross-planar Ratios ===')
print(f'{"Module":<10} {"C-002 co":>10} {"C-002 cross":>12} {"C-002 ratio":>12}'
      f' {"exp2 co":>10} {"exp2 cross":>12} {"exp2 ratio":>12}')
print('-' * 82)
for m, ml in zip(MODULES, MODULE_LABELS):
    rc, coc, crc = compute_ratio(c002[m])
    re, coe, cre = compute_ratio(exp2[m])
    print(f'{ml:<10} {coc:>10.4f} {crc:>12.4f} {rc:>11.1f}x'
          f' {coe:>10.4f} {cre:>12.4f} {re:>11.1f}x')
