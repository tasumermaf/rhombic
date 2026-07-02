"""
TASUMER MAF — Publication Assets for Substack + X Thread.

Generates:
  1. Substack banner (1200x630) — "From sphere to crystal" narrative
  2. X thread icon (1080x1080) — Compact BD + spectral motif

Colors from the 8-Law Weave palette.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# === Palette (8-Law Weave) ===
BG_DARK = '#0A0A12'
FCC_COLOR = '#B34444'
CUBIC_COLOR = '#3D3D6B'
ACCENT_GOLD = '#C4A35A'
ACCENT_TEAL = '#4AAFAF'
TEXT_LIGHT = '#E8E8E0'
TEXT_DIM = '#808080'
GRID_LINE = '#1A1A2E'

# Warm-to-hot colormap for heatmaps (BD structure)
bd_cmap = LinearSegmentedColormap.from_list('bd', [
    (0.0, '#0A0A12'),
    (0.15, '#1A1A3E'),
    (0.3, '#3D3D6B'),
    (0.5, '#6B4444'),
    (0.7, '#B34444'),
    (0.85, '#D4714A'),
    (1.0, '#F0C060'),
])

# Diverging colormap
div_cmap = LinearSegmentedColormap.from_list('div', [
    (0.0, '#2244AA'),
    (0.35, '#1A1A2E'),
    (0.5, '#0A0A12'),
    (0.65, '#3D1A1A'),
    (1.0, '#B34444'),
])


def make_bd_matrix(n=6):
    """Generate a realistic block-diagonal bridge matrix (n=6, 3 blocks)."""
    B = np.random.randn(n, n) * 0.005  # near-zero cross coupling
    # Three 2x2 blocks: (0,5), (1,4), (2,3)
    pairs = [(0, 5), (1, 4), (2, 3)]
    for i, j in pairs:
        strength = np.random.uniform(1.2, 1.8)
        B[i, i] = strength + np.random.randn() * 0.05
        B[j, j] = strength + np.random.randn() * 0.05
        coupling = strength * 0.85 + np.random.randn() * 0.05
        B[i, j] = coupling
        B[j, i] = coupling
    return B


def make_identity_matrix(n=6):
    """Starting point — identity with slight noise."""
    return np.eye(n) * 1.0 + np.random.randn(n, n) * 0.03


def make_formless_matrix(n=6):
    """Non-cybernetic result — no structure."""
    B = np.random.randn(n, n) * 0.3
    B = (B + B.T) / 2  # symmetric
    np.fill_diagonal(B, np.random.uniform(0.8, 1.2, n))
    return B


def make_tesseract_matrix(n=8):
    """4+4 block-diagonal for tesseract (n=8)."""
    B = np.random.randn(n, n) * 0.003
    pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in pairs:
        strength = np.random.uniform(1.0, 1.5)
        B[i, i] = strength + np.random.randn() * 0.03
        B[j, j] = strength + np.random.randn() * 0.03
        coupling = strength * 0.8 + np.random.randn() * 0.03
        B[i, j] = coupling
        B[j, i] = coupling
    return B


def generate_substack_banner(save_path):
    """
    1200x630 banner: three-panel crystallization narrative.
    Left: identity (start). Center: formless blob (no Steersman). Right: crystal (BD).
    """
    fig = plt.figure(figsize=(12, 6.3), facecolor=BG_DARK, dpi=100)

    # Title
    fig.text(0.5, 0.96, 'T A S U M E R   M A F', ha='center', va='top',
             fontsize=24, fontweight='bold', color=TEXT_LIGHT,
             fontfamily='monospace', alpha=0.95)
    fig.text(0.5, 0.90, 'from sphere to crystal — programming geometry into neural networks',
             ha='center', va='top', fontsize=11, color=TEXT_DIM,
             fontfamily='monospace', style='italic')

    matrices = [
        (make_identity_matrix(6), 'Step 0\nIdentity', div_cmap),
        (make_formless_matrix(6), 'Step 10K\nNo Steersman', div_cmap),
        (make_bd_matrix(6), 'Step 10K\nSteersman Active', div_cmap),
    ]

    ratios = ['—', '1 : 1', '70,000 : 1']

    for idx, (mat, title, cmap) in enumerate(matrices):
        ax = fig.add_axes([0.05 + idx * 0.31, 0.12, 0.27, 0.65])
        # Use consistent scale across all three panels for comparison
        global_vmax = 1.8
        ax.imshow(mat, cmap=cmap, vmin=-global_vmax, vmax=global_vmax, aspect='equal',
                  interpolation='nearest')

        # Grid lines
        for k in range(7):
            ax.axhline(k - 0.5, color=GRID_LINE, linewidth=0.5)
            ax.axvline(k - 0.5, color=GRID_LINE, linewidth=0.5)

        # Block boundaries for the BD panel
        if idx == 2:
            pairs_xy = [(0, 5), (1, 4), (2, 3)]
            for (r, c) in pairs_xy:
                for dr, dc in [(r, c), (c, r), (r, r), (c, c)]:
                    rect = plt.Rectangle((dc - 0.48, dr - 0.48), 0.96, 0.96,
                                         linewidth=2.0, edgecolor=ACCENT_GOLD,
                                         facecolor='none', alpha=0.9)
                    ax.add_patch(rect)

        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels([str(i) for i in range(6)], fontsize=7, color=TEXT_DIM)
        ax.set_yticklabels([str(i) for i in range(6)], fontsize=7, color=TEXT_DIM)
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=10, color=TEXT_LIGHT, fontfamily='monospace',
                     fontweight='bold', pad=8)

        # Ratio annotation below
        ax.text(2.5, 6.8, f'co/cross: {ratios[idx]}',
                ha='center', va='center', fontsize=9,
                color=ACCENT_GOLD if idx == 2 else TEXT_DIM,
                fontfamily='monospace',
                fontweight='bold' if idx == 2 else 'normal')

    # Arrows between panels
    for x_start in [0.33, 0.645]:
        fig.text(x_start, 0.44, '→', ha='center', va='center',
                 fontsize=28, color=TEXT_DIM, fontfamily='monospace', alpha=0.5)

    fig.savefig(save_path, dpi=100, facecolor=BG_DARK, bbox_inches='tight',
                pad_inches=0.15)
    plt.close(fig)
    print(f'Substack banner: {save_path}')


def generate_x_icon(save_path):
    """
    1080x1080 X thread icon: BD heatmap (8x8 tesseract) + spectral attractor ring.
    """
    fig = plt.figure(figsize=(10.8, 10.8), facecolor=BG_DARK, dpi=100)

    # Main heatmap — tesseract BD
    ax_main = fig.add_axes([0.1, 0.18, 0.8, 0.65])
    mat = make_tesseract_matrix(8)
    vmax = max(abs(mat.min()), abs(mat.max()))
    ax_main.imshow(mat, cmap=bd_cmap, vmin=0, vmax=vmax, aspect='equal',
                   interpolation='nearest')

    # Block boundaries (4 pairs for tesseract)
    pairs_xy = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for (r, c) in pairs_xy:
        for dr, dc in [(r, c), (c, r), (r, r), (c, c)]:
            rect = plt.Rectangle((dc - 0.45, dr - 0.45), 0.9, 0.9,
                                 linewidth=2.0, edgecolor=ACCENT_GOLD,
                                 facecolor='none', alpha=0.8)
            ax_main.add_patch(rect)

    for k in range(9):
        ax_main.axhline(k - 0.5, color=GRID_LINE, linewidth=0.5, alpha=0.5)
        ax_main.axvline(k - 0.5, color=GRID_LINE, linewidth=0.5, alpha=0.5)

    ax_main.set_xticks(range(8))
    ax_main.set_yticks(range(8))
    ax_main.set_xticklabels([str(i) for i in range(8)], fontsize=10, color=TEXT_DIM)
    ax_main.set_yticklabels([str(i) for i in range(8)], fontsize=10, color=TEXT_DIM)
    ax_main.tick_params(length=0)

    # Title
    fig.text(0.5, 0.92, 'T A S U M E R   M A F', ha='center', va='center',
             fontsize=28, fontweight='bold', color=TEXT_LIGHT,
             fontfamily='monospace')
    fig.text(0.5, 0.87, 'tesseract bridge  ·  n = 8  ·  41,564 : 1',
             ha='center', va='center', fontsize=14, color=ACCENT_GOLD,
             fontfamily='monospace')

    # Spectral attractor bar at bottom
    ax_bar = fig.add_axes([0.15, 0.06, 0.7, 0.06])
    channels = [3, 4, 6, 8, 12]
    fiedler_spectral = [0.0951, 0.0918, 0.00009, 0.0944, 0.1019]
    colors = [ACCENT_TEAL if f > 0.05 else FCC_COLOR for f in fiedler_spectral]
    bars = ax_bar.bar(range(len(channels)), fiedler_spectral, color=colors, alpha=0.8, width=0.6)

    ax_bar.set_xticks(range(len(channels)))
    ax_bar.set_xticklabels([f'n={c}' for c in channels], fontsize=9, color=TEXT_DIM,
                            fontfamily='monospace')
    ax_bar.set_yticks([])
    ax_bar.set_facecolor(BG_DARK)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    ax_bar.spines['bottom'].set_color(TEXT_DIM)
    ax_bar.tick_params(length=0)

    # Attractor line
    ax_bar.axhline(0.10, color=ACCENT_TEAL, linewidth=1, linestyle='--', alpha=0.6)
    ax_bar.text(len(channels) - 0.3, 0.105, 'attractor ~0.10',
                fontsize=8, color=ACCENT_TEAL, fontfamily='monospace', ha='right', alpha=0.7)

    # Mark the BD run
    ax_bar.text(2, fiedler_spectral[2] + 0.008, 'BD',
                fontsize=8, color=FCC_COLOR, fontfamily='monospace',
                ha='center', fontweight='bold')

    fig.savefig(save_path, dpi=100, facecolor=BG_DARK, bbox_inches='tight',
                pad_inches=0.1)
    plt.close(fig)
    print(f'X thread icon: {save_path}')


if __name__ == '__main__':
    out = Path('C:/Users/Timothy Paul Bielec/Desktop/TASUMER_MAF_Deliverables')
    out.mkdir(exist_ok=True)

    np.random.seed(67)  # Primary thread prime

    generate_substack_banner(out / 'substack_banner.png')
    generate_x_icon(out / 'x_thread_icon.png')

    print('\nDone. Assets in:', out)
