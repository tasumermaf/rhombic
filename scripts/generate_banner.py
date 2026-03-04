"""
Generate the rhombic project banner from the library's own code.

The geometry is the argument. The banner IS the science.

Colors from the 8-Law Weave (Vadrashinetal, March 2026):
  Cubic = #3D3D6B (Fall of Neutral Events, prime 11)
  FCC   = #B34444 (Geometric Essence, prime 67)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from rhombic.lattice import CubicLattice, FCCLattice

# Palette
CUBIC_COLOR = '#3D3D6B'
FCC_COLOR = '#B34444'
BG_DARK = '#0D0D0D'
GRID_SILVER = '#C0C0C0'
ARROW_AZURE = '#4A90E2'
TEXT_LIGHT = '#E8E8E0'
ACCENT_DIM = '#2A2A3A'


def draw_lattice_3d(ax, positions, edges, color, alpha_nodes=0.9, alpha_edges=0.15):
    """Draw a 3D lattice with nodes and edges."""
    # Draw edges as a collection
    edge_lines = []
    for i, j in edges:
        p1 = positions[i]
        p2 = positions[j]
        edge_lines.append([p1, p2])

    if edge_lines:
        lc = Line3DCollection(edge_lines, colors=color, alpha=alpha_edges, linewidths=0.3)
        ax.add_collection3d(lc)

    # Draw nodes
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=color, s=8, alpha=alpha_nodes, edgecolors='none', depthshade=True
    )


def generate_banner(save_path='assets/banner.png'):
    """Generate the project banner: cubic vs FCC lattice side by side."""

    matplotlib.rcParams.update({
        'font.family': 'monospace',
        'font.size': 10,
        'text.color': TEXT_LIGHT,
        'axes.labelcolor': TEXT_LIGHT,
        'xtick.color': TEXT_LIGHT,
        'ytick.color': TEXT_LIGHT,
    })

    fig = plt.figure(figsize=(16, 6), facecolor=BG_DARK)

    # Build lattices at visible scale
    n = 4
    cubic = CubicLattice(n)
    fcc = FCCLattice(n)

    # Left panel: Cubic lattice
    ax1 = fig.add_subplot(121, projection='3d', facecolor=BG_DARK)
    draw_lattice_3d(ax1, cubic.positions, cubic.edges, CUBIC_COLOR,
                    alpha_nodes=0.95, alpha_edges=0.25)
    ax1.set_title('CUBIC  (6-connected)', color=CUBIC_COLOR, fontsize=14,
                  fontweight='bold', pad=10, fontfamily='monospace')

    # Right panel: FCC lattice
    ax2 = fig.add_subplot(122, projection='3d', facecolor=BG_DARK)
    draw_lattice_3d(ax2, fcc.positions, fcc.edges, FCC_COLOR,
                    alpha_nodes=0.95, alpha_edges=0.2)
    ax2.set_title('FCC / RHOMBIC  (12-connected)', color=FCC_COLOR, fontsize=14,
                  fontweight='bold', pad=10, fontfamily='monospace')

    # Style both axes
    for ax in [ax1, ax2]:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Remove pane backgrounds
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(ACCENT_DIM)
        ax.yaxis.pane.set_edgecolor(ACCENT_DIM)
        ax.zaxis.pane.set_edgecolor(ACCENT_DIM)
        ax.grid(False)
        # Set viewing angle
        ax.view_init(elev=20, azim=35)

    # Title overlay
    fig.text(0.5, 0.95, 'R H O M B I C', ha='center', va='top',
             fontsize=28, fontweight='bold', color=TEXT_LIGHT,
             fontfamily='monospace', alpha=0.95)
    fig.text(0.5, 0.03, 'what happens when you replace the cube',
             ha='center', va='bottom', fontsize=12, color=GRID_SILVER,
             fontfamily='monospace', style='italic', alpha=0.7)

    # Key numbers — centered between panels
    stats_text = '30% shorter paths  \u00b7  40% smaller diameter  \u00b7  2.4\u00d7 algebraic connectivity'
    fig.text(0.5, 0.89, stats_text, ha='center', va='top',
             fontsize=11, color=ARROW_AZURE, fontfamily='monospace', alpha=0.9)

    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.88])
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=BG_DARK,
                pad_inches=0.3)
    print(f'Banner saved: {save_path}')
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('assets', exist_ok=True)
    generate_banner()
