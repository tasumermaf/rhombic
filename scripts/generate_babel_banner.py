"""
Generate the Substack banner for "What the Tower Was Made Of."

Concept: Two towers as architectural elevation drawings on warm parchment.
Left tower (cubic) crumbles at the top. Right tower (FCC) stands whole.
Clean, light, evocative — distinct from the dark GitHub/data-viz aesthetic.

Colors from the 8-Law Weave:
  Cubic = #3D3D6B (Fall of Neutral Events, prime 11)
  FCC   = #B34444 (Geometric Essence, prime 67)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Palette
CUBIC_COLOR = '#3D3D6B'
FCC_COLOR = '#B34444'
BG_PARCHMENT = '#F5F0E8'
BG_WARM = '#EDE8DD'
TEXT_DARK = '#1A1A2E'
TEXT_MID = '#555566'
DUST_COLOR = '#8888AA'


def cubic_tower_2d(nx, nz):
    """Build a 2D front-elevation cubic lattice (x, z)."""
    positions = []
    for iz in range(nz):
        for ix in range(nx):
            positions.append([float(ix), float(iz)])
    return np.array(positions)


def cubic_edges_2d(nx, nz):
    """Horizontal and vertical edges for the cubic elevation."""
    edges = []
    def idx(ix, iz):
        return iz * nx + ix
    for iz in range(nz):
        for ix in range(nx):
            if ix + 1 < nx:
                edges.append((idx(ix, iz), idx(ix+1, iz)))
            if iz + 1 < nz:
                edges.append((idx(ix, iz), idx(ix, iz+1)))
    return edges


def fcc_tower_2d(nx, nz):
    """Build a 2D front-elevation FCC-like lattice.

    Alternating rows offset by half a unit, with diagonal connections.
    This is the 2D projection of FCC — a triangular/hex lattice.
    """
    positions = []
    for iz in range(nz):
        offset = 0.5 if iz % 2 == 1 else 0.0
        for ix in range(nx):
            positions.append([float(ix) + offset, float(iz) * 0.866])
    return np.array(positions)


def fcc_edges_2d(positions, cutoff=1.1):
    """Connect FCC-2D neighbors within cutoff distance."""
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    pairs = tree.query_pairs(cutoff)
    return list(pairs)


def scatter_top(positions, edges, z_frac=0.65, strength=2.0, seed=42):
    """Displace top nodes outward/upward. Return new positions + edge alphas."""
    z_max = positions[:, 1].max()
    z_thresh = z_max * z_frac
    new_pos = positions.copy()
    rng = np.random.RandomState(seed)
    cx = (positions[:, 0].max() + positions[:, 0].min()) / 2

    for i in range(len(new_pos)):
        z = positions[i, 1]
        if z > z_thresh:
            t = (z - z_thresh) / (z_max - z_thresh)
            dx = new_pos[i, 0] - cx
            radial = strength * t * t
            new_pos[i, 0] += dx * radial + rng.normal(0, 0.2 * t)
            new_pos[i, 1] += t * strength * 0.6 + rng.normal(0, 0.1 * t)

    edge_alphas = []
    for i, j in edges:
        z_e = max(positions[i, 1], positions[j, 1])
        if z_e > z_thresh:
            t = (z_e - z_thresh) / (z_max - z_thresh)
            edge_alphas.append(max(0.03, 0.6 * (1 - t * t)))
        else:
            edge_alphas.append(0.6)

    return new_pos, edge_alphas


def draw_lattice_2d(ax, positions, edges, color, edge_alphas=None,
                    node_size=12, edge_width=0.6, node_alpha=0.85):
    """Draw a 2D lattice as an architectural elevation."""
    # Draw edges
    if edge_alphas is not None:
        for idx, (i, j) in enumerate(edges):
            p1, p2 = positions[i], positions[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, alpha=edge_alphas[idx],
                    linewidth=edge_width, solid_capstyle='round')
    else:
        segments = [[positions[i], positions[j]] for i, j in edges]
        if segments:
            lc = LineCollection(segments, colors=color, alpha=0.45,
                                linewidths=edge_width)
            ax.add_collection(lc)

    # Draw nodes
    ax.scatter(positions[:, 0], positions[:, 1],
               c=color, s=node_size, alpha=node_alpha,
               edgecolors='none', zorder=5)


def generate_babel_banner(save_path):
    """Generate the Babel banner — architectural elevation on parchment."""

    fig = plt.figure(figsize=(16, 8.4), facecolor=BG_PARCHMENT)

    # Single axes spanning the full figure
    ax = fig.add_axes([0.03, 0.10, 0.94, 0.76], facecolor='none')
    ax.set_aspect('equal')
    ax.axis('off')

    # --- LEFT TOWER: Cubic, fragmenting ---
    nx_c, nz_c = 6, 16
    c_pos = cubic_tower_2d(nx_c, nz_c)
    c_edges = cubic_edges_2d(nx_c, nz_c)

    # Shift left
    c_pos[:, 0] -= 1.0

    # Scatter top
    c_pos_s, c_edge_alphas = scatter_top(
        c_pos, c_edges, z_frac=0.62, strength=2.5, seed=42
    )

    # Separate base and top for different rendering
    z_max_c = c_pos[:, 1].max()
    z_thresh_c = z_max_c * 0.62
    base_mask = c_pos[:, 1] <= z_thresh_c
    top_mask = ~base_mask

    # Draw edges
    for idx, (i, j) in enumerate(c_edges):
        p1, p2 = c_pos_s[i], c_pos_s[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=CUBIC_COLOR, alpha=c_edge_alphas[idx],
                linewidth=0.7, solid_capstyle='round')

    # Base nodes
    ax.scatter(c_pos_s[base_mask, 0], c_pos_s[base_mask, 1],
               c=CUBIC_COLOR, s=14, alpha=0.85, edgecolors='none', zorder=5)

    # Top nodes — fading
    ax.scatter(c_pos_s[top_mask, 0], c_pos_s[top_mask, 1],
               c=DUST_COLOR, s=8, alpha=0.3, edgecolors='none', zorder=5)

    # Drifting dust particles above the tower
    rng = np.random.RandomState(99)
    n_dust = 30
    cx_tower = (c_pos[:, 0].max() + c_pos[:, 0].min()) / 2
    dust_x = cx_tower + rng.normal(0, 3.5, n_dust)
    dust_y = z_max_c + rng.uniform(1.5, 7, n_dust)
    dust_sizes = rng.uniform(1, 5, n_dust)
    ax.scatter(dust_x, dust_y, c=DUST_COLOR, s=dust_sizes,
               alpha=0.12, edgecolors='none', zorder=4)

    # --- RIGHT TOWER: FCC, stable ---
    nx_f, nz_f = 7, 18
    f_pos = fcc_tower_2d(nx_f, nz_f)
    f_edges = fcc_edges_2d(f_pos, cutoff=1.1)

    # Shift right
    f_pos[:, 0] += 12.0

    draw_lattice_2d(ax, f_pos, f_edges, FCC_COLOR,
                    node_size=10, edge_width=0.5, node_alpha=0.75)

    # --- Set limits ---
    ax.set_xlim(-6, 22)
    ax.set_ylim(-1.5, 23)

    # --- Thin horizontal line separating towers (ground line) ---
    ax.axhline(y=-0.5, color=TEXT_MID, linewidth=0.3, alpha=0.3,
               xmin=0.02, xmax=0.98)

    # --- Text ---
    # Title — serif, dark, centered
    fig.text(0.5, 0.94, 'WHAT  THE  TOWER  WAS  MADE  OF',
             ha='center', va='top', fontsize=28, fontweight='bold',
             color=TEXT_DARK, fontfamily='serif', alpha=0.9)

    # Subtitle
    fig.text(0.5, 0.04, 'the language before babel was geometry',
             ha='center', va='bottom', fontsize=13, color=TEXT_MID,
             fontfamily='serif', style='italic', alpha=0.7)

    # Tower labels
    fig.text(0.22, 0.13, 'the tower of words',
             ha='center', va='top', fontsize=11, color=CUBIC_COLOR,
             fontfamily='serif', style='italic', alpha=0.7)

    fig.text(0.78, 0.13, 'the pattern which connects',
             ha='center', va='top', fontsize=11, color=FCC_COLOR,
             fontfamily='serif', style='italic', alpha=0.7)

    # Connection count labels (small, technical, below tower labels)
    fig.text(0.22, 0.105, '6 connections per node',
             ha='center', va='top', fontsize=8, color=TEXT_MID,
             fontfamily='monospace', alpha=0.45)

    fig.text(0.78, 0.105, '12 connections per node',
             ha='center', va='top', fontsize=8, color=TEXT_MID,
             fontfamily='monospace', alpha=0.45)

    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=BG_PARCHMENT, pad_inches=0.3)
    plt.close()
    print(f'Babel banner saved: {save_path}')


if __name__ == '__main__':
    import os
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
    os.makedirs(out_dir, exist_ok=True)
    generate_babel_banner(os.path.join(out_dir, 'babel-banner.png'))
