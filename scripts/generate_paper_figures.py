"""Generate publication figures for the rhombic paper.

Figure 1: Voronoi cells (cube vs rhombic dodecahedron) with neighbor connectivity.
Figure 2: Benchmark summary dashboard (4-panel, white background, publication style).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import os

# Publication palette (from 8-Law Weave)
CUBIC_COLOR = '#3D3D6B'
FCC_COLOR = '#B34444'
CUBIC_LIGHT = '#9999BB'
FCC_LIGHT = '#CC8888'
EDGE_COLOR = '#333333'

# Publication rcParams
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
})


# ── Figure 1: Voronoi Cells ──────────────────────────────────────────

def cube_vertices():
    """Unit cube centered at origin."""
    v = np.array([[i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]],
                 dtype=float) * 0.5
    return v


def rhombic_dodecahedron_vertices():
    """Rhombic dodecahedron centered at origin.

    14 vertices: 8 trivalent (cube vertices at ±1/√3) and
    6 tetravalent (axis vertices at ±2/√3).
    Scaled so face-center distance matches the cube for visual comparison.
    """
    s = 1.0 / np.sqrt(3)
    # 8 trivalent vertices (cube corners)
    trivalent = np.array([
        [i, j, k] for i in [-s, s] for j in [-s, s] for k in [-s, s]
    ])
    # 6 tetravalent vertices (axis points)
    t = 2.0 * s
    tetravalent = np.array([
        [t, 0, 0], [-t, 0, 0],
        [0, t, 0], [0, -t, 0],
        [0, 0, t], [0, 0, -t],
    ])
    return np.vstack([trivalent, tetravalent])


def get_faces_from_hull(vertices):
    """Get face polygons from convex hull."""
    hull = ConvexHull(vertices)
    faces = []
    for simplex in hull.simplices:
        faces.append(vertices[simplex])
    return faces


def plot_polyhedron(ax, vertices, color, alpha=0.15, edge_alpha=0.6):
    """Plot a convex polyhedron with transparent faces and visible edges."""
    hull = ConvexHull(vertices)
    # Draw faces
    for simplex in hull.simplices:
        tri = vertices[simplex]
        poly = Poly3DCollection([tri], alpha=alpha, facecolor=color,
                                edgecolor=EDGE_COLOR, linewidth=0.5)
        ax.add_collection3d(poly)


def draw_neighbor_lines(ax, center, offsets, color, alpha=0.4, linewidth=1.0):
    """Draw lines from center to neighbor positions."""
    for offset in offsets:
        end = center + offset
        ax.plot([center[0], end[0]],
                [center[1], end[1]],
                [center[2], end[2]],
                color=color, alpha=alpha, linewidth=linewidth, linestyle='--')
        ax.scatter(*end, color=color, s=15, alpha=0.7, zorder=5)


def generate_figure1(save_path):
    """Side-by-side Voronoi cells: cube (6 neighbors) vs rhombic dodecahedron (12 neighbors)."""
    fig = plt.figure(figsize=(7.0, 3.2))

    # Left panel: Cube
    ax1 = fig.add_subplot(121, projection='3d')
    cube_v = cube_vertices()
    plot_polyhedron(ax1, cube_v, CUBIC_COLOR, alpha=0.12)

    # Show 6 neighbor directions
    cubic_offsets = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float) * 0.65
    draw_neighbor_lines(ax1, np.zeros(3), cubic_offsets, CUBIC_COLOR, alpha=0.5)
    ax1.scatter(0, 0, 0, color=CUBIC_COLOR, s=40, zorder=10)

    ax1.set_title('Cube — 6 neighbors', fontweight='bold', color=CUBIC_COLOR, pad=0)
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_zlim(-1.0, 1.0)
    ax1.set_axis_off()
    ax1.view_init(elev=20, azim=35)

    # Right panel: Rhombic dodecahedron
    ax2 = fig.add_subplot(122, projection='3d')
    rd_v = rhombic_dodecahedron_vertices()
    plot_polyhedron(ax2, rd_v, FCC_COLOR, alpha=0.12)

    # Show 12 neighbor directions (FCC offsets, scaled for visibility)
    fcc_offsets = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    ], dtype=float) * 0.45
    draw_neighbor_lines(ax2, np.zeros(3), fcc_offsets, FCC_COLOR, alpha=0.5)
    ax2.scatter(0, 0, 0, color=FCC_COLOR, s=40, zorder=10)

    ax2.set_title('Rhombic dodecahedron — 12 neighbors', fontweight='bold',
                  color=FCC_COLOR, pad=0)
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_zlim(-1.0, 1.0)
    ax2.set_axis_off()
    ax2.view_init(elev=20, azim=35)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure 1 saved: {save_path}")


# ── Figure 2: Benchmark Dashboard ────────────────────────────────────

def generate_figure2(save_path):
    """4-panel benchmark summary from live data."""
    from rhombic.benchmark import run_suite

    print("Running benchmark suite for Figure 2...")
    results = run_suite([125, 1000, 4000])

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    width = 0.32

    # ── Panel A: Average Shortest Path ──
    ax = axes[0, 0]
    data = [r for r in results if r.cubic_avg_path is not None]
    labels = [f"{r.scale}" for r in data]
    x = np.arange(len(labels))
    c_vals = [r.cubic_avg_path for r in data]
    f_vals = [r.fcc_avg_path for r in data]
    ax.bar(x - width/2, c_vals, width, color=CUBIC_COLOR, label='Cubic (SC)')
    ax.bar(x + width/2, f_vals, width, color=FCC_COLOR, label='FCC')
    for i, r in enumerate(data):
        pct = (1 - r.fcc_avg_path / r.cubic_avg_path) * 100
        ax.annotate(f'{pct:.0f}%$\\downarrow$', xy=(x[i] + width/2, f_vals[i]),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=7, color=FCC_COLOR)
    ax.set_ylabel('Avg. shortest path')
    ax.set_title('(a) Routing efficiency', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Approximate node count')
    ax.legend(fontsize=7, loc='upper left')

    # ── Panel B: Algebraic Connectivity ──
    ax = axes[0, 1]
    data_f = [r for r in results if r.cubic_fiedler is not None]
    labels_f = [f"{r.scale}" for r in data_f]
    x_f = np.arange(len(labels_f))
    c_fied = [r.cubic_fiedler for r in data_f]
    f_fied = [r.fcc_fiedler for r in data_f]
    ax.bar(x_f - width/2, c_fied, width, color=CUBIC_COLOR, label='Cubic (SC)')
    ax.bar(x_f + width/2, f_fied, width, color=FCC_COLOR, label='FCC')
    for i, r in enumerate(data_f):
        ratio = r.fcc_fiedler / r.cubic_fiedler
        ax.annotate(f'{ratio:.1f}$\\times$', xy=(x_f[i] + width/2, f_fied[i]),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=7, color=FCC_COLOR, fontweight='bold')
    ax.set_ylabel('Fiedler value')
    ax.set_title('(b) Algebraic connectivity', fontsize=10)
    ax.set_xticks(x_f)
    ax.set_xticklabels(labels_f)
    ax.set_xlabel('Approximate node count')
    ax.legend(fontsize=7, loc='upper left')

    # ── Panel C: Diameter ──
    ax = axes[1, 0]
    data_d = [r for r in results if r.cubic_diameter is not None]
    labels_d = [f"{r.scale}" for r in data_d]
    x_d = np.arange(len(labels_d))
    c_diam = [r.cubic_diameter for r in data_d]
    f_diam = [r.fcc_diameter for r in data_d]
    ax.bar(x_d - width/2, c_diam, width, color=CUBIC_COLOR, label='Cubic (SC)')
    ax.bar(x_d + width/2, f_diam, width, color=FCC_COLOR, label='FCC')
    for i in range(len(data_d)):
        pct = (1 - f_diam[i] / c_diam[i]) * 100
        ax.annotate(f'{pct:.0f}%$\\downarrow$', xy=(x_d[i] + width/2, f_diam[i]),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=7, color=FCC_COLOR)
    ax.set_ylabel('Diameter (hops)')
    ax.set_xlabel('Approximate node count')
    ax.set_title('(c) Worst-case latency', fontsize=10)
    ax.set_xticks(x_d)
    ax.set_xticklabels(labels_d)
    ax.legend(fontsize=7, loc='upper left')

    # ── Panel D: Fault Tolerance ──
    ax = axes[1, 1]
    r_largest = max(results, key=lambda r: r.scale)
    steps = len(r_largest.cubic_fault_curve)
    removal_pct = np.linspace(0, 50, steps)
    ax.plot(removal_pct, r_largest.cubic_fault_curve, '-', color=CUBIC_COLOR,
            label='Cubic (SC)', linewidth=1.5)
    ax.plot(removal_pct, r_largest.fcc_fault_curve, '-', color=FCC_COLOR,
            label='FCC', linewidth=1.5)
    ax.set_xlabel('Nodes removed (%)')
    ax.set_ylabel('Fraction connected')
    ax.set_title(f'(d) Fault tolerance (~{r_largest.scale} nodes)', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure 2 saved: {save_path}")


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    generate_figure1(os.path.join(out_dir, 'fig1-voronoi-cells.pdf'))
    generate_figure1(os.path.join(out_dir, 'fig1-voronoi-cells.png'))
    generate_figure2(os.path.join(out_dir, 'fig2-graph-benchmarks.pdf'))
    generate_figure2(os.path.join(out_dir, 'fig2-graph-benchmarks.png'))
