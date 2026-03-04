"""
Publication-quality plots for the lattice benchmark results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from rhombic.benchmark import BenchmarkResult

# Use a clean style
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
})

# 8-Law Weave palette (Vadrashinetal, March 2026)
# Each color carries a Primeval Law through its prime thread.
CUBIC_COLOR = '#3D3D6B'    # Fall of Neutral Events (11) — the unexamined default
FCC_COLOR = '#B34444'      # Geometric Essence (67) — the revealed structure

# Accent palette
SYNC_ORANGE = '#FF8C42'    # Synchronicity (89) — comparison points
ARROW_AZURE = '#4A90E2'    # Arrow of Complexity (17) — edges, evolution
JUSTICE_GREEN = '#52A352'  # Sole Atom (19) — performance wins
KAOS_GOLD = '#FFD700'      # Kaos (23) — when assumptions break
GRID_SILVER = '#C0C0C0'    # Time Matrix (29) — measurement substrate


def plot_path_comparison(results: list[BenchmarkResult], save_path: str | None = None):
    """Bar chart comparing average shortest path at each scale."""
    scales_with_paths = [r for r in results if r.cubic_avg_path is not None]
    if not scales_with_paths:
        print("No path data to plot.")
        return

    labels = [f"~{r.scale}" for r in scales_with_paths]
    cubic_vals = [r.cubic_avg_path for r in scales_with_paths]
    fcc_vals = [r.fcc_avg_path for r in scales_with_paths]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, cubic_vals, width, label='Cubic (6-connected)',
                   color=CUBIC_COLOR, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, fcc_vals, width, label='FCC (12-connected)',
                   color=FCC_COLOR, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Approximate node count')
    ax.set_ylabel('Average shortest path length')
    ax.set_title('Routing Efficiency: Cubic vs FCC Lattice')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add ratio annotations
    for i, r in enumerate(scales_with_paths):
        ratio = r.fcc_avg_path / r.cubic_avg_path
        ax.annotate(f'{ratio:.1%}', xy=(x[i] + width/2, fcc_vals[i]),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=9, color=FCC_COLOR)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_diameter_comparison(results: list[BenchmarkResult], save_path: str | None = None):
    """Bar chart comparing graph diameter at each scale."""
    scales_with_diam = [r for r in results if r.cubic_diameter is not None]
    if not scales_with_diam:
        return

    labels = [f"~{r.scale}" for r in scales_with_diam]
    cubic_vals = [r.cubic_diameter for r in scales_with_diam]
    fcc_vals = [r.fcc_diameter for r in scales_with_diam]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, cubic_vals, width, label='Cubic (6-connected)',
           color=CUBIC_COLOR, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, fcc_vals, width, label='FCC (12-connected)',
           color=FCC_COLOR, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Approximate node count')
    ax.set_ylabel('Graph diameter (worst-case hops)')
    ax.set_title('Worst-Case Latency: Cubic vs FCC Lattice')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_fiedler_comparison(results: list[BenchmarkResult], save_path: str | None = None):
    """Bar chart comparing algebraic connectivity."""
    scales_with_fiedler = [r for r in results if r.cubic_fiedler is not None]
    if not scales_with_fiedler:
        return

    labels = [f"~{r.scale}" for r in scales_with_fiedler]
    cubic_vals = [r.cubic_fiedler for r in scales_with_fiedler]
    fcc_vals = [r.fcc_fiedler for r in scales_with_fiedler]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, cubic_vals, width, label='Cubic (6-connected)',
           color=CUBIC_COLOR, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, fcc_vals, width, label='FCC (12-connected)',
           color=FCC_COLOR, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Approximate node count')
    ax.set_ylabel('Algebraic connectivity (Fiedler value)')
    ax.set_title('Structural Robustness: Cubic vs FCC Lattice')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add ratio annotations
    for i, r in enumerate(scales_with_fiedler):
        ratio = r.fcc_fiedler / r.cubic_fiedler
        ax.annotate(f'{ratio:.1f}×', xy=(x[i] + width/2, fcc_vals[i]),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=9, color=FCC_COLOR, fontweight='bold')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_fault_tolerance(results: list[BenchmarkResult], save_path: str | None = None):
    """Line plot showing degradation under random node removal."""
    # Use the largest-scale result
    r = max(results, key=lambda x: x.scale)

    steps = len(r.cubic_fault_curve)
    removal_pct = np.linspace(0, 50, steps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(removal_pct, r.cubic_fault_curve, '-o', color=CUBIC_COLOR,
            label='Cubic (6-connected)', markersize=3, linewidth=2)
    ax.plot(removal_pct, r.fcc_fault_curve, '-s', color=FCC_COLOR,
            label='FCC (12-connected)', markersize=3, linewidth=2)

    ax.set_xlabel('Nodes removed (%)')
    ax.set_ylabel('Fraction in largest connected component')
    ax.set_title(f'Fault Tolerance (~{r.scale} nodes): Random Node Removal')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_summary_dashboard(results: list[BenchmarkResult], save_path: str | None = None):
    """4-panel dashboard with all key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scales_with_data = [r for r in results if r.cubic_avg_path is not None]
    labels = [f"~{r.scale}" for r in scales_with_data]
    x = np.arange(len(labels))
    width = 0.35

    # Panel 1: Average shortest path
    ax = axes[0, 0]
    ax.bar(x - width/2, [r.cubic_avg_path for r in scales_with_data], width,
           label='Cubic', color=CUBIC_COLOR, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, [r.fcc_avg_path for r in scales_with_data], width,
           label='FCC', color=FCC_COLOR, edgecolor='black', linewidth=0.5)
    ax.set_title('Average Shortest Path')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)

    # Panel 2: Diameter
    ax = axes[0, 1]
    scales_diam = [r for r in results if r.cubic_diameter is not None]
    labels_d = [f"~{r.scale}" for r in scales_diam]
    x_d = np.arange(len(labels_d))
    ax.bar(x_d - width/2, [r.cubic_diameter for r in scales_diam], width,
           label='Cubic', color=CUBIC_COLOR, edgecolor='black', linewidth=0.5)
    ax.bar(x_d + width/2, [r.fcc_diameter for r in scales_diam], width,
           label='FCC', color=FCC_COLOR, edgecolor='black', linewidth=0.5)
    ax.set_title('Graph Diameter (worst-case)')
    ax.set_xticks(x_d)
    ax.set_xticklabels(labels_d)
    ax.legend(fontsize=9)

    # Panel 3: Algebraic connectivity
    ax = axes[1, 0]
    scales_f = [r for r in results if r.cubic_fiedler is not None]
    labels_f = [f"~{r.scale}" for r in scales_f]
    x_f = np.arange(len(labels_f))
    ax.bar(x_f - width/2, [r.cubic_fiedler for r in scales_f], width,
           label='Cubic', color=CUBIC_COLOR, edgecolor='black', linewidth=0.5)
    ax.bar(x_f + width/2, [r.fcc_fiedler for r in scales_f], width,
           label='FCC', color=FCC_COLOR, edgecolor='black', linewidth=0.5)
    ax.set_title('Algebraic Connectivity')
    ax.set_xticks(x_f)
    ax.set_xticklabels(labels_f)
    ax.legend(fontsize=9)
    for i, r in enumerate(scales_f):
        ratio = r.fcc_fiedler / r.cubic_fiedler
        ax.annotate(f'{ratio:.1f}×', xy=(x_f[i] + width/2, r.fcc_fiedler),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=9, color=FCC_COLOR, fontweight='bold')

    # Panel 4: Fault tolerance
    ax = axes[1, 1]
    r_largest = max(results, key=lambda x: x.scale)
    steps = len(r_largest.cubic_fault_curve)
    removal_pct = np.linspace(0, 50, steps)
    ax.plot(removal_pct, r_largest.cubic_fault_curve, '-o', color=CUBIC_COLOR,
            label='Cubic', markersize=3, linewidth=2)
    ax.plot(removal_pct, r_largest.fcc_fault_curve, '-s', color=FCC_COLOR,
            label='FCC', markersize=3, linewidth=2)
    ax.set_title(f'Fault Tolerance (~{r_largest.scale} nodes)')
    ax.set_xlabel('Nodes removed (%)')
    ax.set_ylabel('Fraction connected')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Cubic vs FCC Lattice: Graph Theory Benchmarks',
                fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


if __name__ == "__main__":
    from rhombic.benchmark import run_suite
    import os

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'rhombic', 'results')
    os.makedirs(out_dir, exist_ok=True)

    results = run_suite([125, 1000, 8000])

    plot_summary_dashboard(results, os.path.join(out_dir, 'dashboard.png'))
    plot_path_comparison(results, os.path.join(out_dir, 'avg_path.png'))
    plot_diameter_comparison(results, os.path.join(out_dir, 'diameter.png'))
    plot_fiedler_comparison(results, os.path.join(out_dir, 'fiedler.png'))
    plot_fault_tolerance(results, os.path.join(out_dir, 'fault_tolerance.png'))
    plt.show()
