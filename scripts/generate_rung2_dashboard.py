"""
Generate the Rung 2 spatial operations dashboard.

Runs benchmarks at three scales and produces a 2×3 panel visualization
comparing cubic vs FCC performance across all spatial operations.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rhombic.spatial import run_spatial_benchmark

# 8-Law Weave palette
CUBIC_COLOR = '#3D3D6B'
FCC_COLOR = '#B34444'
BG_DARK = '#0D0D0D'
TEXT_LIGHT = '#E8E8E0'
GRID_SILVER = '#C0C0C0'
ARROW_AZURE = '#4A90E2'
ACCENT_DIM = '#2A2A3A'

SCALES = [125, 1000, 8000]


def generate_dashboard(save_path='results/rung-2/dashboard.png'):
    matplotlib.rcParams.update({
        'font.family': 'monospace',
        'font.size': 9,
        'text.color': TEXT_LIGHT,
        'axes.labelcolor': TEXT_LIGHT,
        'axes.edgecolor': ACCENT_DIM,
        'xtick.color': GRID_SILVER,
        'ytick.color': GRID_SILVER,
        'figure.facecolor': BG_DARK,
        'axes.facecolor': BG_DARK,
    })

    print("Running spatial benchmarks...")
    results = []
    for scale in SCALES:
        print(f"  Scale ~{scale}...")
        r = run_spatial_benchmark(scale, n_queries=500)
        results.append(r)

    scales = [r.scale for r in results]
    scale_labels = [f"~{s}" for s in scales]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=BG_DARK)

    # ── Panel 1: Flood fill reach ────────────────────────────────
    ax = axes[0, 0]
    x = np.arange(len(scales))
    w = 0.35
    cubic_vals = [r.cubic_flood_reached for r in results]
    fcc_vals = [r.fcc_flood_reached for r in results]
    ax.bar(x - w/2, cubic_vals, w, color=CUBIC_COLOR, label='Cubic')
    ax.bar(x + w/2, fcc_vals, w, color=FCC_COLOR, label='FCC')
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_title('Flood Fill Reach (same hops)', color=TEXT_LIGHT, fontsize=11)
    ax.set_ylabel('Nodes reached')
    ax.legend(facecolor=BG_DARK, edgecolor=ACCENT_DIM)

    # ── Panel 2: Flood fill ratio ────────────────────────────────
    ax = axes[0, 1]
    ratios = [f/max(1, c) for f, c in zip(fcc_vals, cubic_vals)]
    ax.bar(x, ratios, 0.5, color=ARROW_AZURE)
    ax.axhline(y=1.0, color=GRID_SILVER, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_title('FCC / Cubic Flood Reach Ratio', color=TEXT_LIGHT, fontsize=11)
    ax.set_ylabel('Ratio (>1 = FCC better)')

    # ── Panel 3: NN query time ───────────────────────────────────
    ax = axes[0, 2]
    cubic_nn = [r.cubic_nn_time * 1000 for r in results]
    fcc_nn = [r.fcc_nn_time * 1000 for r in results]
    ax.bar(x - w/2, cubic_nn, w, color=CUBIC_COLOR, label='Cubic')
    ax.bar(x + w/2, fcc_nn, w, color=FCC_COLOR, label='FCC')
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_title('Nearest Neighbor Query (ms)', color=TEXT_LIGHT, fontsize=11)
    ax.set_ylabel('Time (ms)')
    ax.legend(facecolor=BG_DARK, edgecolor=ACCENT_DIM)

    # ── Panel 4: Sphere range query ──────────────────────────────
    ax = axes[1, 0]
    cubic_sp = [r.cubic_sphere_count for r in results]
    fcc_sp = [r.fcc_sphere_count for r in results]
    ax.bar(x - w/2, cubic_sp, w, color=CUBIC_COLOR, label='Cubic')
    ax.bar(x + w/2, fcc_sp, w, color=FCC_COLOR, label='FCC')
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_title('Sphere Query: Avg Nodes Returned', color=TEXT_LIGHT, fontsize=11)
    ax.set_ylabel('Nodes per query')
    ax.legend(facecolor=BG_DARK, edgecolor=ACCENT_DIM)

    # ── Panel 5: Spatial hash query time ─────────────────────────
    ax = axes[1, 1]
    cubic_h = [r.cubic_hash_query_time * 1000 for r in results]
    fcc_h = [r.fcc_hash_query_time * 1000 for r in results]
    ax.bar(x - w/2, cubic_h, w, color=CUBIC_COLOR, label='Cubic')
    ax.bar(x + w/2, fcc_h, w, color=FCC_COLOR, label='FCC')
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels)
    ax.set_title('Spatial Hash Query (ms)', color=TEXT_LIGHT, fontsize=11)
    ax.set_ylabel('Time (ms)')
    ax.legend(facecolor=BG_DARK, edgecolor=ACCENT_DIM)

    # ── Panel 6: Summary ratios ──────────────────────────────────
    ax = axes[1, 2]
    # Use the 8000-node results for the summary
    r = results[-1]
    metrics = ['Flood\nReach', 'NN\nSpeed', 'Hash\nSpeed', 'Sphere\nDensity']
    ratios = [
        r.fcc_flood_reached / max(1, r.cubic_flood_reached),
        r.cubic_nn_time / max(1e-9, r.fcc_nn_time),      # inverted: higher = FCC faster
        r.cubic_hash_query_time / max(1e-9, r.fcc_hash_query_time),
        r.fcc_sphere_count / max(1, r.cubic_sphere_count),
    ]
    colors = [FCC_COLOR if v > 1 else CUBIC_COLOR for v in ratios]
    bars = ax.bar(range(len(metrics)), ratios, 0.6, color=colors)
    ax.axhline(y=1.0, color=GRID_SILVER, linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_title(f'FCC Advantage at ~{r.scale} nodes', color=TEXT_LIGHT, fontsize=11)
    ax.set_ylabel('Ratio (>1 = FCC better)')
    # Add value labels
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}×', ha='center', va='bottom', color=TEXT_LIGHT, fontsize=9)

    fig.suptitle('R U N G  2 :  S P A T I A L  O P E R A T I O N S',
                 color=TEXT_LIGHT, fontsize=16, fontweight='bold',
                 fontfamily='monospace', y=0.98)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=BG_DARK,
                pad_inches=0.3)
    print(f"Dashboard saved: {save_path}")
    return results


if __name__ == '__main__':
    results = generate_dashboard()
