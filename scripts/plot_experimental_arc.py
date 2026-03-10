#!/usr/bin/env python
"""
Paper 3, Graphic 3: Full Experimental Arc
Directed graph showing the 7 experiments, their motivation chain,
key metrics, and the honest NULL result at Exp 2.5.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────
CUBIC     = '#3D3D6B'
FCC       = '#B34444'
GOLD      = '#C49A2A'
TEAL      = '#2A7A7A'
NULL_RED  = '#CC2222'
BG_LIGHT  = '#F8F7F4'
EDGE_CLR  = '#555555'

# ── Experiment data ──────────────────────────────────────────────────────
experiments = [
    {
        'id': 'Exp 1',
        'title': 'Topology\nBenchmarks',
        'metric': '2.3x connectivity',
        'status': 'complete',
        'pos': (1.0, 4.0),
    },
    {
        'id': 'Exp 2',
        'title': 'Spatial\nOperations',
        'metric': '30% shorter paths',
        'status': 'complete',
        'pos': (3.5, 4.0),
    },
    {
        'id': 'Exp 2.5',
        'title': 'Attention\nWeights',
        'metric': 'NULL',
        'status': 'null',
        'pos': (6.0, 4.0),
    },
    {
        'id': 'Exp 2.6',
        'title': 'Task\nFingerprinting',
        'metric': '84.5% accuracy',
        'status': 'complete',
        'pos': (6.0, 2.0),
    },
    {
        'id': 'Exp 2.7',
        'title': 'Overfitting\nDetection',
        'metric': 'r = 0.888',
        'status': 'complete',
        'pos': (8.5, 2.0),
    },
    {
        'id': 'Exp 3',
        'title': 'Cybernetic\nTraining',
        'metric': 'Closed-loop control',
        'status': 'complete',
        'pos': (8.5, 4.0),
    },
    {
        'id': 'Exp 3a',
        'title': 'Overfitting\nPhase Transition',
        'metric': 'Deviation → gap',
        'status': 'complete',
        'pos': (11.0, 3.0),
    },
]

# Edges: (from_id, to_id, label)
edges = [
    ('Exp 1', 'Exp 2', 'Structure\nvalidated'),
    ('Exp 2', 'Exp 2.5', 'Do adapters\nprefer FCC?'),
    ('Exp 2.5', 'Exp 2.6', 'Pivot: topology\nas fingerprint'),
    ('Exp 2.5', 'Exp 3', 'Pivot: lattice\nas feedback'),
    ('Exp 2.6', 'Exp 2.7', 'Deviation\ndetects overfit'),
    ('Exp 2.7', 'Exp 3a', 'Phase\ntransition'),
    ('Exp 3', 'Exp 3a', 'Verify under\ncontrolled overfit'),
]

# ── Figure ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=150)
fig.patch.set_facecolor(BG_LIGHT)
ax.set_facecolor(BG_LIGHT)

# Build position lookup
pos_map = {e['id']: e['pos'] for e in experiments}

# ── Draw edges first (behind nodes) ─────────────────────────────────────
for src, dst, label in edges:
    x0, y0 = pos_map[src]
    x1, y1 = pos_map[dst]

    # Compute midpoint for label
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2

    # Draw arrow
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    # Shorten arrows to not overlap with boxes
    shrink = 0.75 / dist
    sx0 = x0 + dx * shrink
    sy0 = y0 + dy * shrink
    sx1 = x1 - dx * shrink
    sy1 = y1 - dy * shrink

    ax.annotate('', xy=(sx1, sy1), xytext=(sx0, sy0),
                arrowprops=dict(arrowstyle='->', color=EDGE_CLR,
                                linewidth=1.8, connectionstyle='arc3,rad=0.05'))

    # Edge label with offset perpendicular to the edge direction
    perp_x = -dy / dist * 0.25
    perp_y = dx / dist * 0.25
    ax.text(mx + perp_x, my + perp_y, label,
            fontsize=7, ha='center', va='center', color=EDGE_CLR,
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.85))

# ── Draw nodes ───────────────────────────────────────────────────────────
BOX_W = 1.3
BOX_H = 1.4

for exp in experiments:
    x, y = exp['pos']
    is_null = exp['status'] == 'null'

    # Box colors
    if is_null:
        face = '#FFEEEE'
        edge = NULL_RED
        edge_lw = 2.5
    else:
        face = 'white'
        edge = CUBIC
        edge_lw = 2.0

    # Draw box
    box = FancyBboxPatch((x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
                          boxstyle='round,pad=0.1',
                          facecolor=face, edgecolor=edge,
                          linewidth=edge_lw, zorder=5)
    ax.add_patch(box)

    # Experiment ID
    id_color = NULL_RED if is_null else CUBIC
    ax.text(x, y + 0.42, exp['id'], fontsize=10, fontweight='bold',
            ha='center', va='center', color=id_color, zorder=6)

    # Title
    ax.text(x, y + 0.05, exp['title'], fontsize=8.5, ha='center',
            va='center', color='#333333', zorder=6)

    # Metric
    metric_color = NULL_RED if is_null else FCC
    metric_weight = 'bold' if is_null else 'bold'
    metric_size = 9 if is_null else 8.5
    ax.text(x, y - 0.48, exp['metric'], fontsize=metric_size,
            fontweight=metric_weight, ha='center', va='center',
            color=metric_color, zorder=6)

    # NULL cross-out for exp 2.5
    if is_null:
        ax.plot([x - 0.5, x + 0.5], [y - 0.2, y + 0.2],
                color=NULL_RED, linewidth=2.5, alpha=0.3, zorder=7)
        ax.plot([x - 0.5, x + 0.2], [y + 0.2, y - 0.2],
                color=NULL_RED, linewidth=2.5, alpha=0.3, zorder=7)

# ── Legend annotations ───────────────────────────────────────────────────
# "Honest science" callout for the NULL
null_pos = pos_map['Exp 2.5']
ax.annotate('Honest science:\nno topology preference\ndetected in attention',
            xy=(null_pos[0] - 0.3, null_pos[1] - BOX_H/2 - 0.05),
            xytext=(null_pos[0] - 2.2, null_pos[1] - BOX_H/2 - 1.0),
            fontsize=8, ha='center', color=NULL_RED,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEEEE',
                      edgecolor=NULL_RED, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color=NULL_RED, lw=1.5))

# Title arc annotation
ax.annotate('', xy=(11.8, 4.5), xytext=(0.3, 4.5),
            arrowprops=dict(arrowstyle='->', color=GOLD, lw=2,
                            connectionstyle='arc3,rad=-0.15',
                            linestyle='--', alpha=0.4))
ax.text(6.0, 5.2, 'Motivation chain: each experiment drives the next',
        fontsize=9, ha='center', va='center', color=GOLD,
        fontstyle='italic', fontweight='bold')

# ── Formatting ───────────────────────────────────────────────────────────
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(0.0, 5.7)
ax.set_aspect('equal')
ax.axis('off')

fig.suptitle('Experimental Arc — From Benchmarks to Cybernetic Control',
             fontsize=15, fontweight='bold', y=0.98, color=CUBIC)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = Path(__file__).resolve().parent.parent / 'assets' / 'paper3' / 'experimental_arc.png'
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG_LIGHT)
plt.close(fig)
print(f'Saved: {out}')
