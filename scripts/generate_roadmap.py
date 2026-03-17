"""Generate the three-column roadmap visual for investors/stakeholders."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# 8-Law Weave palette
CUBIC = '#3D3D6B'
FCC = '#B34444'
BG = '#0a0a0f'
GOLD = '#c4a265'
DIM_GOLD = '#8a7045'
DARK_GOLD = '#4a3a2a'

def draw_roadmap():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'TASUMER MAF — TeLoRA Roadmap',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=GOLD, fontfamily='serif')

    # Subtitle: three-paper arc
    ax.text(7, 7.0, 'Replace → Mechanism → Embrace',
            ha='center', va='center', fontsize=11, color=DIM_GOLD,
            fontfamily='serif', style='italic')

    # Column headers
    cols = [
        (2.3, 'PROVEN', '(Papers 1-3)', GOLD),
        (7.0, 'BUILDING', '(Now)', '#aaaacc'),
        (11.7, 'FORWARD VISION', '', FCC),
    ]
    for x, title, sub, color in cols:
        ax.text(x, 6.3, title, ha='center', va='center',
                fontsize=14, fontweight='bold', color=color, fontfamily='serif')
        if sub:
            ax.text(x, 5.95, sub, ha='center', va='center',
                    fontsize=9, color=DIM_GOLD, fontfamily='serif')

    # Vertical separator lines
    for x in [4.65, 9.35]:
        ax.plot([x, x], [1.0, 6.0], color=DARK_GOLD, linewidth=0.5, alpha=0.5)

    # Column content
    proven = [
        'Rung 0-4 library',
        'Paper 1: 2.3× connectivity',
        'Paper 2: 6.1× amplification',
        'Paper 3: TeLoRA bridge',
        '255 tests, PyPI, MPL-2.0',
    ]
    building = [
        'Exp 3 cybernetic loop',
        'Hackathon: Hermes Agent',
        'Flimmer integration',
        'Community tools',
        'Consulting pilots',
    ]
    forward = [
        'SuperLoRA product',
        'GNN on lattice',
        'Cross-modal transit',
        'World model architecture',
        '24-cell / D₄ geometry',
    ]

    for items, x, color in [(proven, 2.3, GOLD), (building, 7.0, '#aaaacc'), (forward, 11.7, FCC)]:
        for i, item in enumerate(items):
            y = 5.3 - i * 0.65
            # Bullet
            ax.plot(x - 1.5, y, 'o', color=color, markersize=4, alpha=0.6)
            ax.text(x - 1.3, y, item, ha='left', va='center',
                    fontsize=10, color=color, fontfamily='serif', alpha=0.85)

    # Bottom row: revenue model
    revenue = [
        (2.3, 'Open source credibility', GOLD),
        (7.0, 'Revenue: consulting', '#aaaacc'),
        (11.7, 'Revenue: product', FCC),
    ]
    for x, label, color in revenue:
        # Box
        rect = mpatches.FancyBboxPatch(
            (x - 1.8, 1.15), 3.6, 0.55,
            boxstyle='round,pad=0.1',
            facecolor=color, alpha=0.08,
            edgecolor=color, linewidth=0.8
        )
        ax.add_patch(rect)
        ax.text(x, 1.42, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color, fontfamily='serif')

    # Arrows between columns
    arrow_props = dict(arrowstyle='->', color=DIM_GOLD, lw=1.5, alpha=0.4)
    ax.annotate('', xy=(5.0, 3.5), xytext=(4.3, 3.5), arrowprops=arrow_props)
    ax.annotate('', xy=(9.7, 3.5), xytext=(9.0, 3.5), arrowprops=arrow_props)

    # Cybernetic loop arrow (bottom)
    ax.annotate('', xy=(4.5, 1.42), xytext=(3.9, 1.42),
                arrowprops=dict(arrowstyle='->', color=DIM_GOLD, lw=1.0, alpha=0.3))
    ax.annotate('', xy=(9.2, 1.42), xytext=(8.6, 1.42),
                arrowprops=dict(arrowstyle='->', color=DIM_GOLD, lw=1.0, alpha=0.3))

    # Key metrics callout
    metrics = [
        ('4.6×', 'cross-channel\ncoupling'),
        ('84.5%', 'fingerprint\naccuracy'),
        ('36', 'parameters\nper layer'),
    ]
    for i, (num, label) in enumerate(metrics):
        x = 3.5 + i * 3.5
        ax.text(x, 0.55, num, ha='center', va='center',
                fontsize=16, fontweight='bold', color=GOLD, fontfamily='serif')
        ax.text(x, 0.18, label, ha='center', va='center',
                fontsize=7, color=DIM_GOLD, fontfamily='serif')

    # Attribution
    ax.text(7, -0.15, 'Promptcrafted LLC  •  tasumermaf.github.io/rhombic',
            ha='center', va='center', fontsize=8, color=DARK_GOLD, fontfamily='serif')

    plt.tight_layout(pad=0.5)
    out = Path('assets/paper3/roadmap.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == '__main__':
    draw_roadmap()
