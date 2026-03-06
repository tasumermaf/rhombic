"""Generate publication figures for Paper 2: Pure Number Architecture.

Figure 3: Amplification gradient (Fiedler ratio vs distribution, direction-weighted).
Figure 4: Consensus inversion across scale.
Figure 5: Eigenvalue spectra under 4 distributions (stem plot).
Figure 6: Corpus Fiedler percentile across 5 polytopes (horizontal bar).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Publication palette (from 8-Law Weave, matching Paper 1)
CUBIC_COLOR = '#3D3D6B'
FCC_COLOR = '#B34444'
ACCENT_ORANGE = '#FF8C42'
ACCENT_AZURE = '#4A90E2'

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


# ── Figure 3: Amplification Gradient ────────────────────────────────

def generate_figure3(save_path):
    """Bar chart: Fiedler ratio by distribution for direction-weighted (Exp 5).

    Shows the amplification gradient from uniform (2.3x) through corpus (6.1x)
    at scale 1000 (the most representative scale).
    """
    distributions = ['uniform', 'random', 'power_law', 'corpus']
    labels = ['Uniform', 'Random', 'Power-law', 'Corpus']

    # Direction-weighted Fiedler ratios at scale 1000 (from experiment_5 raw)
    ratios_1000 = [2.55, 3.65, 3.37, 6.11]

    # Edge-cycled Fiedler ratios at scale 1000 (from experiment_1 raw) for comparison
    ratios_1000_ec = [2.55, 2.64, 3.06, 3.11]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars_ec = ax.bar(x - width/2, ratios_1000_ec, width, color=CUBIC_COLOR,
                     label='Edge-cycled (Exp. 1)', alpha=0.85)
    bars_dw = ax.bar(x + width/2, ratios_1000, width, color=FCC_COLOR,
                     label='Direction-weighted (Exp. 5)', alpha=0.85)

    # Annotate direction-weighted bars
    for i, v in enumerate(ratios_1000):
        ax.text(x[i] + width/2, v + 0.08, f'{v:.1f}$\\times$',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=FCC_COLOR)

    ax.set_ylabel('Fiedler ratio (FCC / Cubic)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 7.5)
    ax.axhline(y=2.31, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(3.6, 2.31 + 0.1, 'Paper 1 baseline (2.3$\\times$)',
            fontsize=7, color='grey', ha='right')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Fiedler ratio amplification at scale 1{,}000', fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure 3 saved: {save_path}")


# ── Figure 4: Consensus Inversion ───────────────────────────────────

def generate_figure4(save_path):
    """Line chart: consensus speedup vs scale for corpus weights (Exp 5).

    Shows the inversion from 6.69x advantage at 125 to 0.73x at 1000.
    """
    scales = [125, 1000]

    # Consensus speedup (FCC/cubic rounds, so >1 = FCC faster)
    # From experiment_5 raw data
    consensus_uniform = [1.00, 0.93]
    consensus_random = [2.18, 0.78]
    consensus_power = [4.12, 0.58]
    consensus_corpus = [6.69, 0.73]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    ax.plot(scales, consensus_uniform, 'o-', color='grey', label='Uniform',
            linewidth=1.5, markersize=6, alpha=0.6)
    ax.plot(scales, consensus_random, 's-', color=ACCENT_AZURE, label='Random',
            linewidth=1.5, markersize=6)
    ax.plot(scales, consensus_power, '^-', color=ACCENT_ORANGE, label='Power-law',
            linewidth=1.5, markersize=6)
    ax.plot(scales, consensus_corpus, 'D-', color=FCC_COLOR, label='Corpus',
            linewidth=2.0, markersize=7)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.text(800, 1.06, 'parity', fontsize=7, color='black', alpha=0.5)

    ax.annotate(f'6.69$\\times$', xy=(125, 6.69), xytext=(200, 6.2),
                fontsize=8, fontweight='bold', color=FCC_COLOR,
                arrowprops=dict(arrowstyle='->', color=FCC_COLOR, lw=0.8))
    ax.annotate(f'0.73$\\times$', xy=(1000, 0.73), xytext=(850, 1.5),
                fontsize=8, fontweight='bold', color=FCC_COLOR,
                arrowprops=dict(arrowstyle='->', color=FCC_COLOR, lw=0.8))

    ax.set_xlabel('Lattice scale (nodes)')
    ax.set_ylabel('Consensus speedup (FCC / Cubic)')
    ax.set_title('Consensus inversion under direction-based weighting', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 7.5)
    ax.set_xlim(50, 1100)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure 4 saved: {save_path}")


# ── Figure 5: Spectral Stems ────────────────────────────────────────

def generate_figure5(save_path):
    """Stem plots: RD eigenvalue spectra under 4 distributions (Exp 4).

    Shows degeneracy breaking from uniform (6 distinct) to corpus (14 distinct).
    """
    # Full spectra from experiment_4 raw data
    spectra = {
        'Uniform': [0.0, 1.4384, 1.4384, 1.4384, 3.0, 3.0, 3.0, 3.0,
                     4.0, 4.0, 5.5616, 5.5616, 5.5616, 7.0],
        'Random':  [0.0, 0.5364, 0.8775, 1.0446, 1.3607, 1.7728, 1.8615,
                     2.0701, 2.5610, 2.7517, 3.4269, 3.8069, 3.9805, 4.7654],
        'Power-law': [0.0, 0.1041, 0.1244, 0.1281, 0.1731, 0.2389, 0.2548,
                       0.3350, 0.3936, 0.4711, 0.6054, 0.8545, 1.0617, 2.8073],
        'Corpus':  [0.0, 0.0863, 0.1228, 0.2081, 0.2918, 0.3852, 0.4447,
                     0.5469, 0.5836, 0.8551, 0.9294, 0.9983, 1.8541, 2.7547],
    }
    colors = ['grey', ACCENT_AZURE, ACCENT_ORANGE, FCC_COLOR]
    distinct_counts = [6, 14, 14, 14]

    fig, axes = plt.subplots(4, 1, figsize=(6.0, 6.0), sharex=True)

    for ax, (name, spec), color, dc in zip(axes, spectra.items(), colors, distinct_counts):
        markerline, stemlines, baseline = ax.stem(
            range(len(spec)), spec, linefmt='-', markerfmt='o', basefmt=' ')
        plt.setp(stemlines, color=color, linewidth=1.2)
        plt.setp(markerline, color=color, markersize=4)
        ax.set_ylabel('$\\lambda_i$', fontsize=9)
        ax.set_title(f'{name} ({dc}/14 distinct)', fontsize=9,
                     loc='left', pad=2)
        ax.set_ylim(-0.3, 7.5)
        ax.grid(True, alpha=0.15)
        # Highlight Fiedler value
        if len(spec) > 1:
            ax.axhline(y=spec[1], color=color, linestyle=':', linewidth=0.6, alpha=0.5)

    axes[-1].set_xlabel('Eigenvalue index')
    axes[-1].set_xticks(range(14))

    plt.suptitle('RD Laplacian spectrum under four weight distributions',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure 5 saved: {save_path}")


# ── Figure 6: Polytope Percentiles ──────────────────────────────────

def generate_figure6(save_path):
    """Horizontal bar chart: corpus Fiedler percentile across 5 graphs (Exp 7).

    Shows universality of corpus Fiedler suppression.
    """
    graphs = ['Random G(14,24)', '3-regular (16V)', 'K(4,6) (10V)',
              'RD (14V)', 'Cuboctahedron (12V)']
    percentiles = [5.94, 1.04, 0.36, 0.08, 0.02]
    colors = ['grey', ACCENT_AZURE, ACCENT_ORANGE, FCC_COLOR, CUBIC_COLOR]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    y_pos = np.arange(len(graphs))

    bars = ax.barh(y_pos, percentiles, color=colors, height=0.6, alpha=0.85)

    for i, (v, bar) in enumerate(zip(percentiles, bars)):
        ax.text(v + 0.15, i, f'{v:.2f}%', va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(graphs, fontsize=9)
    ax.set_xlabel('Corpus Fiedler percentile (vs. 10K random weights)')
    ax.set_title('Fiedler suppression is universal across 24-edge graphs', fontsize=11)
    ax.set_xlim(0, 8)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure 6 saved: {save_path}")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    generate_figure3(os.path.join(out_dir, 'fig3-amplification-gradient.pdf'))
    generate_figure3(os.path.join(out_dir, 'fig3-amplification-gradient.png'))
    generate_figure4(os.path.join(out_dir, 'fig4-consensus-inversion.pdf'))
    generate_figure4(os.path.join(out_dir, 'fig4-consensus-inversion.png'))
    generate_figure5(os.path.join(out_dir, 'fig5-spectral-stems.pdf'))
    generate_figure5(os.path.join(out_dir, 'fig5-spectral-stems.png'))
    generate_figure6(os.path.join(out_dir, 'fig6-polytope-percentiles.pdf'))
    generate_figure6(os.path.join(out_dir, 'fig6-polytope-percentiles.png'))

    print("\nAll Paper 2 figures generated.")
