#!/usr/bin/env python3
"""Generate Paper 3 Figure: Temporal Emergence of Block-Diagonal Structure.

Two-panel figure:
  Left: Co-planar vs cross-planar coupling magnitude over training steps
  Right: Co/cross ratio (log scale) over training steps

Three experiments overlaid: Qwen 7B, TinyLlama exp3, C-001.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Color palette from rhombic project
QWEN_COLOR = '#B34444'      # FCC_COLOR (Geometric Essence / prime 67)
TINY_COLOR = '#3D3D6B'      # CUBIC_COLOR (Fall of Neutral Events / prime 11)
C001_COLOR = '#6B8E23'      # OliveDrab for C-001

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 150,
})


def load_data():
    """Load temporal emergence data for all three experiments."""
    datasets = {}
    for name in ['exp3', 'exp3_tinyllama', 'C-001']:
        with open(f'results/{name}_temporal_emergence.json') as f:
            datasets[name] = json.load(f)
    return datasets


def main():
    datasets = load_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Temporal Emergence of Block-Diagonal Bridge Structure', fontsize=13, fontweight='bold')

    # --- Left panel: Co-planar and cross-planar coupling magnitudes ---
    for name, color, label in [
        ('exp3', QWEN_COLOR, 'Qwen 7B'),
        ('exp3_tinyllama', TINY_COLOR, 'TinyLlama 1.1B'),
        ('C-001', C001_COLOR, 'C-001 (identity init)'),
    ]:
        data = datasets[name]
        steps = [r['step'] for r in data]
        co = [r['co_mean'] for r in data]
        cross = [r['cross_mean'] for r in data]

        ax1.plot(steps, co, color=color, linewidth=1.5, label=f'{label} co-planar')
        ax1.plot(steps, cross, color=color, linewidth=1.0, linestyle='--', alpha=0.6, label=f'{label} cross-planar')

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Mean Coupling Magnitude')
    ax1.set_title('Co-planar vs Cross-planar Coupling')
    ax1.legend(fontsize=7, loc='upper left', ncol=1)
    ax1.set_xlim(0, 13000)
    ax1.axhline(y=0.001, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.annotate('cross-planar < 0.001', xy=(6000, 0.001), fontsize=7, color='gray', ha='center', va='bottom')

    # --- Right panel: Co/cross ratio on log scale ---
    for name, color, label in [
        ('exp3', QWEN_COLOR, 'Qwen 7B'),
        ('exp3_tinyllama', TINY_COLOR, 'TinyLlama 1.1B'),
        ('C-001', C001_COLOR, 'C-001 (identity init)'),
    ]:
        data = datasets[name]
        # Skip step 0 (0:0 ratio)
        valid = [r for r in data if r['step'] > 0 and r['ratio'] > 0]
        steps = [r['step'] for r in valid]
        ratios = [r['ratio'] for r in valid]

        ax2.semilogy(steps, ratios, color=color, linewidth=1.5, label=label, marker='.', markersize=2)

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Co/Cross Ratio')
    ax2.set_title('Axis Alignment Ratio (log scale)')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 13000)
    ax2.axhline(y=1, color='gray', linestyle=':', linewidth=0.5)
    ax2.annotate('No alignment (1:1)', xy=(6000, 1), fontsize=7, color='gray', ha='center', va='bottom')

    # Mark the lock point
    ax2.axvline(x=200, color='green', linestyle=':', linewidth=0.7, alpha=0.5)
    ax2.annotate('100% BD locked\n(step 200)', xy=(200, 100), fontsize=7, color='green',
                 ha='left', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig('results/fig_temporal_emergence.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/fig_temporal_emergence.pdf', bbox_inches='tight')
    print('Saved: results/fig_temporal_emergence.png and .pdf')

    # --- Bonus: single-panel version focusing on the first 1000 steps ---
    fig2, ax3 = plt.subplots(figsize=(7, 4))
    fig2.suptitle('Early Emergence: First 1000 Training Steps', fontsize=12, fontweight='bold')

    for name, color, label in [
        ('exp3', QWEN_COLOR, 'Qwen 7B'),
        ('exp3_tinyllama', TINY_COLOR, 'TinyLlama 1.1B'),
        ('C-001', C001_COLOR, 'C-001 (identity init)'),
    ]:
        data = datasets[name]
        valid = [r for r in data if 0 < r['step'] <= 1000 and r['ratio'] > 0]
        steps = [r['step'] for r in valid]
        ratios = [r['ratio'] for r in valid]

        ax3.semilogy(steps, ratios, color=color, linewidth=2, label=label, marker='o', markersize=4)

    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Co/Cross Ratio')
    ax3.legend(fontsize=9)
    ax3.axhline(y=1, color='gray', linestyle=':', linewidth=0.5)
    ax3.axvline(x=200, color='green', linestyle=':', linewidth=0.7, alpha=0.5)
    ax3.annotate('100% BD locked', xy=(220, 10), fontsize=8, color='green')

    plt.tight_layout()
    plt.savefig('results/fig_early_emergence.png', dpi=300, bbox_inches='tight')
    print('Saved: results/fig_early_emergence.png')


if __name__ == '__main__':
    main()
