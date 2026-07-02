#!/usr/bin/env python3
"""Generate updated init convergence and H2 comparison figures.

Uses the latest temporal emergence JSON data from C-001, C-002, C-003.
Also generates H2 (n=4) comparison figure.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Palette from rhombic
CUBIC_COLOR = '#3D3D6B'
FCC_COLOR = '#B34444'
ACCENT = '#4A8B3F'
GRAY = '#666666'

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_json(name):
    path = os.path.join(RESULTS, name)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    if isinstance(d, list):
        return d
    elif isinstance(d, dict) and 'checkpoints' in d:
        return d['checkpoints']
    return d


def plot_init_convergence():
    """Four-panel: all 3 inits — co-planar, cross-planar (log), ratio (log), BD%."""
    c001 = load_json('C-001_temporal_emergence.json')
    c002 = load_json('C-002_temporal_emergence.json')
    c003 = load_json('C-003_temporal_emergence.json')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for data, label, color, ls in [
        (c001, 'C-001 (identity)', CUBIC_COLOR, '-'),
        (c002, 'C-002 (geometric)', FCC_COLOR, '--'),
        (c003, 'C-003 (corpus-coupled)', ACCENT, '-.'),
    ]:
        steps = [d['step'] for d in data]
        co = [d['co_mean'] for d in data]
        cross = [d['cross_mean'] for d in data]
        ratio = [d['ratio'] for d in data]
        bd = [d['block_diag_pct'] for d in data]

        axes[0, 0].plot(steps, co, color=color, ls=ls, lw=2, label=label, marker='o', ms=3)
        axes[0, 1].plot(steps, cross, color=color, ls=ls, lw=2, label=label, marker='o', ms=3)
        axes[1, 0].plot(steps, ratio, color=color, ls=ls, lw=2, label=label, marker='o', ms=3)
        axes[1, 1].plot(steps, bd, color=color, ls=ls, lw=2, label=label, marker='o', ms=3)

    axes[0, 0].set_ylabel('Co-planar coupling (mean)')
    axes[0, 0].set_title('Co-planar Growth')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_ylabel('Cross-planar coupling (mean)')
    axes[0, 1].set_title('Cross-planar Decay')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_ylabel('Co/Cross Ratio')
    axes[1, 0].set_title('Coupling Ratio')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_ylabel('Block-diagonal %')
    axes[1, 1].set_title('BD Percentage')
    axes[1, 1].set_ylim(-5, 105)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('Training step')

    fig.suptitle('Init Convergence — All Three Initializations (Updated)', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(RESULTS, f'fig_init_convergence_comprehensive.{ext}'), dpi=200)
    plt.close(fig)
    print(f"Saved fig_init_convergence_comprehensive.png/pdf")
    print(f"  C-001: {len(c001)} checkpoints, max step {c001[-1]['step']}")
    print(f"  C-002: {len(c002)} checkpoints, max step {c002[-1]['step']}")
    print(f"  C-003: {len(c003)} checkpoints, max step {c003[-1]['step']}")


def plot_convergence_delta():
    """Three-panel: co-planar delta, ratio delta, and co-planar overlay."""
    c001 = load_json('C-001_temporal_emergence.json')
    c002 = load_json('C-002_temporal_emergence.json')
    c003 = load_json('C-003_temporal_emergence.json')

    # Find common steps across C-001 and C-003
    steps_001 = {d['step']: d for d in c001}
    steps_002 = {d['step']: d for d in c002}
    steps_003 = {d['step']: d for d in c003}

    # C-003 vs C-001 (skip step 0 where identity init has co=0)
    common_13 = sorted(s for s in set(steps_001.keys()) & set(steps_003.keys()) if s > 0)
    delta_co_13 = []
    delta_ratio_13 = []
    for s in common_13:
        co1, co3 = steps_001[s]['co_mean'], steps_003[s]['co_mean']
        r1, r3 = steps_001[s]['ratio'], steps_003[s]['ratio']
        denom_co = max(co1, co3, 1e-6)
        denom_r = max(r1, r3, 1e-6)
        delta_co_13.append(100 * abs(co1 - co3) / denom_co)
        delta_ratio_13.append(100 * abs(r1 - r3) / denom_r)

    # C-002 vs C-001 (skip step 0)
    common_12 = sorted(s for s in set(steps_001.keys()) & set(steps_002.keys()) if s > 0)
    delta_co_12 = []
    delta_ratio_12 = []
    for s in common_12:
        co1, co2 = steps_001[s]['co_mean'], steps_002[s]['co_mean']
        r1, r2 = steps_001[s]['ratio'], steps_002[s]['ratio']
        denom_co = max(co1, co2, 1e-6)
        denom_r = max(r1, r2, 1e-6)
        delta_co_12.append(100 * abs(co1 - co2) / denom_co)
        delta_ratio_12.append(100 * abs(r1 - r2) / denom_r)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Co-planar overlay
    for data, label, color, ls in [
        (c001, 'C-001 (identity)', CUBIC_COLOR, '-'),
        (c002, 'C-002 (geometric)', FCC_COLOR, '--'),
        (c003, 'C-003 (corpus)', ACCENT, '-.'),
    ]:
        steps = [d['step'] for d in data]
        co = [d['co_mean'] for d in data]
        axes[0].plot(steps, co, color=color, ls=ls, lw=2, label=label)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Co-planar coupling')
    axes[0].set_title('Co-planar Overlay')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Co-planar delta %
    axes[1].plot(common_13, delta_co_13, color=ACCENT, lw=2, label='C-003 vs C-001', marker='o', ms=3)
    axes[1].plot(common_12, delta_co_12, color=FCC_COLOR, lw=2, label='C-002 vs C-001', marker='s', ms=3)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Delta (%)')
    axes[1].set_title('Co-planar Delta')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Ratio delta %
    axes[2].plot(common_13, delta_ratio_13, color=ACCENT, lw=2, label='C-003 vs C-001', marker='o', ms=3)
    axes[2].plot(common_12, delta_ratio_12, color=FCC_COLOR, lw=2, label='C-002 vs C-001', marker='s', ms=3)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Delta (%)')
    axes[2].set_title('Ratio Delta')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    fig.suptitle('Init Convergence Proof — All Three Inits', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(RESULTS, f'fig_init_convergence_proof.{ext}'), dpi=200)
    plt.close(fig)
    print("Saved fig_init_convergence_proof.png/pdf")


def plot_h2_comparison():
    """Three-panel: H2 (n=4) vs C-001 (n=6) — Fiedler, BD%, val_loss."""
    # H2 data from results.json
    h2_path = os.path.join(RESULTS, 'channel-ablation', 'H-ch4', 'results.json')
    with open(h2_path, encoding='utf-8') as f:
        h2_data = json.load(f)
    h2_ckpts = h2_data['checkpoints']

    # H1 data
    h1_path = os.path.join(RESULTS, 'channel-ablation', 'H-ch3', 'results.json')
    with open(h1_path, encoding='utf-8') as f:
        h1_data = json.load(f)
    h1_ckpts = h1_data['checkpoints']

    # C-001 data
    c001 = load_json('C-001_temporal_emergence.json')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Val loss
    axes[0].plot([c['step'] for c in h1_ckpts], [c['val_loss'] for c in h1_ckpts],
                 color=ACCENT, lw=2, label='H1 (n=3)', alpha=0.7)
    axes[0].plot([c['step'] for c in h2_ckpts], [c['val_loss'] for c in h2_ckpts],
                 color=FCC_COLOR, lw=2, label='H2 (n=4)')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Val Loss: n=3 vs n=4')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Fiedler
    axes[1].plot([c['step'] for c in h1_ckpts], [c['fiedler_mean'] for c in h1_ckpts],
                 color=ACCENT, lw=2, label='H1 (n=3)', alpha=0.7)
    axes[1].plot([c['step'] for c in h2_ckpts], [c['fiedler_mean'] for c in h2_ckpts],
                 color=FCC_COLOR, lw=2, label='H2 (n=4)')
    axes[1].axhline(0.10, color=GRAY, ls=':', lw=1, label='n=6 converged (0.10)')
    axes[1].axhline(0.095, color=ACCENT, ls=':', lw=1, alpha=0.5, label='n=3 converged (0.095)')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Fiedler Eigenvalue')
    axes[1].set_title('Spectral Connectivity')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Eigenvalue spread (λ4/λ2 ratio for H2)
    eig_ratios = []
    eig_steps = []
    for entry in h2_data.get('feedback_log', []):
        eig_pattern = entry.get('eigenvalue_pattern', [])
        if len(eig_pattern) >= 4:
            # Sort by value, skip the ~0 one
            vals = sorted([e[0] for e in eig_pattern])
            if len(vals) >= 4 and vals[1] > 1e-10:
                eig_ratios.append(vals[3] / vals[1])
                eig_steps.append(entry['step'])

    if eig_ratios:
        axes[2].plot(eig_steps, eig_ratios, color=FCC_COLOR, lw=2, marker='o', ms=4, label='H2 λ₄/λ₂')
        axes[2].axhline(1.0, color=GRAY, ls=':', lw=1, label='Perfect degeneracy')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('λ₄ / λ₂ ratio')
        axes[2].set_title('Eigenvalue Spread (n=4)')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

    fig.suptitle('Channel Ablation: H2 (n=4, no contrastive) — Early Results', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(RESULTS, 'channel-ablation',
                                 f'fig_channel_ablation_h2_early.{ext}'), dpi=200)
    plt.close(fig)
    print("Saved channel-ablation/fig_channel_ablation_h2_early.png/pdf")


if __name__ == '__main__':
    plot_init_convergence()
    plot_convergence_delta()
    plot_h2_comparison()
