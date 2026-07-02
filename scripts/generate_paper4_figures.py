#!/usr/bin/env python3
"""Generate Paper 4 figures from experiment trajectory data.

Produces:
  fig_four_regimes.pdf       — 4-panel comparison of the four regimes
  fig_o001_emergence.pdf     — O-001 co/cross emergence curve
  fig_wl_r001_identity.pdf   — WL-001 vs R-001 trajectory overlay
  fig_spectral_attractor.pdf — Fiedler vs n across spectral-only runs
  fig_inverse_n.pdf          — Co/cross vs n for geometric topologies
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# TASUMER MAF color palette (from 8-law weave)
CUBIC_COLOR = '#3D3D6B'    # Fall of Neutral Events (prime 11)
FCC_COLOR = '#B34444'      # Geometric Essence (prime 67)
ACCENT1 = '#4A7C59'        # Green accent
ACCENT2 = '#C4A35A'        # Gold accent
GREY = '#888888'

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

outdir = 'paper/paper4/figures'
import os
os.makedirs(outdir, exist_ok=True)


# ============================================================
# Figure 1: O-001 Emergence Curve
# ============================================================
o001_steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
              5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
o001_co    = [0, 0.086409, 0.185839, 0.282744, 0.376200, 0.465134, 0.548625,
              0.626105, 0.696462, 0.759205, 0.814129, 0.861122, 0.900284,
              0.931891, 0.956740, 0.974851, 0.987204, 0.997083, 1.006926,
              1.016759, 1.026625]
o001_cross = [1e-10, 2.44168e-5, 2.57295e-5, 2.40106e-5, 2.34374e-5, 2.38457e-5,
              2.13579e-5, 1.96819e-5, 1.86880e-5, 1.62698e-5, 1.39064e-5,
              1.22904e-5, 9.8941e-6, 8.1367e-6, 6.0862e-6, 4.3138e-6,
              2.8926e-6, 2.9638e-6, 2.6637e-6, 2.8043e-6, 2.7794e-6]
o001_ratio = [r/c if c > 1e-12 else 0 for r, c in zip(o001_co, o001_cross)]
o001_ratio[0] = 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: coupling magnitudes
ax1.plot(o001_steps, o001_co, color=FCC_COLOR, linewidth=2, label='Co-planar mean')
ax1.plot(o001_steps, o001_cross, color=CUBIC_COLOR, linewidth=2, label='Cross-planar mean')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Mean |B[i,j]|')
ax1.set_title('O-001: Coupling Magnitude (n=4 Octahedral)')
ax1.set_yscale('log')
ax1.set_ylim(1e-6, 2)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: ratio
ax2.plot(o001_steps[1:], [r/1000 for r in o001_ratio[1:]], color=FCC_COLOR, linewidth=2)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Co/Cross Ratio (×10³)')
ax2.set_title('O-001: Block-Diagonal Emergence')
ax2.grid(True, alpha=0.3)
ax2.annotate(f'Final: {o001_ratio[-1]:,.0f}:1',
             xy=(10000, o001_ratio[-1]/1000), fontsize=9,
             ha='right', va='bottom', color=FCC_COLOR)

plt.tight_layout()
plt.savefig(f'{outdir}/fig_o001_emergence.pdf')
plt.savefig(f'{outdir}/fig_o001_emergence.png')
plt.close()
print('Saved fig_o001_emergence')


# ============================================================
# Figure 2: WL-001 vs R-001 Identity
# ============================================================
steps_1k = list(range(1000, 10001, 1000))

wl_cocross = [0.000649, 0.000241, 0.000139, 0.000081, 0.000050,
              0.000036, 0.000019, 0.000009, 0.000008, 0.000009]
r_cocross  = [0.000712, 0.000265, 0.000126, 0.000073, 0.000051,
              0.000035, 0.000018, 0.000009, 0.000008, 0.000009]

wl_fiedler = [0.000163, 0.000149, 0.000107, 0.000088, 0.000060,
              0.000045, 0.000025, 0.000013, 0.000012, 0.000013]
r_fiedler  = [0.000171, 0.000153, 0.000104, 0.000083, 0.000064,
              0.000043, 0.000026, 0.000012, 0.000012, 0.000012]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(steps_1k, wl_cocross, 'o-', color=CUBIC_COLOR, linewidth=2,
         markersize=4, label='WL-001 (random pairs)')
ax1.plot(steps_1k, r_cocross, 's--', color=ACCENT1, linewidth=2,
         markersize=4, label='R-001 (prime-theoretic)')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Co/Cross Ratio')
ax1.set_title('Co/Cross: Wrong-Labels vs Resonance')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(steps_1k, wl_fiedler, 'o-', color=CUBIC_COLOR, linewidth=2,
         markersize=4, label='WL-001')
ax2.plot(steps_1k, r_fiedler, 's--', color=ACCENT1, linewidth=2,
         markersize=4, label='R-001')
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Fiedler Value')
ax2.set_title('Fiedler: Wrong-Labels vs Resonance')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Incoherent Topologies Produce Identical Collapse', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig_wl_r001_identity.pdf')
plt.savefig(f'{outdir}/fig_wl_r001_identity.png')
plt.close()
print('Saved fig_wl_r001_identity')


# ============================================================
# Figure 3: Spectral Attractor — Fiedler vs n
# ============================================================
ns = [3, 4, 8, 12]
fiedlers = [0.0951, 0.0836, 0.0944, 0.1019]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(len(ns)), fiedlers, color=CUBIC_COLOR, width=0.6, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(ns)))
ax.set_xticklabels([f'n={n}' for n in ns])
ax.set_ylabel('Final Fiedler Value')
ax.set_title('Spectral Attractor: Universal Convergence Across Channel Counts')
ax.axhline(y=np.mean(fiedlers), color=FCC_COLOR, linestyle='--', linewidth=1.5,
           label=f'Mean: {np.mean(fiedlers):.4f}')
ax.set_ylim(0, 0.12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Annotate band
band_min, band_max = min(fiedlers), max(fiedlers)
ax.axhspan(band_min, band_max, alpha=0.1, color=FCC_COLOR)
ax.text(3.3, np.mean(fiedlers), f'Band: {(band_max-band_min)/np.mean(fiedlers)*100:.1f}%',
        fontsize=8, color=FCC_COLOR)

plt.tight_layout()
plt.savefig(f'{outdir}/fig_spectral_attractor.pdf')
plt.savefig(f'{outdir}/fig_spectral_attractor.png')
plt.close()
print('Saved fig_spectral_attractor')


# ============================================================
# Figure 4: Inverse n Relationship — Co/Cross vs n
# ============================================================
geo_ns = [4, 6, 6, 6, 8]
geo_labels = ['O-001\n(oct)', 'Seed-43\n(RD)', 'Seed-44\n(RD)', 'H-ch6\n(RD)', 'T-001r2\n(tes)']
geo_ratios = [473622, 73309, 70201, 70404, 41564]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [FCC_COLOR if n == 4 else ACCENT1 if n == 6 else CUBIC_COLOR for n in geo_ns]
bars = ax.bar(range(len(geo_ns)), [r/1000 for r in geo_ratios], color=colors,
              width=0.6, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(geo_ns)))
ax.set_xticklabels(geo_labels, fontsize=8)
ax.set_ylabel('Co/Cross Ratio (×10³)')
ax.set_title('Inverse n Relationship: Minimum Geometry = Maximum Signal')
ax.grid(True, alpha=0.3, axis='y')

for i, (bar, ratio) in enumerate(zip(bars, geo_ratios)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
            f'{ratio:,}:1', ha='center', va='bottom', fontsize=7)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=FCC_COLOR, label='n=4 (octahedral)'),
                   Patch(facecolor=ACCENT1, label='n=6 (RD)'),
                   Patch(facecolor=CUBIC_COLOR, label='n=8 (tesseract)')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(f'{outdir}/fig_inverse_n.pdf')
plt.savefig(f'{outdir}/fig_inverse_n.png')
plt.close()
print('Saved fig_inverse_n')


# ============================================================
# Figure 5: Four Regimes Summary (2x2)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Panel A: Block-Diagonal (O-001 trajectory)
ax = axes[0, 0]
ax.plot(o001_steps[1:], [r/1000 for r in o001_ratio[1:]], color=FCC_COLOR, linewidth=2)
ax.set_xlabel('Step')
ax.set_ylabel('Co/Cross (×10³)')
ax.set_title('(a) Block-Diagonal: O-001 (n=4)', fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel B: Spectral Attractor
ax = axes[0, 1]
ax.bar(range(4), fiedlers, color=CUBIC_COLOR, width=0.6, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels([f'n={n}' for n in ns])
ax.set_ylabel('Fiedler')
ax.set_title('(b) Spectral Attractor (no contrastive)', fontweight='bold')
ax.axhline(y=np.mean(fiedlers), color=FCC_COLOR, linestyle='--', linewidth=1)
ax.set_ylim(0, 0.12)
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Hierarchical Coherence (E-001)
ax = axes[1, 0]
e_steps = steps_1k
e_cocross = [1.052363, 1.055496, 1.056878, 1.067037, 1.083978,
             1.114441, 1.119224, 1.127105, 1.122120, 1.122651]
e_fiedler = [0.058502, 0.067400, 0.071980, 0.075147, 0.078044,
             0.078789, 0.080890, 0.082200, 0.082903, 0.083596]
ax.plot(e_steps, e_cocross, 'o-', color=ACCENT2, linewidth=2, markersize=3, label='Co/Cross')
ax_twin = ax.twinx()
ax_twin.plot(e_steps, e_fiedler, 's--', color=ACCENT1, linewidth=2, markersize=3, label='Fiedler')
ax.set_xlabel('Step')
ax.set_ylabel('Co/Cross Ratio', color=ACCENT2)
ax_twin.set_ylabel('Fiedler', color=ACCENT1)
ax.set_title('(c) Hierarchical Coherence: E-001', fontweight='bold')
ax.set_ylim(1.0, 1.15)
ax_twin.set_ylim(0.05, 0.09)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=7)

# Panel D: Collapse (WL-001 + R-001 overlay)
ax = axes[1, 1]
ax.plot(steps_1k, wl_cocross, 'o-', color=CUBIC_COLOR, linewidth=2,
        markersize=3, label='WL-001 (random)')
ax.plot(steps_1k, r_cocross, 's--', color=ACCENT1, linewidth=2,
        markersize=3, label='R-001 (prime)')
ax.set_xlabel('Step')
ax.set_ylabel('Co/Cross Ratio')
ax.set_title('(d) Connectivity Collapse: WL-001 = R-001', fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.suptitle('Four Regimes of Topology Programming', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{outdir}/fig_four_regimes.pdf')
plt.savefig(f'{outdir}/fig_four_regimes.png')
plt.close()
print('Saved fig_four_regimes')

# ============================================================
# Figure 6: 24-Cell BD Emergence (from results.json)
# ============================================================
import json
import os
_24c_path = os.path.join('results', 'channel-ablation', '24C-001', 'results.json')
if os.path.exists(_24c_path):
    with open(_24c_path) as f:
        _24c_data = json.load(f)
    _24c_ckpts = _24c_data.get('checkpoints', [])
    if len(_24c_ckpts) >= 3:
        _steps = [c['step'] for c in _24c_ckpts if c['step'] > 0]
        _fiedler = [c['fiedler_mean'] for c in _24c_ckpts if c['step'] > 0]
        _co_cross = [c.get('co_cross_ratio') for c in _24c_ckpts if c['step'] > 0]
        _val_loss = [c.get('val_loss', 0) for c in _24c_ckpts if c['step'] > 0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Left: Fiedler trajectory
        ax1.plot(_steps, _fiedler, 'o-', color=FCC_COLOR, linewidth=2, markersize=4)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Fiedler (algebraic connectivity)')
        ax1.set_title('(a) Fiedler Decline: 24-Cell (n=24)', fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Right: co/cross ratio (if measurable)
        _co_valid = [(s, c) for s, c in zip(_steps, _co_cross) if c is not None and c > 1]
        if _co_valid:
            ax2.plot([s for s, _ in _co_valid], [c for _, c in _co_valid],
                     'o-', color=ACCENT1, linewidth=2, markersize=4)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Co/Cross Ratio')
            ax2.set_title('(b) Co/Cross Emergence: 24-Cell', fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Co/Cross not yet measurable\n(too early in training)',
                     transform=ax2.transAxes, ha='center', va='center', fontsize=11,
                     color=GREY, style='italic')
            ax2.set_title('(b) Co/Cross: Pending', fontweight='bold')

        plt.suptitle('24-Cell D4 Root Polytope: 12+12 BD Emergence',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{outdir}/fig_24cell_emergence.pdf')
        plt.savefig(f'{outdir}/fig_24cell_emergence.png')
        plt.close()
        print('Saved fig_24cell_emergence')
    else:
        print(f'  24C-001: only {len(_24c_ckpts)} checkpoints, need >= 3 for figure')
else:
    print(f'  24C-001: results.json not found at {_24c_path} (run locally or copy from Hermes)')

# ============================================================
# Figure 7: FI-002 Bridge Initialization Independence
# ============================================================
_fi002_dir = os.path.join('results', 'fi-002')
_fi002_configs = ['P-000', 'P-001', 'P-002', 'P-CTRL']
_fi002_colors = [FCC_COLOR, ACCENT1, ACCENT2, GREY]
_fi002_labels = ['P-000 (canonical)', 'P-001 (max-diff)', 'P-002 (moderate)', 'P-CTRL (identity)']

_fi002_data = {}
for cname in _fi002_configs:
    rpath = os.path.join(_fi002_dir, cname, 'results.json')
    if os.path.exists(rpath):
        with open(rpath) as f:
            _fi002_data[cname] = json.load(f)

if len(_fi002_data) >= 3:  # Need at least 3 configs to make the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for cname, color, label in zip(_fi002_configs, _fi002_colors, _fi002_labels):
        if cname not in _fi002_data:
            continue
        cps = _fi002_data[cname].get('checkpoints', [])
        if not cps:
            continue
        steps = [c['step'] for c in cps]
        co_cross = [c.get('co_cross_ratio') for c in cps]
        fiedler = [c.get('fiedler_mean') for c in cps]

        # Left: co/cross ratio
        co_valid = [(s, c) for s, c in zip(steps, co_cross) if c is not None and c > 1]
        if co_valid:
            ax1.plot([s for s, _ in co_valid], [c for _, c in co_valid],
                     'o-', color=color, linewidth=2, markersize=3, label=label)

        # Right: Fiedler
        fied_valid = [(s, f) for s, f in zip(steps, fiedler) if f is not None]
        if fied_valid:
            ax2.plot([s for s, _ in fied_valid], [f for _, f in fied_valid],
                     'o-', color=color, linewidth=2, markersize=3, label=label)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Co/Cross Ratio')
    ax1.set_title('(a) Co/Cross Convergence', fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Fiedler (algebraic connectivity)')
    ax2.set_title('(b) Fiedler Trajectory', fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('FI-002: Bridge Initialization Independence',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig_fi002_init_independence.pdf')
    plt.savefig(f'{outdir}/fig_fi002_init_independence.png')
    plt.close()
    print('Saved fig_fi002_init_independence')
else:
    print(f'  FI-002: only {len(_fi002_data)} configs found, need >= 3 for figure')

# ============================================================
# Figure 8: FI-003 Topology Homeostasis (co/cross decay after Steersman removal)
# ============================================================
_fi003_path = os.path.join('results', 'fi-003', 'results.json')
if os.path.exists(_fi003_path):
    with open(_fi003_path) as f:
        _fi003_data = json.load(f)
    _fi003_ckpts = _fi003_data.get('checkpoints', [])
    if len(_fi003_ckpts) >= 3:
        _steps = [c['step'] for c in _fi003_ckpts]
        _cocross = [c.get('co_cross_ratio', 0) for c in _fi003_ckpts]
        _signs = [c.get('sign_stability', 0) for c in _fi003_ckpts]

        # Add the initial state (step 0) from the converged adapter
        _steps = [0] + _steps
        _cocross = [12586.0] + _cocross  # P-000 co/cross at step 3000
        _signs = [1.0] + _signs  # 100% sign stability at start

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Left: co/cross decay (log scale)
        ax1.plot(_steps, _cocross, 'o-', color=FCC_COLOR, linewidth=2, markersize=4)
        ax1.axhline(y=1, color=GREY, linestyle='--', alpha=0.5, label='Isotropic (1:1)')
        ax1.set_xlabel('Steps After Steersman Removal')
        ax1.set_ylabel('Co/Cross Ratio')
        ax1.set_title('(a) Co/Cross Exponential Decay', fontweight='bold')
        ax1.set_yscale('log')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Right: sign stability (linear scale)
        ax2.plot(_steps, _signs, 'o-', color=ACCENT1, linewidth=2, markersize=4)
        ax2.axhline(y=0.5, color=GREY, linestyle='--', alpha=0.5, label='Random (50%)')
        ax2.set_xlabel('Steps After Steersman Removal')
        ax2.set_ylabel('Sign Stability')
        ax2.set_title('(b) Sign Pattern Collapse', fontweight='bold')
        ax2.set_ylim(0.4, 1.05)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('FI-003: Topology Requires Continuous Maintenance',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{outdir}/fig_fi003_homeostasis.pdf')
        plt.savefig(f'{outdir}/fig_fi003_homeostasis.png')
        plt.close()
        print('Saved fig_fi003_homeostasis')
    else:
        print(f'  FI-003: only {len(_fi003_ckpts)} checkpoints, need >= 3 for figure')
else:
    print(f'  FI-003: results.json not found at {_fi003_path}')

# ============================================================
# Figure 9: FI-004 Steersman Annealing — Five Regimes
# ============================================================
_fi004_path = os.path.join('results', 'fi-004', 'results.json')
if os.path.exists(_fi004_path):
    with open(_fi004_path) as f:
        _fi004_data = json.load(f)
    _fi004_ckpts = _fi004_data.get('checkpoints', [])
    if len(_fi004_ckpts) >= 10:
        _steps = [c['step'] for c in _fi004_ckpts]
        _cw = [c.get('contrastive_weight', c.get('c_weight', 0)) for c in _fi004_ckpts]
        _cocross = [c.get('co_cross_ratio', 0) for c in _fi004_ckpts]
        _val = [c.get('val_loss', 0) for c in _fi004_ckpts]

        # Add the initial state (step 0, from P-000 converged adapter)
        _steps = [0] + _steps
        _cw = [0.100] + _cw
        _cocross = [12586.0] + _cocross
        _val = [0.499] + _val

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        # Left: co/cross vs step with regime shading
        ax1.plot(_steps, _cocross, 'o-', color=FCC_COLOR, linewidth=2, markersize=3)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Co/Cross Ratio')
        ax1.set_title('(a) Five Regimes of Steersman Annealing', fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Regime shading
        ax1.axvspan(0, 100, alpha=0.08, color=ACCENT2, label='Maintenance')
        ax1.axvspan(100, 600, alpha=0.08, color=FCC_COLOR, label='Interference')
        ax1.axvspan(600, 2500, alpha=0.08, color=ACCENT1, label='Growth')
        ax1.axvspan(2500, 2900, alpha=0.08, color=CUBIC_COLOR, label='Decay')
        ax1.axvspan(2900, 3000, alpha=0.15, color='red', label='Cliff')

        # Annotate peak and cliff
        ax1.annotate('Peak: 18,671:1\n($c_w$=0.017)',
                     xy=(2500, 18671), fontsize=7.5,
                     ha='right', va='bottom', color=FCC_COLOR,
                     arrowprops=dict(arrowstyle='->', color=FCC_COLOR, lw=0.8),
                     xytext=(1800, 25000))
        ax1.annotate('Cliff: 2,942:1\n($c_w$=0)',
                     xy=(3000, 2942), fontsize=7.5,
                     ha='left', va='top', color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
                     xytext=(2500, 800))
        ax1.legend(fontsize=6.5, loc='lower left', ncol=2)

        # Right: co/cross vs c_w (reversed x axis) — the "operating curve"
        # Filter out step 0 which is pre-anneal
        _cw_plot = _cw[1:]  # skip step 0
        _cc_plot = _cocross[1:]
        ax2.plot(_cw_plot, _cc_plot, 'o-', color=FCC_COLOR, linewidth=2, markersize=4)
        ax2.set_xlabel('Contrastive Weight ($c_w$)')
        ax2.set_ylabel('Co/Cross Ratio')
        ax2.set_title('(b) Operating Curve: Topology vs. Weight', fontweight='bold')
        ax2.set_yscale('log')
        ax2.invert_xaxis()
        ax2.grid(True, alpha=0.3)

        # Mark optimal point
        ax2.axvline(x=0.017, color=ACCENT1, linestyle='--', alpha=0.7, linewidth=1)
        ax2.annotate('Optimal\n$c_w \\approx 0.02$',
                     xy=(0.017, 18671), fontsize=8,
                     ha='left', va='bottom', color=ACCENT1,
                     xytext=(0.035, 20000))

        # Mark interference zone
        ax2.axvspan(0.083, 0.097, alpha=0.1, color=FCC_COLOR)
        ax2.text(0.090, 200, 'Interference\nzone', fontsize=7, ha='center',
                 color=FCC_COLOR, alpha=0.8)

        plt.suptitle('FI-004: Steersman Annealing Reveals Optimal Operating Point',
                     fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{outdir}/fig_fi004_annealing.pdf')
        plt.savefig(f'{outdir}/fig_fi004_annealing.png')
        plt.close()
        print('Saved fig_fi004_annealing')
    else:
        print(f'  FI-004: only {len(_fi004_ckpts)} checkpoints, need >= 10 for figure')
else:
    print(f'  FI-004: results.json not found at {_fi004_path}')

print('\nAll Paper 4 figures generated successfully.')
