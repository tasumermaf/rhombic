"""
Generate the FULL channel ablation figure for Paper 3.

Five data series: n=3,4,8 (spectral-only) + n=6 (RD contrastive) + T-001 (tesseract contrastive).
Four panels:
  (a) Fiedler convergence over training (log scale)
  (b) Eigenvalue spectra at convergence (bar chart)
  (c) Co/Cross ratio for contrastive runs
  (d) Spectral attractor summary (final Fiedler vs n_channels)

Data sources:
  n=3: H-ch3/results.json (local)
  n=4: H-ch4/results_hermes.json (local)
  n=6: H-ch6/metrics_hermes.csv (local)
  n=8: H-ch8-results-hermes.json (pulled from Hermes)
  T-001: ../T-001-full/results.json (local, partial 2700 steps)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\falco\rhombic\results\channel-ablation")

# ===== Load all data =====

# n=3 spectral
with open(BASE / "H-ch3" / "results.json") as f:
    ch3_data = json.load(f)
ch3_fb = ch3_data.get("feedback_log", ch3_data.get("checkpoints", []))
ch3_steps = [c["step"] for c in ch3_fb]
ch3_fiedler = [c.get("fiedler_mean", c.get("fiedler", 0)) for c in ch3_fb]

# n=4 spectral
with open(BASE / "H-ch4" / "results_hermes.json") as f:
    ch4_data = json.load(f)
ch4_fb = ch4_data.get("feedback_log", ch4_data.get("checkpoints", []))
ch4_steps = [c["step"] for c in ch4_fb]
ch4_fiedler = [c.get("fiedler_mean", c.get("fiedler", 0)) for c in ch4_fb]

# n=6 contrastive (CSV format)
ch6_steps, ch6_fiedler, ch6_ratio = [], [], []
with open(BASE / "H-ch6" / "metrics_hermes.csv") as f:
    lines = f.readlines()
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split(",")
    if len(parts) < 7:
        continue
    try:
        ch6_steps.append(int(parts[0]))
        ch6_fiedler.append(float(parts[2]))
        ch6_ratio.append(float(parts[3]))
    except (ValueError, IndexError):
        continue

# n=8 spectral (pulled from Hermes)
with open(BASE / "H-ch8-results-hermes.json") as f:
    ch8_data = json.load(f)
ch8_fb = ch8_data.get("feedback_log", [])
ch8_steps = [c["step"] for c in ch8_fb]
ch8_fiedler = [c.get("fiedler_mean", 0) for c in ch8_fb]
ch8_eigenvalues = ch8_fb[-1].get("eigenvalue_pattern", []) if ch8_fb else []

# T-001 tesseract contrastive (local, partial)
t001_path = Path(r"C:\falco\rhombic\results\T-001-full\results.json")
with open(t001_path) as f:
    t001_data = json.load(f)
t001_fb = t001_data.get("feedback_log", [])
t001_steps = [c["step"] for c in t001_fb]
t001_fiedler = [c.get("fiedler_mean", 0) for c in t001_fb]
t001_cocross = [c.get("co_cross_ratio", 0) for c in t001_fb]
t001_eigenvalues = t001_fb[-1].get("eigenvalue_pattern", []) if t001_fb else []

# ===== Colors =====
SPECTRAL_COLORS = {
    3: '#2196F3',   # blue
    4: '#FF9800',   # orange
    8: '#4CAF50',   # green
    12: '#9C27B0',  # purple (placeholder)
}
C6 = '#B34444'      # FCC red (contrastive)
CT = '#E91E63'      # tesseract pink (contrastive)

LW = 1.8
LABEL_SIZE = 13
TITLE_SIZE = 13
TICK_SIZE = 10

# ===== Figure: 2x2 =====
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# ===== Panel (a): Fiedler convergence over training =====
ax1.plot(ch3_steps, ch3_fiedler, color=SPECTRAL_COLORS[3], lw=LW, label='n=3 spectral', zorder=3)
ax1.plot(ch4_steps, ch4_fiedler, color=SPECTRAL_COLORS[4], lw=LW, label='n=4 spectral', zorder=3)
ax1.plot(ch8_steps, ch8_fiedler, color=SPECTRAL_COLORS[8], lw=LW, label='n=8 spectral', zorder=3)
ax1.plot(ch6_steps, ch6_fiedler, color=C6, lw=LW, ls='--', label='n=6 RD contrastive', zorder=4)
ax1.plot(t001_steps, t001_fiedler, color=CT, lw=LW, ls='--', label='n=8 tesseract contrastive', zorder=4)

ax1.set_yscale('log')
ax1.set_xlabel('Step', fontsize=LABEL_SIZE)
ax1.set_ylabel('Fiedler Eigenvalue (log)', fontsize=LABEL_SIZE)
ax1.set_title('Fiedler Convergence: Spectral Attractor vs Block-Diagonal', fontsize=TITLE_SIZE, fontweight='bold')
ax1.legend(fontsize=9, loc='right', framealpha=0.9)
ax1.tick_params(labelsize=TICK_SIZE)
ax1.grid(True, alpha=0.2, which='both')

# Shade the spectral attractor band
attractor_lo, attractor_hi = 0.088, 0.098
ax1.axhspan(attractor_lo, attractor_hi, alpha=0.1, color='green', zorder=1)
ax1.text(500, 0.093, 'Spectral attractor\n(0.092 ± 0.002)', fontsize=8,
         color='green', fontstyle='italic', va='center')

ax1.text(0.02, 0.02, '(a)', transform=ax1.transAxes, fontsize=13, fontweight='bold', va='bottom')

# ===== Panel (b): Eigenvalue spectra at convergence =====
# n=8 spectral eigenvalues
if ch8_eigenvalues:
    ch8_eigs = sorted([e[0] for e in ch8_eigenvalues])
    x8 = np.arange(len(ch8_eigs))
    ax2.bar(x8 - 0.2, ch8_eigs, width=0.35, color=SPECTRAL_COLORS[8],
            label='n=8 spectral (10K)', alpha=0.8, zorder=3)

# T-001 tesseract eigenvalues
if t001_eigenvalues:
    t001_eigs = sorted([e[0] for e in t001_eigenvalues])
    xt = np.arange(len(t001_eigs))
    ax2.bar(xt + 0.2, t001_eigs, width=0.35, color=CT,
            label='n=8 tesseract (2.7K)', alpha=0.8, zorder=3)

ax2.set_xlabel('Eigenvalue Index', fontsize=LABEL_SIZE)
ax2.set_ylabel('Eigenvalue', fontsize=LABEL_SIZE)
ax2.set_title('Eigenvalue Spectra: Continuous vs 4+4 Block Split', fontsize=TITLE_SIZE, fontweight='bold')
ax2.legend(fontsize=9, framealpha=0.9)
ax2.tick_params(labelsize=TICK_SIZE)
ax2.grid(True, alpha=0.2, axis='y')

# Annotate the gap
if t001_eigenvalues:
    gap_lo = max(t001_eigs[:4])
    gap_hi = min(t001_eigs[4:])
    ax2.annotate(f'3 orders of\nmagnitude gap',
                 xy=(3.5, (gap_lo + gap_hi) / 2),
                 xytext=(5.5, 0.3),
                 fontsize=9, color=CT, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=CT, lw=1.2))

ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=13, fontweight='bold', va='top')

# ===== Panel (c): Co/Cross ratio over training =====
# n=6 contrastive
ax3.plot(ch6_steps, ch6_ratio, color=C6, lw=LW + 0.3, label='n=6 RD', zorder=3)

# T-001 tesseract (filter out None/0 values)
t001_valid = [(s, r) for s, r in zip(t001_steps, t001_cocross) if r and r > 0]
if t001_valid:
    ts, tr = zip(*t001_valid)
    ax3.plot(ts, tr, color=CT, lw=LW + 0.3, label='n=8 tesseract', zorder=3)

ax3.set_yscale('log')
ax3.set_xlabel('Step', fontsize=LABEL_SIZE)
ax3.set_ylabel('Co-Planar / Cross-Planar Ratio', fontsize=LABEL_SIZE)
ax3.set_title('Contrastive Loss Drives Extreme Axis Separation', fontsize=TITLE_SIZE, fontweight='bold')
ax3.legend(fontsize=10, loc='lower right', framealpha=0.9)
ax3.tick_params(labelsize=TICK_SIZE)
ax3.grid(True, alpha=0.2, which='both')

# Mark peaks
if ch6_ratio:
    peak6 = max(ch6_ratio)
    peak6_step = ch6_steps[ch6_ratio.index(peak6)]
    ax3.annotate(f'Peak: {peak6:,.0f}:1', xy=(peak6_step, peak6),
                 xytext=(-80, -20), textcoords='offset points',
                 fontsize=9, color=C6, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C6, lw=1.0))

ax3.text(0.02, 0.02, '(c)', transform=ax3.transAxes, fontsize=13, fontweight='bold', va='bottom')

# ===== Panel (d): Spectral attractor summary =====
# Final Fiedler vs n_channels
spectral_n = [3, 4, 8]
spectral_fiedler = [ch3_fiedler[-1], ch4_fiedler[-1], ch8_fiedler[-1]]
contrastive_n = [6, 8]
contrastive_fiedler = [ch6_fiedler[-1], t001_fiedler[-1]]

ax4.scatter(spectral_n, spectral_fiedler, s=150, c=[SPECTRAL_COLORS[n] for n in spectral_n],
            edgecolors='black', linewidths=1.5, zorder=5, label='Spectral-only')
ax4.scatter(contrastive_n, contrastive_fiedler, s=150, c=[C6, CT],
            edgecolors='black', linewidths=1.5, marker='D', zorder=5, label='Contrastive')

# Attractor band
ax4.axhspan(0.088, 0.098, alpha=0.15, color='green', zorder=1)
ax4.text(2.5, 0.093, 'Spectral attractor\n(3.5% band)', fontsize=9,
         color='green', fontstyle='italic', va='center')

ax4.set_xlabel('Number of Channels', fontsize=LABEL_SIZE)
ax4.set_ylabel('Final Fiedler Eigenvalue', fontsize=LABEL_SIZE)
ax4.set_title('The Spectral Attractor Is Channel-Count-Invariant', fontsize=TITLE_SIZE, fontweight='bold')
ax4.set_yscale('log')
ax4.set_xticks([3, 4, 6, 8])
ax4.legend(fontsize=10, framealpha=0.9)
ax4.tick_params(labelsize=TICK_SIZE)
ax4.grid(True, alpha=0.2, which='both')

# Annotate the contrastive points
for n, f, c in zip(contrastive_n, contrastive_fiedler, [C6, CT]):
    label = 'RD' if n == 6 else 'Tesseract'
    ax4.annotate(f'{label}\n(F={f:.5f})', xy=(n, f),
                 xytext=(15, 20), textcoords='offset points',
                 fontsize=8, color=c,
                 arrowprops=dict(arrowstyle='->', color=c, lw=0.8))

ax4.text(0.02, 0.02, '(d)', transform=ax4.transAxes, fontsize=13, fontweight='bold', va='bottom')

# ===== Save =====
fig.tight_layout(pad=2.5)

out_png = BASE / "fig_full_ablation.png"
out_pdf = BASE / "fig_full_ablation.pdf"
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
print(f"\n-- Summary --")
print(f"Spectral attractor (n=3,4,8): {min(spectral_fiedler):.4f} - {max(spectral_fiedler):.4f}")
print(f"  Range: {(max(spectral_fiedler) - min(spectral_fiedler)) / np.mean(spectral_fiedler) * 100:.1f}%")
print(f"Contrastive n=6 final Fiedler: {ch6_fiedler[-1]:.6f}")
print(f"Contrastive T-001 final Fiedler: {t001_fiedler[-1]:.6f} (at step {t001_steps[-1]})")
print(f"T-001 co/cross at last step: {t001_cocross[-1]}")
