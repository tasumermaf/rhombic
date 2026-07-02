"""Paper 4 summary figure: 4-panel comparison of topology programming results.

Panel 1: T-001r2 co/cross trajectory (tesseract, n=8) — 41,564:1
Panel 2: WL-001 co/cross trajectory (wrong-labels, n=6) — ~0:1
Panel 3: H-ch6 co/cross trajectory (RD, n=6) — 70,404:1 (reference)
Panel 4: Spectral attractor (Fiedler across n=3,4,8,12) vs contrastive (n=6,8)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"C:\falco\rhombic\results")
CA = BASE / "channel-ablation"

# Color palette (from 8-Law Weave)
FCC_COLOR = '#B34444'   # Geometric Essence
CUBIC_COLOR = '#3D3D6B' # Fall of Neutral Events
TESS_COLOR = '#2196F3'  # Blue for tesseract
WL_COLOR = '#999999'    # Grey for wrong-labels
ATTRACTOR_COLOR = '#FF9800'  # Orange for attractor

def load_feedback(path):
    with open(path) as f:
        d = json.load(f)
    fb = d.get("feedback_log", d.get("checkpoints", []))
    return fb

def load_csv(path):
    """Load H-ch6 style CSV: step,val_loss,fiedler,co_cross,..."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header_line = next(reader)  # first line might be just "100" (no header)
        # Check if first line is a header or data
        try:
            step = int(header_line[0])
            # It's data, parse it
            rows.append({
                "step": step,
                "val_loss": float(header_line[1]) if len(header_line) > 1 else 0,
                "fiedler_mean": float(header_line[2]) if len(header_line) > 2 else 0,
                "co_cross_ratio": float(header_line[3]) if len(header_line) > 3 else 0,
            })
        except (ValueError, IndexError):
            pass  # header row, skip
        for row in reader:
            if len(row) >= 4:
                try:
                    rows.append({
                        "step": int(row[0]),
                        "val_loss": float(row[1]),
                        "fiedler_mean": float(row[2]),
                        "co_cross_ratio": float(row[3]),
                    })
                except (ValueError, IndexError):
                    continue
    return rows

# Load all datasets
t001r2 = load_feedback(BASE / "T-001-full-r2" / "results.json")
hch6 = load_csv(CA / "H-ch6" / "metrics_hermes.csv")

# WL-001 uses 'checkpoints' key
with open(CA / "WL-001" / "results.json") as f:
    wl_data = json.load(f)
wl001 = wl_data["checkpoints"]

# Spectral-only runs
hch3 = load_feedback(CA / "H-ch3" / "results.json")
hch4 = load_feedback(CA / "H-ch4" / "results.json")
hch8 = load_feedback(CA / "H-ch8-results-hermes.json")
hch12 = load_feedback(CA / "H-ch12" / "results.json")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Panel 1: T-001r2 Tesseract (n=8) ---
ax = axes[0, 0]
steps = [e["step"] for e in t001r2]
cc = [e.get("co_cross_ratio", 0) for e in t001r2]
cc_valid = [(s, c) for s, c in zip(steps, cc) if c and c > 0]
if cc_valid:
    s, c = zip(*cc_valid)
    ax.plot(s, c, color=TESS_COLOR, lw=2.5)
    ax.fill_between(s, 1, c, alpha=0.1, color=TESS_COLOR)
ax.set_yscale('log')
ax.set_xlabel('Step')
ax.set_ylabel('Co/Cross Ratio')
ax.set_title('Tesseract (n=8) — T-001r2', fontweight='bold', fontsize=13)
ax.axhline(y=41564, color=TESS_COLOR, ls='--', alpha=0.5)
ax.text(0.95, 0.95, f'Final: 41,564:1\n4+4 eigenvalue split',
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightskyblue', alpha=0.8))
ax.grid(True, alpha=0.2, which='both')
ax.set_ylim(bottom=0.1)

# --- Panel 2: WL-001 Wrong-Labels (n=6) ---
ax = axes[0, 1]
steps_wl = [e["step"] for e in wl001]
cc_wl = [e.get("co_cross_ratio", 0) for e in wl001]
ax.plot(steps_wl, cc_wl, color=WL_COLOR, lw=2.5)
ax.set_xlabel('Step')
ax.set_ylabel('Co/Cross Ratio')
ax.set_title('Wrong-Labels Control (n=6) — WL-001', fontweight='bold', fontsize=13)
ax.text(0.95, 0.95, f'Final: ≈0:1\nNo BD structure',
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
ax.grid(True, alpha=0.2, which='both')

# --- Panel 3: H-ch6 RD Reference (n=6) ---
ax = axes[1, 0]
steps_h6 = [e["step"] for e in hch6]
cc_h6 = [e.get("co_cross_ratio", 0) for e in hch6]
cc_h6_valid = [(s, c) for s, c in zip(steps_h6, cc_h6) if c and c > 0]
if cc_h6_valid:
    s, c = zip(*cc_h6_valid)
    ax.plot(s, c, color=FCC_COLOR, lw=2.5)
    ax.fill_between(s, 1, c, alpha=0.1, color=FCC_COLOR)
ax.set_yscale('log')
ax.set_xlabel('Step')
ax.set_ylabel('Co/Cross Ratio')
ax.set_title('Rhombic Dodecahedron (n=6) — H-ch6', fontweight='bold', fontsize=13)
ax.axhline(y=70404, color=FCC_COLOR, ls='--', alpha=0.5)
ax.text(0.95, 0.95, f'Final: 70,404:1\n3-axis BD',
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#FFCCCC', alpha=0.8))
ax.grid(True, alpha=0.2, which='both')
ax.set_ylim(bottom=0.1)

# --- Panel 4: Spectral Attractor vs Contrastive ---
ax = axes[1, 1]
# Spectral-only final Fiedler values
spectral_ns = [3, 4, 8, 12]
spectral_fiedler = []
for fb in [hch3, hch4, hch8, hch12]:
    last = fb[-1]
    spectral_fiedler.append(last.get("fiedler_mean", last.get("fiedler", 0)))

# Contrastive final Fiedler values
contrastive_ns = [6, 8]
contrastive_fiedler = [
    hch6[-1].get("fiedler_mean", hch6[-1].get("fiedler", 0)),
    t001r2[-1].get("fiedler_mean", 0)
]

ax.scatter(spectral_ns, spectral_fiedler, s=150, c=ATTRACTOR_COLOR,
           edgecolors='black', linewidths=1, zorder=5, label='Spectral only')
ax.scatter(contrastive_ns, contrastive_fiedler, s=150, c=FCC_COLOR,
           marker='D', edgecolors='black', linewidths=1, zorder=5, label='Contrastive')

# Attractor band
ax.axhspan(min(spectral_fiedler) * 0.9, max(spectral_fiedler) * 1.1,
           alpha=0.15, color=ATTRACTOR_COLOR, label=f'Attractor band ({min(spectral_fiedler):.4f}–{max(spectral_fiedler):.4f})')

ax.set_yscale('log')
ax.set_xlabel('Channel Count (n)')
ax.set_ylabel('Final Fiedler Eigenvalue')
ax.set_title('Spectral Attractor vs Contrastive Programming', fontweight='bold', fontsize=13)
ax.set_xticks([3, 4, 6, 8, 12])
ax.legend(fontsize=10, loc='center right')
ax.grid(True, alpha=0.2, which='both')

# Annotation
band_pct = (max(spectral_fiedler) - min(spectral_fiedler)) / np.mean(spectral_fiedler) * 100
ax.text(0.05, 0.05, f'Attractor band: {band_pct:.1f}% width\nContrastive breaks by 1,000×+',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.suptitle('Paper 4: Topology Programming — The Steersman as General-Purpose Geometry Programmer',
             fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout(pad=2)

out = CA / "fig_paper4_summary.png"
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"Saved: {out}")
print(f"Saved: {out.with_suffix('.pdf')}")
print(f"\nKey numbers:")
print(f"  T-001r2 (tesseract n=8): {cc_valid[-1][1]:,.0f}:1 co/cross")
print(f"  H-ch6 (RD n=6): {cc_h6_valid[-1][1]:,.0f}:1 co/cross")
print(f"  WL-001 (wrong-labels n=6): {cc_wl[-1]:.6f}:1 co/cross")
print(f"  Spectral attractor: {[f'{f:.4f}' for f in spectral_fiedler]}")
print(f"  Contrastive Fiedler: {[f'{f:.6f}' for f in contrastive_fiedler]}")
