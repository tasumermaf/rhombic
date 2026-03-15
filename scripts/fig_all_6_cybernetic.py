#!/usr/bin/env python
"""
Generate the definitive temporal emergence figure for Paper 3.
Shows ALL 6 cybernetic n=6 experiments on one two-panel plot.
"""

import json
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Try publication style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass  # fall back to default

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')


def load_list_json(path, ratio_key='ratio'):
    """Load list-format JSON: [{step, ratio}, ...]"""
    with open(path) as f:
        data = json.load(f)
    steps = [d['step'] for d in data if d[ratio_key] > 0]
    ratios = [d[ratio_key] for d in data if d[ratio_key] > 0]
    return np.array(steps), np.array(ratios)


def load_dict_json(path):
    """Load dict-format JSON with 'checkpoints' key, each has co_cross_ratio."""
    with open(path) as f:
        data = json.load(f)
    chks = data['checkpoints']
    steps = [c['step'] for c in chks if c.get('co_cross_ratio', 0) > 0]
    ratios = [c['co_cross_ratio'] for c in chks if c.get('co_cross_ratio', 0) > 0]
    return np.array(steps), np.array(ratios)


def load_hermes_csv(path):
    """Load H-ch6 CSV: first line is count, then step,val_loss,...,co_cross_ratio,..."""
    steps, ratios = [], []
    with open(path) as f:
        # First line is the count
        _count = f.readline().strip()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                step = int(parts[0])
                ratio = float(parts[3])  # co_cross_ratio is 4th column
                if ratio > 0:
                    steps.append(step)
                    ratios.append(ratio)
    return np.array(steps), np.array(ratios)


# --- Load all 6 datasets ---
print("Loading data...")

# 1. exp3 Qwen 7B
s_qwen, r_qwen = load_list_json(os.path.join(RESULTS, 'exp3_temporal_emergence.json'))
print(f"  Qwen 7B: {len(s_qwen)} points, steps {s_qwen[0]}..{s_qwen[-1]}")

# 2. exp3 TinyLlama
s_tiny, r_tiny = load_list_json(os.path.join(RESULTS, 'exp3_tinyllama_temporal_emergence.json'))
print(f"  TinyLlama: {len(s_tiny)} points, steps {s_tiny[0]}..{s_tiny[-1]}")

# 3. C-001 identity
s_c001, r_c001 = load_list_json(os.path.join(RESULTS, 'C-001_temporal_emergence.json'))
print(f"  C-001: {len(s_c001)} points, steps {s_c001[0]}..{s_c001[-1]}")

# 4. C-002 geometric
s_c002, r_c002 = load_dict_json(os.path.join(RESULTS, 'corpus-baselines', 'C-002-geometric-default', 'results.json'))
print(f"  C-002: {len(s_c002)} points, steps {s_c002[0]}..{s_c002[-1]}")

# 5. C-003 corpus-coupled
s_c003, r_c003 = load_dict_json(os.path.join(RESULTS, 'corpus-baselines', 'C-003-corpus-coupled-default', 'results.json'))
print(f"  C-003: {len(s_c003)} points, steps {s_c003[0]}..{s_c003[-1]}")

# 6. H-ch6 Hermes replication
s_h3, r_h3 = load_hermes_csv(os.path.join(RESULTS, 'channel-ablation', 'H-ch6', 'metrics_hermes.csv'))
print(f"  H-ch6: {len(s_h3)} points, steps {s_h3[0]}..{s_h3[-1]}")

# --- Find C-003 peak ---
peak_idx = np.argmax(r_c003)
peak_step = s_c003[peak_idx]
peak_val = r_c003[peak_idx]
print(f"\nC-003 peak: step={peak_step}, ratio={peak_val:.1f}")

# Also report H-ch6 peak for reference
h3_peak_idx = np.argmax(r_h3)
print(f"H-ch6 peak: step={s_h3[h3_peak_idx]}, ratio={r_h3[h3_peak_idx]:.1f}")

# --- Create figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Color and style definitions
datasets = [
    (s_qwen, r_qwen, '#B34444', '-',  2.0, 'Qwen 7B (exp3)'),
    (s_tiny, r_tiny, '#B34444', '--', 1.5, 'TinyLlama 1.1B (exp3)'),
    (s_c001, r_c001, '#3D3D6B', '-',  1.5, 'C-001 (identity init)'),
    (s_c002, r_c002, '#4B8B3B', '-',  1.5, 'C-002 (geometric)'),
    (s_c003, r_c003, '#CC7722', '-',  1.5, 'C-003 (corpus-coupled)'),
    (s_h3,   r_h3,   '#7B3F9E', '-',  1.5, 'H-ch6 (Hermes replication)'),
]

# ===== LEFT PANEL: Full range, log scale =====
for steps, ratios, color, ls, lw, label in datasets:
    ax1.plot(steps, ratios, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.9)

# Convergence band
ax1.axhspan(64000, 71000, alpha=0.15, color='#888888', zorder=0)
ax1.text(11500, 67500, '64K\u201371K band', fontsize=7, ha='center', va='center',
         color='#555555', style='italic')

# Peak marker
ax1.plot(peak_step, peak_val, marker='*', markersize=14, color='#CC7722',
         markeredgecolor='black', markeredgewidth=0.5, zorder=5)
ax1.annotate(f'{peak_val:,.0f}:1', xy=(peak_step, peak_val),
             xytext=(peak_step - 2500, peak_val * 1.3),
             fontsize=7, color='#CC7722', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#CC7722', lw=0.8))

ax1.set_yscale('log')
ax1.set_xlabel('Training Step', fontsize=10)
ax1.set_ylabel('Co/Cross Ratio (log scale)', fontsize=10)
ax1.set_title('Temporal Emergence: All 6 Cybernetic Experiments', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 13500)

# ===== RIGHT PANEL: Zoom on first 2000 steps =====
for steps, ratios, color, ls, lw, label in datasets:
    mask = steps <= 2000
    if mask.any():
        ax2.plot(steps[mask], ratios[mask], color=color, linestyle=ls, linewidth=lw,
                 label=label, alpha=0.9)

# Lock-in line — draw AFTER setting scale and limits so text is positioned correctly
ax2.set_yscale('log')
ax2.set_xlabel('Training Step', fontsize=10)
ax2.set_ylabel('Co/Cross Ratio (log scale)', fontsize=10)
ax2.set_title('Universal Lock-in by Step 200', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 2000)

# Lock-in line — positioned after axis limits are set
ax2.axvline(x=200, color='#333333', linestyle='--', linewidth=1.0, alpha=0.7)
# Place label using axes transform for robust positioning
ax2.text(0.13, 0.92, 'Lock-in', fontsize=8, color='#333333', fontweight='bold',
         transform=ax2.transAxes, ha='left', va='top')

plt.tight_layout()

# Save
out_png = os.path.join(RESULTS, 'fig_all_6_cybernetic_definitive.png')
out_pdf = os.path.join(RESULTS, 'fig_all_6_cybernetic_definitive.pdf')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")

# Report summary stats
print("\n--- Summary ---")
for steps, ratios, _, _, _, label in datasets:
    print(f"  {label}: max ratio = {ratios.max():,.1f} at step {steps[np.argmax(ratios)]}")
