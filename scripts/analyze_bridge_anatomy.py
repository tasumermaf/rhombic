"""Phase 0: Bridge Anatomy Analysis — Mine existing Exp 2 data.

Answers three questions from the utility interrogation plan:
  0A: Do bridges at different transformer layers learn different structures?
  0B: Do Q/K/V/O projections develop different bridge structures?
  0C: How does bridge structure evolve through training?

Operates on existing .npy files — no training, no GPU required.

Usage:
  python scripts/analyze_bridge_anatomy.py
  python scripts/analyze_bridge_anatomy.py --exp-dir results/exp2/rhombi_fcc_r24
  python scripts/analyze_bridge_anatomy.py --output results/exp2/bridge_anatomy.md
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


# ── Bridge Loading ────────────────────────────────────────────────────

FILENAME_PATTERN = re.compile(
    r"bridge_(step(\d+)|final)_model_layers_(\d+)_self_attn_([qkvo])_proj\.npy"
)


def parse_bridge_filename(path: Path) -> dict | None:
    """Extract step, layer, module type from bridge filename."""
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        return None
    step_or_final = m.group(1)
    step = int(m.group(2)) if m.group(2) else "final"
    layer = int(m.group(3))
    module = m.group(4) + "_proj"
    return {"step": step, "layer": layer, "module": module, "path": path}


def load_all_bridges(exp_dir: Path) -> list[dict]:
    """Load all bridge .npy files with parsed metadata."""
    bridges = []
    for f in sorted(exp_dir.glob("bridge_*.npy")):
        meta = parse_bridge_filename(f)
        if meta is None:
            continue
        B = np.load(f)
        if B.shape != (6, 6):
            continue
        meta["bridge"] = B
        bridges.append(meta)
    return bridges


# ── Metrics ───────────────────────────────────────────────────────────

def fiedler_value(B: np.ndarray) -> float:
    """Algebraic connectivity of bridge as weighted adjacency graph."""
    n = B.shape[0]
    W = np.abs(B)
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def deviation_from_identity(B: np.ndarray) -> float:
    """Frobenius distance from identity matrix."""
    return float(np.linalg.norm(B - np.eye(B.shape[0]), "fro"))


def eigenvalue_spread(B: np.ndarray) -> float:
    """Max - min eigenvalue of bridge."""
    eigvals = np.linalg.eigvalsh(B)
    return float(eigvals.max() - eigvals.min())


CO_PLANAR = [(0, 1), (2, 3), (4, 5)]
CROSS_PLANAR = [(i, j) for i in range(6) for j in range(i + 1, 6)
                if (i, j) not in CO_PLANAR]


def co_cross_ratio(B: np.ndarray) -> float:
    """Mean |co-planar| / mean |cross-planar| off-diagonal entries."""
    co = np.mean([abs(B[i, j]) for i, j in CO_PLANAR])
    cross = np.mean([abs(B[i, j]) for i, j in CROSS_PLANAR])
    return co / max(cross, 1e-12)


def compute_metrics(B: np.ndarray) -> dict:
    return {
        "fiedler": fiedler_value(B),
        "deviation": deviation_from_identity(B),
        "eig_spread": eigenvalue_spread(B),
        "co_cross": co_cross_ratio(B),
    }


# ── Analysis 0A: Per-Layer ────────────────────────────────────────────

def analyze_per_layer(bridges: list[dict]) -> str:
    """Group final bridges by layer, compute statistics."""
    by_layer = defaultdict(list)
    for b in bridges:
        if b["step"] == "final":
            by_layer[b["layer"]].append(b)

    lines = ["## 0A: Per-Layer Bridge Anatomy\n"]
    lines.append(f"*{sum(len(v) for v in by_layer.values())} final bridges across "
                 f"{len(by_layer)} layers*\n")

    # Compute per-layer statistics
    layer_stats = {}
    for layer in sorted(by_layer.keys()):
        metrics = [compute_metrics(b["bridge"]) for b in by_layer[layer]]
        layer_stats[layer] = {
            k: (np.mean([m[k] for m in metrics]), np.std([m[k] for m in metrics]))
            for k in ["fiedler", "deviation", "eig_spread", "co_cross"]
        }

    # Table
    lines.append("| Layer | Fiedler | Deviation | Eig Spread | Co/Cross |")
    lines.append("|-------|---------|-----------|------------|----------|")
    for layer in sorted(layer_stats.keys()):
        s = layer_stats[layer]
        lines.append(
            f"| {layer:2d} | {s['fiedler'][0]:.5f} ± {s['fiedler'][1]:.5f} "
            f"| {s['deviation'][0]:.4f} ± {s['deviation'][1]:.4f} "
            f"| {s['eig_spread'][0]:.5f} ± {s['eig_spread'][1]:.5f} "
            f"| {s['co_cross'][0]:.3f} ± {s['co_cross'][1]:.3f} |"
        )

    # ANOVA on Fiedler across layers
    fiedler_groups = [
        [compute_metrics(b["bridge"])["fiedler"] for b in by_layer[layer]]
        for layer in sorted(by_layer.keys())
    ]
    if len(fiedler_groups) > 1 and all(len(g) > 1 for g in fiedler_groups):
        f_stat, p_val = stats.f_oneway(*fiedler_groups)
        lines.append(f"\n**ANOVA (Fiedler ~ Layer):** F = {f_stat:.3f}, p = {p_val:.6f}")
        lines.append(f"**{'PASS' if p_val < 0.05 else 'FAIL'}** — "
                     f"{'Significant' if p_val < 0.05 else 'Not significant'} "
                     f"variation across layers (threshold p < 0.05)")

    # Deviation ANOVA
    dev_groups = [
        [compute_metrics(b["bridge"])["deviation"] for b in by_layer[layer]]
        for layer in sorted(by_layer.keys())
    ]
    if len(dev_groups) > 1 and all(len(g) > 1 for g in dev_groups):
        f_stat, p_val = stats.f_oneway(*dev_groups)
        lines.append(f"**ANOVA (Deviation ~ Layer):** F = {f_stat:.3f}, p = {p_val:.6f}")

    # Identify outlier layers (Fiedler > 1.5 IQR above median)
    layer_fiedlers = {
        layer: np.mean([compute_metrics(b["bridge"])["fiedler"] for b in by_layer[layer]])
        for layer in by_layer
    }
    vals = list(layer_fiedlers.values())
    q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    iqr = q3 - q1
    outliers_high = [l for l, v in layer_fiedlers.items() if v > q3 + 1.5 * iqr]
    outliers_low = [l for l, v in layer_fiedlers.items() if v < q1 - 1.5 * iqr]
    if outliers_high or outliers_low:
        lines.append(f"\n**Outlier layers (high Fiedler):** {outliers_high}")
        lines.append(f"**Outlier layers (low Fiedler):** {outliers_low}")

    # Trend test: does Fiedler correlate with layer depth?
    layers_arr = np.array(sorted(layer_fiedlers.keys()))
    fiedlers_arr = np.array([layer_fiedlers[l] for l in layers_arr])
    r, p = stats.pearsonr(layers_arr, fiedlers_arr)
    lines.append(f"\n**Depth correlation (Fiedler ~ Layer index):** r = {r:.4f}, p = {p:.6f}")
    if abs(r) > 0.3 and p < 0.05:
        direction = "increases" if r > 0 else "decreases"
        lines.append(f"Bridge coupling {direction} with depth.")

    return "\n".join(lines)


# ── Analysis 0B: Per-Module-Type ──────────────────────────────────────

def analyze_per_module(bridges: list[dict]) -> str:
    """Group final bridges by module type, compare Q/K/V/O."""
    by_module = defaultdict(list)
    for b in bridges:
        if b["step"] == "final":
            by_module[b["module"]].append(b)

    lines = ["\n## 0B: Per-Module-Type Bridge Anatomy\n"]

    # Per-module statistics
    module_stats = {}
    for mod in sorted(by_module.keys()):
        metrics = [compute_metrics(b["bridge"]) for b in by_module[mod]]
        module_stats[mod] = {
            k: (np.mean([m[k] for m in metrics]), np.std([m[k] for m in metrics]))
            for k in ["fiedler", "deviation", "eig_spread", "co_cross"]
        }

    lines.append("| Module | N | Fiedler | Deviation | Eig Spread | Co/Cross |")
    lines.append("|--------|---|---------|-----------|------------|----------|")
    for mod in sorted(module_stats.keys()):
        s = module_stats[mod]
        n = len(by_module[mod])
        lines.append(
            f"| {mod} | {n} | {s['fiedler'][0]:.5f} ± {s['fiedler'][1]:.5f} "
            f"| {s['deviation'][0]:.4f} ± {s['deviation'][1]:.4f} "
            f"| {s['eig_spread'][0]:.5f} ± {s['eig_spread'][1]:.5f} "
            f"| {s['co_cross'][0]:.3f} ± {s['co_cross'][1]:.3f} |"
        )

    # ANOVA on Fiedler across module types
    fiedler_groups = [
        [compute_metrics(b["bridge"])["fiedler"] for b in by_module[mod]]
        for mod in sorted(by_module.keys())
    ]
    if len(fiedler_groups) > 1:
        f_stat, p_val = stats.f_oneway(*fiedler_groups)
        lines.append(f"\n**ANOVA (Fiedler ~ Module Type):** F = {f_stat:.3f}, p = {p_val:.6f}")
        lines.append(f"**{'PASS' if p_val < 0.05 else 'FAIL'}** — "
                     f"{'Significant' if p_val < 0.05 else 'Not significant'} "
                     f"variation across module types")

    # Pairwise Q-K vs Q-V distance (per layer)
    by_layer_module = defaultdict(dict)
    for b in bridges:
        if b["step"] == "final":
            by_layer_module[b["layer"]][b["module"]] = b["bridge"]

    qk_dists = []
    qv_dists = []
    for layer, mods in by_layer_module.items():
        if all(m in mods for m in ["q_proj", "k_proj", "v_proj"]):
            qk = np.linalg.norm(mods["q_proj"] - mods["k_proj"], "fro")
            qv = np.linalg.norm(mods["q_proj"] - mods["v_proj"], "fro")
            qk_dists.append(qk)
            qv_dists.append(qv)

    if qk_dists and qv_dists:
        lines.append(f"\n**Q-K vs Q-V Bridge Distance (per layer, {len(qk_dists)} layers):**")
        lines.append(f"- Mean Q-K distance: {np.mean(qk_dists):.5f} ± {np.std(qk_dists):.5f}")
        lines.append(f"- Mean Q-V distance: {np.mean(qv_dists):.5f} ± {np.std(qv_dists):.5f}")
        t_stat, p_val = stats.ttest_rel(qk_dists, qv_dists)
        lines.append(f"- Paired t-test: t = {t_stat:.3f}, p = {p_val:.6f}")
        if np.mean(qk_dists) < np.mean(qv_dists) and p_val < 0.05:
            lines.append("- **PASS**: Q and K bridges are more similar than Q and V (as predicted)")
        elif p_val < 0.05:
            lines.append(f"- **Significant but opposite direction**: Q-V closer than Q-K")
        else:
            lines.append("- **FAIL**: No significant difference in Q-K vs Q-V similarity")

    return "\n".join(lines)


# ── Analysis 0C: Evolution Trajectories ───────────────────────────────

def analyze_trajectories(bridges: list[dict]) -> str:
    """Track per-adapter bridge evolution across training steps."""
    # Group by (layer, module) and step
    adapters = defaultdict(dict)
    for b in bridges:
        if b["step"] != "final":
            key = (b["layer"], b["module"])
            adapters[key][b["step"]] = b["bridge"]

    lines = ["\n## 0C: Bridge Evolution Trajectories\n"]

    if not adapters:
        lines.append("*No checkpoint data found.*")
        return "\n".join(lines)

    # Get sorted step list
    all_steps = sorted({s for steps in adapters.values() for s in steps})
    lines.append(f"*{len(adapters)} adapters tracked across {len(all_steps)} checkpoints "
                 f"(steps {all_steps[0]}-{all_steps[-1]})*\n")

    # Compute per-adapter Fiedler at each step
    adapter_trajectories = {}
    for (layer, mod), steps_dict in adapters.items():
        trajectory = []
        for step in all_steps:
            if step in steps_dict:
                trajectory.append(fiedler_value(steps_dict[step]))
            else:
                trajectory.append(np.nan)
        adapter_trajectories[(layer, mod)] = trajectory

    # Statistics at final checkpoint
    final_step = all_steps[-1]
    final_fiedlers = [
        t[-1] for t in adapter_trajectories.values() if not np.isnan(t[-1])
    ]

    if final_fiedlers:
        cv = np.std(final_fiedlers) / max(np.mean(final_fiedlers), 1e-12)
        lines.append(f"**Final step ({final_step}) Fiedler distribution:**")
        lines.append(f"- Mean: {np.mean(final_fiedlers):.5f}")
        lines.append(f"- Std: {np.std(final_fiedlers):.5f}")
        lines.append(f"- Min: {np.min(final_fiedlers):.5f}")
        lines.append(f"- Max: {np.max(final_fiedlers):.5f}")
        lines.append(f"- CV (coefficient of variation): {cv:.3f}")
        lines.append(f"- **{'PASS' if cv > 0.5 else 'FAIL'}** — "
                     f"{'High' if cv > 0.5 else 'Low'} variation "
                     f"(threshold CV > 0.5)\n")

    # Identify early vs late divergers
    # Compare Fiedler at step 1000 vs final — high correlation = uniform evolution
    if len(all_steps) > 2:
        early_step_idx = 1  # step 1000 typically
        early_fiedlers = [
            t[early_step_idx] for t in adapter_trajectories.values()
            if not np.isnan(t[early_step_idx]) and not np.isnan(t[-1])
        ]
        late_fiedlers = [
            t[-1] for t in adapter_trajectories.values()
            if not np.isnan(t[early_step_idx]) and not np.isnan(t[-1])
        ]
        if early_fiedlers and late_fiedlers:
            r, p = stats.pearsonr(early_fiedlers, late_fiedlers)
            lines.append(f"**Early-Late Correlation (step {all_steps[early_step_idx]} vs {final_step}):**")
            lines.append(f"- r = {r:.4f}, p = {p:.6f}")
            if r > 0.7:
                lines.append("- High correlation: adapters that start high stay high. "
                             "Bridge structure is determined early.")
            elif r < 0.3:
                lines.append("- Low correlation: adapters that start high don't necessarily "
                             "stay high. Bridge structure reorganizes during training.")

    # Top 10 and bottom 10 adapters by final Fiedler
    sorted_adapters = sorted(
        [(k, v[-1]) for k, v in adapter_trajectories.items() if not np.isnan(v[-1])],
        key=lambda x: x[1], reverse=True
    )
    if sorted_adapters:
        lines.append("\n**Top 10 adapters by final Fiedler:**")
        lines.append("| Layer | Module | Fiedler |")
        lines.append("|-------|--------|---------|")
        for (layer, mod), fied in sorted_adapters[:10]:
            lines.append(f"| {layer} | {mod} | {fied:.5f} |")

        lines.append("\n**Bottom 10 adapters by final Fiedler:**")
        lines.append("| Layer | Module | Fiedler |")
        lines.append("|-------|--------|---------|")
        for (layer, mod), fied in sorted_adapters[-10:]:
            lines.append(f"| {layer} | {mod} | {fied:.5f} |")

    # Mean trajectory table
    lines.append("\n**Mean Fiedler across all adapters by step:**")
    lines.append("| Step | Mean Fiedler | Std | Min | Max |")
    lines.append("|------|-------------|-----|-----|-----|")
    for i, step in enumerate(all_steps):
        vals = [t[i] for t in adapter_trajectories.values() if not np.isnan(t[i])]
        if vals:
            lines.append(
                f"| {step} | {np.mean(vals):.5f} | {np.std(vals):.5f} "
                f"| {np.min(vals):.5f} | {np.max(vals):.5f} |"
            )

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0: Bridge Anatomy Analysis")
    parser.add_argument("--exp-dir", type=str, default="results/exp2/rhombi_fcc_r24")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    print(f"Loading bridges from {exp_dir}...")
    bridges = load_all_bridges(exp_dir)
    print(f"Loaded {len(bridges)} bridge files")

    report = ["# Phase 0: Bridge Anatomy Analysis\n"]
    report.append(f"> **Source:** `{exp_dir}`")
    report.append(f"> **Total bridges loaded:** {len(bridges)}")
    report.append(f"> **Final bridges:** {sum(1 for b in bridges if b['step'] == 'final')}")

    n_checkpoint = sum(1 for b in bridges if b["step"] != "final")
    steps = sorted({b["step"] for b in bridges if b["step"] != "final"})
    report.append(f"> **Checkpoint bridges:** {n_checkpoint} across steps {steps}\n")

    report.append(analyze_per_layer(bridges))
    report.append(analyze_per_module(bridges))
    report.append(analyze_trajectories(bridges))

    # Summary
    report.append("\n---\n## Summary & Decision Gate\n")
    report.append("Phase 0 informs the direction of Phase 1 (task specificity).")
    report.append("If layer/module structure is found, the bridge captures positional information.")
    report.append("If not, the bridge learns generic coupling — Phase 1 tests whether task")
    report.append("specificity exists despite positional uniformity.\n")
    report.append("*Generated by `scripts/analyze_bridge_anatomy.py`*")

    text = "\n".join(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nReport written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
