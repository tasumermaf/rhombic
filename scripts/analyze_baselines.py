#!/usr/bin/env python3
"""
Corpus Baseline Comparison — Structured Analysis

Loads results from C-001 through C-004 (as available) and produces a
comprehensive comparison report covering val loss, co/cross ratio,
Fiedler value, deviation, convergence speed, and bridge dynamics.

Output: stdout + results/corpus-baselines/BASELINE_COMPARISON.md
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASELINES_DIR = Path(__file__).resolve().parent.parent / "results" / "corpus-baselines"

EXPERIMENTS = {
    "C-001": {
        "dir": "C-001-identity-default",
        "label": "L3 Identity (baseline)",
        "bridge_mode": "identity",
    },
    "C-002": {
        "dir": "C-002-geometric-default",
        "label": "L4 Geometric",
        "bridge_mode": "geometric",
    },
    "C-003": {
        "dir": "C-003-corpus-coupled-default",
        "label": "L5 Corpus-Coupled (default)",
        "bridge_mode": "corpus_coupled",
    },
    "C-004": {
        "dir": "C-004-corpus-coupled-corpus",
        "label": "L5 Corpus-Coupled (corpus)",
        "bridge_mode": "corpus_coupled",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_experiment(exp_id: str) -> Optional[dict]:
    """Load a single experiment's results.json, or None if missing."""
    meta = EXPERIMENTS[exp_id]
    path = BASELINES_DIR / meta["dir"] / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data


def extract_checkpoints(data: dict) -> list[dict]:
    """Return the checkpoint list sorted by step."""
    cps = data.get("checkpoints", [])
    return sorted(cps, key=lambda c: c["step"])


# ---------------------------------------------------------------------------
# Per-experiment analysis
# ---------------------------------------------------------------------------
def analyze_trajectory(values: np.ndarray, steps: np.ndarray) -> dict:
    """Compute first, min, max, last, mean, and linear trend slope."""
    if len(values) == 0:
        return {"first": None, "min": None, "max": None, "last": None,
                "mean": None, "trend_per_100": None}
    result = {
        "first": float(values[0]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "last": float(values[-1]),
        "mean": float(np.mean(values)),
    }
    if len(values) >= 2:
        # Linear regression: value = a*step + b
        coeffs = np.polyfit(steps.astype(float), values.astype(float), 1)
        result["trend_per_100"] = float(coeffs[0] * 100)  # change per 100 steps
    else:
        result["trend_per_100"] = None
    return result


def find_threshold_step(steps: np.ndarray, values: np.ndarray,
                        threshold: float) -> Optional[int]:
    """Return the first step at which values drop below threshold, or None."""
    mask = values < threshold
    if not np.any(mask):
        return None
    return int(steps[np.argmax(mask)])


def find_plateau_step(steps: np.ndarray, values: np.ndarray,
                      window: int = 5, rel_change: float = 0.05) -> Optional[int]:
    """
    Find the step where a metric plateaus.
    Plateau = rolling window where relative change from window mean
    is below rel_change for all points in window.
    """
    if len(values) < window:
        return None
    for i in range(len(values) - window + 1):
        segment = values[i:i + window]
        seg_mean = np.mean(segment)
        if seg_mean == 0:
            continue
        max_dev = np.max(np.abs(segment - seg_mean)) / abs(seg_mean)
        if max_dev < rel_change:
            return int(steps[i])
    return None


def warmup_index(steps: np.ndarray, warmup_steps: int = 200) -> int:
    """Return the index of the first checkpoint after warmup."""
    idxs = np.where(steps > warmup_steps)[0]
    return int(idxs[0]) if len(idxs) > 0 else 0


def analyze_experiment(data: dict, exp_id: str) -> dict:
    """Full analysis of a single experiment."""
    cps = extract_checkpoints(data)
    n = len(cps)
    config = data.get("config", {})

    steps = np.array([c["step"] for c in cps])
    val_loss = np.array([c.get("val_loss", np.nan) for c in cps])
    train_loss = np.array([c.get("train_loss", np.nan) for c in cps])
    co_cross = np.array([c.get("co_cross_ratio", np.nan) for c in cps])
    fiedler = np.array([c.get("fiedler_mean", np.nan) for c in cps])
    deviation = np.array([c.get("deviation_mean", np.nan) for c in cps])
    spectral_gap = np.array([c.get("spectral_gap", np.nan) for c in cps])
    contrastive_loss = np.array([c.get("contrastive_loss", np.nan) for c in cps])
    wall_time = np.array([c.get("wall_time", np.nan) for c in cps])

    wi = warmup_index(steps)

    # Feedback log initial state
    fb = data.get("feedback_log", [])
    init_state = fb[0] if fb else {}

    result = {
        "exp_id": exp_id,
        "label": EXPERIMENTS[exp_id]["label"],
        "bridge_mode": config.get("bridge_mode", "unknown"),
        "n_checkpoints": n,
        "step_range": (int(steps[0]), int(steps[-1])) if n > 0 else (0, 0),
        "total_steps": int(steps[-1]) if n > 0 else 0,
        "trainable_params": data.get("trainable_params", "?"),
        "bridge_params": data.get("bridge_params", "?"),
        "total_params": data.get("total_params", "?"),
    }

    if n == 0:
        result["status"] = "NO DATA"
        return result

    result["status"] = "OK"

    # --- Val loss ---
    result["val_loss"] = analyze_trajectory(val_loss, steps)
    result["val_loss"]["step_below_043"] = find_threshold_step(steps, val_loss, 0.43)
    result["val_loss"]["step_below_042"] = find_threshold_step(steps, val_loss, 0.42)
    result["val_loss"]["step_below_0425"] = find_threshold_step(steps, val_loss, 0.425)
    result["val_loss"]["step_below_0420"] = find_threshold_step(steps, val_loss, 0.420)

    # --- Train loss ---
    result["train_loss"] = analyze_trajectory(train_loss, steps)

    # --- Co/cross ratio ---
    result["co_cross"] = analyze_trajectory(co_cross, steps)
    if n > wi:
        result["co_cross"]["mean_after_warmup"] = float(np.nanmean(co_cross[wi:]))
        result["co_cross"]["plateau_step"] = find_plateau_step(
            steps[wi:], co_cross[wi:], window=5, rel_change=0.10)
    else:
        result["co_cross"]["mean_after_warmup"] = None
        result["co_cross"]["plateau_step"] = None

    # --- Fiedler ---
    result["fiedler"] = analyze_trajectory(fiedler, steps)
    result["fiedler"]["init"] = float(init_state.get("fiedler_mean", 0))
    if n > wi:
        result["fiedler"]["stabilize_step"] = find_plateau_step(
            steps[wi:], fiedler[wi:], window=5, rel_change=0.20)
    else:
        result["fiedler"]["stabilize_step"] = None

    # --- Deviation from init ---
    result["deviation"] = analyze_trajectory(deviation, steps)

    # --- Spectral gap ---
    result["spectral_gap"] = analyze_trajectory(spectral_gap, steps)

    # --- Contrastive loss ---
    result["contrastive"] = analyze_trajectory(contrastive_loss, steps)

    # --- Wall time ---
    if n >= 2 and not np.isnan(wall_time[-1]) and not np.isnan(wall_time[0]):
        total_wall = float(wall_time[-1] - wall_time[0])
        result["wall_time_total_s"] = float(wall_time[-1])
        result["wall_time_per_step"] = total_wall / float(steps[-1] - steps[0])
    else:
        result["wall_time_total_s"] = float(wall_time[0]) if n > 0 else None
        result["wall_time_per_step"] = None

    return result


# ---------------------------------------------------------------------------
# Cross-experiment comparison
# ---------------------------------------------------------------------------
def compare_at_step(analyses: list[dict], all_data: dict, target_step: int) -> list[dict]:
    """
    For each experiment, find the checkpoint closest to target_step
    and extract key metrics for comparison.
    """
    rows = []
    for a in analyses:
        exp_id = a["exp_id"]
        data = all_data[exp_id]
        cps = extract_checkpoints(data)
        if not cps:
            rows.append({"exp_id": exp_id, "label": a["label"], "step": None})
            continue
        # Find closest step <= target_step
        best = None
        for c in cps:
            if c["step"] <= target_step:
                best = c
        if best is None:
            best = cps[0]
        rows.append({
            "exp_id": exp_id,
            "label": a["label"],
            "step": best["step"],
            "val_loss": best.get("val_loss"),
            "co_cross_ratio": best.get("co_cross_ratio"),
            "fiedler_mean": best.get("fiedler_mean"),
            "deviation_mean": best.get("deviation_mean"),
            "spectral_gap": best.get("spectral_gap"),
            "contrastive_loss": best.get("contrastive_loss"),
        })
    return rows


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def fmt(val, decimals=6, width=12):
    """Format a numeric value for table display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—".rjust(width)
    if isinstance(val, int):
        return str(val).rjust(width)
    return f"{val:.{decimals}f}".rjust(width)


def fmt_step(val, width=8):
    if val is None:
        return "—".rjust(width)
    return str(val).rjust(width)


def section_header(title: str) -> str:
    return f"\n{'=' * 72}\n  {title}\n{'=' * 72}\n"


def subsection(title: str) -> str:
    return f"\n--- {title} ---\n"


def format_single_experiment(a: dict) -> str:
    """Format the detailed report for one experiment."""
    lines = []
    lines.append(section_header(f"{a['exp_id']}: {a['label']}"))

    lines.append(f"  Bridge mode:      {a['bridge_mode']}")
    lines.append(f"  Checkpoints:      {a['n_checkpoints']}")
    lines.append(f"  Step range:       {a['step_range'][0]} — {a['step_range'][1]}")
    lines.append(f"  Trainable params: {a['trainable_params']:,}" if isinstance(a['trainable_params'], int) else f"  Trainable params: {a['trainable_params']}")
    lines.append(f"  Bridge params:    {a['bridge_params']:,}" if isinstance(a['bridge_params'], int) else f"  Bridge params:    {a['bridge_params']}")

    if a.get("status") != "OK":
        lines.append(f"\n  STATUS: {a.get('status', 'UNKNOWN')} — insufficient data for analysis.\n")
        return "\n".join(lines)

    # Val loss
    vl = a["val_loss"]
    lines.append(subsection("Val Loss Trajectory"))
    lines.append(f"  First:            {fmt(vl['first'], 6)}")
    lines.append(f"  Min:              {fmt(vl['min'], 6)}")
    lines.append(f"  Last:             {fmt(vl['last'], 6)}")
    lines.append(f"  Trend/100 steps:  {fmt(vl['trend_per_100'], 8)}")
    lines.append(f"  Step < 0.430:     {fmt_step(vl['step_below_043'])}")
    lines.append(f"  Step < 0.425:     {fmt_step(vl['step_below_0425'])}")
    lines.append(f"  Step < 0.420:     {fmt_step(vl['step_below_0420'])}")

    # Train loss
    tl = a["train_loss"]
    lines.append(subsection("Train Loss Trajectory"))
    lines.append(f"  First:            {fmt(tl['first'], 6)}")
    lines.append(f"  Min:              {fmt(tl['min'], 6)}")
    lines.append(f"  Last:             {fmt(tl['last'], 6)}")
    lines.append(f"  Trend/100 steps:  {fmt(tl['trend_per_100'], 8)}")

    # Co/cross ratio
    cc = a["co_cross"]
    lines.append(subsection("Co/Cross Ratio Trajectory"))
    lines.append(f"  First:            {fmt(cc['first'], 2)}")
    lines.append(f"  Peak:             {fmt(cc['max'], 2)}")
    lines.append(f"  Last:             {fmt(cc['last'], 2)}")
    lines.append(f"  Mean (post-warm): {fmt(cc['mean_after_warmup'], 2)}")
    lines.append(f"  Trend/100 steps:  {fmt(cc['trend_per_100'], 4)}")
    lines.append(f"  Plateau step:     {fmt_step(cc['plateau_step'])}")

    # Fiedler
    fi = a["fiedler"]
    lines.append(subsection("Fiedler Value Trajectory"))
    lines.append(f"  Init (step 0):    {fmt(fi['init'], 8)}")
    lines.append(f"  First (step {a['step_range'][0]}): {fmt(fi['first'], 8)}")
    lines.append(f"  Peak:             {fmt(fi['max'], 8)}")
    lines.append(f"  Last:             {fmt(fi['last'], 8)}")
    lines.append(f"  Mean:             {fmt(fi['mean'], 8)}")
    lines.append(f"  Trend/100 steps:  {fmt(fi['trend_per_100'], 10)}")
    lines.append(f"  Stabilize step:   {fmt_step(fi['stabilize_step'])}")

    # Deviation
    dv = a["deviation"]
    lines.append(subsection("Deviation from Init Trajectory"))
    lines.append(f"  First:            {fmt(dv['first'], 6)}")
    lines.append(f"  Last:             {fmt(dv['last'], 6)}")
    lines.append(f"  Trend/100 steps:  {fmt(dv['trend_per_100'], 8)}")

    # Spectral gap
    sg = a["spectral_gap"]
    lines.append(subsection("Spectral Gap"))
    lines.append(f"  First:            {fmt(sg['first'], 6)}")
    lines.append(f"  Last:             {fmt(sg['last'], 6)}")
    lines.append(f"  Mean:             {fmt(sg['mean'], 6)}")

    # Contrastive loss
    cl = a["contrastive"]
    lines.append(subsection("Contrastive Loss"))
    lines.append(f"  First:            {fmt(cl['first'], 8)}")
    lines.append(f"  Last:             {fmt(cl['last'], 8)}")
    lines.append(f"  Trend/100 steps:  {fmt(cl['trend_per_100'], 10)}")

    # Timing
    if a.get("wall_time_per_step"):
        lines.append(subsection("Timing"))
        lines.append(f"  Total wall time:  {a['wall_time_total_s']:.0f}s ({a['wall_time_total_s']/3600:.1f}h)")
        lines.append(f"  Per step:         {a['wall_time_per_step']:.2f}s")

    return "\n".join(lines)


def format_comparison_table(rows: list[dict], step_label: str) -> str:
    """Format a cross-experiment comparison table at a given step."""
    lines = []
    lines.append(f"\n  Comparison at step {step_label}:")
    lines.append(f"  {'Experiment':<35} {'Step':>6} {'Val Loss':>10} {'Co/Cross':>12} {'Fiedler':>12} {'Deviation':>12}")
    lines.append(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    baseline_vl = None
    for r in rows:
        if r.get("step") is None:
            lines.append(f"  {r['label']:<35} {'—':>6} {'—':>10} {'—':>12} {'—':>12} {'—':>12}")
            continue
        vl = r.get("val_loss")
        cc = r.get("co_cross_ratio")
        fi = r.get("fiedler_mean")
        dv = r.get("deviation_mean")
        if baseline_vl is None and vl is not None:
            baseline_vl = vl
        lines.append(
            f"  {r['label']:<35} {r['step']:>6} "
            f"{fmt(vl, 6, 10)} {fmt(cc, 2, 12)} {fmt(fi, 8, 12)} {fmt(dv, 6, 12)}"
        )

    # Delta row if we have a baseline and at least one other
    if baseline_vl is not None and len(rows) > 1:
        lines.append("")
        lines.append(f"  {'DELTA vs baseline':<35} {'':>6} {'Val Loss':>10} {'Co/Cross':>12} {'Fiedler':>12} {'Deviation':>12}")
        lines.append(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
        for r in rows[1:]:
            if r.get("step") is None:
                continue
            vl_d = (r["val_loss"] - baseline_vl) if r.get("val_loss") is not None and baseline_vl is not None else None
            cc_base = rows[0].get("co_cross_ratio")
            cc_d = (r["co_cross_ratio"] - cc_base) if r.get("co_cross_ratio") is not None and cc_base is not None else None
            fi_base = rows[0].get("fiedler_mean")
            fi_d = (r["fiedler_mean"] - fi_base) if r.get("fiedler_mean") is not None and fi_base is not None else None
            dv_base = rows[0].get("deviation_mean")
            dv_d = (r["deviation_mean"] - dv_base) if r.get("deviation_mean") is not None and dv_base is not None else None

            sign_vl = "+" if vl_d and vl_d > 0 else ""
            sign_cc = "+" if cc_d and cc_d > 0 else ""
            sign_fi = "+" if fi_d and fi_d > 0 else ""
            sign_dv = "+" if dv_d and dv_d > 0 else ""

            lines.append(
                f"  {r['label']:<35} {'':>6} "
                f"{sign_vl}{fmt(vl_d, 6, 9)} {sign_cc}{fmt(cc_d, 2, 11)} "
                f"{sign_fi}{fmt(fi_d, 8, 11)} {sign_dv}{fmt(dv_d, 6, 11)}"
            )

    return "\n".join(lines)


def format_convergence_summary(analyses: list[dict]) -> str:
    """Side-by-side convergence speed comparison."""
    lines = []
    lines.append(subsection("Convergence Speed"))
    lines.append(f"  {'Experiment':<35} {'< 0.430':>8} {'< 0.425':>8} {'< 0.420':>8}")
    lines.append(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for a in analyses:
        if a.get("status") != "OK":
            lines.append(f"  {a['label']:<35} {'—':>8} {'—':>8} {'—':>8}")
            continue
        vl = a["val_loss"]
        lines.append(
            f"  {a['label']:<35} "
            f"{fmt_step(vl['step_below_043'])}"
            f"{fmt_step(vl['step_below_0425'])}"
            f"{fmt_step(vl['step_below_0420'])}"
        )
    return "\n".join(lines)


def format_bridge_dynamics(analyses: list[dict]) -> str:
    """Bridge dynamics comparison."""
    lines = []
    lines.append(subsection("Bridge Dynamics"))
    lines.append(f"  {'Experiment':<35} {'Co/Cross Plateau':>18} {'Fiedler Stable':>16} {'Final Co/Cross':>16} {'Final Fiedler':>16}")
    lines.append(f"  {'-'*35} {'-'*18} {'-'*16} {'-'*16} {'-'*16}")
    for a in analyses:
        if a.get("status") != "OK":
            lines.append(f"  {a['label']:<35} {'—':>18} {'—':>16} {'—':>16} {'—':>16}")
            continue
        cc_plat = a["co_cross"].get("plateau_step")
        fi_stab = a["fiedler"].get("stabilize_step")
        cc_final = a["co_cross"]["last"]
        fi_final = a["fiedler"]["last"]
        lines.append(
            f"  {a['label']:<35} "
            f"{fmt_step(cc_plat, 18)}"
            f"{fmt_step(fi_stab, 16)}"
            f"{fmt(cc_final, 1, 16)}"
            f"{fmt(fi_final, 8, 16)}"
        )
    return "\n".join(lines)


def generate_markdown(analyses: list[dict], comparison_rows: list[dict],
                      match_step: int) -> str:
    """Generate the full markdown report."""
    lines = []
    lines.append("# Corpus Baseline Comparison Report")
    lines.append(f"\n*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append(f"\n**Experiments loaded:** {len(analyses)}")
    for a in analyses:
        status = "FULL" if a.get("status") == "OK" else f"PARTIAL ({a['n_checkpoints']} checkpoints)"
        lines.append(f"- **{a['exp_id']}** — {a['label']} [{status}, steps {a['step_range'][0]}–{a['step_range'][1]}]")

    # Per-experiment details
    lines.append("\n## Per-Experiment Analysis")
    for a in analyses:
        lines.append(f"\n### {a['exp_id']}: {a['label']}")
        lines.append("")
        if a.get("status") != "OK":
            lines.append(f"*Insufficient data ({a['n_checkpoints']} checkpoints). Partial metrics only.*\n")
            if a["n_checkpoints"] > 0:
                lines.append(f"| Metric | Value |")
                lines.append(f"|--------|-------|")
                lines.append(f"| Val loss (step {a['step_range'][0]}) | {a['val_loss']['first']:.6f} |")
                lines.append(f"| Co/cross ratio | {a['co_cross']['first']:.2f} |")
                lines.append(f"| Fiedler mean | {a['fiedler']['first']:.8f} |")
                lines.append(f"| Deviation mean | {a['deviation']['first']:.6f} |")
            continue

        vl = a["val_loss"]
        tl = a["train_loss"]
        cc = a["co_cross"]
        fi = a["fiedler"]
        dv = a["deviation"]

        def md_fmt(val, dec):
            """Format a value for markdown, handling None."""
            if val is None:
                return "—"
            return f"{val:.{dec}f}"

        lines.append("| Metric | First | Min | Last | Trend/100 |")
        lines.append("|--------|-------|-----|------|-----------|")
        lines.append(f"| Val loss | {md_fmt(vl['first'],6)} | {md_fmt(vl['min'],6)} | {md_fmt(vl['last'],6)} | {md_fmt(vl['trend_per_100'],8)} |")
        lines.append(f"| Train loss | {md_fmt(tl['first'],6)} | {md_fmt(tl['min'],6)} | {md_fmt(tl['last'],6)} | {md_fmt(tl['trend_per_100'],8)} |")
        lines.append(f"| Co/cross ratio | {md_fmt(cc['first'],1)} | — | {md_fmt(cc['last'],1)} | {md_fmt(cc['trend_per_100'],2)} |")
        lines.append(f"| Fiedler | {md_fmt(fi['first'],8)} | — | {md_fmt(fi['last'],8)} | {md_fmt(fi['trend_per_100'],10)} |")
        lines.append(f"| Deviation | {md_fmt(dv['first'],6)} | — | {md_fmt(dv['last'],6)} | {md_fmt(dv['trend_per_100'],8)} |")

        lines.append("")
        lines.append("**Convergence milestones:**")
        lines.append(f"- Val loss < 0.430 at step **{vl['step_below_043'] or '—'}**")
        lines.append(f"- Val loss < 0.425 at step **{vl['step_below_0425'] or '—'}**")
        lines.append(f"- Val loss < 0.420 at step **{vl['step_below_0420'] or '—'}**")

        lines.append("")
        lines.append("**Bridge dynamics:**")
        lines.append(f"- Co/cross ratio plateau: step **{cc['plateau_step'] or '—'}**")
        lines.append(f"- Co/cross mean (post-warmup): **{cc['mean_after_warmup']:.1f}**")
        lines.append(f"- Fiedler stabilization: step **{fi['stabilize_step'] or '—'}**")

    # Cross-experiment comparison
    if len(analyses) > 1:
        lines.append(f"\n## Cross-Experiment Comparison (at step {match_step})")
        lines.append("")
        lines.append("| Experiment | Step | Val Loss | Co/Cross | Fiedler | Deviation |")
        lines.append("|-----------|------|----------|----------|---------|-----------|")
        baseline = comparison_rows[0] if comparison_rows else None
        for r in comparison_rows:
            if r.get("step") is None:
                lines.append(f"| {r['label']} | — | — | — | — | — |")
                continue
            lines.append(
                f"| {r['label']} | {r['step']} | "
                f"{r['val_loss']:.6f} | {r['co_cross_ratio']:.1f} | "
                f"{r['fiedler_mean']:.8f} | {r['deviation_mean']:.6f} |"
            )

        if baseline and baseline.get("step") is not None:
            lines.append("")
            lines.append("**Deltas vs C-001 baseline:**")
            lines.append("")
            lines.append("| Experiment | Val Loss Delta | Co/Cross Delta | Fiedler Delta | Deviation Delta |")
            lines.append("|-----------|----------------|----------------|---------------|-----------------|")
            for r in comparison_rows[1:]:
                if r.get("step") is None:
                    continue
                vld = r["val_loss"] - baseline["val_loss"] if r.get("val_loss") is not None else None
                ccd = r["co_cross_ratio"] - baseline["co_cross_ratio"] if r.get("co_cross_ratio") is not None else None
                fid = r["fiedler_mean"] - baseline["fiedler_mean"] if r.get("fiedler_mean") is not None else None
                dvd = r["deviation_mean"] - baseline["deviation_mean"] if r.get("deviation_mean") is not None else None

                def delta_str(val, dec):
                    if val is None:
                        return "—"
                    sign = "+" if val > 0 else ""
                    return f"{sign}{val:.{dec}f}"

                lines.append(
                    f"| {r['label']} | "
                    f"{delta_str(vld, 6)} | "
                    f"{delta_str(ccd, 1)} | "
                    f"{delta_str(fid, 8)} | "
                    f"{delta_str(dvd, 6)} |"
                )

        # Convergence speed table
        lines.append(f"\n### Convergence Speed")
        lines.append("")
        lines.append("| Experiment | Step < 0.430 | Step < 0.425 | Step < 0.420 |")
        lines.append("|-----------|-------------|-------------|-------------|")
        for a in analyses:
            if a.get("status") != "OK":
                lines.append(f"| {a['label']} | — | — | — |")
                continue
            vl = a["val_loss"]
            lines.append(
                f"| {a['label']} | {vl['step_below_043'] or '—'} | "
                f"{vl['step_below_0425'] or '—'} | {vl['step_below_0420'] or '—'} |"
            )

    # Interpretation
    lines.append("\n## Key Observations")
    lines.append("")
    for a in analyses:
        if a.get("status") != "OK" or a["n_checkpoints"] < 2:
            lines.append(f"- **{a['exp_id']}**: Only {a['n_checkpoints']} checkpoint(s) — analysis pending more data.")
            if a["n_checkpoints"] == 1:
                vl = a["val_loss"]
                fi = a["fiedler"]
                lines.append(f"  First snapshot: val_loss={vl['first']:.6f}, fiedler={fi['first']:.8f}")
            continue
        vl = a["val_loss"]
        cc = a["co_cross"]
        fi = a["fiedler"]
        trend_1k = vl['trend_per_100'] * 10 if vl['trend_per_100'] is not None else 0
        cc_warmup = cc['mean_after_warmup'] if cc['mean_after_warmup'] is not None else cc['mean']
        lines.append(f"- **{a['exp_id']} ({a['label']})**: Val loss {vl['first']:.4f} -> {vl['last']:.4f} "
                      f"({trend_1k:.5f}/1k steps). "
                      f"Co/cross ratio reached {cc['max']:.0f} (mean post-warmup: {cc_warmup:.0f}). "
                      f"Fiedler {fi['first']:.6f} -> {fi['last']:.6f}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(section_header("CORPUS BASELINE COMPARISON"))
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Source:    {BASELINES_DIR}")
    print()

    # Load all available experiments
    loaded = {}
    for exp_id in EXPERIMENTS:
        data = load_experiment(exp_id)
        if data is not None:
            loaded[exp_id] = data
            cps = extract_checkpoints(data)
            print(f"  {exp_id}: LOADED — {len(cps)} checkpoints, "
                  f"steps {cps[0]['step']}–{cps[-1]['step']}" if cps else
                  f"  {exp_id}: LOADED — 0 checkpoints")
        else:
            print(f"  {exp_id}: not found (skipped)")

    if not loaded:
        print("\n  ERROR: No experiment data found. Exiting.")
        sys.exit(1)

    # Analyze each experiment
    analyses = []
    for exp_id in EXPERIMENTS:
        if exp_id in loaded:
            a = analyze_experiment(loaded[exp_id], exp_id)
            analyses.append(a)

    # Print per-experiment reports
    for a in analyses:
        print(format_single_experiment(a))

    # Cross-experiment comparison
    if len(analyses) > 1:
        # Find the step count of the shortest experiment
        min_step = min(a["total_steps"] for a in analyses if a.get("status") == "OK")
        match_step = min_step

        print(section_header(f"CROSS-EXPERIMENT COMPARISON (at step {match_step})"))
        comparison_rows = compare_at_step(analyses, loaded, match_step)
        print(format_comparison_table(comparison_rows, str(match_step)))
        print(format_convergence_summary(analyses))
        print(format_bridge_dynamics(analyses))
    else:
        match_step = analyses[0]["total_steps"] if analyses else 0
        comparison_rows = []
        print("\n  (Only one experiment loaded — cross-comparison skipped.)")

    # Save markdown report
    md = generate_markdown(analyses, comparison_rows, match_step)
    output_path = BASELINES_DIR / "BASELINE_COMPARISON.md"
    output_path.write_text(md, encoding="utf-8")
    print(f"\n  Report saved to: {output_path}")


if __name__ == "__main__":
    main()
