#!/usr/bin/env python3
"""
Channel Ablation Analysis — Decision Gate for n=6 (RD Geometry)

Loads results from H1-H5 (n_channels={3,4,6,8,12}) and determines whether
n=6 wins on the val_loss/bridge_params ratio, validating the rhombic
dodecahedron geometry thesis.

Decision gate criteria:
  - n=6 should have the BEST val_loss / bridge_params efficiency
  - If n=3 or n=4 matches n=6 on val_loss with fewer params, n=6 thesis weakens
  - If n=8 or n=12 beats n=6 on val_loss, extra connectivity helps (but costs more)

Output: stdout + results/channel-ablation/CHANNEL_ABLATION_REPORT.md
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "channel-ablation"

CHANNELS = {
    "H-ch3":  {"n": 3,  "channel_size": 8, "bridge_params": 9},
    "H-ch4":  {"n": 4,  "channel_size": 6, "bridge_params": 16},
    "H-ch6":  {"n": 6,  "channel_size": 4, "bridge_params": 36},
    "H-ch8":  {"n": 8,  "channel_size": 3, "bridge_params": 64},
    "H-ch12": {"n": 12, "channel_size": 2, "bridge_params": 144},
}

RANK = 24  # All runs use rank 24


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_run(run_id: str) -> Optional[dict]:
    """Load results.json for a run, trying multiple path patterns."""
    meta = CHANNELS[run_id]

    # Try primary path (results.json in run dir)
    for fname in ["results.json", f"{run_id}-results.json"]:
        path = RESULTS_DIR / run_id / fname
        if path.exists():
            with open(path) as f:
                return json.load(f)

    # Try flat path (e.g., H-ch3-results.json at RESULTS_DIR level)
    flat = RESULTS_DIR / f"{run_id}-results.json"
    if flat.exists():
        with open(flat) as f:
            return json.load(f)

    return None


def extract_checkpoints(data: dict) -> list[dict]:
    """Extract checkpoint list from results data."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "checkpoints" in data:
        return data["checkpoints"]
    return []


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def analyze_trajectory(checkpoints: list[dict]) -> dict:
    """Analyze training trajectory from checkpoint list."""
    if not checkpoints:
        return {"status": "no_data"}

    steps = [cp.get("step", 0) for cp in checkpoints]
    losses = [cp.get("val_loss", cp.get("lm_loss", None)) for cp in checkpoints]
    losses = [l for l in losses if l is not None]

    fiedlers = []
    for cp in checkpoints:
        f = cp.get("fiedler_mean", cp.get("fiedler", None))
        if f is not None:
            fiedlers.append(f)

    deviations = []
    for cp in checkpoints:
        d = cp.get("deviation_mean", cp.get("deviation", None))
        if d is not None:
            deviations.append(d)

    result = {
        "total_checkpoints": len(checkpoints),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "status": "complete" if steps and steps[-1] >= 9900 else "in_progress",
    }

    if losses:
        result["final_val_loss"] = losses[-1]
        result["best_val_loss"] = min(losses)
        result["val_loss_trajectory"] = losses

    if fiedlers:
        result["final_fiedler"] = fiedlers[-1]
        result["max_fiedler"] = max(fiedlers)
        result["fiedler_trajectory"] = fiedlers

    if deviations:
        result["final_deviation"] = deviations[-1]
        result["deviation_trajectory"] = deviations

    return result


def compute_efficiency(val_loss: float, bridge_params: int, baseline_loss: float = 0.50) -> float:
    """Compute efficiency: loss improvement per bridge parameter.

    Higher is better. Measures how much loss reduction each bridge parameter buys.
    baseline_loss is the estimated loss at step 0 (before any training).
    """
    improvement = baseline_loss - val_loss
    if bridge_params == 0:
        return float('inf') if improvement > 0 else 0.0
    return improvement / bridge_params


def find_convergence_step(losses: list[float], steps: list[int], threshold: float = 0.001) -> Optional[int]:
    """Find the step where loss improvement drops below threshold (per 100 steps)."""
    for i in range(1, len(losses)):
        rate = (losses[i - 1] - losses[i]) / max(steps[i] - steps[i - 1], 1) * 100
        if rate < threshold and i > len(losses) // 4:  # Only after 25% of training
            return steps[i]
    return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(results: dict[str, dict]) -> str:
    """Generate structured comparison report."""
    lines = []
    lines.append("# Channel Ablation Analysis — Decision Gate Report")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nRank: {RANK} (fixed across all runs)")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Run | n_channels | ch_size | bridge_params | Steps | Final Val Loss | Best Val Loss | Fiedler | Deviation | Efficiency |")
    lines.append("|-----|-----------|---------|--------------|-------|---------------|--------------|---------|-----------|------------|")

    for run_id, meta in CHANNELS.items():
        r = results.get(run_id)
        if r is None or r.get("status") == "no_data":
            lines.append(f"| {run_id} | {meta['n']} | {meta['channel_size']} | {meta['bridge_params']} | — | — | — | — | — | — |")
            continue

        steps = r.get("last_step", "?")
        final_loss = r.get("final_val_loss", None)
        best_loss = r.get("best_val_loss", None)
        fiedler = r.get("final_fiedler", None)
        deviation = r.get("final_deviation", None)

        eff = ""
        if final_loss is not None:
            eff_val = compute_efficiency(final_loss, meta["bridge_params"])
            eff = f"{eff_val:.6f}"

        fl = f"{final_loss:.4f}" if final_loss is not None else "—"
        bl = f"{best_loss:.4f}" if best_loss is not None else "—"
        fi = f"{fiedler:.5f}" if fiedler is not None else "—"
        dv = f"{deviation:.3f}" if deviation is not None else "—"

        lines.append(
            f"| {run_id} | {meta['n']} | {meta['channel_size']} | {meta['bridge_params']} "
            f"| {steps} | {fl} | {bl} | {fi} | {dv} | {eff} |"
        )

    lines.append("")

    # Decision gate evaluation
    lines.append("## Decision Gate: Does n=6 Win?")
    lines.append("")

    complete_runs = {k: v for k, v in results.items() if v and v.get("final_val_loss") is not None}

    if len(complete_runs) < 2:
        lines.append("**INSUFFICIENT DATA** — Need at least 2 completed runs for comparison.")
        lines.append(f"Currently have {len(complete_runs)} run(s) with data.")
    else:
        # Sort by efficiency
        efficiencies = {}
        for run_id, r in complete_runs.items():
            meta = CHANNELS[run_id]
            efficiencies[run_id] = compute_efficiency(r["final_val_loss"], meta["bridge_params"])

        sorted_by_eff = sorted(efficiencies.items(), key=lambda x: x[1], reverse=True)

        lines.append("### Efficiency Ranking (loss improvement per bridge parameter)")
        lines.append("")
        for i, (run_id, eff) in enumerate(sorted_by_eff, 1):
            meta = CHANNELS[run_id]
            r = complete_runs[run_id]
            marker = " **← WINNER**" if i == 1 else ""
            lines.append(f"{i}. **{run_id}** (n={meta['n']}): {eff:.6f} loss/param, "
                        f"val_loss={r['final_val_loss']:.4f}, "
                        f"bridge_params={meta['bridge_params']}{marker}")

        lines.append("")

        # Sort by raw val_loss
        sorted_by_loss = sorted(complete_runs.items(), key=lambda x: x[1]["final_val_loss"])

        lines.append("### Raw Performance Ranking (best val_loss)")
        lines.append("")
        for i, (run_id, r) in enumerate(sorted_by_loss, 1):
            meta = CHANNELS[run_id]
            lines.append(f"{i}. **{run_id}** (n={meta['n']}): val_loss={r['final_val_loss']:.4f}, "
                        f"bridge_params={meta['bridge_params']}")

        lines.append("")

        # The verdict
        eff_winner = sorted_by_eff[0][0]
        loss_winner = sorted_by_loss[0][0]

        lines.append("### Verdict")
        lines.append("")

        if eff_winner == "H-ch6":
            lines.append("**n=6 WINS on efficiency.** The RD geometry thesis is supported: "
                        "6 channels (36 bridge parameters) provides the best loss improvement "
                        "per parameter. The rhombic dodecahedron's 6 face-pair directions are "
                        "the natural decomposition.")
        elif CHANNELS[eff_winner]["n"] < 6:
            lines.append(f"**n={CHANNELS[eff_winner]['n']} WINS on efficiency.** "
                        f"Fewer channels achieve better cost-effectiveness. "
                        f"The RD geometry thesis is WEAKENED — the system doesn't need "
                        f"all 6 face-pair directions. Consider: does the smaller bridge "
                        f"still show geometric structure?")
        else:
            lines.append(f"**n={CHANNELS[eff_winner]['n']} WINS on efficiency.** "
                        f"More channels improve cost-effectiveness. The RD geometry may be "
                        f"a minimum, not an optimum. Investigate whether the extra channels "
                        f"correspond to meaningful geometric directions.")

        if loss_winner != eff_winner:
            lines.append("")
            lines.append(f"Note: n={CHANNELS[loss_winner]['n']} has the best raw val_loss, "
                        f"but n={CHANNELS[eff_winner]['n']} wins on efficiency. "
                        f"The extra parameters of n={CHANNELS[loss_winner]['n']} buy marginal "
                        f"loss improvement at diminishing returns.")

    lines.append("")

    # Fiedler comparison
    fiedler_data = {k: v for k, v in complete_runs.items() if v.get("final_fiedler") is not None}
    if fiedler_data:
        lines.append("## Fiedler Value Comparison")
        lines.append("")
        lines.append("Note: Fiedler value depends on graph size (n_channels). "
                     "Larger bridges produce higher absolute Fiedler values by construction. "
                     "Normalize by n_channels for fair comparison.")
        lines.append("")
        for run_id, r in sorted(fiedler_data.items(), key=lambda x: CHANNELS[x[0]]["n"]):
            meta = CHANNELS[run_id]
            fiedler = r["final_fiedler"]
            normalized = fiedler / meta["n"] if meta["n"] > 0 else 0
            lines.append(f"- **{run_id}** (n={meta['n']}): Fiedler={fiedler:.5f}, "
                        f"normalized={normalized:.5f}")

    lines.append("")

    # Matched-step comparison
    lines.append("## Matched-Step Comparison")
    lines.append("")
    lines.append("Comparing all runs at common checkpoints for fair evaluation.")
    lines.append("")

    # Find common steps
    all_steps = {}
    for run_id, r in complete_runs.items():
        if "val_loss_trajectory" in r:
            cps = extract_checkpoints(load_run(run_id) or {})
            for cp in cps:
                step = cp.get("step", 0)
                loss = cp.get("val_loss", cp.get("lm_loss"))
                if loss is not None:
                    all_steps.setdefault(step, {})[run_id] = loss

    # Find steps where all completed runs have data
    common_steps = sorted(s for s, d in all_steps.items()
                         if len(d) >= len(complete_runs))

    if common_steps:
        milestone_steps = [s for s in common_steps if s % 1000 == 0] or common_steps[-3:]
        lines.append("| Step | " + " | ".join(f"{k} (n={CHANNELS[k]['n']})" for k in complete_runs) + " |")
        lines.append("|------|" + "|".join("------" for _ in complete_runs) + "|")

        for step in milestone_steps:
            vals = all_steps.get(step, {})
            row = [f"{vals.get(k, '—'):.4f}" if k in vals else "—" for k in complete_runs]
            lines.append(f"| {step} | " + " | ".join(row) + " |")
    else:
        lines.append("No common checkpoint steps found across runs.")

    lines.append("")

    # Implications for Paper 3
    lines.append("## Implications for Paper 3 (RhombiLoRA)")
    lines.append("")
    eff_winner_n6 = (len(complete_runs) >= 2 and
                     sorted(complete_runs.items(),
                            key=lambda x: compute_efficiency(x[1]["final_val_loss"],
                                                             CHANNELS[x[0]]["bridge_params"]),
                            reverse=True)[0][0] == "H-ch6")
    if eff_winner_n6:
        lines.append("The channel ablation supports the n=6 design choice for RhombiLoRA. "
                     "The 6 bridge channels correspond to the 6 face-pair directions of the "
                     "rhombic dodecahedron — the Voronoi cell of the FCC lattice. This is "
                     "the geometric argument: the number of bridge directions is not arbitrary "
                     "but determined by the topology.")
    else:
        lines.append("Results pending full ablation completion. The decision gate requires "
                     "all 5 channel counts to be evaluated before concluding.")

    lines.append("")
    lines.append("---")
    lines.append(f"*Analysis generated by analyze_channel_ablation.py at {datetime.now().isoformat()}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Channel Ablation Analysis — Decision Gate for n=6")
    print("=" * 70)
    print()

    # Load all available runs
    results = {}
    for run_id in CHANNELS:
        data = load_run(run_id)
        if data is None:
            print(f"  {run_id}: NOT FOUND")
            results[run_id] = {"status": "no_data"}
            continue

        checkpoints = extract_checkpoints(data)
        analysis = analyze_trajectory(checkpoints)
        results[run_id] = analysis

        status = analysis.get("status", "unknown")
        step = analysis.get("last_step", "?")
        loss = analysis.get("final_val_loss")
        loss_str = f"{loss:.4f}" if loss is not None else "—"
        print(f"  {run_id}: {status}, step {step}, val_loss={loss_str}")

    print()

    # Generate report
    report = generate_report(results)
    print(report)

    # Write to file
    output_path = RESULTS_DIR / "CHANNEL_ABLATION_REPORT.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
