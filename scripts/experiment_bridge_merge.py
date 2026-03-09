"""Phase 2A: Bridge-Level Adapter Merging Experiment.

Tests whether adapters can be merged at the bridge level (36 parameters)
instead of full weight interpolation. INFERENCE ONLY — no training, no GPU.

Given three sets of final bridge matrices (alpaca, code, math) from Phase 1A,
for each pair of tasks and each alpha in [0.0, 0.1, ..., 1.0]:

  1. Weight merge (baseline): merged = (1-alpha)*bridge_A + alpha*bridge_B
  2. Bridge swap: use task A's bridge entirely, or task B's bridge entirely

For each merged bridge, compute:
  - Bridge Fiedler value
  - Deviation from identity
  - Frobenius distance to each source bridge
  - Cosine similarity of eigenspectra between merged and source bridges

Output: results/bridge_merge/
  - MERGE_REPORT.md     — full analysis
  - merge_results.json  — all numerical results

Usage:
  python scripts/experiment_bridge_merge.py \\
    --alpaca results/exp2/rhombi_fcc_r24 \\
    --code results/fingerprints/code \\
    --math results/fingerprints/math \\
    --output results/bridge_merge
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# Import bridge loading from Phase 1A analysis script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_task_bridges import load_final_bridges, parse_bridge_filename


# ── Metrics ──────────────────────────────────────────────────────────────

def fiedler_value(B: np.ndarray) -> float:
    """Algebraic connectivity of bridge treated as weighted adjacency graph."""
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


def frobenius_distance(B1: np.ndarray, B2: np.ndarray) -> float:
    """Frobenius distance between two matrices."""
    return float(np.linalg.norm(B1 - B2, "fro"))


def eigenspectrum(B: np.ndarray) -> np.ndarray:
    """Sorted eigenvalues of a bridge matrix."""
    return np.sort(np.linalg.eigvalsh(B))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def eigenspectrum_cosine(B1: np.ndarray, B2: np.ndarray) -> float:
    """Cosine similarity between eigenspectra of two bridge matrices."""
    return cosine_similarity(eigenspectrum(B1), eigenspectrum(B2))


def compute_merge_metrics(
    merged: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
) -> dict:
    """Compute all metrics for a merged bridge relative to its sources."""
    return {
        "fiedler": fiedler_value(merged),
        "deviation": deviation_from_identity(merged),
        "dist_to_a": frobenius_distance(merged, source_a),
        "dist_to_b": frobenius_distance(merged, source_b),
        "eig_cos_a": eigenspectrum_cosine(merged, source_a),
        "eig_cos_b": eigenspectrum_cosine(merged, source_b),
    }


# ── Random Baseline ─────────────────────────────────────────────────────

def random_bridge(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric bridge matrix with same scale as identity."""
    R = rng.standard_normal(shape)
    return (R + R.T) / 2  # symmetrize


# ── Merge Strategies ─────────────────────────────────────────────────────

def interpolate_bridges(
    bridges_a: dict[str, np.ndarray],
    bridges_b: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    """Linear interpolation: merged = (1-alpha)*A + alpha*B."""
    common = sorted(set(bridges_a.keys()) & set(bridges_b.keys()))
    return {
        key: (1 - alpha) * bridges_a[key] + alpha * bridges_b[key]
        for key in common
    }


# ── Analysis Engine ──────────────────────────────────────────────────────

def extract_module_type(key: str) -> str:
    """Extract q_proj/k_proj/v_proj/o_proj from adapter key like L00_q_proj."""
    parts = key.split("_", 1)
    return parts[1] if len(parts) > 1 else key


def analyze_pair(
    task_a: str,
    bridges_a: dict[str, np.ndarray],
    task_b: str,
    bridges_b: dict[str, np.ndarray],
    alphas: list[float],
    rng: np.random.Generator,
) -> dict:
    """Run the full merge sweep for one task pair."""
    common_keys = sorted(set(bridges_a.keys()) & set(bridges_b.keys()))
    if not common_keys:
        return {"error": "no common adapter keys"}

    pair_name = f"{task_a}_vs_{task_b}"
    n_adapters = len(common_keys)

    # Pre-compute source metrics for reference
    source_a_fiedlers = [fiedler_value(bridges_a[k]) for k in common_keys]
    source_b_fiedlers = [fiedler_value(bridges_b[k]) for k in common_keys]
    source_a_deviations = [deviation_from_identity(bridges_a[k]) for k in common_keys]
    source_b_deviations = [deviation_from_identity(bridges_b[k]) for k in common_keys]

    # Sweep alpha values
    sweep_results = []
    for alpha in alphas:
        merged = interpolate_bridges(bridges_a, bridges_b, alpha)

        # Per-adapter metrics
        per_adapter = {}
        for key in common_keys:
            metrics = compute_merge_metrics(merged[key], bridges_a[key], bridges_b[key])
            metrics["module_type"] = extract_module_type(key)
            per_adapter[key] = metrics

        # Aggregate across all adapters
        all_fiedlers = [per_adapter[k]["fiedler"] for k in common_keys]
        all_deviations = [per_adapter[k]["deviation"] for k in common_keys]
        all_dist_a = [per_adapter[k]["dist_to_a"] for k in common_keys]
        all_dist_b = [per_adapter[k]["dist_to_b"] for k in common_keys]
        all_eig_cos_a = [per_adapter[k]["eig_cos_a"] for k in common_keys]
        all_eig_cos_b = [per_adapter[k]["eig_cos_b"] for k in common_keys]

        # Per module-type aggregation
        module_types = ["q_proj", "k_proj", "v_proj", "o_proj"]
        per_module = {}
        for mod in module_types:
            mod_keys = [k for k in common_keys if extract_module_type(k) == mod]
            if mod_keys:
                per_module[mod] = {
                    "mean_fiedler": float(np.mean([per_adapter[k]["fiedler"] for k in mod_keys])),
                    "mean_deviation": float(np.mean([per_adapter[k]["deviation"] for k in mod_keys])),
                    "mean_dist_a": float(np.mean([per_adapter[k]["dist_to_a"] for k in mod_keys])),
                    "mean_dist_b": float(np.mean([per_adapter[k]["dist_to_b"] for k in mod_keys])),
                    "mean_eig_cos_a": float(np.mean([per_adapter[k]["eig_cos_a"] for k in mod_keys])),
                    "mean_eig_cos_b": float(np.mean([per_adapter[k]["eig_cos_b"] for k in mod_keys])),
                    "n_adapters": len(mod_keys),
                }

        sweep_results.append({
            "alpha": alpha,
            "mean_fiedler": float(np.mean(all_fiedlers)),
            "std_fiedler": float(np.std(all_fiedlers)),
            "mean_deviation": float(np.mean(all_deviations)),
            "std_deviation": float(np.std(all_deviations)),
            "mean_dist_to_a": float(np.mean(all_dist_a)),
            "mean_dist_to_b": float(np.mean(all_dist_b)),
            "mean_eig_cos_a": float(np.mean(all_eig_cos_a)),
            "mean_eig_cos_b": float(np.mean(all_eig_cos_b)),
            "per_module": per_module,
        })

    # Random bridge baseline: merge each source with a random bridge at alpha=0.5
    n_random_trials = 20
    random_fiedlers = []
    random_deviations = []
    random_eig_cos = []
    shape = bridges_a[common_keys[0]].shape
    for _ in range(n_random_trials):
        for key in common_keys:
            R = random_bridge(shape, rng)
            merged_rand = 0.5 * bridges_a[key] + 0.5 * R
            random_fiedlers.append(fiedler_value(merged_rand))
            random_deviations.append(deviation_from_identity(merged_rand))
            random_eig_cos.append(eigenspectrum_cosine(merged_rand, bridges_a[key]))

    # Linearity analysis: check if Fiedler trajectory is linear vs shows phase transitions
    fiedler_trajectory = [r["mean_fiedler"] for r in sweep_results]
    fiedler_arr = np.array(fiedler_trajectory)
    alpha_arr = np.array(alphas)

    # Fit linear model to Fiedler trajectory
    if len(alphas) > 2:
        coeffs = np.polyfit(alpha_arr, fiedler_arr, 1)
        linear_pred = np.polyval(coeffs, alpha_arr)
        residuals = fiedler_arr - linear_pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((fiedler_arr - np.mean(fiedler_arr)) ** 2))
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)
        max_residual = float(np.max(np.abs(residuals)))
    else:
        r_squared = 1.0
        max_residual = 0.0

    # Task structure preservation: at alpha=0.1, how much of source A's
    # eigenspectrum is retained?
    low_alpha_result = next((r for r in sweep_results if abs(r["alpha"] - 0.1) < 0.01), None)
    high_alpha_result = next((r for r in sweep_results if abs(r["alpha"] - 0.9) < 0.01), None)

    return {
        "pair": pair_name,
        "n_adapters": n_adapters,
        "source_a": {
            "task": task_a,
            "mean_fiedler": float(np.mean(source_a_fiedlers)),
            "std_fiedler": float(np.std(source_a_fiedlers)),
            "mean_deviation": float(np.mean(source_a_deviations)),
        },
        "source_b": {
            "task": task_b,
            "mean_fiedler": float(np.mean(source_b_fiedlers)),
            "std_fiedler": float(np.std(source_b_fiedlers)),
            "mean_deviation": float(np.mean(source_b_deviations)),
        },
        "sweep": sweep_results,
        "linearity": {
            "r_squared": r_squared,
            "max_residual": max_residual,
            "is_linear": r_squared > 0.95,
        },
        "random_baseline": {
            "mean_fiedler": float(np.mean(random_fiedlers)),
            "std_fiedler": float(np.std(random_fiedlers)),
            "mean_deviation": float(np.mean(random_deviations)),
            "mean_eig_cos": float(np.mean(random_eig_cos)),
            "n_trials": n_random_trials * n_adapters,
        },
        "structure_preservation": {
            "eig_cos_a_at_alpha_0.1": low_alpha_result["mean_eig_cos_a"] if low_alpha_result else None,
            "eig_cos_b_at_alpha_0.9": high_alpha_result["mean_eig_cos_b"] if high_alpha_result else None,
        },
    }


# ── Report Generation ────────────────────────────────────────────────────

def write_report(pair_results: list[dict], output_dir: Path):
    """Generate MERGE_REPORT.md from all pair results."""
    lines = [
        "# Phase 2A: Bridge-Level Adapter Merging",
        "",
        "> **Experiment:** Interpolate final bridge matrices between task-specific",
        "> adapters and measure structural preservation across the merge trajectory.",
        "> No GPU required — pure linear algebra on 36-parameter bridges.",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Pair | Linear (R^2) | Phase Transition | Best Structure Retention |")
    lines.append("|------|-------------|------------------|-------------------------|")
    for pr in pair_results:
        lin = pr["linearity"]
        label = "YES" if lin["is_linear"] else f"NO (R^2={lin['r_squared']:.3f})"
        phase = "None detected" if lin["is_linear"] else f"Max residual: {lin['max_residual']:.5f}"
        sp = pr["structure_preservation"]
        ret_a = f"{sp['eig_cos_a_at_alpha_0.1']:.4f}" if sp["eig_cos_a_at_alpha_0.1"] is not None else "—"
        lines.append(f"| {pr['pair']} | {label} | {phase} | cos(A)@0.1 = {ret_a} |")

    # Per-pair detailed analysis
    for pr in pair_results:
        lines.append("")
        lines.append(f"## {pr['pair']}")
        lines.append("")
        lines.append(f"**Adapters:** {pr['n_adapters']} common positions")
        lines.append(f"**Source A ({pr['source_a']['task']}):** "
                     f"Fiedler = {pr['source_a']['mean_fiedler']:.5f} +/- {pr['source_a']['std_fiedler']:.5f}, "
                     f"Deviation = {pr['source_a']['mean_deviation']:.4f}")
        lines.append(f"**Source B ({pr['source_b']['task']}):** "
                     f"Fiedler = {pr['source_b']['mean_fiedler']:.5f} +/- {pr['source_b']['std_fiedler']:.5f}, "
                     f"Deviation = {pr['source_b']['mean_deviation']:.4f}")
        lines.append("")

        # Interpolation curve table
        lines.append("### Interpolation Curve")
        lines.append("")
        lines.append("| alpha | Fiedler | Deviation | dist(A) | dist(B) | cos(eig,A) | cos(eig,B) |")
        lines.append("|-------|---------|-----------|---------|---------|------------|------------|")
        for s in pr["sweep"]:
            lines.append(
                f"| {s['alpha']:.1f} | {s['mean_fiedler']:.5f} | "
                f"{s['mean_deviation']:.4f} | {s['mean_dist_to_a']:.5f} | "
                f"{s['mean_dist_to_b']:.5f} | {s['mean_eig_cos_a']:.4f} | "
                f"{s['mean_eig_cos_b']:.4f} |"
            )

        # Linearity analysis
        lin = pr["linearity"]
        lines.append("")
        lines.append("### Linearity Analysis")
        lines.append("")
        lines.append(f"**R^2 (Fiedler vs alpha):** {lin['r_squared']:.5f}")
        lines.append(f"**Max residual:** {lin['max_residual']:.6f}")
        if lin["is_linear"]:
            lines.append("**Verdict:** Fiedler trajectory is LINEAR. No phase transitions detected.")
            lines.append("Bridge interpolation is smooth — no catastrophic collapse or sudden "
                        "reorganization at any mixing ratio.")
        else:
            lines.append(f"**Verdict:** Fiedler trajectory is NON-LINEAR (R^2 = {lin['r_squared']:.3f}).")
            lines.append("Phase transition or curvature detected in the merge trajectory. "
                        "Some alpha values cause disproportionate structural change.")

        # Structure preservation
        sp = pr["structure_preservation"]
        lines.append("")
        lines.append("### Structure Preservation")
        lines.append("")
        if sp["eig_cos_a_at_alpha_0.1"] is not None:
            cos_a = sp["eig_cos_a_at_alpha_0.1"]
            lines.append(f"**Eigenspectrum retention at alpha=0.1 (90% task A):** "
                        f"cos = {cos_a:.4f}")
            if cos_a > 0.99:
                lines.append("Task A's spectral structure is almost perfectly preserved "
                            "with 10% contamination from task B.")
            elif cos_a > 0.95:
                lines.append("Task A's spectral structure is well preserved with "
                            "minor perturbation from task B.")
            else:
                lines.append(f"Task A's spectral structure is significantly perturbed "
                            f"even at 10% mixing (cos = {cos_a:.4f}).")

        if sp["eig_cos_b_at_alpha_0.9"] is not None:
            cos_b = sp["eig_cos_b_at_alpha_0.9"]
            lines.append(f"**Eigenspectrum retention at alpha=0.9 (90% task B):** "
                        f"cos = {cos_b:.4f}")

        # Random baseline comparison
        rb = pr["random_baseline"]
        mid_result = next((s for s in pr["sweep"] if abs(s["alpha"] - 0.5) < 0.01), None)
        lines.append("")
        lines.append("### Random Baseline Comparison (alpha=0.5)")
        lines.append("")
        lines.append("| Metric | Task Merge | Random Merge | Better |")
        lines.append("|--------|-----------|--------------|--------|")
        if mid_result:
            task_fied = mid_result["mean_fiedler"]
            rand_fied = rb["mean_fiedler"]
            fied_better = "Task" if task_fied > rand_fied else "Random"
            lines.append(f"| Fiedler | {task_fied:.5f} | {rand_fied:.5f} | {fied_better} |")

            task_dev = mid_result["mean_deviation"]
            rand_dev = rb["mean_deviation"]
            dev_better = "Task" if task_dev < rand_dev else "Random"
            lines.append(f"| Deviation | {task_dev:.4f} | {rand_dev:.4f} | {dev_better} |")

            task_cos = mid_result["mean_eig_cos_a"]
            rand_cos = rb["mean_eig_cos"]
            cos_better = "Task" if task_cos > rand_cos else "Random"
            lines.append(f"| Eig cos(A) | {task_cos:.4f} | {rand_cos:.4f} | {cos_better} |")

        if mid_result and mid_result["mean_eig_cos_a"] > rb["mean_eig_cos"]:
            lines.append("")
            lines.append("**PASS** — Bridge merging preserves more task structure than "
                        "merging with random noise. The merged bridge retains meaningful "
                        "spectral properties from its sources.")
        elif mid_result:
            lines.append("")
            lines.append("**FAIL** — Bridge merging does NOT preserve more structure than "
                        "random noise. Task-specific bridge information is fragile under "
                        "interpolation.")

        # Per-module-type breakdown
        lines.append("")
        lines.append("### Per-Module-Type Sensitivity")
        lines.append("")
        lines.append("Sensitivity = Fiedler change from alpha=0.0 to alpha=0.5, normalized "
                     "by source Fiedler.")
        lines.append("")
        lines.append("| Module | Fiedler@0.0 | Fiedler@0.5 | Delta | Sensitivity |")
        lines.append("|--------|-------------|-------------|-------|-------------|")

        a0_result = next((s for s in pr["sweep"] if abs(s["alpha"] - 0.0) < 0.01), None)
        a5_result = next((s for s in pr["sweep"] if abs(s["alpha"] - 0.5) < 0.01), None)
        if a0_result and a5_result:
            module_types = ["q_proj", "k_proj", "v_proj", "o_proj"]
            sensitivities = []
            for mod in module_types:
                if mod in a0_result["per_module"] and mod in a5_result["per_module"]:
                    f0 = a0_result["per_module"][mod]["mean_fiedler"]
                    f5 = a5_result["per_module"][mod]["mean_fiedler"]
                    delta = abs(f5 - f0)
                    sens = delta / max(f0, 1e-8)
                    sensitivities.append((mod, f0, f5, delta, sens))
                    lines.append(f"| {mod} | {f0:.5f} | {f5:.5f} | {delta:.5f} | {sens:.3f} |")

            if sensitivities:
                most_sensitive = max(sensitivities, key=lambda x: x[4])
                least_sensitive = min(sensitivities, key=lambda x: x[4])
                lines.append("")
                lines.append(f"**Most affected by merging:** {most_sensitive[0]} "
                            f"(sensitivity = {most_sensitive[4]:.3f})")
                lines.append(f"**Least affected by merging:** {least_sensitive[0]} "
                            f"(sensitivity = {least_sensitive[4]:.3f})")

    # Global conclusions
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Conclusions")
    lines.append("")

    all_linear = all(pr["linearity"]["is_linear"] for pr in pair_results)
    if all_linear:
        lines.append("**Interpolation is smooth across all task pairs.** No phase transitions "
                     "detected. Bridge-level merging produces predictable, monotonic changes "
                     "in spectral structure. This supports the use of bridge interpolation "
                     "as a lightweight adapter merging strategy.")
    else:
        nonlinear = [pr["pair"] for pr in pair_results if not pr["linearity"]["is_linear"]]
        lines.append(f"**Phase transitions detected in {len(nonlinear)}/{len(pair_results)} "
                     f"pairs:** {', '.join(nonlinear)}. Bridge merging is NOT uniformly smooth — "
                     f"certain task combinations exhibit nonlinear structural reorganization "
                     f"at intermediate alpha values.")

    # Structure preservation summary
    all_preserved = True
    for pr in pair_results:
        sp = pr["structure_preservation"]
        if sp["eig_cos_a_at_alpha_0.1"] is not None and sp["eig_cos_a_at_alpha_0.1"] < 0.95:
            all_preserved = False
    if all_preserved:
        lines.append("")
        lines.append("**Task structure is preserved at low alpha values.** At alpha=0.1 "
                     "(10% contamination), the source task's eigenspectrum is retained "
                     "with cos > 0.95 in all pairs. Small amounts of cross-task information "
                     "can be injected without destroying the dominant task's structure.")
    else:
        lines.append("")
        lines.append("**Task structure is fragile.** Some pairs show significant spectral "
                     "degradation even at alpha=0.1. Bridge merging may require careful "
                     "calibration per task pair.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by `scripts/experiment_bridge_merge.py`*")

    report_path = output_dir / "MERGE_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2A: Bridge-Level Adapter Merging Experiment"
    )
    parser.add_argument(
        "--alpaca", type=str, default="results/exp2/rhombi_fcc_r24",
        help="Directory containing alpaca/general task bridge_final_*.npy files",
    )
    parser.add_argument(
        "--code", type=str, default="results/fingerprints/code",
        help="Directory containing code task bridge_final_*.npy files",
    )
    parser.add_argument(
        "--math", type=str, default="results/fingerprints/math",
        help="Directory containing math task bridge_final_*.npy files",
    )
    parser.add_argument(
        "--output", type=str, default="results/bridge_merge",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    alphas = [round(a * 0.1, 1) for a in range(11)]  # 0.0 to 1.0

    # Load task bridges
    task_bridges = {}
    for name, path_str in [("alpaca", args.alpaca), ("code", args.code), ("math", args.math)]:
        p = Path(path_str)
        if not p.exists():
            print(f"WARNING: {p} not found, skipping task '{name}'")
            continue
        bridges = load_final_bridges(p)
        if bridges:
            task_bridges[name] = bridges
            print(f"Loaded {len(bridges)} bridges for task '{name}'")
        else:
            print(f"WARNING: No bridge_final_*.npy files found in {p}")

    if len(task_bridges) < 2:
        print("ERROR: Need at least 2 task types for merge experiment.")
        print("Phase 1A must complete first (training code + math tasks).")
        sys.exit(1)

    # Verify all tasks have compatible bridge shapes
    task_names = list(task_bridges.keys())
    first_task = task_names[0]
    first_keys = set(task_bridges[first_task].keys())
    for t in task_names[1:]:
        common = first_keys & set(task_bridges[t].keys())
        if not common:
            print(f"ERROR: No common adapter keys between {first_task} and {t}")
            sys.exit(1)
        print(f"  {first_task} vs {t}: {len(common)} common adapters")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis for each task pair
    print(f"\nRunning merge sweep with alpha in {alphas}...")
    pair_results = []

    for task_a, task_b in combinations(task_names, 2):
        print(f"\n  Analyzing {task_a} vs {task_b}...")
        result = analyze_pair(
            task_a, task_bridges[task_a],
            task_b, task_bridges[task_b],
            alphas, rng,
        )
        pair_results.append(result)
        print(f"    Linearity R^2: {result['linearity']['r_squared']:.5f}")
        sp = result["structure_preservation"]
        if sp["eig_cos_a_at_alpha_0.1"] is not None:
            print(f"    Structure preservation (cos@0.1): {sp['eig_cos_a_at_alpha_0.1']:.4f}")

    # Write outputs
    print(f"\nWriting results to {output_dir}/...")

    # JSON results
    json_path = output_dir / "merge_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pair_results, f, indent=2)
    print(f"  JSON results: {json_path}")

    # Markdown report
    write_report(pair_results, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
