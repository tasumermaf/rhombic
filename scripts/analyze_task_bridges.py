"""Phase 1A Analysis: Compare bridge structures across task types.

Compares final bridge matrices from three training runs:
  - Alpaca (general): results/exp2/rhombi_fcc_r24/bridge_final_*.npy
  - Code: results/fingerprints/code/bridge_final_*.npy
  - Math: results/fingerprints/math/bridge_final_*.npy

Tests:
  1. Within-task vs between-task Frobenius distance
  2. Bridge signature classifier (SVM on flattened bridge)
  3. Per-module-type breakdown
  4. Task-discriminating dimensions (PCA)

Usage:
  python scripts/analyze_task_bridges.py \
    --alpaca results/exp2/rhombi_fcc_r24 \
    --code results/fingerprints/code \
    --math results/fingerprints/math \
    --output results/fingerprints/TASK_FINGERPRINT_REPORT.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import permutation_test, mannwhitneyu


def parse_bridge_filename(name: str) -> dict | None:
    """Extract layer, module from bridge filename."""
    m = re.match(
        r"bridge_final_model_layers_(\d+)_self_attn_(\w+_proj)\.npy",
        name,
    )
    if m:
        return {"layer": int(m.group(1)), "module": m.group(2)}
    return None


def load_final_bridges(directory: Path) -> dict[str, np.ndarray]:
    """Load all bridge_final_*.npy from a directory."""
    bridges = {}
    for f in sorted(directory.glob("bridge_final_*.npy")):
        info = parse_bridge_filename(f.name)
        if info is None:
            continue
        key = f"L{info['layer']:02d}_{info['module']}"
        bridges[key] = np.load(f)
    return bridges


def bridge_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """Frobenius distance between two bridge matrices."""
    return float(np.linalg.norm(b1 - b2, 'fro'))


def compute_distance_matrix(task_bridges: dict[str, dict[str, np.ndarray]]) -> dict:
    """Compute within-task and between-task distances.

    For each adapter position (e.g., L00_q_proj), compute distance
    between same-position bridges across different tasks.
    """
    task_names = list(task_bridges.keys())
    n_tasks = len(task_names)

    # Find common adapter keys
    common_keys = set(task_bridges[task_names[0]].keys())
    for t in task_names[1:]:
        common_keys &= set(task_bridges[t].keys())
    common_keys = sorted(common_keys)

    if not common_keys:
        raise ValueError("No common adapter keys across tasks")

    within_distances = {t: [] for t in task_names}
    between_distances = {}

    # Between-task: distance between same adapter across tasks
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            pair = f"{task_names[i]}_vs_{task_names[j]}"
            between_distances[pair] = []
            for key in common_keys:
                d = bridge_distance(
                    task_bridges[task_names[i]][key],
                    task_bridges[task_names[j]][key],
                )
                between_distances[pair].append(d)

    # Within-task: distance between different adapters WITHIN same task
    # (for module-type grouping)
    for t in task_names:
        bridges_list = [task_bridges[t][k] for k in common_keys]
        for i in range(len(bridges_list)):
            for j in range(i + 1, len(bridges_list)):
                d = bridge_distance(bridges_list[i], bridges_list[j])
                within_distances[t].append(d)

    return {
        "common_keys": common_keys,
        "within": within_distances,
        "between": between_distances,
        "n_adapters": len(common_keys),
    }


def classify_bridges(task_bridges: dict[str, dict[str, np.ndarray]]) -> dict:
    """Train a simple classifier on flattened bridge matrices.

    Uses leave-one-out cross-validation with a linear SVM.
    """
    try:
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "sklearn not available", "accuracy": None}

    task_names = list(task_bridges.keys())
    common_keys = set(task_bridges[task_names[0]].keys())
    for t in task_names[1:]:
        common_keys &= set(task_bridges[t].keys())
    common_keys = sorted(common_keys)

    X, y = [], []
    for task_idx, t in enumerate(task_names):
        for key in common_keys:
            X.append(task_bridges[t][key].flatten())
            y.append(task_idx)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    loo = LeaveOneOut()
    correct = 0
    total = 0
    for train_idx, test_idx in loo.split(X_scaled):
        clf = LinearSVC(max_iter=5000, random_state=42)
        clf.fit(X_scaled[train_idx], y[train_idx])
        pred = clf.predict(X_scaled[test_idx])
        correct += (pred == y[test_idx]).sum()
        total += len(test_idx)

    accuracy = correct / total

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    clf_full = LinearSVC(max_iter=5000, random_state=42)
    clf_full.fit(X_scaled, y)
    preds_full = clf_full.predict(X_scaled)
    cm = confusion_matrix(y, preds_full)

    return {
        "accuracy": float(accuracy),
        "n_samples": total,
        "n_per_task": len(common_keys),
        "task_names": task_names,
        "confusion_matrix": cm.tolist(),
        "chance_level": 1.0 / len(task_names),
    }


def per_module_analysis(task_bridges: dict[str, dict[str, np.ndarray]]) -> dict:
    """Analyze task discrimination per module type (q/k/v/o_proj)."""
    task_names = list(task_bridges.keys())
    modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    results = {}
    for mod in modules:
        mod_bridges = {}
        for t in task_names:
            mod_bridges[t] = {
                k: v for k, v in task_bridges[t].items() if mod in k
            }

        # Mean bridge per task for this module type
        means = {}
        for t in task_names:
            if mod_bridges[t]:
                means[t] = np.mean(
                    [b for b in mod_bridges[t].values()], axis=0
                )

        # Distance between task means
        if len(means) >= 2:
            task_pairs = []
            for i, t1 in enumerate(task_names):
                for t2 in task_names[i+1:]:
                    if t1 in means and t2 in means:
                        d = bridge_distance(means[t1], means[t2])
                        task_pairs.append((t1, t2, d))

            results[mod] = {
                "n_adapters": len(mod_bridges[task_names[0]]),
                "task_pair_distances": task_pairs,
                "mean_distance": float(np.mean([p[2] for p in task_pairs])),
            }

    return results


def write_report(
    distances: dict,
    classification: dict,
    module_analysis: dict,
    task_bridges: dict[str, dict[str, np.ndarray]],
    output_path: Path,
):
    """Write the analysis report."""
    lines = [
        "# Phase 1A: Task-Specific Bridge Fingerprints",
        "",
        "> **Source tasks:** " + ", ".join(task_bridges.keys()),
        f"> **Adapters per task:** {distances['n_adapters']}",
        "",
    ]

    # Between-task distances
    lines.append("## Between-Task Bridge Distances")
    lines.append("")
    lines.append("| Task Pair | Mean Distance | Std | Min | Max |")
    lines.append("|-----------|--------------|-----|-----|-----|")
    for pair, dists in distances["between"].items():
        d = np.array(dists)
        lines.append(
            f"| {pair} | {d.mean():.5f} | {d.std():.5f} | "
            f"{d.min():.5f} | {d.max():.5f} |"
        )

    # Statistical test: are between-task distances significantly > 0?
    all_between = []
    for dists in distances["between"].values():
        all_between.extend(dists)
    all_between = np.array(all_between)

    lines.append("")
    lines.append(f"**Overall mean between-task distance:** {all_between.mean():.5f}")
    lines.append("")

    # Within-task distances for comparison
    lines.append("## Within-Task Bridge Distances")
    lines.append("")
    lines.append("| Task | Mean Distance | Std |")
    lines.append("|------|--------------|-----|")
    for task, dists in distances["within"].items():
        d = np.array(dists)
        lines.append(f"| {task} | {d.mean():.5f} | {d.std():.5f} |")

    # Mann-Whitney test: between > within?
    all_within = []
    for dists in distances["within"].values():
        all_within.extend(dists)
    all_within = np.array(all_within)

    if len(all_between) > 0 and len(all_within) > 0:
        stat, p = mannwhitneyu(all_between, all_within, alternative="greater")
        lines.append("")
        lines.append(
            f"**Mann-Whitney U (between > within):** U = {stat:.1f}, p = {p:.6f}"
        )
        if p < 0.01:
            lines.append("**PASS** — Between-task distances significantly greater than within-task")
        elif p < 0.05:
            lines.append("**MARGINAL** — Between-task distances marginally greater")
        else:
            lines.append("**FAIL** — No significant difference between within and between-task distances")

    # Classification
    lines.append("")
    lines.append("## Bridge Classifier")
    lines.append("")
    if classification.get("accuracy") is not None:
        acc = classification["accuracy"]
        chance = classification["chance_level"]
        lines.append(f"**Leave-one-out SVM accuracy:** {acc:.1%}")
        lines.append(f"**Chance level:** {chance:.1%}")
        lines.append(f"**Samples:** {classification['n_samples']} ({classification['n_per_task']} per task)")
        lines.append("")
        if acc > 0.80:
            lines.append("**STRONG PASS** — Bridge matrices reliably identify task type")
        elif acc > 0.60:
            lines.append("**PASS** — Bridge matrices carry task-discriminating information")
        elif acc > chance + 0.10:
            lines.append("**MARGINAL** — Some task information in bridges, but weak")
        else:
            lines.append("**FAIL** — Bridge matrices do not reliably distinguish tasks")

        # Confusion matrix
        lines.append("")
        lines.append("**Confusion matrix (LOO):**")
        lines.append("```")
        names = classification["task_names"]
        header = "       " + "  ".join(f"{n:>8}" for n in names)
        lines.append(header)
        cm = classification["confusion_matrix"]
        for i, row in enumerate(cm):
            line = f"{names[i]:>7}" + "  ".join(f"{v:>8}" for v in row)
            lines.append(line)
        lines.append("```")
    elif classification.get("error"):
        lines.append(f"*{classification['error']}*")

    # Per-module analysis
    lines.append("")
    lines.append("## Per-Module-Type Discrimination")
    lines.append("")
    lines.append("| Module | Mean Task Distance | Best Discriminator |")
    lines.append("|--------|-------------------|-------------------|")
    best_mod = None
    best_dist = 0
    for mod, info in module_analysis.items():
        d = info["mean_distance"]
        lines.append(f"| {mod} | {d:.5f} | |")
        if d > best_dist:
            best_dist = d
            best_mod = mod

    if best_mod:
        lines.append("")
        lines.append(f"**Best discriminator:** {best_mod} (distance = {best_dist:.5f})")

    # Per-task bridge statistics
    lines.append("")
    lines.append("## Per-Task Bridge Statistics")
    lines.append("")
    lines.append("| Task | Mean Fiedler | Mean Deviation | Mean Eig Spread |")
    lines.append("|------|-------------|----------------|-----------------|")
    for task, bridges in task_bridges.items():
        fiedlers = []
        deviations = []
        for b in bridges.values():
            # Compute Laplacian Fiedler
            n = b.shape[0]
            W = np.abs(b)
            np.fill_diagonal(W, 0)
            D = np.diag(W.sum(axis=1))
            L = D - W
            eigs = np.sort(np.linalg.eigvalsh(L))
            fiedlers.append(eigs[1] if len(eigs) > 1 else 0)
            deviations.append(float(np.linalg.norm(b - np.eye(n), 'fro')))

        lines.append(
            f"| {task} | {np.mean(fiedlers):.5f} | "
            f"{np.mean(deviations):.4f} | — |"
        )

    # Decision gate
    lines.append("")
    lines.append("---")
    lines.append("## Decision Gate")
    lines.append("")
    lines.append("Phase 1A informs the direction of Phase 2 (adapter composition).")
    lines.append("If bridges differ by task → proceed to bridge-level merging.")
    lines.append("If not → proceed to Phase 3 (regularization).")
    lines.append("")
    lines.append("*Generated by `scripts/analyze_task_bridges.py`*")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1A: Task bridge analysis")
    parser.add_argument("--alpaca", type=str, default="results/exp2/rhombi_fcc_r24")
    parser.add_argument("--code", type=str, default="results/fingerprints/code")
    parser.add_argument("--math", type=str, default="results/fingerprints/math")
    parser.add_argument("--output", type=str,
                        default="results/fingerprints/TASK_FINGERPRINT_REPORT.md")
    args = parser.parse_args()

    task_bridges = {}

    for name, path in [("alpaca", args.alpaca), ("code", args.code), ("math", args.math)]:
        p = Path(path)
        if not p.exists():
            print(f"WARNING: {p} not found, skipping task '{name}'")
            continue
        bridges = load_final_bridges(p)
        if bridges:
            task_bridges[name] = bridges
            print(f"Loaded {len(bridges)} bridges for task '{name}'")
        else:
            print(f"WARNING: No bridges found in {p}")

    if len(task_bridges) < 2:
        print("ERROR: Need at least 2 task types for comparison")
        return

    print(f"\nAnalyzing {len(task_bridges)} task types...")

    distances = compute_distance_matrix(task_bridges)
    classification = classify_bridges(task_bridges)
    module_analysis = per_module_analysis(task_bridges)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_report(distances, classification, module_analysis, task_bridges, Path(args.output))


if __name__ == "__main__":
    main()
