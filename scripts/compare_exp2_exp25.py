"""Compare Exp 2 (Alpaca) vs Exp 2.5 (Geometric) bridge metrics.

Reads results.json and bridge .npy files from both experiment directories.
Produces comparison tables plus Karkada-informed spectral analysis:
  - Standard metrics (Fiedler, deviation, co/cross ratio)
  - Bridge eigendecomposition vs O_h irrep structure
  - Gram matrix alignment with theoretical RD coupling
  - Spectral distance tracking
  - Permutation test for co/cross statistical significance
  - Perturbation test (zero co-planar, check reconstruction)

Usage:
  python scripts/compare_exp2_exp25.py
  python scripts/compare_exp2_exp25.py --exp2 results/exp2 --exp25 results/exp2_5
  python scripts/compare_exp2_exp25.py --deep  # Include spectral analysis on bridge .npy files
"""

from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path

import numpy as np


# ── Theoretical Reference ─────────────────────────────────────────────

def theoretical_coupling_matrix() -> np.ndarray:
    """RD direction-pair coupling matrix from topology.py.

    Diagonal = 4, co-planar off-diag = 4, cross-planar off-diag = 2.
    """
    try:
        from rhombic.nn.topology import direction_pair_coupling
        return direction_pair_coupling()
    except ImportError:
        # Hardcoded fallback (verified from topology.py)
        C = np.full((6, 6), 2.0)
        np.fill_diagonal(C, 4.0)
        # Co-planar pairs: (0,1), (2,3), (4,5) share 4 octahedral vertices
        for i, j in [(0, 1), (2, 3), (4, 5)]:
            C[i, j] = C[j, i] = 4.0
        return C


CO_PLANAR_PAIRS = [(0, 1), (2, 3), (4, 5)]
CROSS_PLANAR_PAIRS = [(i, j) for i in range(6) for j in range(i+1, 6)
                       if (i, j) not in CO_PLANAR_PAIRS]


# ── Basic Comparison (from results.json) ──────────────────────────────

def load_results(exp_dir: Path, config_name: str) -> dict | None:
    path = exp_dir / config_name / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def final_checkpoint(results: dict) -> dict:
    return results["checkpoints"][-1]


def format_comparison(exp2_results: dict, exp25_results: dict,
                       config_name: str, label2: str, label25: str) -> str:
    cp2 = final_checkpoint(exp2_results)
    cp25 = final_checkpoint(exp25_results)

    lines = [
        f"\n### {config_name}",
        f"| Metric | {label2} | {label25} | Δ |",
        f"|--------|---:|---:|---:|",
    ]

    vl2 = cp2.get("val_loss", 0)
    vl25 = cp25.get("val_loss", 0)
    lines.append(f"| Val Loss | {vl2:.4f} | {vl25:.4f} | {vl25 - vl2:+.4f} |")

    tl2 = cp2.get("train_loss", 0)
    tl25 = cp25.get("train_loss", 0)
    lines.append(f"| Train Loss | {tl2:.4f} | {tl25:.4f} | {tl25 - tl2:+.4f} |")

    f2 = cp2.get("bridge_fiedler_mean", 0)
    f25 = cp25.get("bridge_fiedler_mean", 0)
    ratio = f25 / f2 if f2 > 1e-12 else float('inf')
    lines.append(f"| Fiedler | {f2:.5f} | {f25:.5f} | {ratio:.2f}× |")

    d2 = cp2.get("bridge_deviation_mean", 0)
    d25 = cp25.get("bridge_deviation_mean", 0)
    dratio = d25 / d2 if d2 > 1e-12 else float('inf')
    lines.append(f"| Deviation | {d2:.5f} | {d25:.5f} | {dratio:.2f}× |")

    ccp2 = cp2.get("coplanar_crossplanar")
    ccp25 = cp25.get("coplanar_crossplanar")
    if ccp2 and ccp25:
        r2 = ccp2["ratio"]
        r25 = ccp25["ratio"]
        lines.append(f"| **Co/Cross Ratio** | **{r2:.4f}** | **{r25:.4f}** | **{r25 - r2:+.4f}** |")
        lines.append(f"| Co-planar mean | {ccp2['mean_coplanar']:.5f} | {ccp25['mean_coplanar']:.5f} | |")
        lines.append(f"| Cross-planar mean | {ccp2['mean_crossplanar']:.5f} | {ccp25['mean_crossplanar']:.5f} | |")

    return "\n".join(lines)


def learning_curve_comparison(exp2_results: dict, exp25_results: dict) -> str:
    lines = ["\n### Co/Cross Learning Curve"]
    lines.append("| Step | Alpaca | Geometric | Δ |")
    lines.append("|------|--------|----------|---|")

    steps2 = {cp["step"]: cp for cp in exp2_results["checkpoints"]}
    steps25 = {cp["step"]: cp for cp in exp25_results["checkpoints"]}

    for step in sorted(set(steps2.keys()) | set(steps25.keys())):
        r2 = r25 = delta = "—"
        cp2 = steps2.get(step)
        cp25 = steps25.get(step)

        if cp2 and cp2.get("coplanar_crossplanar"):
            r2 = f"{cp2['coplanar_crossplanar']['ratio']:.4f}"
        if cp25 and cp25.get("coplanar_crossplanar"):
            r25 = f"{cp25['coplanar_crossplanar']['ratio']:.4f}"
        if cp2 and cp25 and cp2.get("coplanar_crossplanar") and cp25.get("coplanar_crossplanar"):
            d = cp25["coplanar_crossplanar"]["ratio"] - cp2["coplanar_crossplanar"]["ratio"]
            delta = f"{d:+.4f}"

        lines.append(f"| {step} | {r2} | {r25} | {delta} |")

    return "\n".join(lines)


# ── Karkada-Informed Spectral Analysis ────────────────────────────────

def load_final_bridges(exp_dir: Path, config_name: str) -> list[np.ndarray]:
    """Load all bridge_final_*.npy files for a config."""
    pattern = str(exp_dir / config_name / "bridge_final_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        # Try bridge_stepN for the last checkpoint
        all_bridge = sorted(glob.glob(str(exp_dir / config_name / "bridge_step*.npy")))
        if all_bridge:
            # Get highest step number
            steps = set()
            for f in all_bridge:
                parts = Path(f).stem.split("_")
                for p in parts:
                    if p.startswith("step"):
                        try:
                            steps.add(int(p[4:]))
                        except ValueError:
                            pass
            if steps:
                max_step = max(steps)
                files = [f for f in all_bridge if f"step{max_step}_" in f]
    return [np.load(f) for f in files if np.load(f).shape == (6, 6)]


def gram_alignment(bridges: list[np.ndarray]) -> dict:
    """Measure alignment between learned bridge Gram matrices and theoretical coupling.

    Frobenius distance between normalized learned Gram (B @ B.T) and
    normalized theoretical coupling matrix. Lower = more aligned.
    """
    C_theo = theoretical_coupling_matrix()
    C_norm = C_theo / np.linalg.norm(C_theo, 'fro')

    distances = []
    for B in bridges:
        G = B @ B.T
        G_norm = G / max(np.linalg.norm(G, 'fro'), 1e-12)
        dist = np.linalg.norm(G_norm - C_norm, 'fro')
        distances.append(dist)

    return {
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "n_bridges": len(bridges),
    }


def eigenvalue_analysis(bridges: list[np.ndarray]) -> dict:
    """Analyze eigenvalue structure of bridge matrices.

    The RD coupling matrix has eigenvalues with specific multiplicity
    pattern from O_h symmetry. Check if learned bridges converge toward
    this pattern.
    """
    C_theo = theoretical_coupling_matrix()
    theo_eigvals = np.sort(np.linalg.eigvalsh(C_theo))

    all_eigvals = []
    spectral_distances = []
    for B in bridges:
        eigvals = np.sort(np.linalg.eigvalsh(B))
        all_eigvals.append(eigvals)

        # Spectral distance: L2 norm of sorted eigenvalue differences
        # Normalize both to unit norm for comparison
        e_norm = eigvals / max(np.linalg.norm(eigvals), 1e-12)
        t_norm = theo_eigvals / max(np.linalg.norm(theo_eigvals), 1e-12)
        spectral_distances.append(float(np.linalg.norm(e_norm - t_norm)))

    all_eigvals = np.array(all_eigvals)

    # Eigenvalue multiplicity pattern: cluster eigenvalues and count
    mean_eigvals = all_eigvals.mean(axis=0)

    return {
        "mean_eigenvalues": [float(v) for v in mean_eigvals],
        "theoretical_eigenvalues": [float(v) for v in theo_eigvals],
        "spectral_distance_mean": float(np.mean(spectral_distances)),
        "spectral_distance_std": float(np.std(spectral_distances)),
        "eigval_spread": float(mean_eigvals.max() - mean_eigvals.min()),
    }


def permutation_test(bridges: list[np.ndarray], n_perms: int = 10000,
                      seed: int = 42) -> dict:
    """Permutation test for co/cross ratio significance.

    Shuffle channel labels, recompute co/cross ratio, build null distribution.
    Returns p-value: fraction of permutations with ratio >= observed.
    """
    rng = np.random.RandomState(seed)

    # Compute observed ratio across all bridges
    obs_co = []
    obs_cross = []
    for B in bridges:
        for i, j in CO_PLANAR_PAIRS:
            obs_co.append(abs(B[i, j]))
        for i, j in CROSS_PLANAR_PAIRS:
            obs_cross.append(abs(B[i, j]))

    observed_ratio = np.mean(obs_co) / max(np.mean(obs_cross), 1e-12)

    # Null distribution: permute channel indices
    null_ratios = []
    for _ in range(n_perms):
        perm = rng.permutation(6)
        perm_co = []
        perm_cross = []
        for B in bridges:
            B_perm = B[np.ix_(perm, perm)]
            for i, j in CO_PLANAR_PAIRS:
                perm_co.append(abs(B_perm[i, j]))
            for i, j in CROSS_PLANAR_PAIRS:
                perm_cross.append(abs(B_perm[i, j]))
        null_ratios.append(np.mean(perm_co) / max(np.mean(perm_cross), 1e-12))

    null_ratios = np.array(null_ratios)
    p_value = float(np.mean(null_ratios >= observed_ratio))

    return {
        "observed_ratio": float(observed_ratio),
        "null_mean": float(null_ratios.mean()),
        "null_std": float(null_ratios.std()),
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "n_permutations": n_perms,
    }


def _bridge_fiedler(B: np.ndarray) -> float:
    """Fiedler value of a 6×6 bridge treated as weighted adjacency matrix."""
    from rhombic.spectral import fiedler_value
    n = B.shape[0]
    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(B[i, j]) > 1e-12:
                edges.append((i, j))
                weights.append(abs(B[i, j]))
    if len(edges) < n - 1:
        return 0.0  # disconnected
    return fiedler_value(n, edges, weights)


def perturbation_test(bridges: list[np.ndarray]) -> dict:
    """Perturbation test: zero co-planar entries, measure structure retention.

    Karkada's zeroing experiment adapted for the bridge. If the geometric
    structure is distributed (strong result), the Fiedler value of the
    perturbed bridge should retain a significant fraction of the original.
    """
    original_fiedlers = []
    perturbed_fiedlers = []

    for B in bridges:
        original_fiedlers.append(_bridge_fiedler(B))

        B_perturbed = B.copy()
        for i, j in CO_PLANAR_PAIRS:
            B_perturbed[i, j] = 0.0
            B_perturbed[j, i] = 0.0
        perturbed_fiedlers.append(_bridge_fiedler(B_perturbed))

    orig_mean = float(np.mean(original_fiedlers))
    pert_mean = float(np.mean(perturbed_fiedlers))

    return {
        "original_fiedler_mean": orig_mean,
        "perturbed_fiedler_mean": pert_mean,
        "retention_ratio": pert_mean / max(orig_mean, 1e-12),
        "interpretation": (
            "DISTRIBUTED — structure survives co-planar zeroing"
            if pert_mean / max(orig_mean, 1e-12) > 0.5
            else "LOCAL — structure concentrated in co-planar pairs"
        ),
    }


def format_spectral_analysis(
    label: str,
    bridges: list[np.ndarray],
) -> str:
    """Full Karkada-informed spectral analysis of bridge matrices."""
    if not bridges:
        return f"\n### Spectral Analysis: {label}\n*No bridge files found.*"

    lines = [f"\n### Spectral Analysis: {label}"]
    lines.append(f"*{len(bridges)} bridge matrices analyzed*\n")

    # Gram alignment
    gram = gram_alignment(bridges)
    lines.append("**Gram Matrix Alignment** (lower = more aligned with RD coupling)")
    lines.append(f"- Distance: {gram['mean_distance']:.4f} ± {gram['std_distance']:.4f}")
    lines.append(f"- Range: [{gram['min_distance']:.4f}, {gram['max_distance']:.4f}]")
    lines.append("")

    # Eigenvalue analysis
    eig = eigenvalue_analysis(bridges)
    lines.append("**Eigenvalue Structure**")
    lines.append(f"- Learned:      {[f'{v:.4f}' for v in eig['mean_eigenvalues']]}")
    lines.append(f"- Theoretical:  {[f'{v:.4f}' for v in eig['theoretical_eigenvalues']]}")
    lines.append(f"- Spectral distance: {eig['spectral_distance_mean']:.4f} ± {eig['spectral_distance_std']:.4f}")
    lines.append(f"- Eigenvalue spread: {eig['eigval_spread']:.5f}")
    lines.append("")

    # Permutation test
    perm = permutation_test(bridges)
    sig_marker = "***" if perm["significant_001"] else ("**" if perm["significant_005"] else "ns")
    lines.append("**Permutation Test** (co/cross under shuffled channel labels)")
    lines.append(f"- Observed ratio: {perm['observed_ratio']:.4f}")
    lines.append(f"- Null: {perm['null_mean']:.4f} ± {perm['null_std']:.4f}")
    lines.append(f"- p = {perm['p_value']:.4f} {sig_marker}")
    lines.append("")

    # Perturbation test
    try:
        pert = perturbation_test(bridges)
        lines.append("**Perturbation Test** (zero co-planar entries)")
        lines.append(f"- Original Fiedler: {pert['original_fiedler_mean']:.5f}")
        lines.append(f"- Perturbed Fiedler: {pert['perturbed_fiedler_mean']:.5f}")
        lines.append(f"- Retention: {pert['retention_ratio']:.2%}")
        lines.append(f"- {pert['interpretation']}")
    except Exception as e:
        lines.append(f"**Perturbation Test** — skipped ({e})")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Exp 2 vs Exp 2.5")
    parser.add_argument("--exp2", type=str, default="results/exp2")
    parser.add_argument("--exp25", type=str, default="results/exp2_5")
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown file. Default: print to stdout.")
    parser.add_argument("--deep", action="store_true",
                        help="Include Karkada-informed spectral analysis on bridge .npy files.")
    args = parser.parse_args()

    exp2_dir = Path(args.exp2)
    exp25_dir = Path(args.exp25)

    report = ["# Exp 2 vs Exp 2.5: Dataset Effect on Bridge Behavior\n"]
    report.append("> Exp 2: Alpaca-cleaned (isotropic)")
    report.append("> Exp 2.5: Geometric dataset (directionally structured)")
    report.append("> Key question: Does purpose-built data strengthen co-planar signal?\n")

    if args.deep:
        report.append("> **Deep analysis enabled** — includes Karkada-informed spectral diagnostics:")
        report.append("> Gram alignment, eigenvalue structure, permutation test (p-value),")
        report.append("> perturbation test (distributed vs local structure).\n")

    configs = [
        ("rhombi_fcc_r24", "FCC RhombiLoRA (6-ch)"),
        ("rhombi_cubic_r24", "Cubic RhombiLoRA (3-ch)"),
        ("standard_lora_r24", "Standard LoRA (frozen bridge)"),
    ]

    exp25_prefixes = ["geometric_", "mixed50_", "alpaca_", ""]

    for config_name, display_name in configs:
        exp2_res = load_results(exp2_dir, config_name)
        if exp2_res is None:
            report.append(f"\n### {display_name}\n*Exp 2 results not found.*")
            continue

        found = False
        for prefix in exp25_prefixes:
            exp25_name = f"{prefix}{config_name}" if prefix else config_name
            exp25_res = load_results(exp25_dir, exp25_name)
            if exp25_res:
                label = prefix.rstrip("_") if prefix else "geometric"
                report.append(format_comparison(
                    exp2_res, exp25_res, display_name,
                    "Alpaca", label.title()
                ))

                if config_name == "rhombi_fcc_r24":
                    report.append(learning_curve_comparison(exp2_res, exp25_res))

                # Deep spectral analysis
                if args.deep and config_name == "rhombi_fcc_r24":
                    report.append("\n---\n## Spectral Analysis (Karkada Framework)\n")

                    bridges_exp2 = load_final_bridges(exp2_dir, config_name)
                    bridges_exp25 = load_final_bridges(exp25_dir, exp25_name)

                    report.append(format_spectral_analysis(
                        f"Exp 2 — Alpaca ({config_name})", bridges_exp2))
                    report.append(format_spectral_analysis(
                        f"Exp 2.5 — {label.title()} ({exp25_name})", bridges_exp25))

                    # Comparison summary
                    if bridges_exp2 and bridges_exp25:
                        gram2 = gram_alignment(bridges_exp2)
                        gram25 = gram_alignment(bridges_exp25)
                        perm2 = permutation_test(bridges_exp2)
                        perm25 = permutation_test(bridges_exp25)

                        report.append("\n### Head-to-Head Spectral Comparison")
                        report.append("| Metric | Alpaca | Geometric | Better |")
                        report.append("|--------|--------|----------|--------|")
                        better_gram = "Geometric" if gram25["mean_distance"] < gram2["mean_distance"] else "Alpaca"
                        report.append(f"| Gram alignment (↓ better) | {gram2['mean_distance']:.4f} | {gram25['mean_distance']:.4f} | {better_gram} |")
                        better_p = "Geometric" if perm25["p_value"] < perm2["p_value"] else "Alpaca"
                        report.append(f"| Permutation p-value (↓ better) | {perm2['p_value']:.4f} | {perm25['p_value']:.4f} | {better_p} |")
                        report.append(f"| Co/Cross observed | {perm2['observed_ratio']:.4f} | {perm25['observed_ratio']:.4f} | |")

                found = True

        if not found:
            report.append(f"\n### {display_name}\n*Exp 2.5 results not found.*")

    # Verdict
    report.append("\n---\n## Verdict\n")
    report.append("**Decision criteria for proceeding to Exp 3A-3C:**")
    report.append("1. Co/Cross ratio > 1.10 (up from Exp 2's 1.019)")
    report.append("2. Permutation p-value < 0.05 (statistically significant directional preference)")
    report.append("3. Gram alignment closer to theoretical coupling under geometric data")
    report.append("4. Perturbation test shows distributed structure (retention > 50%)")
    report.append("")
    report.append("Meeting criteria 1+2 is sufficient to proceed. 3+4 strengthen the case.\n")

    text = "\n".join(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Report written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
