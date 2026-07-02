"""
AR-001: Asymmetry Analysis of Bridge Matrices (v2 - corrected co-pairs)

The BD structure uses CONSECUTIVE channel pairs: (0,1), (2,3), ...
NOT the antipodal geometric pairs. This script corrects the co-pair
definitions and re-runs the full analysis.
"""

import numpy as np
import glob
import os
import json
from collections import defaultdict

# ============================================================
# Experiment definitions (corrected co-pairs)
# ============================================================

experiments = {
    "H-ch3": {
        "dir": "C:/falco/rhombic/results/channel-ablation/H-ch3",
        "prefix": "bridge_final",
        "n": 3,
        "geometry": "None (spectral-only)",
        "model": "TinyLlama-1.1B",
        "co_pairs": None,  # n=3: odd, no clean BD pairing
        "bd_expected": False,
    },
    "H-ch4": {
        "dir": "C:/falco/rhombic/results/channel-ablation/H-ch4",
        "prefix": "bridge_step600",
        "n": 4,
        "geometry": "Octahedron",
        "model": "TinyLlama-1.1B",
        "co_pairs": [(0,1),(2,3)],  # consecutive 2x2 blocks
        "bd_expected": True,
    },
    "Seed-43": {
        "dir": "C:/falco/rhombic/results/Seed-43",
        "prefix": "bridge_final",
        "n": 6,
        "geometry": "RD (rhombic dodecahedron)",
        "model": "TinyLlama-1.1B",
        "co_pairs": [(0,1),(2,3),(4,5)],  # consecutive 2x2 blocks
        "bd_expected": True,
    },
    "Seed-44": {
        "dir": "C:/falco/rhombic/results/Seed-44",
        "prefix": "bridge_final",
        "n": 6,
        "geometry": "RD (rhombic dodecahedron)",
        "model": "TinyLlama-1.1B",
        "co_pairs": [(0,1),(2,3),(4,5)],
        "bd_expected": True,
    },
    "exp3-Qwen": {
        "dir": "C:/falco/rhombic/results/exp3",
        "prefix": "bridge_final",
        "n": 6,
        "geometry": "RD (rhombic dodecahedron)",
        "model": "Qwen2.5-7B",
        "co_pairs": [(0,1),(2,3),(4,5)],
        "bd_expected": True,
    },
    "T-001-full": {
        "dir": "C:/falco/rhombic/results/T-001-full",
        "prefix": "bridge_final",
        "n": 8,
        "geometry": "Tesseract",
        "model": "TinyLlama-1.1B",
        "co_pairs": [(0,1),(2,3),(4,5),(6,7)],  # consecutive 2x2 blocks
        "bd_expected": True,
    },
    "T-001r2": {
        "dir": "C:/falco/rhombic/results/T-001-full-r2",
        "prefix": "bridge_final",
        "n": 8,
        "geometry": "Tesseract",
        "model": "TinyLlama-1.1B",
        "co_pairs": [(0,1),(2,3),(4,5),(6,7)],
        "bd_expected": True,
    },
}


def load_bridges(exp):
    """Load all bridge files for an experiment."""
    bridges = {}
    pattern = os.path.join(exp["dir"], exp["prefix"] + "_model_layers_*_self_attn_*_proj.npy")
    files = glob.glob(pattern)
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace(".npy", "").split("_")
        try:
            li = parts.index("layers")
            layer = int(parts[li + 1])
            pi = parts.index("attn")
            proj = parts[pi + 1]
        except (ValueError, IndexError):
            continue
        B = np.load(f)
        bridges[(layer, proj)] = B
    return bridges


def compute_co_cross_ratio(B, co_pairs, n):
    """Compute co/cross ratio from upper triangle of B."""
    if co_pairs is None or n < 4:
        return None, None, None

    co_set = set()
    for i, j in co_pairs:
        co_set.add((min(i, j), max(i, j)))

    co_vals = []
    cross_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            val = abs(B[i, j])
            if (i, j) in co_set:
                co_vals.append(val)
            else:
                cross_vals.append(val)

    mean_co = np.mean(co_vals) if co_vals else 0
    mean_cross = np.mean(cross_vals) if cross_vals else 1e-10
    ratio = mean_co / max(mean_cross, 1e-10)
    return ratio, mean_co, mean_cross


def analyze_bridge(B, co_pairs, n):
    """Full asymmetry analysis of a single bridge matrix."""
    S = (B + B.T) / 2  # symmetric part
    A = (B - B.T) / 2  # antisymmetric part

    norm_B = np.linalg.norm(B, "fro")
    norm_S = np.linalg.norm(S, "fro")
    norm_A = np.linalg.norm(A, "fro")

    asym_ratio = norm_A / max(norm_B, 1e-10)
    sym_ratio = norm_S / max(norm_B, 1e-10)

    # Pythagorean check: ||B||^2 = ||S||^2 + ||A||^2
    pyth_check = norm_B**2 - (norm_S**2 + norm_A**2)

    # Co/cross on B, S, A
    cc_B, co_B, cross_B = compute_co_cross_ratio(B, co_pairs, n)
    cc_S, co_S, cross_S = compute_co_cross_ratio(S, co_pairs, n)
    cc_A, co_A, cross_A = compute_co_cross_ratio(A, co_pairs, n)

    # Eigenvalue analysis of S
    eigs_S = np.sort(np.linalg.eigvalsh(S))[::-1]

    # Singular value analysis of A (skew-symmetric has purely imaginary eigenvalues)
    svd_A = np.linalg.svd(A, compute_uv=False)

    # Directional asymmetry: for each co-pair (i,j), measure B[i,j] vs B[j,i]
    dir_asym = {}
    if co_pairs is not None:
        for i, j in co_pairs:
            dir_asym[(i, j)] = {
                "B_ij": float(B[i, j]),
                "B_ji": float(B[j, i]),
                "S_ij": float(S[i, j]),  # same as S[j,i]
                "A_ij": float(A[i, j]),  # = -A[j,i]
                "ratio_ij_ji": float(abs(B[i, j]) / max(abs(B[j, i]), 1e-10)),
            }

    return {
        "norm_B": float(norm_B),
        "norm_S": float(norm_S),
        "norm_A": float(norm_A),
        "asym_ratio": float(asym_ratio),
        "sym_ratio": float(sym_ratio),
        "pyth_residual": float(pyth_check),
        "cc_B": float(cc_B) if cc_B is not None else None,
        "cc_S": float(cc_S) if cc_S is not None else None,
        "cc_A": float(cc_A) if cc_A is not None else None,
        "co_B": float(co_B) if co_B is not None else None,
        "cross_B": float(cross_B) if cross_B is not None else None,
        "co_S": float(co_S) if co_S is not None else None,
        "cross_S": float(cross_S) if cross_S is not None else None,
        "co_A": float(co_A) if co_A is not None else None,
        "cross_A": float(cross_A) if cross_A is not None else None,
        "eigs_S": eigs_S,
        "svd_A": svd_A,
        "dir_asym": dir_asym,
        "S": S,
        "A": A,
        "B": B,
    }


# ============================================================
# Main analysis
# ============================================================

all_results = {}

for name, exp in experiments.items():
    bridges = load_bridges(exp)
    if not bridges:
        print(f"WARNING: No bridge files found for {name}")
        continue

    n = exp["n"]
    co_pairs = exp["co_pairs"]

    results_per_layer_proj = {}
    asym_ratios = []
    cc_B_vals = []
    cc_S_vals = []
    cc_A_vals = []

    layers = sorted(set(k[0] for k in bridges.keys()))
    projs = sorted(set(k[1] for k in bridges.keys()))

    for layer in layers:
        for proj in projs:
            if (layer, proj) not in bridges:
                continue
            B = bridges[(layer, proj)]
            res = analyze_bridge(B, co_pairs, n)
            results_per_layer_proj[(layer, proj)] = res
            asym_ratios.append(res["asym_ratio"])
            if res["cc_B"] is not None:
                cc_B_vals.append(res["cc_B"])
                cc_S_vals.append(res["cc_S"])
                cc_A_vals.append(res["cc_A"])

    # Per-projection aggregates
    proj_stats = {}
    for proj in projs:
        proj_asym = [results_per_layer_proj[(l, proj)]["asym_ratio"]
                     for l in layers if (l, proj) in results_per_layer_proj]
        proj_cc_B = [results_per_layer_proj[(l, proj)]["cc_B"]
                     for l in layers if (l, proj) in results_per_layer_proj
                     and results_per_layer_proj[(l, proj)]["cc_B"] is not None]
        proj_cc_S = [results_per_layer_proj[(l, proj)]["cc_S"]
                     for l in layers if (l, proj) in results_per_layer_proj
                     and results_per_layer_proj[(l, proj)]["cc_S"] is not None]
        proj_cc_A = [results_per_layer_proj[(l, proj)]["cc_A"]
                     for l in layers if (l, proj) in results_per_layer_proj
                     and results_per_layer_proj[(l, proj)]["cc_A"] is not None]

        proj_stats[proj] = {
            "asym_mean": float(np.mean(proj_asym)),
            "asym_std": float(np.std(proj_asym)),
            "cc_B_mean": float(np.mean(proj_cc_B)) if proj_cc_B else None,
            "cc_S_mean": float(np.mean(proj_cc_S)) if proj_cc_S else None,
            "cc_A_mean": float(np.mean(proj_cc_A)) if proj_cc_A else None,
        }

    all_results[name] = {
        "n": n,
        "geometry": exp["geometry"],
        "model": exp["model"],
        "bd_expected": exp["bd_expected"],
        "n_bridges": len(bridges),
        "layers": layers,
        "projs": projs,
        "asym_ratio_mean": float(np.mean(asym_ratios)),
        "asym_ratio_std": float(np.std(asym_ratios)),
        "asym_ratio_min": float(np.min(asym_ratios)),
        "asym_ratio_max": float(np.max(asym_ratios)),
        "cc_B_mean": float(np.mean(cc_B_vals)) if cc_B_vals else None,
        "cc_B_std": float(np.std(cc_B_vals)) if cc_B_vals else None,
        "cc_S_mean": float(np.mean(cc_S_vals)) if cc_S_vals else None,
        "cc_S_std": float(np.std(cc_S_vals)) if cc_S_vals else None,
        "cc_A_mean": float(np.mean(cc_A_vals)) if cc_A_vals else None,
        "cc_A_std": float(np.std(cc_A_vals)) if cc_A_vals else None,
        "proj_stats": proj_stats,
        "per_lp": results_per_layer_proj,
    }


# ============================================================
# Output
# ============================================================
np.set_printoptions(precision=4, suppress=True, linewidth=140)

print("=" * 95)
print("AR-001: ASYMMETRY ANALYSIS OF BRIDGE MATRICES")
print("Co-pairs = consecutive 2x2 blocks: (0,1), (2,3), ...")
print("=" * 95)

# --- Table 1: Global asymmetry ---
print("\n## Table 1: Global Asymmetry Ratios (||A||_F / ||B||_F)")
hdr = f"{'Experiment':<15} {'n':>3} {'Geometry':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
print(hdr)
print("-" * len(hdr))
for name in experiments:
    if name not in all_results:
        continue
    r = all_results[name]
    print(f"{name:<15} {r['n']:>3} {r['geometry']:<28} "
          f"{r['asym_ratio_mean']:>8.4f} {r['asym_ratio_std']:>8.4f} "
          f"{r['asym_ratio_min']:>8.4f} {r['asym_ratio_max']:>8.4f}")

# --- Table 2: Co/cross decomposition ---
print("\n## Table 2: Co/Cross Ratios (consecutive block pairs)")
print(f"{'Experiment':<15} {'n':>3} {'BD':>3} {'cc(B)':>12} {'cc(S)':>12} {'cc(A)':>12} {'S/B%':>8} {'A/B%':>8}")
print("-" * 80)
for name in experiments:
    if name not in all_results:
        continue
    r = all_results[name]
    if r["cc_B_mean"] is not None:
        s_pct = r["cc_S_mean"] / max(r["cc_B_mean"], 1e-10) * 100
        a_pct = r["cc_A_mean"] / max(r["cc_B_mean"], 1e-10) * 100
        print(f"{name:<15} {r['n']:>3} {'Y' if r['bd_expected'] else 'N':>3} "
              f"{r['cc_B_mean']:>12.1f} {r['cc_S_mean']:>12.1f} "
              f"{r['cc_A_mean']:>12.1f} {s_pct:>7.1f}% {a_pct:>7.1f}%")
    else:
        print(f"{name:<15} {r['n']:>3} {'N':>3} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>8} {'N/A':>8}")

# --- Table 3: Per-projection ---
print("\n## Table 3: Per-Projection Breakdown")
print(f"{'Experiment':<15} {'Proj':>5} {'Asym':>8} {'cc(B)':>10} {'cc(S)':>10} {'cc(A)':>10}")
print("-" * 65)
for name in experiments:
    if name not in all_results:
        continue
    r = all_results[name]
    for proj in ["k", "q", "v", "o"]:
        if proj not in r["proj_stats"]:
            continue
        ps = r["proj_stats"][proj]
        cc_b = f"{ps['cc_B_mean']:.1f}" if ps["cc_B_mean"] is not None else "N/A"
        cc_s = f"{ps['cc_S_mean']:.1f}" if ps["cc_S_mean"] is not None else "N/A"
        cc_a = f"{ps['cc_A_mean']:.1f}" if ps["cc_A_mean"] is not None else "N/A"
        print(f"{name:<15} {proj:>5} {ps['asym_mean']:>8.4f} {cc_b:>10} {cc_s:>10} {cc_a:>10}")
    print()

# --- Directional asymmetry within co-pairs ---
print("\n## Table 4: Directional Asymmetry Within Co-Pairs")
print("  For each block-pair (i,j): B[i,j] vs B[j,i]")
print("  If B[i,j] >> B[j,i]: information flows predominantly i->j")
for name in ["Seed-43", "T-001r2", "H-ch4"]:
    if name not in all_results:
        continue
    r = all_results[name]
    print(f"\n  {name} ({r['geometry']}):")
    # Aggregate directional stats across all layers/projections
    pair_stats = defaultdict(lambda: {"ij": [], "ji": [], "ratio": []})
    for key, res in r["per_lp"].items():
        for (i, j), d in res["dir_asym"].items():
            pair_stats[(i, j)]["ij"].append(d["B_ij"])
            pair_stats[(i, j)]["ji"].append(d["B_ji"])
            pair_stats[(i, j)]["ratio"].append(d["ratio_ij_ji"])

    print(f"  {'Pair':>8} {'Mean B[i,j]':>12} {'Mean B[j,i]':>12} {'Mean |ratio|':>13} {'Sign pattern':>15}")
    for (i, j) in sorted(pair_stats.keys()):
        d = pair_stats[(i, j)]
        mean_ij = np.mean(d["ij"])
        mean_ji = np.mean(d["ji"])
        mean_ratio = np.mean(d["ratio"])
        # Check sign consistency
        signs_ij = np.sign(d["ij"])
        signs_ji = np.sign(d["ji"])
        ij_sign = "+" if np.mean(signs_ij) > 0.8 else "-" if np.mean(signs_ij) < -0.8 else "mixed"
        ji_sign = "+" if np.mean(signs_ji) > 0.8 else "-" if np.mean(signs_ji) < -0.8 else "mixed"
        sign_pat = f"({ij_sign},{ji_sign})"
        print(f"  ({i},{j}){'':<4} {mean_ij:>12.4f} {mean_ji:>12.4f} {mean_ratio:>13.1f} {sign_pat:>15}")

# --- Correlation ---
print("\n## Correlation: Asymmetry vs Co/Cross (per bridge)")
for name in ["Seed-43", "Seed-44", "exp3-Qwen", "T-001-full", "T-001r2", "H-ch4"]:
    if name not in all_results:
        continue
    r = all_results[name]
    asym_vals = []
    cc_vals = []
    for key, res in r["per_lp"].items():
        if res["cc_B"] is not None:
            asym_vals.append(res["asym_ratio"])
            cc_vals.append(np.log10(max(res["cc_B"], 0.01)))  # log scale
    if len(asym_vals) > 2:
        corr = np.corrcoef(asym_vals, cc_vals)[0, 1]
        print(f"  {name:<15} r = {corr:>+.4f}  (N = {len(asym_vals)}, log10 cc(B))")

# --- SVD of A: does the antisymmetric part have rank structure? ---
print("\n## SVD of Antisymmetric Part A (mean singular values across all bridges)")
for name in ["Seed-43", "T-001r2", "H-ch3"]:
    if name not in all_results:
        continue
    r = all_results[name]
    all_svd = []
    for key, res in r["per_lp"].items():
        all_svd.append(res["svd_A"])
    mean_svd = np.mean(all_svd, axis=0)
    std_svd = np.std(all_svd, axis=0)
    print(f"\n  {name} (n={r['n']}):")
    print(f"  {'Sing. val #':>12} {'Mean':>10} {'Std':>10} {'% of total':>12}")
    total = np.sum(mean_svd)
    for i, (m, s) in enumerate(zip(mean_svd, std_svd)):
        pct = m / max(total, 1e-10) * 100
        print(f"  {i+1:>12} {m:>10.6f} {s:>10.6f} {pct:>11.1f}%")

# --- Eigenvalues of S ---
print("\n## Eigenvalues of Symmetric Part S (mean across all bridges)")
for name in ["Seed-43", "T-001r2", "H-ch3"]:
    if name not in all_results:
        continue
    r = all_results[name]
    all_eigs = []
    for key, res in r["per_lp"].items():
        all_eigs.append(res["eigs_S"])
    mean_eigs = np.mean(all_eigs, axis=0)
    std_eigs = np.std(all_eigs, axis=0)
    print(f"\n  {name} (n={r['n']}):")
    for i, (m, s) in enumerate(zip(mean_eigs, std_eigs)):
        print(f"    eig[{i}] = {m:.6f} +/- {s:.6f}")

# --- Layer depth ---
print("\n## Asymmetry and Co/Cross by Layer Depth")
for name in ["Seed-43", "T-001r2", "H-ch3"]:
    if name not in all_results:
        continue
    r = all_results[name]
    layer_data = defaultdict(lambda: {"asym": [], "cc_B": [], "cc_S": [], "cc_A": []})
    for (layer, proj), res in r["per_lp"].items():
        layer_data[layer]["asym"].append(res["asym_ratio"])
        if res["cc_B"] is not None:
            layer_data[layer]["cc_B"].append(res["cc_B"])
            layer_data[layer]["cc_S"].append(res["cc_S"])
            layer_data[layer]["cc_A"].append(res["cc_A"])

    print(f"\n  {name}:")
    print(f"  {'Layer':>5} {'Asym':>8} {'cc(B)':>10} {'cc(S)':>10} {'cc(A)':>10}")
    for layer in sorted(layer_data.keys()):
        d = layer_data[layer]
        asym = np.mean(d["asym"])
        cc_b = f"{np.mean(d['cc_B']):.1f}" if d["cc_B"] else "N/A"
        cc_s = f"{np.mean(d['cc_S']):.1f}" if d["cc_S"] else "N/A"
        cc_a = f"{np.mean(d['cc_A']):.1f}" if d["cc_A"] else "N/A"
        print(f"  {layer:>5} {asym:>8.4f} {cc_b:>10} {cc_s:>10} {cc_a:>10}")

# --- Example matrices ---
print("\n## Example: Highest co/cross bridge in Seed-43")
r = all_results["Seed-43"]
best_key = max(
    [k for k, v in r["per_lp"].items() if v["cc_B"] is not None],
    key=lambda k: r["per_lp"][k]["cc_B"]
)
res = r["per_lp"][best_key]
print(f"  Layer {best_key[0]}, {best_key[1]}_proj")
print(f"  cc(B)={res['cc_B']:.1f}, cc(S)={res['cc_S']:.1f}, cc(A)={res['cc_A']:.1f}")
print(f"  ||A||/||B|| = {res['asym_ratio']:.4f}")
print(f"\n  B (original):")
for row in res["B"]:
    print(f"    [{', '.join(f'{v:>10.5f}' for v in row)}]")
print(f"\n  S (symmetric):")
for row in res["S"]:
    print(f"    [{', '.join(f'{v:>10.5f}' for v in row)}]")
print(f"\n  A (antisymmetric):")
for row in res["A"]:
    print(f"    [{', '.join(f'{v:>10.5f}' for v in row)}]")

# ============================================================
# KEY FINDINGS
# ============================================================
print("\n" + "=" * 95)
print("KEY FINDINGS")
print("=" * 95)

bd_asym = []
non_bd_asym = []
for name, r in all_results.items():
    for key, res in r["per_lp"].items():
        if r["bd_expected"]:
            bd_asym.append(res["asym_ratio"])
        else:
            non_bd_asym.append(res["asym_ratio"])

print(f"\n1. ASYMMETRY IS BIMODAL:")
print(f"   BD experiments:     ||A||/||B|| = {np.mean(bd_asym):.4f} +/- {np.std(bd_asym):.4f}")
print(f"   Non-BD (H-ch3):    ||A||/||B|| = {np.mean(non_bd_asym):.4f} +/- {np.std(non_bd_asym):.4f}")
print(f"   Ratio: {np.mean(bd_asym)/np.mean(non_bd_asym):.1f}x")
print(f"   -> BD experiments are ~{np.mean(bd_asym)/np.mean(non_bd_asym):.0f}x more asymmetric than non-BD")

print(f"\n2. SYMMETRIZATION EFFECT on co/cross:")
for name in experiments:
    if name not in all_results:
        continue
    r = all_results[name]
    if r["cc_B_mean"] is not None and r["cc_B_mean"] > 10:
        retention = r["cc_S_mean"] / r["cc_B_mean"] * 100
        a_retention = r["cc_A_mean"] / r["cc_B_mean"] * 100
        print(f"   {name}: cc(B)={r['cc_B_mean']:.0f} -> cc(S)={r['cc_S_mean']:.0f} ({retention:.0f}%) "
              f"-> cc(A)={r['cc_A_mean']:.0f} ({a_retention:.0f}%)")

print(f"\n3. INFORMATION FLOW DIRECTION:")
for name in ["Seed-43", "T-001r2"]:
    if name not in all_results:
        continue
    r = all_results[name]
    # Check if B[i,j] and B[j,i] have consistent sign/magnitude pattern
    all_ij = []
    all_ji = []
    for key, res in r["per_lp"].items():
        for (i, j), d in res["dir_asym"].items():
            all_ij.append(abs(d["B_ij"]))
            all_ji.append(abs(d["B_ji"]))
    print(f"   {name}: mean|B[i,j]|={np.mean(all_ij):.4f}, mean|B[j,i]|={np.mean(all_ji):.4f}, "
          f"ratio={np.mean(all_ij)/np.mean(all_ji):.2f}")

# Save raw results
json_results = {}
for name, r in all_results.items():
    jr = {k: v for k, v in r.items() if k not in ("per_lp", "layers", "projs")}
    jr["layers"] = r["layers"]
    jr["projs"] = r["projs"]
    jr["per_lp_summary"] = {}
    for (layer, proj), res in r["per_lp"].items():
        jr["per_lp_summary"][f"L{layer}_{proj}"] = {
            k: v for k, v in res.items()
            if k not in ("S", "A", "B", "eigs_S", "svd_A", "dir_asym")
        }
    json_results[name] = jr

with open("C:/falco/rhombic/results/AR-001-raw-results.json", "w") as f:
    json.dump(json_results, f, indent=2, default=str)
print("\n\nRaw results saved to AR-001-raw-results.json")
