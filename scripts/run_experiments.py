"""
Paper 2 experiment runner — all four experiments, reproducible.

Usage:
    python scripts/run_experiments.py

Outputs:
    results/paper2/experiment_1_weighted_benchmarks.txt
    results/paper2/experiment_2_optimal_assignment.txt
    results/paper2/experiment_3_prime_coherence.txt
    results/paper2/experiment_4_spectral.txt
"""

from __future__ import annotations

import os
import sys
import time
import json
import numpy as np
import networkx as nx

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rhombic.polyhedron import RhombicDodecahedron
from rhombic.corpus import (
    edge_values, TRACKED_PRIMES, TRUMP_VALUES, CANONICAL_EDGE_ORDER,
    prime_membership, weight_distributions, corpus_stats,
)
from rhombic.assignment import (
    total_variation, optimal_assignment, random_assignment_tv, compare_graphs,
)
from rhombic.spectral import (
    spectrum, fiedler_value, spectral_gap, compare_spectra,
)
from rhombic.benchmark import matched_lattices, _avg_shortest_path, _diameter, _fiedler_value


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "paper2")
SEED = 42


def ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Experiment 1: Weighted FCC vs Cubic Benchmarks
# ══════════════════════════════════════════════════════════════════════

def experiment_1():
    """Does the FCC topology advantage persist under non-uniform weighting?"""
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Weighted FCC vs Cubic Benchmarks")
    print("=" * 72)

    scales = [125, 1000, 8000]
    dists = weight_distributions(seed=SEED)
    dist_names = ['uniform', 'random', 'power_law', 'corpus']

    lines = ["EXPERIMENT 1: Weighted FCC vs Cubic Benchmarks\n"]
    lines.append(f"{'Scale':>8} {'Distribution':>14} {'Metric':>20} {'Cubic':>12} {'FCC':>12} {'Ratio':>8}\n")
    lines.append("-" * 78 + "\n")

    for scale in scales:
        print(f"\n  Scale ~{scale}...")
        cubic, fcc = matched_lattices(scale)
        Gc = cubic.to_networkx()
        Gf = fcc.to_networkx()

        for dist_name in dist_names:
            weights = dists[dist_name]

            # Apply weights to edges (cycle if lattice has more edges than 24)
            def apply_weights(G, w):
                for idx, (u, v) in enumerate(G.edges()):
                    G[u][v]['weight'] = w[idx % len(w)]
                return G

            Gc_w = apply_weights(Gc.copy(), weights)
            Gf_w = apply_weights(Gf.copy(), weights)

            # Metrics (skip expensive path computation at large scale)
            compute_paths = scale <= 1000
            asp_c = _avg_shortest_path(Gc_w, weight='weight') if compute_paths else None
            asp_f = _avg_shortest_path(Gf_w, weight='weight') if compute_paths else None
            if asp_c and asp_f:
                ratio = asp_f / asp_c
                lines.append(f"{scale:>8} {dist_name:>14} {'Avg Shortest Path':>20} {asp_c:>12.3f} {asp_f:>12.3f} {ratio:>8.3f}\n")
                print(f"    {dist_name} ASP: cubic={asp_c:.3f} fcc={asp_f:.3f} ratio={ratio:.3f}")

            fied_c = _fiedler_value(Gc_w, weight='weight')
            fied_f = _fiedler_value(Gf_w, weight='weight')
            if fied_c and fied_f:
                ratio = fied_f / fied_c
                lines.append(f"{scale:>8} {dist_name:>14} {'Algebraic Connect.':>20} {fied_c:>12.4f} {fied_f:>12.4f} {ratio:>8.2f}\n")
                print(f"    {dist_name} Fiedler: cubic={fied_c:.4f} fcc={fied_f:.4f} ratio={ratio:.2f}")

        lines.append("\n")

    with open(os.path.join(RESULTS_DIR, "experiment_1_weighted_benchmarks.txt"), "w") as f:
        f.writelines(lines)
    print("  -> Results written.")


# ══════════════════════════════════════════════════════════════════════
# Experiment 2: Optimal Assignment on Rhombic Dodecahedron
# ══════════════════════════════════════════════════════════════════════

def experiment_2():
    """Does the rhombic dodecahedron sort structured weights?"""
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Optimal Assignment on Rhombic Dodecahedron")
    print("=" * 72)

    rd = RhombicDodecahedron()
    ve = {v: rd.vertex_star(v) for v in range(14)}
    corpus = [float(v) for v in edge_values()]

    lines = ["EXPERIMENT 2: Optimal Assignment on Rhombic Dodecahedron\n\n"]

    # Test with multiple weight sets
    dists = weight_distributions(seed=SEED)
    for dist_name in ['random', 'power_law', 'corpus']:
        weights = dists[dist_name]
        print(f"\n  Distribution: {dist_name}")

        best_assign, best_tv = optimal_assignment(
            ve, rd.edges, weights, n_restarts=100, max_iters=5000, seed=SEED)
        null_tvs = random_assignment_tv(ve, weights, n_samples=10000, seed=SEED)
        p_value = float(np.mean(null_tvs <= best_tv))

        lines.append(f"Distribution: {dist_name}\n")
        lines.append(f"  Optimal TV: {best_tv:.4f}\n")
        lines.append(f"  Random mean TV: {null_tvs.mean():.4f} +/- {null_tvs.std():.4f}\n")
        lines.append(f"  p-value: {p_value:.4f}\n")
        lines.append(f"  Improvement over random mean: {(1 - best_tv/null_tvs.mean())*100:.1f}%\n\n")

        print(f"    Optimal TV: {best_tv:.4f}")
        print(f"    Random mean: {null_tvs.mean():.4f} +/- {null_tvs.std():.4f}")
        print(f"    p-value: {p_value:.4f}")

    # Also compare RD against a random 24-edge graph
    lines.append("\nComparison: RD vs random 24-edge graph\n")
    rng = np.random.default_rng(SEED)

    # Build a random graph with 14 vertices and 24 edges
    Gr = nx.gnm_random_graph(14, 24, seed=SEED)
    while not nx.is_connected(Gr):
        Gr = nx.gnm_random_graph(14, 24, seed=rng.integers(10000))
    random_edges = list(Gr.edges())
    random_ve: dict[int, list[int]] = {i: [] for i in range(14)}
    for idx, (u, v) in enumerate(random_edges):
        random_ve[u].append(idx)
        random_ve[v].append(idx)

    for dist_name in ['corpus']:
        weights = dists[dist_name]

        _, rd_tv = optimal_assignment(ve, rd.edges, weights, n_restarts=50, seed=SEED)
        _, rand_tv = optimal_assignment(random_ve, random_edges, weights, n_restarts=50, seed=SEED)

        lines.append(f"  {dist_name}: RD optimal TV = {rd_tv:.4f}, Random graph TV = {rand_tv:.4f}\n")
        print(f"\n  RD vs random graph ({dist_name}): RD={rd_tv:.4f}, Random={rand_tv:.4f}")

    with open(os.path.join(RESULTS_DIR, "experiment_2_optimal_assignment.txt"), "w") as f:
        f.writelines(lines)
    print("  -> Results written.")


# ══════════════════════════════════════════════════════════════════════
# Experiment 3: Prime Coherence at Vertex Stars
# ══════════════════════════════════════════════════════════════════════

def experiment_3():
    """Do the 8 tracked primes organize naturally at vertex stars?"""
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Prime Coherence at Vertex Stars")
    print("=" * 72)

    rd = RhombicDodecahedron()
    values = edge_values()

    lines = ["EXPERIMENT 3: Prime Coherence at Vertex Stars\n\n"]

    # Map primes to trivalent vertices (8 primes -> 8 cubic vertices)
    # For each assignment of primes to vertices, measure coherence:
    # coherence(v, p) = fraction of edges incident to v whose values are divisible by p
    primes = TRACKED_PRIMES[:]

    # Under optimal assignment from experiment 2, check if the edge values
    # at each vertex star share the vertex's prime factor
    ve = {v: rd.vertex_star(v) for v in range(14)}
    best_assign, _ = optimal_assignment(
        ve, rd.edges, [float(v) for v in values],
        n_restarts=100, max_iters=5000, seed=SEED)

    # For each trivalent vertex, which edges are incident?
    lines.append("Optimal assignment: edge values at each trivalent vertex star\n\n")
    for v in rd.trivalent_vertices:
        star_edges = rd.vertex_star(v)
        star_values = [best_assign[e] for e in star_edges]
        star_ints = [int(round(sv)) for sv in star_values]

        # Check which tracked primes divide ALL values in this star
        shared = []
        for p in primes:
            if all(prime_membership(sv, p) for sv in star_ints if sv > 1):
                shared.append(p)

            # Check which primes divide at least one value
        partial = []
        for p in primes:
            count = sum(1 for sv in star_ints if prime_membership(sv, p))
            if count > 0:
                partial.append((p, count, len(star_ints)))

        lines.append(f"  Vertex {v} (degree {rd.degree(v)}): values = {star_ints}\n")
        if shared:
            lines.append(f"    Fully shared primes: {shared}\n")
        if partial:
            lines.append(f"    Partial: {[(p, f'{c}/{t}') for p, c, t in partial]}\n")

        print(f"  V{v}: {star_ints} shared={shared}")

    # Null model: random assignment, measure total prime hits
    lines.append("\nPrime coherence score vs null distribution\n")
    rng = np.random.default_rng(SEED)

    def coherence_score(assignment):
        """Sum over trivalent vertices of (max prime hits at that vertex)."""
        score = 0
        for v in rd.trivalent_vertices:
            star_edges = rd.vertex_star(v)
            star_vals = [int(round(assignment[e])) for e in star_edges]
            best_hits = 0
            for p in primes:
                hits = sum(1 for sv in star_vals if sv > 1 and prime_membership(sv, p))
                best_hits = max(best_hits, hits)
            score += best_hits
        return score

    optimal_score = coherence_score(best_assign)

    null_scores = []
    for _ in range(10000):
        perm = list(best_assign)
        rng.shuffle(perm)
        null_scores.append(coherence_score(perm))
    null_scores = np.array(null_scores)

    p_value = float(np.mean(null_scores >= optimal_score))
    lines.append(f"  Optimal coherence score: {optimal_score}\n")
    lines.append(f"  Null mean: {null_scores.mean():.2f} +/- {null_scores.std():.2f}\n")
    lines.append(f"  p-value (one-sided, >= optimal): {p_value:.4f}\n")

    print(f"\n  Coherence score: {optimal_score}")
    print(f"  Null: {null_scores.mean():.2f} +/- {null_scores.std():.2f}")
    print(f"  p-value: {p_value:.4f}")

    with open(os.path.join(RESULTS_DIR, "experiment_3_prime_coherence.txt"), "w") as f:
        f.writelines(lines)
    print("  -> Results written.")


# ══════════════════════════════════════════════════════════════════════
# Experiment 4: Spectral Properties
# ══════════════════════════════════════════════════════════════════════

def experiment_4():
    """Do corpus weights cooperate with rhombic topology?"""
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Spectral Properties")
    print("=" * 72)

    rd = RhombicDodecahedron()
    dists = weight_distributions(seed=SEED)

    lines = ["EXPERIMENT 4: Spectral Properties\n\n"]
    lines.append(f"{'Distribution':>14} {'Fiedler':>12} {'Spec. Gap':>12} {'lambda_max':>12}\n")
    lines.append("-" * 54 + "\n")

    for dist_name in ['uniform', 'random', 'power_law', 'corpus']:
        weights = dists[dist_name]
        fv = fiedler_value(14, rd.edges, weights)
        sg = spectral_gap(14, rd.edges, weights)
        eigs = spectrum(14, rd.edges, weights)
        lmax = float(eigs[-1])

        lines.append(f"{dist_name:>14} {fv:>12.6f} {sg:>12.6f} {lmax:>12.4f}\n")
        print(f"  {dist_name}: Fiedler={fv:.6f}, gap={sg:.6f}, lambda_max={lmax:.4f}")

    # Compare corpus Fiedler against random weight Fiedler distribution
    lines.append("\nCorpus Fiedler vs random weight null distribution\n")
    rng = np.random.default_rng(SEED)
    corpus_fiedler = fiedler_value(14, rd.edges, dists['corpus'])

    null_fiedlers = []
    for _ in range(10000):
        random_weights = rng.uniform(0.0, 1.0, size=24).tolist()
        null_fiedlers.append(fiedler_value(14, rd.edges, random_weights))
    null_fiedlers = np.array(null_fiedlers)

    p_high = float(np.mean(null_fiedlers >= corpus_fiedler))
    lines.append(f"  Corpus Fiedler: {corpus_fiedler:.6f}\n")
    lines.append(f"  Null mean: {null_fiedlers.mean():.6f} +/- {null_fiedlers.std():.6f}\n")
    lines.append(f"  p-value (>= corpus): {p_high:.4f}\n")

    print(f"\n  Corpus Fiedler: {corpus_fiedler:.6f}")
    print(f"  Null: {null_fiedlers.mean():.6f} +/- {null_fiedlers.std():.6f}")
    print(f"  p-value: {p_high:.4f}")

    # Full spectrum for each distribution
    lines.append("\nFull spectra:\n")
    for dist_name in ['uniform', 'random', 'power_law', 'corpus']:
        weights = dists[dist_name]
        eigs = spectrum(14, rd.edges, weights)
        eig_str = ", ".join(f"{e:.4f}" for e in eigs)
        lines.append(f"  {dist_name}: [{eig_str}]\n")

    with open(os.path.join(RESULTS_DIR, "experiment_4_spectral.txt"), "w") as f:
        f.writelines(lines)
    print("  -> Results written.")


# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    ensure_dir()

    print("Pure Number Architecture — Paper 2 Experiments")
    print(f"Seed: {SEED}")

    # Print corpus summary
    stats = corpus_stats()
    print(f"\nCorpus: {stats.n_values} values, range [{stats.min_value}, {stats.max_value}]")
    print(f"Mean: {stats.mean_value:.1f}, Std: {stats.std_value:.1f}")
    print(f"Tracked primes: {TRACKED_PRIMES}")

    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()

    elapsed = time.perf_counter() - t0
    print(f"\nAll experiments complete in {elapsed:.1f}s")
    print(f"Results at: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
