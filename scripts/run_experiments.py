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

from rhombic.polyhedron import RhombicDodecahedron, cuboctahedron_graph
from rhombic.corpus import (
    edge_values, TRACKED_PRIMES, TRUMP_VALUES, CANONICAL_EDGE_ORDER,
    prime_membership, weight_distributions, corpus_stats, direction_weights,
)
from rhombic.assignment import (
    total_variation, optimal_assignment, random_assignment_tv, compare_graphs,
    prime_vertex_score, optimal_prime_assignment, null_prime_scores,
)
from rhombic.spectral import (
    spectrum, fiedler_value, spectral_gap, compare_spectra,
    eigenvalue_multiplicity_pattern, spectral_distance, spectrum_summary,
)
from rhombic.spatial import build_adjacency
from rhombic.context import (
    weighted_information_diffusion, weighted_consensus_speed,
    information_diffusion, consensus_speed, _rounds_to_threshold,
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
# Experiment 5: Multi-Cell Weighted Tessellation (Direction-Based)
# ══════════════════════════════════════════════════════════════════════

def experiment_5():
    """Does the FCC advantage amplify under direction-based weighting?"""
    print("\n" + "=" * 72)
    print("EXPERIMENT 5: Multi-Cell Weighted Tessellation (Direction-Based)")
    print("=" * 72)

    scales = [125, 1000, 8000]
    corpus_vals = [float(v) for v in edge_values()]
    dists = weight_distributions(seed=SEED)

    # Direction weights for each distribution
    dir_weights_fcc = {}
    dir_weights_cubic = {}
    for name, vals in dists.items():
        dir_weights_fcc[name] = direction_weights(vals, 6)
        dir_weights_cubic[name] = direction_weights(vals, 3)

    lines = ["EXPERIMENT 5: Multi-Cell Weighted Tessellation (Direction-Based)\n\n"]
    lines.append("Direction-based weighting: FCC has 6 direction pairs (4 values/pair),\n")
    lines.append("cubic has 3 direction pairs (8 values/pair).\n\n")

    for scale in scales:
        print(f"\n  Scale ~{scale}...")
        cubic, fcc = matched_lattices(scale)

        # Get edge directions
        dirs_c = cubic.edge_directions()
        dirs_f = fcc.edge_directions()

        # Build adjacency
        adj_c = build_adjacency(cubic.edges, cubic.node_count)
        adj_f = build_adjacency(fcc.edges, fcc.node_count)

        for dist_name in ['uniform', 'random', 'power_law', 'corpus']:
            dw_c = dir_weights_cubic[dist_name]
            dw_f = dir_weights_fcc[dist_name]

            # Build per-edge weight dicts
            ew_c = {}
            for dir_idx, edge_indices in dirs_c.items():
                w = dw_c[dir_idx]
                for e_idx in edge_indices:
                    u, v = cubic.edges[e_idx]
                    ew_c[(u, v)] = w
                    ew_c[(v, u)] = w

            ew_f = {}
            for dir_idx, edge_indices in dirs_f.items():
                w = dw_f[dir_idx]
                for e_idx in edge_indices:
                    u, v = fcc.edges[e_idx]
                    ew_f[(u, v)] = w
                    ew_f[(v, u)] = w

            # Weighted Fiedler via networkx
            Gc = cubic.to_networkx()
            Gf = fcc.to_networkx()
            for u, v in Gc.edges():
                Gc[u][v]['weight'] = ew_c.get((u, v), 1.0)
            for u, v in Gf.edges():
                Gf[u][v]['weight'] = ew_f.get((u, v), 1.0)

            fied_c = _fiedler_value(Gc, weight='weight')
            fied_f = _fiedler_value(Gf, weight='weight')

            if fied_c and fied_f:
                ratio = fied_f / fied_c
                lines.append(f"Scale {scale:>5} {dist_name:>12} Fiedler: "
                             f"cubic={fied_c:.4f} fcc={fied_f:.4f} ratio={ratio:.2f}\n")
                print(f"    {dist_name} Fiedler: cubic={fied_c:.4f} fcc={fied_f:.4f} "
                      f"ratio={ratio:.2f}")

            # Weighted diffusion (skip at 8000 — too slow for pure-Python loops)
            if scale <= 1000:
                center_c = np.argmin(np.linalg.norm(
                    cubic.positions - cubic.positions.mean(axis=0), axis=1))
                center_f = np.argmin(np.linalg.norm(
                    fcc.positions - fcc.positions.mean(axis=0), axis=1))

                curve_c = weighted_information_diffusion(
                    adj_c, ew_c, center_c, cubic.node_count, max_rounds=30)
                curve_f = weighted_information_diffusion(
                    adj_f, ew_f, center_f, fcc.node_count, max_rounds=30)

                r50_c = _rounds_to_threshold(curve_c, 0.50)
                r50_f = _rounds_to_threshold(curve_f, 0.50)
                speedup = r50_c / max(1, r50_f)
                lines.append(f"Scale {scale:>5} {dist_name:>12} Diffusion 50%: "
                             f"cubic={r50_c} fcc={r50_f} speedup={speedup:.2f}x\n")
                print(f"    {dist_name} Diffusion 50%: cubic={r50_c} fcc={r50_f} "
                      f"speedup={speedup:.2f}x")

            # Weighted consensus (skip at 8000)
            if scale <= 1000:
                rng = np.random.default_rng(SEED)
                init_c = rng.standard_normal(cubic.node_count)
                init_f = rng.standard_normal(fcc.node_count)

                rounds_c, _ = weighted_consensus_speed(
                    adj_c, ew_c, init_c, max_rounds=500)
                rounds_f, _ = weighted_consensus_speed(
                    adj_f, ew_f, init_f, max_rounds=500)
                speedup = rounds_c / max(1, rounds_f)
                lines.append(f"Scale {scale:>5} {dist_name:>12} Consensus: "
                             f"cubic={rounds_c} fcc={rounds_f} speedup={speedup:.2f}x\n")
                print(f"    {dist_name} Consensus: cubic={rounds_c} fcc={rounds_f} "
                      f"speedup={speedup:.2f}x")

        lines.append("\n")

    with open(os.path.join(RESULTS_DIR, "experiment_5_direction_weighted.txt"), "w") as f:
        f.writelines(lines)
    print("  -> Results written.")


# ══════════════════════════════════════════════════════════════════════
# Experiment 6: Targeted Prime-Vertex Mapping
# ══════════════════════════════════════════════════════════════════════

def experiment_6():
    """Does the determined prime-to-Law mapping produce significant coherence?"""
    print("\n" + "=" * 72)
    print("EXPERIMENT 6: Targeted Prime-Vertex Mapping")
    print("=" * 72)

    rd = RhombicDodecahedron()
    values = edge_values()
    ve = {v: rd.vertex_star(v) for v in range(14)}

    # The DETERMINED prime-to-Law mapping (from MEMORY.md)
    # Map to trivalent vertices (indices 0-7)
    primes = TRACKED_PRIMES[:]  # [67, 23, 29, 17, 19, 31, 11, 89]
    trivalent = rd.trivalent_vertices  # [0, 1, 2, 3, 4, 5, 6, 7]

    lines = ["EXPERIMENT 6: Targeted Prime-Vertex Mapping\n\n"]
    lines.append("Question: Does the determined prime-to-Law mapping produce\n")
    lines.append("significant coherence when primes are assigned to RD vertices?\n\n")
    lines.append(f"Primes: {primes}\n")
    lines.append(f"Target vertices: {trivalent} (trivalent, degree 3)\n\n")

    # First: get optimal edge assignment (TV-optimal from SA)
    print("  Computing TV-optimal edge assignment...")
    best_assign, best_tv = optimal_assignment(
        ve, rd.edges, [float(v) for v in values],
        n_restarts=200, max_iters=10000, seed=SEED)
    best_assign_int = [int(round(v)) for v in best_assign]

    lines.append(f"TV-optimal edge assignment (TV={best_tv:.2f}):\n")
    for v in trivalent:
        star = rd.vertex_star(v)
        star_vals = [best_assign_int[e] for e in star]
        lines.append(f"  Vertex {v}: edges {star} -> values {star_vals}\n")

    # Exhaustive search over all 8! prime-to-vertex mappings
    print("  Exhaustive search over 8! = 40,320 prime-to-vertex mappings...")
    all_scores = null_prime_scores(ve, best_assign_int, primes, trivalent)

    # Find the optimal mapping
    best_mapping, best_score = optimal_prime_assignment(
        ve, best_assign_int, primes, trivalent)

    lines.append(f"\nOptimal prime-to-vertex mapping (score={best_score:.1f}):\n")
    for p, v in sorted(best_mapping.items()):
        star = rd.vertex_star(v)
        star_vals = [best_assign_int[e] for e in star]
        divisible = [sv for sv in star_vals if sv > 0 and sv % p == 0]
        lines.append(f"  Prime {p:>3} -> Vertex {v} "
                     f"(values {star_vals}, divisible: {divisible})\n")

    # Statistics
    mean_score = float(all_scores.mean())
    std_score = float(all_scores.std())
    p_value = float(np.mean(all_scores >= best_score))
    percentile = float(np.mean(all_scores <= best_score)) * 100

    lines.append(f"\nExhaustive null distribution (N=40,320):\n")
    lines.append(f"  Best score: {best_score:.1f}\n")
    lines.append(f"  Mean: {mean_score:.2f} +/- {std_score:.2f}\n")
    lines.append(f"  p-value (>= best): {p_value:.6f}\n")
    lines.append(f"  Percentile of best: {percentile:.2f}%\n")

    # Test the DETERMINED mapping specifically
    determined_mapping = dict(zip(primes, trivalent))  # primes[i] -> vertex i
    determined_score = prime_vertex_score(ve, best_assign_int, determined_mapping)
    p_determined = float(np.mean(all_scores >= determined_score))

    lines.append(f"\nDetermined mapping (identity: prime[i] -> vertex[i]):\n")
    lines.append(f"  Score: {determined_score:.1f}\n")
    lines.append(f"  p-value: {p_determined:.6f}\n")
    lines.append(f"  Rank: {int(np.sum(all_scores > determined_score)) + 1} / 40,320\n")

    print(f"  Optimal score: {best_score:.1f}")
    print(f"  Null mean: {mean_score:.2f} +/- {std_score:.2f}")
    print(f"  Optimal p-value: {p_value:.6f}")
    print(f"  Determined score: {determined_score:.1f}, p={p_determined:.6f}")

    with open(os.path.join(RESULTS_DIR, "experiment_6_prime_vertex.txt"), "w") as f:
        f.writelines(lines)
    print("  -> Results written.")


# ══════════════════════════════════════════════════════════════════════
# Experiment 7: Spectral Comparison Across Polytopes
# ══════════════════════════════════════════════════════════════════════

def experiment_7():
    """Is the RD's spectral response to corpus weights unique among 24-edge graphs?"""
    print("\n" + "=" * 72)
    print("EXPERIMENT 7: Spectral Comparison Across Polytopes")
    print("=" * 72)

    rd = RhombicDodecahedron()
    dists = weight_distributions(seed=SEED)

    # Build comparison graphs
    print("  Building comparison graphs...")

    # 1. Cuboctahedron (12V, 24E, 4-regular)
    G_cuboct = cuboctahedron_graph()

    # 2. K(4,6) complete bipartite (10V, 24E)
    G_k46 = nx.complete_bipartite_graph(4, 6)

    # 3. 3-regular graph on 16V (16V, 24E)
    G_3reg = nx.random_regular_graph(3, 16, seed=SEED)

    # 4. Random G(14,24)
    rng_graph = np.random.default_rng(SEED)
    G_rand = nx.gnm_random_graph(14, 24, seed=SEED)
    while not nx.is_connected(G_rand):
        G_rand = nx.gnm_random_graph(14, 24, seed=int(rng_graph.integers(100000)))

    graphs = {
        "RD (14V,24E bipartite)": {
            "n_vertices": 14, "edges": rd.edges, "graph": rd.to_networkx(),
        },
        "Cuboctahedron (12V,24E)": {
            "n_vertices": 12, "edges": list(G_cuboct.edges()),
            "graph": G_cuboct,
        },
        "K(4,6) (10V,24E)": {
            "n_vertices": 10, "edges": list(G_k46.edges()),
            "graph": G_k46,
        },
        "3-regular (16V,24E)": {
            "n_vertices": 16, "edges": list(G_3reg.edges()),
            "graph": G_3reg,
        },
        "G(14,24) random": {
            "n_vertices": 14, "edges": list(G_rand.edges()),
            "graph": G_rand,
        },
    }

    lines = ["EXPERIMENT 7: Spectral Comparison Across Polytopes\n\n"]

    # Part A: Spectrum under each distribution
    lines.append("=" * 72 + "\n")
    lines.append("PART A: Spectrum summary under 4 distributions\n")
    lines.append("=" * 72 + "\n\n")

    for dist_name in ['uniform', 'random', 'power_law', 'corpus']:
        weights = dists[dist_name]
        lines.append(f"Distribution: {dist_name}\n")
        lines.append(f"{'Graph':>30} {'Fiedler':>10} {'Gap':>10} "
                     f"{'lambda_max':>12} {'Distinct':>10}\n")
        lines.append("-" * 76 + "\n")

        for gname, gdata in graphs.items():
            nv = gdata['n_vertices']
            ed = gdata['edges']
            n_edges = len(ed)
            # Cycle weights to match edge count
            w = [weights[i % len(weights)] for i in range(n_edges)]
            ss = spectrum_summary(gname, nv, ed, w)
            lines.append(f"{gname:>30} {ss.fiedler:>10.4f} {ss.spectral_gap:>10.4f} "
                         f"{ss.lambda_max:>12.4f} {ss.n_distinct_eigenvalues:>10}\n")
            print(f"    {dist_name} | {gname}: Fiedler={ss.fiedler:.4f}")

        lines.append("\n")

    # Part B: Corpus Fiedler percentile vs 10K random weights
    lines.append("=" * 72 + "\n")
    lines.append("PART B: Corpus Fiedler percentile (10K random weights)\n")
    lines.append("=" * 72 + "\n\n")

    rng = np.random.default_rng(SEED)
    corpus_weights = dists['corpus']

    for gname, gdata in graphs.items():
        nv = gdata['n_vertices']
        ed = gdata['edges']
        n_edges = len(ed)
        w_corpus = [corpus_weights[i % len(corpus_weights)] for i in range(n_edges)]
        corpus_fv = fiedler_value(nv, ed, w_corpus)

        null_fvs = []
        for _ in range(10000):
            w_rand = rng.uniform(0.0, 1.0, size=n_edges).tolist()
            null_fvs.append(fiedler_value(nv, ed, w_rand))
        null_fvs = np.array(null_fvs)

        percentile = float(np.mean(null_fvs <= corpus_fv)) * 100
        lines.append(f"{gname:>30}: corpus Fiedler={corpus_fv:.6f}, "
                     f"percentile={percentile:.2f}%\n")
        print(f"  {gname}: corpus Fiedler percentile = {percentile:.1f}%")

    # Part C: Multiplicity patterns (degeneracy breaking)
    lines.append("\n" + "=" * 72 + "\n")
    lines.append("PART C: Eigenvalue multiplicity patterns\n")
    lines.append("=" * 72 + "\n\n")

    for gname, gdata in graphs.items():
        nv = gdata['n_vertices']
        ed = gdata['edges']
        n_edges = len(ed)

        lines.append(f"{gname}:\n")
        for dist_name in ['uniform', 'corpus']:
            weights = dists[dist_name]
            w = [weights[i % len(weights)] for i in range(n_edges)]
            eigs = spectrum(nv, ed, w)
            pattern = eigenvalue_multiplicity_pattern(eigs)
            mults = [m for _, m in pattern]
            lines.append(f"  {dist_name:>12}: {len(pattern)} distinct eigenvalues, "
                         f"multiplicities = {mults}\n")
        lines.append("\n")

    # Part D: Pairwise spectral distances
    lines.append("=" * 72 + "\n")
    lines.append("PART D: Pairwise spectral distances (corpus weights)\n")
    lines.append("=" * 72 + "\n\n")

    spectra_corpus = {}
    for gname, gdata in graphs.items():
        nv = gdata['n_vertices']
        ed = gdata['edges']
        n_edges = len(ed)
        w = [corpus_weights[i % len(corpus_weights)] for i in range(n_edges)]
        spectra_corpus[gname] = spectrum(nv, ed, w)

    gnames = list(graphs.keys())
    lines.append(f"{'':>30}")
    for gn in gnames:
        lines.append(f" {gn[:12]:>12}")
    lines.append("\n")
    for i, g1 in enumerate(gnames):
        lines.append(f"{g1:>30}")
        for j, g2 in enumerate(gnames):
            d = spectral_distance(spectra_corpus[g1], spectra_corpus[g2])
            lines.append(f" {d:>12.4f}")
        lines.append("\n")

    with open(os.path.join(RESULTS_DIR, "experiment_7_spectral_polytopes.txt"), "w") as f:
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
    experiment_5()
    experiment_6()
    experiment_7()

    elapsed = time.perf_counter() - t0
    print(f"\nAll 7 experiments complete in {elapsed:.1f}s")
    print(f"Results at: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
