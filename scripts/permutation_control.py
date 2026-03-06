"""
Permutation control for direction-pair weighting (Paper 2 remediation Step 0).

Tests whether the Fiedler amplification comes from ALIGNMENT (sorted weights
concentrated into coherent directional channels) or merely from BIN COUNT
(6 bins vs 3 bins regardless of assignment order).

Protocol:
  - Sorted baseline: sort 24 corpus values, bucket into 6 groups of 4 (FCC)
    and 3 groups of 8 (SC), average each group, compute Fiedler ratio.
  - Shuffled trials (1000x): randomly permute 24 corpus values, then bucket
    into groups IN SHUFFLED ORDER (no sort), average, compute Fiedler ratio.

If amplification is driven by alignment, sorted >> shuffled.
If driven only by bin count, sorted ~ shuffled.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rhombic.corpus import edge_values, weight_distributions, direction_weights
from rhombic.benchmark import matched_lattices, _fiedler_value
from rhombic.spatial import build_adjacency

SEED = 42
N_TRIALS = 1000


def shuffled_direction_weights(values: list[float], n_directions: int,
                                rng: np.random.Generator) -> list[float]:
    """Like direction_weights but permutes BEFORE bucketing (no sort)."""
    arr = np.array(values)
    rng.shuffle(arr)
    per_group = len(arr) // n_directions
    result = []
    for i in range(n_directions):
        start = i * per_group
        end = start + per_group if i < n_directions - 1 else len(arr)
        result.append(float(arr[start:end].mean()))
    return result


def compute_fiedler_ratio(cubic, fcc, dw_cubic, dw_fcc):
    """Compute FCC/SC Fiedler ratio under direction weights."""
    dirs_c = cubic.edge_directions()
    dirs_f = fcc.edge_directions()

    Gc = cubic.to_networkx()
    Gf = fcc.to_networkx()

    for dir_idx, edge_indices in dirs_c.items():
        w = dw_cubic[dir_idx]
        for e_idx in edge_indices:
            u, v = cubic.edges[e_idx]
            Gc[u][v]['weight'] = w

    for dir_idx, edge_indices in dirs_f.items():
        w = dw_fcc[dir_idx]
        for e_idx in edge_indices:
            u, v = fcc.edges[e_idx]
            Gf[u][v]['weight'] = w

    fied_c = _fiedler_value(Gc, weight='weight')
    fied_f = _fiedler_value(Gf, weight='weight')

    if fied_c and fied_f and fied_c > 0:
        return fied_f / fied_c
    return None


def main():
    t0 = time.perf_counter()

    print("Permutation Control: Direction-Pair Weighting")
    print(f"Seed: {SEED}, Trials: {N_TRIALS}")
    print()

    # Build lattices at scale 1000
    print("Building lattices at scale 1000...")
    cubic, fcc = matched_lattices(1000)

    # Get normalized corpus weights
    dists = weight_distributions(seed=SEED)
    corpus_norm = dists['corpus']

    # Sorted baseline (this is the existing method)
    print("Computing sorted baseline...")
    dw_fcc_sorted = direction_weights(corpus_norm, 6)
    dw_cubic_sorted = direction_weights(corpus_norm, 3)
    sorted_ratio = compute_fiedler_ratio(cubic, fcc, dw_cubic_sorted, dw_fcc_sorted)
    print(f"  Sorted Fiedler ratio: {sorted_ratio:.4f}")

    # Shuffled trials
    print(f"Running {N_TRIALS} shuffled trials...")
    rng = np.random.default_rng(SEED)
    shuffled_ratios = []

    for i in range(N_TRIALS):
        dw_fcc_shuf = shuffled_direction_weights(corpus_norm, 6, rng)
        dw_cubic_shuf = shuffled_direction_weights(corpus_norm, 3, rng)
        ratio = compute_fiedler_ratio(cubic, fcc, dw_cubic_shuf, dw_fcc_shuf)
        if ratio is not None:
            shuffled_ratios.append(ratio)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{N_TRIALS}...")

    shuffled_arr = np.array(shuffled_ratios)
    mean_shuf = float(shuffled_arr.mean())
    std_shuf = float(shuffled_arr.std())
    p_value = float(np.mean(shuffled_arr >= sorted_ratio))

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("RESULTS: Direction-Pair Permutation Control")
    print("=" * 60)
    print(f"  Sorted Fiedler ratio:   {sorted_ratio:.2f}")
    print(f"  Shuffled mean +/- std:  {mean_shuf:.2f} +/- {std_shuf:.2f}")
    print(f"  Shuffled min/max:       {shuffled_arr.min():.2f} / {shuffled_arr.max():.2f}")
    print(f"  p-value (shuffled >= sorted): {p_value}")
    if p_value == 0:
        print(f"  p-value bound: <= {1/N_TRIALS}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()
    print("For paper insertion:")
    print(f"  mean shuffled Fiedler ratio was {mean_shuf:.2f} \\pm {std_shuf:.2f},")
    print(f"  compared to {sorted_ratio:.2f} for the sorted assignment")
    if p_value == 0:
        print(f"  (p \\leq {1/N_TRIALS})")
    else:
        print(f"  (p = {p_value})")


if __name__ == "__main__":
    main()
