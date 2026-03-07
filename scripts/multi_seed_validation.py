"""Multi-seed validation for Paper 2 key ratios.

Runs the direction-weighted Fiedler ratio computation across 100 seeds
to produce confidence intervals for the headline claims.

For deterministic distributions (uniform, power-law, corpus), the ratio
is identical across seeds — the CI collapses to a point. For random
distributions, the seed varies the weight draw, producing a distribution
of ratios.

Usage:
    python scripts/multi_seed_validation.py
"""

import numpy as np
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rhombic.corpus import direction_weights, weight_distributions
from rhombic.benchmark import matched_lattices, _fiedler_value


def apply_direction_weights(lattice, dw, G):
    """Apply direction-pair weights to a NetworkX graph."""
    dirs = lattice.edge_directions()
    for dir_idx, edge_indices in dirs.items():
        w = dw[dir_idx]
        for e_idx in edge_indices:
            u, v = lattice.edges[e_idx]
            G[u][v]['weight'] = w


def direction_weighted_fiedler_ratio(scale: int, seed: int) -> dict:
    """Compute direction-weighted Fiedler ratio for all distributions."""
    cubic, fcc = matched_lattices(scale)
    dists = weight_distributions(seed=seed)

    results = {}

    for dist_name in ['uniform', 'random', 'power_law', 'corpus']:
        vals = dists[dist_name]

        dw_fcc = direction_weights(vals, 6)
        dw_cubic = direction_weights(vals, 3)

        Gc = cubic.to_networkx()
        Gf = fcc.to_networkx()

        apply_direction_weights(cubic, dw_cubic, Gc)
        apply_direction_weights(fcc, dw_fcc, Gf)

        fied_c = _fiedler_value(Gc, weight='weight')
        fied_f = _fiedler_value(Gf, weight='weight')

        if fied_c and fied_f and fied_c > 0:
            results[dist_name] = fied_f / fied_c
        else:
            results[dist_name] = float('nan')

    return results


def permutation_control(scale: int, n_trials: int = 1000, seed: int = 42) -> dict:
    """Permutation control: shuffle corpus before sorted-bucketing."""
    rng = np.random.RandomState(seed)
    dists = weight_distributions(seed=42)
    corpus = dists['corpus']

    cubic, fcc = matched_lattices(scale)

    # Sorted (real) assignment
    dw_fcc = direction_weights(corpus, 6)
    dw_cubic = direction_weights(corpus, 3)

    Gc = cubic.to_networkx()
    Gf = fcc.to_networkx()
    apply_direction_weights(cubic, dw_cubic, Gc)
    apply_direction_weights(fcc, dw_fcc, Gf)

    sorted_ratio = _fiedler_value(Gf, weight='weight') / _fiedler_value(Gc, weight='weight')

    # Shuffled trials
    shuffled_ratios = []
    for i in range(n_trials):
        perm = rng.permutation(corpus).tolist()

        dw_fcc_shuf = direction_weights(perm, 6)
        dw_cubic_shuf = direction_weights(perm, 3)

        Gc2 = cubic.to_networkx()
        Gf2 = fcc.to_networkx()
        apply_direction_weights(cubic, dw_cubic_shuf, Gc2)
        apply_direction_weights(fcc, dw_fcc_shuf, Gf2)

        fc = _fiedler_value(Gc2, weight='weight')
        ff = _fiedler_value(Gf2, weight='weight')
        shuffled_ratios.append(ff / fc if fc and fc > 0 else float('nan'))

        if (i + 1) % 100 == 0:
            print(f"    Permutation trial {i + 1}/{n_trials}")

    shuffled_ratios = np.array(shuffled_ratios)
    valid = shuffled_ratios[~np.isnan(shuffled_ratios)]
    p_value = np.mean(valid >= sorted_ratio)

    return {
        'sorted_ratio': float(sorted_ratio),
        'shuffled_mean': float(np.mean(valid)),
        'shuffled_std': float(np.std(valid)),
        'shuffled_median': float(np.median(valid)),
        'shuffled_p025': float(np.percentile(valid, 2.5)),
        'shuffled_p975': float(np.percentile(valid, 97.5)),
        'p_value': float(p_value),
        'n_trials': n_trials,
        'n_valid': int(len(valid)),
    }


def main():
    output_dir = Path(__file__).resolve().parent.parent / 'results' / 'multi_seed'
    output_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = 100
    scales = [125, 1000]

    print(f"Multi-seed validation: {n_seeds} seeds x {len(scales)} scales")
    print("=" * 60)

    all_results = {}

    for scale in scales:
        print(f"\n--- Scale {scale} ---")
        t0 = time.time()

        seed_results = []
        for seed in range(n_seeds):
            if (seed + 1) % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (seed + 1) * (n_seeds - seed - 1)
                print(f"  Seed {seed + 1}/{n_seeds} "
                      f"({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)")

            result = direction_weighted_fiedler_ratio(scale, seed)
            seed_results.append(result)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        # Aggregate
        distributions = ['uniform', 'random', 'power_law', 'corpus']
        aggregated = {}
        for dist in distributions:
            values = [r[dist] for r in seed_results]
            arr = np.array(values)
            valid = arr[~np.isnan(arr)]
            aggregated[dist] = {
                'mean': float(np.mean(valid)),
                'std': float(np.std(valid)),
                'median': float(np.median(valid)),
                'ci_lower': float(np.percentile(valid, 2.5)),
                'ci_upper': float(np.percentile(valid, 97.5)),
                'min': float(np.min(valid)),
                'max': float(np.max(valid)),
                'n_seeds': n_seeds,
                'n_valid': int(len(valid)),
            }

            if aggregated[dist]['std'] < 1e-10:
                print(f"  {dist:12s}: {aggregated[dist]['mean']:.3f} "
                      f"(deterministic)")
            else:
                print(f"  {dist:12s}: {aggregated[dist]['mean']:.3f} "
                      f"+/- {aggregated[dist]['std']:.3f} "
                      f"[{aggregated[dist]['ci_lower']:.3f}, "
                      f"{aggregated[dist]['ci_upper']:.3f}]")

        all_results[f'scale_{scale}'] = aggregated

    # Permutation control
    print(f"\n--- Permutation control (scale 1000, 1000 trials) ---")
    t0 = time.time()
    perm_result = permutation_control(1000, n_trials=1000, seed=42)
    elapsed = time.time() - t0
    print(f"  Sorted ratio: {perm_result['sorted_ratio']:.3f}")
    print(f"  Shuffled: {perm_result['shuffled_mean']:.3f} "
          f"+/- {perm_result['shuffled_std']:.3f}")
    print(f"  95% CI: [{perm_result['shuffled_p025']:.3f}, "
          f"{perm_result['shuffled_p975']:.3f}]")
    print(f"  p-value: {perm_result['p_value']:.4f}")
    print(f"  Completed in {elapsed:.1f}s")

    all_results['permutation_control'] = perm_result

    # Save
    output_file = output_dir / 'multi_seed_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    for scale in scales:
        key = f'scale_{scale}'
        corpus = all_results[key]['corpus']
        uniform = all_results[key]['uniform']
        random_d = all_results[key]['random']
        print(f"\nScale {scale}:")
        print(f"  Uniform:   {uniform['mean']:.2f} (deterministic)")
        print(f"  Corpus:    {corpus['mean']:.2f} (deterministic)")
        print(f"  Random:    {random_d['mean']:.2f} "
              f"+/- {random_d['std']:.2f} "
              f"[95% CI: {random_d['ci_lower']:.2f}--"
              f"{random_d['ci_upper']:.2f}]")


if __name__ == '__main__':
    main()
