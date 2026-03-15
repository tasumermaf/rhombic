#!/usr/bin/env python3
"""Analyze block structure for arbitrary n-channel bridges.

For n=6, detects the known 3×2×2 co-planar block structure.
For other n, discovers whatever block structure the Steersman enforces.

Works with flat-file bridge checkpoints (bridge_stepN_*.npy or bridge_final_*.npy).
"""
import os
import sys
import numpy as np
import json
import re
from itertools import combinations


def detect_blocks(B, threshold_ratio=5.0):
    """Detect block structure in an n×n bridge matrix.

    Returns a list of detected blocks (channel pairs with coupling
    significantly above the cross-block average).

    Args:
        B: n×n bridge matrix
        threshold_ratio: minimum ratio of within-block to cross-block
                        coupling to consider a pair as "blocked"

    Returns:
        dict with:
          - blocks: list of (i,j) pairs that form blocks
          - block_mean: mean coupling within detected blocks
          - cross_mean: mean coupling between blocks
          - ratio: block_mean / cross_mean
          - n_blocks: number of detected blocks
          - partition: which block each channel belongs to (-1 if unassigned)
    """
    n = B.shape[0]

    # Collect all off-diagonal magnitudes
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append((abs(B[i, j]), (i, j)))

    if not off_diag:
        return None

    off_diag.sort(reverse=True)
    mags = [m for m, _ in off_diag]

    # Find natural gap: largest relative drop in sorted magnitudes
    # This separates "strong" (within-block) from "weak" (cross-block) pairs
    best_gap_idx = 0
    best_gap_ratio = 1.0

    for k in range(1, len(mags)):
        if mags[k] > 1e-12:
            gap_ratio = mags[k - 1] / mags[k]
            if gap_ratio > best_gap_ratio:
                best_gap_ratio = gap_ratio
                best_gap_idx = k

    if best_gap_ratio < threshold_ratio:
        # No clear block structure
        return {
            'blocks': [],
            'block_mean': float(np.mean(mags)),
            'cross_mean': float(np.mean(mags)),
            'ratio': 1.0,
            'n_blocks': 0,
            'partition': [-1] * n,
            'gap_ratio': float(best_gap_ratio),
        }

    # Strong pairs are above the gap
    strong_pairs = [p for m, p in off_diag[:best_gap_idx]]
    weak_pairs = [p for m, p in off_diag[best_gap_idx:]]

    strong_mags = [m for m, _ in off_diag[:best_gap_idx]]
    weak_mags = [m for m, _ in off_diag[best_gap_idx:]]

    # Build partition from strong pairs using union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in strong_pairs:
        union(i, j)

    # Extract partition
    blocks = {}
    for ch in range(n):
        root = find(ch)
        if root not in blocks:
            blocks[root] = []
        blocks[root].append(ch)

    partition = [0] * n
    for block_id, (root, members) in enumerate(sorted(blocks.items())):
        for ch in members:
            partition[ch] = block_id

    block_mean = float(np.mean(strong_mags)) if strong_mags else 0
    cross_mean = float(np.mean(weak_mags)) if weak_mags else 0
    ratio = block_mean / max(cross_mean, 1e-10)

    return {
        'blocks': strong_pairs,
        'block_mean': block_mean,
        'cross_mean': cross_mean,
        'ratio': float(ratio),
        'n_blocks': len(blocks),
        'partition': partition,
        'block_sizes': [len(m) for m in sorted(blocks.values())],
        'gap_ratio': float(best_gap_ratio),
    }


def analyze_experiment(base_dir, pattern="bridge_final_*.npy"):
    """Analyze all bridges matching pattern in a directory."""
    files = [f for f in os.listdir(base_dir)
             if re.match(pattern.replace('*', '.*'), f)]

    if not files:
        # Try step bridges
        all_files = os.listdir(base_dir)
        steps = set()
        for f in all_files:
            m = re.match(r'bridge_step(\d+)_', f)
            if m:
                steps.add(int(m.group(1)))
        if steps:
            last_step = max(steps)
            prefix = f'bridge_step{last_step}_'
            files = [f for f in all_files if f.startswith(prefix)]
            print(f"  Using step {last_step} bridges ({len(files)} files)")

    if not files:
        print(f"  No bridge files found in {base_dir}")
        return None

    n = None
    results = []
    partition_counts = {}

    for bf in files:
        B = np.load(os.path.join(base_dir, bf))
        if n is None:
            n = B.shape[0]
            print(f"  Bridge size: {n}×{n}")

        result = detect_blocks(B)
        if result is None:
            continue
        results.append(result)

        # Track partition patterns
        part_key = tuple(result['partition'])
        partition_counts[part_key] = partition_counts.get(part_key, 0) + 1

    if not results:
        return None

    # Summary statistics
    avg_ratio = np.mean([r['ratio'] for r in results])
    avg_n_blocks = np.mean([r['n_blocks'] for r in results])
    block_pct = 100 * sum(1 for r in results if r['n_blocks'] > 0) / len(results)

    # Most common partition
    most_common_part = max(partition_counts, key=partition_counts.get)
    most_common_count = partition_counts[most_common_part]

    # Block sizes of most common partition
    if any(r['partition'] == list(most_common_part) for r in results):
        rep = next(r for r in results if r['partition'] == list(most_common_part))
        block_sizes = rep.get('block_sizes', [])
    else:
        block_sizes = []

    return {
        'n': n,
        'bridges': len(results),
        'avg_ratio': float(avg_ratio),
        'avg_n_blocks': float(avg_n_blocks),
        'block_pct': float(block_pct),
        'most_common_partition': list(most_common_part),
        'partition_frequency': float(most_common_count / len(results)),
        'block_sizes': block_sizes,
        'unique_partitions': len(partition_counts),
    }


def main():
    ablation_dir = "results/channel-ablation"

    runs = {
        'H-ch3': 3,
        'H-ch4': 4,
        'H-ch6': 6,
        'H-ch8': 8,
        'H-ch12': 12,
    }

    print("=" * 70)
    print("  Block Structure Analysis — Channel Ablation")
    print("=" * 70)

    all_results = {}

    for run_name, n_channels in sorted(runs.items(), key=lambda x: x[1]):
        run_dir = os.path.join(ablation_dir, run_name)
        if not os.path.exists(run_dir):
            print(f"\n{run_name} (n={n_channels}): NOT FOUND")
            continue

        print(f"\n{run_name} (n={n_channels}):")
        result = analyze_experiment(run_dir)
        if result is None:
            continue

        all_results[run_name] = result

        print(f"  Bridges: {result['bridges']}")
        print(f"  Block structure: {result['block_pct']:.0f}% of bridges")
        print(f"  Average ratio: {result['avg_ratio']:.1f}:1")
        print(f"  Average blocks: {result['avg_n_blocks']:.1f}")
        print(f"  Block sizes: {result['block_sizes']}")
        print(f"  Most common partition: {result['most_common_partition']} "
              f"({result['partition_frequency']:.0%})")
        print(f"  Unique partitions: {result['unique_partitions']}")

    # Summary table
    if all_results:
        print(f"\n{'=' * 70}")
        print(f"  Summary")
        print(f"{'=' * 70}")
        print(f"{'Run':>8} | {'n':>3} | {'Blocks':>6} | {'Ratio':>10} | {'BD%':>5} | {'Block Sizes':>15}")
        print(f"{'-'*8}-+-{'-'*3}-+-{'-'*6}-+-{'-'*10}-+-{'-'*5}-+-{'-'*15}")

        for run_name in sorted(all_results, key=lambda x: runs.get(x, 0)):
            r = all_results[run_name]
            n = runs.get(run_name, '?')
            blocks = f"{r['avg_n_blocks']:.1f}"
            ratio = f"{r['avg_ratio']:.1f}:1"
            bd_pct = f"{r['block_pct']:.0f}%"
            sizes = str(r['block_sizes'])
            print(f"{run_name:>8} | {n:>3} | {blocks:>6} | {ratio:>10} | {bd_pct:>5} | {sizes:>15}")

    # Save results
    outfile = os.path.join(ablation_dir, "block_structure_analysis.json")
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
