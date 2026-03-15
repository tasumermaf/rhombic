#!/usr/bin/env python3
"""
Block detection for arbitrary n-channel bridges.
Uses gap detection + union-find to identify block-diagonal structure.

For Paper 3 channel ablation analysis (H1-H5).
Pre-registered predictions in results/channel-ablation/PREDICTIONS_PREREGISTERED.json.
"""
import numpy as np
import os
import json
import sys
from typing import List, Tuple, Dict, Optional


def detect_blocks(bridge: np.ndarray, threshold_ratio: float = 10.0) -> dict:
    """Detect block-diagonal structure in an n x n bridge matrix.

    Uses gap detection on sorted off-diagonal coupling magnitudes,
    then union-find to cluster strongly coupled channels into blocks.

    Args:
        bridge: n x n bridge matrix
        threshold_ratio: minimum ratio between consecutive sorted couplings
                        to declare a gap (default: 10x)

    Returns:
        dict with block structure analysis
    """
    n = bridge.shape[0]

    # Extract off-diagonal couplings
    couplings = {}
    for i in range(n):
        for j in range(i + 1, n):
            couplings[(i, j)] = abs(bridge[i, j])

    if not couplings:
        return {
            'n_blocks': 1, 'block_sizes': [n], 'blocks': [list(range(n))],
            'is_block_diagonal': False, 'within_block_coupling': 0,
            'between_block_coupling': 0, 'ratio': 1.0
        }

    # Sort couplings by magnitude (descending)
    sorted_pairs = sorted(couplings.items(), key=lambda x: x[1], reverse=True)
    values = [v for _, v in sorted_pairs]

    if len(values) < 2:
        return {
            'n_blocks': 1, 'block_sizes': [n], 'blocks': [list(range(n))],
            'is_block_diagonal': False,
            'within_block_coupling': values[0] if values else 0,
            'between_block_coupling': 0, 'ratio': 1.0
        }

    # Union-find
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

    # Find the biggest gap (ratio between consecutive sorted values)
    max_gap_ratio = 0
    gap_idx = 0
    for i in range(len(values) - 1):
        if values[i + 1] > 1e-10:  # Avoid division by near-zero
            ratio = values[i] / values[i + 1]
            if ratio > max_gap_ratio:
                max_gap_ratio = ratio
                gap_idx = i

    # If no significant gap, everything is one block
    if max_gap_ratio < threshold_ratio:
        return {
            'n_blocks': 1, 'block_sizes': [n], 'blocks': [list(range(n))],
            'is_block_diagonal': False,
            'within_block_coupling': float(np.mean(values)),
            'between_block_coupling': float(np.mean(values)),
            'ratio': 1.0,
            'max_gap_ratio': float(max_gap_ratio)
        }

    # Union the strong pairs (above the gap)
    threshold = (values[gap_idx] + values[gap_idx + 1]) / 2
    for (i, j), v in sorted_pairs:
        if v > threshold:
            union(i, j)

    # Extract blocks
    blocks_dict = {}
    for i in range(n):
        root = find(i)
        if root not in blocks_dict:
            blocks_dict[root] = []
        blocks_dict[root].append(i)

    blocks = sorted(blocks_dict.values(), key=lambda x: x[0])
    block_sizes = sorted([len(b) for b in blocks], reverse=True)

    # Compute within-block and between-block coupling
    within = []
    between = []
    for (i, j), v in couplings.items():
        if find(i) == find(j):
            within.append(v)
        else:
            between.append(v)

    within_mean = np.mean(within) if within else 0
    between_mean = np.mean(between) if between else 0
    ratio = within_mean / between_mean if between_mean > 1e-10 else float('inf')

    return {
        'n_blocks': len(blocks),
        'block_sizes': block_sizes,
        'blocks': blocks,
        'is_block_diagonal': True,
        'within_block_coupling': float(within_mean),
        'between_block_coupling': float(between_mean),
        'ratio': float(ratio),
        'max_gap_ratio': float(max_gap_ratio),
        'threshold': float(threshold)
    }


def analyze_bridge_set(bridge_dir: str, n_channels: int,
                       step: str = 'final') -> dict:
    """Analyze all bridges from a training run at a given step."""
    prefix = f'bridge_{step}_' if step == 'final' else f'bridge_step{step}_'
    files = sorted([f for f in os.listdir(bridge_dir) if f.startswith(prefix)])

    if not files:
        return {'error': f'No files matching {prefix}*', 'n_bridges': 0}

    results = []
    for f in files:
        bridge = np.load(os.path.join(bridge_dir, f))
        r = detect_blocks(bridge)
        r['file'] = f
        results.append(r)

    # Aggregate block count distribution
    n_blocks_counts = {}
    for r in results:
        nb = r['n_blocks']
        n_blocks_counts[nb] = n_blocks_counts.get(nb, 0) + 1

    bd_count = sum(1 for r in results if r['is_block_diagonal'])

    # Modal block sizes (from BD bridges only)
    bd_bridges = [r for r in results if r['is_block_diagonal']]
    if bd_bridges:
        # Find most common block size pattern
        size_patterns = {}
        for r in bd_bridges:
            key = tuple(r['block_sizes'])
            size_patterns[key] = size_patterns.get(key, 0) + 1
        modal_pattern = max(size_patterns.items(), key=lambda x: x[1])
        modal_sizes = list(modal_pattern[0])
        modal_count = modal_pattern[1]
    else:
        modal_sizes = []
        modal_count = 0

    return {
        'n_bridges': len(results),
        'n_channels': n_channels,
        'block_count_distribution': n_blocks_counts,
        'bd_percentage': 100 * bd_count / len(results),
        'mean_ratio': float(np.mean([r['ratio'] for r in results
                                     if r['ratio'] != float('inf')])) if any(
            r['ratio'] != float('inf') for r in results) else 0,
        'mean_within': float(np.mean([r['within_block_coupling']
                                      for r in results])),
        'mean_between': float(np.mean([r['between_block_coupling']
                                       for r in results])),
        'modal_block_sizes': modal_sizes,
        'modal_count': modal_count,
        'per_bridge': results
    }


def validate_prediction(run_result: dict, prediction: dict) -> dict:
    """Check a run's results against pre-registered predictions."""
    n = prediction['n']
    expected_blocks = prediction.get('expected_blocks')
    expected_sizes = prediction.get('expected_block_sizes')

    actual_dist = run_result['block_count_distribution']
    modal_n_blocks = max(actual_dist.items(), key=lambda x: x[1])[0]

    verdict = 'UNKNOWN'
    detail = ''

    if expected_blocks is not None:
        if modal_n_blocks == expected_blocks:
            verdict = 'CONFIRMED'
            detail = (f'Modal block count = {modal_n_blocks} '
                      f'(predicted {expected_blocks})')
        else:
            verdict = 'REJECTED'
            detail = (f'Modal block count = {modal_n_blocks} '
                      f'(predicted {expected_blocks})')

    if expected_sizes and run_result['modal_block_sizes']:
        actual_sizes = run_result['modal_block_sizes']
        if actual_sizes == expected_sizes:
            detail += f', sizes match: {actual_sizes}'
        else:
            detail += f', sizes differ: actual={actual_sizes} vs pred={expected_sizes}'

    return {
        'verdict': verdict,
        'detail': detail,
        'modal_n_blocks': modal_n_blocks,
        'block_distribution': actual_dist,
        'bd_percentage': run_result['bd_percentage']
    }


def main():
    """Analyze all available channel ablation runs."""
    results_dir = os.path.join(os.path.dirname(__file__), '..',
                               'results', 'channel-ablation')

    # Load predictions
    pred_path = os.path.join(results_dir, 'PREDICTIONS_PREREGISTERED.json')
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            predictions = json.load(f)
    else:
        predictions = {}

    # Analyze each available run
    runs = {
        'H-ch3': 3, 'H-ch4': 4, 'H-ch6': 6, 'H-ch8': 8, 'H-ch12': 12
    }

    print("=" * 70)
    print("Channel Ablation Block Structure Analysis")
    print("=" * 70)

    for run_name, n_ch in runs.items():
        run_dir = os.path.join(results_dir, run_name)
        if not os.path.exists(run_dir):
            print(f"\n{run_name} (n={n_ch}): NOT AVAILABLE")
            continue

        # Check for bridge files
        has_final = any(f.startswith('bridge_final_')
                        for f in os.listdir(run_dir))
        if not has_final:
            print(f"\n{run_name} (n={n_ch}): NO FINAL BRIDGES")
            continue

        print(f"\n{'=' * 50}")
        print(f"{run_name} (n={n_ch})")
        print(f"{'=' * 50}")

        result = analyze_bridge_set(run_dir, n_ch)
        print(f"  Bridges analyzed: {result['n_bridges']}")
        print(f"  Block count distribution: {result['block_count_distribution']}")
        print(f"  BD percentage: {result['bd_percentage']:.1f}%")
        print(f"  Modal block sizes: {result['modal_block_sizes']}")
        print(f"  Mean within-block coupling: {result['mean_within']:.6f}")
        print(f"  Mean between-block coupling: {result['mean_between']:.6f}")
        if result['mean_ratio'] > 0:
            print(f"  Within/between ratio: {result['mean_ratio']:.1f}")

        # Validate against prediction
        pred_key = f"H{n_ch}" if f"H{n_ch}" in predictions else run_name
        if pred_key not in predictions:
            # Try alternate key formats
            for k in predictions:
                if predictions[k].get('n') == n_ch:
                    pred_key = k
                    break

        if pred_key in predictions:
            v = validate_prediction(result, predictions[pred_key])
            print(f"\n  PREDICTION: {v['verdict']}")
            print(f"  {v['detail']}")

        # Per-layer breakdown for block-diagonal runs
        if result['bd_percentage'] > 50:
            print(f"\n  Per-projection breakdown:")
            proj_blocks = {}
            for r in result['per_bridge']:
                # Extract projection type
                for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if proj in r['file']:
                        if proj not in proj_blocks:
                            proj_blocks[proj] = []
                        proj_blocks[proj].append(r['n_blocks'])
                        break

            for proj, counts in sorted(proj_blocks.items()):
                mean_b = np.mean(counts)
                print(f"    {proj}: mean blocks = {mean_b:.2f}, "
                      f"range = [{min(counts)}, {max(counts)}]")


if __name__ == '__main__':
    main()
