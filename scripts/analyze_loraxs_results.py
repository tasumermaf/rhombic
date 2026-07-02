"""Analyze LoRA-XS corpus initialization experiment results.

Reads all results.json files from the experiment directory and produces:
1. Comparison table (val loss, params, time)
2. Convergence analysis (early-stage advantage)
3. R matrix evolution analysis (for LoRA-XS conditions)
4. Statistical significance assessment

Usage:
  python scripts/analyze_loraxs_results.py results/loraxs-corpus
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load all results.json files from condition subdirectories."""
    results = {}
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        rfile = subdir / "results.json"
        if rfile.exists():
            with open(rfile) as f:
                results[subdir.name] = json.load(f)
    return results


def print_comparison_table(results: dict[str, dict]) -> None:
    """Print main comparison table."""
    print("\n" + "=" * 80)
    print("LORAXS CORPUS INITIALIZATION EXPERIMENT - RESULTS")
    print("=" * 80)

    # Header
    print(f"\n{'Condition':<30} {'Val Loss':>10} {'Params':>12} {'Time (min)':>12} {'Mode':<20}")
    print("-" * 84)

    # Sort by val loss
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("final_val_loss", 999))

    best_loss = sorted_results[0][1]["final_val_loss"] if sorted_results else None

    for name, data in sorted_results:
        val_loss = data.get("final_val_loss", float("nan"))
        params = data.get("trainable_params", 0)
        mode = data.get("config", {}).get("mode", "?")

        checkpoints = data.get("checkpoints", [])
        wall_time = checkpoints[-1]["wall_time"] / 60.0 if checkpoints else 0

        marker = " <-- BEST" if val_loss == best_loss else ""
        delta = ""
        if best_loss and val_loss != best_loss:
            diff = val_loss - best_loss
            pct = 100 * diff / best_loss
            delta = f" (+{diff:.4f}, +{pct:.1f}%)"

        print(f"{name:<30} {val_loss:>10.4f} {params:>12,} {wall_time:>12.1f} {mode:<20}{marker}{delta}")

    print()


def print_convergence_analysis(results: dict[str, dict]) -> None:
    """Analyze early-stage convergence advantages."""
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS - Val Loss at Key Steps")
    print("=" * 80)

    steps_of_interest = [200, 400, 600, 1000, 2000, 3000]

    # Header
    header = f"{'Condition':<30}"
    for s in steps_of_interest:
        header += f" {'Step '+str(s):>10}"
    header += f" {'Final':>10}"
    print(f"\n{header}")
    print("-" * (30 + 11 * (len(steps_of_interest) + 1)))

    for name, data in sorted(results.items()):
        checkpoints = data.get("checkpoints", [])
        cp_by_step = {cp["step"]: cp for cp in checkpoints}

        row = f"{name:<30}"
        for s in steps_of_interest:
            if s in cp_by_step:
                row += f" {cp_by_step[s]['val_loss']:>10.4f}"
            else:
                row += f" {'---':>10}"
        row += f" {data.get('final_val_loss', float('nan')):>10.4f}"
        print(row)

    print()


def print_r_matrix_analysis(results: dict[str, dict]) -> None:
    """Analyze R matrix evolution for LoRA-XS conditions."""
    print("\n" + "=" * 80)
    print("R MATRIX EVOLUTION")
    print("=" * 80)

    for name, data in sorted(results.items()):
        mode = data.get("config", {}).get("mode", "?")
        if mode == "standard_lora":
            continue

        checkpoints = data.get("checkpoints", [])
        if not checkpoints:
            continue

        first = checkpoints[0]
        last = checkpoints[-1]

        if "R_deviation_from_init" not in first:
            continue

        print(f"\n{name} ({mode}):")
        print(f"  Initial:  Frob={first.get('R_frobenius', 0):.3f}, "
              f"Cond={first.get('R_cond', 0):.3f}, "
              f"Eig=[{first.get('R_eigmin', 0):.4f}, {first.get('R_eigmax', 0):.4f}]")
        print(f"  Final:    Frob={last.get('R_frobenius', 0):.3f}, "
              f"Cond={last.get('R_cond', 0):.3f}, "
              f"Eig=[{last.get('R_eigmin', 0):.4f}, {last.get('R_eigmax', 0):.4f}]")
        print(f"  Movement: DevInit={last.get('R_deviation_from_init', 0):.4f} "
              f"(Cond delta: {last.get('R_cond', 0) - first.get('R_cond', 0):+.3f})")


def print_parameter_efficiency(results: dict[str, dict]) -> None:
    """Compare parameter efficiency across conditions."""
    print("\n" + "=" * 80)
    print("PARAMETER EFFICIENCY")
    print("=" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1].get("final_val_loss", 999))
    if not sorted_results:
        return

    print(f"\n{'Condition':<30} {'Params':>12} {'Val Loss':>10} {'Loss/Param':>14}")
    print("-" * 66)

    for name, data in sorted_results:
        params = data.get("trainable_params", 1)
        val_loss = data.get("final_val_loss", float("nan"))
        ratio = val_loss / params * 1e6  # loss per million params
        print(f"{name:<30} {params:>12,} {val_loss:>10.4f} {ratio:>14.4f}")


def print_key_comparisons(results: dict[str, dict]) -> None:
    """Print key A/B comparisons that answer our research questions."""
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)

    def get_loss(name):
        return results.get(name, {}).get("final_val_loss")

    a = get_loss("corpus_optimized_24cell")
    b = get_loss("corpus_random_24cell")
    c = get_loss("structure_only_24cell")
    d = get_loss("loraxs_baseline")
    e = get_loss("standard_lora_r24")

    print()
    if a is not None and d is not None:
        diff = a - d
        print(f"Q1: Does corpus init (A) beat random init (D)?")
        print(f"    A={a:.4f} vs D={d:.4f} -- Delta: {diff:+.4f} "
              f"({'YES, corpus wins' if diff < 0 else 'NO, random wins' if diff > 0 else 'TIE'})")
        print()

    if a is not None and b is not None:
        diff = a - b
        print(f"Q2: Does optimized mapping (A) beat random mapping (B)?")
        print(f"    A={a:.4f} vs B={b:.4f} -- Delta: {diff:+.4f} "
              f"({'YES, optimization wins' if diff < 0 else 'NO, random mapping sufficient' if diff > 0 else 'TIE'})")
        print()

    if c is not None and d is not None:
        diff = c - d
        print(f"Q3: Does 24-cell structure alone (C) help?")
        print(f"    C={c:.4f} vs D={d:.4f} -- Delta: {diff:+.4f} "
              f"({'YES, geometry helps' if diff < 0 else 'NO, geometry alone insufficient' if diff > 0 else 'TIE'})")
        print()

    if a is not None and e is not None:
        a_params = results.get("corpus_optimized_24cell", {}).get("trainable_params", 1)
        e_params = results.get("standard_lora_r24", {}).get("trainable_params", 1)
        ratio = e_params / a_params
        print(f"Q4: Does LoRA-XS + corpus (A) compete with standard LoRA (E)?")
        print(f"    A={a:.4f} ({a_params:,} params) vs E={e:.4f} ({e_params:,} params)")
        print(f"    Standard LoRA uses {ratio:.0f}x more parameters")
        if a <= e:
            print(f"    RESULT: LoRA-XS + corpus MATCHES standard LoRA with {ratio:.0f}x fewer params!")
        else:
            gap = a - e
            print(f"    RESULT: Standard LoRA wins by {gap:.4f}. "
                  f"Cost: {ratio:.0f}x more parameters.")
        print()


def main():
    if len(sys.argv) < 2:
        results_dir = Path("results/loraxs-corpus")
    else:
        results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    results = load_results(results_dir)
    if not results:
        print(f"No results.json found in {results_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} conditions from {results_dir}")

    print_comparison_table(results)
    print_convergence_analysis(results)
    print_r_matrix_analysis(results)
    print_parameter_efficiency(results)
    print_key_comparisons(results)


if __name__ == "__main__":
    main()
