"""Tinker pilot — robustness and mechanism checks on the signal result.

Three questions the headline SIGNAL verdict raises, each answered from the
downloaded adapters (falco env; no network, no spend):

1. MECHANISM (why does RAW fail?). In raw space the SMALL distances are the
   same-SEED pairs regardless of task. Tinker's ``seed`` argument sets the LoRA
   INITIALIZATION, ``lora_A`` is randomly initialized while ``lora_B`` starts at
   zero, and ||A|| >> ||B|| after only 1M tokens. So a feature vector built from
   the raw factors is dominated by initialization, not by learning. Quantified
   here: per-seed cosine similarity of the lora_A blocks, and the ||B||/||A||
   ratio.

2. INFERENCE. With 3 within-task and 12 cross-task pairs, a perfect split is
   the event "the 3 within-task pairs are exactly the 3 smallest of 15", whose
   probability under an exchangeable null is 1 / C(15,3) = 1/455 = 0.0022.
   Reported as an exact combinatorial p-value; also computed by exhaustive
   enumeration over all task-label relabelings as a stricter permutation null.

3. ROBUSTNESS. The canonical margin is thin, so the separation is re-checked
   under (a) the 'sigma' feature variant (log1p singular values only — fully
   rotation-invariant, no random probes at all) and (b) several probe seeds.
   A result that only holds for one probe seed would be an artifact.

Usage
-----
    python scripts/tinker_pilot_signal_checks.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asset1_canonicalize import canonicalize_module, feature_vector  # noqa: E402
from tinker_pilot_signal import (cosine_distances, effective_factors_bridgeless,
                                 evaluate, load_bridgeless_adapter,
                                 scaling_from_config)


def canonical_dict(modules, scaling):
    out = {}
    for name in sorted(modules):
        B_eff, A_eff = effective_factors_bridgeless(modules[name], scaling)
        out[name] = canonicalize_module(B_eff, A_eff)
    return out


def exact_split_p(n_within: int, n_total: int) -> float:
    """P(the n_within designated pairs are exactly the n_within smallest)."""
    return 1.0 / math.comb(n_total, n_within)


def relabeling_p(dists: dict, runs: list[str], observed_margin: float) -> dict:
    """Stricter null: enumerate every way of partitioning the 6 runs into 3
    labelled pairs, and ask how often a partition separates at least as well.

    This holds the distance matrix fixed and permutes only which runs are
    called same-task, so it accounts for the structure of the geometry rather
    than assuming exchangeable pair labels.
    """
    def perfect_matchings(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for i in range(len(rest)):
            pair = (first, rest[i])
            for tail in perfect_matchings(rest[:i] + rest[i + 1:]):
                yield [pair] + tail

    n_ge = 0
    total = 0
    margins = []
    for matching in perfect_matchings(runs):
        within = [dists[tuple(sorted(p))] for p in matching]
        cross = [d for k, d in dists.items()
                 if all(set(k) != set(p) for p in matching)]
        margin = min(cross) - max(within)
        margins.append(margin)
        total += 1
        if margin >= observed_margin:
            n_ge += 1
    return {"n_matchings": total, "n_at_least_as_separating": n_ge,
            "p_value": n_ge / total,
            "best_possible_margin": max(margins)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path,
                    default=REPO_ROOT / "results" / "tinker-pilot")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dirs = sorted(d for d in args.bank.iterdir()
                      if d.is_dir() and (d / "adapter_config.json").exists())
    runs = [d.name for d in run_dirs]
    task_of = {r: r.rsplit("_", 1)[0] for r in runs}
    seed_of = {r: r.rsplit("_", 1)[1] for r in runs}
    print(f"runs: {runs}")

    mods, scal, canon = {}, {}, {}
    a_blocks, ratios = {}, {}
    for d in run_dirs:
        m, cfg = load_bridgeless_adapter(d)
        s, _ = scaling_from_config(cfg, 32)
        mods[d.name], scal[d.name] = m, s
        canon[d.name] = canonical_dict(m, s)
        a_blocks[d.name] = torch.cat(
            [m[k]["lora_A"].reshape(-1).double() for k in sorted(m)]).numpy()
        an = float(torch.cat([m[k]["lora_A"].reshape(-1) for k in sorted(m)]).double().norm())
        bn = float(torch.cat([m[k]["lora_B"].reshape(-1) for k in sorted(m)]).double().norm())
        ratios[d.name] = {"norm_A": an, "norm_B": bn, "ratio_B_over_A": bn / an}
        print(f"  {d.name}: ||A||={an:.4f} ||B||={bn:.4f} "
              f"||B||/||A||={bn / an:.5f}")

    # ── 1. Mechanism: is raw space dominated by the init seed? ──────
    print("\n=== MECHANISM: cosine similarity of the lora_A (init) blocks ===")
    a_cos = {}
    for a, b in itertools.combinations(runs, 2):
        xa, xb = a_blocks[a], a_blocks[b]
        cos = float(np.dot(xa, xb) / (np.linalg.norm(xa) * np.linalg.norm(xb)))
        same_seed = seed_of[a] == seed_of[b]
        a_cos[f"{a}|{b}"] = {"cos": cos, "same_seed": same_seed,
                             "same_task": task_of[a] == task_of[b]}
        print(f"  {a:10s} {b:10s} cos(A,A)={cos:+.6f}  "
              f"same_seed={same_seed}  same_task={task_of[a] == task_of[b]}")

    same_seed_cos = [v["cos"] for v in a_cos.values() if v["same_seed"]]
    diff_seed_cos = [v["cos"] for v in a_cos.values() if not v["same_seed"]]
    print(f"\n  same-seed  cos(A,A): min {min(same_seed_cos):+.6f} "
          f"max {max(same_seed_cos):+.6f}  (n={len(same_seed_cos)})")
    print(f"  diff-seed  cos(A,A): min {min(diff_seed_cos):+.6f} "
          f"max {max(diff_seed_cos):+.6f}  (n={len(diff_seed_cos)})")

    # ── 2 & 3. Inference and robustness of the canonical separation ──
    variants = [("full", 0), ("full", 1), ("full", 7), ("sigma", 0)]
    rob = {}
    print("\n=== ROBUSTNESS of the canonical separation ===")
    for variant, pseed in variants:
        feats = {r: feature_vector(canon[r], variant=variant, proj_dim=16,
                                  proj_seed=pseed).numpy() for r in runs}
        dists = cosine_distances(feats)
        ev = evaluate(dists, task_of)
        key = f"{variant}/proj_seed={pseed}"
        rel = relabeling_p(dists, runs, ev["separation_margin"])
        rob[key] = {"dim": int(next(iter(feats.values())).size),
                    "max_within": ev["max_within_task"],
                    "min_cross": ev["min_cross_task"],
                    "margin": ev["separation_margin"],
                    "separated": ev["separated"],
                    "nn_task_accuracy": ev["nn_task_accuracy"],
                    "exact_split_p": exact_split_p(3, 15),
                    "relabeling_null": rel}
        print(f"  {key:22s} dim {feats[runs[0]].size:>7,}  "
              f"max_within {ev['max_within_task']:.6f}  "
              f"min_cross {ev['min_cross_task']:.6f}  "
              f"margin {ev['separation_margin']:+.6f}  "
              f"sep {ev['separated']}  1NN {ev['nn_task_accuracy']:.3f}  "
              f"relabel_p {rel['p_value']:.4f}")

    payload = {
        "runs": runs,
        "task_of": task_of, "seed_of": seed_of,
        "norms": ratios,
        "mechanism_lora_A_cosine": a_cos,
        "mechanism_summary": {
            "same_seed_cos_A_min": min(same_seed_cos),
            "same_seed_cos_A_max": max(same_seed_cos),
            "diff_seed_cos_A_min": min(diff_seed_cos),
            "diff_seed_cos_A_max": max(diff_seed_cos),
            "interpretation": (
                "Tinker's seed sets the LoRA INITIALIZATION. lora_A is random "
                "and frozen-large while lora_B starts at zero, so after 1M "
                "tokens ||B||/||A|| is small and any feature built from the raw "
                "factors is dominated by initialization identity, not task. "
                "This is why RAW groups by seed and fails to separate task."),
        },
        "exact_split_p_definition": (
            "P(the 3 within-task pairs are exactly the 3 smallest of 15 "
            "distances) = 1/C(15,3) = 1/455 = 0.002198 under exchangeable "
            "pair labels."),
        "robustness": rob,
    }
    out = args.out or (args.bank / "signal_checks.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
