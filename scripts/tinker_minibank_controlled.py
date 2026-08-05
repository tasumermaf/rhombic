"""E-T4 mini-bank — the controlled contrast the crossed seed design was for.

Plain LOO 1-NN over this bank is CONFOUNDED and must not be read as
"raw carries task identity". The design has 3 data seeds x 3 init seeds per
task, so every adapter sits in a (task, init) cell containing 3 adapters that
share BOTH labels. An adapter's nearest neighbour is therefore almost always
a cell-mate, and a cell-mate is simultaneously same-task and same-init — which
is exactly why raw scores 1.000 on task AND 1.000 on init at n=54. The plain
metric cannot attribute the success to either label.

The decisive test restricts the neighbourhood so the two labels come apart:

  CROSS-INIT TASK   for each adapter, its nearest neighbour AMONG ADAPTERS
                    WITH A DIFFERENT INIT SEED. If task identity survives
                    here, the representation carries task independently of
                    initialization. If it collapses, the apparent task
                    signal was init structure.
  CROSS-TASK INIT   the mirror: nearest neighbour among adapters of a
                    DIFFERENT TASK, scored on init identity. Measures how
                    much initialization identity survives across tasks.
  WITHIN-INIT TASK  nearest neighbour restricted to the SAME init seed —
                    task identification with initialization held fixed.

Chance levels are computed for each restricted neighbourhood, not assumed.

Reads the distances already written by tinker_minibank_signal.py, so it
recomputes no features and cannot disagree with the headline numbers.

RUNS UNDER THE ``falco`` CONDA ENV.

Usage
-----
    python scripts/tinker_minibank_controlled.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BANK = REPO_ROOT / "results" / "tinker-minibank"


def restricted_nn(D: np.ndarray, allowed: np.ndarray, y: list) -> tuple:
    """1-NN accuracy where `allowed[i, j]` says j may be i's neighbour."""
    n = len(y)
    correct, used, detail = 0, 0, []
    for i in range(n):
        cand = np.where(allowed[i])[0]
        if cand.size == 0:
            continue
        j = cand[np.argmin(D[i, cand])]
        ok = y[j] == y[i]
        correct += int(ok)
        used += 1
        detail.append({"run": i, "nn": int(j), "same": bool(ok)})
    return (correct / used if used else float("nan")), used, detail


def chance_for(allowed: np.ndarray, y: list) -> float:
    """P(a uniformly random ALLOWED neighbour shares the label)."""
    tot, hit = 0, 0
    for i in range(len(y)):
        cand = np.where(allowed[i])[0]
        tot += cand.size
        hit += sum(1 for j in cand if y[j] == y[i])
    return hit / tot if tot else float("nan")


def perm_p(D, allowed, y, obs, n_perm, rng) -> float:
    """Label-permutation null with the geometry and the mask held fixed."""
    y_arr = np.asarray(y, dtype=object)
    ge = 0
    for _ in range(n_perm):
        perm = list(y_arr[rng.permutation(len(y_arr))])
        acc, _, _ = restricted_nn(D, allowed, perm)
        ge += int(acc >= obs)
    return (ge + 1) / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Controlled cross-init contrast")
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sig = json.loads((args.bank / "signal_results.json").read_text(encoding="utf-8"))
    dists = json.loads((args.bank / "pairwise_distances.json").read_text(encoding="utf-8"))
    runs = sig["runs"]
    labels = sig["labels"]
    idx = {r: i for i, r in enumerate(runs)}
    n = len(runs)

    task = [labels[r]["task"] for r in runs]
    init = [labels[r]["init_seed"] for r in runs]

    diff_init = np.array([[init[i] != init[j] and i != j for j in range(n)]
                          for i in range(n)])
    same_init = np.array([[init[i] == init[j] and i != j for j in range(n)]
                          for i in range(n)])
    diff_task = np.array([[task[i] != task[j] and i != j for j in range(n)]
                          for i in range(n)])

    out = {"n_adapters": n, "n_perm": args.n_perm, "spaces": {}}
    print(f"{'space':18s} {'contrast':18s} {'acc':>7s} {'chance':>7s} {'p':>10s}")
    for space in ("raw", "canonical_full", "canonical_sigma"):
        if space not in dists:
            continue
        D = np.full((n, n), np.inf)
        for key, v in dists[space].items():
            a, b = key.split("|")
            D[idx[a], idx[b]] = D[idx[b], idx[a]] = v

        rng = np.random.default_rng(args.seed)
        entry = {}
        for name, mask, y in (
            ("cross_init_task", diff_init, task),
            ("within_init_task", same_init, task),
            ("cross_task_init", diff_task, init),
        ):
            acc, used, _ = restricted_nn(D, mask, y)
            ch = chance_for(mask, y)
            p = perm_p(D, mask, y, acc, args.n_perm, rng)
            entry[name] = {"accuracy": acc, "chance": ch, "p_value": p,
                           "n_scored": used}
            print(f"{space:18s} {name:18s} {acc:7.3f} {ch:7.3f} {p:10.4g}")
        out["spaces"][space] = entry

    out["interpretation"] = (
        "Plain LOO 1-NN on this bank is confounded: each (task, init) cell "
        "holds 3 adapters sharing both labels, so a nearest neighbour is "
        "typically a cell-mate and cannot attribute the match to either "
        "label. cross_init_task is the decisive figure — task identification "
        "when the neighbour is forced to come from a DIFFERENT "
        "initialization.")
    path = args.bank / "controlled_contrast.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
