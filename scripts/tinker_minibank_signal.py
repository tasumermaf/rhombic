"""E-T4 Tinker mini-bank — task-identity readout over the exported adapters.

The pilot's readout (``scripts/tinker_pilot_signal.py``) at n=6 could only
ask "is every same-task pair closer than every cross-task pair". At n=54
the question becomes a classification one, which is what the mini-bank was
bought for: **leave-one-out task identification over 6 classes**, reported
separately for the gauge-DEPENDENT raw representation and the
gauge-CANONICAL one.

Bindings from PILOT_REPORT §6, honoured here
--------------------------------------------
§6.1  Data seed and init seed vary independently, so INIT IDENTITY is
      carried as a named nuisance label and scored exactly like task
      identity. The pilot's raw-space failure was an initialization
      effect; here that is measured rather than inferred.
§6.2  The ``sigma`` variant (log1p singular values only — fully
      rotation-invariant, no random probes) is carried as a HEADLINE
      representation alongside ``full``, not as a robustness footnote.

Canonicalization is IMPORTED from ``scripts/asset1_canonicalize.py``
(``canonicalize_module``, ``feature_vector``) exactly as the pilot did it,
with the same bridgeless adaptation (bridge absorption = identity, only the
alpha/r scalar remains). Nothing about it is reimplemented.

Memory
------
A raw feature vector is 92,286,976 floats; 54 of them would be ~20 GB in
RAM. So the two spaces are built in two passes with different majorities:

* canonical — ADAPTER-major. One adapter's modules are canonicalized and
  handed to ``feature_vector`` whole (the imported function, unmodified);
  the resulting 267,168-d (full) and 8,096-d (sigma) vectors are kept.
* raw — MODULE-major. For each module the (n_adapters x block) slice is
  read from all adapters at once and accumulated into the Gram matrix,
  ``G += X @ X.T``. This is the EXACT Gram of the full concatenated
  vectors — cosine distance from it is identical to the pilot's
  whole-vector computation — at ~80 MB of working set.

RUNS UNDER THE ``falco`` CONDA ENV (needs torch + safetensors):
    C:\\miniconda3\\envs\\falco\\python.exe

Usage
-----
    python scripts/tinker_minibank_signal.py
    python scripts/tinker_minibank_signal.py --bank results/tinker-minibank
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asset1_canonicalize import canonicalize_module, feature_vector  # noqa: E402
from tinker_pilot_signal import (  # noqa: E402
    _module_key, effective_factors_bridgeless, load_bridgeless_adapter,
    scaling_from_config)

_DTYPE = torch.float64


# ── Discovery and labels ─────────────────────────────────────────────


def discover_runs(bank: Path) -> list[Path]:
    """Immediate children holding an exported adapter. data/ has none."""
    def is_run(d: Path) -> bool:
        return d.is_dir() and bool(list(d.glob("*.safetensors")))
    return sorted(d for d in bank.iterdir() if is_run(d))


def labels_for(run_dir: Path) -> dict:
    """(task, data_seed, init_seed) from run_record.json, else from the name.

    Mini-bank names are ``<task>_d<data_seed>_i<init_seed>``. The PILOT's
    names are ``<task>_<seed>``, where the single seed set BOTH the data
    order and the LoRA init — so it maps to data_seed == init_seed. Keeping
    that fallback lets this readout run unchanged on the pilot bank, which
    is how it is validated against the pilot's published numbers.
    """
    rec_path = run_dir / "run_record.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if all(k in rec for k in ("task", "data_seed", "init_seed")):
            return {"task": rec["task"], "data_seed": int(rec["data_seed"]),
                    "init_seed": int(rec["init_seed"])}
        if "task" in rec and "seed" in rec:          # pilot record
            return {"task": rec["task"], "data_seed": int(rec["seed"]),
                    "init_seed": int(rec["seed"])}
    name = run_dir.name
    if "_d" in name and "_i" in name:
        task, _, rest = name.partition("_d")
        d, _, i = rest.partition("_i")
        return {"task": task, "data_seed": int(d), "init_seed": int(i)}
    task, _, seed = name.rpartition("_")             # pilot form
    return {"task": task, "data_seed": int(seed), "init_seed": int(seed)}


# ── Feature construction ─────────────────────────────────────────────


def canonical_features(run_dirs: list[Path], proj_dim: int, proj_seed: int,
                       rank_fallback: int) -> tuple[dict, dict, dict]:
    """Adapter-major pass -> (full_feats, sigma_feats, per-adapter meta)."""
    full: dict[str, np.ndarray] = {}
    sigma: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for k, run_dir in enumerate(run_dirs, 1):
        t0 = time.time()
        rid = run_dir.name
        modules, cfg = load_bridgeless_adapter(run_dir)
        scaling, scale_meta = scaling_from_config(cfg, rank_fallback)
        canon: dict[str, dict[str, torch.Tensor]] = {}
        for name in sorted(modules):
            B_eff, A_eff = effective_factors_bridgeless(modules[name], scaling)
            canon[name] = canonicalize_module(B_eff, A_eff)
        full[rid] = feature_vector(canon, variant="full", proj_dim=proj_dim,
                                   proj_seed=proj_seed).numpy()
        sigma[rid] = feature_vector(canon, variant="sigma").numpy()
        meta[rid] = {"n_modules": len(modules),
                     "canonical_full_dim": int(full[rid].size),
                     "canonical_sigma_dim": int(sigma[rid].size),
                     **scale_meta}
        print(f"[canon] {k:3d}/{len(run_dirs)} {rid:22s} {len(modules)} mods  "
              f"full {full[rid].size:,}  sigma {sigma[rid].size:,}  "
              f"{time.time() - t0:.0f}s", flush=True)
    return full, sigma, meta


def raw_gram(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, int]:
    """Module-major pass -> (Gram, module_order_hash_ok, raw_dim).

    Returns the EXACT Gram matrix of the full concatenated raw vectors
    (flattened lora_A then lora_B per module, sorted module order — the
    pilot's definition), computed without ever materializing them.
    """
    n = len(run_dirs)
    handles, names_per = [], []
    for run_dir in run_dirs:
        st = sorted(run_dir.glob("adapter_model.safetensors")) or \
            sorted(run_dir.glob("*.safetensors"))
        f = safe_open(str(st[0]), framework="pt", device="cpu")
        handles.append(f)
        mods: dict[str, dict[str, str]] = {}
        for key in f.keys():
            parsed = _module_key(key)
            if parsed is None:
                continue
            module, field = parsed
            mods.setdefault(module, {})[field] = key
        names_per.append(mods)

    module_names = sorted(names_per[0])
    for k, mods in enumerate(names_per[1:], 1):
        if sorted(mods) != module_names:
            raise SystemExit(
                f"module set mismatch: {run_dirs[0].name} has "
                f"{len(module_names)} modules, {run_dirs[k].name} has "
                f"{len(mods)} — cannot compare raw vectors")

    G = np.zeros((n, n), dtype=np.float64)
    raw_dim = 0
    # The unembed module is ~5M floats per adapter (vocab x rank), so a naive
    # stack-then-upcast would hold both a float32 and a float64 copy across
    # all n adapters at once. Fill a preallocated float32 buffer, then
    # accumulate in float64 over COLUMN CHUNKS: exact, and bounded working set.
    chunk_cols = 1_000_000
    for m, module in enumerate(module_names, 1):
        rows = []
        for f, mods in zip(handles, names_per):
            A = f.get_tensor(mods[module]["lora_A"]).to(torch.float32).reshape(-1)
            B = f.get_tensor(mods[module]["lora_B"]).to(torch.float32).reshape(-1)
            rows.append(torch.cat([A, B]).numpy())
        D = rows[0].size
        X = np.empty((n, D), dtype=np.float32)
        for i, r in enumerate(rows):
            if r.size != D:
                raise SystemExit(
                    f"module {module!r}: adapter {run_dirs[i].name} has "
                    f"{r.size} params, expected {D}")
            X[i] = r
        del rows
        for s in range(0, D, chunk_cols):
            Xc = X[:, s:s + chunk_cols].astype(np.float64)
            G += Xc @ Xc.T
        raw_dim += D
        del X
        if m % 50 == 0 or m == len(module_names):
            print(f"[raw]   module {m}/{len(module_names)}  "
                  f"dim so far {raw_dim:,}", flush=True)
    return G, module_names, raw_dim


def gram_to_distances(G: np.ndarray, run_ids: list[str]
                      ) -> dict[tuple[str, str], float]:
    d = np.sqrt(np.diag(G))
    out = {}
    for a, b in itertools.combinations(range(len(run_ids)), 2):
        denom = d[a] * d[b]
        cos = float(G[a, b] / denom) if denom > 0 else float("nan")
        key = tuple(sorted((run_ids[a], run_ids[b])))
        out[key] = 1.0 - cos
    return out


def feats_to_distances(feats: dict[str, np.ndarray]
                       ) -> dict[tuple[str, str], float]:
    runs = sorted(feats)
    X = np.stack([feats[r].astype(np.float64) for r in runs])
    G = X @ X.T
    return gram_to_distances(G, runs)


# ── Classification readout ───────────────────────────────────────────


def dist_matrix(dists: dict[tuple[str, str], float], runs: list[str]
                ) -> np.ndarray:
    n = len(runs)
    D = np.full((n, n), np.inf)
    idx = {r: i for i, r in enumerate(runs)}
    for (a, b), v in dists.items():
        D[idx[a], idx[b]] = D[idx[b], idx[a]] = v
    return D


def loo_knn(D: np.ndarray, y: list, k: int) -> tuple[float, list]:
    """Leave-one-out k-NN accuracy and the predicted labels.

    LOO is enforced HERE, not assumed of the caller: the diagonal is masked
    to +inf on a copy, so a point can never be its own neighbour whatever
    matrix it is handed. (A zero diagonal would make k=1 trivially perfect
    on pure noise — the failure this guard exists to prevent.) Ties in the
    vote are broken by the nearest member of the tied classes.
    """
    D = np.array(D, dtype=float, copy=True)
    np.fill_diagonal(D, np.inf)
    n = len(y)
    preds = []
    for i in range(n):
        order = np.argsort(D[i])
        nbrs = [y[j] for j in order[:k]]
        counts = Counter(nbrs)
        top = max(counts.values())
        tied = {lab for lab, c in counts.items() if c == top}
        if len(tied) == 1:
            preds.append(next(iter(tied)))
        else:
            for j in order:
                if y[j] in tied:
                    preds.append(y[j])
                    break
    acc = float(np.mean([p == t for p, t in zip(preds, y)]))
    return acc, preds


def separation(dists: dict[tuple[str, str], float], label_of: dict) -> dict:
    within = [d for p, d in dists.items() if label_of[p[0]] == label_of[p[1]]]
    cross = [d for p, d in dists.items() if label_of[p[0]] != label_of[p[1]]]
    max_w = max(within) if within else float("nan")
    min_c = min(cross) if cross else float("nan")
    return {
        "n_within_pairs": len(within), "n_cross_pairs": len(cross),
        "mean_within": float(np.mean(within)) if within else float("nan"),
        "mean_cross": float(np.mean(cross)) if cross else float("nan"),
        "max_within": max_w, "min_cross": min_c,
        "margin": min_c - max_w, "separated": bool(max_w < min_c),
    }


def permutation_null(D: np.ndarray, y: list, k: int, n_perm: int,
                     rng: np.random.Generator) -> dict:
    """Label-permutation null for LOO k-NN accuracy.

    The geometry is held fixed and only the labels move, so this asks
    exactly: could this accuracy arise from a random assignment of the
    same class sizes to these same points?
    """
    obs, _ = loo_knn(D, y, k)
    y_arr = np.asarray(y, dtype=object)
    ge = 0
    for _ in range(n_perm):
        perm = list(y_arr[rng.permutation(len(y_arr))])
        acc, _ = loo_knn(D, perm, k)
        ge += int(acc >= obs)
    return {"observed": obs, "n_perm": n_perm, "n_ge": ge,
            "p_value": (ge + 1) / (n_perm + 1)}


def confusion(y_true: list, y_pred: list, classes: list) -> dict:
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return {"classes": classes, "matrix": M.tolist()}


def score_space(name: str, dists, runs, labels: dict[str, dict],
                ks: list[int], n_perm: int, seed: int) -> dict:
    D = dist_matrix(dists, runs)
    out: dict = {"n_adapters": len(runs)}
    rng = np.random.default_rng(seed)
    for label in ("task", "init_seed", "data_seed"):
        y = [labels[r][label] for r in runs]
        classes = sorted(set(y), key=str)
        sizes = Counter(y)
        chance = sum((c - 1) for c in sizes.values()) / (len(y) * (len(y) - 1))
        entry = {
            "classes": [str(c) for c in classes],
            "class_sizes": {str(c): sizes[c] for c in classes},
            "loo_1nn_chance": chance,
            "separation": separation(dists, {r: labels[r][label] for r in runs}),
            "knn": {},
        }
        for k in ks:
            acc, preds = loo_knn(D, y, k)
            entry["knn"][f"k={k}"] = {"loo_accuracy": acc}
            if k == 1:
                entry["confusion_1nn"] = confusion(
                    y, preds, [str(c) for c in classes]) if isinstance(
                        classes[0], str) else confusion(
                        [str(v) for v in y], [str(v) for v in preds],
                        [str(c) for c in classes])
        entry["permutation_null_1nn"] = permutation_null(
            D, y, 1, n_perm, rng)
        out[label] = entry
        print(f"  [{name}] {label:10s} LOO-1NN {entry['knn']['k=1']['loo_accuracy']:.3f} "
              f"(chance {chance:.3f}, p={entry['permutation_null_1nn']['p_value']:.4g})",
              flush=True)
    return out


# ── Orchestration ────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Tinker mini-bank readout")
    ap.add_argument("--bank", type=Path,
                    default=REPO_ROOT / "results" / "tinker-minibank")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--proj-dim", type=int, default=16)
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--rank-fallback", type=int, default=32)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument("--skip-raw", action="store_true")
    args = ap.parse_args()

    run_dirs = discover_runs(args.bank)
    if not run_dirs:
        raise SystemExit(f"no exported adapters under {args.bank}")
    runs = [d.name for d in run_dirs]
    labels = {d.name: labels_for(d) for d in run_dirs}

    tasks = sorted({labels[r]["task"] for r in runs})
    print(f"[bank] {len(runs)} adapters, {len(tasks)} tasks: {tasks}")
    print(f"[bank] init seeds {sorted({labels[r]['init_seed'] for r in runs})}, "
          f"data seeds {sorted({labels[r]['data_seed'] for r in runs})}\n")

    full, sigma, meta = canonical_features(
        run_dirs, args.proj_dim, args.proj_seed, args.rank_fallback)

    spaces: dict[str, dict] = {}
    dist_store: dict[str, dict] = {}

    for name, feats in (("canonical_full", full), ("canonical_sigma", sigma)):
        print(f"\n=== {name} ===")
        dists = feats_to_distances(feats)
        dist_store[name] = dists
        spaces[name] = score_space(name, dists, runs, labels, args.ks,
                                   args.n_perm, args.perm_seed)

    raw_dim = None
    if not args.skip_raw:
        print(f"\n=== raw (module-major exact Gram) ===")
        G, module_names, raw_dim = raw_gram(run_dirs)
        dists = gram_to_distances(G, runs)
        dist_store["raw"] = dists
        spaces["raw"] = score_space("raw", dists, runs, labels, args.ks,
                                    args.n_perm, args.perm_seed)
        print(f"[raw]   raw dim {raw_dim:,} over {len(module_names)} modules")

    payload = {
        "bank": args.bank.as_posix(),
        "n_adapters": len(runs),
        "runs": runs,
        "labels": labels,
        "tasks": tasks,
        "probe": {"proj_dim": args.proj_dim, "proj_seed": args.proj_seed},
        "knn_ks": args.ks,
        "n_perm": args.n_perm,
        "raw_dim": raw_dim,
        "per_adapter": meta,
        "spaces": spaces,
        "readout_definition": (
            "Leave-one-out k-NN identification of TASK (6 classes) and of the "
            "two nuisance labels INIT_SEED and DATA_SEED, under cosine "
            "distance, in three representations: raw (gauge-dependent, exact "
            "Gram), canonical_full and canonical_sigma (both GL(r)-invariant). "
            "p-values are label-permutation nulls on LOO-1NN accuracy with the "
            "geometry held fixed."),
    }
    out = args.out or (args.bank / "signal_results.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n[signal] wrote {out}")

    dpath = out.with_name("pairwise_distances.json")
    dpath.write_text(json.dumps(
        {sp: {f"{a}|{b}": v for (a, b), v in sorted(d.items())}
         for sp, d in dist_store.items()}, indent=2) + "\n", encoding="utf-8")
    print(f"[signal] wrote {dpath}")


if __name__ == "__main__":
    main()
