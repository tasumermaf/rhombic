# -*- coding: utf-8 -*-
"""A7b — D1 replication on L1-taxonomy adapters, exploratory, n = 60.

WHAT THIS IS: the Director's Item 6 ruling of 2026-09-06 ("A7b, RUN IT
bounded"), implemented with its three pins and nothing more.

    (a) It computes LOO accuracy, Cohen's kappa, and the 1,000-permutation
        null for raw / canonical / vocab_signature, and reports Jeffreys
        intervals; it does NOT apply the D10 lock, the D5 ceiling, or any
        registered granularity metric, because IT IS NOT A LEVEL. There is
        no clean-core split, no D6 reference, and NO GATE IS RECORDED —
        `results/granularity/TIER_GATES.json` is never read or written here.
    (b) Every output is labelled "D1 replication on L1-taxonomy adapters,
        exploratory, n = 60".
    (c) The 60 adapters are named by run index, grouped by parent class, in
        the JSON and in the report (and in `docs/A7B_NOTE_2026-09-06.md`,
        written before this script ran).

COHORT: the COMPLETE L1 runs whose replicate index is 0-4 in every one of the
    12 L1 classes. The four replicate-5 runs are dropped so the design is
    balanced. Each adapter is relabelled by its PARENT task, giving K = 6
    classes, 10 adapters per class, n = 60, chance 1/6.

CODE PATH: `asset1_d1_identifiability.analyze_family` is called with the same
    keyword arguments `granularity_analysis.analyze_label_space` uses
    (proj_dim 16, proj_seed 0, chunk_rows 8, svm_c 1.0, one VocabReadout per
    family expanded into its two kv modes). `granularity_analysis` is imported
    for its helpers — it has a `__main__` guard, so importing it runs no
    analysis — and `cohens_kappa` is taken from it rather than reimplemented.
    Only the D10/D5/clean-core/D6 additions are omitted, per pin (a).

NULL STREAM: seed 0, but `level_index = 101` — deliberately outside every
    registered index space (the granularity ladder occupies level_index 0-8,
    len(LEVEL_TIER) == 9; D1's family_index values are 0-1), so this
    exploratory null can never share a permutation stream with a registered
    run. `rep_index` is d1's fixed per-representation map, as usual.

Usage:
    python scripts/a7b_d1_replication_l1.py --smoke --out-dir <scratch>
    python scripts/a7b_d1_replication_l1.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Pins and frozen parameters ──────────────────────────────────────

LABEL = "D1 replication on L1-taxonomy adapters, exploratory, n = 60"

LEVEL = "L1"
MAX_REPLICATE = 4                 # replicates 0-4 -> balanced 60
EXPECTED_N = 60
EXPECTED_PER_PARENT = 10
EXPECTED_K = 6

N_PERMUTATIONS = 1_000
SEED = 0
LEVEL_INDEX = 101                 # see NULL STREAM in the module docstring
SVM_C = 1.0                       # D1 precedent, as granularity_analysis uses
CHUNK_ROWS = 8
PROJ_DIM = 16
PROJ_SEED = 0

REPRESENTATIONS = ("raw", "canonical", "vocab_signature")
VOCAB_ARMS = (("vocab_signature", "zero_pad"),
              ("vocab_signature_kv_exclude", "exclude"))

DEFAULT_OUT = (REPO_ROOT / "results" / "granularity" / "analysis-exploratory"
               / "A7b-d1-replication-L1-2026-09-06")

# Directories this script must never write into.
FORBIDDEN_ROOTS = (
    REPO_ROOT / "results" / "granularity" / "analysis",
    REPO_ROOT / "results" / "asset1-bank",
)


# ── Jeffreys interval ───────────────────────────────────────────────


def _betacf(a: float, b: float, x: float, itmax: int = 300,
            eps: float = 3.0e-16) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), numpy/math only."""
    from math import exp, lgamma, log
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = lgamma(a + b) - lgamma(a) - lgamma(b)
    front = exp(lbeta + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_ppf_numpy(q: float, a: float, b: float) -> float:
    """Beta quantile by bisection on the regularized incomplete beta."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def jeffreys_ci(x: int, n: int) -> tuple[list[float], str]:
    """Jeffreys 95% interval: Beta(x + 0.5, n - x + 0.5) at 0.025 / 0.975.

    Returns (interval, backend). Verified against the Director's stated
    interval for a perfect LOO at n = 60: [0.959, 1.000].
    """
    a, b = x + 0.5, n - x + 0.5
    try:
        from scipy.stats import beta as _beta
        lo = float(_beta.ppf(0.025, a, b))
        hi = float(_beta.ppf(0.975, a, b))
        backend = "scipy.stats.beta.ppf"
    except Exception:  # noqa: BLE001 — numpy/math fallback
        lo = _beta_ppf_numpy(0.025, a, b)
        hi = _beta_ppf_numpy(0.975, a, b)
        backend = "numpy bisection on the regularized incomplete beta"
    return [lo, hi], backend


def _selfcheck_jeffreys() -> dict:
    """The Director's stated anchor, plus the fallback against the primary."""
    ci, backend = jeffreys_ci(EXPECTED_N, EXPECTED_N)
    fb_lo = _beta_ppf_numpy(0.025, EXPECTED_N + 0.5, 0.5)
    fb_hi = _beta_ppf_numpy(0.975, EXPECTED_N + 0.5, 0.5)
    return {
        "x": EXPECTED_N, "n": EXPECTED_N,
        "backend": backend,
        "interval": ci,
        "interval_rounded_3dp": [round(ci[0], 3), round(ci[1], 3)],
        "director_stated": [0.959, 1.000],
        "matches_director": bool(round(ci[0], 3) == 0.959
                                 and round(ci[1], 3) == 1.000),
        "numpy_fallback_interval": [fb_lo, fb_hi],
        "fallback_agrees_to_1e-9": bool(abs(fb_lo - ci[0]) < 1e-9
                                        and abs(fb_hi - ci[1]) < 1e-9),
        "one_error_accuracy": (EXPECTED_N - 1) / EXPECTED_N,
    }


# ── Guards ──────────────────────────────────────────────────────────


def guard_out_dir(out_dir: Path) -> Path:
    out = Path(out_dir).resolve()
    if "asset1-bank" in out.parts:
        raise SystemExit(f"[a7b] REFUSING --out-dir {out}: inside the "
                         f"Asset-1 bank tree.")
    for root in FORBIDDEN_ROOTS:
        root = root.resolve()
        if out == root or root in out.parents:
            raise SystemExit(
                f"[a7b] REFUSING --out-dir {out}: it is at or under {root}, "
                f"which holds registered analysis outputs. A7b is "
                f"EXPLORATORY and is not a level; it never writes there.")
    return out


# ── Cohort ──────────────────────────────────────────────────────────


def build_cohort(limit_per_parent: int = 0) -> tuple[list[dict], np.ndarray,
                                                     list[str], dict]:
    """The balanced replicate-0-4 cohort, relabelled by parent task."""
    from granularity_runner import (live_classes, load_level, plan_runs,
                                    run_dir_for)

    lm, _pools = load_level(LEVEL)
    classes = live_classes(lm)
    n_classes = len(classes)
    plan = plan_runs(LEVEL, lm)

    selected = []
    for e in plan:
        if e["replicate"] > MAX_REPLICATE:
            continue
        d = run_dir_for(LEVEL, e["run_k"])
        if not (d / "COMPLETE").exists():
            raise SystemExit(
                f"[a7b] REFUSING: run_{e['run_k']:03d} (replicate "
                f"{e['replicate']}, class {e['class_id']}) has no COMPLETE "
                f"marker. The A7b cohort is the balanced replicate-0-4 set; "
                f"it is not a partial-set analysis.")
        selected.append({"run_dir": d, "run_index": e["run_k"],
                         "class_id": e["class_id"], "task": e["task"],
                         "replicate": e["replicate"]})

    if limit_per_parent:
        kept, seen = [], {}
        for r in selected:
            if seen.get(r["task"], 0) < limit_per_parent:
                kept.append(r)
                seen[r["task"]] = seen.get(r["task"], 0) + 1
        selected = kept

    parents = sorted({r["task"] for r in selected})
    y = np.asarray([parents.index(r["task"]) for r in selected],
                   dtype=np.int64)

    by_parent: dict[str, list[str]] = {p: [] for p in parents}
    for r in selected:
        by_parent[r["task"]].append(f"run_{r['run_index']:03d}")
    dropped = [f"run_{e['run_k']:03d}" for e in plan
               if e["replicate"] > MAX_REPLICATE
               and (run_dir_for(LEVEL, e["run_k"]) / "COMPLETE").exists()]

    cohort_meta = {
        "rule": "the COMPLETE L1 runs whose replicate index is 0-4 in every "
                "one of the 12 L1 classes; the replicate-5 runs are dropped "
                "so the design is balanced",
        "level": LEVEL,
        "l1_classes_materialized": n_classes,
        "n_adapters": len(selected),
        "k_parent_classes": len(parents),
        "parents": parents,
        "per_parent_n": {p: len(v) for p, v in by_parent.items()},
        "run_indices_by_parent": by_parent,
        "dropped_replicate_5_complete_runs": dropped,
        "label_space": "each adapter relabelled by its PARENT task",
        "chance": 1.0 / len(parents),
    }
    return selected, y, parents, cohort_meta


def assert_cohort(cohort_meta: dict, strict: bool) -> None:
    if not strict:
        print(f"[a7b] SMOKE cohort: n={cohort_meta['n_adapters']} "
              f"K={cohort_meta['k_parent_classes']} — the 60/10 assertions "
              f"are enforced only on the full run.", flush=True)
        return
    n = cohort_meta["n_adapters"]
    k = cohort_meta["k_parent_classes"]
    if n != EXPECTED_N:
        raise SystemExit(f"[a7b] REFUSING: cohort is {n} adapters, "
                         f"expected {EXPECTED_N}.")
    if k != EXPECTED_K:
        raise SystemExit(f"[a7b] REFUSING: {k} parent classes, "
                         f"expected {EXPECTED_K}.")
    bad = {p: c for p, c in cohort_meta["per_parent_n"].items()
           if c != EXPECTED_PER_PARENT}
    if bad:
        raise SystemExit(f"[a7b] REFUSING: unbalanced cohort {bad}, "
                         f"expected {EXPECTED_PER_PARENT} per parent.")
    print(f"[a7b] cohort asserted: n={n}, K={k}, "
          f"{EXPECTED_PER_PARENT} per parent.", flush=True)


# ── One (representation, kv_mode) cell ──────────────────────────────


def expand_cells(representations: tuple[str, ...]
                 ) -> list[tuple[str, str, str | None]]:
    """(output key, d1 representation, kv_mode) — same shape as
    granularity_analysis.expand_reps."""
    out: list[tuple[str, str, str | None]] = []
    for rep in representations:
        if rep == "vocab_signature":
            out += [(key, "vocab_signature", kv) for key, kv in VOCAB_ARMS]
        else:
            out.append((rep, rep, None))
    return out


def analyze_cell(d1, ga, records: list[dict], y: np.ndarray,
                 class_ids: list[str], *, scratch_dir: Path, key: str,
                 representation: str, kv_mode: str | None, vocab_readout,
                 n_permutations: int, seed: int, level_index: int) -> dict:
    """d1's H1 pipeline for one cell, plus kappa and a Jeffreys interval.

    The analyze_family call mirrors granularity_analysis.analyze_label_space
    exactly. Pin (a): NO D10 lock, NO D5 ceiling, no parent collapse (the
    label space here IS the parent space), no clean-core split.
    """
    classes = np.arange(len(class_ids))
    scratch_dir.mkdir(parents=True, exist_ok=True)
    label = f"A7b-{key}"
    scratch = scratch_dir / f"features_{key}.dat"
    extra = {}
    if representation == "vocab_signature":
        if vocab_readout is None:
            raise ValueError("vocab_signature cell requires a vocab_readout")
        extra = {"vocab_readout": vocab_readout, "kv_mode": kv_mode}

    res = d1.analyze_family(
        records, y, classes, class_ids, representation, scratch,
        n_permutations=n_permutations, seed=seed,
        family_index=level_index,
        rep_index=d1._REP_STREAM_INDEX[key],
        chunk_rows=CHUNK_ROWS, svm_c=SVM_C, proj_dim=PROJ_DIM,
        proj_seed=PROJ_SEED, label=label, **extra)

    res["representation"] = representation
    res["kv_mode"] = kv_mode
    cm = np.asarray(res["confusion_matrix"]["rows_true_cols_pred"])
    res["cohens_kappa"] = ga.cohens_kappa(cm)
    n = int(cm.sum())
    correct = int(np.trace(cm))
    ci, backend = jeffreys_ci(correct, n)
    res["n_correct"] = correct
    res["n_errors"] = n - correct
    res["jeffreys_ci_95"] = ci
    res["jeffreys_backend"] = backend
    res["jeffreys_note"] = (
        "Beta(x + 0.5, n - x + 0.5) at quantiles 0.025 / 0.975. Like the "
        "Wilson interval d1 reports, it treats LOO predictions as "
        "independent Bernoulli trials, which they are not; the permutation "
        "p-value is the calibrated inference.")
    res["locks_applied"] = "none"
    res["not_a_level"] = True
    return res


# ── Output ──────────────────────────────────────────────────────────


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default),
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _typed(pairs) -> str:
    width = max(len(k) for k, _ in pairs)
    return "\n".join(f"{k.ljust(width)} = {v}" for k, v in pairs)


def write_report(path: Path, r: dict) -> None:
    coh = r["cohort"]
    parts = [f"# {r['label']}", "",
             "Produced by `scripts/a7b_d1_replication_l1.py` on the "
             "`asset1_d1_identifiability` H1 machinery (imported, not "
             "forked), per the Director's Item 6 ruling of 2026-09-06. "
             "**This is not a level.** No D10 lock, no D5 ceiling, no "
             "registered granularity metric, no clean-core split, no D6 "
             "reference, and no tier gate is recorded. Scope limit: "
             "**llama3.2-1b only**.", ""]

    core = [("label", r["label"]),
            ("exploratory_only", "true"),
            ("not_a_level", "true"),
            ("locks_applied", r["locks_applied"]),
            ("n_adapters", coh["n_adapters"]),
            ("k_parent_classes", coh["k_parent_classes"]),
            ("per_parent_n",
             sorted(set(coh["per_parent_n"].values()))[0]
             if len(set(coh["per_parent_n"].values())) == 1
             else coh["per_parent_n"]),
            ("chance", f"{coh['chance']:.6f}")]
    for key, res in r["results"].items():
        core += [(f"{key}_acc", f"{res['loo_accuracy']:.4f}"),
                 (f"{key}_kappa", f"{res['cohens_kappa']:.4f}"),
                 (f"{key}_p", f"{res['permutation']['p_value']:.6f}"),
                 (f"{key}_jeffreys_ci_95",
                  f"[{res['jeffreys_ci_95'][0]:.4f}, "
                  f"{res['jeffreys_ci_95'][1]:.4f}]")]
    if r.get("delta_canonical_minus_raw") is not None:
        core.append(("delta_canonical_minus_raw",
                     f"{r['delta_canonical_minus_raw']:+.4f}"))
    parts += ["=== VERIFIED STATE ===", _typed(core),
              "=== END VERIFIED STATE ===", ""]

    parts += ["## Per representation", ""]
    for key, res in r["results"].items():
        parts += [f"### {key}", "",
                  _typed([("representation", res["representation"]),
                          ("kv_mode", res["kv_mode"] or "—"),
                          ("loo_accuracy", f"{res['loo_accuracy']:.4f}"),
                          ("n_correct", res["n_correct"]),
                          ("n_errors", res["n_errors"]),
                          ("jeffreys_ci_95",
                           f"[{res['jeffreys_ci_95'][0]:.4f}, "
                           f"{res['jeffreys_ci_95'][1]:.4f}]"),
                          ("wilson_ci_95",
                           f"[{res['wilson_ci_95'][0]:.4f}, "
                           f"{res['wilson_ci_95'][1]:.4f}]"),
                          ("cohens_kappa", f"{res['cohens_kappa']:.4f}"),
                          ("permutation_p",
                           f"{res['permutation']['p_value']:.6f}"),
                          ("null_mean",
                           f"{res['permutation']['null_mean']:.4f}"),
                          ("null_max",
                           f"{res['permutation']['null_max']:.4f}"),
                          ("macro_f1", f"{res['macro_f1']:.4f}"),
                          ("feature_dim", res["feature_dim"]),
                          ("locks_applied", "none")]), ""]

    parts += ["## What n = 60 can and cannot establish", "",
              "At n = 60, K = 6, balanced 10 per class, a perfect LOO gives "
              "a Jeffreys 95% interval of "
              f"[{r['jeffreys_selfcheck']['interval'][0]:.3f}, "
              f"{r['jeffreys_selfcheck']['interval'][1]:.3f}], so the 0.99 "
              "ceiling is inside the interval and A7b cannot distinguish "
              "\"at ceiling\" from \"one error below it\" (one error at "
              f"n = 60 is "
              f"{r['jeffreys_selfcheck']['one_error_accuracy']:.4f}). What "
              "A7b can establish is the direction and rough size of the "
              "raw-vs-canonical gap on adapters trained under the L1 "
              "taxonomy rather than the Asset-1 taxonomy. The numbers are "
              "above; the reading belongs to the Director.", ""]

    parts += ["## Cohort — the 60 adapters by parent class", "",
              _typed([(p, " ".join(coh["run_indices_by_parent"][p]))
                      for p in coh["parents"]]), "",
              _typed([("dropped_replicate_5",
                       " ".join(coh["dropped_replicate_5_complete_runs"])
                       or "—"),
                      ("cohort_rule", coh["rule"])]), ""]

    parts += ["## Provenance", "",
              _typed([("git_commit", r["git_commit"][:12]),
                      ("generated_at_utc", r["generated_at"]),
                      ("n_permutations", r["n_permutations"]),
                      ("seed", r["seed"]),
                      ("null_stream_key", r["null_stream_key"]),
                      ("level_index", r["level_index"]),
                      ("svm_c", SVM_C),
                      ("note_written_before_run",
                       "docs/A7B_NOTE_2026-09-06.md"),
                      ("tier_gates_touched", "none")]), ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# ── Orchestration ───────────────────────────────────────────────────


def run(out_dir: Path, *, n_permutations: int, seed: int,
        representations: tuple[str, ...], strict_cohort: bool,
        limit_per_parent: int) -> dict:
    import asset1_d1_identifiability as d1
    import granularity_analysis as ga           # has a __main__ guard
    from asset1_bank import git_commit_hash, utc_now

    records, y, parents, cohort = build_cohort(limit_per_parent)
    assert_cohort(cohort, strict_cohort)

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"

    cells = expand_cells(representations)
    vocab_readout = None
    vocab_meta = None
    if any(rep == "vocab_signature" for _, rep, _ in cells):
        vocab_readout, vocab_meta = ga.build_vocab_readout()
        print(f"[a7b] vocab readout loaded "
              f"({vocab_meta.get('vocab_size')} x "
              f"{vocab_meta.get('d_model')})", flush=True)

    results: dict[str, dict] = {}
    for key, rep, kv in cells:
        print(f"[a7b] cell {key}: K={len(parents)} n={len(records)}",
              flush=True)
        results[key] = analyze_cell(
            d1, ga, records, y, parents, scratch_dir=scratch, key=key,
            representation=rep, kv_mode=kv, vocab_readout=vocab_readout,
            n_permutations=n_permutations, seed=seed,
            level_index=LEVEL_INDEX)
        print(f"[a7b] cell {key}: acc="
              f"{results[key]['loo_accuracy']:.4f} kappa="
              f"{results[key]['cohens_kappa']:.4f} p="
              f"{results[key]['permutation']['p_value']:.6f}", flush=True)

    delta = None
    if "raw" in results and "canonical" in results:
        delta = (results["canonical"]["loo_accuracy"]
                 - results["raw"]["loo_accuracy"])

    payload = {
        "label": LABEL,
        "analysis": "A7b — held-out replication of D1's label-separability "
                    "result on the completed L1-taxonomy adapters",
        "exploratory_only": True,
        "not_a_level": True,
        "locks_applied": "none",
        "pins": {
            "a": "computes LOO accuracy, Cohen's kappa and the "
                 "1,000-permutation null for raw / canonical / "
                 "vocab_signature and reports Jeffreys intervals; does NOT "
                 "apply the D10 lock, the D5 ceiling, or any registered "
                 "granularity metric, because it is not a level",
            "b": f"labelled \"{LABEL}\" in every output",
            "c": "the 60 adapters used are named by run index so that, when "
                 "240 are complete, the same analysis can be re-run and the "
                 "two compared",
            "source": "Director's review 2026-09-06, Item 6 (A7b, RUN IT "
                      "bounded)",
        },
        "generated_by": "scripts/a7b_d1_replication_l1.py",
        "note": "docs/A7B_NOTE_2026-09-06.md (written before this run)",
        "git_commit": git_commit_hash(),
        "generated_at": utc_now(),
        "family": "meta-llama/Llama-3.2-1B-Instruct",
        "scope_limit": "llama3.2-1b only (D1 APPROVED) — this scope limit "
                       "belongs in every claim derived from this file.",
        "n_permutations": n_permutations,
        "seed": seed,
        "level_index": LEVEL_INDEX,
        "null_stream_key": f"default_rng([seed={seed}, "
                           f"level_index={LEVEL_INDEX}, rep_index]) — "
                           f"level_index {LEVEL_INDEX} is outside every "
                           f"registered index space (granularity levels 0-8, "
                           f"D1 families 0-1) so this exploratory null never "
                           f"shares a stream with a registered run",
        "rep_stream_index": {k: d1._REP_STREAM_INDEX[k] for k, _, _ in cells},
        "parameters": {"svm_c": SVM_C, "chunk_rows": CHUNK_ROWS,
                       "proj_dim": PROJ_DIM, "proj_seed": PROJ_SEED,
                       "representations": list(representations),
                       "code_path": "asset1_d1_identifiability.analyze_family, "
                                    "called with the same keyword arguments "
                                    "granularity_analysis.analyze_label_space "
                                    "uses"},
        "cohort": cohort,
        "jeffreys_selfcheck": _selfcheck_jeffreys(),
        "results": results,
        "delta_canonical_minus_raw": delta,
        "direction_check": "The Director's reading to test: if canonical is "
                           "near 1.0 and raw is near chance on held-out "
                           "adapters that share nothing with the Asset-1 "
                           "bank but the base model, D1's central finding "
                           "replicates out of sample. The numbers above are "
                           "reported without editorial.",
        "tier_gates_touched": "none",
    }
    if vocab_meta:
        payload["vocabsig_readout"] = vocab_meta

    _write_json(out_dir / "A7b_results.json", payload)
    write_report(out_dir / "A7b_REPORT.md", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="A7b — D1 replication on L1-taxonomy adapters, "
                    "exploratory, n = 60. Not a level; no locks applied.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--representation", default="all",
                    choices=["raw", "canonical", "vocab_signature", "all"])
    ap.add_argument("--smoke", action="store_true",
                    help="tooling check: 10 permutations, raw only, into a "
                         "scratch out-dir. NEVER reportable.")
    ap.add_argument("--limit-per-parent", type=int, default=0,
                    help="smoke only — cap adapters per parent class")
    args = ap.parse_args(argv)

    if args.limit_per_parent and not args.smoke:
        ap.error("--limit-per-parent is a smoke-only knob; the full run "
                 "uses all 60 adapters")

    out_dir = guard_out_dir(args.out_dir)
    n_perm = 10 if args.smoke else args.n_permutations
    reps = (("raw",) if args.smoke
            else (REPRESENTATIONS if args.representation == "all"
                  else (args.representation,)))
    if args.smoke and out_dir == guard_out_dir(DEFAULT_OUT):
        raise SystemExit("[a7b] REFUSING: --smoke must write to a scratch "
                         "out-dir, not the reportable default.")

    check = _selfcheck_jeffreys()
    print(f"[a7b] jeffreys selfcheck x=60 n=60: "
          f"[{check['interval'][0]:.6f}, {check['interval'][1]:.6f}] "
          f"backend={check['backend']} "
          f"matches_director={check['matches_director']} "
          f"fallback_agrees={check['fallback_agrees_to_1e-9']}", flush=True)
    if not check["matches_director"]:
        raise SystemExit("[a7b] REFUSING: the Jeffreys implementation does "
                         "not reproduce the Director's [0.959, 1.000] at "
                         "x = n = 60.")

    if args.smoke:
        print("\n*** A7b SMOKE — TOOLING CHECK ONLY. 10 permutations, raw "
              "only. Every number produced here is NON-REPORTABLE. ***\n",
              flush=True)

    print(f"[a7b] {LABEL}", flush=True)
    print(f"[a7b] out_dir = {out_dir}", flush=True)
    print(f"[a7b] n_permutations = {n_perm}  seed = {args.seed}  "
          f"representations = {reps}", flush=True)

    r = run(out_dir, n_permutations=n_perm, seed=args.seed,
            representations=reps, strict_cohort=not args.smoke,
            limit_per_parent=args.limit_per_parent)
    print(f"[a7b] done -> {out_dir / 'A7b_REPORT.md'}", flush=True)
    for key, res in r["results"].items():
        print(f"[a7b] {key}: acc={res['loo_accuracy']:.4f} "
              f"kappa={res['cohens_kappa']:.4f} "
              f"p={res['permutation']['p_value']:.6f} "
              f"jeffreys=[{res['jeffreys_ci_95'][0]:.4f}, "
              f"{res['jeffreys_ci_95'][1]:.4f}]", flush=True)
    if r["delta_canonical_minus_raw"] is not None:
        print(f"[a7b] delta_canonical_minus_raw = "
              f"{r['delta_canonical_minus_raw']:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    # No GPU, no hub calls — set before anything imports torch/transformers.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.exit(main())
