# -*- coding: utf-8 -*-
"""Granularity ladder — per-level analysis (H1 machinery, identical at each level).

REGISTERED CARD: `docs/REGISTRATION_GRANULARITY_2026-07-30.md`, LOCKED by
    `docs/LOCK_GRANULARITY_2026-08-04.md`.

WHAT THIS SCRIPT IS: the design's §5, run per level, on the SAME machinery
    that produced Asset-1's H1. Every classifier, Gram, permutation null,
    Wilson interval and heterogeneity guard comes from
    `asset1_d1_identifiability` by import — `analyze_family()` is called
    unchanged. This module adds only what the ladder's rulings require on
    top of it:

      * **Cohen's kappa at EVERY level, including L0** (D5 PINNED, D10) —
        computed from the LOO confusion matrix `analyze_family` already
        returns, so it is the same predictions, not a second fit.
      * **The D10 triple lock** (Director-set at registration, not tunable
        after a level returns):
            acc > 1.5 x chance(K)  AND  perm p < 0.01  AND  kappa >= 0.40
      * **Clean-core vs all-classes curves** (D3 RESTRICTED): the T1+T2
        subset is analyzed as its own label space alongside all classes. On
        material divergence the clean-core curve is the claim.
      * **The D6 data-space reference** (PINNED; Ask 3 2026-08-04): TF-IDF +
        linear SVM on the training DOCUMENTS under the same labels,
        1,000 examples/class nominal, **cross-validated WITHIN the
        subsample**, with the realized per-class n reported — the three
        conditions the Director attached to the pin.
      * **Parent-collapsed accuracy** (§5.4) — the fine confusion matrix
        collapsed onto the six parent tasks.
      * **Ceiling** (D5): LOO acc >= 0.99.
      * **Three representations** (design §5): raw, canonical, and — wired
        2026-09-05 after the relaunch-readiness read found it registered
        but unimplemented — vocab_signature in both kv modes (zero_pad
        primary, exclude secondary), through the same D1 arm #3 code path
        and null-stream indices. `--representation both` reproduces the
        pre-2026-09-05 two-representation run.

TWO INTERLOCKS, both refusals rather than warnings:
    1. COMPLETENESS — a level is analyzable only when every planned run
       carries a COMPLETE marker (L0: the Asset-1 bank's own
       `require_complete_bank`, which is where L0's 240 llama adapters live).
    2. TIER ORDER — the lock freezes the order L0 -> L1 -> ARMB -> L2 -> L3
       -> D7 and requires each gate to record which tiers were already
       unblinded when it fired. `results/granularity/TIER_GATES.json` is that
       ledger; a level whose predecessors have not fired is refused.

L0 IS ANALYSIS-SIDE: it trains nothing. Its adapters are the existing llama
    cohort of the Asset-1 bank (6 tasks x 40 seeds). The re-baseline exists
    to put kappa and the D6 reference on the anchor so the curve's first
    point is comparable with the rest.

Usage:
    python scripts/granularity_analysis.py --selftest
    python scripts/granularity_analysis.py --level L0
    python scripts/granularity_analysis.py --level L1
    python scripts/granularity_analysis.py --curve
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

os.environ.setdefault("HF_HUB_CACHE", r"C:\falco\hf-cache\hub")
os.environ.setdefault("HF_DATASETS_CACHE", r"C:\falco\hf-cache\datasets")

import asset1_analysis_io as aio  # noqa: E402
import asset1_d1_identifiability as d1  # noqa: E402
import asset1_vocab_signature as vs  # noqa: E402  (arm #3, wired 2026-09-05)
from asset1_bank import MAX_LEN, git_commit_hash, utc_now  # noqa: E402
from asset1_datasets import (  # noqa: E402
    POOL_CAP, TASK_REGISTRY, VAL_SEED, load_hf_dataset_with_fallback)
from granularity_labels import (  # noqa: E402
    CLEAN_CORE_TIERS, D6_SUBSAMPLE_N, LABELS_DIR, OUT_ROOT, TIER_ORDER)
from granularity_runner import FAMILY, live_classes, load_level, plan_runs, \
    run_dir_for  # noqa: E402

# ── Locked / pinned analysis constants ──────────────────────────────

D10_ACC_MULTIPLIER = d1.H1_ACCURACY_MULTIPLIER   # 1.5 x chance — from D1
D10_P_THRESHOLD = d1.H1_P_THRESHOLD              # 0.01 — from D1
D10_KAPPA_FLOOR = 0.40                           # D10, Director-set at reg.
D5_CEILING = 0.99                                # D5 PINNED

N_PERMUTATIONS = 1_000                           # design §5
SVM_C = 1.0                                      # D1 precedent, APPROVED

# D6 frozen configuration (PINNED — pin the method AND the subsample size).
# Same model class as the weight-space probe (a linear SVM at C=1.0); the
# solver is liblinear rather than libsvm because the data-space matrices are
# large and sparse. Recorded in every output.
D6_CONFIG = {
    "vectorizer": "sklearn TfidfVectorizer",
    "lowercase": True,
    "sublinear_tf": True,
    "min_df": 2,
    "max_features": 50_000,
    "ngram_range": [1, 1],
    "classifier": "sklearn LinearSVC(C=1.0, dual='auto', max_iter=5000)",
    "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=6) "
          "WITHIN the subsample (Ask 3 condition i)",
    "subsample_nominal_per_class": D6_SUBSAMPLE_N,
    "documents": "the formatted training documents the adapters saw "
                 "(asset1_datasets task class, keep_text=True) — including "
                 "the token-budget truncation for xsum/squad",
}
D6_CV_FOLDS = 5
D6_CV_SEED = 6

GATES_FILE = OUT_ROOT / "TIER_GATES.json"
ANALYSIS_ROOT = OUT_ROOT / "analysis"

# Level -> tier name in the frozen order.
LEVEL_TIER = {"L0": "L0", "L1": "L1",
              "B2": "ARMB", "B4": "ARMB", "B8": "ARMB", "B16": "ARMB",
              "L2": "L2", "L3": "L3", "D7": "D7"}

REPRESENTATIONS = ("raw", "canonical", "vocab_signature")
# The registered design (DESIGN_GRANULARITY_BRIDGE_2026-07-21.md §5, "per
# representation (raw, canonical, vocab_signature [both kv modes])") names
# three representations; until 2026-09-05 this script implemented two
# (audit 2026-09-01 / relaunch readiness step 4). The vocab arm now runs
# exactly as D1 arm #3 does: one VocabReadout per family, expanded into its
# two kv_mode variants with their reserved null-stream indices (d1
# _REP_STREAM_INDEX 2 and 3). zero_pad is the pinned primary; exclude is the
# secondary (Director condition 2026-07-07).
VOCAB_ARMS = (("vocab_signature", "zero_pad"),
              ("vocab_signature_kv_exclude", "exclude"))


def expand_reps(representations: tuple[str, ...]) -> list[tuple[str, str, str | None]]:
    """(output key, d1 representation, kv_mode) per analysis cell."""
    out: list[tuple[str, str, str | None]] = []
    for rep in representations:
        if rep == "vocab_signature":
            out += [(key, "vocab_signature", kv) for key, kv in VOCAB_ARMS]
        else:
            out.append((rep, rep, None))
    return out


def build_vocab_readout() -> tuple[object, dict]:
    """One frozen-model output readout for the level's (single) family,
    built exactly as asset1_d1_identifiability does for arm #3."""
    info = vs.load_unembedding(FAMILY["model"])
    readout = vs.VocabReadout(info["W_U"], info["norm_g"],
                              sketch_dim=vs.SKETCH_DIM,
                              sketch_seed=vs.DEFAULT_SEED,
                              vocab_chunk=vs.VOCAB_CHUNK)
    meta = {k: info[k] for k in ("vocab_size", "d_model", "loaded_keys",
                                 "files_opened", "tied_embeddings_fallback",
                                 "snapshot") if k in info}
    meta["model_id"] = FAMILY["model"]
    return readout, meta


# ── Small math the rulings add on top of d1 ─────────────────────────


def cohens_kappa(cm: np.ndarray) -> float:
    """Cohen's kappa from a confusion matrix (rows=true, cols=pred).

    kappa = (po - pe) / (1 - pe), po = trace/N, pe = sum_i row_i*col_i / N^2.
    Computed from the SAME LOO predictions that produced the accuracy — not
    a second classifier fit. Required at every level including L0 (D5/D10)
    because the 1.5x-chance bar degenerates to ~0.031 at K=48.
    """
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    if n <= 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=1) * cm.sum(axis=0)).sum()) / (n * n)
    if np.isclose(pe, 1.0):
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def collapse_confusion(cm: np.ndarray, class_parents: list[str]
                       ) -> tuple[np.ndarray, list[str]]:
    """Collapse a fine confusion matrix onto parent tasks (§5.4)."""
    parents = sorted(set(class_parents))
    pos = {p: i for i, p in enumerate(parents)}
    out = np.zeros((len(parents), len(parents)), dtype=np.int64)
    cm = np.asarray(cm)
    for i, pi in enumerate(class_parents):
        for j, pj in enumerate(class_parents):
            out[pos[pi], pos[pj]] += int(cm[i, j])
    return out, parents


def d10_lock(acc: float, p: float, kappa: float, k: int) -> dict:
    """The Director-set lock form, evaluated identically at every level."""
    chance = 1.0 / k
    threshold = D10_ACC_MULTIPLIER * chance
    return {
        "k": k,
        "chance": chance,
        "accuracy_threshold_1p5x_chance": threshold,
        "accuracy": acc,
        "accuracy_exceeds_threshold": bool(acc > threshold),
        "p_value": p,
        "p_below_threshold": bool(p < D10_P_THRESHOLD),
        "kappa": kappa,
        "kappa_floor": D10_KAPPA_FLOOR,
        "kappa_meets_floor": bool(kappa >= D10_KAPPA_FLOOR),
        "pass": bool(acc > threshold and p < D10_P_THRESHOLD
                     and kappa >= D10_KAPPA_FLOOR),
        "form": "acc > 1.5 x chance(K) AND perm p < 0.01 AND kappa >= 0.40 "
                "(D10, Director-set at registration 2026-07-30 — set before "
                "any level was analyzed so it cannot be tuned to a result)",
    }


# ── Interlocks ──────────────────────────────────────────────────────


def load_gates() -> dict:
    if GATES_FILE.exists():
        return json.loads(GATES_FILE.read_text(encoding="utf-8"))
    return {"ledger": [], "note": "Tier gates, in firing order. The lock "
                                  "requires each gate to record which tiers "
                                  "were already unblinded when it fired."}


def require_tier_order(level: str, force: bool = False) -> list[str]:
    """Refuse a level whose predecessors in the frozen order have not fired."""
    tier = LEVEL_TIER[level]
    gates = load_gates()
    fired = [g["tier"] for g in gates["ledger"]]
    idx = TIER_ORDER.index(tier)
    missing = [t for t in TIER_ORDER[:idx] if t not in fired]
    if missing and not force:
        raise SystemExit(
            f"[gran-analysis] REFUSING level {level} (tier {tier}): the "
            f"frozen tier order is {' -> '.join(TIER_ORDER)} and "
            f"{missing} has not fired. The lock freezes this order "
            f"(cheapest-first, S9 doctrine); analyzing out of order changes "
            f"what was unblinded when each gate fired. Resolve by dated "
            f"amendment (L-006), never by a flag on a reportable run.")
    if missing and force:
        print(f"[gran-analysis] *** --force-tier-order: analyzing {level} "
              f"with {missing} unfired. EXPLORATORY ONLY; this run is not "
              f"reportable.", file=sys.stderr)
    return fired


def record_gate(level: str, fired_before: list[str], payload: dict) -> None:
    gates = load_gates()
    gates["ledger"].append({
        "tier": LEVEL_TIER[level],
        "level": level,
        "fired_at": utc_now(),
        "tiers_already_unblinded": fired_before,
        "k": payload.get("k"),
        "n_runs": payload.get("n_runs"),
        "git_commit": git_commit_hash(),
    })
    GATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = GATES_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(gates, indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, GATES_FILE)


def collect_level_records(level: str, allow_partial: bool
                          ) -> tuple[list[dict], np.ndarray, list[str],
                                     list[str], dict]:
    """(records, y, class_ids, class_parents, level manifest) with the
    completeness interlock applied."""
    if level == "L0":
        manifest = aio.require_complete_bank(aio.REAL_BANK_ROOT,
                                             allow_partial=allow_partial)
        recs = [r for r in aio.iter_runs(aio.REAL_BANK_ROOT,
                                         family=FAMILY["short"])]
        tasks = sorted({r["task"] for r in recs})
        y = np.array([tasks.index(r["task"]) for r in recs], dtype=np.int64)
        lm = {"level": "L0", "k_materialized": len(tasks),
              "seeds_per_class": 40, "kind": "analysis-side",
              "balance_policy": "n/a (existing bank adapters)",
              "source": str(aio.REAL_BANK_ROOT),
              "note": "L0 anchor — the existing llama3.2-1b cohort of the "
                      "Asset-1 bank; no new training (design §2)."}
        return recs, y, tasks, tasks, lm

    lm, _pools = load_level(level)
    classes = live_classes(lm)
    class_ids = [c["class_id"] for c in classes]
    parents = [c["task"] for c in classes]
    plan = plan_runs(level, lm)
    missing = [e["run_k"] for e in plan
               if not (run_dir_for(level, e["run_k"]) / "COMPLETE").exists()]
    if missing and not allow_partial:
        raise SystemExit(
            f"[gran-analysis] REFUSING level {level}: "
            f"{len(plan) - len(missing)}/{len(plan)} runs COMPLETE "
            f"(missing run indices {missing[:20]}"
            f"{' ...' if len(missing) > 20 else ''}). The registered card "
            f"analyzes each level on its FULL set of adapters — the "
            f"completeness interlock fires exactly once per level. Re-run "
            f"when the queue finishes, or pass --allow-partial for an "
            f"exploratory tooling check that will NEVER be reported.")
    if missing:
        sys.stderr.write(
            "\n*** PRE-REGISTRATION WARNING — PARTIAL LEVEL ***\n"
            f"{len(plan) - len(missing)}/{len(plan)} runs COMPLETE at "
            f"{level}. Every number produced here is EXPLORATORY ONLY.\n\n")
    recs, y = [], []
    for e in plan:
        d = run_dir_for(level, e["run_k"])
        if not (d / "COMPLETE").exists():
            continue
        recs.append({"run_dir": d, "run_index": e["run_k"],
                     "class_id": e["class_id"], "task": e["task"]})
        y.append(class_ids.index(e["class_id"]))
    return recs, np.asarray(y, dtype=np.int64), class_ids, parents, lm


# ── The label-space analysis (fixture-agnostic — the selftest calls it) ──


def analyze_label_space(records: list[dict], y: np.ndarray,
                        class_ids: list[str], class_parents: list[str],
                        *, scratch_dir: Path, representation: str,
                        n_permutations: int, seed: int, level_index: int,
                        svm_c: float = SVM_C, chunk_rows: int = 8,
                        proj_dim: int = 16, proj_seed: int = 0,
                        label: str = "", key: str | None = None,
                        vocab_readout=None, kv_mode: str | None = None) -> dict:
    """One (label space, representation) cell: d1's H1 pipeline + the
    ladder's kappa / D10 / parent-collapse additions. `key` names the
    output cell (defaults to the representation); the vocab arm passes
    key='vocab_signature' / 'vocab_signature_kv_exclude' with its readout
    and kv_mode, and draws the null stream reserved for that key."""
    key = key or representation
    classes = np.arange(len(class_ids))
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir / f"features_{key}_{label or 'x'}.dat"
    extra = {}
    if representation == "vocab_signature":
        if vocab_readout is None:
            raise ValueError("vocab_signature cell requires a vocab_readout")
        extra = {"vocab_readout": vocab_readout,
                 "kv_mode": kv_mode or vs.KV_MODE}
    res = d1.analyze_family(
        records, y, classes, class_ids, representation, scratch,
        n_permutations=n_permutations, seed=seed,
        family_index=level_index,
        rep_index=d1._REP_STREAM_INDEX[key],
        chunk_rows=chunk_rows, svm_c=svm_c, proj_dim=proj_dim,
        proj_seed=proj_seed, label=label or key, **extra)
    res["representation"] = representation
    res["kv_mode"] = kv_mode

    cm = np.asarray(res["confusion_matrix"]["rows_true_cols_pred"])
    kappa = cohens_kappa(cm)
    acc = res["loo_accuracy"]
    p = res["permutation"]["p_value"]
    res["cohens_kappa"] = kappa
    res["d10_lock"] = d10_lock(acc, p, kappa, len(class_ids))
    res["ceiling"] = {
        "definition": f"LOO acc >= {D5_CEILING} (<=2 errors at N=240) — "
                      f"D5 PINNED",
        "threshold": D5_CEILING,
        "at_ceiling": bool(acc >= D5_CEILING),
        "errors": int(len(y) - np.trace(cm)),
    }
    pcm, parents = collapse_confusion(cm, class_parents)
    res["parent_collapsed"] = {
        "parents": parents,
        "confusion_matrix": pcm.tolist(),
        "accuracy": float(np.trace(pcm) / max(pcm.sum(), 1)),
        "cohens_kappa": cohens_kappa(pcm),
        "note": "§5.4 — fine confusions collapsed onto the parent tasks: "
                "'loses fine structure, keeps coarse' vs 'loses everything'.",
    }
    return res


# ── D6: the data-space separability reference ───────────────────────


def _format_documents(task: str, row_ids: list[int], tokenizer) -> list[str]:
    """The formatted training documents for a set of raw rows — built by the
    unmodified task dataset class, so xsum/squad carry the same token-budget
    truncation the adapters trained on."""
    cls = TASK_REGISTRY[task]
    ds, _source = load_hf_dataset_with_fallback(
        cls.dataset_candidates, cls.dataset_config_name, cls.hf_split)
    subset = ds.select([int(i) for i in row_ids])
    built = cls(tokenizer, "train", data_seed=0, max_len=MAX_LEN,
                val_size=0, val_seed=VAL_SEED, pool_cap=POOL_CAP,
                raw=subset, keep_text=True)
    return list(built.formatted_texts)


def d6_reference(level: str, class_ids: list[str], out_dir: Path) -> dict:
    """TF-IDF + linear SVM on the training documents under the same labels.

    Ask 3 conditions, all three encoded here: (i) CV runs WITHIN the
    subsample, (ii) a class with fewer than the nominal 1,000 eligible
    examples contributes its whole pool and its realized n is reported,
    (iii) the same nominal 1,000 is used at every level.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.svm import LinearSVC
    from transformers import AutoTokenizer

    pools_path = LABELS_DIR / f"{level}_pools.json"
    if not pools_path.exists():
        raise SystemExit(f"[gran-analysis] no {pools_path} — run "
                         f"scripts/granularity_labels.py first.")
    pools = json.loads(pools_path.read_text(encoding="utf-8"))
    manifest = json.loads((LABELS_DIR / f"{level}.json")
                          .read_text(encoding="utf-8"))
    task_of = {c["class_id"]: c["task"] for c in manifest["classes"]}

    # Load from the local snapshot when one is cached: transformers 4.57's
    # tokenizer loader calls the Hub (model_info, a "base Mistral" check)
    # for a repo id even when every file is cached, which is refused under
    # HF_HUB_OFFLINE=1 and needs a token for a gated repo. A local path skips
    # that call; the files are the same snapshot (2026-09-05 dry run).
    try:
        tok_source = str(vs._resolve_snapshot(FAMILY["model"],
                                              vs.resolve_hf_hub_cache()))
    except Exception:  # noqa: BLE001 — no snapshot: fall back to the id
        tok_source = FAMILY["model"]
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    docs: list[str] = []
    labels: list[int] = []
    realized: dict[str, int] = {}
    for ci, cid in enumerate(class_ids):
        ids = pools["d6_subsample_ids"].get(cid)
        if not ids:
            raise SystemExit(f"[gran-analysis] class {cid} has no frozen D6 "
                             f"subsample in {pools_path}")
        texts = _format_documents(task_of[cid], ids, tokenizer)
        docs.extend(texts)
        labels.extend([ci] * len(texts))
        realized[cid] = len(texts)
        print(f"[d6] {level} {cid}: {len(texts)} documents", flush=True)

    y = np.asarray(labels, dtype=np.int64)
    vec = TfidfVectorizer(lowercase=D6_CONFIG["lowercase"],
                          sublinear_tf=D6_CONFIG["sublinear_tf"],
                          min_df=D6_CONFIG["min_df"],
                          max_features=D6_CONFIG["max_features"],
                          ngram_range=tuple(D6_CONFIG["ngram_range"]))
    X = vec.fit_transform(docs)
    cv = StratifiedKFold(n_splits=D6_CV_FOLDS, shuffle=True,
                         random_state=D6_CV_SEED)
    clf = LinearSVC(C=1.0, dual="auto", max_iter=5000)
    preds = cross_val_predict(clf, X, y, cv=cv)
    acc = float(np.mean(preds == y))
    cm, _recalls = d1.confusion_and_recalls(y, preds, np.arange(len(class_ids)))
    kappa = cohens_kappa(cm)
    k = len(class_ids)
    return {
        "config": D6_CONFIG,
        "k": k,
        "chance": 1.0 / k,
        "n_documents": int(len(docs)),
        "n_features": int(X.shape[1]),
        "realized_n_per_class": realized,
        "realized_n_min": min(realized.values()),
        "realized_n_max": max(realized.values()),
        "at_nominal_everywhere": all(v == D6_SUBSAMPLE_N
                                     for v in realized.values()),
        "cv_accuracy": acc,
        "cohens_kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "note": "Bounds attainable weight-space accuracy: a class whose own "
                "text is inseparable cannot be read from weights, so a "
                "failure there is label noise, not a canonicalization "
                "failure (design §6.1; converts outcome (C) from an excuse "
                "into a finding).",
    }


# ── Level orchestration ─────────────────────────────────────────────


def analyze_level(level: str, out_dir: Path, *, n_permutations: int,
                  seed: int, representations: tuple[str, ...],
                  allow_partial: bool, force_tier_order: bool,
                  skip_d6: bool) -> dict:
    fired_before = require_tier_order(level, force=force_tier_order)
    records, y, class_ids, parents, lm = collect_level_records(
        level, allow_partial)
    if not records:
        raise SystemExit(f"[gran-analysis] level {level} has no COMPLETE runs")

    level_index = list(LEVEL_TIER).index(level)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"

    clean_idx = [i for i, cid in enumerate(class_ids)
                 if _is_clean_core(level, cid)]
    results: dict = {
        "analysis": "granularity ladder — per-level label separability",
        "level": level,
        "generated_by": "scripts/granularity_analysis.py",
        "card": ["docs/REGISTRATION_GRANULARITY_2026-07-30.md",
                 "docs/LOCK_GRANULARITY_2026-08-04.md"],
        "git_commit": git_commit_hash(),
        "generated_at": utc_now(),
        "family": FAMILY["model"],
        "scope_limit": "llama3.2-1b only (D1 APPROVED) — this scope limit "
                       "belongs in every claim derived from this file.",
        "k": len(class_ids),
        "n_runs": len(records),
        "chance": 1.0 / len(class_ids),
        "classes": class_ids,
        "class_parents": parents,
        "clean_core_classes": [class_ids[i] for i in clean_idx],
        "level_manifest": {k: lm.get(k) for k in
                           ("kind", "seeds_per_class", "balance_policy",
                            "k_materialized", "k_declared", "note",
                            "source")},
        "parameters": {"n_permutations": n_permutations, "seed": seed,
                       "svm_c": SVM_C,
                       "representations": list(representations),
                       "permutation_stream": "default_rng([seed, "
                                             "level_index, rep_index]) — "
                                             "rep_index is d1's fixed map, "
                                             "so a level's null stream does "
                                             "not depend on which "
                                             "representations ran alongside"},
        "locks": {"d10": d10_lock(0.0, 1.0, 0.0, len(class_ids))["form"],
                  "d5_ceiling": D5_CEILING,
                  "d3_clean_core": "clean-core (T1+T2) reported alongside "
                                   "all-classes at every level; on material "
                                   "divergence the clean-core curve is the "
                                   "claim and all-classes is descriptive"},
        "allow_partial": bool(allow_partial),
        "exploratory_only": bool(allow_partial or force_tier_order),
        "all_classes": {},
        "clean_core": {},
    }

    cells = expand_reps(representations)
    vocab_readout = None
    if any(rep == "vocab_signature" for _, rep, _ in cells):
        vocab_readout, results["vocabsig_readout"] = build_vocab_readout()
        print(f"[gran-analysis] {level}: vocab readout loaded "
              f"({results['vocabsig_readout'].get('vocab_size')} x "
              f"{results['vocabsig_readout'].get('d_model')})", flush=True)

    for key, rep, kv in cells:
        print(f"[gran-analysis] {level} / {key}: all classes "
              f"(K={len(class_ids)}, n={len(records)})", flush=True)
        results["all_classes"][key] = analyze_label_space(
            records, y, class_ids, parents, scratch_dir=scratch,
            representation=rep, n_permutations=n_permutations, seed=seed,
            level_index=level_index, label=f"{level}-all-{key}", key=key,
            vocab_readout=vocab_readout, kv_mode=kv)

    if 0 < len(clean_idx) < len(class_ids):
        keep = {i for i in clean_idx}
        sub_records = [r for r, yy in zip(records, y) if int(yy) in keep]
        remap = {old: new for new, old in enumerate(clean_idx)}
        sub_y = np.array([remap[int(yy)] for yy in y if int(yy) in keep],
                         dtype=np.int64)
        sub_ids = [class_ids[i] for i in clean_idx]
        sub_parents = [parents[i] for i in clean_idx]
        for key, rep, kv in cells:
            print(f"[gran-analysis] {level} / {key}: clean core "
                  f"(K={len(sub_ids)}, n={len(sub_records)})", flush=True)
            results["clean_core"][key] = analyze_label_space(
                sub_records, sub_y, sub_ids, sub_parents, scratch_dir=scratch,
                representation=rep, n_permutations=n_permutations, seed=seed,
                level_index=level_index, label=f"{level}-clean-{key}", key=key,
                vocab_readout=vocab_readout, kv_mode=kv)
        results["clean_core_note"] = (
            f"K={len(sub_ids)} of {len(class_ids)} classes are T1+T2.")
    else:
        # Director condition (c), 2026-08-05, binding on every output: the
        # clean-core consequence is reported as "not testable", NEVER as
        # "passed". Since the G-5 amendment retired the annotator, the whole
        # ladder is T1+T2 and this is the expected branch at every level.
        results["clean_core_note"] = (
            "every class at this level is T1+T2 — NO T3 CELLS, so the D3 "
            "clean-core requirement is NOT TESTABLE at this level. This is "
            "not a statement that it passed: the clean-core and all-classes "
            "label spaces are identical by construction, so the divergence "
            "rule has nothing to bite on (G-4 principle, applied per "
            "Director condition (c) of 2026-08-05).")

    if not skip_d6:
        # Run per level, including L0 — the anchor needs its own reference
        # for the curve to be comparable (D6 PINNED: "run per level").
        results["d6_data_space_reference"] = d6_reference(
            level, class_ids, out_dir)

    _write(out_dir / f"{level}_results.json", results)
    _write_report(out_dir / f"{level}_REPORT.md", results)
    if not (allow_partial or force_tier_order):
        record_gate(level, fired_before, {"k": results["k"],
                                          "n_runs": results["n_runs"]})
    return results


def _is_clean_core(level: str, class_id: str) -> bool:
    if level == "L0":
        return True                     # the six native tasks
    manifest = json.loads((LABELS_DIR / f"{level}.json")
                          .read_text(encoding="utf-8"))
    for c in manifest["classes"]:
        if c["class_id"] == class_id:
            return c["tier"] in CLEAN_CORE_TIERS
    return False


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default),
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _typed(pairs) -> str:
    width = max(len(k) for k, _ in pairs)
    return "\n".join(f"{k.ljust(width)} = {v}" for k, v in pairs)


def _write_report(path: Path, r: dict) -> None:
    parts = [f"# GRANULARITY {r['level']} — LEVEL REPORT", "",
             "Produced by `scripts/granularity_analysis.py` on the "
             "`asset1_d1_identifiability` H1 machinery (imported, not "
             "forked). Scope limit: **llama3.2-1b only** — it belongs in "
             "every claim taken from this file.", ""]
    core = [("level", r["level"]), ("k", r["k"]), ("n_runs", r["n_runs"]),
            ("chance", f"{r['chance']:.6f}"),
            ("exploratory_only", str(r["exploratory_only"]).lower())]
    for rep, res in r["all_classes"].items():
        core += [(f"{rep}_acc", f"{res['loo_accuracy']:.4f}"),
                 (f"{rep}_kappa", f"{res['cohens_kappa']:.4f}"),
                 (f"{rep}_p", f"{res['permutation']['p_value']:.6f}"),
                 (f"{rep}_d10_pass", str(res["d10_lock"]["pass"]).lower()),
                 (f"{rep}_at_ceiling", str(res["ceiling"]["at_ceiling"]).lower())]
    if "raw" in r["all_classes"] and "canonical" in r["all_classes"]:
        delta = (r["all_classes"]["canonical"]["loo_accuracy"]
                 - r["all_classes"]["raw"]["loo_accuracy"])
        core.append(("delta_K_canonical_minus_raw", f"{delta:+.4f}"))
    d6 = r.get("d6_data_space_reference")
    if d6:
        core += [("d6_cv_accuracy", f"{d6['cv_accuracy']:.4f}"),
                 ("d6_kappa", f"{d6['cohens_kappa']:.4f}"),
                 ("d6_realized_n_min", d6["realized_n_min"])]
    parts += ["=== VERIFIED STATE ===", _typed(core),
              "=== END VERIFIED STATE ===", ""]

    for scope in ("all_classes", "clean_core"):
        if not r.get(scope):
            continue
        parts += [f"## {scope.replace('_', ' ').title()}", ""]
        for rep, res in r[scope].items():
            lock = res["d10_lock"]
            parts += [f"### {rep}", "",
                      _typed([("loo_accuracy", f"{res['loo_accuracy']:.4f}"),
                              ("wilson_ci_95",
                               f"[{res['wilson_ci_95'][0]:.4f}, "
                               f"{res['wilson_ci_95'][1]:.4f}]"),
                              ("cohens_kappa", f"{res['cohens_kappa']:.4f}"),
                              ("permutation_p",
                               f"{res['permutation']['p_value']:.6f}"),
                              ("null_mean",
                               f"{res['permutation']['null_mean']:.4f}"),
                              ("macro_f1", f"{res['macro_f1']:.4f}"),
                              ("acc_bar_1p5x_chance",
                               f"{lock['accuracy_threshold_1p5x_chance']:.4f}"),
                              ("D10_accuracy", str(
                                  lock["accuracy_exceeds_threshold"]).lower()),
                              ("D10_p", str(lock["p_below_threshold"]).lower()),
                              ("D10_kappa", str(
                                  lock["kappa_meets_floor"]).lower()),
                              ("D10_PASS", str(lock["pass"]).upper()),
                              ("at_ceiling_0p99",
                               str(res["ceiling"]["at_ceiling"]).lower()),
                              ("errors", res["ceiling"]["errors"]),
                              ("parent_collapsed_accuracy",
                               f"{res['parent_collapsed']['accuracy']:.4f}"),
                              ("feature_dim", res["feature_dim"])]), ""]
    parts += ["## Clean core (D3)", "", r["clean_core_note"], ""]
    if d6:
        parts += ["## D6 data-space reference", "",
                  _typed([("k", d6["k"]),
                          ("n_documents", d6["n_documents"]),
                          ("n_features", d6["n_features"]),
                          ("cv_accuracy", f"{d6['cv_accuracy']:.4f}"),
                          ("cohens_kappa", f"{d6['cohens_kappa']:.4f}"),
                          ("realized_n_min", d6["realized_n_min"]),
                          ("realized_n_max", d6["realized_n_max"]),
                          ("at_nominal_everywhere",
                           str(d6["at_nominal_everywhere"]).lower()),
                          ("cv", d6["config"]["cv"])]), "",
                  d6["note"], ""]
    parts += ["## Provenance", "",
              _typed([("git_commit", r["git_commit"][:12]),
                      ("generated_at_utc", r["generated_at"]),
                      ("n_permutations",
                       r["parameters"]["n_permutations"]),
                      ("seed", r["parameters"]["seed"]),
                      ("svm_c", r["parameters"]["svm_c"]),
                      ("tier_order", " -> ".join(TIER_ORDER))]), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# ── The curve across levels (§5.1–5.3) ──────────────────────────────


def build_curve(out_dir: Path) -> dict:
    rows = []
    for level in ["L0", "L1", "B2", "B4", "B8", "B16", "L2", "L3"]:
        p = out_dir / f"{level}_results.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        row = {"level": level, "k": r["k"], "n_runs": r["n_runs"],
               "chance": r["chance"]}
        for scope in ("all_classes", "clean_core"):
            for rep, res in (r.get(scope) or {}).items():
                row[f"{scope}.{rep}.acc"] = res["loo_accuracy"]
                row[f"{scope}.{rep}.kappa"] = res["cohens_kappa"]
                row[f"{scope}.{rep}.p"] = res["permutation"]["p_value"]
                row[f"{scope}.{rep}.d10"] = res["d10_lock"]["pass"]
        if ("all_classes.raw.acc" in row
                and "all_classes.canonical.acc" in row):
            row["delta_K"] = (row["all_classes.canonical.acc"]
                              - row["all_classes.raw.acc"])
        d6 = r.get("d6_data_space_reference")
        if d6:
            row["d6.acc"] = d6["cv_accuracy"]
            row["d6.kappa"] = d6["cohens_kappa"]
        rows.append(row)
    departure = next((r["level"] for r in rows
                      if r.get("all_classes.canonical.acc") is not None
                      and r["all_classes.canonical.acc"] < D5_CEILING), None)
    curve = {
        "generated_by": "scripts/granularity_analysis.py --curve",
        "generated_at": utc_now(),
        "git_commit": git_commit_hash(),
        "levels": rows,
        "confirmatory_endpoints": {
            "ceiling_departure_level": departure,
            "definition": f"first level whose canonical LOO acc < "
                          f"{D5_CEILING} (D5 PINNED)",
            "delta_K_trend": [{"level": r["level"], "k": r["k"],
                               "delta_K": r.get("delta_K")} for r in rows],
        },
        "scope_limit": "llama3.2-1b only (D1).",
        "note": "Accuracy is not comparable across K; kappa is reported at "
                "every level for that reason (D5). Levels appear here only "
                "once their tier gate has fired.",
    }
    _write(out_dir / "GRANULARITY_CURVE.json", curve)
    lines = ["# GRANULARITY LADDER — CURVE", "",
             "Levels appear only once their tier gate has fired "
             "(`results/granularity/TIER_GATES.json`). Scope limit: "
             "**llama3.2-1b only**.", "",
             "| level | K | n | chance | raw acc | canon acc | Δ(K) | "
             "canon κ | D10 | D6 acc |", "|---|---|---|---|---|---|---|---|"
             "---|---|"]
    for r in rows:
        def g(key, fmt="{:.4f}"):
            v = r.get(key)
            return fmt.format(v) if isinstance(v, float) else "—"
        lines.append(
            f"| {r['level']} | {r['k']} | {r['n_runs']} | "
            f"{r['chance']:.4f} | {g('all_classes.raw.acc')} | "
            f"{g('all_classes.canonical.acc')} | {g('delta_K', '{:+.4f}')} | "
            f"{g('all_classes.canonical.kappa')} | "
            f"{str(r.get('all_classes.canonical.d10', '—')).upper()} | "
            f"{g('d6.acc')} |")
    lines += ["", f"Ceiling departure (canonical < {D5_CEILING}): "
                  f"**{departure or 'none observed yet'}**", ""]
    (out_dir / "GRANULARITY_CURVE.md").write_text("\n".join(lines),
                                                  encoding="utf-8",
                                                  newline="\n")
    return curve


# ── Synthetic self-test (never touches a real level) ────────────────


def run_selftest(out_dir: Path, n_permutations: int = 200) -> dict:
    """Acceptance test on synthetic fixtures — the Asset-1 pattern.

    Builds miniature banks with `asset1_synth.make_synthetic_bank` (the real
    on-disk adapter schema) at task_effect 1.0 and 0.0 and asserts that the
    ladder's additions behave: the planted label space passes the D10 triple
    lock and the null label space fails it; kappa matches a hand-computed
    value; the parent collapse is exact; and the D6 TF-IDF path recovers a
    planted lexical signal and sits at chance on noise.
    """
    import asset1_synth as synth

    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    # 1. Kappa against a hand-computed matrix.
    cm = np.array([[8, 2], [3, 7]])       # po=0.75, pe=0.5 -> kappa=0.5
    k_hand = cohens_kappa(cm)
    checks.append({"check": "cohens_kappa vs hand computation",
                   "expected": 0.5, "got": k_hand,
                   "pass": bool(abs(k_hand - 0.5) < 1e-12)})
    checks.append({"check": "kappa of a chance-level matrix ~ 0",
                   "got": cohens_kappa(np.full((4, 4), 10)),
                   "pass": bool(abs(cohens_kappa(np.full((4, 4), 10)))
                                < 1e-12)})

    # 2. Parent collapse is exact.
    fine = np.array([[5, 1, 0, 0], [1, 5, 0, 0],
                     [0, 0, 4, 2], [0, 0, 2, 4]])
    pcm, parents = collapse_confusion(fine, ["a", "a", "b", "b"])
    checks.append({"check": "parent collapse sums blocks",
                   "expected": [[12, 0], [0, 12]], "got": pcm.tolist(),
                   "pass": bool(pcm.tolist() == [[12, 0], [0, 12]])
                   and parents == ["a", "b"]})

    # 3. Planted vs null label space through the real H1 pipeline.
    for tag, effect, expect_pass in (("effect", 1.0, True),
                                     ("null", 0.0, False)):
        bank = out_dir / f"synthbank_{tag}"
        info = synth.make_synthetic_bank(
            bank, n_families=1, n_tasks=4, n_reps=6, n_layers=2,
            d_model=16, rank=4, n_channels=2, task_effect=effect, seed=7)
        recs = list(aio.iter_runs(bank))
        tasks = sorted({r["task"] for r in recs})
        y = np.array([tasks.index(r["task"]) for r in recs], dtype=np.int64)
        res = analyze_label_space(
            recs, y, tasks, tasks, scratch_dir=out_dir / "scratch",
            representation="raw", n_permutations=n_permutations, seed=0,
            level_index=0, label=f"selftest-{tag}")
        lock = res["d10_lock"]
        checks.append({
            "check": f"D10 triple lock on the {tag} bank",
            "expected_pass": expect_pass, "got_pass": lock["pass"],
            "accuracy": res["loo_accuracy"], "kappa": res["cohens_kappa"],
            "p": res["permutation"]["p_value"], "n_runs": info["n_runs"],
            "pass": bool(lock["pass"] == expect_pass)})

    # 4. The D6 path: planted lexical signal vs noise.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.svm import LinearSVC
    rng = np.random.default_rng(7)
    vocab = [f"w{i}" for i in range(60)]
    for tag, planted in (("signal", True), ("noise", False)):
        docs, ys = [], []
        for c in range(4):
            for _ in range(60):
                body = " ".join(rng.choice(vocab, size=25))
                docs.append(f"marker{c} {body}" if planted else body)
                ys.append(c)
        X = TfidfVectorizer(min_df=1).fit_transform(docs)
        preds = cross_val_predict(
            LinearSVC(C=1.0, dual="auto", max_iter=5000), X,
            np.asarray(ys), cv=StratifiedKFold(n_splits=5, shuffle=True,
                                               random_state=D6_CV_SEED))
        acc = float(np.mean(preds == np.asarray(ys)))
        ok = acc > 0.95 if planted else acc < 0.45
        checks.append({"check": f"D6 TF-IDF+LinearSVC on {tag} documents",
                       "accuracy": acc,
                       "expected": ">0.95" if planted else "<0.45",
                       "pass": bool(ok)})

    verdict = all(c["pass"] for c in checks)
    payload = {"selftest": "granularity_analysis",
               "generated_at": utc_now(), "git_commit": git_commit_hash(),
               "n_permutations": n_permutations,
               "checks": checks, "verdict": "PASS" if verdict else "FAIL"}
    _write(out_dir / "SELFTEST.json", payload)
    for c in checks:
        detail = ""
        if "accuracy" in c:
            detail = f"  acc={c['accuracy']:.4f}"
        elif "got" in c:
            detail = f"  got={c['got']}"
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}{detail}")
    print(f"[selftest] verdict: {payload['verdict']}")
    return payload


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-level granularity analysis on the Asset-1 H1 "
                    "machinery, with the ladder's kappa / D10 / D3 / D6 "
                    "additions and both interlocks.")
    ap.add_argument("--level", choices=list(LEVEL_TIER))
    ap.add_argument("--out-dir", type=Path, default=ANALYSIS_ROOT)
    ap.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--representation", default="all",
                    choices=["raw", "canonical", "vocab_signature", "both",
                             "all"],
                    help="'all' = the three registered representations "
                         "(vocab_signature runs both kv modes); 'both' = "
                         "raw + canonical, the pre-2026-09-05 default")
    ap.add_argument("--allow-partial", action="store_true",
                    help="EXPLORATORY ONLY — analyze an incomplete level")
    ap.add_argument("--force-tier-order", action="store_true",
                    help="EXPLORATORY ONLY — bypass the frozen tier order")
    ap.add_argument("--skip-d6", action="store_true",
                    help="skip the data-space reference (tooling checks)")
    ap.add_argument("--curve", action="store_true",
                    help="rebuild the across-level curve from level results")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic acceptance test; touches no real level")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if "asset1-bank" in out_dir.resolve().parts:
        raise SystemExit(f"[gran-analysis] REFUSING --out-dir {out_dir}: "
                         f"inside the Asset-1 bank tree.")

    if args.selftest:
        payload = run_selftest(OUT_ROOT / "selftest",
                               n_permutations=min(args.n_permutations, 200))
        return 0 if payload["verdict"] == "PASS" else 1
    if args.curve:
        build_curve(out_dir)
        print(f"[gran-analysis] wrote {out_dir / 'GRANULARITY_CURVE.md'}")
        return 0
    if not args.level:
        ap.error("one of --level, --curve, --selftest is required")

    reps = {"both": ("raw", "canonical"),
            "all": REPRESENTATIONS}.get(args.representation,
                                        (args.representation,))
    r = analyze_level(args.level, out_dir, n_permutations=args.n_permutations,
                      seed=args.seed, representations=reps,
                      allow_partial=args.allow_partial,
                      force_tier_order=args.force_tier_order,
                      skip_d6=args.skip_d6)
    print(f"[gran-analysis] {args.level}: K={r['k']} n={r['n_runs']} -> "
          f"{out_dir / f'{args.level}_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
