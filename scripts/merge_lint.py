"""Merge linter — will the MIDPOINT merge of two adapters degrade? (D3 prototype).

Stream A (open) tooling built from the Asset-1 D3 machinery. Given TWO
adapter_state.pt files in the bank's flat format (same family), the linter
computes the D3 weight-only pair features by CALLING asset1_d3_merge
(import, never reimplement), scores them with a reference conflict model
fit at startup from the SHIPPED bank data, and reports the conflict
probability together with the family-matched bank AUC context and an
honest scope banner. It answers exactly one question: "if I average these
two adapters at alpha = 0.5, is the merge likely to degrade either
endpoint?" — nothing more.

The reference model (what it is, and what it is NOT)
----------------------------------------------------
Fit at startup, deterministically (pinned seed 0), from the delivery
bundle (--bank-dir, default results/asset1-delivery-verify):

    features  d3_pairs.json  — the shipped D3 pair-feature dicts (240
              pairs; exact 'full'/'distance' sets defined by
              asset1_d3_merge.assemble_matrix — this module never builds
              a feature by hand)
    labels    d3_labels.json — per-endpoint val perplexities; PRIMARY
              5%-rule binarization via asset1_d3_merge.binarize_primary
              (conflict iff the midpoint merge degrades EITHER endpoint
              by >= 5% relative vs that endpoint's native adapter —
              Director D3 override)
    model     per family: StandardScaler + L2 logistic regression
              (C=1.0, max_iter=2000, random_state=0) on the 'full'
              feature matrix, fit on ALL of that family's pairs — the
              same estimator asset1_d3_merge.oof_scores uses, minus the
              cross-validation.

BANK-TRAINED means exactly this envelope, and the linter says so loudly:

    * 2 families ONLY (qwen2.5-1.5b, llama3.2-1b) — 1–1.5B-scale
      RhombiLoRA adapters in the bank flat format, 6 text tasks;
    * MIDPOINT merges (alpha = 0.5) only — no other merge method or
      mixing weight has ever produced a label;
    * labels from val-loss/perplexity degradation ONLY (no task metrics,
      no human judgment);
    * ~86% of bank pairs are conflicts under the 5% rule — the
      probability inherits that skew and is NOT calibrated for any other
      population.

Anything outside the envelope — other families, scales, ranks, merge
methods, alphas, domains — is EXTRAPOLATION and is labeled as such.

Family matching, fallback, and refusal
--------------------------------------
The input pair's family is matched against the bank families by module
NAME SET (sorted module names must equal a bank family's module_names
exactly). Known limitation: the shipped reference features carry no shape
information, so an input that reuses a bank family's module names at a
different scale/shape would silently extrapolate through the full model —
the linter cannot detect that case from the reference bundle alone.

Out-of-family inputs (module set matches NO bank family) cannot go
through a per-family full model at all (the per-module feature blocks are
family-specific); the linter falls back to the 2-feature distance-only
model pooled over both bank families, prints a loud extrapolation
banner, and quotes the (much weaker) distance-only bank AUCs as the only
honest context.

Cross-family PAIRS (the two inputs' module sets differ from EACH OTHER,
or share names at mismatched shapes) are REFUSED outright (exit code 2):
D3 pair features are defined for same-family pairs only, and a midpoint
average of mismatched adapters is not a merge. The refusal check runs
before any bank data is touched.

AUC context
-----------
The quoted AUC context is read from the shipped d3_report.json (the
group-aware-CV headline — StratifiedGroupKFold over run-overlap
components; see asset1_d3_merge's DEPENDENCE_NOTE). The reference model
itself is fit on ALL pairs with no held-out set, so the AUC context is
the bank experiment's number, NOT a performance guarantee for the input
pair. A missing d3_report.json degrades to "no AUC context available".

Safety
------
Strictly read-only: loads two adapter files and the reference bundle,
writes nothing anywhere. CPU-only (map_location='cpu' via the production
loader; no CUDA is ever touched).

Usage
-----
    python scripts/merge_lint.py A/adapter_state.pt B/adapter_state.pt \
        [--bank-dir results/asset1-delivery-verify] [--json]

Exit codes: 0 = linted; 2 = refused (cross-family pair / CLI error).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asset1_d3_merge as d3m  # noqa: E402
from asset1_canonicalize import load_adapter_modules  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────

SEED = 0                     # pinned -- the reference fit is deterministic
LOGISTIC_C = 1.0             # match asset1_d3_merge.oof_scores defaults
LOGISTIC_MAX_ITER = 2000
BANK_ALPHA = 0.5             # the only merge the labels ever measured
DEFAULT_BANK_DIR = REPO_ROOT / "results" / "asset1-delivery-verify"

SCOPE_BANNER = """\
SCOPE -- the reference model is BANK-TRAINED, and only that:
  * training data: midpoint (alpha=0.5) merges from the Asset-1 bank --
    2 families only (qwen2.5-1.5b, llama3.2-1b; 1-1.5B-scale RhombiLoRA
    adapters in the bank flat format), 6 text tasks;
  * labels: val-loss/perplexity degradation ONLY -- conflict = merge
    degrades EITHER endpoint by >= 5% relative vs its native adapter
    (Director D3 override); most bank pairs are conflicts under this
    rule, so the probability inherits that skew and is NOT calibrated
    for other populations;
  * the model is fit on ALL bank pairs (no held-out set); the quoted AUC
    is the bank experiment's group-aware-CV number from the shipped
    d3_report.json, NOT a performance guarantee for this input pair;
  * family matching is by module NAME SET only -- same-named adapters at
    a different scale/shape would extrapolate undetected;
  * anything outside this envelope (other families, scales, ranks, merge
    methods, alphas, domains) is EXTRAPOLATION."""

EXTRAPOLATION_BANNER = """\
!!! EXTRAPOLATION -- module set matches NO bank family !!!
  The reference model has never seen adapters shaped like these. Falling
  back to the 2-feature distance-only model pooled over both bank
  families -- substantially weaker even in-bank (see the distance-only
  AUCs below). Treat the probability as a rough, uncalibrated hint at
  best; a real answer requires evaluating the merge."""


class CrossFamilyPairError(ValueError):
    """The two input adapters are not a same-family pair."""


# ── Input pair loading / refusal ────────────────────────────────────


def check_pair_compatible(mods_a: dict, mods_b: dict) -> None:
    """Refuse cross-family pairs BEFORE any feature work.

    Same-family means: identical module name sets AND identical tensor
    shapes per module field. D3 pair features (and the midpoint merge
    itself) are undefined otherwise -- Qwen and Llama do not even share
    module names.
    """
    names_a, names_b = set(mods_a), set(mods_b)
    if names_a != names_b:
        only_a = sorted(names_a - names_b)[:3]
        only_b = sorted(names_b - names_a)[:3]
        raise CrossFamilyPairError(
            f"module sets differ ({len(names_a)} vs {len(names_b)} "
            f"modules; e.g. only in A: {only_a}, only in B: {only_b}) -- "
            f"this is a CROSS-FAMILY pair. D3 pair features and midpoint "
            f"merges are defined for same-family pairs only.")
    for name in sorted(names_a):
        ea, eb = mods_a[name], mods_b[name]
        for field in ("lora_A", "lora_B", "bridge"):
            ta, tb = ea.get(field), eb.get(field)
            if ta is None or tb is None:
                continue
            if tuple(ta.shape) != tuple(tb.shape):
                raise CrossFamilyPairError(
                    f"module {name!r} field {field!r}: shape mismatch "
                    f"{tuple(ta.shape)} vs {tuple(tb.shape)} -- same module "
                    f"names but different geometry; not a same-family "
                    f"pair, refusing.")


def featurize_pair(path_a: str | Path, path_b: str | Path
                   ) -> tuple[dict, dict, dict]:
    """Load both adapters (production loader) and compute the D3 features.

    Loading goes through asset1_canonicalize.load_adapter_modules; the
    features come from asset1_d3_merge.pair_features UNCHANGED -- the
    linter adds no feature of its own. Raises CrossFamilyPairError for
    incompatible pairs (checked before feature work).
    """
    mods_a = load_adapter_modules(path_a)
    mods_b = load_adapter_modules(path_b)
    check_pair_compatible(mods_a, mods_b)
    try:
        pf = d3m.pair_features(mods_a, mods_b)
    except ValueError as e:            # belt-and-braces: d3m's own guard
        raise CrossFamilyPairError(str(e)) from e
    return mods_a, mods_b, pf


# ── Reference model (shipped bank data) ─────────────────────────────


def load_reference(bank_dir: str | Path) -> dict:
    """Load the shipped bank data: features + labels (+ AUC report).

    Reads d3_pairs.json (stored pair-feature dicts) and d3_labels.json
    (per-endpoint metrics; parsed by asset1_d3_merge.load_labels and
    binarized by binarize_primary -- the PRIMARY 5% rule). Pairs and
    labels are joined strictly by (family, task_a, run_a, task_b, run_b).
    d3_report.json is optional context (None when absent).

    Returns {"families": {fam: {"features": [...], "y": ndarray,
    "module_names": [...], "n_pairs": int}}, "binarization": meta,
    "alpha": float, "report": dict | None, "n_pairs": int}.
    """
    bank_dir = Path(bank_dir)
    pairs_path = bank_dir / "d3_pairs.json"
    labels_path = bank_dir / "d3_labels.json"
    for p in (pairs_path, labels_path):
        if not p.exists():
            raise FileNotFoundError(
                f"reference bank data not found: {p} -- pass --bank-dir "
                f"pointing at a directory with d3_pairs.json and "
                f"d3_labels.json (default: {DEFAULT_BANK_DIR})")

    payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    if "pairs" not in payload or not payload["pairs"]:
        raise ValueError(f"{pairs_path}: no pair records")
    rows = d3m.load_labels(labels_path)
    y_all, bin_meta = d3m.binarize_primary(rows)

    def _key(r: dict) -> tuple:
        return (r["family_short"], r["task_a"], int(r["run_index_a"]),
                r["task_b"], int(r["run_index_b"]))

    label_index = {_key(r): i for i, r in enumerate(rows)}
    families: dict[str, dict] = {}
    for rec in payload["pairs"]:
        key = _key(rec)
        if key not in label_index:       # tolerate swapped orientation
            key = (rec["family_short"], rec["task_b"],
                   int(rec["run_index_b"]), rec["task_a"],
                   int(rec["run_index_a"]))
        if key not in label_index:
            raise ValueError(
                f"pair {_key(rec)} has no matching row in {labels_path} "
                f"-- pairs and labels are out of sync")
        fam = rec["family_short"]
        slot = families.setdefault(
            fam, {"features": [], "y": [],
                  "module_names": rec["features"]["module_names"]})
        if rec["features"]["module_names"] != slot["module_names"]:
            raise ValueError(f"family {fam!r}: inconsistent module_names "
                             f"across stored pairs")
        slot["features"].append(rec["features"])
        slot["y"].append(int(y_all[label_index[key]]))
    for fam, slot in families.items():
        slot["y"] = np.asarray(slot["y"], dtype=int)
        slot["n_pairs"] = len(slot["features"])

    report_path = bank_dir / "d3_report.json"
    report = (json.loads(report_path.read_text(encoding="utf-8"))
              if report_path.exists() else None)
    return {"families": families, "binarization": bin_meta,
            "alpha": float(payload.get("alpha", BANK_ALPHA)),
            "report": report,
            "n_pairs": int(sum(f["n_pairs"] for f in families.values()))}


def _pipeline(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER,
                           random_state=seed))


def fit_reference(reference: dict, seed: int = SEED) -> dict:
    """Fit the reference models from the loaded bank data (deterministic).

    Per family: the 'full' feature matrix (assemble_matrix -- the exact
    bank feature set) -> StandardScaler + logistic. Families whose labels
    are single-class are recorded as unfittable, not silently dropped.
    Plus ONE pooled distance-only ([cos_distance, l2_distance]) model over
    all families -- the out-of-family fallback (those two features are the
    only family-agnostic ones).
    """
    fitted: dict = {"families": {}, "distance_fallback": None,
                    "seed": seed}
    all_feats: list[dict] = []
    all_y: list[np.ndarray] = []
    for fam in sorted(reference["families"]):
        slot = reference["families"][fam]
        y = slot["y"]
        all_feats.extend(slot["features"])
        all_y.append(y)
        if np.unique(y).size < 2:
            fitted["families"][fam] = {
                "pipeline": None, "n_pairs": slot["n_pairs"],
                "note": "single-class labels -- full model unfittable"}
            continue
        X, _ = d3m.assemble_matrix(slot["features"], "full")
        pipe = _pipeline(seed)
        pipe.fit(X, y)
        fitted["families"][fam] = {"pipeline": pipe,
                                   "n_pairs": slot["n_pairs"],
                                   "note": None}
    y_pool = np.concatenate(all_y)
    if np.unique(y_pool).size >= 2:
        X_dist, _ = d3m.assemble_matrix(all_feats, "distance")
        pipe = _pipeline(seed)
        pipe.fit(X_dist, y_pool)
        fitted["distance_fallback"] = pipe
    return fitted


# ── Lint verdict ────────────────────────────────────────────────────


def _family_context(reference: dict, fam: str) -> dict | None:
    """Family-matched AUC context from the shipped d3_report.json."""
    report = reference.get("report")
    if not report:
        return None
    rep = (report.get("per_family") or {}).get(fam)
    if not rep:
        return {"note": f"family {fam!r} not in d3_report.json"}
    if "auc_full" not in rep:
        return {"note": "no group-aware headline for this family in "
                        "d3_report.json (group CV infeasible)"}
    return {"basis": "group-aware CV (shipped d3_report.json)",
            "auc_full": rep["auc_full"],
            "auc_full_ci": rep.get("auc_full_ci"),
            "auc_distance": rep["auc_distance"],
            "auc_distance_ci": rep.get("auc_distance_ci"),
            "n_pairs": rep.get("n_pairs")}


def _fallback_context(reference: dict) -> dict:
    """Distance-only AUCs per family -- the only honest out-of-family
    context (no pooled distance-only AUC was ever pre-registered)."""
    report = reference.get("report")
    ctx: dict = {"note": ("distance-only bank AUCs per family, group-aware "
                          "CV -- the fallback model's in-bank ceiling")}
    if not report:
        ctx["per_family_auc_distance"] = None
        ctx["note"] = "no d3_report.json -- no AUC context available"
        return ctx
    ctx["per_family_auc_distance"] = {
        fam: rep.get("auc_distance")
        for fam, rep in (report.get("per_family") or {}).items()
        if isinstance(rep, dict)}
    return ctx


def _finite_or_none(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def lint_pair(pf: dict, reference: dict, fitted: dict) -> dict:
    """Score one featurized pair against the reference models.

    In-family (module_names equal a bank family's): per-family full
    model. Out-of-family: pooled distance-only fallback + extrapolation
    flag. Returns the verdict dict (JSON-safe scalars only).
    """
    fam_match = next(
        (fam for fam, slot in reference["families"].items()
         if pf["module_names"] == slot["module_names"]), None)

    verdict: dict = {
        "n_modules": len(pf["module_names"]),
        "alpha": reference["alpha"],
        "binarization": {
            "rule_used": reference["binarization"].get("rule_used"),
            "threshold_rel": reference["binarization"].get("threshold_rel"),
            "frac_positive_bank":
                reference["binarization"].get("frac_positive_relative"),
        },
        "features": {k: _finite_or_none(pf[k])
                     for k in ("cos_distance", "l2_distance",
                               *d3m.AGGREGATE_KEYS)},
        "seed": fitted["seed"],
    }

    fam_slot = fitted["families"].get(fam_match) if fam_match else None
    if fam_match is not None and fam_slot and fam_slot["pipeline"]:
        X, _ = d3m.assemble_matrix([pf], "full")
        assert pf["module_names"] == \
            reference["families"][fam_match]["module_names"]
        prob = float(fam_slot["pipeline"].predict_proba(X)[0, 1])
        verdict.update({
            "in_family": True,
            "family": fam_match,
            "model_used": "full",
            "n_reference_pairs": fam_slot["n_pairs"],
            "conflict_probability": prob,
            "bank_context": _family_context(reference, fam_match),
            "extrapolation": None,
        })
        return verdict

    # Out-of-family (or the matched family's full model was unfittable):
    # the loud path.
    if fitted["distance_fallback"] is None:
        raise ValueError(
            "no usable reference model: the input matches no fittable "
            "bank family and the pooled distance-only fallback could not "
            "be fit (single-class reference labels)")
    X, _ = d3m.assemble_matrix([pf], "distance")
    prob = float(fitted["distance_fallback"].predict_proba(X)[0, 1])
    reason = ("module set matches NO bank family"
              if fam_match is None else
              f"family {fam_match!r} matched but its full model is "
              f"unfittable ({fam_slot['note']})")
    verdict.update({
        "in_family": fam_match is not None,
        "family": fam_match,
        "model_used": "distance_only_fallback",
        "n_reference_pairs": reference["n_pairs"],
        "conflict_probability": prob,
        "bank_context": _fallback_context(reference),
        "extrapolation": reason,
    })
    return verdict


# ── Presentation ────────────────────────────────────────────────────


def _fmt_ci(ci) -> str:
    if not ci or ci[0] is None:
        return ""
    return f" [CI {ci[0]:.3f}, {ci[1]:.3f}]"


def format_verdict(verdict: dict, path_a: str, path_b: str) -> str:
    bank_families = "qwen2.5-1.5b, llama3.2-1b"
    lines = [
        "=== MERGE LINT -- Asset-1 D3 weight-only conflict check ===",
        f"adapter A: {path_a}",
        f"adapter B: {path_b}",
        f"modules:   {verdict['n_modules']}",
    ]
    if verdict["extrapolation"] is not None:
        lines += ["", EXTRAPOLATION_BANNER, "",
                  f"  reason: {verdict['extrapolation']}"]
    else:
        lines.append(f"family:    {verdict['family']} (bank family -- "
                     f"module set exact match)")
    b = verdict["binarization"]
    thr = b["threshold_rel"]
    frac = b["frac_positive_bank"]
    lines += [
        "",
        f"CONFLICT PROBABILITY (midpoint merge, alpha="
        f"{verdict['alpha']:g}): {verdict['conflict_probability']:.4f}",
        f"  model: {verdict['model_used']} "
        f"(fit on {verdict['n_reference_pairs']} bank pairs, "
        f"seed {verdict['seed']})",
    ]
    if thr is not None:
        lines.append(
            f"  conflict = merge degrades EITHER endpoint by >= "
            f"{thr:.0%} relative vs its native adapter "
            f"({b['rule_used']} rule)")
    if frac is not None:
        lines.append(f"  bank base rate: {frac:.1%} of bank pairs are "
                     f"conflicts under this rule")

    ctx = verdict["bank_context"]
    lines.append("")
    if ctx is None:
        lines.append("bank AUC context: no d3_report.json in the bank "
                     "dir -- none available")
    elif "auc_full" in ctx:
        lines += [
            f"bank AUC context (family {verdict['family']}, {ctx['basis']}):",
            f"  full features {ctx['auc_full']:.3f}"
            f"{_fmt_ci(ctx.get('auc_full_ci'))} vs distance-only "
            f"{ctx['auc_distance']:.3f}"
            f"{_fmt_ci(ctx.get('auc_distance_ci'))}",
            f"  (n={ctx.get('n_pairs')} bank pairs; NOT a guarantee for "
            f"this input pair)",
        ]
    else:
        lines.append(f"bank AUC context: {ctx.get('note')}")
        per_fam = ctx.get("per_family_auc_distance") or {}
        for fam, auc in sorted(per_fam.items()):
            if auc is not None:
                lines.append(f"  {fam}: distance-only AUC {auc:.3f}")

    lines += ["", SCOPE_BANNER,
              f"  (bank families: {bank_families})"]
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge linter -- predict whether the MIDPOINT "
                    "(alpha=0.5) merge of two same-family bank-format "
                    "adapters will degrade, using the Asset-1 D3 "
                    "weight-only features and a reference model fit from "
                    "the shipped bank data. Read-only; refuses "
                    "cross-family pairs (exit 2).")
    parser.add_argument("adapter_a", type=Path,
                        help="first adapter_state.pt (bank flat format)")
    parser.add_argument("adapter_b", type=Path,
                        help="second adapter_state.pt (same family)")
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR,
                        help="directory with the shipped reference data "
                             "(d3_pairs.json + d3_labels.json, optional "
                             "d3_report.json) "
                             "(default: results/asset1-delivery-verify)")
    parser.add_argument("--json", action="store_true",
                        help="emit the verdict as JSON instead of text")
    args = parser.parse_args(argv)

    # Load + refusal check FIRST -- a cross-family pair is refused before
    # any reference data is read (works with no bank present).
    try:
        _, _, pf = featurize_pair(args.adapter_a, args.adapter_b)
    except CrossFamilyPairError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        raise SystemExit(2)

    reference = load_reference(args.bank_dir)
    fitted = fit_reference(reference, seed=SEED)
    verdict = lint_pair(pf, reference, fitted)

    if args.json:
        print(json.dumps(verdict, indent=2))
    else:
        print(format_verdict(verdict, str(args.adapter_a),
                             str(args.adapter_b)))


if __name__ == "__main__":
    main()
