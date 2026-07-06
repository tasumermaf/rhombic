"""Asset 1 — D3: WEIGHT-ONLY merge-degradation prediction (post-hoc, no training access).

D3 predicts post-merge degradation from the weights of two adapters ALONE —
post hoc, with no access to training data, gradients, optimizer state, or
the training loop. arXiv:2606.19549 (Jun 17) owns the training-time form of
merge prediction; this tool is deliberately scoped to the weight-only,
after-the-fact regime, per the locked experiment card (2026-07-03). Labels
(actual post-merge degradation) exist only after the post-bank GPU
evaluations, so this module ships the complete feature extractor, merge
constructor, and prediction harness NOW, and consumes a labels file LATER
(schema below). All development validation is synthetic (--selftest); the
real-bank path is gated by asset1_analysis_io.require_complete_bank().

Features per same-family adapter pair
-------------------------------------
(a) Flattened-parameter distances (D1 fingerprint distance): cosine distance
    and L2 distance on asset1_analysis_io.flatten_features raw vectors
    (sorted modules; lora_A / lora_B / bridge order; raw stored values).
(b) Gauge-invariant principal angles per module, on the column spaces of the
    effective updates. For each module, effective_factors (asset1_canonicalize)
    absorbs the bridge and alpha/rank scaling: B_eff = scaling * (B @ E),
    A_eff = A, so DW = B_eff @ A_eff is the TRUE update. col(DW) =
    col(B_eff @ A_eff) ⊆ col(B_eff), with equality when A_eff has full row
    rank (generic for trained adapters) — so angles are computed between
    col(B_eff^i) and col(B_eff^j) WITHOUT materializing the d_out × d_in
    update. Gauge invariance: any GL(r) reparameterization of the absorbed
    factors (B_eff G, G^{-1} A_eff) leaves col(B_eff) unchanged, so the
    angles are invariant (verified to 1e-10 in tests). A corollary worth
    stating: an INVERTIBLE bridge is itself such a gauge — col(B @ E) =
    col(B) for invertible expanded E — so the angle block sees the bridge
    only through rank/degeneracy effects; the bridge's numeric content
    reaches the classifier through the delta-magnitude weights and the raw
    distance features instead. Procedure per module:
    orthonormalize columns of each B_eff by SVD with relative rank tolerance
    RANK_RTOL (columns with sigma > RANK_RTOL * sigma_max; zero-update
    modules — e.g. lora_B still at its zero init — yield an empty basis and
    a NaN angle summary, imputed as 0.0 at matrix-assembly time), then
    SVD of Q_i^T Q_j gives cos(theta_k), clipped to [0, 1].
(c) Per-module distance vector: per-module L2 distance over the module's
    raw (lora_A, lora_B, bridge) block — the same block the D1 classifier
    sees, sliced via feature_layout.

Module aggregation of angles — EXACT WEIGHTING (documented per the card):
    per-module summaries s_m are (i) mean principal angle in radians and
    (ii) chordal RMS sqrt(mean(sin^2 theta)) in [0, 1]. The weighted
    aggregate uses w_m = sqrt(||DW_a^m||_F * ||DW_b^m||_F) — the geometric
    mean of the two adapters' delta magnitudes at module m, computed
    exactly and cheaply via the Gram identity
    ||B_eff A_eff||_F^2 = sum( (B_eff^T B_eff) * (A_eff A_eff^T) )
    (r × r matrices only). Aggregate = sum_m w_m s_m / sum_m w_m over
    modules with finite s_m and w_m > 0. The UNWEIGHTED mean over finite
    s_m is always reported alongside.

Merge constructor
-----------------
merge_adapters(a, b, alpha) returns the bank-format flat state dict
{module}.lora_A / .lora_B / .bridge / .scaling / .n_channels / .rank with
each trainable tensor = (1 - alpha) * a + alpha * b (alpha in [0, 1];
0.5 = midpoint). Metadata (scaling / n_channels / rank) must match between
the two adapters and is copied through. The output is torch.save-able and
loads back through asset1_canonicalize.load_adapter_modules for the later
GPU evaluation. Note: averaging factors is NOT the same as averaging the
updates DW — that discrepancy is precisely what post-merge degradation
measures.

Labels schema (produced by the post-bank GPU evals; consumed here)
------------------------------------------------------------------
Required fields (JSON keys / CSV columns):
    family_short, task_a, run_index_a, task_b, run_index_b, degradation
Optional fields:
    degraded              0|1 explicit binary label
    merged_ppl_a / native_ppl_a / merged_ppl_b / native_ppl_b
                          per-endpoint perplexity (up = degradation)
    merged_score_a / native_score_a / merged_score_b / native_score_b
                          per-endpoint task-metric (down = degradation)

"degradation" is the continuous post-merge metric (definition fixed at eval
time; the harness is agnostic). The per-endpoint metric fields carry the
merged metric and BOTH natives' metrics per endpoint so the PRIMARY
binarization can be applied.

Binarization precedence (DIRECTOR_DECISIONS_2026-07-06.md, D3 OVERRIDE):
1. PRIMARY — fixed relative-degradation threshold. If the per-endpoint
   metric fields are present, a pair is a "conflict" (positive=1) iff the
   merge degrades EITHER endpoint by >= THRESHOLD_REL (=5%) relative vs that
   endpoint's native adapter: perplexity up >= 5% OR task-metric down >= 5%.
   Degenerate fallback: if this yields < DEGENERATE_MIN_FRAC (=10%) positives
   (or < 10% negatives), the imbalance is REPORTED as a finding and the
   headline falls back to the pre-declared median split. The rule actually
   used is recorded in the output.
2. If the per-endpoint metrics are absent but a "degraded" column is present
   on every row, it is used directly (backward-compatible; threshold None).
3. Otherwise labels are binarized as degradation > threshold, where
   --label-threshold is an explicit float or "median" (deterministic median
   split). Median-split is SECONDARY / descriptive under the override.

Dyadic dependence (round-1 review fix — this is NOT an iid sample)
------------------------------------------------------------------
Pairs share endpoint runs, so pair-level rows are dyadically dependent:
a naive StratifiedKFold over PAIRS lets test-fold pairs share a run (and
hence large blocks of module-vector features) with training-fold pairs,
inflating out-of-fold AUC via run-specific components; a pair-iid
bootstrap likewise yields anti-conservative CIs. Three defenses, all
pre-registered here BEFORE the bank completes:
1.  DESIGN: sample_pairs now caps run reuse (``max_run_uses``, default 1
    = vertex-disjoint pairs, making every pair its own run-overlap
    component and removing the dependence at the source). Unlimited
    reuse remains available (CLI --max-run-uses 0) for exploratory work.
2.  GROUP-AWARE CV: pair_group_ids computes connected components of the
    run-overlap graph (union-find); oof_scores(groups=...) uses
    StratifiedGroupKFold over those components, so NO run appears in
    both train and test pairs. The naive StratifiedKFold numbers are
    still computed and reported — as the SECONDARY, explicitly
    anti-conservative diagnostic, never the headline.
3.  CLUSTER BOOTSTRAP: auc_compare(groups=...) resamples run-overlap
    COMPONENTS with replacement (all pairs of a drawn component enter
    together), giving dependence-honest CIs; the pair-iid bootstrap is
    retained only for the naive block.
fit_from_labels reports the group-aware numbers as the per-family
headline (mirrored at the top level) with the naive block alongside; if
the pair graph is so dense that group-aware CV is infeasible (e.g. one
giant component), NO headline is emitted — the naive numbers are
reported with a loud caveat instead of silently standing in.

Prediction harness
------------------
Out-of-fold scores via StratifiedKFold over pairs (naive) or
StratifiedGroupKFold over run-overlap components (group-aware; see
above). Folds are shuffled with the run seed and depend only on
(y, groups, seed, n_splits) — NOT on X — so the full-feature and
distance-only harnesses share IDENTICAL folds (a paired design).
Models: logistic (L2 logistic regression on standardized features,
scores = P(degraded)) or ridge (ridge regression on the continuous
degradation, scores = predictions). Reported: ROC AUC of the full
feature set vs the distance-only baseline (cos + L2 flattened distances
alone), each with a percentile bootstrap CI, plus the PAIRED bootstrap
CI of the AUC difference (same resample draws for both feature sets).
All bootstrap draws come from numpy.random.default_rng seeded with
integer lists — deterministic.

Synthetic selftest (--selftest)
-------------------------------
Two checks, both required for "passed":
(a) PLANTED ANGLES: in-memory pairs where planted angle structure
    determines the label while flattened distances are uninformative:
    label-0 pairs share the B column space (B_b = B_a @ G + noise,
    Frobenius-renormalized; A drawn independently for both, so raw
    distances are dominated by the uninformative A/B entries), label-1
    pairs have independent B (random rank-r subspaces — large principal
    angles). The full-feature AUC must beat the distance-only AUC
    (AUC_full >= AUC_distance + 0.15 and AUC_full >= 0.85).
(b) DEPENDENCE CONTROL (round-1 review fix): a run-latent leakage
    fixture — triangles of runs whose pair labels are determined purely
    by run latents and whose features encode only run identity. Naive
    pair-level CV exploits the endpoint sharing (inflated AUC);
    group-aware CV over run-overlap components must sit near chance.
    Passed = AUC_naive >= AUC_group + 0.15 and AUC_group <= 0.70. This
    is the check the original selftest could not perform because its
    pairs were generated independently.
No bank access, no files required.

Safety
------
Read-only with respect to the bank; the only writes go to --out-dir, which
is REFUSED if it resolves inside the live campaign tree (any path with an
'asset1-bank' component). CPU-only: everything loads map_location='cpu'
via the foundation loader; no CUDA is ever touched.

Usage
-----
    python scripts/asset1_d3_merge.py --selftest [--out-dir DIR] [--seed N]
    python scripts/asset1_d3_merge.py --bank-root results/asset1-bank \
        --out-dir results/asset1-d3 --make-pairs 200 --emit-merges
    python scripts/asset1_d3_merge.py --bank-root results/asset1-bank \
        --out-dir results/asset1-d3 --labels merge_labels.csv
(The two real-bank invocations refuse to run until the manifest shows all
480 runs COMPLETE, unless --allow-partial-bank explicitly overrides with a
loud PRE-REGISTRATION WARNING.)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asset1_analysis_io as aio  # noqa: E402
from asset1_canonicalize import effective_factors  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────

RANK_RTOL = 1e-9        # relative rank tolerance for column-space bases
AGGREGATE_KEYS = ("angle_mean_weighted", "angle_mean_unweighted",
                  "chordal_rms_weighted", "chordal_rms_unweighted")
FEATURE_SETS = ("distance", "full")
_MERGE_TENSOR_FIELDS = ("lora_A", "lora_B")      # always merged
_MERGE_META_FIELDS = ("scaling", "n_channels", "rank")
DEFAULT_MAX_RUN_USES = 1     # vertex-disjoint pairs (dyadic-dependence fix)

# ── Primary label rule (DIRECTOR_DECISIONS_2026-07-06.md, D3 OVERRIDE) ──
# The PRIMARY binarization is a FIXED RELATIVE-DEGRADATION THRESHOLD declared
# now: a pair is a "conflict" (positive) iff the merge degrades EITHER
# endpoint task by >= THRESHOLD_REL relative vs that task's native adapter
# (perplexity up >= 5%, OR task-metric down >= 5%). Median-split is SECONDARY
# / descriptive only. Degenerate fallback: if the fixed rule yields
# < DEGENERATE_MIN_FRAC positives (or < DEGENERATE_MIN_FRAC negatives), that
# is reported as a finding and the headline falls back to the pre-declared
# median split; the rule actually used is recorded.
THRESHOLD_REL = 0.05
DEGENERATE_MIN_FRAC = 0.10
DIRECTOR_DECISIONS_DOC = "docs/DIRECTOR_DECISIONS_2026-07-06.md"

DEPENDENCE_NOTE = (
    "DYADIC DEPENDENCE (round-1 review fix): pairs sharing an endpoint "
    "run are NOT independent observations. The headline numbers use "
    "StratifiedGroupKFold over run-overlap connected components (no run "
    "in both train and test pairs) and a component-cluster bootstrap. "
    "The 'naive' block (StratifiedKFold over pairs, pair-iid bootstrap) "
    "is anti-conservative and reported for comparison only — it must "
    "never be quoted as the result.")


# ── Safety guard (shared with asset1_daux_gap) ──────────────────────


def guard_out_dir(out_dir: str | Path) -> Path:
    """Refuse any output directory inside the LIVE campaign tree.

    The interlock protects reads of the real bank; this guard protects
    writes — no D3/D-aux tool may ever create files under a path with an
    'asset1-bank' component. Returns the resolved path on success.
    """
    resolved = Path(out_dir).resolve()
    if "asset1-bank" in resolved.parts:
        raise ValueError(
            f"REFUSING to write into {resolved}: the path contains an "
            f"'asset1-bank' component — that is the LIVE campaign tree "
            f"and is write-protected.")
    return resolved


# ── Principal angles (gauge-invariant, bridge absorbed) ─────────────


def orthonormal_colspace(M: np.ndarray, rtol: float = RANK_RTOL) -> np.ndarray:
    """Orthonormal basis of col(M) via SVD with relative rank truncation.

    Keeps left singular vectors with sigma > rtol * sigma_max. A zero (or
    empty) matrix returns a (d, 0) basis — the caller treats the resulting
    angle set as undefined (NaN summary) rather than inventing directions.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {M.shape}")
    if M.size == 0:
        return M.reshape(M.shape[0], 0)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0 or s[0] <= 0.0:
        return U[:, :0]
    k = int(np.sum(s > rtol * s[0]))
    return U[:, :k]


def principal_angles(B1: np.ndarray, B2: np.ndarray,
                     rtol: float = RANK_RTOL) -> np.ndarray:
    """Principal angles (radians, ascending) between col(B1) and col(B2).

    Standard Björck–Golub construction: orthonormal bases Q1, Q2, then the
    singular values of Q1^T Q2 are cos(theta_k), clipped to [0, 1]. The
    number of angles is min(rank(B1), rank(B2)); an empty column space on
    either side yields an empty array.
    """
    Q1 = orthonormal_colspace(B1, rtol)
    Q2 = orthonormal_colspace(B2, rtol)
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    c = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.arccos(np.clip(c, 0.0, 1.0))     # cos descending -> ascending


def delta_magnitude(entry: dict[str, torch.Tensor]) -> float:
    """||DW||_F of a module's TRUE effective update, without materializing DW.

    DW = B_eff @ A_eff with the bridge and alpha/rank scaling absorbed by
    asset1_canonicalize.effective_factors. Uses the Gram identity
    ||B A||_F^2 = tr((B^T B)(A A^T)) = sum((B^T B) * (A A^T)) — r x r
    matrices only ((A A^T) is symmetric, so the transpose in the elementwise
    form is free).
    """
    B_eff, A_eff = effective_factors(entry)
    return _delta_magnitude_from_factors(B_eff, A_eff)


def _delta_magnitude_from_factors(B_eff: torch.Tensor,
                                  A_eff: torch.Tensor) -> float:
    GB = B_eff.T @ B_eff
    GA = A_eff @ A_eff.T
    val = torch.clamp((GB * GA).sum(), min=0.0)
    return float(torch.sqrt(val).item())


def module_pair_angles(entry_a: dict[str, torch.Tensor],
                       entry_b: dict[str, torch.Tensor],
                       rtol: float = RANK_RTOL) -> np.ndarray:
    """Principal angles between the effective-update column spaces of one
    module across two adapters (bridge + scaling absorbed first)."""
    Ba, _ = effective_factors(entry_a)
    Bb, _ = effective_factors(entry_b)
    return principal_angles(Ba.numpy(), Bb.numpy(), rtol)


# ── Pair features ───────────────────────────────────────────────────


def pair_features(adapter_a: dict, adapter_b: dict,
                  rtol: float = RANK_RTOL) -> dict:
    """All D3 features for one SAME-FAMILY adapter pair.

    Returns a dict with scalar features (cos_distance, l2_distance, the
    four angle aggregates of AGGREGATE_KEYS), per-module vectors
    (module_l2, module_angle_mean, module_chordal_rms, module_weight —
    sorted-module order), and module_names. Raises ValueError on
    mismatched module sets or shapes (cross-family pairs are undefined
    for D3 — Qwen and Llama do not even share module names).
    """
    if set(adapter_a) != set(adapter_b):
        raise ValueError(
            "adapters have different module sets — D3 pair features are "
            "defined for SAME-FAMILY pairs only")
    names = sorted(adapter_a)

    va = aio.flatten_features(adapter_a).astype(np.float64)
    vb = aio.flatten_features(adapter_b).astype(np.float64)
    if va.size != vb.size:
        raise ValueError(
            "flattened feature sizes differ — D3 pair features are defined "
            "for SAME-FAMILY pairs only")

    diff = va - vb
    l2_distance = float(np.linalg.norm(diff))
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na > 0.0 and nb > 0.0:
        cos_distance = float(1.0 - (va @ vb) / (na * nb))
    else:                       # all-zero adapter: cosine undefined -> max
        cos_distance = 1.0

    # Per-module raw L2 via the SAME layout the D1 classifier uses.
    sq_by_module: dict[str, float] = {n: 0.0 for n in names}
    for module, _field, sl in aio.feature_layout(adapter_a):
        sq_by_module[module] += float(np.sum(diff[sl] ** 2))
    module_l2 = np.array([np.sqrt(sq_by_module[n]) for n in names])

    angle_mean = np.full(len(names), np.nan)
    chordal_rms = np.full(len(names), np.nan)
    weight = np.zeros(len(names))
    for i, name in enumerate(names):
        Ba, Aa = effective_factors(adapter_a[name])
        Bb, Ab = effective_factors(adapter_b[name])
        mag_a = _delta_magnitude_from_factors(Ba, Aa)
        mag_b = _delta_magnitude_from_factors(Bb, Ab)
        weight[i] = float(np.sqrt(mag_a * mag_b))
        angles = principal_angles(Ba.numpy(), Bb.numpy(), rtol)
        if angles.size:
            angle_mean[i] = float(angles.mean())
            chordal_rms[i] = float(np.sqrt(np.mean(np.sin(angles) ** 2)))

    def _agg(summary: np.ndarray) -> tuple[float, float]:
        finite = np.isfinite(summary)
        unweighted = float(summary[finite].mean()) if finite.any() else float("nan")
        usable = finite & (weight > 0.0)
        if usable.any():
            w = weight[usable]
            weighted = float(np.sum(w * summary[usable]) / np.sum(w))
        else:
            weighted = float("nan")
        return weighted, unweighted

    am_w, am_u = _agg(angle_mean)
    cr_w, cr_u = _agg(chordal_rms)

    return {
        "cos_distance": cos_distance,
        "l2_distance": l2_distance,
        "angle_mean_weighted": am_w,
        "angle_mean_unweighted": am_u,
        "chordal_rms_weighted": cr_w,
        "chordal_rms_unweighted": cr_u,
        "module_names": names,
        "module_l2": module_l2,
        "module_angle_mean": angle_mean,
        "module_chordal_rms": chordal_rms,
        "module_weight": weight,
    }


def assemble_matrix(features_list: list[dict],
                    feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Stack pair-feature dicts into a design matrix (rows = pairs).

    'distance' — the baseline: [cos_distance, l2_distance] only.
    'full'     — baseline + the four angle aggregates + the per-module L2
                 vector + the per-module mean-angle vector. NaN angle
                 summaries (zero-update modules) are imputed as 0.0 — "no
                 evidence of rotation" — documented in the module docstring.
    The 'full' set requires every pair to share the same module_names
    (same family); mixing families raises ValueError.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, "
                         f"got {feature_set!r}")
    if not features_list:
        raise ValueError("features_list is empty")
    base_names = ["cos_distance", "l2_distance"]
    if feature_set == "distance":
        X = np.array([[pf["cos_distance"], pf["l2_distance"]]
                      for pf in features_list], dtype=np.float64)
        return X, base_names

    modules0 = features_list[0]["module_names"]
    rows = []
    for pf in features_list:
        if pf["module_names"] != modules0:
            raise ValueError(
                "pairs with different module sets in one matrix — assemble "
                "per family (module vectors are family-specific)")
        row = [pf["cos_distance"], pf["l2_distance"]]
        row += [np.nan_to_num(pf[k], nan=0.0) for k in AGGREGATE_KEYS]
        row += list(pf["module_l2"])
        row += list(np.nan_to_num(pf["module_angle_mean"], nan=0.0))
        rows.append(row)
    names = (base_names + list(AGGREGATE_KEYS)
             + [f"l2:{m}" for m in modules0]
             + [f"angle:{m}" for m in modules0])
    return np.array(rows, dtype=np.float64), names


# ── Merge constructor ───────────────────────────────────────────────


def merge_adapters(adapter_a: dict, adapter_b: dict,
                   alpha: float = 0.5) -> dict[str, torch.Tensor]:
    """Weight-space average of two adapters, in the bank's flat key format.

    merged = (1 - alpha) * a + alpha * b for lora_A, lora_B, and bridge
    (bridge merged when present in both, forbidden one-sided). Metadata
    (scaling, n_channels, rank) must be identical between the adapters and
    is copied through. Result is torch.save-able and round-trips through
    asset1_canonicalize.load_adapter_modules for the later GPU eval.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if set(adapter_a) != set(adapter_b):
        raise ValueError("cannot merge adapters with different module sets")

    out: dict[str, torch.Tensor] = {}
    for name in sorted(adapter_a):
        ea, eb = adapter_a[name], adapter_b[name]
        fields = list(_MERGE_TENSOR_FIELDS)
        has_a, has_b = "bridge" in ea, "bridge" in eb
        if has_a != has_b:
            raise ValueError(f"module {name!r}: bridge present in only one "
                             f"adapter — inconsistent inputs")
        if has_a:
            fields.append("bridge")
        for field in fields:
            if field not in ea or field not in eb:
                raise ValueError(f"module {name!r} missing {field!r}")
            ta, tb = ea[field], eb[field]
            if ta.shape != tb.shape:
                raise ValueError(
                    f"module {name!r} field {field!r}: shape mismatch "
                    f"{tuple(ta.shape)} vs {tuple(tb.shape)}")
            merged = ((1.0 - alpha) * ta.to(torch.float64)
                      + alpha * tb.to(torch.float64))
            out[f"{name}.{field}"] = merged.to(ta.dtype)
        for field in _MERGE_META_FIELDS:
            pa, pb = ea.get(field), eb.get(field)
            if pa is None and pb is None:
                continue
            if pa is None or pb is None:
                raise ValueError(f"module {name!r}: metadata {field!r} "
                                 f"present in only one adapter")
            if not torch.allclose(pa.to(torch.float64), pb.to(torch.float64)):
                raise ValueError(
                    f"module {name!r}: metadata {field!r} differs "
                    f"({pa.tolist()} vs {pb.tolist()}) — refusing to merge")
            out[f"{name}.{field}"] = pa.clone()
    return out


# ── Labels ──────────────────────────────────────────────────────────

_LABEL_REQUIRED = ("family_short", "task_a", "run_index_a",
                   "task_b", "run_index_b", "degradation")
# Optional per-endpoint metric fields (Director override D3): merged + both
# natives per endpoint, so the fixed relative-degradation rule can be applied.
_LABEL_OPTIONAL_METRICS = ("merged_ppl_a", "native_ppl_a",
                           "merged_ppl_b", "native_ppl_b",
                           "merged_score_a", "native_score_a",
                           "merged_score_b", "native_score_b")


def load_labels(path: str | Path) -> list[dict]:
    """Parse the labels file (schema in the module docstring). JSON or CSV
    by extension. Returns canonical rows with coerced types; 'degraded' is
    int or None; the optional per-endpoint metric fields are float or None."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "pairs" not in raw or not isinstance(raw["pairs"], list):
            raise ValueError(f"{path}: JSON labels need a top-level "
                             f"'pairs' list")
        raw_rows = raw["pairs"]
    elif path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"labels file must be .json or .csv, got {path}")

    rows: list[dict] = []
    for i, r in enumerate(raw_rows):
        missing = [k for k in _LABEL_REQUIRED if r.get(k) in (None, "")]
        if missing:
            raise ValueError(f"labels row {i}: missing fields {missing}")
        degraded = r.get("degraded")
        if degraded in (None, ""):
            degraded = None
        else:
            degraded = int(degraded)
            if degraded not in (0, 1):
                raise ValueError(f"labels row {i}: degraded must be 0/1")
        canonical = {
            "family_short": str(r["family_short"]),
            "task_a": str(r["task_a"]),
            "run_index_a": int(r["run_index_a"]),
            "task_b": str(r["task_b"]),
            "run_index_b": int(r["run_index_b"]),
            "degradation": float(r["degradation"]),
            "degraded": degraded,
        }
        for field in _LABEL_OPTIONAL_METRICS:
            val = r.get(field)
            canonical[field] = None if val in (None, "") else float(val)
        rows.append(canonical)
    if not rows:
        raise ValueError(f"{path}: no label rows")
    return rows


def binarize_labels(rows: list[dict],
                    threshold: str | float = "median"
                    ) -> tuple[np.ndarray, float | None]:
    """Binary labels from the rows. If EVERY row carries 'degraded', it is
    used directly (returned threshold None). Mixed presence is an error.
    Otherwise degraded = degradation > threshold, with threshold 'median'
    (deterministic median split) or an explicit float."""
    have = [r["degraded"] is not None for r in rows]
    if all(have):
        return np.array([r["degraded"] for r in rows], dtype=int), None
    if any(have):
        raise ValueError("'degraded' present on some rows but not all — "
                         "make it consistent")
    deg = np.array([r["degradation"] for r in rows], dtype=np.float64)
    thr = float(np.median(deg)) if threshold == "median" else float(threshold)
    return (deg > thr).astype(int), thr


def _relative_degraded_per_row(row: dict, threshold_rel: float
                               ) -> tuple[bool | None, list]:
    """Fixed relative-degradation rule for one row (Director override D3).

    A row is degraded (positive) iff the merge degrades EITHER endpoint by
    >= threshold_rel relative vs that endpoint's native adapter:
      - perplexity form  (merged_ppl / native_ppl):   up   >= threshold_rel
      - task-metric form (merged_score / native_score): down >= threshold_rel
    Returns (degraded_bool, detail) or (None, reason) when the row lacks any
    per-endpoint metric field (so the caller can fall through to the
    median/degraded-column path). Endpoints with a non-positive native
    denominator are skipped (their relative change is undefined).
    """
    detail: list = []
    degraded = False
    for ep in ("a", "b"):
        mp, npp = row.get(f"merged_ppl_{ep}"), row.get(f"native_ppl_{ep}")
        ms, ns = row.get(f"merged_score_{ep}"), row.get(f"native_score_{ep}")
        if mp is not None and npp is not None and npp > 0.0:
            rel = (mp - npp) / npp                # perplexity up = worse
            detail.append({"endpoint": ep, "metric": "ppl", "rel": rel})
            if rel >= threshold_rel:
                degraded = True
        elif ms is not None and ns is not None and ns > 0.0:
            rel = (ns - ms) / ns                  # task-metric down = worse
            detail.append({"endpoint": ep, "metric": "score", "rel": rel})
            if rel >= threshold_rel:
                degraded = True
    if not detail:
        return None, "no per-endpoint metric fields"
    return degraded, detail


def binarize_primary(rows: list[dict], *,
                     threshold_rel: float = THRESHOLD_REL,
                     degenerate_min_frac: float = DEGENERATE_MIN_FRAC,
                     median_threshold: str | float = "median"
                     ) -> tuple[np.ndarray, dict]:
    """PRIMARY binarization (DIRECTOR_DECISIONS_2026-07-06.md, D3 OVERRIDE).

    Fixed relative-degradation threshold when the per-endpoint metric fields
    are present, with a degenerate-balance fallback to the pre-declared
    median split. When those fields are absent, defers to binarize_labels
    (explicit 'degraded' column, else median/explicit split). Returns
    (y, meta); ``meta`` records the constants, the rule actually used, the
    positive fraction, and any degenerate finding — so runs self-document.
    """
    meta: dict = {
        "threshold_rel": threshold_rel,
        "degenerate_min_frac": degenerate_min_frac,
        "source": DIRECTOR_DECISIONS_DOC,
    }
    per_row = [_relative_degraded_per_row(r, threshold_rel) for r in rows]
    have_metrics = all(d is not None for d, _ in per_row)

    if have_metrics:
        y = np.array([int(d) for d, _ in per_row], dtype=int)
        frac_pos = float(y.mean()) if y.size else 0.0
        frac_neg = 1.0 - frac_pos
        degenerate = bool(frac_pos < degenerate_min_frac
                          or frac_neg < degenerate_min_frac)
        meta["rule_available"] = "relative_degradation"
        meta["frac_positive_relative"] = frac_pos
        meta["degenerate"] = degenerate
        if not degenerate:
            meta["rule_used"] = "relative_degradation"
            meta["threshold"] = None
            return y, meta
        # Degenerate balance: report the finding, fall back to median split.
        y_fb, thr = binarize_labels(rows, median_threshold)
        meta["rule_used"] = "median_fallback"
        meta["threshold"] = thr
        meta["degenerate_finding"] = (
            f"the fixed {threshold_rel:.0%} relative-degradation rule "
            f"yielded {frac_pos:.1%} positives — below the "
            f"{degenerate_min_frac:.0%} degenerate floor on one side; the "
            f"headline falls back to the pre-declared median split "
            f"({DIRECTOR_DECISIONS_DOC}).")
        return y_fb, meta

    # No per-endpoint metrics: defer to the secondary/descriptive path.
    y, thr = binarize_labels(rows, median_threshold)
    meta["rule_available"] = "none (per-endpoint metrics absent)"
    if thr is None:
        meta["rule_used"] = "degraded_column"
    elif median_threshold == "median":
        meta["rule_used"] = "median"
    else:
        meta["rule_used"] = "explicit"
    meta["threshold"] = thr
    return y, meta


# ── Prediction harness ──────────────────────────────────────────────


def pair_group_ids(pair_run_keys: list[tuple]) -> np.ndarray:
    """Connected-component id per pair over the run-overlap graph.

    ``pair_run_keys`` is [(key_a, key_b), ...] with hashable run keys
    (e.g. (family_short, task, run_index)). Two pairs belong to the same
    group iff they are connected through shared endpoint runs (union-find
    with path compression). Holding out a whole group therefore holds out
    every pair touching any of its runs — the grouping unit for the
    dyadic-dependence-aware CV and cluster bootstrap (module docstring).
    Returns dense int64 ids (0..n_groups-1, first-appearance order).
    """
    parent: dict = {}

    def find(x):
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[x] != root:            # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b in pair_run_keys:
        parent[find(a)] = find(b)

    ids = np.empty(len(pair_run_keys), dtype=np.int64)
    dense: dict = {}
    for i, (a, _b) in enumerate(pair_run_keys):
        root = find(a)
        ids[i] = dense.setdefault(root, len(dense))
    return ids


def oof_scores(X: np.ndarray, y: np.ndarray, model: str = "logistic",
               n_splits: int = 5, seed: int = 0, C: float = 1.0,
               ridge_alpha: float = 1.0,
               y_cont: np.ndarray | None = None,
               groups: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold prediction scores (higher = more degraded).

    groups is None — NAIVE: StratifiedKFold over pairs (run overlap
    across folds possible; anti-conservative — round-1 review finding).
    groups given  — GROUP-AWARE: StratifiedGroupKFold over the run-overlap
    component ids from pair_group_ids, so no run appears in both train
    and test pairs.

    Folds are shuffled with ``seed`` and depend only on (y, groups, seed,
    n_splits) — NOT on X — so calling this with different X but identical
    (y, groups, seed, n_splits) yields IDENTICAL folds (the paired
    full-vs-baseline design). n_splits is clamped to the minority class
    count (and to the group count when grouped); fewer than 2 per class,
    fewer than 2 groups, or a single-class training fold raises.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model not in ("logistic", "ridge"):
        raise ValueError(f"model must be 'logistic' or 'ridge', got {model!r}")
    y = np.asarray(y, dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 2:
        raise ValueError(f"need exactly 2 classes in y, got {classes.tolist()}")
    min_count = int(counts.min())
    if min_count < 2:
        raise ValueError("need at least 2 samples in each class")

    if groups is None:
        n_eff = min(n_splits, min_count)
        splitter = StratifiedKFold(n_splits=n_eff, shuffle=True,
                                   random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError(f"groups shape {groups.shape} does not match "
                             f"y shape {y.shape}")
        n_groups = int(np.unique(groups).size)
        if n_groups < 2:
            raise ValueError(
                "need at least 2 run-overlap groups for group-aware CV — "
                "the pair graph is a single connected component (consider "
                "max_run_uses=1 pair sampling)")
        n_eff = min(n_splits, min_count, n_groups)
        if n_eff < 2:
            raise ValueError("cannot form 2 group-aware folds with these "
                             "labels/groups")
        splitter = StratifiedGroupKFold(n_splits=n_eff, shuffle=True,
                                        random_state=seed)
        split_iter = splitter.split(X, y, groups)

    scores = np.zeros(len(y), dtype=np.float64)
    target = (np.asarray(y_cont, dtype=np.float64)
              if y_cont is not None else y.astype(np.float64))
    for tr, te in split_iter:
        if np.unique(y[tr]).size < 2:
            raise ValueError(
                "a training fold contains a single class — the group/label "
                "structure cannot support this CV; lower n_splits")
        if model == "logistic":
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, max_iter=2000, random_state=seed))
            pipe.fit(X[tr], y[tr])
            scores[te] = pipe.predict_proba(X[te])[:, 1]
        else:
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
            pipe.fit(X[tr], target[tr])
            scores[te] = pipe.predict(X[te])
    return scores


def auc_compare(y: np.ndarray, scores_full: np.ndarray,
                scores_base: np.ndarray, n_boot: int = 1000,
                seed: int = 0, groups: np.ndarray | None = None) -> dict:
    """ROC AUCs of full vs distance-only scores + paired bootstrap CIs.

    groups is None — pair-iid bootstrap (resamples pair indices with
    replacement; ANTI-CONSERVATIVE under dyadic dependence — round-1
    review finding; used only for the naive diagnostic block).
    groups given  — CLUSTER bootstrap over run-overlap components: each
    draw resamples component ids with replacement and takes ALL pairs of
    every drawn component, so within-component dependence is preserved
    inside each replicate.

    Both score sets are evaluated on the same draw, so auc_diff_ci is a
    paired CI. Single-class draws are skipped (counted via n_boot_valid);
    if fewer than 10 draws are valid the CIs are [None, None].
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    auc_f = float(roc_auc_score(y, scores_full))
    auc_b = float(roc_auc_score(y, scores_base))

    rng = np.random.default_rng([seed, 808])
    n = len(y)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError(f"groups shape {groups.shape} does not match "
                             f"y shape {y.shape}")
        uniq = np.unique(groups)
        idx_by_group = [np.flatnonzero(groups == g) for g in uniq]
        unit = "run-component-cluster"
    else:
        unit = "pair-iid"

    fulls, bases, diffs = [], [], []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(idx_by_group), len(idx_by_group))
            idx = np.concatenate([idx_by_group[p] for p in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        bf = float(roc_auc_score(yy, scores_full[idx]))
        bb = float(roc_auc_score(yy, scores_base[idx]))
        fulls.append(bf)
        bases.append(bb)
        diffs.append(bf - bb)

    def _ci(vals: list[float]) -> list[float | None]:
        if len(vals) < 10:
            return [None, None]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return [float(lo), float(hi)]

    return {
        "auc_full": auc_f,
        "auc_full_ci": _ci(fulls),
        "auc_distance": auc_b,
        "auc_distance_ci": _ci(bases),
        "auc_diff": auc_f - auc_b,
        "auc_diff_ci": _ci(diffs),
        "n_boot_requested": n_boot,
        "n_boot_valid": len(diffs),
        "bootstrap_unit": unit,
    }


# ── Synthetic selftest ──────────────────────────────────────────────


def _well_conditioned(rng: np.random.Generator, r: int,
                      max_cond: float = 100.0) -> np.ndarray:
    for _ in range(64):
        G = rng.standard_normal((r, r))
        if np.linalg.cond(G) < max_cond:
            return G
    raise RuntimeError("could not draw a well-conditioned G")


def make_selftest_pairs(n_per_class: int = 30, n_modules: int = 3,
                        d_out: int = 20, d_in: int = 12, rank: int = 4,
                        n_channels: int = 2, noise: float = 0.05,
                        seed: int = 0) -> tuple[list[dict], np.ndarray]:
    """Synthetic pairs where the label lives in the ANGLES, not distances.

    label 0: B_b = B_a @ G + noise (shared column space, Frobenius-matched);
    label 1: B_b independent (random subspace). A factors are independent
    for both classes with identical scale, so flattened distances carry no
    label signal (documented plant — the selftest's separability claim).
    """
    if rank % n_channels != 0:
        raise ValueError("rank must be divisible by n_channels")
    feats: list[dict] = []
    y: list[int] = []
    for label in (0, 1):
        for i in range(n_per_class):
            rng = np.random.default_rng([seed, 909, label, i])
            a: dict[str, dict[str, torch.Tensor]] = {}
            b: dict[str, dict[str, torch.Tensor]] = {}
            for m in range(n_modules):
                B1 = rng.standard_normal((d_out, rank))
                A1 = 0.3 * rng.standard_normal((rank, d_in))
                A2 = 0.3 * rng.standard_normal((rank, d_in))
                if label == 0:
                    G = _well_conditioned(rng, rank)
                    B2 = B1 @ G + noise * rng.standard_normal((d_out, rank))
                else:
                    B2 = rng.standard_normal((d_out, rank))
                B2 = B2 * (np.linalg.norm(B1)
                           / max(np.linalg.norm(B2), 1e-12))
                br1 = np.eye(n_channels) + 0.05 * rng.standard_normal(
                    (n_channels, n_channels))
                br2 = np.eye(n_channels) + 0.05 * rng.standard_normal(
                    (n_channels, n_channels))
                name = f"m{m:02d}"
                a[name] = {
                    "lora_A": torch.as_tensor(A1, dtype=torch.float32),
                    "lora_B": torch.as_tensor(B1, dtype=torch.float32),
                    "bridge": torch.as_tensor(br1, dtype=torch.float32),
                    "scaling": torch.tensor(1.0),
                }
                b[name] = {
                    "lora_A": torch.as_tensor(A2, dtype=torch.float32),
                    "lora_B": torch.as_tensor(B2, dtype=torch.float32),
                    "bridge": torch.as_tensor(br2, dtype=torch.float32),
                    "scaling": torch.tensor(1.0),
                }
            feats.append(pair_features(a, b))
            y.append(label)
    return feats, np.array(y, dtype=int)


def make_leakage_fixture(n_triangles: int = 20, seed: int = 0
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-latent leakage fixture for the dependence control (docstring).

    3 runs per triangle, all 3 within-triangle pairs. Each run r carries a
    latent u_r ~ N(0,1); pair label = 1 iff u_a + u_b > 0; the feature
    vector is one-hot(run a) + one-hot(run b) — run identity is the ONLY
    signal. Naive pair-level CV can read a held-out pair's run
    coefficients from the triangle's other two edges in the training
    fold; group-aware CV holds out whole triangles (= run-overlap
    components), so the held-out runs' feature columns are all-zero in
    training and scores carry no signal. Returns (X, y, groups).
    """
    rng = np.random.default_rng([seed, 505])
    n_runs = 3 * n_triangles
    u = rng.standard_normal(n_runs)
    rows, labels, keys = [], [], []
    for t in range(n_triangles):
        base = 3 * t
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = base + a, base + b
            x = np.zeros(n_runs, dtype=np.float64)
            x[i] = 1.0
            x[j] = 1.0
            rows.append(x)
            labels.append(int(u[i] + u[j] > 0))
            keys.append((i, j))
    return (np.vstack(rows), np.asarray(labels, dtype=int),
            pair_group_ids(keys))


def run_selftest(seed: int = 0, n_per_class: int = 30, n_splits: int = 5,
                 n_boot: int = 200) -> dict:
    """Full-pipeline selftest (no bank access): planted angles + the
    dyadic-dependence control (module docstring).

    passed = (AUC_full >= AUC_distance + 0.15 AND AUC_full >= 0.85)
             AND (AUC_naive >= AUC_group + 0.15 AND AUC_group <= 0.70
                  on the run-latent leakage fixture).
    """
    feats, y = make_selftest_pairs(n_per_class=n_per_class, seed=seed)
    X_full, _ = assemble_matrix(feats, "full")
    X_base, _ = assemble_matrix(feats, "distance")
    sf = oof_scores(X_full, y, model="logistic", n_splits=n_splits, seed=seed)
    sb = oof_scores(X_base, y, model="logistic", n_splits=n_splits, seed=seed)
    rep = auc_compare(y, sf, sb, n_boot=n_boot, seed=seed)
    rep["n_pairs"] = int(len(y))
    rep["seed"] = seed
    angle_passed = bool(rep["auc_full"] >= rep["auc_distance"] + 0.15
                        and rep["auc_full"] >= 0.85)

    # Dependence control: naive CV must inflate on run-identity leakage;
    # group-aware CV must not (round-1 review fix).
    from sklearn.metrics import roc_auc_score
    X_leak, y_leak, g_leak = make_leakage_fixture(seed=seed)
    s_naive = oof_scores(X_leak, y_leak, model="logistic",
                         n_splits=n_splits, seed=seed)
    s_group = oof_scores(X_leak, y_leak, model="logistic",
                         n_splits=n_splits, seed=seed, groups=g_leak)
    auc_naive = float(roc_auc_score(y_leak, s_naive))
    auc_group = float(roc_auc_score(y_leak, s_group))
    dep_passed = bool(auc_naive >= auc_group + 0.15 and auc_group <= 0.70)
    rep["dependence_control"] = {
        "n_pairs": int(len(y_leak)),
        "n_components": int(np.unique(g_leak).size),
        "auc_naive_cv": auc_naive,
        "auc_group_aware_cv": auc_group,
        "passed": dep_passed,
        "note": DEPENDENCE_NOTE,
    }
    rep["passed"] = bool(angle_passed and dep_passed)
    return rep


# ── Real-bank plumbing ──────────────────────────────────────────────


@lru_cache(maxsize=16)
def _load_adapter_cached(run_dir_str: str) -> dict:
    return aio.load_adapter(Path(run_dir_str))


def build_run_index(bank_root: str | Path) -> dict[tuple[str, str, int], Path]:
    """(family_short, task, run_index) -> run_dir, from the manifest."""
    return {(r["family_short"], r["task"], r["run_index"]): r["run_dir"]
            for r in aio.iter_runs(bank_root)}


def sample_pairs(bank_root: str | Path, n_per_family: int,
                 seed: int = 0,
                 max_run_uses: int | None = DEFAULT_MAX_RUN_USES
                 ) -> list[dict]:
    """Deterministically sample same-family run pairs (for GPU eval).

    Per family (sorted): enumerate all unordered pairs of COMPLETE runs
    in manifest order, then sample from default_rng([seed, 707,
    family_index]).

    max_run_uses (round-1 review fix — dyadic dependence at the source):
    cap on how many sampled pairs any single run may appear in. The
    default 1 yields VERTEX-DISJOINT pairs — every pair is its own
    run-overlap component, so downstream CV folds and bootstrap draws
    are honestly independent by design. Constrained sampling scans a
    seeded random order of all pairs, accepting a pair only if both
    endpoints are under the cap; it may return fewer than n_per_family
    when the cap binds. ``None`` (CLI --max-run-uses 0) reproduces the
    legacy unconstrained draw-without-replacement sampler. Pair-selection
    POLICY for the reportable experiment (stratification over task
    pairs, N, the run-reuse cap) requires Director sign-off — this
    sampler is the deterministic mechanism.
    """
    if max_run_uses is not None and max_run_uses < 1:
        raise ValueError("max_run_uses must be >= 1 or None (unlimited)")
    fams: dict[str, list[dict]] = defaultdict(list)
    for r in aio.iter_runs(bank_root):
        fams[r["family_short"]].append(r)
    out: list[dict] = []
    for fi, fam in enumerate(sorted(fams)):
        runs = fams[fam]
        all_pairs = [(i, j) for i in range(len(runs))
                     for j in range(i + 1, len(runs))]
        if not all_pairs:
            continue
        rng = np.random.default_rng([seed, 707, fi])
        if max_run_uses is None:
            take = min(n_per_family, len(all_pairs))
            sel = sorted(rng.choice(len(all_pairs), size=take,
                                    replace=False).tolist())
        else:
            uses: Counter = Counter()
            accepted: list[int] = []
            for k in rng.permutation(len(all_pairs)).tolist():
                i, j = all_pairs[k]
                if uses[i] < max_run_uses and uses[j] < max_run_uses:
                    accepted.append(k)
                    uses[i] += 1
                    uses[j] += 1
                    if len(accepted) == n_per_family:
                        break
            sel = sorted(accepted)
        for k in sel:
            i, j = all_pairs[k]
            out.append({
                "family_short": fam,
                "task_a": runs[i]["task"],
                "run_index_a": runs[i]["run_index"],
                "task_b": runs[j]["task"],
                "run_index_b": runs[j]["run_index"],
            })
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _pair_key(spec: dict) -> tuple[tuple[str, str, int], tuple[str, str, int]]:
    return ((spec["family_short"], spec["task_a"], spec["run_index_a"]),
            (spec["family_short"], spec["task_b"], spec["run_index_b"]))


def make_pairs_command(bank_root: Path, out_dir: Path, n_per_family: int,
                       seed: int, alpha: float, emit_merges: bool,
                       max_run_uses: int | None = DEFAULT_MAX_RUN_USES
                       ) -> dict:
    """--make-pairs: sample pairs, compute features, optionally write the
    merged adapter states (for the later GPU eval) under out_dir/merges/."""
    index = build_run_index(bank_root)
    pairs = sample_pairs(bank_root, n_per_family, seed,
                         max_run_uses=max_run_uses)
    merges_dir = out_dir / "merges"
    records = []
    for spec in pairs:
        key_a, key_b = _pair_key(spec)
        aa = _load_adapter_cached(str(index[key_a]))
        ab = _load_adapter_cached(str(index[key_b]))
        rec = dict(spec)
        rec["features"] = _jsonable(pair_features(aa, ab))
        if emit_merges:
            merges_dir.mkdir(parents=True, exist_ok=True)
            merged = merge_adapters(aa, ab, alpha=alpha)
            fname = (f"merge_{spec['family_short']}"
                     f"_run{spec['run_index_a']:03d}"
                     f"_run{spec['run_index_b']:03d}_alpha{alpha:g}.pt")
            torch.save(merged, merges_dir / fname)
            rec["merge_file"] = str(Path("merges") / fname)
        records.append(rec)
    payload = {"seed": seed, "alpha": alpha, "n_per_family": n_per_family,
               "max_run_uses": max_run_uses,
               "n_pairs": len(records), "pairs": records}
    (out_dir / "d3_pairs.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def fit_from_labels(bank_root: Path, labels_path: Path,
                    threshold: str | float, model: str, n_splits: int,
                    n_boot: int, seed: int,
                    threshold_rel: float = THRESHOLD_REL,
                    degenerate_min_frac: float = DEGENERATE_MIN_FRAC) -> dict:
    """--labels: join labels to bank adapters, fit per family, report AUCs.

    Labels are binarized by the PRIMARY rule (binarize_primary — fixed
    relative-degradation threshold per DIRECTOR_DECISIONS_2026-07-06.md, with
    a degenerate-balance fallback to the median split); the rule actually
    used is recorded in the returned ``binarization`` block. Median-split is
    secondary/descriptive under the override.

    The harness is fit PER FAMILY (module-vector features are
    family-specific). Per family the HEADLINE numbers (mirrored at the
    block's top level, headline_basis='group_aware') come from
    StratifiedGroupKFold over run-overlap components with a
    component-cluster bootstrap; the naive pair-level numbers are the
    'naive' sub-block, reported for comparison only (round-1 review fix —
    see DEPENDENCE_NOTE). If the pair graph cannot support group-aware CV
    (e.g. one giant component), NO headline is emitted for that family:
    the naive block stands alone with group_cv_feasible=false. A pooled
    AUC over the concatenated GROUP-AWARE out-of-fold scores is reported
    as a descriptive summary (families never share runs, so per-family
    groups remain valid after offsetting) — the pooling method for any
    headline number requires Director sign-off.
    """
    rows = load_labels(labels_path)
    y_all, bin_meta = binarize_primary(
        rows, threshold_rel=threshold_rel,
        degenerate_min_frac=degenerate_min_frac, median_threshold=threshold)
    index = build_run_index(bank_root)

    per_family: dict[str, dict] = {}
    pooled_y, pooled_f, pooled_b, pooled_g = [], [], [], []
    group_offset = 0
    for fam in sorted({r["family_short"] for r in rows}):
        idxs = [i for i, r in enumerate(rows)
                if r["family_short"] == fam]
        feats = []
        pair_keys = []
        for i in idxs:
            r = rows[i]
            key_a = (fam, r["task_a"], r["run_index_a"])
            key_b = (fam, r["task_b"], r["run_index_b"])
            for key in (key_a, key_b):
                if key not in index:
                    raise KeyError(f"labels reference unknown COMPLETE run "
                                   f"{key} — not in the manifest")
            aa = _load_adapter_cached(str(index[key_a]))
            ab = _load_adapter_cached(str(index[key_b]))
            feats.append(pair_features(aa, ab))
            pair_keys.append((key_a, key_b))
        y = y_all[idxs]
        if y.min() == y.max():
            per_family[fam] = {"note": "single-class labels — skipped",
                               "n_pairs": len(idxs)}
            continue
        groups = pair_group_ids(pair_keys)
        comp_sizes = np.bincount(groups)
        dependence = {
            "n_runs": len({k for pair in pair_keys for k in pair}),
            "n_components": int(comp_sizes.size),
            "largest_component_pairs": int(comp_sizes.max()),
            "note": DEPENDENCE_NOTE,
        }
        y_cont = np.array([rows[i]["degradation"] for i in idxs])
        X_full, _ = assemble_matrix(feats, "full")
        X_base, _ = assemble_matrix(feats, "distance")

        # Naive block — anti-conservative diagnostic, never the headline.
        sf_n = oof_scores(X_full, y, model=model, n_splits=n_splits,
                          seed=seed, y_cont=y_cont)
        sb_n = oof_scores(X_base, y, model=model, n_splits=n_splits,
                          seed=seed, y_cont=y_cont)
        naive = auc_compare(y, sf_n, sb_n, n_boot=n_boot, seed=seed)
        naive["cv"] = ("StratifiedKFold over pairs — run overlap across "
                       "folds possible; anti-conservative")

        rep: dict = {"n_pairs": len(idxs), "naive": naive}
        # Group-aware block — the headline when feasible.
        try:
            sf_g = oof_scores(X_full, y, model=model, n_splits=n_splits,
                              seed=seed, y_cont=y_cont, groups=groups)
            sb_g = oof_scores(X_base, y, model=model, n_splits=n_splits,
                              seed=seed, y_cont=y_cont, groups=groups)
            ga = auc_compare(y, sf_g, sb_g, n_boot=n_boot, seed=seed,
                             groups=groups)
            ga["cv"] = ("StratifiedGroupKFold over run-overlap components "
                        "— no run appears in both train and test pairs")
            feasible, reason = True, None
        except ValueError as e:
            feasible, reason = False, str(e)
        dependence["group_cv_feasible"] = feasible
        if feasible:
            rep.update(ga)
            rep["group_aware"] = ga
            rep["headline_basis"] = "group_aware"
            pooled_y.append(y)
            pooled_f.append(sf_g)
            pooled_b.append(sb_g)
            pooled_g.append(groups + group_offset)
            group_offset += int(groups.max()) + 1
        else:
            dependence["group_cv_infeasible_reason"] = reason
            rep["group_aware"] = {
                "note": f"group-aware CV infeasible: {reason}"}
            rep["headline_basis"] = None
        rep["dependence"] = dependence
        per_family[fam] = rep

    if pooled_y:
        yy = np.concatenate(pooled_y)
        if yy.min() != yy.max():
            pooled = auc_compare(yy, np.concatenate(pooled_f),
                                 np.concatenate(pooled_b),
                                 n_boot=n_boot, seed=seed,
                                 groups=np.concatenate(pooled_g))
            pooled["basis"] = "group-aware out-of-fold scores only"
        else:
            pooled = {"note": "single-class pooled labels"}
    else:
        pooled = {"note": "no family produced group-aware scores"}

    return {
        "labels_file": str(labels_path),
        "binarization": bin_meta,
        "threshold": bin_meta["threshold"],
        "threshold_mode": bin_meta["rule_used"],
        "threshold_rel": threshold_rel,
        "degenerate_min_frac": degenerate_min_frac,
        "model": model,
        "seed": seed,
        "n_splits": n_splits,
        "n_boot": n_boot,
        "n_pairs_total": len(rows),
        "dependence_note": DEPENDENCE_NOTE,
        "per_family": per_family,
        "pooled_oof": pooled,
    }


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Asset 1 D3 — weight-only, post-hoc merge-degradation "
                    "prediction (features + merge constructor + harness). "
                    "Real-bank invocations are gated on the FULL 480-run "
                    "bank (pre-registration interlock).")
    parser.add_argument("--selftest", action="store_true",
                        help="run the synthetic planted-angle selftest "
                             "(no bank access)")
    parser.add_argument("--bank-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="master seed for sampling/folds/bootstrap "
                             "(default 0)")
    parser.add_argument("--allow-partial-bank", action="store_true",
                        help="EXPLORATORY ONLY: bypass the completeness "
                             "interlock with a loud pre-registration "
                             "warning; results must never be reported")
    parser.add_argument("--labels", type=Path, default=None,
                        help="labels file (schema in module docstring)")
    parser.add_argument("--label-threshold", default="median",
                        help="SECONDARY/descriptive median-split fallback: "
                             "'median' or a float. Used only when the "
                             "per-endpoint metric fields are absent (or the "
                             "primary rule is degenerate); ignored when a "
                             "'degraded' column is present (default: median)")
    parser.add_argument("--label-threshold-rel", type=float,
                        default=THRESHOLD_REL,
                        help=f"PRIMARY fixed relative-degradation threshold "
                             f"(Director override "
                             f"DIRECTOR_DECISIONS_2026-07-06.md): a merge is "
                             f"a conflict if it degrades EITHER endpoint by "
                             f">= this fraction relative to native "
                             f"(default {THRESHOLD_REL})")
    parser.add_argument("--degenerate-min-frac", type=float,
                        default=DEGENERATE_MIN_FRAC,
                        help=f"degenerate-balance floor: if the fixed rule "
                             f"yields fewer than this fraction of positives "
                             f"(or negatives), report the finding and fall "
                             f"back to the median split "
                             f"(default {DEGENERATE_MIN_FRAC})")
    parser.add_argument("--model", choices=("logistic", "ridge"),
                        default="logistic")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--make-pairs", type=int, default=None, metavar="N",
                        help="sample N pairs per family, compute features, "
                             "write d3_pairs.json")
    parser.add_argument("--max-run-uses", type=int,
                        default=DEFAULT_MAX_RUN_USES,
                        help=f"cap on pairs any single run may appear in "
                             f"(default {DEFAULT_MAX_RUN_USES} = "
                             f"vertex-disjoint pairs — the dyadic-"
                             f"dependence-safe design; 0 = unlimited "
                             f"legacy sampling, exploratory only)")
    parser.add_argument("--emit-merges", action="store_true",
                        help="with --make-pairs: also write merged adapter "
                             "states for GPU eval")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="merge interpolation weight on adapter b "
                             "(default 0.5 = midpoint)")
    args = parser.parse_args(argv)

    if args.selftest:
        rep = run_selftest(seed=args.seed, n_boot=args.n_boot)
        print(json.dumps(_jsonable(rep), indent=2))
        if args.out_dir is not None:
            out = guard_out_dir(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "d3_selftest.json").write_text(
                json.dumps(_jsonable(rep), indent=2), encoding="utf-8")
        if not rep["passed"]:
            raise SystemExit("[asset1-d3] selftest FAILED: full-feature AUC "
                             "did not beat the distance-only baseline")
        print("[asset1-d3] selftest PASSED")
        return

    if args.bank_root is None or args.out_dir is None:
        parser.error("--bank-root and --out-dir are required unless "
                     "--selftest")
    out_dir = guard_out_dir(args.out_dir)
    # THE INTERLOCK — refuse a partial bank (pre-registration hygiene).
    aio.require_complete_bank(args.bank_root,
                              allow_partial=args.allow_partial_bank)
    out_dir.mkdir(parents=True, exist_ok=True)

    did_something = False
    if args.make_pairs is not None:
        max_uses = (None if args.max_run_uses is not None
                    and args.max_run_uses <= 0 else args.max_run_uses)
        payload = make_pairs_command(args.bank_root, out_dir,
                                     args.make_pairs, args.seed, args.alpha,
                                     args.emit_merges,
                                     max_run_uses=max_uses)
        print(f"[asset1-d3] wrote {payload['n_pairs']} pair records to "
              f"{out_dir / 'd3_pairs.json'} "
              f"(max_run_uses={payload['max_run_uses']})")
        did_something = True
    if args.labels is not None:
        report = fit_from_labels(args.bank_root, args.labels,
                                 args.label_threshold, args.model,
                                 args.n_splits, args.n_boot, args.seed,
                                 threshold_rel=args.label_threshold_rel,
                                 degenerate_min_frac=args.degenerate_min_frac)
        (out_dir / "d3_report.json").write_text(
            json.dumps(_jsonable(report), indent=2), encoding="utf-8")
        _bm = report["binarization"]
        print(f"[asset1-d3] binarization rule: {_bm['rule_used']} "
              f"(primary threshold_rel={_bm['threshold_rel']}, "
              f"degenerate_min_frac={_bm['degenerate_min_frac']})")
        if _bm.get("degenerate_finding"):
            print(f"[asset1-d3] FINDING: {_bm['degenerate_finding']}")
        for fam, rep in report["per_family"].items():
            if "auc_full" in rep:
                print(f"[asset1-d3] {fam}: AUC full={rep['auc_full']:.3f} "
                      f"vs distance={rep['auc_distance']:.3f} "
                      f"(n={rep['n_pairs']}, group-aware CV)")
            elif "naive" in rep:
                print(f"[asset1-d3] {fam}: group-aware CV infeasible "
                      f"({rep['dependence']['group_cv_infeasible_reason']})"
                      f" — naive-only numbers are anti-conservative and "
                      f"must not be reported "
                      f"(naive AUC full={rep['naive']['auc_full']:.3f}, "
                      f"n={rep['n_pairs']})")
            else:
                print(f"[asset1-d3] {fam}: {rep['note']}")
        print(f"[asset1-d3] report written to {out_dir / 'd3_report.json'}")
        did_something = True
    if not did_something:
        parser.error("nothing to do — pass --labels, --make-pairs, or "
                     "--selftest")


if __name__ == "__main__":
    main()
