"""Tests for Asset 1 D3 (weight-only merge prediction) + D-aux (overfit
detection).

Covers: principal-angle gauge invariance (GL(r) on the absorbed factors,
1e-10; identity-bridge raw-factor reparameterization), angle correctness on
constructed subspaces (known planted angles, identical and orthogonal
subspaces), the exact SV/delta-magnitude weighting math (Gram-trick
magnitudes vs direct ||scaling * B E A||_F, weighted aggregate identity),
the merge constructor (alpha interpolation, metadata checks, torch.save
round-trip through the production loader), the prediction harness (labels
JSON/CSV round-trip, binarization, separable-toy AUC, paired bootstrap
determinism, class guards), the DYADIC-DEPENDENCE machinery (round-1
review fix: run-overlap connected components via pair_group_ids,
group-aware CV blocking run leakage that naive pair-level CV exploits,
component-cluster bootstrap CIs wider than the anti-conservative pair-iid
CIs under dependence, vertex-disjoint pair sampling, group-infeasible
reporting path), the D3 planted-angle + dependence-control selftest,
D-aux deviation/update/gap metrics on constructed inputs, correlation
machinery (linear/monotone, zero-variance note, bootstrap determinism,
Simpson's-paradox stratification), recovery of the asset1_synth planted
deviation<->gap correlation (and its absence at task_effect=0), the CLI
interlock wiring (--allow-partial-bank), and the out-dir write guard.

Pre-registration hygiene: every statistic below runs on SYNTHETIC data —
in-memory constructions or asset1_synth fixtures in tmp dirs. Nothing
reads or writes results/asset1-bank/, no HF downloads, no network, no GPU
(all torch work is CPU tensors).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

torch = pytest.importorskip("torch")

import asset1_analysis_io as aio  # noqa: E402
import asset1_d3_merge as d3  # noqa: E402
import asset1_daux_gap as daux  # noqa: E402
import asset1_synth as synth  # noqa: E402
from asset1_canonicalize import load_adapter_modules  # noqa: E402
from rhombic.nn.absorb import _expand_bridge  # noqa: E402


# ── Helpers / fixtures ──────────────────────────────────────────────


def _entry(B, A, bridge=None, scaling=1.0):
    """In-memory adapter module entry in the bank's field layout."""
    B = torch.as_tensor(np.asarray(B), dtype=torch.float32)
    A = torch.as_tensor(np.asarray(A), dtype=torch.float32)
    rank = B.shape[1]
    if bridge is None:
        bridge = np.eye(2)
    bridge = torch.as_tensor(np.asarray(bridge), dtype=torch.float32)
    return {"lora_A": A, "lora_B": B, "bridge": bridge,
            "scaling": torch.tensor(float(scaling)),
            "n_channels": torch.tensor(bridge.shape[0]),
            "rank": torch.tensor(rank)}


def _random_adapter(rng, n_modules=2, d_out=16, d_in=12, rank=4,
                    scaling=0.5):
    return {f"m{k:02d}": _entry(rng.standard_normal((d_out, rank)),
                                rng.standard_normal((rank, d_in)),
                                bridge=np.eye(2), scaling=scaling)
            for k in range(n_modules)}


def _well_conditioned(rng, r, max_cond=100.0):
    for _ in range(64):
        G = rng.standard_normal((r, r))
        if np.linalg.cond(G) < max_cond:
            return G
    raise RuntimeError("no well-conditioned G")


@pytest.fixture(scope="module")
def pairs_bank(tmp_path_factory):
    """2 families x 3 tasks x 2 reps = 12 runs (CLI make-pairs/labels)."""
    root = tmp_path_factory.mktemp("d3-pairs") / "bank"
    info = synth.make_synthetic_bank(
        root, n_families=2, n_tasks=3, n_reps=2, n_layers=2,
        d_model=16, rank=4, n_channels=2, task_effect=1.0, seed=7)
    return root, info


@pytest.fixture(scope="module")
def daux_effect_bank(tmp_path_factory):
    """1 family x 2 tasks x 8 reps = 16 runs, planted D-aux correlation."""
    root = tmp_path_factory.mktemp("daux-effect") / "bank"
    info = synth.make_synthetic_bank(
        root, n_families=1, n_tasks=2, n_reps=8, n_layers=1,
        d_model=12, rank=4, n_channels=2, task_effect=1.0, seed=21)
    return root, info


@pytest.fixture(scope="module")
def daux_zero_bank(tmp_path_factory):
    """Same shape, task_effect = 0 — no planted correlation."""
    root = tmp_path_factory.mktemp("daux-zero") / "bank"
    info = synth.make_synthetic_bank(
        root, n_families=1, n_tasks=2, n_reps=8, n_layers=1,
        d_model=12, rank=4, n_channels=2, task_effect=0.0, seed=21)
    return root, info


# ── Principal angles: correctness on constructed subspaces ──────────


def test_principal_angles_known_construction():
    """Planted angle t between 2-D subspaces of R^6: span(e1, e2) vs
    span(cos t * e1 + sin t * e3, e2) -> angles {0, t}, robust to
    invertible within-subspace mixing."""
    t = 0.7
    e = np.eye(6)
    S1 = np.stack([e[0], e[1]], axis=1)                      # (6, 2)
    S2 = np.stack([np.cos(t) * e[0] + np.sin(t) * e[2], e[1]], axis=1)
    rng = np.random.default_rng(0)
    B1 = S1 @ _well_conditioned(rng, 2)
    B2 = S2 @ _well_conditioned(rng, 2)
    angles = d3.principal_angles(B1, B2)
    assert angles.shape == (2,)
    assert angles[0] == pytest.approx(0.0, abs=1e-7)
    assert angles[1] == pytest.approx(t, abs=1e-10)


def test_principal_angles_identical_subspace_zero():
    rng = np.random.default_rng(1)
    B1 = rng.standard_normal((20, 4))
    B2 = B1 @ _well_conditioned(rng, 4)
    angles = d3.principal_angles(B1, B2)
    assert angles.shape == (4,)
    assert np.all(angles < 1e-6)      # arccos noise floor near cos = 1


def test_principal_angles_orthogonal_subspaces():
    e = np.eye(8)
    B1 = np.stack([e[0], e[1]], axis=1)
    B2 = np.stack([e[2], e[3]], axis=1)
    angles = d3.principal_angles(B1, B2)
    assert np.allclose(angles, np.pi / 2, atol=1e-12)


def test_principal_angles_empty_colspace():
    B1 = np.zeros((10, 3))
    B2 = np.random.default_rng(2).standard_normal((10, 3))
    assert d3.principal_angles(B1, B2).size == 0
    assert d3.orthonormal_colspace(B1).shape == (10, 0)


# ── Principal angles: gauge invariance ──────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_principal_angles_gl_invariance(seed):
    """GL(r) on the (bridge-absorbed) factors leaves the angles unchanged
    within 1e-10 — the pre-registered invariance property."""
    rng = np.random.default_rng(seed)
    B1 = rng.standard_normal((40, 6))
    B2 = rng.standard_normal((40, 6))
    G1 = _well_conditioned(rng, 6)
    G2 = _well_conditioned(rng, 6)
    a_ref = d3.principal_angles(B1, B2)
    a_gl = d3.principal_angles(B1 @ G1, B2 @ G2)
    assert a_ref.shape == a_gl.shape
    assert np.max(np.abs(a_ref - a_gl)) < 1e-10


def test_pair_angles_gauge_invariance_identity_bridge():
    """With an identity bridge, (B G, G^{-1} A) preserves the effective
    update, so ALL angle features of pair_features are unchanged (raw
    flattened distances legitimately change — that is the point of the
    gauge-invariant feature block)."""
    rng = np.random.default_rng(3)
    adapter = _random_adapter(rng, n_modules=2, rank=4)
    other = _random_adapter(rng, n_modules=2, rank=4)

    reparam = {}
    for name, e in adapter.items():
        G = torch.as_tensor(_well_conditioned(rng, 4), dtype=torch.float64)
        B2 = (e["lora_B"].to(torch.float64) @ G).to(torch.float32)
        A2 = (torch.linalg.inv(G) @ e["lora_A"].to(torch.float64)
              ).to(torch.float32)
        reparam[name] = {**e, "lora_A": A2, "lora_B": B2}

    pf1 = d3.pair_features(adapter, other)
    pf2 = d3.pair_features(reparam, other)
    for key in d3.AGGREGATE_KEYS:
        assert pf2[key] == pytest.approx(pf1[key], abs=1e-6), key
    assert np.allclose(pf1["module_angle_mean"], pf2["module_angle_mean"],
                       atol=1e-6)
    # Delta magnitudes (the weights) are preserved too: same effective DW.
    assert np.allclose(pf1["module_weight"], pf2["module_weight"],
                       rtol=1e-4)


def test_module_angles_bridge_absorption():
    """An entry with bridge E equals (in angle terms) a twin whose lora_B
    pre-multiplies the expanded E and carries an identity bridge — proves
    the absorption path feeds the angle computation."""
    rng = np.random.default_rng(4)
    B = rng.standard_normal((12, 4))
    A = rng.standard_normal((4, 8))
    bridge = np.array([[1.0, 0.4], [0.2, 0.7]])
    e1 = _entry(B, A, bridge=bridge, scaling=0.5)
    E = _expand_bridge(torch.as_tensor(bridge, dtype=torch.float64), 2)
    B_absorbed = torch.as_tensor(B, dtype=torch.float64) @ E
    e2 = _entry(B_absorbed.numpy(), A, bridge=np.eye(2), scaling=0.5)
    angles = d3.module_pair_angles(e1, e2)
    assert angles.size == 4
    assert np.all(angles < 1e-6)
    # An INVERTIBLE bridge is a gauge on col(B_eff): col(B E) == col(B), so
    # angles vs the identity-bridge twin are ~0 (documented property).
    e3 = _entry(B, A, bridge=np.eye(2), scaling=0.5)
    assert np.max(d3.module_pair_angles(e1, e3)) < 1e-6
    # A SINGULAR bridge shrinks the column space: rank-1 bridge -> rank-2
    # expanded E -> only min(2, 4) = 2 principal angles survive.
    e4 = _entry(B, A, bridge=np.array([[1.0, 0.0], [0.0, 0.0]]),
                scaling=0.5)
    assert d3.module_pair_angles(e4, e3).size == 2


# ── Delta magnitude + SV weighting math ─────────────────────────────


def test_delta_magnitude_matches_direct():
    rng = np.random.default_rng(5)
    B = rng.standard_normal((10, 4))
    A = rng.standard_normal((4, 6))
    bridge = np.eye(2) + 0.3 * rng.standard_normal((2, 2))
    scaling = 0.37
    e = _entry(B, A, bridge=bridge, scaling=scaling)
    E = _expand_bridge(torch.as_tensor(bridge, dtype=torch.float64), 2)
    direct = float(torch.linalg.norm(
        scaling * (torch.as_tensor(B, dtype=torch.float64) @ E
                   @ torch.as_tensor(A, dtype=torch.float64))))
    # float32 storage round-trip bounds the achievable agreement
    assert d3.delta_magnitude(e) == pytest.approx(direct, rel=1e-5)


def test_sv_weighting_math():
    """module_weight == sqrt(mag_a * mag_b) per module, and the weighted
    aggregate == sum(w * s) / sum(w); a heavy near-aligned module pulls
    the weighted mean below the unweighted mean."""
    rng = np.random.default_rng(6)
    # m00: tiny magnitude, orthogonal-ish subspaces (big angles)
    B_small_a = 0.01 * np.stack([np.eye(12)[0], np.eye(12)[1],
                                 np.eye(12)[2], np.eye(12)[3]], axis=1)
    B_small_b = 0.01 * np.stack([np.eye(12)[4], np.eye(12)[5],
                                 np.eye(12)[6], np.eye(12)[7]], axis=1)
    # m01: large magnitude, identical subspace (zero angles)
    B_big = 10.0 * rng.standard_normal((12, 4))
    A = rng.standard_normal((4, 8))
    a = {"m00": _entry(B_small_a, A), "m01": _entry(B_big, A)}
    b = {"m00": _entry(B_small_b, A),
         "m01": _entry(B_big @ _well_conditioned(rng, 4), A)}
    pf = d3.pair_features(a, b)

    for i, name in enumerate(pf["module_names"]):
        expected = np.sqrt(d3.delta_magnitude(a[name])
                           * d3.delta_magnitude(b[name]))
        assert pf["module_weight"][i] == pytest.approx(expected, rel=1e-6)

    s = pf["module_angle_mean"]
    w = pf["module_weight"]
    assert pf["angle_mean_unweighted"] == pytest.approx(s.mean(), rel=1e-12)
    assert pf["angle_mean_weighted"] == pytest.approx(
        float(np.sum(w * s) / np.sum(w)), rel=1e-12)
    # m00 is ~pi/2, m01 ~0; weight sits on m01 -> weighted << unweighted
    assert pf["angle_mean_weighted"] < 0.1 * pf["angle_mean_unweighted"]


def test_pair_features_rejects_mismatched_modules():
    rng = np.random.default_rng(7)
    a = _random_adapter(rng, n_modules=2)
    b = _random_adapter(rng, n_modules=3)
    with pytest.raises(ValueError, match="SAME-FAMILY"):
        d3.pair_features(a, b)


# ── Merge constructor ───────────────────────────────────────────────


def test_merge_alpha_interpolation():
    rng = np.random.default_rng(8)
    a = _random_adapter(rng, n_modules=2)
    b = _random_adapter(rng, n_modules=2)
    alpha = 0.25
    merged = d3.merge_adapters(a, b, alpha=alpha)
    for name in a:
        for field in ("lora_A", "lora_B", "bridge"):
            expected = (1 - alpha) * a[name][field] + alpha * b[name][field]
            got = merged[f"{name}.{field}"]
            assert got.dtype == torch.float32
            assert torch.allclose(got, expected, atol=1e-6), (name, field)
        for field in ("scaling", "n_channels", "rank"):
            assert torch.equal(merged[f"{name}.{field}"], a[name][field])


def test_merge_validation_errors():
    rng = np.random.default_rng(9)
    a = _random_adapter(rng, n_modules=2)
    b = _random_adapter(rng, n_modules=2)
    with pytest.raises(ValueError, match="alpha"):
        d3.merge_adapters(a, b, alpha=1.5)
    b_missing = {k: v for k, v in b.items() if k != "m01"}
    with pytest.raises(ValueError, match="module sets"):
        d3.merge_adapters(a, b_missing)
    b_scale = {k: dict(v) for k, v in b.items()}
    b_scale["m00"]["scaling"] = torch.tensor(999.0)
    with pytest.raises(ValueError, match="scaling"):
        d3.merge_adapters(a, b_scale)
    b_shape = {k: dict(v) for k, v in b.items()}
    b_shape["m00"]["lora_A"] = torch.zeros(4, 99)
    with pytest.raises(ValueError, match="shape"):
        d3.merge_adapters(a, b_shape)


def test_merge_roundtrip_through_production_loader(tmp_path):
    """torch.save(merged) loads back through the exact loader the GPU eval
    will use (asset1_canonicalize.load_adapter_modules, weights_only)."""
    rng = np.random.default_rng(10)
    a = _random_adapter(rng, n_modules=2)
    b = _random_adapter(rng, n_modules=2)
    merged = d3.merge_adapters(a, b, alpha=0.5)
    path = tmp_path / "merged.pt"
    torch.save(merged, path)
    loaded = load_adapter_modules(path)
    assert set(loaded) == set(a)
    for name in a:
        mid = 0.5 * (a[name]["lora_B"] + b[name]["lora_B"])
        assert torch.allclose(loaded[name]["lora_B"], mid, atol=1e-6)


# ── Labels + harness ────────────────────────────────────────────────

# load_labels now canonicalizes the optional per-endpoint metric fields
# (Director override D3) — absent values come back as None.
_METRIC_NONE = {k: None for k in d3._LABEL_OPTIONAL_METRICS}
_LABEL_ROWS = [
    {"family_short": "famX", "task_a": "t0", "run_index_a": 0,
     "task_b": "t1", "run_index_b": 3, "degradation": 0.4, "degraded": None,
     **_METRIC_NONE},
    {"family_short": "famX", "task_a": "t0", "run_index_a": 1,
     "task_b": "t0", "run_index_b": 2, "degradation": 0.1, "degraded": None,
     **_METRIC_NONE},
]


def test_labels_json_csv_roundtrip(tmp_path):
    # The canonical rows now carry the optional per-endpoint metric fields
    # (None when absent) per the extended D3 labels schema; the round-trip
    # still drops None fields in JSON and re-hydrates them on load.
    jpath = tmp_path / "labels.json"
    jpath.write_text(json.dumps(
        {"pairs": [{k: v for k, v in r.items() if v is not None}
                   for r in _LABEL_ROWS]}), encoding="utf-8")
    cpath = tmp_path / "labels.csv"
    cpath.write_text(
        "family_short,task_a,run_index_a,task_b,run_index_b,degradation\n"
        "famX,t0,0,t1,3,0.4\n"
        "famX,t0,1,t0,2,0.1\n", encoding="utf-8")
    assert d3.load_labels(jpath) == _LABEL_ROWS
    assert d3.load_labels(cpath) == _LABEL_ROWS
    bad = tmp_path / "bad.csv"
    bad.write_text("family_short,task_a\nfamX,t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        d3.load_labels(bad)


def test_binarize_labels():
    rows = [dict(r, degradation=d, degraded=None)
            for r, d in zip(_LABEL_ROWS * 2, [0.1, 0.2, 0.3, 0.4])]
    y, thr = d3.binarize_labels(rows, "median")
    assert thr == pytest.approx(0.25)
    assert y.tolist() == [0, 0, 1, 1]
    y2, thr2 = d3.binarize_labels(rows, 0.35)
    assert thr2 == pytest.approx(0.35)
    assert y2.tolist() == [0, 0, 0, 1]
    # explicit degraded column wins; threshold not used
    rows_dc = [dict(r, degraded=g) for r, g in zip(rows, [1, 0, 1, 0])]
    y3, thr3 = d3.binarize_labels(rows_dc, "median")
    assert thr3 is None
    assert y3.tolist() == [1, 0, 1, 0]
    with pytest.raises(ValueError, match="not all"):
        d3.binarize_labels([rows[0], rows_dc[1]], "median")


# ── Primary relative-degradation binarization (Director override D3) ──


def _ppl_row(rel_a, rel_b, base=10.0, degradation=0.0):
    """A labels row carrying per-endpoint perplexity metrics (up = worse)."""
    return {"family_short": "famX", "task_a": "t0", "run_index_a": 0,
            "task_b": "t1", "run_index_b": 1, "degradation": degradation,
            "degraded": None,
            "merged_ppl_a": base * (1.0 + rel_a), "native_ppl_a": base,
            "merged_ppl_b": base * (1.0 + rel_b), "native_ppl_b": base}


def test_d3_relative_constants_recorded():
    """The pinned constants (Director override D3) are exactly 5% / 10%."""
    assert d3.THRESHOLD_REL == 0.05
    assert d3.DEGENERATE_MIN_FRAC == 0.10


def test_binarize_primary_relative_rule():
    """PRIMARY rule: a pair degrading EITHER endpoint by >= 5% relative is
    positive; below 5% on both endpoints is negative; the boundary at
    exactly 5% is positive (>=)."""
    rows = ([_ppl_row(0.06, 0.0) for _ in range(4)]      # a up 6% -> conflict
            + [_ppl_row(0.0, 0.05)]                       # b exactly 5% -> pos
            + [_ppl_row(0.049, 0.0) for _ in range(5)])   # 4.9% -> negative
    y, meta = d3.binarize_primary(rows)
    assert y.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert meta["rule_used"] == "relative_degradation"
    assert meta["threshold_rel"] == d3.THRESHOLD_REL
    assert meta["degenerate_min_frac"] == d3.DEGENERATE_MIN_FRAC
    assert meta["degenerate"] is False
    assert meta["frac_positive_relative"] == pytest.approx(0.5)
    assert meta["threshold"] is None


def test_binarize_primary_score_form():
    """Task-metric DOWN >= 5% relative is also a conflict (score form)."""
    def srow(drop):
        return {"family_short": "f", "task_a": "t0", "run_index_a": 0,
                "task_b": "t1", "run_index_b": 1, "degradation": 0.0,
                "degraded": None,
                "merged_score_a": 1.0 - drop, "native_score_a": 1.0,
                "merged_score_b": 1.0, "native_score_b": 1.0}
    rows = [srow(0.06), srow(0.06), srow(0.06), srow(0.0), srow(0.0),
            srow(0.0)]
    y, meta = d3.binarize_primary(rows)
    assert y.tolist() == [1, 1, 1, 0, 0, 0]
    assert meta["rule_used"] == "relative_degradation"


def test_binarize_primary_degenerate_fallback():
    """When the fixed 5% rule yields < 10% positives the imbalance is
    reported as a finding and the headline falls back to the pre-declared
    median split (Director override D3)."""
    rows = [_ppl_row(0.06, 0.0, degradation=100.0)]           # lone positive
    rows += [_ppl_row(0.0, 0.0, degradation=float(i))         # 19 negatives
             for i in range(19)]
    y, meta = d3.binarize_primary(rows)
    assert meta["degenerate"] is True
    assert meta["frac_positive_relative"] == pytest.approx(1 / 20)
    assert meta["rule_used"] == "median_fallback"
    assert "degenerate_finding" in meta and meta["degenerate_finding"]
    deg = np.array([r["degradation"] for r in rows])
    assert meta["threshold"] == pytest.approx(float(np.median(deg)))
    assert y.tolist() == (deg > meta["threshold"]).astype(int).tolist()


def test_binarize_primary_no_metrics_defers_to_secondary():
    """Without per-endpoint metrics, binarize_primary defers: an explicit
    'degraded' column is used directly (threshold None), else the median
    split (secondary/descriptive under the override)."""
    dc_rows = [dict(r, degraded=1) for r in _LABEL_ROWS]
    y, meta = d3.binarize_primary(dc_rows)
    assert meta["rule_used"] == "degraded_column"
    assert meta["threshold"] is None
    assert y.tolist() == [1, 1]
    med_rows = [dict(r, degradation=d, degraded=None)
                for r, d in zip(_LABEL_ROWS * 2, [0.1, 0.2, 0.3, 0.4])]
    y2, meta2 = d3.binarize_primary(med_rows)
    assert meta2["rule_used"] == "median"
    assert meta2["threshold"] == pytest.approx(0.25)
    assert y2.tolist() == [0, 0, 1, 1]


def test_harness_separable_toy():
    rng = np.random.default_rng(11)
    y = np.array([0] * 20 + [1] * 20)
    X = np.column_stack([2.0 * y + 0.1 * rng.standard_normal(40),
                         rng.standard_normal(40)])
    scores = d3.oof_scores(X, y, model="logistic", n_splits=5, seed=0)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, scores) > 0.95
    # ridge path runs and separates too
    scores_r = d3.oof_scores(X, y, model="ridge", n_splits=5, seed=0,
                             y_cont=y.astype(float))
    assert roc_auc_score(y, scores_r) > 0.95


def test_harness_bootstrap_deterministic():
    rng = np.random.default_rng(12)
    y = np.array([0] * 15 + [1] * 15)
    sf = y + 0.3 * rng.standard_normal(30)
    sb = rng.standard_normal(30)
    r1 = d3.auc_compare(y, sf, sb, n_boot=100, seed=5)
    r2 = d3.auc_compare(y, sf, sb, n_boot=100, seed=5)
    assert r1 == r2
    assert set(r1) >= {"auc_full", "auc_full_ci", "auc_distance",
                       "auc_distance_ci", "auc_diff", "auc_diff_ci",
                       "n_boot_valid"}
    assert r1["auc_full"] > r1["auc_distance"]
    lo, hi = r1["auc_diff_ci"]
    assert lo <= r1["auc_diff"] <= hi


def test_harness_class_guards():
    X = np.zeros((11, 2))
    with pytest.raises(ValueError, match="2 samples"):
        d3.oof_scores(X, np.array([0] * 10 + [1]), seed=0)
    with pytest.raises(ValueError, match="2 classes"):
        d3.oof_scores(X, np.zeros(11, dtype=int), seed=0)
    with pytest.raises(ValueError, match="feature_set"):
        d3.assemble_matrix([{}], "everything")


# ── Dyadic dependence (round-1 review fix) ──────────────────────────


def test_pair_group_ids_connected_components():
    """Union-find correctness: chained pairs merge into one component,
    disjoint pairs stay separate, ids are dense first-appearance ints."""
    keys = [("a", "b"), ("b", "c"),          # component 0: a-b-c chain
            ("d", "e"),                       # component 1
            ("c", "a"),                       # still component 0
            ("f", "g"), ("g", "h"), ("h", "f")]   # component 2: triangle
    g = d3.pair_group_ids(keys)
    assert g.tolist() == [0, 0, 1, 0, 2, 2, 2]
    # vertex-disjoint pairs: every pair its own component
    g2 = d3.pair_group_ids([("p", "q"), ("r", "s"), ("t", "u")])
    assert g2.tolist() == [0, 1, 2]


def test_group_cv_prevents_run_leakage():
    """THE ROUND-1 REGRESSION TEST: pair labels determined purely by run
    latents, features encoding only run identity (triangles of runs).
    Naive StratifiedKFold over pairs reads held-out pairs' run
    coefficients from same-run training pairs -> inflated AUC on what is
    pure run-identity leakage. Group-aware CV over run-overlap components
    holds out whole triangles -> no signal. The original harness had only
    the naive path, so this leakage was invisible."""
    from sklearn.metrics import roc_auc_score
    X, y, groups = d3.make_leakage_fixture(n_triangles=20, seed=2)
    s_naive = d3.oof_scores(X, y, model="logistic", n_splits=5, seed=2)
    s_group = d3.oof_scores(X, y, model="logistic", n_splits=5, seed=2,
                            groups=groups)
    auc_naive = roc_auc_score(y, s_naive)
    auc_group = roc_auc_score(y, s_group)
    assert auc_naive > 0.8, "leakage fixture failed to inflate naive CV"
    assert auc_group < 0.65, \
        f"group-aware CV leaked run identity (AUC {auc_group})"
    assert auc_naive - auc_group > 0.3


def test_oof_scores_group_guards_and_determinism():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((12, 3))
    y = np.array([0, 1] * 6)
    groups = np.repeat(np.arange(4), 3)
    s1 = d3.oof_scores(X, y, n_splits=3, seed=1, groups=groups)
    s2 = d3.oof_scores(X, y, n_splits=3, seed=1, groups=groups)
    assert np.array_equal(s1, s2)
    # a single connected component makes group-aware CV impossible
    with pytest.raises(ValueError, match="single connected component"):
        d3.oof_scores(X, y, n_splits=3, seed=1, groups=np.zeros(12))
    with pytest.raises(ValueError, match="groups shape"):
        d3.oof_scores(X, y, n_splits=3, seed=1, groups=np.zeros(5))


def test_cluster_bootstrap_wider_than_pair_iid_under_dependence():
    """Duplicate every pair 5x within its component (extreme dyadic
    dependence: effective n = 20 components, nominal n = 100 pairs). The
    pair-iid CI treats 100 rows as independent -> too narrow; the
    component-cluster bootstrap resamples the 20 components -> wider,
    honest CIs. This is the anti-conservatism the finding flagged."""
    rng = np.random.default_rng(7)
    n_comp, reps = 20, 5
    y0 = (np.arange(n_comp) % 2).astype(int)
    s_full0 = y0 + 0.8 * rng.standard_normal(n_comp)
    s_base0 = rng.standard_normal(n_comp)
    y = np.repeat(y0, reps)
    sf = np.repeat(s_full0, reps)
    sb = np.repeat(s_base0, reps)
    groups = np.repeat(np.arange(n_comp), reps)
    naive = d3.auc_compare(y, sf, sb, n_boot=400, seed=3)
    clust = d3.auc_compare(y, sf, sb, n_boot=400, seed=3, groups=groups)
    assert naive["bootstrap_unit"] == "pair-iid"
    assert clust["bootstrap_unit"] == "run-component-cluster"
    # point estimates identical (same scores), CIs differ in width
    assert clust["auc_full"] == naive["auc_full"]
    w_naive = naive["auc_full_ci"][1] - naive["auc_full_ci"][0]
    w_clust = clust["auc_full_ci"][1] - clust["auc_full_ci"][0]
    assert w_clust > w_naive * 1.5, (w_naive, w_clust)
    # deterministic
    clust2 = d3.auc_compare(y, sf, sb, n_boot=400, seed=3, groups=groups)
    assert clust == clust2


def test_sample_pairs_max_run_uses(pairs_bank):
    """Default sampling is vertex-disjoint (no run in two pairs) — the
    at-the-source dependence fix; unlimited mode reproduces run reuse."""
    root, _ = pairs_bank
    pairs = d3.sample_pairs(root, n_per_family=10, seed=0)  # default cap 1
    per_fam_runs: dict = {}
    for p in pairs:
        runs = per_fam_runs.setdefault(p["family_short"], [])
        runs.extend([p["run_index_a"], p["run_index_b"]])
    for fam, runs in per_fam_runs.items():
        assert len(runs) == len(set(runs)), f"run reuse in {fam}"
        assert len(runs) <= 6                # 6 runs/family -> <= 3 pairs
    # determinism
    pairs2 = d3.sample_pairs(root, n_per_family=10, seed=0)
    assert pairs == pairs2
    # unlimited mode can reuse runs and reach the full quota
    pairs_u = d3.sample_pairs(root, n_per_family=10, seed=0,
                              max_run_uses=None)
    fam0 = [p for p in pairs_u if p["family_short"] == "synthfam0"]
    assert len(fam0) == 10
    with pytest.raises(ValueError, match="max_run_uses"):
        d3.sample_pairs(root, n_per_family=2, seed=0, max_run_uses=0)


# ── D3 synthetic selftest ───────────────────────────────────────────


def test_d3_selftest_full_beats_distance():
    rep = d3.run_selftest(seed=0, n_per_class=12, n_splits=4, n_boot=60)
    assert rep["passed"], rep
    assert rep["auc_full"] >= rep["auc_distance"] + 0.15
    assert rep["auc_full"] >= 0.85
    # the dependence control (round-1 review fix) is part of the selftest:
    # naive CV inflates on run-identity leakage, group-aware CV does not
    dep = rep["dependence_control"]
    assert dep["passed"]
    assert dep["auc_naive_cv"] >= dep["auc_group_aware_cv"] + 0.15
    assert dep["auc_group_aware_cv"] <= 0.70
    # deterministic under the same seed
    rep2 = d3.run_selftest(seed=0, n_per_class=12, n_splits=4, n_boot=60)
    assert rep2["auc_full"] == rep["auc_full"]
    assert rep2["dependence_control"] == dep


# ── D-aux: metrics on constructed inputs ────────────────────────────


def test_deviation_metrics_constructed():
    delta = np.array([[0.0, 0.3], [0.4, 0.0]])
    bridges = {"b0": np.eye(2), "b1": np.eye(2) + delta}
    m = daux.bridge_deviation_metrics(bridges)
    assert m["dev_mean"] == pytest.approx(0.25)     # (0 + 0.5) / 2
    assert m["dev_max"] == pytest.approx(0.5)       # ||delta||_F = 0.5
    assert m["n_modules"] == 2
    with pytest.raises(ValueError, match="no bridges"):
        daux.bridge_deviation_metrics({})
    with pytest.raises(ValueError, match="square"):
        daux.bridge_deviation_metrics({"x": np.zeros((2, 3))})


def test_update_magnitude_metrics():
    rng = np.random.default_rng(13)
    adapter = _random_adapter(rng, n_modules=3)
    m = daux.update_magnitude_metrics(adapter)
    mags = [d3.delta_magnitude(adapter[n]) for n in sorted(adapter)]
    assert m["update_mag_mean"] == pytest.approx(np.mean(mags))
    assert m["update_mag_max"] == pytest.approx(np.max(mags))


def test_gap_metrics_constructed():
    steps = np.array([0, 100, 200])
    train = np.array([np.nan, 1.2, 1.0])
    val = np.array([1.8, 1.5, 1.4])
    g = daux.gap_metrics(steps, train, val)
    assert g["final_gap"] == pytest.approx(0.4)
    # trapezoid over finite gaps (0.3 @ 100, 0.4 @ 200) = 0.35 * 100
    assert g["gap_auc"] == pytest.approx(35.0)
    assert g["n_finite"] == 2
    g2 = daux.gap_metrics(np.array([0]), np.array([np.nan]),
                          np.array([1.0]))
    assert np.isnan(g2["final_gap"]) and np.isnan(g2["gap_auc"])


# ── D-aux: correlation machinery ────────────────────────────────────


def test_correlation_linear_and_monotone():
    x = np.linspace(0, 1, 20)
    lin = daux._corr_cell(x, 2 * x + 1, n_boot=50, rng_key=[0, 404, 0])
    assert lin["pearson_r"] == pytest.approx(1.0)
    assert lin["spearman_r"] == pytest.approx(1.0)
    mono = daux._corr_cell(x, np.exp(5 * x), n_boot=50, rng_key=[0, 404, 1])
    assert mono["spearman_r"] == pytest.approx(1.0)
    assert mono["pearson_r"] < 0.95


def test_correlation_zero_variance_and_small_n():
    c = daux._corr_cell(np.ones(10), np.arange(10.0), n_boot=50,
                        rng_key=[0, 404, 2])
    assert c["pearson_r"] is None
    assert "zero-variance" in c["note"]
    c2 = daux._corr_cell([1.0, 2.0], [1.0, 2.0], n_boot=50,
                         rng_key=[0, 404, 3])
    assert c2["pearson_r"] is None
    assert "insufficient n" in c2["note"]


def test_correlation_bootstrap_deterministic_and_brackets():
    rng = np.random.default_rng(14)
    x = np.linspace(0, 1, 30)
    y = x + 0.2 * rng.standard_normal(30)
    c1 = daux._corr_cell(x, y, n_boot=200, rng_key=[3, 404, 0])
    c2 = daux._corr_cell(x, y, n_boot=200, rng_key=[3, 404, 0])
    assert c1 == c2
    lo, hi = c1["pearson_ci"]
    assert lo < c1["pearson_r"] < hi
    lo_s, hi_s = c1["spearman_ci"]
    assert lo_s < c1["spearman_r"] < hi_s


def test_simpson_stratification_guard():
    """Within-task slope negative, between-task means positive: the pooled
    cell must NOT hide the within-task reversal — that is exactly what the
    stratified cells are for."""
    rows = []
    for t, (x0, y0) in {"tA": (0.0, 10.0), "tB": (2.0, 14.0)}.items():
        for i in range(6):
            x = x0 + i * 0.2
            rows.append({"family_short": "f0", "task": t,
                         "xv": x, "yv": y0 - (x - x0)})
    rep = daux.correlation_report(rows, "xv", "yv", n_boot=50, seed=0)
    assert rep["cells"]["pooled"]["pearson_r"] > 0.5
    assert rep["cells"]["task:f0/tA"]["pearson_r"] == pytest.approx(-1.0)
    assert rep["cells"]["task:f0/tB"]["pearson_r"] == pytest.approx(-1.0)
    assert rep["cells"]["family:f0"]["n"] == 12


# ── D-aux: recovery of the asset1_synth plant ───────────────────────


def test_daux_run_table_matches_planted_values(daux_effect_bank):
    root, _ = daux_effect_bank
    rows = daux.collect_run_table(root)
    assert len(rows) == 16
    recs = {r["run_index"]: r for r in aio.iter_runs(root)}
    for row in rows:
        plant = recs[row["run_index"]]["config"]["synthetic_generative_model"]
        # ||bridge_final - I||_F == dev_mag exactly, all modules identical
        assert row["dev_mean"] == pytest.approx(plant["dev_mag"], rel=1e-4)
        assert row["dev_max"] == pytest.approx(plant["dev_mag"], rel=1e-4)
        assert row["dev_step0_mean"] == 0.0      # identity init control
        assert row["dev_step0_max"] == 0.0
        assert row["final_gap"] == pytest.approx(plant["gap"], rel=1e-5)
        assert row["update_mag_mean"] > 0.0


def test_daux_recovers_planted_correlation(daux_effect_bank):
    root, _ = daux_effect_bank
    rows = daux.collect_run_table(root)
    rep = daux.correlation_report(rows, "dev_mean", "final_gap",
                                  n_boot=100, seed=0)
    pooled = rep["cells"]["pooled"]
    assert pooled["pearson_r"] > 0.9
    assert pooled["spearman_r"] > 0.9
    lo, hi = pooled["pearson_ci"]
    assert lo > 0.8
    # within-task cells (Simpson guard) also carry the run-level plant
    for name, cell in rep["cells"].items():
        if name.startswith("task:"):
            assert cell["n"] == 8
            assert cell["pearson_r"] > 0.9, name


def test_daux_zero_effect_no_correlation(daux_zero_bank):
    root, _ = daux_zero_bank
    rows = daux.collect_run_table(root)
    rep_dev = daux.correlation_report(rows, "dev_mean", "final_gap",
                                      n_boot=50, seed=0)
    pooled = rep_dev["cells"]["pooled"]
    assert pooled["pearson_r"] is None            # deviation identically 0
    assert "zero-variance" in pooled["note"]
    rep_mag = daux.correlation_report(rows, "update_mag_mean", "final_gap",
                                      n_boot=50, seed=0)
    r = rep_mag["cells"]["pooled"]["pearson_r"]
    assert r is not None and abs(r) < 0.6         # pure noise, n = 16


def test_daux_selftest_end_to_end(tmp_path):
    rep = daux.run_selftest(tmp_path / "work", seed=5, n_boot=80)
    assert rep["passed"], rep["checks"]
    assert rep["n_runs"] == {"effect": 16, "zero": 16}


# ── CLI wiring: interlock + outputs + guards ────────────────────────


def test_daux_cli_refuses_partial_bank(daux_effect_bank, tmp_path):
    root, _ = daux_effect_bank
    with pytest.raises(SystemExit) as exc:
        daux.main(["--bank-root", str(root), "--out-dir", str(tmp_path)])
    assert "REFUSING" in str(exc.value)
    assert "480" in str(exc.value)
    assert not (tmp_path / "daux_report.json").exists()


def test_daux_cli_allow_partial_warns_and_runs(daux_effect_bank, tmp_path,
                                               capsys):
    root, _ = daux_effect_bank
    out = tmp_path / "daux-out"
    daux.main(["--bank-root", str(root), "--out-dir", str(out),
               "--allow-partial-bank", "--n-boot", "50", "--seed", "1"])
    err = capsys.readouterr().err
    assert "PRE-REGISTRATION WARNING" in err
    report = json.loads((out / "daux_report.json").read_text())
    assert report["n_runs"] == 16
    assert report["primary"]["cells"]["pooled"]["pearson_r"] > 0.9
    assert report["step0_control"]["max_dev_step0_max"] == 0.0
    table = (out / "daux_run_table.csv").read_text().strip().splitlines()
    assert len(table) == 17                       # header + 16 rows


def test_d3_cli_refuses_partial_bank(pairs_bank, tmp_path):
    root, _ = pairs_bank
    with pytest.raises(SystemExit) as exc:
        d3.main(["--bank-root", str(root), "--out-dir", str(tmp_path),
                 "--make-pairs", "2"])
    assert "REFUSING" in str(exc.value)


def test_d3_cli_make_pairs_and_merges(pairs_bank, tmp_path):
    root, _ = pairs_bank
    out = tmp_path / "d3-out"
    d3.main(["--bank-root", str(root), "--out-dir", str(out),
             "--make-pairs", "4", "--emit-merges", "--allow-partial-bank",
             "--seed", "2"])
    payload = json.loads((out / "d3_pairs.json").read_text())
    # default --max-run-uses 1: vertex-disjoint pairs -> 6 runs/family
    # support at most 3 pairs, and no run appears twice
    assert payload["max_run_uses"] == 1
    assert payload["n_pairs"] == 6                # 3 per family x 2
    fams = {p["family_short"] for p in payload["pairs"]}
    assert fams == {"synthfam0", "synthfam1"}
    for fam in fams:
        used = [r for p in payload["pairs"] if p["family_short"] == fam
                for r in (p["run_index_a"], p["run_index_b"])]
        assert len(used) == len(set(used))
    for p in payload["pairs"]:
        assert "cos_distance" in p["features"]
        merge_path = out / p["merge_file"]
        assert merge_path.exists()
    # merged states load through the production loader
    loaded = load_adapter_modules(out / payload["pairs"][0]["merge_file"])
    entry = next(iter(loaded.values()))
    assert {"lora_A", "lora_B", "bridge", "scaling"} <= set(entry)
    # determinism: same seed -> same sampled pairs; --max-run-uses 0 is
    # the unlimited legacy sampler and reaches the full quota
    out2 = tmp_path / "d3-out2"
    d3.main(["--bank-root", str(root), "--out-dir", str(out2),
             "--make-pairs", "4", "--allow-partial-bank", "--seed", "2"])
    payload2 = json.loads((out2 / "d3_pairs.json").read_text())
    keys = [(p["family_short"], p["run_index_a"], p["run_index_b"])
            for p in payload["pairs"]]
    keys2 = [(p["family_short"], p["run_index_a"], p["run_index_b"])
             for p in payload2["pairs"]]
    assert keys == keys2
    out3 = tmp_path / "d3-out3"
    d3.main(["--bank-root", str(root), "--out-dir", str(out3),
             "--make-pairs", "4", "--allow-partial-bank", "--seed", "2",
             "--max-run-uses", "0"])
    payload3 = json.loads((out3 / "d3_pairs.json").read_text())
    assert payload3["max_run_uses"] is None
    assert payload3["n_pairs"] == 8               # 4 per family x 2


def test_d3_cli_labels_dense_pairs_no_headline(pairs_bank, tmp_path):
    """Labels harness through main() on a DENSE pair design (all 15 fam0
    pairs over 6 runs = one giant run-overlap component): group-aware CV
    is infeasible, so NO headline AUC may be emitted — only the naive
    block with group_cv_feasible=false (round-1 review fix: the naive
    numbers must never silently stand in for the result)."""
    root, _ = pairs_bank
    runs = [r for r in aio.iter_runs(root, family="synthfam0")]
    lines = ["family_short,task_a,run_index_a,task_b,run_index_b,"
             "degradation,degraded"]
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            cross = int(a["task"] != b["task"])
            lines.append(f"synthfam0,{a['task']},{a['run_index']},"
                         f"{b['task']},{b['run_index']},"
                         f"{cross}.0,{cross}")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "d3-labels-out"
    d3.main(["--bank-root", str(root), "--out-dir", str(out),
             "--labels", str(labels), "--allow-partial-bank",
             "--n-boot", "25", "--seed", "1"])
    report = json.loads((out / "d3_report.json").read_text())
    assert report["n_pairs_total"] == 15
    assert "DYADIC DEPENDENCE" in report["dependence_note"]
    fam = report["per_family"]["synthfam0"]
    assert fam["n_pairs"] == 15
    assert fam["headline_basis"] is None
    assert "auc_full" not in fam                  # no headline
    dep = fam["dependence"]
    assert dep["group_cv_feasible"] is False
    assert dep["n_components"] == 1
    assert dep["largest_component_pairs"] == 15
    assert "single connected component" in dep["group_cv_infeasible_reason"]
    assert 0.0 <= fam["naive"]["auc_full"] <= 1.0
    assert fam["naive"]["bootstrap_unit"] == "pair-iid"
    assert report["pooled_oof"]["note"] == \
        "no family produced group-aware scores"
    assert report["threshold"] is None            # degraded column used


def test_d3_cli_labels_disjoint_pairs_group_aware_headline(
        daux_effect_bank, tmp_path):
    """Labels harness through main() on a VERTEX-DISJOINT pair design
    (16 runs -> 8 pairs, each its own run-overlap component): group-aware
    CV is feasible and its numbers are the headline, mirrored at the
    family block's top level, with cluster-bootstrap CIs."""
    root, _ = daux_effect_bank
    by_task: dict = {}
    for r in aio.iter_runs(root):
        by_task.setdefault(r["task"], []).append(r)
    t0, t1 = (by_task[t] for t in sorted(by_task))
    pair_specs = [(t0[0], t0[1], 0), (t0[2], t0[3], 0),
                  (t1[0], t1[1], 0), (t1[2], t1[3], 0),
                  (t0[4], t1[4], 1), (t0[5], t1[5], 1),
                  (t0[6], t1[6], 1), (t0[7], t1[7], 1)]
    lines = ["family_short,task_a,run_index_a,task_b,run_index_b,"
             "degradation,degraded"]
    for a, b, lab in pair_specs:
        lines.append(f"{a['family_short']},{a['task']},{a['run_index']},"
                     f"{b['task']},{b['run_index']},{lab}.0,{lab}")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "d3-labels-out"
    d3.main(["--bank-root", str(root), "--out-dir", str(out),
             "--labels", str(labels), "--allow-partial-bank",
             "--n-boot", "25", "--seed", "1"])
    report = json.loads((out / "d3_report.json").read_text())
    fam = report["per_family"]["synthfam0"]
    assert fam["headline_basis"] == "group_aware"
    dep = fam["dependence"]
    assert dep["group_cv_feasible"] is True
    assert dep["n_components"] == 8               # disjoint: 1 per pair
    assert dep["largest_component_pairs"] == 1
    assert 0.0 <= fam["auc_full"] <= 1.0
    # headline mirrors the group_aware block exactly
    assert fam["auc_full"] == fam["group_aware"]["auc_full"]
    assert fam["group_aware"]["bootstrap_unit"] == "run-component-cluster"
    assert fam["naive"]["bootstrap_unit"] == "pair-iid"
    # pooled summary built from group-aware scores only
    assert report["pooled_oof"]["basis"] == \
        "group-aware out-of-fold scores only"


def test_d3_cli_labels_relative_rule_headline(daux_effect_bank, tmp_path):
    """PRIMARY binarization via the fixed 5% relative-degradation rule,
    end-to-end (Director override D3). A vertex-disjoint pair design carries
    per-endpoint PERPLEXITY columns and NO 'degraded' column: cross-task
    pairs degrade an endpoint by 10% (>= 5% -> positive), within-task pairs
    do not (negative). The report records rule_used='relative_degradation',
    threshold None, and a group-aware headline."""
    root, _ = daux_effect_bank
    by_task: dict = {}
    for r in aio.iter_runs(root):
        by_task.setdefault(r["task"], []).append(r)
    t0, t1 = (by_task[t] for t in sorted(by_task))
    pair_specs = [(t0[0], t0[1], 0), (t0[2], t0[3], 0),
                  (t1[0], t1[1], 0), (t1[2], t1[3], 0),
                  (t0[4], t1[4], 1), (t0[5], t1[5], 1),
                  (t0[6], t1[6], 1), (t0[7], t1[7], 1)]
    lines = ["family_short,task_a,run_index_a,task_b,run_index_b,"
             "degradation,merged_ppl_a,native_ppl_a,merged_ppl_b,"
             "native_ppl_b"]
    for a, b, lab in pair_specs:
        mpa = 11.0 if lab == 1 else 10.0      # +10% ppl on endpoint a if lab 1
        lines.append(f"{a['family_short']},{a['task']},{a['run_index']},"
                     f"{b['task']},{b['run_index']},0.0,{mpa},10.0,10.0,10.0")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "d3-rel-out"
    d3.main(["--bank-root", str(root), "--out-dir", str(out),
             "--labels", str(labels), "--allow-partial-bank",
             "--n-boot", "25", "--seed", "1"])
    report = json.loads((out / "d3_report.json").read_text())
    bm = report["binarization"]
    assert bm["rule_used"] == "relative_degradation"
    assert bm["threshold_rel"] == 0.05
    assert bm["degenerate_min_frac"] == 0.10
    assert bm["degenerate"] is False
    assert bm["frac_positive_relative"] == pytest.approx(0.5)
    assert report["threshold"] is None
    assert report["threshold_mode"] == "relative_degradation"
    fam = report["per_family"]["synthfam0"]
    assert fam["headline_basis"] == "group_aware"


def test_d3_selftest_cli_writes_report(tmp_path):
    out = tmp_path / "selftest-out"
    d3.main(["--selftest", "--out-dir", str(out), "--seed", "0",
             "--n-boot", "60"])
    rep = json.loads((out / "d3_selftest.json").read_text())
    assert rep["passed"] is True


def test_out_dir_guard_refuses_bank_tree(tmp_path):
    bad = tmp_path / "asset1-bank" / "sub"
    with pytest.raises(ValueError, match="asset1-bank"):
        d3.guard_out_dir(bad)
    with pytest.raises(ValueError, match="asset1-bank"):
        daux.run_selftest(bad)
    ok = d3.guard_out_dir(tmp_path / "fine")
    assert ok.name == "fine"
