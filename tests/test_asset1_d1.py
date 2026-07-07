"""Tests for the D1 cross-family task identifiability analysis.

Covers: Gram-trick vs explicit linear-SVM equivalence, permutation p
calibration on null data, Wilson CI math, heterogeneity-guard math,
per-module Gram/slicing consistency (per-module Grams tile the full
Gram), H2 dimension-agnostic representations (equal length across
mismatched families), metric math (confusion/recall/macro-F1/rank),
the CLI interlock wiring (--allow-partial-bank), the out-dir guard,
and the full synthetic self-test (planted signal detected, null bank
at chance, cross-family transfer at chance).

Pre-registration hygiene: every statistic below runs on SYNTHETIC
fixtures in tmp dirs. Nothing reads or writes results/asset1-bank/,
no HF downloads, no network, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from sklearn.svm import SVC  # noqa: E402

import asset1_analysis_io as aio  # noqa: E402
import asset1_d1_identifiability as d1  # noqa: E402
import asset1_synth as synth  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def effect_bank(tmp_path_factory):
    """2 families x 3 tasks x 4 reps = 24 runs, task_effect = 1.0.
    Families have DIFFERENT d_model / n_layers (cross-family mismatch)."""
    root = tmp_path_factory.mktemp("d1-effect") / "bank"
    info = synth.make_synthetic_bank(
        root, n_families=2, n_tasks=3, n_reps=4, n_layers=2,
        d_model=16, rank=4, n_channels=2, task_effect=1.0, seed=7)
    return root, info


def _toy_data(seed=0, n_per_class=8, n_classes=3, dim=6, sep=4.0):
    """Well-separated Gaussian blobs for SVM equivalence checks."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        mu = np.zeros(dim)
        mu[c % dim] = sep
        X.append(rng.standard_normal((n_per_class, dim)) + mu)
        y.append(np.full(n_per_class, c))
    return np.vstack(X).astype(np.float64), np.concatenate(y)


# ── Gram trick vs explicit linear SVM ───────────────────────────────


def test_precomputed_gram_matches_linear_svc():
    """SVC(precomputed) on G = X X^T must agree with SVC(linear) on X —
    the mathematical identity the whole n<<d design rests on."""
    X, y = _toy_data(seed=1)
    G = X @ X.T
    lin = SVC(kernel="linear", C=1.0).fit(X, y)
    pre = SVC(kernel="precomputed", C=1.0).fit(G, y)
    Xt, _ = _toy_data(seed=2)
    assert np.array_equal(lin.predict(Xt), pre.predict(Xt @ X.T))
    # decision functions agree within numerical tolerance
    np.testing.assert_allclose(lin.decision_function(Xt),
                               pre.decision_function(Xt @ X.T),
                               rtol=1e-8, atol=1e-8)


def test_loo_predict_matches_explicit_loo():
    """d1.loo_predict on the Gram == manual LOO with SVC(linear) on X."""
    X, y = _toy_data(seed=3, n_per_class=6)
    G = X @ X.T
    preds, _ = d1.loo_predict(G, y, C=1.0)
    manual = np.empty_like(y)
    idx = np.arange(y.size)
    for i in range(y.size):
        tr = np.delete(idx, i)
        clf = SVC(kernel="linear", C=1.0).fit(X[tr], y[tr])
        manual[i] = clf.predict(X[i:i + 1])[0]
    assert np.array_equal(preds, manual)
    assert float(np.mean(preds == y)) > 0.9   # separable toy data


def test_loo_predict_scores_shape_and_ranks():
    X, y = _toy_data(seed=4, n_per_class=5)
    G = X @ X.T
    classes = np.unique(y)
    preds, scores = d1.loo_predict(G, y, C=1.0, collect_scores=True,
                                   classes=classes)
    assert scores.shape == (y.size, classes.size)
    assert np.isfinite(scores).all()
    rm = d1.rank_metrics(y, scores, classes)
    # near-perfect separation -> mean rank near 1, MRR near 1
    assert rm["mean_true_class_rank"] < 1.5
    assert rm["mrr"] > 0.75


def test_loo_predict_binary_scores():
    X, y = _toy_data(seed=5, n_per_class=6, n_classes=2)
    G = X @ X.T
    classes = np.unique(y)
    preds, scores = d1.loo_predict(G, y, C=1.0, collect_scores=True,
                                   classes=classes)
    assert scores.shape == (y.size, 2)
    assert np.isfinite(scores).all()
    # binary convention: the two columns are negatives of each other
    np.testing.assert_allclose(scores[:, 0], -scores[:, 1])


# ── Permutation null calibration ────────────────────────────────────


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_permutation_p_not_significant_on_null_data(seed):
    """On label-free Gaussian noise the permutation p must not be small —
    across several seeds (calibration, not luck)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((24, 10))
    y = np.repeat(np.arange(3), 8)
    G = X @ X.T
    preds, _ = d1.loo_predict(G, y)
    acc = float(np.mean(preds == y))
    null = d1.permutation_null(G, y, 99, seed_key=[seed, 0, 0])
    p = d1.permutation_p_value(acc, null)
    assert p > 0.05, f"seed {seed}: p={p} spuriously significant on noise"


def test_permutation_p_small_on_separable_data():
    X, y = _toy_data(seed=6, n_per_class=8)
    G = X @ X.T
    preds, _ = d1.loo_predict(G, y)
    acc = float(np.mean(preds == y))
    null = d1.permutation_null(G, y, 199, seed_key=[0, 0, 0])
    p = d1.permutation_p_value(acc, null)
    assert p == pytest.approx(1.0 / 200.0)   # no null draw reaches real acc
    assert null.max() < acc


def test_permutation_p_value_formula():
    null = np.array([0.1, 0.2, 0.3, 0.4])
    # real 0.25: two null values >= 0.25 -> (1+2)/(1+4)
    assert d1.permutation_p_value(0.25, null) == pytest.approx(3 / 5)
    # real above all nulls -> minimum p = 1/(n+1)
    assert d1.permutation_p_value(0.9, null) == pytest.approx(1 / 5)
    # ties count (>=)
    assert d1.permutation_p_value(0.4, null) == pytest.approx(2 / 5)


def test_permutation_null_deterministic():
    X, y = _toy_data(seed=7, n_per_class=4)
    G = X @ X.T
    a = d1.permutation_null(G, y, 25, seed_key=[3, 1, 0])
    b = d1.permutation_null(G, y, 25, seed_key=[3, 1, 0])
    c = d1.permutation_null(G, y, 25, seed_key=[4, 1, 0])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ── Wilson CI math ──────────────────────────────────────────────────


def test_wilson_interval_known_values():
    # acc=0.5, n=100, z=1.96: standard textbook value ~ (0.4038, 0.5962)
    lo, hi = d1.wilson_interval(0.5, 100)
    assert lo == pytest.approx(0.40383, abs=2e-4)
    assert hi == pytest.approx(0.59617, abs=2e-4)


def test_wilson_interval_properties():
    # independent re-derivation of the closed form
    acc, n, z = 0.8, 40, 1.959963984540054
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    lo, hi = d1.wilson_interval(acc, n)
    assert lo == pytest.approx(center - half)
    assert hi == pytest.approx(center + half)
    # bounds clipped to [0, 1]
    lo0, hi0 = d1.wilson_interval(0.0, 5)
    lo1, hi1 = d1.wilson_interval(1.0, 5)
    assert lo0 == 0.0 and hi1 == 1.0 and hi0 > 0.0 and lo1 < 1.0
    with pytest.raises(ValueError):
        d1.wilson_interval(0.5, 0)


# ── Heterogeneity guard math ────────────────────────────────────────


def test_heterogeneity_from_gram_matches_direct():
    """Gram-derived within-class distances == direct pairwise Euclidean."""
    rng = np.random.default_rng(0)
    # class 0 tight (scale 0.1), class 1 spread (scale 1.0)
    X0 = 0.1 * rng.standard_normal((6, 5))
    X1 = 1.0 * rng.standard_normal((6, 5)) + 10.0
    X = np.vstack([X0, X1])
    y = np.repeat([0, 1], 6)
    classes = np.array([0, 1])
    G = X @ X.T
    per_task, ratio = d1.heterogeneity_from_gram(G, y, classes)

    def direct_mean(Xc):
        ds = [np.linalg.norm(Xc[i] - Xc[j])
              for i in range(len(Xc)) for j in range(i + 1, len(Xc))]
        return float(np.mean(ds))

    assert per_task[0] == pytest.approx(direct_mean(X0), rel=1e-9)
    assert per_task[1] == pytest.approx(direct_mean(X1), rel=1e-9)
    assert ratio == pytest.approx(per_task[1] / per_task[0], rel=1e-12)
    assert ratio >= d1.HETEROGENEITY_TRIGGER   # 10x scale gap trips guard


def test_heterogeneity_ratio_near_one_when_homogeneous():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((18, 5))
    y = np.repeat(np.arange(3), 6)
    G = X @ X.T
    _, ratio = d1.heterogeneity_from_gram(G, y, np.arange(3))
    assert 1.0 <= ratio < d1.HETEROGENEITY_TRIGGER


# ── Metric math ─────────────────────────────────────────────────────


def test_confusion_recall_f1_balanced():
    y = np.array([0, 0, 0, 1, 1, 2])
    p = np.array([0, 0, 1, 1, 1, 0])
    classes = np.arange(3)
    cm, recalls = d1.confusion_and_recalls(y, p, classes)
    assert cm.tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]
    np.testing.assert_allclose(recalls, [2 / 3, 1.0, 0.0])
    assert d1.balanced_accuracy(recalls) == pytest.approx((2 / 3 + 1) / 3)
    # macro-F1 cross-check against sklearn
    from sklearn.metrics import f1_score
    assert d1.macro_f1(cm) == pytest.approx(
        f1_score(y, p, average="macro", labels=classes,
                 zero_division=0))
    mc = d1.most_confused_class(cm, recalls, ["a", "b", "c"])
    assert mc["task"] == "c"
    assert mc["most_confused_with"] == "a"
    assert mc["n_confusions_to_target"] == 1


# ── Per-module Grams / slicing consistency ──────────────────────────


def test_module_grams_tile_full_gram(effect_bank, tmp_path):
    """Sum of per-module Grams == full Gram == explicit X X^T, and each
    per-module Gram == the explicit Gram of that module's feature slice."""
    root, _ = effect_bank
    records = [r for r in aio.iter_runs(root, family="synthfam0")]
    scratch = tmp_path / "feat.dat"
    n, d, groups = d1.build_feature_memmap(records, scratch, "raw", 4, 0)
    assert groups and max(g.stop for g in groups.values()) == d

    grams = d1.grams_from_memmap(scratch, n, d, groups, chunk_rows=3)
    G_sum = sum(grams.values())

    X = np.stack([aio.flatten_features(aio.load_adapter(r["run_dir"]))
                  for r in records]).astype(np.float64)
    assert X.shape == (n, d)
    np.testing.assert_allclose(G_sum, X @ X.T, rtol=1e-10, atol=1e-10)
    for name, sl in groups.items():
        np.testing.assert_allclose(grams[name], X[:, sl] @ X[:, sl].T,
                                   rtol=1e-10, atol=1e-10)


def test_chunk_size_does_not_change_gram(effect_bank, tmp_path):
    root, _ = effect_bank
    records = [r for r in aio.iter_runs(root, family="synthfam0")][:6]
    scratch = tmp_path / "feat.dat"
    n, d, _ = d1.build_feature_memmap(records, scratch, "raw", 4, 0)
    full = {"__full__": slice(0, d)}
    g1 = d1.grams_from_memmap(scratch, n, d, full, chunk_rows=1)["__full__"]
    g2 = d1.grams_from_memmap(scratch, n, d, full, chunk_rows=4)["__full__"]
    g3 = d1.grams_from_memmap(scratch, n, d, full,
                              chunk_rows=100)["__full__"]
    np.testing.assert_allclose(g1, g2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g1, g3, rtol=1e-12, atol=1e-12)


def test_canonical_representation_memmap(effect_bank, tmp_path):
    """Canonical mode featurizes via canonicalize_adapter+feature_vector;
    no module groups (per-module breakdown is raw-only)."""
    root, _ = effect_bank
    records = [r for r in aio.iter_runs(root, family="synthfam0")][:4]
    scratch = tmp_path / "canon.dat"
    n, d, groups = d1.build_feature_memmap(records, scratch, "canonical",
                                           proj_dim=4, proj_seed=0)
    assert groups is None
    assert n == 4 and d > 0
    X = np.memmap(scratch, dtype=np.float32, mode="r", shape=(n, d))
    assert np.isfinite(np.asarray(X)).all()
    del X


# ── H2 dimension-agnostic representations ───────────────────────────


def test_h2_features_equal_length_across_mismatched_families(effect_bank):
    """Families differ in d_model AND layer count; both H2 representations
    must produce identical-length vectors for both."""
    root, _ = effect_bank
    rec0 = next(aio.iter_runs(root, family="synthfam0"))
    rec1 = next(aio.iter_runs(root, family="synthfam1"))
    # raw features are NOT comparable (sanity of the premise)
    v0 = aio.flatten_features(aio.load_adapter(rec0["run_dir"]))
    v1 = aio.flatten_features(aio.load_adapter(rec1["run_dir"]))
    assert v0.size != v1.size

    kw = dict(sigma_slots=4, n_depth_bins=3, proj_dim=5, proj_seed=0)
    s0, p0 = d1.h2_features_for_run(rec0["run_dir"], **kw)
    s1, p1 = d1.h2_features_for_run(rec1["run_dir"], **kw)
    assert s0.shape == s1.shape == (4 * 3 * 4,)          # proj*bins*slots
    assert p0.shape == p1.shape == (4 * 3 * 4 * (1 + 2 * 5),)
    for v in (s0, s1, p0, p1):
        assert v.dtype == np.float32
        assert np.isfinite(v).all()
    # deterministic
    s0b, p0b = d1.h2_features_for_run(rec0["run_dir"], **kw)
    assert np.array_equal(s0, s0b) and np.array_equal(p0, p0b)


def test_parse_module_name():
    assert d1._parse_module_name(
        "model_layers_7_self_attn_k_proj") == (7, "k")
    assert d1._parse_module_name(
        "model.layers.12.self_attn.q_proj") == (12, "q")
    with pytest.raises(ValueError):
        d1._parse_module_name("model_embed_tokens")


def test_h2_transfer_output_shape():
    """Both variants (raw + shift-controlled) and the family probe are
    present per representation — the pinned H2 triviality controls. The
    old single-variant schema (accuracy at the cell top level, no probe)
    is exactly what made the H2 finding unfalsifiable; this asserts the
    controls exist."""
    rng = np.random.default_rng(0)
    fams = ["famA", "famB"]
    features = {f: {"spectrum": rng.standard_normal((12, 8)),
                    "probe": rng.standard_normal((12, 20))}
                for f in fams}
    labels = {f: np.repeat(np.arange(3), 4) for f in fams}
    out = d1.h2_transfer(features, labels, fams, chance=1 / 3, seed=0)
    for rep in ("spectrum", "probe"):
        assert set(out[rep]["pairs"]) == {"famA->famB", "famB->famA"}
        assert set(out[rep]["family_probe"]) == {"raw",
                                                 "family_standardized"}
        for probe in out[rep]["family_probe"].values():
            assert probe["n"] == 24
            assert probe["chance_majority"] == pytest.approx(0.5)
            assert set(probe["per_family_mean_feature_norm"]) == set(fams)
        for cell in out[rep]["pairs"].values():
            assert set(cell) == {"raw", "family_standardized"}
            for c in cell.values():
                assert 0.0 <= c["accuracy"] <= 1.0
                assert 0.0 <= c["binom_p_greater_than_chance"] <= 1.0
                assert c["n_test"] == 12


def test_h2_transfer_includes_within_family_and_decision():
    """h2_transfer now carries within-family accuracy per variant and a
    per-representation H2 decision (the pinned rule) with role labels —
    spectrum PRIMARY, probe corroborating; the decision runs on the
    shift-controlled (family_standardized) variant."""
    rng = np.random.default_rng(0)
    fams = ["famA", "famB"]
    features = {f: {"spectrum": rng.standard_normal((12, 8)),
                    "probe": rng.standard_normal((12, 20))}
                for f in fams}
    labels = {f: np.repeat(np.arange(3), 4) for f in fams}
    out = d1.h2_transfer(features, labels, fams, chance=1 / 3, seed=0)
    assert out["spectrum"]["role"] == "PRIMARY"
    assert out["probe"]["role"] == "corroborating"
    for rep in ("spectrum", "probe"):
        wfa = out[rep]["within_family_accuracy"]
        assert set(wfa) == {"raw", "family_standardized"}
        for variant in wfa.values():
            assert set(variant) == set(fams)
            for f in fams:
                assert 0.0 <= variant[f]["accuracy"] <= 1.0
        dec = out[rep]["decision"]
        assert dec["variant"] == "family_standardized"
        assert dec["alpha"] == d1.H2_ALPHA
        assert dec["margin_pp_threshold"] == d1.H2_MARGIN_PP
        assert set(dec["directions"]) == {"famA->famB", "famB->famA"}
        assert isinstance(dec["supported"], bool)


# ── H2 decision rule (h2_supported — DIRECTOR_DECISIONS_2026-07-06.md) ──


def test_h2_supported_both_conditions_pass():
    """Both directions: cross NOT significant (p >= alpha) AND within − cross
    margin >= 15pp -> H2 supported. alpha/margin constants recorded."""
    directions = {
        "A->B": {"cross_accuracy": 0.34, "cross_binom_p": 0.5,
                 "within_accuracy": 0.90},
        "B->A": {"cross_accuracy": 0.33, "cross_binom_p": 0.7,
                 "within_accuracy": 0.88},
    }
    out = d1.h2_supported(directions)
    assert out["supported"] is True
    assert out["alpha"] == d1.H2_ALPHA == 0.01
    assert out["margin_pp_threshold"] == d1.H2_MARGIN_PP == 15.0
    for d in out["directions"].values():
        assert d["not_above_chance_at_alpha"] is True
        assert d["margin_ge_threshold"] is True
        assert d["direction_supported"] is True


def test_h2_supported_fails_on_significant_cross():
    """Condition (i) fails in one direction: cross-family accuracy
    significantly above chance (p < alpha) -> overall not supported."""
    directions = {
        "A->B": {"cross_accuracy": 0.34, "cross_binom_p": 0.005,   # < 0.01
                 "within_accuracy": 0.90},
        "B->A": {"cross_accuracy": 0.33, "cross_binom_p": 0.7,
                 "within_accuracy": 0.88},
    }
    out = d1.h2_supported(directions)
    assert out["supported"] is False
    assert out["directions"]["A->B"]["not_above_chance_at_alpha"] is False
    assert out["directions"]["A->B"]["direction_supported"] is False
    assert out["directions"]["B->A"]["direction_supported"] is True


def test_h2_supported_fails_on_small_margin():
    """Condition (ii) fails: within − cross < 15pp -> not supported."""
    directions = {
        "A->B": {"cross_accuracy": 0.80, "cross_binom_p": 0.5,
                 "within_accuracy": 0.90},   # margin 10pp < 15
        "B->A": {"cross_accuracy": 0.30, "cross_binom_p": 0.5,
                 "within_accuracy": 0.90},
    }
    out = d1.h2_supported(directions)
    assert out["supported"] is False
    assert out["directions"]["A->B"]["margin_ge_threshold"] is False
    assert out["directions"]["A->B"]["margin_pp"] == pytest.approx(10.0)


def test_h2_supported_requires_both_directions():
    """One direction supported, the other not -> overall not supported;
    an empty direction set is not supported."""
    directions = {
        "A->B": {"cross_accuracy": 0.30, "cross_binom_p": 0.5,
                 "within_accuracy": 0.90},   # supported
        "B->A": {"cross_accuracy": 0.85, "cross_binom_p": 0.5,
                 "within_accuracy": 0.90},   # margin 5pp -> fails
    }
    out = d1.h2_supported(directions)
    assert out["directions"]["A->B"]["direction_supported"] is True
    assert out["directions"]["B->A"]["direction_supported"] is False
    assert out["supported"] is False
    assert d1.h2_supported({})["supported"] is False


def test_h2_supported_alpha_and_margin_boundaries_honored():
    """Boundary: p == alpha counts as NOT significant (>= alpha) and a margin
    of exactly 15pp passes; custom alpha/margin constants are honored."""
    directions = {
        "A->B": {"cross_accuracy": 0.30, "cross_binom_p": d1.H2_ALPHA,
                 "within_accuracy": 0.45},   # margin exactly 15pp
        "B->A": {"cross_accuracy": 0.30, "cross_binom_p": d1.H2_ALPHA,
                 "within_accuracy": 0.45},
    }
    assert d1.h2_supported(directions)["supported"] is True
    # tighten alpha so p is now "significant", and raise the margin bar
    out2 = d1.h2_supported(directions, alpha=0.5, margin_pp=20.0)
    assert out2["supported"] is False
    assert out2["alpha"] == 0.5
    assert out2["margin_pp_threshold"] == 20.0


def test_familywise_standardize_math():
    """Exact per-family per-feature z-scoring, zero-variance columns pass
    through centered, inputs unmodified."""
    rng = np.random.default_rng(1)
    Xa = np.column_stack([rng.standard_normal(10) * 3.0 + 5.0,
                          np.full(10, 7.0)])          # col 1: zero variance
    Xb = rng.standard_normal((8, 2))
    features = {"famA": {"spectrum": Xa.copy()},
                "famB": {"spectrum": Xb.copy()}}
    out = d1.familywise_standardize(features)
    Za, Zb = out["famA"]["spectrum"], out["famB"]["spectrum"]
    for Z in (Za, Zb):
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Za[:, 0].std(), 1.0, atol=1e-12)
    np.testing.assert_allclose(Za[:, 1], 0.0, atol=1e-12)   # centered only
    np.testing.assert_allclose(Zb.std(axis=0), 1.0, atol=1e-12)
    # manual cross-check on the non-degenerate column
    np.testing.assert_allclose(
        Za[:, 0], (Xa[:, 0] - Xa[:, 0].mean()) / Xa[:, 0].std(),
        atol=1e-12)
    # inputs untouched
    assert features["famA"]["spectrum"][0, 1] == 7.0


def test_family_probe_detects_dim_scale_signature(effect_bank):
    """THE ROUND-1 REGRESSION TEST: the synthetic families share the
    generative process and differ only in dims, yet the raw H2
    representations classify FAMILY nearly perfectly — the covariate
    shift that makes an uncontrolled H2 'transfer at chance' finding
    trivially achievable. The probe must expose it, and per-family mean
    feature norms must differ."""
    root, _ = effect_bank
    kw = dict(sigma_slots=4, n_depth_bins=3, proj_dim=5, proj_seed=0)
    feats = {}
    for fam in ("synthfam0", "synthfam1"):
        spec_rows, probe_rows = [], []
        for rec in aio.iter_runs(root, family=fam):
            s, p = d1.h2_features_for_run(rec["run_dir"], **kw)
            spec_rows.append(s)
            probe_rows.append(p)
        feats[fam] = {"spectrum": np.stack(spec_rows),
                      "probe": np.stack(probe_rows)}
    probe = d1.family_identity_probe(
        {f: feats[f]["spectrum"] for f in feats}, seed=0)
    assert probe["accuracy"] is not None
    assert probe["accuracy"] > 0.9, \
        "family probe failed to detect the dims-only scale signature"
    norms = probe["per_family_mean_feature_norm"]
    assert norms["synthfam0"] != pytest.approx(norms["synthfam1"],
                                               rel=0.02)


def test_shift_control_rescues_genuinely_transferable_structure():
    """THE FALSIFIABILITY TEST: construct families with IDENTICAL task
    structure but a gross per-feature affine family shift (per-feature
    scales spanning ~4 orders of magnitude plus offsets — the realistic
    form of the spectra/probe scale signature). Raw transfer collapses
    while per-family z-scoring — which exactly undoes per-feature affine
    maps — recovers near-perfect transfer. Under the OLD uncontrolled H2,
    this transferable structure would have been reported as 'transfer at
    chance' — the exact unfalsifiability the finding flagged. The probe
    fires on raw and collapses after standardization (the shift is pure
    location/scale here)."""
    rng = np.random.default_rng(42)
    n_tasks, n_per_task, dim = 3, 12, 8
    means = 3.0 * rng.standard_normal((n_tasks, dim))
    y = np.repeat(np.arange(n_tasks), n_per_task)

    def draw():
        return np.vstack([means[c] + 0.3 * rng.standard_normal(dim)
                          for c in y])

    scale = 10.0 ** rng.uniform(-2.0, 2.0, size=dim)   # anisotropic
    offset = 10.0 * rng.standard_normal(dim)
    Xa = draw()
    Xb = draw() * scale + offset             # same structure, family shift
    features = {"famA": {"spectrum": Xa, "probe": Xa},
                "famB": {"spectrum": Xb, "probe": Xb}}
    labels = {"famA": y, "famB": y}
    out = d1.h2_transfer(features, labels, ["famA", "famB"],
                         chance=1.0 / n_tasks, seed=0)
    cell = out["spectrum"]["pairs"]["famA->famB"]
    assert cell["family_standardized"]["accuracy"] > 0.9
    assert (cell["family_standardized"]["accuracy"]
            > cell["raw"]["accuracy"] + 0.2)
    probes = out["spectrum"]["family_probe"]
    assert probes["raw"]["accuracy"] > 0.9
    assert probes["family_standardized"]["accuracy"] < 0.8


# ── Interlock wiring + out-dir guard (CLI level) ────────────────────


def test_cli_refuses_partial_bank(effect_bank, tmp_path):
    """A real-bank-capable invocation must hit require_complete_bank:
    a 24-run bank fails the locked 480 expectation without the flag."""
    root, _ = effect_bank
    with pytest.raises(SystemExit) as exc:
        d1.main(["--bank-root", str(root),
                 "--out-dir", str(tmp_path / "out")])
    assert "REFUSING" in str(exc.value)
    assert "480" in str(exc.value)


def test_cli_allow_partial_bank_warns_and_runs(effect_bank, tmp_path,
                                               capsys):
    root, _ = effect_bank
    out = tmp_path / "out"
    d1.main(["--bank-root", str(root), "--out-dir", str(out),
             "--allow-partial-bank", "--n-permutations", "20",
             "--proj-dim", "4"])
    err = capsys.readouterr().err
    assert "PRE-REGISTRATION WARNING" in err
    assert (out / "d1_results.json").exists()
    assert (out / "D1_REPORT.md").exists()
    import json
    res = json.loads((out / "d1_results.json").read_text(encoding="utf-8"))
    assert res["exploratory_only"] is True
    # A2 ADOPTED (DIRECTOR_DECISIONS_2026-07-06.md): --representation now
    # defaults to 'both' (was 'raw'), so each family reports raw AND
    # W2T-canonical H1 results, and the pinned decisions self-document.
    assert res["parameters"]["representation"] == "both"
    assert set(next(iter(res["families"].values()))["representations"]) == \
        {"raw", "canonical"}
    assert res["pinned_decisions"]["h2_alpha"] == d1.H2_ALPHA
    assert res["pinned_decisions"]["h2_margin_pp"] == d1.H2_MARGIN_PP
    assert res["pinned_decisions"]["h2_primary_representation"] == "spectrum"
    report = (out / "D1_REPORT.md").read_text(encoding="utf-8")
    assert "EXPLORATORY ONLY" in report


def test_out_dir_guard_refuses_bank_tree(tmp_path):
    with pytest.raises(SystemExit, match="asset1-bank"):
        d1.analyze_bank(tmp_path / "whatever",
                        tmp_path / "asset1-bank" / "out")


def test_selftest_rejects_too_few_permutations(tmp_path):
    with pytest.raises(ValueError, match="100"):
        d1.run_synthetic_selftest(tmp_path, n_permutations=50)


# ── Full synthetic self-test (the acceptance test) ──────────────────


def test_synthetic_selftest_end_to_end(tmp_path):
    """Planted signal detected (acc >> chance, p < 0.01), null bank at
    chance (lock fails, acc within null band), cross-family transfer at
    chance. All assertions live inside run_synthetic_selftest."""
    res = d1.run_synthetic_selftest(tmp_path / "st", n_permutations=120,
                                    seed=0, chunk_rows=4)
    eff = res["effect"]
    # Headline numbers land where the plant puts them
    for fam_out in eff["families"].values():
        raw = fam_out["representations"]["raw"]
        assert raw["loo_accuracy"] > 0.9          # strongly separable plant
        assert raw["permutation"]["p_value"] == pytest.approx(1 / 121)
        assert raw["macro_f1"] > 0.9
        lo, hi = raw["wilson_ci_95"]
        assert 0.0 <= lo <= raw["loo_accuracy"] <= hi <= 1.0
    # Output artifacts exist and are self-describing
    out_eff = tmp_path / "st" / "selftest" / "out_effect"
    assert (out_eff / "d1_results_selftest_effect.json").exists()
    assert (out_eff / "D1_REPORT_selftest_effect.md").exists()
    md = (out_eff / "D1_REPORT_selftest_effect.md").read_text(
        encoding="utf-8")
    assert "H1 lock: PASS" in md
    assert "H2" in md


def test_h2_supported_passes_through_source_within_accuracy():
    """Director ruling 2026-07-07 (pinning-1 addition): the source family's
    within-accuracy is carried through h2_supported descriptively — present
    when provided, absent when not, and never part of the decision."""
    directions = {
        "A->B": {"cross_accuracy": 0.17, "cross_binom_p": 0.5,
                 "within_accuracy": 0.80, "within_accuracy_source": 0.75},
        "B->A": {"cross_accuracy": 0.17, "cross_binom_p": 0.5,
                 "within_accuracy": 0.80},
    }
    out = d1.h2_supported(directions)
    assert out["directions"]["A->B"]["within_accuracy_source"] == 0.75
    assert "within_accuracy_source" not in out["directions"]["B->A"]
    # decision identical with/without the descriptive field
    stripped = {k: {kk: vv for kk, vv in v.items()
                    if kk != "within_accuracy_source"}
                for k, v in directions.items()}
    assert d1.h2_supported(stripped)["supported"] == out["supported"]
