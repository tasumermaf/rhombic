"""BM-000b hub-motif null family tests (scripts/bm_hub_motif_nulls.py).

Verifies:
  1. Motif generators — degree properties (star hub degree n-1 / leaf 1,
     perfect matchings 1-regular, expander d-regular + connected), mask
     value conventions, and reference strong sets matching the canonical
     pair partitions in train_exp2_scale.
  2. Determinism — same seed => bit-identical tables; different seed =>
     different tables.
  3. Metric-reuse equivalence — the pipeline's metric path produces the
     exact values of the BM-000-cited implementations on known matrices.
  4. Small-n smoke of the full pipeline including file outputs with the
     invented-defaults record (self-documenting runs).

CPU-only: bm_hub_motif_nulls masks CUDA before importing torch-touching
modules; no test initializes CUDA and no bank data is read (moments are
fixed synthetic values, mirroring tests/test_bm000.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bm_hub_motif_nulls as hub  # noqa: E402
import bm000_null_model as bm     # noqa: E402
from train_exp2_scale import (    # noqa: E402
    _coplanar_crossplanar_indices,
    coplanar_crossplanar_ratio,
)
from detect_blocks import detect_blocks  # noqa: E402

# Fixed synthetic moments — tests must not depend on smoke-bank files.
MOMENTS = {
    "diag_mean": 1.0, "diag_std": 0.001,
    "offdiag_mean": 0.0, "offdiag_std": 0.002,
    "n_matrices": 0,
}
N_SMALL = 60


def _strong_degrees(n: int, pairs: list[tuple[int, int]]) -> list[int]:
    deg = [0] * n
    for i, j in pairs:
        deg[i] += 1
        deg[j] += 1
    return deg


# ── 1. Motif generators ──────────────────────────────────────────────


@pytest.mark.parametrize("n", [4, 6, 8])
def test_reference_strong_set_matches_canonical_partition(n):
    """ref motif uses exactly the canonical co-planar/co-axial pairs."""
    co, _ = _coplanar_crossplanar_indices(n)
    assert hub.strong_pairs_polytope(n) == [tuple(p) for p in co]
    # Polytope strong structure is a perfect matching at every family n
    assert sorted(_strong_degrees(n, hub.strong_pairs_polytope(n))) == [1] * n


@pytest.mark.parametrize("n", [4, 6, 8])
def test_star_degrees(n):
    pairs = hub.strong_pairs_star(n)
    deg = _strong_degrees(n, pairs)
    assert deg[hub.STAR_CENTER] == n - 1
    assert sorted(deg[:hub.STAR_CENTER] + deg[hub.STAR_CENTER + 1:]) == [1] * (n - 1)
    assert len(pairs) == n - 1


@pytest.mark.parametrize("n", [4, 6, 8])
def test_random_perfect_matching_properties(n):
    rng = np.random.default_rng(0)
    for _ in range(20):
        pairs = hub.random_perfect_matching(rng, n)
        assert len(pairs) == n // 2
        assert _strong_degrees(n, pairs) == [1] * n  # 1-regular
        assert all(i < j for i, j in pairs)          # normalized


@pytest.mark.parametrize("n", [4, 6, 8])
def test_rewire_matching_stays_matching(n):
    rng = np.random.default_rng(1)
    ref = hub.strong_pairs_polytope(n)
    for _ in range(20):
        pairs = hub.rewire_matching(rng, ref, 10 * len(ref))
        assert len(pairs) == len(ref)
        assert _strong_degrees(n, pairs) == [1] * n


def test_rewire_actually_rewires():
    """Across draws the rewire must produce non-reference matchings."""
    rng = np.random.default_rng(2)
    ref = hub.strong_pairs_polytope(6)
    outcomes = {tuple(hub.rewire_matching(rng, ref, 30)) for _ in range(50)}
    assert len(outcomes) > 1
    assert any(o != tuple(ref) for o in outcomes)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_expander_regular_and_connected(n):
    import networkx as nx
    rng = np.random.default_rng(3)
    d = hub.EXPANDER_DEGREE[n]
    for _ in range(5):
        pairs = hub.expander_strong_pairs(rng, n)
        assert _strong_degrees(n, pairs) == [d] * n
        g = nx.Graph(pairs)
        g.add_nodes_from(range(n))
        assert nx.is_connected(g)


def test_expander_n4_is_k4():
    """Degeneracy documented in D3: the only 3-regular graph on 4 vertices."""
    rng = np.random.default_rng(4)
    pairs = hub.expander_strong_pairs(rng, 4)
    assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


@pytest.mark.parametrize("n", [4, 6, 8])
def test_mask_convention(n):
    """D2: symmetric, diag 1.0, values exactly {0.5, 1.0}."""
    strong = hub.strong_pairs_polytope(n)
    mask = hub.mask_from_strong(n, strong)
    assert mask.shape == (n, n)
    assert np.array_equal(mask, mask.T)
    assert np.all(np.diag(mask) == 1.0)
    off = mask[~np.eye(n, dtype=bool)]
    assert set(np.unique(off)) <= {0.5, 1.0}
    assert int((off == 1.0).sum()) == 2 * len(strong)


def test_ref_rd6_mask_equals_bm000_convention():
    """ref_rd6 strong set = the RD co-planar pairs BM-000's rdmask6 samples
    around: (0,1),(2,3),(4,5) with 3 strong pairs of 15."""
    assert hub.strong_pairs_polytope(6) == [(0, 1), (2, 3), (4, 5)]


def test_strong_edge_counts():
    assert hub.strong_edge_count("ref", 6) == 3
    assert hub.strong_edge_count("star", 6) == 5
    assert hub.strong_edge_count("matched1reg", 8) == 4
    assert hub.strong_edge_count("shuffledRD", 4) == 2
    assert hub.strong_edge_count("expander", 6) == 9
    assert hub.strong_edge_count("expander", 8) == 12


# ── 2. Determinism ───────────────────────────────────────────────────


def test_pipeline_deterministic():
    t1, _ = hub.run_hub_nulls(hub.SEED, N_SMALL, MOMENTS)
    t2, _ = hub.run_hub_nulls(hub.SEED, N_SMALL, MOMENTS)
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)


def test_pipeline_seed_sensitivity():
    t1, _ = hub.run_hub_nulls(hub.SEED, N_SMALL, MOMENTS)
    t3, _ = hub.run_hub_nulls(hub.SEED + 1, N_SMALL, MOMENTS)
    assert json.dumps(t1, sort_keys=True) != json.dumps(t3, sort_keys=True)


def test_ensemble_structure():
    """All 15 ensembles present with the BM-000 metric set summarized."""
    tables, _ = hub.run_hub_nulls(hub.SEED, 30, MOMENTS)
    expected = {hub.ensemble_name(m, n) for n in hub.NS for m in hub.MOTIFS}
    assert set(tables.keys()) == expected
    assert expected == {
        "ref_octa4", "star4", "matched1reg4", "shuffledRD4", "expander4",
        "ref_rd6", "star6", "matched1reg6", "shuffledRD6", "expander6",
        "ref_tess8", "star8", "matched1reg8", "shuffledRD8", "expander8",
    }
    for ens in expected:
        for metric in ("co_cross", "fiedler", "fiedler_atb", "bd_flag",
                       "asymmetry"):
            assert tables[ens][metric]["n"] > 0, (ens, metric)
    # n-gated metrics: populated at n=6, empty elsewhere
    assert tables["ref_rd6"]["co_cross_fullmatrix"]["n"] > 0
    assert tables["ref_octa4"]["co_cross_fullmatrix"]["n"] == 0
    assert tables["star8"]["bd_top3"]["n"] == 0


# ── 3. Metric-reuse equivalence on known matrices ────────────────────


def test_metric_reuse_equivalence_known_matrix():
    """The pipeline scores bridges with bm000_null_model.compute_metrics;
    verify against direct calls to the cited implementations."""
    rng = np.random.default_rng(7)
    B = np.eye(6) + 0.01 * rng.normal(size=(6, 6))
    metrics = bm.compute_metrics([B])
    # M1 — train_exp2_scale.coplanar_crossplanar_ratio (imported by BM-000)
    assert metrics["co_cross"][0] == pytest.approx(
        coplanar_crossplanar_ratio(B)["ratio"])
    # M2 — rhombic.spectral.fiedler_value on upper-tri |B|
    assert metrics["fiedler"][0] == pytest.approx(bm.metric_fiedler(B))
    # M3a — detect_blocks.detect_blocks
    db = detect_blocks(B)
    assert bool(metrics["bd_flag"][0]) == db["is_block_diagonal"]
    # M4 — asymmetry, hand-computed
    A = (B - B.T) / 2
    expected = np.linalg.norm(A, "fro") / np.linalg.norm(B, "fro")
    assert metrics["asymmetry"][0] == pytest.approx(expected)


def test_metric_reuse_planted_bd_matrix():
    """A hand-built block-diagonal 6x6 must be detected and produce a large
    co/cross ratio through the reused metric path."""
    B = np.eye(6)
    for i, j in [(0, 1), (2, 3), (4, 5)]:   # RD co-planar = the 3 blocks
        B[i, j] = B[j, i] = 0.5
    for i in range(6):
        for j in range(6):
            if i != j and B[i, j] == 0.0:
                B[i, j] = 1e-6
    metrics = bm.compute_metrics([B])
    assert metrics["co_cross"][0] > 1e4
    assert metrics["bd_flag"][0] == 1.0
    assert metrics["bd_top3"][0] == 1.0


def test_distinguishability_rule_on_planted_ensembles():
    """D6 rule fires when reference and motif ensembles are separated."""
    raw = {}
    rng = np.random.default_rng(8)
    for n in hub.NS:
        for motif in hub.MOTIFS:
            name = hub.ensemble_name(motif, n)
            # co_cross planted: ref high, motifs low, others neutral
            base = 100.0 if motif == "ref" else 1.0
            raw[name] = {
                "co_cross": base + 0.01 * rng.normal(size=50),
                "fiedler": np.ones(50),
                "fiedler_atb": np.ones(50),
                "asymmetry": np.ones(50),
            }
    rows = hub.distinguishability(raw)
    cc = [r for r in rows if r["metric"] == "co_cross"]
    assert cc and all(r["distinguishable"] for r in cc)
    fv = [r for r in rows if r["metric"] == "fiedler"]
    assert fv and not any(r["distinguishable"] for r in fv)


# ── 4. Small-n smoke of the full pipeline ────────────────────────────


def test_full_pipeline_smoke(tmp_path):
    tables, raw = hub.run_hub_nulls(hub.SEED, 25, MOMENTS)
    payload = hub.write_outputs(tmp_path, MOMENTS, "synthetic (test)",
                                tables, raw, 25)
    nulls = tmp_path / "nulls.json"
    results = tmp_path / "RESULTS.md"
    assert nulls.exists() and results.exists()
    on_disk = json.loads(nulls.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(payload))  # round-trips
    # Self-documenting run: invented defaults recorded
    d = on_disk["invented_defaults"]
    for key in ("seed", "n_samples_per_ensemble", "mask_convention",
                "star_center_channel", "expander_degree",
                "expander_candidates_per_draw", "shuffled_rd_rewire",
                "matched_regular", "distinguishability_rule",
                "moments_source_used"):
        assert key in d, key
    assert d["n_samples_per_ensemble"] == 25
    assert d["moments_source_used"] == "synthetic (test)"
    assert on_disk["seed"] == hub.SEED
    assert len(on_disk["ensembles"]) == 15
    assert len(on_disk["strong_edge_counts"]) == 15
    # Distinguishability rows: 3 n-values x 4 metrics x 4 motifs = 48
    assert len(on_disk["distinguishability"]) == 48
    text = results.read_text(encoding="utf-8")
    assert "Q1" in text and "Q2" in text and "Verdict (honest)" in text
    assert "MASK-LEVEL" in text  # scope limitation stated


def test_untrained_motifs_do_not_reach_trained_signature():
    """Q1 at test scale: no motif null band at n=6 approaches the trained
    headline regime (co/cross ~7e4, Fiedler ~9e-5 with off-diag ~2e-3)."""
    _, raw = hub.run_hub_nulls(hub.SEED, 200, MOMENTS)
    for motif in hub.MOTIFS:
        name = hub.ensemble_name(motif, 6)
        cc = raw[name]["co_cross"]
        cc = cc[np.isfinite(cc)]
        assert cc.max() < 1e3, name  # trained H-ch6 value is 7.04e4
