"""BM-000c Config-D persistence null tests
(scripts/bm_configd_persistence_null.py).

Verifies:
  1. Persistence metric — correctness on hand-constructed SYNTHETIC templates
     (no corpus needed): perfect match -> 1.0, anti-template -> -1.0, positive
     affine invariance, sign flip -> negation, consistent-relabel invariance,
     degenerate -> NaN, and equivalence to a direct off-diagonal corrcoef.
  2. Determinism — same seed => byte-identical tables and byte-identical
     nulls.json; different seed => different tables.
  3. Bank-path guard — assert_safe_out_dir refuses any 'asset1-bank' path.
  4. IP boundary — the output payload carries NO raw template array (only a
     SHA-256); the structural guard fires on a planted matrix; and (gated on
     the real corpus) no raw template value appears among payload numbers.
  5. Corpus-availability — metric-property + determinism tests run WITHOUT the
     corpus (synthetic stand-in template); real-template + real-run tests are
     skipped when corpus_private.json is absent.

CPU-only: the module masks CUDA before importing torch-touching code; no test
initializes CUDA and no bank data is read (moments are fixed synthetic values,
mirroring tests/test_bm_hub_motifs.py).
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

import bm_configd_persistence_null as cd  # noqa: E402
import bm000_null_model as bm             # noqa: E402
from rhombic.corpus import corpus_available  # noqa: E402

# Fixed synthetic moments — tests must not depend on smoke-bank files.
MOMENTS = {
    "diag_mean": 1.0, "diag_std": 0.001,
    "offdiag_mean": 0.0, "offdiag_std": 0.002,
    "n_matrices": 0,
}
N_SMALL = 80
FAKE_SHA = "0" * 64


def _syn_template() -> np.ndarray:
    """A symmetric 6x6 stand-in template: identity diagonal, 15 DISTINCT
    off-diagonal magnitudes. Mirrors the corpus_coupled structure (symmetric,
    identity diagonal) WITHOUT using any proprietary value."""
    M = np.eye(6, dtype=np.float64)
    vals = np.linspace(-0.01, 0.01, 15)
    k = 0
    for i in range(6):
        for j in range(i + 1, 6):
            M[i, j] = M[j, i] = vals[k]
            k += 1
    return M


SYN = _syn_template()
SYN_OFF = cd.offdiag_vector(SYN)


# ── 1. Persistence metric correctness (synthetic; no corpus) ─────────


def test_perfect_template_match_is_one():
    assert cd.persistence_metric(SYN, SYN_OFF) == pytest.approx(1.0)


def test_anti_template_is_minus_one():
    assert cd.persistence_metric(-SYN, SYN_OFF) == pytest.approx(-1.0)


def test_positive_affine_invariance():
    """r is invariant to positive scale + shift of the bridge (D2/D8)."""
    rng = np.random.default_rng(11)
    B = rng.normal(size=(6, 6))                    # asymmetric, generic
    base = cd.persistence_metric(B, SYN_OFF)
    for a, b in [(3.0, 0.5), (0.01, -2.0), (1000.0, 7.0)]:
        assert cd.persistence_metric(a * B + b, SYN_OFF) == pytest.approx(base)


def test_sign_flip_negates():
    rng = np.random.default_rng(12)
    B = rng.normal(size=(6, 6))
    r = cd.persistence_metric(B, SYN_OFF)
    assert cd.persistence_metric(-B, SYN_OFF) == pytest.approx(-r)


def test_consistent_relabel_invariance():
    """Relabeling channels identically on the bridge AND the template leaves
    the off-diagonal vectors as the same permutation of each other -> r is
    unchanged (permutation invariance 'as appropriate')."""
    rng = np.random.default_rng(13)
    B = rng.normal(size=(6, 6))
    base = cd.persistence_metric(B, SYN_OFF)
    for _ in range(10):
        perm = rng.permutation(6)
        Bp = B[np.ix_(perm, perm)]
        Tp = SYN[np.ix_(perm, perm)]
        assert cd.persistence_metric(Bp, cd.offdiag_vector(Tp)) == \
            pytest.approx(base)


def test_degenerate_bridge_is_nan():
    """Zero-variance off-diagonal (constant) -> NaN, not a crash."""
    B = np.full((6, 6), 0.3)
    np.fill_diagonal(B, 1.0)                        # off-diag all 0.3
    assert np.isnan(cd.persistence_metric(B, SYN_OFF))


def test_metric_equals_direct_offdiag_corrcoef():
    """Pins the definition: r over the 30 row-major off-diagonal entries."""
    rng = np.random.default_rng(14)
    B = rng.normal(size=(6, 6))                     # asymmetric on purpose
    idx = [(i, j) for i in range(6) for j in range(6) if i != j]
    b = np.array([B[i, j] for i, j in idx])
    expected = float(np.corrcoef(b, SYN_OFF)[0, 1])
    assert cd.persistence_metric(B, SYN_OFF) == pytest.approx(expected)
    assert len(idx) == 30


def test_offdiag_vector_excludes_diagonal():
    M = np.arange(36, dtype=np.float64).reshape(6, 6)
    v = cd.offdiag_vector(M)
    assert v.size == 30
    assert M[0, 0] not in set(v.tolist())          # diagonal excluded


# ── 2. Determinism ───────────────────────────────────────────────────


def test_calibration_deterministic():
    t1, _ = cd.run_persistence_nulls(cd.SEED, N_SMALL, MOMENTS, SYN_OFF)
    t2, _ = cd.run_persistence_nulls(cd.SEED, N_SMALL, MOMENTS, SYN_OFF)
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)


def test_calibration_seed_sensitivity():
    t1, _ = cd.run_persistence_nulls(cd.SEED, N_SMALL, MOMENTS, SYN_OFF)
    t3, _ = cd.run_persistence_nulls(cd.SEED + 1, N_SMALL, MOMENTS, SYN_OFF)
    assert json.dumps(t1, sort_keys=True) != json.dumps(t3, sort_keys=True)


def test_nulls_json_byte_identical_across_runs(tmp_path):
    """Same seed => byte-identical nulls.json (D7)."""
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        tables, raw = cd.run_persistence_nulls(cd.SEED, N_SMALL, MOMENTS,
                                               SYN_OFF)
        cd.write_outputs(out, tables, MOMENTS, "synthetic (test)", FAKE_SHA,
                         N_SMALL, raw)
    assert (a / "nulls.json").read_bytes() == (b / "nulls.json").read_bytes()


def test_both_nulls_present_and_summarized():
    tables, raw = cd.run_persistence_nulls(cd.SEED, N_SMALL, MOMENTS, SYN_OFF)
    assert set(tables.keys()) == set(cd.NULLS) == {"rewire", "matched_moments"}
    for name in cd.NULLS:
        s = tables[name]
        assert s["n"] == N_SMALL
        for key in ("p2.5", "p97.5", "p99"):
            assert key in s["band"]
        # correlations live in [-1, 1]
        assert -1.0 <= s["band"]["p2.5"] <= s["band"]["p97.5"] <= 1.0
        assert raw[name].shape == (N_SMALL,)


# ── 3. Bank-path guard ───────────────────────────────────────────────


@pytest.mark.parametrize("bad", [
    "results/asset1-bank",
    "results/asset1-bank/sub",
    r"results\asset1-bank-bs2x8-archive",
    "C:/falco/rhombic/results/asset1-bank/x",
])
def test_bank_guard_refuses(bad):
    with pytest.raises(SystemExit):
        cd.assert_safe_out_dir(bad)


def test_bank_guard_allows_default():
    p = cd.assert_safe_out_dir("results/BM-000c-configd-persistence")
    assert isinstance(p, Path)
    assert str(cd.OUT_DIR).endswith("BM-000c-configd-persistence")


# ── 4. IP boundary ───────────────────────────────────────────────────


def test_output_carries_no_raw_matrix():
    """The structural guard passes on a real payload and its keys hold only
    derived statistics + a SHA-256 (no bridge-shaped arrays)."""
    tables, _ = cd.run_persistence_nulls(cd.SEED, N_SMALL, MOMENTS, SYN_OFF)
    payload = cd.build_payload(tables, MOMENTS, "synthetic (test)", FAKE_SHA,
                               N_SMALL)
    cd.assert_no_raw_matrix(payload)               # must not raise
    assert payload["template_identity"]["sha256"] == FAKE_SHA
    assert "shape" in payload["template_identity"]
    # No JSON KEY named like a raw template/matrix dump. (Provenance strings
    # may legitimately mention functions like edge_values(); we forbid the
    # dump keys in quoted-key syntax, not those mentions.)
    text = json.dumps(payload)
    for banned_key in ('"template_values"', '"template_matrix"',
                       '"template_offdiag"', '"template_array"'):
        assert banned_key not in text


def test_ip_guard_fires_on_planted_matrix():
    """Positive control: a smuggled length-30 (or 6x6) array is caught."""
    with pytest.raises(AssertionError):
        cd.assert_no_raw_matrix({"leak": [0.1] * 30})
    with pytest.raises(AssertionError):
        cd.assert_no_raw_matrix({"leak": [[0.0] * 6 for _ in range(6)]})
    # A short grid (the percentile grid) must NOT trip the guard.
    cd.assert_no_raw_matrix({"percentile_grid": list(cd.PERSIST_PCT_GRID)})


# ── 5. Real-corpus template (gated) ──────────────────────────────────


requires_corpus = pytest.mark.skipif(
    not corpus_available(),
    reason="corpus_private.json absent — real-template tests skipped")


@requires_corpus
def test_real_template_loads_and_hashes():
    M, sha = cd.load_template()
    assert M.shape == (6, 6)
    assert np.allclose(np.diag(M), 1.0)            # identity diagonal
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)


@requires_corpus
def test_real_run_hides_template_values(tmp_path):
    """End-to-end on the real template: the emitted nulls.json contains none
    of the template's raw off-diagonal values, only derived stats + SHA-256."""
    M, sha = cd.load_template()
    t_off = cd.offdiag_vector(M)
    tables, raw = cd.run_persistence_nulls(cd.SEED, 300, MOMENTS, t_off)
    payload = cd.write_outputs(tmp_path, tables, MOMENTS, "synthetic (test)",
                               sha, 300, raw)
    cd.assert_no_raw_matrix(payload)

    # Collect every number appearing anywhere in the emitted JSON.
    emitted: list[float] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for e in x:
                walk(e)
        elif isinstance(x, (int, float)):
            emitted.append(float(x))

    on_disk = json.loads((tmp_path / "nulls.json").read_text(encoding="utf-8"))
    walk(on_disk)
    emitted_set = set(np.round(emitted, 10).tolist())

    # No raw template off-diagonal value (excluding trivial 0.0) may appear.
    for v in t_off:
        if abs(v) < 1e-12:
            continue
        assert round(float(v), 10) not in emitted_set, "template value leaked"
    # The SHA-256 (a hash, not a value) IS present — that is the only identity.
    assert sha in json.dumps(on_disk)


@requires_corpus
def test_real_run_bands_are_null_centered():
    """Sanity: against the real template, both BM-000 nulls sit ~0-centered
    (a bridge with no memory of the init is uncorrelated with it)."""
    M, _ = cd.load_template()
    t_off = cd.offdiag_vector(M)
    tables, _ = cd.run_persistence_nulls(cd.SEED, 2000, MOMENTS, t_off)
    for name in cd.NULLS:
        s = tables[name]
        assert abs(s["mean"]) < 0.05               # centered near zero
        assert s["band"]["p97.5"] < 0.6            # nowhere near persistence=1
        assert s["band"]["p99"] < 0.7
