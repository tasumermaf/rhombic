"""Tests for the Level-B J-lens synthetic POSITIVE CONTROL (Stage PC).

The Director's hard condition for treating any Level-B signature as an
arbiter verdict (docs/DIRECTOR_RULING_JLENS_STAGEB_2026-07-09.md): the
lens->signature pathway must recover a planted propagation structure
above a matched null AND read a genuinely output-null planted update as
null — in the same class as Level A's synthetic selftest.

Covered here:
  1. END-TO-END PASS — run_positive_control produces PASS with planted
     LOO == 1.0 above the matched null and the permutation band, and the
     output-null case reads numerically zero while its raw update still
     carries the task.
  2. DETERMINISM — identical seed => byte-identical JSON stats (only the
     created_at timestamp differs).
  3. OUTPUT-NULL GENUINELY NULLS — lensed residual ratio and absolute
     magnitude are float-dust while the raw (identity-lens) readout of the
     SAME updates recovers the tasks perfectly (so it is planted, not
     empty).
  4. SABOTAGE (non-tautology) — signing the planted bank with a DIFFERENT
     toy map's lens collapses recovery toward chance, proving the control
     can detect a broken/mismatched lens.
  5. BANK-PATH GUARD — an --out-dir inside the live campaign tree is
     refused.
  6. CLEANLINESS — the control path imports no transformers and leaves
     CUDA uninitialized (Stage-PC is CPU-only construction class).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")

import asset1_jacobian_lens as jlens  # noqa: E402

SEED = jlens.CONTROL_SEED
CHANCE = 1.0 / jlens.PC_N_TASKS


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("jlens-pc")
    return jlens.run_positive_control(out, seed=SEED)


# ── 1. End-to-end PASS ──────────────────────────────────────────────


def test_control_passes_end_to_end(report):
    assert report["PASS"] is True
    checks = report["checks"]
    assert checks["planted_recovered"] is True
    assert checks["output_null_reads_null"] is True
    assert checks["sabotage_detected"] is True

    res = report["results"]
    # planted recovery is perfect and above every null.
    assert res["planted"]["loo"] == 1.0
    assert res["matched_null"]["loo"] <= jlens.PC_NULL_BAND_HI
    assert res["planted"]["loo"] > res["matched_null"]["loo"]
    assert res["planted"]["loo"] > res["permutation_null"]["max"]
    assert res["chance"] == CHANCE


def test_toy_map_geometry_is_sound(report):
    """Every toy Jacobian is full row-rank with a real null space, an
    exact right inverse, and J @ null_basis ~ 0 — the preconditions the
    planted/output-null constructions rely on."""
    geom = report["toy_map_geometry"]
    assert len(geom) == len(jlens.PC_MODULES)
    for g in geom:
        assert g["J_row_rank"] == jlens.PC_V_TOY
        assert g["null_dim"] == jlens.PC_D_MODEL - jlens.PC_V_TOY
        assert g["J_pinv_identity_maxerr"] < 1e-8
        assert g["J_nullbasis_maxabs"] < 1e-8


# ── 2. Determinism ──────────────────────────────────────────────────


def test_control_deterministic_same_seed(tmp_path):
    r1 = jlens.run_positive_control(tmp_path / "a", seed=SEED)
    r2 = jlens.run_positive_control(tmp_path / "b", seed=SEED)
    for r in (r1, r2):
        r.pop("created_at")
    # Bit-identical stats/booleans/geometry across independent runs.
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    # …and identical on disk too (minus the timestamp line).
    j1 = json.loads((tmp_path / "a" / "jlens_positive_control.json").read_text(
        encoding="utf-8"))
    j2 = json.loads((tmp_path / "b" / "jlens_positive_control.json").read_text(
        encoding="utf-8"))
    j1.pop("created_at"), j2.pop("created_at")
    assert j1 == j2


def test_different_seed_changes_stats(tmp_path):
    """A different seed must actually drive a different toy geometry
    (guards against a hard-coded/ignored seed)."""
    r_alt = jlens.run_positive_control(tmp_path / "alt", seed=SEED + 1)
    ref_geom = jlens.build_toy_frozen_map(SEED)["geometry"]
    assert r_alt["toy_map_geometry"] != ref_geom
    # the control is robust: it still PASSES at another seed.
    assert r_alt["PASS"] is True


# ── 3. Output-null genuinely nulls (norm, not LOO) ──────────────────


def test_output_null_reads_numerically_zero(report):
    onull = report["results"]["output_null"]
    assert onull["residual_ratio"] < jlens.PC_RESIDUAL_RATIO_MAX
    assert onull["lensed_maxabs"] < jlens.PC_ONULL_MAXABS_MAX
    # the update is GENUINELY planted: its raw (identity-lens) readout of
    # the identical updates recovers the tasks perfectly — the lens reads
    # nothing not because there is nothing, but because it lives in
    # null(J_toy).
    assert onull["raw_loo"] == 1.0


def test_output_null_construction_lies_in_null_space():
    """Directly: every output-null update's column space is inside
    null(J_toy), so (S^T J_toy) @ Delta @ X is numerically zero while
    Delta @ X is not."""
    toy = jlens.build_toy_frozen_map(SEED)
    banks, _ = jlens.build_output_null_bank(toy, SEED)
    Xo = jlens.control_signatures(banks, toy["lenses"], seed=SEED)
    Xo_raw = jlens.control_signatures(banks, toy["raw_lenses"], seed=SEED)
    assert np.abs(Xo).max() < 1e-8
    assert np.linalg.norm(Xo_raw, axis=1).min() > 1e-3   # genuinely nonzero


# ── 4. Sabotage — wrong lens must fail (non-tautology) ──────────────


def test_sabotage_wrong_lens_fails_to_recover():
    """Actively: build the planted bank, recover with the CORRECT lens
    (LOO 1.0), then re-sign the very same bank with a DIFFERENT toy map's
    lens — recovery must collapse toward chance. This is what makes the
    control a real positive control rather than a tautology that any lens
    would pass."""
    toy = jlens.build_toy_frozen_map(SEED)
    banks, labels = jlens.build_planted_bank(toy, SEED)

    X_good = jlens.control_signatures(banks, toy["lenses"], seed=SEED)
    loo_good = jlens.loo_nearest_centroid_accuracy(X_good, labels)
    assert loo_good == 1.0

    toy_wrong = jlens.build_toy_frozen_map(SEED + jlens.PC_SABOTAGE_SEED_OFFSET)
    X_bad = jlens.control_signatures(banks, toy_wrong["lenses"], seed=SEED)
    loo_bad = jlens.loo_nearest_centroid_accuracy(X_bad, labels)

    assert loo_bad < loo_good
    assert loo_bad <= jlens.PC_NULL_BAND_HI


def test_sabotage_recorded_in_report(report):
    sab = report["results"]["sabotage"]
    assert sab["wrong_lens_loo"] < report["results"]["planted"]["loo"]
    assert sab["wrong_lens_loo"] <= jlens.PC_NULL_BAND_HI


# ── 5. Bank-path guard ──────────────────────────────────────────────


def test_positive_control_refuses_bank_tree():
    with pytest.raises(SystemExit, match="asset1-bank"):
        jlens.run_positive_control(
            REPO_ROOT / "results" / "asset1-bank" / "pc")


def test_cli_positive_control_refuses_bank_tree():
    bad = str(REPO_ROOT / "results" / "asset1-bank" / "pc")
    with pytest.raises(SystemExit, match="asset1-bank"):
        jlens.main(["--positive-control", "--out-dir", bad])


# ── 6. Construction-class cleanliness (no transformers, no CUDA) ────


def test_control_path_imports_no_transformers_and_no_cuda(tmp_path):
    jlens.run_positive_control(tmp_path / "clean", seed=SEED)
    assert "transformers" not in sys.modules
    assert not torch.cuda.is_initialized()
