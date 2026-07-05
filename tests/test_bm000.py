"""BM-000 null-model generator tests (results/BM-000/PROTOCOL.md section 5).

Verifies:
  1. Determinism — same seed => bit-identical percentile tables across two
     independent generator invocations.
  2. Different seed => different tables (the seed actually matters).
  3. Pair-definition consistency across the two modules that define co/cross
     pairs (train_exp2_scale vs train_contrastive_bridge).
  4. percentile_of correctness on a known array.
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

import bm000_null_model as bm  # noqa: E402

# Fixed synthetic moments — the test must not depend on the smoke bank files.
MOMENTS = {
    "diag_mean": 1.0, "diag_std": 0.001,
    "offdiag_mean": 0.0, "offdiag_std": 0.002,
    "n_matrices": 0,
}
N_SMALL = 150


def test_null_generator_deterministic():
    """Same seed => identical percentile tables (protocol seed 20260704)."""
    t1 = bm.run_nulls(bm.SEED, N_SMALL, MOMENTS)
    t2 = bm.run_nulls(bm.SEED, N_SMALL, MOMENTS)
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)


def test_null_generator_seed_sensitivity():
    """A different seed must change the percentiles (RNG is actually used)."""
    t1 = bm.run_nulls(bm.SEED, N_SMALL, MOMENTS)
    t3 = bm.run_nulls(bm.SEED + 1, N_SMALL, MOMENTS)
    assert json.dumps(t1, sort_keys=True) != json.dumps(t3, sort_keys=True)


def test_ensemble_structure():
    """All protocol ensembles present with all metrics summarized."""
    tables = bm.run_nulls(bm.SEED, 50, MOMENTS)
    expected_ensembles = {
        "gauss6", "rdmask6", "identity+eps0.01", "identity+eps0.05",
        "identity+eps0.1", "gauss4", "gauss8",
    }
    assert set(tables.keys()) == expected_ensembles
    for ens in expected_ensembles:
        for metric in ("co_cross", "fiedler", "fiedler_atb", "bd_flag",
                       "asymmetry"):
            assert tables[ens][metric]["n"] > 0, (ens, metric)


def test_pair_definitions_consistent():
    """train_exp2_scale and train_contrastive_bridge must agree (protocol M1)."""
    bm._verify_pair_definitions()


def test_percentile_of():
    arr = np.arange(100, dtype=np.float64)  # 0..99
    assert bm.percentile_of(50.0, arr) == pytest.approx(50.0)
    assert bm.percentile_of(1000.0, arr) == pytest.approx(100.0)
    assert bm.percentile_of(-1.0, arr) == pytest.approx(0.0)


def test_identity_bridge_degenerate_reference():
    """Frozen-identity reference: Fiedler and asymmetry exactly 0 (protocol
    note — family (c) proper is degenerate; eps ensembles bracket it)."""
    B = np.eye(6)
    assert bm.metric_fiedler(B) == pytest.approx(0.0, abs=1e-12)
    assert bm.metric_asymmetry(B) == pytest.approx(0.0, abs=1e-12)
