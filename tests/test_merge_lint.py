"""Tests for scripts/merge_lint.py (the D3 merge-linter prototype).

Covers: the feature pass-through (merge_lint's featurize_pair must be
bit-identical to calling asset1_canonicalize.load_adapter_modules +
asset1_d3_merge.pair_features directly — the linter adds no feature of
its own), both refusal paths (different module sets; same names at
mismatched shapes — each refused with exit code 2 BEFORE any reference
data is read), the reference-model fit from a synthetic bank bundle in
the shipped d3_pairs.json / d3_labels.json schema (primary 5%-rule
binarization), the in-family full-model verdict, the out-of-family
distance-only fallback with its loud extrapolation banner, JSON output,
and pinned-seed determinism.

Everything above runs WITHOUT the real bank: synthetic adapters in the
bank's flat format (torch.save round-trip through the production
loader) and synthetic reference bundles in tmp dirs. The single
end-to-end test on two real bank adapters is skipif-gated on
results/asset1-bank and the results/asset1-delivery-verify bundle being
present, and is strictly read-only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

torch = pytest.importorskip("torch")

import asset1_d3_merge as d3  # noqa: E402
import merge_lint as ml  # noqa: E402
from asset1_canonicalize import load_adapter_modules  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_BANK = REPO_ROOT / "results" / "asset1-bank"
VERIFY_DIR = REPO_ROOT / "results" / "asset1-delivery-verify"

FAM = "synthfam"


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


def _random_adapter(rng, names=("m00", "m01"), d_out=16, d_in=12, rank=4,
                    scaling=0.5):
    return {n: _entry(rng.standard_normal((d_out, rank)),
                      rng.standard_normal((rank, d_in)),
                      bridge=np.eye(2), scaling=scaling)
            for n in names}


def _save_flat(modules, path):
    """torch.save a module dict in the bank's FLAT key format."""
    state = {f"{name}.{field}": tensor
             for name, entry in modules.items()
             for field, tensor in entry.items()}
    torch.save(state, path)
    return path


@pytest.fixture(scope="module")
def synthetic_reference(tmp_path_factory):
    """Synthetic bank bundle in the shipped schema: 12 pairs, one family,
    per-endpoint ppl fields driving the primary 5% rule (6 conflicts /
    6 clean — comfortably non-degenerate). No d3_report.json, so the
    no-AUC-context path is exercised too."""
    bank = tmp_path_factory.mktemp("merge-lint-ref")
    rng = np.random.default_rng(11)
    pair_recs, label_rows = [], []
    for i in range(12):
        pf = d3.pair_features(_random_adapter(rng), _random_adapter(rng))
        spec = {"family_short": FAM, "task_a": "t0", "run_index_a": 2 * i,
                "task_b": "t1", "run_index_b": 2 * i + 1}
        pair_recs.append({**spec, "features": d3._jsonable(pf)})
        positive = i % 2 == 0
        na, nb = 1.0, 1.2
        ma = na * (1.10 if positive else 1.01)     # >=5% up iff positive
        label_rows.append({**spec, "degradation": (ma - na) / na,
                           "merged_ppl_a": ma, "native_ppl_a": na,
                           "merged_ppl_b": nb, "native_ppl_b": nb})
    (bank / "d3_pairs.json").write_text(json.dumps(
        {"seed": 0, "alpha": 0.5, "n_per_family": 12, "max_run_uses": 1,
         "n_pairs": 12, "pairs": pair_recs}), encoding="utf-8")
    (bank / "d3_labels.json").write_text(
        json.dumps({"pairs": label_rows}), encoding="utf-8")
    return bank


def _fresh_pair_paths(tmp_path, seed=99, names=("m00", "m01"), **kw):
    rng = np.random.default_rng(seed)
    pa = _save_flat(_random_adapter(rng, names=names, **kw),
                    tmp_path / "a.pt")
    pb = _save_flat(_random_adapter(rng, names=names, **kw),
                    tmp_path / "b.pt")
    return pa, pb


def _probability_from_text(out: str) -> float:
    m = re.search(r"alpha=0\.5\): ([0-9.]+)", out)
    assert m, f"no probability line in output:\n{out}"
    return float(m.group(1))


# ── Feature pass-through (never reimplement) ────────────────────────


def test_featurize_pair_matches_d3_directly(tmp_path):
    """merge_lint.featurize_pair == load_adapter_modules + pair_features,
    bit for bit — every scalar and every per-module vector."""
    pa, pb = _fresh_pair_paths(tmp_path, seed=3)
    _, _, pf_ml = ml.featurize_pair(pa, pb)
    pf_d3 = d3.pair_features(load_adapter_modules(pa),
                             load_adapter_modules(pb))
    assert set(pf_ml) == set(pf_d3)
    assert pf_ml["module_names"] == pf_d3["module_names"]
    for key in ("cos_distance", "l2_distance", *d3.AGGREGATE_KEYS):
        assert pf_ml[key] == pf_d3[key]
    for key in ("module_l2", "module_angle_mean", "module_chordal_rms",
                "module_weight"):
        np.testing.assert_array_equal(pf_ml[key], pf_d3[key])


def test_featurize_pair_roundtrips_flat_format(tmp_path):
    """The saved flat states load back through the production loader with
    the exact module/field structure the fixtures built."""
    pa, _ = _fresh_pair_paths(tmp_path, seed=4)
    mods = load_adapter_modules(pa)
    assert sorted(mods) == ["m00", "m01"]
    for entry in mods.values():
        assert {"lora_A", "lora_B", "bridge", "scaling",
                "n_channels", "rank"} <= set(entry)


# ── Refusal paths (no bank needed — checked before any bank read) ───


def test_refuses_different_module_sets(tmp_path, capsys):
    rng = np.random.default_rng(5)
    pa = _save_flat(_random_adapter(rng, names=("m00", "m01")),
                    tmp_path / "a.pt")
    pb = _save_flat(_random_adapter(rng, names=("x00", "x01")),
                    tmp_path / "b.pt")
    with pytest.raises(SystemExit) as ei:
        # Nonexistent --bank-dir: proves refusal precedes any bank read.
        ml.main([str(pa), str(pb),
                 "--bank-dir", str(tmp_path / "no-such-dir")])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "CROSS-FAMILY" in err


def test_refuses_same_names_mismatched_shapes(tmp_path, capsys):
    rng = np.random.default_rng(6)
    pa = _save_flat(_random_adapter(rng, d_out=16), tmp_path / "a.pt")
    pb = _save_flat(_random_adapter(rng, d_out=20), tmp_path / "b.pt")
    with pytest.raises(SystemExit) as ei:
        ml.main([str(pa), str(pb),
                 "--bank-dir", str(tmp_path / "no-such-dir")])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err
    assert "shape mismatch" in err


def test_check_pair_compatible_passes_same_family():
    rng = np.random.default_rng(7)
    ml.check_pair_compatible(_random_adapter(rng), _random_adapter(rng))


# ── Reference model from the synthetic bundle ───────────────────────


def test_load_reference_binarizes_with_primary_rule(synthetic_reference):
    ref = ml.load_reference(synthetic_reference)
    assert ref["alpha"] == 0.5
    assert ref["binarization"]["rule_used"] == "relative_degradation"
    assert ref["binarization"]["threshold_rel"] == d3.THRESHOLD_REL
    slot = ref["families"][FAM]
    assert slot["n_pairs"] == 12
    assert slot["y"].sum() == 6            # 6 planted conflicts of 12
    assert ref["report"] is None           # no d3_report.json shipped here


def test_fit_reference_is_deterministic(synthetic_reference, tmp_path):
    ref = ml.load_reference(synthetic_reference)
    pa, pb = _fresh_pair_paths(tmp_path, seed=42)
    _, _, pf = ml.featurize_pair(pa, pb)
    probs = [ml.lint_pair(pf, ref, ml.fit_reference(ref))
             ["conflict_probability"] for _ in range(2)]
    assert probs[0] == probs[1]            # pinned seed 0, same data


def test_lint_in_family_full_model(synthetic_reference, tmp_path, capsys):
    pa, pb = _fresh_pair_paths(tmp_path, seed=8)
    ml.main([str(pa), str(pb), "--bank-dir", str(synthetic_reference)])
    out = capsys.readouterr().out
    assert "CONFLICT PROBABILITY" in out
    assert FAM in out
    assert "SCOPE" in out
    assert "!!!" not in out                # no extrapolation banner
    assert 0.0 <= _probability_from_text(out) <= 1.0


def test_lint_out_of_family_falls_back_loudly(synthetic_reference,
                                              tmp_path, capsys):
    pa, pb = _fresh_pair_paths(tmp_path, seed=9, names=("z00", "z01"))
    ml.main([str(pa), str(pb), "--bank-dir", str(synthetic_reference)])
    out = capsys.readouterr().out
    assert "!!! EXTRAPOLATION" in out
    assert "distance_only_fallback" in out
    assert 0.0 <= _probability_from_text(out) <= 1.0


def test_json_verdict_shape(synthetic_reference, tmp_path, capsys):
    pa, pb = _fresh_pair_paths(tmp_path, seed=10)
    ml.main([str(pa), str(pb), "--bank-dir", str(synthetic_reference),
             "--json"])
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["in_family"] is True
    assert verdict["family"] == FAM
    assert verdict["model_used"] == "full"
    assert verdict["extrapolation"] is None
    assert verdict["seed"] == 0
    assert verdict["n_reference_pairs"] == 12
    assert 0.0 <= verdict["conflict_probability"] <= 1.0
    for key in ("cos_distance", "l2_distance"):
        assert isinstance(verdict["features"][key], float)


def test_missing_bank_dir_message(tmp_path):
    pa, pb = _fresh_pair_paths(tmp_path, seed=12)
    with pytest.raises(FileNotFoundError, match="d3_pairs.json"):
        ml.main([str(pa), str(pb),
                 "--bank-dir", str(tmp_path / "absent")])


# ── Real bank (skipif — read-only) ──────────────────────────────────


@pytest.mark.skipif(
    not (REAL_BANK.is_dir()
         and (VERIFY_DIR / "d3_pairs.json").exists()
         and (VERIFY_DIR / "d3_labels.json").exists()),
    reason="real Asset-1 bank / delivery-verify bundle not present")
def test_real_bank_end_to_end(capsys):
    """One full run on two real bank adapters (same family, different
    tasks) against the shipped delivery-verify reference data."""
    adapters = sorted(
        REAL_BANK.glob("qwen2.5-1.5b/*/run_*/adapter_state.pt"))
    if len(adapters) < 2:
        pytest.skip("fewer than two qwen2.5-1.5b adapters in the bank")
    a = adapters[0]
    b = next((p for p in adapters if p.parts[-3] != a.parts[-3]),
             adapters[1])                  # prefer a cross-task pair
    ml.main([str(a), str(b), "--bank-dir", str(VERIFY_DIR), "--json"])
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["in_family"] is True
    assert verdict["family"] == "qwen2.5-1.5b"
    assert verdict["model_used"] == "full"
    assert verdict["n_modules"] == 112     # delivery-verify README
    assert 0.0 <= verdict["conflict_probability"] <= 1.0
    ctx = verdict["bank_context"]
    assert ctx and ctx["auc_full"] is not None
    assert ctx["n_pairs"] == 120
