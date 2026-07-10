"""E-5 coherence-interpolation bifurcation sweep tests.

Covers the two deliverables of docs/E5_BIFURCATION_PREREG_2026-07-10.md:

  A. train_cybernetic.apply_pair_correctness — the §2 corruption operator:
     f=1.0 byte-identical no-op (global RNG untouched, exact 4-key record),
     determinism, count preservation, k = round((1-f)*N), realized-agreement
     semantics, global-stream purity at f<1, and n=6 generality. Plus the
     --pair-correctness parser default.
  B. e5_bifurcation_sweep — the runner/endpoint tallies: manifest lifecycle,
     resume-skips-COMPLETE, FAILED + retry, endpoint math against a
     hand-computed value, the f=1 trained==true positive control, band
     boundaries, and atomic manifest JSON.

CPU/stdlib+numpy. train_cybernetic / train_exp2_scale import torch at module
load (slow, present in the env); no model download, no CUDA, no real subprocess.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_cybernetic as tc          # noqa: E402
import e5_bifurcation_sweep as sweep    # noqa: E402


# The n=8 tesseract spec (4 co-axial + 24 cross-axial).
N8_CO = [(0, 1), (2, 3), (4, 5), (6, 7)]
_N8_CO_SET = set(N8_CO)
N8_CROSS = [(i, j) for i in range(8) for j in range(i + 1, 8)
            if (i, j) not in _N8_CO_SET]
GRID_F_BELOW_1 = [0.0, 0.25, 0.5, 0.75]


# ── A. apply_pair_correctness ───────────────────────────────────────


def test_f1_is_byte_identical_noop():
    st = random.getstate()
    co_out, cross_out, rec = tc.apply_pair_correctness(N8_CO, N8_CROSS, 1.0, 42)
    # No RNG constructed → global random stream is bit-for-bit unchanged.
    assert random.getstate() == st
    # Same objects returned, unchanged order.
    assert co_out is N8_CO and cross_out is N8_CROSS
    assert rec == {
        "pair_correctness": 1.0,
        "n_corrupted": 0,
        "corrupted_pairs": [],
        "realized_agreement": 1.0,
    }


def test_f_above_1_also_noop():
    co_out, cross_out, rec = tc.apply_pair_correctness(N8_CO, N8_CROSS, 1.5, 42)
    assert co_out is N8_CO and cross_out is N8_CROSS
    assert rec["n_corrupted"] == 0 and rec["realized_agreement"] == 1.0


def test_determinism_same_args_same_output():
    a = tc.apply_pair_correctness(N8_CO, N8_CROSS, 0.5, 42)
    b = tc.apply_pair_correctness(N8_CO, N8_CROSS, 0.5, 42)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_seed_changes_the_draw():
    a = tc.apply_pair_correctness(N8_CO, N8_CROSS, 0.5, 42)[2]["corrupted_pairs"]
    b = tc.apply_pair_correctness(N8_CO, N8_CROSS, 0.5, 43)[2]["corrupted_pairs"]
    assert a != b


def test_count_preserved_at_every_grid_f():
    for f in GRID_F_BELOW_1:
        co_out, cross_out, _ = tc.apply_pair_correctness(N8_CO, N8_CROSS, f, 42)
        assert len(co_out) == 4 and len(cross_out) == 24
        # The output pairs are exactly a repartition of the same universe.
        assert sorted(co_out + cross_out) == sorted(N8_CO + N8_CROSS)


def test_k_equals_round_one_minus_f_times_n():
    for f in GRID_F_BELOW_1:
        rec = tc.apply_pair_correctness(N8_CO, N8_CROSS, f, 42)[2]
        expected_k = round((1.0 - f) * 28)
        assert rec["k"] == expected_k
        assert rec["n_corrupted"] == expected_k
        assert len(rec["corrupted_pairs"]) == expected_k
        assert len(rec["pre_labels"]) == expected_k
        assert len(rec["post_labels"]) == expected_k


def test_realized_agreement_in_unit_interval_and_iff_fixed():
    for f in GRID_F_BELOW_1:
        co_out, cross_out, rec = tc.apply_pair_correctness(N8_CO, N8_CROSS, f, 42)
        ra = rec["realized_agreement"]
        assert 0.0 <= ra <= 1.0
        # realized == 1 exactly when the shuffle left the true labels intact,
        # i.e. the output partition equals the input partition.
        fixed = (co_out == N8_CO and cross_out == N8_CROSS)
        assert (ra == 1.0) == fixed
        # post labels are a permutation of the pre labels (count-preserving).
        assert sorted(rec["pre_labels"]) == sorted(rec["post_labels"])


def test_global_streams_untouched_at_f_below_1():
    r_before = random.getstate()
    np_before = np.random.get_state()
    tc.apply_pair_correctness(N8_CO, N8_CROSS, 0.5, 42)
    assert random.getstate() == r_before
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0]
    assert np.array_equal(np_before[1], np_after[1])
    assert (np_before[2], np_before[3], np_before[4]) == \
        (np_after[2], np_after[3], np_after[4])


def test_works_for_n6_pair_sets():
    co6 = [(0, 1), (2, 3), (4, 5)]
    cross6 = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3),
              (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]  # 12
    assert len(co6) + len(cross6) == 15
    for f in GRID_F_BELOW_1:
        co_out, cross_out, rec = tc.apply_pair_correctness(co6, cross6, f, 7)
        assert len(co_out) == 3 and len(cross_out) == 12
        assert rec["k"] == round((1.0 - f) * 15)


def test_pair_correctness_parser_default_and_parse():
    ns = tc.build_parser().parse_args([])
    assert ns.pair_correctness == 1.0
    ns2 = tc.build_parser().parse_args(["--pair-correctness", "0.5"])
    assert ns2.pair_correctness == 0.5


# ── B. sweep runner + endpoints ─────────────────────────────────────


def _make_bridge(co_pairs, co_val=10.0, cross_val=1.0, n=8):
    """Symmetric bridge: |B[i,j]| = co_val on co pairs, cross_val elsewhere."""
    B = np.eye(n, dtype=np.float64)
    co_set = {tuple(p) for p in co_pairs}
    for i in range(n):
        for j in range(i + 1, n):
            v = co_val if (i, j) in co_set else cross_val
            B[i, j] = v
            B[j, i] = v
    return B


def _write_run(run_dir, co_pairs_trained, pair_correctness, *,
               n_adapters=1, co_val=10.0, cross_val=1.0,
               realized_agreement=1.0, checkpoint=None):
    """Fake trainer output dir: config.json + bridge_final_*.npy + results.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    co = [list(p) for p in co_pairs_trained]
    all_pairs = [[i, j] for i in range(8) for j in range(i + 1, 8)]
    cross = [p for p in all_pairs if p not in co]
    config = {
        "pair_correctness": pair_correctness,
        "realized_agreement": realized_agreement,
        "co_pairs_trained": co,
        "cross_pairs_trained": cross,
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    B = _make_bridge(co_pairs_trained, co_val, cross_val)
    for a in range(n_adapters):
        np.save(run_dir / f"bridge_final_layer{a}.npy", B)
    cp = checkpoint or {"step": 3000, "co_cross_ratio": 10.0,
                        "fiedler_mean": 0.2, "val_loss": 2.5}
    (run_dir / "results.json").write_text(
        json.dumps({"checkpoints": [cp]}), encoding="utf-8")
    return run_dir


def _stub_factory(fail_ids=()):
    """A _invoke_training stub that writes a true-spec fake run (or fails)."""
    calls = []

    def stub(cmd, log_path, cwd):
        out = Path(cmd[cmd.index("--output") + 1])
        f = float(cmd[cmd.index("--pair-correctness") + 1])
        rid = out.name
        calls.append(rid)
        if rid in fail_ids:
            return 1  # crash before writing artifacts
        _write_run(out, N8_CO, f)  # true spec ⇒ f=1 positive control holds
        return 0

    stub.calls = calls
    return stub


def test_endpoint_math_uses_post_corruption_pairs(tmp_path):
    # A corrupted spec: (6,7) demoted to cross, (0,2) promoted to co.
    trained_co = [(0, 1), (2, 3), (4, 5), (0, 2)]
    run_dir = _write_run(tmp_path / "f0.50_s42", trained_co, 0.5, n_adapters=2)
    rec = sweep.compute_run_endpoint(run_dir, "f0.50_s42", 0.5, 42)
    # Trained: co={10,10,10,10}, cross=all 1 → 10.0
    assert rec["co_cross_trained"] == pytest.approx(10.0)
    # True: co={(0,1),(2,3),(4,5),(6,7)} → mean 7.75; cross has one 10 → 1.375
    assert rec["co_cross_true"] == pytest.approx(62.0 / 11.0)
    # Pooled magnitude: (4*10 + 24*1)/28 = 16/7
    assert rec["mean_magnitude"] == pytest.approx(16.0 / 7.0)
    assert rec["fiedler_mean"] == 0.2
    assert rec["val_loss"] == 2.5
    assert rec["final_co_cross_ratio"] == 10.0
    assert rec["n_adapters"] == 2
    assert rec["band"] == "Intermediate"   # 10.0 sits at the [10^1,10^3) floor


def test_f1_positive_control_trained_equals_true(tmp_path):
    run_dir = _write_run(tmp_path / "f1.00_s42", N8_CO, 1.0)
    rec = sweep.compute_run_endpoint(run_dir, "f1.00_s42", 1.0, 42)
    assert rec["co_cross_trained"] == pytest.approx(rec["co_cross_true"])
    assert rec["co_cross_trained"] == pytest.approx(10.0)


def test_f1_positive_control_raises_when_specs_diverge(tmp_path):
    # A config that claims f=1 but records a corrupted trained spec must trip
    # the positive-control assertion (trained != true).
    run_dir = _write_run(tmp_path / "bad", [(0, 1), (2, 3), (4, 5), (0, 2)], 1.0)
    with pytest.raises(AssertionError, match="positive control"):
        sweep.compute_run_endpoint(run_dir, "bad", 1.0, 42)


def test_band_classification_boundaries():
    assert sweep.classify_band(1e3) == "Structured"        # >= 10^3
    assert sweep.classify_band(1e3 - 1) == "Intermediate"
    assert sweep.classify_band(1e1) == "Intermediate"      # [10^1, ...)
    assert sweep.classify_band(1e1 - 1e-6) == "Unstructured"
    assert sweep.classify_band(1e5) == "Structured"
    assert sweep.classify_band(1.0) == "Unstructured"
    assert sweep.classify_band(float("inf")) == "Structured"
    # NaN is a diverged/defective endpoint, NOT a basin (prereg amendment 1)
    assert sweep.classify_band(float("nan")) == "NonFinite"
    assert sweep.classify_band(None) == "NonFinite"


def test_manifest_lifecycle_pending_to_complete(tmp_path, monkeypatch):
    stub = _stub_factory()
    monkeypatch.setattr(sweep, "_invoke_training", stub)
    outdir = tmp_path / "sweep"
    manifest = sweep.run_sweep(outdir=outdir, steps=3000, repo_root=tmp_path)

    # All 15 grid cells ran, in f-ascending / seed-ascending order.
    expected_ids = [sweep.run_id_for(f, s)
                    for f in sweep.F_GRID for s in sweep.SEED_GRID]
    assert stub.calls == expected_ids
    assert len(manifest["runs"]) == 15
    assert all(e["status"] == "COMPLETE" for e in manifest["runs"])
    for e in manifest["runs"]:
        assert e["return_code"] == 0
        assert e["started_at"] and e["finished_at"]

    # Manifest on disk is valid JSON with no lingering temp file.
    mpath = (outdir).resolve() / "sweep_manifest.json"
    disk = json.loads(mpath.read_text(encoding="utf-8"))
    assert all(e["status"] == "COMPLETE" for e in disk["runs"])
    assert not (mpath.parent / (mpath.name + ".tmp")).exists()

    # Endpoints were computed incrementally (all 15 present, f=1 control held).
    epath = mpath.parent / "endpoints.json"
    endpoints = json.loads(epath.read_text(encoding="utf-8"))
    assert len(endpoints["runs"]) == 15


def test_resume_skips_complete_runs(tmp_path, monkeypatch):
    stub1 = _stub_factory()
    monkeypatch.setattr(sweep, "_invoke_training", stub1)
    outdir = tmp_path / "sweep"
    sweep.run_sweep(outdir=outdir, steps=3000, repo_root=tmp_path)
    assert len(stub1.calls) == 15

    # Second pass: everything COMPLETE ⇒ the stub is never called again.
    stub2 = _stub_factory()
    monkeypatch.setattr(sweep, "_invoke_training", stub2)
    manifest = sweep.run_sweep(outdir=outdir, steps=3000, repo_root=tmp_path)
    assert stub2.calls == []
    assert all(e["status"] == "COMPLETE" for e in manifest["runs"])


def test_failed_run_then_retry(tmp_path, monkeypatch):
    first_id = sweep.run_id_for(sweep.F_GRID[0], sweep.SEED_GRID[0])
    stub1 = _stub_factory(fail_ids={first_id})
    monkeypatch.setattr(sweep, "_invoke_training", stub1)
    outdir = tmp_path / "sweep"
    manifest = sweep.run_sweep(outdir=outdir, steps=3000, repo_root=tmp_path)

    entry = next(e for e in manifest["runs"] if e["run_id"] == first_id)
    assert entry["status"] == "FAILED" and entry["return_code"] == 1
    # The other 14 completed despite the failure (continue-on-nonzero).
    assert sum(e["status"] == "COMPLETE" for e in manifest["runs"]) == 14

    # Retry: a healthy stub is called ONLY for the failed run.
    stub2 = _stub_factory()
    monkeypatch.setattr(sweep, "_invoke_training", stub2)
    manifest2 = sweep.run_sweep(outdir=outdir, steps=3000, repo_root=tmp_path)
    assert stub2.calls == [first_id]
    assert all(e["status"] == "COMPLETE" for e in manifest2["runs"])


def test_dry_run_executes_nothing(tmp_path, monkeypatch):
    stub = _stub_factory()
    monkeypatch.setattr(sweep, "_invoke_training", stub)
    outdir = tmp_path / "sweep"
    manifest = sweep.run_sweep(outdir=outdir, steps=3000, repo_root=tmp_path,
                               dry_run=True)
    assert stub.calls == []
    assert all(e["status"] == "PENDING" for e in manifest["runs"])


def test_atomic_manifest_write_leaves_valid_json(tmp_path):
    path = tmp_path / "m.json"
    sweep._atomic_write_json(path, {"runs": [{"run_id": "x", "status": "OK"}]})
    assert json.loads(path.read_text(encoding="utf-8"))["runs"][0]["run_id"] == "x"
    assert not (path.parent / (path.name + ".tmp")).exists()


def test_summarize_writes_tally(tmp_path, monkeypatch):
    stub = _stub_factory()
    monkeypatch.setattr(sweep, "_invoke_training", stub)
    outdir = (tmp_path / "sweep").resolve()
    sweep.run_sweep(outdir=outdir, steps=3000, repo_root=tmp_path)
    out = sweep.summarize(outdir)
    text = Path(out).read_text(encoding="utf-8")
    assert "NOT the confirmatory analysis" in text
    assert "Intermediate" in text
    assert "15 / 15" in text


# ── pre-launch verification fixes (prereg amendment 2026-07-10) ──────


def test_resume_refuses_steps_mismatch(tmp_path):
    path = tmp_path / "sweep_manifest.json"
    sweep._atomic_write_json(path, sweep._init_manifest(3000))
    with pytest.raises(SystemExit, match="mixed-depth"):
        sweep._load_or_init_manifest(path, 2000)
    # Same steps resumes fine.
    man = sweep._load_or_init_manifest(path, 3000)
    assert man["steps"] == 3000


def test_endpoint_error_isolated_per_run(tmp_path):
    outdir = (tmp_path / "sweep").resolve()
    good_id = sweep.run_id_for("1.00", 42)
    bad_id = sweep.run_id_for("1.00", 43)
    _write_run(outdir / good_id, N8_CO, 1.0)
    (outdir / bad_id).mkdir(parents=True)   # no artifacts: poisoned run dir
    man = sweep._init_manifest(3000)
    for e in man["runs"]:
        if e["run_id"] in (good_id, bad_id):
            e["status"] = "COMPLETE"
    sweep._atomic_write_json(outdir / "sweep_manifest.json", man)
    records = sweep._recompute_endpoints(outdir)
    assert [r["run_id"] for r in records] == [good_id]
    saved = json.loads((outdir / "endpoints.json").read_text(encoding="utf-8"))
    assert [e["run_id"] for e in saved["errors"]] == [bad_id]


def test_endpoint_aggregation_finite_filter_with_distinct_adapters(tmp_path):
    # Adapter 1: finite ratio 10.0. Adapter 2: all-zero cross entries → inf
    # ratio, filtered out. Mean over FINITE per-adapter ratios = 10.0 —
    # locks the trainer-parity aggregation (finite-only mean of per-adapter
    # ratios, NOT a pooled ratio), which identical-adapter fixtures cannot.
    run_dir = _write_run(tmp_path / "f1.00_s42", N8_CO, 1.0)
    B_inf = _make_bridge(N8_CO, 5.0, 0.0)   # cross mean <= 1e-12 → inf
    np.save(run_dir / "bridge_final_layer1.npy", B_inf)
    rec = sweep.compute_run_endpoint(run_dir, "f1.00_s42", 1.0, 42)
    assert rec["co_cross_trained"] == pytest.approx(10.0)
    assert rec["n_adapters"] == 2


def test_invoke_training_prepends_repo_root_to_pythonpath(tmp_path,
                                                          monkeypatch):
    # The child must import THIS tree's rhombic package, never a stale
    # editable install elsewhere in the env (the Hermes failure mode).
    captured = {}

    def fake_run(cmd, cwd=None, stdout=None, stderr=None, env=None):
        captured["env"] = env
        captured["cwd"] = cwd
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/pre/existing")
    rc = sweep._invoke_training(["python", "x.py"], tmp_path / "r.log",
                                tmp_path)
    assert rc == 0
    pp = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert pp[0] == str(Path(tmp_path).resolve())
    assert "/pre/existing" in pp
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"


def test_trainer_rejects_out_of_range_f(monkeypatch):
    import train_cybernetic as tc
    for bad in ("1.5", "-0.1"):
        monkeypatch.setattr(sys, "argv",
                            ["train_cybernetic.py", "--pair-correctness", bad])
        with pytest.raises(SystemExit):
            tc.main()   # parser.error fires before any model/data loading
