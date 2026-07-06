"""Tests for the Asset 1 analysis IO layer + synthetic bank generator.

Covers: the pre-registration interlock (complete / partial / allow-partial /
wrong-total / missing-manifest), synthetic-bank schema round-trip through
EVERY io function (manifest, iter_runs, load_adapter, flatten_features,
feature_layout, load_gap_trajectory, load_bridges), deterministic
flattening order, layout slice correctness, cross-family dimension
differences, the planted task-separation / D-aux structure, and the synth
generator's safety guards.

Pre-registration hygiene: every statistic below runs on SYNTHETIC fixtures
generated in tmp dirs. Nothing reads or writes results/asset1-bank/, no HF
downloads, no network, no GPU.
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
import asset1_bank as bank  # noqa: E402
import asset1_synth as synth  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def effect_bank(tmp_path_factory):
    """2 families x 3 tasks x 2 reps = 12 runs, task_effect = 1.0."""
    root = tmp_path_factory.mktemp("synth-effect") / "bank"
    info = synth.make_synthetic_bank(
        root, n_families=2, n_tasks=3, n_reps=2, n_layers=2,
        d_model=16, rank=4, n_channels=2, task_effect=1.0, seed=7)
    return root, info


@pytest.fixture(scope="module")
def zero_bank(tmp_path_factory):
    """1 family x 2 tasks x 4 reps = 8 runs, task_effect = 0.0."""
    root = tmp_path_factory.mktemp("synth-zero") / "bank"
    info = synth.make_synthetic_bank(
        root, n_families=1, n_tasks=2, n_reps=4, n_layers=2,
        d_model=16, rank=4, n_channels=2, task_effect=0.0, seed=7)
    return root, info


@pytest.fixture()
def tiny_bank(tmp_path):
    """1 x 1 x 2 = 2 runs — function-scoped, safe to mutate."""
    root = tmp_path / "bank"
    synth.make_synthetic_bank(
        root, n_families=1, n_tasks=1, n_reps=2, n_layers=1,
        d_model=8, rank=2, n_channels=2, task_effect=1.0, seed=3)
    return root


def _set_status(root: Path, run_index: int, status: str) -> None:
    path = root / "bank_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["runs"][run_index]["status"] = status
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _runs(root, **kw):
    return list(aio.iter_runs(root, **kw))


# ── Manifest + schema fidelity ──────────────────────────────────────


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        aio.load_manifest(tmp_path)


def test_manifest_entry_fields_match_runspec(effect_bank):
    """Synthetic manifest entries carry EXACTLY the RunSpec field names."""
    root, _ = effect_bank
    spec = bank.make_campaign_spec(smoke=True)
    real_entry = bank.generate_manifest(spec)[0].to_manifest_entry("PENDING")
    synth_entry = aio.load_manifest(root)["runs"][0]
    assert set(synth_entry) == set(real_entry)


def test_run_status_counts(effect_bank):
    root, info = effect_bank
    counts = aio.run_status_counts(aio.load_manifest(root))
    assert counts == {"COMPLETE": info["n_runs"]}
    assert info["n_runs"] == 12


# ── The interlock (require_complete_bank) ───────────────────────────


def test_interlock_passes_when_complete(tiny_bank, capsys):
    manifest = aio.require_complete_bank(tiny_bank, expected_total=2)
    assert len(manifest["runs"]) == 2
    assert "WARNING" not in capsys.readouterr().err


def test_interlock_refuses_partial(tiny_bank):
    _set_status(tiny_bank, 0, "PENDING")
    with pytest.raises(SystemExit) as exc:
        aio.require_complete_bank(tiny_bank, expected_total=2)
    assert "REFUSING" in str(exc.value)
    assert "1/2" in str(exc.value)


def test_interlock_refuses_wrong_total(tiny_bank):
    # All runs COMPLETE, but the default expected_total is the locked 480 —
    # a 2-run bank must be refused by a real-bank tool.
    with pytest.raises(SystemExit) as exc:
        aio.require_complete_bank(tiny_bank)
    assert "480" in str(exc.value)


def test_interlock_allow_partial_warns(tiny_bank, capsys):
    _set_status(tiny_bank, 0, "PENDING")
    manifest = aio.require_complete_bank(tiny_bank, allow_partial=True,
                                         expected_total=2)
    assert len(manifest["runs"]) == 2   # manifest still returned
    err = capsys.readouterr().err
    assert "PRE-REGISTRATION WARNING" in err
    assert "EXPLORATORY ONLY" in err


def test_interlock_missing_manifest_systemexit(tmp_path):
    with pytest.raises(SystemExit) as exc:
        aio.require_complete_bank(tmp_path)
    assert "REFUSING" in str(exc.value)


# ── iter_runs (manifest-driven enumeration) ─────────────────────────


def test_iter_runs_yields_all_with_fields(effect_bank):
    root, info = effect_bank
    records = _runs(root)
    assert len(records) == 12
    for rec in records:
        assert set(rec) == {"run_dir", "config", "family_short", "task",
                            "run_index", "replicate"}
        assert isinstance(rec["run_dir"], Path)
        assert rec["run_dir"].is_dir()
        cfg = rec["config"]
        assert cfg is not None
        assert cfg["run_index"] == rec["run_index"]
        assert cfg["family_short"] == rec["family_short"]
        assert cfg["task"] == rec["task"]
        assert cfg["replicate"] == rec["replicate"]
        assert cfg["seed"] == 10_000 + rec["run_index"]
        assert cfg["data_seed"] == 20_000 + rec["run_index"]


def test_iter_runs_manifest_order(effect_bank):
    root, _ = effect_bank
    indices = [r["run_index"] for r in _runs(root)]
    assert indices == sorted(indices)


def test_iter_runs_family_filter(effect_bank):
    root, _ = effect_bank
    short = _runs(root, family="synthfam0")
    assert len(short) == 6
    assert {r["family_short"] for r in short} == {"synthfam0"}
    # Full family id matches too
    full = _runs(root, family="synthetic/family-1")
    assert len(full) == 6
    assert {r["family_short"] for r in full} == {"synthfam1"}


def test_iter_runs_task_filter(effect_bank):
    root, _ = effect_bank
    recs = _runs(root, task="task01")
    assert len(recs) == 4   # 2 families x 2 reps
    assert {r["task"] for r in recs} == {"task01"}


def test_iter_runs_only_complete(tiny_bank):
    _set_status(tiny_bank, 0, "PENDING")
    assert len(_runs(tiny_bank)) == 1
    assert len(_runs(tiny_bank, only_complete=False)) == 2


# ── load_adapter ────────────────────────────────────────────────────


def test_load_adapter_schema(effect_bank):
    root, _ = effect_bank
    rec = _runs(root, family="synthfam0")[0]
    adapter = aio.load_adapter(rec["run_dir"])
    assert len(adapter) == 2 * 4          # n_layers=2 x {q,k,v,o}
    for name, entry in adapter.items():
        assert set(entry) == {"lora_A", "lora_B", "bridge", "scaling",
                              "n_channels", "rank"}
        assert entry["lora_A"].shape == (4, 16)          # (rank, d_in)
        d_out = 8 if ("_k_proj" in name or "_v_proj" in name) else 16
        assert entry["lora_B"].shape == (d_out, 4)       # (d_out, rank)
        assert entry["bridge"].shape == (2, 2)
        for f in ("lora_A", "lora_B", "bridge"):
            assert entry[f].dtype == torch.float32
        assert entry["scaling"].ndim == 0
        assert entry["scaling"].item() == pytest.approx(16.0 / 4)
        assert entry["rank"].item() == 4
        assert entry["n_channels"].item() == 2


def test_load_adapter_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        aio.load_adapter(tmp_path)


def test_cross_family_dims_differ(effect_bank):
    root, _ = effect_bank
    a0 = aio.load_adapter(_runs(root, family="synthfam0")[0]["run_dir"])
    a1 = aio.load_adapter(_runs(root, family="synthfam1")[0]["run_dir"])
    assert len(a0) == 8 and len(a1) == 12         # 2 vs 3 layers
    d_in0 = next(iter(a0.values()))["lora_A"].shape[1]
    d_in1 = next(iter(a1.values()))["lora_A"].shape[1]
    assert (d_in0, d_in1) == (16, 12)             # d_model mismatch
    v0 = aio.flatten_features(a0)
    v1 = aio.flatten_features(a1)
    assert v0.size != v1.size                     # not naively comparable


# ── flatten_features / feature_layout ───────────────────────────────


def test_flatten_deterministic_and_dtype(effect_bank):
    root, _ = effect_bank
    adapter = aio.load_adapter(_runs(root)[0]["run_dir"])
    v1 = aio.flatten_features(adapter)
    v2 = aio.flatten_features(adapter)
    assert v1.dtype == np.float32 and v1.ndim == 1
    assert np.array_equal(v1, v2)


def test_flatten_module_order_sorted(effect_bank):
    root, _ = effect_bank
    adapter = aio.load_adapter(_runs(root)[0]["run_dir"])
    layout = aio.feature_layout(adapter)
    modules_in_layout = [m for m, _, _ in layout]
    # Each module appears as a contiguous A, B, bridge triple, modules sorted
    assert modules_in_layout == [m for m in sorted(adapter)
                                 for _ in range(3)]
    fields = [f for _, f, _ in layout]
    assert fields == ["lora_A", "lora_B", "bridge"] * len(adapter)


def test_layout_slices_tile_vector(effect_bank):
    root, _ = effect_bank
    adapter = aio.load_adapter(_runs(root)[0]["run_dir"])
    vec = aio.flatten_features(adapter)
    layout = aio.feature_layout(adapter)
    pos = 0
    for _, _, sl in layout:
        assert sl.start == pos
        pos = sl.stop
    assert pos == vec.size


def test_layout_slice_correctness(effect_bank):
    root, _ = effect_bank
    adapter = aio.load_adapter(_runs(root)[0]["run_dir"])
    vec = aio.flatten_features(adapter)
    for module, field, sl in aio.feature_layout(adapter):
        expected = adapter[module][field].numpy().reshape(-1)
        assert np.array_equal(vec[sl], expected), (module, field)


def test_include_subsets(effect_bank):
    root, _ = effect_bank
    adapter = aio.load_adapter(_runs(root)[0]["run_dir"])
    bridges_only = aio.flatten_features(adapter, include={"bridge"})
    assert bridges_only.size == len(adapter) * 2 * 2
    a_only = aio.flatten_features(adapter, include=("A",))
    assert a_only.size == sum(e["lora_A"].numel() for e in adapter.values())
    # Caller's container order is irrelevant — canonical order is fixed
    v1 = aio.flatten_features(adapter, include=("bridge", "A"))
    v2 = aio.flatten_features(adapter, include=("A", "bridge"))
    assert np.array_equal(v1, v2)


def test_modules_subset_and_errors(effect_bank):
    root, _ = effect_bank
    adapter = aio.load_adapter(_runs(root)[0]["run_dir"])
    two = sorted(adapter)[:2]
    vec = aio.flatten_features(adapter, modules=two)
    expected = sum(adapter[m][f].numel() for m in two
                   for f in ("lora_A", "lora_B", "bridge"))
    assert vec.size == expected
    with pytest.raises(ValueError):
        aio.flatten_features(adapter, modules=["nonexistent_module"])
    with pytest.raises(ValueError):
        aio.flatten_features(adapter, modules=[two[0], two[0]])
    with pytest.raises(ValueError):
        aio.flatten_features(adapter, include={"C"})
    with pytest.raises(ValueError):
        aio.flatten_features(adapter, include=())
    with pytest.raises(ValueError):
        aio.flatten_features(adapter, modules="some")


# ── load_gap_trajectory ─────────────────────────────────────────────


def test_gap_trajectory(effect_bank):
    root, _ = effect_bank
    rec = _runs(root)[0]
    steps, train, val = aio.load_gap_trajectory(rec["run_dir"])
    assert steps.tolist() == [0, 100, 200]
    assert steps.dtype == np.int64
    assert np.isnan(train[0])                 # step-0 null -> NaN
    assert np.isfinite(train[1:]).all()
    assert np.isfinite(val).all()
    gap = val[-1] - train[-1]
    assert gap > 0
    # Planted gap recorded by the generator matches what we parse back
    assert gap == pytest.approx(
        rec["config"]["synthetic_generative_model"]["gap"], rel=1e-6)


def test_gap_trajectory_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        aio.load_gap_trajectory(tmp_path)


# ── load_bridges ────────────────────────────────────────────────────


def test_load_bridges_final_and_step0(effect_bank):
    root, _ = effect_bank
    rec = _runs(root)[0]
    adapter = aio.load_adapter(rec["run_dir"])
    final = aio.load_bridges(rec["run_dir"], which="final")
    step0 = aio.load_bridges(rec["run_dir"], which="step0")
    assert set(final) == set(step0) == set(adapter)
    eye = np.eye(2, dtype=np.float32)
    for name in adapter:
        assert final[name].shape == (2, 2)
        assert np.array_equal(step0[name], eye)          # identity at init
        assert not np.allclose(final[name], eye)         # trained deviation
        # bridge in adapter_state == bridge_final npy (same tensor saved)
        assert np.array_equal(final[name], adapter[name]["bridge"].numpy())


def test_load_bridges_validation(effect_bank, tmp_path):
    root, _ = effect_bank
    rec = _runs(root)[0]
    with pytest.raises(ValueError):
        aio.load_bridges(rec["run_dir"], which="initial")
    with pytest.raises(FileNotFoundError):
        aio.load_bridges(tmp_path)


def test_bridge_deviation_matches_planted_dev_mag(effect_bank):
    root, _ = effect_bank
    for rec in _runs(root, family="synthfam0"):
        dev_mag = rec["config"]["synthetic_generative_model"]["dev_mag"]
        bridges = aio.load_bridges(rec["run_dir"], which="final")
        eye = np.eye(2)
        for b in bridges.values():
            assert np.linalg.norm(b - eye) == pytest.approx(dev_mag,
                                                            rel=1e-4)


# ── Planted structure (SYNTHETIC-only validation) ───────────────────


def _task_mean_features(root, family, task):
    vecs = [aio.flatten_features(aio.load_adapter(r["run_dir"]))
            for r in aio.iter_runs(root, family=family, task=task)]
    return np.mean(np.stack(vecs), axis=0)


def test_task_separation_when_effect_positive(effect_bank):
    root, info = effect_bank
    means = {t: _task_mean_features(root, "synthfam0", t)
             for t in info["tasks"]}
    tasks = info["tasks"]
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            d = np.linalg.norm(means[tasks[i]] - means[tasks[j]])
            assert d > 3.0, (tasks[i], tasks[j], d)


def test_no_separation_when_effect_zero(zero_bank, effect_bank):
    zroot, zinfo = zero_bank
    d_zero = np.linalg.norm(
        _task_mean_features(zroot, "synthfam0", zinfo["tasks"][0])
        - _task_mean_features(zroot, "synthfam0", zinfo["tasks"][1]))
    eroot, einfo = effect_bank
    d_eff = np.linalg.norm(
        _task_mean_features(eroot, "synthfam0", einfo["tasks"][0])
        - _task_mean_features(eroot, "synthfam0", einfo["tasks"][1]))
    assert d_zero < 2.0            # noise floor only
    assert d_eff > 2.0 * d_zero    # the planted effect dominates


def test_zero_effect_bridges_are_identity(zero_bank):
    root, _ = zero_bank
    eye = np.eye(2, dtype=np.float32)
    for rec in _runs(root):
        for b in aio.load_bridges(rec["run_dir"], which="final").values():
            assert np.array_equal(b, eye)     # dev_mag == 0 exactly


def test_daux_planted_correlation(effect_bank):
    root, _ = effect_bank
    devs, gaps = [], []
    for rec in _runs(root, family="synthfam0"):
        eye = np.eye(2)
        bridges = aio.load_bridges(rec["run_dir"], which="final")
        devs.append(np.mean([np.linalg.norm(b - eye)
                             for b in bridges.values()]))
        _, train, val = aio.load_gap_trajectory(rec["run_dir"])
        gaps.append(val[-1] - train[-1])
    r = np.corrcoef(devs, gaps)[0, 1]
    assert r > 0.9, f"planted deviation<->gap correlation too weak: r={r}"


# ── Synth generator guards + determinism ────────────────────────────


def test_synth_refuses_real_bank_path(tmp_path):
    with pytest.raises(ValueError, match="asset1-bank"):
        synth.make_synthetic_bank(tmp_path / "asset1-bank" / "sub",
                                  n_families=1, n_tasks=1, n_reps=1)


def test_synth_refuses_foreign_nonempty_dir(tmp_path):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "precious.txt").write_text("do not clobber")
    with pytest.raises(ValueError, match="non-empty"):
        synth.make_synthetic_bank(target, n_families=1, n_tasks=1, n_reps=1)


def test_synth_regenerates_over_own_marker(tmp_path):
    target = tmp_path / "bank"
    synth.make_synthetic_bank(target, n_families=1, n_tasks=1, n_reps=1,
                              n_layers=1, d_model=8, rank=2)
    # Second invocation into the marked dir succeeds (fixture refresh)
    info = synth.make_synthetic_bank(target, n_families=1, n_tasks=1,
                                     n_reps=1, n_layers=1, d_model=8, rank=2)
    assert info["n_runs"] == 1


def test_synth_rank_divisibility_check(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        synth.make_synthetic_bank(tmp_path / "bad", rank=3, n_channels=2)


def test_synth_deterministic_across_invocations(tmp_path):
    kw = dict(n_families=1, n_tasks=2, n_reps=1, n_layers=1,
              d_model=8, rank=2, n_channels=2, task_effect=1.0, seed=11)
    synth.make_synthetic_bank(tmp_path / "a", **kw)
    synth.make_synthetic_bank(tmp_path / "b", **kw)
    for rec_a, rec_b in zip(_runs(tmp_path / "a"), _runs(tmp_path / "b")):
        va = aio.flatten_features(aio.load_adapter(rec_a["run_dir"]))
        vb = aio.flatten_features(aio.load_adapter(rec_b["run_dir"]))
        assert np.array_equal(va, vb)
        _, ta, vla = aio.load_gap_trajectory(rec_a["run_dir"])
        _, tb, vlb = aio.load_gap_trajectory(rec_b["run_dir"])
        assert np.array_equal(vla, vlb)
