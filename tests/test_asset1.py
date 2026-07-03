"""Tests for the Asset 1 bank campaign infrastructure.

Covers: manifest generation (count / balance / round-robin ordering /
determinism), seed derivation, prompt templates, and the fixed-val /
train-pool split determinism. HF downloads are never touched — dataset
tests inject raw examples and a stub tokenizer.
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

import asset1_bank as bank  # noqa: E402
import asset1_datasets as a1d  # noqa: E402


# ── Manifest generation ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def full_manifest():
    spec = bank.make_campaign_spec(smoke=False)
    return spec, bank.generate_manifest(spec)


def test_manifest_count_480(full_manifest):
    _, runs = full_manifest
    assert len(runs) == 480


def test_manifest_balance(full_manifest):
    _, runs = full_manifest
    cells = {}
    for r in runs:
        cells.setdefault((r.family_short, r.task), []).append(r)
    assert len(cells) == 12
    for key, cell_runs in cells.items():
        assert len(cell_runs) == 40, f"cell {key} has {len(cell_runs)} runs"
        # replicates 0..39 exactly once each
        assert sorted(r.replicate for r in cell_runs) == list(range(40))


def test_manifest_round_robin_replicate_major(full_manifest):
    spec, runs = full_manifest
    n_cells = spec.n_cells
    assert n_cells == 12
    # First cycle covers each (family, task) cell exactly once
    first_cycle = [(r.family_short, r.task) for r in runs[:n_cells]]
    assert len(set(first_cycle)) == n_cells
    # Replicate-major: replicate == run_index // n_cells for every run
    for r in runs:
        assert r.replicate == r.run_index // n_cells
    # Cell order is identical in every cycle (balanced interruption property)
    for cycle_start in range(0, len(runs), n_cells):
        cycle = [(r.family_short, r.task)
                 for r in runs[cycle_start: cycle_start + n_cells]]
        assert cycle == first_cycle


def test_manifest_deterministic(full_manifest):
    spec, runs = full_manifest
    runs2 = bank.generate_manifest(bank.make_campaign_spec(smoke=False))
    assert [r.to_manifest_entry("PENDING") for r in runs] == \
           [r.to_manifest_entry("PENDING") for r in runs2]


def test_manifest_run_indices_sequential(full_manifest):
    _, runs = full_manifest
    assert [r.run_index for r in runs] == list(range(480))


def test_run_dir_layout(full_manifest):
    spec, runs = full_manifest
    r = runs[13]  # second cycle, second cell
    expected = spec.bank_root / r.family_short / r.task / f"run_{r.run_index:03d}"
    assert r.run_dir == expected


# ── Seed derivation ─────────────────────────────────────────────────


def test_seed_derivation(full_manifest):
    _, runs = full_manifest
    for r in runs:
        assert r.seed == 10_000 + r.run_index
        assert r.data_seed == 20_000 + r.run_index
    seeds = [r.seed for r in runs]
    data_seeds = [r.data_seed for r in runs]
    assert len(set(seeds)) == 480          # all training seeds distinct
    assert len(set(data_seeds)) == 480     # all data seeds distinct
    assert not set(seeds) & set(data_seeds)  # ranges never collide


# ── Smoke manifest ──────────────────────────────────────────────────


def test_smoke_manifest():
    spec = bank.make_campaign_spec(smoke=True)
    runs = bank.generate_manifest(spec)
    assert len(runs) == 4  # 2 families x (alpaca, squad) x 1 replicate
    assert {r.task for r in runs} == {"alpaca", "squad"}
    per_family = {}
    for r in runs:
        per_family.setdefault(r.family_short, set()).add(r.task)
    for fam, tasks in per_family.items():
        assert tasks == {"alpaca", "squad"}, fam
    assert spec.max_steps == 50
    assert spec.bank_root != bank.make_campaign_spec(smoke=False).bank_root
    # Smoke seeds follow the same derivation
    for r in runs:
        assert r.seed == 10_000 + r.run_index


# ── Prompt templates ────────────────────────────────────────────────


def test_prompt_templates_cover_all_tasks():
    assert set(a1d.TASK_TEMPLATE_KEYS) == set(a1d.TASK_REGISTRY) == set(bank.TASKS)
    for task, keys in a1d.TASK_TEMPLATE_KEYS.items():
        for key in keys:
            assert key in a1d.PROMPT_TEMPLATES, (task, key)


def test_prompt_templates_format_cleanly():
    fills = {
        "alpaca_with_input": dict(instruction="I", input="X", output="O"),
        "alpaca_no_input": dict(instruction="I", output="O"),
        "code_with_input": dict(instruction="I", input="X", output="O"),
        "code_no_input": dict(instruction="I", output="O"),
        "math": dict(question="Q", answer="A"),
        "xsum": dict(document="D", summary="S"),
        "squad": dict(context="C", question="Q", answer="A"),
        "agnews": dict(text="T", label="Sports"),
    }
    assert set(fills) == set(a1d.PROMPT_TEMPLATES)
    for key, kwargs in fills.items():
        text = a1d.PROMPT_TEMPLATES[key].format(**kwargs)
        assert "### Instruction:" in text
        assert "### Response:" in text
        assert "{" not in text and "}" not in text  # no unfilled fields


def test_summarization_template_content():
    text = a1d.PROMPT_TEMPLATES["xsum"].format(document="DOC", summary="SUM")
    assert "Summarize" in text
    assert text.index("DOC") < text.index("SUM")  # target last (in-window)


# ── Fixed-val / train-pool split ────────────────────────────────────


def test_split_ids_fixed_val_across_data_seeds():
    val1, train1 = a1d.split_ids(5000, data_seed=1, val_size=500,
                                 val_seed=777, pool_cap=40_000)
    val2, train2 = a1d.split_ids(5000, data_seed=2, val_size=500,
                                 val_seed=777, pool_cap=40_000)
    # Val split identical regardless of data_seed
    assert np.array_equal(val1, val2)
    assert len(val1) == 500
    # Pool composition identical, order differs
    assert set(train1.tolist()) == set(train2.tolist())
    assert not np.array_equal(train1, train2)
    # Val and pool are disjoint and cover the dataset
    assert not set(val1.tolist()) & set(train1.tolist())
    assert len(set(val1.tolist()) | set(train1.tolist())) == 5000


def test_split_ids_same_data_seed_is_deterministic():
    val1, train1 = a1d.split_ids(9000, data_seed=42)
    val2, train2 = a1d.split_ids(9000, data_seed=42)
    assert np.array_equal(val1, val2)
    assert np.array_equal(train1, train2)


def test_split_ids_pool_cap():
    val, train = a1d.split_ids(100_000, data_seed=3, val_size=500,
                               val_seed=777, pool_cap=40_000)
    assert len(train) == 40_000
    # Pool composition (canonical positions 500..40500) is data_seed-invariant
    _, train_b = a1d.split_ids(100_000, data_seed=4, val_size=500,
                               val_seed=777, pool_cap=40_000)
    assert set(train.tolist()) == set(train_b.tolist())


def test_split_ids_rejects_tiny_dataset():
    with pytest.raises(ValueError):
        a1d.split_ids(400, data_seed=0, val_size=500)


def test_ids_sha256_stable():
    h1 = a1d.ids_sha256([1, 2, 3])
    h2 = a1d.ids_sha256(np.array([1, 2, 3]))
    assert h1 == h2
    assert h1 != a1d.ids_sha256([3, 2, 1])


# ── Dataset class behaviour (stub tokenizer, injected raw examples) ──


class StubTokenizer:
    """Minimal tokenizer standing in for a HF tokenizer — no downloads."""

    pad_token = "<pad>"

    def __call__(self, text, truncation=True, max_length=32,
                 padding="max_length", return_tensors=None,
                 add_special_tokens=True):
        ids = [(hash(text) + i) % 1000 for i in range(max_length)]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, max_length, dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [1] * max_length}

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(ids)


def _raw_alpaca(n=60):
    return [{"instruction": f"instr-{k}", "input": "", "output": f"out-{k}"}
            for k in range(n)]


def test_dataset_fixed_val_identical_across_data_seeds():
    tok = StubTokenizer()
    ds_a = a1d.AlpacaTaskDataset(tok, "val", data_seed=101, max_len=32,
                                 val_size=10, pool_cap=100, raw=_raw_alpaca())
    ds_b = a1d.AlpacaTaskDataset(tok, "val", data_seed=202, max_len=32,
                                 val_size=10, pool_cap=100, raw=_raw_alpaca())
    assert ds_a.example_ids == ds_b.example_ids  # SAME val ids, any data_seed
    assert len(ds_a) == 10
    assert a1d.ids_sha256(ds_a.example_ids) == a1d.ids_sha256(ds_b.example_ids)


def test_dataset_train_pool_shuffled_by_data_seed():
    tok = StubTokenizer()
    ds_a = a1d.AlpacaTaskDataset(tok, "train", data_seed=101, max_len=32,
                                 val_size=10, pool_cap=100, raw=_raw_alpaca())
    ds_b = a1d.AlpacaTaskDataset(tok, "train", data_seed=202, max_len=32,
                                 val_size=10, pool_cap=100, raw=_raw_alpaca())
    ds_a2 = a1d.AlpacaTaskDataset(tok, "train", data_seed=101, max_len=32,
                                  val_size=10, pool_cap=100, raw=_raw_alpaca())
    # Same composition, different order across data seeds
    assert set(ds_a.example_ids) == set(ds_b.example_ids)
    assert ds_a.example_ids != ds_b.example_ids
    # Same data_seed -> identical order (determinism)
    assert ds_a.example_ids == ds_a2.example_ids
    # Train pool disjoint from val
    val_ids = set(int(i) for i in ds_a.val_ids)
    assert not val_ids & set(ds_a.example_ids)


def test_dataset_example_tensors_follow_pilot_convention():
    tok = StubTokenizer()
    ds = a1d.AlpacaTaskDataset(tok, "val", data_seed=0, max_len=32,
                               val_size=5, pool_cap=50, raw=_raw_alpaca(20))
    ex = ds[0]
    assert set(ex) == {"input_ids", "attention_mask", "labels"}
    assert ex["input_ids"].shape == (32,)
    # Pilot convention: labels are a clone of input_ids (full-sequence loss)
    assert torch.equal(ex["labels"], ex["input_ids"])
    assert ex["labels"].data_ptr() != ex["input_ids"].data_ptr()


def test_dataset_rejects_bad_split_pool():
    with pytest.raises(ValueError):
        a1d.AlpacaTaskDataset(StubTokenizer(), "test", data_seed=0,
                              max_len=32, val_size=5, pool_cap=50,
                              raw=_raw_alpaca(20))


def test_build_dataset_rejects_unknown_task():
    with pytest.raises(ValueError):
        a1d.build_dataset("poetry", StubTokenizer(), "train", 0)


def test_agnews_label_fallback_names():
    raw = [{"text": f"t{k}", "label": k % 4} for k in range(40)]
    ds = a1d.AGNewsTaskDataset(StubTokenizer(), "val", data_seed=0,
                               max_len=32, val_size=5, pool_cap=30,
                               raw=raw, keep_text=True)
    assert ds.label_names == ["World", "Sports", "Business", "Sci/Tech"]
    assert any(name in ds.formatted_texts[0] for name in ds.label_names)


# ── Runner utilities ────────────────────────────────────────────────


def test_blocked_error_classification():
    class GatedRepoError(Exception):
        pass

    assert bank.is_blocked_error(
        GatedRepoError("403 Client Error. Cannot access gated repo"))
    assert bank.is_blocked_error(
        OSError("401 Unauthorized for url https://huggingface.co/..."))
    assert not bank.is_blocked_error(
        RuntimeError("CUDA out of memory"))
    assert not bank.is_blocked_error(
        ConnectionError("Connection reset by peer"))


def test_status_scan_and_markers(tmp_path):
    spec = bank.make_campaign_spec(smoke=True)
    runs = bank.generate_manifest(spec)
    run = runs[0]
    # Redirect the run dir into tmp for the scan test
    run = bank.RunSpec(**{**run.__dict__, "run_dir": tmp_path / "run_000"})
    assert bank.scan_run_status(run) == "PENDING"
    run.run_dir.mkdir(parents=True)
    (run.run_dir / "FAILED").write_text("x")
    assert bank.scan_run_status(run) == "FAILED"
    (run.run_dir / "COMPLETE").write_text("x")
    assert bank.scan_run_status(run) == "COMPLETE"  # COMPLETE wins


def test_atomic_write_json(tmp_path):
    p = tmp_path / "m.json"
    bank.atomic_write_json(p, {"a": 1})
    import json
    assert json.loads(p.read_text()) == {"a": 1}
    assert not (tmp_path / "m.json.tmp").exists()
