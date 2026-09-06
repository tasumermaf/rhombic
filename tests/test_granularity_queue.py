"""Tests for the training-side tier-order interlock in scripts/granularity_queue.py.

Director review 2026-09-06, Item 3: "a unit test that asserts the three-triple
above, plus L3 -> [L1, ARMB, L2], belongs in the suite before the next tier is
enqueued." The ledger fixtures use the ANALYSIS side's own format —
{"ledger": [{"tier": ..., "level": ..., ...}, ...], "note": ...} — because that
is the file granularity_analysis.record_gate writes and the queue reads.

Pre-registration hygiene: GATES_FILE is redirected to tmp_path; nothing reads
or writes results/granularity/, no HF downloads, no network, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import granularity_queue as q  # noqa: E402

FROZEN_ORDER = ["L0", "L1", "ARMB", "L2", "L3", "D7"]
ARMB_RUNGS = ("B2", "B4", "B8", "B16")


def _entry(tier: str, level: str | None = None) -> dict:
    """One ledger entry in the analysis side's record_gate shape."""
    return {"tier": tier, "level": level or tier,
            "fired_at": "2026-09-06T00:00:00+00:00",
            "tiers_already_unblinded": [], "k": 6, "n_runs": 240,
            "git_commit": "0" * 40}


def _write_ledger(path: Path, tiers: list[str]) -> None:
    path.write_text(json.dumps({"ledger": [_entry(t) for t in tiers],
                                "note": "test ledger"}, indent=1),
                    encoding="utf-8")


@pytest.fixture
def gates_file(monkeypatch, tmp_path):
    p = tmp_path / "TIER_GATES.json"
    monkeypatch.setattr(q, "GATES_FILE", p)
    monkeypatch.setattr(q, "log", lambda msg: None)   # keep the real queue log untouched
    return p


# ── The frozen order itself ─────────────────────────────────────────


def test_frozen_order_is_the_registered_one():
    assert list(q.TIER_ORDER) == FROZEN_ORDER
    assert tuple(q.ARMB_RUNGS) == ARMB_RUNGS


def test_tier_of_maps_rungs_to_armb_and_levels_to_themselves():
    for rung in ARMB_RUNGS:
        assert q.tier_of(rung) == "ARMB"
    for level in ("L0", "L1", "L2", "L3", "D7"):
        assert q.tier_of(level) == level


# ── The Director's triple, plus L3 (Item 3, 2026-09-06) ─────────────


def test_interlock_with_no_ledger(gates_file):
    assert not gates_file.exists()
    assert q.fired_gates() == []
    assert q.training_interlock("L0") == []            # analysis-side; never a training requirement
    assert q.training_interlock("L1") == []            # L0 gates L1's ANALYSIS, not its training
    for rung in ARMB_RUNGS:
        assert q.training_interlock(rung) == ["L1"]
    assert q.training_interlock("L2") == ["L1", "ARMB"]
    assert q.training_interlock("L3") == ["L1", "ARMB", "L2"]
    assert q.training_interlock("D7") == ["L1", "ARMB", "L2", "L3"]


def test_l0_gate_alone_changes_nothing_on_the_training_side(gates_file):
    _write_ledger(gates_file, ["L0"])
    assert q.fired_gates() == ["L0"]
    assert q.training_interlock("L1") == []
    for rung in ARMB_RUNGS:
        assert q.training_interlock(rung) == ["L1"]
    assert q.training_interlock("L2") == ["L1", "ARMB"]
    assert q.training_interlock("L3") == ["L1", "ARMB", "L2"]


def test_l1_gate_clears_armb_and_moves_l2_to_armb_only(gates_file):
    _write_ledger(gates_file, ["L0", "L1"])
    assert q.fired_gates() == ["L0", "L1"]
    for rung in ARMB_RUNGS:
        assert q.training_interlock(rung) == []
    assert q.training_interlock("L2") == ["ARMB"]
    assert q.training_interlock("L3") == ["ARMB", "L2"]


def test_armb_gate_clears_l2(gates_file):
    _write_ledger(gates_file, ["L0", "L1", "ARMB"])
    assert q.training_interlock("L2") == []
    assert q.training_interlock("L3") == ["L2"]
    assert q.training_interlock("D7") == ["L2", "L3"]


# ── fired_gates() parses what record_gate writes ────────────────────


def test_fired_gates_reads_the_analysis_side_format_in_firing_order(gates_file):
    _write_ledger(gates_file, ["L0", "L1", "ARMB", "L2"])
    assert q.fired_gates() == ["L0", "L1", "ARMB", "L2"]


def test_fired_gates_counts_a_tier_once_across_its_rungs(gates_file):
    # The analysis side records one entry per LEVEL: the four ARMB rungs share a tier.
    gates_file.write_text(json.dumps({"ledger": [
        _entry("L0"), _entry("L1"),
        _entry("ARMB", "B2"), _entry("ARMB", "B4"),
        _entry("ARMB", "B8"), _entry("ARMB", "B16")]}), encoding="utf-8")
    assert q.fired_gates() == ["L0", "L1", "ARMB"]


def test_fired_gates_never_returns_top_level_keys(gates_file):
    """The defect found 2026-09-06: the dict branch returned ['ledger', 'note']."""
    _write_ledger(gates_file, ["L0"])
    fired = q.fired_gates()
    assert "ledger" not in fired
    assert "note" not in fired
    assert fired == ["L0"]


def test_unreadable_ledger_is_no_gates_not_an_exception(gates_file):
    gates_file.write_text("{not json", encoding="utf-8")
    assert q.fired_gates() == []
    for rung in ARMB_RUNGS:
        assert q.training_interlock(rung) == ["L1"]


def test_legacy_ledger_forms_still_accepted(gates_file):
    gates_file.write_text(json.dumps(["L0", "L1"]), encoding="utf-8")
    assert q.fired_gates() == ["L0", "L1"]
    gates_file.write_text(json.dumps({"L0": True, "L1": False}), encoding="utf-8")
    assert q.fired_gates() == ["L0"]


def test_ledger_entries_without_a_tier_are_ignored(gates_file):
    gates_file.write_text(json.dumps({"ledger": [
        _entry("L0"), {"level": "L1"}, "junk", {"tier": None}]}), encoding="utf-8")
    assert q.fired_gates() == ["L0"]
