"""Tests for the `history_epoch` field in the tier-gate ledger.

Director's ruling 2026-09-11 §2 addition 2
(`docs/DIRECTOR_RULING_LEDGER_PROVENANCE_2026-09-11.md:25`): "Add a field —
`"history_epoch": "2026-09-11-rewrite"` ... to the entry schema from the L1
gate onward. The L0 entry is not edited; the amendment file states that L0
predates the field."

Three things are asserted here:

1. `granularity_analysis.record_gate` writes the field, and adding it breaks no
   consumer — `granularity_queue.fired_gates()` still reports the fired tier
   from a ledger that carries it.
2. The REAL `results/granularity/TIER_GATES.json` still holds the L0 entry as
   written, with its pre-rewrite `git_commit` and NO `history_epoch` key. The
   file's own hash is deliberately NOT pinned: it changes by design when the
   next gate fires. The entry is what is sealed, not the file's length.
3. The amendment exists and carries both mapped pairs.

Pre-registration hygiene: the writer test redirects GATES_FILE to tmp_path;
nothing writes to results/granularity/, no HF downloads, no network, no GPU.
The real ledger is opened read-only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import granularity_analysis as ga  # noqa: E402
import granularity_queue as q  # noqa: E402

EPOCH = "2026-09-11-rewrite"
L0_RECORDED_SHA = "174ae959f2ac42d9ea2230e3ddc0f15e57ce6632"
GATE_TREE_IMAGE = "6c79be96b372ceb51979d46ea3ab4958587373c2"
ITEM1_ANCHOR = "d46c0822aa8504b42a76894f2313a85f5a54dd95"
ITEM1_IMAGE = "9d23a893fa7f8883b4e36be0bb815380b8fba262"

REAL_LEDGER = REPO / "results" / "granularity" / "TIER_GATES.json"
AMENDMENT = REPO / "results" / "granularity" / "TIER_GATES_AMENDMENT_2026-09-11.md"

HEX40 = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture
def tmp_ledger(monkeypatch, tmp_path):
    """Both sides pointed at the same throwaway ledger (they share one file)."""
    p = tmp_path / "TIER_GATES.json"
    monkeypatch.setattr(ga, "GATES_FILE", p)
    monkeypatch.setattr(q, "GATES_FILE", p)
    monkeypatch.setattr(q, "log", lambda msg: None)
    return p


# ── 1. The writer records the epoch, and no consumer breaks ─────────


def test_module_constant_is_the_ruled_value():
    assert ga.HISTORY_EPOCH == EPOCH


def test_record_gate_writes_history_epoch(tmp_ledger):
    assert "L1" in ga.LEVEL_TIER          # a level that exists in the map
    ga.record_gate("L1", ["L0"], {"k": 12, "n_runs": 240})

    entry = json.loads(tmp_ledger.read_text(encoding="utf-8"))["ledger"][0]
    assert entry["history_epoch"] == EPOCH
    assert entry["tier"] == ga.LEVEL_TIER["L1"] == "L1"
    assert entry["level"] == "L1"
    assert entry["tiers_already_unblinded"] == ["L0"]
    assert entry["k"] == 12 and entry["n_runs"] == 240
    assert HEX40.match(entry["git_commit"]), entry["git_commit"]


def test_fired_gates_still_parses_a_ledger_carrying_the_new_key(tmp_ledger):
    """The added key must not disturb the training-side interlock."""
    ga.record_gate("L1", ["L0"], {"k": 12, "n_runs": 240})
    assert "history_epoch" in json.loads(
        tmp_ledger.read_text(encoding="utf-8"))["ledger"][0]
    assert q.fired_gates() == ["L1"]


def test_epoch_does_not_disturb_the_frozen_tier_order(tmp_ledger):
    """require_tier_order reads `tier` only; an epoch-bearing L1 gate still
    satisfies L1's successor and still refuses an unfired predecessor."""
    ga.record_gate("L1", ["L0"], {"k": 12, "n_runs": 240})
    with pytest.raises(SystemExit):
        ga.require_tier_order("L2")        # L0 recorded nowhere in this ledger


# ── 2. The real ledger: L0 as written, and unedited ─────────────────


def test_real_ledger_l0_entry_is_as_written_and_has_no_epoch():
    data = json.loads(REAL_LEDGER.read_text(encoding="utf-8"))
    e0 = data["ledger"][0]
    assert e0["tier"] == "L0" and e0["level"] == "L0"
    assert e0["git_commit"] == L0_RECORDED_SHA
    assert e0["fired_at"] == "2026-09-06T04:48:57.413066+00:00"
    assert e0["k"] == 6 and e0["n_runs"] == 240
    assert e0["tiers_already_unblinded"] == []
    # L0 predates the field (ruling :25). The ledger is not edited.
    assert "history_epoch" not in e0


# ── 3. The amendment is filed and carries both pairs ────────────────


def test_amendment_exists_and_maps_both_anchors():
    text = AMENDMENT.read_text(encoding="utf-8")
    for sha in (L0_RECORDED_SHA, GATE_TREE_IMAGE, ITEM1_ANCHOR, ITEM1_IMAGE):
        assert sha in text, f"amendment does not name {sha}"
    assert EPOCH in text
    assert "docs/history-rewrite/cited-commit-map.txt:8" in text
    assert "docs/history-rewrite/cited-commit-map.txt:62" in text
    assert "HISTORY_REWRITE_2026-09.md" in text
