"""Interlock tests for scripts/bm004_runner.py (F2 hard gate — A5-2).

Enforces the Director's condition A5-2 (ruling 2026-07-07,
docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md): the F2 positive-control
gate is a PRECONDITION of every BM-004 arm launch, not a post-hoc check.

  1. Missing gate artifact  -> launch refuses (SystemExit), and the
     refusal happens BEFORE any other validation (precedence proof).
  2. Recorded FAIL          -> refuses (outcome recorded either way,
     launching stays blocked).
  3. PASS with off-criteria fields (steps over cap, wrong model, wrong
     corpus kind)           -> refuses.
  4. passed-flag true but losses contradict it -> refuses (the verdict
     must RE-DERIVE from the recorded numbers; flags are never trusted).
  5. Genuine PASS           -> arm 4 (identity x generic) dry-run
     launches; launch record carries the gate verification.
  6. Arm 6b requires the dated amendment (pre-registered conditional).
  7. Transit arms refuse while the trainer lacks --transit-corpus;
     arm 6a refuses while the trainer lacks the shuffled_rd mode.
  8. Bank-tree paths are refused everywhere.
  9. evaluate_gate computes the verdict — callers cannot assert a pass.
 10. There is NO bypass parameter on require_f2_gate/launch.

CPU/stdlib only; no torch, no CUDA, no subprocess execution (dry runs).
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bm004_runner as br  # noqa: E402


def _passing_gate(tmp_path: Path, **over) -> Path:
    kw = dict(identity_loss=2.31, rd_graph_loss=2.27, max_steps=1000)
    kw.update({k: over.pop(k) for k in list(over)
               if k in ("identity_loss", "rd_graph_loss", "max_steps",
                        "model", "corpus_kind")})
    rec = br.evaluate_gate(kw.pop("identity_loss"),
                           kw.pop("rd_graph_loss"), **kw)
    rec.update(over)  # post-hoc field tampering for adversarial tests
    return br.write_gate_result(rec, results_root=tmp_path)


def _corpus(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / br.CORPUS_MANIFEST).write_text("{}", encoding="utf-8")
    return d


# ── 1-4: the interlock refuses ───────────────────────────────────────


def test_missing_gate_refuses(tmp_path):
    with pytest.raises(SystemExit, match="no F2 gate artifact"):
        br.require_f2_gate(results_root=tmp_path)


def test_missing_gate_refusal_precedes_all_other_validation(tmp_path):
    # Nonsense arm + no corpus + no amendment: the FIRST failure must
    # still be the gate — proving the interlock is a precondition.
    with pytest.raises(SystemExit, match="no F2 gate artifact"):
        br.launch("nonsense-arm", results_root=tmp_path,
                  out_root=tmp_path / "out")


def test_recorded_fail_blocks_launch(tmp_path):
    rec = br.evaluate_gate(2.31, 2.35, max_steps=1000)  # rd_graph worse
    assert rec["verdict"] == "FAIL"
    path = br.write_gate_result(rec, results_root=tmp_path)
    assert path.exists()  # outcome recorded either way
    with pytest.raises(SystemExit, match="REFUSING to launch"):
        br.require_f2_gate(results_root=tmp_path)


def test_steps_over_cap_refuses(tmp_path):
    _passing_gate(tmp_path, max_steps=2000)
    with pytest.raises(SystemExit, match="max_steps"):
        br.require_f2_gate(results_root=tmp_path)


def test_wrong_model_refuses(tmp_path):
    _passing_gate(tmp_path, model="Qwen/Qwen2.5-1.5B-Instruct")
    with pytest.raises(SystemExit, match="model"):
        br.require_f2_gate(results_root=tmp_path)


def test_wrong_corpus_kind_refuses(tmp_path):
    _passing_gate(tmp_path, corpus_kind="alpaca")
    with pytest.raises(SystemExit, match="corpus_kind"):
        br.require_f2_gate(results_root=tmp_path)


def test_tampered_pass_flag_refuses(tmp_path):
    # passed=True asserted over contradicting losses: must re-derive.
    _passing_gate(tmp_path, identity_loss=2.31, rd_graph_loss=2.35,
                  passed=True, verdict="PASS")
    with pytest.raises(SystemExit, match="re-deriv"):
        br.require_f2_gate(results_root=tmp_path)


def test_corrupt_artifact_refuses(tmp_path):
    p = br.gate_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="unreadable"):
        br.require_f2_gate(results_root=tmp_path)


# ── 5: genuine PASS lets a supported arm through ─────────────────────


def test_pass_allows_arm4_dry_run(tmp_path):
    _passing_gate(tmp_path)
    rec = br.launch("4", results_root=tmp_path,
                    out_root=tmp_path / "out", dry_run=True)
    assert rec["dry_run"] and rec["arm"] == "4"
    assert rec["gate_verified"]["verdict"] == "PASS"
    cmd = rec["command"]
    assert "--no-bridge-training" in cmd          # identity = exact LoRA
    assert cmd[cmd.index("--dataset") + 1] == "alpaca"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert cmd[cmd.index("--max-steps") + 1] == "4000"
    assert cmd[cmd.index("--batch-size") + 1] == "4"          # A1
    assert cmd[cmd.index("--gradient-accumulation") + 1] == "4"


def test_gate_subcommand_records_and_exits_nonzero_on_fail(tmp_path,
                                                           monkeypatch):
    monkeypatch.setattr(br, "RESULTS_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        br.main(["gate", "--identity-loss", "2.0",
                 "--rd-graph-loss", "2.5", "--max-steps", "1000"])
    assert json.loads(br.gate_path(tmp_path).read_text(
        encoding="utf-8"))["verdict"] == "FAIL"


# ── 6: the 6b conditional ────────────────────────────────────────────


def test_6b_requires_amendment(tmp_path):
    _passing_gate(tmp_path)
    with pytest.raises(SystemExit, match="amendment"):
        br.launch("6b", results_root=tmp_path, out_root=tmp_path / "out",
                  corpus_dir=_corpus(tmp_path))


# ── 7: refuse-don't-improvise trainer capability guards ─────────────


def test_transit_arm_refuses_until_trainer_wired(tmp_path):
    _passing_gate(tmp_path)
    if br.trainer_supports("transit_corpus"):
        pytest.skip("trainer transit wiring has landed")
    with pytest.raises(SystemExit, match="transit-corpus"):
        br.launch("1", results_root=tmp_path, out_root=tmp_path / "out",
                  corpus_dir=_corpus(tmp_path))


def test_transit_arm_requires_corpus_manifest(tmp_path):
    _passing_gate(tmp_path)
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(SystemExit,
                       match=f"{br.CORPUS_MANIFEST}|transit-corpus"):
        br.launch("1", results_root=tmp_path, out_root=tmp_path / "out",
                  corpus_dir=bare)


# ── 8: bank-tree guard ───────────────────────────────────────────────


def test_bank_path_refused_for_output(tmp_path):
    _passing_gate(tmp_path)
    bank_out = tmp_path / "results" / "asset1-bank" / "evil"
    with pytest.raises(SystemExit, match="asset1-bank"):
        br.launch("4", results_root=tmp_path, out_root=bank_out,
                  dry_run=True)


def test_bank_path_refused_for_gate_write(tmp_path):
    rec = br.evaluate_gate(2.3, 2.2, max_steps=1000)
    with pytest.raises(SystemExit, match="asset1-bank"):
        br.write_gate_result(rec,
                             results_root=tmp_path / "asset1-bank")


# ── 9-10: verdict computation + no bypass surface ────────────────────


def test_evaluate_gate_computes_verdict():
    assert br.evaluate_gate(2.31, 2.31, max_steps=1000)["passed"]
    assert not br.evaluate_gate(2.31, 2.3101, max_steps=1000)["passed"]
    assert not br.evaluate_gate(2.31, 2.0, max_steps=1001)["passed"]


def test_no_bypass_parameter_exists():
    for fn in (br.require_f2_gate, br.launch):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"allow_partial", "force", "skip_gate",
                             "allow_unggated", "bypass", "override"}, \
            f"{fn.__name__} grew a bypass-shaped parameter: {params}"


def test_run_table_matches_prereg_section6():
    assert set(br.ARMS) == {"1", "2", "3", "4", "5", "6a", "6b"}
    assert br.ARMS["1"] == {"mask": "rd_graph", "data": "matched",
                            "role": "thesis cell (core)"}
    assert br.ARMS["6a"]["mask"] == "shuffled_rd"      # hard fix F3
    assert "CONDITIONAL" in br.ARMS["6b"]["role"]
    assert br.PINNED_TRAIN["max_steps"] == 4000        # pinned default 2
    assert br.PINNED_TRAIN["batch_size"] == 4          # A1 geometry
    assert br.PINNED_TRAIN["gradient_accumulation"] == 4
    assert br.PINNED_TRAIN["seed"] == 42
