"""Tests for scripts/xr001_externalization_pilot.py (XR-001 pilot).

Enforces the pre-registered protocol
(results/XR-001-externalization-pilot/PROTOCOL.md) at three layers:

  generate   determinism (byte-identical re-run), and every §2 construction
             constraint — value uniqueness, designed +-1 pairs ONLY, distinct
             stated sums, questions referencing real entities, conflation-set
             correctness, distinct multi-hop truths, segment word-count band.
  score      one case per §5 corruption class incl. the off_by_one-over-
             conflation precedence, comma tolerance, omission on garbage;
             exploratory compaction-stage scoring for R1 prose and R2 blocks.
  run        cascade mechanics under a MOCK transport (state k carries only
             state k-1 + segment k, never raw segment k-1), think-flag routing,
             manifest resume (COMPLETE cells are never re-called), and the
             FAILED path on a raised transport error.
  analyze    Wilson CI sanity + a hand-checked exact McNemar value.

Stdlib/CPU only; no Ollama, no network — the transport is always injected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import xr001_externalization_pilot as xr  # noqa: E402


# ── Mock transport ───────────────────────────────────────────────────


class MockTransport:
    """Records every call; returns a recognizable state / answer.

    Compaction calls (num_predict == COMPACT_NUM_PREDICT) return a unique
    ``STATEMARK<n>`` marker so cascade wiring can be inspected; answer calls
    return a fixed parseable ``ANSWER: 42``.
    """

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    def __call__(self, messages, *, model, num_predict, think, base_url):
        self.calls.append({"model": model, "num_predict": num_predict,
                           "think": think, "base_url": base_url,
                           "messages": messages})
        if self.fail:
            raise xr.TransportError("mock forced failure")
        if num_predict == xr.COMPACT_NUM_PREDICT:
            content = (f"=== STATE ===\nvalues:\n  STATEMARK{len(self.calls)} "
                       f"= 1\n=== END STATE ===")
        else:
            content = "The value is clear.\nANSWER: 42"
        return {"content": content, "prompt_eval_count": 100,
                "eval_count": 50}


@pytest.fixture()
def tasks_file(tmp_path):
    path = tmp_path / "tasks.json"
    xr.generate(path, episodes=2, base_seed=xr.BASE_SEED)
    return path


# ── generate: determinism ────────────────────────────────────────────


def test_generate_byte_identical(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    xr.generate(a, episodes=6, base_seed=xr.BASE_SEED)
    xr.generate(b, episodes=6, base_seed=xr.BASE_SEED)
    assert a.read_bytes() == b.read_bytes()


def test_generate_no_timestamps(tmp_path):
    p = tmp_path / "t.json"
    xr.generate(p, episodes=3)
    txt = p.read_text(encoding="utf-8").lower()
    for token in ("timestamp", "generated_at", "created_at", "date"):
        assert token not in txt


def test_different_seed_differs(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    xr.generate(a, episodes=3, base_seed=77001)
    xr.generate(b, episodes=3, base_seed=88001)
    assert a.read_bytes() != b.read_bytes()


# ── generate: §2 construction constraints ────────────────────────────


@pytest.fixture(scope="module")
def episodes():
    return [xr.build_episode(i, xr.BASE_SEED) for i in range(15)]


def test_value_uniqueness_and_range(episodes):
    for ep in episodes:
        vals = [e["value"] for seg in ep["segments"]
                for e in xr._iter_entities(seg)]
        assert len(vals) == 20 and len(set(vals)) == 20
        assert all(xr.VALUE_LO <= v <= xr.VALUE_HI for v in vals)


def test_name_uniqueness_and_pattern(episodes):
    for ep in episodes:
        names = [e["name"] for seg in ep["segments"]
                 for e in xr._iter_entities(seg)]
        assert len(names) == 20 and len(set(names)) == 20
        for nm in names:
            assert len(nm) == 6 and nm.isupper()
            # strict CV alternation
            for i, ch in enumerate(nm):
                assert (ch in xr._CONS) if i % 2 == 0 else (ch in xr._VOW)


def test_only_designed_plus_minus_one_pairs(episodes):
    for ep in episodes:
        designed = set()
        for seg in ep["segments"]:
            b = seg["entities"]["base"]["value"]
            c = seg["entities"]["neighbor"]["value"]
            assert c == b + 1
            designed.add((b, c))
        vals = sorted(e["value"] for seg in ep["segments"]
                      for e in xr._iter_entities(seg))
        for x, y in zip(vals, vals[1:]):
            if y - x == 1:
                assert (x, y) in designed, f"undesigned adjacency {x},{y}"


def test_stated_sums_distinct_from_values_and_each_other(episodes):
    for ep in episodes:
        sums = [seg["combined"]["value"] for seg in ep["segments"]]
        vals = {e["value"] for seg in ep["segments"]
                for e in xr._iter_entities(seg)}
        assert len(set(sums)) == 4
        assert not (set(sums) & vals)


def test_multihop_truths_distinct_from_everything(episodes):
    for ep in episodes:
        vals = {e["value"] for seg in ep["segments"]
                for e in xr._iter_entities(seg)}
        sums = {seg["combined"]["value"] for seg in ep["segments"]}
        mh = [q["truth"] for q in ep["questions"] if q["type"] == "multihop"]
        assert len(mh) == 2 and len(set(mh)) == 2
        assert not (set(mh) & vals) and not (set(mh) & sums)


def test_conflation_set_correctness(episodes):
    for ep in episodes:
        vals = {e["value"] for seg in ep["segments"]
                for e in xr._iter_entities(seg)}
        sums = {seg["combined"]["value"] for seg in ep["segments"]}
        assert set(ep["conflation_set"]) == vals | sums
        assert len(ep["conflation_set"]) == 24
        assert ep["conflation_set"] == sorted(ep["conflation_set"])


def test_questions_reference_existing_entities(episodes):
    for ep in episodes:
        names = {e["name"] for seg in ep["segments"]
                 for e in xr._iter_entities(seg)}
        assert len(ep["questions"]) == 8
        for q in ep["questions"]:
            refs = q.get("entities") or ([q["entity"]] if "entity" in q
                                         else [])
            assert refs and all(n in names for n in refs)


def test_question_truths_match_construction(episodes):
    for ep in episodes:
        by = {e["name"]: e["value"] for seg in ep["segments"]
              for e in xr._iter_entities(seg)}
        for q in ep["questions"]:
            if q["type"] in ("direct", "conflation"):
                assert q["truth"] == by[q["entity"]]
            elif q["type"] == "combined":
                assert q["truth"] == sum(by[n] for n in q["entities"])
            else:
                assert q["truth"] == sum(by[n] for n in q["entities"])


def test_segment_word_counts_in_band(episodes):
    for ep in episodes:
        for seg in ep["segments"]:
            assert 120 <= len(seg["text"].split()) <= 180


def test_segment_text_embeds_facts_unambiguously(episodes):
    ep = episodes[0]
    for seg in ep["segments"]:
        e = seg["entities"]
        text = seg["text"]
        assert f"{e['base']['name']}" in text and str(e["base"]["value"]) in text
        assert str(e["partner"]["value"]) in text
        assert str(e["neighbor"]["value"]) in text  # base+1 present -> the trap
        assert str(seg["combined"]["value"]) in text


# ── score: §5 corruption classes ─────────────────────────────────────

CSET = {499, 500, 700, 900}


def test_classify_correct():
    assert xr.classify(500, 500, CSET) == "correct"


def test_classify_off_by_one():
    assert xr.classify(501, 500, CSET) == "off_by_one"
    assert xr.classify(499, 500, {700}) == "off_by_one"


def test_classify_off_by_one_precedes_conflation():
    # 499 is BOTH truth-1 and a member of the conflation set: off_by_one wins.
    assert 499 in CSET
    assert xr.classify(499, 500, CSET) == "off_by_one"


def test_classify_conflation():
    assert xr.classify(700, 500, CSET) == "conflation"


def test_classify_other_wrong():
    assert xr.classify(123, 500, CSET) == "other_wrong"


def test_classify_omission():
    assert xr.classify(None, 500, CSET) == "omission"


def test_parse_answer_comma_and_last_match():
    assert xr.parse_answer("junk\nANSWER: 1,234") == 1234
    assert xr.parse_answer("ANSWER: 1\nmore\nANSWER: 5") == 5
    assert xr.parse_answer("ANSWER: -3") == -3
    assert xr.parse_answer("answer: 7 (lowercase tag)") == 7


def test_parse_answer_omission_on_garbage():
    assert xr.parse_answer("no integer answer at all") is None
    assert xr.parse_answer("") is None
    assert xr.parse_answer(None) is None
    assert xr.classify(xr.parse_answer("garbage"), 500, CSET) == "omission"


# ── score: exploratory compaction-stage scoring ──────────────────────


def test_score_state_r2_pairs():
    emap = {"DALPOR": 412, "MIVEK": 319}
    state = ("=== STATE ===\nvalues:\n  DALPOR = 412\n  MIVEK = 999\n"
             "combined:\n  DALPOR + MIVEK = 731\n=== END STATE ===")
    res = xr._score_state_r2(state, emap)
    assert res["retained"] == 2 and res["correct"] == 1 and res["wrong"] == 1
    # the combined line must NOT be misread as a values pair
    assert res["correct"] + res["wrong"] == 2


def test_score_state_r1_nearest_number():
    emap = {"DALPOR": 412, "MIVEK": 319}
    state = "DALPOR is 412 and separately MIVEK is 319 in the ledger."
    res = xr._score_state_r1(state, emap)
    assert res["retained"] == 2 and res["correct"] == 2 and res["wrong"] == 0


def _fabricate_manifest(outdir, tasks_path, cells, *, models, regimes,
                        n_episodes):
    """Write a minimal manifest for fabricated-raw score() tests.

    ``cells`` is an iterable of (model, regime, episode, status).
    """
    man = {"tasks": str(tasks_path),
           "tasks_sha256": xr._tasks_sha256(tasks_path),
           "models": models, "regimes": regimes, "n_episodes": n_episodes,
           "cells": {xr._cell_key(m, r, e): {
               "model": m, "regime": r, "episode": e, "status": st}
               for (m, r, e, st) in cells}}
    (Path(outdir) / "manifest.json").write_text(json.dumps(man),
                                                encoding="utf-8")


def test_score_integration(tasks_file, tmp_path):
    # Fabricate a raw log: R0 answers for one model, all "ANSWER: 42".
    outdir = tmp_path
    rawdir = outdir / "raw"
    rawdir.mkdir()
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    lines = []
    for ep in tasks["episodes"]:
        for q in ep["questions"]:
            lines.append(json.dumps({
                "model": "m", "regime": "R0", "episode": ep["episode"],
                "kind": f"question_{q['id']}", "question_id": q["id"],
                "response": "ANSWER: 42"}))
    (rawdir / "m_R0.jsonl").write_text("\n".join(lines), encoding="utf-8")
    _fabricate_manifest(outdir, tasks_file,
                        [("m", "R0", 0, "COMPLETE"),
                         ("m", "R0", 1, "COMPLETE")],
                        models=["m"], regimes=["R0"], n_episodes=2)
    out = xr.score(tasks_file, outdir)
    assert len(out["probes"]) == 2 * 8
    assert (outdir / "results.json").exists()
    assert all(p["answer"] == 42 for p in out["probes"])
    assert out["cells"] == {"complete": 2, "not_complete": 0, "expected": 2,
                            "expected_probes_per_cell": 16}


def _compact_line(model, regime, ep, seg, ec=50):
    return json.dumps({"model": model, "regime": regime, "episode": ep,
                       "kind": f"compact_{seg}", "segment": seg,
                       "response": "=== STATE ===\nvalues:\n=== END STATE ===",
                       "eval_count": ec})


def _question_line(model, regime, ep, qid, answer="ANSWER: 42"):
    return json.dumps({"model": model, "regime": regime, "episode": ep,
                       "kind": f"question_{qid}", "question_id": qid,
                       "response": answer, "eval_count": 20})


def test_score_excludes_failed_and_partial_cells(tasks_file, tmp_path):
    # ep0 COMPLETE; ep1 FAILED mid-cascade. FAILED records must contribute
    # nothing: no probes, no stage entries, no token-mean contamination.
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    lines = [_compact_line("m", "R1", 0, s) for s in range(1, 5)]
    lines += [_question_line("m", "R1", 0, q) for q in range(1, 9)]
    lines += [_compact_line("m", "R1", 1, s) for s in range(1, 3)]  # partial
    (rawdir / "m_R1.jsonl").write_text("\n".join(lines), encoding="utf-8")
    _fabricate_manifest(tmp_path, tasks_file,
                        [("m", "R1", 0, "COMPLETE"),
                         ("m", "R1", 1, "FAILED")],
                        models=["m"], regimes=["R1"], n_episodes=2)
    out = xr.score(tasks_file, tmp_path)
    assert {p["episode"] for p in out["probes"]} == {0}
    assert out["compaction_tokens"]["R1"]["n"] == 4          # ep0 only
    assert [s["episode"] for s in out["compaction_stage"]] == [0]
    assert out["cells"]["not_complete"] == 1


def test_score_dedups_retried_compactions(tasks_file, tmp_path):
    # A failed-then-resumed cell logs compact_1/compact_2 twice; the token
    # means must count each segment once (last record wins).
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    lines = [_compact_line("m", "R1", 0, 1, ec=999),   # stale failed attempt
             _compact_line("m", "R1", 0, 2, ec=999)]
    lines += [_compact_line("m", "R1", 0, s, ec=50) for s in range(1, 5)]
    lines += [_question_line("m", "R1", 0, q) for q in range(1, 9)]
    (rawdir / "m_R1.jsonl").write_text("\n".join(lines), encoding="utf-8")
    _fabricate_manifest(tmp_path, tasks_file, [("m", "R1", 0, "COMPLETE")],
                        models=["m"], regimes=["R1"], n_episodes=1)
    out = xr.score(tasks_file, tmp_path)
    tok = out["compaction_tokens"]["R1"]
    assert tok["n"] == 4 and tok["mean"] == 50


def test_score_requires_manifest(tasks_file, tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(SystemExit):
        xr.score(tasks_file, tmp_path)


# ── analyze: Wilson + McNemar ────────────────────────────────────────


def test_wilson_ci_sanity():
    lo, hi = xr.wilson_ci(5, 10)
    assert 0 < lo < 0.5 < hi < 1
    lo0, hi0 = xr.wilson_ci(0, 10)
    assert lo0 == pytest.approx(0.0, abs=1e-9) and 0 < hi0 < 1
    lo1, hi1 = xr.wilson_ci(10, 10)
    assert 0 < lo1 < 1 and hi1 == pytest.approx(1.0, abs=1e-9)
    assert xr.wilson_ci(0, 0) == (0.0, 0.0)


def test_mcnemar_exact_hand_values():
    # b=8, c=1: n=9, k=1, tail=(C(9,0)+C(9,1))*2^-9=10/512, p=20/512.
    assert xr.mcnemar_exact(8, 1) == pytest.approx(20 / 512)
    # b=10, c=0: n=10, k=0, tail=1/1024, p=2/1024.
    assert xr.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert xr.mcnemar_exact(0, 0) == 1.0
    assert xr.mcnemar_exact(3, 3) == 1.0        # symmetric -> capped at 1


# ── run: cascade, routing, resume, failure ───────────────────────────


def test_r1_cascade_carries_state_not_raw_segments(tasks_file, tmp_path):
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    seg_texts = [s["text"] for s in tasks["episodes"][0]["segments"]]
    mock = MockTransport()
    xr.run(tasks_file, ["gemma3:4b"], ["R1"], tmp_path,
           base_url="http://x", episodes_limit=1, transport=mock,
           retry_backoff=0, max_retries=1, progress=False)

    compaction = [c for c in mock.calls
                  if c["num_predict"] == xr.COMPACT_NUM_PREDICT]
    answers = [c for c in mock.calls
               if c["num_predict"] == xr.ANSWER_NUM_PREDICT]
    assert len(compaction) == 4 and len(answers) == 8  # 4 segs + 8 questions

    # Segment 1 compaction sees "none" and raw segment 1.
    assert "none (this is the first segment)" in compaction[0]["messages"][-1][
        "content"]
    # For k>=2: prompt carries state k-1 (STATEMARK from prior call) and the
    # raw segment k, but NOT the raw text of any earlier segment.
    for k in range(1, 4):
        user = compaction[k]["messages"][-1]["content"]
        assert f"STATEMARK{k}" in user            # state k-1 carried forward
        assert seg_texts[k] in user               # this segment's raw text
        for earlier in range(k):
            assert seg_texts[earlier] not in user  # earlier raw text is gone

    # Answer calls see ONLY the final state, not raw segments.
    for a in answers:
        user = a["messages"][-1]["content"]
        assert "STATEMARK4" in user
        for t in seg_texts:
            assert t not in user


def test_r2_cell_call_shape(tasks_file, tmp_path):
    mock = MockTransport()
    xr.run(tasks_file, ["gemma3:4b"], ["R2"], tmp_path,
           base_url="http://x", episodes_limit=1, transport=mock,
           retry_backoff=0, max_retries=1, progress=False)
    assert len(mock.calls) == 12                  # 4 compaction + 8 answers
    # R2 compaction prompt must specify the typed block format.
    first = mock.calls[0]["messages"][-1]["content"]
    assert "=== STATE ===" in first and "relations:" in first


def test_r0_uses_full_transcript(tasks_file, tmp_path):
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    seg_texts = [s["text"] for s in tasks["episodes"][0]["segments"]]
    mock = MockTransport()
    xr.run(tasks_file, ["gemma3:4b"], ["R0"], tmp_path,
           base_url="http://x", episodes_limit=1, transport=mock,
           retry_backoff=0, max_retries=1, progress=False)
    assert len(mock.calls) == 8                   # one per question, no compaction
    user = mock.calls[0]["messages"][-1]["content"]
    for t in seg_texts:
        assert t in user                          # ceiling: full transcript


def test_think_flag_routing(tasks_file, tmp_path):
    m_qwen = MockTransport()
    xr.run(tasks_file, ["qwen3:14b"], ["R0"], tmp_path / "q",
           base_url="http://x", episodes_limit=1, transport=m_qwen,
           retry_backoff=0, max_retries=1, progress=False)
    assert all(c["think"] is False for c in m_qwen.calls)

    m_other = MockTransport()
    xr.run(tasks_file, ["gemma3:4b"], ["R0"], tmp_path / "g",
           base_url="http://x", episodes_limit=1, transport=m_other,
           retry_backoff=0, max_retries=1, progress=False)
    assert all(c["think"] is None for c in m_other.calls)


def test_manifest_resume_skips_complete(tasks_file, tmp_path):
    mock1 = MockTransport()
    xr.run(tasks_file, ["gemma3:4b"], ["R0"], tmp_path,
           base_url="http://x", transport=mock1, retry_backoff=0,
           max_retries=1, progress=False)
    assert len(mock1.calls) == 2 * 8              # 2 episodes fully run
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert all(c["status"] == "COMPLETE" for c in man["cells"].values())

    # Second run over the same outdir must re-call nothing.
    mock2 = MockTransport()
    xr.run(tasks_file, ["gemma3:4b"], ["R0"], tmp_path,
           base_url="http://x", transport=mock2, retry_backoff=0,
           max_retries=1, progress=False)
    assert mock2.calls == []


def test_manifest_resume_preseeded_cell(tasks_file, tmp_path):
    # Pre-mark episode 0 COMPLETE; only episode 1's 8 answers should run.
    outdir = tmp_path
    (outdir / "raw").mkdir(parents=True)
    key = xr._cell_key("gemma3:4b", "R0", 0)
    (outdir / "manifest.json").write_text(json.dumps(
        {"cells": {key: {"model": "gemma3:4b", "regime": "R0",
                         "episode": 0, "status": "COMPLETE"}}}),
        encoding="utf-8")
    mock = MockTransport()
    xr.run(tasks_file, ["gemma3:4b"], ["R0"], outdir,
           base_url="http://x", transport=mock, retry_backoff=0,
           max_retries=1, progress=False)
    assert len(mock.calls) == 8                   # episode 1 only


def test_failed_path_marks_cell_and_continues(tasks_file, tmp_path):
    mock = MockTransport(fail=True)
    man = xr.run(tasks_file, ["gemma3:4b"], ["R0"], tmp_path,
                 base_url="http://x", episodes_limit=1, transport=mock,
                 retry_backoff=0, max_retries=1, progress=False)
    key = xr._cell_key("gemma3:4b", "R0", 0)
    assert man["cells"][key]["status"] == "FAILED"
    assert man["cells"][key]["reason"]
    # retried max_retries+1 = 2 times before giving up on the first call
    assert len(mock.calls) == 2
    # manifest on disk reflects the FAILED status (atomic write happened)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["cells"][key]["status"] == "FAILED"


def test_raw_log_records_calls(tasks_file, tmp_path):
    mock = MockTransport()
    xr.run(tasks_file, ["gemma3:4b"], ["R1"], tmp_path,
           base_url="http://x", episodes_limit=1, transport=mock,
           retry_backoff=0, max_retries=1, progress=False)
    raw = (tmp_path / "raw" / "gemma3_4b_R1.jsonl").read_text().splitlines()
    recs = [json.loads(x) for x in raw if x.strip()]
    kinds = [r["kind"] for r in recs]
    assert kinds[:4] == [f"compact_{k}" for k in range(1, 5)]
    assert sorted(r["kind"] for r in recs if r["kind"].startswith(
        "question_")) == [f"question_{i}" for i in range(1, 9)]
    for r in recs:
        assert "messages" in r and "response" in r
        assert r["eval_count"] == 50


# ── analyze: pairing, completeness gate ──────────────────────────────


def _probe(model, regime, ep, qid, cls):
    return {"model": model, "regime": regime, "episode": ep,
            "question_id": qid, "type": "direct", "truth": 1,
            "answer": 1 if cls == "correct" else 2, "class": cls}


def _results_file(tmp_path, probes, cells):
    res = {"tasks": "t", "models": sorted({p["model"] for p in probes}),
           "regimes": sorted({p["regime"] for p in probes}),
           "cells": cells, "probes": probes,
           "compaction_stage": [], "compaction_tokens": {}}
    p = tmp_path / "results.json"
    p.write_text(json.dumps(res), encoding="utf-8")
    return p


def test_mcnemar_pairing_excludes_unpaired_and_r0(tmp_path):
    probes = [
        # m1 ep0 q1: R1 correct / R2 wrong  -> b
        _probe("m1", "R1", 0, 1, "correct"),
        _probe("m1", "R2", 0, 1, "other_wrong"),
        # m1 ep0 q2: R1 wrong / R2 correct  -> c
        _probe("m1", "R1", 0, 2, "omission"),
        _probe("m1", "R2", 0, 2, "correct"),
        # m1 ep0 q3: R2 only (unpaired) -> excluded
        _probe("m1", "R2", 0, 3, "correct"),
        # R0 probes never enter the McNemar
        _probe("m1", "R0", 0, 1, "correct"),
        _probe("m1", "R0", 0, 2, "correct"),
        # m2 ep0 q1: R1 wrong / R2 correct  -> c
        _probe("m2", "R1", 0, 1, "conflation"),
        _probe("m2", "R2", 0, 1, "correct"),
    ]
    ncells = len({(p["model"], p["regime"]) for p in probes})
    cells = {"complete": ncells, "not_complete": 0, "expected": ncells,
             "expected_probes_per_cell": 8}
    path = _results_file(tmp_path, probes, cells)
    s = xr.analyze(path, tmp_path)
    pooled = s["mcnemar_R1_vs_R2_pooled"]
    assert pooled["b_R1correct_R2wrong"] == 1
    assert pooled["c_R2correct_R1wrong"] == 2
    assert pooled["direction"] == "R2_better"
    assert pooled["p_value"] == pytest.approx(xr.mcnemar_exact(1, 2))
    per = s["mcnemar_R1_vs_R2_by_model"]
    assert per["m1"]["b_R1correct_R2wrong"] == 1
    assert per["m1"]["c_R2correct_R1wrong"] == 1
    assert per["m2"]["b_R1correct_R2wrong"] == 0
    assert per["m2"]["c_R2correct_R1wrong"] == 1


def test_analyze_completeness_gate(tmp_path):
    probes = [_probe("m1", "R1", 0, 1, "correct"),
              _probe("m1", "R2", 0, 1, "correct")]
    cells = {"complete": 1, "not_complete": 1, "expected": 2,
             "expected_probes_per_cell": 8}
    path = _results_file(tmp_path, probes, cells)
    with pytest.raises(SystemExit):
        xr.analyze(path, tmp_path)
    s = xr.analyze(path, tmp_path, allow_partial=True)
    assert s["partial_bank_non_confirmatory"] is True
    md = (tmp_path / "analysis.md").read_text(encoding="utf-8")
    assert "NON-CONFIRMATORY" in md


# ── manifest hardening ───────────────────────────────────────────────


def test_manifest_corruption_refuses(tasks_file, tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        xr._load_manifest(bad, tasks_file, ["m"], ["R0"], 2)


def test_manifest_tasks_hash_mismatch_refuses(tasks_file, tmp_path):
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"tasks_sha256": "0" * 64, "cells": {}}),
                   encoding="utf-8")
    with pytest.raises(SystemExit):
        xr._load_manifest(man, tasks_file, ["m"], ["R0"], 2)


def test_manifest_records_tasks_hash(tasks_file, tmp_path):
    man = xr._load_manifest(tmp_path / "manifest.json", tasks_file,
                            ["m"], ["R0"], 2)
    assert man["tasks_sha256"] == xr._tasks_sha256(tasks_file)


# ── import safety ────────────────────────────────────────────────────


def test_import_is_side_effect_free():
    # Module imported at top of file with no crash; main is guarded (only
    # runs under __main__) and generation requires an explicit output path —
    # there is no module-level execution to leave stray artifacts behind.
    assert callable(xr.main)
    assert callable(xr.generate) and callable(xr.run) and callable(xr.score)
