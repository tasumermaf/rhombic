"""XR-001 externalization-robustness pilot — generator, runner, scorer, analysis.

Implements the pre-registered protocol at
``results/XR-001-externalization-pilot/PROTOCOL.md`` exactly. Nothing in this
module changes the protocol; it operationalizes it.

The pilot measures whether the FORMAT of a compacted agent state (prose vs.
typed block) affects numeric fidelity across a cascading multi-segment task,
at matched token budgets, over three local Ollama models. Four subcommands:

    generate   deterministic seeded task episodes + ground truth -> tasks.json
    run        execute the three memory regimes over Ollama (resumable)
    score      classify every probe answer + exploratory compaction-stage metric
    analyze    Wilson CIs, regime x class table, paired exact McNemar (R1 vs R2)

Stdlib only (urllib, json, argparse, random, re, math, time, os, pathlib). The
runner talks to Ollama's ``POST /api/chat`` (non-streaming) at localhost:11434;
it is import-safe (no side effects at import; ``main`` is guarded), bounded
(manifest rewritten after every episode-cell), and safe to kill and resume.

Usage
-----
    python scripts/xr001_externalization_pilot.py generate \
        --out results/XR-001-externalization-pilot/tasks.json
    python scripts/xr001_externalization_pilot.py run \
        --tasks results/XR-001-externalization-pilot/tasks.json \
        --models gemma3:4b,qwen3:14b,qwen3-coder:30b --regimes R0,R1,R2
    python scripts/xr001_externalization_pilot.py score \
        --tasks .../tasks.json --outdir results/XR-001-externalization-pilot
    python scripts/xr001_externalization_pilot.py analyze \
        --results .../results.json --outdir results/XR-001-externalization-pilot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import hashlib
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL = "results/XR-001-externalization-pilot/PROTOCOL.md"

# ── Pinned protocol constants (§2, §3, §4) ───────────────────────────────
BASE_SEED = 77001
N_EPISODES = 15
N_SEGMENTS = 4
VALUE_LO, VALUE_HI = 101, 987           # inclusive entity-value range (§2)
DEFAULT_MODELS = ["gemma3:4b", "qwen3:14b", "qwen3-coder:30b"]
DEFAULT_REGIMES = ["R0", "R1", "R2"]
THINK_DISABLED_MODEL = "qwen3:14b"      # only this model gets think=false (§4)
COMPACT_NUM_PREDICT = 256               # §3 hard cap for compaction calls
ANSWER_NUM_PREDICT = 192               # answer calls (protocol amendment 1)
OLLAMA_NUM_CTX = 4096                   # pinned so R0's full transcript is
                                        # never silently head-truncated by a
                                        # model-default context window
CALL_TIMEOUT_S = 600.0
GENERATOR_VERSION = "1"

SYSTEM_PREAMBLE = (
    "You are a meticulous analytical assistant tracking numeric facts stated "
    "across a multi-segment working session. Report values exactly as given; "
    "never round, guess, or infer unstated numbers."
)

# Compaction prompts are format-symmetric (§3): both state the transcript is
# REPLACED and anything omitted is lost, both instruct merging previous state
# with the new segment, both cap at 120 words, neither reveals the questions.
_COMPACT_R1 = (
    "You are maintaining a running prose summary of an analytical session. "
    "This summary will REPLACE the full transcript: any fact you leave out is "
    "permanently lost and cannot be recovered when later questions are asked."
    "\n\nPREVIOUS SUMMARY:\n{prev}\n\nNEW SEGMENT:\n{seg}\n\n"
    "Merge the previous summary with the new segment into a single updated "
    "prose summary of at most 120 words. Preserve every named entity and its "
    "exact numeric value. Output only the summary; do not answer or "
    "anticipate any questions."
)
_COMPACT_R2 = (
    "You are maintaining a running state block for an analytical session. "
    "This state block will REPLACE the full transcript: any fact you leave "
    "out is permanently lost and cannot be recovered when later questions are "
    "asked.\n\nPREVIOUS STATE:\n{prev}\n\nNEW SEGMENT:\n{seg}\n\n"
    "Merge the previous state with the new segment into a single updated "
    "state block of at most 120 words, using EXACTLY this format:\n\n"
    "=== STATE ===\nvalues:\n  NAME = N\ncombined:\n  NAME1 + NAME2 = N\n"
    "relations:\n  NAMEC = NAMEA + 1\nnotes: <free text>\n=== END STATE ===\n\n"
    "Preserve every named entity and its exact numeric value. Output only the "
    "state block; do not answer or anticipate any questions."
)

# ── Task construction (§2) ───────────────────────────────────────────────

# CV-pattern pseudoword alphabets. Alternating consonant/vowel over 6 slots
# avoids real English words by construction; a tiny blocklist is belt-and-
# suspenders against the rare accidental word.
_CONS = "BDFGKLMNPRSTVZ"
_VOW = "AEIOU"
_WORD_BLOCKLIST = {"banana", "tomato", "cabana", "wasabi", "salami",
                   "pajama", "karate", "safari", "levity", "malady"}

_OPENERS = [
    "the session moved into a fresh block of readings",
    "another round of measurements was opened in the log",
    "the analysts began the next block of the working session",
    "recording resumed after a short handover between operators",
]
_FILLER = [
    "the operator confirmed each reading aloud before moving on",
    "notes were transcribed directly into the shared ledger",
    "instrument drift stayed within the accepted tolerance for the run",
    "a brief pause was taken to re-seat the calibration jig",
    "the analyst cross-checked the entry against the previous pass",
    "ambient conditions in the lab held steady throughout",
    "the bench technician initialled the margin of the page",
    "no anomalies were flagged during this stretch of the work",
    "the reference standard was left untouched between readings",
    "the running tally was mirrored to the backup ledger as usual",
    "each figure was read back once for the record",
    "the panel lights remained nominal for the duration",
]
_BASE_T = [
    "the calibration value of {n} is {v}",
    "{n} carries a calibration value of {v}",
    "we recorded {n} at a calibration value of {v}",
]
_PARTNER_T = [
    "the calibration value of {n} is {v}",
    "{n} was measured at a calibration value of {v}",
    "for {n} the calibration value came in at {v}",
]
_COMBINED_T = [
    "the combined value of {a} and {b} is {s}",
    "taken together, {a} and {b} give a combined value of {s}",
    "{a} and {b} combine to a stated total of {s}",
]
_NEIGHBOR_T = [
    "{c}'s value sits exactly one above {a}, at {v}",
    "{c} lands one unit above {a}, at {v}",
    "{c} is pinned one above {a}, giving {v}",
]
_DISTRACT_T = [
    "the calibration value of {n} is {v}",
    "{n} was logged at a calibration value of {v}",
    "separately, {n} held a calibration value of {v}",
]

# Fixed, per-episode-invariant probe plan (§2 table). 0-based segment indices;
# role in {base, partner, neighbor}. Multi-hop operands name a role+segment
# pair drawn from DIFFERENT segments. This fixed plan makes the design fully
# paired across regimes and models.
_QUESTION_PLAN = [
    {"id": 1, "type": "direct", "seg": 0, "role": "base"},
    {"id": 2, "type": "direct", "seg": 1, "role": "partner"},
    {"id": 3, "type": "direct", "seg": 2, "role": "base"},
    {"id": 4, "type": "conflation", "seg": 0, "role": "neighbor"},
    {"id": 5, "type": "conflation", "seg": 3, "role": "neighbor"},
    {"id": 6, "type": "combined", "seg": 0},
    {"id": 7, "type": "multihop", "ops": [("base", 1), ("base", 3)]},
    {"id": 8, "type": "multihop", "ops": [("partner", 0), ("base", 2)]},
]


def _make_name(rng: random.Random) -> str:
    return "".join(rng.choice(_CONS if i % 2 == 0 else _VOW)
                   for i in range(6))


def _gen_names(rng: random.Random, n: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < n:
        nm = _make_name(rng)
        if nm in seen or nm.lower() in _WORD_BLOCKLIST:
            continue
        seen.add(nm)
        out.append(nm)
    return out


def _assign_values(rng: random.Random) -> dict:
    """Draw entity values satisfying every §2 constraint by rejection.

    16 independent values (4 base, 4 partner, 8 distractor) are drawn distinct;
    the 4 neighbors are base+1. The whole 20-value set must be distinct with no
    accidental +-1 adjacency except the 4 designed (base, base+1) pairs; the 4
    stated sums must be distinct from each other and from all entity values;
    the 2 multi-hop truths must be distinct from everything and each other.
    """
    span = range(VALUE_LO, VALUE_HI + 1)
    for _ in range(200_000):
        picks = rng.sample(span, 16)
        base = picks[0:4]
        partner = picks[4:8]
        distract = picks[8:16]
        neighbor = [b + 1 for b in base]
        if any(c > VALUE_HI for c in neighbor):
            continue
        entity_vals = base + partner + neighbor + distract
        if len(set(entity_vals)) != 20:
            continue
        designed = {(b, b + 1) for b in base}
        ordered = sorted(entity_vals)
        adj_ok = True
        for i in range(len(ordered) - 1):
            if ordered[i + 1] - ordered[i] == 1 and \
                    (ordered[i], ordered[i + 1]) not in designed:
                adj_ok = False
                break
        if not adj_ok:
            continue
        stated = [base[s] + partner[s] for s in range(4)]
        if len(set(stated)) != 4:
            continue
        val_set = set(entity_vals)
        if any(s in val_set for s in stated):
            continue
        # Multi-hop truths per the fixed plan (base1+base3, partner0+base2).
        multi = [base[1] + base[3], partner[0] + base[2]]
        if multi[0] == multi[1]:
            continue
        if any(m in val_set for m in multi):
            continue
        if any(m in set(stated) for m in multi):
            continue
        return {"base": base, "partner": partner,
                "neighbor": neighbor, "distract": distract,
                "stated": stated, "multi": multi}
    raise RuntimeError("value constraints unsatisfiable — check VALUE range")


def _wc(fragments: list[str]) -> int:
    return len((" ".join(fragments)).split())


def _sentence(fragment: str) -> str:
    return fragment[:1].upper() + fragment[1:] + "."


def _pick_filler(rng: random.Random, avoid: str | None) -> str:
    # Never repeat the immediately preceding filler — keeps the narrative
    # readable while staying deterministic in the seed.
    pool = [f for f in _FILLER if f != avoid] if avoid else _FILLER
    return rng.choice(pool)


def _compose_segment(rng: random.Random, ent: dict) -> str:
    a, b, c = ent["base"], ent["partner"], ent["neighbor"]
    d1, d2 = ent["distractors"]
    last_filler = None

    def filler() -> str:
        nonlocal last_filler
        last_filler = _pick_filler(rng, last_filler)
        return last_filler

    frags = [
        rng.choice(_OPENERS),
        rng.choice(_BASE_T).format(n=a["name"], v=a["value"]),
        filler(),
        rng.choice(_PARTNER_T).format(n=b["name"], v=b["value"]),
        rng.choice(_COMBINED_T).format(a=a["name"], b=b["name"],
                                       s=ent["combined"]),
        filler(),
        rng.choice(_NEIGHBOR_T).format(c=c["name"], a=a["name"], v=c["value"]),
        rng.choice(_DISTRACT_T).format(n=d1["name"], v=d1["value"]),
        rng.choice(_DISTRACT_T).format(n=d2["name"], v=d2["value"]),
    ]
    while _wc(frags) < 120:
        frags.append(filler())
        if _wc(frags) > 175:            # unreachable in practice; hard guard
            break
    text = " ".join(_sentence(f) for f in frags)
    wc = len(text.split())
    if not 120 <= wc <= 180:
        raise RuntimeError(f"segment word count {wc} out of [120,180]")
    return text


def build_episode(index: int, base_seed: int = BASE_SEED) -> dict:
    """Construct one fully-specified episode (deterministic in the seed)."""
    seed = base_seed + index
    rng = random.Random(seed)
    names = _gen_names(rng, 20)
    vals = _assign_values(rng)

    segments = []
    for s in range(N_SEGMENTS):
        base = {"name": names[5 * s + 0], "value": vals["base"][s],
                "role": "base"}
        partner = {"name": names[5 * s + 1], "value": vals["partner"][s],
                   "role": "partner"}
        neighbor = {"name": names[5 * s + 2], "value": vals["neighbor"][s],
                    "role": "neighbor"}
        distractors = [
            {"name": names[5 * s + 3], "value": vals["distract"][2 * s]},
            {"name": names[5 * s + 4], "value": vals["distract"][2 * s + 1]},
        ]
        combined = vals["stated"][s]
        ent = {"base": base, "partner": partner, "neighbor": neighbor,
               "distractors": distractors, "combined": combined}
        text = _compose_segment(rng, ent)
        segments.append({
            "index": s + 1,
            "text": text,
            "entities": {"base": base, "partner": partner,
                         "neighbor": neighbor, "distractors": distractors},
            "combined": {"names": [base["name"], partner["name"]],
                         "value": combined},
        })

    questions = _build_questions(segments, vals)
    conflation_set = sorted(
        {e["value"] for seg in segments for e in _iter_entities(seg)}
        | set(vals["stated"]))
    return {"episode": index, "seed": seed, "segments": segments,
            "questions": questions, "conflation_set": conflation_set}


def _role_entity(segments: list[dict], seg: int, role: str) -> dict:
    return segments[seg]["entities"][role]


def _build_questions(segments: list[dict], vals: dict) -> list[dict]:
    q = []
    for plan in _QUESTION_PLAN:
        if plan["type"] in ("direct", "conflation"):
            ent = _role_entity(segments, plan["seg"], plan["role"])
            prompt = f"What is the calibration value of {ent['name']}?"
            q.append({"id": plan["id"], "type": plan["type"],
                      "segment": plan["seg"] + 1, "entity": ent["name"],
                      "prompt": prompt, "truth": ent["value"]})
        elif plan["type"] == "combined":
            seg = segments[plan["seg"]]
            a, b = seg["combined"]["names"]
            prompt = f"What is the combined value of {a} and {b}?"
            q.append({"id": plan["id"], "type": "combined",
                      "segment": plan["seg"] + 1, "entities": [a, b],
                      "prompt": prompt, "truth": seg["combined"]["value"]})
        else:  # multihop
            ents = [_role_entity(segments, s, r) for r, s in plan["ops"]]
            n1, n2 = ents[0]["name"], ents[1]["name"]
            prompt = ("What is the sum of the calibration values of "
                      f"{n1} and {n2}?")
            q.append({"id": plan["id"], "type": "multihop",
                      "entities": [n1, n2],
                      "prompt": prompt,
                      "truth": ents[0]["value"] + ents[1]["value"]})
    return q


def _iter_entities(seg: dict):
    e = seg["entities"]
    yield e["base"]
    yield e["partner"]
    yield e["neighbor"]
    yield from e["distractors"]


def generate(out_path: str | os.PathLike, episodes: int = N_EPISODES,
             base_seed: int = BASE_SEED) -> dict:
    obj = {
        "protocol": PROTOCOL,
        "generator_version": GENERATOR_VERSION,
        "base_seed": base_seed,
        "n_episodes": episodes,
        "n_segments": N_SEGMENTS,
        "value_range": [VALUE_LO, VALUE_HI],
        "episodes": [build_episode(i, base_seed) for i in range(episodes)],
    }
    _write_json(out_path, obj)
    return obj


# ── Ollama transport (§3) ────────────────────────────────────────────────


class TransportError(Exception):
    """A recoverable Ollama call failure (connection/timeout/bad body)."""


def call_ollama(messages: list[dict], *, model: str, num_predict: int,
                base_url: str, think: bool | None = None,
                timeout: float = CALL_TIMEOUT_S) -> dict:
    """One non-streaming POST /api/chat. Raises TransportError on failure.

    ``think`` is sent as a top-level field ONLY when not None (qwen3:14b gets
    think=false; every other model omits it entirely).
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_predict": num_predict,
                    "num_ctx": OLLAMA_NUM_CTX},
    }
    if think is not None:
        payload["think"] = think
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError,
            OSError) as e:
        raise TransportError(f"{type(e).__name__}: {e}") from e
    except json.JSONDecodeError as e:
        raise TransportError(f"bad JSON from ollama: {e}") from e
    return {"content": body.get("message", {}).get("content", ""),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count")}


def _retry(transport, *, messages, model, num_predict, think, base_url,
           max_retries: int, backoff: float):
    last = None
    for attempt in range(max_retries + 1):
        try:
            return transport(messages, model=model, num_predict=num_predict,
                             think=think, base_url=base_url)
        except TransportError as e:
            last = e
            if attempt < max_retries and backoff:
                time.sleep(backoff)
    raise TransportError(
        f"all {max_retries + 1} attempts failed: {last}")


# ── Prompt assembly (§3) ─────────────────────────────────────────────────


def _sys() -> dict:
    return {"role": "system", "content": SYSTEM_PREAMBLE}


def _answer_block(prompt: str) -> str:
    return (f"{prompt}\n\nAnswer with a single integer only. End your reply "
            "with a line of the form\nANSWER: <integer>")


def _transcript(segments: list[dict]) -> str:
    return "\n\n".join(f"Segment {seg['index']}:\n{seg['text']}"
                       for seg in segments)


def _compaction_prompt(regime: str, state: str | None, seg_text: str) -> str:
    prev = state if state is not None else "none (this is the first segment)"
    tmpl = _COMPACT_R1 if regime == "R1" else _COMPACT_R2
    return tmpl.format(prev=prev, seg=seg_text)


# ── Runner (§3, §7) ──────────────────────────────────────────────────────


def _sanitize(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def _cell_key(model: str, regime: str, episode: int) -> str:
    return f"{model}|{regime}|{episode}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _tasks_sha256(tasks_path) -> str:
    return hashlib.sha256(Path(tasks_path).read_bytes()).hexdigest()


def _load_manifest(path: Path, tasks_path, models, regimes, n_episodes):
    sha = _tasks_sha256(tasks_path)
    if path.exists():
        # A corrupted manifest must STOP the run, not silently restart the
        # whole sweep from scratch (hours of duplicate unattended work).
        try:
            man = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(
                f"manifest {path} exists but is unreadable ({e}); refusing "
                "to silently restart — inspect or delete it explicitly")
        man.setdefault("cells", {})
        stored = man.get("tasks_sha256")
        if stored is not None and stored != sha:
            raise SystemExit(
                f"manifest {path} was created against a different tasks.json "
                f"(sha256 {stored[:12]}... != {sha[:12]}...); refusing to mix "
                "episodes across task sets")
        man["tasks_sha256"] = sha
        return man
    return {"tasks": str(tasks_path), "tasks_sha256": sha, "models": models,
            "regimes": regimes, "n_episodes": n_episodes, "cells": {}}


def _exec_cell(ep, model, regime, think, transport, base_url, log,
               backoff, max_retries):
    """Execute one (model, regime, episode) cell; log every LLM call.

    R0 asks each question against the full transcript. R1/R2 cascade a
    compaction after every segment (state k is produced from state k-1 + the
    new segment ONLY), then ask each question against the FINAL state.
    """
    segments, questions = ep["segments"], ep["questions"]

    def call(messages, num_predict, kind, **extra):
        t0 = time.time()
        resp = _retry(transport, messages=messages, model=model,
                      num_predict=num_predict, think=think, base_url=base_url,
                      max_retries=max_retries, backoff=backoff)
        log(kind=kind, messages=messages, response=resp["content"],
            prompt_eval_count=resp.get("prompt_eval_count"),
            eval_count=resp.get("eval_count"),
            wall_s=round(time.time() - t0, 3), **extra)
        return resp

    if regime == "R0":
        transcript = _transcript(segments)
        for q in questions:
            msgs = [_sys(), {"role": "user",
                             "content": transcript + "\n\n"
                             + _answer_block(q["prompt"])}]
            call(msgs, ANSWER_NUM_PREDICT, f"question_{q['id']}",
                 question_id=q["id"])
        return

    state = None
    for k, seg in enumerate(segments, 1):
        prompt = _compaction_prompt(regime, state, seg["text"])
        msgs = [_sys(), {"role": "user", "content": prompt}]
        resp = call(msgs, COMPACT_NUM_PREDICT, f"compact_{k}", segment=k)
        state = resp["content"]          # cascade: only this carries forward
    for q in questions:
        msgs = [_sys(), {"role": "user",
                         "content": (state or "") + "\n\n"
                         + _answer_block(q["prompt"])}]
        call(msgs, ANSWER_NUM_PREDICT, f"question_{q['id']}",
             question_id=q["id"])


def run(tasks_path, models, regimes, outdir, *, base_url,
        episodes_limit=None, transport=None, retry_backoff=30.0,
        max_retries=3, progress=True):
    transport = transport or call_ollama
    outdir = Path(outdir)
    rawdir = outdir / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    episodes = tasks["episodes"]
    if episodes_limit is not None:
        episodes = episodes[:episodes_limit]

    manifest_path = outdir / "manifest.json"
    manifest = _load_manifest(manifest_path, tasks_path, models, regimes,
                              len(episodes))
    # An outdir whose manifest was created with a different episode count
    # (e.g. an --episodes-limit smoke) must not host the full run — the
    # completeness accounting would be silently wrong.
    if manifest.get("n_episodes") != len(episodes):
        raise SystemExit(
            f"manifest n_episodes={manifest.get('n_episodes')} != this run's "
            f"{len(episodes)} (--episodes-limit mismatch?); use a fresh "
            "outdir for smoke runs")
    ok = failed = 0
    for model in models:
        think = False if model == THINK_DISABLED_MODEL else None
        for regime in regimes:
            rawfile = rawdir / f"{_sanitize(model)}_{regime}.jsonl"
            for ep in episodes:
                key = _cell_key(model, regime, ep["episode"])
                cell = manifest["cells"].get(key)
                if cell and cell.get("status") == "COMPLETE":
                    continue
                t0 = time.time()
                records: list[dict] = []

                def log(**rec):
                    rec.update(model=model, regime=regime,
                               episode=ep["episode"])
                    records.append(rec)

                try:
                    _exec_cell(ep, model, regime, think, transport, base_url,
                               log, retry_backoff, max_retries)
                    status, reason = "COMPLETE", None
                    ok += 1
                except TransportError as e:
                    status, reason = "FAILED", str(e)
                    failed += 1
                with rawfile.open("a", encoding="utf-8") as fh:
                    for rec in records:
                        fh.write(json.dumps(rec) + "\n")
                manifest["cells"][key] = {
                    "model": model, "regime": regime,
                    "episode": ep["episode"], "status": status,
                    "n_calls": len(records), "reason": reason,
                    "updated_at": _now()}
                _atomic_write_json(manifest_path, manifest)
                if progress:
                    print(f"[{model} {regime} ep {ep['episode'] + 1}/"
                          f"{len(episodes)}] ok={ok} failed={failed} "
                          f"elapsed={time.time() - t0:.1f}s", flush=True)
    return manifest


# ── Scoring (§5) ─────────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r"ANSWER:\s*(-?\d[\d,]*)", re.IGNORECASE)
_R2_PAIR_RE = re.compile(r"^\s*([A-Z][A-Z0-9]*)\s*=\s*(-?\d[\d,]*)\s*$",
                         re.MULTILINE)
_INT_TOKEN_RE = re.compile(r"-?\d[\d,]*")


def parse_answer(text: str | None):
    """Last ``ANSWER: <int>`` in the reply; commas tolerated. None if absent."""
    matches = _ANSWER_RE.findall(text or "")
    if not matches:
        return None
    try:
        return int(matches[-1].replace(",", ""))
    except ValueError:
        return None


def classify(answer, truth, conflation_set) -> str:
    """§5 corruption classes.

    off_by_one is checked BEFORE conflation: a truth+-1 answer can also be a
    member of the conflation set (the neighbor's base value is exactly truth-1
    on a conflation probe), and the protocol's off_by_one class takes
    precedence over conflation in that overlap.
    """
    if answer is None:
        return "omission"
    if answer == truth:
        return "correct"
    if abs(answer - truth) == 1:
        return "off_by_one"
    if answer in conflation_set and answer != truth:
        return "conflation"
    return "other_wrong"


def _nearest_number_to_name(text: str, name: str):
    idx = text.find(name)
    if idx < 0:
        return None
    tokens = [(m.start(), int(m.group().replace(",", "")))
              for m in _INT_TOKEN_RE.finditer(text)]
    if not tokens:
        return None
    return min(tokens, key=lambda t: abs(t[0] - idx))[1]


def _score_state_r2(state: str, emap: dict) -> dict:
    retained, correct, wrong = set(), 0, 0
    for name, num in _R2_PAIR_RE.findall(state):
        if name in emap:
            retained.add(name)
            if int(num.replace(",", "")) == emap[name]:
                correct += 1
            else:
                wrong += 1
    return {"retained": len(retained), "correct": correct, "wrong": wrong,
            "n_entities": len(emap)}


def _score_state_r1(state: str, emap: dict) -> dict:
    retained = correct = wrong = 0
    for name, val in emap.items():
        if name in state:
            retained += 1
            if _nearest_number_to_name(state, name) == val:
                correct += 1
            else:
                wrong += 1
    return {"retained": retained, "correct": correct, "wrong": wrong,
            "n_entities": len(emap)}


def score(tasks_path, outdir) -> dict:
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    episodes = {ep["episode"]: ep for ep in tasks["episodes"]}
    outdir = Path(outdir)
    rawdir = outdir / "raw"

    # Only manifest-COMPLETE cells are scored (protocol amendment 3): records
    # from FAILED/partial cells, and stale attempts of later-resumed cells,
    # must not contaminate probes or the matched-budget token means.
    manifest_path = outdir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(
            f"score requires a readable run manifest at {manifest_path}: {e}")
    complete: set = set()
    n_not_complete = 0
    for cell in manifest.get("cells", {}).values():
        if cell.get("status") == "COMPLETE":
            complete.add((cell["model"], cell["regime"], cell["episode"]))
        else:
            n_not_complete += 1

    question_recs: dict = {}      # (model,regime,ep,qid) -> record (last wins)
    compact_recs: dict = {}       # (model,regime,ep,segment) -> record (last wins)
    for f in sorted(rawdir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.get("kind", "")
            key = (rec["model"], rec["regime"], rec["episode"])
            if key not in complete:
                continue
            if kind.startswith("question_"):
                question_recs[key + (rec.get("question_id"),)] = rec
            elif kind.startswith("compact_"):
                compact_recs[key + (rec.get("segment", 0),)] = rec

    # Derive the final state and the token means from the DEDUPED compaction
    # records — a failed-then-resumed cell contributes each segment once.
    compact_final: dict = {}      # (model,regime,ep) -> final compaction record
    compact_tokens: dict = {}     # regime -> [(model, eval_count), ...]
    for (model, regime, ep_id, seg), rec in sorted(
            compact_recs.items(), key=lambda kv: kv[0]):
        k3 = (model, regime, ep_id)
        cur = compact_final.get(k3)
        if cur is None or seg >= cur.get("segment", 0):
            compact_final[k3] = rec
        ec = rec.get("eval_count")
        if ec is not None:
            compact_tokens.setdefault(regime, []).append((model, ec))

    probes = []
    for (model, regime, ep_id, qid), rec in sorted(
            question_recs.items(), key=lambda kv: kv[0]):
        ep = episodes[ep_id]
        q = next(qq for qq in ep["questions"] if qq["id"] == qid)
        cset = set(ep["conflation_set"])
        ans = parse_answer(rec.get("response", ""))
        probes.append({"model": model, "regime": regime, "episode": ep_id,
                       "question_id": qid, "type": q["type"],
                       "truth": q["truth"], "answer": ans,
                       "class": classify(ans, q["truth"], cset)})

    stage = []
    for (model, regime, ep_id), rec in sorted(
            compact_final.items(), key=lambda kv: kv[0]):
        if regime == "R0":
            continue
        emap = {e["name"]: e["value"]
                for seg in episodes[ep_id]["segments"]
                for e in _iter_entities(seg)}
        state = rec.get("response", "")
        res = (_score_state_r2 if regime == "R2" else _score_state_r1)(
            state, emap)
        res.update(model=model, regime=regime, episode=ep_id)
        stage.append(res)

    tokens = {}
    for regime, lst in compact_tokens.items():
        vals = [ec for _, ec in lst]
        by_model: dict = {}
        for m, ec in lst:
            by_model.setdefault(m, []).append(ec)
        tokens[regime] = {
            "n": len(vals),
            "mean": (sum(vals) / len(vals)) if vals else None,
            "by_model": {m: {"n": len(v), "mean": sum(v) / len(v)}
                         for m, v in by_model.items()}}

    n_expected = (manifest.get("n_episodes", 0)
                  * len(manifest.get("models", []))
                  * len(manifest.get("regimes", [])))
    cells_info = {"complete": len(complete),
                  "not_complete": n_not_complete,
                  "expected": n_expected,
                  "expected_probes_per_cell":
                      manifest.get("n_episodes", 0) * 8}
    if len(complete) != n_expected or n_not_complete:
        print(f"WARNING: cells {len(complete)}/{n_expected} COMPLETE "
              f"({n_not_complete} not complete) — analyze will refuse "
              "without --allow-partial", flush=True)

    out = {"tasks": str(tasks_path),
           "models": sorted({p["model"] for p in probes}),
           "regimes": sorted({p["regime"] for p in probes}),
           "cells": cells_info,
           "probes": probes, "compaction_stage": stage,
           "compaction_tokens": tokens}
    _write_json(Path(outdir) / "results.json", out)
    return out


# ── Analysis (§6) ────────────────────────────────────────────────────────

_Z95 = 1.959963984540054
_CLASSES = ["correct", "off_by_one", "conflation", "other_wrong", "omission"]


def wilson_ci(k: int, n: int, z: float = _Z95) -> tuple:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def analyze(results_path, outdir, *, allow_partial: bool = False) -> dict:
    res = json.loads(Path(results_path).read_text(encoding="utf-8"))
    probes = res["probes"]
    models = res["models"]
    regimes = res["regimes"]

    # Completeness gate (protocol amendment 3): the confirmatory analysis
    # runs only on a fully COMPLETE bank; interim looks require an explicit
    # flag and are marked NON-CONFIRMATORY.
    ci = res.get("cells") or {}
    partial = bool(ci) and (ci.get("complete") != ci.get("expected")
                            or ci.get("not_complete"))
    if partial and not allow_partial:
        raise SystemExit(
            f"XR-001 bank incomplete: {ci.get('complete')}/"
            f"{ci.get('expected')} cells COMPLETE "
            f"({ci.get('not_complete')} not complete). Pass --allow-partial "
            "for an interim, NON-CONFIRMATORY look.")
    if partial:
        print("WARNING (pre-registration): incomplete bank — this output is "
              "NON-CONFIRMATORY", flush=True)

    # per (model x regime) corruption rate + Wilson CI
    cells: dict = {}
    for p in probes:
        d = cells.setdefault((p["model"], p["regime"]),
                             {"n": 0, "corrupt": 0})
        d["n"] += 1
        if p["class"] != "correct":
            d["corrupt"] += 1
    corruption = {}
    for (m, r), d in cells.items():
        lo, hi = wilson_ci(d["corrupt"], d["n"])
        corruption[f"{m}|{r}"] = {
            "model": m, "regime": r, "n": d["n"], "corrupt": d["corrupt"],
            "rate": d["corrupt"] / d["n"] if d["n"] else None,
            "ci95": [lo, hi]}

    # regime x class breakdown (pooled across models)
    breakdown = {r: {c: 0 for c in _CLASSES} for r in regimes}
    for p in probes:
        breakdown[p["regime"]][p["class"]] += 1

    # paired exact McNemar R1 vs R2 on identical probes, pooled + per-model
    def _mcnemar(model_filter=None):
        by_probe: dict = {}
        for p in probes:
            if model_filter and p["model"] != model_filter:
                continue
            key = (p["model"], p["episode"], p["question_id"])
            by_probe.setdefault(key, {})[p["regime"]] = \
                (p["class"] == "correct")
        b = c = 0
        for d in by_probe.values():
            if "R1" in d and "R2" in d:
                if d["R1"] and not d["R2"]:
                    b += 1                 # R1 correct, R2 wrong
                elif d["R2"] and not d["R1"]:
                    c += 1                 # R2 correct, R1 wrong
        return {"b_R1correct_R2wrong": b, "c_R2correct_R1wrong": c,
                "p_value": mcnemar_exact(b, c),
                "direction": ("R2_better" if c > b else
                              "R1_better" if b > c else "tie")}

    mcnemar_pooled = _mcnemar()
    mcnemar_by_model = {m: _mcnemar(m) for m in models}

    # multi-hop completion (questions 7-8) per (model x regime)
    multihop: dict = {}
    for p in probes:
        if p["question_id"] in (7, 8):
            d = multihop.setdefault(f"{p['model']}|{p['regime']}",
                                    {"model": p["model"],
                                     "regime": p["regime"], "n": 0, "ok": 0})
            d["n"] += 1
            if p["class"] == "correct":
                d["ok"] += 1
    for d in multihop.values():
        d["rate"] = d["ok"] / d["n"] if d["n"] else None

    summary = {
        "partial_bank_non_confirmatory": partial,
        "cells": ci,
        "corruption": corruption,
        "regime_class_breakdown": breakdown,
        "mcnemar_R1_vs_R2_pooled": mcnemar_pooled,
        "mcnemar_R1_vs_R2_by_model": mcnemar_by_model,
        "compaction_tokens": res.get("compaction_tokens", {}),
        "multihop_completion": multihop,
    }
    md = _render_analysis_md(models, regimes, summary)
    _write_text(Path(outdir) / "analysis.md", md)
    print(md)
    return summary


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _render_analysis_md(models, regimes, s) -> str:
    lines = ["# XR-001 - Externalization-Robustness Pilot: Analysis", "",
             f"Protocol: `{PROTOCOL}`. Confirmatory test = pre-registered "
             "paired McNemar (R1 vs R2); everything else is descriptive.", ""]
    if s.get("partial_bank_non_confirmatory"):
        ci = s.get("cells", {})
        lines += ["**WARNING: INCOMPLETE BANK ("
                  f"{ci.get('complete')}/{ci.get('expected')} cells) - "
                  "NON-CONFIRMATORY INTERIM OUTPUT.**", ""]

    lines += ["## Numeric-corruption rate (per model x regime, 95% Wilson CI)",
              "", "| Model | Regime | n | corrupt | rate | 95% CI |",
              "|---|---|---|---|---|---|"]
    for m in models:
        for r in regimes:
            d = s["corruption"].get(f"{m}|{r}")
            if not d:
                continue
            lo, hi = d["ci95"]
            lines.append(f"| {m} | {r} | {d['n']} | {d['corrupt']} | "
                         f"{_pct(d['rate'])} | [{_pct(lo)}, {_pct(hi)}] |")

    lines += ["", "## Regime x class breakdown (pooled across models)", "",
              "signature = off_by_one + conflation (P2 union class, "
              "protocol amendment 1)", "",
              "| Regime | " + " | ".join(_CLASSES) + " | signature |",
              "|---|" + "---|" * (len(_CLASSES) + 1)]
    for r in regimes:
        row = s["regime_class_breakdown"].get(r, {})
        sig = row.get("off_by_one", 0) + row.get("conflation", 0)
        lines.append(f"| {r} | "
                     + " | ".join(str(row.get(c, 0)) for c in _CLASSES)
                     + f" | {sig} |")

    mp = s["mcnemar_R1_vs_R2_pooled"]
    lines += ["", "## Paired exact McNemar - R1 vs R2 (confirmatory)", "",
              f"Pooled across models: b (R1 correct, R2 wrong) = "
              f"{mp['b_R1correct_R2wrong']}, c (R2 correct, R1 wrong) = "
              f"{mp['c_R2correct_R1wrong']}, two-sided exact p = "
              f"{mp['p_value']:.4g}, direction = {mp['direction']}.", "",
              "| Model | b (R1>R2) | c (R2>R1) | p | direction |",
              "|---|---|---|---|---|"]
    for m in models:
        d = s["mcnemar_R1_vs_R2_by_model"].get(m)
        if d:
            lines.append(f"| {m} | {d['b_R1correct_R2wrong']} | "
                         f"{d['c_R2correct_R1wrong']} | {d['p_value']:.4g} | "
                         f"{d['direction']} |")

    lines += ["", "## Mean realized compaction tokens (matched-budget check)",
              "", "| Regime | n calls | mean eval_count |", "|---|---|---|"]
    for r, d in s["compaction_tokens"].items():
        mean = d.get("mean")
        lines.append(f"| {r} | {d.get('n', 0)} | "
                     + ("—" if mean is None else f"{mean:.1f}") + " |")

    lines += ["", "## Multi-hop completion (questions 7-8)", "",
              "| Model | Regime | n | correct | rate |", "|---|---|---|---|---|"]
    for m in models:
        for r in regimes:
            d = s["multihop_completion"].get(f"{m}|{r}")
            if d:
                lines.append(f"| {m} | {r} | {d['n']} | {d['ok']} | "
                             f"{_pct(d['rate'])} |")
    lines.append("")
    return "\n".join(lines)


# ── IO helpers ───────────────────────────────────────────────────────────


def _write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_json(path, obj) -> None:
    _write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _atomic_write_json(path, obj) -> None:
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    _write_text(tmp, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ── CLI ──────────────────────────────────────────────────────────────────


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="XR-001 externalization-robustness pilot "
                    f"(protocol: {PROTOCOL}).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write seeded tasks.json")
    g.add_argument("--out", required=True)
    g.add_argument("--episodes", type=int, default=N_EPISODES)
    g.add_argument("--base-seed", type=int, default=BASE_SEED)

    r = sub.add_parser("run", help="execute regimes over Ollama (resumable)")
    r.add_argument("--tasks", required=True)
    r.add_argument("--models", type=_csv, default=list(DEFAULT_MODELS))
    r.add_argument("--regimes", type=_csv, default=list(DEFAULT_REGIMES))
    r.add_argument("--episodes-limit", type=int, default=None)
    r.add_argument("--base-url", default="http://localhost:11434")
    r.add_argument("--outdir",
                   default="results/XR-001-externalization-pilot")

    s = sub.add_parser("score", help="classify probes -> results.json")
    s.add_argument("--tasks", required=True)
    s.add_argument("--outdir",
                   default="results/XR-001-externalization-pilot")

    a = sub.add_parser("analyze", help="Wilson/McNemar -> analysis.md")
    a.add_argument("--results", required=True)
    a.add_argument("--outdir",
                   default="results/XR-001-externalization-pilot")
    a.add_argument("--allow-partial", action="store_true",
                   help="permit an interim NON-CONFIRMATORY analysis on an "
                        "incomplete bank (loud warning; prereg amendment 3)")

    args = parser.parse_args(argv)
    if args.cmd == "generate":
        obj = generate(args.out, episodes=args.episodes,
                       base_seed=args.base_seed)
        print(f"[xr001] wrote {len(obj['episodes'])} episodes -> {args.out}")
    elif args.cmd == "run":
        run(args.tasks, args.models, args.regimes, args.outdir,
            base_url=args.base_url, episodes_limit=args.episodes_limit)
    elif args.cmd == "score":
        out = score(args.tasks, args.outdir)
        print(f"[xr001] scored {len(out['probes'])} probes -> "
              f"{Path(args.outdir) / 'results.json'}")
    else:  # analyze
        analyze(args.results, args.outdir, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
