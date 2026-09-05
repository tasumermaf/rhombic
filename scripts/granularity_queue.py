# -*- coding: utf-8 -*-
"""Granularity ladder — sequential queue over the FROZEN tier order.

TIER ORDER (frozen at lock, `docs/LOCK_GRANULARITY_2026-08-04.md`):

    L0 re-baseline -> L1 -> Arm B -> L2 -> L3 -> D7
    (D6-per-level runs alongside each level's analysis)

This queue implements the first two rungs and STOPS:

  * **L0 is ANALYSIS-SIDE.** The design's §2 table gives L0 "new runs = 0" —
    its 240 adapters are the existing llama cohort of the Asset-1 bank
    (`results/asset1-bank/llama3.2-1b/`, 6 tasks x 40 seeds, verified
    480/480 COMPLETE). The queue asserts that cohort is present and complete,
    logs it, and enqueues nothing. `scripts/granularity_analysis.py` runs the
    L0 re-baseline (which exists to put kappa and the D6 reference on the
    anchor, per D5/D10).
  * **L1 is enqueued** — 240 runs, 12 classes x 20 seeds.
  * **Everything after L1 is GATED** and this queue refuses to enqueue it.
    Arm B, L2, L3 and D7 fire only when the PREVIOUS tier's gate has been
    recorded by the analysis side in `results/granularity/TIER_GATES.json`
    (`granularity_analysis.record_gate`). The queue enforces that order on
    the training side too (`training_interlock`, added 2026-09-05 per audit
    action A18): a level whose predecessor tier has not fired is refused
    before any run launches. Ambiguity G-5 (the T3 annotator) was RESOLVED
    2026-08-05 by dated amendment — xsum moved T3 -> T2 — so L2/L3 are
    launchable by their manifests and blocked only by tier order.
  * **Rate.** The GPU-day projection uses the mean `wall_clock_min` of this
    level's COMPLETE runs (their TIMING.md), over the PENDING remainder only;
    with no completed run it falls back to the card basis (30.56 min, llama
    n=240, RATE_EXTRACT.md) and says so. A literal 30.56 over the full plan
    (the 2026-08-04 version) overstated the remaining work.

Discipline (the house pattern, `scripts/s2_queue_runner.py`):
  - sequential by construction; one run per fresh subprocess
  - waits out a run already in progress from a prior detached launch
  - skips runs whose COMPLETE marker exists
  - `gpu_guard.pause_gate()` between EVERY run, in addition to the guard
    inside each run (a refused start is a log line; an OOM is a failure)
  - a STOP file in results/granularity/ ends the queue gracefully after the
    current run
  - three consecutive failures abort the level (loop stop rule: do not loop
    on a failing step, escalate); a single retry pass follows the main pass
  - QUEUE_STATE.md is rewritten from disk after EVERY run, so the ledger can
    never go stale relative to the filesystem

Launch detached (PowerShell). Set $env:HF_TOKEN in that shell FIRST — the
queue refuses to start without it (audit D-18), and HF_HUB_OFFLINE=1 is not a
substitute on transformers 4.57 (tokenizer loads by repo id call the Hub):
  Start-Process C:\\miniconda3\\envs\\falco\\python.exe -ArgumentList
    "scripts\\granularity_queue.py" -RedirectStandardOutput
    "results\\granularity\\logs\\queue_runner.log" -WindowStyle Hidden
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_CACHE", r"C:\falco\hf-cache\hub")
os.environ.setdefault("HF_DATASETS_CACHE", r"C:\falco\hf-cache\datasets")

import gpu_guard  # noqa: E402
import asset1_analysis_io as aio  # noqa: E402
from asset1_bank import utc_now  # noqa: E402
from granularity_labels import OUT_ROOT, TIER_ORDER  # noqa: E402
from granularity_runner import (  # noqa: E402
    FAMILY, live_classes, load_level, plan_runs, run_dir_for)

PY = sys.executable or r"C:\miniconda3\envs\falco\python.exe"
LOG_DIR = OUT_ROOT / "logs"
STOP_FILE = OUT_ROOT / "STOP"
STATE_FILE = OUT_ROOT / "QUEUE_STATE.md"

# The queue enqueues this and no more. Later tiers are gated (see docstring).
ENQUEUED_LEVELS = ["L1"]
GATED_LEVELS = ["ARMB (B2 B4 B8 B16)", "L2", "L3", "D7"]

MAX_CONSECUTIVE_FAILURES = 3
IN_PROGRESS_POLL_S = 120

L0_EXPECTED_LLAMA_RUNS = 240


def log(msg: str) -> None:
    line = f"{utc_now()} | {msg}"
    print(f"[queue] {line}", flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "queue.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_state(level: str, run_k: int) -> str:
    d = run_dir_for(level, run_k)
    if (d / "COMPLETE").exists():
        return "COMPLETE"
    if (d / "FAILED").exists():
        return "FAILED"
    if d.exists():
        return "IN_PROGRESS"
    return "PENDING"


def write_state(level: str, plan: list[dict], note: str = "") -> None:
    counts: dict[str, int] = {}
    for e in plan:
        s = run_state(level, e["run_k"])
        counts[s] = counts.get(s, 0) + 1
    pairs = [("level", level), ("n_runs", len(plan))]
    pairs += [(k.lower(), counts.get(k, 0))
              for k in ("COMPLETE", "FAILED", "IN_PROGRESS", "PENDING")]
    pairs += [("tier_order_frozen", " -> ".join(TIER_ORDER)),
              ("enqueued_by_this_queue", " ".join(ENQUEUED_LEVELS)),
              ("gated_not_enqueued", " | ".join(GATED_LEVELS)),
              ("updated_at_utc", utc_now())]
    width = max(len(k) for k, _ in pairs)
    body = "\n".join(f"{k.ljust(width)} = {v}" for k, v in pairs)
    text = ("# GRANULARITY QUEUE STATE\n\n"
            "Rewritten from the filesystem after every run by "
            "`scripts/granularity_queue.py` — never carried forward from a "
            "previous version of this file.\n\n"
            "=== VERIFIED STATE ===\n" + body + "\n"
            "=== END VERIFIED STATE ===\n")
    if note:
        text += f"\n{note}\n"
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, STATE_FILE)


def check_l0_rebaseline() -> bool:
    """L0 trains nothing; assert the existing llama cohort is complete."""
    try:
        manifest = aio.require_complete_bank(aio.REAL_BANK_ROOT)
    except SystemExit as e:
        log(f"L0 GATE FAILED: {e}")
        return False
    n_llama = sum(1 for r in manifest["runs"]
                  if r["family_short"] == FAMILY["short"]
                  and r["status"] == "COMPLETE")
    ok = n_llama == L0_EXPECTED_LLAMA_RUNS
    log(f"L0 re-baseline: ANALYSIS-SIDE, 0 runs enqueued. "
        f"llama3.2-1b COMPLETE adapters in the Asset-1 bank = {n_llama} "
        f"(expected {L0_EXPECTED_LLAMA_RUNS}) -> {'OK' if ok else 'MISMATCH'}")
    if ok:
        log("L0 analysis is run by scripts/granularity_analysis.py "
            "--level L0 (kappa + D6 on the anchor; no training).")
    return ok


def launch(level: str, run_k: int) -> int:
    cmd = [PY, str(SCRIPTS_DIR / "granularity_runner.py"),
           "--level", level, "--run", str(run_k)]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    tail = (r.stdout or "")[-500:] + (r.stderr or "")[-500:]
    log(f"{level} run {run_k} exit {r.returncode}\n{tail}")
    return r.returncode


CARD_RATE_BASIS_MIN = 30.56   # llama3.2-1b, n=240, results/asset1-bank/RATE_EXTRACT.md
GATES_FILE = OUT_ROOT / "TIER_GATES.json"   # written by granularity_analysis.record_gate
ARMB_RUNGS = ("B2", "B4", "B8", "B16")


def tier_of(level: str) -> str:
    return "ARMB" if level in ARMB_RUNGS else level


def fired_gates() -> list[str]:
    """Tiers whose gate the analysis side has recorded (same file it writes)."""
    if not GATES_FILE.exists():
        return []
    try:
        data = json.loads(GATES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log(f"TIER_GATES.json unreadable ({e}); treating as no gates fired")
        return []
    if isinstance(data, dict):
        return [t for t, v in data.items() if v]
    return [str(t) for t in data]


def training_interlock(level: str) -> list[str]:
    """Tiers that must have FIRED before this level may train. Empty = clear.

    Frozen order L0 -> L1 -> ARMB -> L2 -> L3 -> D7. L0 is analysis-side (its
    gate is the re-baseline, `granularity_analysis.py --level L0`) and gates
    L1's ANALYSIS via `require_tier_order`, not L1's training — the cohort
    check in `check_l0_rebaseline` is the training-side condition for L1.
    Every later tier trains only after its predecessor's gate is recorded.
    """
    tier = tier_of(level)
    idx = TIER_ORDER.index(tier)
    required = [t for t in TIER_ORDER[1:idx]]      # training tiers before this one
    return [t for t in required if t not in fired_gates()]


def measured_rate_min(level: str, plan: list[dict]) -> tuple[float, str]:
    """Mean wall_clock_min over this level's COMPLETE runs, else the card basis."""
    vals = []
    for e in plan:
        d = run_dir_for(level, e["run_k"])
        if not (d / "COMPLETE").exists():
            continue
        try:
            for ln in (d / "TIMING.md").read_text(encoding="utf-8").split("\n"):
                if ln.startswith("wall_clock_min"):
                    vals.append(float(ln.split("=", 1)[1].strip()))
                    break
        except (OSError, ValueError):
            continue
    if vals:
        return sum(vals) / len(vals), f"measured, mean of {len(vals)} COMPLETE runs"
    return CARD_RATE_BASIS_MIN, "card basis (no completed run yet)"


def drain_level(level: str) -> bool:
    """Run every pending run of a level. Returns False if the queue stopped."""
    missing = training_interlock(level)
    if missing:
        log(f"{level} REFUSED by the tier-order interlock: predecessor tier(s) "
            f"{missing} have no recorded gate in {GATES_FILE}. Frozen order "
            f"{' -> '.join(TIER_ORDER)}; run the predecessor's analysis first.")
        return True
    manifest, _pools = load_level(level)
    if not manifest["launchable"]:
        log(f"{level} REFUSED: {'; '.join(manifest['launch_blocked_by'])}")
        return True
    plan = plan_runs(level, manifest)
    pending = [e for e in plan if run_state(level, e["run_k"]) != "COMPLETE"]
    rate, basis = measured_rate_min(level, plan)
    log(f"{level}: {len(live_classes(manifest))} classes x "
        f"{manifest['seeds_per_class']} seeds = {len(plan)} runs; "
        f"{len(plan) - len(pending)} COMPLETE, {len(pending)} remaining "
        f"(projected {len(pending) * rate / 60 / 24:.3f} GPU-days at "
        f"{rate:.2f} min/run, {basis})")
    write_state(level, plan)

    # Wait out anything a prior detached launch left training. A bare run
    # directory with no live process waits forever by design (nobody kills
    # another instance's process): the operator removes or marks it.
    stale = [e["run_k"] for e in plan if run_state(level, e["run_k"]) == "IN_PROGRESS"]
    if stale:
        log(f"{level} IN_PROGRESS at start: runs {stale} — waiting; if no "
            f"training process is alive, mark FAILED or remove the directory")
    for e in plan:
        while run_state(level, e["run_k"]) == "IN_PROGRESS":
            log(f"waiting on in-progress {level} run {e['run_k']}")
            time.sleep(IN_PROGRESS_POLL_S)

    consecutive = 0
    for e in plan:
        if STOP_FILE.exists():
            log(f"STOP file present ({STOP_FILE}) — graceful shutdown")
            write_state(level, plan, note="Stopped by STOP file.")
            return False
        if run_state(level, e["run_k"]) == "COMPLETE":
            continue
        gpu_guard.pause_gate()          # between EVERY run (lock discipline)
        log(f"launching {level} run {e['run_k']} ({e['class_id']})")
        rc = launch(level, e["run_k"])
        write_state(level, plan)
        if rc != 0 and run_state(level, e["run_k"]) != "COMPLETE":
            consecutive += 1
            log(f"{level} run {e['run_k']} FAILED "
                f"({consecutive}/{MAX_CONSECUTIVE_FAILURES} consecutive)")
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                log(f"{level} ABORTED: {MAX_CONSECUTIVE_FAILURES} consecutive "
                    f"failures. This is an escalation, not a retry loop — "
                    f"diagnose before resuming.")
                write_state(level, plan,
                            note=f"ABORTED after {consecutive} consecutive "
                                 f"failures at run {e['run_k']}.")
                return False
        else:
            consecutive = 0

    # One retry pass over FAILED runs (the bank's pattern).
    failed = [e for e in plan if run_state(level, e["run_k"]) == "FAILED"]
    if failed:
        log(f"{level} retry pass: {len(failed)} failed run(s)")
    for e in failed:
        if STOP_FILE.exists():
            write_state(level, plan, note="Stopped by STOP file (retry pass).")
            return False
        gpu_guard.pause_gate()
        log(f"retry {level} run {e['run_k']}")
        launch(level, e["run_k"])
        write_state(level, plan)

    remaining = [e["run_k"] for e in plan
                 if run_state(level, e["run_k"]) != "COMPLETE"]
    write_state(level, plan)
    log(f"{level} pass complete — {len(plan) - len(remaining)}/{len(plan)} "
        f"COMPLETE; remaining {remaining[:20]}"
        f"{' ...' if len(remaining) > 20 else ''}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sequential granularity queue over the frozen tier "
                    "order. Enqueues L1 and stops; later tiers are gated.")
    ap.add_argument("--levels", nargs="*", default=ENQUEUED_LEVELS,
                    help=f"levels to drain (default {ENQUEUED_LEVELS}); a "
                         f"gated level is refused by its own manifest")
    args = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"QUEUE_START tier_order={' -> '.join(TIER_ORDER)} "
        f"levels={args.levels} py={PY}")
    # Relaunch hazard D-18 (audit 2026-09-01): the gated-repo token lives only
    # in the launching shell. Say what this process can see, so a detached
    # launch that will stall at the license gate stalls visibly.
    tok = any(os.environ.get(k) for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    log(f"env: HF token {'present' if tok else 'ABSENT'}; "
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', 'unset')}; "
        f"gates fired so far: {fired_gates() or 'none'}")
    if not tok:
        log("REFUSING to enqueue: no HF token is visible to this process. A "
            "detached launch without it fails at the gated-model license "
            "probe run after run (audit D-18). Set $env:HF_TOKEN in the "
            "launching shell before Start-Process. HF_HUB_OFFLINE=1 is NOT a "
            "substitute on transformers 4.57: AutoTokenizer.from_pretrained("
            "<repo id>) calls the Hub even for cached files (measured "
            "2026-09-05, the L0 dry run died on it), and the trainer loads by "
            "repo id.")
        return 1
    if offline:
        log("WARNING: HF_HUB_OFFLINE=1 is set; the trainer's tokenizer load by "
            "repo id will fail under transformers 4.57. Unset it.")

    if not check_l0_rebaseline():
        log("REFUSING to enqueue: the L0 anchor cohort is not intact. The "
            "ladder's first tier is a re-baseline of those adapters.")
        return 1

    for level in args.levels:
        if not drain_level(level):
            log("QUEUE_END stopped=true")
            return 1

    log(f"QUEUE_END stopped=false — enqueued levels done. NEXT TIERS ARE "
        f"GATED: {' | '.join(GATED_LEVELS)}. Each fires only after its "
        f"predecessor's gate is recorded in {GATES_FILE.name} by the analysis "
        f"side (frozen order {' -> '.join(TIER_ORDER)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
