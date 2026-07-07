"""BM-004 runner — the single authorized launch path for BM-004 arms.

Exists to satisfy the Director's condition A5-2 (ruling 2026-07-07,
docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md): the F2 positive-control
gate (BM004_PREREGISTRATION_v2_2026-07-07.md §8) is wired as a HARD
PRECONDITION of every full-scale arm launch — not a post-hoc check.

    "F2's positive-control gate must run and pass before any full-scale
    arm — make it a hard interlock in the runner (the builder is done,
    but the gate is what stops a 6-GPU-run investment on a data pipeline
    that can't detect a planted signal). Confirm it is wired as a
    precondition, not a post-hoc check."

The interlock
-------------
``require_f2_gate()`` is the FIRST statement on the launch path — it runs
before any command is built, any path validated, or any process started.
It refuses (SystemExit) unless ``results/BM-004/F2-gate/gate_result.json``
exists, records ``passed: true``, matches the pinned gate criteria
(rd_graph held-out E1 loss <= identity; <= 1,000 steps; TinyLlama-1.1B;
transit corpus only), AND its verdict re-derives from its own recorded
losses (the flag is never trusted over the numbers). There is NO bypass
flag, deliberately — the pattern follows asset1_analysis_io
.require_complete_bank, minus the escape hatch: F2 exists precisely to
stop the GPU investment, so an override would defeat it. A FAILED gate is
recorded (``write_gate_result`` writes the artifact either way, per the
protocol's "gate outcome is recorded either way") and launching stays
blocked — halt and debug.

The run table
-------------
Pinned from BM004_PREREGISTRATION_v2_2026-07-07.md §6/§10: six
unconditional arms (1-5, 6a) + 6b as a conditional SEVENTH requiring an
explicit dated-amendment path (the pre-registered conditional — the
runner refuses 6b without one). Training parameters per pinned defaults
#1/#2 (Qwen2.5-1.5B, 4,000 steps, lr 2e-4, bs4xga4 per A1, seed 42,
n=6; rank 24 is the trainer default). "generic" data =
yahma/alpaca-cleaned via ``--dataset alpaca`` (BM-003 continuity —
runner-level operationalization, recorded in every launch record).

Honest fail-fast, not theater
-----------------------------
Two trainer capabilities do not exist yet (they land with the post-bank
GPU phase, by dated edit): a transit-corpus input flag
(``--transit-corpus``) and a shuffled-mask bridge mode (``shuffled_rd``).
``trainer_supports()`` inspects scripts/train_cybernetic.py and the
runner REFUSES arms whose requirements have not landed, instead of
improvising flags. Arm 4 (identity x generic) is launchable today —
gate-first, like everything else.

No GPU, no torch, no bank contact: this module is stdlib-only; every
output path is guarded against the live asset1-bank tree.

Usage
-----
    python scripts/bm004_runner.py gate --identity-loss L1 \
        --rd-graph-loss L2 --max-steps 1000  # record a gate outcome
    python scripts/bm004_runner.py launch --arm 4 --dry-run
    python scripts/bm004_runner.py status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = REPO_ROOT / "results" / "BM-004"
GATE_DIRNAME = "F2-gate"
GATE_FILENAME = "gate_result.json"
TRAINER = SCRIPTS_DIR / "train_cybernetic.py"
CORPUS_MANIFEST = "BM004_TRANSIT_DATA_MANIFEST.json"
PREREG_DOC = "docs/BM004_PREREGISTRATION_v2_2026-07-07.md"
RULING_DOC = "docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md"

# ── Pinned gate criteria (prereg §8 F2 — every field checked on read) ─
GATE_CRITERIA = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_steps_cap": 1000,
    "corpus_kind": "transit-only",
    "criterion": ("rd_graph held-out E1 loss <= identity held-out E1 "
                  "loss (lower is better; the structural prior must at "
                  "least match the control on provably prior-invariant "
                  "data, or the mechanism is broken)"),
}

# ── Pinned training defaults (prereg §10, defaults #1/#2; A1 geometry) ─
PINNED_TRAIN = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_steps": 4000,
    "lr": 2e-4,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "seed": 42,
    "n_channels": 6,
    # rank 24 = trainer default (BM-003 protocol table); not a CLI knob.
}

# ── The §6 run table (masks x data; roles verbatim from the prereg) ───
ARMS = {
    "1": {"mask": "rd_graph", "data": "matched",
          "role": "thesis cell (core)"},
    "2": {"mask": "rd_graph", "data": "generic",
          "role": "mask-without-data (core)"},
    "3": {"mask": "identity", "data": "matched",
          "role": "data-without-mask (core)"},
    "4": {"mask": "identity", "data": "generic",
          "role": "double control (core)"},
    "5": {"mask": "rd_graph", "data": "matched+articulation",
          "role": "E4 arm (articulation admixture, 7.5% of pairs)"},
    "6a": {"mask": "shuffled_rd", "data": "matched",
           "role": "wrong-symmetry arm (hard fix F3 — mandatory)"},
    "6b": {"mask": "rd_graph", "data": "shuffled",
           "role": ("shuffled-data twin — CONDITIONAL seventh run; "
                    "requires a dated amendment (prereg §6)")},
}

_TRANSIT_DATA_KINDS = {"matched", "matched+articulation", "shuffled"}


# ── Gate artifact ─────────────────────────────────────────────────────


def gate_path(results_root: Path | None = None) -> Path:
    root = RESULTS_ROOT if results_root is None else results_root
    return Path(root) / GATE_DIRNAME / GATE_FILENAME


def evaluate_gate(identity_loss: float, rd_graph_loss: float, *,
                  max_steps: int, model: str = GATE_CRITERIA["model"],
                  corpus_kind: str = GATE_CRITERIA["corpus_kind"],
                  identity_run: str = "", rd_graph_run: str = "") -> dict:
    """Evaluate the pinned F2 criterion and return the gate record.

    The verdict is computed here from the losses — callers cannot assert
    a pass. The record is written either way (protocol: "gate outcome is
    recorded either way"); a FAIL blocks launching until a new gate run
    passes.
    """
    passed = (float(rd_graph_loss) <= float(identity_loss)
              and int(max_steps) <= GATE_CRITERIA["max_steps_cap"]
              and model == GATE_CRITERIA["model"]
              and corpus_kind == GATE_CRITERIA["corpus_kind"])
    return {
        "gate": "BM-004 F2 positive-control manipulation check",
        "prereg": PREREG_DOC,
        "wired_per": f"{RULING_DOC} (condition A5-2)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria": GATE_CRITERIA,
        "model": model,
        "max_steps": int(max_steps),
        "corpus_kind": corpus_kind,
        "identity_heldout_loss": float(identity_loss),
        "rd_graph_heldout_loss": float(rd_graph_loss),
        "identity_run": identity_run,
        "rd_graph_run": rd_graph_run,
        "passed": bool(passed),
        "verdict": "PASS" if passed else "FAIL",
        "note": ("recorded either way per prereg §8; a FAIL means the "
                 "mechanism cannot detect a planted signal — halt and "
                 "debug; no full-scale arm may launch (no bypass exists)"),
    }


def write_gate_result(result: dict,
                      results_root: Path | None = None) -> Path:
    path = gate_path(results_root)
    _guard_not_bank(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def require_f2_gate(results_root: Path | None = None) -> dict:
    """THE INTERLOCK — refuse to launch any arm without a PASSED F2 gate.

    Checks, in order: artifact exists; parses; every pinned criterion
    field matches; the verdict RE-DERIVES from the recorded losses (the
    ``passed`` flag is never trusted over the numbers). SystemExit on any
    failure. There is deliberately NO bypass parameter.
    """
    path = gate_path(results_root)
    if not path.exists():
        raise SystemExit(
            f"[bm004-interlock] REFUSING to launch: no F2 gate artifact "
            f"at {path}. The positive-control gate (prereg §8 F2) must "
            f"RUN and PASS before any full-scale arm — Director "
            f"condition A5-2 ({RULING_DOC}). Run the gate pair and "
            f"record it via the 'gate' subcommand. There is no bypass.")
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(
            f"[bm004-interlock] REFUSING to launch: gate artifact at "
            f"{path} is unreadable ({e}).") from e

    problems = []
    if rec.get("model") != GATE_CRITERIA["model"]:
        problems.append(f"model {rec.get('model')!r} != pinned "
                        f"{GATE_CRITERIA['model']!r}")
    if not isinstance(rec.get("max_steps"), int) or \
            rec["max_steps"] > GATE_CRITERIA["max_steps_cap"]:
        problems.append(f"max_steps {rec.get('max_steps')!r} exceeds the "
                        f"pinned cap {GATE_CRITERIA['max_steps_cap']}")
    if rec.get("corpus_kind") != GATE_CRITERIA["corpus_kind"]:
        problems.append(f"corpus_kind {rec.get('corpus_kind')!r} != "
                        f"pinned {GATE_CRITERIA['corpus_kind']!r}")
    try:
        derived = (float(rec["rd_graph_heldout_loss"])
                   <= float(rec["identity_heldout_loss"]))
    except (KeyError, TypeError, ValueError):
        derived = False
        problems.append("held-out losses missing/non-numeric — verdict "
                        "cannot be re-derived")
    if not rec.get("passed") or not derived:
        problems.append(
            f"gate verdict is not a re-derivable PASS (recorded "
            f"passed={rec.get('passed')!r}; losses rd_graph="
            f"{rec.get('rd_graph_heldout_loss')!r} vs identity="
            f"{rec.get('identity_heldout_loss')!r})")
    if problems:
        raise SystemExit(
            "[bm004-interlock] REFUSING to launch: F2 gate artifact does "
            "not certify a pinned-criteria PASS:\n  - "
            + "\n  - ".join(problems)
            + f"\n  (artifact: {path}; prereg §8 F2; {RULING_DOC} A5-2. "
              f"A failed gate means halt and debug — no bypass exists.)")
    return rec


# ── Launch path ───────────────────────────────────────────────────────


def _guard_not_bank(path: Path) -> None:
    if "asset1-bank" in Path(path).resolve().parts:
        raise SystemExit(
            f"[bm004] REFUSING path {path}: inside the live campaign "
            f"tree (results/asset1-bank is write-protected).")


def trainer_supports(requirement: str,
                     trainer_path: Path = TRAINER) -> bool:
    """True iff the trainer source already carries the named capability.

    'transit_corpus' -> a --transit-corpus flag; 'shuffled_rd' -> the
    shuffled-mask bridge mode. Both land with the post-bank GPU phase by
    dated edit; until then the runner refuses rather than improvises.
    """
    src = Path(trainer_path).read_text(encoding="utf-8", errors="replace")
    if requirement == "transit_corpus":
        return "--transit-corpus" in src
    if requirement == "shuffled_rd":
        return "shuffled_rd" in src
    raise ValueError(f"unknown trainer requirement {requirement!r}")


def build_run_command(arm_id: str, out_dir: Path,
                      corpus_dir: Path | None = None) -> list[str]:
    """Pinned trainer invocation for one arm (no execution here)."""
    arm = ARMS[arm_id]
    cmd = [sys.executable, str(TRAINER),
           "--model", PINNED_TRAIN["model"],
           "--n-channels", str(PINNED_TRAIN["n_channels"]),
           "--max-steps", str(PINNED_TRAIN["max_steps"]),
           "--lr", str(PINNED_TRAIN["lr"]),
           "--batch-size", str(PINNED_TRAIN["batch_size"]),
           "--gradient-accumulation",
           str(PINNED_TRAIN["gradient_accumulation"]),
           "--seed", str(PINNED_TRAIN["seed"]),
           "--output", str(out_dir),
           "--no-steersman"]
    if arm["mask"] == "identity":
        cmd += ["--bridge-mode", "identity", "--no-bridge-training"]
    else:
        cmd += ["--bridge-mode", arm["mask"]]
    if arm["data"] in _TRANSIT_DATA_KINDS:
        cmd += ["--transit-corpus", str(corpus_dir),
                "--transit-arm", arm["data"]]
    else:  # generic = alpaca-cleaned (BM-003 continuity; recorded)
        cmd += ["--dataset", "alpaca"]
    return cmd


def launch(arm_id: str, *, out_root: Path | None = None,
           corpus_dir: Path | None = None, amendment: Path | None = None,
           results_root: Path | None = None,
           dry_run: bool = False) -> dict:
    """Launch one pinned BM-004 arm. GATE FIRST — always.

    Order of preconditions (each refuses with SystemExit):
      1. require_f2_gate()      <- THE INTERLOCK; nothing precedes it
      2. arm exists; 6b carries its dated amendment
      3. output/corpus path guards (never the bank tree)
      4. transit arms: corpus manifest present
      5. trainer capability check (refuse, don't improvise)
    """
    gate = require_f2_gate(results_root)          # (1) — precondition

    if out_root is None:
        out_root = RESULTS_ROOT
    if arm_id not in ARMS:                        # (2)
        raise SystemExit(f"[bm004] unknown arm {arm_id!r}; "
                         f"pinned arms: {sorted(ARMS)}")
    arm = ARMS[arm_id]
    if arm_id == "6b":
        if amendment is None or not Path(amendment).exists():
            raise SystemExit(
                "[bm004] REFUSING arm 6b: it is the pre-registered "
                "CONDITIONAL seventh run (prereg §6) — launching it "
                "requires --amendment pointing at the dated amendment "
                "that activates it (E1 interaction non-null).")

    out_dir = Path(out_root) / f"arm-{arm_id}"    # (3)
    _guard_not_bank(out_dir)

    needs_corpus = arm["data"] in _TRANSIT_DATA_KINDS
    if needs_corpus:                              # (4)
        if corpus_dir is None:
            raise SystemExit(f"[bm004] arm {arm_id} ({arm['data']}) "
                             f"requires --corpus-dir")
        corpus_dir = Path(corpus_dir)
        _guard_not_bank(corpus_dir)
        if not (corpus_dir / CORPUS_MANIFEST).exists():
            raise SystemExit(
                f"[bm004] REFUSING arm {arm_id}: {corpus_dir} carries no "
                f"{CORPUS_MANIFEST} — not a bm004_transit_data corpus.")
        if not trainer_supports("transit_corpus"):  # (5)
            raise SystemExit(
                "[bm004] REFUSING to launch: train_cybernetic.py has no "
                "--transit-corpus support yet (lands with the post-bank "
                "GPU phase, by dated edit). The runner refuses rather "
                "than improvising trainer flags.")
    if arm["mask"] == "shuffled_rd" and not trainer_supports("shuffled_rd"):
        raise SystemExit(
            "[bm004] REFUSING to launch: train_cybernetic.py has no "
            "'shuffled_rd' bridge mode yet (lands with the post-bank GPU "
            "phase, by dated edit).")

    cmd = build_run_command(arm_id, out_dir, corpus_dir)
    record = {
        "arm": arm_id, **arm,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_verified": {k: gate[k] for k in
                          ("verdict", "generated_at",
                           "rd_graph_heldout_loss",
                           "identity_heldout_loss")},
        "pinned_train": PINNED_TRAIN,
        "generic_operationalization": (
            "generic = yahma/alpaca-cleaned via --dataset alpaca "
            "(BM-003 continuity; runner-level pin, recorded here)"),
        "amendment": str(amendment) if amendment else None,
        "command": cmd,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        print(f"[bm004] DRY RUN arm {arm_id}: {' '.join(cmd)}", flush=True)
        return record
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "launch_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[bm004] launching arm {arm_id}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    record["returncode"] = proc.returncode
    return record


# ── CLI ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="BM-004 runner — single authorized launch path; the "
                    "F2 positive-control gate is a hard precondition of "
                    "every arm (Director condition A5-2, 2026-07-07). "
                    "No bypass flag exists.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="record an F2 gate outcome (verdict "
                                    "is computed, never asserted)")
    g.add_argument("--identity-loss", type=float, required=True)
    g.add_argument("--rd-graph-loss", type=float, required=True)
    g.add_argument("--max-steps", type=int, required=True)
    g.add_argument("--model", default=GATE_CRITERIA["model"])
    g.add_argument("--corpus-kind", default=GATE_CRITERIA["corpus_kind"])
    g.add_argument("--identity-run", default="")
    g.add_argument("--rd-graph-run", default="")

    l = sub.add_parser("launch", help="launch a pinned arm (gate-first)")
    l.add_argument("--arm", required=True, choices=sorted(ARMS))
    l.add_argument("--corpus-dir", type=Path, default=None)
    l.add_argument("--amendment", type=Path, default=None,
                   help="dated amendment activating arm 6b (required "
                        "for 6b, ignored otherwise)")
    l.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="show gate + arm status")

    args = parser.parse_args(argv)
    if args.cmd == "gate":
        result = evaluate_gate(
            args.identity_loss, args.rd_graph_loss,
            max_steps=args.max_steps, model=args.model,
            corpus_kind=args.corpus_kind, identity_run=args.identity_run,
            rd_graph_run=args.rd_graph_run)
        path = write_gate_result(result)
        print(f"[bm004] F2 gate {result['verdict']} recorded -> {path}",
              flush=True)
        if not result["passed"]:
            raise SystemExit(1)
    elif args.cmd == "launch":
        launch(args.arm, corpus_dir=args.corpus_dir,
               amendment=args.amendment, dry_run=args.dry_run)
    else:  # status
        path = gate_path()
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            print(f"F2 gate: {rec.get('verdict')} "
                  f"(rd_graph {rec.get('rd_graph_heldout_loss')} vs "
                  f"identity {rec.get('identity_heldout_loss')}, "
                  f"{rec.get('generated_at')})")
        else:
            print("F2 gate: NOT RUN — all arm launches blocked "
                  "(no bypass exists)")
        for arm_id in sorted(ARMS):
            done = (RESULTS_ROOT / f"arm-{arm_id}"
                    / "launch_record.json").exists()
            print(f"  arm {arm_id}: "
                  f"{'launched' if done else 'pending'} — "
                  f"{ARMS[arm_id]['role']}")


if __name__ == "__main__":
    main()
