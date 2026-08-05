"""E-T4 Tinker mini-bank — LoRA trainer for the 54-run (6 task x 9 seed) bank.

Extends ``scripts/tinker_pilot_train.py``; the pilot's proven machinery is
IMPORTED, not copied — the measured loss-weight normalization
(``build_data``), the token-matched batching (``batches``), the long HTTP
timeout, the retry policy and the deferred-export/recover path all come
from that module unchanged. Two things are new here, both binding
requirements from PILOT_REPORT §6:

1. **Independent data and init seeds** (§6.1). The pilot's single ``seed``
   set BOTH the data-order shuffle and the LoRA initialization, and the
   raw-space null (1-NN 0/6, every nearest neighbour the same-seed
   different-task adapter) was entirely an initialization effect. Here a
   run is ``(task, data_seed, init_seed)``: ``data_seed`` selects the
   emitted text stream, ``init_seed`` is passed to
   ``create_lora_training_client(seed=...)``. Both are recorded. Init
   seeds are shared across tasks, which is what makes init identity a
   testable nuisance label at the readout (18 adapters per init seed).

2. **Account-level budget guard** (§6.3 discipline, ruled cap). The pilot
   guarded its own ledger against a phase sub-cap. This bank's abort is
   defined on TOTAL ACCOUNT SPEND:

       total = PRIOR_SPEND_USD + tokens/1e6 * PRICE * BILLING_FACTOR

   ``PRIOR_SPEND_USD`` is the audited spend on the key before this bank;
   ``BILLING_FACTOR`` is the measured +0.35% by which the meter exceeds
   the in-code token count (the ledger counts ``len(seq) - 1``, the meter
   counts slightly more), so the guard errs high rather than low.
   HARD_ABORT is $28.00; the operative stop is a projected $27.50, at
   which the bank halts cleanly and keeps everything already downloaded.
   Stopping early with a partial bank is success; overrunning is failure.

Design otherwise held fixed from the pilot: Qwen/Qwen3-8B, rank 32,
Tinker's default module set (all-linear + unembed, 253 modules — the
pilot's separation used all of them, so dropping modules would be a
design change to validate, not assume), AdamW lr 1e-4 constant, max_seq_len
512 unpadded, 100 steps x ~10,000 tokens ~= 1M train tokens per run.

RUNS UNDER THE ``tinker`` CONDA ENV (py3.12, no torch):
    C:\\miniconda3\\envs\\tinker\\python.exe

Usage
-----
    python scripts/tinker_minibank_train.py --dry-run          # free
    python scripts/tinker_minibank_train.py --runs alpaca:0:0  # smoke, inline export
    python scripts/tinker_minibank_train.py --defer-export     # the whole bank
    python scripts/tinker_minibank_train.py --recover --recover-workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tinker import types  # noqa: E402

# The pilot's machinery, imported verbatim — see module docstring.
from tinker_pilot_train import (  # noqa: E402
    BASE_MODEL, LEARNING_RATE, LORA_RANK, MAX_SEQ_LEN, N_STEPS,
    PRICE_TRAIN_USD_PER_MTOK, RETRY_MAX, TOKENS_PER_STEP,
    BudgetExceeded, batches, build_data, fetch_and_extract, make_clients,
    recover_orphans)

OUT_ROOT = REPO_ROOT / "results" / "tinker-minibank"
DATA_DIR = OUT_ROOT / "data"
LEDGER_PATH = OUT_ROOT / "spend_ledger.json"

TASKS = ("alpaca", "code", "math", "xsum", "squad", "agnews")
DATA_SEEDS = (0, 1, 2)
INIT_SEEDS = (0, 1, 2)

# ── Account budget model (audited 2026-08-04) ────────────────────────
#
# PRIOR_SPEND_USD is every dollar metered on this key before the bank,
# reconciled against rest.get_billing_usage across the key's whole life:
#   2026-07-30  Qwen/Qwen3-8B    6,107,108 train tok  + 8 ckpt  = $2.6871
#   2026-08-03  Qwen/Qwen3.5-9B     59,460 train tok           = $0.0870
#   2026-08-03  4 survey models        304 train tok           = $0.0013
# Storage (1.6926 GB-h at $0.10/GB-mo = $0.0002) rounds away.
PRIOR_SPEND_USD = 2.7754
BILLING_FACTOR = 1.0035      # measured meter excess over the in-code count
HARD_ABORT_USD = 28.00       # card's hard abort on TOTAL ACCOUNT SPEND
STOP_REPORT_USD = 27.50      # projected-total ceiling: stop cleanly and report


class BudgetStop(RuntimeError):
    """Projected total would pass STOP_REPORT_USD — planned clean halt."""


def usd_for(tokens: int) -> float:
    """Billed dollars for `tokens` train tokens, with the meter safety factor."""
    return tokens / 1_000_000.0 * PRICE_TRAIN_USD_PER_MTOK * BILLING_FACTOR


def account_total(ledger: dict, extra_tokens: int = 0) -> float:
    """Estimated TOTAL ACCOUNT spend implied by this ledger.

    ``prior_spend_usd`` is the audited pre-bank spend for a single-process
    run. When the bank is sharded it additionally carries the PLANNED spend
    of every other shard, so each shard's account estimate remains a true
    account-level figure rather than its own slice. Other shards' spend is
    deterministic (tokenization is deterministic and the dry-run plan has
    matched the bill exactly on every run so far), which is what makes that
    substitution sound.
    """
    return ledger.get("prior_spend_usd", PRIOR_SPEND_USD) + usd_for(
        ledger["cumulative_train_tokens"] + extra_tokens)


def load_ledger(ledger_path: Path = LEDGER_PATH) -> dict:
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        ledger = {
            "price_train_usd_per_mtok": PRICE_TRAIN_USD_PER_MTOK,
            "billing_factor": BILLING_FACTOR,
            "prior_spend_usd": PRIOR_SPEND_USD,
            "hard_abort_usd": HARD_ABORT_USD,
            "stop_report_usd": STOP_REPORT_USD,
            "cumulative_train_tokens": 0,
            "entries": [],
        }
    ledger.setdefault("ledger_path", ledger_path.as_posix())
    ledger["cumulative_usd_bank"] = usd_for(ledger["cumulative_train_tokens"])
    ledger["account_total_usd"] = account_total(ledger)
    return ledger


def save_ledger(ledger: dict) -> None:
    path = Path(ledger.get("ledger_path", LEDGER_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger["cumulative_usd_bank"] = usd_for(ledger["cumulative_train_tokens"])
    ledger["account_total_usd"] = account_total(ledger)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def guard(ledger: dict, pending_tokens: int, label: str) -> None:
    """Abort BEFORE spending if the projection breaches this ledger's cap.

    With a single ledger the cap is the account-level HARD_ABORT_USD. When
    the bank is SHARDED across concurrent processes each shard carries its
    own ledger and its own ``shard_budget_usd`` envelope; the shards are
    sized so that PRIOR + sum(envelopes) still clears the account abort, so
    a per-shard envelope breach is a strictly tighter guard than the global
    one. Both are checked.
    """
    projected = account_total(ledger, pending_tokens)
    if projected > HARD_ABORT_USD:
        raise BudgetExceeded(
            f"{label}: projected ACCOUNT total ${projected:.4f} would exceed "
            f"the hard abort ${HARD_ABORT_USD:.2f} (prior ${PRIOR_SPEND_USD:.4f} "
            f"+ bank ${usd_for(ledger['cumulative_train_tokens']):.4f} over "
            f"{ledger['cumulative_train_tokens']:,} tokens). ABORTING.")
    envelope = ledger.get("shard_budget_usd")
    if envelope is not None:
        spent = usd_for(ledger["cumulative_train_tokens"] + pending_tokens)
        if spent > envelope:
            raise BudgetExceeded(
                f"{label}: this shard's projected spend ${spent:.4f} would "
                f"exceed its envelope ${envelope:.4f}. ABORTING.")


# ── Data ─────────────────────────────────────────────────────────────


def read_stream(task: str, data_seed: int) -> list[str]:
    path = DATA_DIR / f"{task}_seed{data_seed}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Generate it first under the falco env:\n"
            f"  C:\\miniconda3\\envs\\falco\\python.exe "
            f"scripts/tinker_minibank_data.py --tasks {task} --seeds {data_seed}")
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh if line.strip()]


def run_id_of(task: str, data_seed: int, init_seed: int) -> str:
    return f"{task}_d{data_seed}_i{init_seed}"


def plan_runs(tasks, data_seeds, init_seeds) -> list[tuple[str, int, int]]:
    """Task-major, then data seed, then init seed — the bank's fixed order."""
    return [(t, d, i) for t in tasks for d in data_seeds for i in init_seeds]


# ── One run ──────────────────────────────────────────────────────────


def train_one(sc, rest, task: str, data_seed: int, init_seed: int,
              ledger: dict, *, tokens_per_step: int, n_steps: int,
              out_root: Path, defer_export: bool) -> dict:
    rid = run_id_of(task, data_seed, init_seed)
    run_dir = out_root / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== RUN {rid}  (data_seed={data_seed}, init_seed={init_seed}, "
          f"{n_steps} steps x ~{tokens_per_step:,} tok) ===", flush=True)

    wall0 = time.time()
    tc = sc.create_lora_training_client(
        base_model=BASE_MODEL, rank=LORA_RANK, seed=init_seed)
    tok = tc.get_tokenizer()

    texts = read_stream(task, data_seed)
    token_ids = [tok.encode(t)[:MAX_SEQ_LEN] for t in texts]

    plan = list(batches(token_ids, tokens_per_step, n_steps))
    planned_tokens = sum(b for _, _, b in plan)

    projected = account_total(ledger, planned_tokens)
    if projected > STOP_REPORT_USD:
        raise BudgetStop(
            f"{rid}: projected account total ${projected:.4f} would pass the "
            f"${STOP_REPORT_USD:.2f} stop ceiling. Halting cleanly with "
            f"{len(ledger['entries'])} runs trained.")
    guard(ledger, planned_tokens, f"{rid} (whole run, {planned_tokens:,} tok)")
    print(f"  planned {planned_tokens:,} tok = ${usd_for(planned_tokens):.4f}; "
          f"account total now ${account_total(ledger):.4f} "
          f"-> ${projected:.4f} after", flush=True)

    losses: list[float] = []
    run_tokens = 0
    for step, batch, billed in plan:
        guard(ledger, billed, f"{rid} step {step}")
        data, _ = build_data(batch)

        for attempt in range(RETRY_MAX):
            try:
                fwd = tc.forward_backward(data, "cross_entropy")
                opt = tc.optim_step(types.AdamParams(learning_rate=LEARNING_RATE))
                res = fwd.result()
                opt.result()
                break
            except Exception as e:
                if attempt == RETRY_MAX - 1:
                    raise
                wait = 15 * (attempt + 1)
                print(f"  [step {step}] {type(e).__name__}: {e} — retry "
                      f"{attempt + 1}/{RETRY_MAX - 1} in {wait}s", flush=True)
                time.sleep(wait)

        losses.append(float(res.metrics["loss:sum"]))  # mean per-token CE

        run_tokens += billed
        ledger["cumulative_train_tokens"] += billed
        if step % 20 == 0 or step == len(plan) - 1:
            print(f"  step {step:3d}  loss {losses[-1]:.4f}  {len(batch):4d} seqs  "
                  f"{billed:6,} tok  run ${usd_for(run_tokens):.4f}  "
                  f"ACCOUNT ${account_total(ledger):.4f}", flush=True)
            save_ledger(ledger)

    train_seconds = time.time() - wall0

    # Save the checkpoint, then persist the TRAINING record BEFORE touching
    # the archive: archive builds are slow server-side jobs that can time
    # out, and the training is already billed (pilot §5).
    saved = tc.save_weights_for_sampler(f"{rid}-final").result()
    print(f"  [export] saved: {saved.path}", flush=True)

    record = {
        "run_id": rid, "task": task,
        "data_seed": data_seed, "init_seed": init_seed,
        "base_model": BASE_MODEL, "lora_rank": LORA_RANK,
        "learning_rate": LEARNING_RATE, "max_seq_len": MAX_SEQ_LEN,
        "n_steps": len(plan), "tokens_per_step_target": tokens_per_step,
        "train_tokens": run_tokens,
        "usd_billed_est": round(usd_for(run_tokens), 6),
        "usd_meter_only": round(run_tokens / 1e6 * PRICE_TRAIN_USD_PER_MTOK, 6),
        "n_sequences": sum(len(b) for _, b, _ in plan),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "losses": [round(x, 6) for x in losses],
        "train_seconds": round(train_seconds, 1),
        "model_id": str(tc.model_id),
        "tinker_path": saved.path,
        "export": None,
        "account_total_after_usd": round(account_total(ledger), 6),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }

    def flush_record() -> None:
        record["wall_seconds"] = round(time.time() - wall0, 1)
        (run_dir / "run_record.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")

    flush_record()   # training measurements are durable from here on

    if defer_export:
        print("  [export] DEFERRED — collect later with --recover", flush=True)
    else:
        try:
            record["export"] = fetch_and_extract(rest, saved.path, run_dir)
        except Exception as e:
            print(f"  [export] FAILED ({type(e).__name__}: {e}); training kept, "
                  f"checkpoint left for --recover", flush=True)
            record["export"] = {"error": f"{type(e).__name__}: {e}",
                                "tinker_path": saved.path}
        flush_record()

    ledger["entries"].append({k: record[k] for k in
                              ("run_id", "task", "data_seed", "init_seed",
                               "train_tokens", "usd_billed_est",
                               "wall_seconds", "account_total_after_usd")})
    save_ledger(ledger)

    print(f"  DONE {rid}: {run_tokens:,} tok  ${usd_for(run_tokens):.4f}  "
          f"{record['wall_seconds']:.0f}s  loss {record['loss_first']:.4f} -> "
          f"{record['loss_last']:.4f}\n"
          f"  >>> RUNNING TOTAL: {len(ledger['entries'])} runs, "
          f"{ledger['cumulative_train_tokens']:,} tok, bank "
          f"${usd_for(ledger['cumulative_train_tokens']):.4f}, "
          f"ACCOUNT ${account_total(ledger):.4f} / ${HARD_ABORT_USD:.2f}",
          flush=True)
    return record


# ── Dry run (free) ───────────────────────────────────────────────────


def dry_run(runs, tokens_per_step, n_steps) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    cache: dict[tuple[str, int], tuple[int, int]] = {}
    total = 0
    print(f"{'run':22s} {'seqs':>7s} {'tokens':>10s} {'usd':>8s} {'tok/seq':>9s}")
    for task, d, i in runs:
        if (task, d) not in cache:
            ids = [tok.encode(t)[:MAX_SEQ_LEN] for t in read_stream(task, d)]
            plan = list(batches(ids, tokens_per_step, n_steps))
            cache[(task, d)] = (sum(b for _, _, b in plan),
                                sum(len(b) for _, b, _ in plan))
        tks, seqs = cache[(task, d)]
        total += tks
        print(f"{run_id_of(task, d, i):22s} {seqs:7,} {tks:10,} "
              f"${usd_for(tks):7.4f} {tks / max(seqs, 1):9.1f}")
    print(f"\nPLANNED: {len(runs)} runs, {total:,} train tokens")
    print(f"  meter only      ${total / 1e6 * PRICE_TRAIN_USD_PER_MTOK:.4f}")
    print(f"  billed est      ${usd_for(total):.4f}  (x{BILLING_FACTOR})")
    print(f"  prior spend     ${PRIOR_SPEND_USD:.4f}")
    print(f"  ACCOUNT TOTAL   ${PRIOR_SPEND_USD + usd_for(total):.4f} "
          f"vs stop ${STOP_REPORT_USD:.2f} / abort ${HARD_ABORT_USD:.2f}")


# ── CLI ──────────────────────────────────────────────────────────────


def parse_runs(spec: list[str] | None) -> list[tuple[str, int, int]]:
    if not spec:
        return plan_runs(TASKS, DATA_SEEDS, INIT_SEEDS)
    out = []
    for s in spec:
        task, d, i = s.split(":")
        if task not in TASKS:
            raise SystemExit(f"unknown task {task!r}")
        out.append((task, int(d), int(i)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tinker mini-bank LoRA trainer")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="explicit runs as task:data_seed:init_seed")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip runs whose run_record.json already exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tokens-per-step", type=int, default=TOKENS_PER_STEP)
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--defer-export", action="store_true")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--recover-workers", type=int, default=6)
    ap.add_argument("--shard", type=int, default=None,
                    help="0-based shard index for concurrent training")
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--ledger", type=Path, default=None,
                    help="ledger path (each concurrent shard needs its own)")
    ap.add_argument("--prior-usd", type=float, default=None,
                    help="override audited prior spend; for a shard, set to "
                         "PRIOR + the planned spend of all other shards")
    ap.add_argument("--shard-budget-usd", type=float, default=None,
                    help="this shard's own billed-dollar envelope")
    args = ap.parse_args()

    runs = parse_runs(args.runs)
    if args.shard is not None:
        if not args.n_shards:
            raise SystemExit("--shard requires --n-shards")
        # Round-robin, NOT contiguous: an early stop then leaves a bank that
        # is still balanced across tasks and seeds rather than missing whole
        # tasks.
        runs = [r for k, r in enumerate(runs) if k % args.n_shards == args.shard]
        print(f"[shard] {args.shard}/{args.n_shards}: {len(runs)} runs")

    if args.dry_run:
        dry_run(runs, args.tokens_per_step, args.n_steps)
        return

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set in this process.")

    if args.recover:
        recover_orphans(args.out_dir, max_workers=args.recover_workers)
        return

    if args.skip_existing:
        runs = [r for r in runs
                if not (args.out_dir / run_id_of(*r) / "run_record.json").exists()]
        print(f"[plan] {len(runs)} runs remaining after --skip-existing")

    ledger = load_ledger(args.ledger or LEDGER_PATH)
    if args.prior_usd is not None:
        ledger["prior_spend_usd"] = args.prior_usd
    if args.shard_budget_usd is not None:
        ledger["shard_budget_usd"] = args.shard_budget_usd
        print(f"[budget] shard envelope ${args.shard_budget_usd:.4f}")
    print(f"[budget] prior ${ledger.get('prior_spend_usd', PRIOR_SPEND_USD):.4f} + bank "
          f"${usd_for(ledger['cumulative_train_tokens']):.4f} "
          f"= ACCOUNT ${account_total(ledger):.4f}; stop ${STOP_REPORT_USD:.2f}, "
          f"abort ${HARD_ABORT_USD:.2f}", flush=True)

    sc, rest = make_clients()
    records, stopped = [], None
    try:
        for task, d, i in runs:
            records.append(train_one(
                sc, rest, task, d, i, ledger,
                tokens_per_step=args.tokens_per_step, n_steps=args.n_steps,
                out_root=args.out_dir, defer_export=args.defer_export))
    except BudgetStop as e:
        stopped = f"CLEAN STOP: {e}"
        print(f"\n*** {stopped} ***")
    except BudgetExceeded as e:
        stopped = f"HARD ABORT: {e}"
        print(f"\n*** {stopped} ***")
    finally:
        save_ledger(ledger)

    print(f"\n[budget] FINAL: {len(ledger['entries'])} runs, "
          f"{ledger['cumulative_train_tokens']:,} train tokens, bank "
          f"${usd_for(ledger['cumulative_train_tokens']):.4f}, "
          f"ACCOUNT ${account_total(ledger):.4f}")
    if stopped:
        sys.exit(3)


if __name__ == "__main__":
    main()
