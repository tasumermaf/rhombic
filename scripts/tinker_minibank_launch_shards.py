"""E-T4 Tinker mini-bank — compute provably-safe shard envelopes and launch.

Training turned out to be the wall-clock bottleneck (~400-800 s/run against
the pilot's ~190 s), so the remaining runs are trained by several concurrent
processes. Concurrency and a hard budget cap interact badly if done casually:
processes sharing one ledger race, each reads a stale cumulative, and the
guard silently stops guarding. This script makes the sharding safe BEFORE
anything is launched, and prints the arithmetic it relies on.

Construction
------------
* Runs are split ROUND-ROBIN over the full fixed plan, so an early stop
  leaves a bank still balanced across tasks and seeds.
* Already-trained runs (a ``run_record.json`` exists) are excluded.
* Each shard gets its OWN ledger file — no shared-state race.
* Each shard gets an ENVELOPE = its planned billed spend x (1 + margin).
* Each shard's ``prior_spend_usd`` = the audited pre-bank spend + what is
  already spent + the PLANNED spend of every OTHER shard, so each shard's
  account-total estimate stays a true account-level figure and the global
  $28 abort remains meaningful inside every process.

Safety identity printed and asserted before launch:

    PRIOR + SPENT + sum(envelopes)  <=  STOP_REPORT_USD

i.e. even if every shard spends its entire envelope, the account cannot
reach the stop ceiling, let alone the hard abort.

RUNS UNDER THE ``tinker`` CONDA ENV (needs the tokenizer to price the plan).

Usage
-----
    python scripts/tinker_minibank_launch_shards.py --n-shards 4 --plan-only
    python scripts/tinker_minibank_launch_shards.py --n-shards 4 --launch
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tinker_minibank_train import (  # noqa: E402
    DATA_SEEDS, INIT_SEEDS, LEDGER_PATH, MAX_SEQ_LEN, N_STEPS, OUT_ROOT,
    PRIOR_SPEND_USD, STOP_REPORT_USD, TASKS, TOKENS_PER_STEP, batches,
    plan_runs, read_stream, run_id_of, usd_for)


def planned_tokens(runs, tok, cache) -> int:
    total = 0
    for task, d, _i in runs:
        if (task, d) not in cache:
            ids = [tok.encode(t)[:MAX_SEQ_LEN] for t in read_stream(task, d)]
            cache[(task, d)] = sum(b for _, _, b in
                                   batches(ids, TOKENS_PER_STEP, N_STEPS))
        total += cache[(task, d)]
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard launcher for the mini-bank")
    ap.add_argument("--n-shards", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.02,
                    help="envelope headroom over the planned spend")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    # Default beside the bank, not "/tmp": on Windows Path("/tmp") resolves
    # to C:\tmp, which is NOT the MSYS /tmp a bash shell reads, and the logs
    # go somewhere the operator is not looking.
    ap.add_argument("--log-dir", type=Path, default=OUT_ROOT / "logs")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from tinker_minibank_train import BASE_MODEL
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    all_runs = plan_runs(TASKS, DATA_SEEDS, INIT_SEEDS)
    done = {run_id_of(*r) for r in all_runs
            if (args.out_dir / run_id_of(*r) / "run_record.json").exists()}

    spent_usd = 0.0
    if LEDGER_PATH.exists():
        led = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        spent_usd = usd_for(led["cumulative_train_tokens"])

    shards, cache = [], {}
    for k in range(args.n_shards):
        runs = [r for j, r in enumerate(all_runs)
                if j % args.n_shards == k and run_id_of(*r) not in done]
        tokens = planned_tokens(runs, tok, cache)
        shards.append({"shard": k, "n_runs": len(runs), "tokens": tokens,
                       "planned_usd": usd_for(tokens),
                       "envelope_usd": usd_for(tokens) * (1 + args.margin)})

    total_env = sum(s["envelope_usd"] for s in shards)
    worst_case = PRIOR_SPEND_USD + spent_usd + total_env

    print(f"[plan] {len(all_runs)} total runs, {len(done)} already trained, "
          f"{sum(s['n_runs'] for s in shards)} remaining over "
          f"{args.n_shards} shards")
    print(f"{'shard':>6s} {'runs':>5s} {'tokens':>12s} {'planned$':>10s} {'envelope$':>10s}")
    for s in shards:
        print(f"{s['shard']:6d} {s['n_runs']:5d} {s['tokens']:12,} "
              f"{s['planned_usd']:10.4f} {s['envelope_usd']:10.4f}")
    print(f"\nPRIOR                 = ${PRIOR_SPEND_USD:.4f}")
    print(f"SPENT_SO_FAR          = ${spent_usd:.4f}")
    print(f"SUM_ENVELOPES         = ${total_env:.4f}  (margin {args.margin:.0%})")
    print(f"WORST_CASE_ACCOUNT    = ${worst_case:.4f}")
    print(f"STOP_REPORT_USD       = ${STOP_REPORT_USD:.2f}")

    if worst_case > STOP_REPORT_USD:
        raise SystemExit(
            f"REFUSING TO LAUNCH: worst case ${worst_case:.4f} exceeds the "
            f"${STOP_REPORT_USD:.2f} stop ceiling. Reduce shards or margin.")
    print(f"OK: even at full envelopes the account cannot reach the stop "
          f"ceiling (headroom ${STOP_REPORT_USD - worst_case:.4f}).")

    if args.plan_only or not args.launch:
        print("\n[plan-only] not launching. Re-run with --launch.")
        return

    for s in shards:
        if s["n_runs"] == 0:
            continue
        k = s["shard"]
        others = total_env - s["envelope_usd"]
        # prior for THIS shard = audited prior + already spent + what the
        # other shards are entitled to spend (their envelopes, the
        # conservative bound rather than their plans).
        prior_k = PRIOR_SPEND_USD + spent_usd + others
        ledger_k = args.out_dir / f"spend_ledger_shard{k}.json"
        log = args.log_dir / f"bank_shard{k}.log"
        cmd = [sys.executable, str(SCRIPTS_DIR / "tinker_minibank_train.py"),
               "--shard", str(k), "--n-shards", str(args.n_shards),
               "--skip-existing", "--defer-export",
               "--ledger", str(ledger_k),
               "--prior-usd", f"{prior_k:.6f}",
               "--shard-budget-usd", f"{s['envelope_usd']:.6f}"]
        fh = open(log, "w", encoding="utf-8")
        subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                         cwd=str(REPO_ROOT))
        print(f"[launch] shard {k}: {s['n_runs']} runs, envelope "
              f"${s['envelope_usd']:.4f}, prior ${prior_k:.4f} -> {log}")


if __name__ == "__main__":
    main()
