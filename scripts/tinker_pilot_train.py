"""Tinker signal pilot (E-T4 pre-step) — LoRA trainer with a hard budget guard.

Trains one rank-32 LoRA adapter per (task, seed) on Qwen/Qwen3-8B through the
Tinker API, exports the adapter as raw tensors, downloads it, and DELETES the
remote checkpoint (storage is billed at $0.10/GB-mo).

RUNS UNDER THE ``tinker`` CONDA ENV (py3.12, no torch):
    C:\\miniconda3\\envs\\tinker\\python.exe

Design (locked by the pilot brief; deviations only downward in cost)
-------------------------------------------------------------------
base model   Qwen/Qwen3-8B          train meter $0.44 / M tokens (verified
                                    2026-07-30 against the live pricing page)
rank         32 (Tinker default), Tinker's default module set
             (train_attn + train_mlp + train_unembed), i.e. STANDARD
             BRIDGELESS LoRA — no RhombiLoRA bridge anywhere
loss         cross_entropy, next-token shift
optimizer    AdamW, lr 1e-4 constant. The LoRA primer states the optimal LoRA
             LR is ~10x full-FT and is INDEPENDENT of rank, so 1e-4 stands
             for rank 32.
budget       ~1,000,000 train tokens per run, spent as 100 optim steps of
             ~10,000 tokens each.

LOSS NORMALIZATION (measured, not assumed)
------------------------------------------
The backend cross-entropy is an UNNORMALIZED SUM: ``loss = sum(-logprob * w)``.
Measured 2026-07-30 on a 60-loss-token batch: weights=1.0 reported
loss:sum = 210.6964, weights=1/N reported 3.51160728931427, and
210.6964 / 60 = 3.51161 exactly. So every weight is set to
1 / (loss tokens in the batch), making the reported loss the MEAN per-token
cross-entropy and keeping the gradient scale independent of batch size —
which is what lr=1e-4 assumes. Feeding weights=1.0 with a ~10,000-token
batch would apply a gradient ~10^4 too large.

TOKEN-MATCHED, NOT SEQUENCE-MATCHED
-----------------------------------
Each step is filled with whole sequences until the step's token budget is
reached, so every run consumes the same train tokens (= the same cost and the
same gradient scale) while the number of sequences per step varies by task
(agnews sequences are short, math long). Sequences are truncated to 512
tokens and NOT padded: Tinker has no fixed-shape requirement and padded
positions would be billed.

BUDGET GUARD
------------
A ledger at ``results/tinker-pilot/spend_ledger.json`` accumulates train
tokens and dollars ACROSS invocations, so the cap holds even if this script
is run repeatedly. Before every optim step the projected cumulative cost is
checked against --hard-abort-usd (default $4.50); on breach the run aborts
immediately, having already written the partial ledger. Stopping early with a
partial report is success; overrunning is failure.

Usage
-----
    # free: tokenize locally, print exact token counts and projected cost
    python scripts/tinker_pilot_train.py --dry-run

    python scripts/tinker_pilot_train.py --smoke                 # ~50k tokens
    python scripts/tinker_pilot_train.py --tasks alpaca math agnews --seeds 0 1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import tinker
from tinker import types

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "results" / "tinker-pilot"
DATA_DIR = OUT_ROOT / "data"
LEDGER_PATH = OUT_ROOT / "spend_ledger.json"

BASE_MODEL = "Qwen/Qwen3-8B"
LORA_RANK = 32
LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 512

PRICE_TRAIN_USD_PER_MTOK = 0.44   # verified on the live pricing page 2026-07-30
DEFAULT_HARD_ABORT_USD = 4.50

TOKENS_PER_STEP = 10_000
N_STEPS = 100
SMOKE_TOKENS_PER_STEP = 5_000
SMOKE_N_STEPS = 10

RETRY_MAX = 3

# The SDK default is httpx.Timeout(60s). Building the archive for a ~370 MB
# rank-32 all-modules adapter is a slow server-side job, and the archive-URL
# request blocks while it runs: at 60s it raises APITimeoutError AFTER the
# checkpoint has been written, orphaning billed storage (observed 2026-07-30 on
# the first smoke attempt). A long client timeout is required, not optional.
HTTP_TIMEOUT_SECONDS = 1800.0


def make_clients():
    """ServiceClient + RestClient with a timeout that survives archive builds."""
    sc = tinker.ServiceClient(
        timeout=httpx.Timeout(timeout=HTTP_TIMEOUT_SECONDS, connect=10.0))
    return sc, sc.create_rest_client()


class BudgetExceeded(RuntimeError):
    """Projected cumulative spend would breach the hard cap."""


# ── Spend ledger (persists across invocations) ───────────────────────


def load_ledger(hard_abort_usd: float) -> dict:
    if LEDGER_PATH.exists():
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    else:
        ledger = {
            "price_train_usd_per_mtok": PRICE_TRAIN_USD_PER_MTOK,
            "hard_abort_usd": hard_abort_usd,
            "cumulative_train_tokens": 0,
            "cumulative_usd": 0.0,
            "entries": [],
        }
    ledger["hard_abort_usd"] = hard_abort_usd
    return ledger


def save_ledger(ledger: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    tmp.replace(LEDGER_PATH)


def usd_for(tokens: int) -> float:
    return tokens / 1_000_000.0 * PRICE_TRAIN_USD_PER_MTOK


def guard(ledger: dict, pending_tokens: int, label: str) -> None:
    """Abort BEFORE spending if the projection breaches the cap."""
    projected = ledger["cumulative_usd"] + usd_for(pending_tokens)
    if projected > ledger["hard_abort_usd"]:
        raise BudgetExceeded(
            f"{label}: projected cumulative ${projected:.4f} would exceed the "
            f"hard cap ${ledger['hard_abort_usd']:.2f} "
            f"(already spent ${ledger['cumulative_usd']:.4f} over "
            f"{ledger['cumulative_train_tokens']:,} tokens). ABORTING.")


# ── Data ─────────────────────────────────────────────────────────────


def read_stream(task: str, seed: int) -> list[str]:
    """Read a generated text stream. Iterating the handle (not splitlines)
    keeps U+2028-bearing records intact; the emitter also escapes them."""
    path = DATA_DIR / f"{task}_seed{seed}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"missing {path}. Generate it first under the falco env:\n"
            f"  C:\\miniconda3\\envs\\falco\\python.exe "
            f"scripts/tinker_pilot_data.py --tasks {task} --seeds {seed}")
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh if line.strip()]


def batches(token_ids: list[list[int]], tokens_per_step: int, n_steps: int):
    """Yield (step, [seq, ...]) filling each step to >= tokens_per_step BILLED
    tokens, where a sequence of length L bills L-1 (the forward length after
    the next-token shift). Raises if the stream runs dry."""
    i = 0
    for step in range(n_steps):
        batch, billed = [], 0
        while billed < tokens_per_step:
            if i >= len(token_ids):
                raise SystemExit(
                    f"stream exhausted at step {step} ({i} sequences used); "
                    f"regenerate data with a larger --max-examples")
            seq = token_ids[i]
            i += 1
            if len(seq) < 2:
                continue
            batch.append(seq)
            billed += len(seq) - 1
        yield step, batch, billed


def build_data(batch: list[list[int]]) -> tuple[list[types.Datum], int]:
    """Datums with mean-normalizing weights (see module docstring)."""
    n_loss_tokens = sum(len(s) - 1 for s in batch)
    w = 1.0 / n_loss_tokens
    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(seq[:-1]),
            loss_fn_inputs={"target_tokens": seq[1:],
                            "weights": [w] * (len(seq) - 1)},
        )
        for seq in batch
    ]
    return data, n_loss_tokens


# ── Export ───────────────────────────────────────────────────────────


def _extract(blob: bytes, dest: Path) -> list[str]:
    """Extract a checkpoint archive (tar/tar.gz/zip), rejecting absolute paths,
    traversal and symlinks. Returns the flat list of extracted file names."""
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def safe(name: str) -> str:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"unsafe archive member {name!r}")
        return p.name  # flatten

    if blob[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = safe(info.filename)
                (dest / name).write_bytes(zf.read(info))
                written.append(name)
    else:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            for member in tf.getmembers():
                if member.issym() or member.islnk():
                    raise ValueError(f"archive contains a link: {member.name}")
                if not member.isfile():
                    continue
                name = safe(member.name)
                src = tf.extractfile(member)
                if src is None:
                    continue
                (dest / name).write_bytes(src.read())
                written.append(name)
    return sorted(written)


def fetch_and_extract(rest, tinker_path: str, run_dir: Path) -> dict:
    """Signed archive URL -> download -> extract -> DELETE the remote
    checkpoint. Separated from training so an orphaned checkpoint (training
    paid for, export interrupted) can be recovered without retraining."""
    t0 = time.time()
    url_resp = None
    for attempt in range(RETRY_MAX):
        try:
            url_resp = rest.get_checkpoint_archive_url_from_tinker_path(
                tinker_path).result()
            break
        except Exception as e:
            if attempt == RETRY_MAX - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"  [export] archive-url {type(e).__name__}: {e} — retry "
                  f"{attempt + 1}/{RETRY_MAX - 1} in {wait}s", flush=True)
            time.sleep(wait)

    with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True) as client:
        resp = client.get(url_resp.url)
        resp.raise_for_status()
        blob = resp.content
    print(f"  [export] archive {len(blob) / 1e6:.1f} MB")

    files = _extract(blob, run_dir)
    print(f"  [export] extracted: {files}")

    deleted = False
    try:
        rest.delete_checkpoint_from_tinker_path(tinker_path).result()
        deleted = True
        print("  [export] remote checkpoint deleted")
    except Exception as e:
        print(f"  [export] WARNING delete failed ({type(e).__name__}: {e}); "
              f"storage may bill until TTL")

    return {
        "tinker_path": tinker_path,
        "archive_bytes": len(blob),
        "files": files,
        "remote_deleted": deleted,
        "export_seconds": round(time.time() - t0, 1),
    }


def export_adapter(tc, rest, run_dir: Path, run_id: str) -> dict:
    """save_weights_for_sampler, then download + extract + delete."""
    saved = tc.save_weights_for_sampler(f"{run_id}-final").result()
    print(f"  [export] saved: {saved.path}")
    return fetch_and_extract(rest, saved.path, run_dir)


def recover_orphans(out_root: Path, max_workers: int = 3) -> None:
    """Download + delete every checkpoint still sitting in remote storage.

    Used when training completed and was billed but the export was deferred or
    interrupted: the weights are already paid for, so recover them rather than
    retrain, and clear the billed storage either way.

    Archives are fetched CONCURRENTLY. The first archive-URL request for a
    checkpoint triggers a slow server-side build (~40 min observed for a 369 MB
    rank-32 adapter) and blocks while it runs, so serial recovery of 6 adapters
    would cost hours of pure waiting. These are HTTP waits, not local compute,
    so a small thread pool overlaps the builds; max_workers stays low to avoid
    hammering the service.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _, rest = make_clients()
    resp = rest.list_user_checkpoints(limit=100).result()
    if not resp.checkpoints:
        print("[recover] no remote checkpoints; nothing to do")
        return

    targets = []
    for c in resp.checkpoints:
        # 'sampler_weights/<run_id>-final' -> '<run_id>'
        run_id = c.checkpoint_id.rsplit("/", 1)[-1].removesuffix("-final")
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        targets.append((run_id, c.tinker_path, c.size_bytes, run_dir))
        print(f"[recover] queued {run_id} ({c.size_bytes / 1e6:.1f} MB)")

    def work(target):
        run_id, tinker_path, _size, run_dir = target
        info = fetch_and_extract(rest, tinker_path, run_dir)
        (run_dir / "export_record.json").write_text(
            json.dumps(info, indent=2) + "\n", encoding="utf-8")
        # Fold the export back into run_record.json when training wrote one.
        rec_path = run_dir / "run_record.json"
        if rec_path.exists():
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            rec["export"] = info
            rec_path.write_text(json.dumps(rec, indent=2) + "\n",
                                encoding="utf-8")
        return run_id, info

    n_ok = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
        futures = {pool.submit(work, t): t[0] for t in targets}
        for fut in as_completed(futures):
            run_id = futures[fut]
            try:
                _, info = fut.result()
                n_ok += 1
                print(f"[recover] OK {run_id}: {info['files']} "
                      f"({info['archive_bytes'] / 1e6:.1f} MB, "
                      f"{info['export_seconds']:.0f}s)", flush=True)
            except Exception as e:
                print(f"[recover] FAILED {run_id}: {type(e).__name__}: {e}",
                      flush=True)
    print(f"[recover] {n_ok}/{len(targets)} recovered")


# ── One run ──────────────────────────────────────────────────────────


def train_one(sc, rest, task: str, seed: int, ledger: dict, *,
              tokens_per_step: int, n_steps: int, out_root: Path,
              defer_export: bool = False) -> dict:
    run_id = f"{task}_{seed}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== RUN {run_id} "
          f"({n_steps} steps x ~{tokens_per_step:,} tokens) ===")

    wall0 = time.time()
    tc = sc.create_lora_training_client(
        base_model=BASE_MODEL, rank=LORA_RANK, seed=seed)
    tok = tc.get_tokenizer()

    texts = read_stream(task, seed)
    token_ids = [tok.encode(t)[:MAX_SEQ_LEN] for t in texts]

    plan = list(batches(token_ids, tokens_per_step, n_steps))
    planned_tokens = sum(b for _, _, b in plan)
    guard(ledger, planned_tokens, f"{run_id} (whole run, {planned_tokens:,} tok)")
    print(f"  planned {planned_tokens:,} train tokens "
          f"= ${usd_for(planned_tokens):.4f}; "
          f"ledger so far ${ledger['cumulative_usd']:.4f}")

    losses: list[float] = []
    run_tokens = 0
    for step, batch, billed in plan:
        guard(ledger, billed, f"{run_id} step {step}")
        data, n_loss = build_data(batch)

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
                      f"{attempt + 1}/{RETRY_MAX - 1} in {wait}s")
                time.sleep(wait)

        loss = float(res.metrics["loss:sum"])   # mean per-token CE by construction
        losses.append(loss)

        # Bill only AFTER the tokens were actually spent.
        run_tokens += billed
        ledger["cumulative_train_tokens"] += billed
        ledger["cumulative_usd"] = usd_for(ledger["cumulative_train_tokens"])
        if step % 10 == 0 or step == len(plan) - 1:
            print(f"  step {step:3d}  loss {loss:.4f}  "
                  f"{len(batch):3d} seqs  {billed:6,} tok  "
                  f"run ${usd_for(run_tokens):.4f}  "
                  f"TOTAL ${ledger['cumulative_usd']:.4f}", flush=True)
            save_ledger(ledger)

    train_seconds = time.time() - wall0

    # Save the checkpoint immediately; then persist the TRAINING record before
    # touching the archive. Archive builds are slow server-side jobs that can
    # time out (observed 2026-07-30), and the training is already billed — the
    # losses and timings must survive an export failure. Export errors are
    # non-fatal so one slow archive cannot kill the remaining runs; the
    # checkpoint is left for --recover.
    saved = tc.save_weights_for_sampler(f"{run_id}-final").result()
    print(f"  [export] saved: {saved.path}")

    record = {
        "run_id": run_id,
        "task": task,
        "seed": seed,
        "base_model": BASE_MODEL,
        "lora_rank": LORA_RANK,
        "learning_rate": LEARNING_RATE,
        "max_seq_len": MAX_SEQ_LEN,
        "n_steps": len(plan),
        "tokens_per_step_target": tokens_per_step,
        "train_tokens": run_tokens,
        "usd": round(usd_for(run_tokens), 6),
        "n_sequences": sum(len(b) for _, b, _ in plan),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "losses": [round(x, 6) for x in losses],
        "train_seconds": round(train_seconds, 1),
        "model_id": str(tc.model_id),
        "tinker_path": saved.path,
        "export": None,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }

    def flush_record() -> None:
        record["wall_seconds"] = round(time.time() - wall0, 1)
        (run_dir / "run_record.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")

    flush_record()   # training measurements are now durable

    if defer_export:
        print("  [export] DEFERRED — collect later with --recover")
    else:
        try:
            record["export"] = fetch_and_extract(rest, saved.path, run_dir)
        except Exception as e:
            print(f"  [export] FAILED ({type(e).__name__}: {e}); training is "
                  f"kept, checkpoint left for --recover")
            record["export"] = {"error": f"{type(e).__name__}: {e}",
                                "tinker_path": saved.path}
        flush_record()

    ledger["entries"].append({k: record[k] for k in
                              ("run_id", "train_tokens", "usd", "wall_seconds")})
    save_ledger(ledger)

    print(f"  DONE {run_id}: {run_tokens:,} tok  ${usd_for(run_tokens):.4f}  "
          f"{record['wall_seconds']:.0f}s  "
          f"loss {record['loss_first']:.4f} -> {record['loss_last']:.4f}")
    return record


# ── Dry run (free) ───────────────────────────────────────────────────


def dry_run(tasks, seeds, tokens_per_step, n_steps) -> None:
    """Tokenize locally and price the plan without touching the training API."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    total = 0
    print(f"{'run':16s} {'seqs':>7s} {'tokens':>10s} {'usd':>8s} "
          f"{'mean_tok/seq':>13s}")
    for task in tasks:
        for seed in seeds:
            texts = read_stream(task, seed)
            ids = [tok.encode(t)[:MAX_SEQ_LEN] for t in texts]
            plan = list(batches(ids, tokens_per_step, n_steps))
            tks = sum(b for _, _, b in plan)
            seqs = sum(len(b) for _, b, _ in plan)
            total += tks
            print(f"{task + '_' + str(seed):16s} {seqs:7,} {tks:10,} "
                  f"${usd_for(tks):7.4f} {tks / max(seqs, 1):13.1f}")
    print(f"\nPLANNED TOTAL: {total:,} train tokens = ${usd_for(total):.4f} "
          f"at ${PRICE_TRAIN_USD_PER_MTOK}/M")
    print(f"pool depth OK for every run (no stream exhausted)")


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Tinker pilot LoRA trainer")
    ap.add_argument("--tasks", nargs="+", default=["alpaca", "math", "agnews"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--dry-run", action="store_true",
                    help="tokenize + price locally, spend nothing")
    ap.add_argument("--smoke", action="store_true",
                    help="one tiny run (~50k tokens) end-to-end incl. export")
    ap.add_argument("--tokens-per-step", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--hard-abort-usd", type=float, default=DEFAULT_HARD_ABORT_USD)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--recover", action="store_true",
                    help="download + delete any orphaned remote checkpoint "
                         "(training already billed) and exit")
    ap.add_argument("--recover-workers", type=int, default=3,
                    help="concurrent archive fetches for --recover (default 3)")
    ap.add_argument("--defer-export", action="store_true",
                    help="train and save checkpoints only; collect the "
                         "archives afterwards with --recover. Keeps slow "
                         "server-side archive builds off the training path")
    args = ap.parse_args()

    if args.smoke:
        tasks, seeds = ["alpaca"], [0]
        tps = args.tokens_per_step or SMOKE_TOKENS_PER_STEP
        steps = args.n_steps or SMOKE_N_STEPS
        out_root = args.out_dir or (OUT_ROOT / "smoke")
    else:
        tasks, seeds = args.tasks, args.seeds
        tps = args.tokens_per_step or TOKENS_PER_STEP
        steps = args.n_steps or N_STEPS
        out_root = args.out_dir or OUT_ROOT

    if args.dry_run:
        dry_run(tasks, seeds, tps, steps)
        return

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set in this process.")

    if args.recover:
        recover_orphans(args.out_dir or OUT_ROOT,
                        max_workers=args.recover_workers)
        return

    ledger = load_ledger(args.hard_abort_usd)
    print(f"[budget] hard cap ${ledger['hard_abort_usd']:.2f}; already spent "
          f"${ledger['cumulative_usd']:.4f} over "
          f"{ledger['cumulative_train_tokens']:,} tokens")

    sc, rest = make_clients()

    records = []
    try:
        for task in tasks:
            for seed in seeds:
                records.append(train_one(
                    sc, rest, task, seed, ledger,
                    tokens_per_step=tps, n_steps=steps, out_root=out_root,
                    defer_export=args.defer_export))
    except BudgetExceeded as e:
        print(f"\n*** BUDGET ABORT ***\n{e}")
        save_ledger(ledger)
        sys.exit(3)
    finally:
        save_ledger(ledger)

    print(f"\n[budget] FINAL: {ledger['cumulative_train_tokens']:,} train "
          f"tokens = ${ledger['cumulative_usd']:.4f}")
    for r in records:
        exported = isinstance(r.get("export"), dict) and "files" in r["export"]
        print(f"  {r['run_id']:14s} {r['train_tokens']:9,} tok  "
              f"${r['usd']:.4f}  {r.get('wall_seconds', 0):6.0f}s  "
              f"loss {r['loss_first']:.3f} -> {r['loss_last']:.3f}  "
              f"export={'ok' if exported else 'pending'}")
    if any(not (isinstance(r.get("export"), dict) and "files" in r["export"])
           for r in records):
        print("\n[export] some adapters are not downloaded yet; collect them "
              "with:  python scripts/tinker_pilot_train.py --recover")


if __name__ == "__main__":
    main()
