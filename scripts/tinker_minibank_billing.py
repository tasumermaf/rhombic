"""E-T4 Tinker mini-bank — billing reconciliation via the usage API.

Queries ``rest.get_billing_usage`` over the whole life of the key and writes
``results/tinker-minibank/billing_usage.json``: per (day, base model, usage
kind) metered totals, plus the derived dollar figures.

The usage API returns NO dollar amounts — only metered quantities — so the
dollars here are quantity x the published price table, and the billing page
remains the authority. That is exactly why the trainer's guard carries the
+0.35% BILLING_FACTOR: the meter counts slightly more than the in-code
``len(seq) - 1`` ledger, and the guard must err high.

Windows are capped at 14 days by the API, so the range is walked in chunks.

RUNS UNDER THE ``tinker`` CONDA ENV:
    C:\\miniconda3\\envs\\tinker\\python.exe

Usage
-----
    python scripts/tinker_minibank_billing.py
    python scripts/tinker_minibank_billing.py --start 2026-07-22 --end 2026-08-06
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import tinker

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "results" / "tinker-minibank"

# Published price table (verified on the live models & pricing page).
PRICE_TRAIN_USD_PER_MTOK = {"Qwen/Qwen3-8B": 0.44}
PRICE_STORAGE_USD_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 730.0


def walk_windows(start: datetime, end: datetime, days: int = 7):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def main() -> None:
    ap = argparse.ArgumentParser(description="Tinker mini-bank billing audit")
    ap.add_argument("--start", default="2026-07-22")
    ap.add_argument("--end", default=None, help="default: tomorrow UTC")
    ap.add_argument("--out", type=Path, default=OUT_ROOT / "billing_usage.json")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           if args.end else
           (datetime.now(timezone.utc) + timedelta(days=1)).replace(
               hour=0, minute=0, second=0, microsecond=0))

    sc = tinker.ServiceClient(timeout=httpx.Timeout(timeout=300.0, connect=10.0))
    rest = sc.create_rest_client()

    agg: dict[tuple[str, str, str], float] = defaultdict(float)
    events: dict[tuple[str, str, str], int] = defaultdict(int)
    for a, b in walk_windows(start, end):
        resp = rest.get_billing_usage(
            a.strftime("%Y-%m-%dT%H:00:00Z"),
            b.strftime("%Y-%m-%dT%H:00:00Z")).result()
        for ev in resp.data:
            info = ev.event_info
            qty = getattr(info, "token_count", None)
            if qty is None:
                qty = getattr(info, "gigabyte_hours", None)
            if qty is None:
                qty = getattr(info, "count", 0)
            key = (str(ev.bucket_start)[:10], str(ev.base_model), str(info.type))
            agg[key] += float(qty)
            events[key] += 1

    rows = []
    train_tokens_by_model: dict[str, float] = defaultdict(float)
    storage_gbh = 0.0
    checkpoints = 0.0
    for (day, model, kind), qty in sorted(agg.items()):
        rows.append({"day": day, "base_model": model, "kind": kind,
                     "quantity": qty, "n_events": events[(day, model, kind)]})
        if kind == "training":
            train_tokens_by_model[model] += qty
        elif kind == "storage":
            storage_gbh += qty
        elif kind == "checkpoint":
            checkpoints += qty

    priced, unpriced = 0.0, {}
    for model, tokens in train_tokens_by_model.items():
        price = PRICE_TRAIN_USD_PER_MTOK.get(model)
        if price is None:
            unpriced[model] = tokens
        else:
            priced += tokens / 1e6 * price
    storage_usd = storage_gbh / HOURS_PER_MONTH * PRICE_STORAGE_USD_PER_GB_MONTH

    payload = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "price_table_train_usd_per_mtok": PRICE_TRAIN_USD_PER_MTOK,
        "price_storage_usd_per_gb_month": PRICE_STORAGE_USD_PER_GB_MONTH,
        "rows": rows,
        "train_tokens_by_model": dict(train_tokens_by_model),
        "total_train_tokens": sum(train_tokens_by_model.values()),
        "storage_gigabyte_hours": storage_gbh,
        "storage_usd": storage_usd,
        "checkpoint_events": checkpoints,
        "priced_training_usd": priced,
        "unpriced_models_train_tokens": unpriced,
        "note": ("The usage API returns metered quantities only, no dollars. "
                 "Dollars here are quantity x the published price table; the "
                 "billing page is the authority. Models absent from the price "
                 "table are listed under unpriced_models_train_tokens."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for r in rows:
        print(f"{r['day']}  {r['base_model']:26s} {r['kind']:12s} "
              f"{r['quantity']:>16,.4f}  ({r['n_events']} ev)")
    print(f"\nTOTAL_TRAIN_TOKENS = {payload['total_train_tokens']:,.0f}")
    print(f"PRICED_TRAINING_USD = ${priced:.4f}   (priced models only)")
    if unpriced:
        print(f"UNPRICED (not in table): {unpriced}")
    print(f"STORAGE = {storage_gbh:.4f} GB-h = ${storage_usd:.4f}   "
          f"CHECKPOINT_EVENTS = {checkpoints:.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
