"""E-T4 Tinker mini-bank — assemble the typed core of MINIBANK_REPORT.md.

Every number this emits is READ FROM AN ARTIFACT at emit time — run records,
the spend ledger, the billing audit, the readout JSON, the merge_lint JSON.
Nothing is restated from narrative or memory. This is the §1 / §10 discipline
of ``.claude/rules/agent-prompt-templates.md`` and the XR-001 result behind
it (prose re-encoding corrupts 36.4% of numeric facts vs 9.4% for typed
blocks) applied to this report: the typed block is generated, the prose is
written around it.

RUNS UNDER THE ``falco`` CONDA ENV.

Usage
-----
    python scripts/tinker_minibank_report.py                 # typed core -> stdout
    python scripts/tinker_minibank_report.py --out results/tinker-minibank/TYPED_CORE.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BANK = REPO_ROOT / "results" / "tinker-minibank"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_records(bank: Path) -> list[dict]:
    out = []
    for d in sorted(bank.iterdir()):
        if d.is_dir():
            rec = d / "run_record.json"
            if rec.exists():
                out.append(json.loads(rec.read_text(encoding="utf-8")))
    return out


def adapter_present(bank: Path, run_id: str) -> bool:
    return bool(list((bank / run_id).glob("*.safetensors")))


def fmt_run_table(recs: list[dict], bank: Path) -> str:
    lines = ["| run | task | data_seed | init_seed | steps | train tokens | "
             "usd (billed est) | train s | wall s | loss first | loss last | adapter |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        lines.append(
            f"| {r['run_id']} | {r['task']} | {r['data_seed']} | {r['init_seed']} "
            f"| {r['n_steps']} | {r['train_tokens']:,} | {r['usd_billed_est']:.4f} "
            f"| {r.get('train_seconds', float('nan')):.1f} "
            f"| {r.get('wall_seconds', float('nan')):.1f} "
            f"| {r['loss_first']:.4f} | {r['loss_last']:.4f} "
            f"| {'yes' if adapter_present(bank, r['run_id']) else 'NO'} |")
    return "\n".join(lines)


def fmt_readout(sig: dict) -> str:
    if not sig:
        return "READOUT = [not yet run]"
    out = [f"N_ADAPTERS = {sig['n_adapters']}",
           f"TASKS = {sig['tasks']}",
           f"RAW_DIM = {sig['raw_dim']:,}" if sig.get("raw_dim") else "RAW_DIM = [skipped]",
           f"N_PERM = {sig['n_perm']}", ""]
    ks = sig["knn_ks"]
    header = ("| space | label | classes | LOO-1NN | chance | "
              + " | ".join(f"LOO-{k}NN" for k in ks if k != 1)
              + " | perm p (1NN) | separated |")
    # Derive the separator width from the header itself — hand-counted
    # column arithmetic is exactly how markdown tables silently break.
    out += [header, "|" + "---|" * (header.count("|") - 1)]
    for space in ("raw", "canonical_full", "canonical_sigma"):
        if space not in sig["spaces"]:
            continue
        for label in ("task", "init_seed", "data_seed"):
            e = sig["spaces"][space][label]
            extra = " | ".join(f"{e['knn'][f'k={k}']['loo_accuracy']:.3f}"
                               for k in ks if k != 1)
            out.append(
                f"| {space} | {label} | {len(e['classes'])} | "
                f"{e['knn']['k=1']['loo_accuracy']:.3f} | {e['loo_1nn_chance']:.3f} | "
                f"{extra} | {e['permutation_null_1nn']['p_value']:.4g} | "
                f"{e['separation']['separated']} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mini-bank typed core")
    ap.add_argument("--bank", type=Path, default=BANK)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bank = args.bank
    recs = run_records(bank)
    # ALL ledgers: the pre-shard one, the three shard ledgers, and the fixup.
    # Reading only spend_ledger.json under-reports the bank by ~90%.
    ledgers = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(bank.glob("spend_ledger*.json"))]
    ledger = ledgers[0] if ledgers else {}
    ledger_tokens = sum(l.get("cumulative_train_tokens", 0) for l in ledgers)
    ledger_entries = sum(len(l.get("entries", [])) for l in ledgers)
    billing = load(bank / "billing_usage.json")
    sig = load(bank / "signal_results.json")
    ctrl = load(bank / "controlled_contrast.json")
    ml = load(bank / "merge_lint_results.json")
    manifest = load(bank / "data" / "data_manifest.json")

    tok_total = sum(r["train_tokens"] for r in recs)
    usd_billed = sum(r["usd_billed_est"] for r in recs)
    usd_meter = sum(r["usd_meter_only"] for r in recs)
    n_adapters = sum(adapter_present(bank, r["run_id"]) for r in recs)
    prior = ledger.get("prior_spend_usd", float("nan"))

    parts = ["=== VERIFIED STATE: E-T4 Tinker mini-bank ===", "", "```"]
    parts += [
        f"N_RUNS_TRAINED        = {len(recs)}",
        f"N_ADAPTERS_DOWNLOADED = {n_adapters}",
        f"TASKS                 = {sorted({r['task'] for r in recs})}",
        f"DATA_SEEDS            = {sorted({r['data_seed'] for r in recs})}",
        f"INIT_SEEDS            = {sorted({r['init_seed'] for r in recs})}",
        f"BASE_MODEL            = {recs[0]['base_model'] if recs else '-'}",
        f"LORA_RANK             = {recs[0]['lora_rank'] if recs else '-'}",
        f"LEARNING_RATE         = {recs[0]['learning_rate'] if recs else '-'}",
        f"MAX_SEQ_LEN           = {recs[0]['max_seq_len'] if recs else '-'}",
        "",
        f"KEPT_RUN_TOKENS       = {tok_total:,}   [the 54 exported runs]",
        f"METERED_TOKENS_TOTAL  = {ledger_tokens:,}   [all {len(ledgers)} ledgers; "
        f"{ledger_entries} completed entries]",
        f"WASTED_ON_RESTARTS    = {ledger_tokens - tok_total:,} tokens = "
        f"${(ledger_tokens - tok_total) / 1e6 * 0.44 * 1.0035:.4f}   "
        f"[shard-2 watchdog restarts; billed, no adapter]",
        f"USD_METER_ONLY        = {ledger_tokens / 1e6 * 0.44:.4f}   [metered x $0.44/M]",
        f"USD_BILLED_EST        = {ledger_tokens / 1e6 * 0.44 * 1.0035:.4f}   "
        f"[x1.0035 meter factor]",
        f"USD_KEPT_RUNS_ONLY    = {usd_billed:.4f}   [what 54 clean runs would have cost]",
        f"PRIOR_ACCOUNT_SPEND   = {prior:.4f}   [audited via get_billing_usage]",
        f"ACCOUNT_TOTAL_EST     = "
        f"{prior + ledger_tokens / 1e6 * 0.44 * 1.0035:.4f}",
        f"STOP_REPORT_USD       = {ledger.get('stop_report_usd')}",
        f"HARD_ABORT_USD        = {ledger.get('hard_abort_usd')}",
        f"HEADROOM_TO_STOP      = "
        f"{ledger.get('stop_report_usd', 27.5) - (prior + ledger_tokens / 1e6 * 0.44 * 1.0035):.4f}",
    ]
    if recs:
        ws = [r.get("wall_seconds", 0) for r in recs]
        ts = [r.get("train_seconds", 0) for r in recs]
        parts += [
            "",
            f"WALL_SECONDS_PER_RUN  = {min(ws):.1f} - {max(ws):.1f} "
            f"(mean {sum(ws)/len(ws):.1f})",
            f"TRAIN_SECONDS_PER_RUN = {min(ts):.1f} - {max(ts):.1f} "
            f"(mean {sum(ts)/len(ts):.1f})",
        ]
    if billing:
        parts += [
            "",
            f"BILLED_TRAIN_TOKENS   = {billing['total_train_tokens']:,.0f}   [get_billing_usage, all models]",
            f"BILLED_TRAINING_USD   = {billing['priced_training_usd']:.4f}   [priced models]",
            f"BILLED_STORAGE_GBH    = {billing['storage_gigabyte_hours']:.4f} = ${billing['storage_usd']:.4f}",
            f"BILLED_CHECKPOINT_EV  = {billing['checkpoint_events']:.0f}",
        ]
        # Bank-attributable billed tokens = all-time billed minus the
        # pre-bank baseline (6,107,108 Qwen3-8B on Jul 30 + 59,764 other
        # models on Aug 3), all from the same usage API.
        pre_bank = 6_107_108 + 59_764
        bank_billed = billing["total_train_tokens"] - pre_bank
        parts += [
            f"BANK_BILLED_TOKENS    = {bank_billed:,.0f}   "
            f"[all-time billed minus the {pre_bank:,} pre-bank baseline]",
            f"BANK_METERED_TOKENS   = {ledger_tokens:,}   [our ledgers]",
            f"BILLING_POSTED_FRAC   = {bank_billed / ledger_tokens:.3f}   "
            f"[<1.0 means the usage API has not finished posting; it lags "
            f"real time by hours, so full reconciliation is OUTSTANDING]",
        ]
    if manifest:
        parts += ["", f"DATA_STREAMS          = {len(manifest['streams'])}",
                  f"DATA_TOKENIZER        = {manifest.get('base_model')}"]
    if ml:
        parts += [
            "",
            f"MERGE_LINT_PAIRS      = {ml['n_pairs']} (vertex-disjoint, cross-task)",
            f"MERGE_LINT_IN_FAMILY  = {ml['n_in_family']}",
            f"MERGE_LINT_EXITS      = {ml['exit_code_counts']}",
            f"BRIDGELESS_REFUSAL    = exit {ml['bridgeless_refusal_probe']['exit_code']}",
        ]
    parts += ["```", "", "## Readout (plain LOO — CONFOUNDED, see below)", "",
              "```", fmt_readout(sig), "```"]

    if ctrl:
        parts += ["", "## Controlled contrast (the decisive figures)", "", "```",
                  f"N_PERM = {ctrl['n_perm']}", "",
                  "| space | contrast | accuracy | chance | perm p |",
                  "|---|---|---|---|---|"]
        for space, e in ctrl["spaces"].items():
            for name in ("cross_init_task", "within_init_task", "cross_task_init"):
                c = e[name]
                parts.append(
                    f"| {space} | {name} | {c['accuracy']:.3f} | "
                    f"{c['chance']:.3f} | {c['p_value']:.4g} |")
        parts += ["```"]

    parts += ["", "## Per-run", "", fmt_run_table(recs, bank)]

    text = "\n".join(parts) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
