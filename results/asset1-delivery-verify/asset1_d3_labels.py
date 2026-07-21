#!/usr/bin/env python
"""Asset 1 — D3 label generation (post-bank GPU evals; delivery-day runner).

Evaluates every merged adapter emitted by asset1_d3_merge.py --emit-merges on
BOTH endpoint tasks' fixed 500-example val splits and writes the labels file
in the schema documented in asset1_d3_merge.py's module docstring.

Design decisions, fixed at eval time (2026-07-20, before any label existed):

- Metric: perplexity = exp(val_loss), emitted in the schema's merged_ppl_* /
  native_ppl_* fields. val_loss comes from the SAME instrument as the bank:
  asset1_bank.evaluate_val_loss on asset1_datasets.build_dataset('val')
  (locked val_seed=777 split), via the Stage B harness functions imported
  from asset1_d2_swap (not copied).
- Natives: taken from each run's trainer-logged metrics.json final val loss.
  VERIFIED EQUIVALENT before this runner was written: all 36 of D2 Stage B's
  fresh native evals match the trainer finals with 0.00000% relative
  divergence (same function, same split, deterministic eval) — see the
  delivery report. This halves the GPU work.
- degradation (continuous, required by the harness): the max over the two
  endpoints of relative perplexity degradation,
  max((m_a - n_a)/n_a, (m_b - n_b)/n_b) — aligned with the DECIDED primary
  rule (conflict iff EITHER endpoint degrades >= 5% relative), so
  degradation >= 0.05 <=> primary-rule conflict.

The interlock applies: refuses unless the manifest shows 480/480 COMPLETE.
Labels checkpoint to the output file after every pair (rewrite-in-full, like
Stage B); an interrupted run restarts from pair 1.

Usage:
    python scripts/asset1_d3_labels.py --bank-root results/asset1-bank \
        --d3-dir results/asset1-d3 --out results/asset1-d3/d3_labels.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asset1_analysis_io as aio
from asset1_canonicalize import load_adapter_modules
from asset1_d2_swap import _install_state, _load_family_model


def _run_dir(bank_root: Path, manifest: dict, run_index: int) -> Path:
    """Re-derive a run dir from bank_root — NEVER trust the manifest's
    machine-absolute run_dir field (asset1_analysis_io foundation rule:
    the bank may be moved/copied; paths are derived, not stored)."""
    run = next(r for r in manifest["runs"] if r["run_index"] == run_index)
    return (bank_root / run["family_short"] / run["task"]
            / f"run_{run_index:03d}")


def native_final_val(bank_root: Path, manifest: dict, run_index: int) -> float:
    """Trainer-logged final val loss for a run (verified == harness eval)."""
    metrics_path = _run_dir(bank_root, manifest, run_index) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    recs = metrics["records"] if isinstance(metrics, dict) and "records" in metrics else metrics
    vals = [r for r in recs if r.get("val_loss") is not None]
    if not vals:
        raise RuntimeError(f"run {run_index}: no val_loss records in metrics.json")
    return float(vals[-1]["val_loss"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank-root", default="results/asset1-bank")
    parser.add_argument("--d3-dir", default="results/asset1-d3")
    parser.add_argument("--out", default="results/asset1-d3/d3_labels.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    bank_root = Path(args.bank_root)
    d3_dir = Path(args.d3_dir)
    out_path = Path(args.out)

    # Pre-registration interlock — same gate as every other real-bank CLI.
    manifest = aio.require_complete_bank(bank_root)

    pairs_doc = json.loads((d3_dir / "d3_pairs.json").read_text(encoding="utf-8"))
    pairs = pairs_doc["pairs"]
    missing = [p for p in pairs if "merge_file" not in p]
    if missing:
        raise SystemExit(f"[d3-labels] {len(missing)} pair records carry no "
                         "merge_file — regenerate pairs with --emit-merges")

    fam_model = {f["short"]: f["model"] for f in manifest["campaign"]["families"]}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[d3-labels] REFUSING: cuda requested but unavailable")

    # Lazy heavy imports (mirrors Stage B).
    from torch.utils.data import DataLoader

    import asset1_bank as bank_mod
    from asset1_datasets import build_dataset

    out = {
        "generator": "scripts/asset1_d3_labels.py",
        "metric": "perplexity = exp(val_loss); natives from trainer metrics.json "
                  "finals (verified 0.00000% divergence vs harness on 36 D2 natives)",
        "degradation_def": "max over endpoints of (merged_ppl - native_ppl) / native_ppl",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "pairs": [],
    }

    def flush() -> None:
        out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    n_total = len(pairs)
    n_done = 0
    for fam in sorted({p["family_short"] for p in pairs}):
        fam_pairs = [p for p in pairs if p["family_short"] == fam]
        # Merges are saved FLAT ({module}.{field}); parse to nested via the
        # production loader — same form _install_state expects (verifier #1).
        first = load_adapter_modules(d3_dir / fam_pairs[0]["merge_file"])
        entry = next(iter(first.values()))
        rank = int(entry["rank"].item())
        n_channels = int(entry["n_channels"].item())

        tokenizer, model = _load_family_model(fam_model[fam], device)
        injected = bank_mod.inject_rhombi_lora(
            model, rank, n_channels, bank_mod.TARGET_MODULES)
        safe_to_dotted = {n.replace(".", "_"): n for n in injected}

        loaders: dict[str, DataLoader] = {}

        def val_loader(task: str) -> DataLoader:
            if task not in loaders:
                ds = build_dataset(task, tokenizer, "val", data_seed=0,
                                   max_len=bank_mod.MAX_LEN,
                                   pool_cap=bank_mod.POOL_CAP)
                loaders[task] = DataLoader(
                    ds, batch_size=bank_mod.EVAL_BATCH_SIZE, shuffle=False,
                    num_workers=0, pin_memory=True)
            return loaders[task]

        for p in fam_pairs:
            state = load_adapter_modules(d3_dir / p["merge_file"])
            _install_state(injected, safe_to_dotted, state)
            merged_loss_a = bank_mod.evaluate_val_loss(
                model, val_loader(p["task_a"]), device)
            merged_loss_b = (merged_loss_a if p["task_b"] == p["task_a"]
                             else bank_mod.evaluate_val_loss(
                                 model, val_loader(p["task_b"]), device))
            m_a, m_b = math.exp(merged_loss_a), math.exp(merged_loss_b)
            n_a = math.exp(native_final_val(bank_root, manifest, p["run_index_a"]))
            n_b = math.exp(native_final_val(bank_root, manifest, p["run_index_b"]))
            degradation = max((m_a - n_a) / n_a, (m_b - n_b) / n_b)
            out["pairs"].append({
                "family_short": fam,
                "task_a": p["task_a"], "run_index_a": p["run_index_a"],
                "task_b": p["task_b"], "run_index_b": p["run_index_b"],
                "degradation": degradation,
                "merged_ppl_a": m_a, "native_ppl_a": n_a,
                "merged_ppl_b": m_b, "native_ppl_b": n_b,
            })
            n_done += 1
            print(f"  [{n_done}/{n_total}] {fam} run{p['run_index_a']:03d}"
                  f"({p['task_a']}) + run{p['run_index_b']:03d}({p['task_b']})"
                  f" | deg={degradation:+.4f}", flush=True)
            flush()

        del model, tokenizer, injected
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out["finished_at"] = datetime.now(timezone.utc).isoformat()
    flush()
    print(f"[d3-labels] wrote {len(out['pairs'])} labeled pairs to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
