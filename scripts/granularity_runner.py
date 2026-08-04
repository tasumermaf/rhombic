# -*- coding: utf-8 -*-
"""Granularity ladder — one-adapter runner (a thin wrapper on the Asset-1 trainer).

REGISTERED CARD: `docs/REGISTRATION_GRANULARITY_2026-07-30.md`, LOCKED by
    `docs/LOCK_GRANULARITY_2026-08-04.md`. Family: llama3.2-1b ONLY (D1).

WHAT THIS SCRIPT IS: `asset1_bank.run_single` called UNMODIFIED, on a
    class-restricted training pool. The recipe is the bank's recipe by
    construction — 2,000 steps, rank 24, n_channels 6, identity bridge,
    bs4xga4 (effective batch 16), seq 512, bf16 base / fp32 adapters,
    LM loss only, val loss every 100 steps — because every constant is
    imported from `asset1_bank` and the training function is the bank's own.
    Nothing in `asset1_bank.py` or `asset1_datasets.py` is edited or forked.

HOW THE CLASS POOL GETS IN: by DELEGATION, the S2-pilot pattern. The single
    seam is `asset1_bank.build_dataset`, which the runner temporarily rebinds
    to a wrapper that constructs the SAME task dataset class with
    `raw=ds.select(row_ids)` — the frozen row ids from
    `results/granularity/labels/<level>_pools.json`. Everything downstream
    (the canonical val_seed=777 shuffle, the per-class fixed 500-row val, the
    data_seed pool ordering, tokenization, the loss convention) is the
    unmodified `asset1_datasets` machinery operating on the class's rows.
    The wrapper is removed in a `finally`.

GPU DISCIPLINE (LOCK, non-negotiable): every launch is wrapped in
    `gpu_guard.guarded(...)` — PAUSE sentinel, VRAM preflight, and a claim
    file, so a shared-card user is never OOM'd and never waits more than one
    run.

WHAT IS REFUSED: any write under `results/asset1-bank*`; any level whose
    manifest is not launchable (e.g. L2/L3 while xsum's T3 classes are
    unmaterialized — ambiguity G-5); any L0 run (L0 is analysis-side: it
    reuses the 240 existing llama bank adapters and trains nothing). The
    Asset-1 bank manifest is sha256'd before and after each run and asserted
    unchanged.

EXCLUSION: every run dir carries `EXCLUDED_FROM_BANK`, sits outside every
    bank root, and its `config.json` campaign_tag is `granularity-<level>` —
    so the Asset-1 analysis CLIs (which walk `results/asset1-bank/...`)
    cannot see these adapters.

Usage:
    python scripts/granularity_runner.py --level L1 --run 0 --dry-run
    python scripts/granularity_runner.py --level L1 --run 0
    python scripts/granularity_runner.py --level L1 --plan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

PROCESS_T0 = time.monotonic()

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_CACHE", r"C:\falco\hf-cache\hub")
os.environ.setdefault("HF_DATASETS_CACHE", r"C:\falco\hf-cache\datasets")

import gpu_guard  # noqa: E402
import asset1_bank as ab  # noqa: E402
from asset1_bank import (  # noqa: E402  (constants imported, never redeclared)
    BATCH_SIZE,
    CampaignSpec,
    GRAD_ACCUM,
    MAX_LEN,
    MAX_STEPS,
    RunSpec,
    git_commit_hash,
    library_versions,
    probe_family_access,
    utc_now,
)
from asset1_datasets import (  # noqa: E402
    POOL_CAP,
    TASK_REGISTRY,
    load_hf_dataset_with_fallback,
)
# Timing instrumentation reused verbatim from the S2 pilots, so granularity
# wall-clock is measured by the same formulas as the 30.56 min/run cost basis.
from s2_timing_pilot import (  # noqa: E402
    Probe,
    _fmt,
    derive_breakdown,
    hf_cache_is_warm,
    project_full_run,
    tokens_trained,
    typed_block,
)
from granularity_labels import (  # noqa: E402
    EMITTED_LEVELS,
    LABELS_DIR,
    LEVEL_SEEDS,
    OUT_ROOT,
)

# ── Locked runner constants ─────────────────────────────────────────

FAMILY = {"model": "meta-llama/Llama-3.2-1B-Instruct", "short": "llama3.2-1b"}

# Seed bands: 1,000 wide per level, disjoint from the bank (10,000 / 20,000)
# and the S2 pilots (90,000 / 91,000). seed is a pure function of
# (level, run index) — design §4 "pure function of index".
GRAN_SEED_BASE = 300_000
GRAN_DATA_SEED_BASE = 310_000
LEVEL_BAND = 1_000

# gpu_guard parameters (LOCK: mandatory on every launch).
GUARD_NEEDED_GB = 14
GUARD_EXPECTED_MIN = 32

DRY_RUN_STEPS = 10

FORBIDDEN_ROOT = (REPO_ROOT / "results").resolve()


# ── Hard refusal: nothing may be written under a bank root ──────────


def assert_granularity_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(FORBIDDEN_ROOT)
    except ValueError:
        rel = None
    if rel is not None:
        for part in rel.parts:
            if part.lower().startswith("asset1-bank"):
                raise SystemExit(
                    f"REFUSED: {resolved} lies under the Asset-1 bank root "
                    f"'{part}'. The granularity ladder never writes there.")
    root = OUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise SystemExit(f"REFUSED: {resolved} is outside {root}.")
    return resolved


def _bank_manifest_digest() -> str | None:
    p = ab.DEFAULT_BANK_ROOT / "bank_manifest.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def write_text(path: Path, text: str) -> None:
    assert_granularity_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# ── Level manifest -> run plan ──────────────────────────────────────


def load_level(level: str) -> tuple[dict, dict]:
    """(manifest, pools) for a level, from the frozen Stage-0 artifacts."""
    if level not in EMITTED_LEVELS:
        raise SystemExit(f"[gran] unknown level {level!r}; expected one of "
                         f"{EMITTED_LEVELS}")
    man_path = LABELS_DIR / f"{level}.json"
    pool_path = LABELS_DIR / f"{level}_pools.json"
    if not man_path.exists() or not pool_path.exists():
        raise SystemExit(
            f"[gran] no frozen label manifest for {level} at {man_path} — "
            f"run scripts/granularity_labels.py first.")
    return (json.loads(man_path.read_text(encoding="utf-8")),
            json.loads(pool_path.read_text(encoding="utf-8")))


def live_classes(manifest: dict) -> list[dict]:
    """Materialized classes, in the frozen (class_id-sorted) order."""
    return [c for c in manifest["classes"] if c.get("materialized")]


def plan_runs(level: str, manifest: dict) -> list[dict]:
    """Round-robin across classes, replicate-major — the bank's ordering, so
    an interrupted level leaves a class-balanced partial set."""
    classes = live_classes(manifest)
    seeds = LEVEL_SEEDS[level]
    band = EMITTED_LEVELS.index(level) * LEVEL_BAND
    plan = []
    k = 0
    for rep in range(seeds):
        for ci, cls in enumerate(classes):
            plan.append({
                "run_k": k,
                "level": level,
                "class_index": ci,
                "class_id": cls["class_id"],
                "task": cls["task"],
                "tier": cls["tier"],
                "clean_core": cls["clean_core"],
                "replicate": rep,
                "seed": GRAN_SEED_BASE + band + k,
                "data_seed": GRAN_DATA_SEED_BASE + band + k,
                "n_train_pool": cls["n_train_pool"],
                "row_ids_sha256": cls["row_ids_sha256"],
            })
            k += 1
    return plan


def run_dir_for(level: str, run_k: int, dry: bool = False) -> Path:
    """Real runs and dry runs NEVER share a directory.

    A dry run writes a COMPLETE marker (it calls the real trainer for 10
    steps), so if it landed in `run_{k}/` the queue would skip that cell and
    the analysis completeness interlock would admit a 10-step adapter into a
    2,000-step level. Dry runs therefore live in `dryrun_{k}/`, outside the
    run plan entirely.
    """
    name = f"{'dryrun' if dry else 'run'}_{run_k:03d}"
    return assert_granularity_path(OUT_ROOT / level / name)


def refuse_unlaunchable(level: str, manifest: dict) -> None:
    if level == "L0":
        raise SystemExit(
            "[gran] REFUSED: L0 is ANALYSIS-SIDE. Its 240 adapters already "
            "exist (results/asset1-bank/llama3.2-1b/, 6 tasks x 40 seeds); "
            "the design's §2 table gives L0 'new runs = 0'. The L0 "
            "re-baseline is run by scripts/granularity_analysis.py.")
    if not manifest.get("launchable"):
        blocked = manifest.get("launch_blocked_by") or [
            f"{manifest['k_declared'] - manifest['k_materialized']} of "
            f"{manifest['k_declared']} classes are not materialized"]
        raise SystemExit(
            f"[gran] REFUSED: level {level} is not launchable — "
            f"{'; '.join(blocked)}. The registered ladder fixes K at this "
            f"level; training a partial label space would answer a "
            f"different question. Resolve by dated amendment (L-006).")


# ── The single seam: a class-restricted dataset factory ─────────────


def install_class_pool(task: str, row_ids: list[int]):
    """Rebind ``asset1_bank.build_dataset`` to serve the class's rows.

    Returns (restore, source_id, n_rows). The dataset CLASS, the split
    function, the prompt template, the tokenizer conventions and the label
    convention are all the unmodified `asset1_datasets` ones; only the row
    universe changes, and it changes through the class's own public `raw=`
    argument.
    """
    cls = TASK_REGISTRY[task]
    ds, source = load_hf_dataset_with_fallback(
        cls.dataset_candidates, cls.dataset_config_name, cls.hf_split)
    subset = ds.select([int(i) for i in row_ids])
    original = ab.build_dataset

    def class_build_dataset(task_key, tokenizer, split_pool, data_seed,
                            max_len=MAX_LEN, pool_cap=POOL_CAP,
                            keep_text=False):
        if task_key != task:
            raise RuntimeError(
                f"granularity seam misuse: trainer asked for task "
                f"{task_key!r}, this run's class is a {task!r} class")
        return cls(tokenizer, split_pool, data_seed, max_len,
                   pool_cap=pool_cap, keep_text=keep_text, raw=subset)

    ab.build_dataset = class_build_dataset

    def restore() -> None:
        ab.build_dataset = original

    return restore, source, len(subset)


# ── Execution ───────────────────────────────────────────────────────


def gran_spec(level: str, max_steps: int) -> CampaignSpec:
    """A CampaignSpec whose bank_root is the granularity root. run_single
    reads only tag / max_steps / pool_cap; no campaign orchestration
    function (which would write a bank manifest) is ever called."""
    return CampaignSpec(
        tag=f"granularity-{level}", bank_root=assert_granularity_path(OUT_ROOT),
        families=[], tasks=[], n_replicates=0, max_steps=max_steps,
        pool_cap=POOL_CAP)


def write_granularity_json(run_dir: Path, entry: dict, manifest: dict,
                           source: str, n_rows: int, dry: bool) -> None:
    write_text(run_dir / "granularity.json", json.dumps({
        "asset": "granularity-ladder",
        "level": entry["level"],
        "run_k": entry["run_k"],
        "class_id": entry["class_id"],
        "class_index": entry["class_index"],
        "task": entry["task"],
        "tier": entry["tier"],
        "clean_core": entry["clean_core"],
        "replicate": entry["replicate"],
        "seed": entry["seed"],
        "data_seed": entry["data_seed"],
        "family": FAMILY["model"],
        "family_short": FAMILY["short"],
        "n_class_rows_served": n_rows,
        "n_train_pool_expected": entry["n_train_pool"],
        "row_ids_sha256": entry["row_ids_sha256"],
        "dataset_source": source,
        "labels_git_commit": manifest.get("git_commit"),
        "labels_created_at": manifest.get("created_at"),
        "balance_policy": manifest.get("balance_policy"),
        "k_level": manifest.get("k_materialized"),
        "chance": manifest.get("chance"),
        "cohort": f"bs{BATCH_SIZE}xga{GRAD_ACCUM}",
        "excluded_from_bank": True,
        "dry_run": dry,
        "written_at": utc_now(),
    }, indent=1))


def write_timing_md(run_dir: Path, entry: dict, *, status: str, steps: int,
                    wall_run_s: float, peak_alloc_gb, peak_reserved_gb,
                    probe: Probe, bd: dict, warm: bool,
                    manifest_before, manifest_after, manifest: dict,
                    error_excerpt: str | None, dry: bool,
                    projection: dict | None) -> Path:
    cfg = {}
    cpath = run_dir / "config.json"
    if cpath.exists():
        try:
            cfg = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    versions = library_versions()

    core = [("level", entry["level"]),
            ("run_k", entry["run_k"]),
            ("class_id", entry["class_id"]),
            ("status", status),
            ("steps", steps),
            ("wall_clock_min", _fmt(wall_run_s / 60.0)),
            ("peak_vram_gb", _fmt(peak_alloc_gb)),
            ("tokens_trained", tokens_trained(steps)),
            ("date", date.today().isoformat())]
    context = [("model_id", FAMILY["model"]),
               ("task", entry["task"]),
               ("tier", entry["tier"]),
               ("clean_core", str(entry["clean_core"]).lower()),
               ("replicate", entry["replicate"]),
               ("seed", entry["seed"]),
               ("data_seed", entry["data_seed"]),
               ("batch_geometry", f"bs{BATCH_SIZE}xga{GRAD_ACCUM}"),
               ("effective_batch", BATCH_SIZE * GRAD_ACCUM),
               ("max_len", MAX_LEN),
               ("n_train_pool_expected", entry["n_train_pool"]),
               ("n_pool_trainer_reported",
                (cfg.get("dataset") or {}).get("n_pool")),
               ("n_val_trainer_reported",
                (cfg.get("dataset") or {}).get("n_val")),
               ("val_ids_sha256_trainer",
                ((cfg.get("dataset") or {}).get("val_ids_sha256") or "—")[:16]),
               ("k_level", manifest.get("k_materialized")),
               ("chance", manifest.get("chance")),
               ("hf_cache_warm_at_start", str(warm).lower()),
               ("peak_vram_reserved_gb", _fmt(peak_reserved_gb)),
               ("n_params_base_measured", probe.n_params_base),
               ("attn_implementation", probe.attn_implementation),
               ("model_dtype", probe.model_dtype)]
    breakdown = [("model_load_s", _fmt(probe.model_load_s)),
                 ("setup_s", _fmt(bd["setup_s"])),
                 ("step_loop_s", _fmt(bd["step_loop_s"])),
                 ("steps_recorded", bd["steps_recorded"]),
                 ("mean_step_s", _fmt(bd["mean_step_s"], 4)),
                 ("val_evals", bd["val_evals"]),
                 ("mean_val_eval_s", _fmt(bd["mean_val_eval_s"])),
                 ("save_s", _fmt(bd["save_s"]))]
    provenance = [("EXCLUDED_FROM_BANK", "true"),
                  ("computes_ladder_statistics", "false"),
                  ("campaign_tag", cfg.get("campaign_tag", "—")),
                  ("trainer", "asset1_bank.run_single (imported unmodified)"),
                  ("dataset_seam",
                   "asset1_bank.build_dataset rebound to serve "
                   "ds.select(frozen row_ids) via the task class's raw= "
                   "argument; restored after the run"),
                  ("labels_manifest_git", manifest.get("git_commit", "—")[:12]),
                  ("row_ids_sha256", (entry["row_ids_sha256"] or "—")[:16]),
                  ("gpu_guard", f"guarded(needed_gb={GUARD_NEEDED_GB}, "
                                f"expected_min={GUARD_EXPECTED_MIN})"),
                  ("bank_manifest_sha256_before", (manifest_before or "—")[:16]),
                  ("bank_manifest_sha256_after", (manifest_after or "—")[:16]),
                  ("bank_manifest_unchanged",
                   str(manifest_before == manifest_after).lower()),
                  ("git_commit", cfg.get("git_commit", git_commit_hash())),
                  ("gpu", versions.get("gpu", "—")),
                  ("python", versions.get("python")),
                  ("torch", versions.get("torch")),
                  ("transformers", versions.get("transformers")),
                  ("finished_at_utc", utc_now())]

    kind = "DRY RUN" if dry else f"run {entry['run_k']:03d}"
    parts = [f"# GRANULARITY {entry['level']} — {kind} · "
             f"{entry['class_id']}", "",
             "Training run under the LOCKED granularity card "
             "(`docs/LOCK_GRANULARITY_2026-08-04.md`). Excluded from every "
             "bank. No ladder statistic is computed here. Every value below "
             "is measured, or derived from measured values by the formulas "
             "in `scripts/s2_timing_pilot.py` (reused unchanged, so these "
             "rates are comparable to the S2 rate basis).", "",
             "=== VERIFIED STATE ===", typed_block(core),
             "=== END VERIFIED STATE ===",
             "", "## Context", "", typed_block(context),
             "", "## Measured breakdown", "", typed_block(breakdown)]
    if projection:
        proj = [("projection_basis_steps", bd["steps_recorded"]),
                ("n_evals_in_full_run", projection["n_evals_full"]),
                ("PROJECTED_min_per_full_run", _fmt(projection["projected_min"])),
                ("PROJECTED_min_excl_model_load",
                 _fmt(projection.get("projected_min_excl_model_load"))),
                ("measured_llama_basis_min_per_run", 30.56),
                ("basis_source",
                 "results/asset1-bank/RATE_EXTRACT.md (n=240)"),
                ("projection_bias",
                 "upper (first-step cost amortized over few steps)")]
        parts += ["", "## Projection (NOT a measurement)", "", typed_block(proj)]
    if error_excerpt:
        parts += ["", "## Failure detail (recorded as a finding)", "",
                  "```", error_excerpt.strip(), "```"]
    parts += ["", "## Provenance", "", typed_block(provenance), ""]

    path = run_dir / ("DRYRUN.md" if dry else "TIMING.md")
    write_text(path, "\n".join(parts))
    return path


def execute(level: str, run_k: int, dry: bool, steps_override: int) -> int:
    manifest, pools = load_level(level)
    refuse_unlaunchable(level, manifest)
    plan = plan_runs(level, manifest)
    if not (0 <= run_k < len(plan)):
        raise SystemExit(f"[gran] --run {run_k} out of range for {level} "
                         f"(0..{len(plan) - 1})")
    entry = plan[run_k]

    access = probe_family_access(FAMILY["model"])
    if access == "BLOCKED":
        print(f"REFUSED: {FAMILY['model']} is license-gated on this account. "
              f"The ladder is llama3.2-1b only (D1); this is a finding, not "
              f"something to work around.")
        return 78

    steps = steps_override if dry else MAX_STEPS
    spec = gran_spec(level, steps)
    run_dir = run_dir_for(level, run_k, dry=dry)
    run_dir.mkdir(parents=True, exist_ok=True)

    if dry:
        for name in ("COMPLETE", "FAILED", "error.log"):
            p = run_dir / name
            if p.exists():
                p.unlink()
    elif (run_dir / "COMPLETE").exists():
        print(f"{run_dir} already COMPLETE — refusing to overwrite a trained "
              f"adapter. Delete it deliberately to re-train.")
        return 0

    run = RunSpec(
        run_index=entry["run_k"], family=FAMILY["model"],
        family_short=FAMILY["short"], task=entry["task"],
        replicate=entry["replicate"], seed=entry["seed"],
        data_seed=entry["data_seed"], run_dir=run_dir)

    row_ids = pools["row_ids"][entry["class_id"]]
    restore_pool, source, n_rows = install_class_pool(entry["task"], row_ids)
    write_granularity_json(run_dir, entry, manifest, source, n_rows, dry)

    warm = hf_cache_is_warm(FAMILY["model"])
    manifest_before = _bank_manifest_digest()

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    probe = Probe()
    restore_probe = probe.install()
    print(f"\n{'=' * 70}")
    print(f"GRANULARITY {level} | run {run_k}/{len(plan) - 1} | "
          f"{entry['class_id']} ({entry['tier']})")
    print(f"{'dry run' if dry else 'training'} | steps={steps} | "
          f"pool={entry['n_train_pool']} | seed={entry['seed']} | "
          f"K={manifest['k_materialized']}")
    print(f"out: {run_dir}")
    print(f"{'=' * 70}\n", flush=True)

    t0 = time.monotonic()
    try:
        with gpu_guard.guarded(
                purpose=f"granularity {level} run {run_k}",
                needed_gb=GUARD_NEEDED_GB,
                expected_min=GUARD_EXPECTED_MIN):
            rc = ab.run_single(spec, run)
    finally:
        restore_probe()
        restore_pool()
    wall_run_s = time.monotonic() - t0

    peak_alloc_gb = peak_reserved_gb = None
    if torch.cuda.is_available():
        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

    status = {0: "COMPLETE", 78: "BLOCKED"}.get(rc, "FAILED")
    error_excerpt = None
    if status == "FAILED":
        elog = run_dir / "error.log"
        if elog.exists():
            txt = elog.read_text(encoding="utf-8", errors="replace").strip()
            lines = txt.splitlines()
            error_excerpt = ("\n".join(lines[:6] + ["..."] + lines[-6:])
                             if len(lines) > 12 else txt)

    bd = derive_breakdown(run_dir, probe, steps)
    projection = project_full_run(bd, probe) if dry else None
    manifest_after = _bank_manifest_digest()

    path = write_timing_md(
        run_dir, entry, status=status, steps=bd["steps_recorded"] or steps,
        wall_run_s=wall_run_s, peak_alloc_gb=peak_alloc_gb,
        peak_reserved_gb=peak_reserved_gb, probe=probe, bd=bd, warm=warm,
        manifest_before=manifest_before, manifest_after=manifest_after,
        manifest=manifest, error_excerpt=error_excerpt, dry=dry,
        projection=projection)
    if not dry:
        write_text(run_dir / "EXCLUDED_FROM_BANK",
                   "Granularity-ladder adapter — excluded from every Asset-1 "
                   "bank per the locked granularity card.\n" + utc_now() + "\n")

    # The trainer records the realized pool; assert it against the frozen
    # manifest so a seam failure can never pass silently as a trained run.
    cfg_path = run_dir / "config.json"
    if status == "COMPLETE" and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        n_pool = cfg.get("dataset", {}).get("n_pool")
        if n_pool != entry["n_train_pool"]:
            print(f"*** POOL MISMATCH: trainer served {n_pool} training rows, "
                  f"frozen manifest says {entry['n_train_pool']} — this run "
                  f"is NOT usable; investigate the seam before continuing.")
            return 1

    print(f"\n{'-' * 70}")
    print(f"status         = {status}")
    print(f"wall_clock_min = {_fmt(wall_run_s / 60.0)}")
    print(f"peak_vram_gb   = {_fmt(peak_alloc_gb)}")
    print(f"mean_step_s    = {_fmt(bd['mean_step_s'], 4)}")
    if projection:
        print(f"PROJECTED full-run min = {_fmt(projection['projected_min'])} "
              f"(measured llama basis 30.56)")
    if manifest_before != manifest_after:
        print("*** WARNING: bank_manifest.json changed during this run.")
    print(f"wrote          {path}")
    print(f"{'-' * 70}\n")
    return 0 if status == "COMPLETE" else 1


def show_plan(level: str) -> int:
    manifest, _pools = load_level(level)
    plan = plan_runs(level, manifest)
    print(f"\nLEVEL {level} — {manifest['k_materialized']} classes x "
          f"{LEVEL_SEEDS[level]} seeds = {len(plan)} runs")
    print(f"launchable = {manifest['launchable']}  "
          f"balance = {manifest['balance_policy']}")
    print(f"seed band  = {GRAN_SEED_BASE + EMITTED_LEVELS.index(level) * LEVEL_BAND}"
          f" + run_k   (data_seed band "
          f"{GRAN_DATA_SEED_BASE + EMITTED_LEVELS.index(level) * LEVEL_BAND})")
    print(f"projected  = {len(plan)} x 30.56 min = "
          f"{len(plan) * 30.56 / 60 / 24:.3f} GPU-days "
          f"(measured llama basis, RATE_EXTRACT.md)")
    for e in plan[:len(live_classes(manifest))]:
        d = run_dir_for(level, e["run_k"])
        state = ("COMPLETE" if (d / "COMPLETE").exists()
                 else "FAILED" if (d / "FAILED").exists() else "pending")
        print(f"  run_{e['run_k']:03d} {e['class_id']:34s} {e['tier']} "
              f"seed={e['seed']} pool={e['n_train_pool']:6d} {state}")
    if len(plan) > len(live_classes(manifest)):
        print(f"  ... (first replicate shown; {len(plan)} runs total)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train ONE granularity-ladder adapter with the "
                    "unmodified Asset-1 trainer on a frozen class pool.")
    ap.add_argument("--level", required=True, choices=EMITTED_LEVELS)
    ap.add_argument("--run", type=int, help="run index within the level")
    ap.add_argument("--dry-run", action="store_true",
                    help=f"{DRY_RUN_STEPS}-step wiring check + projection")
    ap.add_argument("--steps", type=int, default=DRY_RUN_STEPS,
                    help="dry-run step count (ignored for real runs)")
    ap.add_argument("--plan", action="store_true",
                    help="print the level's run plan and exit")
    args = ap.parse_args(argv)

    if args.plan:
        return show_plan(args.level)
    if args.run is None:
        ap.error("--run is required unless --plan is given")
    return execute(args.level, args.run, args.dry_run, args.steps)


if __name__ == "__main__":
    sys.exit(main())
