"""S2 timing pilots — the registered H2-at-scale precondition.

REGISTERED CARD: docs/REGISTRATION_H2_SCALE_2026-07-30.md
    + docs/DIRECTOR_RULINGS_LADDER_ERRATA_REGISTRATIONS_2026-07-30.md Part 4.
    "S2 - Timing pilots: APPROVED, and PROMOTED to a precondition. Three
    timing-only runs per family, excluded from the bank, before any S1
    commitment. Timing-only runs are not unblinding: they compute no H2
    statistic, and wall-clock carries no task-transfer information."

WHAT THIS SCRIPT IS: a thin wrapper around the Asset-1 trainer. It imports
    asset1_bank.run_single and calls it unmodified, so a pilot run executes
    the IDENTICAL recipe (2000 steps, bs4xga4 = effective batch 16, 512
    tokens, rank 24, n_channels 6, identity bridge, LM loss only, bf16 base /
    fp32 adapters, gradient checkpointing, val loss every 100 steps over the
    fixed val_seed=777 split). The trainer logic is NOT forked or edited.
    Constants are imported, never redeclared.

WHAT IS ADDED: timing instrumentation only, by delegation (wrappers that
    record and call through), never by editing the training path:
      * time.monotonic around run_single, and around the process
      * torch.cuda.max_memory_allocated / max_memory_reserved (peak reset
        immediately before the run)
      * transformers.AutoModelForCausalLM.from_pretrained wrapped to record
        model-load seconds, measured base parameter count, and the attention
        implementation actually selected
      * asset1_bank.evaluate_val_loss wrapped to record per-eval seconds, so
        the step loop can be isolated from validation exactly
    Every number written is measured or derived from measured numbers by a
    stated formula. Nothing is estimated.

WHAT IS REFUSED: any write under results/asset1-bank/ (or any sibling whose
    first path component starts with "asset1-bank"). Every output path is
    checked against the pilot root before use, and bank_manifest.json is
    sha256'd before and after each run and asserted unchanged. Pilots are
    also excluded structurally: they live outside every bank root, so the
    analysis CLIs (which glob results/asset1-bank/...) cannot see them, and
    each run dir carries an EXCLUDED_FROM_BANK marker.

NO H2 STATISTIC IS COMPUTED HERE. No adapter is compared to any other. The
    outputs are wall clock, peak VRAM, and the recorded config.

Family ids: the draft's Section 3 names families ("Gemma-2-2B", "Qwen2.5-3B",
    "Qwen2.5-7B", "Llama-3.1-8B") and the recipe line requires "instruct
    checkpoints"; it does not print HF repo ids. The ids below are that
    resolution, each verified to exist on the Hub by --gate-check. They are
    flagged for Director confirmation in GATING.md.

Usage (always with HF_HUB_CACHE / HF_DATASETS_CACHE set - see GATING.md):
    python scripts/s2_timing_pilot.py --gate-check
    python scripts/s2_timing_pilot.py --family gemma2-2b --dry-run
    python scripts/s2_timing_pilot.py --family gemma2-2b --run 1
    python scripts/s2_timing_pilot.py --rates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
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

# The trainer and its locked constants — imported, never redeclared.
import gpu_guard  # noqa: E402  (retrofit 2026-08-11: pilot predates the arbiter)
import asset1_bank as ab  # noqa: E402
from asset1_bank import (  # noqa: E402
    BATCH_SIZE,
    CampaignSpec,
    EVAL_INTERVAL,
    GRAD_ACCUM,
    MAX_LEN,
    MAX_STEPS,
    RunSpec,
    git_commit_hash,
    library_versions,
    probe_family_access,
    run_single,
    utc_now,
)
from asset1_datasets import POOL_CAP  # noqa: E402

# ── Pilot constants (S2) ────────────────────────────────────────────

PILOT_ROOT = REPO_ROOT / "results" / "s2-timing-pilots"
PILOT_TAG = "s2-timing-pilot"

# Draft Section 3 family set, resolved to instruct checkpoints.
# est_min_run / approx_params_b are the DRAFT'S ESTIMATES, carried here only
# so the report can state measured-vs-estimated. They are never used in any
# computation.
FAMILIES: dict[str, dict] = {
    "gemma2-2b": {
        "model": "google/gemma-2-2b-it",
        "draft_label": "Gemma-2-2B",
        "draft_approx_params_b": 2.6,
        "draft_est_min_run": 79,
        "draft_reps": 20,
        "draft_runs": 120,
        "guard_needed_gb": 15,      # MEASURED peak 14.07 GB (RATES.md)
        "guard_expected_min": 76,   # MEASURED mean 75.51 min/run (RATES.md)
    },
    "qwen2.5-3b": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "draft_label": "Qwen2.5-3B",
        "draft_approx_params_b": 3.1,
        "draft_est_min_run": 93,
        "draft_reps": 20,
        "draft_runs": 120,
        "guard_needed_gb": 12,      # MEASURED peak 11.05 GB (RATES.md)
        "guard_expected_min": 83,   # MEASURED mean 82.86 min/run (RATES.md)
    },
    "qwen2.5-7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "draft_label": "Qwen2.5-7B",
        "draft_approx_params_b": 7.6,
        "draft_est_min_run": 230,
        "draft_reps": 10,
        "draft_runs": 60,
        "guard_needed_gb": 21,      # MEASURED peak 20.30 GB (RATES.md)
        "guard_expected_min": 154,  # MEASURED mean 153.01 min/run (RATES.md)
    },
    "llama3.1-8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "draft_label": "Llama-3.1-8B",
        "draft_approx_params_b": 8.0,
        "draft_est_min_run": 243,
        "draft_reps": 10,
        "draft_runs": 60,
        "guard_needed_gb": 22,      # PROJECTED (unmeasured; 8.0B vs qwen-7b's
                                    # 7.6B at 20.30 GB — replaced by the pilot's
                                    # own measurement once it exists)
        "guard_expected_min": 160,  # PROJECTED 159.20 from the affine fit
                                    # (docs/S2_COST_RESTATEMENT_2026-08-04.md)
    },
}

# Family order = the draft's table order (cheapest first), which is also the
# frozen S9 tier order. Pilots run in this order.
FAMILY_ORDER = ["gemma2-2b", "qwen2.5-3b", "qwen2.5-7b", "llama3.1-8b"]

N_PILOT_RUNS = 3

# Task assignment: MERIDIAN'S CHOICE, not a Director ruling — the registered
# card specifies "3 timing-only runs per family" without naming tasks. The
# three runs take the first three tasks in the bank's task order so the pilot
# samples dataset-build cost across three different sources; per-step cost is
# task-invariant by construction (every sequence is padded to max_len, so
# every batch carries an identical token count). Recorded per run and raised
# for the Director in the S2 report.
PILOT_TASKS = ["alpaca", "code", "math"]

# Distinct seed bases so a pilot seed can never be confused with a bank seed
# (bank: 10000 + idx / 20000 + idx).
PILOT_SEED_BASE = 90_000
PILOT_DATA_SEED_BASE = 91_000

DRY_RUN_STEPS = 10

BANK_ROOTS_FORBIDDEN = (REPO_ROOT / "results").resolve()


# ── Hard refusal: nothing may be written under a bank root ──────────


def _forbidden_component(resolved: Path) -> str | None:
    """Return the offending path component if `resolved` sits in a bank root.

    Covers results/asset1-bank/ and results/asset1-bank-bs2x8-archive/.
    """
    try:
        rel = resolved.relative_to(BANK_ROOTS_FORBIDDEN)
    except ValueError:
        return None
    for part in rel.parts:
        if part.lower().startswith("asset1-bank"):
            return part
    return None


def assert_pilot_path(path: Path) -> Path:
    """Every output path passes through here before it is written."""
    resolved = Path(path).resolve()
    offender = _forbidden_component(resolved)
    if offender is not None:
        raise SystemExit(
            f"REFUSED: {resolved} lies under the bank root '{offender}'. "
            f"S2 pilots are excluded from every bank by the registered card; "
            f"this script never writes there."
        )
    pilot = PILOT_ROOT.resolve()
    if resolved != pilot and pilot not in resolved.parents:
        raise SystemExit(
            f"REFUSED: {resolved} is outside the pilot root {pilot}."
        )
    return resolved


def _bank_manifest_digest() -> str | None:
    """sha256 of the Asset-1 bank manifest, or None if absent."""
    p = ab.DEFAULT_BANK_ROOT / "bank_manifest.json"
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── Small helpers ───────────────────────────────────────────────────


def _fmt(v, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def typed_block(pairs: list[tuple[str, object]]) -> str:
    width = max(len(k) for k, _ in pairs)
    lines = [f"{k.ljust(width)} = {v}" for k, v in pairs]
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    assert_pilot_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def hf_cache_is_warm(model_id: str) -> bool:
    """True if a snapshot for `model_id` already exists in HF_HUB_CACHE.

    A cold cache means the run's wall clock includes the weight download,
    which is why this is recorded next to wall_clock_min.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return False
    d = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not d.exists():
        return False
    for snap in d.iterdir():
        if any(snap.glob("*.safetensors")) or any(snap.glob("*.bin")):
            return True
    return False


# ── Instrumentation by delegation (no trainer logic is forked) ──────


class Probe:
    """Records timings/facts as the unmodified trainer runs through it."""

    def __init__(self) -> None:
        self.model_load_s: float | None = None
        self.n_params_base: int | None = None
        self.attn_implementation: str | None = None
        self.model_dtype: str | None = None
        self.eval_seconds: list[float] = []

    def install(self):
        import transformers

        auto = transformers.AutoModelForCausalLM
        orig_load = auto.from_pretrained
        had_own = "from_pretrained" in auto.__dict__
        own = auto.__dict__.get("from_pretrained")
        orig_eval = ab.evaluate_val_loss

        def timed_load(*args, **kwargs):
            t0 = time.monotonic()
            model = orig_load(*args, **kwargs)
            self.model_load_s = time.monotonic() - t0
            try:
                self.n_params_base = sum(p.numel() for p in model.parameters())
                self.attn_implementation = getattr(
                    model.config, "_attn_implementation", None)
                self.model_dtype = str(next(model.parameters()).dtype)
            except Exception:
                pass
            return model

        def timed_eval(*args, **kwargs):
            t0 = time.monotonic()
            out = orig_eval(*args, **kwargs)
            self.eval_seconds.append(time.monotonic() - t0)
            return out

        auto.from_pretrained = timed_load
        ab.evaluate_val_loss = timed_eval

        def restore() -> None:
            if had_own:
                auto.from_pretrained = own
            else:
                # It was inherited (_BaseAutoModelClass) — remove the override
                # rather than pinning a bound method onto the subclass.
                del auto.from_pretrained
            ab.evaluate_val_loss = orig_eval

        return restore


# ── Pilot run construction ─────────────────────────────────────────


def pilot_spec(max_steps: int) -> CampaignSpec:
    """A CampaignSpec whose bank_root is the pilot root.

    run_single() reads only tag / max_steps / pool_cap from the spec; the
    campaign orchestration functions (which write manifests) are never called.
    """
    root = assert_pilot_path(PILOT_ROOT)
    return CampaignSpec(
        tag=PILOT_TAG, bank_root=root, families=[], tasks=[],
        n_replicates=0, max_steps=max_steps, pool_cap=POOL_CAP,
    )


def pilot_run(family_key: str, run_k: int, dry: bool) -> RunSpec:
    fam = FAMILIES[family_key]
    fam_index = FAMILY_ORDER.index(family_key)
    # run_index is a pilot-local index; run_k=0 is reserved for the dry run.
    run_index = fam_index * 10 + run_k
    task = PILOT_TASKS[0] if dry else PILOT_TASKS[(run_k - 1) % len(PILOT_TASKS)]
    sub = "dryrun" if dry else f"run_{run_k}"
    run_dir = assert_pilot_path(PILOT_ROOT / family_key / sub)
    return RunSpec(
        run_index=run_index,
        family=fam["model"],
        family_short=family_key,
        task=task,
        replicate=0,
        seed=PILOT_SEED_BASE + run_index,
        data_seed=PILOT_DATA_SEED_BASE + run_index,
        run_dir=run_dir,
    )


# ── Derivation of the timing breakdown from measured quantities ─────


def derive_breakdown(run_dir: Path, probe: Probe, steps: int) -> dict:
    """Split the run into setup / step loop / validation / save.

    All inputs are measured: metrics.json's per-eval wall_time_s stamps (the
    trainer's own clock, relative to run entry) and the probe's per-eval
    seconds. Formulas:
        setup_s      = records[0].wall_time_s - eval_s[0]
        step_loop_s  = (records[-1].wall - records[0].wall) - sum(eval_s[1:])
        save_s       = metrics.wall_time_seconds - records[-1].wall_time_s
        mean_step_s  = step_loop_s / steps_recorded
    """
    out: dict = {"setup_s": None, "step_loop_s": None, "save_s": None,
                 "mean_step_s": None, "val_evals": len(probe.eval_seconds),
                 "mean_val_eval_s": None, "steps_recorded": None}
    if probe.eval_seconds:
        out["mean_val_eval_s"] = statistics.fmean(probe.eval_seconds)
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return out
    try:
        m = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return out
    recs = m.get("records") or []
    if not recs:
        return out
    first, last = recs[0], recs[-1]
    out["steps_recorded"] = last.get("step")
    if probe.eval_seconds:
        out["setup_s"] = first["wall_time_s"] - probe.eval_seconds[0]
        loop = (last["wall_time_s"] - first["wall_time_s"]
                - sum(probe.eval_seconds[1:]))
        out["step_loop_s"] = loop
        if last.get("step"):
            out["mean_step_s"] = loop / last["step"]
    total = m.get("wall_time_seconds")
    if total is not None:
        out["save_s"] = total - last["wall_time_s"]
    return out


def project_full_run(bd: dict, probe: Probe) -> dict:
    """Project a 2000-step run from a short dry run. STATED AS A PROJECTION.

    projected_s = setup + n_evals_full * mean_eval + MAX_STEPS * mean_step
                  + save,  n_evals_full = 1 + MAX_STEPS // EVAL_INTERVAL
    Upper-biased: mean_step is amortized over few steps, so it carries the
    one-time first-step cost (allocator warm-up + the trainer's
    gradient-checkpointing verification pass).
    """
    n_evals_full = 1 + MAX_STEPS // EVAL_INTERVAL
    if bd["mean_step_s"] is None or bd["mean_val_eval_s"] is None:
        return {"projected_min": None, "n_evals_full": n_evals_full}
    total_s = ((bd["setup_s"] or 0.0)
               + n_evals_full * bd["mean_val_eval_s"]
               + MAX_STEPS * bd["mean_step_s"]
               + (bd["save_s"] or 0.0))
    # On a cold cache, model_load_s contains the weight download, which a
    # campaign pays once rather than per run. Reported separately, not netted
    # out of the headline projection.
    warm = total_s - (probe.model_load_s or 0.0)
    return {"projected_min": total_s / 60.0, "n_evals_full": n_evals_full,
            "projected_min_excl_model_load": warm / 60.0,
            "projected_gpu_days_per_60_runs": total_s * 60 / 86400.0}


# ── TIMING.md ───────────────────────────────────────────────────────


def tokens_trained(steps: int) -> int:
    """steps x batch x grad_accum x max_len — exact, not sampled.

    drop_last=True and every sequence padded to max_len, so each optimizer
    step consumes exactly batch*accum*max_len token positions. Counts padded
    positions: the pilot inherits the bank's label convention (labels over the
    full padded sequence).
    """
    return steps * BATCH_SIZE * GRAD_ACCUM * MAX_LEN


def write_timing_md(run: RunSpec, family_key: str, *, status: str, steps: int,
                    wall_run_s: float, wall_proc_s: float, peak_alloc_gb,
                    peak_reserved_gb, probe: Probe, bd: dict, warm: bool,
                    manifest_before: str | None, manifest_after: str | None,
                    error_excerpt: str | None, dry: bool,
                    projection: dict | None) -> Path:
    fam = FAMILIES[family_key]
    cfg = {}
    cpath = run.run_dir / "config.json"
    if cpath.exists():
        try:
            cfg = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    versions = library_versions()

    core: list[tuple[str, object]] = [
        ("model_id", run.family),
        ("steps", steps),
        ("batch_geometry", f"bs{BATCH_SIZE}xga{GRAD_ACCUM}"),
        ("wall_clock_min", _fmt(wall_run_s / 60.0)),
        ("peak_vram_gb", _fmt(peak_alloc_gb)),
        ("tokens_trained", tokens_trained(steps)),
        ("date", date.today().isoformat()),
    ]
    context: list[tuple[str, object]] = [
        ("family", family_key),
        ("draft_label", fam["draft_label"]),
        ("run_kind", "dry-run (projection only)" if dry else
         f"timing run {run.run_dir.name.replace('run_', '')} of {N_PILOT_RUNS}"),
        ("status", status),
        ("task", run.task),
        ("task_choice_authority", "Meridian (card specifies count, not tasks)"),
        ("seed", run.seed),
        ("data_seed", run.data_seed),
        ("effective_batch", BATCH_SIZE * GRAD_ACCUM),
        ("max_len", MAX_LEN),
        ("wall_clock_incl_process_min", _fmt(wall_proc_s / 60.0)),
        ("peak_vram_reserved_gb", _fmt(peak_reserved_gb)),
        ("tokens_per_second", _fmt(tokens_trained(steps) / wall_run_s
                                   if wall_run_s else None, 1)),
        ("hf_cache_warm_at_start", str(warm).lower()),
        ("n_params_base_measured", probe.n_params_base),
        ("n_injected_modules", cfg.get("n_injected_modules")),
        ("attn_implementation", probe.attn_implementation),
        ("model_dtype", probe.model_dtype),
    ]
    breakdown: list[tuple[str, object]] = [
        ("model_load_s", _fmt(probe.model_load_s)),
        ("setup_s", _fmt(bd["setup_s"])),
        ("step_loop_s", _fmt(bd["step_loop_s"])),
        ("steps_recorded", bd["steps_recorded"]),
        ("mean_step_s", _fmt(bd["mean_step_s"], 4)),
        ("val_evals", bd["val_evals"]),
        ("mean_val_eval_s", _fmt(bd["mean_val_eval_s"])),
        ("save_s", _fmt(bd["save_s"])),
    ]
    provenance: list[tuple[str, object]] = [
        ("EXCLUDED_FROM_BANK", "true"),
        ("computes_h2_statistics", "false"),
        ("campaign_tag", cfg.get("campaign_tag", PILOT_TAG)),
        ("config_asset_field", cfg.get("asset", "—")),
        ("trainer", "asset1_bank.run_single (imported unmodified)"),
        ("recipe_deviations", "none" if not dry else
         f"steps={steps} (dry run); LR schedule truncated — timing only"),
        ("bank_manifest_sha256_before", (manifest_before or "—")[:16]),
        ("bank_manifest_sha256_after", (manifest_after or "—")[:16]),
        ("bank_manifest_unchanged",
         str(manifest_before == manifest_after).lower()),
        ("git_commit", cfg.get("git_commit", git_commit_hash())),
        ("gpu", versions.get("gpu", "—")),
        ("python", versions.get("python")),
        ("torch", versions.get("torch")),
        ("transformers", versions.get("transformers")),
        ("finished_at_utc", utc_now()),
    ]

    title = (f"# S2 TIMING PILOT — {family_key} "
             f"{'DRY RUN' if dry else run.run_dir.name}")
    parts = [
        title, "",
        "Timing-only run under the registered H2-at-scale card "
        "(`docs/REGISTRATION_H2_SCALE_2026-07-30.md`, S2). Excluded from "
        "every bank. No H2 statistic is computed. Every value below is "
        "measured, or derived from measured values by the formula stated in "
        "`scripts/s2_timing_pilot.py`.", "",
        "=== VERIFIED STATE ===", typed_block(core), "=== END VERIFIED STATE ===",
        "", "## Context", "", typed_block(context),
        "", "## Measured breakdown", "", typed_block(breakdown),
    ]
    if projection:
        proj = [("projection_basis_steps", bd["steps_recorded"]),
                ("n_evals_in_full_run", projection["n_evals_full"]),
                ("PROJECTED_min_per_full_run",
                 _fmt(projection["projected_min"])),
                ("PROJECTED_min_excl_model_load",
                 _fmt(projection.get("projected_min_excl_model_load"))),
                ("PROJECTED_gpu_days_per_60_runs",
                 _fmt(projection.get("projected_gpu_days_per_60_runs"))),
                ("draft_estimate_min_per_run", fam["draft_est_min_run"]),
                ("projection_bias", "upper (first-step cost amortized over "
                                    "the dry run's few steps)")]
        parts += ["", "## Projection (NOT a measurement)", "",
                  typed_block(proj)]
    if error_excerpt:
        parts += ["", "## Failure detail (recorded as a finding)", "",
                  "```", error_excerpt.strip(), "```"]
    parts += ["", "## Provenance", "", typed_block(provenance), ""]

    path = run.run_dir / ("DRYRUN.md" if dry else "TIMING.md")
    write_text(path, "\n".join(parts))
    if not dry:
        write_text(run.run_dir / "EXCLUDED_FROM_BANK",
                   "S2 timing pilot — excluded from every bank per the "
                   "registered H2-at-scale card.\n" + utc_now() + "\n")
    return path


# ── Aggregate rates table ──────────────────────────────────────────


def parse_typed(path: Path) -> dict:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " not in line or line.startswith("#"):
            continue
        k, _, v = line.partition(" = ")
        out.setdefault(k.strip(), v.strip())
    return out


def rebuild_rates() -> Path:
    rows: list[dict] = []
    for fam in FAMILY_ORDER:
        for k in range(1, N_PILOT_RUNS + 1):
            p = PILOT_ROOT / fam / f"run_{k}" / "TIMING.md"
            if p.exists():
                d = parse_typed(p)
                d["_family"], d["_k"] = fam, str(k)
                rows.append(d)
    lines = [
        "# S2 measured rates — running aggregate", "",
        "Rebuilt by `python scripts/s2_timing_pilot.py --rates` from the "
        "per-run TIMING.md files. Measured only; no estimates. The Section 3 "
        "cost-table restatement required before the card locks is written "
        "separately, from these numbers.", "",
        "| family | run | task | status | wall_clock_min | peak_vram_gb | "
        "mean_step_s | tokens_trained | cache_warm |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in rows:
        lines.append(
            f"| {d['_family']} | {d['_k']} | {d.get('task', '—')} | "
            f"{d.get('status', '—')} | {d.get('wall_clock_min', '—')} | "
            f"{d.get('peak_vram_gb', '—')} | {d.get('mean_step_s', '—')} | "
            f"{d.get('tokens_trained', '—')} | "
            f"{d.get('hf_cache_warm_at_start', '—')} |")
    if not rows:
        lines.append("| — | — | — | — | — | — | — | — | — |")

    lines += ["", "## Per-family mean (completed runs only)", ""]
    agg: list[tuple[str, object]] = []
    for fam in FAMILY_ORDER:
        vals = [float(d["wall_clock_min"]) for d in rows
                if d["_family"] == fam and d.get("status") == "COMPLETE"
                and d.get("wall_clock_min") not in (None, "—")]
        if vals:
            agg.append((f"{fam}_mean_min_per_run", f"{statistics.fmean(vals):.2f}"))
            agg.append((f"{fam}_n_runs_measured", len(vals)))
        else:
            agg.append((f"{fam}_mean_min_per_run", "—"))
    lines += [typed_block(agg) if agg else "—", ""]
    path = PILOT_ROOT / "RATES.md"
    write_text(path, "\n".join(lines))
    return path


# ── Gate check ─────────────────────────────────────────────────────


def gate_check() -> dict[str, str]:
    print("S2 gating check — config-only probe per family "
          "(hf_hub_download of config.json; model_info cannot be trusted on "
          "gated repos).\n")
    results: dict[str, str] = {}
    details: dict[str, str] = {}
    for fam in FAMILY_ORDER:
        model = FAMILIES[fam]["model"]
        access = probe_family_access(model)
        results[fam] = access
        if access != "OK":
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(model, "config.json")
            except Exception as e:
                raw = [ln.strip() for ln in
                       f"{type(e).__name__}: {e}".splitlines() if ln.strip()]
                details[fam] = "\n".join(raw[:3])[:600]
        print(f"  {fam:<12} {model:<36} {access}")

    lines = [
        "# S2 timing pilots — family gating status", "",
        f"Probed {date.today().isoformat()} on Timothy's workstation "
        "(single RTX 6000 Ada 48GB), account `timotheospaul`, via "
        "`asset1_bank.probe_family_access` (config-only download; the same "
        "classifier the Asset-1 campaign used).", "",
        "## Model-id resolution (needs Director confirmation)", "",
        "The registered card and the draft's Section 3 name families and "
        "require instruct checkpoints; neither prints HF repo ids. The ids "
        "below are Meridian's resolution of those labels, each verified to "
        "exist on the Hub. `Qwen/Qwen2.5-7B-Instruct` is independently "
        "attested in this repo (the pilot bank and BM-003 both used it).", "",
        "=== VERIFIED STATE ===",
        typed_block([(f"{fam}_model_id", FAMILIES[fam]["model"])
                     for fam in FAMILY_ORDER]
                    + [(f"{fam}_access", results[fam]) for fam in FAMILY_ORDER]),
        "=== END VERIFIED STATE ===", "",
        "## Consequence", "",
    ]
    blocked = [f for f in FAMILY_ORDER if results[f] == "BLOCKED"]
    ok = [f for f in FAMILY_ORDER if results[f] == "OK"]
    lines.append(f"Accessible, pilots run: {', '.join(ok) if ok else 'none'}.")
    if blocked:
        lines += [
            "", f"SKIPPED (license gate not accepted on this account): "
                f"{', '.join(blocked)}. Not a blocker — Timothy accepts model "
                f"gates manually; re-run `--gate-check` after acceptance and "
                f"the skipped family's pilots proceed unchanged. Until then "
                f"its measured rate does not exist, so the Section 3 cost "
                f"table cannot be restated for it.",
        ]
        for fam in blocked:
            if fam in details:
                lines += ["", f"`{fam}` probe error, verbatim first line:",
                          "", "```", details[fam], "```"]
        if "llama3.1-8b" in blocked:
            lines += [
                "", "Note for the Director: `meta-llama/Llama-3.2-1B-Instruct` "
                "(an Asset-1 anchor family) IS accessible on this account — "
                "Llama-3.1-8B is a separately gated repo, not a lapsed "
                "account. The draft's ALTERNATE for a license-blocked "
                "Llama-3.1-8B is Mistral-7B (Section 3), and its drop option "
                "is Llama-3.1-8B itself; both remain open. No substitution is "
                "made here — that is an S1 decision.",
            ]
    lines += [
        "", "## Environment required on every model-load step", "",
        typed_block([
            ("HF_HUB_CACHE", "C:\\falco\\hf-cache\\hub"),
            ("HF_DATASETS_CACHE", "C:\\falco\\hf-cache\\datasets"),
            ("reason", "the default HF cache is a junction onto a full drive; "
                       "launching without these fails on xsum/squad"),
        ]), "",
    ]
    path = PILOT_ROOT / "GATING.md"
    write_text(path, "\n".join(lines))
    print(f"\nWrote {path}")
    return results


# ── The pilot run ──────────────────────────────────────────────────


def execute(family_key: str, run_k: int, dry: bool, steps_override: int) -> int:
    fam = FAMILIES[family_key]
    access = probe_family_access(fam["model"])
    if access == "BLOCKED":
        print(f"REFUSED: {fam['model']} is license-gated on this account "
              f"(BLOCKED). Run --gate-check; the family is skipped, not "
              f"worked around.")
        return 78

    steps = steps_override if dry else MAX_STEPS
    spec = pilot_spec(steps)
    run = pilot_run(family_key, run_k, dry)
    assert_pilot_path(run.run_dir)
    run.run_dir.mkdir(parents=True, exist_ok=True)

    # A dry run is re-runnable: clear its COMPLETE marker so the trainer does
    # not skip. Timing runs are never silently overwritten.
    if dry:
        for name in ("COMPLETE", "FAILED", "error.log"):
            p = run.run_dir / name
            if p.exists():
                p.unlink()
    elif (run.run_dir / "COMPLETE").exists():
        print(f"{run.run_dir} already COMPLETE — refusing to overwrite a "
              f"measured timing run. Delete it deliberately to re-measure.")
        return 0

    warm = hf_cache_is_warm(fam["model"])
    manifest_before = _bank_manifest_digest()

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    probe = Probe()
    restore = probe.install()
    print(f"\n{'=' * 70}")
    print(f"S2 timing pilot | {family_key} | {fam['model']}")
    print(f"{'dry run' if dry else 'timing run ' + str(run_k)} | task="
          f"{run.task} | steps={steps} | bs{BATCH_SIZE}xga{GRAD_ACCUM} | "
          f"hf_cache_warm={warm}")
    print(f"out: {run.run_dir}")
    print(f"{'=' * 70}\n", flush=True)

    # gpu_guard retrofit (2026-08-11): explicit calls rather than guarded()
    # so any pause/preflight wait happens OUTSIDE the timed window —
    # wall_clock_min is the rate basis and a guard wait inside it would
    # corrupt the measurement.
    gpu_guard.pause_gate()
    gpu_guard.preflight(fam["guard_needed_gb"])
    gpu_guard.claim(f"s2-timing-pilot {family_key} run_{run_k}",
                    fam["guard_needed_gb"], fam["guard_expected_min"])

    t0 = time.monotonic()
    try:
        rc = run_single(spec, run)
    finally:
        restore()
        gpu_guard.release()
    wall_run_s = time.monotonic() - t0

    peak_alloc_gb = peak_reserved_gb = None
    if torch.cuda.is_available():
        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

    status = {0: "COMPLETE", 78: "BLOCKED"}.get(rc, "FAILED")
    error_excerpt = None
    if status == "FAILED":
        elog = run.run_dir / "error.log"
        if elog.exists():
            txt = elog.read_text(encoding="utf-8", errors="replace").strip()
            lines = txt.splitlines()
            error_excerpt = ("\n".join(lines[:6] + (["..."] if len(lines) > 12
                                                    else []) + lines[-6:])
                             if len(lines) > 12 else txt)

    bd = derive_breakdown(run.run_dir, probe, steps)
    projection = project_full_run(bd, probe) if dry else None
    manifest_after = _bank_manifest_digest()

    path = write_timing_md(
        run, family_key, status=status, steps=bd["steps_recorded"] or steps,
        wall_run_s=wall_run_s, wall_proc_s=time.monotonic() - PROCESS_T0,
        peak_alloc_gb=peak_alloc_gb, peak_reserved_gb=peak_reserved_gb,
        probe=probe, bd=bd, warm=warm, manifest_before=manifest_before,
        manifest_after=manifest_after, error_excerpt=error_excerpt, dry=dry,
        projection=projection)

    print(f"\n{'-' * 70}")
    print(f"status           = {status}")
    print(f"wall_clock_min   = {_fmt(wall_run_s / 60.0)}")
    print(f"peak_vram_gb     = {_fmt(peak_alloc_gb)} "
          f"(reserved {_fmt(peak_reserved_gb)})")
    print(f"mean_step_s      = {_fmt(bd['mean_step_s'], 4)}")
    if projection:
        print(f"PROJECTED full-run min = "
              f"{_fmt(projection['projected_min'])}  "
              f"(draft estimate {fam['draft_est_min_run']})")
    if manifest_before != manifest_after:
        print("*** WARNING: bank_manifest.json changed during this run — "
              "investigate before trusting anything here.")
    print(f"wrote            {path}")
    print(f"{'-' * 70}\n")

    if not dry:
        print(f"rates table: {rebuild_rates()}")
    return 0 if status == "COMPLETE" else 1


# ── CLI ────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="S2 timing-only pilots for the registered H2-at-scale "
                    "card. Excluded from every bank; computes no H2 statistic.")
    ap.add_argument("--family", choices=FAMILY_ORDER,
                    help="family to run (draft table order = frozen tier order)")
    ap.add_argument("--run", type=int, choices=range(1, N_PILOT_RUNS + 1),
                    help=f"timing run k of {N_PILOT_RUNS} (full length)")
    ap.add_argument("--dry-run", action="store_true",
                    help=f"load the model, run {DRY_RUN_STEPS} steps, report a "
                         f"projected min/run (no timing run is recorded)")
    ap.add_argument("--dry-run-steps", type=int, default=DRY_RUN_STEPS,
                    help="steps for --dry-run (default %(default)s; more steps "
                         "tighten the projection)")
    ap.add_argument("--gate-check", action="store_true",
                    help="probe every family's license gate, write GATING.md")
    ap.add_argument("--rates", action="store_true",
                    help="rebuild RATES.md from the per-run TIMING.md files")
    ap.add_argument("--list", action="store_true",
                    help="print the pilot plan and each run's status")
    args = ap.parse_args()

    if args.gate_check:
        gate_check()
        return
    if args.rates:
        print(f"wrote {rebuild_rates()}")
        return
    if args.list:
        print(f"S2 pilot plan — {N_PILOT_RUNS} timing runs per family, "
              f"tasks {PILOT_TASKS} (run k -> task k)")
        for fam in FAMILY_ORDER:
            for k in range(1, N_PILOT_RUNS + 1):
                d = PILOT_ROOT / fam / f"run_{k}"
                state = ("COMPLETE" if (d / "COMPLETE").exists()
                         else "FAILED" if (d / "FAILED").exists()
                         else "pending")
                print(f"  {fam:<12} run_{k}  {state:<9} {FAMILIES[fam]['model']}")
        return

    if not args.family:
        ap.error("--family is required (or use --gate-check / --rates / --list)")
    if args.dry_run:
        sys.exit(execute(args.family, 0, True, args.dry_run_steps))
    if not args.run:
        ap.error("--run k is required for a timing run (or use --dry-run)")
    sys.exit(execute(args.family, args.run, False, 0))


if __name__ == "__main__":
    main()
