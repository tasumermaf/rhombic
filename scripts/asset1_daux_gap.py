"""Asset 1 — D-aux: overfit detection via bridge deviation <-> generalization gap.

Re-verifies the pilot's r = 0.888 deviation<->gap correlation on the D1
bank (experiment card 2026-07-03, "D-aux — rides along"; prior art in kind:
WeightWatcher-PEFT). Pure post-hoc analysis of saved artifacts — no
training access, no GPU.

WHAT "DEVIATION" IS IN THIS BANK (verified by code reading, 2026-07-06)
-----------------------------------------------------------------------
The bank trains with bridge_mode='identity'. The question the card ordered
answered first: are bridges TRAINABLE in this regime, so that
deviation-from-identity of bridge_final is a trained quantity? YES:

* rhombic/nn/topology.py:133-134 — bridge_init('identity') returns
  np.eye(n_channels): every bridge starts exactly at I (the saved
  bridge_step0 npys are the identity — our step-0 control).
* rhombic/nn/rhombi_lora.py:165-170 — in standard mode (bridge_mode=
  'identity', no master_bridge, not rd_graph) the bridge is stored as
  nn.Parameter(...), which has requires_grad=True by default. The only
  freezing path is freeze_bridge() (rhombi_lora.py:252-261).
* scripts/train_task_fingerprint.py:163-200 — the bank's injector (reused
  via asset1_bank.inject_rhombi_lora, scripts/asset1_bank.py:397-411)
  constructs RhombiLoRALinear(bridge_mode='identity') and freezes ONLY the
  wrapped base Linear's parameters (train_task_fingerprint.py:194-195).
  freeze_bridge() is never called anywhere in the bank pipeline.
* scripts/asset1_bank.py:585-588 — the AdamW optimizer is built over every
  requires_grad parameter of the wrapped model, which includes all bridges.
* scripts/asset1_bank.py:437-475 — verify_gradient_checkpointing HARD-FAILS
  the run unless every module's bridge has a non-None, finite gradient
  after the first backward. The live campaign passing this check is
  production proof that bridges receive gradients. (The note at lines
  441-444 says bridge grads are exactly zero AT STEP 0 only because lora_B
  is zero-initialized — a step-0 fact, not a frozen parameter.)
* rhombic/nn/rhombi_lora.py:196-212 — effective_bridge in standard mode
  returns self.bridge directly, and scripts/asset1_bank.py:770-776 saves
  effective_bridge into both adapter_state.pt and the bridge_final npys.

Therefore deviation-from-identity of bridge_final IS the right x-variable,
and the bridge_step0 files provide the zero control. The frozen-bridge
fallback ordered by the card is NOT needed; nevertheless the adapter
update magnitude (per-module ||DW||_F = ||scaling * B E A||_F aggregated
by mean over modules) is computed as a SUPPLEMENTARY descriptive
covariate, both for diagnostics and because it gives the task_effect=0
synthetic selftest a non-degenerate x-variable (planted deviation is
identically zero there, which makes the deviation<->gap correlation
undefined — the honest "no correlation" outcome).

Metrics per run
---------------
Deviation : mean and max over modules of ||bridge_final - I||_F, plus the
            same on bridge_step0 (control — identically 0 at identity init).
Update    : mean/max over modules of ||DW||_F (supplementary; Gram-trick
            evaluation via asset1_d3_merge.delta_magnitude).
Gap       : final gap = val_loss - train_loss at the LAST metrics record
            (step 2000 on the real bank), and gap_auc = trapezoidal AUC of
            the gap trajectory over steps, restricted to finite records
            (step 0 has train_loss = null -> NaN gap by design; see
            asset1_analysis_io.load_gap_trajectory).

Correlations
------------
Pearson AND Spearman r with percentile bootstrap CIs (resampling runs with
replacement; numpy default_rng seeded with integer lists — deterministic).
Reported pooled, per family, AND within each (family, task) cell — the
within-task breakdown is the Simpson's-paradox guard, a pre-registered-
compatible DESCRIPTIVE detail (the pre-registered claim is the pooled
bank-level correlation; the stratified views diagnose whether it is driven
by between-task mean differences rather than run-level variation).
Zero-variance inputs (e.g. deviation identically 0) yield r = None with an
explanatory note instead of a fabricated number.

Synthetic selftest (--selftest)
-------------------------------
asset1_synth plants gap = 0.05 + 0.5 * dev_mag + 0.01 * noise with
||bridge_final - I||_F == dev_mag exactly (see its module docstring), so:
task_effect=1 bank -> pooled AND within-task Pearson r > 0.9 must be
recovered; task_effect=0 bank -> deviation is identically 0 (correlation
undefined, note emitted) and the supplementary update-magnitude covariate
shows |r| < 0.6 vs the pure-noise gap. Banks are written to --out-dir
(never the live tree; guarded).

Safety
------
Real-bank reads are gated by require_complete_bank() (the pre-registration
interlock; --allow-partial-bank prints the loud warning and is for tooling
checks only). Writes go only to --out-dir, refused inside the live
campaign tree. CPU-only throughout.

Usage
-----
    python scripts/asset1_daux_gap.py --selftest --out-dir results/asset1-daux-selftest
    python scripts/asset1_daux_gap.py --bank-root results/asset1-bank \
        --out-dir results/asset1-daux
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asset1_analysis_io as aio  # noqa: E402
from asset1_d3_merge import delta_magnitude, guard_out_dir  # noqa: E402

MIN_CELL_N = 3          # below this, report "insufficient n" instead of r
MIN_VALID_BOOT = 10     # below this, CI = [None, None]


# ── Per-run metrics ─────────────────────────────────────────────────


def bridge_deviation_metrics(bridges: dict[str, np.ndarray]) -> dict:
    """mean/max over modules of ||bridge - I||_F (Frobenius)."""
    if not bridges:
        raise ValueError("no bridges given")
    devs = []
    for name in sorted(bridges):
        b = np.asarray(bridges[name], dtype=np.float64)
        if b.ndim != 2 or b.shape[0] != b.shape[1]:
            raise ValueError(f"bridge {name!r} is not square: {b.shape}")
        devs.append(float(np.linalg.norm(b - np.eye(b.shape[0]))))
    arr = np.array(devs)
    return {"dev_mean": float(arr.mean()), "dev_max": float(arr.max()),
            "dev_per_module": arr, "n_modules": int(arr.size)}


def update_magnitude_metrics(adapter: dict) -> dict:
    """mean/max over modules of the true effective ||DW||_F (supplementary
    covariate; bridge + scaling absorbed via asset1_d3_merge.delta_magnitude)."""
    mags = np.array([delta_magnitude(adapter[n]) for n in sorted(adapter)])
    if mags.size == 0:
        raise ValueError("empty adapter")
    return {"update_mag_mean": float(mags.mean()),
            "update_mag_max": float(mags.max())}


def gap_metrics(steps: np.ndarray, train: np.ndarray,
                val: np.ndarray) -> dict:
    """final gap (last record) + trapezoidal AUC over finite gap records.

    gap = val - train elementwise; step 0 is NaN by design (train_loss is
    null before training). gap_auc integrates only finite records (needs
    >= 2, else NaN)."""
    gap = np.asarray(val, dtype=np.float64) - np.asarray(train,
                                                         dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    finite = np.isfinite(gap)
    final_gap = float(gap[-1]) if np.isfinite(gap[-1]) else float("nan")
    if finite.sum() >= 2:
        gap_auc = float(np.trapezoid(gap[finite], steps[finite]))
    else:
        gap_auc = float("nan")
    return {"final_gap": final_gap, "gap_auc": gap_auc,
            "n_finite": int(finite.sum())}


def collect_run_table(bank_root: str | Path,
                      family: str | None = None,
                      include_update_mag: bool = True) -> list[dict]:
    """One flat row per COMPLETE run: identity + deviation + gap metrics.

    Enumerates via the manifest (asset1_analysis_io.iter_runs — half-written
    live run dirs are invisible). Row keys: family_short, task, run_index,
    dev_mean, dev_max, dev_step0_mean, dev_step0_max, final_gap, gap_auc,
    and (when include_update_mag) update_mag_mean, update_mag_max.
    """
    rows: list[dict] = []
    for rec in aio.iter_runs(bank_root, family=family):
        final = bridge_deviation_metrics(
            aio.load_bridges(rec["run_dir"], which="final"))
        step0 = bridge_deviation_metrics(
            aio.load_bridges(rec["run_dir"], which="step0"))
        g = gap_metrics(*aio.load_gap_trajectory(rec["run_dir"]))
        row = {
            "family_short": rec["family_short"],
            "task": rec["task"],
            "run_index": rec["run_index"],
            "dev_mean": final["dev_mean"],
            "dev_max": final["dev_max"],
            "dev_step0_mean": step0["dev_mean"],
            "dev_step0_max": step0["dev_max"],
            "final_gap": g["final_gap"],
            "gap_auc": g["gap_auc"],
        }
        if include_update_mag:
            row.update(update_magnitude_metrics(
                aio.load_adapter(rec["run_dir"])))
        rows.append(row)
    return rows


# ── Correlation machinery ───────────────────────────────────────────


def _corr_cell(x, y, n_boot: int, rng_key: list[int]) -> dict:
    """Pearson + Spearman r with percentile bootstrap CIs for one cell.

    Deterministic: the bootstrap rng is default_rng(rng_key) where rng_key
    is an integer list derived from (seed, stream tag, cell index).
    Zero-variance input or n < MIN_CELL_N produce r = None with a note.
    """
    from scipy import stats

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(x.size)
    cell = {"n": n, "pearson_r": None, "pearson_ci": [None, None],
            "spearman_r": None, "spearman_ci": [None, None], "note": None}
    if n < MIN_CELL_N:
        cell["note"] = f"insufficient n (< {MIN_CELL_N})"
        return cell
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        cell["note"] = ("zero-variance input — correlation undefined "
                        "(see module docstring: for deviation this is the "
                        "honest no-drift outcome, not a failure)")
        return cell

    cell["pearson_r"] = float(stats.pearsonr(x, y).statistic)
    cell["spearman_r"] = float(stats.spearmanr(x, y).statistic)

    rng = np.random.default_rng(rng_key)
    pear, spear = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
            continue
        pear.append(float(np.corrcoef(xs, ys)[0, 1]))
        rx = stats.rankdata(xs)
        ry = stats.rankdata(ys)
        spear.append(float(np.corrcoef(rx, ry)[0, 1]))

    def _ci(vals: list[float]) -> list[float | None]:
        if len(vals) < MIN_VALID_BOOT:
            return [None, None]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return [float(lo), float(hi)]

    cell["pearson_ci"] = _ci(pear)
    cell["spearman_ci"] = _ci(spear)
    cell["n_boot_valid"] = len(pear)
    return cell


def correlation_report(rows: list[dict], x_key: str, y_key: str,
                       n_boot: int = 1000, seed: int = 0) -> dict:
    """Pooled + per-family + within-(family, task) correlation cells.

    Cell enumeration order is deterministic (pooled, then sorted families,
    then sorted family/task pairs), and each cell's bootstrap rng is keyed
    [seed, 404, cell_index] — reproducible regardless of dict order. The
    within-task cells are the Simpson's-paradox guard (descriptive,
    pre-registered-compatible; the pooled cell carries the D-aux claim).
    """
    fams = sorted({r["family_short"] for r in rows})
    cells: list[tuple[str, list[dict]]] = [("pooled", rows)]
    for f in fams:
        cells.append((f"family:{f}",
                      [r for r in rows if r["family_short"] == f]))
    for f in fams:
        tasks = sorted({r["task"] for r in rows if r["family_short"] == f})
        for t in tasks:
            cells.append((f"task:{f}/{t}",
                          [r for r in rows
                           if r["family_short"] == f and r["task"] == t]))
    out = {"x": x_key, "y": y_key, "n_boot": n_boot, "seed": seed,
           "cells": {}}
    for i, (name, subset) in enumerate(cells):
        x = [r[x_key] for r in subset]
        y = [r[y_key] for r in subset]
        out["cells"][name] = _corr_cell(x, y, n_boot, [seed, 404, i])
    return out


# ── Synthetic selftest ──────────────────────────────────────────────


def run_selftest(work_dir: str | Path, seed: int = 0,
                 n_boot: int = 300) -> dict:
    """Recover the planted deviation<->gap correlation from synthetic banks.

    Builds a task_effect=1.0 bank (planted near-perfect correlation) and a
    task_effect=0.0 bank (no plant: deviation identically zero, gap pure
    noise) under work_dir, runs the full pipeline, and checks:
      1. effect bank: pooled Pearson r > 0.9 (dev_mean vs final_gap);
      2. effect bank: every within-task cell r > 0.9 (run-level plant);
      3. effect bank: step-0 control deviation == 0 exactly;
      4. zero bank: deviation cell reports the zero-variance note;
      5. zero bank: |r| < 0.6 for update_mag_mean vs final_gap (noise).
    """
    import asset1_synth as synth

    work = guard_out_dir(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    common = dict(n_families=1, n_tasks=2, n_reps=8, n_layers=1,
                  d_model=12, rank=4, n_channels=2, seed=seed)
    synth.make_synthetic_bank(work / "effect-bank", task_effect=1.0,
                              **common)
    synth.make_synthetic_bank(work / "zero-bank", task_effect=0.0,
                              **common)

    rows_e = collect_run_table(work / "effect-bank")
    rows_z = collect_run_table(work / "zero-bank")

    rep_e = correlation_report(rows_e, "dev_mean", "final_gap",
                               n_boot=n_boot, seed=seed)
    rep_z_dev = correlation_report(rows_z, "dev_mean", "final_gap",
                                   n_boot=n_boot, seed=seed)
    rep_z_mag = correlation_report(rows_z, "update_mag_mean", "final_gap",
                                   n_boot=n_boot, seed=seed)

    task_cells = [c for name, c in rep_e["cells"].items()
                  if name.startswith("task:")]
    zero_pooled = rep_z_dev["cells"]["pooled"]
    mag_pooled = rep_z_mag["cells"]["pooled"]
    checks = {
        "effect_pooled_r_gt_0.9":
            (rep_e["cells"]["pooled"]["pearson_r"] or 0.0) > 0.9,
        "effect_within_task_r_gt_0.9":
            bool(task_cells) and all((c["pearson_r"] or 0.0) > 0.9
                                     for c in task_cells),
        "effect_step0_control_zero":
            max(r["dev_step0_max"] for r in rows_e) == 0.0,
        "zero_bank_deviation_undefined":
            zero_pooled["pearson_r"] is None
            and "zero-variance" in (zero_pooled["note"] or ""),
        "zero_bank_update_mag_r_near_0":
            mag_pooled["pearson_r"] is not None
            and abs(mag_pooled["pearson_r"]) < 0.6,
    }
    return {
        "effect_dev_vs_final_gap": rep_e,
        "zero_dev_vs_final_gap": rep_z_dev,
        "zero_update_mag_vs_final_gap": rep_z_mag,
        "n_runs": {"effect": len(rows_e), "zero": len(rows_z)},
        "checks": checks,
        "passed": all(checks.values()),
    }


# ── Output helpers ──────────────────────────────────────────────────


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ── CLI ─────────────────────────────────────────────────────────────

# (x_key, y_key) pairs. The FIRST is the pre-registered D-aux claim
# (pilot r = 0.888 was deviation vs final gap); the rest are descriptive.
PRIMARY_PAIR = ("dev_mean", "final_gap")
DESCRIPTIVE_PAIRS = (("dev_max", "final_gap"),
                     ("dev_mean", "gap_auc"),
                     ("update_mag_mean", "final_gap"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Asset 1 D-aux — bridge deviation <-> generalization "
                    "gap (overfit detection). Real-bank invocations are "
                    "gated on the FULL 480-run bank (pre-registration "
                    "interlock).")
    parser.add_argument("--selftest", action="store_true",
                        help="build synthetic banks and verify recovery of "
                             "the planted correlation (no real-bank access)")
    parser.add_argument("--bank-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="bootstrap / fixture seed (default 0)")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--allow-partial-bank", action="store_true",
                        help="EXPLORATORY ONLY: bypass the completeness "
                             "interlock with a loud pre-registration "
                             "warning; results must never be reported")
    args = parser.parse_args(argv)

    if args.selftest:
        if args.out_dir is None:
            parser.error("--out-dir is required with --selftest (synthetic "
                         "banks are written there)")
        out = guard_out_dir(args.out_dir)
        rep = run_selftest(out, seed=args.seed, n_boot=args.n_boot)
        out.mkdir(parents=True, exist_ok=True)
        (out / "daux_selftest.json").write_text(
            json.dumps(_jsonable(rep), indent=2), encoding="utf-8")
        print(json.dumps(_jsonable(rep["checks"]), indent=2))
        if not rep["passed"]:
            raise SystemExit("[asset1-daux] selftest FAILED — see "
                             f"{out / 'daux_selftest.json'}")
        print("[asset1-daux] selftest PASSED")
        return

    if args.bank_root is None or args.out_dir is None:
        parser.error("--bank-root and --out-dir are required unless "
                     "--selftest")
    out_dir = guard_out_dir(args.out_dir)
    # THE INTERLOCK — refuse a partial bank (pre-registration hygiene).
    aio.require_complete_bank(args.bank_root,
                              allow_partial=args.allow_partial_bank)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_run_table(args.bank_root)
    if not rows:
        raise SystemExit("[asset1-daux] no COMPLETE runs in the manifest")
    _write_csv(rows, out_dir / "daux_run_table.csv")

    primary = correlation_report(rows, *PRIMARY_PAIR,
                                 n_boot=args.n_boot, seed=args.seed)
    descriptive = {
        f"{x}__vs__{y}": correlation_report(rows, x, y, n_boot=args.n_boot,
                                            seed=args.seed)
        for x, y in DESCRIPTIVE_PAIRS
    }
    report = {
        "n_runs": len(rows),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "primary": primary,
        "descriptive": descriptive,
        "step0_control": {
            "max_dev_step0_max": max(r["dev_step0_max"] for r in rows),
            "expectation": "0.0 exactly (identity init; see docstring)",
        },
    }
    (out_dir / "daux_report.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8")

    pooled = primary["cells"]["pooled"]
    print(f"[asset1-daux] n={len(rows)} runs; primary "
          f"{PRIMARY_PAIR[0]} vs {PRIMARY_PAIR[1]}: "
          f"pearson r={pooled['pearson_r']} ci={pooled['pearson_ci']}, "
          f"spearman r={pooled['spearman_r']}")
    print(f"[asset1-daux] report written to {out_dir / 'daux_report.json'}")


if __name__ == "__main__":
    main()
