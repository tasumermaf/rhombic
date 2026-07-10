"""E-5 coherence-interpolation bifurcation sweep — runner + endpoint tallies.

Drives the frozen pre-registration docs/E5_BIFURCATION_PREREG_2026-07-10.md:
the pair-correctness axis f ∈ {0.0, 0.25, 0.5, 0.75, 1.0} × seed ∈ {42, 43, 44},
15 sequential runs of scripts/train_cybernetic.py, each corrupting the n=8
tesseract contrastive spec by f (the trainer's --pair-correctness edit) and
landing in a coherence basin. This module is a bounded-cycle, kill-safe
sequential chain (context-engineering §4): the manifest is atomically rewritten
after every state change, so a crash resumes skipping COMPLETE runs and retries
FAILED/RUNNING ones.

Subcommands:
  run        — iterate the frozen grid; invoke the trainer per run; after each
               COMPLETE run compute its endpoint record so partial sweeps stay
               analyzable.
  endpoints  — (re)compute endpoints.json for every COMPLETE run: primary
               co/cross_trained (post-corruption spec), secondary co/cross_true
               (native spec), fiedler_mean, mean coupling magnitude, val_loss.
  summarize  — write summary.md: the 15-run table + band classification (§4) +
               the P1/P2 tallies. This is a TALLY, NOT the confirmatory analysis.

Stdlib + numpy only (train_exp2_scale, imported for the co/cross helpers, pulls
in torch transitively — the endpoint math mirrors its indexing convention).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# The endpoint math MUST mirror the trainer's own logged ratio exactly — reuse
# the very helpers the trainer's oscilloscope uses (train_exp2_scale).
from train_exp2_scale import (  # noqa: E402
    coplanar_crossplanar_ratio,
    _coplanar_crossplanar_indices,
)

PREREG_DOC = "docs/E5_BIFURCATION_PREREG_2026-07-10.md"
DEFAULT_OUTDIR = "results/E-5-bifurcation"
TRAINER_REL = "scripts/train_cybernetic.py"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Frozen grid (prereg §3). f as literal grid strings (F formatting); seeds are
# BOTH the corruption draw seed and the training seed.
F_GRID = ["0.00", "0.25", "0.50", "0.75", "1.00"]
SEED_GRID = [42, 43, 44]

# Pre-registered classification bands on endpoint co/cross_trained (prereg §4):
#   Structured    >= 10^3
#   Intermediate  [10^1, 10^3)
#   Unstructured  < 10^1
BAND_STRUCTURED = 1e3
BAND_INTERMEDIATE = 1e1


# ── small utilities ──────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON via temp + os.replace so the file on disk is never partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _iter_grid():
    """Deterministic run order: f ascending, seed ascending."""
    for f_str in F_GRID:
        for seed in SEED_GRID:
            yield f_str, seed


def run_id_for(f_str: str, seed: int) -> str:
    return f"f{f_str}_s{seed}"


# ── manifest ─────────────────────────────────────────────────────────


def _init_manifest(steps: int) -> dict:
    runs = []
    for f_str, seed in _iter_grid():
        runs.append({
            "run_id": run_id_for(f_str, seed),
            "f": float(f_str),
            "seed": seed,
            "status": "PENDING",
            "started_at": None,
            "finished_at": None,
            "return_code": None,
        })
    return {
        "prereg": PREREG_DOC,
        "steps": steps,
        "grid": {"f": list(F_GRID), "seed": list(SEED_GRID)},
        "runs": runs,
    }


def _load_or_init_manifest(path: Path, steps: int) -> dict:
    path = Path(path)
    if path.exists():
        man = json.loads(path.read_text(encoding="utf-8"))
        # A resume must not silently mix run depths inside one sweep
        # (completed runs at one --steps, remaining runs at another).
        if man.get("steps") != steps:
            raise SystemExit(
                f"sweep manifest was created with steps={man.get('steps')} "
                f"but --steps {steps} was passed; refusing a mixed-depth "
                "sweep — rerun with the original value or a fresh outdir")
        return man
    return _init_manifest(steps)


def _entry(manifest: dict, run_id: str) -> dict:
    for e in manifest["runs"]:
        if e["run_id"] == run_id:
            return e
    raise KeyError(run_id)


# ── run subcommand ───────────────────────────────────────────────────


def build_run_command(python: str, repo_root: Path, steps: int,
                      f_str: str, seed: int, run_dir: Path) -> list[str]:
    """Pinned trainer invocation for one grid cell (prereg §3/§6)."""
    return [
        str(python), TRAINER_REL,
        "--model", MODEL,
        "--max-steps", str(steps),
        "--feedback-interval", "100",
        "--lr", "2e-4",
        "--batch-size", "2",
        "--gradient-accumulation", "8",
        "--warmup-steps", "200",
        "--checkpoint-steps", "100",
        "--bridge-mode", "identity",
        "--n-channels", "8",
        "--steersman-preset", "default",
        "--seed", str(seed),
        "--pair-correctness", f_str,
        "--output", str(run_dir),
    ]


def _invoke_training(cmd: list[str], log_path: Path, cwd: Path) -> int:
    """Run one trainer subprocess; stdout+stderr appended to log_path.

    Factored out as the single execution seam so tests can monkeypatch it with a
    stub that writes a fake run dir. Returns the process return code.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log,
                              stderr=subprocess.STDOUT)
    return proc.returncode


def run_sweep(outdir=DEFAULT_OUTDIR, python=None, steps=3000,
              dry_run=False, repo_root=".") -> dict:
    """Iterate the frozen grid, resuming from the manifest.

    Skips COMPLETE runs; retries FAILED and RUNNING (a RUNNING entry at startup
    is a prior crash). Manifest atomically rewritten after every state change.
    After each COMPLETE run its endpoint record is recomputed immediately.
    """
    python = python or sys.executable
    outdir = Path(outdir).resolve()
    repo_root = Path(repo_root).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "sweep_manifest.json"
    manifest = _load_or_init_manifest(manifest_path, steps)

    for f_str, seed in _iter_grid():
        rid = run_id_for(f_str, seed)
        entry = _entry(manifest, rid)
        if entry["status"] == "COMPLETE":
            print(f"[e5] skip {rid}: COMPLETE")
            continue

        run_dir = outdir / rid
        log_path = outdir / f"{rid}.log"
        cmd = build_run_command(python, repo_root, steps, f_str, seed, run_dir)

        if dry_run:
            print(f"[e5] DRY RUN {rid}: {' '.join(cmd)}")
            continue

        entry["status"] = "RUNNING"
        entry["started_at"] = _now()
        entry["finished_at"] = None
        entry["return_code"] = None
        _atomic_write_json(manifest_path, manifest)

        rc = _invoke_training(cmd, log_path, repo_root)

        entry["return_code"] = rc
        entry["finished_at"] = _now()
        entry["status"] = "COMPLETE" if rc == 0 else "FAILED"
        _atomic_write_json(manifest_path, manifest)
        print(f"[e5] {rid}: {entry['status']} (rc={rc})")

        if entry["status"] == "COMPLETE":
            try:
                _recompute_endpoints(outdir, manifest)
            except Exception as exc:  # keep the chain alive; endpoints retryable
                print(f"[e5] endpoint compute for {rid} deferred: {exc}")

    return manifest


# ── endpoint computation ─────────────────────────────────────────────


def _mean_finite(values) -> float:
    """Mean over finite values only (mirrors the trainer's ratio aggregation)."""
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _ratio_over_pairs(B: np.ndarray, co_pairs, cross_pairs) -> float:
    """Per-adapter mean|co| / mean|cross|, mirroring coplanar_crossplanar_ratio.

    Reads the SAME upper-triangle entries abs(B[i, j]) (i < j) and applies the
    same inf convention (ratio is +inf when the cross mean is <= 1e-12).
    """
    co_vals = [abs(B[i, j]) for i, j in co_pairs]
    cross_vals = [abs(B[i, j]) for i, j in cross_pairs]
    mean_co = float(np.mean(co_vals)) if co_vals else 0.0
    mean_cross = float(np.mean(cross_vals)) if cross_vals else 0.0
    return mean_co / mean_cross if mean_cross > 1e-12 else float("inf")


def _post_corruption_pairs(config: dict, n: int):
    """The spec the run actually optimized (post-corruption co/cross pairs).

    Read from the trainer-recorded co_pairs_trained / cross_pairs_trained; if
    absent (an uncorrupted run that predates the record) fall back to the true
    n-channel partition — identical to the true spec, which is correct.
    """
    co = config.get("co_pairs_trained")
    cross = config.get("cross_pairs_trained")
    if co is not None and cross is not None:
        return ([tuple(int(x) for x in p) for p in co],
                [tuple(int(x) for x in p) for p in cross])
    co_t, cross_t = _coplanar_crossplanar_indices(n)
    return list(co_t or []), list(cross_t or [])


def _final_checkpoint(run_dir: Path) -> dict:
    results_path = Path(run_dir) / "results.json"
    if not results_path.exists():
        return {}
    results = json.loads(results_path.read_text(encoding="utf-8"))
    checkpoints = results.get("checkpoints", [])
    return checkpoints[-1] if checkpoints else {}


def _assert_close_rel(a: float, b: float, rtol: float, rid: str) -> None:
    """f=1 positive control: co/cross_trained ≡ co/cross_true by construction."""
    a_nan = a is None or (isinstance(a, float) and np.isnan(a))
    b_nan = b is None or (isinstance(b, float) and np.isnan(b))
    if a_nan and b_nan:
        return
    denom = abs(b) if (b is not None and abs(b) > 0) else 1.0
    if a_nan or b_nan or abs(a - b) > rtol * denom:
        raise AssertionError(
            f"[e5] {rid}: f=1 positive control failed — co_cross_trained {a!r} "
            f"!= co_cross_true {b!r} within {rtol} relative")


def compute_run_endpoint(run_dir, run_id, f, seed) -> dict:
    """Endpoint record for one COMPLETE run (prereg §4)."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    bridges = [np.load(p) for p in sorted(run_dir.glob("bridge_final_*.npy"))]
    if not bridges:
        raise FileNotFoundError(f"no bridge_final_*.npy in {run_dir}")
    n = bridges[0].shape[0]

    co_pairs, cross_pairs = _post_corruption_pairs(config, n)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Primary: co/cross on the corrupted spec the run optimized.
    co_cross_trained = _mean_finite(
        [_ratio_over_pairs(B, co_pairs, cross_pairs) for B in bridges])

    # Secondary: co/cross on the native n=8 spec (the trainer's logged ratio).
    true_ratios = []
    for B in bridges:
        r = coplanar_crossplanar_ratio(B)
        if r is not None:
            true_ratios.append(r["ratio"])
    co_cross_true = _mean_finite(true_ratios)

    # Mean coupling magnitude: pooled mean|B[i,j]| over all pairs, per adapter.
    magnitudes = [float(np.mean([abs(B[i, j]) for i, j in all_pairs]))
                  for B in bridges]
    mean_magnitude = float(np.mean(magnitudes)) if magnitudes else float("nan")

    final_cp = _final_checkpoint(run_dir)
    pair_correctness = config.get("pair_correctness", f)

    if float(pair_correctness) >= 1.0:
        _assert_close_rel(co_cross_trained, co_cross_true, 1e-9, run_id)

    return {
        "run_id": run_id,
        "f": float(f),
        "seed": int(seed),
        "pair_correctness": float(pair_correctness),
        "realized_agreement": config.get("realized_agreement"),
        "co_cross_trained": co_cross_trained,
        "co_cross_true": co_cross_true,
        "fiedler_mean": final_cp.get("fiedler_mean"),
        "final_co_cross_ratio": final_cp.get("co_cross_ratio"),
        "val_loss": final_cp.get("val_loss"),
        "mean_magnitude": mean_magnitude,
        "n_adapters": len(bridges),
        "band": classify_band(co_cross_trained),
    }


def _recompute_endpoints(outdir, manifest=None) -> list[dict]:
    outdir = Path(outdir).resolve()
    if manifest is None:
        manifest = json.loads(
            (outdir / "sweep_manifest.json").read_text(encoding="utf-8"))
    records = []
    errors = []
    for entry in manifest["runs"]:
        if entry["status"] != "COMPLETE":
            continue
        # One poisoned run dir must not block endpoints for every other
        # COMPLETE run: record the error and keep going.
        try:
            rec = compute_run_endpoint(
                outdir / entry["run_id"], entry["run_id"],
                entry["f"], entry["seed"])
            records.append(rec)
        except Exception as e:  # noqa: BLE001 — recorded, never swallowed
            errors.append({"run_id": entry["run_id"], "error": str(e)})
            print(f"[e5] endpoint ERROR {entry['run_id']}: {e}", flush=True)
    _atomic_write_json(outdir / "endpoints.json",
                       {"prereg": PREREG_DOC, "runs": records,
                        "errors": errors})
    return records


# ── band classification (prereg §4) ──────────────────────────────────


def classify_band(co_cross_trained) -> str:
    v = co_cross_trained
    if v is None or (isinstance(v, float) and np.isnan(v)):
        # NaN is a diverged/defective endpoint, not a basin. The prereg
        # bands are defined on the ratio only for finite/inf values;
        # NonFinite runs are tallied separately and never enter P1/P2.
        return "NonFinite"
    if isinstance(v, float) and np.isinf(v):
        # All per-adapter cross means <= 1e-12: inf >= 10^3 is band-
        # consistent, but a COLLAPSED bridge (co ~ 0 too) also lands here —
        # summarize flags Structured rows with vanishing magnitude for
        # manual review rather than silently reclassifying.
        return "Structured"
    if v >= BAND_STRUCTURED:
        return "Structured"
    if v >= BAND_INTERMEDIATE:
        return "Intermediate"
    return "Unstructured"


# ── summarize subcommand ─────────────────────────────────────────────


def _fmt(x, spec=".4g") -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    try:
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def build_summary(records: list[dict]) -> str:
    """Assemble summary.md text from the endpoint records."""
    rows = sorted(records, key=lambda r: (r["f"], r["seed"]))
    lines = [
        "# E-5 Bifurcation Sweep — Endpoint Tally",
        "",
        f"Pre-registration: `{PREREG_DOC}`",
        "",
        "**This is a tally, NOT the confirmatory analysis.** It reports the "
        "endpoint co/cross ratios and the pre-registered band classification "
        "(§4) so a partial or complete sweep can be inspected. The frozen "
        "predictions (P1 bimodality, P2 controls, the graded falsifier) are "
        "evaluated in the confirmatory analysis, not here.",
        "",
        f"Runs summarized: {len(rows)} / 15.",
        "",
        "| f | seed | co/cross_trained | co/cross_true | fiedler_mean | "
        "magnitude | band |",
        "|---|------|------------------|---------------|--------------|"
        "-----------|------|",
    ]
    flagged = []
    for r in rows:
        band = r["band"]
        mag = r.get("mean_magnitude")
        # A Structured band with vanishing coupling magnitude is the
        # collapsed-bridge-inf signature — flag it for manual review.
        if (band == "Structured" and isinstance(mag, (int, float))
                and np.isfinite(mag) and mag < 1e-6):
            band = "Structured ⚠mag"
            flagged.append(r["run_id"])
        lines.append(
            f"| {r['f']:.2f} | {r['seed']} | "
            f"{_fmt(r['co_cross_trained'])} | {_fmt(r['co_cross_true'])} | "
            f"{_fmt(r.get('fiedler_mean'))} | {_fmt(r.get('mean_magnitude'))} | "
            f"{band} |")

    nonfinite = [r for r in rows if r["band"] == "NonFinite"]
    intermediate = [r for r in rows if r["band"] == "Intermediate"]
    f1 = [r for r in rows if r["f"] == 1.0]
    f0 = [r for r in rows if r["f"] == 0.0]
    f1_all_structured = bool(f1) and all(r["band"] == "Structured" for r in f1)
    f0_all_unstructured = bool(f0) and all(
        r["band"] == "Unstructured" for r in f0)

    lines += [
        "",
        "## Tally (descriptive only)",
        "",
        f"- Runs in the Intermediate band [10¹, 10³): "
        f"**{len(intermediate)}** / {len(rows)} "
        f"(P1 predicts ≤ 2 across the full 15; falsifier ≥ 5).",
        f"- f = 1.0 runs all Structured (P2 positive control): "
        f"**{'YES' if f1_all_structured else 'NO'}** "
        f"({len(f1)}/3 present).",
        f"- f = 0.0 runs all Unstructured (P2 negative control): "
        f"**{'YES' if f0_all_unstructured else 'NO'}** "
        f"({len(f0)}/3 present).",
        "",
        "Bands (§4): Structured ≥ 10³; Intermediate [10¹, 10³); "
        "Unstructured < 10¹. NonFinite (NaN endpoint) runs are defective, "
        "tallied separately, and never enter P1/P2.",
        "",
    ]
    if nonfinite:
        lines += [f"- **NonFinite runs: {len(nonfinite)}** "
                  f"({', '.join(r['run_id'] for r in nonfinite)}) — "
                  "diverged/defective; excluded from every tally above.", ""]
    if flagged:
        lines += [f"- **⚠ magnitude flags: {len(flagged)}** "
                  f"({', '.join(flagged)}) — Structured band via inf ratio "
                  "with vanishing coupling magnitude; review before any "
                  "basin claim.", ""]
    return "\n".join(lines)


def summarize(outdir) -> Path:
    outdir = Path(outdir).resolve()
    endpoints_path = outdir / "endpoints.json"
    records = []
    if endpoints_path.exists():
        records = json.loads(endpoints_path.read_text(encoding="utf-8")).get(
            "runs", [])
    md = build_summary(records)
    out = outdir / "summary.md"
    out.write_text(md, encoding="utf-8")
    print(f"[e5] wrote {out} ({len(records)} runs)")
    return out


# ── CLI ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E-5 coherence-interpolation bifurcation sweep "
                    f"({PREREG_DOC}). Sequential, kill-safe, resumable.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the frozen 15-cell grid (resumable)")
    r.add_argument("--outdir", default=DEFAULT_OUTDIR)
    r.add_argument("--python", default=None,
                   help="python interpreter for trainer subprocesses "
                        "(default: this interpreter)")
    r.add_argument("--steps", type=int, default=3000)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--repo-root", default=".")

    e = sub.add_parser("endpoints",
                       help="(re)compute endpoints.json for COMPLETE runs")
    e.add_argument("--outdir", default=DEFAULT_OUTDIR)

    s = sub.add_parser("summarize", help="write summary.md (tally, not "
                                         "confirmatory)")
    s.add_argument("--outdir", default=DEFAULT_OUTDIR)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        run_sweep(args.outdir, python=args.python, steps=args.steps,
                  dry_run=args.dry_run, repo_root=args.repo_root)
    elif args.cmd == "endpoints":
        _recompute_endpoints(args.outdir)
    elif args.cmd == "summarize":
        summarize(args.outdir)


if __name__ == "__main__":
    main()
