"""Asset 1 — SYNTHETIC bank generator (validation fixtures for the analysis
pipeline; pre-registration hygiene).

Why this exists
---------------
The pre-registration forbids running any classifier, correlation, or summary
statistic over the REAL bank before all 480 runs are COMPLETE. Every D1/D2/
D3/D-aux tool is therefore developed and validated against miniature
synthetic banks produced HERE, which replicate the real on-disk schema
exactly (manifest field names via asset1_bank.RunSpec itself, adapter-state
key naming, config.json / metrics.json shapes, COMPLETE markers, bridge
npys) while planting KNOWN structure the tools must recover.

Generative model (verifiers rely on this being exact)
-----------------------------------------------------
Let f = family index, t = task index, r = global run index, m = module.
All randomness comes from ``numpy.random.default_rng`` seeded with integer
lists derived from (seed, fixed stream tag, indices) — fully deterministic,
never date- or entropy-based.

Family geometry (cross-family dimension mismatch is deliberate):
    d_model_f = d_model - 4*f          (e.g. 16, 12, ...)
    n_layers_f = n_layers + f          (e.g. 2, 3, ...)
    modules: model_layers_{L}_self_attn_{p}_proj, p in {q,k,v,o},
    d_out = d_model_f for q/o, d_model_f // 2 for k/v (heterogeneous
    output dims, mirroring GQA); d_in = d_model_f everywhere;
    lora_A: (rank, d_in), lora_B: (d_out, rank).

Fixed task directions (identical across runs; the separable signal):
    For every (f, t, m): D_A = outer(a, w), D_B = outer(u, b) with the
    factor vectors drawn N(0,1) from rng([seed, 101, f, t, L, p]) and each
    D_* normalized to unit Frobenius norm.
    For every (f, t): Delta = C x C matrix from rng([seed, 202, f, t]),
    normalized to unit Frobenius norm (the bridge's task deviation).

Per run r (rng_run = default_rng([seed, 303, r])):
    dev_mag  = task_effect * (0.5 + U)           with U ~ Uniform[0, 1)
    lora_A   = NOISE_SCALE * N(0,1)^{rank x d_in} + task_effect * D_A
    lora_B   = NOISE_SCALE * N(0,1)^{d_out x rank} + task_effect * D_B
    bridge_final(m) = I_C + dev_mag * Delta(f, t)     (all modules)
    bridge_step0(m) = I_C exactly (bridge_mode='identity' at init)
    gap      = GAP_BASE + GAP_COEF * dev_mag + GAP_NOISE * N(0,1)

Planted properties, all switched by ``task_effect``:
    * task_effect > 0: class means of the flattened features differ by
      task_effect * (unit-Frobenius directions per tensor) against iid
      noise of scale NOISE_SCALE — a linear classifier CAN separate tasks.
      task_effect = 0: features are pure iid noise — it CANNOT.
    * ||bridge_final - I||_F == dev_mag exactly for every module, and the
      final val - train gap in metrics.json equals GAP_BASE + GAP_COEF *
      dev_mag + GAP_NOISE * noise, so D-aux has a near-perfect planted
      deviation<->gap correlation when task_effect > 0 and none when 0
      (deviation is identically zero).

metrics.json: records at steps (0, 100, 200); step 0 has train_loss = None
(real schema); train decays 1.5 -> 1.25 -> 1.0; val = train + gap for
steps > 0 (val at step 0 = 1.6 + gap). "final" mirrors the last record;
wall_time_seconds is a fixed placeholder. Timestamps everywhere are fixed
literal placeholders — the fixture is byte-deterministic in its arguments.

Safety
------
This generator refuses to write into (a) any path containing an
'asset1-bank' component — the LIVE campaign tree — and (b) any existing
non-empty directory that does not carry the SYNTHETIC_BANK.json marker
from a previous invocation. It therefore cannot point at the real bank,
which is why its CLI carries no require_complete_bank() gate (the gate
protects reads of the real bank; this tool cannot touch it at all).

Usage
-----
    python scripts/asset1_synth.py --out-dir results/asset1-synth-fixture
    (all knobs have documented defaults; --seed defaults to 0)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from asset1_bank import RunSpec  # noqa: E402  (manifest field fidelity)

# ── Generative-model constants (documented in the module docstring) ──

NOISE_SCALE = 0.05      # iid parameter noise (std) per run
DEV_BASE = 0.5          # dev_mag = task_effect * (DEV_BASE + Uniform[0,1))
GAP_BASE = 0.05         # baseline train/val gap
GAP_COEF = 0.5          # gap slope against dev_mag (the D-aux plant)
GAP_NOISE = 0.01        # gap observation noise (std)
METRIC_STEPS = (0, 100, 200)
TRAIN_CURVE = {100: 1.25, 200: 1.0}   # train loss at each step > 0
VAL_STEP0 = 1.6         # val loss baseline at step 0 (before training)

SEED_BASE = 10_000      # mirrors asset1_bank (seed = 10000 + run_index)
DATA_SEED_BASE = 20_000
LORA_ALPHA = 16.0       # scaling = LORA_ALPHA / rank, as in the real bank

MARKER_NAME = "SYNTHETIC_BANK.json"
PROJECTIONS = ("q", "k", "v", "o")

# Fixed literal placeholders — byte-deterministic fixtures (no wall clock).
_PLACEHOLDER_START = "2026-01-01T00:00:00+00:00"
_PLACEHOLDER_END = "2026-01-01T01:00:00+00:00"


# ── Safety guard ────────────────────────────────────────────────────


def _guard_out_dir(out_dir: Path) -> None:
    """Refuse the live campaign tree and refuse clobbering foreign dirs."""
    resolved = out_dir.resolve()
    if "asset1-bank" in resolved.parts:
        raise ValueError(
            f"REFUSING to write a synthetic bank into {resolved}: the path "
            f"contains an 'asset1-bank' component — that is the LIVE "
            f"campaign tree and is write-protected.")
    if resolved.exists():
        has_contents = any(resolved.iterdir())
        if has_contents and not (resolved / MARKER_NAME).exists():
            raise ValueError(
                f"REFUSING to write into existing non-empty directory "
                f"{resolved}: it does not carry {MARKER_NAME}, so it is not "
                f"a previous synthetic bank. Choose an empty/new directory.")


# ── Fixed structure builders ────────────────────────────────────────


def _family_geometry(f: int, d_model: int, n_layers: int) -> dict:
    dm = d_model - 4 * f
    if dm < 4:
        raise ValueError(
            f"d_model={d_model} too small for family index {f} "
            f"(d_model_f = {dm} < 4); raise d_model or lower n_families")
    return {"d_model": dm, "n_layers": n_layers + f}


def _module_names(geom: dict) -> list[str]:
    return [f"model_layers_{L}_self_attn_{p}_proj"
            for L in range(geom["n_layers"]) for p in PROJECTIONS]


def _module_dims(name: str, geom: dict) -> tuple[int, int]:
    """(d_out, d_in) for a module. k/v get d_model // 2 (GQA-style
    heterogeneous output dims); q/o get d_model."""
    dm = geom["d_model"]
    proj = name.rsplit("_", 2)[1]          # ..._{p}_proj -> p
    d_out = dm // 2 if proj in ("k", "v") else dm
    return d_out, dm


def _unit_frobenius(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def _task_directions(seed: int, f: int, t: int, layer: int, proj_idx: int,
                     rank: int, d_out: int, d_in: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Fixed unit-Frobenius rank-1 task directions (D_A, D_B) for one
    (family, task, module) — identical across all runs of the cell."""
    rng = np.random.default_rng([seed, 101, f, t, layer, proj_idx])
    a = rng.standard_normal(rank)
    w = rng.standard_normal(d_in)
    u = rng.standard_normal(d_out)
    b = rng.standard_normal(rank)
    return _unit_frobenius(np.outer(a, w)), _unit_frobenius(np.outer(u, b))


def _bridge_delta(seed: int, f: int, t: int, n_channels: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 202, f, t])
    return _unit_frobenius(rng.standard_normal((n_channels, n_channels)))


# ── Per-run writers ─────────────────────────────────────────────────


def _write_run(run: RunSpec, f: int, t: int, geom: dict, rank: int,
               n_channels: int, task_effect: float, seed: int) -> None:
    rng_run = np.random.default_rng([seed, 303, run.run_index])
    run.run_dir.mkdir(parents=True, exist_ok=True)

    dev_mag = task_effect * (DEV_BASE + rng_run.uniform())
    delta = _bridge_delta(seed, f, t, n_channels)
    eye = np.eye(n_channels, dtype=np.float32)
    bridge_final = (eye + dev_mag * delta).astype(np.float32)

    adapter_state: dict[str, torch.Tensor] = {}
    names = _module_names(geom)
    for L in range(geom["n_layers"]):
        for p_idx, proj in enumerate(PROJECTIONS):
            safe = f"model_layers_{L}_self_attn_{proj}_proj"
            d_out, d_in = _module_dims(safe, geom)
            D_A, D_B = _task_directions(seed, f, t, L, p_idx,
                                        rank, d_out, d_in)
            lora_A = (NOISE_SCALE * rng_run.standard_normal((rank, d_in))
                      + task_effect * D_A).astype(np.float32)
            lora_B = (NOISE_SCALE * rng_run.standard_normal((d_out, rank))
                      + task_effect * D_B).astype(np.float32)
            adapter_state[f"{safe}.lora_A"] = torch.from_numpy(lora_A)
            adapter_state[f"{safe}.lora_B"] = torch.from_numpy(lora_B)
            adapter_state[f"{safe}.bridge"] = torch.from_numpy(
                bridge_final.copy())
            adapter_state[f"{safe}.scaling"] = torch.tensor(
                LORA_ALPHA / rank)
            adapter_state[f"{safe}.n_channels"] = torch.tensor(n_channels)
            adapter_state[f"{safe}.rank"] = torch.tensor(rank)
            np.save(run.run_dir / f"bridge_final_{safe}.npy", bridge_final)
            np.save(run.run_dir / f"bridge_step0_{safe}.npy", eye)
    torch.save(adapter_state, run.run_dir / "adapter_state.pt")

    # ── metrics.json (planted gap; real record schema) ──
    gap = GAP_BASE + GAP_COEF * dev_mag + GAP_NOISE * rng_run.standard_normal()
    records = []
    for step in METRIC_STEPS:
        if step == 0:
            records.append({"step": 0, "train_loss": None,
                            "val_loss": VAL_STEP0 + gap, "lr": 0.0,
                            "wall_time_s": 0.0})
        else:
            tr = TRAIN_CURVE[step]
            records.append({"step": step, "train_loss": tr,
                            "val_loss": tr + gap, "lr": 2e-4,
                            "wall_time_s": float(step) / 10.0})
    final = {"step": records[-1]["step"],
             "train_loss": records[-1]["train_loss"],
             "val_loss": records[-1]["val_loss"]}
    (run.run_dir / "metrics.json").write_text(
        json.dumps({"records": records, "final": final,
                    "wall_time_seconds": 42.0}, indent=2),
        encoding="utf-8")

    # ── config.json (every field the IO layer reads, real names) ──
    config = {
        "asset": "asset1-synth",
        "campaign_tag": "synthetic",
        "run_index": run.run_index,
        "replicate": run.replicate,
        "family": run.family,
        "family_short": run.family_short,
        "task": run.task,
        "task_name": f"synthetic task {run.task}",
        "model": run.family,
        "dataset": {
            "source": "synthetic",
            "config_name": None,
            "hf_split": "train",
            "split_design": "synthetic fixture — no real data",
            "val_seed": 777,
            "val_size": 50,
            "pool_cap": 1000,
            "n_total_raw": 1000,
            "n_val": 50,
            "n_pool": 950,
            "val_ids_sha256": hashlib.sha256(
                f"synthetic-val:{run.task}".encode()).hexdigest(),
        },
        "seed": run.seed,
        "data_seed": run.data_seed,
        "steps": METRIC_STEPS[-1],
        "rank": rank,
        "n_channels": n_channels,
        "lora_alpha": LORA_ALPHA,
        "bridge_mode": "identity",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "n_injected_modules": len(names),
        "batch_size": 2,
        "gradient_accumulation": 8,
        "git_commit": "synthetic",
        "library_versions": {"numpy": np.__version__,
                             "torch": torch.__version__},
        "started_at": _PLACEHOLDER_START,
        "finished_at": _PLACEHOLDER_END,
        "wall_time_seconds": 42.0,
        "synthetic_generative_model": {
            "task_effect": task_effect,
            "dev_mag": dev_mag,
            "gap": gap,
            "noise_scale": NOISE_SCALE,
        },
    }
    (run.run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    # COMPLETE marker written LAST (real protocol)
    (run.run_dir / "COMPLETE").write_text("synthetic\n", encoding="utf-8")


# ── Public entry point ──────────────────────────────────────────────


def make_synthetic_bank(out_dir: str | Path,
                        n_families: int = 2,
                        n_tasks: int = 3,
                        n_reps: int = 4,
                        n_layers: int = 2,
                        d_model: int = 16,
                        rank: int = 4,
                        n_channels: int = 2,
                        task_effect: float = 1.0,
                        seed: int = 0) -> dict:
    """Write a miniature synthetic Asset-1 bank with the real on-disk schema.

    NOTE on defaults: rank must be divisible by n_channels (the RhombiLoRA
    bridge block-expansion constraint enforced by
    asset1_canonicalize.effective_factors), so the default is rank=4 with
    n_channels=2. Layout, manifest fields (via asset1_bank.RunSpec), file
    names, and key naming all match the real bank byte-for-byte in
    structure; see the module docstring for the planted generative model.

    Returns {"out_dir": Path, "manifest": dict, "n_runs": int,
             "families": [...], "tasks": [...]}.
    """
    if rank % n_channels != 0:
        raise ValueError(
            f"rank ({rank}) must be divisible by n_channels ({n_channels}) "
            f"— RhombiLoRA bridge block-expansion constraint")
    if min(n_families, n_tasks, n_reps, n_layers) < 1:
        raise ValueError("n_families, n_tasks, n_reps, n_layers must be >= 1")

    out_dir = Path(out_dir)
    _guard_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families = [{"model": f"synthetic/family-{f}", "short": f"synthfam{f}"}
                for f in range(n_families)]
    tasks = [f"task{t:02d}" for t in range(n_tasks)]
    geoms = [_family_geometry(f, d_model, n_layers)
             for f in range(n_families)]

    # Round-robin across (family, task) cells, replicate-major — the real
    # generate_manifest ordering, with RunSpec supplying the field names.
    cells = [(f, t) for f in range(n_families) for t in range(n_tasks)]
    runs: list[tuple[RunSpec, int, int]] = []
    idx = 0
    for rep in range(n_reps):
        for f, t in cells:
            fam = families[f]
            runs.append((RunSpec(
                run_index=idx,
                family=fam["model"],
                family_short=fam["short"],
                task=tasks[t],
                replicate=rep,
                seed=SEED_BASE + idx,
                data_seed=DATA_SEED_BASE + idx,
                run_dir=out_dir / fam["short"] / tasks[t] / f"run_{idx:03d}",
            ), f, t))
            idx += 1

    for run, f, t in runs:
        _write_run(run, f, t, geoms[f], rank, n_channels, task_effect, seed)

    manifest = {
        "updated_at": _PLACEHOLDER_END,
        "campaign": {
            "tag": "synthetic",
            "bank_root": str(out_dir),
            "families": families,
            "tasks": tasks,
            "n_replicates": n_reps,
            "n_runs": len(runs),
            "max_steps": METRIC_STEPS[-1],
            "pool_cap": 1000,
            "seed_base": SEED_BASE,
            "data_seed_base": DATA_SEED_BASE,
            "val_seed": 777,
            "val_size": 50,
        },
        "status_counts": {"COMPLETE": len(runs)},
        "runs": [run.to_manifest_entry("COMPLETE") for run, _, _ in runs],
    }
    (out_dir / "bank_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    marker = {
        "synthetic": True,
        "generator": "scripts/asset1_synth.py",
        "params": {
            "n_families": n_families, "n_tasks": n_tasks, "n_reps": n_reps,
            "n_layers": n_layers, "d_model": d_model, "rank": rank,
            "n_channels": n_channels, "task_effect": task_effect,
            "seed": seed,
        },
        "n_runs": len(runs),
    }
    (out_dir / MARKER_NAME).write_text(json.dumps(marker, indent=2),
                                       encoding="utf-8")

    return {"out_dir": out_dir, "manifest": manifest, "n_runs": len(runs),
            "families": families, "tasks": tasks}


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a miniature SYNTHETIC Asset-1 bank (real "
                    "schema, planted structure) for analysis-tool "
                    "validation. Refuses to touch the real bank tree.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-families", type=int, default=2)
    parser.add_argument("--n-tasks", type=int, default=3)
    parser.add_argument("--n-reps", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--n-channels", type=int, default=2)
    parser.add_argument("--task-effect", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0,
                        help="master seed for the whole fixture (default 0)")
    args = parser.parse_args()

    info = make_synthetic_bank(
        args.out_dir, n_families=args.n_families, n_tasks=args.n_tasks,
        n_reps=args.n_reps, n_layers=args.n_layers, d_model=args.d_model,
        rank=args.rank, n_channels=args.n_channels,
        task_effect=args.task_effect, seed=args.seed)
    print(f"Synthetic bank written: {info['out_dir']} "
          f"({info['n_runs']} runs, families="
          f"{[f['short'] for f in info['families']]}, tasks={info['tasks']})")


if __name__ == "__main__":
    main()
