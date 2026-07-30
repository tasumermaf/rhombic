"""Tinker signal pilot (E-T4 pre-step) — task-identity readout on Tinker adapters.

Answers the pilot's one question: at ~1M train tokens per run, is TASK IDENTITY
legible from the exported adapter weights?

    SIGNAL = every same-task pair is closer than every cross-task pair,
             reported independently for the RAW and the GAUGE-CANONICAL
             feature space.

RUNS UNDER THE ``falco`` CONDA ENV (needs torch + safetensors):
    C:\\miniconda3\\envs\\falco\\python.exe

Bridgeless adaptation of the Asset-1 canonicalization
-----------------------------------------------------
Tinker exports standard PEFT adapters: per module a ``lora_A`` (r, d_in) and a
``lora_B`` (d_out, r), with NO RhombiLoRA bridge. The Asset-1 readout absorbs
bridge + alpha/rank scaling into B before decomposing
(``asset1_canonicalize.effective_factors``); here the bridge-absorption step
becomes the IDENTITY (E = I) and only the scalar alpha/rank scaling remains:

    B_eff = (lora_alpha / r) * B        A_eff = A
    DW    = B_eff @ A_eff               (the true update the model applies)

The canonical object is then the same GL(r)-invariant exact r-slot SVD of DW
computed WITHOUT materializing DW, via the identical QR->SVD utilities and the
identical deterministic sign convention imported from
``scripts/asset1_canonicalize.py`` (W2T arXiv:2603.15990 Sec. 3). Nothing about
the canonicalization is re-implemented here — only the factor construction
differs, and only by dropping the bridge.

Two feature spaces
------------------
raw        per module, flattened A then B, modules in sorted-name order. This
           is the gauge-DEPENDENT representation: (B G, G^-1 A) leaves DW
           unchanged for any G in GL(r) but moves this vector arbitrarily.
canonical  ``feature_vector(..., variant='full')`` — per module
           [log1p(sigma); (U^T P_u).flatten(); (V^T P_v).flatten()] with the
           shared fixed seeded Gaussian probes. GL(r)-invariant.

Expectation going in: canonical separates; raw may not. Whichever way it lands
is the finding — a null result here is exactly the S12 lesson applied to
ourselves, and stops the 60-run mini-bank before it is paid for.

Metrics reported
----------------
* all 15 pairwise cosine distances (1 - cosine similarity) per space
* max(within-task) vs min(cross-task) and the separation margin
* 1-NN task-identity accuracy (is each adapter's nearest neighbour same-task?)

Usage
-----
    python scripts/tinker_pilot_signal.py
    python scripts/tinker_pilot_signal.py --bank results/tinker-pilot
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The canonicalization itself is imported, never reimplemented.
from asset1_canonicalize import (  # noqa: E402
    canonicalize_module, feature_vector)

_DTYPE = torch.float64


# ── Adapter loading (standard PEFT, bridgeless) ──────────────────────


def _module_key(tensor_name: str) -> tuple[str, str] | None:
    """('<module path>', 'lora_A'|'lora_B') for a PEFT tensor name, else None.

    Tolerates the common PEFT spellings, e.g.
      base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
      ...q_proj.lora_A.default.weight
    """
    for field in ("lora_A", "lora_B"):
        marker = f".{field}"
        if marker in tensor_name:
            return tensor_name.split(marker, 1)[0], field
    return None


def load_bridgeless_adapter(run_dir: Path) -> tuple[dict[str, dict], dict]:
    """{module: {'lora_A': (r,d_in), 'lora_B': (d_out,r)}} plus the config."""
    st = list(run_dir.glob("adapter_model.safetensors"))
    if not st:
        st = sorted(run_dir.glob("*.safetensors"))
    if not st:
        raise FileNotFoundError(f"no .safetensors under {run_dir}")

    cfg_path = run_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    tensors: dict[str, torch.Tensor] = {}
    for path in st:
        tensors.update(load_file(str(path)))

    modules: dict[str, dict] = {}
    skipped: list[str] = []
    for name, tensor in tensors.items():
        parsed = _module_key(name)
        if parsed is None:
            skipped.append(name)
            continue
        module, field = parsed
        modules.setdefault(module, {})[field] = tensor
    for module, entry in list(modules.items()):
        if "lora_A" not in entry or "lora_B" not in entry:
            raise ValueError(f"{run_dir.name}: module {module!r} is missing "
                             f"{'lora_A' if 'lora_A' not in entry else 'lora_B'}")
    if not modules:
        raise ValueError(f"{run_dir.name}: no lora_A/lora_B pairs found; "
                         f"tensor names sampled: {sorted(tensors)[:5]}")
    if skipped:
        cfg = {**cfg, "_non_lora_tensors": sorted(skipped)[:10]}
    return modules, cfg


def effective_factors_bridgeless(entry: dict[str, torch.Tensor], scaling: float
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """(B_eff, A_eff) with DW = B_eff @ A_eff. Bridge absorption = identity."""
    A = entry["lora_A"].detach().to(_DTYPE)      # (r, d_in)
    B = entry["lora_B"].detach().to(_DTYPE)      # (d_out, r)
    if A.ndim != 2 or B.ndim != 2 or B.shape[1] != A.shape[0]:
        raise ValueError(f"bad LoRA shapes: A {tuple(A.shape)}, B {tuple(B.shape)}")
    return scaling * B, A


def scaling_from_config(cfg: dict, rank_fallback: int) -> tuple[float, dict]:
    """alpha / r, with use_rslora honoured if the config declares it."""
    r = int(cfg.get("r", rank_fallback) or rank_fallback)
    alpha = float(cfg.get("lora_alpha", r))
    if cfg.get("use_rslora"):
        scaling = alpha / (r ** 0.5)
        mode = "rslora: alpha/sqrt(r)"
    else:
        scaling = alpha / r
        mode = "alpha/r"
    return scaling, {"r": r, "lora_alpha": alpha, "scaling": scaling,
                     "scaling_mode": mode}


# ── Feature construction ─────────────────────────────────────────────


def raw_feature(modules: dict[str, dict]) -> np.ndarray:
    """Gauge-DEPENDENT: flattened A then B per module, sorted module order."""
    blocks = []
    for name in sorted(modules):
        entry = modules[name]
        blocks.append(entry["lora_A"].detach().to(torch.float32).reshape(-1))
        blocks.append(entry["lora_B"].detach().to(torch.float32).reshape(-1))
    return torch.cat(blocks).numpy()


def canonical_feature(modules: dict[str, dict], scaling: float,
                      proj_dim: int, proj_seed: int) -> np.ndarray:
    """GL(r)-invariant: W2T canonical (U, S, V) per module -> 'full' features."""
    canonical: dict[str, dict[str, torch.Tensor]] = {}
    for name in sorted(modules):
        B_eff, A_eff = effective_factors_bridgeless(modules[name], scaling)
        canonical[name] = canonicalize_module(B_eff, A_eff)
    return feature_vector(canonical, variant="full",
                          proj_dim=proj_dim, proj_seed=proj_seed).numpy()


# ── Pairwise geometry ────────────────────────────────────────────────


def cosine_distances(feats: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    runs = sorted(feats)
    norms = {r: float(np.linalg.norm(feats[r].astype(np.float64))) for r in runs}
    out: dict[tuple[str, str], float] = {}
    for a, b in itertools.combinations(runs, 2):
        xa = feats[a].astype(np.float64)
        xb = feats[b].astype(np.float64)
        denom = norms[a] * norms[b]
        cos = float(np.dot(xa, xb) / denom) if denom > 0 else float("nan")
        out[(a, b)] = 1.0 - cos
    return out


def evaluate(dists: dict[tuple[str, str], float], task_of: dict[str, str]) -> dict:
    within = {p: d for p, d in dists.items() if task_of[p[0]] == task_of[p[1]]}
    cross = {p: d for p, d in dists.items() if task_of[p[0]] != task_of[p[1]]}
    max_w = max(within.values()) if within else float("nan")
    min_c = min(cross.values()) if cross else float("nan")

    runs = sorted({r for p in dists for r in p})
    nn_correct = 0
    nn_detail = {}
    for r in runs:
        cand = {(o): dists[tuple(sorted((r, o)))] for o in runs if o != r}
        nearest = min(cand, key=cand.get)
        ok = task_of[nearest] == task_of[r]
        nn_correct += int(ok)
        nn_detail[r] = {"nearest": nearest, "distance": cand[nearest],
                        "same_task": ok}

    return {
        "n_pairs": len(dists),
        "n_within_task_pairs": len(within),
        "n_cross_task_pairs": len(cross),
        "within_task_distances": {f"{a}|{b}": d for (a, b), d in sorted(within.items())},
        "cross_task_distances": {f"{a}|{b}": d for (a, b), d in sorted(cross.items())},
        "max_within_task": max_w,
        "min_cross_task": min_c,
        "separation_margin": min_c - max_w,
        "separated": bool(max_w < min_c),
        "nn_task_accuracy": nn_correct / len(runs) if runs else float("nan"),
        "nn_detail": nn_detail,
    }


# ── Orchestration ────────────────────────────────────────────────────


def discover_runs(bank: Path) -> list[Path]:
    """Run dirs are the immediate children holding adapter files. The sibling
    data/ and smoke/ dirs hold no adapter at their top level and are skipped."""
    def is_run(d: Path) -> bool:
        if not d.is_dir():
            return False
        return ((d / "adapter_config.json").exists()
                or any(d.glob("*.safetensors")))

    return sorted(d for d in bank.iterdir() if is_run(d))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tinker pilot signal check")
    ap.add_argument("--bank", type=Path,
                    default=REPO_ROOT / "results" / "tinker-pilot")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--proj-dim", type=int, default=16)
    ap.add_argument("--proj-seed", type=int, default=0)
    ap.add_argument("--rank-fallback", type=int, default=32)
    args = ap.parse_args()

    run_dirs = discover_runs(args.bank)
    if not run_dirs:
        raise SystemExit(f"no adapter run dirs under {args.bank}")

    raw_feats: dict[str, np.ndarray] = {}
    can_feats: dict[str, np.ndarray] = {}
    task_of: dict[str, str] = {}
    meta: dict[str, dict] = {}

    for run_dir in run_dirs:
        run_id = run_dir.name
        modules, cfg = load_bridgeless_adapter(run_dir)
        scaling, scale_meta = scaling_from_config(cfg, args.rank_fallback)
        task_of[run_id] = run_id.rsplit("_", 1)[0]

        rf = raw_feature(modules)
        cf = canonical_feature(modules, scaling, args.proj_dim, args.proj_seed)
        raw_feats[run_id] = rf
        can_feats[run_id] = cf

        meta[run_id] = {
            "task": task_of[run_id],
            "n_modules": len(modules),
            "raw_dim": int(rf.size),
            "canonical_dim": int(cf.size),
            "raw_l2": float(np.linalg.norm(rf.astype(np.float64))),
            **scale_meta,
        }
        print(f"[signal] {run_id}: {len(modules)} modules, "
              f"raw dim {rf.size:,}, canonical dim {cf.size:,}, "
              f"{scale_meta['scaling_mode']} = {scaling:.4f}", flush=True)

    tasks = sorted(set(task_of.values()))
    print(f"\n[signal] {len(run_dirs)} adapters over {len(tasks)} tasks: {tasks}")

    results = {}
    for space, feats in (("raw", raw_feats), ("canonical", can_feats)):
        dists = cosine_distances(feats)
        ev = evaluate(dists, task_of)
        results[space] = ev
        print(f"\n=== {space.upper()} cosine distances ===")
        print("  within-task:")
        for k, v in ev["within_task_distances"].items():
            print(f"    {k:34s} {v:.6f}")
        print("  cross-task:")
        for k, v in ev["cross_task_distances"].items():
            print(f"    {k:34s} {v:.6f}")
        print(f"  max_within={ev['max_within_task']:.6f}  "
              f"min_cross={ev['min_cross_task']:.6f}  "
              f"margin={ev['separation_margin']:+.6f}")
        print(f"  SEPARATED: {ev['separated']}   "
              f"1-NN task accuracy: {ev['nn_task_accuracy']:.3f}")

    raw_ok = results["raw"]["separated"]
    can_ok = results["canonical"]["separated"]
    verdict = ("YES" if (raw_ok and can_ok)
               else "MIXED" if (raw_ok or can_ok) else "NO")
    print(f"\nSIGNAL = {verdict}  (raw separated={raw_ok}, "
          f"canonical separated={can_ok})")

    payload = {
        "bank": args.bank.as_posix(),
        "n_adapters": len(run_dirs),
        "tasks": tasks,
        "probe": {"proj_dim": args.proj_dim, "proj_seed": args.proj_seed},
        "per_adapter": meta,
        "spaces": results,
        "signal": verdict,
        "signal_definition": (
            "SIGNAL=YES iff every same-task pair is closer than every "
            "cross-task pair (max within < min cross) in BOTH the raw and the "
            "gauge-canonical feature space; MIXED if exactly one space "
            "separates; NO if neither does."),
    }
    out = args.out or (args.bank / "signal_results.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[signal] wrote {out}")


if __name__ == "__main__":
    main()
