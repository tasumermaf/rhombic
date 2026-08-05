"""E-T4 Tinker mini-bank — merge_lint over a vertex-disjoint adapter pairing.

Two jobs, both thin:

1. **Bridge the format.** ``scripts/merge_lint.py`` consumes Asset-1 bank
   ``adapter_state.pt`` files — a flat dict of ``{module}.lora_A``,
   ``{module}.lora_B``, ``{module}.bridge``, ``{module}.scaling``. Tinker
   exports standard PEFT safetensors and is BRIDGELESS. No tensor value is
   altered by the conversion; the only question is the bridge key.

   **Measured finding, recorded rather than papered over: merge_lint as
   shipped REFUSES a bridgeless adapter.** Omitting the key — which
   ``asset1_canonicalize.effective_factors`` handles by construction ("a
   missing bridge (plain LoRA) degrades gracefully to E = I") — exits 2
   with ``REFUSED: module '...' has no 'bridge' tensor (include requested
   'bridge')``, because ``asset1_analysis_io.flatten_features`` defaults to
   ``include=("A", "B", "bridge")`` and is stricter than the absorption
   path. That refusal is reported as a result of the demo, not hidden.

   To then obtain a verdict, the bridge is written as the EXACT bridgeless
   identity. ``_expand_bridge(I_C, rank // C) == I_rank`` for any C
   dividing rank, so ``C = 1`` with ``bridge = [[1.0]]`` reproduces E = I
   exactly while adding the smallest possible block to the flattened
   vector. Consequences, both measured on a real pair rather than assumed:

     * ``l2_distance`` and every per-module L2 are EXACTLY unchanged —
       identical bridge blocks cancel in the difference (measured delta
       3.6e-15, i.e. float noise).
     * ``cos_distance`` IS perturbed, because ``flatten_features``
       concatenates RAW stored parameters and a constant identity block
       adds a common component to both vectors. Measured on
       alpaca_d0_i0 x pilot agnews_0: true bridgeless cos 0.065207769,
       C=1 cos 0.060392835 (delta -4.8e-03). The C=32 encoding, equally
       valid as E = I, would shift it ten times further (-4.7e-02), which
       is why C=1 is the encoding used.

   Each pair's TRUE bridgeless ``cos_distance``/``l2_distance`` (computed
   with ``include=("A", "B")``) is recorded alongside merge_lint's own
   values so the perturbation is visible in the artifact.

2. **Pick a vertex-disjoint pairing.** Each adapter appears in AT MOST ONE
   pair, so the linted pairs are statistically independent draws rather
   than an overlapping web sharing endpoints. Pairs are drawn across
   DIFFERENT tasks (the interesting merge case) by a fixed-seed maximum
   matching over a shuffled adapter list.

The linter itself is invoked unmodified as a subprocess, and its verdicts
are recorded verbatim. Tinker adapters are expected to land OUT-OF-FAMILY
(their Qwen3-8B module name set is not one of the shipped bank families),
in which case merge_lint prints its EXTRAPOLATION banner and falls back to
the pooled distance-only model. That is a real, reportable outcome — the
demo shows the linter's refusal/fallback discipline working on genuinely
foreign input, not a calibrated merge prediction.

RUNS UNDER THE ``falco`` CONDA ENV (needs torch + safetensors):
    C:\\miniconda3\\envs\\falco\\python.exe

Usage
-----
    python scripts/tinker_minibank_merge_lint.py
    python scripts/tinker_minibank_merge_lint.py --pair-seed 0 --keep-converted
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

from tinker_minibank_signal import discover_runs, labels_for  # noqa: E402
from tinker_pilot_signal import _module_key, scaling_from_config  # noqa: E402


def convert_to_bank_format(run_dir: Path, dest: Path,
                           rank_fallback: int = 32,
                           write_bridge: bool = True) -> dict:
    """PEFT safetensors -> Asset-1 flat ``adapter_state.pt``. Values unchanged.

    ``scaling`` carries alpha/r. ``bridge`` is the 1x1 identity — the exact
    bridgeless E = I, minimal-footprint (see module docstring). Pass
    ``write_bridge=False`` to emit the strictly-bridgeless file that
    merge_lint refuses; that refusal is itself recorded by the demo.
    """
    st = sorted(run_dir.glob("adapter_model.safetensors")) or \
        sorted(run_dir.glob("*.safetensors"))
    if not st:
        raise FileNotFoundError(f"no .safetensors under {run_dir}")
    cfg_path = run_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    scaling, scale_meta = scaling_from_config(cfg, rank_fallback)

    tensors: dict[str, torch.Tensor] = {}
    for p in st:
        tensors.update(load_file(str(p)))

    flat: dict[str, torch.Tensor] = {}
    modules: set[str] = set()
    for name, tensor in tensors.items():
        parsed = _module_key(name)
        if parsed is None:
            continue
        module, field = parsed
        flat[f"{module}.{field}"] = tensor.contiguous()
        modules.add(module)
    for module in modules:
        flat[f"{module}.scaling"] = torch.tensor(float(scaling))
        flat[f"{module}.rank"] = torch.tensor(int(scale_meta["r"]))
        if write_bridge:
            flat[f"{module}.bridge"] = torch.eye(1, dtype=torch.float32)
            flat[f"{module}.n_channels"] = torch.tensor(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(flat, str(dest))
    return {"run_id": run_dir.name, "n_modules": len(modules),
            "path": dest.as_posix(), "bridge": "I_1 (exact E=I)" if write_bridge
            else None, **scale_meta}


def true_bridgeless_distances(run_dir_a: Path, run_dir_b: Path) -> dict:
    """cos/l2 over ('A','B') only — the undistorted reference for a pair."""
    import asset1_analysis_io as aio
    from tinker_pilot_signal import load_bridgeless_adapter
    ma, _ = load_bridgeless_adapter(run_dir_a)
    mb, _ = load_bridgeless_adapter(run_dir_b)
    va = aio.flatten_features(ma, include=("A", "B")).astype(np.float64)
    vb = aio.flatten_features(mb, include=("A", "B")).astype(np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return {"cos_distance_true_bridgeless": float(1.0 - (va @ vb) / (na * nb)),
            "l2_distance_true_bridgeless": float(np.linalg.norm(va - vb))}


def vertex_disjoint_pairs(runs: list[str], labels: dict[str, dict],
                          seed: int) -> list[tuple[str, str]]:
    """Greedy maximum matching on a shuffled list, cross-task edges only.

    Vertex-disjoint by construction: an adapter is removed from the pool the
    moment it is matched, so no adapter appears in two pairs.
    """
    rng = np.random.default_rng(seed)
    pool = list(runs)
    rng.shuffle(pool)
    pairs: list[tuple[str, str]] = []
    used: set[str] = set()
    for a in pool:
        if a in used:
            continue
        for b in pool:
            if b in used or b == a:
                continue
            if labels[b]["task"] == labels[a]["task"]:
                continue
            pairs.append((a, b))
            used.add(a)
            used.add(b)
            break
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="merge_lint over the mini-bank")
    ap.add_argument("--bank", type=Path,
                    default=REPO_ROOT / "results" / "tinker-minibank")
    ap.add_argument("--bank-dir", type=Path,
                    default=REPO_ROOT / "results" / "asset1-delivery-verify",
                    help="shipped reference data for merge_lint")
    ap.add_argument("--pair-seed", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--keep-converted", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dirs = discover_runs(args.bank)
    if len(run_dirs) < 2:
        raise SystemExit(f"need >=2 exported adapters under {args.bank}")
    runs = [d.name for d in run_dirs]
    labels = {d.name: labels_for(d) for d in run_dirs}
    by_id = {d.name: d for d in run_dirs}

    pairs = vertex_disjoint_pairs(runs, labels, args.pair_seed)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    matched = {r for p in pairs for r in p}
    print(f"[pairs] {len(pairs)} vertex-disjoint cross-task pairs over "
          f"{len(matched)}/{len(runs)} adapters (seed {args.pair_seed})")

    conv_dir = args.bank / "_merge_lint_converted"
    converted: dict[str, dict] = {}
    for rid in sorted(matched):
        converted[rid] = convert_to_bank_format(
            by_id[rid], conv_dir / rid / "adapter_state.pt")
    print(f"[convert] {len(converted)} adapters -> bank flat format "
          f"({conv_dir})")

    # Record, once, that merge_lint as shipped refuses a strictly bridgeless
    # adapter. This is a finding about the linter's interface, produced by
    # running it on genuinely foreign input — not a step to be skipped.
    a0, b0 = pairs[0]
    strict_dir = conv_dir / "_strict_bridgeless"
    sa = convert_to_bank_format(by_id[a0], strict_dir / "a" / "adapter_state.pt",
                                write_bridge=False)
    sb = convert_to_bank_format(by_id[b0], strict_dir / "b" / "adapter_state.pt",
                                write_bridge=False)
    strict = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "merge_lint.py"), sa["path"],
         sb["path"], "--bank-dir", str(args.bank_dir), "--json"],
        capture_output=True, text=True)
    bridgeless_probe = {
        "pair": [a0, b0], "exit_code": strict.returncode,
        "stderr": strict.stderr.strip()[-500:],
        "note": ("merge_lint as shipped requires a 'bridge' tensor "
                 "(flatten_features defaults to include=('A','B','bridge')); "
                 "a strictly bridgeless PEFT adapter is refused."),
    }
    print(f"[probe] strictly-bridgeless input -> exit {strict.returncode}: "
          f"{strict.stderr.strip()[-140:]}")

    results = []
    for a, b in pairs:
        cmd = [sys.executable, str(SCRIPTS_DIR / "merge_lint.py"),
               converted[a]["path"], converted[b]["path"],
               "--bank-dir", str(args.bank_dir), "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        entry = {
            "adapter_a": a, "adapter_b": b,
            "task_a": labels[a]["task"], "task_b": labels[b]["task"],
            "exit_code": proc.returncode,
            **true_bridgeless_distances(by_id[a], by_id[b]),
        }
        try:
            entry["verdict"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            entry["verdict"] = None
            entry["stdout"] = proc.stdout[-2000:]
            entry["stderr"] = proc.stderr[-2000:]
        results.append(entry)
        v = entry["verdict"] or {}
        print(f"[lint] {a:22s} x {b:22s} exit={proc.returncode} "
              f"in_family={v.get('in_family')} "
              f"p={v.get('probability_degrade', v.get('probability'))}",
              flush=True)

    if not args.keep_converted:
        import shutil
        shutil.rmtree(conv_dir, ignore_errors=True)
        print(f"[convert] removed {conv_dir}")

    exits = {}
    for r in results:
        exits[r["exit_code"]] = exits.get(r["exit_code"], 0) + 1
    in_fam = sum(1 for r in results if (r["verdict"] or {}).get("in_family"))
    payload = {
        "bank": args.bank.as_posix(),
        "pair_seed": args.pair_seed,
        "n_adapters": len(runs),
        "n_pairs": len(pairs),
        "pairing": "vertex-disjoint greedy maximum matching, cross-task edges",
        "n_in_family": in_fam,
        "exit_code_counts": {str(k): v for k, v in sorted(exits.items())},
        "bridgeless_refusal_probe": bridgeless_probe,
        "bridge_encoding": (
            "1x1 identity == exact E=I; l2 distances unchanged, cos_distance "
            "perturbed by the constant block (see module docstring)"),
        "converted_meta": converted,
        "results": results,
    }
    out = args.out or (args.bank / "merge_lint_results.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n[lint] {len(pairs)} pairs, in_family={in_fam}, "
          f"exits={payload['exit_code_counts']}")
    print(f"[lint] wrote {out}")


if __name__ == "__main__":
    main()
