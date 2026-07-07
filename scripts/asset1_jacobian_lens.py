"""Asset 1 — TRUE J-lens (Jacobian) output-referenced adapter signatures.

Level B of the output-referenced canonicalization lane
(GLOBAL_WORKSPACE_MAPPING_2026-07-07.md §2.2 / shortlist item 1). Level A
(asset1_vocab_signature.py) reads module updates through the FINAL-LAYER
linearization W_eff = W_U * g — the depth-0 special case. This tool
estimates the full linearized propagation from a module-output
perturbation to the final logits, averaged over contexts — a
J-lens-STYLE estimator (the paper's J_l composed through the
unembedding and restricted to within-position terms; both deviations
are pinned defaults requiring Director sign-off, see PINNED DEFAULTS),
J_m = E_c[ d logits_{t} / d h_{m,t} ] — and applies it to
adapter effective updates:

    jlens signature block per module = (S^T J_m) @ Delta_m @ X

with the IDENTICAL vocab sketch S and input probes X as Level A
(imported from asset1_vocab_signature — same seeds, same stream tags).
The two levels therefore differ ONLY in the readout map (W_eff vs
S^T J_m): comparing them isolates exactly what mid-network causal
propagation adds to the output-referenced axis.

Three cleanly separated stages
------------------------------
STAGE A (CPU, now): ``build_plan`` / --plan-only. Writes jlens_plan.json:
the pinned context set (or --contexts-file), sketch config, target
module pattern, seeds, and every pinned default. No bank access, no HF
import, no network.

STAGE B (GPU, POST-BANK / HERMES ONLY): ``estimate_lenses`` / --estimate.
Loads the frozen base model ONCE per family (one command per family,
exactly like asset1_d2_swap.py Stage B), hooks every target module, and
for each (context, sketch row) pair runs ONE backward pass, reading the
gradient of s_i . logits[t_last] at every hooked module output
simultaneously. Cost per family: n_contexts x sketch_dim backward
passes (default 32 x 8 = 256) — minutes on Hermes. transformers is
imported LAZILY inside Stage B functions only; the CLI refuses
--estimate without the explicit --i-have-gpu-and-bank-is-complete flag,
checked BEFORE any lazy import. NOT RUN NOW: the campaign owns the
local GPU; this runs on Hermes after ~Jul 8 or locally post-bank.

STAGE C (CPU, post-Stage-B): ``jlens_signature_for_adapter`` / --sign.
Applies saved lenses to bank adapters (bridge + scaling absorbed via
asset1_canonicalize.effective_factors — the same effective DW as every
other representation). Gauge-invariant for the same reason as Level A:
the signature is a fixed linear map of Delta. Real-bank invocations are
gated by asset1_analysis_io.require_complete_bank.

Estimator definition (pinned defaults, Director sign-off items marked)
----------------------------------------------------------------------
* Contexts: DEFAULT_CONTEXTS — 32 short neutral English sentences,
  pinned below (sha256 recorded in the plan). PINNED DEFAULT — DIRECTOR
  SIGN-OFF: neutral built-in prompts vs task-distribution prompts is an
  open design choice; task prompts would couple the lens to the bank's
  data and are deliberately avoided at this stage.
* Position: t = the last token of each context (batch size 1, no
  padding). Perturbation site and readout position are the SAME t —
  the within-position lens; cross-position terms are dropped. PINNED
  DEFAULT — DIRECTOR SIGN-OFF.
* Sketch: the logits are read through the SAME seeded vocab sketch S
  as Level A (asset1_vocab_signature.vocab_sketch, stream tag 72), so
  only S^T J_m in R^{sketch_dim x d_out} is ever stored per module
  (full J_m at V x d_out x n_modules is not storable).
* Averaging: arithmetic mean over contexts, float32 accumulation on
  device, float64 on save.
* Model dtype: float32 (Jacobian fidelity; 1.5B fits Hermes 4090 16GB).
  PINNED DEFAULT.
* Frozen model, no adapter installed: the lens measures the BASE
  model's propagation geometry, applied post-hoc to every adapter of
  the family (one lens per family, reused across all 240 adapters).
  PINNED DEFAULT — DIRECTOR SIGN-OFF (alternative: per-adapter lenses,
  240x the cost, couples the readout to the object being measured).

Import hygiene (tested): importing this module never imports
transformers and never initializes CUDA.

Usage
-----
    # now (CPU):
    python scripts/asset1_jacobian_lens.py --plan-only \
        --out-dir results/asset1-jlens
    # Hermes after ~Jul 8 / locally post-bank (per family):
    python scripts/asset1_jacobian_lens.py --estimate \
        --family qwen2.5-1.5b --out-dir results/asset1-jlens \
        --i-have-gpu-and-bank-is-complete
    # then (CPU):
    python scripts/asset1_jacobian_lens.py --sign \
        --bank-root results/asset1-bank --family qwen2.5-1.5b \
        --out-dir results/asset1-jlens
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asset1_analysis_io as aio  # noqa: E402  (torch-only)
from asset1_canonicalize import (  # noqa: E402
    effective_factors, load_adapter_modules)
from asset1_vocab_signature import (  # noqa: E402  (shared probes/sketch)
    DEFAULT_SEED, N_PROBES, SKETCH_DIM, probe_inputs, vocab_sketch)

# ── Pinned defaults ─────────────────────────────────────────────────

TARGET_MODULE_RE = r"model\.layers\.(\d+)\.self_attn\.([qkvo])_proj$"
MODEL_DTYPE = "float32"     # Jacobian fidelity (see docstring)
GATE_FLAG = "--i-have-gpu-and-bank-is-complete"

# 32 short neutral English sentences — the pinned context set. Chosen
# for topic diversity and zero overlap with the bank's task prompts;
# sha256 of the joined list is recorded in every plan/lens artifact.
DEFAULT_CONTEXTS: list[str] = [
    "The river froze solid before the first snow arrived.",
    "She counted the coins twice and wrote the total in the ledger.",
    "A cargo ship waited outside the harbor for the tide to turn.",
    "The recipe calls for two eggs, flour, and a pinch of salt.",
    "He tuned the violin slowly, listening for the fifth.",
    "The museum's east wing closes early on winter afternoons.",
    "Rain collected in the gutter and spilled over the window.",
    "The committee postponed its vote until the following week.",
    "A single lamp lit the corner of the reading room.",
    "The train to the coast leaves from platform nine.",
    "Farmers rotated the crops to keep the soil healthy.",
    "The bridge was painted gray after decades of rust.",
    "Her notebook filled with sketches of leaves and beetles.",
    "The bakery on the corner sells out of bread by noon.",
    "Two chess players sat in silence as the clock ticked.",
    "The lighthouse keeper logged the weather every morning.",
    "A gentle wind moved the curtains in the empty hall.",
    "The mechanic replaced the belt and checked the brakes.",
    "Students lined up outside the library before it opened.",
    "The orchard smelled of apples after the September rain.",
    "He folded the map along its worn creases.",
    "The clerk stamped each form and filed it by date.",
    "Snow settled on the statue in the quiet square.",
    "The ferry crossed the strait twice a day in summer.",
    "She repaired the hem with three quick stitches.",
    "The observatory dome opened as the sky darkened.",
    "A kettle whistled somewhere down the corridor.",
    "The gardener pruned the roses before the frost.",
    "The archive keeps letters older than the city itself.",
    "Wooden crates stacked neatly along the warehouse wall.",
    "The tide left a line of shells along the beach.",
    "He wound the old clock and set its hands by the radio.",
]

PINNED_DEFAULTS = {
    "contexts": (
        "32 built-in neutral English sentences (sha256 recorded); "
        "PINNED — DIRECTOR SIGN-OFF (alternative: task-distribution "
        "prompts, deliberately avoided to keep the lens decoupled from "
        "bank data)."),
    "position": (
        "last token; perturbation site == readout position "
        "(within-position lens; cross-position terms dropped). PINNED "
        "— DIRECTOR SIGN-OFF."),
    "sketch": (
        f"vocab sketch shared with asset1_vocab_signature (stream tag "
        f"72), sketch_dim={SKETCH_DIM}; only S^T J_m stored per module."),
    "averaging": "arithmetic mean over contexts",
    "model_dtype": MODEL_DTYPE + " — PINNED (Jacobian fidelity)",
    "lens_target": (
        "frozen BASE model, no adapter installed — one lens per family "
        "reused across all its adapters. PINNED — DIRECTOR SIGN-OFF "
        "(alternative: per-adapter lenses at 240x cost)."),
    "target_modules": TARGET_MODULE_RE,
    "probes": (
        f"input probes shared with asset1_vocab_signature (stream tag "
        f"71), n_probes={N_PROBES} — the two output-referenced levels "
        "differ ONLY in the readout map."),
}


def _contexts_sha256(contexts: list[str]) -> str:
    return hashlib.sha256("\n".join(contexts).encode("utf-8")).hexdigest()


# ── STAGE A — plan (CPU, no bank access, no HF) ─────────────────────


def build_plan(*, contexts: list[str] | None = None,
               sketch_dim: int = SKETCH_DIM, seed: int = DEFAULT_SEED
               ) -> dict:
    """Deterministic Stage-A plan. Pure CPU; touches nothing."""
    ctx = list(contexts) if contexts is not None else list(DEFAULT_CONTEXTS)
    if not ctx:
        raise ValueError("empty context list")
    return {
        "tool": "scripts/asset1_jacobian_lens.py",
        "stage": "A (plan)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_contexts": len(ctx),
        "contexts_sha256": _contexts_sha256(ctx),
        "contexts": ctx,
        "sketch_dim": sketch_dim,
        "seed": seed,
        "target_module_re": TARGET_MODULE_RE,
        "model_dtype": MODEL_DTYPE,
        "pinned_defaults": PINNED_DEFAULTS,
        "gate": (f"Stage B requires {GATE_FLAG}; runs on Hermes after "
                 f"~Jul 8 or locally post-bank ONLY."),
    }


def write_plan(out_dir: Path, plan: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "jlens_plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


# ── STAGE B — GPU lens estimation (POST-BANK ONLY) ──────────────────
#
# Nothing below runs unless the operator passes BOTH --estimate and
# --i-have-gpu-and-bank-is-complete. transformers is imported lazily
# INSIDE these functions so importing this module never touches HF.


def _load_frozen_model(model_id: str, device):
    """GPU GATE — POST-BANK ONLY. Frozen float32 base model + tokenizer.
    Lazy-imports transformers; never call from tests or pre-bank code."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


def estimate_lenses(model_id: str, plan: dict, out_dir: Path,
                    family_short: str, device_str: str = "cuda") -> dict:
    """GPU GATE — POST-BANK ONLY. Estimate S^T J_m for every target
    module of one family and save jlens_{family}.npz + metadata JSON.

    For each context (batch 1): forward with hooks replacing each target
    module's output by a grad-requiring clone; for each sketch row i,
    backward of s_i . logits[0, t_last] accumulates s_i^T dlogits/dh_m
    at EVERY hooked module output in one pass; the position-t row is
    read from each retained grad. Mean over contexts.
    """
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[jlens] REFUSING: --estimate on cuda but no "
                         "CUDA device is available")

    tokenizer, model = _load_frozen_model(model_id, device)
    pattern = re.compile(plan["target_module_re"])
    targets = {name: mod for name, mod in model.named_modules()
               if pattern.search(name)}
    if not targets:
        raise SystemExit(f"[jlens] no modules match "
                         f"{plan['target_module_re']!r} on {model_id}")

    vocab_size = model.get_output_embeddings().weight.shape[0]
    S = torch.from_numpy(
        vocab_sketch(vocab_size, plan["sketch_dim"], plan["seed"])
    ).to(device=device, dtype=torch.float32)          # (V, s)

    captured: dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def hook(_module, _inputs, output):
            out = output.clone().requires_grad_(True)
            captured[name] = out
            return out
        return hook

    handles = [mod.register_forward_hook(_make_hook(name))
               for name, mod in targets.items()]

    sums = {name: torch.zeros(plan["sketch_dim"],
                              mod.weight.shape[0],  # d_out
                              dtype=torch.float32)
            for name, mod in targets.items()}
    try:
        for ci, text in enumerate(plan["contexts"]):
            enc = tokenizer(text, return_tensors="pt").to(device)
            captured.clear()
            with torch.enable_grad():
                logits = model(**enc).logits          # (1, T, V)
                t_last = enc["input_ids"].shape[1] - 1
                for i in range(plan["sketch_dim"]):
                    scalar = (S[:, i] * logits[0, t_last]).sum()
                    retain = i < plan["sketch_dim"] - 1
                    grads = torch.autograd.grad(
                        scalar, list(captured.values()),
                        retain_graph=retain, allow_unused=True)
                    for name, g in zip(captured, grads):
                        if g is not None:
                            sums[name][i] += g[0, t_last].detach().cpu()
            print(f"[jlens] {family_short}: context {ci + 1}/"
                  f"{len(plan['contexts'])}", flush=True)
    finally:
        for h in handles:
            h.remove()

    n = len(plan["contexts"])
    lenses = {name: (t / n).to(torch.float64).numpy()
              for name, t in sums.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"jlens_{family_short}.npz"
    np.savez_compressed(npz_path, **lenses)
    meta = {
        "tool": "scripts/asset1_jacobian_lens.py",
        "stage": "B (lens estimation)",
        "family_short": family_short,
        "model": model_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_contexts": n,
        "contexts_sha256": plan["contexts_sha256"],
        "sketch_dim": plan["sketch_dim"],
        "seed": plan["seed"],
        "vocab_size": int(vocab_size),
        "n_modules": len(lenses),
        "pinned_defaults": PINNED_DEFAULTS,
        "npz": str(npz_path),
    }
    (out_dir / f"jlens_{family_short}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[jlens] lenses saved: {npz_path} ({len(lenses)} modules)",
          flush=True)
    return meta


# ── STAGE C — CPU adapter signatures from saved lenses ──────────────


def _lens_key_for_module(adapter_module_name: str,
                         lens_keys: list[str]) -> str | None:
    """Map a bank adapter module name (dot-safe, e.g.
    'model_layers_0_self_attn_q_proj') to the saved lens key
    ('model.layers.0.self_attn.q_proj'). Exact match after normalizing
    both to underscore form."""
    normalized = {k.replace(".", "_"): k for k in lens_keys}
    return normalized.get(adapter_module_name)


def jlens_signature_for_modules(modules: dict[str, dict[str, torch.Tensor]],
                                lenses: dict[str, np.ndarray], *,
                                n_probes: int = N_PROBES,
                                seed: int = DEFAULT_SEED
                                ) -> tuple[np.ndarray, dict]:
    """J-lens signature: concat over sorted modules of
    (S^T J_m) @ Delta_m @ X, flattened sketch-major — the Level-B
    analog of the Level-A sketch block (identical X; readout map
    S^T J_m instead of S^T W_eff). Gauge-invariant: a fixed linear map
    of Delta. Modules with no matching lens are excluded and recorded.
    """
    lens_keys = list(lenses)
    kept, excluded, blocks = [], [], []
    for name in sorted(modules):
        key = _lens_key_for_module(name, lens_keys)
        if key is None:
            excluded.append(name)
            continue
        L = np.asarray(lenses[key], dtype=np.float64)     # (s, d_out)
        B_eff, A_eff = effective_factors(modules[name])
        Bn, An = B_eff.numpy(), A_eff.numpy()
        if L.shape[1] != Bn.shape[0]:
            raise ValueError(
                f"lens/adapter d_out mismatch for {name}: "
                f"{L.shape[1]} vs {Bn.shape[0]}")
        X = probe_inputs(An.shape[1], n_probes, seed)
        blocks.append((L @ (Bn @ (An @ X))).ravel())      # (s*p,)
        kept.append(name)
    if not kept:
        raise ValueError("no adapter module matched a saved lens")
    sig = np.concatenate(blocks).astype(np.float32)
    layout = {
        "n_modules_kept": len(kept),
        "modules_excluded": excluded,
        "per_module_dim": int(blocks[0].size),
        "dim": int(sig.size),
        "n_probes": n_probes,
        "seed": seed,
    }
    return sig, layout


def jlens_signature_for_adapter(adapter_state_path: str | Path,
                                lenses: dict[str, np.ndarray], **cfg
                                ) -> tuple[np.ndarray, dict]:
    return jlens_signature_for_modules(
        load_adapter_modules(adapter_state_path), lenses, **cfg)


def sign_bank(bank_root: Path, out_dir: Path, family_short: str, *,
              n_probes: int = N_PROBES, seed: int = DEFAULT_SEED,
              expected_total: int = aio.EXPECTED_TOTAL_RUNS,
              allow_partial: bool = False) -> dict:
    """Stage C bank pass (CPU): apply saved lenses to every COMPLETE run
    of one family. Gated by require_complete_bank; refuses an out-dir
    inside the bank tree."""
    bank_root = Path(bank_root)
    out_dir = Path(out_dir)
    if "asset1-bank" in out_dir.resolve().parts:
        raise SystemExit(f"[jlens] REFUSING --out-dir {out_dir}: inside "
                         f"the live campaign tree.")
    aio.require_complete_bank(bank_root, allow_partial=allow_partial,
                              expected_total=expected_total)
    npz_path = out_dir / f"jlens_{family_short}.npz"
    if not npz_path.exists():
        raise SystemExit(f"[jlens] no lenses at {npz_path} — run Stage B "
                         f"first (post-bank, {GATE_FLAG}).")
    with np.load(npz_path) as data:
        lenses = {k: data[k] for k in data.files}

    records = [r for r in aio.iter_runs(bank_root)
               if r["family_short"] == family_short]
    rows, layout = [], None
    for rec in records:
        sig, layout = jlens_signature_for_adapter(
            Path(rec["run_dir"]) / "adapter_state.pt", lenses,
            n_probes=n_probes, seed=seed)
        rows.append(sig)
    X = np.stack(rows)
    sig_path = out_dir / f"jlens_signatures_{family_short}.npz"
    np.savez_compressed(
        sig_path, signatures=X,
        run_index=np.array([r["run_index"] for r in records]),
        task=np.array([r["task"] for r in records]),
        replicate=np.array([r["replicate"] for r in records]))
    meta = {
        "tool": "scripts/asset1_jacobian_lens.py",
        "stage": "C (adapter signatures)",
        "family_short": family_short,
        "n_runs": len(records),
        "signature_dim": int(X.shape[1]),
        "layout": layout,
        "lenses_npz": str(npz_path),
        "pinned_defaults": PINNED_DEFAULTS,
        "npz": str(sig_path),
        "exploratory_only": bool(allow_partial),
    }
    (out_dir / f"jlens_signatures_{family_short}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[jlens] signatures: {sig_path} "
          f"({X.shape[0]} runs x {X.shape[1]} dims)", flush=True)
    return meta


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TRUE J-lens output-referenced adapter signatures. "
                    "Stage A (--plan-only) is CPU/now; Stage B "
                    f"(--estimate) is GPU-gated by {GATE_FLAG} and runs "
                    "post-bank/Hermes ONLY; Stage C (--sign) applies "
                    "saved lenses on CPU.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true",
                        help="Stage A: write jlens_plan.json and exit")
    parser.add_argument("--estimate", action="store_true",
                        help="Stage B: estimate lenses (GPU, gated)")
    parser.add_argument("--sign", action="store_true",
                        help="Stage C: adapter signatures from saved "
                             "lenses (CPU, bank-gated)")
    parser.add_argument("--family", type=str, default=None,
                        help="family_short (required for --estimate / "
                             "--sign; one command per family)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="HF model id for --estimate (default: "
                             "resolved from the bank manifest)")
    parser.add_argument("--bank-root", type=Path,
                        default=aio.REAL_BANK_ROOT)
    parser.add_argument("--contexts-file", type=Path, default=None,
                        help="One context per line (default: the pinned "
                             "32-sentence built-in set)")
    parser.add_argument("--sketch-dim", type=int, default=SKETCH_DIM)
    parser.add_argument("--n-probes", type=int, default=N_PROBES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--allow-partial-bank", action="store_true")
    parser.add_argument("--i-have-gpu-and-bank-is-complete",
                        action="store_true",
                        help="Explicit Stage-B acknowledgment: a GPU is "
                             "free for this work AND the bank campaign "
                             "is complete (Hermes post ~Jul 8, or local "
                             "post-bank)")
    args = parser.parse_args(argv)

    contexts = None
    if args.contexts_file is not None:
        contexts = [ln for ln in args.contexts_file.read_text(
            encoding="utf-8").splitlines() if ln.strip()]

    if args.plan_only:
        plan = build_plan(contexts=contexts, sketch_dim=args.sketch_dim,
                          seed=args.seed)
        path = write_plan(args.out_dir, plan)
        print(f"[jlens] plan written: {path} "
              f"({plan['n_contexts']} contexts, "
              f"sketch_dim={plan['sketch_dim']})", flush=True)
        return

    if args.estimate:
        # Gate BEFORE any bank access or lazy heavyweight import.
        if not args.i_have_gpu_and_bank_is_complete:
            raise SystemExit(
                "[jlens] REFUSING --estimate: this is the GPU stage and "
                "the local GPU belongs to the live asset-1 campaign. "
                f"Run on Hermes after ~Jul 8 or locally post-bank, and "
                f"pass {GATE_FLAG} to acknowledge.")
        if not args.family:
            raise SystemExit("[jlens] --estimate requires --family")
        plan_path = args.out_dir / "jlens_plan.json"
        plan = (json.loads(plan_path.read_text(encoding="utf-8"))
                if plan_path.exists()
                else build_plan(contexts=contexts,
                                sketch_dim=args.sketch_dim,
                                seed=args.seed))
        model_id = args.model_id
        if model_id is None:
            manifest = aio.require_complete_bank(
                args.bank_root, allow_partial=args.allow_partial_bank)
            fam = next((f for f in manifest["campaign"]["families"]
                        if f["short"] == args.family), None)
            if fam is None:
                raise SystemExit(f"[jlens] family {args.family!r} not in "
                                 f"the bank manifest")
            model_id = fam["model"]
        estimate_lenses(model_id, plan, args.out_dir, args.family,
                        device_str=args.device)
        return

    if args.sign:
        if not args.family:
            raise SystemExit("[jlens] --sign requires --family")
        sign_bank(args.bank_root, args.out_dir, args.family,
                  n_probes=args.n_probes, seed=args.seed,
                  allow_partial=args.allow_partial_bank)
        return

    parser.error("choose one of --plan-only / --estimate / --sign")


if __name__ == "__main__":
    main()
