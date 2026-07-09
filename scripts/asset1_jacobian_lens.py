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
    DEFAULT_SEED, N_PROBES, SKETCH_DIM, loo_nearest_centroid_accuracy,
    probe_inputs, vocab_sketch)

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
        "(within-position lens; cross-position terms dropped). "
        "DIRECTOR APPROVED 2026-07-09 "
        "(docs/DIRECTOR_RULING_JLENS_STAGEB_2026-07-09.md) — accepted "
        "as conservative-for-arbiter; see stageb_ruling in this "
        "artifact for the two disclosure conditions."),
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


# Director's Stage-B ruling (2026-07-09), encoded into EVERY output
# artifact (plan, lens, signature) per the ruling's disclosure
# conditions — a reader of any result must see the bias direction and
# the reporting gate without opening this file.
STAGEB_RULING = {
    "status": ("Stage-B defaults 1-4 APPROVED for reportable use — "
               "docs/DIRECTOR_RULING_JLENS_STAGEB_2026-07-09.md"),
    "conservative_disclosure": (
        "Cross-position terms are DROPPED (within-position lens): Level "
        "B UNDERESTIMATES propagation rather than fabricating it — the "
        "conservative direction for the arbiter role. A null Level-B "
        "reading is therefore protected by construction (Director "
        "condition 2(i), 2026-07-09)."),
    "lower_bound_framing": (
        "Any Level-B CONFIRMATION of propagation structure is a LOWER "
        "BOUND: the true cross-position propagation is at least this "
        "strong (Director condition 2(ii), 2026-07-09)."),
    "positive_control_gate": (
        "HARD CONDITION: no Level-B signature is REPORTED as an arbiter "
        "verdict until the synthetic positive control passes — planted "
        "propagation recovered above a matched null AND a planted "
        "output-null update reading as null, Level-A-selftest class. "
        "Lens construction/estimation are unaffected (null-class)."),
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
        "stageb_ruling": STAGEB_RULING,
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
        "stageb_ruling": STAGEB_RULING,
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
        "stageb_ruling": STAGEB_RULING,
        "npz": str(sig_path),
        "exploratory_only": bool(allow_partial),
    }
    (out_dir / f"jlens_signatures_{family_short}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[jlens] signatures: {sig_path} "
          f"({X.shape[0]} runs x {X.shape[1]} dims)", flush=True)
    return meta


# ── STAGE PC — synthetic positive control (Director condition) ──────
#
# The one HARD condition of docs/DIRECTOR_RULING_JLENS_STAGEB_2026-07-09.md:
# before any Level-B signature is reported as an arbiter verdict, a
# synthetic positive control (Level-A-selftest class) must show the lens
#   (a) RECOVERS a planted propagation structure above a matched null, and
#   (b) reads a genuinely OUTPUT-NULL planted update as null.
#
# What is validated is the LENS -> SIGNATURE pathway itself: that
# (S^T J) @ Delta @ X — the exact production code path
# (jlens_signature_for_modules, with the import-shared stream-71 probes
# and stream-72 sketch) — recovers task identity that a TOY FROZEN MAP
# J_toy carries to the output, and only when the lens matches J_toy. No
# bank contact, no transformers, no GPU, CPU/seconds. The toy map stands
# in for a Stage-B estimated lens; the arithmetic exercised is identical.
#
# Design in three sentences: a toy frozen map is a set of seeded linear
# Jacobians J_toy[m] (V_toy x d_out, V_toy < d_out so each has a real
# null space), whose production lens is S^T J_toy[m]; the PLANTED bank
# writes, per task, an effective update whose column space J_toy carries
# to a fixed output direction (recoverable through the lens) plus a large
# OUTPUT-NULL confound that the correct lens annihilates but a wrong lens
# cannot — this is what makes recovery lens-specific rather than a
# tautology. The MATCHED-NULL bank runs the identical construction with a
# per-run random plant (no task->direction association), and the
# OUTPUT-NULL bank writes each task's identity entirely inside null(J_toy)
# so the lens output is numerically zero while the raw update still
# carries the task (proven by reading it through an identity lens). The
# acceptance statistic is Level A's own loo_nearest_centroid_accuracy on
# L2-normalized signatures.

CONTROL_SEED = 20260709          # PINNED — never date/entropy-based
PC_D_MODEL = 16                  # toy module output / residual width (d_out)
PC_D_IN = 16                     # toy module input width (d_in)
PC_V_TOY = 8                     # toy output ("logit") dim; < d_out => null
PC_N_TASKS = 4                   # planted tasks K (chance = 1/K = 0.25)
PC_N_REPS = 6                    # replicates R per task (K*R = 24 runs/bank)
PC_MODULES: tuple[str, ...] = (  # dotted lens keys (production naming)
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.1.self_attn.o_proj",
)
PC_CONF_SCALE = 8.0              # output-null confound magnitude — invisible
                                 # to the correct lens, defeats a wrong lens
PC_VIS_NOISE = 0.03              # within-task spread in the output-VISIBLE
                                 # subspace (seen by the correct lens)
PC_ONULL_NOISE = 0.05            # within-task spread INSIDE null(J_toy)
PC_GAIN_LO = 0.5                 # radial per-run gain in [LO, LO+1) (Level A
                                 # dev_mag analog; removed by L2-normalization)
PC_N_PERM = 500                  # permutation-null replicates
PC_SABOTAGE_SEED_OFFSET = 0x5A80  # wrong-map seed = CONTROL_SEED + this

# Pinned PASS thresholds (every gate a named constant, recorded in JSON).
PC_RESIDUAL_RATIO_MAX = 1e-6     # ||lensed output-null|| / ||planted||
PC_ONULL_MAXABS_MAX = 1e-8       # absolute "reads nothing" ceiling
PC_NULL_BAND_HI = 0.60           # matched/sabotage LOO must sit at/below this
                                 # (chance 0.25; 0.60 is a wide safety band)

# rng stream tags — house convention (default_rng([seed, tag, *idx])),
# disjoint from the shared 71/72/73 and from each other.
_PC_STREAM_J = 80                # toy Jacobian per module
_PC_STREAM_U = 81                # visible task directions u_k
_PC_STREAM_W = 82                # per-module input direction w_m
_PC_STREAM_GAIN = 83             # per-run radial gain
_PC_STREAM_CONF = 84             # per-run/module output-null confound
_PC_STREAM_VISN = 85             # per-run/module visible within-task noise
_PC_STREAM_CNULL = 86            # output-null task directions c_k
_PC_STREAM_ONULLN = 87           # output-null within-task null noise
_PC_STREAM_RANDU = 88            # matched-null per-run random plant
_PC_STREAM_PERM = 89            # permutation-null shuffles


def _pc_guard_out_dir(out_dir: Path) -> None:
    """Refuse an --out-dir inside the live campaign tree (same discipline
    as sign_bank / vocab_signature)."""
    if "asset1-bank" in Path(out_dir).resolve().parts:
        raise SystemExit(
            f"[jlens] REFUSING --out-dir {out_dir}: inside the live "
            f"campaign tree (results/asset1-bank is write-protected).")


def build_toy_frozen_map(seed: int = CONTROL_SEED) -> dict:
    """The TOY FROZEN MAP: a seeded linear Jacobian per module and the
    production lens S^T J_toy built from it.

    For each module J_toy[m] in R^{V_toy x d_out} (V_toy < d_out, so a
    real null space of dimension d_out - V_toy exists), with:
      * J_pinv[m] = right inverse (J J^T)^{-1} folded — J @ J_pinv = I;
      * null_basis[m] = orthonormal basis of null(J_toy[m]) (J @ N = 0);
      * lenses[m]     = S^T @ J_toy[m] in R^{sketch_dim x d_out} — the
                        EXACT shape/role of a real Stage-B lens;
      * raw_lenses[m] = I_{d_out} — reads the raw update Delta @ X through
                        the identical code path (no propagation map).
    S is the import-shared vocab_sketch (stream 72). Returns the maps plus
    per-module geometry diagnostics (rank, ||J N||, ||J J_pinv - I||).
    """
    S = vocab_sketch(PC_V_TOY, SKETCH_DIM, seed)          # (V_toy, s)
    J_toy, J_pinv, null_basis, lenses, raw_lenses = {}, {}, {}, {}, {}
    geometry = []
    for mi, name in enumerate(PC_MODULES):
        d_out = PC_D_MODEL
        rng = np.random.default_rng([seed, _PC_STREAM_J, mi])
        J = rng.standard_normal((PC_V_TOY, d_out))
        rank = int(np.linalg.matrix_rank(J))
        if rank < PC_V_TOY:                                # generically never
            raise RuntimeError(f"toy J for {name} is rank-deficient "
                               f"({rank} < {PC_V_TOY})")
        Jp = J.T @ np.linalg.inv(J @ J.T)                 # (d_out, V_toy)
        _, _, Vh = np.linalg.svd(J, full_matrices=True)   # Vh: (d_out, d_out)
        N = Vh[PC_V_TOY:].T                               # (d_out, null_dim)
        J_toy[name] = J
        J_pinv[name] = Jp
        null_basis[name] = N
        lenses[name] = S.T @ J                            # (s, d_out)
        raw_lenses[name] = np.eye(d_out)
        geometry.append({
            "module": name,
            "d_out": d_out, "d_in": PC_D_IN, "V_toy": PC_V_TOY,
            "null_dim": int(N.shape[1]),
            "J_row_rank": rank,
            "J_pinv_identity_maxerr": float(
                np.abs(J @ Jp - np.eye(PC_V_TOY)).max()),
            "J_nullbasis_maxabs": float(np.abs(J @ N).max()),
        })
    return {"seed": int(seed), "S": S, "module_names": list(PC_MODULES),
            "J_toy": J_toy, "J_pinv": J_pinv, "null_basis": null_basis,
            "lenses": lenses, "raw_lenses": raw_lenses, "geometry": geometry}


def _pc_task_dirs(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal task directions: u_k in R^{V_toy} (output-VISIBLE) and
    c_k in R^{null_dim} (null coordinates). Shared across all modules and
    banks for a given seed."""
    u, _ = np.linalg.qr(np.random.default_rng(
        [seed, _PC_STREAM_U]).standard_normal((PC_V_TOY, PC_N_TASKS)))
    null_dim = PC_D_MODEL - PC_V_TOY
    c, _ = np.linalg.qr(np.random.default_rng(
        [seed, _PC_STREAM_CNULL]).standard_normal((null_dim, PC_N_TASKS)))
    return u[:, :PC_N_TASKS], c[:, :PC_N_TASKS]


def _pc_input_dir(seed: int, mi: int) -> np.ndarray:
    w = np.random.default_rng([seed, _PC_STREAM_W, mi]).standard_normal(PC_D_IN)
    return w / np.linalg.norm(w)


def _pc_gain(seed: int, run_index: int) -> float:
    return PC_GAIN_LO + float(
        np.random.default_rng([seed, _PC_STREAM_GAIN, run_index]).uniform())


def _pc_module(delta: np.ndarray) -> dict[str, torch.Tensor]:
    """Wrap an effective update Delta (d_out x d_in) as a plain-LoRA module
    dict so effective_factors returns exactly (Delta, I): lora_B = Delta,
    lora_A = I. The production signature path then computes Delta @ X."""
    return {"lora_A": torch.eye(delta.shape[1], dtype=torch.float64),
            "lora_B": torch.from_numpy(np.ascontiguousarray(delta))}


def build_planted_bank(toy: dict, seed: int = CONTROL_SEED, *,
                       randomize_task: bool = False
                       ) -> tuple[list[dict], np.ndarray]:
    """The PLANTED (or, with randomize_task, the MATCHED-NULL) bank.

    Per run (task k, replicate j) and module m the effective update is
        Delta = g * (J_pinv[m] u_k) w_m^T           # visible, task-carrying
              + PC_VIS_NOISE * (J_pinv[m] E)         # visible within-task noise
              + PC_CONF_SCALE * (N_m Z)              # OUTPUT-NULL confound
    with g a per-run radial gain, E/Z per-run/module Gaussians. Under the
    correct lens the confound vanishes (J_toy N = 0) so the signature is
    g (S^T u_k)(w_m^T X) + small visible noise — the planted task
    direction. randomize_task=True draws u per RUN instead of per task:
    identical machinery, no task->direction plant => the matched null."""
    u, _ = _pc_task_dirs(seed)
    banks: list[dict] = []
    labels: list[str] = []
    run_index = 0
    for k in range(PC_N_TASKS):
        for _j in range(PC_N_REPS):
            g = _pc_gain(seed, run_index)
            modules: dict = {}
            for mi, name in enumerate(toy["module_names"]):
                Jp = toy["J_pinv"][name]
                N = toy["null_basis"][name]
                w = _pc_input_dir(seed, mi)
                if randomize_task:
                    uk = np.random.default_rng(
                        [seed, _PC_STREAM_RANDU, run_index, mi]
                    ).standard_normal(PC_V_TOY)
                    uk = uk / np.linalg.norm(uk)
                else:
                    uk = u[:, k]
                delta = g * np.outer(Jp @ uk, w)
                E = np.random.default_rng(
                    [seed, _PC_STREAM_VISN, run_index, mi]
                ).standard_normal((PC_V_TOY, PC_D_IN))
                delta = delta + PC_VIS_NOISE * (Jp @ E)
                Z = np.random.default_rng(
                    [seed, _PC_STREAM_CONF, run_index, mi]
                ).standard_normal((N.shape[1], PC_D_IN))
                delta = delta + PC_CONF_SCALE * (N @ Z)
                modules[name.replace(".", "_")] = _pc_module(delta)
            banks.append(modules)
            labels.append(f"task{k}")
            run_index += 1
    return banks, np.array(labels)


def build_output_null_bank(toy: dict, seed: int = CONTROL_SEED
                           ) -> tuple[list[dict], np.ndarray]:
    """The OUTPUT-NULL bank: each task's identity lives ENTIRELY inside
    null(J_toy). Per run/module
        Delta = g * (N_m c_k) w_m^T + PC_ONULL_NOISE * (N_m Z),
    so the whole column space is in null(J_toy): the lens output
    (S^T J_toy) Delta X is numerically zero, yet the RAW update Delta @ X
    still carries c_k (recoverable through an identity lens) — a genuinely
    planted update that the lens correctly reads as nothing."""
    _, c = _pc_task_dirs(seed)
    banks: list[dict] = []
    labels: list[str] = []
    run_index = 0
    for k in range(PC_N_TASKS):
        for _j in range(PC_N_REPS):
            g = _pc_gain(seed, run_index)
            modules: dict = {}
            for mi, name in enumerate(toy["module_names"]):
                N = toy["null_basis"][name]
                w = _pc_input_dir(seed, mi)
                delta = g * np.outer(N @ c[:, k], w)
                Z = np.random.default_rng(
                    [seed, _PC_STREAM_ONULLN, run_index, mi]
                ).standard_normal((N.shape[1], PC_D_IN))
                delta = delta + PC_ONULL_NOISE * (N @ Z)
                modules[name.replace(".", "_")] = _pc_module(delta)
            banks.append(modules)
            labels.append(f"task{k}")
            run_index += 1
    return banks, np.array(labels)


def control_signatures(banks: list[dict], lenses: dict[str, np.ndarray], *,
                       seed: int = CONTROL_SEED,
                       n_probes: int = N_PROBES) -> np.ndarray:
    """Sign every run of a control bank through the PRODUCTION code path
    (jlens_signature_for_modules — same probes, same sketch). Returns
    (n_runs x dim) float64."""
    rows = [jlens_signature_for_modules(m, lenses, n_probes=n_probes,
                                        seed=seed)[0].astype(np.float64)
            for m in banks]
    return np.stack(rows)


def _pc_permutation_null(X: np.ndarray, labels: np.ndarray, seed: int,
                         n_perm: int = PC_N_PERM) -> dict:
    """Label-permutation null band of the LOO statistic on the planted
    signatures — the chance distribution for this exact geometry."""
    rng = np.random.default_rng([seed, _PC_STREAM_PERM])
    accs = np.array([loo_nearest_centroid_accuracy(X, labels[rng.permutation(
        len(labels))]) for _ in range(n_perm)])
    return {"n_perm": int(n_perm), "mean": float(accs.mean()),
            "min": float(accs.min()), "max": float(accs.max()),
            "p95": float(np.percentile(accs, 95)),
            "p99": float(np.percentile(accs, 99))}


def run_positive_control(out_dir: Path, seed: int = CONTROL_SEED) -> dict:
    """Synthetic positive control for the Level-B J-lens (Director's hard
    condition, 2026-07-09). CPU-only, seconds, no bank/transformers/GPU.
    Writes jlens_positive_control.json and returns the report dict."""
    out_dir = Path(out_dir)
    _pc_guard_out_dir(out_dir)
    chance = 1.0 / PC_N_TASKS
    toy = build_toy_frozen_map(seed)

    # (a) PLANTED PROPAGATION — correct lens must recover the tasks.
    planted_banks, planted_labels = build_planted_bank(toy, seed)
    Xp = control_signatures(planted_banks, toy["lenses"], seed=seed)
    planted_loo = loo_nearest_centroid_accuracy(Xp, planted_labels)

    # (b) MATCHED NULL — identical machinery, random per-run plant.
    mnull_banks, mnull_labels = build_planted_bank(
        toy, seed, randomize_task=True)
    Xn = control_signatures(mnull_banks, toy["lenses"], seed=seed)
    matched_null_loo = loo_nearest_centroid_accuracy(Xn, mnull_labels)
    perm = _pc_permutation_null(Xp, planted_labels, seed)

    # (c) OUTPUT-NULL planted update — lens must read numerically nothing,
    #     while the RAW update (identity lens) still carries the task.
    onull_banks, onull_labels = build_output_null_bank(toy, seed)
    Xo = control_signatures(onull_banks, toy["lenses"], seed=seed)
    Xo_raw = control_signatures(onull_banks, toy["raw_lenses"], seed=seed)
    onull_lensed_loo = loo_nearest_centroid_accuracy(Xo, onull_labels)
    onull_raw_loo = loo_nearest_centroid_accuracy(Xo_raw, onull_labels)
    planted_norm = float(np.linalg.norm(Xp, axis=1).mean())
    onull_norm = float(np.linalg.norm(Xo, axis=1).mean())
    residual_ratio = onull_norm / planted_norm
    onull_maxabs = float(np.abs(Xo).max())

    # NON-TAUTOLOGY — a WRONG frozen map's lens must FAIL to recover the
    # very same planted bank (the confound it cannot null swamps it).
    toy_wrong = build_toy_frozen_map(seed + PC_SABOTAGE_SEED_OFFSET)
    Xsab = control_signatures(planted_banks, toy_wrong["lenses"], seed=seed)
    sabotage_loo = loo_nearest_centroid_accuracy(Xsab, planted_labels)

    planted_recovered = bool(
        planted_loo == 1.0
        and planted_loo > matched_null_loo
        and planted_loo > perm["max"]
        and matched_null_loo <= PC_NULL_BAND_HI)
    output_null_reads_null = bool(
        residual_ratio < PC_RESIDUAL_RATIO_MAX
        and onull_maxabs < PC_ONULL_MAXABS_MAX
        and onull_raw_loo == 1.0)
    sabotage_detected = bool(
        sabotage_loo < planted_loo and sabotage_loo <= PC_NULL_BAND_HI)
    passed = bool(planted_recovered and output_null_reads_null
                  and sabotage_detected)

    report = {
        "tool": "scripts/asset1_jacobian_lens.py",
        "stage": "PC (synthetic positive control)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "PASS": passed,
        "purpose": (
            "Director hard condition (docs/DIRECTOR_RULING_JLENS_STAGEB_"
            "2026-07-09.md): validate the Level-B lens->signature pathway "
            "recovers planted mid-network propagation above a matched null "
            "and reads a genuinely output-null planted update as null — "
            "Level-A-selftest class, so a null Stage-C result is "
            "interpretable as 'no structure' rather than 'lens too weak'."),
        "design": {
            "toy_frozen_map": (
                "per module J_toy in R^{V_toy x d_out}, V_toy < d_out so "
                "null(J_toy) has dim d_out - V_toy; production lens = "
                "S^T J_toy (exact Stage-B shape); raw readout = identity "
                "lens (Delta @ X, no propagation)."),
            "planted_update": (
                "Delta = g (J_pinv u_k) w^T [visible, carried by J_toy to "
                "output dir u_k] + PC_VIS_NOISE (J_pinv E) [visible spread] "
                "+ PC_CONF_SCALE (N Z) [OUTPUT-NULL confound: annihilated "
                "by the correct lens, swamps a wrong lens — the "
                "non-tautology mechanism]."),
            "output_null_update": (
                "Delta = g (N c_k) w^T + PC_ONULL_NOISE (N Z): column space "
                "entirely in null(J_toy); lens output numerically 0, raw "
                "update still carries c_k."),
            "matched_null": ("identical construction, per-run RANDOM plant "
                             "(no task->direction association)."),
            "statistic": ("loo_nearest_centroid_accuracy on L2-normalized "
                          "signatures — Level A's own selftest metric "
                          "(imported, not re-implemented)."),
            "shared_probes_sketch": (
                "probe_inputs (stream 71) via jlens_signature_for_modules; "
                "vocab_sketch (stream 72) builds every lens — condition (d) "
                "of the ruling, import-shared with Level A."),
        },
        "pinned": {
            "control_seed": int(seed), "d_model": PC_D_MODEL,
            "d_in": PC_D_IN, "V_toy": PC_V_TOY, "n_tasks": PC_N_TASKS,
            "n_reps": PC_N_REPS, "n_runs_per_bank": PC_N_TASKS * PC_N_REPS,
            "chance": chance, "modules": list(PC_MODULES),
            "conf_scale": PC_CONF_SCALE, "vis_noise": PC_VIS_NOISE,
            "onull_noise": PC_ONULL_NOISE, "gain_lo": PC_GAIN_LO,
            "n_perm": PC_N_PERM, "sketch_dim": SKETCH_DIM,
            "n_probes": N_PROBES,
            "sabotage_seed": int(seed + PC_SABOTAGE_SEED_OFFSET),
            "stream_tags": {
                "J": _PC_STREAM_J, "u": _PC_STREAM_U, "w": _PC_STREAM_W,
                "gain": _PC_STREAM_GAIN, "conf": _PC_STREAM_CONF,
                "vis_noise": _PC_STREAM_VISN, "c_null": _PC_STREAM_CNULL,
                "onull_noise": _PC_STREAM_ONULLN, "rand_u": _PC_STREAM_RANDU,
                "perm": _PC_STREAM_PERM, "probe": 71, "sketch": 72},
            "thresholds": {
                "planted_loo_required": 1.0,
                "residual_ratio_max": PC_RESIDUAL_RATIO_MAX,
                "onull_maxabs_max": PC_ONULL_MAXABS_MAX,
                "null_band_hi": PC_NULL_BAND_HI,
                "onull_raw_loo_required": 1.0},
        },
        "toy_map_geometry": toy["geometry"],
        "results": {
            "planted": {
                "loo": planted_loo, "signature_dim": int(Xp.shape[1]),
                "mean_signature_norm": planted_norm},
            "matched_null": {"loo": matched_null_loo},
            "permutation_null": perm,
            "output_null": {
                "lensed_loo": onull_lensed_loo,
                "lensed_loo_note": (
                    "computed on numerically-zero vectors (maxabs "
                    f"{onull_maxabs:.2e}); not operationally meaningful — "
                    "'reads null' is evidenced by the norm, not this LOO."),
                "raw_loo": onull_raw_loo,
                "mean_lensed_norm": onull_norm,
                "residual_ratio": residual_ratio,
                "lensed_maxabs": onull_maxabs},
            "sabotage": {
                "wrong_lens_loo": sabotage_loo,
                "note": ("planted bank signed with a DIFFERENT toy map's "
                         "lens; recovery must collapse toward chance — "
                         "proof the control is not a tautology.")},
            "chance": chance,
        },
        "checks": {
            "planted_recovered": planted_recovered,
            "output_null_reads_null": output_null_reads_null,
            "sabotage_detected": sabotage_detected,
        },
        "stageb_ruling": STAGEB_RULING,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "jlens_positive_control.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    verdict = "PASS" if passed else "FAIL"
    print(f"[jlens] positive control {verdict}: planted_loo={planted_loo} "
          f"(chance {chance}, matched_null {matched_null_loo}, perm_max "
          f"{perm['max']}); output-null residual_ratio={residual_ratio:.2e} "
          f"maxabs={onull_maxabs:.2e} raw_loo={onull_raw_loo}; "
          f"sabotage_loo={sabotage_loo}", flush=True)
    print(f"[jlens] positive control written: {path}", flush=True)
    return report


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
    parser.add_argument("--positive-control", action="store_true",
                        help="Stage PC: synthetic lens positive control "
                             "(CPU/seconds, no bank/GPU/transformers) — "
                             "the Director's hard condition for arbiter "
                             "use; writes jlens_positive_control.json")
    parser.add_argument("--control-seed", type=int, default=CONTROL_SEED,
                        help=f"pinned seed for --positive-control "
                             f"(default {CONTROL_SEED})")
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

    if args.positive_control:
        run_positive_control(args.out_dir, seed=args.control_seed)
        return

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

    parser.error("choose one of --plan-only / --estimate / --sign / "
                 "--positive-control")


if __name__ == "__main__":
    main()
