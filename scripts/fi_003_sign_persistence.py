#!/usr/bin/env python3
"""FI-003: Sign Persistence Under Unguided Fine-Tuning

Tests whether the corpus-derived sign structure survives when the Steersman
is REMOVED. Takes a converged BD checkpoint (FI-002 P-000 at step 3000)
and continues training with ONLY language modeling loss --- no spectral
regularization, no contrastive loss.

FI-001 KEY FINDING: Signs freeze at 98.2% stability after 1200 steps
during guided training. The question is whether this stability is
maintained by the Steersman (homeostatic maintenance) or whether the
signs are self-sustaining once established (topological memory).

PREDICTION: Signs should remain frozen. The Steersman wrote the topology;
removing it should not erase it. If signs DO rotate, this reveals the
Steersman is continuously maintaining the topology against gradient
pressure --- a homeostatic maintenance function rather than a one-time
programming event.

DESIGN:
  Phase 1 (steps 0-3000): Load converged P-000 adapter, snapshot initial signs
  Phase 2 (steps 3001-6000): Continue training, LM loss only, track signs every 100 steps

METRICS:
  - Sign stability: fraction of cross-planar signs unchanged from step 3000
  - BD metrics: Fiedler, co/cross ratio, eigenvalue pattern
  - Val loss: should continue improving (LM training unimpaired)
  - Bridge magnitude: are bridges growing/shrinking without Steersman?

Usage:
  python scripts/fi_003_sign_persistence.py --source-adapter results/fi-002/P-000/adapter_state.pt
  python scripts/fi_003_sign_persistence.py --analyze-only results/fi-003

TASUMER MAF --- The geometry IS the argument.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rhombic.nn.rhombi_lora import RhombiLoRALinear
from rhombic.nn.topology import direction_pair_coupling

# spectral functions computed inline (rhombic.spectral uses graph-level API)

# Reuse from Exp 2
from train_exp2_scale import (
    ExperimentConfig,
    AlpacaDataset,
    inject_lora,
    collect_metrics,
)

# Reuse Steersman feedback infrastructure (for measurement only, not control)
from train_cybernetic import (
    contrastive_bridge_loss,
    _compute_pair_indices,
)


# -- Sign Analysis Utilities ------------------------------------------------

def extract_cross_planar_signs(
    injected: dict[str, RhombiLoRALinear],
    co_pairs: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Extract sign vectors from cross-planar bridge entries.

    Returns dict mapping layer name to a 1D boolean array where True = positive.
    """
    co_set = set(co_pairs) | {(j, i) for i, j in co_pairs}
    signs = {}
    for name, lora in injected.items():
        bridge = lora.effective_bridge.detach().cpu().numpy()
        n = bridge.shape[0]
        cross_signs = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in co_set:
                    cross_signs.append(bridge[i, j] > 0)
        signs[name] = np.array(cross_signs, dtype=bool)
    return signs


def sign_stability(
    signs_now: dict[str, np.ndarray],
    signs_ref: dict[str, np.ndarray],
) -> dict:
    """Compare current signs to reference. Returns per-layer and aggregate stats."""
    layer_stab = {}
    all_match = 0
    all_total = 0
    for name in signs_ref:
        if name not in signs_now:
            continue
        match = np.sum(signs_now[name] == signs_ref[name])
        total = len(signs_ref[name])
        layer_stab[name] = float(match / total)
        all_match += match
        all_total += total
    return {
        "per_layer": layer_stab,
        "aggregate": float(all_match / all_total) if all_total > 0 else 0.0,
        "total_positions": all_total,
        "positions_changed": all_total - all_match,
    }


def bridge_magnitude_stats(injected: dict[str, RhombiLoRALinear]) -> dict:
    """Compute bridge magnitude statistics across all layers."""
    magnitudes = []
    for lora in injected.values():
        bridge = lora.effective_bridge.detach().cpu().numpy()
        np.fill_diagonal(bridge, 0)
        magnitudes.append(np.abs(bridge).mean())
    return {
        "mean_magnitude": float(np.mean(magnitudes)),
        "std_magnitude": float(np.std(magnitudes)),
        "min_magnitude": float(np.min(magnitudes)),
        "max_magnitude": float(np.max(magnitudes)),
    }


# -- Adapter Loading ---------------------------------------------------------

def load_adapter_state(
    injected: dict[str, RhombiLoRALinear],
    adapter_path: Path,
    device: torch.device,
):
    """Load adapter_state.pt into injected LoRA layers.

    The adapter state dict has keys like:
      model_layers_N_self_attn_PROJ.lora_A
      model_layers_N_self_attn_PROJ.lora_B
      model_layers_N_self_attn_PROJ.bridge
    """
    state = torch.load(adapter_path, map_location=device, weights_only=True)

    loaded = 0
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        key_a = f"{safe}.lora_A"
        key_b = f"{safe}.lora_B"
        key_bridge = f"{safe}.bridge"

        if key_a not in state:
            print(f"  WARNING: {key_a} not found in adapter state")
            continue

        lora.lora_A.data.copy_(state[key_a].to(device))
        lora.lora_B.data.copy_(state[key_b].to(device))
        # Set bridge parameter from saved effective bridge
        lora.bridge.data.copy_(state[key_bridge].to(device))
        loaded += 1

    print(f"  Loaded adapter state into {loaded}/{len(injected)} layers")
    return loaded


# -- Measurement (no control) ------------------------------------------------

def measure_bd_state(
    injected: dict[str, RhombiLoRALinear],
    co_pairs: list[tuple[int, int]],
    cross_pairs: list[tuple[int, int]],
    n_channels: int,
) -> dict:
    """Measure BD metrics without modifying anything."""
    fiedler_vals = []
    co_cross_vals = []
    all_eigs = []

    for name, lora in injected.items():
        bridge = lora.effective_bridge.detach().cpu().numpy()
        n = bridge.shape[0]
        # Build Laplacian from bridge: L = D - |W|
        W = np.abs(bridge.copy())
        np.fill_diagonal(W, 0)
        D = np.diag(W.sum(axis=1))
        L = D - W
        eigs = np.sort(np.linalg.eigvalsh(L))
        fiedler_vals.append(float(eigs[1]))

        # Co/cross ratio
        co_sum = sum(abs(bridge[i, j]) for i, j in co_pairs)
        cross_sum = sum(abs(bridge[i, j]) for i, j in cross_pairs)
        if cross_sum > 0:
            co_cross_vals.append(co_sum / cross_sum)

        all_eigs.append(eigs)

    # Aggregate
    result = {
        "fiedler_mean": float(np.mean(fiedler_vals)),
        "fiedler_std": float(np.std(fiedler_vals)),
        "co_cross_ratio": float(np.mean(co_cross_vals)) if co_cross_vals else None,
    }

    # Representative eigenvalue pattern (from first layer)
    if all_eigs:
        result["eigenvalue_pattern"] = [float(v) for v in all_eigs[0]]

    return result


# -- Main Training -----------------------------------------------------------

def train_unguided(
    source_adapter: Path,
    output_dir: Path,
    max_steps: int = 3000,
    feedback_interval: int = 100,
    checkpoint_steps: int = 100,
    lr: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation: int = 8,
    warmup_steps: int = 200,
    seed: int = 42,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_channels: int = 6,
):
    """Continue training from a converged adapter with NO Steersman."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save config
    config = ExperimentConfig(
        name="fi_003_sign_persistence",
        rank=24,
        n_channels=n_channels,
        bridge_mode="corpus_coupled",  # Starting from corpus-coupled state
        bridge_trainable=True,
        model_name=model_name,
        max_steps=max_steps,
        lr=lr,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        warmup_steps=warmup_steps,
        checkpoint_steps=checkpoint_steps,
        seed=seed,
    )

    config_dict = asdict(config)
    config_dict["experiment"] = "FI-003"
    config_dict["source_adapter"] = str(source_adapter)
    config_dict["steersman_enabled"] = False
    config_dict["contrastive_weight"] = 0.0
    config_dict["spectral_weight"] = 0.0
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Geometry
    co_pairs, cross_pairs = _compute_pair_indices(n_channels)
    print(f"FI-003: Sign Persistence Under Unguided Fine-Tuning")
    print(f"=" * 60)
    print(f"Source adapter: {source_adapter}")
    print(f"Model: {model_name}")
    print(f"Steps: {max_steps} (LM loss only, NO Steersman)")
    print(f"Co-planar pairs: {len(co_pairs)}")
    print(f"Cross-planar pairs: {len(cross_pairs)}")
    print()

    # Load model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)

    # Inject LoRA (with identity bridge -- will be overwritten)
    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    # Load the converged adapter state
    print(f"Loading converged adapter from {source_adapter}...")
    n_loaded = load_adapter_state(injected, source_adapter, device)
    if n_loaded == 0:
        print("ERROR: No adapter layers loaded. Aborting.")
        sys.exit(1)

    # Snapshot initial signs (the reference from BD convergence)
    initial_signs = extract_cross_planar_signs(injected, co_pairs)
    initial_bd = measure_bd_state(injected, co_pairs, cross_pairs, n_channels)
    initial_magnitudes = bridge_magnitude_stats(injected)

    print(f"\nInitial state (from converged adapter):")
    print(f"  Fiedler: {initial_bd['fiedler_mean']:.6f}")
    print(f"  Co/Cross: {initial_bd['co_cross_ratio']:.0f}:1" if initial_bd['co_cross_ratio'] else "  Co/Cross: N/A")
    print(f"  Mean bridge magnitude: {initial_magnitudes['mean_magnitude']:.6f}")
    n_pos = sum(s.sum() for s in initial_signs.values())
    n_total = sum(len(s) for s in initial_signs.values())
    print(f"  Cross-planar sign balance: {n_pos}/{n_total} positive ({100*n_pos/n_total:.1f}%)")

    # Save initial snapshot
    np.savez(
        output_dir / "initial_signs.npz",
        **{k: v for k, v in initial_signs.items()},
    )
    with open(output_dir / "initial_state.json", "w") as f:
        json.dump({
            "bd_metrics": initial_bd,
            "magnitudes": initial_magnitudes,
            "sign_positive_frac": float(n_pos / n_total),
        }, f, indent=2, default=str)

    # Optimizer -- ALL params (LoRA + bridge) get gradients from LM loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "lr": lr, "name": "all"}],
        weight_decay=0.01,
    )

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_schedule])

    # Dataset
    print("\nLoading Alpaca-cleaned dataset...")
    dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len)
    val_dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len, max_samples=1000)
    print(f"Training examples: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Training state
    checkpoints = []
    sign_evolution = []
    model.train()
    global_step = 0
    accumulation_loss = 0.0
    start_time = time.time()
    done = False

    print(f"\n{'=' * 60}")
    print(f"TRAINING: LM loss only, NO Steersman, NO contrastive, NO spectral")
    print(f"{'=' * 60}\n")

    for epoch in range(1000):
        if done:
            break
        for batch_idx, batch in enumerate(dataloader):
            if done:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass -- ONLY language modeling loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss / gradient_accumulation
            lm_loss.backward()
            accumulation_loss += lm_loss.item()

            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # -- Feedback interval: measure (but never control) --
                if global_step % feedback_interval == 0:
                    wall_time = time.time() - start_time

                    # Measure BD state
                    bd = measure_bd_state(injected, co_pairs, cross_pairs, n_channels)

                    # Measure sign stability
                    current_signs = extract_cross_planar_signs(injected, co_pairs)
                    stability = sign_stability(current_signs, initial_signs)
                    magnitudes = bridge_magnitude_stats(injected)

                    # Validation loss
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vi, vb in enumerate(val_dataloader):
                            if vi >= 50:
                                break
                            vo = model(
                                input_ids=vb["input_ids"].to(device),
                                attention_mask=vb["attention_mask"].to(device),
                                labels=vb["labels"].to(device),
                            )
                            val_losses.append(vo.loss.item())
                    val_loss = float(np.mean(val_losses))
                    model.train()

                    avg_loss = accumulation_loss / feedback_interval
                    accumulation_loss = 0.0

                    entry = {
                        "step": global_step,
                        "wall_time": wall_time,
                        "train_loss": avg_loss,
                        "val_loss": val_loss,
                        "fiedler_mean": bd["fiedler_mean"],
                        "co_cross_ratio": bd["co_cross_ratio"],
                        "eigenvalue_pattern": bd.get("eigenvalue_pattern", []),
                        "sign_stability": stability["aggregate"],
                        "positions_changed": stability["positions_changed"],
                        "total_positions": stability["total_positions"],
                        "mean_magnitude": magnitudes["mean_magnitude"],
                    }
                    checkpoints.append(entry)
                    sign_evolution.append({
                        "step": global_step,
                        "aggregate_stability": stability["aggregate"],
                        "per_layer": stability["per_layer"],
                        "positions_changed": stability["positions_changed"],
                    })

                    # Print status
                    co_str = f"{bd['co_cross_ratio']:,.0f}:1" if bd['co_cross_ratio'] else "N/A"
                    print(
                        f"Step {global_step:>5d} | "
                        f"LM={avg_loss:.4f} Val={val_loss:.4f} | "
                        f"Fiedler={bd['fiedler_mean']:.6f} Co/Cross={co_str} | "
                        f"Sign={stability['aggregate']:.3f} "
                        f"({stability['positions_changed']} changed) | "
                        f"Mag={magnitudes['mean_magnitude']:.6f}"
                    )

                    # Save results.json
                    results = {
                        "experiment": "FI-003",
                        "config": config_dict,
                        "initial_state": {
                            "bd_metrics": initial_bd,
                            "magnitudes": initial_magnitudes,
                            "sign_positive_frac": float(n_pos / n_total),
                        },
                        "checkpoints": checkpoints,
                        "sign_evolution": sign_evolution,
                    }
                    with open(output_dir / "results.json", "w") as f:
                        json.dump(results, f, indent=2, default=str)

                # -- Checkpoint interval: save bridge state --
                if global_step % checkpoint_steps == 0:
                    for name, lora in injected.items():
                        safe = name.replace(".", "_")
                        np.save(
                            output_dir / f"bridge_step{global_step}_{safe}.npy",
                            lora.effective_bridge.detach().cpu().numpy(),
                        )

                if global_step >= max_steps:
                    done = True
                    break

    # Final measurements
    wall_time = time.time() - start_time
    final_signs = extract_cross_planar_signs(injected, co_pairs)
    final_stability = sign_stability(final_signs, initial_signs)
    final_bd = measure_bd_state(injected, co_pairs, cross_pairs, n_channels)
    final_magnitudes = bridge_magnitude_stats(injected)

    print(f"\n{'=' * 60}")
    print(f"FI-003 COMPLETE — {global_step} steps in {wall_time/3600:.2f}h")
    print(f"{'=' * 60}")
    print(f"\n  SIGN PERSISTENCE: {final_stability['aggregate']:.4f}")
    print(f"    ({final_stability['positions_changed']}/{final_stability['total_positions']} positions changed)")
    print(f"\n  Initial Fiedler: {initial_bd['fiedler_mean']:.6f}")
    print(f"  Final Fiedler:   {final_bd['fiedler_mean']:.6f}")
    co_init = f"{initial_bd['co_cross_ratio']:,.0f}:1" if initial_bd['co_cross_ratio'] else "N/A"
    co_final = f"{final_bd['co_cross_ratio']:,.0f}:1" if final_bd['co_cross_ratio'] else "N/A"
    print(f"  Initial Co/Cross: {co_init}")
    print(f"  Final Co/Cross:   {co_final}")
    print(f"\n  Initial magnitude: {initial_magnitudes['mean_magnitude']:.6f}")
    print(f"  Final magnitude:   {final_magnitudes['mean_magnitude']:.6f}")

    # Verdict
    if final_stability["aggregate"] > 0.95:
        print(f"\n  VERDICT: TOPOLOGICAL MEMORY — signs self-sustaining without Steersman")
    elif final_stability["aggregate"] > 0.80:
        print(f"\n  VERDICT: PARTIAL PERSISTENCE — signs slowly drifting without Steersman")
    elif final_stability["aggregate"] > 0.60:
        print(f"\n  VERDICT: SIGNIFICANT DRIFT — Steersman actively maintains topology")
    else:
        print(f"\n  VERDICT: COLLAPSE — topology is Steersman-dependent")

    # Save final adapter state
    adapter_state = {}
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        adapter_state[f"{safe}.lora_A"] = lora.lora_A.detach().cpu()
        adapter_state[f"{safe}.lora_B"] = lora.lora_B.detach().cpu()
        adapter_state[f"{safe}.bridge"] = lora.effective_bridge.detach().cpu()
        adapter_state[f"{safe}.scaling"] = torch.tensor(lora.scaling)
        adapter_state[f"{safe}.n_channels"] = torch.tensor(lora.n_channels)
        adapter_state[f"{safe}.rank"] = torch.tensor(lora.rank)
    torch.save(adapter_state, output_dir / "adapter_state_final.pt")

    # Save final results
    results = {
        "experiment": "FI-003",
        "config": config_dict,
        "initial_state": {
            "bd_metrics": initial_bd,
            "magnitudes": initial_magnitudes,
            "sign_positive_frac": float(n_pos / n_total),
        },
        "final_state": {
            "bd_metrics": final_bd,
            "magnitudes": final_magnitudes,
            "sign_stability": final_stability["aggregate"],
            "positions_changed": final_stability["positions_changed"],
        },
        "checkpoints": checkpoints,
        "sign_evolution": sign_evolution,
        "wall_time": wall_time,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_dir}/results.json")


# -- Analysis Mode -----------------------------------------------------------

def analyze_results(results_dir: Path):
    """Analyze FI-003 results from disk."""
    with open(results_dir / "results.json") as f:
        data = json.load(f)

    print(f"{'=' * 60}")
    print(f"FI-003: Sign Persistence Analysis")
    print(f"{'=' * 60}")

    init = data.get("initial_state", {})
    final = data.get("final_state", {})
    checkpoints = data.get("checkpoints", [])
    sign_evo = data.get("sign_evolution", [])

    print(f"\nInitial state:")
    init_bd = init.get("bd_metrics", {})
    print(f"  Fiedler: {init_bd.get('fiedler_mean', 0):.6f}")
    co_init = init_bd.get("co_cross_ratio")
    print(f"  Co/Cross: {co_init:,.0f}:1" if co_init else "  Co/Cross: N/A")

    print(f"\nSign stability over time:")
    print(f"  {'Step':>6} {'Stability':>10} {'Changed':>8} {'Fiedler':>10} {'Co/Cross':>12}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*12}")
    for chk in checkpoints:
        co = chk.get("co_cross_ratio")
        co_str = f"{co:,.0f}:1" if co else "N/A"
        print(
            f"  {chk['step']:>6} "
            f"{chk.get('sign_stability', 0):>10.4f} "
            f"{chk.get('positions_changed', 0):>8} "
            f"{chk.get('fiedler_mean', 0):>10.6f} "
            f"{co_str:>12}"
        )

    if final:
        print(f"\nFinal state:")
        print(f"  Sign stability: {final.get('sign_stability', 0):.4f}")
        print(f"  Positions changed: {final.get('positions_changed', 0)}")
        final_bd = final.get("bd_metrics", {})
        print(f"  Fiedler: {final_bd.get('fiedler_mean', 0):.6f}")

    print(f"\n{'=' * 60}")


# -- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FI-003: Sign Persistence Under Unguided Fine-Tuning"
    )
    parser.add_argument(
        "--source-adapter", type=str, default=None,
        help="Path to adapter_state.pt from FI-002 P-000",
    )
    parser.add_argument(
        "--output", type=str, default="results/fi-003",
        help="Output directory",
    )
    parser.add_argument(
        "--max-steps", type=int, default=3000,
        help="Steps of unguided training",
    )
    parser.add_argument(
        "--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument(
        "--n-channels", type=int, default=6,
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=8,
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=200,
    )
    parser.add_argument(
        "--feedback-interval", type=int, default=100,
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, default=100,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--analyze-only", type=str, default=None,
        help="Path to results directory for analysis only",
    )
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(Path(args.analyze_only))
        return

    if args.source_adapter is None:
        # Default: look for FI-002 P-000 adapter
        default_path = Path("results/fi-002/P-000/adapter_state.pt")
        if default_path.exists():
            args.source_adapter = str(default_path)
            print(f"Using default source adapter: {default_path}")
        else:
            print("ERROR: No --source-adapter specified and no default found at")
            print(f"  {default_path}")
            print("Run FI-002 first, then provide the adapter path.")
            sys.exit(1)

    train_unguided(
        source_adapter=Path(args.source_adapter),
        output_dir=Path(args.output),
        max_steps=args.max_steps,
        model_name=args.model,
        n_channels=args.n_channels,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        feedback_interval=args.feedback_interval,
        checkpoint_steps=args.checkpoint_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
