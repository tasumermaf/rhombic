#!/usr/bin/env python3
"""FI-004: Steersman Annealing — Mapping the Maintenance Boundary

Tests at what contrastive weight the Steersman can no longer maintain
block-diagonal topology. Takes a converged BD checkpoint and continues
training with LINEARLY ANNEALING contrastive/spectral weights.

DEPENDS ON FI-003:
  - If FI-003 shows full persistence (>95% sign stability): FI-004
    becomes less important but maps the force field for completeness.
  - If FI-003 shows decay: FI-004 identifies the critical threshold
    below which topology collapses.

DESIGN:
  Load converged P-000 adapter, continue training 3000 steps.
  Contrastive weight: 0.1 -> 0.0 (linear anneal over 3000 steps)
  Spectral weight: 0.05 -> 0.0 (linear anneal over 3000 steps)
  Track BD metrics, sign stability, and identify the critical step
  where topology begins to degrade.

METRICS:
  - Sign stability vs. step (should be monotonic decline if decay occurs)
  - BD metrics at each checkpoint
  - Critical step: first checkpoint where sign stability < 95%
  - Critical weight: the contrastive weight at that critical step

Usage:
  python scripts/fi_004_steersman_annealing.py --source-adapter results/fi-002/P-000/adapter_state.pt
  python scripts/fi_004_steersman_annealing.py --analyze-only results/fi-004

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
from rhombic.spectral import (
    weighted_laplacian,
    fiedler_value,
    spectral_gap,
    spectrum,
    eigenvalue_multiplicity_pattern,
)

from train_exp2_scale import (
    ExperimentConfig,
    AlpacaDataset,
    inject_lora,
    collect_metrics,
)

from train_cybernetic import (
    contrastive_bridge_loss,
    spectral_reg_loss,
    _compute_pair_indices,
)

# Reuse FI-003 utilities
from fi_003_sign_persistence import (
    extract_cross_planar_signs,
    sign_stability,
    bridge_magnitude_stats,
    load_adapter_state,
    measure_bd_state,
)


def train_annealing(
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
    initial_contrastive: float = 0.1,
    initial_spectral: float = 0.05,
):
    """Continue training from converged adapter with linearly annealing Steersman."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save config
    config = ExperimentConfig(
        name="fi_004_steersman_annealing",
        rank=24,
        n_channels=n_channels,
        bridge_mode="corpus_coupled",
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
    config_dict["experiment"] = "FI-004"
    config_dict["source_adapter"] = str(source_adapter)
    config_dict["steersman_enabled"] = True
    config_dict["anneal_mode"] = "linear_to_zero"
    config_dict["initial_contrastive"] = initial_contrastive
    config_dict["initial_spectral"] = initial_spectral
    config_dict["final_contrastive"] = 0.0
    config_dict["final_spectral"] = 0.0
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Geometry
    co_pairs, cross_pairs = _compute_pair_indices(n_channels)
    print(f"FI-004: Steersman Annealing — Mapping the Maintenance Boundary")
    print(f"=" * 60)
    print(f"Source adapter: {source_adapter}")
    print(f"Model: {model_name}")
    print(f"Steps: {max_steps}")
    print(f"Contrastive: {initial_contrastive} -> 0.0 (linear)")
    print(f"Spectral: {initial_spectral} -> 0.0 (linear)")
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

    # Inject LoRA
    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    # Load converged adapter
    print(f"Loading converged adapter from {source_adapter}...")
    n_loaded = load_adapter_state(injected, source_adapter, device)
    if n_loaded == 0:
        print("ERROR: No adapter layers loaded. Aborting.")
        sys.exit(1)

    # Snapshot initial state
    initial_signs = extract_cross_planar_signs(injected, co_pairs)
    initial_bd = measure_bd_state(injected, co_pairs, cross_pairs, n_channels)
    initial_magnitudes = bridge_magnitude_stats(injected)

    print(f"\nInitial state (from converged adapter):")
    print(f"  Fiedler: {initial_bd['fiedler_mean']:.6f}")
    co_init_str = f"{initial_bd['co_cross_ratio']:,.0f}:1" if initial_bd['co_cross_ratio'] else "N/A"
    print(f"  Co/Cross: {co_init_str}")
    print(f"  Mean bridge magnitude: {initial_magnitudes['mean_magnitude']:.6f}")

    # Save initial snapshot
    np.savez(output_dir / "initial_signs.npz", **{k: v for k, v in initial_signs.items()})

    # Pair spec for contrastive loss
    pair_spec = {
        "co_pairs": co_pairs,
        "cross_pairs": cross_pairs,
    }

    # Optimizer
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
    critical_step = None  # First step where sign stability < 0.95

    print(f"\n{'=' * 60}")
    print(f"TRAINING: Steersman annealing {initial_contrastive} -> 0.0 over {max_steps} steps")
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

            # Compute annealing weights
            anneal_frac = min(global_step / max_steps, 1.0)
            current_contrastive = initial_contrastive * (1.0 - anneal_frac)
            current_spectral = initial_spectral * (1.0 - anneal_frac)

            # Forward pass — LM loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss = outputs.loss / gradient_accumulation

            # Add Steersman losses (with annealed weights)
            if current_contrastive > 1e-8:
                c_loss = contrastive_bridge_loss(injected, co_pairs, cross_pairs)
                total_loss = total_loss + (current_contrastive * c_loss) / gradient_accumulation

            if current_spectral > 1e-8:
                s_loss = spectral_reg_loss(injected)
                total_loss = total_loss + (current_spectral * s_loss) / gradient_accumulation

            total_loss.backward()
            accumulation_loss += (outputs.loss.item() / gradient_accumulation)

            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Feedback interval
                if global_step % feedback_interval == 0:
                    wall_time = time.time() - start_time

                    # Current anneal state
                    anneal_frac = min(global_step / max_steps, 1.0)
                    cw = initial_contrastive * (1.0 - anneal_frac)
                    sw = initial_spectral * (1.0 - anneal_frac)

                    # Measure BD state
                    bd = measure_bd_state(injected, co_pairs, cross_pairs, n_channels)

                    # Measure signs
                    current_signs = extract_cross_planar_signs(injected, co_pairs)
                    stability = sign_stability(current_signs, initial_signs)
                    magnitudes = bridge_magnitude_stats(injected)

                    # Check for critical step
                    if critical_step is None and stability["aggregate"] < 0.95:
                        critical_step = global_step
                        critical_weight = cw
                        print(f"\n  *** CRITICAL STEP {global_step}: "
                              f"sign stability dropped below 95% "
                              f"(contrastive weight = {cw:.4f}) ***\n")

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
                        "contrastive_weight": cw,
                        "spectral_weight": sw,
                    }
                    checkpoints.append(entry)
                    sign_evolution.append({
                        "step": global_step,
                        "aggregate_stability": stability["aggregate"],
                        "contrastive_weight": cw,
                    })

                    co_str = f"{bd['co_cross_ratio']:,.0f}:1" if bd['co_cross_ratio'] else "N/A"
                    print(
                        f"Step {global_step:>5d} | "
                        f"c_w={cw:.4f} s_w={sw:.4f} | "
                        f"LM={avg_loss:.4f} Val={val_loss:.4f} | "
                        f"Fiedler={bd['fiedler_mean']:.6f} Co/Cross={co_str} | "
                        f"Sign={stability['aggregate']:.3f}"
                    )

                    # Save results
                    results = {
                        "experiment": "FI-004",
                        "config": config_dict,
                        "initial_state": {
                            "bd_metrics": initial_bd,
                            "magnitudes": initial_magnitudes,
                        },
                        "checkpoints": checkpoints,
                        "sign_evolution": sign_evolution,
                        "critical_step": critical_step,
                    }
                    with open(output_dir / "results.json", "w") as f:
                        json.dump(results, f, indent=2, default=str)

                # Checkpoint: save bridges
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
    print(f"FI-004 COMPLETE — {global_step} steps in {wall_time/3600:.2f}h")
    print(f"{'=' * 60}")
    print(f"\n  SIGN STABILITY: {final_stability['aggregate']:.4f}")
    print(f"    ({final_stability['positions_changed']}/{final_stability['total_positions']} changed)")
    if critical_step:
        print(f"  CRITICAL STEP: {critical_step} (contrastive weight = {critical_weight:.4f})")
    else:
        print(f"  NO CRITICAL STEP: topology persisted through full anneal")
    print(f"\n  Initial Fiedler: {initial_bd['fiedler_mean']:.6f}")
    print(f"  Final Fiedler:   {final_bd['fiedler_mean']:.6f}")
    co_fin_str = f"{final_bd['co_cross_ratio']:,.0f}:1" if final_bd['co_cross_ratio'] else "N/A"
    print(f"  Initial Co/Cross: {co_init_str}")
    print(f"  Final Co/Cross:   {co_fin_str}")

    # Save final results
    results = {
        "experiment": "FI-004",
        "config": config_dict,
        "initial_state": {
            "bd_metrics": initial_bd,
            "magnitudes": initial_magnitudes,
        },
        "final_state": {
            "bd_metrics": final_bd,
            "magnitudes": final_magnitudes,
            "sign_stability": final_stability["aggregate"],
        },
        "checkpoints": checkpoints,
        "sign_evolution": sign_evolution,
        "critical_step": critical_step,
        "wall_time": wall_time,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save final adapter
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
    print(f"\n  Results saved to {output_dir}")


def analyze_results(results_dir: Path):
    """Analyze FI-004 results."""
    with open(results_dir / "results.json") as f:
        data = json.load(f)

    print(f"FI-004 Analysis: Steersman Annealing")
    print(f"=" * 60)

    cps = data.get("checkpoints", [])
    if not cps:
        print("No checkpoints found.")
        return

    print(f"\nAnnealing trajectory ({len(cps)} checkpoints):")
    print(f"{'Step':>6} {'c_weight':>8} {'Fiedler':>10} {'Co/Cross':>12} {'Sign%':>8} {'Val Loss':>10}")
    print(f"{'-'*60}")
    for cp in cps:
        co = f"{cp['co_cross_ratio']:,.0f}" if cp.get('co_cross_ratio') else "N/A"
        print(f"{cp['step']:>6d} {cp.get('contrastive_weight',0):>8.4f} "
              f"{cp['fiedler_mean']:>10.6f} {co:>12} "
              f"{cp.get('sign_stability',0)*100:>7.1f}% {cp['val_loss']:>10.4f}")

    critical = data.get("critical_step")
    if critical:
        # Find the weight at critical step
        for cp in cps:
            if cp["step"] == critical:
                print(f"\nCRITICAL STEP: {critical}")
                print(f"  Contrastive weight at failure: {cp.get('contrastive_weight', 'N/A')}")
                print(f"  Sign stability at failure: {cp.get('sign_stability', 'N/A'):.3f}")
                break
    else:
        print(f"\nNO CRITICAL STEP — topology persisted through complete anneal")

    # Final state
    final = data.get("final_state", {})
    if final:
        print(f"\nFinal state:")
        bd = final.get("bd_metrics", {})
        print(f"  Fiedler: {bd.get('fiedler_mean', 'N/A')}")
        co = bd.get("co_cross_ratio")
        print(f"  Co/Cross: {co:,.0f}:1" if co else "  Co/Cross: N/A")
        print(f"  Sign stability: {final.get('sign_stability', 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FI-004: Steersman Annealing")
    parser.add_argument("--source-adapter", type=str,
                        help="Path to converged adapter (e.g., results/fi-002/P-000/adapter_state.pt)")
    parser.add_argument("--output-dir", type=str, default="results/fi-004",
                        help="Output directory")
    parser.add_argument("--analyze-only", type=str, default=None,
                        help="Path to results directory for analysis mode")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--initial-contrastive", type=float, default=0.1)
    parser.add_argument("--initial-spectral", type=float, default=0.05)

    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(Path(args.analyze_only))
    else:
        if not args.source_adapter:
            print("ERROR: --source-adapter required for training mode")
            sys.exit(1)
        train_annealing(
            source_adapter=Path(args.source_adapter),
            output_dir=Path(args.output_dir),
            max_steps=args.max_steps,
            initial_contrastive=args.initial_contrastive,
            initial_spectral=args.initial_spectral,
        )
