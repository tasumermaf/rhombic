#!/usr/bin/env python3
"""FI-005: Topology Reprogramming

Tests whether the Steersman can REPROGRAM an existing topology.

DESIGN:
  Phase 1 (steps 0-3000): Load converged P-000 adapter (RD, n=6, 3+3 BD at ~50K co/cross)
  Phase 2 (steps 3001-6000): Switch pair specification, continue training WITH Steersman

The pair spec switch changes which channels are co-axial. The bridge matrices
remain 6×6 — only the Steersman's contrastive feedback changes.

TARGET TOPOLOGIES:
  'reduced': 2 co-axial pairs (channels 0-5, 1-3) instead of 3. Target: 2+4 split.
  'rotated': 3 co-axial pairs but DIFFERENT channels (0-1, 2-3, 4-5). Target: 3+3 split
             but along different axes — tests whether structure transfers.
  'random': random 3-pair partition (WL-001-like). Tests transition to collapse.

KEY QUESTION: Can a bridge that has converged to one topology be reprogrammed
to another? How fast? Does it pass through a collapsed intermediate state?

PREDICTIONS:
  reduced: Bridge transitions from 3+3 to 2+4. The pair that loses co-axial
           status should see its eigenvalue rise from ~0 toward the cross-axial
           cluster. Speed of transition reveals the Steersman's force.
  rotated: Most interesting. If BD transfers across channel reassignment, the
           topology is more abstract than channel-specific. If it collapses
           first, the structure is channel-indexed.
  random:  Bridge should collapse (wrong-labels result). Tests that
           reprogramming to an invalid spec fails predictably.

Usage:
  python scripts/fi_005_reprogramming.py --target reduced
  python scripts/fi_005_reprogramming.py --target rotated
  python scripts/fi_005_reprogramming.py --analyze-only results/fi-005/reduced

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

# Reuse from Exp 2
from train_exp2_scale import (
    ExperimentConfig,
    AlpacaDataset,
    inject_lora,
    collect_metrics,
)

# Reuse Steersman infrastructure
from train_cybernetic import (
    contrastive_bridge_loss,
    spectral_bridge_loss,
    _compute_pair_indices,
    steersman_feedback,
)

# Reuse from FI-003
from fi_003_sign_persistence import (
    extract_cross_planar_signs,
    sign_stability,
    bridge_magnitude_stats,
    load_adapter_state,
    measure_bd_state,
)


# -- Target Topologies ---------------------------------------------------------

def compute_rd_pairs() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Standard RD co-planar pairs for n=6."""
    return _compute_pair_indices(6)


def compute_reduced_pairs() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Reduced topology: 2 co-axial pairs from the original 3.

    Keeps the first 2 RD co-planar pairs, demotes the 3rd to cross-planar.
    Target: 2+4 eigenvalue split (2 near-zero, 4 elevated).
    """
    rd_co, _ = compute_rd_pairs()
    # Keep first 2 pairs as co-axial
    new_co = rd_co[:2]
    # Everything else is cross-axial
    new_co_set = set(new_co) | {(j, i) for i, j in new_co}
    new_cross = [
        (i, j) for i in range(6) for j in range(i + 1, 6)
        if (i, j) not in new_co_set and (j, i) not in new_co_set
    ]
    return new_co, new_cross


def compute_rotated_pairs() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Rotated topology: 3 co-axial pairs with DIFFERENT channel pairing.

    Instead of RD geometry pairs, use sequential pairing: (0,1), (2,3), (4,5).
    Same count, different topology. Tests channel-specificity of programmed structure.
    """
    new_co = [(0, 1), (2, 3), (4, 5)]
    new_co_set = set(new_co)
    new_cross = [
        (i, j) for i in range(6) for j in range(i + 1, 6)
        if (i, j) not in new_co_set
    ]
    return new_co, new_cross


def compute_random_pairs(seed: int = 99) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Random 3-pair partition (WL-001-like). Should cause collapse."""
    rng = np.random.RandomState(seed)
    channels = list(range(6))
    rng.shuffle(channels)
    new_co = [(channels[0], channels[1]), (channels[2], channels[3]), (channels[4], channels[5])]
    new_co_set = set(new_co) | {(j, i) for i, j in new_co}
    new_cross = [
        (i, j) for i in range(6) for j in range(i + 1, 6)
        if (i, j) not in new_co_set and (j, i) not in new_co_set
    ]
    return new_co, new_cross


TARGET_FUNCTIONS = {
    "reduced": compute_reduced_pairs,
    "rotated": compute_rotated_pairs,
    "random": compute_random_pairs,
}


# -- Main Training -----------------------------------------------------------

def train_reprogramming(
    source_adapter: Path,
    output_dir: Path,
    target: str,
    phase2_steps: int = 3000,
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
    """Reprogram a converged BD adapter to a new topology."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute pair specifications
    old_co, old_cross = compute_rd_pairs()
    new_co, new_cross = TARGET_FUNCTIONS[target]()

    # Save config
    config = ExperimentConfig(
        name=f"fi_005_reprogramming_{target}",
        rank=24,
        n_channels=n_channels,
        bridge_mode="corpus_coupled",
        bridge_trainable=True,
        model_name=model_name,
        max_steps=phase2_steps,
        lr=lr,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        warmup_steps=warmup_steps,
        checkpoint_steps=checkpoint_steps,
        seed=seed,
    )

    config_dict = asdict(config)
    config_dict["experiment"] = "FI-005"
    config_dict["target_topology"] = target
    config_dict["source_adapter"] = str(source_adapter)
    config_dict["initial_contrastive"] = initial_contrastive
    config_dict["initial_spectral"] = initial_spectral
    config_dict["old_co_pairs"] = old_co
    config_dict["new_co_pairs"] = new_co
    config_dict["old_cross_pairs"] = old_cross
    config_dict["new_cross_pairs"] = new_cross

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"FI-005: Topology Reprogramming ({target})")
    print(f"{'=' * 60}")
    print(f"Source adapter: {source_adapter}")
    print(f"Target topology: {target}")
    print(f"Old co-axial pairs: {old_co}")
    print(f"New co-axial pairs: {new_co}")
    print(f"Phase 2 steps: {phase2_steps}")
    print(f"Steersman: ACTIVE (contrastive={initial_contrastive}, spectral={initial_spectral})")
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

    # Load converged adapter state
    print(f"Loading converged adapter from {source_adapter}...")
    n_loaded = load_adapter_state(injected, source_adapter, device)
    if n_loaded == 0:
        print("ERROR: No adapter layers loaded. Aborting.")
        sys.exit(1)

    # Measure initial state (with OLD pair spec)
    initial_bd_old = measure_bd_state(injected, old_co, old_cross, n_channels)
    initial_bd_new = measure_bd_state(injected, new_co, new_cross, n_channels)
    initial_signs_old = extract_cross_planar_signs(injected, old_co)
    initial_signs_new = extract_cross_planar_signs(injected, new_co)
    initial_magnitudes = bridge_magnitude_stats(injected)

    print(f"\nInitial state (measured with OLD RD pairs):")
    print(f"  Fiedler: {initial_bd_old['fiedler_mean']:.6f}")
    co_old = initial_bd_old['co_cross_ratio']
    print(f"  Co/Cross (old spec): {co_old:,.0f}:1" if co_old else "  Co/Cross: N/A")

    print(f"\nInitial state (measured with NEW {target} pairs):")
    print(f"  Fiedler: {initial_bd_new['fiedler_mean']:.6f}")
    co_new = initial_bd_new['co_cross_ratio']
    print(f"  Co/Cross (new spec): {co_new:,.0f}:1" if co_new else "  Co/Cross: N/A")

    # Save initial snapshot
    with open(output_dir / "initial_state.json", "w") as f:
        json.dump({
            "bd_old_spec": initial_bd_old,
            "bd_new_spec": initial_bd_new,
            "magnitudes": initial_magnitudes,
        }, f, indent=2, default=str)

    # Save initial bridge states
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        np.save(
            output_dir / f"bridge_step0_{safe}.npy",
            lora.effective_bridge.detach().cpu().numpy(),
        )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    bridge_params = []
    lora_params = []
    for name, lora in injected.items():
        bridge_params.append(lora.bridge)
        lora_params.extend([lora.lora_A, lora.lora_B])

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr, "name": "lora"},
        {"params": bridge_params, "lr": lr * 0.5, "name": "bridge"},
    ], weight_decay=0.01)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(phase2_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_schedule, lr_schedule])

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
    model.train()
    global_step = 0
    accumulation_loss = 0.0
    start_time = time.time()
    done = False

    # Steersman state (adaptive weights)
    contrastive_weight = initial_contrastive
    spectral_weight = initial_spectral
    bridge_lr_scale = 1.0

    print(f"\n{'=' * 60}")
    print(f"PHASE 2: Training with NEW {target} pair spec + Steersman")
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

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss / gradient_accumulation

            # Steersman losses with NEW pair spec
            c_loss = contrastive_bridge_loss(injected, new_co, new_cross) / gradient_accumulation
            s_loss = spectral_bridge_loss(injected, n_channels) / gradient_accumulation

            total_loss = lm_loss + contrastive_weight * c_loss + spectral_weight * s_loss
            total_loss.backward()
            accumulation_loss += lm_loss.item()

            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Steersman feedback
                if global_step % feedback_interval == 0:
                    wall_time = time.time() - start_time

                    # Measure BD state with BOTH pair specs
                    bd_new = measure_bd_state(injected, new_co, new_cross, n_channels)
                    bd_old = measure_bd_state(injected, old_co, old_cross, n_channels)

                    # Sign stability relative to initial (old spec)
                    current_signs_old = extract_cross_planar_signs(injected, old_co)
                    stab_old = sign_stability(current_signs_old, initial_signs_old)

                    # Sign stability relative to initial (new spec)
                    current_signs_new = extract_cross_planar_signs(injected, new_co)
                    stab_new = sign_stability(current_signs_new, initial_signs_new)

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

                    # Steersman adaptive feedback
                    feedback = steersman_feedback(
                        injected, n_channels,
                        spectral_target=0.1,
                        spectral_weight=spectral_weight,
                        contrastive_weight=contrastive_weight,
                    )
                    contrastive_weight = feedback.get("contrastive_weight", contrastive_weight)
                    spectral_weight = feedback.get("spectral_weight", spectral_weight)
                    bridge_lr_scale = feedback.get("bridge_lr_scale", bridge_lr_scale)

                    entry = {
                        "step": global_step,
                        "wall_time": wall_time,
                        "train_loss": avg_loss,
                        "val_loss": val_loss,
                        # New spec metrics
                        "fiedler_mean": bd_new["fiedler_mean"],
                        "co_cross_ratio_new": bd_new["co_cross_ratio"],
                        "eigenvalue_pattern": bd_new.get("eigenvalue_pattern", []),
                        # Old spec metrics (watching RD structure decay)
                        "co_cross_ratio_old": bd_old["co_cross_ratio"],
                        "fiedler_old": bd_old["fiedler_mean"],
                        # Sign tracking
                        "sign_stability_old": stab_old["aggregate"],
                        "sign_stability_new": stab_new["aggregate"],
                        # Steersman state
                        "contrastive_weight": contrastive_weight,
                        "spectral_weight": spectral_weight,
                        "bridge_lr_scale": bridge_lr_scale,
                        "mean_magnitude": magnitudes["mean_magnitude"],
                    }
                    checkpoints.append(entry)

                    # Print status
                    co_new_str = f"{bd_new['co_cross_ratio']:,.0f}:1" if bd_new['co_cross_ratio'] else "N/A"
                    co_old_str = f"{bd_old['co_cross_ratio']:,.0f}:1" if bd_old['co_cross_ratio'] else "N/A"
                    print(
                        f"Step {global_step:>5d} | "
                        f"LM={avg_loss:.4f} Val={val_loss:.4f} | "
                        f"NEW co/x={co_new_str} Fiedler={bd_new['fiedler_mean']:.6f} | "
                        f"OLD co/x={co_old_str} | "
                        f"Sign(old)={stab_old['aggregate']:.3f} Sign(new)={stab_new['aggregate']:.3f}"
                    )

                    # Save results.json
                    results = {
                        "experiment": "FI-005",
                        "config": config_dict,
                        "initial_state": {
                            "bd_old_spec": initial_bd_old,
                            "bd_new_spec": initial_bd_new,
                            "magnitudes": initial_magnitudes,
                        },
                        "checkpoints": checkpoints,
                    }
                    with open(output_dir / "results.json", "w") as f:
                        json.dump(results, f, indent=2, default=str)

                # Save bridge snapshots
                if global_step % checkpoint_steps == 0:
                    for name, lora in injected.items():
                        safe = name.replace(".", "_")
                        np.save(
                            output_dir / f"bridge_step{global_step}_{safe}.npy",
                            lora.effective_bridge.detach().cpu().numpy(),
                        )

                if global_step >= phase2_steps:
                    done = True
                    break

    # Final measurements
    wall_time = time.time() - start_time
    final_bd_new = measure_bd_state(injected, new_co, new_cross, n_channels)
    final_bd_old = measure_bd_state(injected, old_co, old_cross, n_channels)
    final_magnitudes = bridge_magnitude_stats(injected)

    print(f"\n{'=' * 60}")
    print(f"FI-005 COMPLETE — {target} reprogramming, {global_step} steps in {wall_time/3600:.2f}h")
    print(f"{'=' * 60}")

    co_new_f = final_bd_new['co_cross_ratio']
    co_old_f = final_bd_old['co_cross_ratio']
    print(f"\n  NEW spec ({target}):")
    print(f"    Fiedler: {final_bd_new['fiedler_mean']:.6f}")
    print(f"    Co/Cross: {co_new_f:,.0f}:1" if co_new_f else "    Co/Cross: N/A")
    print(f"\n  OLD spec (RD):")
    print(f"    Fiedler: {final_bd_old['fiedler_mean']:.6f}")
    print(f"    Co/Cross: {co_old_f:,.0f}:1" if co_old_f else "    Co/Cross: N/A")

    # Verdict
    if co_new_f and co_new_f > 1000:
        print(f"\n  VERDICT: REPROGRAMMING SUCCESSFUL — new topology established at {co_new_f:,.0f}:1")
    elif co_new_f and co_new_f > 10:
        print(f"\n  VERDICT: PARTIAL REPROGRAMMING — new topology emerging ({co_new_f:,.0f}:1)")
    else:
        print(f"\n  VERDICT: REPROGRAMMING FAILED — no directional content under new spec")

    # Save adapter
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
        "experiment": "FI-005",
        "config": config_dict,
        "initial_state": {
            "bd_old_spec": initial_bd_old,
            "bd_new_spec": initial_bd_new,
            "magnitudes": initial_magnitudes,
        },
        "final_state": {
            "bd_new_spec": final_bd_new,
            "bd_old_spec": final_bd_old,
            "magnitudes": final_magnitudes,
        },
        "checkpoints": checkpoints,
        "wall_time": wall_time,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_dir}/results.json")


# -- Analysis Mode -----------------------------------------------------------

def analyze_results(results_dir: Path):
    """Analyze FI-005 results from disk."""
    with open(results_dir / "results.json") as f:
        data = json.load(f)

    config = data.get("config", {})
    target = config.get("target_topology", "unknown")

    print(f"{'=' * 60}")
    print(f"FI-005: Topology Reprogramming Analysis ({target})")
    print(f"{'=' * 60}")

    init = data.get("initial_state", {})
    final = data.get("final_state", {})
    checkpoints = data.get("checkpoints", [])

    print(f"\nOld co-axial pairs: {config.get('old_co_pairs', '?')}")
    print(f"New co-axial pairs: {config.get('new_co_pairs', '?')}")

    init_old = init.get("bd_old_spec", {})
    init_new = init.get("bd_new_spec", {})
    co_init_old = init_old.get("co_cross_ratio")
    co_init_new = init_new.get("co_cross_ratio")

    print(f"\nInitial state:")
    print(f"  OLD spec co/cross: {co_init_old:,.0f}:1" if co_init_old else "  OLD spec co/cross: N/A")
    print(f"  NEW spec co/cross: {co_init_new:,.0f}:1" if co_init_new else "  NEW spec co/cross: N/A")

    print(f"\nTrajectory:")
    print(f"  {'Step':>6} {'New Co/X':>12} {'Old Co/X':>12} {'Fiedler':>10} {'Val Loss':>9} {'Sign(old)':>10} {'Sign(new)':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*9} {'-'*10} {'-'*10}")
    for chk in checkpoints:
        co_n = chk.get("co_cross_ratio_new")
        co_o = chk.get("co_cross_ratio_old")
        co_n_str = f"{co_n:,.0f}:1" if co_n else "N/A"
        co_o_str = f"{co_o:,.0f}:1" if co_o else "N/A"
        print(
            f"  {chk['step']:>6} "
            f"{co_n_str:>12} "
            f"{co_o_str:>12} "
            f"{chk.get('fiedler_mean', 0):>10.6f} "
            f"{chk.get('val_loss', 0):>9.4f} "
            f"{chk.get('sign_stability_old', 0):>10.4f} "
            f"{chk.get('sign_stability_new', 0):>10.4f}"
        )

    if final:
        final_new = final.get("bd_new_spec", {})
        final_old = final.get("bd_old_spec", {})
        co_f_n = final_new.get("co_cross_ratio")
        co_f_o = final_old.get("co_cross_ratio")
        print(f"\nFinal state:")
        print(f"  NEW spec co/cross: {co_f_n:,.0f}:1" if co_f_n else "  NEW spec co/cross: N/A")
        print(f"  OLD spec co/cross: {co_f_o:,.0f}:1" if co_f_o else "  OLD spec co/cross: N/A")

    print(f"\n{'=' * 60}")


# -- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FI-005: Topology Reprogramming"
    )
    parser.add_argument(
        "--target", type=str, default="reduced",
        choices=["reduced", "rotated", "random"],
        help="Target topology to reprogram to",
    )
    parser.add_argument(
        "--source-adapter", type=str, default=None,
        help="Path to adapter_state.pt from converged BD run",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/fi-005/<target>)",
    )
    parser.add_argument("--phase2-steps", type=int, default=3000)
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--n-channels", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--feedback-interval", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-contrastive", type=float, default=0.1)
    parser.add_argument("--initial-spectral", type=float, default=0.05)
    parser.add_argument(
        "--analyze-only", type=str, default=None,
        help="Path to results directory for analysis only",
    )
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(Path(args.analyze_only))
        return

    if args.output is None:
        args.output = f"results/fi-005/{args.target}"

    if args.source_adapter is None:
        default_path = Path("results/fi-002/P-000/adapter_state.pt")
        if default_path.exists():
            args.source_adapter = str(default_path)
            print(f"Using default source adapter: {default_path}")
        else:
            print("ERROR: No --source-adapter specified and no default found at")
            print(f"  {default_path}")
            sys.exit(1)

    train_reprogramming(
        source_adapter=Path(args.source_adapter),
        output_dir=Path(args.output),
        target=args.target,
        phase2_steps=args.phase2_steps,
        model_name=args.model,
        n_channels=args.n_channels,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        feedback_interval=args.feedback_interval,
        checkpoint_steps=args.checkpoint_steps,
        seed=args.seed,
        initial_contrastive=args.initial_contrastive,
        initial_spectral=args.initial_spectral,
    )


if __name__ == "__main__":
    main()
