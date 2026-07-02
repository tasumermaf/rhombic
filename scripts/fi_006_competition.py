#!/usr/bin/env python3
"""FI-006: Topology Competition

Tests what happens when TWO conflicting pair specifications are applied
simultaneously. Both contrastive losses are active at equal weight.

DESIGN:
  Train n=6 from identity bridges with TWO competing pair specs:
    Spec A (RD): 3 co-axial pairs from RD geometry
    Spec B (rotated): 3 co-axial pairs from sequential pairing (0,1), (2,3), (4,5)

  Both specs have 3 pairs but they partially overlap and partially conflict.
  The Steersman receives contradictory instructions about which channels
  should couple.

PREDICTIONS:
  1. One topology dominates (selection pressure from geometry)
  2. Collapse — mutual cancellation destroys all structure
  3. Compromise — some hybrid topology emerges
  4. Layer-wise separation — different layers adopt different topologies

KEY METRIC: Compare co/cross under BOTH specs at each checkpoint.
If one goes to >>1 while the other collapses, we have selection.
If both stay ~1, we have mutual annihilation.

Usage:
  python scripts/fi_006_competition.py
  python scripts/fi_006_competition.py --weight-ratio 2.0  # 2:1 bias toward spec A
  python scripts/fi_006_competition.py --analyze-only results/fi-006

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
from rhombic.spectral import (
    weighted_laplacian,
    fiedler_value,
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
    spectral_bridge_loss,
    _compute_pair_indices,
    steersman_feedback,
)

from fi_003_sign_persistence import (
    bridge_magnitude_stats,
    measure_bd_state,
)

from fi_005_reprogramming import (
    compute_rd_pairs,
    compute_rotated_pairs,
)


def train_competition(
    output_dir: Path,
    max_steps: int = 5000,
    weight_ratio: float = 1.0,
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
    """Train with two competing pair specifications."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Two competing pair specs
    co_a, cross_a = compute_rd_pairs()
    co_b, cross_b = compute_rotated_pairs()

    # Weight distribution: weight_ratio determines A:B
    # weight_ratio=1.0 → equal, weight_ratio=2.0 → A gets 2x
    weight_a = initial_contrastive * weight_ratio / (1.0 + weight_ratio)
    weight_b = initial_contrastive * 1.0 / (1.0 + weight_ratio)

    config = ExperimentConfig(
        name=f"fi_006_competition_ratio{weight_ratio:.1f}",
        rank=24,
        n_channels=n_channels,
        bridge_mode="identity",
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
    config_dict["experiment"] = "FI-006"
    config_dict["weight_ratio"] = weight_ratio
    config_dict["weight_a"] = weight_a
    config_dict["weight_b"] = weight_b
    config_dict["co_pairs_a"] = co_a
    config_dict["co_pairs_b"] = co_b
    config_dict["initial_spectral"] = initial_spectral

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"FI-006: Topology Competition (ratio={weight_ratio:.1f})")
    print(f"{'=' * 60}")
    print(f"Spec A (RD): {co_a} weight={weight_a:.4f}")
    print(f"Spec B (rotated): {co_b} weight={weight_b:.4f}")
    print(f"Steps: {max_steps}")
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

    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    bridge_params = [lora.bridge for lora in injected.values()]
    lora_params = []
    for lora in injected.values():
        lora_params.extend([lora.lora_A, lora.lora_B])

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr, "name": "lora"},
        {"params": bridge_params, "lr": lr * 0.5, "name": "bridge"},
    ], weight_decay=0.01)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
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

    checkpoints = []
    model.train()
    global_step = 0
    accumulation_loss = 0.0
    start_time = time.time()
    spectral_weight = initial_spectral
    done = False

    print(f"\n{'=' * 60}")
    print(f"TRAINING: Two competing contrastive specs")
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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss / gradient_accumulation

            # Two competing contrastive losses
            c_loss_a = contrastive_bridge_loss(injected, co_a, cross_a) / gradient_accumulation
            c_loss_b = contrastive_bridge_loss(injected, co_b, cross_b) / gradient_accumulation
            s_loss = spectral_bridge_loss(injected, n_channels) / gradient_accumulation

            total_loss = lm_loss + weight_a * c_loss_a + weight_b * c_loss_b + spectral_weight * s_loss
            total_loss.backward()
            accumulation_loss += lm_loss.item()

            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % feedback_interval == 0:
                    wall_time = time.time() - start_time

                    bd_a = measure_bd_state(injected, co_a, cross_a, n_channels)
                    bd_b = measure_bd_state(injected, co_b, cross_b, n_channels)
                    magnitudes = bridge_magnitude_stats(injected)

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
                        "fiedler_mean": bd_a["fiedler_mean"],
                        "co_cross_a": bd_a["co_cross_ratio"],
                        "co_cross_b": bd_b["co_cross_ratio"],
                        "eigenvalue_pattern": bd_a.get("eigenvalue_pattern", []),
                        "mean_magnitude": magnitudes["mean_magnitude"],
                    }
                    checkpoints.append(entry)

                    co_a_str = f"{bd_a['co_cross_ratio']:,.0f}:1" if bd_a['co_cross_ratio'] else "N/A"
                    co_b_str = f"{bd_b['co_cross_ratio']:,.0f}:1" if bd_b['co_cross_ratio'] else "N/A"
                    print(
                        f"Step {global_step:>5d} | "
                        f"LM={avg_loss:.4f} Val={val_loss:.4f} | "
                        f"A(RD)={co_a_str} B(rot)={co_b_str} | "
                        f"Fiedler={bd_a['fiedler_mean']:.6f}"
                    )

                    results = {
                        "experiment": "FI-006",
                        "config": config_dict,
                        "checkpoints": checkpoints,
                    }
                    with open(output_dir / "results.json", "w") as f:
                        json.dump(results, f, indent=2, default=str)

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

    wall_time = time.time() - start_time
    final_a = measure_bd_state(injected, co_a, cross_a, n_channels)
    final_b = measure_bd_state(injected, co_b, cross_b, n_channels)

    print(f"\n{'=' * 60}")
    print(f"FI-006 COMPLETE — {global_step} steps in {wall_time/3600:.2f}h")
    print(f"{'=' * 60}")
    co_fa = final_a['co_cross_ratio']
    co_fb = final_b['co_cross_ratio']
    print(f"  Spec A (RD):     co/cross = {co_fa:,.0f}:1" if co_fa else "  Spec A: N/A")
    print(f"  Spec B (rotated): co/cross = {co_fb:,.0f}:1" if co_fb else "  Spec B: N/A")

    if co_fa and co_fb:
        if co_fa > 100 and co_fb < 10:
            print(f"\n  VERDICT: SELECTION — Spec A (RD geometry) wins")
        elif co_fb > 100 and co_fa < 10:
            print(f"\n  VERDICT: SELECTION — Spec B (rotated) wins")
        elif co_fa > 100 and co_fb > 100:
            print(f"\n  VERDICT: COEXISTENCE — both topologies active")
        elif co_fa < 10 and co_fb < 10:
            print(f"\n  VERDICT: MUTUAL ANNIHILATION — both collapsed")
        else:
            print(f"\n  VERDICT: INTERMEDIATE — neither dominant nor collapsed")

    # Save
    adapter_state = {}
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        adapter_state[f"{safe}.lora_A"] = lora.lora_A.detach().cpu()
        adapter_state[f"{safe}.lora_B"] = lora.lora_B.detach().cpu()
        adapter_state[f"{safe}.bridge"] = lora.effective_bridge.detach().cpu()
    torch.save(adapter_state, output_dir / "adapter_state_final.pt")

    results = {
        "experiment": "FI-006",
        "config": config_dict,
        "final_state": {
            "bd_a": final_a,
            "bd_b": final_b,
        },
        "checkpoints": checkpoints,
        "wall_time": wall_time,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_dir}/results.json")


def analyze_results(results_dir: Path):
    """Analyze FI-006 results."""
    with open(results_dir / "results.json") as f:
        data = json.load(f)

    print(f"{'=' * 60}")
    print(f"FI-006: Topology Competition Analysis")
    print(f"{'=' * 60}")

    config = data.get("config", {})
    print(f"Weight ratio: {config.get('weight_ratio', '?')}")
    print(f"Spec A weight: {config.get('weight_a', '?')}")
    print(f"Spec B weight: {config.get('weight_b', '?')}")

    print(f"\n  {'Step':>6} {'A(RD)':>12} {'B(rot)':>12} {'Fiedler':>10} {'Val Loss':>9}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*9}")
    for chk in data.get("checkpoints", []):
        co_a = chk.get("co_cross_a")
        co_b = chk.get("co_cross_b")
        print(
            f"  {chk['step']:>6} "
            f"{(f'{co_a:,.0f}:1' if co_a else 'N/A'):>12} "
            f"{(f'{co_b:,.0f}:1' if co_b else 'N/A'):>12} "
            f"{chk.get('fiedler_mean', 0):>10.6f} "
            f"{chk.get('val_loss', 0):>9.4f}"
        )

    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="FI-006: Topology Competition")
    parser.add_argument("--output", type=str, default="results/fi-006")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--weight-ratio", type=float, default=1.0,
                        help="A:B weight ratio (1.0=equal, 2.0=A gets 2x)")
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
    parser.add_argument("--analyze-only", type=str, default=None)
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(Path(args.analyze_only))
        return

    train_competition(
        output_dir=Path(args.output),
        max_steps=args.max_steps,
        weight_ratio=args.weight_ratio,
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
