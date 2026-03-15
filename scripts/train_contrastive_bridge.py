"""Experiment 2.6: Contrastive Bridge Pre-Training.

Tests whether directional pressure during training creates a co-planar vs
cross-planar preference in the RhombiLoRA bridge that SURVIVES normal
fine-tuning.

Background: Experiments 2 and 2.5 showed the bridge learns connectivity
(Fiedler value increases) but NOT directionality (co/cross ratio stays near
1.0). The null result means the bridge cannot discover direction from data
alone. This experiment adds a contrastive loss that explicitly tells the
bridge which channel pairs are co-planar (same RD face) vs cross-planar
(across faces).

Contrastive loss:
  L_bridge = -lambda * (mean(|B[co_planar_pairs]|) - mean(|B[cross_planar_pairs]|))
  Encourages co-planar pairs to have LARGER absolute values than cross-planar.

Three scheduling modes:
  warmup_only  — contrastive active during warmup steps, then off (default)
  always       — contrastive active throughout training
  decay        — linear decay from lambda to 0 over total training steps

Uses Qwen2.5-7B-Instruct with Alpaca-cleaned for clean comparison to Exp 2.

Usage:
  # Default: warmup_only mode, 500 warmup steps
  python scripts/train_contrastive_bridge.py --output results/exp2_6

  # Always-on contrastive with higher weight
  python scripts/train_contrastive_bridge.py --contrastive-mode always --contrastive-lambda 0.2

  # Quick validation run
  python scripts/train_contrastive_bridge.py --max-steps 2000 --contrastive-warmup-steps 200

  # Decay mode with custom steps
  python scripts/train_contrastive_bridge.py --contrastive-mode decay --max-steps 10000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rhombic.nn.rhombi_lora import RhombiLoRALinear
from rhombic.nn.topology import direction_pair_coupling

# Reuse infrastructure from Exp 2
from train_exp2_scale import (
    ExperimentConfig,
    AlpacaDataset,
    inject_lora,
    collect_metrics,
    gradient_effective_rank,
    coplanar_crossplanar_ratio,
    CheckpointMetrics,
    evaluate,
    _save_results,
)


# ── Co-planar / cross-planar pair indices from RD topology ────────────


def _compute_pair_indices(
    n_channels: int = 6,
    topology: str = "auto",
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Derive co-planar and cross-planar channel pairs from geometry.

    For n=6 (RD): Co-planar pairs share 4 octahedral vertices (same axis-plane).
    Cross-planar pairs share 2 octahedral vertices (across planes).
    3 co-planar, 12 cross-planar.

    For n=8 (tesseract): Co-axial pairs along 4 coordinate axes of the 4D
    hypercube: (0,1)=±w, (2,3)=±x, (4,5)=±y, (6,7)=±z.
    4 co-axial, 24 cross-axial.

    topology='resonance': Uses prime-threading topology instead of geometry.
    Delegates to _compute_resonance_pairs(). Only valid for n=6.

    Returns (co_planar, cross_planar) as lists of (i, j) index pairs.
    """
    if topology == "resonance":
        return _compute_resonance_pairs(n_channels)
    if n_channels == 6:
        coupling = direction_pair_coupling()
        co_planar: list[tuple[int, int]] = []
        cross_planar: list[tuple[int, int]] = []
        n = coupling.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if coupling[i, j] >= 4:
                    co_planar.append((i, j))
                else:
                    cross_planar.append((i, j))
        return co_planar, cross_planar
    elif n_channels == 8:
        # Tesseract geometry: 8 cubic cells pair into 4 opposite pairs
        # along 4 coordinate axes of the 4D hypercube
        coaxial = [(0, 1), (2, 3), (4, 5), (6, 7)]
        coaxial_set = set(coaxial)
        crossaxial = [
            (i, j) for i in range(8) for j in range(i + 1, 8)
            if (i, j) not in coaxial_set
        ]
        return coaxial, crossaxial
    else:
        return [], []


def _compute_wrong_label_pairs(seed: int = 42) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Random partition of 6 channels into 3 pairs (WRONG labels).

    Tests whether the bridge accepts arbitrary topology or whether RD is special.
    Uses a fixed seed for reproducibility. The partition is random — NOT aligned
    with RD face pairs.
    """
    import random
    rng = random.Random(seed)
    channels = list(range(6))
    rng.shuffle(channels)
    # Partition into 3 random pairs
    wrong_co = [(channels[0], channels[1]), (channels[2], channels[3]), (channels[4], channels[5])]
    # Normalize pair ordering
    wrong_co = [tuple(sorted(p)) for p in wrong_co]
    wrong_co_set = set(wrong_co)
    wrong_cross = [
        (i, j) for i in range(6) for j in range(i + 1, 6)
        if (i, j) not in wrong_co_set
    ]
    return wrong_co, wrong_cross


# Precompute RD pairs at module load (backward compat)
_CO_PLANAR_PAIRS, _CROSS_PLANAR_PAIRS = _compute_pair_indices(6)


# ── Contrastive loss ──────────────────────────────────────────────────


def contrastive_bridge_loss(
    injected: dict[str, RhombiLoRALinear],
    co_pairs: list[tuple[int, int]],
    cross_pairs: list[tuple[int, int]],
) -> torch.Tensor:
    """Contrastive loss: encourage co-planar > cross-planar coupling.

    L = -(mean(|B[co_planar]|) - mean(|B[cross_planar]|))

    Averaged across all adapters. Returns a scalar tensor with grad.
    Works for any n where co_pairs and cross_pairs are defined
    (n=6 RD geometry, n=8 tesseract geometry).
    """
    device = next(iter(injected.values())).bridge.device
    total_loss = torch.tensor(0.0, device=device)

    if not co_pairs or not cross_pairs:
        return total_loss

    count = 0
    for lora in injected.values():
        B = lora.bridge
        # Validate that pair indices are within bridge dimensions
        max_idx = max(max(i, j) for i, j in co_pairs + cross_pairs)
        if B.shape[0] <= max_idx:
            continue

        co_vals = torch.stack([B[i, j].abs() for i, j in co_pairs])
        cross_vals = torch.stack([B[i, j].abs() for i, j in cross_pairs])

        co_mean = co_vals.mean()
        cross_mean = cross_vals.mean()

        # Negative: we MAXIMIZE co - cross (minimize the negative)
        total_loss = total_loss - (co_mean - cross_mean)
        count += 1

    return total_loss / max(count, 1)


# ── Circular resonance loss (prime threading topology) ────────────


# Prime threading from the corpus
CORPUS_PRIMES = [67, 23, 29, 17, 19, 31, 11, 89]


def _compute_resonance_pairs(
    n_channels: int = 6,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Compute resonant and orthogonal channel pairs from corpus prime threading.

    Channel-to-prime assignment: {0: 11, 1: 67, 2: 23, 3: 31, 4: 29, 5: 17}

    Three resonance criteria define the resonant pairs:
      - Sophie Germain: 2*11 + 1 = 23  =>  channels (0, 2) resonate
      - Consecutive primes: (29, 31)    =>  channels (4, 3) = (3, 4) resonate
      - Same residue mod 6: 11 ≡ 5, 17 ≡ 5  =>  channels (0, 5) resonate

    All other pairs from range(6) are orthogonal (no prime-theoretic bond).

    Returns (resonant_pairs, orthogonal_pairs).
    """
    if n_channels != 6:
        return [], []

    # Channel prime assignments (from corpus.py CHANNEL_PRIME_MAP)
    # {0: 11, 1: 67, 2: 23, 3: 31, 4: 29, 5: 17}

    # Resonant pairs from prime-theoretic relationships:
    #   Sophie Germain: 2(11)+1 = 23  => (0, 2)
    #   Consecutive primes: (29, 31)  => (3, 4)  [sorted: 3 < 4]
    #   Same residue mod 6: 11 % 6 = 5, 17 % 6 = 5  => (0, 5)
    resonant_pairs: list[tuple[int, int]] = [(0, 2), (3, 4), (0, 5)]

    resonant_set = set(resonant_pairs)
    orthogonal_pairs: list[tuple[int, int]] = [
        (i, j) for i in range(6) for j in range(i + 1, 6)
        if (i, j) not in resonant_set
    ]

    return resonant_pairs, orthogonal_pairs


def circular_resonance_loss(
    injected: dict,  # dict[str, RhombiLoRALinear]
    resonant_pairs: list[tuple[int, int]],
    orthogonal_pairs: list[tuple[int, int]],
) -> torch.Tensor:
    """Circular resonance loss: encourage prime-threaded channels to correlate.

    Works exactly like contrastive_bridge_loss but with resonant/orthogonal
    pairs instead of co-planar/cross-planar. Resonant pairs (channels whose
    primes share a number-theoretic relationship) are pushed toward high
    coupling; orthogonal pairs (no prime relationship) toward low coupling.

    L = -(mean(|B[resonant]|) - mean(|B[orthogonal]|))

    Averaged across all adapters. Returns a scalar tensor with grad.
    """
    device = next(iter(injected.values())).bridge.device
    total_loss = torch.tensor(0.0, device=device)

    if not resonant_pairs or not orthogonal_pairs:
        return total_loss

    count = 0
    for lora in injected.values():
        B = lora.bridge
        max_idx = max(max(i, j) for i, j in resonant_pairs + orthogonal_pairs)
        if B.shape[0] <= max_idx:
            continue

        res_vals = torch.stack([B[i, j].abs() for i, j in resonant_pairs])
        orth_vals = torch.stack([B[i, j].abs() for i, j in orthogonal_pairs])

        res_mean = res_vals.mean()
        orth_mean = orth_vals.mean()

        total_loss = total_loss - (res_mean - orth_mean)
        count += 1

    return total_loss / max(count, 1)


# ── Contrastive scheduling ──────────────────────────────────────────


def contrastive_weight(
    step: int,
    mode: str,
    lambda_base: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Compute the contrastive loss weight at a given step.

    Parameters
    ----------
    step : int
        Current global training step.
    mode : str
        'warmup_only' — lambda during warmup, 0 after.
        'always' — constant lambda throughout.
        'decay' — linear decay from lambda to 0 over total_steps.
    lambda_base : float
        Base contrastive weight.
    warmup_steps : int
        Number of warmup steps (relevant for warmup_only mode).
    total_steps : int
        Total training steps (relevant for decay mode).
    """
    if mode == "warmup_only":
        return lambda_base if step <= warmup_steps else 0.0
    elif mode == "always":
        return lambda_base
    elif mode == "decay":
        progress = min(step / max(total_steps, 1), 1.0)
        return lambda_base * (1.0 - progress)
    else:
        raise ValueError(f"Unknown contrastive mode: {mode!r}")


# ── Extended checkpoint metrics ───────────────────────────────────────


@dataclass
class ContrastiveCheckpointMetrics:
    step: int
    train_loss: float
    val_loss: float | None
    contrastive_loss: float
    contrastive_weight: float
    co_cross_ratio: float | None
    bridge_fiedler_mean: float
    bridge_fiedler_std: float
    bridge_deviation_mean: float
    bridge_deviation_std: float
    coplanar_crossplanar: dict | None
    grad_effective_rank_mean: float
    n_adapters: int
    wall_time: float


# ── Training ──────────────────────────────────────────────────────────


def train_contrastive(
    config: ExperimentConfig,
    output_dir: Path,
    contrastive_lambda: float = 0.1,
    contrastive_warmup_steps: int = 500,
    contrastive_mode: str = "warmup_only",
):
    """Run Exp 2.6: contrastive bridge training.

    Uses the same ExperimentConfig as Exp 2 for direct comparison.
    The contrastive loss is an auxiliary signal added to the standard
    LM loss, weighted and scheduled according to mode.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config with contrastive parameters
    config_dict = asdict(config)
    config_dict["experiment"] = "2.6"
    config_dict["contrastive_lambda"] = contrastive_lambda
    config_dict["contrastive_warmup_steps"] = contrastive_warmup_steps
    config_dict["contrastive_mode"] = contrastive_mode
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Experiment 2.6: Contrastive Bridge Pre-Training")
    print(f"Model: {config.model_name}")
    print(f"Rank: {config.rank}, Channels: {config.n_channels}")
    print(f"Bridge: {config.bridge_mode}, Trainable: {config.bridge_trainable}")
    print(f"Steps: {config.max_steps}")
    print(f"Contrastive lambda: {contrastive_lambda}")
    print(f"Contrastive warmup: {contrastive_warmup_steps}")
    print(f"Contrastive mode: {contrastive_mode}")
    print(f"Co-planar pairs: {_CO_PLANAR_PAIRS}")
    print(f"Cross-planar pairs: {len(_CROSS_PLANAR_PAIRS)} pairs")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    bridge_params = sum(
        lora.bridge.numel() for lora in injected.values()
        if lora.bridge.requires_grad
    )
    print(f"Trainable: {trainable_params:,} / {total_params:,}")
    print(f"Bridge params: {bridge_params:,}")

    # Dataset — Alpaca-cleaned for clean comparison to Exp 2
    print("Loading Alpaca-cleaned dataset...")
    dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len)
    val_dataset = AlpacaDataset(
        tokenizer, max_len=config.max_seq_len, max_samples=1000
    )
    print(f"Training examples: {len(dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=0.01,
    )

    total_steps = config.max_steps

    def lr_schedule(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    checkpoints: list[dict] = []
    model.train()
    global_step = 0
    accumulation_loss = 0.0
    accumulation_contrast = 0.0
    steps_since_ckpt = 0
    start_time = time.time()

    print(f"\nTraining for {total_steps} steps...")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation}")
    print()

    # Save step-0 bridges
    for name, lora in injected.items():
        safe_name = name.replace(".", "_")
        B = lora.bridge.detach().cpu().numpy()
        np.save(output_dir / f"bridge_step0_{safe_name}.npy", B)

    done = False
    for epoch in range(config.num_epochs):
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
            lm_loss = outputs.loss / config.gradient_accumulation

            # Compute contrastive weight for current step
            c_weight = contrastive_weight(
                global_step + 1,  # +1 because step increments after accumulation
                contrastive_mode,
                contrastive_lambda,
                contrastive_warmup_steps,
                total_steps,
            )

            # Contrastive bridge loss (only if weight > 0 and 6-channel)
            if c_weight > 0.0 and config.n_channels == 6:
                c_loss = contrastive_bridge_loss(
                    injected, _CO_PLANAR_PAIRS, _CROSS_PLANAR_PAIRS
                )
                scaled_c_loss = c_weight * c_loss / config.gradient_accumulation
                total_loss = lm_loss + scaled_c_loss
                accumulation_contrast += scaled_c_loss.item()
            else:
                total_loss = lm_loss

            total_loss.backward()
            accumulation_loss += lm_loss.item()

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                steps_since_ckpt += 1

                # Progress every 100 steps
                if global_step % 100 == 0:
                    avg_loss = accumulation_loss / steps_since_ckpt
                    avg_contrast = accumulation_contrast / max(steps_since_ckpt, 1)
                    cur_c_weight = contrastive_weight(
                        global_step, contrastive_mode,
                        contrastive_lambda, contrastive_warmup_steps,
                        total_steps,
                    )
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    eta = (total_steps - global_step) / steps_per_sec

                    # Compute co/cross ratio for progress display
                    co_cross_str = ""
                    if config.n_channels == 6:
                        # Quick ratio from first adapter
                        first_lora = next(iter(injected.values()))
                        B_np = first_lora.bridge.detach().cpu().numpy()
                        ccp = coplanar_crossplanar_ratio(B_np)
                        if ccp is not None:
                            co_cross_str = f" | Co/Cross: {ccp['ratio']:.3f}"

                    print(
                        f"  Step {global_step:>5}/{total_steps} | "
                        f"LM: {avg_loss:.4f} | "
                        f"C: {avg_contrast:.6f} (w={cur_c_weight:.4f}) | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Speed: {steps_per_sec:.2f} step/s{co_cross_str} | "
                        f"ETA: {eta/3600:.1f}h"
                    )

                # Checkpoint every 100 steps (denser than Exp 2 for bridge tracking)
                if global_step % 100 == 0 or global_step >= total_steps:
                    avg_loss = accumulation_loss / steps_since_ckpt
                    avg_contrast = accumulation_contrast / max(steps_since_ckpt, 1)
                    cur_c_weight = contrastive_weight(
                        global_step, contrastive_mode,
                        contrastive_lambda, contrastive_warmup_steps,
                        total_steps,
                    )

                    f_mean, f_std, d_mean, d_std, ccp = collect_metrics(injected)
                    eff_rank = gradient_effective_rank(injected)
                    val_loss = evaluate(model, val_dataloader, device)

                    # Co/cross ratio (aggregated)
                    co_cross_ratio = None
                    if ccp is not None:
                        co_cross_ratio = ccp["ratio"]

                    cp = ContrastiveCheckpointMetrics(
                        step=global_step,
                        train_loss=avg_loss,
                        val_loss=val_loss,
                        contrastive_loss=avg_contrast,
                        contrastive_weight=cur_c_weight,
                        co_cross_ratio=co_cross_ratio,
                        bridge_fiedler_mean=f_mean,
                        bridge_fiedler_std=f_std,
                        bridge_deviation_mean=d_mean,
                        bridge_deviation_std=d_std,
                        coplanar_crossplanar=ccp,
                        grad_effective_rank_mean=eff_rank,
                        n_adapters=len(injected),
                        wall_time=time.time() - start_time,
                    )
                    checkpoints.append(asdict(cp))

                    # Save bridge .npy files
                    for name, lora in injected.items():
                        safe_name = name.replace(".", "_")
                        B = lora.bridge.detach().cpu().numpy()
                        np.save(
                            output_dir / f"bridge_step{global_step}_{safe_name}.npy",
                            B,
                        )

                    ccp_str = ""
                    if co_cross_ratio is not None:
                        ccp_str = f" | Co/Cross: {co_cross_ratio:.3f}"

                    phase_label = "CONTRASTIVE" if cur_c_weight > 0 else "NORMAL"
                    print(
                        f"\n  *** Checkpoint {global_step}/{total_steps} [{phase_label}] ***\n"
                        f"      Train loss:      {avg_loss:.4f}\n"
                        f"      Val loss:        {val_loss:.4f}\n"
                        f"      Contrastive:     {avg_contrast:.6f} (w={cur_c_weight:.4f})\n"
                        f"      Fiedler:         {f_mean:.5f} +/- {f_std:.5f}\n"
                        f"      Deviation:       {d_mean:.5f} +/- {d_std:.5f}\n"
                        f"      Eff rank:        {eff_rank:.2f}{ccp_str}\n"
                        f"      Wall time:       {cp.wall_time/3600:.2f}h\n"
                    )

                    accumulation_loss = 0.0
                    accumulation_contrast = 0.0
                    steps_since_ckpt = 0

                    # Save intermediate results
                    _save_contrastive_results(
                        output_dir, config, checkpoints,
                        trainable_params, total_params, bridge_params,
                        injected, contrastive_lambda, contrastive_warmup_steps,
                        contrastive_mode,
                    )

                if global_step >= total_steps:
                    done = True

    # Final save
    _save_contrastive_results(
        output_dir, config, checkpoints,
        trainable_params, total_params, bridge_params,
        injected, contrastive_lambda, contrastive_warmup_steps,
        contrastive_mode,
    )

    elapsed = time.time() - start_time
    print(f"\nExperiment 2.6 complete: {config.name}")
    print(f"Total time: {elapsed/3600:.2f}h")
    print(f"Results saved to {output_dir}")

    # Summary: co/cross trajectory
    if checkpoints:
        co_cross_trajectory = [
            (cp["step"], cp.get("co_cross_ratio"))
            for cp in checkpoints
            if cp.get("co_cross_ratio") is not None
        ]
        if co_cross_trajectory:
            print(f"\nCo/Cross ratio trajectory:")
            for step, ratio in co_cross_trajectory:
                c_w = contrastive_weight(
                    step, contrastive_mode, contrastive_lambda,
                    contrastive_warmup_steps, total_steps,
                )
                label = "*" if c_w > 0 else " "
                print(f"  {label} Step {step:>5}: {ratio:.4f}")
            final_ratio = co_cross_trajectory[-1][1]
            print(f"\nFinal Co/Cross: {final_ratio:.4f}")
            if contrastive_mode == "warmup_only" and len(co_cross_trajectory) > 1:
                # Find the checkpoint closest to the warmup boundary
                warmup_ratios = [
                    (s, r) for s, r in co_cross_trajectory
                    if s <= contrastive_warmup_steps
                ]
                if warmup_ratios:
                    warmup_end_ratio = warmup_ratios[-1][1]
                    print(f"Co/Cross at warmup end: {warmup_end_ratio:.4f}")
                    survived = final_ratio > 1.05
                    print(
                        f"Directional preference "
                        f"{'SURVIVED' if survived else 'DID NOT SURVIVE'} "
                        f"post-warmup training"
                    )

    return checkpoints


def _save_contrastive_results(
    output_dir, config, checkpoints,
    trainable_params, total_params, bridge_params,
    injected, contrastive_lambda, contrastive_warmup_steps,
    contrastive_mode,
):
    """Save results with contrastive-specific metadata."""
    results = {
        "config": asdict(config),
        "experiment": "2.6",
        "contrastive": {
            "lambda": contrastive_lambda,
            "warmup_steps": contrastive_warmup_steps,
            "mode": contrastive_mode,
            "co_planar_pairs": _CO_PLANAR_PAIRS,
            "cross_planar_pairs": _CROSS_PLANAR_PAIRS,
        },
        "checkpoints": checkpoints,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "bridge_params": bridge_params,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save final bridge matrices
    for name, lora in injected.items():
        safe_name = name.replace(".", "_")
        B = lora.bridge.detach().cpu().numpy()
        np.save(output_dir / f"bridge_final_{safe_name}.npy", B)


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2.6: Contrastive Bridge Pre-Training"
    )
    parser.add_argument(
        "--output", type=str, default="results/exp2_6",
        help="Output directory. Default: results/exp2_6",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name. Default: Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000,
        help="Total training steps. Default: 10000",
    )
    parser.add_argument(
        "--contrastive-lambda", type=float, default=0.1,
        help="Contrastive loss weight. Default: 0.1",
    )
    parser.add_argument(
        "--contrastive-warmup-steps", type=int, default=500,
        help="Steps during which contrastive loss is active (warmup_only mode). Default: 500",
    )
    parser.add_argument(
        "--contrastive-mode", type=str, default="warmup_only",
        choices=["warmup_only", "always", "decay"],
        help=(
            "Contrastive scheduling mode. "
            "warmup_only: active during warmup then off (default). "
            "always: constant weight throughout. "
            "decay: linear decay from lambda to 0."
        ),
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, default=100,
        help="Bridge checkpoint interval. Default: 100",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate. Default: 2e-4",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Micro batch size. Default: 2",
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=8,
        help="Gradient accumulation steps. Default: 8",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=200,
        help="LR warmup steps. Default: 200",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        name="exp2_6_contrastive_bridge",
        rank=24,
        n_channels=6,
        bridge_mode="identity",
        bridge_trainable=True,
        model_name=args.model,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        checkpoint_steps=args.checkpoint_steps,
        seed=args.seed,
    )

    train_contrastive(
        config,
        Path(args.output),
        contrastive_lambda=args.contrastive_lambda,
        contrastive_warmup_steps=args.contrastive_warmup_steps,
        contrastive_mode=args.contrastive_mode,
    )


if __name__ == "__main__":
    main()
