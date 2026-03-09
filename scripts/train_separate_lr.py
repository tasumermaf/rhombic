"""Experiment 2.7: Separate Bridge Learning Rate

Tests whether bridge parameters need a higher learning rate than A/B projections
to learn directional structure. The bridge has 36 parameters vs ~15M for A/B —
at the same LR, bridge gradients may be too small relative to noise, preventing
the bridge from learning co-planar vs cross-planar preferences.

Design: Two optimizer parameter groups:
  - Group 0: lora_A, lora_B at base LR (2e-4)
  - Group 1: bridge at multiplied LR (default 10x = 2e-3)

Three bridge LR schedules:
  - constant:  bridge_lr = base_lr * multiplier for all steps
  - warmup:    bridge_lr = base_lr * multiplier for first N steps,
               then snaps to base_lr (let bridge learn fast, then freeze rate)
  - decay:     bridge_lr starts at base_lr * multiplier, linearly decays
               to base_lr over N steps (gradual convergence)

Trains on Qwen2.5-7B-Instruct with Alpaca-cleaned (same as Exp 2).
Saves bridge checkpoints every 100 steps as .npy files.
Logs bridge LR, co/cross ratio, Fiedler value at each checkpoint.

Pass criterion:
  - Fiedler at 10K steps significantly higher than Exp 2 baseline (0.040)
  - OR co/cross ratio > 1.05 (bridge LR enables directional preference)

Usage:
  # Default: 10x constant bridge LR
  python -u scripts/train_separate_lr.py --output results/exp2_7

  # Warmup schedule: high bridge LR for 1000 steps then match base
  python -u scripts/train_separate_lr.py --bridge-lr-schedule warmup --output results/exp2_7_warmup

  # Decay schedule: 20x bridge LR decaying to 1x over 2000 steps
  python -u scripts/train_separate_lr.py --bridge-lr-multiplier 20 --bridge-lr-schedule decay \\
      --bridge-lr-warmup-steps 2000 --output results/exp2_7_decay

  # Quick test: 2K steps
  python -u scripts/train_separate_lr.py --max-steps 2000 --output results/exp2_7_quick
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rhombic.nn.rhombi_lora import RhombiLoRALinear
from rhombic.nn.topology import direction_pair_coupling


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class SeparateLRConfig:
    """Experiment 2.7 configuration."""
    name: str = "exp2_7_separate_lr"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    rank: int = 24
    n_channels: int = 6
    alpha: float = 16.0
    max_seq_len: int = 512
    batch_size: int = 2
    gradient_accumulation: int = 8
    base_lr: float = 2e-4
    bridge_lr_multiplier: float = 10.0
    bridge_lr_schedule: str = "constant"  # constant | warmup | decay
    bridge_lr_warmup_steps: int = 1000    # for warmup/decay schedules
    warmup_steps: int = 200               # global warmup for both groups
    checkpoint_steps: int = 1000
    max_steps: int = 10_000
    num_epochs: int = 4
    seed: int = 42
    gradient_checkpointing: bool = True
    bridge_mode: str = "identity"
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    @property
    def bridge_lr(self) -> float:
        """Initial bridge LR (before schedule adjustments)."""
        return self.base_lr * self.bridge_lr_multiplier


# ── Co-planar / Cross-planar from RD geometry ────────────────────────


_COUPLING = direction_pair_coupling()


def _coplanar_crossplanar_indices(n_channels: int):
    """Return co-planar and cross-planar pair indices from RD geometry.

    Co-planar pairs share 4 octahedral vertices; cross-planar share 2.
    Only meaningful for n_channels=6.
    """
    if n_channels != 6:
        return None, None

    coplanar = []
    crossplanar = []
    for i in range(6):
        for j in range(i + 1, 6):
            if _COUPLING[i, j] >= 4:
                coplanar.append((i, j))
            else:
                crossplanar.append((i, j))
    return coplanar, crossplanar


def coplanar_crossplanar_ratio(bridge: np.ndarray) -> dict | None:
    """Compute co-planar vs cross-planar coupling ratio for a 6x6 bridge.

    Uses actual RD geometry (shared octahedral vertex count) to classify
    pairs, not axis-pair heuristic.
    """
    n = bridge.shape[0]
    coplanar_idx, crossplanar_idx = _coplanar_crossplanar_indices(n)
    if coplanar_idx is None:
        return None

    coplanar_vals = [abs(bridge[i, j]) for i, j in coplanar_idx]
    crossplanar_vals = [abs(bridge[i, j]) for i, j in crossplanar_idx]

    mean_co = float(np.mean(coplanar_vals)) if coplanar_vals else 0.0
    mean_cross = float(np.mean(crossplanar_vals)) if crossplanar_vals else 0.0
    ratio = mean_co / mean_cross if mean_cross > 1e-12 else float('inf')

    return {
        "mean_coplanar": mean_co,
        "mean_crossplanar": mean_cross,
        "ratio": ratio,
        "n_coplanar_pairs": len(coplanar_idx),
        "n_crossplanar_pairs": len(crossplanar_idx),
    }


# ── Dataset ──────────────────────────────────────────────────────────


class AlpacaDataset(Dataset):
    """Alpaca-cleaned in instruction format."""

    def __init__(self, tokenizer, max_len: int = 512, max_examples: int = 0):
        from datasets import load_dataset

        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        if max_examples > 0:
            ds = ds.select(range(min(max_examples, len(ds))))

        self.examples = []
        for item in ds:
            instruction = item["instruction"]
            inp = item.get("input", "")
            output = item["output"]

            if inp:
                prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{inp}\n\n"
                    f"### Response:\n{output}"
                )
            else:
                prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Response:\n{output}"
                )

            encoded = tokenizer(
                prompt, truncation=True, max_length=max_len,
                padding="max_length", return_tensors="pt",
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0).clone(),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ── LoRA injection ───────────────────────────────────────────────────


def inject_lora(
    model: nn.Module, config: SeparateLRConfig
) -> dict[str, RhombiLoRALinear]:
    """Replace target linear layers with LoRA-wrapped versions."""
    injected: dict[str, RhombiLoRALinear] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short_name = name.split(".")[-1]
        if short_name not in config.target_modules:
            continue

        lora = RhombiLoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=config.rank,
            n_channels=config.n_channels,
            alpha=config.alpha,
            bridge_mode=config.bridge_mode,
        )
        lora = lora.to(device=module.weight.device, dtype=torch.float32)
        injected[name] = lora
        _wrap_forward(model, name, module, lora)

    return injected


def _wrap_forward(model, name, original, lora):
    """Monkey-patch the module's forward to add LoRA output."""
    parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
    parent = model.get_submodule(parent_name) if parent_name else model

    class LoRAWrappedLinear(nn.Module):
        def __init__(self, base, adapter):
            super().__init__()
            self.base = base
            self.adapter = adapter
            for p in self.base.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.base(x) + self.adapter(x)

    setattr(parent, attr_name, LoRAWrappedLinear(original, lora))


# ── Bridge LR schedule ──────────────────────────────────────────────


def bridge_lr_multiplier_at_step(
    step: int,
    schedule: str,
    multiplier: float,
    warmup_steps: int,
) -> float:
    """Compute the bridge LR multiplier at a given step.

    Returns a multiplier relative to base_lr. The actual bridge LR is
    base_lr * this_return_value.

    Schedules:
      constant — always returns `multiplier`
      warmup   — returns `multiplier` for step < warmup_steps, then 1.0
      decay    — linearly interpolates from `multiplier` to 1.0 over
                 warmup_steps, then stays at 1.0
    """
    if schedule == "constant":
        return multiplier
    elif schedule == "warmup":
        if step < warmup_steps:
            return multiplier
        return 1.0
    elif schedule == "decay":
        if step >= warmup_steps:
            return 1.0
        # Linear decay from multiplier to 1.0
        progress = step / max(warmup_steps, 1)
        return multiplier + (1.0 - multiplier) * progress
    else:
        raise ValueError(f"Unknown bridge LR schedule: {schedule!r}")


# ── Metrics ──────────────────────────────────────────────────────────


def collect_bridge_metrics(
    injected: dict[str, RhombiLoRALinear],
) -> dict:
    """Collect aggregated bridge metrics across all adapters."""
    fiedlers = []
    deviations = []
    bridge_grad_norms = []
    ab_grad_norms = []
    all_ccp = []

    for lora in injected.values():
        fiedlers.append(lora.bridge_fiedler())
        deviations.append(lora.bridge_deviation())

        if lora.bridge.grad is not None:
            bridge_grad_norms.append(float(lora.bridge.grad.norm().item()))
        if lora.lora_A.grad is not None:
            ab_grad_norms.append(float(lora.lora_A.grad.norm().item()))

        B = lora.bridge.detach().cpu().numpy()
        ccp = coplanar_crossplanar_ratio(B)
        if ccp is not None:
            all_ccp.append(ccp)

    result = {
        "fiedler_mean": float(np.mean(fiedlers)),
        "fiedler_std": float(np.std(fiedlers)),
        "deviation_mean": float(np.mean(deviations)),
        "deviation_std": float(np.std(deviations)),
    }

    if bridge_grad_norms:
        result["bridge_grad_norm_mean"] = float(np.mean(bridge_grad_norms))
        result["bridge_grad_norm_std"] = float(np.std(bridge_grad_norms))
    if ab_grad_norms:
        result["ab_grad_norm_mean"] = float(np.mean(ab_grad_norms))
        result["ab_grad_norm_std"] = float(np.std(ab_grad_norms))

    # Aggregate co-planar/cross-planar across adapters
    if all_ccp:
        all_co = [r["mean_coplanar"] for r in all_ccp]
        all_cross = [r["mean_crossplanar"] for r in all_ccp]
        mean_co = float(np.mean(all_co))
        mean_cross = float(np.mean(all_cross))
        result["co_cross_ratio"] = (
            float(mean_co / mean_cross) if mean_cross > 1e-12 else float('inf')
        )
        result["mean_coplanar"] = mean_co
        result["mean_crossplanar"] = mean_cross
        result["co_cross_std"] = float(np.std([r["ratio"] for r in all_ccp]))
        result["n_adapters_with_ccp"] = len(all_ccp)

    return result


def gradient_effective_rank(
    injected: dict[str, RhombiLoRALinear],
) -> float:
    """Mean effective rank of gradient across all adapters."""
    ranks = []
    for lora in injected.values():
        if lora.lora_A.grad is None:
            continue
        with torch.no_grad():
            G = lora.lora_A.grad.float()
            s = torch.linalg.svdvals(G)
            s = s / (s.sum() + 1e-12)
            entropy = -(s * torch.log(s + 1e-12)).sum()
            ranks.append(float(torch.exp(entropy).item()))
    return float(np.mean(ranks)) if ranks else 0.0


# ── Validation ───────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches: int = 50) -> float:
    """Compute mean validation loss over max_batches."""
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ── Training loop ───────────────────────────────────────────────────


def train(config: SeparateLRConfig, output_dir: Path):
    """Run Experiment 2.7 with separate bridge learning rate."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = asdict(config)
    config_dict["experiment"] = "2.7"
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    effective_bridge_lr = config.bridge_lr
    print(f"\n{'='*70}")
    print(f"Experiment 2.7: Separate Bridge Learning Rate")
    print(f"Model: {config.model_name}")
    print(f"Rank: {config.rank}, Channels: {config.n_channels}")
    print(f"Base LR: {config.base_lr}")
    print(f"Bridge LR: {effective_bridge_lr} ({config.bridge_lr_multiplier}x)")
    print(f"Bridge LR schedule: {config.bridge_lr_schedule}")
    if config.bridge_lr_schedule != "constant":
        print(f"Bridge LR warmup/decay steps: {config.bridge_lr_warmup_steps}")
    print(f"Steps: {config.max_steps}, Checkpoint every: {config.checkpoint_steps}")
    print(f"Bridge .npy saved every: 100 steps")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=torch.bfloat16, device_map=device,
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

    # Inject LoRA
    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")
    for name in sorted(injected.keys())[:8]:
        print(f"  {name}")
    if len(injected) > 8:
        print(f"  ... and {len(injected) - 8} more")

    # ── TWO OPTIMIZER PARAMETER GROUPS ──
    bridge_params = []
    ab_params = []
    for lora in injected.values():
        bridge_params.append(lora.bridge)
        ab_params.extend([lora.lora_A, lora.lora_B])

    trainable_bridge = sum(p.numel() for p in bridge_params)
    trainable_ab = sum(p.numel() for p in ab_params)
    trainable_total = trainable_bridge + trainable_ab
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable A/B: {trainable_ab:,} @ LR {config.base_lr}")
    print(f"Trainable bridge: {trainable_bridge:,} @ LR {effective_bridge_lr} "
          f"({config.bridge_lr_multiplier}x)")
    print(f"Total trainable: {trainable_total:,} / {total_params:,} "
          f"({100 * trainable_total / total_params:.4f}%)")
    print(f"Bridge fraction: {100 * trainable_bridge / trainable_total:.3f}% "
          f"of trainable")

    optimizer = torch.optim.AdamW([
        {
            "params": ab_params,
            "lr": config.base_lr,
            "weight_decay": 0.01,
        },
        {
            "params": bridge_params,
            "lr": config.bridge_lr,
            "weight_decay": 0.0,  # No weight decay on 36-param bridge
        },
    ])

    total_steps = config.max_steps

    # Base LR schedule (cosine with warmup) — applied to BOTH groups
    def base_lr_schedule(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # Bridge LR schedule: base cosine * bridge multiplier schedule
    def bridge_lr_schedule(step: int) -> float:
        base_factor = base_lr_schedule(step)
        bridge_mult = bridge_lr_multiplier_at_step(
            step,
            config.bridge_lr_schedule,
            config.bridge_lr_multiplier,
            config.bridge_lr_warmup_steps,
        )
        # The optimizer's group 1 has lr=config.bridge_lr (= base * multiplier).
        # The scheduler multiplies group lr by the lambda return value.
        # We want effective_lr = base_lr * base_factor * bridge_mult_at_step
        # Group 1 base lr = base_lr * config.bridge_lr_multiplier
        # So lambda = base_factor * bridge_mult_at_step / config.bridge_lr_multiplier
        return base_factor * bridge_mult / config.bridge_lr_multiplier

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [base_lr_schedule, bridge_lr_schedule]
    )

    # Load dataset
    print("Loading Alpaca-cleaned dataset...")
    dataset = AlpacaDataset(tokenizer, config.max_seq_len)
    val_dataset = AlpacaDataset(tokenizer, config.max_seq_len, max_examples=1000)
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

    # Save step-0 bridges
    for name, lora in injected.items():
        safe_name = name.replace(".", "_")
        B = lora.bridge.detach().cpu().numpy()
        np.save(output_dir / f"bridge_step0_{safe_name}.npy", B)

    # Training loop
    checkpoints: list[dict] = []
    bridge_lr_log: list[dict] = []  # Track bridge LR at every 100-step log
    model.train()
    global_step = 0
    acc_loss = 0.0
    steps_since_ckpt = 0
    start_time = time.time()

    print(f"\nTraining for {total_steps} steps...")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation}")
    print(f"Dataset: {len(dataset)} examples, {len(dataloader)} batches/epoch")
    print()

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
            loss = outputs.loss / config.gradient_accumulation
            loss.backward()
            acc_loss += loss.item()

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    bridge_params + ab_params, max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                steps_since_ckpt += 1

                # ── Every 100 steps: log + save bridge .npy ──
                if global_step % 100 == 0:
                    avg_loss = acc_loss / max(steps_since_ckpt, 1)
                    elapsed = time.time() - start_time
                    sps = global_step / max(elapsed, 0.01)
                    eta = (total_steps - global_step) / max(sps, 0.01)

                    # Current LRs from scheduler
                    current_lrs = scheduler.get_last_lr()
                    ab_lr = current_lrs[0]
                    current_bridge_lr = current_lrs[1]

                    # Quick co/cross and Fiedler
                    fiedlers = [
                        lora.bridge_fiedler() for lora in injected.values()
                    ]
                    fiedler_mean = float(np.mean(fiedlers))

                    all_ccp = []
                    for lora in injected.values():
                        B = lora.bridge.detach().cpu().numpy()
                        ccp = coplanar_crossplanar_ratio(B)
                        if ccp is not None:
                            all_ccp.append(ccp["ratio"])
                    co_cross = float(np.mean(all_ccp)) if all_ccp else 1.0

                    # Compute bridge LR multiplier at this step
                    effective_mult = bridge_lr_multiplier_at_step(
                        global_step,
                        config.bridge_lr_schedule,
                        config.bridge_lr_multiplier,
                        config.bridge_lr_warmup_steps,
                    )

                    print(
                        f"  Step {global_step:>5}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"AB LR: {ab_lr:.2e} | "
                        f"Br LR: {current_bridge_lr:.2e} ({effective_mult:.1f}x) | "
                        f"Fiedler: {fiedler_mean:.5f} | "
                        f"Co/Cross: {co_cross:.4f} | "
                        f"ETA: {eta / 3600:.1f}h"
                    )

                    # Save bridge .npy at every 100 steps
                    for name, lora in injected.items():
                        safe_name = name.replace(".", "_")
                        B = lora.bridge.detach().cpu().numpy()
                        np.save(
                            output_dir / f"bridge_step{global_step}_{safe_name}.npy",
                            B,
                        )

                    # Track bridge LR evolution
                    bridge_lr_log.append({
                        "step": global_step,
                        "ab_lr": float(ab_lr),
                        "bridge_lr": float(current_bridge_lr),
                        "effective_multiplier": float(effective_mult),
                        "fiedler_mean": fiedler_mean,
                        "co_cross": co_cross,
                        "train_loss": avg_loss,
                    })

                # ── Full checkpoint at checkpoint_steps ──
                if (
                    global_step % config.checkpoint_steps == 0
                    or global_step >= total_steps
                ):
                    avg_loss = acc_loss / max(steps_since_ckpt, 1)
                    metrics = collect_bridge_metrics(injected)
                    eff_rank = gradient_effective_rank(injected)
                    val_loss = evaluate(model, val_dataloader, device)

                    current_lrs = scheduler.get_last_lr()
                    effective_mult = bridge_lr_multiplier_at_step(
                        global_step,
                        config.bridge_lr_schedule,
                        config.bridge_lr_multiplier,
                        config.bridge_lr_warmup_steps,
                    )

                    cp = {
                        "step": global_step,
                        "train_loss": avg_loss,
                        "val_loss": val_loss,
                        "ab_lr": float(current_lrs[0]),
                        "bridge_lr": float(current_lrs[1]),
                        "bridge_lr_effective_multiplier": float(effective_mult),
                        "grad_effective_rank": eff_rank,
                        "wall_time": time.time() - start_time,
                        **metrics,
                    }
                    checkpoints.append(cp)

                    # Save bridges at checkpoint (may duplicate 100-step save
                    # — that's fine, ensures final checkpoint is always saved)
                    for name, lora in injected.items():
                        safe_name = name.replace(".", "_")
                        B = lora.bridge.detach().cpu().numpy()
                        np.save(
                            output_dir / f"bridge_step{global_step}_{safe_name}.npy",
                            B,
                        )

                    co_cross_str = ""
                    if "co_cross_ratio" in metrics:
                        co_cross_str = (
                            f"      Co/Cross:   {metrics['co_cross_ratio']:.4f} "
                            f"± {metrics.get('co_cross_std', 0):.4f}\n"
                        )

                    grad_str = ""
                    if "bridge_grad_norm_mean" in metrics:
                        grad_str = (
                            f"      Bridge grad: {metrics['bridge_grad_norm_mean']:.6f} | "
                            f"A/B grad: {metrics.get('ab_grad_norm_mean', 0):.6f}\n"
                        )

                    print(
                        f"\n  *** Checkpoint {global_step}/{total_steps} ***\n"
                        f"      Train loss: {avg_loss:.4f}\n"
                        f"      Val loss:   {val_loss:.4f}\n"
                        f"      AB LR:      {current_lrs[0]:.2e}\n"
                        f"      Bridge LR:  {current_lrs[1]:.2e} "
                        f"({effective_mult:.1f}x base)\n"
                        f"      Fiedler:    {metrics['fiedler_mean']:.5f} "
                        f"± {metrics['fiedler_std']:.5f}\n"
                        f"      Deviation:  {metrics['deviation_mean']:.5f} "
                        f"± {metrics['deviation_std']:.5f}\n"
                        f"{co_cross_str}"
                        f"{grad_str}"
                        f"      Eff rank:   {eff_rank:.2f}\n"
                        f"      Wall time:  {cp['wall_time'] / 3600:.2f}h\n"
                    )

                    acc_loss = 0.0
                    steps_since_ckpt = 0

                    # Save results incrementally
                    _save_results(
                        output_dir, config, checkpoints, bridge_lr_log,
                        trainable_ab, trainable_bridge, total_params,
                        injected,
                    )

                if global_step >= total_steps:
                    done = True

    # Final save
    _save_results(
        output_dir, config, checkpoints, bridge_lr_log,
        trainable_ab, trainable_bridge, total_params,
        injected,
    )

    elapsed = time.time() - start_time
    print(f"\nExperiment 2.7 complete: {config.name}")
    print(f"Schedule: {config.bridge_lr_schedule} "
          f"(multiplier={config.bridge_lr_multiplier}x)")
    print(f"Total time: {elapsed / 3600:.2f}h")
    print(f"Results saved to {output_dir}")
    return checkpoints


def _save_results(
    output_dir: Path,
    config: SeparateLRConfig,
    checkpoints: list[dict],
    bridge_lr_log: list[dict],
    trainable_ab: int,
    trainable_bridge: int,
    total_params: int,
    injected: dict[str, RhombiLoRALinear],
):
    """Save results, bridge LR log, and final bridge matrices to disk."""
    results = {
        "config": asdict(config),
        "experiment": "2.7",
        "checkpoints": checkpoints,
        "bridge_lr_log": bridge_lr_log,
        "trainable_ab_params": trainable_ab,
        "trainable_bridge_params": trainable_bridge,
        "total_params": total_params,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save final bridge matrices
    for name, lora in injected.items():
        safe_name = name.replace(".", "_")
        B = lora.bridge.detach().cpu().numpy()
        np.save(output_dir / f"bridge_final_{safe_name}.npy", B)


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2.7: Separate Bridge Learning Rate"
    )
    parser.add_argument(
        "--output", type=str, default="results/exp2_7",
        help="Output directory. Default: results/exp2_7",
    )
    parser.add_argument(
        "--bridge-lr-multiplier", type=float, default=10.0,
        help="Bridge LR = base_lr * multiplier. Default: 10.0",
    )
    parser.add_argument(
        "--bridge-lr-schedule", type=str, default="constant",
        choices=["constant", "warmup", "decay"],
        help=(
            "Bridge LR schedule. "
            "'constant': always multiplied. "
            "'warmup': high for N steps, then match base. "
            "'decay': start high, linearly decay to 1x base. "
            "Default: constant"
        ),
    )
    parser.add_argument(
        "--bridge-lr-warmup-steps", type=int, default=1000,
        help="Steps for warmup/decay bridge LR schedule. Default: 1000",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10_000,
        help="Max training steps. Default: 10000",
    )
    parser.add_argument(
        "--base-lr", type=float, default=2e-4,
        help="Base learning rate for A/B parameters. Default: 2e-4",
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, default=1000,
        help="Full checkpoint interval. Default: 1000",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42",
    )
    args = parser.parse_args()

    # Build config name from schedule parameters
    schedule_tag = args.bridge_lr_schedule
    mult_tag = f"{args.bridge_lr_multiplier:.0f}x"
    name = f"exp2_7_{schedule_tag}_{mult_tag}"

    config = SeparateLRConfig(
        name=name,
        base_lr=args.base_lr,
        bridge_lr_multiplier=args.bridge_lr_multiplier,
        bridge_lr_schedule=args.bridge_lr_schedule,
        bridge_lr_warmup_steps=args.bridge_lr_warmup_steps,
        max_steps=args.max_steps,
        checkpoint_steps=args.checkpoint_steps,
        seed=args.seed,
    )
    train(config, Path(args.output))


if __name__ == "__main__":
    main()
