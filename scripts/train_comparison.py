"""Experiment 1: RhombiLoRA vs Standard LoRA training comparison.

Five configurations on Qwen2.5-1.5B-Instruct with Alpaca-cleaned:
  1. Standard LoRA rank 24 (control)
  2. RhombiLoRA rank 24, bridge frozen at identity (sanity — must match #1)
  3. RhombiLoRA rank 24, bridge learnable, identity init
  4. Standard LoRA rank 48 (parameter-budget control)
  5. RhombiLoRA rank 24, 3 channels (cubic ablation)

Metrics at each checkpoint (every 500 steps):
  - Training loss, validation perplexity
  - Bridge matrix snapshot (6x6 heatmap)
  - Bridge Fiedler value
  - Bridge deviation from identity
  - Gradient effective rank through the bottleneck

Usage:
  python scripts/train_comparison.py --config 3 --output results/exp1
  python scripts/train_comparison.py --config all --output results/exp1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rhombic.nn.rhombi_lora import RhombiLoRALinear


# ── Configuration ───────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    name: str
    rank: int
    n_channels: int
    bridge_mode: str
    bridge_trainable: bool
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name: str = "yahma/alpaca-cleaned"
    max_seq_len: int = 512
    batch_size: int = 4
    gradient_accumulation: int = 4
    lr: float = 2e-4
    num_epochs: int = 1
    warmup_steps: int = 100
    checkpoint_steps: int = 500
    max_steps: int = 0  # 0 = full epoch
    seed: int = 42
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )


CONFIGS: dict[int, ExperimentConfig] = {
    1: ExperimentConfig(
        name="standard_lora_r24",
        rank=24, n_channels=6,
        bridge_mode="identity", bridge_trainable=False,
    ),
    2: ExperimentConfig(
        name="rhombi_frozen_r24",
        rank=24, n_channels=6,
        bridge_mode="identity", bridge_trainable=False,
    ),
    3: ExperimentConfig(
        name="rhombi_learnable_r24",
        rank=24, n_channels=6,
        bridge_mode="identity", bridge_trainable=True,
    ),
    4: ExperimentConfig(
        name="standard_lora_r48",
        rank=48, n_channels=6,
        bridge_mode="identity", bridge_trainable=False,
    ),
    5: ExperimentConfig(
        name="rhombi_cubic_r24",
        rank=24, n_channels=3,
        bridge_mode="identity", bridge_trainable=True,
    ),
}


# ── PEFT-style injection ───────────────────────────────────────────


def inject_lora(model: nn.Module, config: ExperimentConfig) -> dict[str, RhombiLoRALinear]:
    """Replace target linear layers with LoRA-wrapped versions.

    For configs 1 and 4 (standard LoRA), we use RhombiLoRA with frozen
    identity bridge — mathematically identical to standard LoRA.

    Returns dict of injected module names -> RhombiLoRALinear instances.
    """
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
            bridge_mode=config.bridge_mode,
        )

        if not config.bridge_trainable:
            lora.freeze_bridge()

        # Move adapter to same device/dtype as the base layer
        lora = lora.to(device=module.weight.device, dtype=torch.float32)

        injected[name] = lora

        # Replace the forward to add LoRA delta
        _wrap_forward(model, name, module, lora)

    return injected


def _wrap_forward(
    model: nn.Module, name: str, original: nn.Linear, lora: RhombiLoRALinear
):
    """Monkey-patch the module's forward to add LoRA output."""
    parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
    parent = model.get_submodule(parent_name) if parent_name else model

    class LoRAWrappedLinear(nn.Module):
        def __init__(self, base: nn.Linear, adapter: RhombiLoRALinear):
            super().__init__()
            self.base = base
            self.adapter = adapter
            # Freeze base weights
            for p in self.base.parameters():
                p.requires_grad = False

        def forward(self, x):
            return self.base(x) + self.adapter(x)

    wrapped = LoRAWrappedLinear(original, lora)
    setattr(parent, attr_name, wrapped)


# ── Metrics ─────────────────────────────────────────────────────────


@dataclass
class CheckpointMetrics:
    step: int
    train_loss: float
    val_perplexity: float | None
    bridge_matrices: dict[str, list[list[float]]]  # name -> 6x6
    bridge_fiedler: dict[str, float]
    bridge_deviation: dict[str, float]
    grad_effective_rank: dict[str, float]
    wall_time: float


def collect_bridge_metrics(
    injected: dict[str, RhombiLoRALinear],
) -> tuple[dict, dict, dict]:
    """Collect bridge snapshot, Fiedler, and deviation from all adapters."""
    matrices = {}
    fiedlers = {}
    deviations = {}
    for name, lora in injected.items():
        B = lora.bridge.detach().cpu().numpy()
        matrices[name] = B.tolist()
        fiedlers[name] = lora.bridge_fiedler()
        deviations[name] = lora.bridge_deviation()
    return matrices, fiedlers, deviations


def gradient_effective_rank(
    injected: dict[str, RhombiLoRALinear],
) -> dict[str, float]:
    """Effective rank of gradient through the bottleneck (via SVD)."""
    result = {}
    for name, lora in injected.items():
        if lora.lora_A.grad is None:
            result[name] = 0.0
            continue
        with torch.no_grad():
            G = lora.lora_A.grad.float()
            s = torch.linalg.svdvals(G)
            s = s / (s.sum() + 1e-12)
            entropy = -(s * torch.log(s + 1e-12)).sum()
            result[name] = float(torch.exp(entropy).item())
    return result


# ── Dataset ─────────────────────────────────────────────────────────


class AlpacaDataset(Dataset):
    """Minimal Alpaca dataset wrapper."""

    def __init__(self, tokenizer, max_len: int = 512, max_samples: int = 0):
        from datasets import load_dataset

        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        self.examples = []
        for item in ds:
            instruction = item["instruction"]
            inp = item.get("input", "")
            output = item["output"]

            if inp:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
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


# ── Training loop ──────────────────────────────────────────────────


def train(config: ExperimentConfig, output_dir: Path):
    """Run one experiment configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment: {config.name}")
    print(f"Rank: {config.rank}, Channels: {config.n_channels}")
    print(f"Bridge: {config.bridge_mode}, Trainable: {config.bridge_trainable}")
    print(f"{'='*60}\n")

    # Load model
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
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA
    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.4f}%)")

    # Dataset
    print("Loading dataset...")
    dataset = AlpacaDataset(
        tokenizer, max_len=config.max_seq_len,
        max_samples=config.max_steps * config.batch_size * config.gradient_accumulation
        if config.max_steps > 0 else 0,
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    # Optimizer — only trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=0.01,
    )

    # LR schedule with linear warmup
    total_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation
    if config.max_steps > 0:
        total_steps = min(total_steps, config.max_steps)

    def lr_schedule(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
        )
        return max(0.1, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training
    checkpoints: list[CheckpointMetrics] = []
    model.train()
    global_step = 0
    accumulation_loss = 0.0
    start_time = time.time()

    print(f"Training for {total_steps} steps...")

    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
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
            accumulation_loss += loss.item()

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Checkpoint
                if global_step % config.checkpoint_steps == 0 or global_step == total_steps:
                    matrices, fiedlers, deviations = collect_bridge_metrics(injected)
                    eff_ranks = gradient_effective_rank(injected)

                    cp = CheckpointMetrics(
                        step=global_step,
                        train_loss=accumulation_loss / config.checkpoint_steps,
                        val_perplexity=None,  # TODO: validation loop
                        bridge_matrices=matrices,
                        bridge_fiedler=fiedlers,
                        bridge_deviation=deviations,
                        grad_effective_rank=eff_ranks,
                        wall_time=time.time() - start_time,
                    )
                    checkpoints.append(cp)

                    print(
                        f"  Step {global_step}/{total_steps} | "
                        f"Loss: {cp.train_loss:.4f} | "
                        f"Fiedler: {list(fiedlers.values())[0]:.4f} | "
                        f"Dev: {list(deviations.values())[0]:.4f} | "
                        f"Time: {cp.wall_time:.0f}s"
                    )
                    accumulation_loss = 0.0

                if config.max_steps > 0 and global_step >= config.max_steps:
                    break

        if config.max_steps > 0 and global_step >= config.max_steps:
            break

    # Save results
    results = {
        "config": asdict(config),
        "checkpoints": [asdict(cp) for cp in checkpoints],
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save final bridge matrices as numpy
    for name, lora in injected.items():
        safe_name = name.replace(".", "_")
        B = lora.bridge.detach().cpu().numpy()
        np.save(output_dir / f"bridge_final_{safe_name}.npy", B)

    print(f"\nResults saved to {output_dir}")
    return results


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="RhombiLoRA vs Standard LoRA training comparison"
    )
    parser.add_argument(
        "--config", type=str, default="3",
        help="Config number (1-5) or 'all'. Default: 3 (learnable bridge)",
    )
    parser.add_argument(
        "--output", type=str, default="results/exp1",
        help="Output directory. Default: results/exp1",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Max training steps (0 = full epoch). Default: 0",
    )
    args = parser.parse_args()

    output_base = Path(args.output)

    if args.config == "all":
        config_ids = list(CONFIGS.keys())
    else:
        config_ids = [int(args.config)]

    for cid in config_ids:
        config = CONFIGS[cid]
        if args.max_steps > 0:
            config.max_steps = args.max_steps
        out = output_base / config.name
        train(config, out)


if __name__ == "__main__":
    main()
