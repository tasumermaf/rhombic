"""Experiment 2: RhombiLoRA at scale — 7B model, 10K steps, 4 target modules.

Three configurations on Qwen2.5-7B-Instruct with Alpaca-cleaned:
  1. Standard LoRA rank 24 (control — frozen identity bridge)
  2. RhombiLoRA rank 24, 6 channels, bridge learnable (FCC thesis)
  3. RhombiLoRA rank 24, 3 channels, bridge learnable (cubic ablation)

Changes from Experiment 1:
  - 7B model (28 layers, 28 heads) vs 1.5B (24 layers, 12 heads)
  - 10K steps (5x longer) — let bridge dynamics saturate
  - q/k/v/o_proj targets (4 per layer) — 112 adapters for statistical power
  - Gradient checkpointing enabled — fits in 48GB
  - Co-planar vs cross-planar coupling tracked per checkpoint

Goals:
  - Resolve the weak co-planar/cross-planar signal (r=0.31 in Exp 1)
  - See if bridge structure produces measurable loss improvement at scale
  - Provide data foundation for MoE routing and cross-modal experiments

Usage:
  python scripts/train_exp2_scale.py --config 2 --output results/exp2
  python scripts/train_exp2_scale.py --config all --output results/exp2
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


# ── Configuration ───────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    name: str
    rank: int
    n_channels: int
    bridge_mode: str
    bridge_trainable: bool
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_name: str = "yahma/alpaca-cleaned"
    max_seq_len: int = 512
    batch_size: int = 2
    gradient_accumulation: int = 8
    lr: float = 2e-4
    num_epochs: int = 4
    warmup_steps: int = 200
    checkpoint_steps: int = 1000
    max_steps: int = 10000
    seed: int = 42
    gradient_checkpointing: bool = True
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    dynamic_bridge: bool = False
    gate_temperature_start: float = 5.0
    gate_temperature_end: float = 0.1


CONFIGS: dict[int, ExperimentConfig] = {
    1: ExperimentConfig(
        name="standard_lora_r24",
        rank=24, n_channels=6,
        bridge_mode="identity", bridge_trainable=False,
    ),
    2: ExperimentConfig(
        name="rhombi_fcc_r24",
        rank=24, n_channels=6,
        bridge_mode="identity", bridge_trainable=True,
    ),
    3: ExperimentConfig(
        name="rhombi_cubic_r24",
        rank=24, n_channels=3,
        bridge_mode="identity", bridge_trainable=True,
    ),
}


# ── Co-planar / Cross-planar analysis ─────────────────────────────


# Precompute which pairs are co-planar vs cross-planar from RD geometry
_COUPLING = direction_pair_coupling()


def _coplanar_crossplanar_indices(n_channels: int):
    """Return indices of co-planar and cross-planar pairs for the bridge.

    For n_channels=6: co-planar pairs share 4 octahedral vertices (RD geometry),
    cross-planar share 2. 3 co-planar pairs, 12 cross-planar.
    For n_channels=8: co-axial pairs along tesseract coordinate axes,
    (0,1)=±w, (2,3)=±x, (4,5)=±y, (6,7)=±z. 4 co-axial, 24 cross-axial.
    For other n_channels: returns None (geometric analysis not applicable).
    """
    if n_channels == 6:
        coplanar = []
        crossplanar = []
        for i in range(6):
            for j in range(i + 1, 6):
                if _COUPLING[i, j] >= 4:
                    coplanar.append((i, j))
                else:
                    crossplanar.append((i, j))
        return coplanar, crossplanar
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
        return None, None


def coplanar_crossplanar_ratio(bridge: np.ndarray) -> dict | None:
    """Compute co-planar vs cross-planar coupling ratio for a 6x6 bridge.

    Returns dict with mean co-planar, mean cross-planar, ratio, and
    individual pair values. Returns None for non-6x6 bridges.
    """
    n = bridge.shape[0]
    coplanar_idx, crossplanar_idx = _coplanar_crossplanar_indices(n)
    if coplanar_idx is None:
        return None

    coplanar_vals = [abs(bridge[i, j]) for i, j in coplanar_idx]
    crossplanar_vals = [abs(bridge[i, j]) for i, j in crossplanar_idx]

    mean_co = np.mean(coplanar_vals) if coplanar_vals else 0.0
    mean_cross = np.mean(crossplanar_vals) if crossplanar_vals else 0.0
    ratio = mean_co / mean_cross if mean_cross > 1e-12 else float('inf')

    return {
        "mean_coplanar": float(mean_co),
        "mean_crossplanar": float(mean_cross),
        "ratio": float(ratio),
        "coplanar_pairs": list(coplanar_idx),
        "crossplanar_pairs": list(crossplanar_idx),
        "coplanar_values": [float(v) for v in coplanar_vals],
        "crossplanar_values": [float(v) for v in crossplanar_vals],
    }


# ── PEFT-style injection ───────────────────────────────────────────


def inject_lora(model: nn.Module, config: ExperimentConfig) -> dict[str, RhombiLoRALinear]:
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
            bridge_mode=config.bridge_mode,
            dynamic_bridge=config.dynamic_bridge,
            gate_temperature=config.gate_temperature_start,
        )

        if not config.bridge_trainable:
            lora.freeze_bridge()

        lora = lora.to(device=module.weight.device, dtype=torch.float32)
        injected[name] = lora
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
    val_loss: float | None
    bridge_fiedler_mean: float
    bridge_fiedler_std: float
    bridge_deviation_mean: float
    bridge_deviation_std: float
    coplanar_crossplanar: dict | None  # aggregated across all 6-ch adapters
    grad_effective_rank_mean: float
    n_adapters: int
    wall_time: float


def collect_metrics(
    injected: dict[str, RhombiLoRALinear],
) -> tuple[float, float, float, float, dict | None]:
    """Collect aggregated bridge metrics across all adapters."""
    fiedlers = []
    deviations = []
    all_ratios = []

    for name, lora in injected.items():
        fiedlers.append(lora.bridge_fiedler())
        deviations.append(lora.bridge_deviation())

        B = lora.bridge.detach().cpu().numpy()
        ratio_info = coplanar_crossplanar_ratio(B)
        if ratio_info is not None:
            all_ratios.append(ratio_info)

    fiedler_mean = float(np.mean(fiedlers))
    fiedler_std = float(np.std(fiedlers))
    dev_mean = float(np.mean(deviations))
    dev_std = float(np.std(deviations))

    # Aggregate co-planar/cross-planar across adapters
    agg_ccp = None
    if all_ratios:
        all_co = [r["mean_coplanar"] for r in all_ratios]
        all_cross = [r["mean_crossplanar"] for r in all_ratios]
        mean_co = float(np.mean(all_co))
        mean_cross = float(np.mean(all_cross))
        agg_ccp = {
            "mean_coplanar": mean_co,
            "mean_crossplanar": mean_cross,
            "ratio": float(mean_co / mean_cross) if mean_cross > 1e-12 else float('inf'),
            "n_adapters": len(all_ratios),
            "coplanar_std": float(np.std(all_co)),
            "crossplanar_std": float(np.std(all_cross)),
            # Correlation with geometric prediction
            "per_adapter_ratios": [r["ratio"] for r in all_ratios],
        }

    return fiedler_mean, fiedler_std, dev_mean, dev_std, agg_ccp


def gradient_effective_rank(
    injected: dict[str, RhombiLoRALinear],
) -> float:
    """Mean effective rank of gradient across all adapters."""
    ranks = []
    for name, lora in injected.items():
        if lora.lora_A.grad is None:
            continue
        with torch.no_grad():
            G = lora.lora_A.grad.float()
            s = torch.linalg.svdvals(G)
            s = s / (s.sum() + 1e-12)
            entropy = -(s * torch.log(s + 1e-12)).sum()
            ranks.append(float(torch.exp(entropy).item()))
    return float(np.mean(ranks)) if ranks else 0.0


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


# ── Validation ──────────────────────────────────────────────────────


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


# ── Training loop ──────────────────────────────────────────────────


def train(config: ExperimentConfig, output_dir: Path):
    """Run one experiment configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\n{'='*70}")
    print(f"Experiment 2: {config.name}")
    print(f"Model: {config.model_name}")
    print(f"Rank: {config.rank}, Channels: {config.n_channels}")
    print(f"Bridge: {config.bridge_mode}, Trainable: {config.bridge_trainable}")
    print(f"Targets: {config.target_modules}")
    print(f"Steps: {config.max_steps}, Checkpoint every: {config.checkpoint_steps}")
    print(f"Gradient checkpointing: {config.gradient_checkpointing}")
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

    # Enable gradient checkpointing to save VRAM
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required for gradient checkpointing with LoRA
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    bridge_params = sum(
        lora.bridge.numel() for lora in injected.values()
        if lora.bridge.requires_grad
    )
    print(f"Trainable: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.4f}%)")
    print(f"Bridge params: {bridge_params:,} "
          f"({100*bridge_params/trainable_params:.2f}% of trainable)")

    # Dataset — load full dataset, we'll loop through epochs
    print("Loading dataset...")
    dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # Validation split — use last 1000 examples
    val_dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len, max_samples=1000)
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Optimizer
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

    # Training
    checkpoints: list[dict] = []
    model.train()
    global_step = 0
    accumulation_loss = 0.0
    steps_since_ckpt = 0
    start_time = time.time()

    print(f"\nTraining for {total_steps} steps...")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation}")
    print(f"Dataset: {len(dataset)} examples, {len(dataloader)} batches/epoch")
    print()

    # Save bridge matrices at step 0 for comparison
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
                steps_since_ckpt += 1

                # Progress every 100 steps
                if global_step % 100 == 0:
                    avg_loss = accumulation_loss / steps_since_ckpt
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    eta = (total_steps - global_step) / steps_per_sec
                    print(
                        f"  Step {global_step:>5}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Speed: {steps_per_sec:.2f} step/s | "
                        f"ETA: {eta/3600:.1f}h"
                    )

                # Checkpoint
                if global_step % config.checkpoint_steps == 0 or global_step >= total_steps:
                    avg_loss = accumulation_loss / steps_since_ckpt

                    # Bridge metrics
                    f_mean, f_std, d_mean, d_std, ccp = collect_metrics(injected)
                    eff_rank = gradient_effective_rank(injected)

                    # Validation
                    val_loss = evaluate(model, val_dataloader, device)

                    cp = CheckpointMetrics(
                        step=global_step,
                        train_loss=avg_loss,
                        val_loss=val_loss,
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

                    # Save bridge matrices at this checkpoint
                    for name, lora in injected.items():
                        safe_name = name.replace(".", "_")
                        B = lora.bridge.detach().cpu().numpy()
                        np.save(
                            output_dir / f"bridge_step{global_step}_{safe_name}.npy",
                            B,
                        )

                    ccp_str = ""
                    if ccp is not None:
                        ccp_str = f" | Co/Cross: {ccp['ratio']:.3f}"

                    print(
                        f"\n  *** Checkpoint {global_step}/{total_steps} ***\n"
                        f"      Train loss: {avg_loss:.4f}\n"
                        f"      Val loss:   {val_loss:.4f}\n"
                        f"      Fiedler:    {f_mean:.5f} ± {f_std:.5f}\n"
                        f"      Deviation:  {d_mean:.5f} ± {d_std:.5f}\n"
                        f"      Eff rank:   {eff_rank:.2f}{ccp_str}\n"
                        f"      Wall time:  {cp.wall_time/3600:.2f}h\n"
                    )

                    accumulation_loss = 0.0
                    steps_since_ckpt = 0

                    # Save intermediate results
                    _save_results(
                        output_dir, config, checkpoints,
                        trainable_params, total_params, bridge_params,
                        injected,
                    )

                if global_step >= total_steps:
                    done = True

    # Final save
    _save_results(
        output_dir, config, checkpoints,
        trainable_params, total_params, bridge_params,
        injected,
    )

    elapsed = time.time() - start_time
    print(f"\nExperiment complete: {config.name}")
    print(f"Total time: {elapsed/3600:.2f}h")
    print(f"Results saved to {output_dir}")
    return checkpoints


def _save_results(
    output_dir, config, checkpoints,
    trainable_params, total_params, bridge_params,
    injected,
):
    """Save results and bridge matrices to disk."""
    results = {
        "config": asdict(config),
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


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: RhombiLoRA at scale"
    )
    parser.add_argument(
        "--config", type=str, default="2",
        help="Config number (1-3) or 'all'. Default: 2 (FCC learnable)",
    )
    parser.add_argument(
        "--output", type=str, default="results/exp2",
        help="Output directory. Default: results/exp2",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Override max training steps. Default: use config value",
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
