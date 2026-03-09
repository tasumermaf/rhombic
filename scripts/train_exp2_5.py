"""Experiment 2.5: Geometric Dataset Validation.

Tests whether purpose-built training data with directional structure produces
stronger co-planar vs cross-planar signal in the RhombiLoRA bridge.

Hypothesis: Alpaca-cleaned is isotropic (co/cross = 1.02×). The geometric
dataset should push this above 1.10× if directional data teaches directional
structure.

Runs the same 3 configs as Exp 2 (standard LoRA, FCC, cubic) but with:
  - Geometric dataset only
  - Geometric 50/50 mixed with Alpaca
  - Alpaca only (control — reproduces Exp 2)

Usage:
  # Full 9-config sweep (3 topologies × 3 datasets)
  python scripts/train_exp2_5.py --output results/exp2_5

  # Quick validation: FCC only on geometric data, 2K steps
  python scripts/train_exp2_5.py --config 2 --dataset-mode geometric --max-steps 2000

  # Single config with custom dataset file
  python scripts/train_exp2_5.py --config 2 --dataset-path data/geometric/geometric_combined.json
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

# Reuse everything from Exp 2
from train_exp2_scale import (
    ExperimentConfig,
    CONFIGS,
    inject_lora,
    collect_metrics,
    gradient_effective_rank,
    coplanar_crossplanar_ratio,
    CheckpointMetrics,
    evaluate,
    _save_results,
)


# ── Local JSON Dataset ────────────────────────────────────────────────


class LocalJSONDataset(Dataset):
    """Load Alpaca-format JSON from a local file.

    Expects list of {"instruction": ..., "input": ..., "output": ...}
    Same format as yahma/alpaca-cleaned.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_len: int = 512,
        max_samples: int = 0,
    ):
        with open(path) as f:
            data = json.load(f)

        if max_samples > 0:
            data = data[:max_samples]

        self.examples = []
        for item in data:
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


# ── Dataset Modes ─────────────────────────────────────────────────────


DATASET_MODES = {
    "geometric": "Geometric dataset only (Components 1+3+4)",
    "mixed50": "50% geometric + 50% Alpaca",
    "alpaca": "Alpaca only (Exp 2 control reproduction)",
}


def get_dataset_path(mode: str, data_dir: Path) -> Path | str:
    """Resolve dataset path for a given mode."""
    if mode == "geometric":
        return data_dir / "geometric_combined.json"
    elif mode == "mixed50":
        return data_dir / "mixed_geo50_alpaca50.json"
    elif mode == "alpaca":
        return "alpaca"  # sentinel for HuggingFace loader
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")


# ── Training (adapted from Exp 2) ────────────────────────────────────


def train_exp25(
    config: ExperimentConfig,
    output_dir: Path,
    dataset_path: str | Path,
    val_dataset_path: str | Path | None = None,
):
    """Run one Exp 2.5 configuration with a custom dataset."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config with dataset info
    config_dict = asdict(config)
    config_dict["dataset_path"] = str(dataset_path)
    config_dict["experiment"] = "2.5"
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Experiment 2.5: {config.name}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Rank: {config.rank}, Channels: {config.n_channels}")
    print(f"Bridge: {config.bridge_mode}, Trainable: {config.bridge_trainable}")
    print(f"Steps: {config.max_steps}")
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
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    bridge_params = sum(
        lora.bridge.numel() for lora in injected.values()
        if lora.bridge.requires_grad
    )
    print(f"Trainable: {trainable_params:,} / {total_params:,}")
    print(f"Bridge params: {bridge_params:,}")

    # Load dataset — local JSON or Alpaca from HuggingFace
    print(f"Loading dataset from {dataset_path}...")
    if str(dataset_path) == "alpaca":
        from train_exp2_scale import AlpacaDataset
        dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len)
        val_dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len, max_samples=1000)
    else:
        if val_dataset_path:
            dataset = LocalJSONDataset(dataset_path, tokenizer, max_len=config.max_seq_len)
            val_dataset = LocalJSONDataset(
                val_dataset_path, tokenizer, max_len=config.max_seq_len, max_samples=1000
            )
        else:
            # Split: last 1000 examples for validation, rest for training
            with open(dataset_path) as f:
                all_data = json.load(f)
            val_size = min(1000, len(all_data) // 10)
            train_data = all_data[:-val_size]
            val_data = all_data[-val_size:]
            # Write split files to temp dir
            import tempfile
            split_dir = Path(tempfile.mkdtemp(prefix="rhombic_split_"))
            train_path = split_dir / "train.json"
            val_path = split_dir / "val.json"
            with open(train_path, "w") as f:
                json.dump(train_data, f)
            with open(val_path, "w") as f:
                json.dump(val_data, f)
            dataset = LocalJSONDataset(train_path, tokenizer, max_len=config.max_seq_len)
            val_dataset = LocalJSONDataset(val_path, tokenizer, max_len=config.max_seq_len)
            print(f"Split: {len(train_data)} train / {len(val_data)} val")

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
        progress = (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    checkpoints: list[dict] = []
    model.train()
    global_step = 0
    accumulation_loss = 0.0
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

                if global_step % config.checkpoint_steps == 0 or global_step >= total_steps:
                    avg_loss = accumulation_loss / steps_since_ckpt
                    f_mean, f_std, d_mean, d_std, ccp = collect_metrics(injected)
                    eff_rank = gradient_effective_rank(injected)
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

                    for name, lora in injected.items():
                        safe_name = name.replace(".", "_")
                        B = lora.bridge.detach().cpu().numpy()
                        np.save(output_dir / f"bridge_step{global_step}_{safe_name}.npy", B)

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

                    _save_results(
                        output_dir, config, checkpoints,
                        trainable_params, total_params, bridge_params,
                        injected,
                    )

                if global_step >= total_steps:
                    done = True

    _save_results(
        output_dir, config, checkpoints,
        trainable_params, total_params, bridge_params,
        injected,
    )

    elapsed = time.time() - start_time
    print(f"\nExperiment complete: {config.name}")
    print(f"Total time: {elapsed/3600:.2f}h")
    return checkpoints


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2.5: Geometric Dataset Validation"
    )
    parser.add_argument(
        "--config", type=str, default="2",
        help="Config: 1 (standard), 2 (FCC), 3 (cubic), or 'all'. Default: 2",
    )
    parser.add_argument(
        "--dataset-mode", type=str, default=None,
        choices=["geometric", "mixed50", "alpaca"],
        help="Dataset mode. Overridden by --dataset-path.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to local JSON dataset. Overrides --dataset-mode.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/geometric",
        help="Directory containing generated geometric datasets.",
    )
    parser.add_argument(
        "--output", type=str, default="results/exp2_5",
        help="Output directory.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Override max training steps.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run full 9-config sweep (3 topologies × 3 datasets).",
    )
    args = parser.parse_args()

    output_base = Path(args.output)
    data_dir = Path(args.data_dir)

    if args.config == "all":
        config_ids = [1, 2, 3]
    else:
        config_ids = [int(args.config)]

    if args.sweep:
        # Full sweep: 3 topologies × 3 datasets
        modes = ["geometric", "mixed50", "alpaca"]
        config_ids = [1, 2, 3]
    elif args.dataset_path:
        modes = [None]
    elif args.dataset_mode:
        modes = [args.dataset_mode]
    else:
        modes = ["geometric"]  # Default

    for mode in modes:
        for cid in config_ids:
            config = CONFIGS[cid]
            if args.max_steps > 0:
                config.max_steps = args.max_steps

            if args.dataset_path:
                ds_path = args.dataset_path
                ds_label = Path(args.dataset_path).stem
            elif mode:
                ds_path = get_dataset_path(mode, data_dir)
                ds_label = mode
            else:
                continue

            out = output_base / f"{ds_label}_{config.name}"
            print(f"\n{'='*70}")
            print(f"Config: {config.name} | Dataset: {ds_label}")
            print(f"Output: {out}")
            print(f"{'='*70}")

            train_exp25(config, out, ds_path)

    print(f"\n{'='*70}")
    print(f"All Exp 2.5 runs complete.")
    print(f"Results in: {output_base}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
