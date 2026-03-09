"""Phase 3A: Overfit Diagnostic — Bridge Structure as Overfit Predictor

Tests whether bridge Fiedler/deviation correlates with the train-val loss gap
(overfitting signal). Deliberately overfits on 500 Alpaca examples for 10K steps,
checkpointing every 100 steps with full validation loss computation.

Pass criterion:
  - Bridge metric (Fiedler or deviation) correlates with train_val_gap (Pearson r > 0.7)
  - OR shows a phase transition (sudden structural change when overfitting begins)

Usage:
  python -u scripts/train_overfit_diagnostic.py --output results/overfit_diagnostic
  python -u scripts/train_overfit_diagnostic.py --output results/overfit_diagnostic --max-steps 2000
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


# ── Configuration ───────────────────────────────────────────────────


@dataclass
class OverfitConfig:
    name: str = "overfit_diagnostic"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    rank: int = 24
    n_channels: int = 6
    max_seq_len: int = 512
    batch_size: int = 2
    gradient_accumulation: int = 8
    lr: float = 2e-4
    warmup_steps: int = 100
    checkpoint_steps: int = 100
    max_steps: int = 10_000
    seed: int = 42
    gradient_checkpointing: bool = True
    train_samples: int = 500
    val_samples: int = 500
    bridge_mode: str = "identity"
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


# ── Dataset ─────────────────────────────────────────────────────────


class AlpacaSliceDataset(Dataset):
    """Alpaca-cleaned with explicit start/end indices for train/val split."""

    def __init__(self, tokenizer, max_len: int, start: int, end: int):
        from datasets import load_dataset

        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        ds = ds.select(range(start, min(end, len(ds))))

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


# ── LoRA injection ──────────────────────────────────────────────────


def inject_lora(model: nn.Module, config: OverfitConfig) -> dict[str, RhombiLoRALinear]:
    """Inject RhombiLoRA adapters into target modules with learnable bridge."""
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
        lora = lora.to(device=module.weight.device, dtype=torch.float32)
        injected[name] = lora
        _wrap_forward(model, name, module, lora)
    return injected


def _wrap_forward(model, name, original, lora):
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


# ── Metrics ─────────────────────────────────────────────────────────


def collect_bridge_metrics(injected: dict[str, RhombiLoRALinear]) -> dict:
    """Collect aggregated bridge Fiedler and deviation metrics."""
    fiedlers, deviations = [], []
    for lora in injected.values():
        fiedlers.append(lora.bridge_fiedler())
        deviations.append(lora.bridge_deviation())
    return {
        "fiedler_mean": float(np.mean(fiedlers)),
        "fiedler_std": float(np.std(fiedlers)),
        "deviation_mean": float(np.mean(deviations)),
        "deviation_std": float(np.std(deviations)),
    }


# ── Validation ──────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, dataloader, device) -> float:
    """Compute mean loss over the FULL validation set (no batch limit)."""
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in dataloader:
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


def train(config: OverfitConfig, output_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\n{'='*70}")
    print(f"Phase 3A: Overfit Diagnostic")
    print(f"Model: {config.model_name}")
    print(f"Train: {config.train_samples} examples | Val: {config.val_samples} examples")
    print(f"Steps: {config.max_steps} | Checkpoint every: {config.checkpoint_steps}")
    print(f"Rank: {config.rank}, Channels: {config.n_channels}, Bridge: {config.bridge_mode} (learnable)")
    print(f"Batch: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")
    print(f"LR: {config.lr}, Warmup: {config.warmup_steps}, Cosine schedule")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────
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
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ── Inject LoRA ─────────────────────────────────────────────────
    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bridge_p = sum(l.bridge.numel() for l in injected.values() if l.bridge.requires_grad)
    print(f"Trainable: {trainable:,} | Bridge: {bridge_p:,}")

    # ── Load datasets ───────────────────────────────────────────────
    print("Loading dataset (Alpaca-cleaned, first 1000 examples)...")
    train_dataset = AlpacaSliceDataset(
        tokenizer, config.max_seq_len, start=0, end=config.train_samples,
    )
    val_dataset = AlpacaSliceDataset(
        tokenizer, config.max_seq_len,
        start=config.train_samples,
        end=config.train_samples + config.val_samples,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    print(f"Train: {len(train_dataset)} examples, {len(train_dataloader)} batches/epoch")
    print(f"Val:   {len(val_dataset)} examples, {len(val_dataloader)} batches")

    eff_batch = config.batch_size * config.gradient_accumulation
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation
    total_epochs = config.max_steps / max(steps_per_epoch, 1)
    print(f"Effective batch: {eff_batch} | ~{steps_per_epoch} steps/epoch | ~{total_epochs:.0f} epochs")

    # ── Optimizer + scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=0.01,
    )

    def lr_schedule(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Save step 0 bridges ─────────────────────────────────────────
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        np.save(output_dir / f"bridge_step0_{safe}.npy", lora.bridge.detach().cpu().numpy())

    # ── Compute step 0 val loss and bridge metrics ──────────────────
    print("Computing initial validation loss...")
    init_val_loss = evaluate(model, val_dataloader, device)
    init_metrics = collect_bridge_metrics(injected)
    init_cp = {
        "step": 0,
        "train_loss": None,
        "val_loss": init_val_loss,
        "train_val_gap": None,
        **init_metrics,
    }
    checkpoints = [init_cp]
    print(f"  Step 0 — Val loss: {init_val_loss:.4f} | Fiedler: {init_metrics['fiedler_mean']:.5f}")

    # Save initial results
    with open(output_dir / "results.json", "w") as f:
        json.dump({"config": asdict(config), "checkpoints": checkpoints}, f, indent=2)

    # ── Training loop ───────────────────────────────────────────────
    model.train()
    global_step = 0
    acc_loss = 0.0
    steps_since = 0
    start = time.time()

    done = False
    epoch = 0
    while not done:
        epoch += 1
        for batch_idx, batch in enumerate(train_dataloader):
            if done:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.gradient_accumulation
            loss.backward()
            acc_loss += loss.item()

            if (batch_idx + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                steps_since += 1

                # Progress log every 50 steps (between checkpoints)
                if global_step % 50 == 0 and global_step % config.checkpoint_steps != 0:
                    elapsed = time.time() - start
                    avg = acc_loss / max(steps_since, 1)
                    sps = global_step / max(elapsed, 0.01)
                    eta = (config.max_steps - global_step) / max(sps, 0.01)
                    print(f"  Step {global_step:>5}/{config.max_steps} | "
                          f"Loss: {avg:.4f} | Speed: {sps:.2f} s/s | ETA: {eta/3600:.1f}h")

                # Checkpoint: save bridges, compute val loss, record metrics
                if global_step % config.checkpoint_steps == 0 or global_step >= config.max_steps:
                    train_loss_avg = acc_loss / max(steps_since, 1)

                    # Compute val loss over full validation set
                    val_loss = evaluate(model, val_dataloader, device)

                    # Bridge metrics
                    metrics = collect_bridge_metrics(injected)
                    elapsed = time.time() - start

                    train_val_gap = val_loss - train_loss_avg

                    cp = {
                        "step": global_step,
                        "train_loss": train_loss_avg,
                        "val_loss": val_loss,
                        "train_val_gap": train_val_gap,
                        **metrics,
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "wall_time": elapsed,
                    }
                    checkpoints.append(cp)

                    # Save bridge matrices
                    for name, lora in injected.items():
                        safe = name.replace(".", "_")
                        np.save(
                            output_dir / f"bridge_step{global_step}_{safe}.npy",
                            lora.bridge.detach().cpu().numpy(),
                        )

                    print(f"\n  *** Checkpoint {global_step} ***")
                    print(f"      Train loss: {train_loss_avg:.4f}")
                    print(f"      Val loss:   {val_loss:.4f}")
                    print(f"      Gap (val-train): {train_val_gap:+.4f}")
                    print(f"      Fiedler: {metrics['fiedler_mean']:.5f} +/- {metrics['fiedler_std']:.5f}")
                    print(f"      Deviation: {metrics['deviation_mean']:.5f} +/- {metrics['deviation_std']:.5f}")
                    print(f"      Epoch: {epoch} | Wall time: {elapsed/3600:.2f}h\n")

                    # Reset running averages
                    acc_loss = 0.0
                    steps_since = 0

                    # Save results (incremental — survives crashes)
                    with open(output_dir / "results.json", "w") as f:
                        json.dump(
                            {"config": asdict(config), "checkpoints": checkpoints},
                            f, indent=2,
                        )

                if global_step >= config.max_steps:
                    done = True

    # ── Save final bridges ──────────────────────────────────────────
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        np.save(output_dir / f"bridge_final_{safe}.npy", lora.bridge.detach().cpu().numpy())

    elapsed = time.time() - start

    # ── Post-training correlation analysis ──────────────────────────
    print(f"\n{'='*70}")
    print(f"Post-Training Correlation Analysis")
    print(f"{'='*70}\n")

    # Filter checkpoints with valid train_loss (skip step 0 which has None)
    valid_cps = [cp for cp in checkpoints if cp["train_loss"] is not None]
    if len(valid_cps) >= 5:
        from scipy import stats as sp_stats

        gaps = [cp["train_val_gap"] for cp in valid_cps]
        fiedlers = [cp["fiedler_mean"] for cp in valid_cps]
        deviations = [cp["deviation_mean"] for cp in valid_cps]
        steps = [cp["step"] for cp in valid_cps]

        # Pearson correlations
        r_fiedler, p_fiedler = sp_stats.pearsonr(gaps, fiedlers)
        r_deviation, p_deviation = sp_stats.pearsonr(gaps, deviations)

        print(f"  Checkpoints analyzed: {len(valid_cps)}")
        print(f"  Gap range: [{min(gaps):.4f}, {max(gaps):.4f}]")
        print(f"  Fiedler range: [{min(fiedlers):.5f}, {max(fiedlers):.5f}]")
        print(f"  Deviation range: [{min(deviations):.5f}, {max(deviations):.5f}]")
        print()
        print(f"  Fiedler ~ Gap:    r = {r_fiedler:+.4f}  (p = {p_fiedler:.4e})")
        print(f"  Deviation ~ Gap:  r = {r_deviation:+.4f}  (p = {p_deviation:.4e})")
        print()

        # Pass criterion
        pass_fiedler = abs(r_fiedler) > 0.7
        pass_deviation = abs(r_deviation) > 0.7
        if pass_fiedler or pass_deviation:
            winner = "Fiedler" if abs(r_fiedler) > abs(r_deviation) else "Deviation"
            print(f"  PASS: {winner} correlates with overfit gap (r > 0.7)")
        else:
            print(f"  NO PASS on correlation criterion (both |r| < 0.7)")

        # Phase transition detection: look for sudden jumps in bridge metrics
        # Compare max step-to-step change in Fiedler to median change
        if len(fiedlers) >= 3:
            fiedler_diffs = [abs(fiedlers[i+1] - fiedlers[i]) for i in range(len(fiedlers)-1)]
            median_diff = float(np.median(fiedler_diffs))
            max_diff = max(fiedler_diffs)
            max_diff_step = steps[fiedler_diffs.index(max_diff) + 1]
            ratio = max_diff / median_diff if median_diff > 1e-10 else float('inf')

            print(f"\n  Phase transition check (Fiedler):")
            print(f"    Median step-to-step change: {median_diff:.6f}")
            print(f"    Max step-to-step change:    {max_diff:.6f} (at step {max_diff_step})")
            print(f"    Ratio (max/median):         {ratio:.1f}x")
            if ratio > 5.0:
                print(f"    PHASE TRANSITION detected at step {max_diff_step} ({ratio:.1f}x median)")
            else:
                print(f"    No clear phase transition (ratio < 5x)")

        # Save correlation results
        correlation_results = {
            "n_checkpoints": len(valid_cps),
            "fiedler_gap_pearson_r": r_fiedler,
            "fiedler_gap_pearson_p": p_fiedler,
            "deviation_gap_pearson_r": r_deviation,
            "deviation_gap_pearson_p": p_deviation,
            "pass_fiedler": pass_fiedler,
            "pass_deviation": pass_deviation,
        }
        with open(output_dir / "correlation_results.json", "w") as f:
            json.dump(correlation_results, f, indent=2)
    else:
        print(f"  Insufficient checkpoints ({len(valid_cps)}) for correlation analysis")

    # Final save with all data
    with open(output_dir / "results.json", "w") as f:
        json.dump({"config": asdict(config), "checkpoints": checkpoints}, f, indent=2)

    print(f"\nOverfit diagnostic complete in {elapsed/3600:.2f}h")
    print(f"Checkpoints: {len(checkpoints)} (incl. step 0)")
    print(f"Results: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3A: Overfit diagnostic — bridge structure as overfit predictor"
    )
    parser.add_argument(
        "--output", type=str, default="results/overfit_diagnostic",
        help="Output directory (default: results/overfit_diagnostic)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Override max steps (default: 10000)",
    )
    args = parser.parse_args()

    config = OverfitConfig()
    if args.max_steps > 0:
        config.max_steps = args.max_steps

    train(config, Path(args.output))


if __name__ == "__main__":
    main()
