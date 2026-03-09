"""Phase 1A: Task-Specific Bridge Fingerprints

Tests whether different training tasks produce measurably different bridge structures.

Three task types:
  A. General (Alpaca) — REUSE Exp 2 bridges (no training needed)
  B. Code (CodeAlpaca-20k) — 2K steps, FCC RhombiLoRA r=24
  C. Math (GSM8K) — 2K steps, FCC RhombiLoRA r=24

Same model (Qwen2.5-7B-Instruct), same architecture, same hyperparameters.
Only the data changes. If bridges differ by task → task fingerprinting story.

Pass criterion: within-task distance < between-task distance (paired test, p < 0.01)
Strong pass: classifier accuracy > 80%

Usage:
  python scripts/train_task_fingerprint.py --task code --output results/fingerprints
  python scripts/train_task_fingerprint.py --task math --output results/fingerprints
  python scripts/train_task_fingerprint.py --task all --output results/fingerprints
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
class TaskConfig:
    task_name: str
    dataset_name: str
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    rank: int = 24
    n_channels: int = 6
    max_seq_len: int = 512
    batch_size: int = 2
    gradient_accumulation: int = 8
    lr: float = 2e-4
    warmup_steps: int = 100
    checkpoint_steps: int = 500
    max_steps: int = 2000
    seed: int = 42
    gradient_checkpointing: bool = True
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


TASK_CONFIGS = {
    "code": TaskConfig(
        task_name="code",
        dataset_name="sahil2801/CodeAlpaca-20k",
    ),
    "math": TaskConfig(
        task_name="math",
        dataset_name="openai/gsm8k",
    ),
}


# ── Datasets ────────────────────────────────────────────────────────


class CodeAlpacaDataset(Dataset):
    """CodeAlpaca-20k in Alpaca instruction format."""

    def __init__(self, tokenizer, max_len: int = 512):
        from datasets import load_dataset

        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

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


class GSM8KDataset(Dataset):
    """GSM8K math problems in instruction format.

    Note: GSM8K has only 7473 training examples (vs 20K code, 52K Alpaca).
    With batch_size=2 and grad_accum=8, effective batch=16, so ~467 steps/epoch.
    2K steps = ~4.3 epochs. Acceptable for fingerprinting (we're measuring
    bridge structure, not task performance).
    """

    def __init__(self, tokenizer, max_len: int = 512):
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="train")

        self.examples = []
        for item in ds:
            question = item["question"]
            answer = item["answer"]

            prompt = f"### Instruction:\nSolve the following math problem step by step.\n\n{question}\n\n### Response:\n{answer}"
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


def get_dataset(task_name: str, tokenizer, max_len: int):
    if task_name == "code":
        return CodeAlpacaDataset(tokenizer, max_len)
    elif task_name == "math":
        return GSM8KDataset(tokenizer, max_len)
    else:
        raise ValueError(f"Unknown task: {task_name}")


# ── LoRA injection (reused from train_exp2_scale) ──────────────────


def inject_lora(model: nn.Module, config: TaskConfig) -> dict[str, RhombiLoRALinear]:
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
            bridge_mode="identity",
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


def collect_bridge_metrics(injected):
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


# ── Training loop ──────────────────────────────────────────────────


def train(config: TaskConfig, output_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\n{'='*70}")
    print(f"Phase 1A: Task Fingerprint — {config.task_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Steps: {config.max_steps}")
    print(f"{'='*70}\n")

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bridge_p = sum(l.bridge.numel() for l in injected.values() if l.bridge.requires_grad)
    print(f"Trainable: {trainable:,} | Bridge: {bridge_p:,}")

    print("Loading dataset...")
    dataset = get_dataset(config.task_name, tokenizer, config.max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    print(f"Dataset: {len(dataset)} examples, {len(dataloader)} batches/epoch")

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

    # Save step 0 bridges
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        np.save(output_dir / f"bridge_step0_{safe}.npy", lora.bridge.detach().cpu().numpy())

    checkpoints = []
    model.train()
    global_step = 0
    acc_loss = 0.0
    steps_since = 0
    start = time.time()

    done = False
    epoch = 0
    while not done:
        epoch += 1
        for batch_idx, batch in enumerate(dataloader):
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

                if global_step % 100 == 0:
                    elapsed = time.time() - start
                    avg = acc_loss / max(steps_since, 1)
                    sps = global_step / max(elapsed, 0.01)
                    eta = (config.max_steps - global_step) / max(sps, 0.01)
                    print(f"  Step {global_step:>5}/{config.max_steps} | "
                          f"Loss: {avg:.4f} | Speed: {sps:.2f} s/s | ETA: {eta/3600:.1f}h")

                if global_step % config.checkpoint_steps == 0 or global_step >= config.max_steps:
                    avg = acc_loss / max(steps_since, 1)
                    metrics = collect_bridge_metrics(injected)
                    elapsed = time.time() - start

                    cp = {
                        "step": global_step,
                        "train_loss": avg,
                        "wall_time": elapsed,
                        **metrics,
                    }
                    checkpoints.append(cp)

                    for name, lora in injected.items():
                        safe = name.replace(".", "_")
                        np.save(output_dir / f"bridge_step{global_step}_{safe}.npy",
                                lora.bridge.detach().cpu().numpy())

                    print(f"\n  *** Checkpoint {global_step} ***")
                    print(f"      Loss: {avg:.4f} | Fiedler: {metrics['fiedler_mean']:.5f}")
                    print(f"      Wall time: {elapsed/3600:.2f}h\n")

                    acc_loss = 0.0
                    steps_since = 0

                    # Save results
                    with open(output_dir / "results.json", "w") as f:
                        json.dump({"config": asdict(config), "checkpoints": checkpoints}, f, indent=2)

                    # Save final bridges
                    for name, lora in injected.items():
                        safe = name.replace(".", "_")
                        np.save(output_dir / f"bridge_final_{safe}.npy",
                                lora.bridge.detach().cpu().numpy())

                if global_step >= config.max_steps:
                    done = True

    elapsed = time.time() - start
    print(f"\nTask '{config.task_name}' complete in {elapsed/3600:.2f}h")
    print(f"Results: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1A: Task fingerprints")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name: 'code', 'math', or 'all'")
    parser.add_argument("--output", type=str, default="results/fingerprints",
                        help="Output directory")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Override max steps")
    args = parser.parse_args()

    if args.task == "all":
        tasks = list(TASK_CONFIGS.keys())
    else:
        tasks = [args.task]

    for task in tasks:
        config = TASK_CONFIGS[task]
        if args.max_steps > 0:
            config.max_steps = args.max_steps
        out = Path(args.output) / task
        train(config, out)


if __name__ == "__main__":
    main()
