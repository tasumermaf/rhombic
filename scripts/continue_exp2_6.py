#!/usr/bin/env python3
"""Continue Exp 2.6 training from saved bridge weights.

Loads the model, injects LoRA, overwrites bridges with the numpy arrays
saved at the latest checkpoint, then continues training for additional steps.
Writes to the SAME training log and results file for seamless continuation.

Usage:
    python scripts/continue_exp2_6.py --extra-steps 10000
    python scripts/continue_exp2_6.py --extra-steps 20000 --lr 1e-4
"""
import sys
sys.path.insert(0, "/workspace")

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_exp2_scale import ExperimentConfig
from scripts.train_contrastive_bridge import (
    _CO_PLANAR_PAIRS,
    _CROSS_PLANAR_PAIRS,
    ContrastiveCheckpointMetrics,
    coplanar_crossplanar_ratio,
    _save_contrastive_results,
)
from rhombic.nn import RhombiLoRALinear, inject_lora, collect_metrics
from rhombic.nn.experiment import (
    AlpacaDataset,
    gradient_effective_rank,
    evaluate,
)


def main():
    parser = argparse.ArgumentParser(description="Continue Exp 2.6")
    parser.add_argument("--extra-steps", type=int, default=10000,
                        help="Additional steps beyond resume point")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for continuation (default: 1e-4)")
    parser.add_argument("--source-dir", type=str, default="results/exp2_6",
                        help="Directory with saved bridge .npy files")
    parser.add_argument("--output", type=str, default="results/exp2_6",
                        help="Output directory (same as source to continue)")
    parser.add_argument("--resume-step", type=int, default=None,
                        help="Step to resume from (auto-detected if not set)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original config
    with open(source_dir / "config.json") as f:
        cfg = json.load(f)

    # Auto-detect resume step from latest bridge files
    if args.resume_step is not None:
        resume_step = args.resume_step
    else:
        bridge_files = sorted(source_dir.glob("bridge_step*_*.npy"))
        steps_found = set()
        for bf in bridge_files:
            parts = bf.stem.split("_")
            if len(parts) >= 2 and parts[1].startswith("step"):
                try:
                    steps_found.add(int(parts[1].replace("step", "")))
                except ValueError:
                    pass
        steps_found.discard(0)
        if not steps_found:
            print("ERROR: No bridge checkpoint files found!")
            sys.exit(1)
        resume_step = max(steps_found)

    total_steps = resume_step + args.extra_steps
    print(f"\n{'='*70}")
    print(f"Experiment 2.6 CONTINUATION")
    print(f"Resuming from step {resume_step}, continuing to step {total_steps}")
    print(f"Learning rate: {args.lr} (original: {cfg['lr']})")
    print(f"Bridge weights from: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    config = ExperimentConfig(
        name=cfg["name"],
        rank=cfg["rank"],
        n_channels=cfg["n_channels"],
        bridge_mode=cfg["bridge_mode"],
        bridge_trainable=cfg["bridge_trainable"],
        model_name=cfg["model_name"],
        max_seq_len=cfg["max_seq_len"],
        batch_size=cfg["batch_size"],
        gradient_accumulation=cfg["gradient_accumulation"],
        lr=args.lr,
        num_epochs=cfg["num_epochs"],
        warmup_steps=100,
        checkpoint_steps=cfg["checkpoint_steps"],
        max_steps=total_steps,
        seed=cfg["seed"] + resume_step,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        target_modules=cfg["target_modules"],
    )

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    # Load saved bridge weights
    loaded = 0
    for name, lora in injected.items():
        safe_name = name.replace(".", "_")
        bridge_file = source_dir / f"bridge_step{resume_step}_{safe_name}.npy"
        if bridge_file.exists():
            saved_bridge = np.load(bridge_file)
            with torch.no_grad():
                lora.bridge.copy_(torch.from_numpy(saved_bridge).to(lora.bridge.device))
            loaded += 1
        else:
            print(f"  WARNING: No bridge file for {name} at step {resume_step}")
    print(f"Loaded {loaded}/{len(injected)} bridge weights from step {resume_step}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    bridge_params = sum(
        lora.bridge.numel() for lora in injected.values() if lora.bridge.requires_grad
    )
    print(f"Trainable: {trainable_params:,} / {total_params:,}")
    print(f"Bridge params: {bridge_params:,}")

    print("Loading Alpaca-cleaned dataset...")
    dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len)
    val_dataset = AlpacaDataset(tokenizer, max_len=config.max_seq_len, max_samples=1000)
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
        lr=config.lr, weight_decay=0.01,
    )

    continuation_steps = args.extra_steps
    def lr_schedule(step):
        rel_step = step - resume_step
        if rel_step < config.warmup_steps:
            return rel_step / max(config.warmup_steps, 1)
        progress = (rel_step - config.warmup_steps) / max(
            continuation_steps - config.warmup_steps, 1
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Load existing checkpoints for seamless continuation
    checkpoints = []
    existing_results = source_dir / "contrastive_results.json"
    if existing_results.exists():
        with open(existing_results) as f:
            existing = json.load(f)
        if "checkpoints" in existing:
            checkpoints = existing["checkpoints"]
            print(f"Loaded {len(checkpoints)} existing checkpoints")

    # Append to existing training log
    log_file = output_dir / "training.log"
    log_mode = "a" if log_file.exists() else "w"
    log_fh = open(log_file, log_mode)

    def log(msg):
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"\n{'='*70}")
    log(f"CONTINUATION from step {resume_step} to {total_steps}")
    log(f"LR: {args.lr} | Optimizer: fresh AdamW (momentum reset)")
    log(f"{'='*70}\n")

    model.train()
    global_step = resume_step
    accumulation_loss = 0.0
    steps_since_ckpt = 0
    start_time = time.time()

    log(f"Continuing training from step {resume_step} to {total_steps}...")
    log(f"Effective batch size: {config.batch_size * config.gradient_accumulation}")
    log(f"NOTE: Optimizer state reset -- expect ~100 step transient\n")

    done = False
    for epoch in range(config.num_epochs):
        if done:
            break
        for batch in dataloader:
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

            accumulation_loss += loss.item() * config.gradient_accumulation

            if (global_step + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            steps_since_ckpt += 1

            if global_step % 100 == 0:
                avg_loss = accumulation_loss / steps_since_ckpt
                elapsed = time.time() - start_time
                steps_done = global_step - resume_step
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
                eta = (total_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0

                co_cross_ratio = None
                try:
                    ccp = coplanar_crossplanar_ratio(
                        injected, _CO_PLANAR_PAIRS, _CROSS_PLANAR_PAIRS
                    )
                    if ccp and "ratio" in ccp:
                        co_cross_ratio = ccp["ratio"]
                except Exception:
                    pass

                co_cross_str = ""
                if co_cross_ratio is not None:
                    co_cross_str = f" | Co/Cross: {co_cross_ratio:.3f}"

                lr_now = optimizer.param_groups[0]["lr"]
                msg = (
                    f"  Step {global_step:>5}/{total_steps} | "
                    f"LM: {avg_loss:.4f} | "
                    f"LR: {lr_now:.2e} | "
                    f"Speed: {steps_per_sec:.2f} step/s{co_cross_str} | "
                    f"ETA: {eta/3600:.1f}h"
                )
                log(msg)

            if global_step % 100 == 0 or global_step >= total_steps:
                avg_loss = accumulation_loss / max(steps_since_ckpt, 1)

                val_loss = evaluate(model, val_dataloader, device)

                f_mean, f_std, d_mean, d_std = 0, 0, 0, 0
                try:
                    metrics = collect_metrics(injected)
                    fiedlers = [m.bridge_fiedler for m in metrics]
                    devs = [m.bridge_deviation for m in metrics]
                    f_mean = float(np.mean(fiedlers)) if fiedlers else 0
                    f_std = float(np.std(fiedlers)) if fiedlers else 0
                    d_mean = float(np.mean(devs)) if devs else 0
                    d_std = float(np.std(devs)) if devs else 0
                except Exception:
                    pass

                ccp = None
                co_cross_ratio = None
                try:
                    ccp = coplanar_crossplanar_ratio(
                        injected, _CO_PLANAR_PAIRS, _CROSS_PLANAR_PAIRS
                    )
                    if ccp and "ratio" in ccp:
                        co_cross_ratio = ccp["ratio"]
                except Exception:
                    pass

                eff_rank = 0.0
                try:
                    eff_rank = gradient_effective_rank(injected)
                except Exception:
                    pass

                cp = ContrastiveCheckpointMetrics(
                    step=global_step,
                    train_loss=avg_loss,
                    val_loss=val_loss,
                    contrastive_loss=0.0,
                    contrastive_weight=0.0,
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

                for name, lora in injected.items():
                    safe_name = name.replace(".", "_")
                    B = lora.bridge.detach().cpu().numpy()
                    np.save(output_dir / f"bridge_step{global_step}_{safe_name}.npy", B)

                ccp_str = ""
                if co_cross_ratio is not None:
                    ccp_str = f" | Co/Cross: {co_cross_ratio:.3f}"

                total_wall = time.time() - start_time
                ckpt_msg = (
                    f"\n  *** Checkpoint {global_step}/{total_steps} [CONTINUATION] ***\n"
                    f"      Train loss:      {avg_loss:.4f}\n"
                    f"      Val loss:        {val_loss:.4f}\n"
                    f"      Contrastive:     0.000000 (w=0.0000)\n"
                    f"      Fiedler:         {f_mean:.5f} +/- {f_std:.5f}\n"
                    f"      Deviation:       {d_mean:.5f} +/- {d_std:.5f}\n"
                    f"      Eff rank:        {eff_rank:.2f}{ccp_str}\n"
                    f"      Wall time:       {total_wall/3600:.2f}h\n"
                )
                log(ckpt_msg)

                steps_since_ckpt = 0
                accumulation_loss = 0.0

                _save_contrastive_results(
                    output_dir, config, checkpoints,
                    trainable_params, total_params, bridge_params,
                    injected, 0.0, 500, "warmup_only",
                )

            if global_step >= total_steps:
                done = True

    _save_contrastive_results(
        output_dir, config, checkpoints,
        trainable_params, total_params, bridge_params,
        injected, 0.0, 500, "warmup_only",
    )

    elapsed = time.time() - start_time
    log(f"\nContinuation complete: {resume_step} -> {global_step}")
    log(f"Wall time: {elapsed/3600:.2f}h")
    log(f"Results saved to {output_dir}")
    log_fh.close()


if __name__ == "__main__":
    main()
