#!/usr/bin/env python3
"""FI-003: Bridge Sign Persistence Under Task Fine-Tuning

Tests whether the corpus-derived sign fingerprint (FI-001) survives
subsequent fine-tuning WITHOUT the Steersman active.

Design:
  1. Load a completed checkpoint (adapter_state.pt + bridge_final files)
  2. Continue training with standard cross-entropy only (no Steersman)
  3. Checkpoint bridges every N steps
  4. Measure sign agreement with the baseline at each checkpoint
  5. Report sign persistence trajectory

Usage:
  # Primary (corpus-coupled, from FI-002 P-000):
  python scripts/fi_003_persistence.py \\
    --checkpoint results/fi-002/P-000 \\
    --output results/fi-003/corpus \\
    --steps 5000 --sign-interval 200

  # Control (identity-init, from Seed-43):
  python scripts/fi_003_persistence.py \\
    --checkpoint results/Seed-43 \\
    --output results/fi-003/identity \\
    --steps 5000 --sign-interval 200

  # Analysis (after both complete):
  python scripts/fi_003_persistence.py --analyze results/fi-003

The Steersman's contrastive loss operates on |B[cross]| — it doesn't
see signs. If signs persist, they are a structural property of the
parameter space, not an artifact of training pressure.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Constants ────────────────────────────────────────────────────────

LAYERS = 22  # TinyLlama
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
CO_PAIRS = [(0, 1), (2, 3), (4, 5)]  # RD face-pair axes


# ── Sign Analysis (shared with FI-002) ───────────────────────────────


def load_bridges_from_dir(experiment_dir: Path, prefix: str = "bridge_final") -> dict[str, np.ndarray]:
    """Load bridge matrices from an experiment directory."""
    bridges = {}
    for layer_idx in range(LAYERS):
        for mod in MODULES:
            fname = f"{prefix}_model_layers_{layer_idx}_self_attn_{mod}.npy"
            fpath = experiment_dir / fname
            if fpath.exists():
                bridges[f"L{layer_idx}_{mod}"] = np.load(fpath)
    return bridges


def get_cross_positions() -> list[tuple[int, int]]:
    """Get the lower-triangle cross-planar positions."""
    co_set = set(CO_PAIRS) | set((j, i) for i, j in CO_PAIRS)
    positions = []
    for i in range(6):
        for j in range(i):
            if (i, j) not in co_set and (j, i) not in co_set:
                positions.append((i, j))
    return positions


def compute_sign_vector(bridge: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    """Compute sign vector for a single bridge matrix."""
    return np.array([1 if bridge[i, j] >= 0 else -1 for i, j in positions])


def compute_sign_profile(bridges: dict[str, np.ndarray]) -> dict:
    """Compute sign structure of cross-planar residuals."""
    positions = get_cross_positions()
    sign_patterns = []
    for key in sorted(bridges.keys()):
        signs = compute_sign_vector(bridges[key], positions)
        sign_patterns.append(signs)

    sign_arr = np.array(sign_patterns)
    frac_pos = (sign_arr > 0).mean(axis=0)
    total_pos = (sign_arr > 0).mean()
    consensus = np.sign(sign_arr.mean(axis=0))
    consistency = np.abs(sign_arr.mean(axis=0))

    return {
        "positions": positions,
        "n_adapters": sign_arr.shape[0],
        "n_positions": sign_arr.shape[1],
        "frac_positive": frac_pos.tolist(),
        "total_positive": float(total_pos),
        "consensus_signs": consensus.tolist(),
        "consensus_strength": consistency.tolist(),
        "sign_array": sign_arr,
    }


def sign_agreement_vs_baseline(
    current_bridges: dict[str, np.ndarray],
    baseline_bridges: dict[str, np.ndarray],
) -> dict:
    """Compute per-adapter sign agreement between current and baseline.

    Returns per-adapter agreement fraction and per-position flip rates.
    """
    positions = get_cross_positions()
    keys = sorted(set(current_bridges.keys()) & set(baseline_bridges.keys()))

    per_adapter_agreement = []
    per_position_flips = np.zeros(len(positions))

    for key in keys:
        current_signs = compute_sign_vector(current_bridges[key], positions)
        baseline_signs = compute_sign_vector(baseline_bridges[key], positions)
        agreement = (current_signs == baseline_signs).mean()
        per_adapter_agreement.append(float(agreement))

        flips = (current_signs != baseline_signs).astype(float)
        per_position_flips += flips

    per_adapter_agreement = np.array(per_adapter_agreement)
    per_position_flips /= len(keys)  # fraction of adapters that flipped at each position

    return {
        "mean_agreement": float(per_adapter_agreement.mean()),
        "std_agreement": float(per_adapter_agreement.std()),
        "min_agreement": float(per_adapter_agreement.min()),
        "max_agreement": float(per_adapter_agreement.max()),
        "per_position_flip_rate": per_position_flips.tolist(),
        "positions": positions,
        "n_adapters": len(keys),
    }


# ── Training ─────────────────────────────────────────────────────────


def run_persistence(
    checkpoint_dir: Path,
    output_dir: Path,
    max_steps: int = 5000,
    sign_interval: int = 200,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lr: float = 2e-4,
    batch_size: int = 2,
    gradient_accumulation: int = 8,
    seed: int = 42,
):
    """Resume training from checkpoint WITHOUT the Steersman.

    Tracks sign persistence at regular intervals.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Add parent dir to path for imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_exp2_scale import ExperimentConfig, AlpacaDataset, inject_lora

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load baseline bridge signs ──────────────────────────────
    print(f"\n{'='*70}")
    print("FI-003: Bridge Sign Persistence Under Task Fine-Tuning")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Steps: {max_steps}")
    print(f"  Sign checkpoint interval: {sign_interval}")
    print(f"  Steersman: OFF (standard cross-entropy only)")
    print(f"{'='*70}\n")

    # Load baseline bridges for comparison
    baseline_bridges = load_bridges_from_dir(checkpoint_dir, "bridge_final")
    if not baseline_bridges:
        print("ERROR: No bridge_final files found in checkpoint directory.")
        return None

    baseline_profile = compute_sign_profile(baseline_bridges)
    print(f"Baseline: {len(baseline_bridges)} bridges loaded")
    print(f"  Total positive: {baseline_profile['total_positive']*100:.1f}%")
    print(f"  Consensus signs: {baseline_profile['consensus_signs']}")

    # Save baseline profile
    baseline_meta = {k: v for k, v in baseline_profile.items() if k != "sign_array"}
    with open(output_dir / "baseline_profile.json", "w") as f:
        json.dump(baseline_meta, f, indent=2)

    # ── Load adapter state ──────────────────────────────────────
    adapter_state_path = checkpoint_dir / "adapter_state.pt"
    if not adapter_state_path.exists():
        print(f"ERROR: No adapter_state.pt in {checkpoint_dir}")
        print("FI-003 requires a checkpoint with saved adapter weights.")
        return None

    adapter_state = torch.load(adapter_state_path, map_location="cpu", weights_only=True)
    print(f"Adapter state loaded: {len(adapter_state)} tensors")

    # ── Load checkpoint config ──────────────────────────────────
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            source_config = json.load(f)
        print(f"Source config: {source_config.get('name', 'unknown')}")
        print(f"  Bridge mode: {source_config.get('bridge_mode', 'unknown')}")
        print(f"  Original steps: {source_config.get('max_steps', 'unknown')}")
    else:
        source_config = {}

    # ── Setup model and LoRA ────────────────────────────────────
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device,
    )

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

    # Inject LoRA with identity bridge (we'll load trained weights next)
    config = ExperimentConfig(
        name=f"fi003_persistence",
        rank=source_config.get("rank", 24),
        n_channels=source_config.get("n_channels", 6),
        bridge_mode="identity",  # placeholder — we overwrite everything
        bridge_trainable=True,
        model_name=model_name,
        max_steps=max_steps,
        lr=lr,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        warmup_steps=200,
        checkpoint_steps=sign_interval,
        seed=seed,
    )
    injected = inject_lora(model, config)
    print(f"Injected {len(injected)} LoRA adapters")

    # ── Load trained weights into LoRA modules ──────────────────
    loaded_count = 0
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        # Load lora_A, lora_B
        key_a = f"{safe}.lora_A"
        key_b = f"{safe}.lora_B"
        key_bridge = f"{safe}.bridge"

        if key_a in adapter_state and key_b in adapter_state:
            lora.lora_A.data.copy_(adapter_state[key_a].to(device))
            lora.lora_B.data.copy_(adapter_state[key_b].to(device))
            loaded_count += 1

        if key_bridge in adapter_state:
            lora.bridge.data.copy_(adapter_state[key_bridge].to(device))
        else:
            # Try loading from bridge_final npy
            bridge_path = checkpoint_dir / f"bridge_final_{safe}.npy"
            if bridge_path.exists():
                bridge_np = np.load(bridge_path)
                lora.bridge.data.copy_(torch.tensor(bridge_np, dtype=lora.bridge.dtype, device=device))

    print(f"Loaded trained weights for {loaded_count}/{len(injected)} adapters")

    # Verify loaded bridges match baseline
    loaded_bridges = {}
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        loaded_bridges[safe.replace("model_layers_", "L").replace("_self_attn_", "_")] = (
            lora.bridge.detach().cpu().numpy()
        )

    # Quick sanity check — loaded bridge signs should match baseline
    # (Use the original key format for comparison)
    verify_count = 0
    for key in sorted(baseline_bridges.keys()):
        if key in loaded_bridges:
            verify_count += 1

    print(f"Bridge verification: {verify_count} keys matched")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,}")

    # ── Dataset ─────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = AlpacaDataset(tokenizer, max_len=512)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    val_dataset = AlpacaDataset(tokenizer, max_len=512, max_samples=1000)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ── Optimizer (all trainable params, including bridge) ──────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    def lr_schedule(step):
        if step < 200:
            return step / 200
        progress = (step - 200) / max(max_steps - 200, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Save config ─────────────────────────────────────────────
    run_config = {
        "experiment": "FI-003",
        "description": "Bridge sign persistence under unguided fine-tuning",
        "checkpoint_source": str(checkpoint_dir),
        "source_bridge_mode": source_config.get("bridge_mode", "unknown"),
        "source_steps": source_config.get("max_steps", "unknown"),
        "persistence_steps": max_steps,
        "sign_interval": sign_interval,
        "steersman": "OFF",
        "model_name": model_name,
        "lr": lr,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "seed": seed,
        "baseline_total_positive": baseline_profile["total_positive"],
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # ── Sign checkpoint at step 0 ───────────────────────────────
    sign_trajectory = []

    def checkpoint_signs(step: int, val_loss: float = 0.0, train_loss: float = 0.0):
        """Checkpoint bridge signs and compute agreement with baseline."""
        current_bridges = {}
        for name, lora in injected.items():
            safe = name.replace(".", "_")
            key = safe.replace("model_layers_", "L").replace("_self_attn_", "_")
            B = lora.bridge.detach().cpu().numpy()
            current_bridges[key] = B
            # Save bridge snapshot
            np.save(output_dir / f"bridge_step{step}_{safe}.npy", B)

        # Compute sign agreement with baseline
        agreement = sign_agreement_vs_baseline(current_bridges, baseline_bridges)

        # Compute current profile
        current_profile = compute_sign_profile(current_bridges)

        # Frobenius distance from baseline
        frob_distances = []
        for key in sorted(set(current_bridges.keys()) & set(baseline_bridges.keys())):
            dist = np.linalg.norm(current_bridges[key] - baseline_bridges[key], 'fro')
            frob_distances.append(float(dist))

        entry = {
            "step": step,
            "sign_agreement": agreement["mean_agreement"],
            "sign_agreement_std": agreement["std_agreement"],
            "sign_agreement_min": agreement["min_agreement"],
            "per_position_flip_rate": agreement["per_position_flip_rate"],
            "total_positive": current_profile["total_positive"],
            "consensus_signs": current_profile["consensus_signs"],
            "frobenius_mean": float(np.mean(frob_distances)),
            "frobenius_std": float(np.std(frob_distances)),
            "val_loss": val_loss,
            "train_loss": train_loss,
            "wall_time": time.time() - start_time if step > 0 else 0.0,
        }
        sign_trajectory.append(entry)

        print(f"\n  Sign checkpoint @ step {step}:")
        print(f"    Agreement with baseline: {agreement['mean_agreement']*100:.1f}% "
              f"(±{agreement['std_agreement']*100:.1f}%)")
        print(f"    Total positive: {current_profile['total_positive']*100:.1f}% "
              f"(baseline: {baseline_profile['total_positive']*100:.1f}%)")
        print(f"    Position flip rates: "
              f"{', '.join(f'{r:.2f}' for r in agreement['per_position_flip_rate'])}")
        print(f"    Frobenius distance: {np.mean(frob_distances):.6f} "
              f"(±{np.std(frob_distances):.6f})")

        return entry

    # ── Training loop ───────────────────────────────────────────
    model.train()
    global_step = 0
    accumulation_loss = 0.0
    steps_since_log = 0
    start_time = time.time()

    # Step 0 checkpoint
    checkpoint_signs(0)

    print(f"\nTraining for {max_steps} steps (NO Steersman)...")
    print(f"Effective batch size: {batch_size * gradient_accumulation}")
    print()

    done = False
    for epoch in range(100):  # enough epochs
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
            loss = outputs.loss / gradient_accumulation
            loss.backward()
            accumulation_loss += loss.item()

            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                steps_since_log += 1

                # Progress
                if global_step % 100 == 0:
                    avg_loss = accumulation_loss / steps_since_log
                    elapsed = time.time() - start_time
                    speed = global_step / elapsed
                    eta = (max_steps - global_step) / speed
                    print(f"Step {global_step:>5}/{max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"ETA: {eta/3600:.1f}h")

                # Sign checkpoint
                if global_step % sign_interval == 0 or global_step >= max_steps:
                    avg_loss = accumulation_loss / steps_since_log

                    # Quick validation
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vi, vb in enumerate(val_dataloader):
                            if vi >= 50:
                                break
                            v_ids = vb["input_ids"].to(device)
                            v_mask = vb["attention_mask"].to(device)
                            v_labels = vb["labels"].to(device)
                            v_out = model(input_ids=v_ids, attention_mask=v_mask, labels=v_labels)
                            val_losses.append(v_out.loss.item())
                    val_loss = float(np.mean(val_losses))
                    model.train()

                    checkpoint_signs(global_step, val_loss=val_loss, train_loss=avg_loss)
                    accumulation_loss = 0.0
                    steps_since_log = 0

                    # Save intermediate results
                    _save_results(output_dir, run_config, sign_trajectory, baseline_profile)

                if global_step >= max_steps:
                    done = True
                    break

    # ── Save final results ──────────────────────────────────────
    # Save final adapter state
    adapter_state_final = {}
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        adapter_state_final[f"{safe}.lora_A"] = lora.lora_A.detach().cpu()
        adapter_state_final[f"{safe}.lora_B"] = lora.lora_B.detach().cpu()
        adapter_state_final[f"{safe}.bridge"] = lora.bridge.detach().cpu()
        # Save bridge_final
        np.save(
            output_dir / f"bridge_final_{safe}.npy",
            lora.bridge.detach().cpu().numpy(),
        )
    torch.save(adapter_state_final, output_dir / "adapter_state.pt")

    _save_results(output_dir, run_config, sign_trajectory, baseline_profile)

    print(f"\n{'='*70}")
    print("FI-003 COMPLETE")
    print(f"{'='*70}")
    print(f"Total steps: {global_step}")
    print(f"Wall time: {(time.time() - start_time)/3600:.1f}h")

    if sign_trajectory:
        final = sign_trajectory[-1]
        initial = sign_trajectory[0]
        print(f"\nSign persistence summary:")
        print(f"  Step 0 agreement:     {initial['sign_agreement']*100:.1f}%")
        print(f"  Final agreement:      {final['sign_agreement']*100:.1f}%")
        print(f"  Agreement delta:      {(final['sign_agreement'] - initial['sign_agreement'])*100:+.1f}%")
        print(f"  Final Frobenius dist: {final['frobenius_mean']:.6f}")
        print(f"  Final val loss:       {final['val_loss']:.4f}")

    return sign_trajectory


def _save_results(output_dir, config, trajectory, baseline_profile):
    """Save results JSON (idempotent — called at each checkpoint)."""
    results = {
        "config": config,
        "baseline_profile": {k: v for k, v in baseline_profile.items() if k != "sign_array"},
        "sign_trajectory": trajectory,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


# ── Analysis ─────────────────────────────────────────────────────────


def analyze_persistence(results_dir: Path):
    """Compare persistence results across conditions."""
    print(f"\n{'='*70}")
    print("FI-003: Persistence Analysis")
    print(f"{'='*70}")

    conditions = {}
    for subdir in sorted(results_dir.iterdir()):
        results_path = subdir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            conditions[subdir.name] = data
            print(f"  Loaded: {subdir.name} ({len(data['sign_trajectory'])} checkpoints)")

    if not conditions:
        print("ERROR: No completed conditions found.")
        return

    # ── Trajectory comparison ───────────────────────────────────
    print(f"\n{'-'*70}")
    print("Sign Agreement Trajectory")
    print(f"{'-'*70}")

    # Header
    names = list(conditions.keys())
    header = f"{'Step':>6}"
    for name in names:
        header += f"  {name:>12}"
    print(header)
    print("-" * len(header))

    # Find all unique steps
    all_steps = set()
    for data in conditions.values():
        for entry in data["sign_trajectory"]:
            all_steps.add(entry["step"])

    for step in sorted(all_steps):
        line = f"{step:>6}"
        for name in names:
            entry = next(
                (e for e in conditions[name]["sign_trajectory"] if e["step"] == step),
                None,
            )
            if entry:
                line += f"  {entry['sign_agreement']*100:>11.1f}%"
            else:
                line += f"  {'—':>12}"
        print(line)

    # ── Position-level analysis ─────────────────────────────────
    print(f"\n{'-'*70}")
    print("Per-Position Final Flip Rates")
    print(f"{'-'*70}")

    positions = get_cross_positions()
    for name in names:
        traj = conditions[name]["sign_trajectory"]
        if traj:
            final = traj[-1]
            print(f"\n  {name}:")
            for i, (pos, flip) in enumerate(zip(positions, final["per_position_flip_rate"])):
                marker = "***" if flip > 0.5 else "   "
                print(f"    B[{pos[0]},{pos[1]}]: {flip*100:5.1f}% flipped {marker}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'-'*70}")
    print("Summary")
    print(f"{'-'*70}")

    for name in names:
        traj = conditions[name]["sign_trajectory"]
        if len(traj) >= 2:
            initial = traj[0]["sign_agreement"]
            final = traj[-1]["sign_agreement"]
            baseline_pos = conditions[name]["baseline_profile"]["total_positive"]
            final_pos = traj[-1]["total_positive"]
            final_frob = traj[-1]["frobenius_mean"]
            steps = traj[-1]["step"]

            print(f"\n  {name}:")
            print(f"    Baseline % positive: {baseline_pos*100:.1f}%")
            print(f"    Final % positive:    {final_pos*100:.1f}%")
            print(f"    Initial agreement:   {initial*100:.1f}%")
            print(f"    Final agreement:     {final*100:.1f}%")
            print(f"    Agreement delta:     {(final-initial)*100:+.1f}%")
            print(f"    Frobenius drift:     {final_frob:.6f}")
            print(f"    Steps:               {steps}")

            if final > 0.8:
                print(f"    VERDICT: STRONG PERSISTENCE — signs survive {steps} steps of unguided training")
            elif final > 0.6:
                print(f"    VERDICT: MODERATE PERSISTENCE — partial sign survival")
            elif final > 0.5:
                print(f"    VERDICT: WEAK PERSISTENCE — approaching random")
            else:
                print(f"    VERDICT: NO PERSISTENCE — signs randomized")

    # Save analysis
    analysis = {}
    for name in names:
        traj = conditions[name]["sign_trajectory"]
        if len(traj) >= 2:
            analysis[name] = {
                "initial_agreement": traj[0]["sign_agreement"],
                "final_agreement": traj[-1]["sign_agreement"],
                "delta": traj[-1]["sign_agreement"] - traj[0]["sign_agreement"],
                "final_frobenius": traj[-1]["frobenius_mean"],
                "steps": traj[-1]["step"],
                "baseline_positive": conditions[name]["baseline_profile"]["total_positive"],
                "final_positive": traj[-1]["total_positive"],
            }

    with open(results_dir / "fi-003-analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {results_dir / 'fi-003-analysis.json'}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="FI-003: Bridge Sign Persistence")
    parser.add_argument("--checkpoint", type=Path, help="Source checkpoint directory")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps (default: 5000)")
    parser.add_argument("--sign-interval", type=int, default=200,
                        help="Sign checkpoint interval (default: 200)")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze", type=Path, help="Analyze completed results directory")

    args = parser.parse_args()

    if args.analyze:
        analyze_persistence(args.analyze)
    elif args.checkpoint and args.output:
        run_persistence(
            checkpoint_dir=args.checkpoint,
            output_dir=args.output,
            max_steps=args.steps,
            sign_interval=args.sign_interval,
            model_name=args.model,
            lr=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation=args.grad_accum,
            seed=args.seed,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
