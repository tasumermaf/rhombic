"""Bridge-swap evaluation: can you mix bridges between task adapters?

Loads two (or more) trained RhombiLoRA adapters from fingerprint experiments.
For each adapter pair, keeps lora_A/B from one adapter and swaps in the
bridge matrices from the other. Measures perplexity on held-out validation
data for all combinations.

If bridge-swap preserves task behavior (low perplexity degradation when
using the matching bridge), then bridges encode task-specific behavior
independently of the projection matrices. This is the deployment killer
feature: one set of lora_A/B weights, N task behaviors via bridge selection
(36 parameters per behavior).

This tests the 36-parameter composition thesis (L-016).

Adapter state requirements:
  Each adapter directory must contain bridge_final_*.npy files. If an
  adapter_state.pt file exists, trained lora_A/B weights are loaded from
  it. Otherwise, lora_A/B are freshly initialized (same seed = same init
  across adapters, which is the correct control: if both adapters share
  init, the bridge IS the only difference).

Usage:
  # Two-adapter comparison
  python scripts/eval_bridge_swap.py \\
    --adapter-a results/fingerprints/code \\
    --adapter-b results/fingerprints/math \\
    --output results/bridge-swap/code_vs_math.json

  # Multi-adapter from fingerprint directory
  python scripts/eval_bridge_swap.py \\
    --fingerprint-dir results/fingerprints \\
    --output results/bridge-swap/all.json

  # Custom validation dataset
  python scripts/eval_bridge_swap.py \\
    --fingerprint-dir results/fingerprints \\
    --eval-dataset wikitext \\
    --output results/bridge-swap/wikitext.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ── Bridge I/O ────────────────────────────────────────────────────


def load_adapter_bridges(adapter_dir: Path) -> dict[str, np.ndarray]:
    """Load all bridge_final_*.npy files from an adapter directory.

    Returns dict mapping sanitized module names to bridge matrices.
    """
    bridges = {}
    for f in sorted(adapter_dir.glob("bridge_final_*.npy")):
        name = f.stem.replace("bridge_final_", "")
        bridges[name] = np.load(f)
    return bridges


def inject_bridges(
    injected: dict[str, object],
    bridges: dict[str, np.ndarray],
) -> int:
    """Replace bridge weights in injected LoRA modules.

    Returns the number of bridges successfully replaced.
    """
    replaced = 0
    for name, lora in injected.items():
        safe_name = name.replace(".", "_")
        if safe_name in bridges:
            with torch.no_grad():
                lora.bridge.copy_(
                    torch.from_numpy(bridges[safe_name]).to(lora.bridge.device)
                )
            replaced += 1
    return replaced


def load_adapter_state(
    adapter_dir: Path,
    injected: dict[str, object],
    device: torch.device,
) -> bool:
    """Load full adapter state (lora_A, lora_B, bridge) if available.

    Returns True if adapter_state.pt was found and loaded.
    """
    state_file = adapter_dir / "adapter_state.pt"
    if not state_file.exists():
        return False

    state = torch.load(state_file, map_location=device, weights_only=True)
    for name, lora in injected.items():
        safe = name.replace(".", "_")
        if f"{safe}.lora_A" in state:
            with torch.no_grad():
                lora.lora_A.copy_(state[f"{safe}.lora_A"].to(lora.lora_A.device))
                lora.lora_B.copy_(state[f"{safe}.lora_B"].to(lora.lora_B.device))
                if f"{safe}.bridge" in state:
                    lora.bridge.copy_(state[f"{safe}.bridge"].to(lora.bridge.device))
    return True


# ── Validation dataset ────────────────────────────────────────────


class WikitextDataset(Dataset):
    """Wikitext-2 validation set for perplexity evaluation."""

    def __init__(self, tokenizer, max_len: int = 512, max_samples: int = 500):
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

        self.examples = []
        for item in ds:
            text = item["text"].strip()
            if len(text) < 50:
                continue

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            ids = encoded["input_ids"].squeeze(0)
            mask = encoded["attention_mask"].squeeze(0)
            labels = ids.clone()
            # Mask padding tokens in labels
            labels[mask == 0] = -100

            self.examples.append({
                "input_ids": ids,
                "attention_mask": mask,
                "labels": labels,
            })

            if max_samples > 0 and len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_eval_dataset(
    name: str,
    tokenizer,
    max_len: int = 512,
    max_samples: int = 500,
) -> Dataset:
    """Create a validation dataset by name."""
    if name == "wikitext":
        return WikitextDataset(tokenizer, max_len, max_samples)
    elif name == "alpaca":
        # Import AlpacaDataset from the training script
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from train_exp2_scale import AlpacaDataset
        return AlpacaDataset(tokenizer, max_len, max_samples)
    else:
        raise ValueError(
            f"Unknown eval dataset: {name!r}. "
            f"Available: 'wikitext', 'alpaca'"
        )


# ── Perplexity computation ────────────────────────────────────────


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
) -> float:
    """Compute perplexity on a held-out dataset.

    Returns perplexity (exp of mean cross-entropy loss). Capped at
    exp(20) ≈ 485M to avoid overflow from degenerate configurations.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

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

        # Weight by number of non-masked tokens for proper averaging
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return float(np.exp(min(avg_loss, 20.0)))


# ── Bridge-swap experiment ────────────────────────────────────────


def run_bridge_swap(
    model_name: str,
    task_adapters: dict[str, Path],
    eval_dataset_name: str,
    batch_size: int,
    max_batches: int,
    max_eval_samples: int,
    device_name: str,
) -> dict:
    """Run the full bridge-swap experiment.

    For each pair of tasks (A, B), creates four configurations:
      1. A_lora + A_bridge (native A)
      2. A_lora + B_bridge (A projections, B routing)
      3. B_lora + A_bridge (B projections, A routing)
      4. B_lora + B_bridge (native B)

    Measures perplexity for all four and reports the swap penalty.
    """
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train_exp2_scale import ExperimentConfig, inject_lora
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation dataset
    print(f"\nLoading eval dataset: {eval_dataset_name} ({max_eval_samples} samples)")
    eval_dataset = get_eval_dataset(
        eval_dataset_name, tokenizer, max_len=512, max_samples=max_eval_samples,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    print(f"  {len(eval_dataset)} examples, {len(eval_loader)} batches")

    # Load base model
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Determine adapter configuration from first adapter
    first_adapter_dir = next(iter(task_adapters.values()))
    saved_config = {}
    config_file = first_adapter_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            saved_config = json.load(f)

    config = ExperimentConfig(
        name="eval_bridge_swap",
        rank=saved_config.get("rank", 24),
        n_channels=saved_config.get("n_channels", 6),
        bridge_mode="identity",
        bridge_trainable=False,
        model_name=model_name,
    )

    # Inject LoRA adapters (fresh init — will be overwritten per-task)
    injected = inject_lora(model, config)
    n_adapters = len(injected)
    print(f"Injected {n_adapters} adapters (rank={config.rank}, ch={config.n_channels})")

    # Pre-load all task bridges
    task_bridges: dict[str, dict[str, np.ndarray]] = {}
    for task_name, task_dir in task_adapters.items():
        bridges = load_adapter_bridges(task_dir)
        if not bridges:
            print(f"WARNING: No bridge files found in {task_dir}, skipping {task_name}")
            continue
        task_bridges[task_name] = bridges
        print(f"  {task_name}: {len(bridges)} bridge matrices loaded")

    if len(task_bridges) < 2:
        print("ERROR: Need at least 2 tasks with bridge files for swap evaluation")
        return {}

    # Save initial lora_A/B state (shared init from the injection)
    # These are the "fresh" weights before any task-specific loading
    initial_lora_state = {}
    for name, lora in injected.items():
        initial_lora_state[name] = {
            "lora_A": lora.lora_A.detach().clone(),
            "lora_B": lora.lora_B.detach().clone(),
        }

    # Check if any adapters have saved lora_A/B state
    task_has_state = {}
    for task_name, task_dir in task_adapters.items():
        if task_name not in task_bridges:
            continue
        state_file = task_dir / "adapter_state.pt"
        task_has_state[task_name] = state_file.exists()

    any_has_state = any(task_has_state.values())
    all_have_state = all(task_has_state.get(t, False) for t in task_bridges)

    if not any_has_state:
        print(
            "\n  NOTE: No adapter_state.pt found — using shared lora_A/B initialization.\n"
            "  Bridge swap tests whether bridge matrices alone differentiate tasks\n"
            "  when the underlying projections are identical (same init seed).\n"
            "  This is a valid control — the bridge IS the only trained variable.\n"
        )

    # Run evaluations
    task_names = sorted(task_bridges.keys())
    results = {
        "model": model_name,
        "eval_dataset": eval_dataset_name,
        "n_eval_samples": len(eval_dataset),
        "tasks": task_names,
        "adapter_config": {
            "rank": config.rank,
            "n_channels": config.n_channels,
        },
        "has_trained_lora_AB": {t: task_has_state.get(t, False) for t in task_names},
        "native_perplexity": {},
        "swap_matrix": {},
        "swap_penalties": {},
    }

    start = time.time()

    # Phase 1: Measure native perplexity for each task
    print(f"\n{'='*60}")
    print(f"Phase 1: Native Perplexity (each task with its own bridge)")
    print(f"{'='*60}")

    for task_name in task_names:
        task_dir = task_adapters[task_name]

        # Restore lora_A/B: either from saved state or from fresh init
        if task_has_state.get(task_name, False):
            load_adapter_state(task_dir, injected, device)
        else:
            for name, lora in injected.items():
                with torch.no_grad():
                    lora.lora_A.copy_(initial_lora_state[name]["lora_A"])
                    lora.lora_B.copy_(initial_lora_state[name]["lora_B"])

        # Load this task's bridges
        inject_bridges(injected, task_bridges[task_name])

        ppl = compute_perplexity(model, eval_loader, device, max_batches)
        results["native_perplexity"][task_name] = ppl
        print(f"  {task_name}: PPL = {ppl:.2f}")

    # Phase 2: Swap matrix — every (lora, bridge) combination
    print(f"\n{'='*60}")
    print(f"Phase 2: Bridge Swap Matrix")
    print(f"{'='*60}")

    for lora_task in task_names:
        lora_dir = task_adapters[lora_task]
        results["swap_matrix"][lora_task] = {}

        # Load lora_A/B for this task
        if task_has_state.get(lora_task, False):
            load_adapter_state(lora_dir, injected, device)
        else:
            for name, lora in injected.items():
                with torch.no_grad():
                    lora.lora_A.copy_(initial_lora_state[name]["lora_A"])
                    lora.lora_B.copy_(initial_lora_state[name]["lora_B"])

        for bridge_task in task_names:
            # Inject the bridge from bridge_task
            inject_bridges(injected, task_bridges[bridge_task])

            ppl = compute_perplexity(model, eval_loader, device, max_batches)
            results["swap_matrix"][lora_task][bridge_task] = ppl

            is_native = " (native)" if lora_task == bridge_task else ""
            print(f"  lora={lora_task} + bridge={bridge_task}: PPL = {ppl:.2f}{is_native}")

    # Phase 3: Compute swap penalties
    print(f"\n{'='*60}")
    print(f"Phase 3: Swap Penalties")
    print(f"{'='*60}")

    for lora_task in task_names:
        native_ppl = results["swap_matrix"][lora_task][lora_task]
        penalties = {}
        for bridge_task in task_names:
            if bridge_task == lora_task:
                continue
            swap_ppl = results["swap_matrix"][lora_task][bridge_task]
            penalty = swap_ppl - native_ppl
            penalty_pct = (penalty / native_ppl) * 100 if native_ppl > 0 else float("inf")
            penalties[bridge_task] = {
                "absolute": penalty,
                "percentage": penalty_pct,
                "swap_ppl": swap_ppl,
                "native_ppl": native_ppl,
            }
            direction = "+" if penalty > 0 else ""
            print(
                f"  lora={lora_task}, swap bridge to {bridge_task}: "
                f"{direction}{penalty:.2f} ({direction}{penalty_pct:.1f}%)"
            )
        results["swap_penalties"][lora_task] = penalties

    results["wall_time_seconds"] = time.time() - start
    return results


def print_swap_matrix(results: dict) -> None:
    """Print a formatted swap matrix table."""
    if "swap_matrix" not in results:
        return

    task_names = results.get("tasks", sorted(results["swap_matrix"].keys()))

    print(f"\n{'='*70}")
    print(f"Bridge-Swap Perplexity Matrix")
    print(f"  Rows = lora_A/B source, Columns = bridge source")
    print(f"{'='*70}")

    # Header
    col_label = "lora \\ bridge"
    header = f"{col_label:<15}"
    for t in task_names:
        header += f" | {t:>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for lora_task in task_names:
        row = f"{lora_task:<15}"
        for bridge_task in task_names:
            ppl = results["swap_matrix"].get(lora_task, {}).get(bridge_task)
            if ppl is not None:
                marker = " *" if lora_task == bridge_task else "  "
                row += f" | {ppl:>8.2f}{marker}"
            else:
                row += f" | {'N/A':>10}"
        print(row)

    print(f"\n  * = native configuration (lora and bridge from same task)")

    # Summary statistics
    all_native = [
        results["swap_matrix"][t][t]
        for t in task_names
        if t in results["swap_matrix"] and t in results["swap_matrix"][t]
    ]
    all_swapped = [
        results["swap_matrix"][lt][bt]
        for lt in task_names
        for bt in task_names
        if lt != bt
        and lt in results["swap_matrix"]
        and bt in results["swap_matrix"].get(lt, {})
    ]

    if all_native and all_swapped:
        mean_native = np.mean(all_native)
        mean_swapped = np.mean(all_swapped)
        mean_penalty = mean_swapped - mean_native
        print(f"\n  Mean native PPL:  {mean_native:.2f}")
        print(f"  Mean swapped PPL: {mean_swapped:.2f}")
        print(f"  Mean swap penalty: {mean_penalty:+.2f} ({mean_penalty/mean_native*100:+.1f}%)")

        if mean_penalty > mean_native * 0.1:
            print(f"\n  RESULT: Bridges encode task-specific behavior (>{mean_native*0.1:.1f} PPL penalty)")
        elif mean_penalty < mean_native * 0.02:
            print(f"\n  RESULT: Bridges are interchangeable (<2% penalty)")
        else:
            print(f"\n  RESULT: Moderate bridge specificity ({mean_penalty/mean_native*100:.1f}% penalty)")

    print(f"{'='*70}")


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Bridge-swap evaluation: test whether bridges encode task behavior independently",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Two specific adapters\n"
            "  python scripts/eval_bridge_swap.py \\\n"
            "    --adapter-a results/fingerprints/code \\\n"
            "    --adapter-b results/fingerprints/math\n"
            "\n"
            "  # All adapters in fingerprint directory\n"
            "  python scripts/eval_bridge_swap.py \\\n"
            "    --fingerprint-dir results/fingerprints\n"
        ),
    )

    # Input: either two specific adapters or a fingerprint directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fingerprint-dir", type=str, default=None,
        help="Directory containing task adapter subdirectories (each with bridge .npy files)",
    )
    group.add_argument(
        "--adapter-a", type=str, default=None,
        help="First adapter directory (use with --adapter-b)",
    )
    parser.add_argument(
        "--adapter-b", type=str, default=None,
        help="Second adapter directory (required if --adapter-a is used)",
    )

    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or path (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--eval-dataset", type=str, default="wikitext",
        choices=["wikitext", "alpaca"],
        help="Validation dataset for perplexity measurement (default: wikitext)",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=500,
        help="Maximum validation examples to use (default: 500)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Batch size for evaluation (default: 2)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=100,
        help="Maximum batches per perplexity evaluation (default: 100)",
    )
    parser.add_argument(
        "--output", type=str, default="results/bridge-swap/results.json",
        help="Output JSON path (default: results/bridge-swap/results.json)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for evaluation (default: cuda)",
    )
    args = parser.parse_args()

    # Resolve adapter directories
    task_adapters: dict[str, Path] = {}

    if args.fingerprint_dir:
        fp_dir = Path(args.fingerprint_dir)
        if not fp_dir.exists():
            print(f"ERROR: Fingerprint directory not found: {fp_dir}")
            sys.exit(1)
        task_dirs = sorted([d for d in fp_dir.iterdir() if d.is_dir()])
        # Filter to directories that contain bridge files
        for d in task_dirs:
            bridges = list(d.glob("bridge_final_*.npy"))
            if bridges:
                task_adapters[d.name] = d
        if len(task_adapters) < 2:
            print(
                f"ERROR: Need at least 2 task directories with bridge files.\n"
                f"Found: {list(task_adapters.keys())}\n"
                f"Searched in: {fp_dir}"
            )
            sys.exit(1)
    else:
        if args.adapter_a is None or args.adapter_b is None:
            print("ERROR: --adapter-a and --adapter-b are both required")
            sys.exit(1)
        path_a = Path(args.adapter_a)
        path_b = Path(args.adapter_b)
        if not path_a.exists():
            print(f"ERROR: Adapter A directory not found: {path_a}")
            sys.exit(1)
        if not path_b.exists():
            print(f"ERROR: Adapter B directory not found: {path_b}")
            sys.exit(1)
        task_adapters[path_a.name] = path_a
        task_adapters[path_b.name] = path_b

    # Print configuration
    print(f"{'='*70}")
    print(f"Bridge-Swap Evaluation")
    print(f"  Model:    {args.model}")
    print(f"  Tasks:    {', '.join(sorted(task_adapters.keys()))}")
    print(f"  Eval set: {args.eval_dataset} ({args.max_eval_samples} samples)")
    print(f"  Batch:    {args.batch_size}")
    print(f"  Device:   {args.device}")
    print(f"  Output:   {args.output}")
    print(f"{'='*70}")

    # Run experiment
    results = run_bridge_swap(
        model_name=args.model,
        task_adapters=task_adapters,
        eval_dataset_name=args.eval_dataset,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_eval_samples=args.max_eval_samples,
        device_name=args.device,
    )

    if not results:
        print("No results produced.")
        sys.exit(1)

    # Print formatted swap matrix
    print_swap_matrix(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
