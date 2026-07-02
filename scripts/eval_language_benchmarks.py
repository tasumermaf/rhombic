"""Language benchmark evaluation for RhombiLoRA adapters.

Compares: base model, base + standard LoRA, base + RhombiLoRA
Benchmarks: MMLU, ARC-Challenge, HellaSwag

Wraps the EleutherAI lm_eval harness to evaluate Qwen 7B with and without
RhombiLoRA adapters on standard NLP benchmarks. Since RhombiLoRA is not a
standard PEFT adapter type, the script manually loads the base model,
injects the adapter architecture, restores saved bridge weights, and
passes the composed model into lm_eval's HFLM wrapper.

Note on adapter loading: the existing training scripts (train_exp2_scale,
train_cybernetic, train_task_fingerprint) save only bridge matrices as
.npy files — lora_A/B weights are not persisted separately. If an
adapter_state.pt file exists (containing full adapter state_dict), the
script will load trained lora_A/B from it. Otherwise, lora_A/B are
freshly initialized (deterministic from seed). The bridge weights carry
the task-specific signal in either case.

Requirements:
  pip install lm-eval>=0.4.0

Usage:
  # Evaluate base model only
  python scripts/eval_language_benchmarks.py --output results/benchmarks/base.json

  # Evaluate with a RhombiLoRA adapter
  python scripts/eval_language_benchmarks.py \\
    --adapter results/exp3 \\
    --output results/benchmarks/exp3.json

  # Custom tasks and fewshot
  python scripts/eval_language_benchmarks.py \\
    --adapter results/exp3 \\
    --tasks mmlu arc_challenge hellaswag winogrande \\
    --num-fewshot 0 \\
    --output results/benchmarks/exp3_0shot.json

  # Full three-way comparison (run each, then compare JSONs)
  python scripts/eval_language_benchmarks.py --output results/benchmarks/base.json
  python scripts/eval_language_benchmarks.py --adapter results/exp2/standard_lora_r24 --output results/benchmarks/standard_lora.json
  python scripts/eval_language_benchmarks.py --adapter results/exp3 --output results/benchmarks/rhombi.json
  python scripts/eval_language_benchmarks.py --compare results/benchmarks/base.json results/benchmarks/standard_lora.json results/benchmarks/rhombi.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# ── Dependency checks ─────────────────────────────────────────────


def _check_lm_eval():
    """Verify lm_eval is installed and return the module."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        return lm_eval, HFLM
    except ImportError:
        print(
            "ERROR: lm_eval not installed.\n"
            "\n"
            "Install with:\n"
            "  pip install lm-eval>=0.4.0\n"
            "\n"
            "For GPU acceleration:\n"
            "  pip install lm-eval[vllm]\n"
        )
        sys.exit(1)


def _check_torch():
    """Verify torch and transformers are available."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return torch, AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(
            f"ERROR: Missing dependency: {e}\n"
            "\n"
            "Install with:\n"
            "  pip install torch transformers\n"
        )
        sys.exit(1)


# ── Adapter loading ───────────────────────────────────────────────


def _load_adapter_config(adapter_path: Path) -> dict:
    """Load adapter configuration from config.json in the adapter directory."""
    config_file = adapter_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def _load_model_with_adapter(
    model_name: str,
    adapter_path: Path,
    device: str,
):
    """Load base model, inject RhombiLoRA, restore saved weights.

    Returns (model, tokenizer) ready for evaluation.
    """
    torch, AutoModelForCausalLM, AutoTokenizer = _check_torch()
    import numpy as np

    # Ensure scripts dir is importable for train_exp2_scale
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train_exp2_scale import ExperimentConfig, inject_lora

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"  Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=dev,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Build ExperimentConfig from saved config
    saved_config = _load_adapter_config(adapter_path)
    config = ExperimentConfig(
        name="eval",
        rank=saved_config.get("rank", 24),
        n_channels=saved_config.get("n_channels", 6),
        bridge_mode=saved_config.get("bridge_mode", "identity"),
        bridge_trainable=False,
        model_name=model_name,
    )

    # Inject LoRA adapters
    injected = inject_lora(model, config)
    n_adapters = len(injected)
    print(f"  Injected {n_adapters} RhombiLoRA adapters (rank={config.rank}, ch={config.n_channels})")

    # Restore full adapter state if available (lora_A, lora_B, bridge)
    adapter_state_file = adapter_path / "adapter_state.pt"
    if adapter_state_file.exists():
        print(f"  Loading full adapter state from {adapter_state_file}")
        state = torch.load(adapter_state_file, map_location=dev, weights_only=True)
        loaded = 0
        for name, lora in injected.items():
            safe = name.replace(".", "_")
            if f"{safe}.lora_A" in state:
                with torch.no_grad():
                    lora.lora_A.copy_(state[f"{safe}.lora_A"].to(lora.lora_A.device))
                    lora.lora_B.copy_(state[f"{safe}.lora_B"].to(lora.lora_B.device))
                    if f"{safe}.bridge" in state:
                        lora.bridge.copy_(state[f"{safe}.bridge"].to(lora.bridge.device))
                loaded += 1
        print(f"  Restored lora_A/B + bridge for {loaded}/{n_adapters} adapters")
    else:
        # Fall back to bridge-only .npy files (existing save format)
        loaded = 0
        for name, lora in injected.items():
            safe = name.replace(".", "_")
            bridge_file = adapter_path / f"bridge_final_{safe}.npy"
            if bridge_file.exists():
                bridge_np = np.load(bridge_file)
                with torch.no_grad():
                    lora.bridge.copy_(
                        torch.from_numpy(bridge_np).to(lora.bridge.device)
                    )
                loaded += 1
        print(f"  Restored bridge weights for {loaded}/{n_adapters} adapters (bridge-only)")
        if loaded < n_adapters:
            missing = n_adapters - loaded
            print(f"  WARNING: {missing} adapters have no saved bridge (using identity init)")

    return model, tokenizer


# ── Evaluation ────────────────────────────────────────────────────


def run_evaluation(
    model_name: str,
    adapter_path: Path | None,
    tasks: list[str],
    num_fewshot: int,
    batch_size: str,
    device: str,
) -> dict:
    """Run lm_eval benchmarks and return results dict."""
    lm_eval, HFLM = _check_lm_eval()
    start = time.time()

    # Parse batch_size
    bs = batch_size if batch_size == "auto" else int(batch_size)

    if adapter_path is not None:
        # Load model with adapter
        print(f"\nLoading model with RhombiLoRA adapter...")
        model, tokenizer = _load_model_with_adapter(model_name, adapter_path, device)
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=bs,
        )
        adapter_label = str(adapter_path)
    else:
        # Evaluate base model directly
        print(f"\nLoading base model for evaluation...")
        lm = HFLM(
            pretrained=model_name,
            dtype="bfloat16",
            batch_size=bs,
            device=device,
        )
        adapter_label = None

    # Run evaluation
    print(f"\nRunning evaluation on: {', '.join(tasks)}")
    print(f"  Few-shot: {num_fewshot}")
    print(f"  Batch size: {batch_size}")

    eval_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
    )

    # Extract and organize results
    results = {
        "model": model_name,
        "adapter": adapter_label,
        "tasks": {},
        "num_fewshot": num_fewshot,
        "wall_time_seconds": time.time() - start,
    }

    for task_name in tasks:
        if task_name in eval_results["results"]:
            task_data = eval_results["results"][task_name]
            # Keep only scalar metrics
            results["tasks"][task_name] = {
                k: v for k, v in task_data.items()
                if isinstance(v, (int, float, str))
            }

    return results


def compare_results(result_files: list[str]) -> None:
    """Load multiple result JSONs and print a comparison table."""
    all_results = []
    for f in result_files:
        path = Path(f)
        if not path.exists():
            print(f"WARNING: {f} not found, skipping")
            continue
        with open(path) as fh:
            all_results.append(json.load(fh))

    if len(all_results) < 2:
        print("Need at least 2 result files to compare.")
        return

    # Collect all tasks across all results
    all_tasks = set()
    for r in all_results:
        all_tasks.update(r.get("tasks", {}).keys())
    all_tasks = sorted(all_tasks)

    # Find best accuracy key per task
    def get_accuracy(result, task):
        """Extract the primary accuracy metric from a task result."""
        if task not in result.get("tasks", {}):
            return None
        metrics = result["tasks"][task]
        # Prefer acc_norm, then acc
        for key in sorted(metrics.keys()):
            if "acc" in key.lower() and "norm" in key.lower():
                return metrics[key], key
        for key in sorted(metrics.keys()):
            if "acc" in key.lower():
                return metrics[key], key
        return None

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"Benchmark Comparison")
    print(f"{'='*80}")

    # Header
    labels = []
    for i, r in enumerate(all_results):
        adapter = r.get("adapter")
        if adapter:
            label = Path(adapter).name
        else:
            label = "base"
        labels.append(label)

    header = f"{'Task':<20}"
    for label in labels:
        header += f" | {label:>14}"
    print(header)
    print("-" * len(header))

    for task in all_tasks:
        row = f"{task:<20}"
        for r in all_results:
            acc_result = get_accuracy(r, task)
            if acc_result is not None:
                acc_val, acc_key = acc_result
                row += f" | {acc_val:>13.4f}"
            else:
                row += f" | {'N/A':>14}"
        print(row)

    # Wall times
    print("-" * len(header))
    time_row = f"{'Wall time (min)':<20}"
    for r in all_results:
        wt = r.get("wall_time_seconds", 0) / 60
        time_row += f" | {wt:>13.1f}"
    print(time_row)
    print(f"{'='*80}")


def print_results(results: dict) -> None:
    """Print a summary table of results."""
    print(f"\n{'='*70}")
    print(f"Results Summary")
    print(f"  Model:   {results['model']}")
    print(f"  Adapter: {results.get('adapter') or 'None (base model)'}")
    print(f"{'='*70}")

    if not results.get("tasks"):
        print("  No task results found.")
        return

    for task, metrics in sorted(results["tasks"].items()):
        # Find best accuracy metric
        acc_key = None
        for k in sorted(metrics.keys()):
            if "acc" in k.lower() and "norm" in k.lower():
                acc_key = k
                break
        if acc_key is None:
            for k in sorted(metrics.keys()):
                if "acc" in k.lower():
                    acc_key = k
                    break

        if acc_key and isinstance(metrics[acc_key], (int, float)):
            print(f"  {task:<25} {metrics[acc_key]:.4f}  ({acc_key})")
        else:
            print(f"  {task:<25} (no accuracy metric found)")

    wt = results.get("wall_time_seconds", 0)
    print(f"\n  Wall time: {wt/60:.1f}m")
    print(f"{'='*70}")


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Language benchmark evaluation for RhombiLoRA adapters via lm_eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Base model only\n"
            "  python scripts/eval_language_benchmarks.py --output results/benchmarks/base.json\n"
            "\n"
            "  # With adapter\n"
            "  python scripts/eval_language_benchmarks.py --adapter results/exp3 --output results/benchmarks/exp3.json\n"
            "\n"
            "  # Compare multiple runs\n"
            "  python scripts/eval_language_benchmarks.py --compare results/benchmarks/base.json results/benchmarks/exp3.json\n"
        ),
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or path (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Directory containing RhombiLoRA adapter weights (bridge .npy files "
             "and/or adapter_state.pt). If omitted, evaluates base model only.",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+",
        default=["mmlu", "arc_challenge", "hellaswag"],
        help="lm_eval task names to evaluate (default: mmlu arc_challenge hellaswag)",
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=5,
        help="Number of few-shot examples (default: 5)",
    )
    parser.add_argument(
        "--batch-size", type=str, default="auto",
        help="Batch size for evaluation — integer or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--output", type=str, default="results/benchmarks/results.json",
        help="Path for output JSON (default: results/benchmarks/results.json)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for evaluation (default: cuda)",
    )
    parser.add_argument(
        "--compare", type=str, nargs="+", default=None,
        help="Compare multiple result JSON files (no evaluation run)",
    )
    args = parser.parse_args()

    # Compare mode — just print table from existing results
    if args.compare:
        compare_results(args.compare)
        return

    # Validate adapter path if specified
    adapter_path = None
    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.exists():
            print(f"ERROR: Adapter directory not found: {adapter_path}")
            sys.exit(1)

    # Print configuration
    print(f"{'='*70}")
    print(f"Language Benchmark Evaluation")
    print(f"  Model:    {args.model}")
    print(f"  Adapter:  {args.adapter or 'None (base model)'}")
    print(f"  Tasks:    {', '.join(args.tasks)}")
    print(f"  Few-shot: {args.num_fewshot}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  Device:   {args.device}")
    print(f"  Output:   {args.output}")
    print(f"{'='*70}")

    # Run evaluation
    results = run_evaluation(
        model_name=args.model,
        adapter_path=adapter_path,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Print results
    print_results(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
