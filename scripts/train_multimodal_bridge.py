"""Multi-modal bridge signature experiment.

Trains RhombiLoRA adapters on three different modalities using the same
model architecture, then compares bridge signatures to test whether
different MODALITIES produce fundamentally different bridge structures
(tessellation thesis).

Modalities:
  - Text: Qwen 7B on alpaca-cleaned (10K steps)
  - Code: Qwen 7B on CodeAlpaca (10K steps)
  - Vision: Qwen-VL-7B on LLaVA-Instruct-150K (10K steps)

All with cybernetic training, 6-channel FCC bridge.

Usage:
  python scripts/train_multimodal_bridge.py --modality text
  python scripts/train_multimodal_bridge.py --modality code
  python scripts/train_multimodal_bridge.py --modality vision
  python scripts/train_multimodal_bridge.py --modality text --output results/multimodal/text
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_cybernetic import train_cybernetic
from train_exp2_scale import ExperimentConfig


MODALITY_CONFIGS = {
    "text": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "yahma/alpaca-cleaned",
        "description": "General instruction following (text)",
    },
    "code": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "sahil2801/CodeAlpaca-20k",
        "description": "Code generation and explanation",
    },
    "vision": {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "dataset": "liuhaotian/LLaVA-Instruct-150K",
        "description": "Vision-language instruction following",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Multi-modal bridge signature training"
    )
    parser.add_argument(
        "--modality", type=str, required=True,
        choices=list(MODALITY_CONFIGS.keys()),
        help="Which modality to train",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/multimodal/{modality})",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000,
    )
    parser.add_argument(
        "--feedback-interval", type=int, default=100,
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=8,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    modality_cfg = MODALITY_CONFIGS[args.modality]
    output_dir = Path(args.output) if args.output else Path(
        f"results/multimodal/{args.modality}"
    )

    config = ExperimentConfig(
        name=f"multimodal_{args.modality}",
        rank=24,
        n_channels=6,
        bridge_mode="identity",
        bridge_trainable=True,
        model_name=modality_cfg["model"],
        dataset_name=modality_cfg["dataset"],
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        seed=args.seed,
    )

    # Save modality metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "modality.json", "w") as f:
        json.dump({
            "modality": args.modality,
            **modality_cfg,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Multi-modal Bridge Training: {args.modality}")
    print(f"  {modality_cfg['description']}")
    print(f"  Model:   {modality_cfg['model']}")
    print(f"  Dataset: {modality_cfg['dataset']}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*70}\n")

    # NOTE: For vision modality, the dataset loading in train_cybernetic
    # will need to be extended to handle image-text pairs. For now, the
    # AlpacaDataset class only handles text. Vision training requires:
    #   1. A custom dataset class that loads LLaVA-format data
    #   2. Qwen2-VL processor instead of tokenizer-only
    #   3. Image preprocessing in the collator
    # This script sets up the infrastructure; vision-specific data loading
    # is a follow-up.

    if args.modality == "vision":
        print("WARNING: Vision modality uses text-only data pipeline.")
        print("Vision-specific data loading is a TODO.")
        print("Running with text data on the VL model for structural comparison.\n")

    train_cybernetic(
        config,
        output_dir,
        feedback_interval=args.feedback_interval,
    )


if __name__ == "__main__":
    main()
