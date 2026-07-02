#!/usr/bin/env python3
"""Absorb TeLoRA bridge from AI Toolkit checkpoint into standard LoRA weights.

After training with aitoolkit_telora.py, the saved checkpoint contains
lora_mid.bridge parameters alongside standard lora_down/lora_up weights.
This script folds the bridge into the LoRA weights so the result loads
in any standard LoRA pipeline (ComfyUI, PEFT, etc.).

The absorption is exact: no approximation, no information loss.

Usage:
    python absorb_aitoolkit_bridge.py input.safetensors output.safetensors
    python absorb_aitoolkit_bridge.py /workspace/output/telora/ /workspace/output/telora_absorbed.safetensors
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
import numpy as np

try:
    from safetensors.torch import load_file, save_file
except ImportError:
    print("pip install safetensors")
    sys.exit(1)


def absorb_bridge(
    lora_down_weight: torch.Tensor,
    lora_up_weight: torch.Tensor,
    bridge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold bridge into lora_down, preserving lora_up unchanged.

    Standard LoRA:  out = x @ A.T @ B.T
    TeLoRA:         out = x @ A.T @ E @ B.T
    where E is the block-diagonal expansion of the C×C bridge.

    We absorb E into A: new_A = E.T @ A
    Then: out = x @ new_A.T @ B.T = x @ (E.T @ A).T @ B.T = x @ A.T @ E @ B.T ✓

    Parameters
    ----------
    lora_down_weight : (rank, in_features) tensor — lora_A
    lora_up_weight : (out_features, rank) tensor — lora_B
    bridge : (C, C) tensor — bridge matrix

    Returns
    -------
    new_down, new_up : absorbed weight tensors (same shapes as input)
    """
    C = bridge.shape[0]
    rank = lora_down_weight.shape[0]
    R = rank // C  # channel size

    # Build block-diagonal expansion E: (rank, rank)
    # The bridge operates as: bridged[b, i*R+k] = sum_j bridge[i,j] * h[b, j*R+k]
    # So E[i*R+k, j*R+k] = bridge[i,j] (only when channel offsets match)
    E = torch.zeros(rank, rank, dtype=bridge.dtype, device=bridge.device)
    for i in range(C):
        for j in range(C):
            for r in range(R):
                E[i * R + r, j * R + r] = bridge[i, j]

    # Absorb: out = x @ A.T @ E.T @ B.T = x @ (E @ A).T @ B.T
    # So new_A = E @ A
    new_down = (E @ lora_down_weight.float()).to(lora_down_weight.dtype)
    return new_down, lora_up_weight


def process_checkpoint(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Process a full state dict, absorbing all bridges.

    AI Toolkit keys look like:
        <prefix>.lora_down.weight
        <prefix>.lora_up.weight
        <prefix>.lora_mid.bridge

    After absorption, lora_mid.bridge keys are removed.
    """
    # Find all bridge keys
    bridge_keys = [k for k in state_dict if '.lora_mid.bridge' in k]

    if not bridge_keys:
        print("No bridge parameters found — already standard LoRA format.")
        return state_dict

    print(f"Found {len(bridge_keys)} bridge parameters to absorb")

    output = {}
    absorbed_prefixes = set()

    for bkey in bridge_keys:
        prefix = bkey.replace('.lora_mid.bridge', '')
        down_key = f"{prefix}.lora_down.weight"
        up_key = f"{prefix}.lora_up.weight"

        if down_key not in state_dict or up_key not in state_dict:
            print(f"WARNING: Missing lora_down/up for {prefix}, skipping")
            continue

        bridge = state_dict[bkey]
        new_down, new_up = absorb_bridge(
            state_dict[down_key],
            state_dict[up_key],
            bridge,
        )

        output[down_key] = new_down
        output[up_key] = new_up
        absorbed_prefixes.add(prefix)

    # Copy non-bridge, non-absorbed keys
    for key, value in state_dict.items():
        if '.lora_mid.' in key:
            continue  # skip bridge keys
        if key in output:
            continue  # already absorbed
        output[key] = value

    print(f"Absorbed {len(absorbed_prefixes)} bridges")
    bridge_params = sum(state_dict[k].numel() for k in bridge_keys)
    print(f"Removed {bridge_params:,} bridge parameters")

    return output


def main():
    parser = argparse.ArgumentParser(description="Absorb TeLoRA bridge into standard LoRA")
    parser.add_argument("input", help="Input .safetensors file or directory containing one")
    parser.add_argument("output", help="Output .safetensors file path")
    parser.add_argument("--verify", action="store_true", help="Verify absorption correctness")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Find the safetensors file
    if input_path.is_dir():
        candidates = sorted(input_path.glob("*.safetensors"))
        if not candidates:
            print(f"No .safetensors files found in {input_path}")
            sys.exit(1)
        input_file = candidates[-1]  # latest
        print(f"Using: {input_file}")
    else:
        input_file = input_path

    # Load
    state_dict = load_file(str(input_file))
    print(f"Loaded {len(state_dict)} keys from {input_file}")

    if args.verify:
        original = {k: v.clone() for k, v in state_dict.items()}

    # Absorb
    absorbed = process_checkpoint(state_dict)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(absorbed, str(output_path))
    print(f"Saved absorbed LoRA to {output_path}")
    print(f"Keys: {len(absorbed)} (was {len(state_dict)})")

    if args.verify:
        print("\n=== Verification ===")
        # Check a few layers: original forward should match absorbed forward
        bridge_prefixes = [k.replace('.lora_mid.bridge', '')
                          for k in original if '.lora_mid.bridge' in k]

        errors = []
        for prefix in bridge_prefixes[:5]:  # check first 5
            A = original[f"{prefix}.lora_down.weight"]
            B = original[f"{prefix}.lora_up.weight"]
            bridge = original[f"{prefix}.lora_mid.bridge"]
            new_A = absorbed[f"{prefix}.lora_down.weight"]
            new_B = absorbed[f"{prefix}.lora_up.weight"]

            # Random input
            x = torch.randn(4, A.shape[1], dtype=torch.float32)

            # Original path: x @ A.T @ E @ B.T
            C = bridge.shape[0]
            R = A.shape[0] // C
            h = x @ A.float().T
            h_reshaped = h.view(4, C, R)
            h_bridged = torch.einsum('ij,bjk->bik', bridge.float(), h_reshaped)
            h_flat = h_bridged.reshape(4, A.shape[0])
            out_orig = h_flat @ B.float().T

            # Absorbed path: x @ new_A.T @ new_B.T
            out_absorbed = (x @ new_A.float().T) @ new_B.float().T

            max_err = (out_orig - out_absorbed).abs().max().item()
            errors.append(max_err)
            status = "PASS" if max_err < 1e-4 else "FAIL"
            print(f"  {prefix[:60]}... max_err={max_err:.2e} [{status}]")

        if all(e < 1e-4 for e in errors):
            print(f"\nAll {len(errors)} checked layers PASS (max error < 1e-4)")
        else:
            print(f"\nWARNING: Some layers have high absorption error!")


if __name__ == "__main__":
    main()
