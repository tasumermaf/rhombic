"""Merge adapter_state.pt into base model and save as HF model for lm-eval.

Usage:
    python scripts/merge_adapter.py --adapter results/T-001-full/adapter_state.pt
    python scripts/merge_adapter.py --adapter results/T-001-full/adapter_state.pt --output results/T-001-full/merged_model
"""
import argparse
import copy
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(model_name, adapter_path, output_dir):
    """Load base model, apply adapter deltas, save merged model."""
    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading adapter state: {adapter_path}")
    adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)

    # Discover all adapted layers from adapter_state keys
    layers = set()
    for key in adapter_state:
        layer_name = key.rsplit(".", 1)[0]  # Remove .lora_A, .lora_B, .bridge, etc.
        layers.add(layer_name)

    print(f"Found {len(layers)} adapted layers")

    merged = copy.deepcopy(model)
    for layer_name in sorted(layers):
        lora_A = adapter_state[f"{layer_name}.lora_A"].float()
        lora_B = adapter_state[f"{layer_name}.lora_B"].float()
        bridge = adapter_state[f"{layer_name}.bridge"].float()
        scaling = adapter_state[f"{layer_name}.scaling"].item()
        n_channels = adapter_state[f"{layer_name}.n_channels"].item()
        rank = adapter_state[f"{layer_name}.rank"].item()

        C = int(n_channels)
        R = int(rank) // C  # channel_size = rank / n_channels

        # Build expanded bridge: (rank, rank) block-diagonal
        bridge_expanded = torch.zeros(int(rank), int(rank))
        for i in range(C):
            for j in range(C):
                bridge_expanded[i*R:(i+1)*R, j*R:(j+1)*R] = bridge[i, j] * torch.eye(R)

        # delta_W = scaling * lora_B @ bridge_expanded @ lora_A
        delta_W = scaling * (lora_B @ bridge_expanded @ lora_A)

        # Navigate to the weight in the merged model
        # Convert safe name back to module path: model_layers_0_self_attn_q_proj -> model.layers.0.self_attn.q_proj
        parts = layer_name.split("_")
        # Reconstruct dotted path
        module_path = _safe_name_to_path(layer_name)

        # Get the base weight
        obj = merged
        for p in module_path.split("."):
            obj = getattr(obj, p)

        obj.weight.data += delta_W.to(obj.weight.dtype)
        print(f"  Merged: {module_path} (C={C}, R={R}, scaling={scaling:.4f})")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved merged model to {output_dir}")
    return output_dir


def _safe_name_to_path(safe_name):
    """Convert adapter_state key prefix to model attribute path.

    E.g. 'model_layers_0_self_attn_q_proj' -> 'model.layers.0.self_attn.q_proj'

    The pattern is: model.layers.{N}.self_attn.{proj_type}
    where N is a digit and proj_type is q_proj, k_proj, v_proj, o_proj.
    """
    # Known structure for LLaMA-family: model.layers.N.self_attn.X_proj
    # The safe name uses _ everywhere, so we need to reconstruct dots
    import re
    # Match: model_layers_{digits}_self_attn_{type}_proj
    m = re.match(r'model_layers_(\d+)_self_attn_([a-z])_proj', safe_name)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.{m.group(2)}_proj"

    # Fallback: try MLP patterns
    m = re.match(r'model_layers_(\d+)_mlp_(\w+)', safe_name)
    if m:
        return f"model.layers.{m.group(1)}.mlp.{m.group(2)}"

    raise ValueError(f"Cannot parse safe name: {safe_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge adapter into base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter_state.pt")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: adapter dir / merged_model)")
    args = parser.parse_args()

    output = args.output or str(Path(args.adapter).parent / "merged_model")
    merge_adapter(args.model, args.adapter, output)
