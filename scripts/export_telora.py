#!/usr/bin/env python3
"""Export a TeLoRA checkpoint as a standard PEFT-compatible LoRA adapter.

TeLoRA (Topological LoRA) trains a small C x C bridge matrix between the
down-projection (lora_A) and up-projection (lora_B). For deployment, this
bridge must be folded ("absorbed") into A and B so the adapter loads in any
LoRA-compatible framework (PEFT, vLLM, llama.cpp, etc.) without custom code.

The absorption is exact: out = x @ A.T @ E @ B.T = x @ new_A.T @ new_B.T
where E is the block-diagonal expansion of the bridge. No approximation, no
information loss. The geometric structure travels invisibly inside standard
LoRA weights.

Handles two checkpoint formats:
  1. Research (adapter_state.pt + bridge_final_*.npy + config.json)
  2. Competition/PEFT (adapter_model.safetensors + adapter_config.json)

Usage:
  python scripts/export_telora.py --checkpoint results/exp3/ --output exports/exp3-absorbed/
  python scripts/export_telora.py --checkpoint results/exp3/ --output exports/exp3-absorbed/ --verify
  python scripts/export_telora.py --checkpoint results/exp3/ --strategy balanced --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root so rhombic is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rhombic.nn.absorb import _expand_bridge, _balanced_factors


# ── Checkpoint detection ────────────────────────────────────────────


def _detect_format(checkpoint_dir: Path) -> str:
    """Detect whether this is a research or PEFT-format checkpoint."""
    if (checkpoint_dir / "adapter_state.pt").exists():
        return "research"
    safetensors = list(checkpoint_dir.glob("*.safetensors"))
    if safetensors and (checkpoint_dir / "adapter_config.json").exists():
        return "peft"
    # Check for adapter/ subdirectory (competition layout)
    adapter_sub = checkpoint_dir / "adapter"
    if adapter_sub.exists():
        if list(adapter_sub.glob("*.safetensors")):
            return "peft-subdir"
    raise FileNotFoundError(
        f"Cannot detect checkpoint format in {checkpoint_dir}.\n"
        f"Expected either adapter_state.pt (research) or "
        f"adapter_model.safetensors + adapter_config.json (PEFT)."
    )


# ── Research format loader ──────────────────────────────────────────


def _load_research_checkpoint(checkpoint_dir: Path) -> dict:
    """Load research-format checkpoint (adapter_state.pt + config.json).

    Returns dict with:
        layers: dict[safe_name -> {lora_A, lora_B, bridge, scaling, n_channels, rank}]
        config: dict from config.json
        model_name: str
        target_modules: list[str]
    """
    state = torch.load(
        checkpoint_dir / "adapter_state.pt",
        map_location="cpu",
        weights_only=True,
    )

    # Load config
    config = {}
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Group keys by layer
    layers = {}
    seen_keys = set()
    for key in state:
        if key in seen_keys:
            continue
        # Keys are like: model_layers_0_self_attn_q_proj.lora_A
        if ".lora_A" in key:
            safe_name = key.split(".lora_A")[0]
            lora_A = state[f"{safe_name}.lora_A"]
            lora_B = state[f"{safe_name}.lora_B"]
            bridge = state.get(f"{safe_name}.bridge")
            scaling_t = state.get(f"{safe_name}.scaling")
            n_channels_t = state.get(f"{safe_name}.n_channels")
            rank_t = state.get(f"{safe_name}.rank")

            scaling = float(scaling_t.item()) if scaling_t is not None else None
            n_channels = int(n_channels_t.item()) if n_channels_t is not None else None
            rank = int(rank_t.item()) if rank_t is not None else int(lora_A.shape[0])

            # If bridge not in state dict, try loading from .npy
            if bridge is None:
                npy_path = checkpoint_dir / f"bridge_final_{safe_name}.npy"
                if npy_path.exists():
                    bridge = torch.from_numpy(np.load(str(npy_path))).float()

            layers[safe_name] = {
                "lora_A": lora_A,
                "lora_B": lora_B,
                "bridge": bridge,
                "scaling": scaling,
                "n_channels": n_channels,
                "rank": rank,
            }
            seen_keys.update([
                f"{safe_name}.lora_A", f"{safe_name}.lora_B",
                f"{safe_name}.bridge", f"{safe_name}.scaling",
                f"{safe_name}.n_channels", f"{safe_name}.rank",
            ])

    model_name = config.get("model_name", "unknown")
    target_modules = config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Infer n_channels and rank from first layer if not in state dict
    if layers:
        first = next(iter(layers.values()))
        default_rank = first["rank"]
        default_n_channels = first.get("n_channels") or config.get("n_channels", 6)
        default_alpha = config.get("alpha", 16.0)
        for info in layers.values():
            if info["n_channels"] is None:
                info["n_channels"] = default_n_channels
            if info["scaling"] is None:
                info["scaling"] = default_alpha / info["rank"]

    return {
        "layers": layers,
        "config": config,
        "model_name": model_name,
        "target_modules": target_modules,
        "format": "research",
    }


# ── PEFT format loader ─────────────────────────────────────────────


def _load_peft_checkpoint(checkpoint_dir: Path) -> dict:
    """Load PEFT-format checkpoint (safetensors + adapter_config.json).

    Returns dict with:
        weights: dict[str -> Tensor] (all weight tensors)
        config: dict from adapter_config.json
        bridge_keys: list of keys that are bridge tensors
    """
    from safetensors.torch import load_file

    config_path = checkpoint_dir / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load all weights
    all_weights = {}
    for wf in sorted(checkpoint_dir.glob("*.safetensors")):
        all_weights.update(load_file(str(wf)))

    bridge_keys = [k for k in all_weights if k.endswith(".bridge")]

    return {
        "weights": all_weights,
        "config": config,
        "bridge_keys": bridge_keys,
        "format": "peft",
    }


# ── Safe name to module path ───────────────────────────────────────


def _safe_name_to_module_path(safe_name: str) -> str:
    """Convert underscore-delimited safe name back to dotted module path.

    model_layers_0_self_attn_q_proj -> model.layers.0.self_attn.q_proj
    """
    import re

    # LLaMA/Qwen attention
    m = re.match(r'model_layers_(\d+)_self_attn_([a-z])_proj', safe_name)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.{m.group(2)}_proj"

    # MLP
    m = re.match(r'model_layers_(\d+)_mlp_(\w+)', safe_name)
    if m:
        return f"model.layers.{m.group(1)}.mlp.{m.group(2)}"

    # Fallback: replace _ with . (lossy but best-effort)
    return safe_name.replace("_", ".")


# ── Absorption core ─────────────────────────────────────────────────


def absorb_single_layer(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    bridge: torch.Tensor,
    n_channels: int,
    strategy: str = "into_B",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absorb a bridge matrix into lora_A and lora_B.

    Returns (new_A, new_B) with bridge folded in.
    """
    rank = lora_A.shape[0]
    channel_size = rank // n_channels
    E = _expand_bridge(bridge, channel_size)

    if strategy == "into_A":
        return E @ lora_A, lora_B.clone()
    elif strategy == "into_B":
        return lora_A.clone(), lora_B @ E
    elif strategy == "balanced":
        L, R = _balanced_factors(E)
        return R @ lora_A, lora_B @ L
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ── Verification ────────────────────────────────────────────────────


def verify_layer(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    bridge: torch.Tensor,
    new_A: torch.Tensor,
    new_B: torch.Tensor,
    n_channels: int,
    scaling: float,
    n_samples: int = 4,
    seq_len: int = 8,
) -> dict:
    """Verify absorbed weights match original TeLoRA forward pass.

    Returns dict with max_abs_err, mean_abs_err, passed (bool).
    """
    rank = lora_A.shape[0]
    in_features = lora_A.shape[1]
    C = n_channels
    R = rank // C
    channel_size = R

    E = _expand_bridge(bridge, channel_size)

    with torch.no_grad():
        x = torch.randn(n_samples, seq_len, in_features)

        # Original TeLoRA path
        h = x @ lora_A.T
        h = h.view(n_samples, seq_len, C, R)
        h = torch.einsum("ij,...jk->...ik", bridge, h)
        h = h.reshape(n_samples, seq_len, rank)
        y_original = h @ lora_B.T * scaling

        # Absorbed standard LoRA path
        y_absorbed = (x @ new_A.T) @ new_B.T * scaling

        abs_err = (y_original - y_absorbed).abs()
        max_abs_err = float(abs_err.max().item())
        mean_abs_err = float(abs_err.mean().item())
        passed = torch.allclose(y_original, y_absorbed, atol=1e-5, rtol=1e-5)

    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "passed": passed,
    }


# ── Bridge diagnostics ──────────────────────────────────────────────


def bridge_stats(bridge: torch.Tensor) -> dict:
    """Compute diagnostic statistics for a bridge matrix."""
    C = bridge.shape[0]
    eye = torch.eye(C, dtype=bridge.dtype)
    deviation = float(torch.norm(bridge - eye).item())

    # Fiedler value (algebraic connectivity)
    try:
        from rhombic.spectral import fiedler_value
        B = bridge.detach().cpu().numpy()
        edges = [(i, j) for i in range(C) for j in range(i + 1, C)]
        weights = [float(abs(B[i, j])) for i, j in edges]
        fiedler = fiedler_value(C, edges, weights)
    except Exception:
        fiedler = None

    # Co-planar / cross-planar ratio (for C=6, RD topology)
    co_ratio = None
    cross_ratio = None
    if C == 6:
        try:
            B_np = bridge.detach().cpu().numpy()
            # RD co-planar pairs: (0,1), (2,3), (4,5) — opposite face pairs
            co_vals = [abs(B_np[i, j]) for i, j in [(0, 1), (2, 3), (4, 5)]]
            # Cross-planar: all other off-diagonal pairs
            cross_vals = []
            for i in range(C):
                for j in range(i + 1, C):
                    if (i, j) not in [(0, 1), (2, 3), (4, 5)]:
                        cross_vals.append(abs(B_np[i, j]))
            if cross_vals and sum(cross_vals) > 0:
                co_ratio = float(np.mean(co_vals))
                cross_ratio = float(np.mean(cross_vals))
        except Exception:
            pass

    return {
        "deviation": deviation,
        "fiedler": fiedler,
        "co_mean": co_ratio,
        "cross_mean": cross_ratio,
    }


# ── Main export logic ───────────────────────────────────────────────


def export_telora(
    checkpoint_dir: Path,
    output_dir: Path,
    strategy: str = "into_B",
    do_verify: bool = False,
) -> None:
    """Export a TeLoRA checkpoint as standard PEFT LoRA.

    Detects checkpoint format, absorbs all bridges, writes:
      - adapter_model.safetensors (standard LoRA weights)
      - adapter_config.json (PEFT-compatible, no TeLoRA metadata)
    """
    from safetensors.torch import save_file

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = _detect_format(checkpoint_dir)
    print(f"Checkpoint format: {fmt}")
    print(f"Strategy: {strategy}")
    print()

    if fmt == "peft-subdir":
        checkpoint_dir = checkpoint_dir / "adapter"
        fmt = "peft"

    # ── Research format ──────────────────────────────────────────
    if fmt == "research":
        data = _load_research_checkpoint(checkpoint_dir)
        layers = data["layers"]
        config = data["config"]

        rank = config.get("rank", 24)
        n_channels_default = config.get("n_channels", 6)
        alpha = config.get("alpha", 16.0)
        model_name = data["model_name"]
        target_modules = data["target_modules"]

        absorbed_weights = {}
        total_bridges = 0
        total_identity = 0
        all_stats = []
        verify_results = []

        for safe_name, info in sorted(layers.items()):
            lora_A = info["lora_A"].float()
            lora_B = info["lora_B"].float()
            bridge = info["bridge"]
            n_ch = info["n_channels"] or n_channels_default
            layer_rank = info["rank"]
            scaling = info["scaling"] or (alpha / layer_rank)

            module_path = _safe_name_to_module_path(safe_name)

            if bridge is not None:
                bridge = bridge.float()
                stats = bridge_stats(bridge)
                all_stats.append((module_path, stats))

                new_A, new_B = absorb_single_layer(
                    lora_A, lora_B, bridge, n_ch, strategy
                )

                if do_verify:
                    vr = verify_layer(
                        lora_A, lora_B, bridge, new_A, new_B,
                        n_ch, scaling,
                    )
                    verify_results.append((module_path, vr))

                total_bridges += 1
            else:
                # No bridge — identity (pure LoRA)
                new_A = lora_A
                new_B = lora_B
                total_identity += 1

            # PEFT key format: base_model.model.{path}.lora_A.default.weight
            peft_prefix = f"base_model.{module_path}"
            absorbed_weights[f"{peft_prefix}.lora_A.default.weight"] = new_A
            absorbed_weights[f"{peft_prefix}.lora_B.default.weight"] = new_B

        # Build PEFT adapter_config.json
        adapter_config = {
            "base_model_name_or_path": model_name,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "peft_type": "LORA",
            "r": rank,
            "target_modules": target_modules,
            "task_type": "CAUSAL_LM",
        }

    # ── PEFT format ──────────────────────────────────────────────
    elif fmt == "peft":
        data = _load_peft_checkpoint(checkpoint_dir)
        config = data["config"]
        all_weights = data["weights"]
        bridge_keys = data["bridge_keys"]

        n_channels_default = (
            config.get("telora_n_channels")
            or config.get("rhombi_n_channels")
            or config.get("n_channels", 6)
        )

        absorbed_weights = {}
        total_bridges = 0
        total_identity = 0
        all_stats = []
        verify_results = []

        # Map bridge keys by prefix for lookup
        bridge_map = {}
        for bk in bridge_keys:
            prefix = bk.rsplit(".bridge", 1)[0]
            bridge_map[prefix] = all_weights[bk]

        for key, tensor in all_weights.items():
            if key.endswith(".bridge"):
                continue  # Skip bridge tensors

            if "lora_A" in key:
                prefix = key.split(".lora_A")[0]
                bridge = bridge_map.get(prefix)

                if bridge is not None:
                    bridge = bridge.float()
                    A = tensor.float()
                    B_key = key.replace("lora_A", "lora_B")
                    B = all_weights.get(B_key)

                    if B is not None:
                        B = B.float()
                        n_ch = n_channels_default
                        layer_rank = A.shape[0]
                        alpha = config.get("lora_alpha", 16.0)
                        scaling = alpha / layer_rank

                        stats = bridge_stats(bridge)
                        all_stats.append((prefix, stats))

                        new_A, new_B = absorb_single_layer(
                            A, B, bridge, n_ch, strategy
                        )

                        if do_verify:
                            vr = verify_layer(
                                A, B, bridge, new_A, new_B,
                                n_ch, scaling,
                            )
                            verify_results.append((prefix, vr))

                        absorbed_weights[key] = new_A
                        absorbed_weights[B_key] = new_B
                        total_bridges += 1
                        continue

                # No bridge for this layer — pass through
                absorbed_weights[key] = tensor
                total_identity += 1

            elif "lora_B" in key:
                if key not in absorbed_weights:
                    absorbed_weights[key] = tensor
            else:
                # Non-LoRA tensors — pass through
                absorbed_weights[key] = tensor

        # Clean adapter config (strip TeLoRA/rhombi metadata)
        adapter_config = {
            k: v for k, v in config.items()
            if not k.startswith("telora") and not k.startswith("rhombi")
        }
        adapter_config.pop("bridge_mode", None)
        adapter_config.pop("n_channels", None)
        adapter_config.pop("bridge_trainable", None)

    # ── Save ─────────────────────────────────────────────────────

    save_file(absorbed_weights, str(output_dir / "adapter_model.safetensors"))
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    # ── Report ───────────────────────────────────────────────────

    total_params = sum(t.numel() for t in absorbed_weights.values())
    size_mb = sum(t.numel() * t.element_size() for t in absorbed_weights.values()) / 1e6

    print(f"{'='*60}")
    print(f"Export complete")
    print(f"{'='*60}")
    print(f"  Output:          {output_dir}")
    print(f"  Bridges absorbed: {total_bridges}")
    print(f"  Identity layers:  {total_identity}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size:            {size_mb:.1f} MB")
    print()

    # Bridge statistics
    if all_stats:
        deviations = [s["deviation"] for _, s in all_stats]
        fiedlers = [s["fiedler"] for _, s in all_stats if s["fiedler"] is not None]

        print(f"Bridge statistics ({len(all_stats)} layers):")
        print(f"  Deviation from I:  mean={np.mean(deviations):.4f}  "
              f"min={np.min(deviations):.4f}  max={np.max(deviations):.4f}")
        if fiedlers:
            print(f"  Fiedler value:     mean={np.mean(fiedlers):.4f}  "
                  f"min={np.min(fiedlers):.4f}  max={np.max(fiedlers):.4f}")

        # Co/cross for 6-channel bridges
        co_vals = [s["co_mean"] for _, s in all_stats if s["co_mean"] is not None]
        cross_vals = [s["cross_mean"] for _, s in all_stats if s["cross_mean"] is not None]
        if co_vals and cross_vals:
            print(f"  Co-planar mean:    {np.mean(co_vals):.4f}")
            print(f"  Cross-planar mean: {np.mean(cross_vals):.4f}")
            avg_ratio = np.mean(co_vals) / max(np.mean(cross_vals), 1e-8)
            print(f"  Co/Cross ratio:    {avg_ratio:.2f}")
        print()

    # Verification results
    if verify_results:
        all_passed = all(vr["passed"] for _, vr in verify_results)
        max_err = max(vr["max_abs_err"] for _, vr in verify_results)
        print(f"Verification: {'PASSED' if all_passed else 'FAILED'} "
              f"(max abs err: {max_err:.2e})")
        if not all_passed:
            for name, vr in verify_results:
                if not vr["passed"]:
                    print(f"  FAIL: {name}  "
                          f"max_err={vr['max_abs_err']:.2e}  "
                          f"mean_err={vr['mean_abs_err']:.2e}")
            sys.exit(1)
        print()

    print(f"Files written:")
    print(f"  {output_dir / 'adapter_model.safetensors'}")
    print(f"  {output_dir / 'adapter_config.json'}")
    print(f"\nThis adapter loads in any PEFT-compatible framework.")


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Export TeLoRA checkpoint as standard PEFT LoRA adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/export_telora.py --checkpoint results/exp3/ "
            "--output exports/exp3-absorbed/\n"
            "  python scripts/export_telora.py --checkpoint results/exp3/ "
            "--output exports/exp3-absorbed/ --verify\n"
            "  python scripts/export_telora.py --checkpoint competition/output-rhombi/adapter "
            "--strategy balanced --verify\n"
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="TeLoRA checkpoint directory (contains adapter_state.pt or adapter_model.safetensors)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for the standard LoRA adapter",
    )
    parser.add_argument(
        "--strategy", type=str, default="into_B",
        choices=["into_A", "into_B", "balanced"],
        help="Bridge absorption strategy (default: into_B). All three are exact.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify absorption with random input test per layer",
    )
    args = parser.parse_args()

    export_telora(
        checkpoint_dir=Path(args.checkpoint),
        output_dir=Path(args.output),
        strategy=args.strategy,
        do_verify=args.verify,
    )


if __name__ == "__main__":
    main()
