"""Holly Battery Bridge Analysis — CPU-only post-hoc analysis.

The Holly Battery's flimmer-trainer monkey-patch saved lora_A/lora_B in
standard PEFT format. The 6×6 bridge matrices were NOT persisted (only
logged to WandB as summary statistics). From the experiment log:

    RhombiLoRA: wrapped 400 modules (rank=24, channels=6, bridge=identity)
    bridge/mean_deviation: 0.1848
    bridge/mean_fiedler: 0.10829

This script reconstructs the bridge's effect on weight structure by:
  1. Partitioning rank-24 lora_A into 6 channels of 4 rows each
  2. Computing the cross-channel correlation matrix (6×6)
  3. Analyzing this correlation for RD axis alignment (co/cross ratio)
  4. Comparing rhombi vs. control (standard LoRA, no channel structure)

The correlation matrix is an IMPLICIT bridge signature — it reveals
whether training under the bridge induced channel-structured correlations.

Usage:
  python scripts/analyze_holly_bridges.py
  python scripts/analyze_holly_bridges.py --backup-dir /path/to/backup
  python scripts/analyze_holly_bridges.py --output results/holly_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from rhombic.nn.topology import direction_pair_coupling
from rhombic.spectral import fiedler_value


# ── Implicit bridge from weight correlations ─────────────────────────


def extract_lora_pairs(checkpoint_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Extract lora_A and lora_B pairs from a .safetensors checkpoint.

    Returns dict mapping module name to {'A': array, 'B': array}.
    Uses PyTorch framework for bfloat16 support.
    """
    from safetensors import safe_open

    pairs: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        for key in f.keys():
            if "lora_A" in key or "lora_B" in key:
                tensor = f.get_tensor(key).float().numpy()
                # Normalize key to module name
                base = key.replace(".lora_A", "").replace(".lora_B", "")
                which = "A" if "lora_A" in key else "B"
                if base not in pairs:
                    pairs[base] = {}
                pairs[base][which] = tensor
    return pairs


def channel_correlation_matrix(
    lora_A: np.ndarray, n_channels: int = 6
) -> np.ndarray:
    """Compute cross-channel correlation from lora_A weight structure.

    Partitions the rank-R lora_A (R, in_features) into n_channels groups
    of R/n_channels rows. Computes cosine similarity between channel
    centroids.

    Returns (n_channels, n_channels) correlation matrix.
    """
    rank, hidden = lora_A.shape
    if rank % n_channels != 0:
        raise ValueError(f"rank {rank} not divisible by n_channels {n_channels}")

    channel_size = rank // n_channels

    # Channel centroids: mean of each channel's rows
    centroids = np.zeros((n_channels, hidden))
    for c in range(n_channels):
        start = c * channel_size
        end = start + channel_size
        centroids[c] = lora_A[start:end].mean(axis=0)

    # Cosine similarity matrix between centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = centroids / norms
    corr = normalized @ normalized.T

    return corr


def channel_gram_matrix(
    lora_A: np.ndarray, n_channels: int = 6
) -> np.ndarray:
    """Compute cross-channel Gram matrix: how much do channels overlap?

    For each channel pair (i, j), computes the mean absolute dot product
    between all row pairs. This captures not just centroid alignment but
    the full subspace relationship.

    Returns (n_channels, n_channels) Gram matrix.
    """
    rank, hidden = lora_A.shape
    channel_size = rank // n_channels

    gram = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        rows_i = lora_A[i * channel_size:(i + 1) * channel_size]
        for j in range(n_channels):
            rows_j = lora_A[j * channel_size:(j + 1) * channel_size]
            # Mean absolute dot product between all pairs
            dots = np.abs(rows_i @ rows_j.T)
            gram[i, j] = dots.mean()

    # Normalize by diagonal
    diag = np.sqrt(np.diag(gram))
    diag = np.maximum(diag, 1e-12)
    gram = gram / (diag[:, None] * diag[None, :])

    return gram


def analyze_implicit_bridge(
    corr: np.ndarray, method: str = "correlation"
) -> dict:
    """Compute axis alignment metrics from an implicit bridge matrix.

    Works with either correlation or Gram matrices — any 6×6 matrix
    where off-diagonal values indicate channel coupling.
    """
    n = corr.shape[0]
    if n != 6:
        return {"error": f"Expected 6x6, got {n}x{n}"}

    coupling = direction_pair_coupling()

    # Co-planar and cross-planar pairs from RD geometry
    co_pairs = []
    cross_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if coupling[i, j] >= 4:
                co_pairs.append((i, j))
            else:
                cross_pairs.append((i, j))

    # Absolute off-diagonal values
    co_vals = [abs(corr[i, j]) for i, j in co_pairs]
    cross_vals = [abs(corr[i, j]) for i, j in cross_pairs]

    co_mean = float(np.mean(co_vals)) if co_vals else 0.0
    cross_mean = float(np.mean(cross_vals)) if cross_vals else 0.0
    ratio = co_mean / max(cross_mean, 1e-12)

    # Fiedler value of the off-diagonal structure
    off_diag = corr.copy()
    np.fill_diagonal(off_diag, 0.0)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = [float(abs(off_diag[i, j])) for i, j in edges]
    fiedler = fiedler_value(n, edges, weights)

    return {
        "method": method,
        "co_mean": round(co_mean, 6),
        "cross_mean": round(cross_mean, 6),
        "co_cross_ratio": round(ratio, 3),
        "fiedler": round(fiedler, 6),
        "n_co_pairs": len(co_pairs),
        "n_cross_pairs": len(cross_pairs),
    }


def analyze_checkpoint(
    checkpoint_path: Path, n_channels: int = 6
) -> dict:
    """Full implicit bridge analysis of a checkpoint."""
    pairs = extract_lora_pairs(checkpoint_path)

    if not pairs:
        return {"error": "No LoRA parameters found", "path": str(checkpoint_path)}

    attn_pairs = {
        k: v for k, v in pairs.items()
        if "A" in v and v["A"].shape[0] == 24  # rank=24 only
    }

    if not attn_pairs:
        return {"error": "No rank-24 LoRA modules found", "path": str(checkpoint_path)}

    # Analyze correlation and Gram for each module
    all_corr_ratios = []
    all_gram_ratios = []
    all_corr_fiedlers = []
    per_module = {}

    for name, weights in sorted(attn_pairs.items()):
        lora_A = weights["A"]

        try:
            corr = channel_correlation_matrix(lora_A, n_channels)
            gram = channel_gram_matrix(lora_A, n_channels)
        except ValueError:
            continue

        corr_metrics = analyze_implicit_bridge(corr, method="correlation")
        gram_metrics = analyze_implicit_bridge(gram, method="gram")

        short_name = name.split(".")[-2] if "." in name else name
        per_module[short_name] = {
            "correlation": corr_metrics,
            "gram": gram_metrics,
        }

        if "co_cross_ratio" in corr_metrics:
            all_corr_ratios.append(corr_metrics["co_cross_ratio"])
            all_corr_fiedlers.append(corr_metrics["fiedler"])
        if "co_cross_ratio" in gram_metrics:
            all_gram_ratios.append(gram_metrics["co_cross_ratio"])

    summary = {
        "n_modules_analyzed": len(per_module),
        "correlation": {
            "mean_ratio": round(float(np.mean(all_corr_ratios)), 3) if all_corr_ratios else None,
            "median_ratio": round(float(np.median(all_corr_ratios)), 3) if all_corr_ratios else None,
            "std_ratio": round(float(np.std(all_corr_ratios)), 3) if all_corr_ratios else None,
            "mean_fiedler": round(float(np.mean(all_corr_fiedlers)), 6) if all_corr_fiedlers else None,
        },
        "gram": {
            "mean_ratio": round(float(np.mean(all_gram_ratios)), 3) if all_gram_ratios else None,
            "median_ratio": round(float(np.median(all_gram_ratios)), 3) if all_gram_ratios else None,
            "std_ratio": round(float(np.std(all_gram_ratios)), 3) if all_gram_ratios else None,
        },
    }
    return summary


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Holly Battery Bridge Analysis (CPU-only, implicit reconstruction)"
    )
    parser.add_argument(
        "--backup-dir", type=str,
        default="rhombi-experiment-backup",
        help="Path to Holly Battery backup directory",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for analysis JSON (default: {backup-dir}/holly_bridge_analysis.json)",
    )
    args = parser.parse_args()

    backup = Path(args.backup_dir)
    if not backup.exists():
        print(f"ERROR: Backup directory not found: {backup}")
        return

    # Find all .safetensors files
    checkpoints = sorted(backup.rglob("*.safetensors"))
    if not checkpoints:
        print(f"No .safetensors files found in {backup}")
        return

    # Group by pod
    pods = {}
    for cp in checkpoints:
        pod = cp.parts[-3] if len(cp.parts) >= 3 else "unknown"
        if pod not in pods:
            pods[pod] = []
        pods[pod].append(cp)

    print(f"Found {len(checkpoints)} checkpoint(s) across {len(pods)} pod(s)")
    print(f"{'='*70}")

    # Known metadata from experiment logs
    print("\nWandB-logged bridge metrics (from experiment log):")
    print("  bridge/mean_deviation: 0.1848")
    print("  bridge/mean_fiedler:   0.10829")
    print("  (Bridge was trained but NOT saved in safetensors)")
    print(f"\n{'='*70}")

    all_results = {}

    for pod_name, pod_checkpoints in sorted(pods.items()):
        is_rhombi = "rhombi" in pod_name.lower()
        pod_type = "RHOMBI" if is_rhombi else "CONTROL"

        print(f"\n--- {pod_name} ({pod_type}) ---")
        print(f"    {len(pod_checkpoints)} checkpoint(s)")

        for cp_path in pod_checkpoints:
            epoch_match = re.search(r"epoch(\d+)", cp_path.name)
            epoch = int(epoch_match.group(1)) if epoch_match else -1

            print(f"\n  {cp_path.name} (epoch {epoch}):")
            result = analyze_checkpoint(cp_path)

            if "error" in result:
                print(f"    ERROR: {result['error']}")
                continue

            rel_path = str(cp_path.relative_to(backup))
            all_results[rel_path] = {
                "pod": pod_name,
                "type": pod_type,
                "epoch": epoch,
                **result,
            }

            n = result["n_modules_analyzed"]
            c = result["correlation"]
            g = result["gram"]
            print(f"    Modules analyzed: {n}")
            if c["mean_ratio"] is not None:
                print(f"    Correlation — ratio: {c['mean_ratio']:.3f} "
                      f"(±{c['std_ratio']:.3f}), "
                      f"fiedler: {c['mean_fiedler']:.6f}")
            if g["mean_ratio"] is not None:
                print(f"    Gram       — ratio: {g['mean_ratio']:.3f} "
                      f"(±{g['std_ratio']:.3f})")

    # Summary comparison: rhombi vs control (epoch 50 only)
    rhombi_50 = [v for v in all_results.values()
                 if v["type"] == "RHOMBI" and v["epoch"] == 50]
    control_50 = [v for v in all_results.values()
                  if v["type"] == "CONTROL" and v["epoch"] == 50]

    if rhombi_50 and control_50:
        print(f"\n{'='*70}")
        print("COMPARISON: RHOMBI vs CONTROL (epoch 50)")
        print(f"{'='*70}")

        r = rhombi_50[0]
        for c in control_50:
            r_ratio = r["correlation"]["mean_ratio"]
            c_ratio = c["correlation"]["mean_ratio"]
            if r_ratio is not None and c_ratio is not None:
                diff = r_ratio - c_ratio
                print(f"\n  {c['pod']} vs rhombi:")
                print(f"    Correlation co/cross ratio:")
                print(f"      Rhombi:  {r_ratio:.3f}")
                print(f"      Control: {c_ratio:.3f}")
                print(f"      Delta:   {diff:+.3f}")

                r_gram = r["gram"]["mean_ratio"]
                c_gram = c["gram"]["mean_ratio"]
                if r_gram is not None and c_gram is not None:
                    print(f"    Gram co/cross ratio:")
                    print(f"      Rhombi:  {r_gram:.3f}")
                    print(f"      Control: {c_gram:.3f}")
                    print(f"      Delta:   {r_gram - c_gram:+.3f}")

    # Save JSON
    output_path = Path(args.output) if args.output else backup / "holly_bridge_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
