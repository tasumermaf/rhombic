"""Peek at bridge matrices from an in-progress training run.

Downloads the latest checkpoint bridges from RunPod and runs quick
spectral diagnostics. Useful for early-stop decisions.

Usage:
  python scripts/peek_bridges.py HOST PORT
  python scripts/peek_bridges.py 213.173.111.6 44315
"""

from __future__ import annotations

import sys
import subprocess
import tempfile
import glob
from pathlib import Path

import numpy as np


# RD direction pair topology
CO_PLANAR_PAIRS = [(0, 1), (2, 3), (4, 5)]
CROSS_PLANAR_PAIRS = [
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 4), (2, 5), (3, 4), (3, 5),
]


def quick_bridge_analysis(bridges: list[np.ndarray], label: str = "") -> dict:
    """Quick spectral analysis of bridge matrices."""
    if not bridges:
        return {"error": "no bridges"}

    # Co/cross ratio
    co_vals = []
    cross_vals = []
    for B in bridges:
        for i, j in CO_PLANAR_PAIRS:
            co_vals.append(abs(B[i, j]))
        for i, j in CROSS_PLANAR_PAIRS:
            cross_vals.append(abs(B[i, j]))
    co_mean = np.mean(co_vals)
    cross_mean = np.mean(cross_vals)
    ratio = co_mean / max(cross_mean, 1e-12)

    # Eigenvalue analysis
    all_eigvals = []
    for B in bridges:
        eigvals = np.sort(np.linalg.eigvalsh(B))
        all_eigvals.append(eigvals)
    all_eigvals = np.array(all_eigvals)
    mean_eigvals = all_eigvals.mean(axis=0)
    spread = float(mean_eigvals.max() - mean_eigvals.min())

    # Distance from identity
    dists = [float(np.linalg.norm(B - np.eye(6))) for B in bridges]

    return {
        "label": label,
        "n_bridges": len(bridges),
        "co_cross_ratio": float(ratio),
        "co_mean": float(co_mean),
        "cross_mean": float(cross_mean),
        "eigenvalues": [float(v) for v in mean_eigvals],
        "eigval_spread": spread,
        "identity_distance": float(np.mean(dists)),
    }


def print_analysis(result: dict):
    """Pretty-print bridge analysis."""
    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    ratio = result["co_cross_ratio"]
    spread = result["eigval_spread"]
    signal = "SIGNAL" if ratio > 1.10 else "no signal"

    print(f"  {result['label']} ({result['n_bridges']} bridges)")
    print(f"  Co/Cross ratio:   {ratio:.4f}  [{signal}]")
    print(f"  Co-planar mean:   {result['co_mean']:.6f}")
    print(f"  Cross-planar mean:{result['cross_mean']:.6f}")
    print(f"  Eigenvalues:      {[f'{v:.4f}' for v in result['eigenvalues']]}")
    print(f"  Eigval spread:    {spread:.5f}")
    print(f"  Dist from I_6:    {result['identity_distance']:.4f}")
    print()


def load_bridges_at_step(directory: Path, step: int) -> list[np.ndarray]:
    """Load all bridge matrices for a given step."""
    pattern = str(directory / f"bridge_step{step}_*.npy")
    files = sorted(glob.glob(pattern))
    return [np.load(f) for f in files if np.load(f).shape == (6, 6)]


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/peek_bridges.py HOST PORT")
        sys.exit(1)

    host = sys.argv[1]
    port = sys.argv[2]

    # Find available steps on remote
    print(f"Checking RunPod at {host}:{port}...")
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
         "-p", port, f"root@{host}",
         "ls /workspace/rhombic/results/exp2_5/geometric_rhombi_fcc_r24/ 2>/dev/null | "
         "grep '^bridge_step' | sed 's/bridge_step//' | sed 's/_.*//' | sort -un"],
        capture_output=True, text=True, timeout=15
    )

    if result.returncode != 0:
        print(f"SSH failed: {result.stderr}")
        sys.exit(1)

    steps = [int(s) for s in result.stdout.strip().split('\n') if s.strip()]
    print(f"Available steps: {steps}")

    if len(steps) < 2:
        print("Only step 0 available — training hasn't checkpointed yet.")
        sys.exit(0)

    # Get the latest non-zero step
    latest = max(s for s in steps if s > 0)
    print(f"\nDownloading bridges at step {latest}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download latest step bridges
        scp_result = subprocess.run(
            ["scp", "-P", port, "-o", "StrictHostKeyChecking=no",
             f"root@{host}:/workspace/rhombic/results/exp2_5/geometric_rhombi_fcc_r24/bridge_step{latest}_*.npy",
             tmpdir],
            capture_output=True, text=True, timeout=60
        )

        if scp_result.returncode != 0:
            print(f"SCP failed: {scp_result.stderr}")
            sys.exit(1)

        bridges = load_bridges_at_step(Path(tmpdir), latest)
        print(f"Loaded {len(bridges)} bridges at step {latest}")
        print()

        # Analyze
        result = quick_bridge_analysis(bridges, f"Exp 2.5 Geometric (step {latest})")
        print("=== Early Bridge Analysis ===")
        print_analysis(result)

        # Compare with Exp 2 baseline
        exp2_bridges = []
        exp2_dir = Path("results/exp2/rhombi_fcc_r24")
        if exp2_dir.exists():
            all_npy = sorted(glob.glob(str(exp2_dir / "bridge_step*.npy")))
            if all_npy:
                step_nums = set()
                for f in all_npy:
                    for p in Path(f).stem.split("_"):
                        if p.startswith("step"):
                            try:
                                step_nums.add(int(p[4:]))
                            except ValueError:
                                pass
                max_step = max(step_nums)
                exp2_bridges = [np.load(f) for f in all_npy
                                if f"step{max_step}_" in f and np.load(f).shape == (6, 6)]

        if exp2_bridges:
            exp2_result = quick_bridge_analysis(exp2_bridges, f"Exp 2 Alpaca (step {max_step})")
            print("=== Exp 2 Baseline ===")
            print_analysis(exp2_result)

            # Verdict
            print("=== Early Verdict ===")
            r25 = result["co_cross_ratio"]
            r2 = exp2_result["co_cross_ratio"]
            delta = r25 - r2
            print(f"  Co/Cross: {r25:.4f} (Exp 2.5) vs {r2:.4f} (Exp 2)")
            print(f"  Delta:    {delta:+.4f}")
            if r25 > 1.10:
                print(f"  >>> THRESHOLD MET at step {latest}! Proceed to full comparison.")
            elif r25 > r2 * 1.05:
                print(f"  >>> Trending positive ({delta:+.4f}). Continue training.")
            else:
                print(f"  >>> No signal yet. May need dataset iteration or more steps.")


if __name__ == "__main__":
    main()
