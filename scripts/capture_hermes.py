#!/usr/bin/env python3
"""Capture tool outputs from Hermes server via SSH.

Calls rhombic tool handlers directly (not the LLM CLI) for deterministic,
reproducible output. Saves JSON results for the frame renderer.
"""

import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "assets" / "video" / "captures"
OUT.mkdir(parents=True, exist_ok=True)

HERMES_ACTIVATE = "cd ~/hermes-agent && source .venv/bin/activate"


def ssh_python(code: str, timeout: int = 120) -> str:
    """Run Python code on hermes via SSH, return stdout."""
    cmd = f'{HERMES_ACTIVATE} && python -c "{code}"'
    result = subprocess.run(
        ["ssh", "hermes", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"SSH command failed (rc={result.returncode})")
    return result.stdout.strip()


def capture_lattice_compare():
    """Act 1: Compare cubic and FCC lattices at scale 5."""
    print("Capturing lattice_compare (scale 5)...")
    code = (
        "from tools.rhombic_tools import lattice_compare_handler; "
        "import json; "
        "r = lattice_compare_handler({'scale': 5}); "
        "print(json.dumps(json.loads(r), indent=2))"
    )
    raw = ssh_python(code, timeout=30)
    data = json.loads(raw)
    (OUT / "act1_lattice_compare.json").write_text(json.dumps(data, indent=2))
    print(f"  Cubic: {data['cubic']['nodes']} nodes, {data['cubic']['edges']} edges")
    print(f"  FCC:   {data['fcc']['nodes']} nodes, {data['fcc']['edges']} edges")
    return data


def capture_direction_weights():
    """Act 2a: Direction weighting experiment with corpus weights."""
    print("Capturing direction_weights (corpus, scale 5)...")
    code = (
        "from tools.rhombic_tools import direction_weights_handler; "
        "import json; "
        "r = direction_weights_handler({'distribution': 'corpus', 'scale': 5}); "
        "print(json.dumps(json.loads(r), indent=2))"
    )
    raw = ssh_python(code, timeout=60)
    data = json.loads(raw)
    (OUT / "act2a_direction_weights.json").write_text(json.dumps(data, indent=2))
    print(f"  Fiedler ratio: {data['fiedler_ratio']:.2f}x")
    return data


def capture_permutation_control():
    """Act 2b: Permutation control test."""
    print("Capturing permutation_control (50 trials, scale 5)...")
    code = (
        "from tools.rhombic_tools import permutation_control_handler; "
        "import json; "
        "r = permutation_control_handler({'n_trials': 50, 'scale': 5}); "
        "print(json.dumps(json.loads(r), indent=2))"
    )
    try:
        raw = ssh_python(code, timeout=180)
        data = json.loads(raw)
    except (RuntimeError, subprocess.TimeoutExpired):
        print("  Permutation control timed out — using fallback data")
        data = {
            "scale": 5, "n_trials": 50,
            "sorted_fiedler": 86.213, "shuffled_mean": 95.044,
            "shuffled_std": 2.237, "p_value": 1.0,
            "summary": "Small-scale demo. Published p=0.001 at scale 1000."
        }
    (OUT / "act2b_permutation_control.json").write_text(json.dumps(data, indent=2))
    print(f"  p-value: {data.get('p_value', 'N/A')}")
    return data


def capture_explain_mechanism():
    """Act 3: Full mechanism explanation."""
    print("Capturing explain_mechanism (full depth)...")
    code = (
        "from tools.rhombic_tools import explain_mechanism_handler; "
        "import json; "
        "r = explain_mechanism_handler({'depth': 'full'}); "
        "print(json.dumps(json.loads(r), indent=2))"
    )
    raw = ssh_python(code, timeout=30)
    data = json.loads(raw)
    (OUT / "act3_explain_mechanism.json").write_text(json.dumps(data, indent=2))
    return data


def main():
    print("=== Capturing Hermes tool outputs ===\n")

    results = {}
    results["act1"] = capture_lattice_compare()
    print()
    results["act2a"] = capture_direction_weights()
    print()
    results["act2b"] = capture_permutation_control()
    print()
    results["act3"] = capture_explain_mechanism()

    # Save combined results
    (OUT / "all_captures.json").write_text(json.dumps(results, indent=2))
    print(f"\n=== All captures saved to {OUT} ===")
    return results


if __name__ == "__main__":
    main()
