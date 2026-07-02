#!/bin/bash
# G2-001: Bridge-swap evaluation — do bridges encode task-specific behavior?
# Run AFTER P0 completes. Inference-only, ~1h, ~6 GB VRAM.
#
# Tests: Can you swap bridge matrices between task adapters and preserve behavior?
# Uses existing fingerprint bridges (code vs math, Qwen 1.5B, n=6, 6×6 bridges).
# The 36-parameter composition thesis (L-016): one set of lora_A/B weights,
# N task behaviors via bridge selection.

set -e
cd "$(dirname "$0")/.."

echo "=== G2-001: Bridge-Swap Evaluation ==="
echo "Model: Qwen/Qwen2.5-1.5B-Instruct (matching fingerprint training)"
echo "Fingerprints: results/fingerprints/ (code, math)"
echo "Starting: $(date)"
echo ""

python scripts/eval_bridge_swap.py \
  --fingerprint-dir results/fingerprints \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --eval-dataset wikitext \
  --max-eval-samples 500 \
  --batch-size 4 \
  --output results/bridge-swap/g2-001.json

echo ""
echo "G2-001 complete: $(date)"
echo "Results: results/bridge-swap/g2-001.json"
