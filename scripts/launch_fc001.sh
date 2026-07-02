#!/bin/bash
# FC-001: Fixed contrastive weight c_w=0.02, n=6, 10K steps
# Run AFTER P0 completes. Overnight experiment.
#
# Tests: Does a fixed c_w produce comparable BD ratios to adaptive?
# Control: H-ch6 baseline (adaptive c_w at n=6) achieved 22,477:1.
# Hypothesis: Fixed c_w=0.02 from FI-004 annealing (optimal at 0.017) should still
# produce strong BD with simpler implementation (no Steersman needed).

set -e
cd "$(dirname "$0")/.."

echo "=== FC-001: Fixed Contrastive Weight ==="
echo "c_w=0.02, n=6, 10K steps, Qwen2.5-1.5B-Instruct"
echo "Starting: $(date)"
echo ""

python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --n-channels 6 \
  --max-steps 10000 \
  --fixed-contrastive 0.02 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --warmup-steps 200 \
  --checkpoint-steps 100 \
  --seed 42 \
  --output results/fc-001

echo ""
echo "FC-001 complete: $(date)"
echo "Results: results/fc-001/"
