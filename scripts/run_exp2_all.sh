#!/bin/bash
# Chain all 3 Experiment 2 configs sequentially.
# Expected total runtime: ~31.5 hours on RTX 6000 Ada 48GB.
# Launch with: nohup bash rhombic/scripts/run_exp2_all.sh > results/exp2/training_all.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

PYTHON="C:/miniconda3/envs/falco/python.exe"
SCRIPT="rhombic/scripts/train_exp2_scale.py"
OUTPUT="results/exp2"

echo "=== Experiment 2: Full 3-config chain ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

echo "=== Config 2: RhombiLoRA FCC r24 (learnable bridge) ==="
echo "Started: $(date)"
$PYTHON $SCRIPT --config 2 --output $OUTPUT
echo "Config 2 complete: $(date)"
echo ""

echo "=== Config 3: RhombiLoRA cubic r24 (learnable bridge, 3-ch) ==="
echo "Started: $(date)"
$PYTHON $SCRIPT --config 3 --output $OUTPUT
echo "Config 3 complete: $(date)"
echo ""

echo "=== Config 1: Standard LoRA r24 (full 10K re-run) ==="
echo "Started: $(date)"
$PYTHON $SCRIPT --config 1 --output $OUTPUT
echo "Config 1 complete: $(date)"
echo ""

echo "=== All 3 configs complete ==="
echo "Finished: $(date)"
