#!/bin/bash
# Pull Exp 2.5 results from RunPod and run comparison
#
# Usage: bash rhombic/scripts/pull_runpod_results.sh HOST PORT
# Example: bash rhombic/scripts/pull_runpod_results.sh 213.173.111.6 44315
#
# Prerequisites:
#   - RunPod training must be complete
#   - SSH key must be configured
#   - Local results/exp2/ must exist with Exp 2 data

set -e

HOST="${1:?Usage: pull_runpod_results.sh HOST PORT}"
PORT="${2:?Usage: pull_runpod_results.sh HOST PORT}"

PYTHON="C:/miniconda3/envs/falco/python.exe"
LOCAL_DIR="results/exp2_5"

echo "=== Pulling Exp 2.5 Results from RunPod ==="
echo "Host: $HOST:$PORT"
echo ""

# Check if training is done
echo "Checking training status..."
STATUS=$(ssh -o ConnectTimeout=10 -p "$PORT" "root@$HOST" \
    "if pgrep -f train_exp2_5 > /dev/null; then echo RUNNING; else echo DONE; fi" 2>/dev/null)

if [ "$STATUS" = "RUNNING" ]; then
    echo "Training still running. Last log lines:"
    ssh -p "$PORT" "root@$HOST" "tail -3 /workspace/training.log" 2>/dev/null
    echo ""
    echo "Wait for training to complete, then re-run this script."
    exit 1
fi

echo "Training complete. Downloading results..."
echo ""

# Create local directory
mkdir -p "$LOCAL_DIR"

# Download results
scp -P "$PORT" -r "root@$HOST:/workspace/rhombic/results/exp2_5/*" "$LOCAL_DIR/"

echo ""
echo "Downloaded to: $LOCAL_DIR/"
ls -la "$LOCAL_DIR/"
echo ""

# Show quick summary
echo "=== Quick Results Summary ==="
$PYTHON -c "
import json, sys
from pathlib import Path

d = Path('$LOCAL_DIR')
for sub in sorted(d.iterdir()):
    if not sub.is_dir():
        continue
    rf = sub / 'results.json'
    if not rf.exists():
        print(f'{sub.name}: no results.json')
        continue
    with open(rf) as f:
        data = json.load(f)
    ckpts = data.get('checkpoints', [])
    if not ckpts:
        print(f'{sub.name}: no checkpoints')
        continue
    last = ckpts[-1]
    ccp = last.get('coplanar_crossplanar', {})
    ratio = ccp.get('ratio', 0)
    print(f'{sub.name}:')
    print(f'  Steps: {last[\"step\"]}, Val loss: {last[\"val_loss\"]:.4f}')
    print(f'  Fiedler: {last.get(\"bridge_fiedler_mean\", 0):.5f}')
    print(f'  Co/Cross ratio: {ratio:.4f}')
    if ratio > 1.10:
        print(f'  >>> SIGNAL DETECTED: ratio {ratio:.3f} exceeds 1.10 threshold')
    print()
"

# Run comparison
echo ""
echo "=== Running Full Comparison (with spectral analysis) ==="
$PYTHON rhombic/scripts/compare_exp2_exp25.py \
    --exp2 results/exp2 \
    --exp25 "$LOCAL_DIR" \
    --deep \
    --output "$LOCAL_DIR/comparison_report.md"

echo ""
echo "=== Report written to: $LOCAL_DIR/comparison_report.md ==="
echo ""
echo "Don't forget to terminate the RunPod pod to stop billing!"
