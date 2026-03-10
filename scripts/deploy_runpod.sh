#!/bin/bash
# Deploy training script to a RunPod pod
# Usage: bash scripts/deploy_runpod.sh HOST PORT SCRIPT_NAME [EXTRA_ARGS]
# Example: bash scripts/deploy_runpod.sh 213.173.111.6 44315 train_overfit_diagnostic.py "--max-steps 10000"
# Example: bash scripts/deploy_runpod.sh 213.173.111.6 44316 train_contrastive_bridge.py "--phase-a-steps 1000"

set -e

HOST=$1
PORT=$2
SCRIPT=$3
EXTRA_ARGS="${4:-}"

if [ -z "$HOST" ] || [ -z "$PORT" ] || [ -z "$SCRIPT" ]; then
    echo "Usage: bash scripts/deploy_runpod.sh HOST PORT SCRIPT_NAME [EXTRA_ARGS]"
    exit 1
fi

REMOTE="root@$HOST"
SSH_CMD="ssh -o StrictHostKeyChecking=no -p $PORT $REMOTE"
SCP_CMD="scp -o StrictHostKeyChecking=no -P $PORT"

echo "=== Deploying $SCRIPT to $HOST:$PORT ==="

# 1. Install dependencies
echo "--- Installing dependencies ---"
$SSH_CMD "pip install -q rhombic transformers datasets accelerate numpy torch 2>&1 | tail -3"

# 2. Create workspace
echo "--- Setting up workspace ---"
$SSH_CMD "mkdir -p /workspace/rhombic/scripts /workspace/rhombic/results"

# 3. Transfer the rhombic package and script
echo "--- Transferring files ---"
$SCP_CMD -r rhombic/nn $REMOTE:/workspace/rhombic/
$SCP_CMD -r rhombic/spectral.py $REMOTE:/workspace/rhombic/ 2>/dev/null || true
$SCP_CMD -r rhombic/polyhedron.py $REMOTE:/workspace/rhombic/ 2>/dev/null || true
$SCP_CMD -r rhombic/__init__.py $REMOTE:/workspace/rhombic/ 2>/dev/null || true
# Transfer ALL training scripts (cybernetic trainer imports from exp2 and contrastive)
$SCP_CMD scripts/train_exp2_scale.py $REMOTE:/workspace/rhombic/scripts/ 2>/dev/null || true
$SCP_CMD scripts/train_contrastive_bridge.py $REMOTE:/workspace/rhombic/scripts/ 2>/dev/null || true
$SCP_CMD scripts/$SCRIPT $REMOTE:/workspace/rhombic/scripts/

# 4. Make rhombic importable
echo "--- Setting up Python path ---"
$SSH_CMD "cd /workspace && pip install -e rhombic 2>/dev/null || PYTHONPATH=/workspace python -c 'import rhombic; print(\"rhombic OK\")' 2>/dev/null || echo 'Will use PYTHONPATH'"

# 5. Launch training in tmux (survives SSH disconnect)
echo "--- Launching training ---"
$SSH_CMD "tmux kill-session -t training 2>/dev/null; true"
$SSH_CMD "cd /workspace && tmux new-session -d -s training 'PYTHONPATH=/workspace python -u rhombic/scripts/$SCRIPT $EXTRA_ARGS 2>&1 | tee /workspace/training.log'"

echo ""
echo "=== Training launched in tmux session 'training' ==="
echo "Monitor: ssh -p $PORT root@$HOST 'tail -f /workspace/training.log'"
echo "Attach:  ssh -p $PORT root@$HOST -t 'tmux attach -t training'"
echo "Status:  ssh -p $PORT root@$HOST 'tail -5 /workspace/training.log'"
