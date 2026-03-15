#!/bin/bash
# Deploy Holly Multi-Seed to 6 RunPod pods in parallel
# Run from: /c/Falco/rhombic/
#
# Requires: SSH access to each pod via RunPod proxy
# Format: ssh root@{POD_ID}-ssh.proxy.runpod.net

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Pod manifest
declare -A PODS
PODS[ku57em9ko4ac2x]="control 42"
PODS[aybdjh0chqm2g4]="control 123"
PODS[7trt449enqipmr]="control 456"
PODS[nagjuyp22j8tqw]="rhombi 42"
PODS[upyyxx6ocaeuxt]="rhombi 123"
PODS[r1ao6zjz02yfug]="rhombi 456"

deploy_pod() {
    local POD_ID=$1
    local CONFIG=$2
    local SEED=$3
    local SSH_HOST="root@${POD_ID}-ssh.proxy.runpod.net"

    echo "[$POD_ID] Deploying $CONFIG seed=$SEED..."

    # Upload setup script
    scp -o StrictHostKeyChecking=no "$SCRIPT_DIR/holly_pod_setup.sh" "$SSH_HOST:/workspace/" 2>/dev/null

    # Launch in tmux (survives SSH disconnect)
    ssh -o StrictHostKeyChecking=no "$SSH_HOST" \
        "tmux new-session -d -s training 'bash /workspace/holly_pod_setup.sh $CONFIG $SEED 2>&1 | tee /workspace/deploy.log'"

    echo "[$POD_ID] Launched $CONFIG seed=$SEED in tmux"
}

echo "=== Deploying to 6 pods ==="
echo ""

for POD_ID in "${!PODS[@]}"; do
    read CONFIG SEED <<< "${PODS[$POD_ID]}"
    deploy_pod "$POD_ID" "$CONFIG" "$SEED" &
done

wait
echo ""
echo "=== All 6 pods deploying ==="
echo ""
echo "Monitor any pod:"
echo "  ssh root@{POD_ID}-ssh.proxy.runpod.net 'tail -f /workspace/deploy.log'"
echo ""
echo "Check all pods:"
echo "  python scripts/check_holly_fleet.py"
