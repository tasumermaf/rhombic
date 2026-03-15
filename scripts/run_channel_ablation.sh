#!/bin/bash
# H1-H5 Channel Ablation — Track 3 (Hermes RTX 4090 16GB)
#
# Tests n_channels={3,4,6,8,12} to validate that n=6 (RD geometry)
# wins on val_loss/bridge_params ratio.
#
# All runs: TinyLlama 1.1B, identity init, default Steersman,
# 10K steps, rank=24, same training config as L3-L6.
#
# Usage: bash scripts/run_channel_ablation.sh

set -euo pipefail

PYTHON="/home/timm156/miniforge3/envs/isopsephy-mcp/bin/python"
export PYTHONPATH="/home/timm156/rhombic:$PYTHONPATH"
export WANDB_PROJECT="rhombic-channel-ablation"
export WANDB_ENTITY="promptcrafted"

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
STEPS=10000
FEEDBACK=100
LR="2e-4"
BATCH=2
GRAD_ACCUM=8
WARMUP=200
RANK=24
SEED=42
OUTPUT_BASE="/home/timm156/rhombic/results/channel-ablation"

mkdir -p "$OUTPUT_BASE"

echo "======================================================================"
echo "Track 3: Channel Ablation (H1-H5)"
echo "  Machine: hermes (RTX 4090 16GB)"
echo "  Model: $MODEL"
echo "  Steps: $STEPS"
echo "======================================================================"

for CHANNELS in 3 4 6 8 12; do
    CHANNEL_SIZE=$((RANK / CHANNELS))
    BRIDGE_PARAMS=$((CHANNELS * CHANNELS))
    RUN_NAME="H-ch${CHANNELS}"
    OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"

    # Skip if rank not divisible by channels
    if [ $((RANK % CHANNELS)) -ne 0 ]; then
        echo "SKIP: n_channels=$CHANNELS does not divide rank=$RANK"
        continue
    fi

    echo ""
    echo "----------------------------------------------------------------------"
    echo "  $RUN_NAME: n_channels=$CHANNELS, channel_size=$CHANNEL_SIZE, bridge_params=$BRIDGE_PARAMS"
    echo "----------------------------------------------------------------------"

    mkdir -p "$OUTPUT_DIR"

    $PYTHON scripts/train_cybernetic.py \
        --model "$MODEL" \
        --max-steps "$STEPS" \
        --feedback-interval "$FEEDBACK" \
        --lr "$LR" \
        --batch-size "$BATCH" \
        --gradient-accumulation "$GRAD_ACCUM" \
        --warmup-steps "$WARMUP" \
        --bridge-mode identity \
        --n-channels "$CHANNELS" \
        --steersman-preset default \
        --seed "$SEED" \
        --output "$OUTPUT_DIR" \
        2>&1 | tee "${OUTPUT_DIR}/train.log"

    echo "  $RUN_NAME COMPLETE."
    echo ""
done

echo "======================================================================"
echo "Track 3 COMPLETE. All channel ablation runs finished."
echo "Results in: $OUTPUT_BASE"
echo "======================================================================"
