#!/bin/bash
# Experiment 2.5: Geometric Dataset Validation
#
# Step 1: Generate geometric dataset
# Step 2: Run FCC config on geometric data (quick validation, 2K steps)
# Step 3: If co/cross ratio improves → run full 9-config sweep
#
# Expected runtime:
#   Quick validation: ~2.2h (2K steps FCC only)
#   Full sweep: ~99h (9 configs × 11h each)
#
# Launch:
#   Quick: nohup bash rhombic/scripts/run_exp2_5.sh quick > results/exp2_5/training.log 2>&1 &
#   Full:  nohup bash rhombic/scripts/run_exp2_5.sh full > results/exp2_5/training.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

PYTHON="C:/miniconda3/envs/falco/python.exe"
DATA_DIR="rhombic/data/geometric"
OUTPUT="results/exp2_5"
MODE="${1:-quick}"

mkdir -p "$OUTPUT"

echo "=== Experiment 2.5: Geometric Dataset Validation ==="
echo "Mode: $MODE"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Step 1: Generate geometric dataset
echo "=== Step 1: Generate Geometric Dataset ==="
echo "Started: $(date)"
$PYTHON rhombic/scripts/generate_geometric_dataset.py \
    --output "$DATA_DIR" \
    --components 1,3,4 \
    --n1 10000 \
    --n3 8000 \
    --n4 5000 \
    --coplanar-bias 0.6

echo "Dataset generation complete: $(date)"
echo ""

# Generate mixed dataset (need Alpaca downloaded first)
echo "=== Downloading Alpaca for mixing ==="
$PYTHON -c "
from datasets import load_dataset
import json
ds = load_dataset('yahma/alpaca-cleaned', split='train')
data = [{'instruction': x['instruction'], 'input': x.get('input',''), 'output': x['output']} for x in ds]
with open('$DATA_DIR/alpaca_cleaned.json', 'w') as f:
    json.dump(data, f)
print(f'Saved {len(data)} Alpaca examples')
"

echo "=== Generating mixed dataset ==="
$PYTHON rhombic/scripts/generate_geometric_dataset.py \
    --output "$DATA_DIR" \
    --components 1,3,4 \
    --mix-with "$DATA_DIR/alpaca_cleaned.json" \
    --ratio 0.5

echo ""

if [ "$MODE" = "quick" ]; then
    # Quick validation: FCC only, geometric data, 2K steps
    echo "=== Quick Validation: FCC on Geometric (2K steps) ==="
    echo "Started: $(date)"
    $PYTHON rhombic/scripts/train_exp2_5.py \
        --config 2 \
        --dataset-mode geometric \
        --data-dir "$DATA_DIR" \
        --output "$OUTPUT" \
        --max-steps 3000
    echo "Quick validation complete: $(date)"

elif [ "$MODE" = "full" ]; then
    # Full 9-config sweep
    echo "=== Full Sweep: 3 topologies × 3 datasets ==="
    echo "Started: $(date)"
    $PYTHON rhombic/scripts/train_exp2_5.py \
        --sweep \
        --data-dir "$DATA_DIR" \
        --output "$OUTPUT"
    echo "Full sweep complete: $(date)"

elif [ "$MODE" = "fcc-only" ]; then
    # FCC on all 3 datasets, full 10K steps
    echo "=== FCC on all 3 datasets (10K steps each) ==="
    for DS_MODE in geometric mixed50 alpaca; do
        echo "=== FCC + $DS_MODE ==="
        echo "Started: $(date)"
        $PYTHON rhombic/scripts/train_exp2_5.py \
            --config 2 \
            --dataset-mode "$DS_MODE" \
            --data-dir "$DATA_DIR" \
            --output "$OUTPUT"
        echo "FCC + $DS_MODE complete: $(date)"
        echo ""
    done

else
    echo "Unknown mode: $MODE"
    echo "Usage: run_exp2_5.sh [quick|full|fcc-only]"
    exit 1
fi

echo ""
echo "=== Experiment 2.5 complete ==="
echo "Finished: $(date)"
echo "Results in: $OUTPUT"
