#!/bin/bash
# Retrieve completed channel ablation results from Hermes server
# Run this after H1-H5 complete on Hermes

HERMES="timm156@hermes"
HERMES_PORT=2222
HERMES_BASE="rhombic/results/channel-ablation"
LOCAL_BASE="results/channel-ablation"

mkdir -p "$LOCAL_BASE"

for run in H-ch3 H-ch4 H-ch6 H-ch8 H-ch12; do
    echo "=== Retrieving $run ==="

    # Check if the run exists and has final bridges
    FINAL_COUNT=$(ssh -p $HERMES_PORT $HERMES \
        "ls $HERMES_BASE/$run/bridge_final_*.npy 2>/dev/null | wc -l" 2>/dev/null)

    if [ -z "$FINAL_COUNT" ] || [ "$FINAL_COUNT" -eq 0 ]; then
        echo "  $run: No final bridges found (run may not be complete yet)"

        # Check if running
        STEP_COUNT=$(ssh -p $HERMES_PORT $HERMES \
            "ls $HERMES_BASE/$run/bridge_step*.npy 2>/dev/null | wc -l" 2>/dev/null)
        if [ -n "$STEP_COUNT" ] && [ "$STEP_COUNT" -gt 0 ]; then
            LAST_STEP=$(ssh -p $HERMES_PORT $HERMES \
                "ls $HERMES_BASE/$run/bridge_step*.npy | sed 's/.*bridge_step//' | sed 's/_.*//' | sort -un | tail -1" 2>/dev/null)
            echo "  $run: In progress, last step $LAST_STEP ($STEP_COUNT bridge files)"
        fi
        continue
    fi

    echo "  $run: $FINAL_COUNT final bridges found"

    mkdir -p "$LOCAL_BASE/$run"

    # Copy results.json and config.json
    scp -P $HERMES_PORT "$HERMES:$HERMES_BASE/$run/results.json" "$LOCAL_BASE/$run/" 2>/dev/null
    scp -P $HERMES_PORT "$HERMES:$HERMES_BASE/$run/config.json" "$LOCAL_BASE/$run/" 2>/dev/null

    # Copy final bridges
    scp -P $HERMES_PORT "$HERMES:$HERMES_BASE/$run/bridge_final_*.npy" "$LOCAL_BASE/$run/" 2>/dev/null

    # Count what we got
    LOCAL_COUNT=$(ls "$LOCAL_BASE/$run/bridge_final_"*.npy 2>/dev/null | wc -l)
    echo "  $run: Retrieved $LOCAL_COUNT final bridges"
done

echo ""
echo "=== Retrieval complete ==="
echo "Now run: python scripts/analyze_channel_ablation.py"
