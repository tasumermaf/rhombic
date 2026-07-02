#!/bin/bash
# RunPod A100: Phase 2 Evals — L7 (Language Benchmarks) + L2 (Bridge-Swap)
#
# Runs AFTER L5/L6 complete on the same RunPod (38.140.51.195:11483).
# L5/L6 used TinyLlama 1.1B; L7 uses Qwen 7B for language benchmarks.
# L2 uses Qwen 7B + fingerprint bridge adapters for bridge-swap eval.
#
# Pre-flight: SCP the fingerprint adapters and this script to RunPod.
# See SETUP section below.
#
# Usage: bash scripts/run_runpod_phase2.sh

set -euo pipefail

export PYTHONPATH="/workspace:/workspace/scripts"
export WANDB_PROJECT="rhombic-phase2-evals"
export WANDB_ENTITY="promptcrafted"

QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_BASE="/workspace/results/phase2-evals"
FINGERPRINT_DIR="/workspace/results/fingerprints"

mkdir -p "$OUTPUT_BASE"

echo "======================================================================"
echo "RunPod A100: Phase 2 Evals (L7 + L2)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Model: $QWEN_MODEL"
echo "  Output: $OUTPUT_BASE"
echo "======================================================================"

# -----------------------------------------------------------------------
# STEP 0: Install lm_eval if not present
# -----------------------------------------------------------------------
if ! python -c "import lm_eval" 2>/dev/null; then
    echo ""
    echo "Installing lm-eval..."
    pip install lm-eval 2>&1 | tail -5
    echo "lm-eval installed."
fi

# -----------------------------------------------------------------------
# L7: Language Benchmarks (base model, no adapter)
# -----------------------------------------------------------------------
echo ""
echo "----------------------------------------------------------------------"
echo "  L7a: Language Benchmarks — Base Qwen 7B (no adapter)"
echo "----------------------------------------------------------------------"

L7_BASE_OUTPUT="${OUTPUT_BASE}/L7-base-qwen7b.json"
mkdir -p "$OUTPUT_BASE"

python /workspace/scripts/eval_language_benchmarks.py \
    --model "$QWEN_MODEL" \
    --tasks mmlu arc_challenge hellaswag \
    --num-fewshot 5 \
    --batch-size auto \
    --output "$L7_BASE_OUTPUT" \
    2>&1 | tee "${OUTPUT_BASE}/L7-base.log"

echo "  L7a COMPLETE. Results: $L7_BASE_OUTPUT"

# -----------------------------------------------------------------------
# L7b: Language Benchmarks — with corpus-coupled adapter from L5
#   (only if L5 produced bridge weights)
# -----------------------------------------------------------------------
L5_OUTPUT="/workspace/results/corpus-baselines/C-003-corpus-coupled-default"

if [ -d "$L5_OUTPUT" ] && ls "$L5_OUTPUT"/bridge_final_*.npy 1>/dev/null 2>&1; then
    echo ""
    echo "----------------------------------------------------------------------"
    echo "  L7b: Language Benchmarks — Qwen 7B + L5 corpus-coupled adapter"
    echo "----------------------------------------------------------------------"

    L7_ADAPTER_OUTPUT="${OUTPUT_BASE}/L7-corpus-coupled-qwen7b.json"

    python /workspace/scripts/eval_language_benchmarks.py \
        --model "$QWEN_MODEL" \
        --adapter-dir "$L5_OUTPUT" \
        --tasks mmlu arc_challenge hellaswag \
        --num-fewshot 5 \
        --batch-size auto \
        --output "$L7_ADAPTER_OUTPUT" \
        2>&1 | tee "${OUTPUT_BASE}/L7-corpus-coupled.log"

    echo "  L7b COMPLETE. Results: $L7_ADAPTER_OUTPUT"
else
    echo ""
    echo "  SKIP L7b: No L5 bridge weights found at $L5_OUTPUT"
    echo "  (L5 may have used TinyLlama — adapters won't be compatible with Qwen 7B)"
fi

# -----------------------------------------------------------------------
# L2: Bridge-Swap Eval
#   Requires fingerprint adapters (code/, math/, alpaca/) at FINGERPRINT_DIR
# -----------------------------------------------------------------------
echo ""
echo "----------------------------------------------------------------------"
echo "  L2: Bridge-Swap Generative Eval"
echo "----------------------------------------------------------------------"

if [ -d "$FINGERPRINT_DIR" ]; then
    TASK_COUNT=$(find "$FINGERPRINT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "  Fingerprint dir: $FINGERPRINT_DIR ($TASK_COUNT task dirs found)"

    if [ "$TASK_COUNT" -ge 2 ]; then
        L2_OUTPUT="${OUTPUT_BASE}/L2-bridge-swap.json"

        python /workspace/scripts/eval_bridge_swap.py \
            --fingerprint-dir "$FINGERPRINT_DIR" \
            --model "$QWEN_MODEL" \
            --output "$L2_OUTPUT" \
            --max-batches 100 \
            --batch-size 2 \
            2>&1 | tee "${OUTPUT_BASE}/L2-bridge-swap.log"

        echo "  L2 COMPLETE. Results: $L2_OUTPUT"
    else
        echo "  SKIP L2: Need >= 2 task dirs in $FINGERPRINT_DIR, found $TASK_COUNT"
        echo "  Available: $(ls -d $FINGERPRINT_DIR/*/ 2>/dev/null | xargs -n1 basename)"
    fi
else
    echo "  SKIP L2: Fingerprint directory not found at $FINGERPRINT_DIR"
    echo ""
    echo "  To fix: SCP fingerprint adapters from local machine:"
    echo "    scp -P 11483 -r results/fingerprints/ root@38.140.51.195:/workspace/results/fingerprints/"
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "RunPod Phase 2 Evals COMPLETE."
echo "Results in: $OUTPUT_BASE"
echo ""
ls -la "$OUTPUT_BASE"/*.json 2>/dev/null || echo "  (no JSON results found)"
ls -la "$OUTPUT_BASE"/*.log 2>/dev/null || echo "  (no logs found)"
echo "======================================================================"
echo ""
echo "To retrieve results:"
echo "  scp -P 11483 -r root@38.140.51.195:$OUTPUT_BASE/ results/phase2-evals/"
