#!/bin/bash
# Local auto-chain: watches T-001r2 (PID 14056), then runs lm-eval benchmarks
# Launched: 2026-03-15
# Machine: Local RTX 6000 Ada 48GB

T001R2_PID=14056
RESULTS_DIR="/c/Falco/rhombic/results"
SCRIPTS_DIR="/c/Falco/rhombic/scripts"
LOG="/c/Falco/rhombic/results/local_auto_chain.log"

echo "[$(date)] Local auto-chain started. Watching T-001r2 PID=$T001R2_PID" | tee -a "$LOG"

# Wait for T-001r2 to finish
while kill -0 $T001R2_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date)] T-001r2 finished!" | tee -a "$LOG"

# Step 1: Baseline benchmarks (TinyLlama base model)
echo "[$(date)] Running baseline lm-eval..." | tee -a "$LOG"
python -m lm_eval \
    --model hf \
    --model_args pretrained=TinyLlama/TinyLlama-1.1B-Chat-v1.0,dtype=bfloat16 \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq \
    --batch_size auto \
    --output_path "$RESULTS_DIR/benchmarks/tinyllama_baseline" \
    --device cuda:0 2>&1 | tee -a "$LOG"
echo "[$(date)] Baseline benchmarks complete." | tee -a "$LOG"

# Step 2: Check if merged model exists
MERGED_DIR="$RESULTS_DIR/T-001-full-r2/merged_model"
if [ -d "$MERGED_DIR" ]; then
    echo "[$(date)] Running adapted model lm-eval on $MERGED_DIR..." | tee -a "$LOG"
    python -m lm_eval \
        --model hf \
        --model_args pretrained="$MERGED_DIR",dtype=bfloat16 \
        --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq \
        --batch_size auto \
        --output_path "$RESULTS_DIR/benchmarks/tinyllama_t001r2_adapted" \
        --device cuda:0 2>&1 | tee -a "$LOG"
    echo "[$(date)] Adapted model benchmarks complete." | tee -a "$LOG"
else
    echo "[$(date)] WARNING: No merged model at $MERGED_DIR — skipping adapted benchmarks." | tee -a "$LOG"
    echo "[$(date)] Run manually after merge: python -m lm_eval --model hf --model_args pretrained=$MERGED_DIR,dtype=bfloat16 --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq --batch_size auto --output_path $RESULTS_DIR/benchmarks/tinyllama_t001r2_adapted --device cuda:0" | tee -a "$LOG"
fi

echo "[$(date)] Local auto-chain complete!" | tee -a "$LOG"
