#!/bin/bash
# Chain conditions C, D, E after B completes
# Usage: bash scripts/run_remaining_conditions.sh

cd /c/falco/rhombic

echo "[$(date)] Waiting for condition B (PID 11704) to finish..."
while tasklist /FI "PID eq 11704" /FO CSV /NH 2>/dev/null | grep -q "11704"; do
    sleep 30
done
echo "[$(date)] Condition B finished."

echo "[$(date)] Starting condition C (structure_only)..."
python -u scripts/train_loraxs_corpus.py --config C --output results/loraxs-corpus 2>&1 | tee results/loraxs-corpus/log_C.txt
echo "[$(date)] Condition C finished with exit code $?"

echo "[$(date)] Starting condition D (loraxs_baseline)..."
python -u scripts/train_loraxs_corpus.py --config D --output results/loraxs-corpus 2>&1 | tee results/loraxs-corpus/log_D.txt
echo "[$(date)] Condition D finished with exit code $?"

echo "[$(date)] Starting condition E (standard_lora)..."
python -u scripts/train_loraxs_corpus.py --config E --output results/loraxs-corpus 2>&1 | tee results/loraxs-corpus/log_E.txt
echo "[$(date)] Condition E finished with exit code $?"

echo "[$(date)] ALL CONDITIONS COMPLETE"
echo "Results:"
for d in results/loraxs-corpus/*/; do
    if [ -f "$d/results.json" ]; then
        name=$(basename "$d")
        val=$(python -c "import json; d=json.load(open('$d/results.json')); print(f'{d[\"final_val_loss\"]:.4f}')" 2>/dev/null)
        params=$(python -c "import json; d=json.load(open('$d/results.json')); print(d['trainable_params'])" 2>/dev/null)
        echo "  $name: val_loss=$val params=$params"
    fi
done
