#!/bin/bash
# Complete P0 analysis pipeline — run immediately when config 3 finishes.
# Generates visualization + extracts data for Paper 4.
set -e
cd "$(dirname "$0")/.."

echo "=== P0 Complete Analysis Pipeline ==="
echo "Time: $(date)"
echo ""

# 1. Check both results exist
if [ ! -f results/p0-proof/standard_lora_r24/results.json ]; then
    echo "ERROR: Config 1 (standard LoRA) results missing"
    exit 1
fi
if [ ! -f results/p0-proof/rhombi_learnable_r24/results.json ]; then
    echo "ERROR: Config 3 (TeLoRA) results missing"
    exit 1
fi

echo "Both configs present. Running analysis..."
echo ""

# 2. Generate visualization
echo "--- Generating P0 comparison figure ---"
python scripts/visualize_p0_comparison.py \
    --results-dir results/p0-proof \
    --output figures/p0_proof_comparison.png

echo ""

# 3. Extract Paper 4 metrics
echo "--- Extracting Paper 4 metrics ---"
python -c "
import json
import numpy as np

# Load both results
with open('results/p0-proof/standard_lora_r24/results.json') as f:
    lora = json.load(f)
with open('results/p0-proof/rhombi_learnable_r24/results.json') as f:
    telora = json.load(f)

lora_cps = lora['checkpoints']
telora_cps = telora['checkpoints']

# Final metrics
lora_loss = lora_cps[-1]['train_loss']
telora_loss = telora_cps[-1]['train_loss']
improvement = (lora_loss - telora_loss) / lora_loss * 100

lora_time = lora_cps[-1]['wall_time']
telora_time = telora_cps[-1]['wall_time']
overhead = (telora_time - lora_time) / lora_time * 100

print('='*60)
print('P0 PROOF — PAPER 4 METRICS')
print('='*60)
print()
print(f'Model: Qwen/Qwen2.5-1.5B-Instruct')
print(f'Dataset: yahma/alpaca-cleaned')
print(f'Rank: 24, Seed: 42')
print()
print(f'--- Loss ---')
print(f'Standard LoRA final loss: {lora_loss:.6f}')
print(f'TeLoRA final loss:        {telora_loss:.6f}')
print(f'Improvement:              {abs(improvement):.2f}% {\"better\" if improvement > 0 else \"worse\"}')
print()
print(f'--- Training Time ---')
print(f'Standard LoRA: {lora_time/60:.1f} min ({lora_time:.0f}s)')
print(f'TeLoRA:        {telora_time/60:.1f} min ({telora_time:.0f}s)')
print(f'Overhead:      {overhead:+.1f}%')
print()

# Bridge metrics (TeLoRA only)
first_key = list(telora_cps[0]['bridge_fiedler'].keys())[0]
print(f'--- Bridge Evolution (module: {first_key.split(\".\")[-2]}.{first_key.split(\".\")[-1]}) ---')
for cp in telora_cps:
    f = cp['bridge_fiedler'][first_key]
    d = cp['bridge_deviation'][first_key]
    B = np.array(cp['bridge_matrices'][first_key])
    n = B.shape[0]
    half = n // 2
    co_vals = np.abs(np.concatenate([
        B[:half, :half][np.triu_indices(half, 1)],
        B[half:, half:][np.triu_indices(half, 1)]
    ]))
    cross_vals = np.abs(B[:half, half:].flatten())
    co_mean = co_vals.mean() if len(co_vals) > 0 else 0
    cross_mean = cross_vals.mean() if len(cross_vals) > 0 else 1e-10
    cc = co_mean / max(cross_mean, 1e-10)
    print(f'  Step {cp[\"step\"]:5d}: Fiedler={f:.6f}, Dev={d:.4f}, Co/Cross={cc:.1f}:1')

# Final bridge matrix statistics (aggregate across all modules)
print()
print(f'--- Final Bridge Aggregate ---')
final = telora_cps[-1]
fiedlers = list(final['bridge_fiedler'].values())
devs = list(final['bridge_deviation'].values())
print(f'N modules: {len(fiedlers)}')
print(f'Fiedler: mean={np.mean(fiedlers):.6f}, std={np.std(fiedlers):.6f}')
print(f'Deviation: mean={np.mean(devs):.4f}, std={np.std(devs):.4f}')

# Compute aggregate co/cross from all bridges
all_cc = []
for name, mat in final['bridge_matrices'].items():
    B = np.array(mat)
    n = B.shape[0]
    half = n // 2
    co_vals = np.abs(np.concatenate([
        B[:half, :half][np.triu_indices(half, 1)],
        B[half:, half:][np.triu_indices(half, 1)]
    ]))
    cross_vals = np.abs(B[:half, half:].flatten())
    co_mean = co_vals.mean() if len(co_vals) > 0 else 0
    cross_mean = cross_vals.mean() if len(cross_vals) > 0 else 1e-10
    all_cc.append(co_mean / max(cross_mean, 1e-10))
print(f'Co/Cross ratio: median={np.median(all_cc):.1f}:1, mean={np.mean(all_cc):.1f}:1')
print()

# Parameters
print(f'--- Parameters ---')
print(f'Standard LoRA: {lora[\"trainable_params\"]:,} trainable / {lora[\"total_params\"]:,} total')
print(f'TeLoRA:        {telora[\"trainable_params\"]:,} trainable / {telora[\"total_params\"]:,} total')
bridge_extra = telora['trainable_params'] - lora['trainable_params']
print(f'Bridge overhead: {bridge_extra:,} params ({bridge_extra/lora[\"trainable_params\"]*100:.2f}%)')
print()
print('='*60)
"

echo ""
echo "=== Analysis complete ==="
echo "Figures: figures/p0_proof_comparison.png, .pdf"
echo "Time: $(date)"
