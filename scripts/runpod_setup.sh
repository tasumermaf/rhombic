#!/bin/bash
# RunPod Setup for Experiment 2.5 — Quick Validation
#
# USAGE:
#   1. Create RunPod pod with A100 40GB, PyTorch 2.x template
#   2. Upload rhombic_exp25.tar.gz (created by pack_for_runpod.sh)
#   3. Paste this entire script into the terminal
#
# Expected runtime: ~45min (A100) for 3K steps FCC on geometric data
# Expected cost: ~$1.50 (A100 40GB spot @ ~$1.64/hr)

set -e

echo "=== RunPod Setup for Exp 2.5 ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'checking...')"
echo ""

# ── Step 1: Install dependencies ─────────────────────────────────────
echo "=== Installing dependencies ==="
pip install -q transformers datasets accelerate numpy scipy networkx

# ── Step 2: Unpack files ─────────────────────────────────────────────
echo "=== Unpacking experiment files ==="
cd /workspace
if [ -f rhombic_exp25.tar.gz ]; then
    tar xzf rhombic_exp25.tar.gz
    echo "Unpacked from tarball"
elif [ -d rhombic ]; then
    echo "Files already in place"
else
    echo "ERROR: No rhombic_exp25.tar.gz found in /workspace"
    echo "Upload it first, then re-run this script"
    exit 1
fi

cd /workspace/rhombic

# ── Step 3: Install rhombic library + verify imports ─────────────────
echo "=== Installing rhombic library ==="
pip install -q -e .
echo "=== Verifying imports ==="
python -c "
from rhombic.nn.rhombi_lora import RhombiLoRALinear
from rhombic.nn.topology import direction_pair_coupling, bridge_init
print('rhombic.nn OK')
import sys; sys.path.insert(0, 'scripts')
from train_exp2_scale import ExperimentConfig, CONFIGS, inject_lora
print('train_exp2_scale OK')
from train_exp2_5 import LocalJSONDataset, train_exp25
print('train_exp2_5 OK')
print('All imports verified.')
"

# ── Step 4: Generate geometric dataset ───────────────────────────────
echo ""
echo "=== Generating geometric dataset ==="
python scripts/generate_geometric_dataset.py \
    --output data/geometric \
    --components 1,3,4 \
    --n1 10000 \
    --n3 8000 \
    --n4 5000 \
    --coplanar-bias 0.6

echo ""
echo "Dataset files:"
ls -lh data/geometric/*.json

# ── Step 5: Run Exp 2.5 Quick Validation (FCC, 3K steps) ────────────
echo ""
echo "=== Running Exp 2.5: FCC on Geometric (3K steps) ==="
echo "Started: $(date)"

mkdir -p results/exp2_5

python scripts/train_exp2_5.py \
    --config 2 \
    --dataset-mode geometric \
    --data-dir data/geometric \
    --output results/exp2_5 \
    --max-steps 3000

echo ""
echo "=== Training complete: $(date) ==="
echo ""

# ── Step 6: Show results ─────────────────────────────────────────────
echo "=== Results ==="
if [ -f results/exp2_5/geometric_fcc_rhombilora/results.json ]; then
    python -c "
import json
with open('results/exp2_5/geometric_fcc_rhombilora/results.json') as f:
    data = json.load(f)
ckpts = data.get('checkpoints', [])
if ckpts:
    last = ckpts[-1]
    print(f\"Final step: {last['step']}\")
    print(f\"Val loss:   {last['val_loss']:.4f}\")
    print(f\"Fiedler:    {last['bridge_fiedler_mean']:.5f}\")
    ccp = last.get('coplanar_crossplanar')
    if ccp:
        print(f\"Co/Cross:   {ccp['ratio']:.4f}\")
    print(f\"Eff rank:   {last['grad_effective_rank_mean']:.2f}\")
    print()
    print('=== KEY METRIC: Co/Cross ratio ==='
          )
    if ccp and ccp['ratio'] > 1.10:
        print(f\"  {ccp['ratio']:.3f}x — SIGNAL DETECTED. Geometric data teaches direction.\")
    elif ccp:
        print(f\"  {ccp['ratio']:.3f}x — Below 1.10 threshold. May need dataset iteration.\")
else:
    print('No checkpoints found in results.')
"
else
    echo "Results file not found — check training output above for errors"
fi

echo ""
echo "=== Done. Download results/exp2_5/ for local analysis ==="
echo "For deep comparison: python scripts/compare_exp2_exp25.py --deep"
