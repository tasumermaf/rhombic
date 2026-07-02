#!/bin/bash
# BM-002: Seeded Bridge Transfer — Does Alpaca-trained topology help on Code?
# Three configs on CodeAlpaca-20k, 5000 steps (smaller dataset):
#   Config C: Standard LoRA baseline (frozen identity bridge)
#   Config D: TeLoRA with identity init (Steersman discovers from scratch)
#   Config E: TeLoRA with Alpaca-seeded bridges (THE test)
set -e

echo "=============================================="
echo "BM-002: Seeded Bridge Transfer"
echo "$(date)"
echo "=============================================="

cd /workspace/rhombic

# Install dependencies
pip install -q transformers datasets peft accelerate sentencepiece protobuf
pip install -q 'lm-eval>=0.4.0'
pip install -e . 2>/dev/null || pip install -q numpy networkx

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"

echo ""
echo "=============================================="
echo "Config C: Standard LoRA on Code (control)"
echo "Started: $(date)"
echo "=============================================="

python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --bridge-mode identity \
  --no-bridge-training \
  --dataset code \
  --max-steps 5000 \
  --output results/BM-002-code-stdlora \
  --save-merged \
  --seed 42

echo ""
echo "Config C DONE: $(date)"
echo ""

echo "=============================================="
echo "Config D: TeLoRA on Code (identity init)"
echo "Started: $(date)"
echo "=============================================="

python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --dataset code \
  --max-steps 5000 \
  --output results/BM-002-code-telora-fresh \
  --save-merged \
  --feedback-interval 100 \
  --seed 42

echo ""
echo "Config D DONE: $(date)"
echo ""

echo "=============================================="
echo "Config E: TeLoRA on Code (Alpaca-seeded bridges)"
echo "  THE EXPERIMENT — does prior topology transfer?"
echo "Started: $(date)"
echo "=============================================="

python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --dataset code \
  --max-steps 5000 \
  --output results/BM-002-code-telora-seeded \
  --save-merged \
  --feedback-interval 100 \
  --seed-bridges results/BM-001-telora \
  --seed 42

echo ""
echo "Config E DONE: $(date)"
echo ""

echo "=============================================="
echo "ALL TRAINING DONE: $(date)"
echo "=============================================="

# Quick comparison
python -c "
import json, sys

configs = [
    ('C: Std LoRA', 'results/BM-002-code-stdlora/results.json'),
    ('D: TeLoRA fresh', 'results/BM-002-code-telora-fresh/results.json'),
    ('E: TeLoRA seeded', 'results/BM-002-code-telora-seeded/results.json'),
]

print('=' * 60)
print('BM-002 RESULTS')
print('=' * 60)
for label, path in configs:
    try:
        r = json.load(open(path))
        cp = r['checkpoints'][-1]
        co = cp.get('co_cross_ratio')
        co_s = f'{co:.0f}:1' if co else 'N/A'
        print(f'{label}:')
        print(f'  train_loss={cp[\"train_loss\"]:.4f}  val_loss={cp[\"val_loss\"]:.4f}')
        print(f'  co/cross={co_s}  fiedler={cp.get(\"fiedler_mean\",0):.5f}')
        print(f'  wall_time={cp[\"wall_time\"]/3600:.2f}h')
        print()
    except Exception as e:
        print(f'{label}: ERROR - {e}')
        print()

# Head-to-head at every 500 steps: D vs E
try:
    d = json.load(open(configs[1][1]))
    e = json.load(open(configs[2][1]))
    d_by = {cp['step']: cp for cp in d['checkpoints']}
    e_by = {cp['step']: cp for cp in e['checkpoints']}
    common = sorted(set(d_by) & set(e_by))
    print('HEAD-TO-HEAD: Fresh vs Seeded')
    print(f'  {\"Step\":>5}  {\"Fresh Val\":>10}  {\"Seeded Val\":>10}  {\"Delta\":>8}')
    for s in common:
        if s % 500 == 0 or s == common[-1]:
            dv = d_by[s]['val_loss']
            ev = e_by[s]['val_loss']
            print(f'  {s:>5}  {dv:>10.4f}  {ev:>10.4f}  {ev-dv:>+8.4f}')
except Exception as ex:
    print(f'Comparison failed: {ex}')
"
