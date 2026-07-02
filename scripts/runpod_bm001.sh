#!/bin/bash
# BM-001: TeLoRA vs Standard LoRA Benchmark — RunPod Auto-Chain
# Estimated: ~10-12h on A100 80GB (5-6h per config)
set -e

echo "=============================================="
echo "BM-001: TeLoRA Benchmark Validation"
echo "$(date)"
echo "=============================================="

# --- Setup ---
cd /workspace

# Clone or update repo
if [ -d "rhombic" ]; then
    echo "Repo exists, pulling latest..."
    cd rhombic && git pull && cd ..
else
    echo "ERROR: repo not synced. Run: rsync -avz from local machine"
    exit 1
fi

cd rhombic

# Install dependencies
pip install -q transformers datasets peft accelerate sentencepiece protobuf
pip install -q lm-eval>=0.4.0
pip install -e . 2>/dev/null || pip install -q numpy networkx

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"

echo ""
echo "=============================================="
echo "Config A: Standard LoRA (bridge frozen)"
echo "Started: $(date)"
echo "=============================================="

python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --bridge-mode identity \
  --no-bridge-training \
  --max-steps 10000 \
  --output results/BM-001-standard-lora \
  --save-merged \
  --seed 42

echo ""
echo "Config A DONE: $(date)"
echo ""

echo "=============================================="
echo "Config B: TeLoRA with Steersman"
echo "Started: $(date)"
echo "=============================================="

python scripts/train_cybernetic.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-channels 6 \
  --max-steps 10000 \
  --output results/BM-001-telora \
  --save-merged \
  --feedback-interval 100 \
  --seed 42

echo ""
echo "Config B DONE: $(date)"
echo ""

echo "=============================================="
echo "Benchmark Evaluation"
echo "Started: $(date)"
echo "=============================================="

mkdir -p results/BM-001

# Base model
echo "Evaluating base model..."
python scripts/eval_language_benchmarks.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --tasks mmlu arc_challenge hellaswag winogrande \
  --num-fewshot 5 \
  --output results/BM-001/base.json

# Standard LoRA
echo "Evaluating standard LoRA..."
python scripts/eval_language_benchmarks.py \
  --model results/BM-001-standard-lora/merged_model \
  --tasks mmlu arc_challenge hellaswag winogrande \
  --num-fewshot 5 \
  --output results/BM-001/standard-lora.json

# TeLoRA
echo "Evaluating TeLoRA..."
python scripts/eval_language_benchmarks.py \
  --model results/BM-001-telora/merged_model \
  --tasks mmlu arc_challenge hellaswag winogrande \
  --num-fewshot 5 \
  --output results/BM-001/telora.json

echo ""
echo "=============================================="
echo "ALL DONE: $(date)"
echo "=============================================="
echo ""
echo "Results at: results/BM-001/"
ls -la results/BM-001/
