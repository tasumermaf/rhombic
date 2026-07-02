#!/bin/bash
# Wanlantis A/B Test — RunPod Setup
#
# USAGE:
#   1. Create RunPod pod: A100 80GB, PyTorch 2.x template
#   2. SSH in and paste this entire script
#
# This sets up both Standard LoRA and TeLoRA training configs
# matching the Replicate ground truth (sunday-film/wanlantis).
#
# Ground truth settings (from Replicate training xdae34etgxrm):
#   trainer: ostris/wan-lora-trainer (AI Toolkit)
#   rank: 42, steps: 6000, lr: 1e-4, optimizer: adamw8bit
#   resolution: 948, caption_dropout: 0.05
#   trigger_word: "atlantis style"
#   dataset: 58 Damanhur Atlantis image/caption pairs

set -e

WORKSPACE="/workspace"
MODELS_DIR="$WORKSPACE/models"
DATASET_DIR="$WORKSPACE/dataset"

echo "=== Wanlantis A/B Test Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"

# ── Step 1: Install dependencies ────────────────────────────────────
echo "=== Installing dependencies ==="
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
pip install -q transformers diffusers accelerate safetensors peft 2>&1 | tail -2
pip install -q bitsandbytes wandb huggingface_hub 2>&1 | tail -2

# ── Step 2: Clone and install AI Toolkit ────────────────────────────
echo "=== Installing AI Toolkit ==="
cd $WORKSPACE
if [ ! -d "ai-toolkit" ]; then
    git clone https://github.com/ostris/ai-toolkit.git 2>&1 | tail -2
    cd ai-toolkit
    git submodule update --init --recursive 2>&1 | tail -2
    pip install -e . 2>&1 | tail -3
else
    cd ai-toolkit
    git pull 2>&1 | tail -2
fi

# ── Step 3: Clone rhombic (for TeLoRA integration) ───────────────────
echo "=== Installing rhombic ==="
cd $WORKSPACE
if [ ! -d "rhombic" ]; then
    git clone https://github.com/tasumermaf/rhombic.git 2>&1 | tail -2
    cd rhombic
    pip install -e . 2>&1 | tail -3
else
    cd rhombic
    git pull 2>&1 | tail -2
fi

# Copy the AI Toolkit integration module to workspace root
cp $WORKSPACE/rhombic/scripts/aitoolkit_telora.py $WORKSPACE/aitoolkit_telora.py
echo "TeLoRA integration module: $WORKSPACE/aitoolkit_telora.py"

# ── Step 4: Download Wan 2.1 T2V 14B model ─────────────────────────
echo "=== Downloading Wan 2.1 T2V 14B ==="
mkdir -p $MODELS_DIR
python3 << 'DLEOF'
from huggingface_hub import snapshot_download
import os

models_dir = os.environ.get("MODELS_DIR", "/workspace/models")
print("Downloading Wan 2.1 T2V 14B (Diffusers)...")
snapshot_download(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    local_dir=os.path.join(models_dir, "Wan2.1-T2V-14B-Diffusers"),
    local_dir_use_symlinks=False,
)
print("Model downloaded.")
DLEOF

echo "Model path: $MODELS_DIR/Wan2.1-T2V-14B-Diffusers"

# ── Step 5: Download Atlantis dataset ───────────────────────────────
echo "=== Downloading Atlantis dataset ==="
mkdir -p $DATASET_DIR
cd $DATASET_DIR
if [ ! -d "dataset_claude_captions" ]; then
    curl -L -o dataset_claude_captions.zip \
        "https://replicate.delivery/pbxt/Mld5j7IFwEjNnNpGFg9mLY75VJriqTOAA5WQKwxczKAsMpOO/dataset_claude_captions.zip"
    unzip -q dataset_claude_captions.zip
    echo "Dataset extracted: $(ls dataset_claude_captions/*.jpg dataset_claude_captions/*.JPG 2>/dev/null | wc -l) images"
else
    echo "Dataset already present"
fi

# ── Step 6: Create Standard LoRA config (control) ──────────────────
# Matches Replicate ground truth: rank=42, steps=6000, lr=1e-4
echo "=== Creating training configs ==="
cd $WORKSPACE

cat > config_standard_lora.yaml << 'YAML'
---
job: extension
config:
  name: wanlantis_standard_lora
  process:
    - type: sd_trainer
      training_folder: /workspace/output/standard_lora
      device: cuda:0
      trigger_word: "atlantis style"
      network:
        type: lora
        linear: 42
        linear_alpha: 42
      save:
        dtype: float16
        save_every: 1000
        max_step_saves_to_keep: 3
      datasets:
        - folder_path: /workspace/dataset/dataset_claude_captions
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [948]
      train:
        batch_size: 1
        steps: 6000
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        timestep_type: sigmoid
        optimizer: adamw8bit
        lr: 1e-4
        optimizer_params:
          weight_decay: 1e-4
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: /workspace/models/Wan2.1-T2V-14B-Diffusers
        arch: wan21
        quantize: true
        quantize_te: true
      sample:
        sampler: flowmatch
        sample_every: 1000
        width: 832
        height: 480
        num_frames: 40
        fps: 15
        prompts:
          - "atlantis style, a majestic crystal city rising from the ocean depths, bioluminescent coral towers"
        seed: 42
        walk_seed: true
        guidance_scale: 5
        sample_steps: 30
meta:
  name: "[name]"
  version: "1.0"
YAML

echo "Standard LoRA config: config_standard_lora.yaml"

# ── Step 7: Create TeLoRA config ────────────────────────────────────
# Same settings as standard, but with telora network type + bridge
# rank=42, n_channels=6, channel_rank=7

cat > config_telora.yaml << 'YAML'
---
job: extension
config:
  name: wanlantis_telora
  process:
    - type: sd_trainer
      training_folder: /workspace/output/telora
      device: cuda:0
      trigger_word: "atlantis style"
      network:
        type: telora
        linear: 42
        linear_alpha: 42
        network_kwargs:
          n_channels: 6
          bridge_mode: identity
      save:
        dtype: float16
        save_every: 1000
        max_step_saves_to_keep: 3
      datasets:
        - folder_path: /workspace/dataset/dataset_claude_captions
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [948]
      train:
        batch_size: 1
        steps: 6000
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        timestep_type: sigmoid
        optimizer: adamw8bit
        lr: 1e-4
        optimizer_params:
          weight_decay: 1e-4
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: /workspace/models/Wan2.1-T2V-14B-Diffusers
        arch: wan21
        quantize: true
        quantize_te: true
      sample:
        sampler: flowmatch
        sample_every: 1000
        width: 832
        height: 480
        num_frames: 40
        fps: 15
        prompts:
          - "atlantis style, a majestic crystal city rising from the ocean depths, bioluminescent coral towers"
        seed: 42
        walk_seed: true
        guidance_scale: 5
        sample_steps: 30
meta:
  name: "[name]"
  version: "1.0"
YAML

echo "TeLoRA config: config_telora.yaml"

# ── Step 8: Instructions ────────────────────────────────────────────
echo ""
echo "=== SETUP COMPLETE ==="
echo ""
echo "To run Standard LoRA (control):"
echo "  cd /workspace/ai-toolkit"
echo "  python run.py /workspace/config_standard_lora.yaml"
echo ""
echo "To run TeLoRA (experiment):"
echo "  cd /workspace"
echo "  PYTHONPATH=/workspace:\$PYTHONPATH python aitoolkit_telora.py config_telora.yaml"
echo ""
echo "Run standard first (~90 min on A100), verify output, then run TeLoRA."
echo "Results in: /workspace/output/{standard_lora,telora}/"
