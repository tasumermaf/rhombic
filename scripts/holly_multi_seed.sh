#!/bin/bash
# Holly Multi-Seed Validation — RunPod A100 80GB
# Usage: ssh into RunPod, then: bash holly_multi_seed.sh
#
# Runs Holly Battery (Wan 2.1 14B) with 3 seeds for both
# standard LoRA and TeLoRA to get confidence intervals
# on the 3.8% loss improvement finding.
#
# Seeds: 42 (original), 123, 456
# Estimated: ~20h per pair (control + rhombi) = ~60h total

set -e

WORKSPACE="/workspace"
MODELS_DIR="$WORKSPACE/models"
DATASET_DIR="$WORKSPACE/dataset"
OUTPUT_DIR="$WORKSPACE/output"
RESULTS_DIR="$WORKSPACE/results"
SEEDS=(42 123 456)

echo "=== Holly Multi-Seed Validation ==="
echo "Seeds: ${SEEDS[*]}"
echo "Start time: $(date)"
echo ""

# ===== PHASE 1: Environment Setup =====
echo "--- Phase 1: Environment Setup ---"

# Install flimmer-trainer
if [ ! -d "$WORKSPACE/flimmer-trainer" ]; then
    cd $WORKSPACE
    git clone https://github.com/promptcrafted/flimmer-trainer.git
    cd flimmer-trainer
    pip install -e ".[training]" 2>&1 | tail -3
else
    echo "flimmer-trainer already installed"
fi

# Install rhombic (for bridge injection)
if ! python -c "import rhombic" 2>/dev/null; then
    cd $WORKSPACE
    if [ ! -d "rhombic" ]; then
        git clone https://github.com/tasumermaf/rhombic.git
    fi
    cd rhombic
    pip install -e . 2>&1 | tail -3
else
    echo "rhombic already installed"
fi

# Additional deps
pip install -q wandb safetensors 2>&1 | tail -1

echo "Environment ready."
echo ""

# ===== PHASE 2: Model Download =====
echo "--- Phase 2: Model Download ---"
mkdir -p $MODELS_DIR

# Wan 2.1 T2V 14B DiT
if [ ! -f "$MODELS_DIR/wan2.1_t2v_14B_fp16.safetensors" ]; then
    echo "Downloading Wan 2.1 T2V 14B..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Wan-AI/Wan2.1-T2V-14B', 'diffusion_pytorch_model.safetensors',
                local_dir='$MODELS_DIR', local_dir_use_symlinks=False)
import os
os.rename('$MODELS_DIR/diffusion_pytorch_model.safetensors',
          '$MODELS_DIR/wan2.1_t2v_14B_fp16.safetensors')
"
    echo "DiT downloaded."
else
    echo "DiT already present."
fi

# VAE
if [ ! -f "$MODELS_DIR/wan_2.1_vae.safetensors" ]; then
    echo "Downloading VAE..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Wan-AI/Wan2.1-T2V-14B', 'Wan2.1_VAE.pth',
                local_dir='$MODELS_DIR', local_dir_use_symlinks=False)
import os
os.rename('$MODELS_DIR/Wan2.1_VAE.pth', '$MODELS_DIR/wan_2.1_vae.safetensors')
"
    echo "VAE downloaded."
else
    echo "VAE already present."
fi

# T5-XXL
if [ ! -f "$MODELS_DIR/umt5_xxl_fp16.safetensors" ]; then
    echo "Downloading T5-XXL..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Wan-AI/Wan2.1-T2V-14B', 'models_t5_umt5-xxl-enc-bf16.pth',
                local_dir='$MODELS_DIR', local_dir_use_symlinks=False)
import os
os.rename('$MODELS_DIR/models_t5_umt5-xxl-enc-bf16.pth',
          '$MODELS_DIR/umt5_xxl_fp16.safetensors')
"
    echo "T5 downloaded."
else
    echo "T5 already present."
fi

echo "All models ready."
echo ""

# ===== PHASE 3: Dataset Setup =====
echo "--- Phase 3: Dataset Setup ---"
mkdir -p $DATASET_DIR

# Holly mini dataset (from HF or local)
if [ ! -f "$DATASET_DIR/flimmer_data.yaml" ]; then
    echo "Setting up Holly mini dataset..."
    # Create minimal dataset config
    cat > "$DATASET_DIR/flimmer_data.yaml" << 'YAML'
# Holly mini — Audrey Hepburn character study
datasets:
  - name: holly
    path: /workspace/dataset/holly
    resolution: [512, 320]
    num_frames: 81
    caption_column: text
YAML
    echo "Dataset config created. Upload holly clips to $DATASET_DIR/holly/"
    echo "WARNING: If holly clips are not present, training will fail."
else
    echo "Dataset config present."
fi

echo ""

# ===== PHASE 4: Multi-Seed Training Loop =====
echo "--- Phase 4: Multi-Seed Training ---"
mkdir -p $RESULTS_DIR

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  SEED $SEED — Starting $(date)"
    echo "=========================================="

    # --- Control (Standard LoRA) ---
    CONTROL_DIR="$OUTPUT_DIR/holly-control-seed$SEED"
    if [ -f "$CONTROL_DIR/holly_standard_r24_full_noise_epoch050.safetensors" ]; then
        echo "  Control seed $SEED already complete, skipping."
    else
        echo "  Running CONTROL (standard LoRA) seed=$SEED ..."
        mkdir -p $CONTROL_DIR

        cat > "/tmp/control_seed${SEED}.yaml" << YAML
model:
  variant: 2.1_t2v
  dit: $MODELS_DIR/wan2.1_t2v_14B_fp16.safetensors
  vae: $MODELS_DIR/wan_2.1_vae.safetensors
  t5: $MODELS_DIR/umt5_xxl_fp16.safetensors

data_config: $DATASET_DIR/flimmer_data.yaml

lora:
  rank: 24
  alpha: 24
  loraplus_lr_ratio: 1.0
  dropout: 0.0

optimizer:
  type: prodigy
  learning_rate: 1.0
  weight_decay: 0.01

scheduler:
  type: constant
  warmup_steps: 0

training:
  mixed_precision: bf16
  base_model_precision: bf16
  gradient_checkpointing: true
  blocks_to_swap: 28
  seed: $SEED
  unified_epochs: 50
  batch_size: 1
  gradient_accumulation_steps: 1
  caption_dropout_rate: 0.10
  timestep_sampling: shift

save:
  output_dir: $CONTROL_DIR
  name: holly_standard_r24
  save_every_n_epochs: 10
  save_last: true
  format: safetensors

logging:
  backends: [console, wandb]
  wandb_project: holly-multi-seed
  wandb_run_name: holly-control-seed$SEED
  wandb_tags: [control, standard-lora, wan21, prodigy, seed$SEED]
  log_every_n_steps: 5
YAML

        python -m flimmer.training.train "/tmp/control_seed${SEED}.yaml" 2>&1 | tee "$RESULTS_DIR/control_seed${SEED}.log"
        echo "  Control seed $SEED DONE at $(date)"
    fi

    # --- TeLoRA ---
    RHOMBI_DIR="$OUTPUT_DIR/holly-rhombi-seed$SEED"
    if [ -f "$RHOMBI_DIR/holly_rhombi_r24_full_noise_epoch050.safetensors" ]; then
        echo "  TeLoRA seed $SEED already complete, skipping."
    else
        echo "  Running RHOMBI (TeLoRA) seed=$SEED ..."
        mkdir -p $RHOMBI_DIR

        cat > "/tmp/rhombi_seed${SEED}.yaml" << YAML
model:
  variant: 2.1_t2v
  dit: $MODELS_DIR/wan2.1_t2v_14B_fp16.safetensors
  vae: $MODELS_DIR/wan_2.1_vae.safetensors
  t5: $MODELS_DIR/umt5_xxl_fp16.safetensors

data_config: $DATASET_DIR/flimmer_data.yaml

lora:
  rank: 24
  alpha: 24
  loraplus_lr_ratio: 1.0
  dropout: 0.0
  rhombi: true

optimizer:
  type: prodigy
  learning_rate: 1.0
  weight_decay: 0.01

scheduler:
  type: constant
  warmup_steps: 0

training:
  mixed_precision: bf16
  base_model_precision: bf16
  gradient_checkpointing: true
  blocks_to_swap: 28
  seed: $SEED
  unified_epochs: 50
  batch_size: 1
  gradient_accumulation_steps: 1
  caption_dropout_rate: 0.10
  timestep_sampling: shift

save:
  output_dir: $RHOMBI_DIR
  name: holly_rhombi_r24
  save_every_n_epochs: 10
  save_last: true
  format: safetensors

logging:
  backends: [console, wandb]
  wandb_project: holly-multi-seed
  wandb_run_name: holly-rhombi-seed$SEED
  wandb_tags: [experiment, rhombi-lora, wan21, prodigy, seed$SEED]
  log_every_n_steps: 5
YAML

        python -m flimmer.training.train --rhombi "/tmp/rhombi_seed${SEED}.yaml" 2>&1 | tee "$RESULTS_DIR/rhombi_seed${SEED}.log"
        echo "  TeLoRA seed $SEED DONE at $(date)"
    fi
done

echo ""
echo "=========================================="
echo "  ALL SEEDS COMPLETE at $(date)"
echo "=========================================="

# ===== PHASE 5: Results Summary =====
echo "--- Results Summary ---"
python << 'PYEND'
import json, glob, os

results = {}
for log in sorted(glob.glob("/workspace/results/*.log")):
    name = os.path.basename(log).replace('.log', '')
    # Extract final loss from last line containing 'loss'
    with open(log) as f:
        lines = f.readlines()
    losses = [l for l in lines if 'loss' in l.lower() and 'val' not in l.lower()]
    if losses:
        results[name] = losses[-1].strip()

print("\n=== Final Losses ===")
for k, v in sorted(results.items()):
    print(f"  {k}: {v}")

print("\nFull results at /workspace/results/")
print("Copy results: scp -P PORT root@IP:/workspace/results/* .")
PYEND
