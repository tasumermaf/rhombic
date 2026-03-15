#!/bin/bash
# Holly Pod Setup — runs on each RunPod A100 pod
# Usage: bash holly_pod_setup.sh <config_type> <seed>
#   config_type: "control" or "rhombi"
#   seed: integer (42, 123, 456)
#
# Example: bash holly_pod_setup.sh rhombi 42

set -e

CONFIG_TYPE=${1:?"Usage: holly_pod_setup.sh <control|rhombi> <seed>"}
SEED=${2:?"Usage: holly_pod_setup.sh <control|rhombi> <seed>"}

WORKSPACE="/workspace"
MODELS_DIR="$WORKSPACE/models"
VOLUME="/runpod-volume"

echo "=== Holly $CONFIG_TYPE seed=$SEED ==="
echo "Start: $(date)"

# ===== Install dependencies =====
echo "--- Installing packages ---"
pip install -q wandb safetensors huggingface_hub accelerate 2>&1 | tail -2

# Clone and install flimmer-trainer
if [ ! -d "$WORKSPACE/flimmer-trainer" ]; then
    cd $WORKSPACE
    git clone https://github.com/promptcrafted/flimmer-trainer.git 2>&1 | tail -2
    cd flimmer-trainer
    pip install -e ".[training]" 2>&1 | tail -3
fi

# Clone and install rhombic
if [ ! -d "$WORKSPACE/rhombic" ]; then
    cd $WORKSPACE
    git clone https://github.com/promptcrafted/rhombic.git 2>&1 | tail -2
    cd rhombic
    pip install -e . 2>&1 | tail -3
fi

echo "Packages installed."

# ===== Download models =====
echo "--- Downloading models ---"
mkdir -p $MODELS_DIR

python3 << DLEOF
from huggingface_hub import hf_hub_download, snapshot_download
import os

models_dir = "$MODELS_DIR"

# Wan 2.1 T2V 14B (all components)
print("Downloading Wan 2.1 T2V 14B components...")
snapshot_download(
    "Wan-AI/Wan2.1-T2V-14B",
    local_dir=os.path.join(models_dir, "Wan2.1-T2V-14B"),
    local_dir_use_symlinks=False,
)
print("Models downloaded.")
DLEOF

# Find the actual model files
DIT=$(find $MODELS_DIR -name "diffusion_pytorch_model*.safetensors" -o -name "wan2.1_t2v_14B*.safetensors" | head -1)
VAE=$(find $MODELS_DIR -name "*VAE*" -o -name "*vae*" | head -1)
T5=$(find $MODELS_DIR -name "*t5*" -o -name "*umt5*" | head -1)

echo "DiT: $DIT"
echo "VAE: $VAE"
echo "T5:  $T5"

# ===== Setup Holly dataset =====
echo "--- Setting up dataset ---"
DATASET_DIR="$WORKSPACE/dataset"
mkdir -p $DATASET_DIR/holly

# Download Holly mini from HF (if available) or create placeholder
cat > "$DATASET_DIR/flimmer_data.yaml" << YAML
datasets:
  - name: holly
    path: $DATASET_DIR/holly
    resolution: [512, 320]
    num_frames: 81
    caption_column: text
YAML

echo "NOTE: Holly clips must be in $DATASET_DIR/holly/"
echo "If not present, upload them now before training starts."

# ===== Create training config =====
echo "--- Creating $CONFIG_TYPE config (seed=$SEED) ---"

OUTPUT_DIR="$WORKSPACE/output/holly-${CONFIG_TYPE}-seed${SEED}"
mkdir -p $OUTPUT_DIR

RHOMBI_FLAG=""
RHOMBI_YAML=""
RUN_NAME="holly-${CONFIG_TYPE}-seed${SEED}"

if [ "$CONFIG_TYPE" = "rhombi" ]; then
    RHOMBI_FLAG="--rhombi"
    RHOMBI_YAML="  rhombi: true"
fi

cat > "$WORKSPACE/config_${CONFIG_TYPE}_s${SEED}.yaml" << YAML
model:
  variant: 2.1_t2v
  dit: $DIT
  vae: $VAE
  t5: $T5

data_config: $DATASET_DIR/flimmer_data.yaml

lora:
  rank: 24
  alpha: 24
  loraplus_lr_ratio: 1.0
  dropout: 0.0
$RHOMBI_YAML

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
  output_dir: $OUTPUT_DIR
  name: holly_${CONFIG_TYPE}_r24
  save_every_n_epochs: 10
  save_last: true
  format: safetensors

logging:
  backends: [console, wandb]
  wandb_project: holly-multi-seed
  wandb_run_name: $RUN_NAME
  wandb_tags: [$CONFIG_TYPE, wan21, prodigy, seed$SEED, multi-seed]
  log_every_n_steps: 5
YAML

echo "Config written to config_${CONFIG_TYPE}_s${SEED}.yaml"

# ===== Launch training =====
echo "--- Launching training ($CONFIG_TYPE seed=$SEED) ---"
echo "Start: $(date)"

cd $WORKSPACE
python -m flimmer.training.train $RHOMBI_FLAG "config_${CONFIG_TYPE}_s${SEED}.yaml" \
    2>&1 | tee "$WORKSPACE/training_${CONFIG_TYPE}_s${SEED}.log"

echo ""
echo "=== COMPLETE: $CONFIG_TYPE seed=$SEED ==="
echo "End: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Log: $WORKSPACE/training_${CONFIG_TYPE}_s${SEED}.log"

# Save results to volume for persistence
cp -r $OUTPUT_DIR "$VOLUME/" 2>/dev/null || true
cp "$WORKSPACE/training_${CONFIG_TYPE}_s${SEED}.log" "$VOLUME/" 2>/dev/null || true
echo "Results copied to $VOLUME/"
