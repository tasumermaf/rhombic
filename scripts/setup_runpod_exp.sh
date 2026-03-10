#!/bin/bash
# RunPod setup script for RhombiLoRA experiments
# Usage: ssh -p PORT root@IP 'bash -s' < scripts/setup_runpod_exp.sh

set -e

echo "=== Installing dependencies ==="
pip install -q transformers accelerate datasets bitsandbytes scipy numpy torch

echo "=== Creating workspace ==="
mkdir -p /workspace/rhombic/rhombic/nn
mkdir -p /workspace/rhombic/scripts
mkdir -p /workspace/rhombic/results

echo "=== Setup complete ==="
echo "Ready for file upload and training launch."
