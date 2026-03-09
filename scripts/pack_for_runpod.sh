#!/bin/bash
# Create minimal tarball for RunPod deployment
# Run from the Falco project root: bash rhombic/scripts/pack_for_runpod.sh
#
# Creates: rhombic_exp25.tar.gz (~50KB code + no data — data generated on pod)

set -e
cd "$(dirname "$0")/../.."

echo "Packing RhombiLoRA Exp 2.5 for RunPod..."

# Create staging area
STAGE=$(mktemp -d)
mkdir -p "$STAGE/rhombic/rhombic/nn"
mkdir -p "$STAGE/rhombic/scripts"
mkdir -p "$STAGE/rhombic/data/geometric"
mkdir -p "$STAGE/rhombic/results/exp2_5"

# Core library
cp rhombic/rhombic/__init__.py "$STAGE/rhombic/rhombic/"
cp rhombic/rhombic/nn/__init__.py "$STAGE/rhombic/rhombic/nn/"
cp rhombic/rhombic/nn/rhombi_lora.py "$STAGE/rhombic/rhombic/nn/"
cp rhombic/rhombic/nn/topology.py "$STAGE/rhombic/rhombic/nn/"

# Training scripts
cp rhombic/scripts/train_exp2_scale.py "$STAGE/rhombic/scripts/"
cp rhombic/scripts/train_exp2_5.py "$STAGE/rhombic/scripts/"
cp rhombic/scripts/generate_geometric_dataset.py "$STAGE/rhombic/scripts/"
cp rhombic/scripts/compare_exp2_exp25.py "$STAGE/rhombic/scripts/"

# pyproject.toml for pip install -e .
cp rhombic/pyproject.toml "$STAGE/rhombic/"

# Pack
cd "$STAGE"
tar czf rhombic_exp25.tar.gz rhombic/
mv rhombic_exp25.tar.gz "$(dirname "$0")/../.."/ 2>/dev/null || mv "$STAGE/rhombic_exp25.tar.gz" /c/Falco/

echo "Created: rhombic_exp25.tar.gz"
echo "Size: $(ls -lh /c/Falco/rhombic_exp25.tar.gz 2>/dev/null | awk '{print $5}' || echo 'check root')"
echo ""
echo "Upload to RunPod /workspace/, then run:"
echo "  bash /workspace/runpod_setup.sh"

# Cleanup
rm -rf "$STAGE"
