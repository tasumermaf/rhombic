# G2-001: Bridge-Swap Evaluation — INCONCLUSIVE (Confound Identified)

## Run Date: 2026-03-18
## Wall time: 50 seconds

## Configuration
- Model: Qwen/Qwen2.5-7B-Instruct
- Tasks: code, math (from fingerprint experiments)
- Eval: wikitext-2 (300 samples, 50 batches)
- Adapter: rank=24, n_channels=6

## Result
All four configurations produce **exactly identical** perplexity: 9.1574.
Swap penalty: 0.0% across all combinations.

## Diagnosis: Experimental Confound

The fingerprint experiments saved `bridge_final_*.npy` files but did NOT
save `adapter_state.pt` (trained lora_A/B weights). The eval script
detected this correctly and used fresh random initialization for lora_A/B.

**Why this makes bridges invisible:** LoRA initializes lora_B to zero,
making ΔW = B·bridge·A = 0 regardless of bridge values. The bridge
multiplies zero. PPL is determined entirely by the frozen base model.

The bridges ARE structurally different (SVM classifier: 72.3% accuracy
from bridge matrices alone). But their effect on generation requires
trained lora_A/B that have co-adapted with the bridge during training.

## Required Fix: G2-001v2

Train task-specific adapters with full state saving:

```bash
# Add to training script: save adapter_state.pt containing
# all of lora_A, lora_B, and bridge for every injected module
torch.save({
    f"{name}.lora_A": lora.lora_A.data,
    f"{name}.lora_B": lora.lora_B.data,
    f"{name}.bridge": lora.bridge.data,
}, output_dir / "adapter_state.pt")
```

Then re-run bridge-swap with trained projections.

**Available trained adapters with full state:**
- `results/Seed-43/adapter_state.pt` — RD n=6, TinyLlama, seed 43
- `results/Seed-44/adapter_state.pt` — RD n=6, TinyLlama, seed 44
- `results/T-001-full/adapter_state.pt` — tesseract n=8, TinyLlama

These are same-task (alpaca) different-seed/topology. For proper bridge-swap,
need DIFFERENT task adapters with full state.

## What This DOES Tell Us

The experimental apparatus works correctly:
- Model loads, adapters inject, bridges swap, PPL computes
- The script correctly detected missing adapter state and warned
- 50-second runtime means the eval is cheap to re-run

## Priority: G2-001v2

1. Modify fingerprint training script to save full adapter state
2. Re-train code and math fingerprints (~2 hours each)
3. Re-run bridge-swap with trained lora_A/B
4. ALTERNATIVELY: train two TinyLlama adapters on different datasets
   (alpaca vs code-alpaca) with cybernetic training, save full state
