# Experiment Queue — Updated March 15, 2026 (Session 2)

## Priority 1: Tesseract Contrastive (THE 4D experiment)

**Goal:** Test whether the bridge can discover 4-dimensional coordinate geometry.

**Design:** n=8, TinyLlama 1.1B, identity init, Steersman with tesseract-encoded
contrastive loss. The tesseract's 8 cubic cells pair into 4 opposite pairs along
4 coordinate axes: (0,1)=±w, (2,3)=±x, (4,5)=±y, (6,7)=±z.

**Code change:** COMPLETE. `_compute_pair_indices(8)` returns tesseract co-axial
pairs. `contrastive_bridge_loss` now works for any n where pairs are defined.

**Prediction A (mechanism is general):** 4 independent 2×2 blocks. Effective
dimensionality = 4. Bridge Fiedler collapses. 4-fold eigenvalue degeneracy.

**Prediction B (RD is special):** Structure collapses to 3 axes despite 4-axis
signal. Would indicate 3D is an intrinsic attractor of the parameter space.

**Machine:** Local GPU (RTX 6000 Ada, 48 GB). ~12 hours.
**Status:** RUNNING — `results/tesseract-contrastive-full/`, step 0+ (just launched)

**First-run data (100 steps before crash):** co_cross=13.56, Fiedler=0.007,
8 distinct eigenvalues (all multiplicity 1). Promising — co-axial preference
emerging immediately. Need 10K steps for full convergence.

## Priority 2: Language Benchmarks

**Goal:** Translate val loss into standard benchmark scores.

**Design:** Qwen 7B + exp3 adapter, lm_eval on MMLU/ARC/HellaSwag.
Compare: base Qwen, Qwen+standard LoRA, Qwen+RhombiLoRA.

**Machine:** Local GPU. ~3 hours.
**Dependencies:** eval_language_benchmarks.py (IMPLEMENTING — agent working)
**Status:** CODE IN PROGRESS

## Priority 3: Bridge-Swap Evaluation

**Goal:** Can you swap bridges between task-specific adapters and get meaningful
behavior changes? If yes → deployment killer feature (one adapter, N behaviors).

**Design:** Load Qwen 7B + task adapters from results/fingerprints/. For each pair:
keep lora_A/B, swap bridge. Measure perplexity on held-out validation.

**Machine:** Local GPU. ~4 hours.
**Dependencies:** eval_bridge_swap.py (IMPLEMENTING — agent working)
**Status:** CODE IN PROGRESS

## Priority 4: Multi-Seed Holly Validation

**Goal:** Statistical confidence on the 3.8% Holly improvement.

**Design:** 3 seeds RhombiLoRA (43, 44, 45) + 2 seeds standard LoRA (43, 44).
Wan 2.1 14B, Prodigy optimizer, 10 epochs.

**Machine:** RunPod A100 80GB. ~60 hours, ~$60.
**Status:** BLOCKED — RunPod API key expired/unauthorized. Need new key.

## Priority 5: Corpus-Seeded Dynamic Bridge

**Goal:** Test corpus values as persistent architectural influence rather than
overwritable initialization.

**Design:** Modify rhombi_lora.py to add optional gate_proj: nn.Linear(hidden_size, n*n).
Gate output is sigmoid-scaled and element-wise multiplied with static bridge.
Seed gate bias with corpus arithmetic: hexagram_coupling × thread_density × corpus_value.
Temperature annealing: T from 5.0 → 0.1 over training (soft → hard gating).

**Two experiments:**
- D-001: Dynamic bridge, random gate init (control)
- D-002: Dynamic bridge, corpus-seeded gate init

**Machine:** Local GPU. ~6 hours each.
**Code:** `--dynamic-bridge` flag IMPLEMENTED in train_cybernetic.py.
**Status:** READY — launch after T-001 completes

## Priority 6: Contrastive-With-Wrong-Labels

**Goal:** Is the bridge programmable with arbitrary topology, or does RD have
special status?

**Design:** n=6, but with WRONG co-planar labels — random partition of 6 channels
into 3 pairs instead of RD face pairs. If the bridge accepts arbitrary labels
→ Steersman is a general topology programmer. If it rejects → RD is special.

**Code:** `--contrastive-topology wrong-labels` IMPLEMENTED and wired.
Wrong pairs (seed=42): co=[(1,3), (2,4), (0,5)], 12 orthogonal pairs.

**Machine:** Local GPU. ~12 hours.
**Status:** READY — launch after T-001 or on Hermes after H5

## Priority 7: Circular Resonance Loss (NEW — Bateson/corpus-inspired)

**Goal:** Test whether prime-threading topology from the corpus can serve as
a contrastive signal. If the bridge responds to number-theoretic structure
the same way it responds to geometric structure → the corpus encodes
something the bridge can learn from.

**Design:** 6 channels, each mapped to a corpus prime: {0:11, 1:67, 2:23, 3:31, 4:29, 5:17}.
Three resonant pairs from prime-theoretic relationships:
  - Sophie Germain: 2(11)+1=23 → channels (0,2)
  - Consecutive primes: (29,31) → channels (3,4)
  - Same residue mod 6: 11≡5, 17≡5 → channels (0,5)

**Code:** `--contrastive-topology resonance` IMPLEMENTED. `circular_resonance_loss`
function in train_contrastive_bridge.py.

**Machine:** Local GPU. ~12 hours.
**Status:** READY — launch after higher priorities

## Priority 8: Emanation Architecture (NEW — Plotinus-inspired)

**Goal:** Test whether a hierarchical bridge (one master emanating into per-layer
variants) produces different structure than independent per-layer bridges.

**Design:** Single master bridge + small per-layer offsets. The master IS the
bridge; layers are perturbations. Coherence metric tracks how much layers
deviate from the master. Steersman monitors coherence and pulls fragmenting
layers back.

**Code:** `--emanation` flag IMPLEMENTED. `EmanationBridge` class in rhombi_lora.py.
Integrated with Steersman (coherence monitoring).

**Machine:** Local GPU. ~12 hours.
**Status:** READY — launch after higher priorities

## Currently Running

| Experiment | Machine | Status | ETA |
|-----------|---------|--------|-----|
| T-001 (n=8, tesseract contrastive) | Local GPU | Step 400/10000, co/cross 1921:1 | ~10 hours |
| H-ch8 (n=8, spectral-only) | Hermes | Step 5500/10000 | ~6 hours |
| H-ch12 (n=12, spectral-only) | Hermes | Queued after H-ch8 | ~12 hours after H-ch8 |

**Hermes auto-chain (watcher PID 519826):** After H-ch8 completes →
WL-001 (wrong-labels) → R-001 (resonance) → E-001 (emanation)

## Completed (This Sprint)

| Experiment | Result | Key Finding |
|-----------|--------|-------------|
| C-002 (geometric init) | 71,337:1, 100% BD | Init is cosmetic |
| C-003 (corpus-coupled) | 64,168:1, 100% BD, peak 82,854:1 | Corpus overwritten but highest peak |
| H-ch3 (n=3) | Val loss 0.4020, Fiedler 0.095 | Spectral-only → distributed, no BD |
| H-ch4 (n=4) | Val loss 0.4022, Fiedler 0.092 | Spectral-only → distributed, no BD |
| H-ch6 (n=6) | 70,404:1, 100% BD, val 0.4015 | ONLY n=6 develops BD (has RD contrastive) |

## Key Insight: Channel Ablation Summary

**Only n=6 develops block-diagonal structure.** All spectral-only runs (n=3,4,8)
converge to Fiedler ~0.09, deviation ~0.2 — generic distributed structure. Only
n=6, which activates RD contrastive loss, develops the extreme co/cross ratio
(70,404:1) and near-zero Fiedler characteristic of block-diagonal organization.

This means: **the contrastive topology IS the structure signal.** Without it, the
Steersman drives Fiedler connectivity but produces no directional preference.

T-001 (n=8 + tesseract contrastive) is the decisive test of whether ANY geometric
contrastive topology produces structure, or whether RD is special.

## Execution Order (Local GPU)

1. T-001 tesseract contrastive (~12h) ← RUNNING (step 400/10000)
2. Language benchmarks (P2) (~3h) ← after T-001
3. Bridge-swap eval (P3) (~4h)
4. Dynamic bridge D-001/D-002 (P5) (~12h)

Total local GPU time remaining: ~31 hours (~1.3 days) sequential after T-001.

## Execution Order (Hermes — auto-chain after H-ch8)

1. H-ch12 (n=12, spectral-only) (~12h)
2. WL-001 wrong-labels (P6) (~12h)
3. R-001 circular resonance (P7) (~12h)
4. E-001 emanation (P8) (~12h)

Total Hermes time remaining: ~48 hours (~2 days) sequential.

## Blockers

- **RunPod key RESOLVED (Mar 15).** New key validated. Config at ~/.runpod/config.toml.
  Use `runpod.api_key = 'key'` direct assignment (NOT env var).
  **Need to launch:** P4 multi-seed Holly pod (A100 80GB, ~$60).
- **Hermes auto-chain watcher** monitors H-ch8 PID specifically. H-ch12 runs via the
  channel ablation script. Watcher fires run_next_experiments.sh AFTER H-ch8 ends.
  Verify sequencing on restart.
