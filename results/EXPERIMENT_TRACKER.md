# Rhombic Experiment Tracker
> Last updated: 2026-03-15 ~12:00 local

## Active Experiments

| ID | Machine | n | Topology | Status | Step | Fiedler | Co/Cross | Paper | PID |
|----|---------|---|----------|--------|------|---------|----------|-------|-----|
| T-001r2 | Local (RTX 6000 Ada) | 8 | tesseract contrastive | TRAINING | 1100/10K | 0.000270 | 5,037:1 | **4** | 14056 |
| H-ch12 | Hermes (RTX 4090) | 12 | spectral-only | TRAINING | 3200/10K | 0.0843 | N/A | **4** | 522079 |

## Queued Experiments (Auto-Chain)

| ID | Machine | n | Topology | Trigger | Paper | Notes |
|----|---------|---|----------|---------|-------|-------|
| Baseline lm-eval | Local | — | — | After T-001r2 | **3/4** | TinyLlama base, 6 tasks |
| Adapted lm-eval | Local | — | — | After baseline | **3/4** | T-001r2 merged model, 6 tasks |
| WL-001 | Hermes | 6 | wrong-labels | After H-ch12 | **4** | Random pairs — decisive test |
| R-001 | Hermes | 8 | resonance | After WL-001 | **4** | Prime threading topology |
| E-001 | Hermes | 8 | emanation | After R-001 | **4** | Master bridge + per-layer offsets |

## Completed Experiments

### Paper 3 — The Learnable Bridge (ALL DATA IN HAND)

| ID | Model | n | Init | Steersman | Steps | Co/Cross | BD? | Paper | Data Location |
|----|-------|---|------|-----------|-------|----------|-----|-------|---------------|
| exp3 | Qwen 7B | 6 | identity | Yes | 12.9K | 18,248:1 | **YES** | **3** | results/exp3/ |
| exp3_tiny | TinyLlama | 6 | identity | Yes | 10K | 37,929:1 | **YES** | **3** | results/exp3_tiny/ |
| C-001 | TinyLlama | 6 | identity | Yes | 4K | 10,118:1 | **YES** | **3** | results/C-001/ |
| C-002 | TinyLlama | 6 | geometric | Yes | 10K | 71,337:1 | **YES** | **3** | results/C-002/ |
| C-003 | TinyLlama | 6 | corpus | Yes | 10K | 64,168:1 | **YES** | **3** | results/C-003/ |
| exp1a-e | Qwen 1.5B | 6 | various | No | 2K | ~1:1 | No | **3** | results/exp1/ |
| exp2 | Qwen 7B | 6 | FCC | No | 10K | ~1:1 | No | **3** | results/exp2/ |
| exp2.5 | Qwen 7B | 6 | geometric | No | 3K | ~1:1 | No | **3** | results/exp2_5/ |
| exp2.6 | Qwen 7B | 6 | warmup | Partial | 6.5K | 2660→16:1 | — | **3** | results/exp2_5/ |
| exp2.7 | Qwen 7B | 6 | identity | No (high LR) | 3K | ~1:1 | — | **3** | results/exp2_5/ |
| Holly | Wan 2.1 14B | 6 | identity | No | 10 ep | 1.07:1 | No | **3** | holly/ |

### Paper 4 — Channel Ablation Series (spectral attractor + mechanism proof)

| ID | Machine | n | Topology | Steps | Fiedler (final) | Co/Cross (final) | BD? | Paper | Data Location |
|----|---------|---|----------|-------|-----------------|------------------|-----|-------|---------------|
| H-ch3 | Hermes | 3 | spectral | 10K | 0.0951 | N/A | No | **3+4** | channel-ablation/H-ch3/ |
| H-ch4 | Hermes | 4 | spectral | 10K | 0.0918 | N/A | No | **3+4** | channel-ablation/H-ch4/ |
| H-ch6 | Hermes | 6 | RD contrastive | 10K | 0.00009 | 70,404:1 | **YES** | **3+4** | channel-ablation/H-ch6/ |
| H-ch8 | Hermes | 8 | spectral | 10K | 0.0944 | N/A | No | **3+4** | channel-ablation/H-ch8-results-hermes.json |
| T-001r1 | Local | 8 | tesseract contrastive | 2,700* | 0.00070 | 5,395:1 | **YES** (4+4) | **4** | T-001-full/ |

*T-001r1 was a partial run (terminated at step 2,700). T-001r2 is the full 10K replication.

### Paper 2 — Weighted Extensions

| ID | Model | n | Steps | Result | Paper | Data Location |
|----|-------|---|-------|--------|-------|---------------|
| TinyLlama baseline | 1.1B | 8 | 10K | Axis 22,477:1, Fiedler ~0.10 | **2** | results/exp2/ |
| Qwen2.5 baseline | 7B | 8 | 10K | Axis 47,145:1, Fiedler ~0.10 | **2** | results/exp2/ |

### Retracted

| ID | Reason | Date |
|----|--------|------|
| L-026 | Corpus weights misapplied (diagonal not off-diagonal) | 2026-03-13 |
| Holly Battery | Dataset provenance unclear, L-026 contamination | 2026-03-13 |

## Key Findings

1. **Spectral attractor CONFIRMED**: Fiedler → 0.0918–0.0951 across n=3,4,8 (3.5% band). H-ch12 at 0.0843 (step 3200), converging. [Paper 4]
2. **Contrastive topology IS the structure signal**: Only runs with co-axial pairs develop BD. [Paper 3]
3. **4D programming works**: T-001r2 at step 1100, co/cross 5,037:1, clean 4+4 eigenvalue split. [Paper 4]
4. **T-001r2 reproduces r1**: r=1.0000 at 6 matching steps, max deviation 3.5%. [Paper 4]
5. **Init is cosmetic**: 3 strategies converge to 64K–71K:1 band, 0.12% val loss delta. [Paper 3]
6. **Scale invariance**: BD at 1.1B + 7B; null at 14B without Steersman. [Paper 3]
7. **Task performance orthogonal**: 0.17% max val loss delta across n=3,4,6. [Paper 3]

## Paper → Experiment Map

| Paper | Experiments | Status |
|-------|-----------|--------|
| **2** | 7 weighted extension experiments | ALL COMPLETE |
| **3** | exp3, exp3_tiny, C-001/2/3, H-ch3/4/6/8, Holly, exp1-2.7 | ALL COMPLETE |
| **4** | T-001r1/r2, H-ch12, WL-001, R-001, E-001, H-ch3/4/8 (shared) | 3 complete, 2 running, 3 queued |
| **5** | D-001, D-002, bridge-swap, transit, Hum | NOT STARTED |

## Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| Local auto-chain | ARMED | Python watcher, polls results.json every 5 min |
| Hermes auto-chain | ARMED (PID 524141) | After H-ch12 → WL-001 → R-001 → E-001 |
| Sleep/hibernate | DISABLED (local) | Power management off for long training |
| RunPod fleet | 6 PODS STOPPED | Dataset blocker — Holly clips not on pods |
| merge_and_save() | VERIFIED | diff = 0.000001, correct delta merge |
| adapter_state.pt | ENABLED | Saves with every run |

## Figures

| Figure | File | Content | Paper |
|--------|------|---------|-------|
| Full ablation (4-panel) | channel-ablation/fig_full_ablation.png | Fiedler, eigenvalues, co/cross, attractor | 3+4 |
| T-001 reproducibility | channel-ablation/fig_t001_reproducibility.png | r1 vs r2, 3-panel comparison | 4 |
| All cybernetic temporal | fig_all_cybernetic_temporal.png | 5 cybernetic runs overlaid | 3 |
| Coupling dynamics | fig_temporal_emergence.png | Exp/linear fits | 3 |
| Init convergence | fig_init_convergence_comprehensive.png | 3-way convergence | 3 |
| Corpus dismantling | fig_corpus_dismantling_full.png | Per-pair trajectory | 3 |
| Per-layer coupling | fig_per_layer_coupling.png | Cross-scale comparison | 3 |
| Bridge heatmap | fig_heatmap_comparison.png | BD vs non-BD | 3 |
| Init growth analysis | fig_init_growth_analysis.png | Power-law fits | 3 |
| Per-module coupling | fig_per_module_coupling_c002.png | q/k/v/o_proj | 3 |
