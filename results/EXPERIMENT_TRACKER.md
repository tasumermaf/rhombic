# Rhombic Experiment Tracker
> Last updated: 2026-03-17 ~14:30 UTC

## Active Experiments

| ID | Machine | n | Topology | Status | Step | Fiedler | Co/Cross | Paper | PID |
|----|---------|---|----------|--------|------|---------|----------|-------|-----|
| E-001 (hermes) | Hermes (RTX 4090) | 6 | emanation | TRAINING | 3100/10K | 0.0723 | 1.05:1 | **4** | tmux:e001 |
| Seed-44 (local) | Local (RTX 6000 Ada) | 6 | RD contrastive | TRAINING | early/10K | — | — | **3** | PID:52060 |

E-001 at step 2500: **Third regime confirmed** — Fiedler slowly rising toward spectral attractor (~0.07 vs target ~0.09), co/cross flat at 1.05:1 (no BD, no null). Neither spectral-only nor contrastive — intermediate emanation behavior. ETA complete: ~10:00 UTC Mar 18.

Seed-44 auto-launched after Seed-43 completion + lm-eval. Chain watcher PID 43436 fired successfully.

**Bug fix applied:** `torch.zeros(lora.rank, lora.rank, device=bridge.device)` in `merge_and_save()` — both Hermes and local copies fixed. This bug crashed WL-001 and T-001r2 post-training (all training data intact).

**RunPod ABANDONED** — enochi-lora pod container restarts kill tmux sessions. ~$40+ wasted.

## Queued Experiments (Auto-Chain)

| ID | Machine | n | Topology | Trigger | Paper | Notes |
|----|---------|---|----------|---------|-------|-------|
| O-001 (hermes) | Hermes | 4 | octahedral contrastive | After E-001 | **4** | tmux:o001watch, 2 co / 4 cross pairs |
| Seed-44 (local) | Local | 6 | RD contrastive (multi-seed) | LAUNCHED (Seed-43 done) | **3** | PID 52060 |

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
| Seed-43 | TinyLlama | 6 | identity | Yes | 10K | 73,309:1 | **YES** | **3** | results/Seed-43/ |

### Paper 4 — Channel Ablation Series (spectral attractor + mechanism proof)

| ID | Machine | n | Topology | Steps | Fiedler (final) | Co/Cross (final) | BD? | Paper | Data Location |
|----|---------|---|----------|-------|-----------------|------------------|-----|-------|---------------|
| H-ch3 | Hermes | 3 | spectral | 10K | 0.0951 | N/A | No | **3+4** | channel-ablation/H-ch3/ |
| H-ch4 | Hermes | 4 | spectral | 10K | 0.0918 | N/A | No | **3+4** | channel-ablation/H-ch4/ |
| H-ch6 | Hermes | 6 | RD contrastive | 10K | 0.00009 | 70,404:1 | **YES** | **3+4** | channel-ablation/H-ch6/ |
| H-ch8 | Hermes | 8 | spectral | 10K | 0.0944 | N/A | No | **3+4** | channel-ablation/H-ch8-results-hermes.json |
| H-ch12 | Hermes | 12 | spectral | 10K | 0.1019 | N/A | No | **3+4** | channel-ablation/H-ch12/ |
| T-001r1 | Local | 8 | tesseract contrastive | 2,700* | 0.00070 | 5,395:1 | **YES** (4+4) | **4** | T-001-full/ |
| T-001r2 | Local | 8 | tesseract contrastive | 10K | 0.000191 | 41,564:1 | **YES** (4+4) | **4** | T-001-full-r2/ |
| WL-001 | Hermes | 6 | wrong-labels | 10K | 0.0000126 | ~0:1 | **No** | **4** | channel-ablation/WL-001/ |
| R-001 | Hermes | 6 | resonance (no contrastive) | 10K | 0.000012 | 0.0:1 | **No** | **4** | channel-ablation/R-001/ |

*T-001r1 was a partial run (terminated at step 2,700). T-001r2 is the full 10K replication.

### Paper 2 — Weighted Extensions

| ID | Model | n | Steps | Result | Paper | Data Location |
|----|-------|---|-------|--------|-------|---------------|
| TinyLlama baseline | 1.1B | 8 | 10K | Axis 22,477:1, Fiedler ~0.10 | **2** | results/exp2/ |
| Qwen2.5 baseline | 7B | 8 | 10K | Axis 47,145:1, Fiedler ~0.10 | **2** | results/exp2/ |

### lm-eval Benchmarks (task performance verification)

| ID | Model | Tasks | Baseline | Adapted | Mean Delta | Paper | Data Location |
|----|-------|-------|----------|---------|------------|-------|---------------|
| lm-eval baseline | TinyLlama 1.1B | 6 (0-shot) | — | — | — | **3/4** | results/lm-eval/tinyllama-baseline/ |
| lm-eval adapted | T-001r1 merged (n=8) | 6 (0-shot) | 0.5786 avg | 0.5741 avg | **-0.75%** | **3/4** | results/lm-eval/t001-adapted/ |
| lm-eval Seed-43 | Seed-43 merged (n=6) | 6 (0-shot) | 0.5900 avg | 0.5863 avg | **-0.36%** | **3** | results/lm-eval/seed43-adapted/ |

### Retracted

| ID | Reason | Date |
|----|--------|------|
| L-026 | Corpus weights misapplied (diagonal not off-diagonal) | 2026-03-13 |
| Holly Battery | Dataset provenance unclear, L-026 contamination | 2026-03-13 |

## Key Findings

1. **Spectral attractor CONFIRMED**: Fiedler → 0.0918–0.1019 across n=3,4,8,12 (10.4% band). H-ch12 COMPLETE: final Fiedler 0.1019 at step 10K. [Paper 4]
2. **Contrastive topology IS the structure signal**: Only RD/tesseract pairs develop BD. WL-001 wrong-labels co/cross ~0 (10K). **R-001 resonance COMPLETE**: co/cross **0.0**, Fiedler **1.2e-05**, val loss **0.4008** at 10K — **definitive null**, spectral loss alone produces NO topology. [Paper 3+4]
3. **4D programming works**: T-001r2 COMPLETE at step 10K, co/cross **41,564:1**, Fiedler 0.000191. Clean 4+4 eigenvalue split. [Paper 4]
4. **T-001r2 reproduces r1**: r=1.0000 at 6 matching steps, max deviation 3.5%. r2 converged to 41,564:1 (r1 partial: 5,395:1 at step 2700). [Paper 4]
8. **Multi-seed replication**: **Seed-43 COMPLETE** — co/cross **73,309:1**, val loss **0.3997**, lm-eval **-0.36%** mean delta. Third independent seed in the 64K–73K band. Seed-44 running. [Paper 3]
9. **Emanation architecture**: E-001 at step 2500 shows intermediate behavior (Fiedler=0.070, co/cross=1.05) — Fiedler rising toward spectral attractor (~0.09) but not there yet. Neither BD nor null. Third regime. [Paper 4]
5. **Init is cosmetic**: 3 strategies converge to 64K–71K:1 band, 0.12% val loss delta. [Paper 3]
6. **Scale invariance**: BD at 1.1B + 7B; null at 14B without Steersman. [Paper 3]
7. **Task performance orthogonal**: 0.17% max val loss delta across n=3,4,6. lm-eval confirms: mean -0.75% across 6 benchmarks (mixed direction). [Paper 3+4]

## Paper → Experiment Map

| Paper | Experiments | Status |
|-------|-----------|--------|
| **2** | 7 weighted extension experiments | ALL COMPLETE |
| **3** | exp3, exp3_tiny, C-001/2/3, H-ch3/4/6/8, Holly, exp1-2.7 | ALL COMPLETE |
| **4** | T-001r1/r2, H-ch12, WL-001, R-001, E-001, O-001, Seed-43/44, H-ch3/4/8 (shared) | 6 complete, 2 running, 2 queued |
| **5** | D-001, D-002, bridge-swap, transit, Hum | NOT STARTED |

## Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| Local auto-chain | ARMED | Python watcher, polls results.json every 5 min |
| Hermes auto-chain | RUNNING (tmux:e001 + tmux:o001watch) | E-001 training, O-001 watches for completion |
| Sleep/hibernate | DISABLED (local) | Power management off for long training |
| RunPod A100 | UNSUITABLE | Container restarts kill training; shared with Flux-2 LoRA |
| merge_and_save() | BUG FIXED | `torch.eye(R, device=bridge.device)` — crashed WL-001 + T-001r2 post-training |
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
