# TeLoRA Experiment Tracker — Master Document

> **Purpose:** Single source of truth for all experiments, their status,
> success criteria, decision gates, and dependencies.
> **Last updated:** 2026-03-31 (FC-001 COMPLETE 67,501:1, FO-001 COMPLETE 262,920:1, LORAXS-CORPUS COMPLETE negative result)
> **Notion mirror:** https://www.notion.so/31d6930d2e1181bab622d6b3a9720b24

---

## Experiment Summary Table

| Exp | Model | Steps | Status | Key Finding |
|-----|-------|-------|--------|-------------|
| 1 | Qwen 1.5B | 2K | COMPLETE | FCC > cubic 4.6× Fiedler |
| 2 | Qwen 7B | 10K | COMPLETE | Baseline. Fiedler 0.0401, co/cross 1.019 |
| 2.5 | Qwen 7B | 3K | COMPLETE — NULL | Geometric data doesn't drive directionality |
| 2.7 | Qwen 7B | 2K | COMPLETE — NULL | Higher bridge LR doesn't help |
| **3.0** | **Qwen 7B** | **12,900** | **COMPLETE — BREAKTHROUGH** | **22,477:1 axis alignment** |
| 3.0-TL | TinyLlama 1.1B | 10K | COMPLETE | 47,145:1 alignment (scale-invariant) |
| Phase 0 | Qwen 7B | — | COMPLETE | Q-proj 40% more coupling (p=0.0008) |
| Phase 1A | Qwen 7B | — | COMPLETE | 72.3% LOO SVM (all modules); Q-proj-only claim retracted pending re-run |
| Phase 2A | Qwen 7B | — | COMPLETE | Eigenspectrum cos > 0.999 under merging |
| **Phase 3A** | **Qwen 7B** | **10K** | **COMPLETE** | **r=0.888 deviation~gap, phase transition step 400** |
| **Holly** | **Wan 2.1 14B** | **full** | **COMPLETE (WandB)** | **3.8% better loss, 9.15 GB less VRAM** |
| H-ch3 | TinyLlama 1.1B | 10K | COMPLETE | Fiedler 0.095, no BD (spectral-only) |
| H-ch4 | TinyLlama 1.1B | 10K | COMPLETE | Fiedler 0.092, no BD (spectral-only) |
| H-ch6 | TinyLlama 1.1B | 10K | COMPLETE | **70,404:1 co/cross, full BD** |
| H-ch8 | TinyLlama 1.1B | 10K | COMPLETE | Fiedler 0.094, no BD (spectral-only) |
| T-001r2 | TinyLlama 1.1B | 10K | **COMPLETE** | **41,564:1 co/cross, 4+4 BD (tesseract)** |
| H-ch12 | TinyLlama 1.1B | 10K | **COMPLETE** | Fiedler 0.102, spectral attractor at n=12 |
| WL-001 | TinyLlama 1.1B | 10K | **COMPLETE** | **co/cross ~0, 3+3 eigenvalues but WRONG direction** |
| R-001 | TinyLlama 1.1B | 10K | **COMPLETE** | co/cross ~0, chain-like eigenvalues, COLLAPSE |
| E-001 | TinyLlama 1.1B | 10K | **COMPLETE** | Fiedler 0.084, co/cross 1.12, spectral attractor |
| O-001 | TinyLlama 1.1B | 10K | **COMPLETE** | **clean 2+2 BD (octahedral, n=4)** |
| 24C-001 | TinyLlama 1.1B | 10K | **COMPLETE** | **co/cross 35,808:1** (PC-001 recovery), Fiedler 0.000555, c_w FIXED at 0.1 (CL2 logging bug hid growth from 5,673→35,808). Stabilization band 34,600–37,600:1 from step 8000. |
| FI-001 | TinyLlama 1.1B | 3K | COMPLETE | Init fingerprint: topology universal, signs init-specific |
| FI-002 | TinyLlama 1.1B | 3K×4 | **COMPLETE (P-CTRL stalled 1300)** | P-000=50K, P-001=52K, P-002=50K. P-CTRL plateau onset at step 1300 (10,654:1, 93% growth deceleration). Init independence CONFIRMED |
| FI-003 | TinyLlama 1.1B | 3K | **STOPPED (step 1200)** | **Signs collapse to random in 100 steps. Co/cross exponential decay: 12,586→10:1 in 1,200 steps. Steersman = homeostatic maintenance.** |
| FI-004 | TinyLlama 1.1B | 3K | **COMPLETE** | **5-regime annealing: peak 18,671:1 (c_w=0.017), cliff to 2,942:1 at c_w=0. Optimal c_w~0.02.** |
| AR-001 | Multiple | 0 (analysis) | **COMPLETE** | **Bridge asymmetry IS the BD signal.** Symmetrization destroys 99.95% of co/cross. Directed flow: 30:1 within-block coupling. |
| CW-001 | TinyLlama 1.1B | 10K | **COMPLETE** | **Three-phase trajectory: plateau→breakout→BD. Final 13,456:1 (peak 15,183:1 step 8300). Fiedler 0.00046. c_w controls SPEED (and possibly ceiling). Gap claim #7 confirmed.** |
| **P0** | **Qwen 1.5B** | **3,235** | **COMPLETE** | **Zero-cost bridge. Loss parity: 0.1762 vs 0.1763 (0.01%). Bridge overhead: 2,016 params (0.06%). Time: 58.3 min (+36.2%). Bridge settles near spectral attractor (Fiedler 0.038, co/cross 1:1).** |
| **24C-002** | **TinyLlama 1.1B** | **10K** | **COMPLETE** | **Adaptive c_w at n=24. Final co/cross 5,983:1 (peak 6,691:1 step 9800). Fiedler 0.00461. c_w settled at 0.025. Val 0.4025. 35h wall. 6× below 24C-001 (fixed c_w=0.1, 35,808:1). Confirms ceiling governed by c_w: adaptive Steersman undershoots at n=24 where 264 cross-axial pairs need more pressure. Non-monotonic trajectory: dip to 936:1 at step 3000 then growth.** |
| **FC-002** | **Qwen 1.5B** | **10K** | **COMPLETE** | **Fixed c_w=0.02, n=12. Fiedler 0.1036, co/cross null (no directional coherence). Spectral attractor at n=12 confirmed with fixed c_w on different model (Qwen 1.5B vs TinyLlama H-ch12). Val 0.3158. Crashed step 7900 (cp1252 encoding), resumed to 10K.** |
| FC-001 | Qwen 1.5B | 10K | **COMPLETE** | Fixed c_w=0.02 at n=6. Resumed from step 2300 (SIGPIPE crash). **Final: val 0.3129, co/cross 67,501:1, Fiedler 0.000086.** Fixed c_w comparable to H-ch6 adaptive (70,404:1). Confirms fixed contrastive weight is viable at n=6. |
| FO-001 | TinyLlama 1.1B | 10K | **COMPLETE** | Fixed c_w=0.02, n=4 (octahedron). Hermes RTX 4090. **Final: val 0.4113, co/cross 262,920:1, Fiedler 0.000032.** Fixed c_w=0.02 at n=4 produces 262,920:1 vs O-001 adaptive 473,622:1 (56%). Confirms fixed slightly undershoots adaptive at n=4 but still produces extreme directionality. |
| LORAXS-CORPUS | Qwen 1.5B | 3,073 | **COMPLETE — NEGATIVE** | LoRA-XS pivot: frozen A/B from SVD, train only R (24×24). 5 conditions. **D (LoRA-XS baseline, R≈0) = 0.4124 BEATS all corpus-initialized variants** (A=0.4262, B=0.4268, C=0.4257). Near-identity R init hurts by +3.2% vs near-zero. E (standard LoRA, 101× params) = 0.3729. Corpus values provide zero benefit for R initialization. |
| **G2-001** | **Qwen 1.5B** | **—** | **COMPLETE — DEGENERATE** | Bridge-swap eval: all PPL identical (14.23). **Null control only** — fingerprint experiments have no trained lora_A/B. Bridges on untrained projections trivially identical. L-016 composition thesis requires redesigned experiment with trained A/B projections + swapped bridges. 2 min runtime. |

---

## Exp 3.0: Cybernetic Bridge Training — COMPLETE, BREAKTHROUGH

**Status:** COMPLETE — the system's most significant result
**Model:** Qwen/Qwen2.5-7B-Instruct, rank 24, 6-channel FCC
**Steps:** 12,900 (4 epochs × ~3,225 steps/epoch — epoch-limited, not step-limited)
**Config max_steps:** 20,000 (training exhausted data at 12,900)
**Results local:** `results/exp3/`
**Training log:** `results/exp3/train.log` (234 KB, 12,900 entries)
**Bridge checkpoints:** 129 checkpoints (every 100 steps) + 112 `bridge_final` files
**Script:** `scripts/train_cybernetic.py` (36K)

### Architecture: SENSOR → OSCILLOSCOPE → STEERSMAN → ACTUATOR → SYSTEM → loop

**Key innovation:** `differentiable_fiedler()` — constructs weighted Laplacian from
bridge matrix off-diagonal elements, computes second eigenvalue via
`torch.linalg.eigvalsh` (differentiable through autograd). Algebraic connectivity
becomes a direct training objective, not post-hoc measurement.

**Three control laws:**
1. CONNECTIVITY: Fiedler trend declining → increase spectral regularization
2. DIRECTIONALITY: co/cross ratio stagnant → increase contrastive weight
3. STABILITY: deviation growing too fast → dampen bridge learning rate

### Results

| Metric | Value | Context |
|--------|-------|---------|
| **Axis alignment (median)** | **22,477:1** | Co-planar mean 1.517, cross-planar 7.3e-5 |
| **Fiedler** | 0.102 | Across all projections |
| **Final loss** | 0.293 (train), 0.241 (val) | |
| **Wall time** | 28.68 hours | Local RTX 6000 Ada |
| **Checkpoints** | 129 + final | Every 100 steps |

**Bug fixed (Mar 12):** `train_cybernetic.py` line 305 had `break` causing
single-bridge sampling in Steersman. Fixed to aggregate across all bridges.
Prior session had reported co/cross = 3.0 using wrong pair definitions
(hemisphere groups). Actual ratio using RD geometry: **22,477:1**.

### Exp 3.0 — TinyLlama 1.1B (Scale Validation)

**Status:** COMPLETE
**Model:** TinyLlama-1.1B, same cybernetic protocol, 88 adapters (22 layers × 4)
**Steps:** 10,000
**Results local:** `results/exp3_tinyllama/`

| Metric | Value | vs Qwen 7B |
|--------|-------|-----------|
| **Axis alignment (median)** | **47,145:1** | Even stronger at smaller scale |
| **Co-planar mean** | 0.778 | |
| **Cross-planar mean** | 2.1e-5 | |
| **Fiedler** | 0.1006 | Same ~0.10 convergence point |

**Finding:** Fiedler converges to ~0.10 regardless of model scale (1.1B, 7B).
RD geometry appears MORE cleanly in smaller models. Axis alignment emerges
immediately (step 200 already at 816:1).

---

## Holly Battery: Wan 2.1 14B T2V — COMPLETE

**Status:** COMPLETE — 3 production runs finished, 5 adamw8bit runs crashed
**Platform:** Minta's RunPod (alvdansen-labs)
**WandB project:** `alvdansen-labs/rhombi-experiment`
**Verified:** Mar 12, against both WandB API and raw .npy files

### Production Results (Prodigy Optimizer)

| Run | WandB ID | Final Loss EMA | Min Loss EMA | Peak VRAM (GB) | Runtime |
|-----|----------|---------------|-------------|----------------|---------|
| Standard LoRA | `rxhm9a4i` | 1.6137 | 1.6132 | 75.75 | 27.1h |
| **TeLoRA** | `u2acmrs0` | **1.5517** | **1.5447** | **66.60** | 25.5h |
| Corpus-Weighted | `n9t7op19` | 1.6453 | 1.6362 | 66.60 | 25.5h |

**Deltas:**
- **Loss:** 3.8% improvement (TeLoRA vs standard)
- **VRAM:** 9.15 GB less (66.60 vs 75.75)
- **Speed:** 6% faster (25.5h vs 27.1h)
- **Corpus weighting hurts** — worse than both standard and TeLoRA

### Crashed adamw8bit Runs

| WandB ID | Status | Runtime | Notes |
|----------|--------|---------|-------|
| `yi68ouj9` | crashed | 23.3h | Standard, VRAM reservation issue |
| `em6bc79n` | crashed | 3.5h | TeLoRA |
| `oy53awuf` | crashed | 1.0h | TeLoRA |
| `ola8rw8n` | crashed | 12.9h | TeLoRA |
| `ls150mpp` | crashed | 11.7h | Standard |

**Root cause:** VRAM reservation issue on 80 GB cards with adamw8bit. Minta
reports the issue was solved but doesn't recall the fix. Pods 3/4 on alvdansen
GitHub have the most up-to-date scripts but need further debugging.

### Pending

- [ ] Holly .safetensors weights from Minta (via alvdansen-labs GitHub)
- [ ] Timothy's access to alvdansen-labs GitHub org (timm156 invite sent)
- [ ] Minta observed TeLoRA "converging and training faster" — qualitative
- [ ] Wan 2.1 training pipeline bug (not inference) — Minta investigating

---

## Phase 3A: Overfitting Diagnostic — COMPLETE

**Status:** COMPLETE — RunPod pod terminated, results successfully recovered
**Model:** Qwen2.5-7B-Instruct, 500 train / 500 val split, 10K steps
**Results local:** `results/exp3a-overfit/`
**Script:** `scripts/train_overfit_diagnostic.py`

### Key Findings

| Metric | Value | p-value |
|--------|-------|---------|
| **deviation ~ train-val gap** | **r = 0.888** | **7.3e-35** |
| **Fiedler ~ train-val gap** | r = 0.825 | 5.6e-26 |
| **Phase transition** | Step 400 | 807× median deviation jump |

**Finding:** Bridge spectral properties DO correlate with overfitting. NOT null.
Deviation and Fiedler both track train-val gap with strong correlation.
Phase transition at step 400 (807× median) marks onset of memorization.

**Report:** Raw data in `results/exp3a-overfit/results.json` (100 checkpoint
entries, step 0 through 10,000). No dedicated synthesis report written yet.

---

## Exp 2.7: Separate Bridge Learning Rate — COMPLETE, NULL

**Status:** COMPLETE — NULL on directionality
**Model:** Qwen2.5-7B-Instruct, 2K steps, bridge LR = 10× base, constant schedule
**Results local:** `results/exp2_7/`

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| Co/Cross | 0.96-1.05 | Never breaks past 1.05 through 2000 steps |
| Fiedler | ~0.15 | Higher than Exp 2 (higher LR → more coupling overall) |

**Finding:** Higher bridge LR increases total coupling but produces no directional
preference. Confirms the null from Exp 2.5 — the mechanism itself needs modification,
not just the hyperparameters.

**Note:** No bridge .npy files saved — only JSON metrics in results.json.

---

## Completed Experiments (Earlier)

### Exp 1: 1.5B Scale Proof of Concept
- **Model:** Qwen2.5-1.5B-Instruct, rank 24
- **Result:** Bridge learns, FCC > cubic 4.6× Fiedler. Architecture works.
- **Location:** `results/exp1/`

### Exp 2: 7B Scale Baseline (Alpaca)
- **Model:** Qwen2.5-7B-Instruct, 10K steps, Alpaca-cleaned
- **Key results:**
  - FCC Fiedler: 0.0401, Cubic: 0.0231 → **1.73× ratio**
  - Co/Cross: 1.019 → weak directional signal (isotropic data expected)
  - Permutation p: 0.332 → not significant
  - Transient peak at step 3000 (1.091) → decays
- **Location:** `results/exp2/`

### Exp 2.5: 7B Geometric Data (NULL)
- **Model:** Qwen2.5-7B-Instruct, 3K steps, geometric dataset (23K examples)
- **Key results:**
  - Co/Cross: 1.002 (NULL — no directional signal, p=0.474)
  - Fiedler: 0.030 (0.76× Exp 2 final)
- **Finding:** Prompt-level co-planar bias does not translate to channel-level
  co-activation through 28 transformer layers. Channel assignment is arbitrary.
- **Root cause identified:** L-001 (rank dimensions are rotationally symmetric)
- **Location:** `results/exp2_5/`

### Phase 0: Bridge Anatomy (FREE)
- **Results:** `results/exp2/bridge_anatomy.md`
- **Key findings:** Q-proj 40% more coupling (p=0.0008). Layer depth
  affects deviation (p=0.001). Q-V closer than Q-K (p=0.042).

### Phase 1A: Task Fingerprints (STRONG SIGNAL)
- **Results:** `results/fingerprints/`
- **Key findings:** 72.3% LOO SVM (all modules, 336 adapters) / 73.5% (2-way). Code most
  distinctive (97.3%). Mann-Whitney p = 0.000000.
  **CORRECTION (Apr 6, 2026):** Q-proj-only SVM re-run produces **69.0%** (84 samples, 36 params). Previously reported as 84.5% — that number was never backed by reproducible computation. Q-proj is the best single module but WORSE than all-modules-combined. Per-module breakdown: q_proj=69.0%, o_proj=60.7%, k_proj=58.3%, v_proj=51.2%.

### Phase 2A: Bridge-Level Merging (MIXED)
- **Results:** `results/bridge_merge/`
- **Key findings:** alpaca↔code R²=0.956 (linear), alpaca↔math R²=0.735
  (non-linear), code↔math R²=0.823 (non-linear). All pass random baseline.
  Eigenspectrum cos > 0.999. Safe mixing: up to 10%.

---

## Channel Ablation Series — COMPLETE (Mar 13-15)

**Thesis:** Only contrastive topology produces block-diagonal structure.
Spectral-only training converges to generic distributed connectivity
regardless of channel count.

**Model:** TinyLlama-1.1B, rank 24, identity init, default Steersman, 10K steps.
**Location (Hermes):** `results/channel-ablation/H-ch{3,4,6,8,12}`
**Location (Local):** `results/tesseract-contrastive/` (500 steps), `results/T-001-full/` (restarted)

### Results Summary

| Run | n_ch | Topology | Val Loss | Co/Cross | Fiedler | Deviation | Block-Diag? | Status |
|-----|------|----------|----------|----------|---------|-----------|-------------|--------|
| H-ch3 | 3 | spectral | 0.4020 | N/A | 0.0951 | 0.2020 | **NO** | DONE |
| H-ch4 | 4 | spectral | 0.4022 | N/A | 0.0918 | 0.2234 | **NO** | DONE |
| H-ch6 | 6 | RD contrastive | 0.4015 | **70,404** | 0.00009 | 1.3668 | **YES** | DONE |
| H-ch8 | 8 | spectral | 0.4022 | N/A | 0.0889 | 0.214 | **NO** | DONE |
| T-001r2 | 8 | tesseract contrastive | 0.439 | **41,564** | 0.00019 | — | **YES (4+4)** | DONE |
| H-ch12 | 12 | spectral | 0.4020 | N/A | 0.102 | — | **NO** | DONE |

### Key Findings

1. **Spectral-only convergence is universal.** n=3, n=4, and n=8 (projected)
   all converge to Fiedler ~0.09, deviation ~0.2. Channel count doesn't matter.
   The Steersman creates connectivity but no directional preference.

2. **Contrastive topology IS the structure signal.** n=6 with RD contrastive
   produces 70,404:1 co/cross ratio. n=8 with tesseract contrastive produces
   2,835:1 at step 500 (still rising when run terminated).

3. **Prediction A confirmed:** The Steersman is a general topology programmer.
   It programs ANY specified geometry — RD (3 co-axial pairs) or tesseract
   (4 co-axial pairs). The block-diagonal structure comes from the co-axial
   pair specification, not from anything intrinsic to 6 channels.

4. **T-001 eigenvalue pattern (step 200):** Clean 4+4 split emerging — 4 near-zero
   eigenvalues and 4 large (~0.05). By step 500: [~0, 0.00018, 0.00020, 0.00026]
   vs [0.164, 0.171, 0.172, 0.173]. The tesseract's 4 axes are being resolved.

### All Channel Ablation Runs COMPLETE

- **H-ch8, H-ch12:** Both confirmed spectral attractor (~0.09-0.10 Fiedler, no BD).
- **T-001r2:** 41,564:1 co/cross, clean 4+4 BD at 10K steps (tesseract).
- **WL-001:** co/cross ~0, 3+3 eigenvalues but WRONG direction.
- **R-001:** co/cross ~0, chain-like eigenvalues, COLLAPSE.
- **E-001:** Fiedler 0.084, co/cross 1.12, spectral attractor.
- **O-001:** Clean 2+2 BD (octahedral, n=4).
- **24C-001:** COMPLETE — 35,808:1, 12+12 split, Fiedler 5.55×10⁻⁴ (see dedicated section below).

---

## Completed Negative Controls — WL-001, R-001, E-001

### WL-001: Wrong-Labels Control — COMPLETE
- **n=6, random partition into 3 pairs** (no geometric prior)
- **Result:** co/cross ~0, 3+3 eigenvalues but in WRONG direction. Confirms geometry matters.
- **Location:** `results/wrong-labels/WL-001/`

### R-001: Circular Resonance — COMPLETE (COLLAPSE)
- **n=6, prime-threading topology**
- **Result:** co/cross ~0, chain-like eigenvalues. Resonance loss produces spectral structure
  but no BD. The topology collapses.
- **Location:** `results/resonance/R-001/`

### E-001: Emanation Architecture — COMPLETE
- **n=6, master bridge + per-layer offsets** (Plotinus-inspired hierarchy)
- **Result:** Fiedler 0.084, co/cross 1.12. Spectral attractor but no BD.
  Shared master bridge does not create directional structure.
- **Location:** `results/emanation/E-001/`

### O-001: Octahedral Contrastive — COMPLETE
- **n=4, octahedral geometry, 2 co-axial pairs**
- **Result:** Clean 2+2 BD. Confirms the Steersman programs any valid geometry.
- **Location:** `results/octahedral/O-001/`

---

## Planned Experiments — Not Started

### Exp 2.6: Contrastive Bridge Pre-Training (Deprioritized)
- **Prerequisite:** None — null branch option 1
- **Status:** Deprioritized. Exp 3.0 solved the directionality problem via
  cybernetic closed-loop training. Contrastive pre-training on isotropic data
  may still be informative as a simpler alternative but is no longer critical path.

### Exp 2.8: Input-Dependent Bridge (Dynamic Routing)
- **Prerequisite:** None — null branch option 3
- **Status:** Designed but not started. The cybernetic bridge result (Exp 3.0)
  may make static bridges sufficient if contrastive training installs durable
  directional preference.

### Generative Evaluation (Critical Missing Experiment)
- **Status:** Designed, not started
- **Design:** Swap bridges between task-specific adapters and measure perplexity
  on held-out task data. Tests whether bridge-only swapping produces usable
  task behavior changes at inference.
- **Importance:** Paper 3 limitations section identifies this as the critical gap.

---

## 24C-001: 24-Cell D4 Root Polytope — COMPLETE

**Status:** COMPLETE (step 10000, finished ~Mar 21 2026)
**Model:** TinyLlama-1.1B-Chat, rank 24, n=24 channels (channel_size=1)
**Pair spec:** 12 antipodal co-axial pairs from D4 root polytope
  - Vertices: permutations of (±1,±1,0,0) in R⁴
  - 6 coordinate-plane groups × 2 pairs each = 12 co-axial, 264 cross-axial
**Bridge params:** 50,688 (24×24=576 per bridge × 88 bridges)
**Prediction:** 12+12 block-diagonal eigenvalue split
**Estimated completion:** ~30h from launch (~Mar 21 00:00 UTC)

### Checkpoint Data

Co/cross computed post-hoc from bridge .npy files using exact D4 antipodal pairs from
`_compute_pair_indices()`: (0,3),(1,2),(4,7),(5,6),(8,11),(9,10),(12,15),(13,14),(16,19),(17,18),(20,23),(21,22).
Pooled ratio = mean(|co-pair elements|) / mean(|cross-pair elements|) across all 88 bridges.
Block separation from mean Laplacian eigenvalue spectrum across all 88 bridges.

| Step | Fiedler | Co/Cross | Block Sep | Val Loss |
|------|---------|----------|-----------|----------|
| 0 | 0.000 | — | 1.0× | — |
| 100 | 0.043 | 2.6:1 | 1.0× | 0.476 |
| 300 | 0.015 | 33.6:1 | 2.0× | 0.443 |
| 500 | 0.006 | 126.1:1 | 5.3× | 0.439 |
| 1000 | 0.003 | 479.5:1 | 16.8× | 0.434 |
| 1500 | 0.0019 | 1,174.5:1 | 38.1× | 0.431 |
| 2000 | 0.0019 | 1,810.2:1 | 71.0× | 0.426 |
| 2500 | 0.0020 | 2,205.1:1 | 102.0× | 0.424 |
| 2800 | 0.0023 | 2,343.8:1 | 115.6× | 0.422 |
| **3000** | **0.0023** | **2,489.7:1** | **127.1×** | **0.422** |
| **3100** | **0.0023** | **2,564.2:1** | **132.8×** | **0.421** |
| 3200 | 0.0023 | 2,561:1 | — | 0.421 |
| 3500 | 0.0026 | — | — | 0.420 |
| 3900 | 0.0028 | 2,637:1 | — | 0.419 |
| 4000 | 0.0026 | — | — | 0.418 |
| 4100 | 0.0027 | — | — | 0.417 |
| 4200 | 0.0028 | — | — | 0.417 |
| **4300** | **0.0026** | **5,673:1†** | — | **0.416** |

†Step 4300 co/cross = aggregate mean across all 88 bridge_final layers (median 3,468:1,
range 989:1 to 36,673:1). All earlier values computed from single-layer or subset samples.

**NOTE (Mar 19-20, 2026):** The training script's built-in co/cross metric returns null for n≠6
because the running process loaded code BEFORE the n=24 `_coplanar_crossplanar_indices`
handler was added. Co/cross values in this table are computed post-hoc from saved bridge files.

**CRITICAL FINDING (Mar 20, 2026): 24C-001 is an accidental fixed-c_w experiment.**
Control Law 2 (directionality, line 361 of train_cybernetic.py) guards on `co_cross is not None`.
Since co_cross is always None for this run, the entire directionality control law is inoperative.
**c_w has been fixed at 0.1 (the default `base_contrastive_weight`) for the entire run.**
This is 5× above FI-004's optimal c_w=0.02. Result: massive BD structure (5,673:1 at step 4300)
without any adaptive decay bottleneck — the strongest evidence for the fixed-contrastive hypothesis.

**Fiedler dynamics (step 1900-4300):** Fiedler reached minimum 0.00186 at step 1900,
rebounded to 0.00282 at step 3900 (52% increase), then entered oscillation (0.0025-0.0028)
from step 3900-4300. The rebound phase may be transitioning into reconvergence.

**Co/cross growth rate:** Decelerating — ~100/100 steps (steps 1000-2000), ~70/100 steps
(steps 2000-3000), ~11/100 steps (steps 3200-3900). The step 4300 aggregate (5,673:1)
suggests continued growth when measured across all layers rather than single-layer samples.

Key dynamics:
- **c_w fixed at 0.1** — Control Law 2 never fires (co_cross always None)
- **Fiedler oscillating** in 0.0025-0.0028 band (rebound→reconvergence transition)
- **Val loss still improving** monotonically (0.476→0.416)
- **Co/cross massive** even at 5× above FI-004 "optimal" — questions whether 0.02 is truly optimal

**Signal density comparison (co/cross at step 1000):**
- n=4 (octahedron): 7,224:1 — signal density 0.50
- n=6 (RD): 7,246:1 — signal density 0.25
- n=8 (tesseract): 4,611:1 — signal density 0.17
- **n=24 (24-cell): 20.6:1** — signal density 0.045

The 99.7% reduction from n=6 to n=24 vastly exceeds the 3.7× reduction
in signal density, suggesting a critical threshold in suppression load.

---

## Paper 4 Status — DRAFT ASSEMBLED

**Title:** "The Topology Programmer: Cybernetic Feedback as a General-Purpose
Geometric Prior for Neural Network Adapters"
**File:** `paper/rhombic-paper4.tex` (16 pages, compiles clean, 0 undefined refs)
**Sections:** 10 files in `paper/paper4-sections/`

### Section Status

| Section | Status | File |
|---------|--------|------|
| §1 Introduction | **WRITTEN** | `01_introduction.tex` |
| §2 Background | **WRITTEN** | `02_background.tex` |
| §3 Method | **WRITTEN** | `03_method.tex` |
| §4 Three Geometries | **WRITTEN** | `04_three_geometries.tex` |
| §4b Four Dimensions | **WRITTEN** (24C pending) | `04b_four_dimensions.tex` |
| §5 Spectral Attractor | **WRITTEN** | `05_spectral_attractor.tex` |
| §6 Negative Controls | **WRITTEN** | `06_negative_controls.tex` |
| §7 Four Regimes | **WRITTEN** | `04c_regimes_synthesis.tex` |
| §8 Discussion | **WRITTEN** | `07_discussion.tex` |
| §9 Conclusion | **WRITTEN** | `08_conclusion.tex` |

### Remaining for Submission
- [x] 24C-001 results — COMPLETE (35,808:1, PC-001 recovery, LaTeX updated)
- [x] FI-002 init independence results — COMPLETE (100% sign convergence)
- [ ] Generate figures (polytope diagrams, regime taxonomy, trajectories)
- [ ] Final proofread and number verification
- [ ] arXiv endorsement (BLOCKED)

---

## Paper 3 Status — SUBSTANTIALLY COMPLETE

**Title:** "The Learnable Bridge: Task Fingerprinting and Adapter Composition
via Structured Coupling in Low-Rank Adaptation"

**File:** `paper/rhombic-paper3.tex` + 6 section files. PDF compiles clean.
**Italian translation:** `paper/rhombic-paper3_IT.tex` — also compiled.

### Section Status

| Section | Status | Source |
|---------|--------|--------|
| §1 Introduction | **WRITTEN** | `paper/section1_introduction.tex` |
| §2 Architecture | **WRITTEN** | `paper/section2_architecture.tex` |
| §3 Theoretical Grounding | **WRITTEN** | Inline in main .tex |
| §4 Experiments | **WRITTEN** | `paper/section4_experiments.tex` |
| §5 Related Work | **WRITTEN** | Inline from `docs/competitive_landscape.md` |
| §6 Discussion + Conclusion | **WRITTEN** | `paper/section6_discussion.tex` |

### Remaining for arXiv Submission
- [ ] Final proofread
- [ ] Update Exp 3.0 numbers with corrected 22,477:1 axis alignment
- [ ] Add Holly Battery results (Wan 2.1 14B — real-world validation)
- [ ] Verify all cross-references and citations
- [ ] Generate final figures with `scripts/generate_paper3_figures.py`

---

## Hackathon Status — Nous Research Hermes Agent

**Deadline:** EOD Sunday March 16, 2026
**Progress:** 8/10 checklist items done

### Completed
- [x] 9/9 custom tools written, tested, registered
- [x] 3/3 skills deployed to `~/.hermes/skills/rhombic/`
- [x] Context file at `~/hermes-agent/rhombic-context.md`
- [x] model_tools.py + toolsets.py patched, `rhombic` toolset available
- [x] Presentation website built (`rhombic/website/`)
- [x] 3D RD hero animation (Three.js)
- [x] CSS metric bars
- [x] Interactive Fiedler slider descoped to static bars

### Remaining
- [ ] Deploy website (GitHub Pages)
- [ ] Record video demo (60-90s)
- [ ] Discord verify
- [ ] Draft tweet + submit

**Plan:** `rhombic/docs/HACKATHON_SPRINT_PLAN.md`

---

## WandB Project Inventory (Complete)

### alvdansen-labs (38 runs total)

| Project | Runs | Finished | Crashed/Failed | Purpose |
|---------|------|----------|----------------|---------|
| rhombi-experiment | 8 | 3 | 5 | Holly Battery (Wan 2.1 14B) |
| holly_i2v_test | 11 | 1 | 10 | Wan 2.2 I2V experiments |
| dimljus-base-comparison | 4 | 4 | 0 | Minta's character LoRA |
| dimljus-stills | 4 | 2 | 2 | |
| dimljus-resume | 6 | 3 | 3 | |
| dimljus-isolation | 5 | 4 | 1 | |

### timotheospaul-tasumer-maf (3 runs)

| Project | Runs | Finished | Failed | Purpose |
|---------|------|----------|--------|---------|
| enochiatron-training | 3 | 2 | 1 | LTX-2.3 22B on Kallisti |

---

## Infrastructure Checklist

### Tools — Ready
- [x] `train_cybernetic.py` — Exp 3.0 cybernetic bridge trainer (Steersman bug FIXED)
- [x] `train_comparison.py` — Exp 1 cubic vs FCC
- [x] `train_exp2_5.py` — Exp 2.5 geometric data
- [x] `train_exp2_scale.py` — Exp 2 7B baseline
- [x] `train_separate_lr.py` — Exp 2.7 bridge LR
- [x] `train_overfit_diagnostic.py` — Phase 3A
- [x] `train_task_fingerprint.py` — Phase 1A
- [x] `train_contrastive_bridge.py` — contrastive pre-training
- [x] `analyze_cybernetic.py` — Exp 3 analysis
- [x] `compare_exp2_exp25.py` — full spectral comparison (4 tools)
- [x] `peek_bridges.py` — early checkpoint inspection
- [x] `pull_runpod_results.sh` — download results pipeline
- [x] `deploy_runpod.sh` — generic pod deployment
- [x] `pack_for_runpod.sh` — minimal tarball for upload

### Tools — Not Yet Implemented
- [ ] Input-dependent bridge architecture (Exp 2.8)
- [ ] Generative evaluation script (bridge-swap + perplexity)
- [ ] Flimmer integration (LoRAState topology field)

### Bug Tracker
| Bug | Severity | Status | Notes |
|-----|----------|--------|-------|
| Val split overlap | Low | Fixed locally | Bridge metrics unaffected |
| Eff rank = 0.00 | Low | Known | Gradient timing bug, not important |
| `_bridge_fiedler` adapter | Medium | **Fixed** | Was calling fiedler_value(B) incorrectly |
| **Steersman single-bridge sampling** | **High** | **Fixed Mar 12** | `break` at line 305 caused single-bridge sampling. Fixed to aggregate across all bridges. Prior co/cross numbers were wrong (3.0 → 22,477:1). |
| flimmer-trainer CI | Medium | **ACTIVE** | `test_block_swap_called_after_initial_model_load` fails after commit `678644c` (block swap + LoRA compat). 1569/1570 pass. alvdansen repo. |

---

## Cost Tracking

| Item | Cost | Date | Status |
|------|------|------|--------|
| RTX 6000 Ada (Exp 1, 2, 3.0, 3.0-TL, Phases 0/1A-code/2A/3A-local) | Electric only | Mar 6-12 | COMPLETE |
| RunPod 4090 (Exp 2.5) | ~$1.40 | Mar 8 | COMPLETE |
| RunPod 4090 (Phase 1A math) | ~$0.95 | Mar 8 | COMPLETE |
| RunPod 4090 (Phase 3A overfit) | ~$4.00 | Mar 8 | COMPLETE, pod terminated |
| Minta's RunPod (Holly Battery ×3) | ~$50+ | Mar 11 | COMPLETE (alvdansen account) |
| Minta's RunPod (Holly adamw8bit ×5) | ~$25+ | Mar 11-12 | CRASHED |

---

## Local Results Directory Inventory (20 directories)

```
results/
├── exp1/                    Exp 1: 1.5B PoC — COMPLETE
├── exp2/                    Exp 2: 7B Baseline — COMPLETE
├── exp2_5/                  Exp 2.5: Geometric Data — COMPLETE (NULL)
├── exp2_7/                  Exp 2.7: Separate Bridge LR — COMPLETE (NULL)
├── exp3/                    Exp 3.0: Cybernetic Bridge (Qwen 7B) — COMPLETE
├── exp3_test/               Exp 3.0 test run (50 steps) — test only
├── exp3_tinyllama/          Exp 3.0: Cybernetic Bridge (TinyLlama) — COMPLETE
├── exp3a-overfit/           Phase 3A: Overfitting Diagnostic — COMPLETE
├── fingerprints/            Phase 1A: Task Fingerprints — COMPLETE
├── bridge_merge/            Phase 2A: Bridge-Level Merging — COMPLETE
├── paper2/                  Paper 2 experiments — COMPLETE
├── multi_seed/              Multi-seed validation — COMPLETE
├── index/                   Rung 4 embedding benchmarks — COMPLETE
├── rung-1/                  Library Rung 1 — COMPLETE
├── rung-2/                  Library Rung 2 — COMPLETE
├── rung-3/                  Library Rung 3 — COMPLETE
├── rung-4/                  Library Rung 4 — COMPLETE
├── SYNTHESIS.md             Master synthesis (17 KB)
├── CROSS_PHASE_SYNTHESIS.md Cross-phase analysis (10 KB)
├── VERIFIED_FINDINGS_2026_03_12.md  Latest verified numbers
├── tesseract-contrastive/   T-001: Tesseract n=8 partial (500 steps)
├── tesseract-contrastive-full/  T-001: Earlier attempt (300 steps)
└── T-001-full/              T-001: Tesseract restart — COMPLETE (10K, 41,564:1)
```

**On Hermes (`/home/timm156/rhombic/results/`):**
```
channel-ablation/
├── H-ch3/   n=3 spectral — COMPLETE
├── H-ch4/   n=4 spectral — COMPLETE
├── H-ch6/   n=6 RD contrastive — COMPLETE
├── H-ch8/   n=8 spectral — COMPLETE
├── H-ch12/  n=12 spectral — COMPLETE
├── 24C-001/ n=24 D4 root — COMPLETE (10K, 35,808:1)
wrong-labels/WL-001/   — COMPLETE
resonance/R-001/       — COMPLETE
emanation/E-001/       — COMPLETE
```

---

## Key Numbers (Verified Mar 12, 2026 + Channel Ablation Mar 15)

| Claim | Value | Source |
|-------|-------|--------|
| FCC > cubic Fiedler (1.5B) | 4.6× | Exp 1 |
| FCC > cubic Fiedler (7B) | 1.73× | Exp 2 |
| Task fingerprint LOO SVM | 72.3% (all modules) / 73.5% (2-way) | Phase 1A |
| Bridge merge eigenspectrum | cos > 0.999 | Phase 2A |
| Deviation ~ overfit gap | r = 0.888, p = 7.3e-35 | Phase 3A |
| Fiedler ~ overfit gap | r = 0.825, p = 5.6e-26 | Phase 3A |
| Axis alignment (7B) | 22,477:1 | Exp 3.0 |
| Axis alignment (1.1B) | 47,145:1 | Exp 3.0-TL |
| Fiedler convergence | ~0.10 (scale-invariant) | Exp 3.0 + 3.0-TL |
| Holly loss improvement | 3.8% (1.5517 vs 1.6137) | Holly Battery |
| Holly VRAM savings | 9.15 GB (66.60 vs 75.75) | Holly Battery |
| Holly speed improvement | 6% (25.5h vs 27.1h) | Holly Battery |
| **Spectral-only Fiedler (n=3)** | **0.0951** | Channel Ablation H-ch3 |
| **Spectral-only Fiedler (n=4)** | **0.0918** | Channel Ablation H-ch4 |
| **Spectral-only Fiedler (n=8)** | **0.0889** | Channel Ablation H-ch8 (proj) |
| **RD contrastive co/cross (n=6)** | **70,404:1** | Channel Ablation H-ch6 |
| **Tesseract contrastive co/cross (n=8)** | **41,564:1** | T-001r2 (10K, COMPLETE) |
| **Tesseract 4+4 split ratio** | **~800:1** (0.17 / 0.00021) | T-001 eigenvalues |
| **24-Cell co/cross (n=24, final)** | **35,808:1** | 24C-001 (COMPLETE, PC-001 recovery, c_w FIXED at 0.1) |
| **Octahedral co/cross (n=4)** | **473,622:1** | O-001 (COMPLETE, strongest BD signal) |
| **Whisper-strength final** | **13,456:1** (peak 15,183:1 step 8300) | CW-001 (COMPLETE, Fiedler 0.00046) |
| **FI-004 peak (annealing)** | **18,671:1 (c_w=0.017)** | FI-004 (COMPLETE) |
| **Asymmetry ratio (BD vs non-BD)** | **0.30-0.50 vs 0.02 (16×)** | AR-001 (COMPLETE) |
| **Symmetrization signal loss** | **99.95%** | AR-001 (COMPLETE) |
| **Within-block directional coupling** | **30:1** | AR-001 (COMPLETE) |

---

---

## Falco Intelligence Experiments (Stream B — Proprietary)

### FI-001: Init-Strategy Fingerprint — COMPLETE + EXTENDED

**Status:** COMPLETE (Mar 18, 2026)
**Question:** Does initialization strategy leave a detectable fingerprint in trained bridges?
**Key finding:** Bridge = topology (universal) + signs (init-determined) + magnitudes (~0).
Signs frozen at 98.2% after 1200 steps. Only geometric/corpus-coupled init produces structured signs.
**Results:** `results/fi-001/`

### FI-002: Corpus-Derived Pair Specification — COMPLETE (P-CTRL stalled at 1300)

**Status:** P-000/P-001/P-002 DONE. P-CTRL stalled at step 1300/3000 (process hung then died).
**Question:** Do different corpus encodings produce distinguishable sign patterns?
**Model:** TinyLlama 1.1B, rank 24, 6-channel, 3000 steps each
**Script:** `scripts/fi_002_corpus_pairs.py`
**Results:** `results/fi-002/`

| Config | Channel Perm | Init Hamming | Steps | Co/Cross | Status |
|--------|-------------|-------------|-------|----------|--------|
| P-000 (canonical) | [0,1,2,3,4,5] | 0/12 | 3000 | 50,344:1 | DONE |
| P-001 (max-diff) | [0,2,4,1,5,3] | 9/12 | 3000 | 51,677:1 | DONE |
| P-002 (moderate) | [0,1,4,3,5,2] | 7/12 | 3000 | 50,382:1 | DONE |
| P-CTRL (identity) | None | N/A | 1300 | 10,654:1 | STALLED (hung after 1300) |

**P-CTRL trajectory (identity init):**
| Step | Co/Cross | Fiedler | Val Loss | Eigenvalues |
|------|----------|---------|----------|-------------|
| 100 | 74.6:1 | 0.0031 | 0.4766 | [0.000, 0.001, 0.001, 0.015, 0.015, 0.015] |
| 200 | 699:1 | 0.00044 | 0.4476 | [0.000, 0.000, 0.000, 0.050, 0.051, 0.051] |
| 300 | 1,600:1 | 0.00014 | 0.4435 | [0.000, 0.000, 0.000, 0.093, 0.095, 0.096] |
| 400 | 2,752:1 | 0.000125 | 0.4404 | [0.000, 0.000, 0.000, 0.130, 0.134, 0.135] |
| **500** | **3,487:1** | **0.000123** | **0.4389** | **[0.000, 0.000, 0.000, 0.163, 0.170, 0.171]** |
| **600** | **4,817:1** | **0.000102** | **0.4378** | **[0.000, 0.000, 0.000, 0.194, 0.202, 0.204]** |
| **700** | **5,172:1** | **0.000111** | **0.4371** | **[0.000, 0.000, 0.000, 0.222, 0.232, 0.233]** |
| **800** | **6,424:1** | **0.000100** | **0.4359** | **[0.000, 0.000, 0.000, 0.247, 0.259, 0.260]** |
| **900** | **7,244:1** | **0.000107** | **0.4350** | **[0.000, 0.000, 0.000, 0.270, 0.283, 0.284]** |
| **1000** | **8,085:1** | **0.000103** | **0.4345** | **[0.000, 0.000, 0.000, 0.291, 0.305, 0.306]** |
| **1100** | **9,227:1** | **0.000100** | **0.4338** | **[0.000, 0.000, 0.000, 0.310, 0.324, 0.326]** |
| **1200** | **10,556:1** | **0.000099** | **0.4338** | — |
| **1300** | **10,654:1** | **0.000107** | **0.4332** | — |

**PLATEAU ONSET at step 1300:** Co/cross delta collapsed from +1,329 to +98 (93%
deceleration). Fiedler bounced from minimum (0.0000990) to 0.000107. This matches
the H-ch6 two-phase pattern: rapid climb (steps 0-1200) → plateau/oscillation
(steps 1200+). Process hung at 100% CPU for 7h after step 1300, then died.
Data is scientifically sufficient for the initialization independence claim.

**Key finding:** Identity initialization produces the SAME 3+3 BD topology
as corpus-coupled initialization — clean eigenvalue split visible by step 100.
Co/cross ratio ~15× behind corpus configs at same step count, but trajectory
is the same. Topology is pair-specification-determined, not init-determined.

**Corpus configs final:** 50-52K co/cross range. P-002 peaked at 57,230 (step 2900),
settled to 50,382 (step 3000) — consistent with Fiedler reconvergence dynamics.

**FI-002 ANALYSIS COMPLETE (Mar 21, 2026):**
- **100% identical trained sign patterns** across all 3 corpus configs despite
  Hamming 9/12 initial differences
- **Frobenius distances:** 1.05e-4, 1.47e-4, 1.93e-4 (all ~10^-4)
- **Sign correction:** P-001 init→trained 25%→100% (75% corrected),
  P-002 41.7%→100% (58.3% corrected)
- **Verdict:** "CORPUS ENCODINGS ARE INDISTINGUISHABLE" — confirms Paper 4 §6
- **Analysis results:** `results/fi-002/fi-002-results.json`

**Growth rate:** 75→699→1,600→2,752→3,487→4,817→5,172→6,424→7,244→8,085→9,227→10,556→10,654.
Co/cross growth oscillates **anti-correlated with Fiedler** (steps 500-1200), then
enters plateau at step 1300 with Fiedler rebound.

**Comparison with H-ch6:** P-CTRL tracks H-ch6 to ±20% through step 600, then begins
to pull ahead. At step 1200: P-CTRL=10,556 vs H-ch6=8,123 (30% ahead). Identity
initialization is NOT slower than corpus-coupled — the convergence rate is wholly
pair-specification-determined. Plateau onset at step 1300 matches H-ch6's two-phase
pattern (rapid climb then oscillation/plateau).

### FI-003: Sign Persistence Under Unguided Fine-Tuning — STOPPED (step 1100/3000)

**Status:** STOPPED at step 1100 (process died; sufficient data for characterization)
**Question:** Do the init-determined signs persist when the Steersman is REMOVED
and the bridge continues training with gradient flow only?
**Design:**
1. Take FI-002 P-000 checkpoint at step 3000 (full BD, corpus-coupled signs)
2. Continue training for 3000 more steps with Steersman DISABLED (no spectral or contrastive loss)
3. Track sign evolution: do signs rotate toward the universal attractor or remain locked?
4. Compare sign pattern at step 6000 (post-removal) to step 3000 (pre-removal)

**Prediction:** Signs should remain frozen (from FI-001: 98.2% stability). The
Steersman wrote the topology; removing the Steersman should not erase it.
If signs DO rotate, this reveals the Steersman is continuously maintaining
the topology against gradient pressure — a homeostatic maintenance function.

**GPU:** Local RTX 6000 Ada. Duration: ~3000 steps × 5s/step = ~4h.

**RESULT (step 600 of 3000): PREDICTION FALSIFIED — Steersman = homeostatic maintenance**

The topology decays immediately and completely:

| Step | Sign Stability | Co/Cross | Val Loss | Magnitude |
|------|---------------|----------|----------|-----------|
| 0 (initial) | 100% | 12,586:1 | — | — |
| 100 | 50.85% | 178.7:1 | 0.499 | 0.0277 |
| 200 | 51.33% | 60.0:1 | 0.590 | 0.0278 |
| 300 | 49.15% | 29.9:1 | 0.608 | 0.0280 |
| 400 | 50.28% | 21.8:1 | 0.603 | 0.0280 |
| 500 | 48.77% | 18.1:1 | 0.568 | — |
| 600 | 49.24% | 15.6:1 | 0.566 | — |
| 700 | 48.39% | 13.7:1 | 0.568 | — |
| 800 | 49.43% | 12.5:1 | 0.557 | — |
| 900 | 48.96% | 11.5:1 | 0.545 | — |
| 1000 | 49.91% | 10.8:1 | 0.528 | — |
| 1100 | 50.57% | 10.2:1 | 0.516 | — |
| 1200 | 52.08% | 9.8:1 | — | — |

**Key findings:**
1. **Sign collapse is immediate and total.** By step 100 (3.3% of original training),
   sign stability = 50.85% = random. The ±2% oscillation around 50% is noise.
2. **Co/cross decays exponentially.** 12,586 → 179 → 60 → 30 → 22 → 18 → 16 → 14 → 13 → 12 → 11 → 10 over 1,100 steps.
   Double-exponential fit (R²=0.9999998): fast (half-life 15 steps) + slow (half-life 230 steps).
   Predicted to reach 1:1 (isotropic) by ~step 2500.
3. **Magnitudes barely change.** 0.0277 → 0.0280 — the bridge weights aren't changing
   magnitude, only losing directional coherence. The BD is dissolving into isotropy.
4. **Val loss rises then stabilizes.** 0.499 → 0.608 → 0.603 — the adapter still
   functions but loses geometric organization. The model degradation is bounded.
5. **Crystal analogy FAILS.** The topology is NOT self-sustaining once established.
   Better analogy: a spinning top that topples when the driving force is removed.
6. **Combined with FI-002:** The Steersman is both ROBUST (same topology from any
   initialization) and NECESSARY (topology dissolves without it). The BD configuration
   is a Steersman-maintained fixed point, accessible from any init but unstable under
   LM loss alone.

### FI-004: Steersman Annealing — COMPLETE (5-Regime Model)

**Status:** COMPLETE (2026-03-19). 30 checkpoints, 3000 steps.
**Question:** At what contrastive weight does the Steersman lose its ability to maintain topology?
**Design:**
1. Take FI-002 P-000 checkpoint (same as FI-003)
2. Continue training 3000 steps with LINEAR ANNEALING of Steersman weights
3. Contrastive: 0.1 → 0.0, Spectral: 0.05 → 0.0
4. Track BD metrics vs. contrastive weight to find critical threshold

**RESULT (step 700): U-SHAPED RECOVERY — interference weakening as weight drops**

| Step | c_weight | s_weight | Fiedler | Co/Cross | Sign Stability | Val Loss |
|------|----------|----------|---------|----------|---------------|----------|
| 0 | 0.1000 | 0.0500 | 0.100 | 12,586:1 | 100% | — |
| 100 | 0.0967 | 0.0483 | 0.125 | 40:1 | 64.5% | 0.499 |
| 200 | 0.0933 | 0.0467 | 0.185 | 12:1 | 66.5% | 0.590 |
| 300 | 0.0900 | 0.0450 | 0.245 | 8.2:1 | 66.0% | 0.609 |
| 400 | 0.0867 | 0.0433 | 0.292 | 7.3:1 | 66.4% | 0.602 |
| 500 | 0.0833 | 0.0417 | 0.326 | 7.5:1 | 66.0% | 0.568 |
| 600 | 0.0800 | 0.0400 | 0.353 | 8.3:1 | 65.9% | 0.563 |
| 700 | 0.0767 | 0.0383 | 0.373 | 9.6:1 | 62.7% | 0.571 |
| 800 | 0.0733 | 0.0367 | 0.389 | 11.7:1 | 64.3% | 0.547 |
| 900 | 0.0700 | 0.0350 | 0.402 | 15.2:1 | 61.9% | 0.547 |
| 1000 | 0.0667 | 0.0333 | 0.412 | 38.9:1 | 61.2% | — |
| 1100 | 0.0633 | 0.0317 | 0.422 | 83.6:1 | 57.9% | — |
| 1200 | 0.0600 | 0.0300 | 0.430 | 74.7:1 | 55.9% | — |
| 1300 | 0.0567 | 0.0283 | 0.437 | 279.1:1 | 54.0% | — |
| 1400 | 0.0533 | 0.0267 | 0.443 | 228.4:1 | 54.6% | — |
| 1500 | 0.0500 | 0.0250 | 0.448 | 625.3:1 | 55.7% | — |
| 1600 | 0.0467 | 0.0233 | 0.452 | 684.5:1 | 53.8% | — |
| 1700 | 0.0433 | 0.0217 | 0.455 | 1,399.6:1 | 54.7% | — |
| 1800 | 0.0400 | 0.0200 | 0.458 | **2,325.6:1** | 51.2% | — |
| 1900 | 0.0367 | 0.0183 | 0.460 | **3,768.7:1** | 55.5% | — |
| 2000 | 0.0333 | 0.0167 | 0.461 | **4,611.0:1** | 53.6% | — |
| 2100 | 0.0300 | 0.0150 | 0.463 | **6,307.0:1** | 52.0% | — |
| 2200 | 0.0267 | 0.0133 | 0.464 | **7,379.8:1** | 52.9% | — |
| 2300 | 0.0233 | 0.0117 | 0.464 | **10,824.6:1** | 52.3% | — |
| 2400 | 0.0200 | 0.0100 | 0.465 | **16,768.0:1** | 52.8% | — |
| 2500 | 0.0167 | 0.0083 | 0.465 | **18,670.5:1** | 54.3% | — |
| 2600 | 0.0133 | 0.0067 | 0.465 | **17,059.5:1** | 51.3% | — |
| 2700 | 0.0100 | 0.0050 | 0.465 | **15,737.3:1** | 51.4% | — |
| 2800 | 0.0067 | 0.0033 | 0.466 | **14,740.8:1** | 52.1% | — |
| 2900 | 0.0033 | 0.0017 | 0.466 | **12,466.9:1** | 55.1% | — |
| **3000** | **0.0000** | **0.0000** | **0.466** | **2,941.7:1** | **54.1%** | **0.315** |

**Key findings (COMPLETE — 5-REGIME MODEL CONFIRMED):**
1. **EXPONENTIAL CO/CROSS GROWTH.** After the interference floor (step 400, 7.3:1),
   co/cross grows exponentially:
   - Floor: step 400 (7.3:1, c_w=0.087)
   - Step 900: 15.2 → Step 1100: 83.6 → Step 1300: 279
   - Step 1500: 625 → Step 1700: 1,400 → Step 1800: 2,326
   - Step 2400: 16,768 -> Step 2500: **18,671 (PEAK)** -> Step 2600: 17,060 -> Step 2700: 15,737
   **PEAK CONFIRMED at step 2500 (c_weight=0.017).** Two consecutive declines (0.91x, 0.92x).
   **Five-regime model:**
   (a) **Maintenance** (c_weight=0.10): 12,586:1 — inherited from P-000
   (b) **Interference** (0.087-0.097): rapid drop to 7.3:1 floor — gradient conflict
   (c) **Growth** (0.017-0.087): exponential rise 7->18,671:1 — LM amplifies weak bias
   (d) **Decay** (c_weight 0.003-0.017): gradual decline 18,671->17,060->15,737->14,741->12,467 (~6%/step)
   (e) **Cliff** (c_weight 0.003->0.000): 12,467->**2,942:1** (76% drop in last 0.3% of weight)
   **The cliff at c_weight=0 confirms: even infinitesimal contrastive signal maintains
   directional coherence. The bridge retains 2,942:1 at zero weight — 294x higher than
   FI-003 at comparable training fraction (~10:1 at step 1200 LM-only).**
   The residual 2,942:1 is the topology imprinted by the annealing; it will presumably
   decay under continued LM-only training (cf. FI-003 exponential decay pattern).
   Val loss improved throughout: 0.499 (step 100) -> 0.315 (step 3000). LM performance
   is not degraded by the annealing process.
   **Key insight: optimal Steersman weight is ~0.02, not 0.10.** Full-strength is in
   the interference regime where contrastive and LM gradients conflict.
   Peak (18,671:1) exceeds H-ch6 converged value (10K at step 10K) by 1.87x.
   Peak is 37% of P-000 converged (50,344:1). FI-003 at step 1200 was ~10:1.
2. **Signs perfectly random at co/cross 4,611:1.** Sign stability = 53.6% (random).
   **Directional coherence and sign structure are completely orthogonal.**
   The bridge can develop extreme axis alignment in magnitudes while signs
   are random. The Steersman's sign-maintenance function is entirely separate
   from the co/cross magnitude alignment function.
3. **Mechanism hypothesis.** The vanishing contrastive loss acts as an
   infinitesimal directional bias that the LM optimizer amplifies through
   continued training. Unlike the maintained regime (where full contrastive
   force fights the LM gradient to a stalemate), the weak bias allows the
   LM optimizer to find a direction that satisfies both objectives —
   creating ever-stronger axis alignment as the contrastive weight diminishes.
   **The Steersman's most efficient mode may be at near-zero weight.**
4. **Four-regime model:**
   (a) **Maintenance** (c_weight > 0.09): BD maintained, co/cross > 1000.
   (b) **Interference** (0.07 < c_weight < 0.09): destructive gradient conflict, minimum.
   (c) **Resonance** (0.04 < c_weight < 0.07): oscillatory co/cross spikes, up to 279:1.
       Signs at random. The dying Steersman creates transient directional bias
       in bridge magnitudes without maintaining sign structure.
   (d) **Free decay** (c_weight < 0.03?): expected convergence to FI-003 trajectory.
5. **Fiedler monotonically rising.** 0.10 → 0.44 in 1400 steps. Exceeds spectral
   attractor (0.09) by 5×. Bridge over-connected due to residual spectral loss.

### CW-001: Whisper-Strength Training — COMPLETE (10K steps)

**Status:** COMPLETE. Final: 13,456:1 co/cross, Fiedler 0.00046, val loss 0.4017.
**Question:** Does starting training at the FI-004-optimal weight (c_w=0.02) produce
comparable or better BD than the default (c_w=0.10)?
**Design:**
- TinyLlama 1.1B, rank 24, 6-channel RD topology, identity init
- `--initial-contrastive 0.02` (vs H-ch6's default 0.10)
- `--initial-spectral 0.01` (proportionally reduced from 0.05)
- 10K steps, all other hyperparameters identical to H-ch6
- Steersman adaptive control still active (will modulate from 0.02 base)

**Prediction:** If FI-004's optimal weight is generalizable, CW-001 should:
- Avoid the interference regime entirely (never reach 0.087-0.097)
- Reach H-ch6-comparable BD (>10,000:1) at similar or earlier step count
- Potentially exceed H-ch6's final co/cross (70,404:1) since it starts in the growth regime

**Comparison baseline:** H-ch6 at same step milestones:
- Step 1000: 7,246:1 (c_w~0.10 baseline)
- Step 5000: ~20,000:1
- Step 10000: 70,404:1

**Results:** `results/cw-001/`

**Full trajectory (21 checkpoints through step 2100):**

| Step | Co/Cross | Fiedler | c_w | Val Loss |
|------|----------|---------|-----|----------|
| 100 | 20.2:1 | 0.0073 | 0.020 | 0.477 |
| 200 | 268.1:1 | 0.0029 | 0.020 | 0.448 |
| 300 | 713.1:1 | 0.0013 | 0.018 | 0.444 |
| 400 | 1,137.4:1 | 0.0007 | 0.016 | 0.440 |
| 500 | 1,591.6:1 | 0.0004 | 0.015 | 0.439 |
| 600 | 1,412.5:1 | 0.0008 | 0.013 | 0.438 |
| 700 | 1,529.1:1 | 0.0010 | 0.012 | 0.437 |
| 800 | 1,652.1:1 | 0.0011 | 0.011 | 0.436 |
| 900 | 1,812.7:1 | 0.0011 | 0.010 | 0.435 |
| 1000 | 1,700.6:1 | 0.0013 | 0.009 | 0.434 |
| 1100 | 1,960.3:1 | 0.0013 | 0.008 | 0.434 |
| 1200 | 1,923.5:1 | 0.0013 | 0.007 | 0.434 |
| 1300 | 1,747.4:1 | 0.0014 | 0.006 | 0.433 |
| 1400 | 1,898.4:1 | 0.0015 | 0.006 | 0.432 |
| 1500 | 1,867.1:1 | 0.0016 | 0.005 | 0.431 |
| 1600 | 1,563.7:1 | 0.0018 | 0.005 | 0.430 |
| 1700 | 1,648.2:1 | 0.0018 | 0.005 | 0.429 |
| 1800 | 1,580.1:1 | 0.0017 | 0.005 | 0.428 |
| 1900 | 1,685.1:1 | 0.0016 | 0.005 | 0.427 |
| 2000 | 1,732.4:1 | 0.0016 | 0.005 | 0.426 |
| 2100 | 1,575.6:1 | 0.0018 | 0.005 | 0.426 |
| 2400 | 1,944.4:1 | 0.0017 | 0.005 | 0.425 |
| 2700 | 1,802.6:1 | 0.0018 | 0.005 | 0.423 |
| 3000 | 1,865.8:1 | 0.0016 | 0.005 | 0.422 |
| **3300** | **1,854.8:1** | **0.0018** | **0.005** | **0.421** |

34 checkpoints. Plateau fully characterized since step 500 — remaining 6,600 steps have
no scientific value but process continues (~10h to completion).

**KEY FINDING: Adaptive c_w decay defeats the whisper-strength hypothesis.**

c_w decayed from 0.020 to 0.005 by step 1500 (75% reduction), then floored at 0.005.
Co/cross plateaued at ~1,700:1 (oscillating 1,412-1,960) since step 500 — a full
1,600 steps with no trend. Meanwhile H-ch6 (c_w=0.10 default) reached 7,246:1 at
step 1000 and 70,404:1 at step 10K.

**Why the plateau:** The Steersman's adaptive control detects BD formation and reduces
c_w to avoid over-regularization. Starting at c_w=0.02 (already low), the adaptive
decay quickly pushes c_w below the threshold needed for continued BD growth. The system
stabilizes at ~1,700:1 — enough to maintain existing structure but too weak to grow it.

**Comparison with P-CTRL (FI-002, c_w=0.10):**
| Step | CW-001 (c_w=0.02 start) | P-CTRL (c_w=0.10 start) | Ratio |
|------|-------------------------|-------------------------|-------|
| 100 | 20.2:1 | 74.6:1 | 3.7× behind |
| 500 | 1,591.6:1 | 3,487:1 | 2.2× behind |
| 1000 | 1,700.6:1 | 8,085:1 | 4.8× behind |
| 1300 | 1,747.4:1 | 10,654:1 | 6.1× behind |

CW-001 falls progressively further behind as c_w decays. By step 1300, P-CTRL has
6.1× more BD despite starting with the "sub-optimal" c_w=0.10.

**Fiedler dynamics:** Fiedler rebound at step 600 (0.0004→0.0008) did NOT reconverge —
instead, Fiedler continued rising (now 0.0018) as c_w dropped. The rebound dynamics
require sufficient contrastive force for reconvergence; without it, Fiedler drifts upward.

**Implication for experiment design:** FI-004's optimal c_w≈0.02 finding applies to FIXED
weights, not adaptive. Next experiment should use `--fixed-contrastive 0.02` (bypassing
adaptive decay) to test whether the optimal weight truly produces better BD.

---

*Tracker created March 8, 2026. Major update March 13, 2026: full inventory
reconciliation. Updated March 19, 2026 (session 7 continued): FI-004 COMPLETE
(5-regime model, peak 18,671:1 at c_w=0.017). AR-001 asymmetry analysis COMPLETE.
Updated March 21, 2026: **ALL EXPERIMENTS COMPLETE.** 24C-001 COMPLETE at step
10000 — PC-001 recovery: 35,808:1 co/cross, Fiedler 0.000555. CW-001 COMPLETE —
final 13,456:1, peak 15,183:1, Fiedler 0.00046. FI-002 analysis: 100% sign
convergence, Frobenius ~10^-4, confirms init independence. O-001 co/cross:
473,622:1 (strongest BD signal). Both GPUs IDLE. Paper 4 experiment programme
finished — remaining: figures, proofread, arXiv endorsement.*
