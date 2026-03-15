# RhombiLoRA Experiment Tracker — Master Document

> **Purpose:** Single source of truth for all experiments, their status,
> success criteria, decision gates, and dependencies.
> **Last updated:** 2026-03-13 (Full inventory reconciliation post-crash recovery)
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
| Phase 1A | Qwen 7B | — | COMPLETE | 84.5% LOO SVM task fingerprinting |
| Phase 2A | Qwen 7B | — | COMPLETE | Eigenspectrum cos > 0.999 under merging |
| **Phase 3A** | **Qwen 7B** | **10K** | **COMPLETE** | **r=0.888 deviation~gap, phase transition step 400** |
| **Holly** | **Wan 2.1 14B** | **full** | **COMPLETE (WandB)** | **3.8% better loss, 9.15 GB less VRAM** |
| H-ch3 | TinyLlama 1.1B | 10K | COMPLETE | Fiedler 0.095, no BD (spectral-only) |
| H-ch4 | TinyLlama 1.1B | 10K | COMPLETE | Fiedler 0.092, no BD (spectral-only) |
| H-ch6 | TinyLlama 1.1B | 10K | COMPLETE | **70,404:1 co/cross, full BD** |
| H-ch8 | TinyLlama 1.1B | 10K | RUNNING (step 6400) | Fiedler 0.089, no BD (spectral-only) |
| T-001 | TinyLlama 1.1B | 500/10K | RESTARTING | **2,835:1 co/cross, 4+4 split (tesseract)** |
| H-ch12 | TinyLlama 1.1B | 10K | QUEUED | Channel count scaling test |
| WL-001 | TinyLlama 1.1B | 10K | QUEUED | Wrong-labels control (random pairs) |
| R-001 | TinyLlama 1.1B | 10K | QUEUED | Circular resonance topology |
| E-001 | TinyLlama 1.1B | 10K | QUEUED | Emanation architecture |

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
| **RhombiLoRA** | `u2acmrs0` | **1.5517** | **1.5447** | **66.60** | 25.5h |
| Corpus-Weighted | `n9t7op19` | 1.6453 | 1.6362 | 66.60 | 25.5h |

**Deltas:**
- **Loss:** 3.8% improvement (RhombiLoRA vs standard)
- **VRAM:** 9.15 GB less (66.60 vs 75.75)
- **Speed:** 6% faster (25.5h vs 27.1h)
- **Corpus weighting hurts** — worse than both standard and RhombiLoRA

### Crashed adamw8bit Runs

| WandB ID | Status | Runtime | Notes |
|----------|--------|---------|-------|
| `yi68ouj9` | crashed | 23.3h | Standard, VRAM reservation issue |
| `em6bc79n` | crashed | 3.5h | RhombiLoRA |
| `oy53awuf` | crashed | 1.0h | RhombiLoRA |
| `ola8rw8n` | crashed | 12.9h | RhombiLoRA |
| `ls150mpp` | crashed | 11.7h | Standard |

**Root cause:** VRAM reservation issue on 80 GB cards with adamw8bit. Minta
reports the issue was solved but doesn't recall the fix. Pods 3/4 on alvdansen
GitHub have the most up-to-date scripts but need further debugging.

### Pending

- [ ] Holly .safetensors weights from Minta (via alvdansen-labs GitHub)
- [ ] Timothy's access to alvdansen-labs GitHub org (timm156 invite sent)
- [ ] Minta observed RhombiLoRA "converging and training faster" — qualitative
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
- **Key findings:** 84.5% LOO SVM (Q-proj only, 1,008 params). Code most
  distinctive (97.3%). Mann-Whitney p = 0.000000.

### Phase 2A: Bridge-Level Merging (MIXED)
- **Results:** `results/bridge_merge/`
- **Key findings:** alpaca↔code R²=0.956 (linear), alpaca↔math R²=0.735
  (non-linear), code↔math R²=0.823 (non-linear). All pass random baseline.
  Eigenspectrum cos > 0.999. Safe mixing: up to 10%.

---

## Channel Ablation Series — IN PROGRESS (Mar 13-15)

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
| H-ch8 | 8 | spectral | ~0.411 | N/A | 0.0889 | 0.214 | **NO** | Step 6400/10K |
| T-001 | 8 | tesseract contrastive | 0.439 | **2,835** | 0.00021 | 0.182 | **YES (4+4)** | 500/10K (restarting) |
| H-ch12 | 12 | spectral | — | — | — | — | — | QUEUED |

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

### Remaining Runs

- **H-ch8 (spectral):** Step 6400/10K, ETA ~4.7 hours. Expected: same as n=3,4.
- **H-ch12 (spectral):** Auto-launches after H-ch8. Tests whether more channels
  create emergent structure. Expected: same baseline.
- **T-001-full (tesseract contrastive):** Restarted 2026-03-15 02:55 AM.
  Full 10,000 steps. Previous run terminated early (Windows sleep).

### Auto-Chain (Hermes)

After H-ch8 → H-ch12 → WL-001 (wrong-labels) → R-001 (resonance) → E-001 (emanation).
Watcher PID 520682 on Hermes monitors H-ch8 PID 397169.

---

## Queued Experiments — Auto-Launching

### WL-001: Wrong-Labels Control (Priority 6)
- **n=6, random partition into 3 pairs** (no geometric prior)
- **Hypothesis:** If WL-001 produces BD, geometry is irrelevant (any partition works).
  If NULL, RD geometry IS special.
- **Prediction:** NULL — random pairs should produce distributed structure like spectral-only.
- **Location:** `results/wrong-labels/WL-001/`

### R-001: Circular Resonance (Priority 7)
- **n=6, prime-threading topology** (Sophie Germain, consecutive primes, mod-6 residue)
- **Hypothesis:** Prime-theoretic relationships add structure beyond geometry.
- **Prediction:** BD emerges (it's still a valid partition), but co/cross may differ
  from RD contrastive — the SHAPE of the BD could vary.
- **Location:** `results/resonance/R-001/`

### E-001: Emanation Architecture (Priority 8)
- **n=6, master bridge + per-layer offsets** (Plotinus-inspired hierarchy)
- **Hypothesis:** Sharing a master bridge across layers creates coherence.
- **Prediction:** More uniform BD across layers. Fiedler std should be lower.
- **Location:** `results/emanation/E-001/`

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
└── T-001-full/              T-001: Tesseract restart — RUNNING (10K target)
```

**On Hermes (`/home/timm156/rhombic/results/`):**
```
channel-ablation/
├── H-ch3/   n=3 spectral — COMPLETE
├── H-ch4/   n=4 spectral — COMPLETE
├── H-ch6/   n=6 RD contrastive — COMPLETE
├── H-ch8/   n=8 spectral — RUNNING (step 6400)
└── H-ch12/  n=12 spectral — QUEUED
wrong-labels/WL-001/   — QUEUED
resonance/R-001/       — QUEUED
emanation/E-001/       — QUEUED
```

---

## Key Numbers (Verified Mar 12, 2026 + Channel Ablation Mar 15)

| Claim | Value | Source |
|-------|-------|--------|
| FCC > cubic Fiedler (1.5B) | 4.6× | Exp 1 |
| FCC > cubic Fiedler (7B) | 1.73× | Exp 2 |
| Task fingerprint LOO SVM | 84.5% (Q-proj) | Phase 1A |
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
| **Tesseract contrastive co/cross (n=8)** | **2,835:1 @ step 500** | T-001 (partial) |
| **Tesseract 4+4 split ratio** | **~800:1** (0.17 / 0.00021) | T-001 eigenvalues |

---

*Tracker created March 8, 2026. Major update March 13, 2026: full inventory
reconciliation. Exp 3.0 (breakthrough), Phase 3A (complete), Exp 2.7 (null),
Holly Battery (verified), Paper 3 (written), all section statuses corrected.
20 learnings in LEARNINGS.md. 25,656 bridge checkpoint files on disk.*
