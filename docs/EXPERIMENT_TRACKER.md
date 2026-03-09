# RhombiLoRA Experiment Tracker — Master Document

> **Purpose:** Single source of truth for all experiments, their status,
> success criteria, decision gates, and dependencies.
> **Last updated:** 2026-03-08 (Phases 0/1A/2A COMPLETE, cross-phase synthesis done, 3A training)
> **Notion mirror:** https://www.notion.so/31d6930d2e1181bab622d6b3a9720b24

---

## Exp 2.5: Geometric Data (7B) — COMPLETE, NULL ON DIRECTION

**Status:** COMPLETE — NULL on directionality, positive on connectivity
**Pod:** `7aeyf5vr1vifxl` (4090, $0.59/hr) — TERMINATE
**Results local:** `results/exp2_5/geometric_rhombi_fcc_r24/`
**Comparison report:** `results/exp2_5/COMPARISON_REPORT.md`

### Checkpoint Log

| Step | Co/Cross | Fiedler | Val Loss | Wall Time | Notes |
|------|----------|---------|----------|-----------|-------|
| 0    | 1.000    | init    | —        | 0h        | Identity bridge |
| 1000 | 0.992    | 0.029   | 0.0199*  | 0.79h     | Higher than Exp 2 (0.929/0.018) |
| 2000 | **1.003**| **0.030**| 0.0193   | 1.58h     | FLAT — Exp 2 was 1.023 here |
| 3000 | **1.002**| **0.030**| 0.0194   | 2.36h     | NULL on direction. 2x Fiedler. |

### Step 2000 Analysis

**Co/Cross = 1.003** — essentially identity. Exp 2 was HIGHER (1.023).
**Fiedler = 0.030** — 1.67x Exp 2 (0.017). Bridge learning more structure overall.

Trajectory: Exp 2 accelerating (+0.094/1K steps), Exp 2.5 flat (+0.011/1K steps).
Geometric data teaches CONNECTIVITY but not DIRECTIONALITY.

Trending toward marginal/null branch of decision tree.

### Step 3000 Final Results — NULL ON DIRECTION

**Co/Cross = 1.002** — dead flat. No directional signal.
**Fiedler = 0.030** — 2× Exp 2 (0.015). More connectivity, not more directionality.

The geometric dataset teaches the bridge to be MORE CONNECTED but does NOT
create co-planar vs cross-planar preference. Exp 2's isotropic Alpaca data
produced a STRONGER transient peak (1.091) than geometric data's best (1.003).

**Finding:** The prompt-level co-planar bias (95% perspective pairing) does not
translate to channel-level co-activation patterns through 28 transformer layers.
Channel assignment is arbitrary — the bridge has no way to map "analytical +
empirical" to channels 0-1 vs channels 2-3.

**Decision:** NULL BRANCH activated.

**Next steps (from decision tree):**
1. Contrastive bridge pre-training (L_bridge = -λ * (mean(|B[co]|) - mean(|B[cross]|)))
2. Separate bridge learning rate (try 10× higher than base, not lower)
3. Input-dependent bridge (dynamic routing per-example)
4. Explicit channel supervision (assign perspectives to channels during training)
5. Publish the null result WITH the Fiedler finding (connectivity without direction)

**The Fiedler finding is itself publishable:** geometric data produces 2× more
bridge connectivity than isotropic data, even without directional preference.
This means the bridge IS responding to data structure — just not in the
co-planar/cross-planar axis we predicted.

*Val loss meaningless (train/val overlap bug). Bridge metrics are the comparison.

### Full Karkada Spectral Analysis (4 tools, 112 bridges each)

| Metric | Exp 2 (Alpaca 10K) | Exp 2.5 (Geometric 3K) | Better |
|--------|-------------------|----------------------|--------|
| **Co/Cross** | **1.019** | **1.002** | Alpaca |
| **Permutation p** | 0.332 (ns) | 0.474 (ns) | Alpaca |
| **Gram distance** | 0.921 | 0.921 | Tie |
| **Eigenvalue spread** | 0.107 | 0.062 | Alpaca |
| **Perturbation retention** | 68.8% | 69.7% | Tie |
| **Fiedler (final)** | 0.040 | 0.030 | Alpaca |
| **Deviation (final)** | 0.199 | 0.058 | Alpaca |

**Alpaca outperforms geometric data on every directional metric.** The step 1000
Fiedler comparison (0.029 vs 0.018) was misleading — Exp 2 at step 10K reached
0.040. The geometric data produced LESS bridge structure overall, not more.

**What IS identical:** Perturbation retention (~69%) and Gram distance (0.921).
Bridge structure is distributed (not concentrated in co-planar pairs) regardless
of data type. The architecture's structural properties are data-independent.

**Root cause:** Prompt-level perspective pairing (95% co-planar) cannot create
channel-level co-activation preference because channel assignment is arbitrary.
The bridge has no mechanism to associate "analytical + empirical" with channels
0-1 versus "creative + systematic" with channels 2-3. The 28 transformer layers
between prompt structure and bridge weights erase the prompt-level signal.

### Success Criteria

| Metric | Pass | Strong Pass | Exp 2 Baseline |
|--------|------|-------------|----------------|
| Co/Cross ratio @ 3K | > 1.10 | > 1.20 | 1.091 (peak, transient) |
| Permutation p | < 0.05 | < 0.01 | 0.332 |
| Gram distance | < 0.85 | < 0.70 | 0.921 |
| Eigenvalue spread | > 0.30 | > 0.50 | 0.107 |
| Perturbation retention | > 50% | > 60% | 68.8% |

**Primary gate:** Co/Cross > 1.10 AND p < 0.05

### The Transient Peak Hypothesis

Exp 2 co/cross trajectory under isotropic (Alpaca) data:
```
Step 1000: 0.929 → Step 2000: 1.023 → Step 3000: 1.091 (PEAK) → Step 10000: 1.019
```

Three phases: Exploration → Transient bias → Regularization (washed out).

**Exp 2.5 must beat this trajectory at step 3000.** If co/cross is sustained
or growing at step 3000 (vs Exp 2's peak-then-decay), the geometric data
creates a *stable* directional signal, not a transient artifact.

### Decision Tree (Post Exp 2.5)

```
Exp 2.5 Results
├── co/cross > 1.10 AND p < 0.05 (PASS)
│   ├── Run FCC-only sweep: 3 dataset configs × 10K steps
│   │   ├── Config A: Multi-perspective only (10K examples)
│   │   ├── Config B: Cross-domain transit only (8K examples)
│   │   └── Config C: Full geometric (23K, current dataset)
│   ├── Identify which component drives the signal
│   └── Proceed to Exp 3A (Flimmer integration)
│
├── co/cross 1.05–1.10 (MARGINAL)
│   ├── Extend training to 10K steps (does signal sustain?)
│   ├── Increase co-planar bias from 95% to 99%
│   ├── Separate bridge LR (try 0.5× and 0.1× of base LR)
│   └── If still marginal → contrastive bridge pre-training
│
└── co/cross ≈ 1.0 (NULL)
    ├── Pre-train bridge with contrastive loss
    │   L_bridge = -λ * (mean(|B[co]|) - mean(|B[cross]|))
    ├── Input-dependent bridge (dynamic routing per-example)
    ├── Reduce bridge LR significantly (1/10th of base)
    └── Publish null result with transient peak analysis
```

---

## Completed Experiments

### Exp 1: 1.5B Scale Proof of Concept
- **Model:** Qwen2.5-1.5B-Instruct
- **Result:** Bridge learns, FCC > cubic Fiedler. Proof that the architecture works.
- **Location:** `results/exp1/`

### Exp 2: 7B Scale Baseline (Alpaca)
- **Model:** Qwen2.5-7B-Instruct, 10K steps, Alpaca-cleaned
- **Key results:**
  - FCC Fiedler: 0.0401, Cubic: 0.0231 → **1.73× ratio**
  - Co/Cross: 1.019 → weak directional signal (isotropic data expected)
  - Permutation p: 0.332 → not significant
  - Gram distance: 0.921 → far from RD coupling
  - Transient peak at step 3000 (1.091) → decays
- **Location:** `results/exp2/`
- **Karkada tools:** All 4 validated on 112 FCC bridges

### Exp 2.5: 7B Geometric Data (COMPLETE — NULL)
- **Model:** Qwen2.5-7B-Instruct, 3K steps, geometric dataset (23K examples)
- **Key results:**
  - Co/Cross: 1.002 (NULL — no directional signal)
  - Permutation p: 0.474 (ns) — worse than Alpaca
  - Fiedler: 0.030 (0.76× Exp 2 final, despite early lead at step 1K)
  - Gram distance: 0.921 (identical to Exp 2)
  - Perturbation retention: 69.7% (identical to Exp 2)
- **Finding:** Prompt-level co-planar bias does not translate to channel-level
  co-activation through 28 transformer layers. Channel assignment is arbitrary.
- **Publishable sub-finding:** Bridge structure is DISTRIBUTED regardless of
  data type (~69% retention under co-planar zeroing). This is an architectural
  property, not a data-dependent one.
- **Location:** `results/exp2_5/`
- **Full comparison:** `results/exp2_5/COMPARISON_REPORT.md`

---

## Utility Interrogation — Active

### Phase 0: Bridge Anatomy (COMPLETE — FREE)
- **Status:** COMPLETE
- **Results:** `results/exp2/bridge_anatomy.md`
- **Key findings:**
  - **Module type differentiates bridges (p = 0.0008).** Q-projections develop 40% more
    coupling than V-projections (Fiedler 0.050 vs 0.036). ANOVA F = 6.022.
  - **Deviation varies by layer depth (p = 0.001).** Later layers diverge more from
    identity. Fiedler itself is depth-invariant (r = 0.03, p = 0.87).
  - **Q-V closer than Q-K (p = 0.042).** Opposite to naive hypothesis. Bridge tracks
    information flow (Q queries, V supplies), not similarity computation (Q-K).
  - **Bridges reorganize during training.** Early-late Fiedler correlation r = 0.099
    (p = 0.30). Step 1K structure doesn't predict step 10K structure.
  - **Fiedler uniform across layers (p = 0.74).** ANOVA F = 0.799. No layer preference.
  - **CV = 0.408** — moderate variation across adapters, below 0.5 threshold but
    Top adapter (L2 q_proj, 0.099) is 9× bottom (L1 v_proj, 0.011).
- **Implication:** Bridge is sensitive to WHERE in the attention head it sits (module type),
  which supports Phase 1A — if it differentiates by position, it may differentiate by task.

### Phase 1A: Task Fingerprints — PASS (STRONG SIGNAL)
- **Status:** COMPLETE — all three tasks analyzed, both decision criteria met
  - Task A: Alpaca (general) — Exp 2 bridges, 112 adapters, 10K steps
  - Task B: Code (CodeAlpaca-20k) — local RTX 6000 Ada, 2K steps (1500 at analysis)
  - Task C: Math (GSM8K) — RunPod RTX 4090, 2K steps, COMPLETE, pod terminated
- **Results (Mar 8):**
  - **Mann-Whitney U:** p = 0.000000 — between-task > within-task distances. **PASS.**
  - **LOO SVM accuracy: 73.5%** (chance = 33.3%) — task identity from 36 parameters. **PASS.**
  - Confusion: Alpaca 93.8%, Code 97.3%, Math 70.5% (confused w/ alpaca & code)
  - **Q-proj best discriminator** (distance 0.162) — consistent with Phase 0 L-011
  - Fiedler by task: Alpaca 0.040, Math 0.028, Code 0.017
  - Code vs Math distance (0.066) << Alpaca vs Code (0.190) or Alpaca vs Math (0.171)
  - Tasks change coupling MAGNITUDE, not STRUCTURE (eigenspectrum cosine ≈ 1.0)
- **Reports:** `results/fingerprints/TASK_FINGERPRINT_REPORT.md`,
  `results/fingerprints/EARLY_SIGNAL_REPORT.md`
- **Decision gate:** PASS → proceed to Phase 2A (bridge-level merging)
- **Paper 3 headline:** "36-parameter task fingerprint at 73.5% LOO accuracy"

### Phase 2A: Bridge-Level Merging — COMPLETE (MIXED)
- **Status:** COMPLETE — smooth interpolation confirmed, non-linearity in 2/3 pairs
- **Results (Mar 8):**
  - **alpaca↔code:** R²=0.956 (LINEAR). Fiedler trajectory smooth, no phase transitions.
    Eigenspectrum cos > 0.9997 at all α values. Most affected: q_proj (sensitivity 0.435).
  - **alpaca↔math:** R²=0.735 (NON-LINEAR). Curvature at intermediate α, max residual 0.00437.
    Most affected: o_proj (sensitivity 0.395).
  - **code↔math:** R²=0.823 (NON-LINEAR). Fiedler DIP at α≈0.2 (from 0.017→0.015→recovery).
    Most affected: q_proj (sensitivity 0.241), but overall low sensitivity (v_proj 0.023).
  - **All 3 pairs PASS random baseline** — task merge preserves eigenspectrum structure
    (cos > 0.999) where random merge destroys it (cos ≈ 0.5).
  - Structure retention at 10% contamination: cos = 1.0000 in all pairs.
- **Report:** `results/bridge_merge/MERGE_REPORT.md`
- **Finding:** Bridge interpolation preserves task structure, but is NOT uniformly linear.
  Some task combinations exhibit nonlinear reorganization at intermediate α.
  The 36-parameter bridge is a viable merge target — merge there, not at the full weight level.
- **Paper 3 angle:** "Adapter composition via 36-parameter bridge interpolation"

### Phase 3A: Overfitting Diagnostic (TRAINING — RunPod)
- **Status:** TRAINING on RunPod RTX 4090 (`w1ri4p5ggpkuzc`)
- **Design:** 500 train / 500 val split from Alpaca. 10K steps, checkpoint every 100.
  Track Fiedler + deviation alongside train-val loss gap.
- **Script:** `scripts/train_overfit_diagnostic.py` (514 lines)
- **Pass:** Bridge metric correlates with train-val gap (r > 0.7) or shows phase transition.
- **Cost:** ~$4 RunPod

---

## Planned Experiments — NULL Branch (Deprioritized pending Utility Interrogation)

### Exp 2.6: Contrastive Bridge Pre-Training
- **Prerequisite:** None — implements null branch option 1
- **Design:** Add contrastive loss to bridge:
  `L_bridge = -λ * (mean(|B[co]|) - mean(|B[cross]|))`
  Train bridge with explicit directional pressure, then fine-tune normally.
- **Hypothesis:** If the bridge can't discover direction from data alone,
  TELL it which pairs are co-planar during a pre-training phase, then see
  if the preference survives normal fine-tuning.
- **Model:** Qwen2.5-7B-Instruct, Alpaca-cleaned (use isotropic data to
  isolate the contrastive loss effect)
- **Compute:** RunPod 4090 (~$4) or local RTX 6000 Ada
- **Implementation needed:** Contrastive bridge pre-training loss in training script

### Exp 2.7: Separate Bridge Learning Rate
- **Prerequisite:** None — implements null branch option 2
- **Design:** Bridge LR = 10× base LR. The bridge has 36 parameters vs 15M for
  A/B; at the same LR, bridge gradients may be too small relative to noise.
- **Model:** Qwen2.5-7B-Instruct, Alpaca-cleaned
- **Compute:** RunPod 4090 (~$4) or local
- **Implementation needed:** Separate optimizer group for bridge parameters

### Exp 2.8: Input-Dependent Bridge (Dynamic Routing)
- **Prerequisite:** None — implements null branch option 3
- **Design:** Replace static B with B(x) = σ(W_gate × h) ⊙ B_base, where
  h is a pooled hidden state. The bridge activates differently per example.
- **Hypothesis:** A static bridge must compromise across all inputs. A dynamic
  bridge can route specific content through co-planar channels.
- **Model:** Qwen2.5-7B-Instruct
- **Compute:** RunPod 4090 (~$8, longer training needed)
- **Implementation needed:** InputDependentBridge module in rhombic.nn

### Exp 2.5 Follow-up: FCC-Only Sweep — DEPRIORITIZED
- **Prerequisite:** Exp 2.5 passes (IT DID NOT)
- **Status:** Deprioritized. The dataset is not the bottleneck — the bridge
  mechanism itself needs modification before dataset sweeps are meaningful.

### Exp 3A-3C: Deferred until Bridge Direction Problem Resolved
- **Prerequisite:** One of Exp 2.6/2.7/2.8 produces co/cross > 1.10 with p < 0.05
- **Status:** Design complete, waiting on bridge mechanism validation
- **Location:** `rhombic/docs/EXP3_EXPANDED_PLAN.md`

---

## Infrastructure Checklist

### Tools — Ready
- [x] `rhombic/scripts/train_exp2_5.py` — training script
- [x] `rhombic/scripts/compare_exp2_exp25.py` — full comparison (4 spectral tools)
- [x] `rhombic/scripts/peek_bridges.py` — early-peek from in-progress training
- [x] `rhombic/scripts/pull_runpod_results.sh` — download + comparison pipeline
- [x] `rhombic/scripts/generate_geometric_dataset.py` — dataset generator
- [x] `rhombic/data/geometric/geometric_combined.json` — 23K examples
- [x] `rhombic/docs/gradient_analysis.md` — Paper 3 §3
- [x] `rhombic/docs/competitive_landscape.md` — Paper 3 §5

### Tools — Needed
- [ ] Val split fix deployed to RunPod (local fix only)
- [ ] Flimmer integration (LoRAState topology field)
- [ ] Bridge-specific learning rate support
- [ ] Contrastive bridge pre-training loss
- [ ] Input-dependent bridge architecture

### Bug Tracker
| Bug | Severity | Status | Notes |
|-----|----------|--------|-------|
| Val split overlap | Low | Fixed locally | Bridge metrics unaffected |
| Eff rank = 0.00 | Low | Known | Gradient timing bug, not important |
| `_bridge_fiedler` adapter | Medium | Fixed | Was calling fiedler_value(B) incorrectly |

---

## Paper 3 Section Status

| Section | Status | Source |
|---------|--------|--------|
| §1 Introduction | Not started | — |
| §2 Architecture | Not started | `rhombic/nn/rhombi_lora.py` |
| §3 Theoretical Grounding | **DONE** | `rhombic/docs/gradient_analysis.md` |
| §4 Experiments | **OUTLINED** | `rhombic/docs/paper3_section4_outline.md` |
| §5 Related Work | **DONE** | `rhombic/docs/competitive_landscape.md` |
| §6 Discussion | Not started | — |

### §4 Data Readiness

| Subsection | Status | Key Figures/Tables |
|-----------|--------|-------------------|
| §4.1 Setup | Ready | Table 1 (training configs) |
| §4.2 Phase 0 | Ready | Figure 1 (Fiedler evolution), Table 2 (per-module ANOVA) |
| §4.3 Phase 1A | Ready | Figure 2-4 (distances, confusion, heatmap), Table 3-4 |
| §4.4 Phase 2A | Ready | Figure 5-6 (interpolation, sensitivity), Table 5-6 |
| §4.5 Cross-Phase | Ready | Table 7, Figure 7 (schematic) |
| §4.6 Phase 3A | PENDING | ~7h remaining on RunPod |
| §4.7 Ablations | Partial | Random baseline done, step comparison pending |

---

## Monitoring Commands

```bash
# Check training status
ssh -p 44315 root@213.173.111.6 "tail -20 /workspace/training.log"

# Check available checkpoints
ssh -p 44315 root@213.173.111.6 "ls /workspace/rhombic/results/exp2_5/geometric_rhombi_fcc_r24/bridge_step*.npy | grep -o 'step[0-9]*' | sort -un"

# Early peek at bridge metrics
python scripts/peek_bridges.py 213.173.111.6 44315

# Download final results + run comparison
bash rhombic/scripts/pull_runpod_results.sh 213.173.111.6 44315

# Full spectral comparison
python scripts/compare_exp2_exp25.py --deep
```

---

## Cost Tracking

| Item | Cost | Date | Notes |
|------|------|------|-------|
| RunPod 4090 (Exp 2.5) | ~$1.40 (2.36h) | Mar 8 | $0.59/hr, COMPLETE |
| RTX 6000 Ada (Exp 2) | Electric only | Mar 7-8 | Local GPU |
| RunPod 4090 (Phase 1A math) | ~$0.95 (1.60h) | Mar 8 | COMPLETE — terminated |
| RunPod 4090 (Phase 3A overfit) | ~$4 (est) | Mar 8 | TRAINING — 21%, ETA ~7h |
| RunPod 4090 (Exp 2.6, est) | ~$4 | Pending | Contrastive bridge |
| RunPod 4090 (Exp 2.7, est) | ~$4 | Pending | Separate bridge LR |
| RunPod 4090 (Exp 2.8, est) | ~$8 | Pending | Dynamic bridge |

---

*Tracker created March 8, 2026. Updated as experiments progress.*
