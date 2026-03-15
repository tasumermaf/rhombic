# Paper 3 Update Notes — March 13, 2026

## Status

Paper 3 ("The Learnable Bridge") covers Experiments 1-3 (Qwen 7B non-cybernetic).
The cybernetic training experiments (Steersman) produce results that
fundamentally revise several findings. This document tracks what needs updating.

---

## UPDATE 1: Block-Diagonal Discovery (NEW SECTION)

**Affects:** §4.3 (Direction vs Connectivity), §4.4 (Contrastive Pre-Training),
§6.6 (Connection to Lattice Topology), Conclusion

**Finding:** The Steersman (cybernetic feedback loop with contrastive + spectral
losses maintained throughout training) causes 100% of 6×6 bridges to self-organize
into 3 independent 2×2 co-planar blocks aligned with the RD's coordinate axes.

**Evidence (27,748 bridges across 10 experiments, 3 model scales, 272 steps):**

| Experiment | Model | Cybernetic | Bridges | Block-Diagonal | Co/Cross Ratio |
|-----------|-------|-----------|---------|---------------|---------------|
| exp3 | Qwen 7B | Yes | 14,560 | **100%** | **18,248:1** |
| exp3_tinyllama | TinyLlama 1.1B | Yes | 8,888 | **100%** | **37,929:1** |
| C-001 (identity) | TinyLlama 1.1B | Yes | 3,608 | **100%** | **5,438:1** |
| C-002 (geometric) | TinyLlama 1.1B | Yes | 3,520 | **100%** | **10,181:1** |
| exp1 learnable | Qwen 7B | No | 56 | 0% | 1:1 |
| exp2 FCC | Qwen 7B | No | 112 | 0% | 1:1 |
| exp2.5 geometric | Qwen 7B | No | 112 | 1% | 1:1 |
| fingerprint code | Qwen 7B | No | 112 | 0% | 1:1 |
| fingerprint math | Qwen 7B | No | 112 | 1% | 1:1 |
| Holly Battery | Wan 2.1 14B | No | 66 | 0% | 1.07:1 |

**Temporal emergence (3 experiments, 27,056 checkpoint bridges):**
- All three cybernetic experiments lock to 100% block-diagonal by step 200
- Qwen 7B: 0:1 → 14:1 (step 100) → 240:1 (step 200) → **18,248:1** (step 12,900)
- TinyLlama: 0:1 → 29:1 (step 100) → **37,929:1** (step 10,000)
- C-001: 0:1 → 10:1 (step 100) → **5,438:1** (step 4,000, still growing)
- Cross-planar coupling saturates < 0.001 across all experiments

**Axis symmetry:** All three co-planar axes develop within <2% of identical
coupling strength. No preferred axis. x-axis (0,1), y-axis (2,3), z-axis (4,5)
treated symmetrically by the Steersman.

**Coupling polarity:** ~50/50 positive/negative signs across layers. The Steersman
constrains TOPOLOGY (which channels couple) but leaves POLARITY to the optimizer.

**Per-module:** o_proj weakest (19,080:1 Qwen), k_proj/v_proj strongest (44,000:1).
Coupling remarkably uniform across layers (±5% in Qwen 7B, ±3% in TinyLlama).

**Pair identity is universal:** (0,1), (2,3), (4,5) in 100% of bridges,
100% of layers, both init modes (identity/geometric), both model scales.

**Revision to §4.3:** The "null result on direction" is only null WITHOUT
the Steersman. With the Steersman, direction emerges perfectly. The section
should present both: non-cybernetic = null, cybernetic = 100% block-diagonal.

**Revision to §4.4:** The contrastive warmup (500 steps → unsupervised) is
superseded. The Steersman maintains contrastive + spectral losses throughout,
producing PERSISTENT (not decaying) block-diagonal structure. The 2,660:1 →
16:1 decay story is replaced by 100% block-diagonal maintenance.

**Key interpretive revision:** The Steersman is an ARCHITECTURAL SELECTOR.
It discovers the 3-axis coordinate structure from the 6-face RD geometry.
Without it, the bridge is a generic coupling matrix.

**Analysis scripts:**
- `scripts/analyze_bridge_structure.py` — Cross-experiment final-state analysis
- `scripts/analyze_temporal_emergence.py` — Checkpoint-by-checkpoint emergence
- `scripts/analyze_per_layer.py` — Per-layer and per-module coupling + heatmaps
- `scripts/analyze_holly_bridges.py` — Holly Battery implicit bridge reconstruction
- `scripts/plot_temporal_emergence.py` — Generate temporal emergence figures

**Reports:**
- `results/BRIDGE_STRUCTURE_ANALYSIS.md` — Full cross-experiment analysis
- `results/BRIDGE_BLOCK_DIAGONAL_FINDING.md` — Core finding (27,748 bridges)
- `results/PAPER3_FIGURE_INVENTORY.md` — Publication figure + data inventory

**Figures (6 total):**
1. `results/fig_temporal_emergence.png/.pdf` — Two-panel main figure
2. `results/fig_early_emergence.png` — First 1000 steps zoom
3. `results/fig_heatmap_comparison.png` — 4-panel cybernetic vs non-cybernetic
4. `results/fig_heatmap_cybernetic.png` — Single cybernetic bridge
5. `results/fig_per_layer_coupling.png` — Per-layer comparison
6. `results/fig_init_convergence.png` — Identity vs geometric init convergence

---

## UPDATE 2: Corrected Co/Cross Ratios

**Affects:** §4.2 (per-module Table 2, Co/Cross column), §4.3, §6.6

**Current paper values (Table 2, non-cybernetic Qwen 7B):**
- q_proj: 1.050 ± 0.593
- k_proj: 1.074 ± 0.528
- o_proj: 1.096 ± 0.411
- v_proj: 1.061 ± 0.448

**These are correct for non-cybernetic training.** They should remain BUT be
contrasted with the cybernetic values:

**Cybernetic co/cross ratios (from VERIFIED_FINDINGS_2026_03_12.md):**
- Qwen 7B cybernetic (exp3): co/cross = 22,477:1
- TinyLlama 1.1B cybernetic (exp3_tinyllama): co/cross = 47,145:1
- Fiedler converges to ~0.10 (scale-invariant across 1.1B/7B/14B)

---

## UPDATE 3: Holly Battery Results (if included in this paper)

**Decision needed:** Holly Battery is a Wan 2.1 14B video diffusion result.
May belong in a separate paper or in this one as a section on scale.

**Key numbers:**
- 3.8% better final loss (1.552 → 1.493 val loss)
- 9.15 GB less VRAM (37.59 GB → 28.44 GB)
- 6% faster training (3.12 → 2.94 s/step)
- 50% smaller checkpoint (2.1 GB → 1.05 GB)
- Domain: video diffusion (Wan 2.1 14B), NOT text (Qwen)

---

## UPDATE 4: Bridge Init Mode Is Cosmetic

**Affects:** §4.1 (may need note), new section on corpus baselines

**Finding (CONFIRMED — both runs COMPLETE at 4000 steps):**

| Metric | C-001 (identity) | C-002 (geometric) | Delta |
|--------|------------------|-------------------|-------|
| Final val loss | 0.4178 | 0.4177 | -0.0002 |
| Final co/cross | 10,118:1 | 10,181:1 | +62 |
| Final BD% | 100% | 100% | 0 |
| Final co-planar | 0.4801 | 0.4938 | +0.014 |
| Final cross-planar | 0.000088 | 0.000092 | negligible |

The geometric init starts slightly biased (2:1 at step 0 vs 0:0 for identity)
but converges to identical topology by step ~1000. At step 4000, the two
experiments are indistinguishable. **Bridge initialization is cosmetic — the
Steersman overwrites any initial structure.** See `results/fig_init_convergence.png`.

Interesting detail: geometric init has LOWER ratio than identity through
steps 200-900 (~22:1 vs ~189:1 at step 200). The pre-built structure appears
to briefly interfere with the Steersman's self-organization before being
overwritten. By step 1000 both converge.

---

## UPDATE 5: Limitations Section Revisions

**Current limitation: "Single model family"**
→ NOW partially addressed. Two model families (TinyLlama 1.1B, Qwen 7B).
  Block-diagonal structure is scale-invariant. Fiedler ~0.10 at both scales.

**Current limitation: "Fixed bridge rank"**
→ NOW partially addressed. Channel ablation (H1-H5) testing n={3,4,6,8,12}.
  Results pending (~Saturday). Prediction: n=3 wins on efficiency because
  Steersman forces 3-block structure.

**Current limitation: "No generative evaluation"**
→ STILL OPEN. Holly results provide partial evidence but for a different
  domain (video diffusion, not text).

---

## UPDATE 6: New Findings Table for Abstract

**Current abstract claims:**
1. Bridge encodes task identity (82.1% → 84.5%) ✓
2. FCC learns 4.6× more coupling than cubic ✓
3. Bridge interpolation preserves eigenspectrum ✓
4. Contrastive pre-training installs persistent direction (2,600× → 16×)

**Revision for claim 4:**
"Cybernetic training (contrastive + spectral feedback maintained throughout)
produces 100% block-diagonal bridge structure aligned with the RD's 3-axis
coordinate geometry, across both TinyLlama 1.1B and Qwen 7B (co/cross
ratio 16-67:1, 692 bridges examined). This structural self-organization
requires the cybernetic feedback loop; without it, bridges fill all DOF
uniformly (ratio ~1:1, 0% block-diagonal, 504 bridges)."

---

## Experiment Inventory for Paper

| Exp | Model | Training | Steps | Status | Key Finding |
|-----|-------|----------|-------|--------|-------------|
| 1a-e | Qwen 1.5B | Non-cyber | 2K | Done | Bridge learns task identity |
| 2 | Qwen 7B | Non-cyber | 10K | Done | FCC 4.6× more coupling |
| 2.5 | Qwen 7B | Non-cyber, geo data | 3K | Done | Direction null w/o Steersman |
| 2.6 | Qwen 7B | Contrastive warmup | 6.5K | Done | Warmup insufficient |
| 2.7 | Qwen 7B | Non-cyber, high LR | 3K | Done | LR null |
| 3 | Qwen 7B | Cybernetic | 12.9K | **Done** | 100% BD, 18,248:1 |
| 3_tinyllama | TinyLlama 1.1B | Cybernetic | 10K | **Done** | 100% BD, 37,929:1 |
| C-001 | TinyLlama 1.1B | Cyber, identity init | 4K | **Done** | Init cosmetic |
| C-002 | TinyLlama 1.1B | Cyber, geometric init | 4K | **Done** | Converges to C-001 |
| C-003 | TinyLlama 1.1B | Cyber, corpus-coupled | 10K | **Running** | Stream B test |
| H1 | TinyLlama 1.1B | Cyber, n=3 | 10K | Running (Hermes) | Channel ablation |
| H2-H5 | TinyLlama 1.1B | Cyber, n={4,6,8,12} | 10K | Queued (Hermes) | Channel ablation |
| Holly | Wan 2.1 14B | Non-cyber | 10 ep | **Done** | 1.07:1, no BD at 14B |

### Total Evidence

| Category | Experiments | Bridges | Models | Scales |
|----------|------------|---------|--------|--------|
| Cybernetic | 4 | 30,576 | 2 | 1.1B, 7B |
| Non-cybernetic | 6 | 570 | 3 | 1.1B, 7B, 14B |
| **Total** | **10** | **31,146** | **3** | **1.1B–14B** |

---

## UPDATE 7: C-003 Corpus-Coupled Bridge (CONFIRMED)

**Status:** Running (step 200+/10000). **100% BD at step 200 CONFIRMED.**
**Init:** corpus_coupled (hexagram × geometric × thread_density).
**Steersman:** default.

**Key finding:** Corpus-coupled init assigns highest coupling to CROSS-planar
pairs (I Ching complementary trigrams: (0,3), (1,4), (2,5)). The init
actively OPPOSES the RD co-planar structure. Despite this:

| Step | Co/Cross | BD% | Note |
|------|----------|-----|------|
| 0 | 1.36:1 | 0% | Cross-planar dominant (I Ching) |
| 100 | 5.1:1 | 70% | Steersman overwriting |
| 200 | 19.4:1 | 100% | **Steersman wins** |

**Conclusion:** The Steersman overwrites ANY initialization:
- Identity (no bias) → 100% BD by step 100
- Geometric (weak co-planar bias) → 100% BD by step 200
- Corpus-coupled (ANTI co-planar bias) → 100% BD by step 200

Init is cosmetic regardless of whether it agrees or opposes the
Steersman's target topology. The block-diagonal structure is an
**attractor** of the cybernetic feedback loop.

## UPDATE 8: Channel Ablation (EARLY RESULTS)

**Status:** H1 (n=3) at step 8200/10000. H2-H5 auto-queued after completion.

**EARLY FINDING — n=3 achieves IDENTICAL val loss to n=6:**

| Step | n=3 (H1) | n=6 (C-001) | Delta |
|------|----------|-------------|-------|
| 1000 | 0.4344 | 0.4344 | +0.0000 |
| 2000 | 0.4261 | 0.4261 | +0.0001 |
| 4000 | 0.4181 | 0.4178 | +0.0002 |

**Bridge params:** n=3 = 9, n=6 = 36 (4× more).
**BD structure:** n=3 = N/A (all 3 pairs active), n=6 = 100%.
**Fiedler:** n=3 = 0.091, n=6 = 0.0004 (n=6 low because blocks → disconnected).

**Interpretation:** n=3 is the **effective rank** of the Steersman's
discovered structure. The Steersman at n=6 drives 12/15 off-diagonal entries
to zero, leaving exactly 3 active DOF. n=3 naturally provides exactly 3 DOF
and matches performance identically.

**This validates the RD geometry thesis:** the 6-channel bridge is a
redundant parameterization of a 3-axis coordinate system. n=3 APPROXIMATES
what the Steersman discovers from n=6. Both are complementary:
n=3 = efficiency, n=6 = structure.

**Decision gate (still pending):** Does n=6 show unique structure not present
in n=4? If yes, the 3 opposing-face-pair geometry is specifically discovered.

**Report:** `results/channel-ablation/EARLY_CHANNEL_ABLATION_RESULTS.md`

---

*Notes compiled March 13, 2026 from sprint analysis sessions. Updated with
temporal emergence (27,056 checkpoint bridges), init convergence (C-001 vs C-002
COMPLETE), Holly Battery (14B non-cybernetic control), per-layer/per-module analysis,
6 publication figures, C-003 corpus-coupled 100% BD confirmed, H1 early channel
ablation (n=3 = n=6 val loss), and pending experiments (C-003 completion, H2-H5).*
