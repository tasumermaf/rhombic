# Cross-Phase Module Synthesis — Phases 0, 1A, 2A

> **Purpose:** Correlate module-type findings across three independent experiments
> to identify which components of the attention head carry the most structural,
> task-specific, and merge-sensitive bridge information.

## Summary

**Q-projections are the dominant bridge component across all three phases.**
They develop the most coupling (Phase 0), carry the most task-discriminating
information (Phase 1A), and are the most sensitive to cross-task interference
during merging (Phase 2A). This is a single coherent finding measured three
independent ways.

## Per-Module Metrics Across Phases

| Module | Ph0 Fiedler | Ph1A Task Distance | Ph2A Merge Sensitivity | Rank (0/1A/2A) |
|--------|------------|-------------------|----------------------|----------------|
| q_proj | **0.050** | **0.162** | **0.353** | 1 / 1 / 1 |
| k_proj | 0.039 | 0.075 | 0.204 | 2 / 3 / 4 |
| v_proj | 0.036 | 0.072 | 0.244 | 3 / 4 / 3 |
| o_proj | 0.035 | 0.133 | 0.276 | 4 / 2 / 2 |

## Cross-Phase Correlations (n=4 module types)

| Pair | Spearman ρ | Pearson r | p (Pearson) |
|------|-----------|-----------|-------------|
| Ph0 Fiedler × Ph1A Distance | 0.400 | 0.647 | 0.353 |
| Ph0 Fiedler × Ph2A Sensitivity | 0.200 | 0.746 | 0.254 |
| **Ph1A Distance × Ph2A Sensitivity** | **0.800** | **0.921** | **0.079** |

**Strongest correlation:** Phase 1A task discrimination and Phase 2A merge
sensitivity (r = 0.921). The modules that carry the most task identity are
the most vulnerable to cross-task interference. This is not surprising — it
would be suspicious if it were otherwise — but quantifying it matters for
practical adapter composition.

With n=4, no correlation reaches p < 0.05. The pattern is interpretive, not
statistical proof. But the rank consistency (q_proj always #1) is stronger
evidence than any single correlation.

## Key Observations

### 1. Q-proj Dominance Is Consistent

Q-projections rank #1 in all three phases. The query projection computes
attention patterns over the full input space — it is the most expressive
component of the attention head. More expressiveness = more coupling = more
task specificity = more merge sensitivity.

### 2. O-proj Rises from Phase 0 to Phases 1A/2A

The output projection ranks last in Phase 0 (least total coupling) but
second in Phases 1A and 2A (second-most task-specific and merge-sensitive).
Interpretation: O-proj doesn't develop the most bridge structure overall,
but what it develops is highly task-dependent. It may be encoding how the
attention output is reformatted for the specific task — a task-specific
"output formatting" layer.

### 3. K-proj and V-proj Are Structurally Paired

K and V have nearly identical Phase 1A distances (0.075 vs 0.072) and
both show low merge sensitivity in the code×math pair (0.031 vs 0.023).
They form a structurally conserved pair that doesn't differentiate much
by task. This is consistent with Phase 0 finding L-012: Q-V proximity
exceeds Q-K, meaning K is the odd one out — but here K and V are paired
because both are *non-discriminative*.

The resolution: L-012 measured structural SIMILARITY (bridge pattern
shape). Here we measure task SENSITIVITY (how much the pattern changes
across tasks). K and V have similar patterns (L-012) AND both change
little across tasks (this analysis). Q and O have dissimilar patterns
AND both change substantially across tasks.

### 4. Implications for Practical Adapter Composition

If bridge-level merging becomes operational:
- **Q-proj bridges need the most careful handling** during composition.
  They carry the task fingerprint and are most damaged by naive interpolation.
- **K-proj and V-proj bridges are robust** to cross-task mixing. They could
  potentially be shared across task adapters without significant degradation.
- **A practical composition strategy:** Merge K/V bridges freely, interpolate
  O bridges cautiously, and use task-specific Q bridges (no merging).
  This would require storing only 28 Q-proj bridges per task (one per layer)
  while sharing the other 84.

## Per-Layer Task Discrimination

**Depth is significant.** Spearman ρ = 0.701 (p < 0.0001) between layer index
and mean between-task Frobenius distance. Later layers carry more task identity.

| Rank | Layer | Mean Distance | Note |
|------|-------|--------------|------|
| 1 | 22 | 0.2131 | |
| 2 | 21 | 0.2109 | |
| 3 | 24 | 0.2047 | Contains single most discriminative position |
| 4 | 20 | 0.1861 | |
| 5 | 23 | 0.1826 | |
| ... | | | |
| 26 | 5 | 0.0996 | |
| 27 | 0 | 0.0768 | |
| 28 | 1 | 0.0724 | Contains least discriminative position |

**Dynamic range:** 10.3× (Layer 24 o_proj = 0.3188, Layer 1 v_proj = 0.0309).

**Layer × Module interaction:** The module-averaged view (q_proj always #1)
hides a critical interaction. In the final layers (20-24), **o_proj surpasses
q_proj** at individual positions. Layer 24 o_proj (0.3188) is the single most
discriminative bridge position in the entire model. Layer 21 o_proj (0.3035)
is third. The output projection specializes by task in the late layers where
the model is "formatting" its output.

## Subset Classification — Feature Selection

LOO SVM accuracy using different bridge subsets, consistent pipeline:

| Subset | Positions | Parameters | LOO Accuracy | vs Full Set |
|--------|-----------|-----------|-------------|-------------|
| All 112 | 112 | 4,032 | 72.9% | baseline |
| Q-proj only | 28 | 1,008 | 82.1% | +9.2pp |
| O-proj only | 28 | 1,008 | 76.2% | +3.3pp |
| **Q+O combined** | **56** | **2,016** | **83.3%** | **+10.4pp** |
| Top 5 layers, all | 20 | 720 | 78.3%* | +5.4pp |
| Top 5 layers, Q+O | 10 | 360 | 76.7%* | +3.8pp |

*Top-5-layer values from step-1500 analysis; qualitative ranking unchanged.

**Key finding:** Removing K-proj and V-proj bridges IMPROVES classification
by up to 10.4 percentage points. The uninformative modules add noise that
degrades the SVM's decision boundary. **56 Q+O bridges (2,016 parameters)
are the optimal task fingerprint** — the O-proj late-layer signal adds value
beyond Q-proj alone once code training reaches 2000 steps.

**Per-task breakdown (Q-proj vs Full):**

| Task | Full Set | Q-proj Only | Q+O | Delta (Q+O) |
|------|----------|-------------|-----|-------------|
| alpaca | 82.1% | **100.0%** | 92.9% | +10.8pp |
| code | 81.2% | 96.4% | 91.1% | +9.9pp |
| math | 55.4% | 50.0% | **66.1%** | +10.7pp |

**Q+O is the optimal subset** because it improves math classification (the
hard case) while maintaining strong alpaca/code performance. Q-proj alone
makes alpaca/code perfect but sacrifices math.

The improvement is NOT uniform. Q-proj selection makes alpaca and code
**perfectly** separable (100%) while math degrades slightly. Math bridges
are confused with code (11/28 misclassified as code) because code and math
are structurally close (Phase 1A between-task distance: 0.066, vs 0.190
for alpaca↔code). The overall accuracy improvement comes from cleaning up
the easy separations, not from resolving the hard one.

**Practical implications:**
- Task identification requires reading only 28 bridges, not 112
- The bridge fingerprint costs 1,008 parameters — less than a single
  hidden state vector (4,096 dimensions in Qwen2.5-7B)
- K/V bridges can be ignored for task classification purposes
- For distinguishing structurally similar tasks (code↔math), additional
  signals beyond Q-proj bridges are needed — possibly O-proj late-layer
  bridges, or generative evaluation

## Fingerprint Emergence Over Training

### Classification Accuracy Trajectory (Q-proj only)

**3-task LOO SVM:**

| Step | Accuracy | Note |
|------|---------|------|
| 0 | 33.3% (chance) | All bridges identical |
| 1000 | 51.2% | Signal emerging, above chance |
| Final | **84.5%** | Strong separation |

**Code↔Math binary LOO SVM (finer resolution):**

| Step | Accuracy | Note |
|------|---------|------|
| 0 | 0% | Bridges identical, SVM confused |
| 500 | 66.1% | Signal detectable after ~80 minutes training |
| 1000 | 87.5% | Strong signal |
| 1500 | **92.9%** | Still climbing — not yet plateau |

The hardest pair to separate (code↔math, distance = 0.066) achieves 93%
binary accuracy at step 1500. The 3-way classifier's low per-class math
accuracy (53.6%) is a multi-class difficulty, not a separation failure — in
the binary comparison, code and math are well-separated and still diverging.

### Distance Trajectories

**Step 0 between-task distance: exactly 0.000000 for all pairs.** All bridges
initialize identically (identity matrix). Task fingerprints emerge entirely
from training — no initialization confound.

| Step | alpaca↔code | alpaca↔math | code↔math |
|------|-----------|-----------|---------|
| 0 | 0.0000 | 0.0000 | 0.0000 |
| 1000 | 0.0313 | 0.0488 | 0.0483 |
| final* | 0.1902 | 0.1713 | 0.0664 |

*Alpaca final = step 10K; code final = step 1500; math final = step 2000.

**Code↔math fine-grained trajectory:** 0.000 → 0.034 (500) → 0.048 (1000)
→ 0.060 (1500). Monotonically diverging. Still separating at the last
checkpoint — additional training would increase separation further.

**Within-task evolution (step 0 → final):** Alpaca 0.199, Math 0.069,
Code 0.032. Proportional to both training duration (10K >> 2K > 1.5K)
and task diversity (general instruction following requires more bridge
adaptation than focused code or math tasks).

## Source Data

- Phase 0: `results/exp2/bridge_anatomy.md` — 112 bridges from Exp 2 (Alpaca)
- Phase 1A: `results/fingerprints/TASK_FINGERPRINT_REPORT.md` — 336 bridges (3 tasks × 112)
- Phase 2A: `results/bridge_merge/MERGE_REPORT.md` — 3 task pairs × 11 α values
- Cross-phase: This document — per-layer discrimination + subset classification

---

*Generated March 8, 2026. Updated with per-layer analysis and subset classification.
Three independent measurements + one cross-analysis, one consistent story.*
