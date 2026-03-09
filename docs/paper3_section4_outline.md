# Paper 3 §4 — Experiments: Structural Outline

> **Working title:** "The Learnable Bridge: Task Fingerprinting and Adapter
> Composition via 36-Parameter Coupling Matrices"
>
> **Status:** Outline from available results (Mar 8, 2026). Pending: Phase 3A
> overfit diagnostic, generative evaluation of bridge merging.

---

## §4.1 Experimental Setup

- **Model:** Qwen2.5-7B-Instruct (28 layers, 4 target modules per layer)
- **Architecture:** RhombiLoRA r=24 with FCC topology → 6×6 bridge (36 params/layer)
- **Tasks:** Alpaca-cleaned (general), CodeAlpaca-20k (code), GSM8K (math)
- **Training:** 2K-10K steps, AdamW, LR=2e-4, bf16, QLoRA
- **Bridge extraction:** 112 bridge matrices per adapter (28 layers × 4 modules)
- **Total bridges analyzed:** 336 (Phase 1A) + 33 interpolated (Phase 2A)

**Table 1:** Training configurations per task (steps, dataset size, hardware, checkpoint timing)

## §4.2 Phase 0: Bridge Anatomy (Single-Task Analysis)

**Research question:** Does the bridge develop non-trivial structure during training?

**Figure 1:** Fiedler value evolution across 10K training steps (from Exp 2)
- Show monotonic growth, contrast with co/cross flatness
- Annotation: "The bridge learns COUPLING but not DIRECTION"

**Table 2:** Per-module-type bridge statistics (Fiedler, deviation) with ANOVA
- Q > K > V > O on Fiedler (F=6.022, p=0.0008)
- Q-V closer than Q-K (t=2.13, p=0.042) — tracks information flow, not computation

**Key finding for narrative:** The bridge is architecturally sensitive to position
within the attention head. This motivates Phase 1A.

## §4.3 Phase 1A: Task Fingerprinting

**Research question:** Do bridges trained on different tasks develop distinguishable structures?

**Figure 2:** Between-task vs within-task distance distributions (violin plot or histogram)
- Mann-Whitney U, p = 0.000000
- Visually show the separation

**Table 3:** Between-task and within-task distances (from TASK_FINGERPRINT_REPORT)

**Figure 3:** Confusion matrices — Full set (74.1%) vs Q-proj only (84.5%)
- Side by side, showing alpaca/code going to 100% while math stays confused

**Table 4:** Per-module discrimination and subset classification accuracy
- Q-proj: 84.5%, O-proj: 77.4%, Q+O: 83.3%, Full: 74.1%
- Headline: "1,008 parameters outperform 4,032 for task identification"

**Figure 4:** Per-layer task discrimination (heatmap, 28 layers × 4 modules)
- Color = mean between-task Frobenius distance
- Shows depth gradient (ρ=0.701, p<0.0001) and late-layer O-proj peaks

**Figure 4b (NEW):** Task fingerprint emergence over training steps
- Step 0: all distances = 0.000000 (identical initialization)
- Monotonic divergence through training
- No initialization confound — cleanest possible control

**Key finding:** 28 Q-proj bridges (1,008 params) classify task type at 84.5%.
Tasks change bridge coupling MAGNITUDE, not STRUCTURE (eigenspectrum cosine ≈ 1.0).
Fingerprints emerge entirely from training (zero at step 0, monotonically diverging).

## §4.4 Phase 2A: Bridge-Level Adapter Merging

**Research question:** Can adapter composition operate at the bridge level?

**Figure 5:** Fiedler interpolation curves for 3 task pairs (α = 0.0 to 1.0)
- Alpaca↔code: linear (R²=0.956)
- Alpaca↔math and code↔math: non-linear (R²=0.735, 0.823)
- Mark the Fiedler dip at code↔math α≈0.2

**Table 5:** Structure preservation at α=0.1 (cos=1.0000 for all pairs)
- "10% contamination preserves source eigenspectrum perfectly"

**Table 6:** Random baseline comparison at α=0.5
- Task merge: cos > 0.999. Random merge: cos ≈ 0.5.
- "Bridge merging is structured, not noise."

**Figure 6:** Per-module merge sensitivity across 3 pairs
- Q-proj most sensitive (consistent with Phase 1A — most task-specific
  modules are most merge-sensitive, r=0.921)

**Key finding:** Bridge interpolation preserves task structure but is not
uniformly linear. Non-linearity correlates with structural dissimilarity
between source tasks.

## §4.5 Cross-Phase Synthesis

**Research question:** Is there a unified model of bridge behavior across experiments?

**Table 7:** Module ranking across phases (the "Q-proj always #1" table)
- Cross-phase Pearson r = 0.921 between Phase 1A distance and Phase 2A sensitivity

**Figure 7:** Schematic of the "bridge behavior model"
- Q-proj: high coupling, high task specificity, high merge sensitivity
- K/V: low coupling, low specificity, merge-robust
- O-proj: moderate coupling, high LATE-LAYER specificity
- Depth gradient: early=general, late=task-specific

**The narrative:** The bridge provides a compact, interpretable view into
what training discovered. Different attention components specialize
differently, and the bridge reveals this at 36 parameters per layer.

## §4.6 Phase 3A: Overfitting Diagnostic (pending)

**Research question:** Does bridge structure predict overfitting?

**Figure 8:** Fiedler and deviation vs train-val loss gap (if r > 0.7)
- If PASS: "The bridge is an early warning system for fine-tuning degradation"
- If FAIL: "Bridge structure is training-phase-invariant, useful for other purposes"

## §4.7 Ablations and Controls

- **Random baseline:** 20-trial random bridge comparison (already computed)
- **Training step sensitivity:** Code at step 1500 vs step 2000 (pending)
- **Topological baseline:** If time permits, compare FCC vs cubic bridge fingerprinting

---

## Figure Budget: 7-8 figures, 6-7 tables

## Data Availability Statement

"All bridge matrices, analysis scripts, and reproduction instructions are
available at [GitHub/HuggingFace]. The training infrastructure uses
standard HuggingFace Transformers and PEFT. The RhombiLoRA implementation
is available as the `rhombic` Python library."

---

*Outlined March 8, 2026. Based on completed Phases 0, 1A, 2A and
cross-phase synthesis. Phase 3A pending (~7h).*
