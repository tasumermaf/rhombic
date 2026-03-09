# Paper 3 — Abstract Draft (v0.1)

> **Working title:** "The Learnable Bridge: Task Fingerprinting and Adapter
> Composition via Structured Coupling Matrices in Low-Rank Adaptation"

---

## Abstract

Low-rank adaptation (LoRA) decomposes weight updates into two projection
matrices A and B, but provides no structured intermediate representation
of how these projections interact. We introduce a learnable 6×6 coupling
matrix — the *bridge* — between A and B in the RhombiLoRA architecture,
adding 36 parameters per adapter layer. Through systematic analysis of
336 bridge matrices across three tasks (general instruction following,
code generation, mathematical reasoning) on Qwen2.5-7B, we demonstrate
three findings:

**(1) Task fingerprinting.** Bridge matrices encode task identity: a
leave-one-out SVM classifies task type at 83.3% accuracy using only
the 56 query+output-projection bridges (2,016 parameters total), compared
to 33.3% chance. Tasks modulate bridge coupling *magnitude* — general
instruction following produces 2.2× more cross-channel coupling than
code generation (Fiedler 0.040 vs 0.018) — while preserving coupling *structure* (eigenspectrum
cosine similarity > 0.999 between tasks).

**(2) Efficient adapter composition.** Linear interpolation of bridge
matrices between task-specific adapters preserves eigenspectrum structure
(cosine > 0.999) at all mixing ratios, with 10% cross-task contamination
causing no measurable structural degradation. Non-linear Fiedler
trajectories in 2/3 task pairs reveal that structurally dissimilar tasks
exhibit destructive interference at intermediate mixing ratios — a
phenomenon invisible to full-weight merging diagnostics.

**(3) Non-uniform informativeness.** Query projections carry the most
task-discriminative bridge structure across all analyses: highest coupling
(Fiedler), highest between-task distance, highest merge sensitivity. This
ranking is consistent across three independent experimental phases (cross-
phase Pearson r = 0.921 between task discrimination and merge sensitivity).
Later transformer layers are significantly more task-specific than earlier
layers (Spearman ρ = 0.701, p < 0.0001), with the output projection
surpassing the query projection in the final five layers.

These results establish the bridge as a compact, interpretable diagnostic
of adapter behavior — a 36-parameter summary of what training discovered,
readable without inference or evaluation on held-out data.

---

## Key Numbers for Quick Reference

| Finding | Metric | Value |
|---------|--------|-------|
| Task classification (Q+O) | LOO SVM accuracy | 83.3% |
| Task classification (Q-proj) | LOO SVM accuracy | 82.1% |
| Task classification (all) | LOO SVM accuracy | 72.9% |
| Chance level | 3 tasks | 33.3% |
| Between > within task | Mann-Whitney p | < 1e-6 |
| Bridge parameters per layer | 6×6 matrix | 36 |
| Optimal fingerprint size | 56 Q+O bridges | 2,016 params |
| Eigenspectrum preservation | cosine at α=0.1 | 1.0000 |
| Task merge vs random | eigenspectrum cos | 0.999 vs 0.5 |
| Module ranking consistency | Ph1A × Ph2A Pearson r | 0.921 |
| Depth gradient | Spearman ρ | 0.701 (p < 0.0001) |
| Dynamic range | max/min discrimination | 10.3× |
| Fiedler ratio (alpaca/code) | coupling magnitude | 2.2× (0.040/0.018) |

---

*Draft v0.1, March 8, 2026. Pending: Phase 3A results (overfitting diagnostic),
generative evaluation of bridge merging, step 2000 code re-run.*
