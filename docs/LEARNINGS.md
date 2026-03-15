# RhombiLoRA — Research Learnings Knowledge Base

> **Purpose:** Institutional memory of what was learned, how, and why it matters.
> Not just results — the reasoning errors, the corrections, and the principles
> extracted from them. This document serves future researchers working on
> geometric inductive biases in neural network adapters.
>
> **Started:** 2026-03-08, after the Exp 2.5 null result.

---

## L-001: Rank Dimensions Are Rotationally Symmetric

**The error:** Assuming that labeling LoRA rank dimensions with geometric
meaning (direction pairs of a rhombic dodecahedron) would cause training
to respect those labels.

**The reality:** LoRA's rank dimensions are a low-rank matrix factorization.
The output `B @ A` is invariant under rotation of the rank basis: for any
orthogonal matrix R, `B @ R^T @ R @ A = B @ A`. The channels are
interchangeable coordinates. Channel 0 does not "know" it should carry
analytical content. The optimizer sees them as equivalent.

**The principle:** You cannot get semantic channel assignment for free from
data structure. If you want specific channels to carry specific content,
you must supervise it explicitly (contrastive loss, channel labeling during
training, explicit routing). Inductive biases at the rank level do not
emerge from structure at the input level — transformer depth erases the
signal.

**Evidence:** Exp 2.5 (23K geometric dataset, 95% co-planar perspective
pairing) produced co/cross = 1.002 (p = 0.474). Three experiments, two
model scales (1.5B, 7B), two data types (isotropic Alpaca, directional
geometric). 224+ bridge matrices measured. The null is comprehensive.

---

## L-002: Total Connectivity ≠ Directional Preference

**The finding:** FCC topology (6 channels) consistently produces more
cross-channel coupling than cubic topology (3 channels): 4.6× at 1.5B,
1.73× at 7B. This is the algebraic connectivity advantage predicted by
Papers 1-2.

**The distinction:** More total connectivity (higher Fiedler value) does
NOT imply directional preference (co-planar > cross-planar). The bridge
learns that mixing helps. More channels = more mixing. But the mixing
is uniform — no preferred direction.

**The principle:** Topology affects how MUCH information flows between
channels. It does not affect WHICH channels prefer each other. These are
independent properties. Algebraic connectivity (Fiedler) measures the
first. Co/cross ratio measures the second. Both should be tracked; neither
implies the other.

---

## L-003: Prompt Structure Does Not Survive Transformer Depth

**The hypothesis:** If training data is structured with explicit
perspective pairing (analytical+empirical together 95% of the time),
this structure should imprint on the bridge weights.

**Why it fails:** The gradient path from "this prompt pairs perspectives
A and B" to "bridge entry B[i,j] should increase" passes through 28
transformer layers and billions of parameters. By the time that signal
reaches the 36-parameter bridge, it is indistinguishable from noise.

**The principle:** The deeper an architectural element sits in the
network, the less it can be influenced by surface-level input structure.
The bridge sits at the LoRA rank dimension — far from the input. For
the bridge to develop input-correlated structure, it needs either (a)
explicit supervision at the bridge level, or (b) a dynamic routing
mechanism that conditions on input features directly.

---

## L-004: The Transient Peak Is an Optimizer Artifact

**The observation:** Under isotropic Alpaca data, co/cross ratio follows
a trajectory: 0.929 (1K) → 1.023 (2K) → 1.091 (3K peak) → decays to
1.019 (10K). Under geometric data, co/cross is flat: 0.992 → 1.003 →
1.002.

**The interpretation:** The transient peak under isotropic data is the
optimizer's early exploration phase — random walk in bridge space that
temporarily creates directional appearance. It is not a signal; it is
noise that regularization washes out. The geometric data starts closer
to 1.0 (less noise, more constrained optimization landscape) and stays
there.

**The principle:** Transient training dynamics are not evidence of learned
structure. Only sustained, statistically significant effects count.
Decision criteria should be evaluated at convergence, not at intermediate
checkpoints. The decision tree design (requiring >1.10 at step 3000)
was correct methodology — the transient peak at 1.091 correctly failed
the threshold.

---

## L-005: Bridge Structure Is Data-Independent

**The finding:** Perturbation retention (~69%) and Gram alignment (0.921)
are identical between Alpaca and geometric data. These are architectural
properties of the RhombiLoRA bridge, not data-dependent learned features.

**The implication:** The bridge's structural properties — distributed
coupling, connectivity level — come from the architecture and
initialization, not from training. This is both a limitation (the
bridge doesn't capture data structure) and an asset (the bridge provides
consistent structural guarantees regardless of training data).

**The principle:** When a metric doesn't change across experimental
conditions, it's measuring the architecture, not the learning. This
distinction matters for knowing what to claim: "our architecture
provides X" vs "training on this data produces X."

---

## L-006: Decision Trees Prevent Cherry-Picking

**The practice:** Before running Exp 2.5, we wrote explicit decision
criteria: co/cross > 1.10 AND p < 0.05 → PASS, 1.05-1.10 → MARGINAL,
~1.0 → NULL. Each branch had pre-specified next steps.

**Why it matters:** When the result was co/cross = 1.002, there was no
ambiguity about what to do. No temptation to squint at one checkpoint,
one layer, one adapter that might show a higher number. The null branch
activated cleanly.

**The anti-pattern:** Running an experiment, getting a null, re-running
with tweaked parameters, finding one favorable subset, and publishing
that subset as the result. This is how scientific embarrassment happens.
Pre-registered decision criteria prevent it.

**The principle:** Define pass/fail before you have data. Commit to the
decision tree in writing. When the data arrives, execute the tree. This
is not pessimism — it is the mechanism that makes positive results
trustworthy.

---

## L-007: "The Cube Does Cube Things Well"

**The strategic insight:** Standard LoRA (the "cube" — 6-connected,
orthogonal, the unexamined default) already handles the core task of
low-rank weight adaptation. RhombiLoRA should not try to do that job
better. It should find what the bridge enables ON TOP of what the cube
already does.

**Candidate unique capabilities (under investigation):**
1. **Adapter composition** — merge at the 36-parameter bridge level
   instead of the full weight level
2. **Controllable sparsity** — the bridge spans a continuum from
   block-diagonal (MELoRA) to fully coupled
3. **Interpretability** — the bridge is a readable 6×6 matrix
4. **Task fingerprinting** — different tasks may produce different
   bridge signatures
5. **Regularization** — the bridge constrains A/B interaction

**The principle:** When your innovation doesn't beat the default at the
default's own job, find the job that only your innovation can do.
Don't reinvent the wheel; build the axle.

---

## L-008: Null Results Are Publishable and Valuable

**The finding:** "Architectural topology alone does not create directional
channel preference in LoRA adapters" is an informative result that saves
other researchers from attempting the same approach.

**What makes it publishable:**
- Clear hypothesis with falsifiable predictions
- Rigorous experimental design with pre-registered criteria
- Root cause analysis (rotational symmetry of rank dimensions)
- Positive sub-findings (Fiedler advantage is real, distributed structure)
- Clear implications for future work (explicit supervision needed)

**The principle:** A well-characterized null result advances the field
more than a weakly-supported positive claim. The scientific community
needs to know what doesn't work and why.

---

## L-009: Separate What the Architecture Provides from What Training Discovers

**The confusion:** Early analysis mixed architectural properties (FCC >
cubic connectivity, 69% perturbation retention) with training outcomes
(co/cross ratio, Gram alignment). When the training outcomes were null,
it felt like everything had failed.

**The clarification:** The architectural properties ARE the positive
results. They are deterministic consequences of the topology. They don't
need to be "discovered" by training — they are provided by design. The
question "what does training add?" is separate from "what does the
architecture guarantee?"

**The principle:** In any neural architecture experiment, clearly separate:
1. Properties of the architecture (testable without training)
2. Properties of the initialization (testable at step 0)
3. Properties that emerge from training (require the full experiment)

The first two are your floor. The third is your hypothesis.

---

## L-010: The Difference Between Emergence and Engineering

**The original hope:** The bridge would DISCOVER geometric order from
data — emergence.

**The revised approach:** We BUILD geometric order into the bridge via
explicit supervision — engineering.

**Why this is better:** Emergence is beautiful but fragile and
unpredictable. Engineering is reliable and controllable. A bridge with
engineered directional preference is more useful than one that might
or might not develop it depending on training conditions.

**For the broader vision:** The cybernetic argument (examine every
default, 12-connected > 6-connected) is about engineering, not
emergence. The lattice topology papers prove the engineering advantage.
The neural experiments tried to show emergence and got null. The path
forward is engineering: contrastive pre-training, explicit channel
supervision, input-dependent routing. Build the structure; don't wait
for it to appear.

---

## L-011: Module Type Differentiates Bridge Structure

**The finding:** Q-projections develop 40% more bridge coupling (Fiedler
0.050) than V-projections (0.036). ANOVA across q/k/v/o_proj: F = 6.022,
p = 0.0008. The bridge is NOT uniform across attention head components.

**Why it matters:** Standard LoRA treats all target modules identically —
same rank, same initialization, same learning rate. The bridge reveals
that different modules in the attention head develop different coupling
structures. Q-projections (which compute queries) need more cross-channel
mixing than V-projections (which carry values).

**The principle:** When an architectural element differentiates by
position within a larger structure, it is encoding positional information.
The bridge "knows" whether it sits at Q, K, V, or O — even though nothing
in the architecture explicitly tells it. This is a data-driven discovery,
not an architectural guarantee.

**Evidence:** Phase 0 analysis of 112 Exp 2 bridges (28 layers × 4 modules).

---

## L-012: Q-V Proximity Contradicts the Naive Hypothesis

**The finding:** Bridge structures at Q-projections are MORE similar to
V-projections than to K-projections (paired t-test: t = 2.13, p = 0.042).
This is the opposite of the naive hypothesis that Q and K (which attend
to each other via dot product) would develop similar bridges.

**The interpretation:** The bridge tracks information flow direction, not
computation similarity. Q asks the question; V provides the answer. They
are partners in the same information retrieval operation. K is the
indexing mechanism — it serves Q but carries different structural
requirements. The bridge couples Q and V because they are the input and
output of the same semantic operation.

**The principle:** In attention mechanisms, proximity of learned structure
does not follow the computational graph (Q·K → attention scores). It
follows the semantic graph (Q asks → V answers). This distinction
matters for any architecture that adds cross-module coupling.

---

## R-001: Smooth Approximations for Hard Routing (Reference)

**Source:** https://balthasar.is/blog/2024/04/24/non-differentiable-functions-in-machine-learning/

**The technique:** Temperature-parameterized sigmoid functions can
approximate hard step functions while remaining differentiable. Start
soft (high temperature) during training, anneal toward hard routing at
inference. Generalizes via mollifiers (convolution with smooth kernels).

**Relevance to Phase 4 (dynamic bridge):** The planned gated bridge
`B(x) = I + σ(W_gate @ pool(h)) ⊙ B_learnable` uses a sigmoid gate.
If we want the gate to approach hard channel routing (directing specific
content to specific channels), temperature annealing is the mechanism.
Start with soft routing so gradients flow everywhere, then sharpen so
the bridge learns to commit to specific channel assignments.

**Relevance to channel supervision (L-001):** If explicit channel
assignment is needed (because data-driven discovery fails per L-001),
a soft-to-hard assignment schedule could enforce directional preference
while keeping the training differentiable. The bridge starts isotropic
and gradually crystallizes into co-planar/cross-planar structure.

**Not yet tested.** Filed for Phase 4 implementation.

---

## L-013: The Bridge Encodes Task Identity

**The finding:** A leave-one-out SVM trained on flattened 6×6 bridge matrices
classifies task type (alpaca/code/math) at **73.5% accuracy** (chance = 33.3%).
Mann-Whitney U between-task vs within-task: p = 0.000000. Code is most
distinctive (97.3% correct); math is least (70.5%, confused with both others).

**Why this matters:** The bridge has 36 learnable parameters per layer. Standard
LoRA has no structured intermediate element at all — task information is
distributed across A and B. The bridge provides a compact, interpretable
summary of what A and B jointly learned. No prior work uses adapter internal
structure as a task fingerprint (competitive landscape, March 2026).

**The mechanism:** Tasks change coupling MAGNITUDE, not STRUCTURE. Eigenspectrum
cosine similarity between tasks ≈ 1.0. The bridge develops the same pattern
at different intensities: Alpaca 0.040 > Math 0.028 > Code 0.017 (Fiedler).
General instruction following (diverse, multi-domain) requires more cross-channel
mixing than structured, formal tasks.

**The principle:** When a small, interpretable matrix distinguishes conditions
that the full weight matrices also distinguish, the small matrix is a
diagnostic. It is a compressed representation of what training discovered.
This is the bridge's unique value proposition — not doing what standard LoRA
does better, but providing a readable summary of what standard LoRA does
silently.

**Evidence:** Phase 1A analysis of 336 bridges (112 per task × 3 tasks).
Alpaca from Exp 2 (10K steps), Code from local RTX 6000 Ada (1500 steps),
Math from RunPod RTX 4090 (2000 steps).

---

## L-014: Q-Projections Are the Best Task Discriminator

**The finding:** Q-projections produce the highest mean between-task distance
(0.162), followed by O-projections (0.133), then K and V (0.075, 0.072).
This is consistent with Phase 0 finding L-011 (Q-projections develop 40%
more bridge coupling than V-projections).

**The implication:** If bridge fingerprinting becomes a practical tool,
you may only need Q-projection bridges (28 per model instead of 112) for
task identification — a further 4× compression of the diagnostic.

**The principle:** The most functionally differentiated component of a
system also carries the most diagnostic information. Q-projections do
the most complex work in attention (computing queries across the full
input space); they also develop the most task-specific bridge structure.
The diagnostic power follows the computational complexity.

**Evidence:** Phase 1A per-module analysis, 112 adapters per module type.

---

## L-015: Bridge Merging Is Smooth but Not Uniformly Linear

**The finding:** Linear interpolation of 6×6 bridge matrices between task-specific
adapters preserves eigenspectrum structure (cosine > 0.999) at all mixing ratios.
However, the Fiedler trajectory is LINEAR for alpaca↔code (R²=0.956) but
NON-LINEAR for alpaca↔math (R²=0.735) and code↔math (R²=0.823).

**The non-linearity:** Code↔math interpolation exhibits a Fiedler DIP at α≈0.2
(from 0.017 to 0.015, then recovery to 0.028). This is destructive interference —
the two bridge patterns partially cancel before the math pattern dominates. The
eigenspectrum remains intact (cos=1.0000), so the interference affects connectivity
magnitude, not connectivity structure.

**Why alpaca↔code is linear:** Alpaca (Fiedler 0.040) and code (0.017) differ
primarily in magnitude, not pattern. Interpolating between them smoothly reduces
coupling strength. Alpaca↔math (0.040 vs 0.028) and code↔math (0.017 vs 0.028)
involve BOTH magnitude and pattern differences, creating curvature.

**The principle:** When merging adapters at the bridge level, task pair similarity
matters. Dissimilar-magnitude pairs interpolate linearly. Pairs with structural
differences (different coupling patterns, not just intensities) exhibit non-linear
trajectories. Adapter merging strategies should test for linearity before assuming
a simple weighted average works.

**Practical implication:** At 10% contamination (α=0.1), ALL pairs preserve the
dominant task's structure with cos=1.0000. Safe mixing ratio for all task
combinations: up to 10% of a foreign bridge. Beyond that, test first.

**Evidence:** Phase 2A, 3 task pairs, 11 α values each, random baseline
comparison (20 trials). Report at `results/bridge_merge/MERGE_REPORT.md`.

---

## L-016: The Bridge Is a Viable Merge Target

**The finding:** All three task pairs pass the random baseline comparison at
α=0.5. Task-merged bridges retain eigenspectrum cosine > 0.999 with the source
tasks, while random-merged bridges retain only cos ≈ 0.5. Bridge interpolation
preserves meaningful structure; random noise destroys it.

**Why this matters for adapter composition:** Standard adapter merging operates
on the full weight matrices (millions of parameters). Bridge-level merging
operates on 36 parameters per layer (6×6 matrix). If task structure is encoded
in the bridge (Phase 1A: 73.5% task classification accuracy), and bridge
interpolation preserves that structure (Phase 2A: cos > 0.999), then the bridge
is a sufficient target for adapter composition.

**The 36-parameter composition thesis:** Keep task A's A/B projections (the
"heavy" parameters). Swap or interpolate only the bridge (the "light" structural
parameter). The bridge carries the task fingerprint; A/B carry the projection.
This separates WHAT the adapter projects from HOW it couples channels.

**Not yet tested:** Whether bridge-only swapping actually produces usable
task behavior changes at inference. Phase 2A measured structural metrics only.
Generative evaluation (perplexity on held-out task data) is the next step
and requires GPU.

**Evidence:** Phase 2A bridge interpolation experiment, 3 task pairs.

---

## L-017: Less Data, Better Classification — Feature Selection Matters

**The finding:** LOO SVM on 28 Q-proj bridges achieves **84.5% accuracy** —
10.4 percentage points HIGHER than the full 112-bridge set (74.1%). Removing
K-proj and V-proj bridges improves classification because they add noise without
task-discriminative signal.

**The ranking of subsets:**
- Q-proj only (28 bridges, 1,008 params): **84.5%**
- Q+O combined (56 bridges, 2,016 params): 83.3%
- O-proj only (28 bridges, 1,008 params): 77.4%
- All 112 (4,032 params): 74.1%

**Why Q-proj is the optimal subset:** Q-projections compute attention queries
over the full input space. They are the most expressive attention component —
and the bridge at Q-projections develops the most task-specific coupling
pattern. K and V bridges are structurally conserved across tasks (Phase 1A
distances: 0.075 and 0.072 respectively), so including them dilutes the
task signal with task-invariant noise.

**The principle:** In classification from structured intermediate representations,
feature selection can matter more than feature quantity. The bridge provides a
natural feature decomposition (Q/K/V/O × 28 layers) where only a subset carries
diagnostic information. Using all features assumes uniform informativeness —
selecting the right subset assumes non-uniform informativeness and benefits from it.

**Practical implication:** A task fingerprint based on 28 Q-proj bridges costs
1,008 parameters — smaller than a single hidden state vector (4,096 dimensions
in Qwen2.5-7B). This is the most compact task diagnostic known to us as of
March 2026.

**Evidence:** Cross-phase synthesis analysis, LOO SVM with consistent pipeline
across 7 subsets. Full data at `results/CROSS_PHASE_SYNTHESIS.md`.

---

## L-018: Task Discrimination Increases with Depth

**The finding:** Spearman ρ = 0.701 (p < 0.0001) between transformer layer
index and mean between-task bridge distance. Later layers are more task-specific.
Top 5 discriminative layers: 22, 21, 24, 20, 23 — all in the final quarter of
the 28-layer model.

**The layer × module interaction:** The module-averaged view (q_proj #1 across
all layers) hides a critical interaction. In layers 20-24, **o_proj surpasses
q_proj** at individual positions. Layer 24 o_proj (distance 0.3188) is the
single most discriminative bridge position in the entire model.

**Interpretation:** Early layers handle general features; late layers specialize
for the task. The output projection (o_proj) reformats the attention output for
downstream consumption — in the final layers, this reformatting becomes
task-specific. Q-proj carries the most task signal ON AVERAGE because it is
consistently discriminative across all depths. O-proj is less consistent but
peaks higher in the task-critical late layers.

**The principle:** When a structural element (the bridge) is replicated across
network depth, its informativeness is not uniform. Depth-dependent analysis
reveals structure that layer-averaged analysis obscures. This applies generally
to any adapter architecture with per-layer parameters.

**Evidence:** Per-layer × per-module Frobenius distance matrix, 28 layers ×
4 modules × 3 task pairs. Full data at `results/CROSS_PHASE_SYNTHESIS.md`.

---

## L-019: Task Fingerprints Emerge From Training, Not Initialization

**The finding:** At step 0, all three tasks produce IDENTICAL bridges (between-
task distance = 0.000000). The identity initialization is the same regardless
of what data will be used. All task differentiation emerges from gradient descent.

**The trajectory:** Code↔math distance evolves monotonically: 0.000 (step 0) →
0.034 (step 500) → 0.048 (step 1000) → 0.060 (step 1500). No plateau, no
collapse — the tasks continue separating as training proceeds. Longer training
= more separation.

**Why this matters:** If bridges started with task-correlated noise (e.g., from
a pre-trained initializer), the fingerprinting result would be confounded. The
zero-initialization confirmation means the SVM is classifying genuine learned
structure, not initialization artifacts. This is the cleanest possible control
for the task fingerprinting claim.

**Within-task evolution is proportional to task complexity:** Alpaca (general,
diverse) moves 0.199 from initialization. Math (structured) moves 0.069.
Code (most constrained) moves 0.032. The bridge adapts MORE when the task
demands more diverse cross-channel coupling.

**Evidence:** Step-0 bridge comparison across 3 tasks × 112 adapter positions.
Code↔math fine-grained trajectory from 500-step checkpoints.

---

## L-020: Classification Accuracy Tracks Training Monotonically

**The finding:** Code↔math binary SVM accuracy on Q-proj bridges:
0% (step 0) → 66% (step 500) → 88% (step 1000) → 93% (step 1500).
No plateau, no collapse, monotonically improving.

**The practical implication:** Task fingerprinting works at ANY training
duration — you don't need to train to completion. After 500 steps (~80
minutes on an A6000), the bridge already carries enough task information
for 66% binary classification. By step 1000, it's 88%.

**The contrast:** 3-way classification at step 1000 is only 51%. Binary
classification is much easier because the SVM doesn't need to draw
boundaries between ALL tasks simultaneously. The multi-class difficulty
doesn't mean the bridges lack information — it means the decision
boundaries are more complex in 36-dimensional space.

**For Paper 3 Figure:** Plot accuracy vs training step as a "readability
curve" showing when the bridge becomes a reliable diagnostic. This gives
practitioners a decision criterion: "train for N steps before checking
the bridge fingerprint."

**Evidence:** LOO SVM at 4 training steps, Q-proj only, code vs math binary.

---

## L-021: Cybernetic Closed-Loop Training Solves the Directionality Problem

**The problem:** Experiments 2.5 and 2.7 produced comprehensive nulls on
directional channel preference. Data structure does not imprint on bridge
weights (L-001, L-003). Higher bridge learning rate increases total
coupling but not directional preference (Exp 2.7). The bridge needed
explicit supervision, not data-driven discovery (L-010).

**The solution:** Cybernetic closed-loop training with three components:
1. `differentiable_fiedler()` — weighted Laplacian from bridge off-diagonal
   elements, second eigenvalue via `torch.linalg.eigvalsh` (differentiable)
2. Contrastive loss separating co-planar from cross-planar coupling
3. Steersman feedback adjusting regularization weights based on bridge state

**The result:** Axis alignment of **22,477:1** (Qwen 7B) and **47,145:1**
(TinyLlama 1.1B). Cross-planar coupling suppressed by 4-5 orders of
magnitude. The bridge learns the RD geometry's three coordinate axes with
near-perfect fidelity.

**The principle:** When emergence fails (L-010), engineering succeeds —
but the engineering must be cybernetic. A static loss function cannot
adapt to the bridge's evolving state. The Steersman's three control laws
(connectivity trending, directionality stagnation, stability monitoring)
create a feedback loop that adjusts the training signal as the bridge
develops. The bridge doesn't just learn structure; it learns structure
under adaptive supervision that responds to what it has learned so far.

**Evidence:** Exp 3.0 (Qwen 7B, 12,900 steps) and Exp 3.0-TL (TinyLlama
1.1B, 10,000 steps). Verified findings at
`results/VERIFIED_FINDINGS_2026_03_12.md`.

---

## L-022: Axis Alignment Emerges Immediately

**The finding:** Co/cross ratio at step 200 is already **816:1**. By
step 500 it reaches 3,621:1. The final value of 22,477:1 is refinement,
not discovery — the bridge identifies the RD's coordinate axes within
the first ~1% of training.

**The trajectory (Qwen 7B):**
```
Step   100:    26:1
Step   200:   816:1
Step   500: 3,621:1
Step 1,000: 8,408:1
Step 3,000: 11,800:1
Step12,900: 22,477:1
```

**The principle:** When the correct inductive bias is provided (RD
topology + contrastive supervision), the network converges to the
geometric structure almost instantly. The long tail of training refines
the ratio by suppressing residual cross-planar leakage, but the
qualitative result is established by step 200. This has practical
implications: bridge diagnostics can be extracted very early in training
without waiting for convergence.

**Evidence:** Exp 3.0 step-by-step analysis from results.json.

---

## L-023: Fiedler Eigenvalue Is Scale-Invariant

**The finding:** The algebraic connectivity (Fiedler eigenvalue) of the
trained bridge converges to **~0.10** across three model scales:
- TinyLlama 1.1B: 0.1006
- Qwen 7B: 0.102
- Wan 2.1 14B (Holly Battery): 0.108

The absolute coupling magnitude varies (co-planar mean: 0.778 at 1.1B,
1.517 at 7B) but the Fiedler value — which measures the bridge's
algebraic connectivity independent of scale — settles to the same point.

**The principle:** The bridge's connectivity structure is determined by
the RD topology and the cybernetic training protocol, not by the model
it's embedded in. This suggests a universal property of the 6-channel
FCC bridge under contrastive supervision: the eigenstructure converges
to a fixed point regardless of how many parameters surround it.

**Implication:** The Fiedler value could serve as a convergence
diagnostic — when it stabilizes near 0.10, the bridge has learned its
geometric structure and further training primarily refines coupling
magnitudes.

**Evidence:** Three independent experiments across three model scales.

---

## L-024: Wrong Pair Definitions Produce Misleadingly Low Ratios

**The error:** Prior analysis reported co/cross ratio as ~3.0 using
hemisphere groupings ({0,1,2} vs {3,4,5}) as the pair definition.
The correct pairs use the RD's `direction_pair_coupling()` function:
co-planar = axis-aligned pairs (0,1), (2,3), (4,5); cross-planar =
all 12 remaining pairs.

**The correct number:** 22,477:1 (Qwen 7B), 47,145:1 (TinyLlama 1.1B).
The hemisphere grouping dilutes the signal because it mixes some
axis-aligned pairs with cross-planar ones within each hemisphere.

**The principle:** Metric definitions matter as much as the values
they produce. A 3× ratio and a 22,000× ratio describe the same
underlying data — the difference is entirely in how "co-planar" and
"cross-planar" are defined. Always use the geometric definition from
the library (`direction_pair_coupling()`), not ad-hoc groupings.

**Additionally:** The Steersman bug (`break` at line 305 of
`train_cybernetic.py`) caused results.json to report single-bridge
metrics instead of aggregates. Two bugs compounding — wrong pairs AND
single-bridge sampling — produced the misleading 3.0 number. Fixed
March 12, 2026.

**Evidence:** Re-analysis of Exp 3.0 raw bridge .npy files.

---

## L-025: Bridge Spectral Properties Track Overfitting

**The finding:** Deviation from identity correlates with train-val gap
at **r = 0.888** (p = 7.3e-35). Fiedler eigenvalue correlates at
**r = 0.825** (p = 5.6e-26). Both bridge spectral properties rise in
lockstep with the onset of memorization.

**The phase transition:** At step 400 (epoch 13 of 500-sample training),
deviation jumps 2× (0.046 → 0.093) while the train-val gap crosses from
0.062 to 0.132. This marks the onset of overfitting — the model begins
memorizing, and the bridge immediately reflects it.

**Saturation:** Bridge metrics plateau by step ~1000 (deviation ~0.174,
Fiedler ~0.070) even as the train-val gap continues growing (1.057 at
step 1000 → 1.500 at step 10000). The bridge saturates before the model
does. This means bridge metrics are most diagnostic in the EARLY phase
of overfitting (steps 200-1000), before saturation limits their
dynamic range.

**The principle:** The bridge is a leading indicator of overfitting,
not a lagging one. It responds to the onset of memorization before
the gap has fully developed. A practitioner monitoring bridge deviation
during training would see the warning signal at step 400, when the gap
is only 0.13 — long before the gap reaches catastrophic levels (>1.0).

**Evidence:** Phase 3A, 500/500 train/val split, 101 checkpoints over
10,000 steps. Raw data at `results/exp3a-overfit/results.json`.

---

## L-026: Corpus Weighting Was Misapplied (RETRACTED, March 13 2026)

**The original claim:** Holly Battery (Wan 2.1 14B) corpus-weighted
RhombiLoRA achieved loss 1.645 — worse than unweighted RhombiLoRA (1.552).

**What actually happened:** The corpus weights were placed on the
**diagonal** (scaling individual channels) rather than the **off-diagonal**
(coupling between channels). This is fundamentally wrong: the corpus
arithmetic describes *relationships between* prime threads, not the
magnitude of individual threads. Diagonal weighting over-constrains
channel scaling while providing zero coupling information. The experiment
tested the wrong thing.

**The correction:** `corpus_coupled` mode (added March 13, 2026) places
hexagram-derived coupling × geometric coupling × thread density on the
off-diagonal, with identity on the diagonal. This tests the actual
hypothesis: that corpus arithmetic encodes inter-channel relationships,
not channel magnitudes.

**The principle STILL HOLDS (refined):** Theoretical advantages in
simplified models do not automatically transfer to full-scale systems.
But the specific failure here was not "corpus weights don't work" — it
was "corpus weights were applied to the wrong matrix elements." The
diagonal-vs-off-diagonal distinction is the difference between asking
"how important is each channel?" (wrong question) and "how do channels
relate to each other?" (the actual corpus arithmetic question).

**Status:** Retracted pending re-evaluation with correct off-diagonal
application (experiments C-001 through C-004).

**Evidence:** Holly Battery, 3 runs on Wan 2.1 14B, WandB
`alvdansen-labs/rhombi-experiment`. See `corpus_coupled_matrix()` in
`rhombic/corpus.py` for the corrected implementation.

---

## L-027: RhombiLoRA Produces Smaller Checkpoints

**The finding:** Holly Battery final checkpoints:
- Standard LoRA r24: **439 MB**
- RhombiLoRA r24: **220 MB** (50% smaller)

Both at identical rank (24), alpha (12), and target modules. The
bridge architecture constrains the effective parameter space, producing
a checkpoint approximately half the size of standard LoRA.

**The principle:** The bridge doesn't just change HOW parameters
couple — it reduces the effective parameter count by constraining
A/B interaction through a 36-parameter bottleneck. This has practical
deployment implications: smaller checkpoints = faster loading, less
storage, easier distribution. The 9.15 GB VRAM saving during training
and the 50% checkpoint size reduction are likely the same underlying
phenomenon: the bridge makes the adapter more parameter-efficient.

**Evidence:** File sizes from `aramintak/rhombi-experiment-backup`.

---

## L-028: Contrastive Topology IS the Structure Signal

**The finding:** Channel ablation across n=3, 4, 6, 8 proves that
block-diagonal structure emerges ONLY when contrastive topology (co-axial
pair specification) is active. Spectral-only training (eigenvalue
regularization without co/cross pair definitions) converges to Fiedler
~0.09 regardless of channel count. The bridge learns connectivity but
not directionality.

**The data:**

| Run | Channels | Topology | Co/Cross | Fiedler | BD? |
|-----|----------|----------|----------|---------|-----|
| H-ch3 | 3 | spectral-only | N/A | 0.0951 | NO |
| H-ch4 | 4 | spectral-only | N/A | 0.0918 | NO |
| H-ch6 | 6 | RD contrastive | 70,404:1 | 0.00009 | YES |
| H-ch8 | 8 | spectral-only | N/A | 0.0944 | NO |
| T-001 | 8 | tesseract contrastive | 5,395:1 | 0.00070 | YES |

**Three conclusions from one table:**

1. **Spectral-only is channel-count-invariant.** n=3,4,8 all converge to
   Fiedler 0.0918-0.0951 — a 3.5% spread across a 2.7× range of channel
   counts. The bridge finds the same connectivity level regardless of how
   many channels exist. The spectral loss creates a universal attractor
   near Fiedler ≈ 0.09.

2. **Contrastive topology creates block-diagonal immediately.** Both H-ch6
   (RD, 70,404:1) and T-001 (tesseract, 5,395:1 at step 2700) produce extreme co/cross
   ratios. The Fiedler drops 3 orders of magnitude (from ~0.09 to ~0.0002)
   because the bridge suppresses cross-planar coupling, which is precisely
   the algebraic connectivity the spectral loss was maintaining.

3. **The Steersman is a general topology programmer.** T-001 uses tesseract
   (4D hypercube) pairs — 4 co-axial, 24 cross-planar — and the Steersman
   programs the bridge into 4+4 block-diagonal structure without any change
   to the feedback laws. The same three control laws (connectivity trending,
   directionality stagnation, stability monitoring) work for ANY co-axial
   pair specification. The Steersman is not RD-specific; it is topology-
   agnostic.

**The eigenvalue signature (T-001 at step 500):**
```
Block A: [~0, 0.000177, 0.000197, 0.000255]  ← suppressed
Block B: [0.1638, 0.1708, 0.1722, 0.1726]    ← active
```
Four eigenvalues near zero, four near 0.17 — a clean 4+4 split matching
the tesseract's 4 co-axial pairs. The bridge has decomposed into two
independent 4-channel blocks connected only by near-zero coupling.

**The principle:** Structural inductive bias in neural adapters requires
explicit supervision at the coupling level. Spectral regularization alone
(encouraging connectivity via eigenvalue targets) produces quantitatively
consistent but qualitatively undifferentiated bridges. Only contrastive
topology — telling the bridge WHICH pairs should couple and which should
not — produces the block-diagonal structure that encodes geometric meaning.
The bridge needs both the map (pair definitions) and the compass (Steersman
feedback) to navigate to structured solutions.

**For Paper 3:** This is the ablation study. The table above is the
definitive evidence that the contrastive loss is necessary and sufficient
for geometric structure. Spectral-only is the control condition that proves
the effect.

**Evidence:** Channel ablation series H-ch3, H-ch4, H-ch6, H-ch8 (Hermes
RTX 4090), T-001 partial (local RTX 6000 Ada). Full data at
`rhombic/docs/EXPERIMENT_TRACKER.md` Channel Ablation section.

---

*Knowledge base started March 8, 2026. Each entry is a lesson extracted
from experimental results. Updated March 15, 2026: L-028 added for
channel ablation key finding — contrastive topology is necessary and
sufficient for block-diagonal structure. The Steersman generalizes to
arbitrary topologies (tesseract confirmed).
L-026 RETRACTED March 13: corpus weights were misapplied (diagonal not
off-diagonal); corpus_coupled mode implements the correction.
The goal is not to record what happened, but to record what was learned
— so the same errors are never repeated and the same insights don't need
to be re-derived.*
