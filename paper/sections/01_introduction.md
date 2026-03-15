# 1. Introduction

Low-rank adaptation (LoRA; Hu et al., 2021) has become the dominant method
for parameter-efficient fine-tuning of large language models. The standard
formulation decomposes a weight update as $\Delta W = BA$, where
$B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ are
low-rank projections and $r \ll d$. This factorization is efficient,
composable, and well-understood. It is also structurally opaque: rank
dimensions are rotationally symmetric — for any orthogonal $R$,
$BR^\top RA = BA$ — and the adapter provides no compact representation
of the coupling structure it has learned.

Multi-channel extensions of LoRA partition the rank space into $n$ parallel
channels of width $r/n$ and insert a learnable $n \times n$ **bridge matrix**
$\mathcal{B}$ between the down- and up-projections. The bridge parameterizes
inter-channel coupling: $\mathcal{B}_{ij}$ controls how strongly channel $j$'s
activations influence channel $i$'s output. When $\mathcal{B} = I_n$, channels
are independent and the architecture reduces to standard LoRA. When
$\mathcal{B}$ departs from the identity, the bridge encodes learned
relationships between channels — relationships that the training objective
and data jointly determine.

The bridge matrix creates a new object of study. Unlike the high-dimensional
projection matrices $A$ and $B$, the bridge is small ($n^2$ parameters per
adapted module), invariant-rich (its eigenspectrum is rotation-invariant), and
directly interpretable (each entry quantifies a specific inter-channel
interaction). Prior work (Bielec, 2026a) established that untrained bridges
learn task-discriminative structure: a leave-one-out SVM on 28 query-projection
bridges (1,008 parameters) classifies task type at 84.5% accuracy, and bridge
interpolation between adapters preserves eigenspectrum structure (cosine
similarity $> 0.999$) at all mixing ratios. That same work reported a null
result: without explicit supervision, the bridge learns *that* cross-channel
coupling helps but not *which* channels to couple preferentially
(co-planar/cross-planar ratio $= 1.002$, $p = 0.474$).

This paper asks what happens when the bridge receives structured feedback.
We introduce the **Steersman**, a second-order cybernetic feedback mechanism
that monitors the bridge's spectral properties during training and adjusts
optimization dynamics accordingly. The Steersman encodes a geometric prior
derived from the rhombic dodecahedron (RD): its 12 faces decompose into 3
pairs of opposing faces along the $x$, $y$, and $z$ coordinate axes, and the
contrastive loss component rewards coupling between co-planar face-pairs
(channels sharing an axis) while penalizing coupling between cross-planar
pairs (channels on different axes). The spectral component regularizes overall
bridge connectivity via the Fiedler eigenvalue. Together, these components
define a target topology — block-diagonal structure with three independent
$2 \times 2$ blocks — without directly imposing it.

The question is whether the bridge discovers this structure, and if so,
whether the discovery is robust, fast, and mechanistically attributable.

Across 6 cybernetic experiments encompassing 42,500+ bridge matrices, 2 model
scales (TinyLlama 1.1B, Qwen 7B), and 3 initialization strategies, the
answer is unambiguous:

**Block-diagonal emergence is universal.** Every cybernetic experiment with
$n = 6$ channels produces 100% block-diagonal bridge structure. Every
non-cybernetic experiment (6 experiments, 570 bridges, 3 model scales
including Wan 2.1 at 14B) produces 0%. The Steersman is necessary and
sufficient.

**The structure is initialization-independent.** Three initialization
strategies — identity (no bias), geometric (weak co-planar bias), and
corpus-coupled (I Ching complementary trigrams that actively oppose the
target topology) — converge to the same final state: 100% block-diagonal,
co-planar/cross-planar ratios in the 64,000:1–71,000:1 band, and
validation loss within 0.12% of each other. The corpus-coupled
initialization starts with cross-planar dominance; the Steersman inverts the
coupling hierarchy within 75 steps and suppresses the opposing structure by
99.5% within 900 steps (I Ching complementary coupling: $0.029 \to 0.0001$).
Initialization is cosmetic.

**Lock-in is fast.** Block-diagonal structure reaches 100% by step 200 in all
experiments. Cross-planar coupling decays exponentially with a half-life of
123 steps. Co-planar coupling grows linearly. The topology is selected in the
first 200 steps; the remaining training deepens the separation, driving the
co-planar/cross-planar ratio from ${\sim}200$:1 at lock-in to a peak of
82,854:1 (C-003, step 9,000) and a convergence band of ${\sim}70{,}000$:1
at 10,000 steps.

**Task performance is orthogonal to bridge topology.** Validation loss at
10,000 steps is 0.4010–0.4022 across channel counts $n \in \{3, 4, 6\}$,
a maximum delta of 0.17%. Block-diagonal structure provides no measurable
performance benefit. The bridge topology is a structural signature, not a
performance feature.

**The contrastive loss is necessary and sufficient.** Channel-count ablation
($n \in \{3, 4, 6, 8\}$) isolates the mechanism. The contrastive loss
component encodes RD face-pair geometry and is defined only for $n = 6$; for
$n \neq 6$ it returns zero, leaving the Steersman with spectral
regularization alone. The result: $n = 6$ produces Bridge Fiedler eigenvalue
$0.00009$ with 70,404:1 co-planar/cross-planar ratio; $n \in \{3, 4, 8\}$
(spectral-only) produce Bridge Fiedler ${\sim}0.09$ with ${\sim}1$:1 ratio
and 0% block structure — a 1,020$\times$ bifurcation in spectral connectivity.
Spectral regularization drives connectivity but not topology. The geometric
prior is the causal mechanism.

**The effective dimensionality is 3.** At $n = 3$, the bridge has 9
parameters (3 off-diagonal pairs). At $n = 6$ with contrastive loss active,
the Steersman drives 12 of 15 off-diagonal entries to zero, leaving exactly
3 active degrees of freedom — the same count as $n = 3$. Both channel counts
produce identical validation loss (maximum delta: 0.16% across 100
checkpoints), but $n = 3$ uses $4\times$ fewer bridge parameters (792 vs.
3,168 total across all modules). The 6-channel bridge is a redundant
parameterization of a 3-axis coordinate system. The redundancy is informative:
it reveals *which* structure the system discovers.

**The finding is scale-invariant.** Block-diagonal structure appears at both
1.1B (TinyLlama) and 7B (Qwen) under cybernetic training. The Holly Battery
experiment (Wan 2.1, 14B parameters, video diffusion) without Steersman
produces 0% block structure at co-planar/cross-planar ratio 1.07:1, confirming
that the effect is caused by the Steersman, not by model scale or architecture.
Cross-layer correlation Fiedler converges to ${\sim}0.10$ across all three
scales — a scale-invariant structural consistency metric.

We make the following contributions:

1. **Block-diagonal emergence as a universal property of cybernetic
   multi-channel LoRA training.** 100% block-diagonal structure across 6
   experiments, 42,500+ bridges, 2 model scales, with 0% in all non-cybernetic
   controls. The Steersman's contrastive loss, encoding rhombic dodecahedral
   face-pair geometry, is necessary and sufficient.

2. **Initialization independence.** Three qualitatively different initialization
   strategies — including one that actively opposes the target topology —
   converge to the same block-diagonal attractor within 200 steps. Bridge
   initialization is cosmetic; the Steersman overwrites any initial structure.

3. **Coupling dynamics characterization.** Cross-planar coupling decays
   exponentially (half-life: 123 steps). Co-planar coupling grows linearly
   (${\sim}0.00012$/step). The dynamics define three phases: topology selection
   (steps 0–200), exponential ratio growth (steps 200–1,000), and plateau
   (steps 1,000+). Peak co-planar/cross-planar ratio reaches 82,854:1.

4. **Channel count as effective rank.** $n = 3$ matches $n = 6$ validation
   loss with $4\times$ fewer bridge parameters. The Steersman at $n = 6$ uses
   3 degrees of freedom from 15 available, recovering the 3-axis coordinate
   geometry of the rhombic dodecahedron. Effective dimensionality equals the
   number of coordinate axes, not the number of faces.

5. **Contrastive loss as the necessary and sufficient mechanism.** Spectral
   regularization alone (active at all channel counts) produces uniform
   coupling with Bridge Fiedler ${\sim}0.09$. Adding the contrastive
   component (active only at $n = 6$) produces 1,020$\times$ lower Bridge
   Fiedler ($0.00009$) and 70,000:1 co-planar/cross-planar separation. The
   geometric prior determines topology; spectral loss determines connectivity
   magnitude.

6. **Scale invariance from 1.1B to 14B parameters.** Block-diagonal structure
   emerges identically at TinyLlama 1.1B and Qwen 7B under cybernetic
   training. Cross-layer structural consistency (correlation Fiedler
   ${\sim}0.10$) is invariant across 1.1B, 7B, and 14B scales.

The remainder of this paper is organized as follows. Section 2 reviews
background and related work in parameter-efficient fine-tuning, multi-channel
adaptation, and cybernetic optimization. Section 3 describes the RhombiLoRA
architecture, the Steersman feedback mechanism, and the rhombic dodecahedral
geometry that motivates the contrastive loss. Section 4 presents the core
experimental evidence for block-diagonal emergence, lock-in dynamics, and
initialization independence. Section 5 reports the channel-count ablation that
isolates the contrastive loss as the causal mechanism. Section 6 examines scale
invariance across three model architectures. Section 7 discusses implications,
limitations, and connections to broader questions about learned structure in
parameter-efficient fine-tuning.
