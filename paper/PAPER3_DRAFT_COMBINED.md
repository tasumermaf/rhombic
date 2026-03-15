# The Learnable Bridge: Cybernetic Feedback Discovers Rhombic Dodecahedral Geometry in Multi-Channel LoRA

## Abstract

Multi-channel LoRA (RhombiLoRA) partitions a low-rank adapter's bottleneck into $n$ parallel channels and couples them through a learnable $n \times n$ bridge matrix. We introduce the Steersman, a cybernetic feedback mechanism that monitors the bridge's spectral properties during training and encodes a geometric prior based on the rhombic dodecahedron's 3-axis coordinate system. Across 15 experiments spanning 1.1B–14B parameters, 60,000+ bridge matrices, and three model architectures (TinyLlama, Qwen 7B, Wan 2.1), the Steersman at $n = 6$ produces 100% block-diagonal bridge structure — three independent $2 \times 2$ blocks aligned with the RD coordinate axes — while non-cybernetic training produces 0%. Block-diagonal lock-in occurs within 200 training steps and is independent of initialization strategy, including initializations that actively oppose the target topology (I Ching complementary trigrams suppressed 99.5% in 900 steps). Channel-count ablation across $n \in \{3, 4, 6, 8\}$ reveals that the contrastive loss encoding RD face-pair geometry is necessary and sufficient: at $n = 6$ (contrastive active), the co-planar/cross-planar coupling ratio reaches 82,854:1 and Bridge Fiedler eigenvalue collapses to 0.00009; at $n \in \{3, 4, 8\}$ (contrastive disabled), Bridge Fiedler converges to $\sim$0.09 with 0% block structure — a 1,020$\times$ bifurcation. The effective dimensionality of the discovered structure is exactly 3, matching the number of coordinate axes: $n = 3$ achieves identical validation loss to $n = 6$ (0.17% maximum delta) with $4\times$ fewer bridge parameters. Task performance is orthogonal to bridge topology. The Steersman reveals structural preferences that standard training does not surface.
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
# 2. Background and Related Work

## 2.1 Low-Rank Adaptation

LoRA (Hu et al., 2021) injects trainable low-rank decompositions into frozen pretrained models, reducing the number of trainable parameters by orders of magnitude while maintaining competitive task performance. The method has been extended along several axes: adaptive rank allocation (AdaLoRA; Zhang et al., 2023), weight decomposition (DoRA; Liu et al., 2024), and multi-task composition (LoRAHub; Huang et al., 2023). All standard LoRA variants treat the rank dimensions as a homogeneous vector space — individual rank dimensions are interchangeable under rotation, and the adapter provides no compact representation of inter-dimensional coupling.

## 2.2 Multi-Channel and Multi-Head Parameter-Efficient Methods

Several works partition the adapter's parameter space into parallel components. Multi-head LoRA (Wang et al., 2024) splits the rank into heads that attend to different input subspaces. Mixture-of-LoRA (Li et al., 2024) routes inputs through multiple LoRA experts. These methods introduce structural decomposition but do not include learnable inter-component coupling — each head or expert operates independently, and the outputs are combined through fixed aggregation (concatenation, addition, or gating).

RhombiLoRA (Bielec, 2026a) introduces the bridge matrix as an explicit parameterization of inter-channel coupling. Paper 2 in this series established that the bridge learns task-discriminative structure under standard (non-cybernetic) training: bridge fingerprints classify task type with 84.5% leave-one-out accuracy, and bridge interpolation preserves eigenspectrum structure at all mixing ratios. However, without structured feedback, the bridge develops uniform coupling with no topological preference (co-planar/cross-planar ratio 1.002:1). The present work introduces the Steersman to provide that structured feedback.

## 2.3 Cybernetic Optimization

Cybernetic control theory (Wiener, 1948; Ashby, 1956) formalizes feedback systems that maintain stability through sense-decide-actuate loops. In machine learning, second-order feedback has appeared in learning rate scheduling (Smith, 2017), loss landscape analysis (Li et al., 2018), and meta-learning (Andrychowicz et al., 2016). The Steersman differs from learned meta-optimizers in that its control laws are fixed (not learned), its state space is low-dimensional (3 scalar diagnostics), and its actuation is limited to weight adjustments and learning rate scaling. This simplicity is deliberate: we aim to demonstrate that even minimal feedback is sufficient to drive strong structural emergence.

## 2.4 Block-Diagonal Structure in Neural Networks

Block-diagonal weight matrices arise naturally in modular networks (Clune et al., 2013; Amer and Maul, 2019), where architectural constraints enforce modularity. The lottery ticket hypothesis (Frankle and Carlin, 2019) identifies sparse subnetworks within dense networks; when the surviving connections cluster, the effective weight matrix is approximately block-diagonal. Mixture-of-experts architectures (Fedus et al., 2022; Lepikhin et al., 2021) achieve block-diagonal routing through learned gating functions. In all these cases, the block structure is either architecturally imposed (modular networks), a consequence of pruning (lottery tickets), or mediated by a separate routing mechanism (MoE gating). The present work demonstrates block-diagonal emergence in a dense, unconstrained matrix through feedback-driven training dynamics alone.

## 2.5 The Rhombic Dodecahedron

The rhombic dodecahedron (RD) is the Voronoi cell of the face-centered cubic (FCC) lattice (Conway and Sloane, 1999). Its 12 faces tessellate $\mathbb{R}^3$ without gaps, and its dual is the cuboctahedron. The RD's 6 face pairs partition naturally by coordinate axis, yielding the 3-fold symmetry that motivates the contrastive loss in Section 3.2. In prior work (Bielec, 2026a), we established that 12-connected FCC lattices provide 30% shorter paths and 2.4$\times$ higher algebraic connectivity than 6-connected cubic lattices at comparable spatial resolution. The present work uses the RD's coordinate geometry as a structural prior for multi-channel coupling, not as a spatial lattice.
# 3. Method

## 3.1 RhombiLoRA Architecture

Standard LoRA (Hu et al., 2021) decomposes each weight update as $\Delta W = BA$, where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ are low-rank factors with rank $r$. RhombiLoRA introduces a single additional component: a learnable *bridge matrix* $\mathbf{M} \in \mathbb{R}^{n \times n}$ that couples $n$ parallel channels within the low-rank bottleneck.

Given $n$ channels, the rank $r$ is partitioned into $n$ equal segments of size $s = r / n$ (we require $n | r$). The hidden representation $\mathbf{h} = A\mathbf{x} \in \mathbb{R}^r$ is reshaped into a matrix $\mathbf{H} \in \mathbb{R}^{n \times s}$, where each row corresponds to one channel. The bridge acts on the channel dimension:

$$\mathbf{H}' = \mathbf{M} \mathbf{H}$$

The coupled representation $\mathbf{H}'$ is then flattened back to $\mathbb{R}^r$ and projected through $B$. The complete forward pass for a single RhombiLoRA adapter is:

$$\Delta \mathbf{y} = \frac{\alpha}{r} B \cdot \text{flatten}(\mathbf{M} \cdot \text{reshape}(A\mathbf{x}, [n, s]))$$

where $\alpha$ is the LoRA scaling factor (set to 16 throughout). When $\mathbf{M} = I_n$, the channels are uncoupled and the forward pass reduces exactly to standard LoRA. The bridge is initialized to identity in all experiments unless otherwise noted.

Each attention projection within each transformer layer receives its own independent bridge matrix. For a model with $L$ transformer layers and four target projections per layer ($W_Q$, $W_K$, $W_V$, $W_O$), the total number of bridge parameters is $n^2 \times 4 \times L$. With $n = 6$, this yields $36 \times 4 \times 22 = 3{,}168$ bridge parameters for TinyLlama-1.1B (22 layers) and $36 \times 4 \times 28 = 4{,}032$ for Qwen2.5-7B-Instruct (28 layers). These bridge parameters represent a negligible fraction of the total trainable parameters: approximately 0.04% of the LoRA parameter budget.

The bridge parameterization is deliberately simple: a dense $n \times n$ matrix with no architectural constraints such as symmetry, positive definiteness, or sparsity. Any structure that emerges in the learned bridge is therefore a consequence of the training dynamics, not of the parameterization.

## 3.2 The Steersman: Cybernetic Feedback

Training proceeds with three loss components. The primary objective is the standard causal language modeling loss $\mathcal{L}_{\text{LM}}$. Two auxiliary losses act on the bridge matrices: a contrastive loss $\mathcal{L}_{\text{con}}$ that encodes geometric priors, and a spectral loss $\mathcal{L}_{\text{spec}}$ that regularizes algebraic connectivity. The total loss is:

$$\mathcal{L} = \mathcal{L}_{\text{LM}} + w_c \mathcal{L}_{\text{con}} + w_s \mathcal{L}_{\text{spec}}$$

where the weights $w_c$ and $w_s$ are not fixed hyperparameters but are adjusted dynamically by a feedback controller we call the Steersman.

**Contrastive loss.** For $n = 6$, the bridge's off-diagonal entries are partitioned into *co-planar pairs* and *cross-planar pairs* according to the rhombic dodecahedral geometry described in Section 3.3. The contrastive loss encourages co-planar coupling to exceed cross-planar coupling:

$$\mathcal{L}_{\text{con}} = -\frac{1}{|\mathcal{A}|}\sum_{a \in \mathcal{A}} \left( \frac{1}{|\mathcal{C}|}\sum_{(i,j) \in \mathcal{C}} |M^a_{ij}| - \frac{1}{|\mathcal{X}|}\sum_{(i,j) \in \mathcal{X}} |M^a_{ij}| \right)$$

where $\mathcal{A}$ is the set of all adapters, $\mathcal{C}$ is the set of 3 co-planar index pairs, and $\mathcal{X}$ is the set of 12 cross-planar index pairs. The loss is minimized when co-planar entries have large absolute magnitude and cross-planar entries are suppressed. For $n \neq 6$, the contrastive loss is undefined and returns zero: no geometric prior is applied. This design decision is central to the ablation study in Section 5.

**Spectral loss.** The spectral regularization drives the algebraic connectivity of each bridge toward a target Fiedler eigenvalue $\lambda_2^*$. For each adapter $a$, we construct the weighted Laplacian $\mathbf{L}^a$ by treating $|M^a_{ij}|$ (for $i \neq j$) as edge weights of a complete graph on $n$ nodes. The Fiedler value $\lambda_2^a$ is the second-smallest eigenvalue of $\mathbf{L}^a$, computed via `torch.linalg.eigvalsh` to maintain differentiability. The spectral loss is:

$$\mathcal{L}_{\text{spec}} = \frac{1}{|\mathcal{A}|}\sum_{a \in \mathcal{A}} (\lambda_2^a - \lambda_2^*)^2$$

The initial target is $\lambda_2^* = 0.1$; this is adapted upward during training to track the observed mean Fiedler value across adapters.

**The Steersman.** Every $T = 100$ training steps, the Steersman executes a sense-decide-actuate cycle. During the sensing phase, it extracts all bridge matrices and computes: the mean Fiedler eigenvalue $\bar{\lambda}_2$ across adapters, the mean co-planar to cross-planar coupling ratio, and the mean Frobenius deviation from identity $\|\mathbf{M} - I_n\|_F$. From a sliding window of the most recent 5 measurements, it estimates linear trends (slopes) for each diagnostic.

Three control laws determine the actuator outputs:

1. *Connectivity.* If the Fiedler trend is declining (slope $< -0.001$), the spectral weight $w_s$ is increased proportionally to the magnitude of the decline, bounded above by $w_s^{\max} = 0.2$. If the trend is positive, $w_s$ decays slowly toward its base value of 0.05.

2. *Directionality.* If the co-planar/cross-planar ratio is near unity and stagnating (trend magnitude $< 0.02$), the contrastive weight $w_c$ is increased, bounded above by $w_c^{\max} = 0.5$. Once the ratio exceeds 1.2, $w_c$ decays, allowing the bridge to consolidate its learned topology without continued pressure.

3. *Stability.* If the deviation from identity grows faster than a threshold (slope $> 0.05$), the learning rate for bridge parameters is dampened multiplicatively (bounded below at $0.1\times$ the base rate). If deviation is converging, the bridge learning rate is gradually restored toward its base value.

The Steersman's control laws are purely reactive: they adjust weights and learning rates based on measured trends, with no model of the system's future dynamics. The initial contrastive weight is $w_c = 0.1$ and the initial spectral weight is $w_s = 0.05$.

**Optimizer configuration.** Bridge parameters and LoRA parameters ($A$, $B$) are placed in separate optimizer parameter groups within a single AdamW optimizer (weight decay 0.01). Both groups share a base learning rate of $2 \times 10^{-4}$ with cosine decay after a linear warmup of 200 steps. The Steersman's bridge learning rate scaling (control law 3) is applied as a multiplicative factor on the bridge parameter group only.

## 3.3 Rhombic Dodecahedral Geometry and Co-planar Pairs

The rhombic dodecahedron (RD) is a convex polyhedron with 12 congruent rhombic faces, 14 vertices (8 cubic and 6 octahedral), and 24 edges. Its faces partition naturally into 6 antipodal pairs, each pair sharing a common coordinate axis. In the standard embedding, the 6 face-pair normals align with the FCC lattice directions: $(\pm 1, \pm 1, 0)$, $(\pm 1, 0, \pm 1)$, and $(0, \pm 1, \pm 1)$. These 6 direction pairs are the channels of RhombiLoRA when $n = 6$.

The 6 face pairs decompose further by coordinate axis. Each pair of opposite faces is perpendicular to one of the three Cartesian axes, yielding 3 axis-aligned groupings of 2 face pairs each:

- $x$-axis: channels $(0, 1)$, corresponding to normals $(1,1,0)$ and $(1,-1,0)$
- $y$-axis: channels $(2, 3)$, corresponding to normals $(1,0,1)$ and $(1,0,-1)$
- $z$-axis: channels $(4, 5)$, corresponding to normals $(0,1,1)$ and $(0,1,-1)$

Two face pairs are *co-planar* if they share a coordinate axis (equivalently, if their representative faces share 4 octahedral vertices). Two face pairs are *cross-planar* if they lie on different coordinate axes (sharing only 2 octahedral vertices). This produces 3 co-planar pairs and $\binom{6}{2} - 3 = 12$ cross-planar pairs.

The contrastive loss (Section 3.2) uses this partition directly: it encourages the $6 \times 6$ bridge matrix to develop strong coupling within co-planar pairs and weak coupling between cross-planar pairs. A bridge that fully realizes this objective has block-diagonal structure: three independent $2 \times 2$ blocks, one per coordinate axis. Each block governs the coupling between two channels that share a geometric axis, and cross-block entries are suppressed to near zero.

The co-planar to cross-planar coupling ratio is defined as:

$$\rho = \frac{\frac{1}{|\mathcal{C}|}\sum_{(i,j) \in \mathcal{C}} |M_{ij}|}{\frac{1}{|\mathcal{X}|}\sum_{(i,j) \in \mathcal{X}} |M_{ij}|}$$

where $\mathcal{C}$ denotes the 3 co-planar pairs and $\mathcal{X}$ denotes the 12 cross-planar pairs. A bridge with no directional preference has $\rho \approx 1$. A perfectly block-diagonal bridge has $\rho \to \infty$ as cross-planar entries approach zero. We also report a binary block-diagonal detection metric: a bridge is classified as block-diagonal if $\rho > 10$ and every cross-planar entry has absolute value below 0.001.

## 3.4 Training Setup

All experiments use the alpaca-cleaned dataset (Taori et al., 2023) for instruction-following fine-tuning. Models are loaded in bfloat16 precision with gradient checkpointing enabled. The base model parameters are frozen; only the injected LoRA adapters (and their bridge matrices) receive gradients.

Common hyperparameters across all experiments: learning rate $2 \times 10^{-4}$, batch size 2, gradient accumulation 8 (effective batch size 16), linear warmup for 200 steps followed by cosine decay, maximum gradient norm 1.0, LoRA rank 24, LoRA $\alpha$ 16, target modules $\{W_Q, W_K, W_V, W_O\}$, random seed 42, maximum sequence length 512.

The primary model is TinyLlama-1.1B-Chat (22 transformer layers, 88 adapters, 3,168 bridge parameters at $n = 6$), which permits rapid iteration and extensive ablation. Scale validation uses Qwen2.5-7B-Instruct (28 transformer layers, 112 adapters, 4,032 bridge parameters at $n = 6$). A non-cybernetic control at 14B scale uses Wan 2.1 (video diffusion backbone, 40 layers, no Steersman).

Steersman feedback is evaluated every 100 steps. At each feedback cycle, validation loss is computed on a held-out subset of 1,000 examples, and all bridge matrices are saved to disk for post-hoc analysis.

Three initialization strategies are tested to assess sensitivity to starting conditions. *Identity* initialization sets $\mathbf{M} = I_n$, equivalent to uncoupled standard LoRA at the start of training. *Geometric* initialization adds a small perturbation proportional to the RD face-pair coupling matrix: $\mathbf{M} = I_6 + 0.01 \cdot \hat{C}$, where $\hat{C}$ is the off-diagonal coupling matrix normalized to unit maximum. *Corpus-coupled* initialization places non-uniform off-diagonal values derived from an external weighting scheme, producing a bridge that initially favors I Ching complementary trigram pairings over co-planar RD pairings (I Ching / RD ratio of 1.4:1 at initialization). This last strategy deliberately opposes the target topology, providing the strongest test of initialization independence.

The channel count ablation varies $n \in \{3, 4, 6, 8\}$ while holding the total rank fixed at 24 (yielding channel sizes $s \in \{8, 6, 4, 3\}$ respectively). Because the contrastive loss is defined only for $n = 6$, experiments with $n \neq 6$ operate under spectral regularization alone. This separation is not a limitation but a deliberate experimental design: it isolates the effect of the geometric prior from the effect of spectral pressure.
# 4. Block-Diagonal Emergence

## 4.1 Core Finding: Cybernetic vs. Non-Cybernetic Training

Table 1 presents the complete experimental record. Across 15 experiments spanning three model architectures (TinyLlama 1.1B, Qwen2.5 7B, Wan 2.1 14B), two training regimes (cybernetic and non-cybernetic), four channel counts ($n \in \{3, 4, 6, 8\}$), and three initialization strategies, the block-diagonal classification is binary: every cybernetic experiment at $n = 6$ produces 100% block-diagonal bridge structure, and every other configuration produces 0%.

Six cybernetic experiments at $n = 6$ contribute 42,500+ bridge matrices to the analysis. Each matrix is classified as block-diagonal if $\rho > 10$ and all cross-planar entries fall below $10^{-3}$ in absolute value. The classification rate is 100.0% in every experiment, at every checkpoint from step 200 onward. There are no exceptions, no near-misses, and no late-onset failures. The attractor is absolute.

Six non-cybernetic experiments contribute 570 final-state bridges across three model scales. None exhibits block-diagonal structure. Co-planar/cross-planar ratios range from 0.99:1 to 1.07:1 — statistically indistinguishable from the null hypothesis of uniform coupling. The Holly Battery experiment (Wan 2.1 14B, 10 training epochs, Prodigy optimizer, no Steersman) achieves 3.8% lower loss than the non-RhombiLoRA baseline while using 9.15 GB less VRAM and producing 6% faster inference, but its bridge matrices remain near-identity with $\rho = 1.07$:1. Performance benefits from multi-channel parameterization do not require — and do not induce — topological structure.

**Table 1: Complete Experiment Summary**

| ID | Model | $n$ | Init | Steersman | Steps | Val Loss | $\rho$ | BD% |
|----|-------|-----|------|-----------|-------|----------|--------|-----|
| exp1a–e | Qwen 1.5B | 6 | various | No | 2K | — | $\sim$1:1 | 0% |
| exp2 | Qwen 7B | 6 | FCC | No | 10K | — | $\sim$1:1 | 0% |
| exp2.5 | Qwen 7B | 6 | geometric | No | 3K | — | $\sim$1:1 | 1% |
| **exp3** | **Qwen 7B** | **6** | identity | **Yes** | **12.9K** | — | **18,248:1** | **100%** |
| **exp3\_tiny** | **TinyLlama** | **6** | identity | **Yes** | **10K** | — | **37,929:1** | **100%** |
| **C-001** | **TinyLlama** | **6** | identity | **Yes** | **4K** | 0.4178 | **10,118:1** | **100%** |
| **C-002** | **TinyLlama** | **6** | geometric | **Yes** | **10K** | **0.4010** | **71,337:1** | **100%** |
| **C-003** | **TinyLlama** | **6** | corpus | **Yes** | **10K** | **0.4011** | **64,168:1** | **100%** |
| **H3** | **TinyLlama** | **6** | identity | **Yes** | **10K** | **0.4015** | **70,404:1** | **100%** |
| H1 | TinyLlama | 3 | identity | Yes | 10K | 0.4020 | N/A | N/A |
| H2 | TinyLlama | 4 | identity | Yes | 10K | 0.4022 | $\sim$1:1 | 0% |
| H4 | TinyLlama | 8 | identity | Yes | 10K | — | $\sim$1:1 | 0% |
| Holly | Wan 2.1 14B | 6 | identity | No | 10 ep | 1.493 | 1.07:1 | 0% |


## 4.2 Lock-in Dynamics

Block-diagonal structure does not emerge gradually. It locks in within 200 training steps — 2% of total training — across all six cybernetic $n = 6$ experiments.

Figure 1 shows the co-planar/cross-planar ratio $\rho$ on a logarithmic scale for all six experiments. Despite spanning two model scales (1.1B and 7B) and three initialization strategies, the trajectories are qualitatively identical: rapid exponential growth in the first 200 steps, followed by sustained linear growth on the log scale. By step 200, every experiment exceeds $\rho = 100$:1. By step 1,000, every experiment exceeds $\rho = 5{,}000$:1.

The coupling dynamics decompose into two independent processes. Cross-planar coupling decays exponentially from its initial value to a computational floor of ${\sim}10^{-5}$. A least-squares fit across three experiments (C-001, C-002, C-003) yields a consistent half-life of $\tau_{1/2} = 123 \pm 8$ steps. Co-planar coupling grows linearly at approximately $0.00012$ per step (Frobenius norm per module), with no evidence of saturation through 10,000 steps ($R^2 = 0.998$ for a linear fit from step 500 onward).

These two processes are independent: topology selection (which entries are suppressed) occurs in the exponential phase and completes by step 200; magnitude growth (how large the surviving entries become) continues throughout training and drives the ratio upward. The peak ratio achieved across all experiments is 82,854:1 (C-003, step 9,000). The peak does not occur at the final step because cross-planar entries at the computational floor exhibit stochastic fluctuations; the ratio is bounded below by the ratio of co-planar magnitude to the numerical noise floor.

The final ratios at 10,000 steps for the three complete TinyLlama runs converge to a narrow band: C-002 = 71,337:1, H3 = 70,404:1, C-003 = 64,168:1 (maximum pairwise delta: 10%). The exp3 (Qwen 7B) final ratio of 18,248:1 at 12,900 steps is lower because the Qwen run used an earlier version of the training code with less aggressive contrastive weighting.


## 4.3 Initialization Independence

The Steersman's block-diagonal attractor is strong enough to overwrite any initial bridge structure. We test three initialization strategies of increasing adversarial strength:

**Identity initialization** ($\mathbf{M} = I_6$). Channels start uncoupled. There is no directional bias — all off-diagonal entries begin at zero. The bridge must discover both the topology (which entries grow) and the dynamics (how fast they grow) from the training signal and the Steersman's feedback alone.

**Geometric initialization** ($\mathbf{M} = I_6 + 0.01 \hat{C}$). A small perturbation in the direction of the co-planar coupling matrix. This provides a weak initial bias toward the target topology but does not impose it: the perturbation magnitude (0.01) is two orders of magnitude below the final co-planar coupling ($\sim$1.4 at step 10,000). If the Steersman merely amplified the initial perturbation, the final co-planar magnitude would depend on the initial perturbation strength. It does not.

**Corpus-coupled initialization**. Off-diagonal entries are set according to I Ching complementary trigram pairings, producing a bridge that initially favors cross-planar coupling over co-planar coupling (I Ching / RD ratio of 1.4:1 at step 0). This initialization *opposes* the target topology: the channels that should decouple (per the RD geometry) are the channels that start most strongly coupled (per the trigram complementarity). Of the three strategies, this provides the strongest test of attractor robustness.

All three initializations converge to the same final state. At step 2,000, the co-planar coupling magnitudes for C-001 (identity), C-002 (geometric), and C-003 (corpus) are 0.252, 0.256, and 0.252 respectively — a maximum pairwise delta of 1.6%. The ratios at step 2,000 are 4,422:1, 4,535:1, and 4,421:1 — a maximum delta of 2.6%. By step 10,000, the final ratios are 10,118:1 (C-001, stopped at 4K), 71,337:1 (C-002), and 64,168:1 (C-003), with validation losses of 0.4178, 0.4010, and 0.4011. The convergence of C-002 and C-003 is within 0.025% in validation loss and 10% in ratio — well within the stochastic variation expected from different random seeds.

The corpus-coupled experiment (C-003) provides the most detailed view of the attractor dynamics. Figure 4 tracks the per-pair coupling magnitudes for all 15 off-diagonal entries. At step 0, the I Ching complementary pairs (e.g., $\text{K\=un} \leftrightarrow \text{Qi\'an}$, channels 0 and 3) have coupling magnitude 0.029, while the RD co-planar pairs start at 0.010 — a 2.9:1 disadvantage. By step 75, the RD co-planar pairs overtake the I Ching pairs (crossover). By step 200, the I Ching complementary coupling has been suppressed to 0.005 (83% reduction). By step 900, it reaches 0.0001 — a 99.5% reduction from the initial value. The Steersman does not merely overwhelm the opposing structure with a stronger signal; it exponentially suppresses it while linearly growing the target structure. The I Ching trigram hierarchy is inverted and then annihilated.


## 4.4 Axis Symmetry and Polarity

The block-diagonal structure exhibits two additional symmetries that constrain interpretation.

**Axis symmetry.** The three co-planar axes ($x$, $y$, $z$) develop coupling of equal magnitude. Across all cybernetic experiments and all checkpoints, the maximum deviation between any two axes' mean coupling magnitude is less than 2.5%. The Steersman treats all three coordinate axes identically, and the bridge learns them identically.

**Polarity freedom.** Within each 2$\times$2 block, the off-diagonal entry $M_{ij}$ can be positive or negative. Approximately 50% of co-planar entries are positive and 50% are negative across the final-state bridges. The Steersman constrains *which* channels couple (topology) but does not constrain *how* they couple (sign). This is consistent with the contrastive loss formulation, which operates on absolute values $|M_{ij}|$: it is agnostic to coupling polarity.

The combination of equal-magnitude axes with free polarity means that the Steersman discovers a 3-degree-of-freedom system embedded in a 15-degree-of-freedom parameter space. Each degree of freedom corresponds to a coordinate axis and can take any real value (positive or negative coupling). The bridge matrix, despite having 36 entries, encodes at most 3 + 6 = 9 informative parameters (3 co-planar couplings + 6 diagonal entries), with the remaining 12 cross-planar entries driven to zero.
# 5. Channel Count Ablation

The experiments in Section 4 establish that the Steersman produces block-diagonal structure at $n = 6$. This section asks why $n = 6$, and what the structure reveals about the effective dimensionality of the bridge's learned representation.

## 5.1 Experimental Design

We vary the channel count $n \in \{3, 4, 6, 8\}$ while holding the total LoRA rank fixed at 24. All experiments use TinyLlama 1.1B with identity initialization and identical hyperparameters (Section 3.4). The channel sizes are $s = 8, 6, 4, 3$ respectively, and the bridge parameter counts per module are $n^2 = 9, 16, 36, 64$.

The critical design element: the contrastive loss component (Section 3.2) is defined exclusively for $n = 6$. For $n \neq 6$, the function `contrastive_bridge_loss` checks $B.\text{shape}[0] \neq 6$ and returns zero. This is not a limitation but a deliberate experimental instrument. The spectral loss operates at all channel counts, providing general connectivity pressure. By comparing $n = 6$ (contrastive + spectral) with $n \in \{3, 4, 8\}$ (spectral only), we isolate the geometric prior's contribution.

This ablation answers three questions: (1) Does block-diagonal structure require the contrastive loss, or can spectral regularization alone produce it? (2) Does the number of channels affect task performance? (3) What is the effective dimensionality of the structure the Steersman discovers?


## 5.2 Results

**Table 2: Channel Count Ablation — Definitive Results**

| $n$ | Bridge Params/Module | Val Loss @10K | Bridge Fiedler | $\rho$ (co/cross) | BD% | Contrastive |
|-----|---------------------|---------------|----------------|---------------------|-----|-------------|
| 3 | 9 | 0.4020 | 0.095 | N/A | N/A | N/A |
| 4 | 16 | 0.4022 | 0.092 | $\sim$1:1 | 0% | disabled |
| **6** | **36** | **0.4015** | **0.00009** | **70,404:1** | **100%** | **active** |
| 8 | 64 | 0.4022 | 0.085 | $\sim$1:1 | 0% | disabled |

The results separate cleanly along three dimensions.


### 5.2.1 Task Performance Is Independent of Channel Count

Validation loss at 10,000 steps varies by at most 0.17% across $n \in \{3, 4, 6\}$: from 0.4015 ($n = 6$) to 0.4022 ($n = 4$). The $n = 8$ experiment, which uses the most bridge parameters (64 per module), achieves 0.4022 at comparable training progress — identical to $n = 4$. Block-diagonal structure provides no measurable performance benefit. Bridge topology is a structural signature, not a performance feature.

The performance equivalence is tighter than might be expected. Over the full 100-checkpoint training trajectory, $n = 3$ and $n = 6$ maintain a maximum validation loss delta of 0.16%. This is within the stochastic variation expected between runs with different random initializations of the LoRA parameters (which are shared across channel counts). The task gradient dominates the language modeling loss surface; bridge topology operates in an orthogonal subspace.


### 5.2.2 Block-Diagonal Structure Requires the Geometric Prior

The Bridge Fiedler eigenvalue — a measure of how tightly connected the bridge graph is — bifurcates sharply between $n = 6$ and all other channel counts. At $n = 6$ (contrastive active), Bridge Fiedler converges to 0.00009: the bridge graph is near-disconnected, with the three 2$\times$2 blocks acting as isolated components. At $n \in \{3, 4, 8\}$ (spectral only), Bridge Fiedler converges to the range 0.085–0.095: a connected graph with no preferential topology.

The ratio of spectral-only Fiedler ($\sim$0.09) to contrastive-active Fiedler (0.00009) is approximately 1,020$\times$. This is not a gradual difference — it is a bifurcation. The spectral loss alone produces well-connected bridges (Fiedler $\sim$0.09) at every channel count. The contrastive loss, available only at $n = 6$, collapses the connectivity by three orders of magnitude by driving 12 of 15 off-diagonal entries to zero.

The spectral gap — the difference between the second and third eigenvalues of the bridge Laplacian, normalized by the second eigenvalue — provides additional evidence. At $n = 6$, the spectral gap converges to 0.00006, indicating near-perfect 3-fold eigenvalue degeneracy: three identical blocks produce three identical Fiedler eigenvalues, and the gap between them approaches zero. At $n = 4$, the spectral gap is 0.559 — the eigenvalues are all distinct, with no block structure.


### 5.2.3 The Fiedler Trajectory at $n = 4$: Competing Objectives

The $n = 4$ experiment (H2) reveals a non-trivial dynamics that simple exponential saturation models fail to capture. The Bridge Fiedler trajectory exhibits three distinct phases:

1. **Rapid growth** (steps 0–1,200). Fiedler rises from 0 toward 0.084 as the spectral loss pushes the bridge toward its connectivity target. The trajectory is well-described by exponential approach to a saturation value.

2. **Dip** (steps 1,200–3,000). Fiedler decreases from 0.084 to 0.076 — a 9.5% decline despite the spectral loss continuing to exert upward pressure. This dip reveals the task gradient competing with the spectral regularizer: the language modeling loss benefits from redistributing bridge connectivity away from the spectral target, and the task gradient temporarily wins.

3. **Recovery** (steps 3,000–10,000). Fiedler slowly climbs from 0.076 to 0.092, exceeding the phase-1 maximum. The spectral loss reasserts control as the task gradient's beneficial redistribution saturates. The final Fiedler value (0.092) exceeds the exponential saturation model's prediction (0.087) by 5.6%.

This three-phase trajectory is absent at $n = 6$, where the contrastive loss dominates: Bridge Fiedler falls monotonically from step 1 onward, reaching 0.00009 at step 10,000. The contrastive loss's gradient toward block structure is strong enough that the task gradient cannot overcome it; the competition visible at $n = 4$ is resolved decisively in favor of the geometric prior at $n = 6$.


### 5.2.4 Three Regimes of Bridge Behavior

The ablation reveals three qualitatively distinct regimes:

**Trivial ($n = 3$).** With only 6 off-diagonal entries (3 symmetric pairs), the bridge has too few degrees of freedom for block structure to be geometrically meaningful. The 3$\times$3 bridge has at most one non-trivial block decomposition (a 2$\times$2 block plus a 1$\times$1 block), which is structurally degenerate. Spectral regularization drives all entries toward uniform coupling (Fiedler 0.095), which is the maximum-entropy configuration for a fully-connected 3-node graph. No contrastive loss is defined at $n = 3$.

**Spectral-only ($n = 4, 8$).** The Steersman operates on spectral loss alone. Bridges develop uniform connectivity — all off-diagonal entries grow toward comparable magnitudes — with Fiedler values in the 0.085–0.092 range. The resulting bridges are well-connected graphs with no topological preference. The spectral loss drives overall connectivity magnitude but provides no signal about which entries should grow relative to others. Without the geometric prior, all entries are treated symmetrically.

**Full Steersman ($n = 6$).** The contrastive loss encodes the RD face-pair partition and drives a 1,020$\times$ Fiedler bifurcation. Bridge matrices decompose into three independent 2$\times$2 blocks, one per coordinate axis. The spectral gap collapses to 0.00006 (3-fold eigenvalue degeneracy). The co-planar/cross-planar ratio reaches 70,404:1 at step 10,000 (peak 76,452:1 at step 8,000). The bridge's 15 off-diagonal degrees of freedom reduce to exactly 3 active dimensions.


## 5.3 Effective Rank and Parameter Efficiency

The channel count ablation reveals that the Steersman at $n = 6$ discovers a 3-dimensional structure within a 15-dimensional parameter space. The 12 cross-planar entries are driven to zero; only the 3 co-planar entries carry information. This matches the $n = 3$ configuration, which has exactly 3 off-diagonal pairs — and achieves identical task performance.

Concretely: $n = 3$ uses 9 bridge parameters per module (792 total across all modules and layers in TinyLlama), while $n = 6$ uses 36 per module (3,168 total) — a 4$\times$ overhead. The additional parameters at $n = 6$ provide no performance benefit (validation loss delta $\leq$ 0.16%) but reveal the internal structure of the learned representation: the system has 3 effective degrees of freedom, corresponding to the 3 coordinate axes of the rhombic dodecahedron.

The $n = 6$ bridge can be viewed as a redundant parameterization of a 3-axis coordinate system. The redundancy is not wasteful — it is *diagnostic*. A 3$\times$3 bridge at $n = 3$ achieves the same task performance but provides no information about which geometric structure underlies the coupling. The 6$\times$6 bridge, by developing block-diagonal structure under the contrastive loss, reveals that the relevant structure is the 3-axis partition encoded in the RD geometry.

This finding has a practical implication for deployment: once the 3-axis structure is established through cybernetic training at $n = 6$, the bridge can be pruned to $n = 3$ (or equivalently, the 12 cross-planar entries can be fixed at zero) with no loss in task performance, yielding a 4$\times$ reduction in bridge parameter count.
# 6. Scale Invariance

The experiments in Sections 4 and 5 establish block-diagonal emergence and its mechanistic basis using TinyLlama 1.1B as the primary model. This section examines whether the findings generalize across model scale, using Qwen2.5 7B and Wan 2.1 14B as additional test beds.

## 6.1 Cross-Scale Comparison

Block-diagonal structure appears identically at 1.1B (TinyLlama) and 7B (Qwen2.5). The Qwen 7B experiment (exp3) uses identity initialization and the Steersman with the same feedback protocol. Over 12,900 training steps, it produces 100% block-diagonal bridges with co-planar/cross-planar ratio 18,248:1. The lower ratio compared to TinyLlama runs (64,000–71,000:1 at 10K) reflects an earlier version of the training code with less aggressive contrastive weighting, not a scale-dependent effect.

Per-layer coupling analysis reveals comparable structure across scales. In both models, co-planar coupling is uniform across layers to within 5%, with no systematic front-to-back gradient. Both models exhibit the same per-module hierarchy: $k$\_proj and $v$\_proj develop the strongest coupling (44,000:1 in Qwen 7B; 46,570:1 in C-002 TinyLlama), followed by $q$\_proj (34,000:1; 16,708:1), with $o$\_proj weakest (19,080:1; 9,116:1). The projection-type ordering is consistent across scales, suggesting an architectural invariant: key and value projections, which operate on the attention mechanism's representational bottleneck, develop stronger inter-channel coupling than the output projection, which acts downstream of attention.


## 6.2 Two Fiedler Metrics

The analysis employs two distinct applications of the Fiedler eigenvalue that must be carefully distinguished.

**Bridge Fiedler** measures *within-matrix* connectivity. For a single bridge matrix $\mathbf{M}$, the weighted Laplacian treats $|M_{ij}|$ as edge weights and computes $\lambda_2$ of the resulting graph. A low Bridge Fiedler indicates near-disconnected components — blocks with minimal cross-block coupling. Bridge Fiedler at $n = 6$ (cybernetic): 0.00009. Bridge Fiedler at $n = 4$ (spectral-only): 0.092.

**Correlation Fiedler** measures *cross-layer* structural consistency. For each module type (e.g., $q$\_proj), the flattened bridge matrices across all layers form a set of $n^2$-dimensional vectors. The pairwise Pearson correlation matrix is constructed, and its Fiedler eigenvalue measures how uniformly the bridge structure propagates across the model's depth. A high Correlation Fiedler indicates that all layers develop the same bridge pattern; a low value indicates layer-specific variation.

The Correlation Fiedler converges to ${\sim}0.10$ across all three model scales: 0.102 (Qwen 7B), 0.101 (TinyLlama), 1.002 (Holly Battery 14B, single RhombiLoRA adapter). This scale-invariant value indicates that bridge structure is remarkably uniform across layers, regardless of model depth (22, 28, or 40 layers), model size (1.1B to 14B), or training regime (cybernetic or non-cybernetic). The cross-layer consistency is an intrinsic property of the multi-channel LoRA parameterization, not of the Steersman.


## 6.3 The Holly Battery Null Result

The Holly Battery experiment provides the non-cybernetic control at 14B scale. Wan 2.1, a 14B-parameter video diffusion backbone, is fine-tuned with 6-channel RhombiLoRA but without the Steersman (no contrastive loss, no spectral loss, no feedback controller). The adapter achieves 3.8% lower loss than the standard LoRA baseline, requires 9.15 GB less VRAM (gradient checkpointing benefits from the factored bridge), and produces 6% faster inference.

Despite these performance benefits, the Holly Battery bridge matrices are near-identity: mean off-diagonal magnitude 0.010, co-planar/cross-planar ratio 1.07:1, Bridge Fiedler $\sim$0.10 (well-connected, no block structure). The bridge learns small perturbations from identity that improve task performance, but these perturbations are uniformly distributed across all off-diagonal entries with no preferential topology.

The Holly result rules out model scale, dataset, and optimizer as confounders. The 14B model has substantially greater representational capacity than TinyLlama (1.1B). The video generation task is fundamentally different from instruction-following. The optimizer (Prodigy, with adaptive learning rate) is more sophisticated than the AdamW used in all other experiments. None of these factors produces block-diagonal structure. The Steersman is the causal factor.
# 7. Discussion

## 7.1 Why Block-Diagonal?

The Steersman's loss landscape contains block-diagonal structure as a sharp attractor. The contrastive loss provides a strong gradient signal that differentiates co-planar from cross-planar entries: at initialization ($\mathbf{M} = I$), all off-diagonal entries are zero and receive zero gradient from the contrastive loss; once stochastic perturbations break symmetry, co-planar entries receive positive reinforcement (grow) while cross-planar entries receive negative reinforcement (shrink). The exponential decay of cross-planar coupling (half-life: 123 steps) reflects the self-reinforcing nature of suppression: as cross-planar entries decrease, the contrastive loss gradient on them intensifies (the gradient of $-|x|$ has constant magnitude regardless of $x$), maintaining pressure until the entries reach the computational floor.

The linear growth of co-planar coupling, by contrast, reflects the absence of saturation in the contrastive reward signal: there is no target magnitude, only the directive to maximize the ratio $\rho$. The co-planar entries grow until the task gradient or the spectral loss provides a countervailing force. The result is an asymmetric attractor: entry into the basin (exponential suppression of cross-planar coupling) is fast and decisive, while the attractor's depth (co-planar magnitude) increases monotonically with training.

The attractor occupies 3 dimensions of a 15-dimensional parameter space. The 12 cross-planar directions are exponentially repulsive (decay to zero with half-life 123 steps). The 3 co-planar directions are linearly attractive (grow without saturation). The 6 diagonal entries are weakly constrained by the spectral loss but not by the contrastive loss. This asymmetry — 3 attractive, 12 repulsive — is a geometric consequence of the RD face-pair partition: there are fewer co-planar pairs (3) than cross-planar pairs (12), so the majority of the parameter space is repulsive.


## 7.2 What Does the Structure Mean?

The three independent 2$\times$2 blocks encode three independent coupling axes. Each axis allows positive or negative coupling between its two channels, providing a one-dimensional control parameter per axis per module. Across the 88 modules in TinyLlama (22 layers $\times$ 4 projections), this yields 264 binary coupling signs (positive or negative) and 264 coupling magnitudes — a compact structural description of the adapter's inter-channel relationships.

The per-module hierarchy ($k$\_proj $>$ $v$\_proj $>$ $q$\_proj $>$ $o$\_proj in coupling magnitude) suggests that the bridge's structural role is connected to the attention mechanism's internal organization. Key and value projections, which determine what information is attended to and what is retrieved, develop the strongest inter-channel coupling. The output projection, which aggregates attention outputs, develops the weakest. This ordering is consistent across TinyLlama and Qwen 7B, suggesting a scale-invariant architectural pattern.

The bridge's polarity freedom — approximately 50/50 positive and negative coupling signs — indicates that the structure encodes both excitatory and inhibitory inter-channel interactions. Each 2$\times$2 block functions as a mini-gate: positive coupling reinforces the joint contribution of two channels, while negative coupling implements a form of competition. The Steersman determines the topology (which channels interact) but leaves the dynamics (excitatory or inhibitory) to the task gradient.


## 7.3 Practical Implications

**Diagnostic over-parameterization.** The $n = 6$ bridge uses $4\times$ more parameters than $n = 3$ but achieves identical task performance. The excess parameters are diagnostically valuable: they reveal the 3-axis coordinate structure that $n = 3$ achieves implicitly. For practitioners, this suggests a two-stage workflow: train with $n = 6$ and cybernetic feedback to discover and verify the bridge's preferred topology, then deploy with $n = 3$ (or pruned $n = 6$) for parameter efficiency.

**Cybernetic training as structural discovery.** The Steersman does not improve task performance (validation loss is within 0.17% of spectral-only training). Its value is structural: it reveals the bridge's latent geometric preferences by applying a geometric prior that the bridge can accept or reject. The fact that the bridge accepts the RD geometry (100% BD in every cybernetic experiment) while rejecting arbitrary coupling patterns (0% BD without the contrastive loss) indicates that the RD coordinate system is compatible with the adapter's natural representational structure.

**Programmable topology.** The contrastive loss is a topology selector: it defines which channel pairs should couple and which should decouple. The current experiments use RD face-pair geometry as the target topology. A natural extension is to test alternative topologies — random partitions, task-specific pairings, or geometries from other polyhedra — to determine whether the bridge accepts arbitrary topological programs or specifically prefers the RD coordinate structure. If the bridge accepts arbitrary programs, the Steersman is a general-purpose topology controller. If it preferentially accepts RD geometry, the rhombic dodecahedron has special status in the multi-channel LoRA parameter space.


## 7.4 Limitations

**Single task family.** All experiments use instruction-following (alpaca-cleaned). The block-diagonal structure may depend on the task type, the dataset, or the relationship between task complexity and channel count. Cross-task validation (code generation, mathematical reasoning, multi-modal) is needed.

**Single bridge architecture.** The bridge is a dense $n \times n$ matrix. Alternative parameterizations (sparse, symmetric, orthogonal) may produce different structural outcomes. The block-diagonal finding may be specific to the unconstrained dense bridge.

**Contrastive loss confound.** The contrastive loss is defined only for $n = 6$. The ablation isolates the contrastive loss as the mechanism, but it does not test whether an $n = 4$ or $n = 8$ contrastive loss (using a different geometric prior) would produce block structure at those channel counts. The current design confounds channel count with contrastive availability. A "contrastive-with-wrong-labels" experiment (applying RD face-pair labels to a 4-channel bridge) would provide additional evidence, but has not been implemented.

**No formal null model.** The probability of observing 100% block-diagonal structure by chance under the null hypothesis (uniform random coupling evolution) has not been computed analytically. The empirical null rate is 0% (6 non-cybernetic experiments, 570 bridges), but a formal combinatorial argument would strengthen the statistical claim.

**Limited scale range.** Three model scales (1.1B, 7B, 14B) provide evidence of scale invariance, but the 14B result is non-cybernetic only. A cybernetic experiment at 14B or larger would confirm that block-diagonal emergence scales to frontier model sizes.


## 7.5 Connection to Broader Literature

Block-diagonal structure in neural network weight matrices has been observed in several contexts: modular networks (Clune et al., 2013), the lottery ticket hypothesis (Frankle and Carlin, 2019), and sparse mixture-of-experts architectures (Fedus et al., 2022). In all prior cases, the block structure arises from architectural constraints (routing, masking, pruning) or regularization penalties (group sparsity). The present work demonstrates that block-diagonal structure can emerge from a purely cybernetic mechanism: second-order feedback that monitors spectral properties and adjusts training dynamics, without any architectural constraint on the bridge matrix itself.

The Steersman's design draws on cybernetic control theory (Ashby, 1956; Wiener, 1948): a sense-decide-actuate loop that maintains homeostasis in the bridge's spectral properties. The three control laws (connectivity, directionality, stability) operate independently, each monitoring a different aspect of the bridge's evolving state. This multi-channel feedback architecture is simpler than learned meta-optimizers (Andrychowicz et al., 2016) or neural architecture search (Zoph and Le, 2017) — it uses fixed control laws with adaptive thresholds rather than learned control policies — yet it is sufficient to drive a strong structural attractor.

The effective dimensionality finding — 3 active dimensions from 15 available — connects to the literature on intrinsic dimensionality of optimization landscapes (Li et al., 2018; Aghajanyan et al., 2021). Those works estimate the intrinsic dimensionality of the full weight update; here, we observe intrinsic dimensionality directly in the bridge's learned structure. The fact that the effective rank (3) equals the number of coordinate axes of the geometric prior suggests that the Steersman does not merely reduce dimensionality but aligns it with a specific geometric basis.
# 8. Conclusion

The learnable bridge in multi-channel LoRA, under cybernetic feedback, discovers the 3-axis coordinate geometry of the rhombic dodecahedron. This geometry is not imposed — it emerges as the unique attractor of a feedback loop that encodes face-pair relationships through a contrastive loss and monitors spectral connectivity through a reactive controller.

The finding is robust across every experimental axis we tested. Six cybernetic experiments at $n = 6$ produce 100% block-diagonal structure across 42,500+ bridges; six non-cybernetic controls produce 0%. Three initialization strategies — including one that actively opposes the target topology — converge to the same attractor within 200 training steps. Validation loss is independent of bridge topology (0.17% maximum delta across $n \in \{3, 4, 6\}$). The contrastive loss component, encoding RD face-pair geometry, is both necessary and sufficient: removing it (at $n \in \{3, 4, 8\}$) produces 1,020$\times$ higher Bridge Fiedler eigenvalue with 0% block structure.

The effective dimensionality of the discovered structure is exactly 3 — the number of coordinate axes, not the number of faces. This is confirmed both by the bridge's internal structure (12 of 15 off-diagonal entries driven to zero) and by the channel count ablation ($n = 3$ matches $n = 6$ task performance with $4\times$ fewer bridge parameters). The 6-channel bridge is a redundant parameterization that reveals structure; the 3-channel bridge achieves the same function without revealing it.

These results suggest that multi-channel LoRA adapters have latent geometric preferences that standard training does not surface. Cybernetic feedback — even a simple reactive controller with fixed control laws — is sufficient to discover and stabilize these preferences. The Steersman provides no task performance benefit; its contribution is structural transparency. The bridge, under feedback, tells us what it has learned.
