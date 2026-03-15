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
