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


### 5.2.1 Task Performance Is Insensitive to Channel Count

Validation loss at 10,000 steps varies by at most 0.17% across $n \in \{3, 4, 6\}$: from 0.4015 ($n = 6$) to 0.4022 ($n = 4$). The $n = 8$ experiment, which uses the most bridge parameters (64 per module), achieves 0.4022 at comparable training progress — identical to $n = 4$. Block-diagonal structure provides no measurable performance benefit. Bridge topology is a structural signature, not a performance feature.

The performance equivalence is tighter than might be expected. Over the full 100-checkpoint training trajectory, $n = 3$ and $n = 6$ maintain a maximum validation loss delta of 0.16%. This is within the stochastic variation expected between runs with different random initializations of the LoRA parameters (which are shared across channel counts). The task gradient dominates the language modeling loss surface; bridge topology operates in an orthogonal subspace.


### 5.2.2 Block-Diagonal Structure Emerges Only When the Geometric Prior Is Active

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
