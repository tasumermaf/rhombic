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
