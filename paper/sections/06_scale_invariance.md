# 6. Scale Invariance

The experiments in Sections 4 and 5 establish block-diagonal emergence and its mechanistic basis using TinyLlama 1.1B as the primary model. This section examines whether the findings generalize across model scale, using Qwen2.5 7B and Wan 2.1 14B as additional test beds.

## 6.1 Cross-Scale Comparison

Block-diagonal structure appears identically at 1.1B (TinyLlama) and 7B (Qwen2.5). The Qwen 7B experiment (exp3) uses identity initialization and the Steersman with the same feedback protocol. Over 12,900 training steps, it produces 100% block-diagonal bridges with co-planar/cross-planar ratio 18,248:1. The lower ratio compared to TinyLlama runs (64,000–71,000:1 at 10K) reflects an earlier version of the training code with less aggressive contrastive weighting, not a scale-dependent effect.

Per-layer coupling analysis reveals comparable structure across scales. In both models, co-planar coupling is uniform across layers to within 5%, with no systematic front-to-back gradient. Both models exhibit the same per-module hierarchy: $k$\_proj develops the strongest coupling (44,000:1 in Qwen 7B; 46,570:1 in C-002 TinyLlama), followed by $v$\_proj (44,000:1; 24,462:1), then $q$\_proj (22,477:1; 16,708:1), with $o$\_proj weakest (19,080:1; 9,116:1). The ordering $k > v > q > o$ is consistent across scales, suggesting an architectural invariant: key and value projections, which operate on the attention mechanism's representational bottleneck, develop stronger inter-channel coupling than the output projection, which acts downstream of attention.


## 6.2 Two Fiedler Metrics

The analysis employs two distinct applications of the Fiedler eigenvalue that must be carefully distinguished.

**Bridge Fiedler** measures *within-matrix* connectivity. For a single bridge matrix $\mathbf{M}$, the weighted Laplacian treats $|M_{ij}|$ as edge weights and computes $\lambda_2$ of the resulting graph. A low Bridge Fiedler indicates near-disconnected components — blocks with minimal cross-block coupling. Bridge Fiedler at $n = 6$ (cybernetic): 0.00009. Bridge Fiedler at $n = 4$ (spectral-only): 0.092.

**Correlation Fiedler** measures *cross-layer* structural consistency. For each module type (e.g., $q$\_proj), the flattened bridge matrices across all layers form a set of $n^2$-dimensional vectors. The pairwise Pearson correlation matrix is constructed, and its Fiedler eigenvalue measures how uniformly the bridge structure propagates across the model's depth. A high Correlation Fiedler indicates that all layers develop the same bridge pattern; a low value indicates layer-specific variation.

The Correlation Fiedler converges to ${\sim}0.10$ for cybernetic text models: 0.102 (Qwen 7B) and 0.101 (TinyLlama). The Holly Battery 14B (non-cybernetic) has Correlation Fiedler 1.002---near-perfect cross-layer correlation arising from its near-identity bridges. The ${\sim}0.10$ convergence for cybernetic models indicates that bridge structure is uniform across layers regardless of model depth or size, while the Holly value reflects a qualitatively different regime where minimal bridge perturbations are trivially correlated.


## 6.3 The Holly Battery Null Result

The Holly Battery experiment provides the non-cybernetic control at 14B scale. Wan 2.1, a 14B-parameter video diffusion backbone, is fine-tuned with 6-channel RhombiLoRA but without the Steersman (no contrastive loss, no spectral loss, no feedback controller). The adapter achieves 3.8% lower loss than the standard LoRA baseline, requires 9.15 GB less VRAM (gradient checkpointing benefits from the factored bridge), and produces 6% faster inference.

Despite these performance benefits, the Holly Battery bridge matrices are near-identity: mean off-diagonal magnitude 0.010, co-planar/cross-planar ratio 1.07:1, Bridge Fiedler $\sim$0.10 (well-connected, no block structure). The bridge learns small perturbations from identity that improve task performance, but these perturbations are uniformly distributed across all off-diagonal entries with no preferential topology.

The Holly result rules out model scale, dataset, and optimizer as confounders. The 14B model has substantially greater representational capacity than TinyLlama (1.1B). The video generation task is fundamentally different from instruction-following. The optimizer (Prodigy, with adaptive learning rate) is more sophisticated than the AdamW used in all other experiments. None of these factors produces block-diagonal structure. The Steersman is the causal factor.
