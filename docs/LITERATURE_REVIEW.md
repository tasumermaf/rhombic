# RhombiLoRA Literature Review: Deep Research Survey

> **Date:** March 8, 2026
> **Scope:** Papers and repositories relevant to RhombiLoRA's research program
> **Coverage:** 2023-2026, prioritizing 2024-2026
> **Sources:** HuggingFace Papers, arXiv, Semantic Scholar, web search

---

## Table of Contents

1. [Tier 1: Directly Actionable Papers](#tier-1-directly-actionable)
2. [Tier 2: Strong Architectural Relevance](#tier-2-strong-architectural-relevance)
3. [Tier 3: Methodological Foundations](#tier-3-methodological-foundations)
4. [Tier 4: Contextual / Background](#tier-4-contextual--background)
5. [Gap Analysis](#gap-analysis)
6. [Actionable Insights by Experiment](#actionable-insights-by-experiment)

---

## Tier 1: Directly Actionable

These papers have immediate, concrete applicability to RhombiLoRA experiments.

### 1.1 LoRAN: Enhancing Low-Rank Adaptation with Structured Nonlinear Transformations
- **Authors:** Aochuan Chen et al.
- **Venue:** EMNLP 2024 Findings; extended version arXiv Sep 2025
- **Links:** [arXiv](https://arxiv.org/abs/2509.21870) | [EMNLP](https://aclanthology.org/2024.findings-emnlp.177.pdf)
- **Core contribution:** Introduces a lightweight nonlinear transformation (Sinter, a sine-based activation) **between LoRA's A and B projection matrices**. This is the closest existing work to RhombiLoRA's bridge concept. Sinter adds structured perturbations without increasing parameter count. Improves ROUGE/accuracy by 0.47/0.79 points on 7B+ models.
- **Relevance to RhombiLoRA:** **CRITICAL.** LoRAN proves that inserting a transformation between A and B projections is beneficial. RhombiLoRA's 6-channel geometric bridge is a specific instantiation of this idea with crystallographic structure rather than a generic sine activation. LoRAN is the primary comparison baseline.
- **Actionable insight:** Benchmark RhombiLoRA against LoRAN directly. The Sinter activation is parameter-free; RhombiLoRA's bridge adds 0.03% parameters but with geometric structure. Test whether the FCC topology of the bridge outperforms generic nonlinearity.

### 1.2 DoRA: Weight-Decomposed Low-Rank Adaptation
- **Authors:** Shih-Yang Liu et al. (NVIDIA)
- **Venue:** ICML 2024 (Oral)
- **Links:** [arXiv](https://arxiv.org/abs/2402.09353) | [GitHub](https://github.com/NVlabs/DoRA)
- **Core contribution:** Decomposes weight updates into magnitude and direction components, applying LoRA only to the directional matrix. Shows that LoRA fails to make subtle directional changes that full fine-tuning achieves. Outperforms LoRA at half the rank.
- **Relevance to RhombiLoRA:** DoRA's magnitude/direction decomposition is orthogonal to RhombiLoRA's bridge. The two could be combined: DoRA for the A/B decomposition, RhombiLoRA bridge for the directional update. DoRA's finding that **directional updates are the bottleneck** directly supports the hypothesis that a geometrically structured bridge (which operates on directions via its 6 FCC channels) would be beneficial.
- **Actionable insight:** Test DoRA + RhombiLoRA bridge as a combined method. If the bridge improves directional updates specifically, this combination could be synergistic.

### 1.3 RiemannLoRA: A Unified Riemannian Framework for LoRA Optimization
- **Authors:** Bogachev et al.
- **Venue:** arXiv Jul 2025
- **Links:** [HF Papers](https://hf.co/papers/2507.12142)
- **Core contribution:** Treats LoRA matrices as elements on a smooth manifold. Uses Riemannian optimization to eliminate overparameterization and find optimal initialization. Consistently improves convergence speed and final performance on both LLMs and diffusion models.
- **Relevance to RhombiLoRA:** The Riemannian perspective validates treating LoRA's parameter space geometrically. RhombiLoRA's bridge matrix lives on a specific geometric manifold (FCC lattice topology). RiemannLoRA's manifold is generic (fixed-rank matrices); RhombiLoRA's is crystallographic. The theoretical framework of RiemannLoRA could be adapted to analyze the bridge's geometric properties.
- **Actionable insight:** Use RiemannLoRA's manifold analysis tools to characterize the bridge matrix's learned geometry. Does the bridge converge to a point on a specific sub-manifold of the FCC topology space?

### 1.4 EMoE: Eigenbasis-Guided Routing for Mixture-of-Experts
- **Authors:** Cheng et al.
- **Venue:** arXiv Jan 2026
- **Links:** [arXiv](https://arxiv.org/abs/2601.12137) | [GitHub](https://github.com/Belis0811/EMoE)
- **Core contribution:** Routes tokens to MoE experts based on alignment with a learned orthonormal eigenbasis. Geometric partitioning of input space intrinsically promotes both balanced utilization and diverse specialization, without auxiliary loss functions.
- **Relevance to RhombiLoRA:** **Directly relevant to Exp 3A/3C.** EMoE proves that geometric routing works for MoE. RhombiLoRA's Exp 3A (bridge between MoE experts) could use a similar eigenbasis approach but with the bridge matrix encoding the FCC topology between experts. EMoE's geometric partitioning is isotropic (orthonormal basis); RhombiLoRA's would be anisotropic (FCC-derived).
- **Actionable insight:** For Exp 3A, compare FCC-topology bridge routing against EMoE's eigenbasis routing. The bridge gives directional structure (6 direction pairs); EMoE gives orthonormal partitioning. These may be complementary.

### 1.5 Chain-of-Experts: Sequential Expert Communication
- **Authors:** Wang et al.
- **Venue:** arXiv Jun 2025
- **Links:** [HF Papers](https://hf.co/papers/2506.18945) | [GitHub](https://github.com/ZihanWang314/coe)
- **Core contribution:** Tokens pass through a chain of experts sequentially within a layer, with re-routing at each step. Introduces expert communication as a scaling axis (depth through iteration). 2x iterations matches 3x expert selections (width) while reducing memory by 17-42%.
- **Relevance to RhombiLoRA:** **Directly relevant to Exp 3A.** Chain-of-Experts establishes that inter-expert communication matters. RhombiLoRA's bridge between experts is a different communication mechanism: instead of sequential chaining, it provides a geometric coupling matrix derived from FCC adjacency. CoE communicates via residual iteration; RhombiLoRA would communicate via learned bridge weights on lattice edges.
- **Actionable insight:** Compare bridge-mediated expert communication against CoE's sequential communication. They may be combinable: experts on FCC nodes, with CoE-style iteration along lattice edges weighted by the bridge.

---

## Tier 2: Strong Architectural Relevance

### 2.1 Modality Gap-Driven Subspace Alignment (ReVision)
- **Authors:** Yu et al.
- **Venue:** arXiv Feb 2026
- **Links:** [HF Papers](https://hf.co/papers/2602.07026)
- **Core contribution:** Precisely characterizes the geometric shape of the modality gap between text and image embeddings. Proposes ReAlign: a three-step process (Anchor, Trace, Centroid Alignment) to bridge modalities without paired data. Demonstrates that statistically aligned unpaired data substitutes for expensive paired datasets.
- **Relevance to RhombiLoRA:** **Directly relevant to Exp 3B (cross-modal bridge).** ReVision shows the modality gap has specific geometric structure (anisotropic, with stable biases in a fixed frame). RhombiLoRA's cross-modal bridge needs to account for this geometry. The 6 FCC channels might correspond to principal directions of the modality gap.
- **Actionable insight:** Use ReVision's Fixed-frame Modality Gap Theory to initialize the cross-modal bridge. Align the 6 bridge channels with the principal components of the measured modality gap.

### 2.2 Latent Space Translation via Inverse Relative Projection
- **Authors:** Maiorca et al.
- **Venue:** arXiv Jun 2024
- **Links:** [HF Papers](https://hf.co/papers/2406.15057)
- **Core contribution:** Translates between latent spaces using a relative (angle-preserving) intermediate space. Demonstrates zero-shot stitching between arbitrary pre-trained text and image encoders, even across modalities.
- **Relevance to RhombiLoRA:** Validates the concept of an intermediate bridging space for cross-modal alignment (Exp 3B). The relative space is angle-preserving; RhombiLoRA's bridge is topology-preserving (FCC structure). These are different geometric invariants and could be compared.
- **Actionable insight:** Test whether the FCC bridge achieves angle-preservation as an emergent property, or whether explicit angle-preservation constraints improve the bridge.

### 2.3 TextME: Bridging Unseen Modalities Through Text
- **Authors:** Hong et al.
- **Venue:** arXiv Feb 2026
- **Links:** [HF Papers](https://hf.co/papers/2602.03098)
- **Core contribution:** Exploits the geometric structure of pre-trained contrastive encoders to enable zero-shot cross-modal transfer using only text descriptions. Demonstrates that consistent "modality gaps" exist across image, video, audio, 3D, X-ray, and molecular domains.
- **Relevance to RhombiLoRA:** Confirms that modality gaps are structurally consistent and predictable -- exactly the kind of regularity a geometric bridge should exploit. The universality of the gap pattern supports the idea that a small, structured bridge (6 FCC channels) could capture the essential transformation.
- **Actionable insight:** If the modality gap is low-dimensional and consistent, 6 channels may be sufficient. Measure the effective dimensionality of the modality gap to validate the 6-channel bridge capacity.

### 2.4 LaVi-Bridge: Bridging Language and Vision Models
- **Authors:** Zhao et al.
- **Venue:** arXiv Mar 2024
- **Links:** [HF Papers](https://hf.co/papers/2403.07860) | [GitHub](https://github.com/ShihaoZhaoZSH/LaVi-Bridge)
- **Core contribution:** Uses LoRA and adapters to bridge arbitrary pre-trained language models with generative vision models for text-to-image generation. Plug-and-play without modifying original weights.
- **Relevance to RhombiLoRA:** Direct proof-of-concept for the LoRA-based cross-modal bridge idea (Exp 3B). LaVi-Bridge uses generic LoRA; RhombiLoRA would replace this with a geometrically structured bridge.
- **Actionable insight:** Use LaVi-Bridge as baseline for Exp 3B. Replace their generic adapters with RhombiLoRA bridges and measure whether FCC topology improves alignment quality.

### 2.5 MoE-LoRA with Riemannian Preconditioners
- **Authors:** Sun et al.
- **Venue:** arXiv Feb 2025
- **Links:** [HF Papers](https://hf.co/papers/2502.15828) | [GitHub](https://github.com/THUDM/MoELoRA_Riemannian)
- **Core contribution:** Stabilizes MoE-LoRA training by treating each LoRA expert as a sub-space projector and using Riemannian preconditioners for multi-space projections. Demonstrates improved robustness.
- **Relevance to RhombiLoRA:** Validates geometric treatment of MoE-LoRA (relevant to Exp 3A). The sub-space projector view aligns with RhombiLoRA's bridge as a mapping between expert sub-spaces organized on an FCC lattice.
- **Actionable insight:** Apply Riemannian preconditioners to RhombiLoRA bridge training for stability.

### 2.6 CDSP-MoE: Gradient Conflict-Driven Subspace Topology Pruning
- **Authors:** Gan & Lei
- **Venue:** arXiv Dec 2025
- **Links:** [HF Papers](https://hf.co/papers/2512.20291) | [GitHub](https://github.com/konodiodaaaaa1/Conflict-Driven-Subspace-Pruning-Mixture-of-Experts)
- **Core contribution:** Shifts from isolated expert containers to dynamic expert instantiation within a shared physical subspace. Uses gradient conflict as a structural supervisory signal to prune conflicting pathways, enabling topology to spontaneously evolve interpretable modular structures.
- **Relevance to RhombiLoRA:** Demonstrates that MoE topology can be learned from gradient signals. In Exp 3C (Large Geometric Model), the question is whether FCC topology is optimal or whether the topology should be learned. CDSP-MoE suggests a hybrid: start with FCC topology and allow gradient-driven pruning to refine it.
- **Actionable insight:** For Exp 3C, initialize with FCC topology and add a topology refinement mechanism inspired by CDSP-MoE's gradient conflict pruning. Compare FCC-fixed vs. FCC-initialized-then-refined.

### 2.7 LiON-LoRA: Orthogonality and Norm Consistency in Video Diffusion LoRA Fusion
- **Authors:** Zhang et al.
- **Venue:** arXiv Jul 2025
- **Links:** [HF Papers](https://hf.co/papers/2507.05678)
- **Core contribution:** Rethinks LoRA fusion for video diffusion through three principles: Linear scalability, Orthogonality, and Norm consistency. Analyzes orthogonality of LoRA features in shallow layers to enable decoupled controllability.
- **Relevance to RhombiLoRA:** **Directly relevant to Exp 3A (Wan 2.2).** LiON-LoRA works specifically on video diffusion models and provides principles (orthogonality, norm consistency) for LoRA fusion that the RhombiLoRA bridge should respect.
- **Actionable insight:** Ensure the RhombiLoRA bridge for Wan 2.2 maintains orthogonality between its 6 directional channels. Use LiON-LoRA's norm consistency criterion as a regularization loss.

---

## Tier 3: Methodological Foundations

### 3.1 Eigenspectrum Analysis (FARMS)
- **Authors:** Hu et al.
- **Venue:** arXiv Jun 2025
- **Links:** [HF Papers](https://hf.co/papers/2506.06280)
- **Core contribution:** Addresses aspect ratio bias in weight matrix spectral analysis. Proposes Fixed-Aspect-Ratio Matrix Subsampling (FARMS) for accurate heavy-tailness estimation. Reduces LLaMA-7B perplexity by 17.3% in pruning experiments.
- **Relevance:** Your Fiedler value measurements (1.73x FCC vs cubic) may be subject to aspect ratio bias if bridge matrices have different dimensions. FARMS provides the diagnostic methodology.
- **Actionable insight:** Apply FARMS to bridge matrix spectral analysis to confirm the 1.73x Fiedler value advantage is not an artifact of matrix dimensions.

### 3.2 Spectral Scaling Laws in LLMs
- **Authors:** Jha & Reagen
- **Venue:** arXiv Oct 2025
- **Links:** [HF Papers](https://hf.co/papers/2510.00537)
- **Core contribution:** Discovers asymmetric spectral scaling: soft rank follows a power law with FFN width, while hard rank grows sublinearly. Widening FFNs mostly adds low-energy tail directions while dominant-mode subspaces saturate.
- **Relevance:** The bridge matrix's 6 channels should show spectral utilization patterns. If FCC topology concentrates energy into fewer but more effective modes (higher Fiedler value = better connectivity), the spectral signature should differ from cubic topology.
- **Actionable insight:** Measure hard rank, soft rank, and spectral concentration of bridge matrices for FCC vs cubic topologies. FCC should show higher spectral utilization if the topology hypothesis is correct.

### 3.3 SETOL: Semi-Empirical Theory of Learning (Heavy-Tailed Self-Regularization)
- **Authors:** Martin & Hinrichs
- **Venue:** arXiv Jul 2025
- **Links:** [HF Papers](https://hf.co/papers/2507.17912)
- **Core contribution:** Formal derivation of heavy-tailed power-law layer quality metrics (alpha, alpha-hat) that predict test accuracy without test data. Introduces ERG (Exact Renormalization Group) as a layer quality metric.
- **Relevance:** HTSR metrics could be applied to diagnose bridge matrix quality. A well-trained bridge should show specific spectral signatures. The alpha metric could indicate whether FCC topology produces better-conditioned bridge matrices.
- **Actionable insight:** Compute alpha-hat for bridge matrices across training. If FCC bridges show more favorable alpha-hat trajectories, this provides a theoretical explanation for the connectivity advantage.

### 3.4 Spectral Geometry for Deep Learning
- **Authors:** Ettori
- **Venue:** arXiv Jan 2026
- **Links:** [HF Papers](https://hf.co/papers/2601.17357)
- **Core contribution:** Unified framework based on spectral geometry and random matrix theory for both hallucination detection (EigenTrack) and model compression (RMT-KD) using eigenvalue structure of hidden activations.
- **Relevance:** Provides the theoretical scaffolding for interpreting bridge matrix spectra. EigenTrack's approach of monitoring spectral features over time could be adapted to track bridge matrix evolution during training.
- **Actionable insight:** Apply EigenTrack-style spectral monitoring to bridge matrices during training. Track whether FCC topology produces more stable spectral evolution.

### 3.5 Representation Geometry Predicts Generalization
- **Authors:** Yadav
- **Venue:** arXiv Jan 2026
- **Links:** [HF Papers](https://hf.co/papers/2602.00130)
- **Core contribution:** Shows that effective dimension (an unsupervised geometric metric) predicts accuracy with partial r=0.75 across 52 models and 13 architectures. Establishes bidirectional causality between geometry and performance.
- **Relevance:** Effective dimension of representations passing through the bridge could predict whether the bridge is helping. If FCC-bridged models show higher effective dimension in their representations, this would validate the geometric hypothesis.
- **Actionable insight:** Measure effective dimension before and after the bridge for FCC vs cubic topologies. Higher post-bridge effective dimension = better geometric transformation.

### 3.6 Lattice Physics Approaches for Neural Networks
- **Authors:** Multiple (ScienceDirect / PMC)
- **Venue:** iScience, Dec 2024
- **Links:** [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11638618/) | [arXiv](https://arxiv.org/abs/2405.12022)
- **Core contribution:** Mathematical framework describing neural network interactions using lattice field theory from particle physics. Shows that cortical recordings behave as renormalized field theories, with renormalization by decimation connecting lattice scales.
- **Relevance:** Provides theoretical grounding for the idea that neural networks can be productively modeled on lattices. The FCC lattice in RhombiLoRA is not just an architectural choice -- it connects to the deepest theoretical framework for understanding spatially structured computation.
- **Actionable insight:** Frame the FCC bridge as a specific lattice field theory with 12 nearest neighbors (FCC coordination number). The bridge weights are coupling constants on the lattice.

### 3.7 A Spectral Condition for Feature Learning
- **Authors:** Yang, Simon & Bernstein
- **Venue:** arXiv Oct 2023
- **Links:** [HF Papers](https://hf.co/papers/2310.17813)
- **Core contribution:** Shows that feature learning is achieved by scaling the spectral norm of weight matrices like fan-out/fan-in. Provides elementary derivation of maximal update parametrization (muP).
- **Relevance:** Bridge matrix initialization and scaling should follow spectral norm principles. The fan-out/fan-in ratio for a 6-channel FCC bridge is specific and should be set correctly for feature learning to occur in the bridge.
- **Actionable insight:** Ensure bridge matrix initialization follows the spectral norm scaling condition. Compute fan-out/fan-in for the FCC topology and set initialization accordingly.

---

## Tier 4: Contextual / Background

### 4.1 Unified Study of LoRA Variants (LoRAFactory)
- **Authors:** He et al.
- **Venue:** arXiv Jan 2026
- **Links:** [HF Papers](https://hf.co/papers/2601.22708)
- **Key finding:** With proper hyperparameters, vanilla LoRA matches or surpasses most variants. Learning rate is the most sensitive hyperparameter.
- **Implication:** RhombiLoRA must demonstrate improvements beyond what hyperparameter tuning of vanilla LoRA achieves. The bar is higher than it appears.

### 4.2 RandLoRA: Full-Rank Updates via Random Matrices
- **Authors:** Albert et al.
- **Venue:** arXiv Feb 2025
- **Links:** [HF Papers](https://hf.co/papers/2502.00987)
- **Key finding:** Full-rank updates (via learned linear combinations of random matrices) eliminate the performance gap between LoRA and full fine-tuning, especially for vision-language tasks.
- **Implication:** The bridge matrix in RhombiLoRA could serve a similar function -- increasing effective rank through geometric structure rather than random projections.

### 4.3 Null-LoRA: Low-Rank Adaptation on Null Space
- **Authors:** Zhang et al.
- **Venue:** arXiv Dec 2025
- **Links:** [HF Papers](https://hf.co/papers/2512.15233)
- **Key finding:** Constraining LoRA updates to the null space of pre-trained weights reduces redundancy and enhances effective rank.
- **Implication:** The FCC bridge might naturally project updates toward less redundant subspaces (higher algebraic connectivity = better information flow = less redundancy).

### 4.4 Flora: Low-Rank Adapters Are Secretly Gradient Compressors
- **Authors:** Hao et al.
- **Venue:** arXiv Feb 2024
- **Links:** [HF Papers](https://hf.co/papers/2402.03293)
- **Key finding:** LoRA can be approximated by random projection. Resampling projection matrices during training achieves high-rank updates.
- **Implication:** Understanding LoRA as gradient compression suggests the bridge matrix might function as a structured (non-random) gradient compressor.

### 4.5 Representation Collapse in Sparse MoE (Hypersphere Routing)
- **Authors:** Chi et al.
- **Venue:** arXiv 2022
- **Links:** [HF Papers](https://hf.co/papers/2204.09179)
- **Key finding:** MoE routing encourages token clustering around expert centroids, causing representation collapse. Routing on a low-dimensional hypersphere alleviates this.
- **Implication:** For Exp 3C, FCC lattice nodes provide a specific non-hyperspherical geometry for routing. The rhombic dodecahedron (Voronoi cell of FCC) provides a natural partitioning of representation space.

### 4.6 From Bricks to Bridges: Product of Invariances for Latent Space Communication
- **Authors:** Cannistraci et al.
- **Venue:** arXiv Oct 2023
- **Links:** [HF Papers](https://hf.co/papers/2310.01211)
- **Key finding:** Incorporating geometric invariances directly into representations enables zero-shot stitching across modalities and architectures.
- **Implication:** The FCC bridge encodes a specific set of geometric invariances (12-fold coordination, rhombic symmetry). These could serve as the invariance product for cross-modal stitching (Exp 3B).

### 4.7 Knowledge Composition via Learned Anisotropic Scaling (aTLAS)
- **Authors:** Zhang et al.
- **Venue:** arXiv Jul 2024
- **Links:** [HF Papers](https://hf.co/papers/2407.02880)
- **Key finding:** Task vectors can be composed via learned anisotropic (directional) scaling of parameter blocks. Anisotropic scaling produces more disentangled task vectors.
- **Implication:** The 6 FCC channels in the bridge ARE anisotropic scaling channels. If they correspond to the 6 co-planar/cross-planar direction pairs, they provide a principled decomposition for anisotropic task adaptation.

### 4.8 The Standing Committee in MoE Models
- **Authors:** Wang et al.
- **Venue:** arXiv Jan 2026
- **Links:** [HF Papers](https://hf.co/papers/2601.03425)
- **Key finding:** MoE models don't truly specialize -- a compact "Standing Committee" of experts handles most routing mass across all domains. Load-balancing losses work against the model's natural optimization.
- **Implication:** For Exp 3C, FCC topology might provide a more natural structure than forced load balancing. If some FCC nodes naturally become Standing Committee members, the topology should support this rather than fight it.

---

## Gap Analysis

### What Exists
- **LoRA + intermediate transformation:** LoRAN proves the concept works (Tier 1.1)
- **Geometric MoE routing:** EMoE and ERMoE prove eigenbasis-guided routing works (Tier 1.4)
- **Cross-modal bridges via LoRA:** LaVi-Bridge proves the pattern works (Tier 2.4)
- **Video diffusion + LoRA fusion:** LiON-LoRA provides principles (Tier 2.7)
- **Spectral analysis of NN weights:** FARMS, SETOL, HTSR provide tools (Tier 3)
- **Modality gap geometry:** ReVision precisely characterizes it (Tier 2.1)

### What Does NOT Exist (RhombiLoRA's Novelty)
1. **Crystallographic topology in adapter design.** No paper uses FCC, BCC, or any sphere-packing lattice as the organizing topology for a LoRA variant. This is genuinely novel.
2. **Direction-pair decomposition in LoRA.** While aTLAS uses anisotropic scaling, no one decomposes the bridge into co-planar vs. cross-planar direction pairs derived from a specific polyhedron.
3. **Fiedler value as a design criterion for adapter topology.** Algebraic connectivity is used extensively in GNNs and network science but has never been applied to choose adapter topology.
4. **Lattice-organized MoE.** Experts on FCC lattice nodes with bridge weights on lattice edges has no precedent.
5. **Training data designed to activate directional structure in adapters.** Purpose-built geometric training data to strengthen directional signals in the bridge is entirely novel.

### RhombiLoRA's Position in the Landscape
RhombiLoRA sits at the intersection of three active research directions that have not been connected:
- **Structured LoRA** (LoRAN, DoRA, RiemannLoRA) -- internal structure of adapters
- **Geometric MoE** (EMoE, CoE, CDSP-MoE) -- geometric organization of experts
- **Spectral network analysis** (FARMS, SETOL, spectral scaling laws) -- diagnostic tools

The contribution is the crystallographic bridge: using sphere-packing geometry to provide both the internal structure of adapters AND the topology of expert communication. No existing work makes this connection.

---

## Actionable Insights by Experiment

### Exp 2.5: Geometric Training Data for Directional Signal

| Source | Insight | Priority |
|--------|---------|----------|
| Euclid (Tier 4) | Synthetic geometric data + multi-stage curriculum enables learning of geometry tasks that fail from scratch. Use curriculum: simple geometric patterns first, then complex. | HIGH |
| EfficientTrain (Tier 4) | Start with low-frequency components of training data (easier patterns), progressively introduce high-frequency. Apply to geometric data design. | MEDIUM |
| aTLAS (Tier 4.7) | Anisotropic scaling produces more disentangled representations. Design training data that explicitly requires different FCC direction pairs to carry different signals. | HIGH |
| Spectral Scaling Laws (Tier 3.2) | Measure spectral utilization of bridge channels during training. Geometric data should produce higher spectral utilization (more channels meaningfully activated) than generic data. | HIGH |

**Specific recommendation:** Create training data with 6 classes of geometric transformations, each aligned to one FCC direction pair. Train and measure per-channel activation. Compare against random/generic data. The co-planar vs. cross-planar signal (currently 1.02x) should strengthen dramatically.

### Exp 3A: RhombiLoRA Bridge for MoE (Wan 2.2)

| Source | Insight | Priority |
|--------|---------|----------|
| EMoE (Tier 1.4) | Use eigenbasis-derived routing as the comparison baseline. FCC bridge provides anisotropic routing where EMoE provides isotropic. | HIGH |
| Chain-of-Experts (Tier 1.5) | Expert communication along FCC edges is an alternative to sequential chaining. Can combine: iterative message passing along FCC lattice edges. | HIGH |
| LiON-LoRA (Tier 2.7) | Ensure orthogonality between bridge channels and norm consistency across layers when applying to video diffusion. | HIGH |
| Standing Committee (Tier 4.8) | Don't force load balancing. Let FCC topology determine natural expert utilization patterns. | MEDIUM |
| CDSP-MoE (Tier 2.6) | Allow gradient-driven topology refinement on top of FCC initialization. | MEDIUM |

**Specific recommendation:** For Wan 2.2, replace the standard MoE router with an FCC-topology bridge. Each expert sits on an FCC lattice node. The bridge matrix encodes coupling between adjacent experts. Use LiON-LoRA's orthogonality principle for the 6 direction-pair channels. Compare against EMoE's eigenbasis routing and CoE's sequential chaining.

### Exp 3B: Cross-Modal Bridge (LLM + Image Model)

| Source | Insight | Priority |
|--------|---------|----------|
| ReVision (Tier 2.1) | Modality gap has specific geometric structure. Initialize bridge channels along principal gap directions. | CRITICAL |
| LaVi-Bridge (Tier 2.4) | Use as direct baseline. Replace generic LoRA bridge with RhombiLoRA bridge. | HIGH |
| TextME (Tier 2.3) | Modality gaps are consistent across domains. 6 channels may be sufficient if aligned to gap structure. | HIGH |
| Latent Space Translation (Tier 2.2) | Test whether bridge learns angle-preservation as emergent property. | MEDIUM |
| From Bricks to Bridges (Tier 4.6) | FCC invariances as the product of invariances for cross-modal stitching. | MEDIUM |

**Specific recommendation:** Measure the modality gap using ReVision's Fixed-frame Theory. Decompose into principal components. Initialize the 6 bridge channels along the top 6 principal directions of the gap. Train the bridge to transform text representations into image representation space. Compare against LaVi-Bridge's generic LoRA approach.

### Exp 3C: Large Geometric Model (FCC-lattice MoE)

| Source | Insight | Priority |
|--------|---------|----------|
| EMoE (Tier 1.4) | Eigenbasis routing for expert specialization without load-balancing loss. | HIGH |
| CDSP-MoE (Tier 2.6) | Gradient conflict as structural signal for topology refinement. | HIGH |
| CoE (Tier 1.5) | Sequential communication as depth scaling axis. | HIGH |
| MoUE (Tier 4) | Universal expert pool with Staggered Rotational Topology for structured sharing. | MEDIUM |
| Representation Collapse (Tier 4.5) | Hypersphere routing alleviates collapse; FCC Voronoi cells provide alternative partitioning geometry. | MEDIUM |
| Lattice Physics (Tier 3.6) | Theoretical framework: experts as lattice field theory variables, bridge weights as coupling constants. | MEDIUM |

**Specific recommendation:** Design the Large Geometric Model as follows: N experts on FCC lattice nodes (start with N=12, the coordination number). Each expert has 12 neighbors. Bridge weights on lattice edges encode inter-expert coupling. Use eigenbasis-guided routing within each expert's Voronoi cell (rhombic dodecahedron). Allow gradient conflict to prune weak edges. Measure Fiedler value of the resulting topology throughout training.

---

## Key Repositories to Monitor

| Repository | Relevance | URL |
|------------|-----------|-----|
| DoRA (NVIDIA) | Baseline comparison | https://github.com/NVlabs/DoRA |
| EMoE | Geometric MoE routing | https://github.com/Belis0811/EMoE |
| Chain-of-Experts | Expert communication | https://github.com/ZihanWang314/coe |
| LaVi-Bridge | Cross-modal bridge baseline | https://github.com/ShihaoZhaoZSH/LaVi-Bridge |
| CDSP-MoE | Topology pruning | https://github.com/konodiodaaaaa1/Conflict-Driven-Subspace-Pruning-Mixture-of-Experts |
| LoRAFactory | Unified LoRA benchmarking | (from HF paper 2601.22708) |
| MoELoRA Riemannian | Riemannian MoE-LoRA | https://github.com/THUDM/MoELoRA_Riemannian |
| FlyLoRA | Bio-inspired implicit MoE | https://github.com/gfyddha/FlyLoRA |

---

*Survey conducted March 8, 2026. 50+ papers reviewed across HuggingFace Papers, arXiv, and web sources. Focused on 2023-2026 literature with direct or methodological relevance to RhombiLoRA's research program.*
