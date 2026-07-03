# Research Suite — Master Index

**Papers:** 345 · **Non-paper resources:** 97 · Built by `tools/build_suite.py` from `data/papers.json`.

Navigation for LLMs and humans: this index → per-paper cards in `papers/` (metadata + abstract + every program assessment, verbatim) → `SYNTHESIS.md` for cross-cutting conclusions → `tools/fetch_papers.py` to materialize full-text PDFs locally.

## Bridge-Matrix Lineage (between A and B)

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| A Unified Study of LoRA Variants: Taxonomy, Review, Codebase, and Empirical Eval | 2601.22708 | arXiv Jan 2026 | Hierarchical LoRA taxonomy; no mention of spectral topology, | [card](papers/arxiv-2601-22708.md) |
| AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning | 2303.10512 | arXiv preprint | cited-in-paper-3; cited-in-paper-4 | [card](papers/arxiv-2303-10512.md) |
| BoRA: Towards More Expressive Low-Rank Adaptation with Block Diversity | 2508.06953 | 2025-08-09 | claim-1 lineage (diagonal block bridges) + claim-4 imposed-b | [card](papers/arxiv-2508-06953.md) |
| CRMA: A Spectrally-Bounded Backbone for Modular Continual Fine-Tuning of LLMs | 2606.00382 | 2026-05-29 | claim-2 static-constraint neighbor (Sinkhorn-bounded mixing  | [card](papers/arxiv-2606-00382.md) |
| Cross-LoRA: A Data-Free LoRA Transfer Framework across Heterogeneous LLMs | 2508.05232 | 2025-08-07 | Tier-3 baseline to beat — Frobenius-optimal closed-form map; | [card](papers/arxiv-2508-05232.md) |
| EDoRA: Efficient Weight-Decomposed Low-Rank Adaptation via Singular Value Decomp | 2501.12067 | arXiv Jan 2025 | Closest LoRA-XS follow-up to TeLoRA in architecture; directl | [card](papers/arxiv-2501-12067.md) |
| CeRA: Breaking the Linear Ceiling of Low-Rank Adaptation with Non-linearity Reta | 2602.22911 | ICLR 2026 | Spectral energy entropy for rank allocation — adjacent spect | [card](papers/arxiv-2602-22911.md) |
| GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning | 2505.20355 | NeurIPS 2025 | cited-in-paper-4 | [card](papers/arxiv-2505-20355.md) |
| GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommen | 2606.07526 | 2026-04-20 | claim-1 lineage (trainable message-passing in the LoRA pathw | [card](papers/arxiv-2606-07526.md) |
| ID-LoRA | — | — | claim-1 lineage (frozen-outers family) | [card](papers/id-lora.md) |
| Low-Rank Interconnected Adaptation across Layers | 2407.09946 | 2024-07-13 | claim-1 lineage (routed A–B coupling) | [card](papers/arxiv-2407-09946.md) |
| LoRA-Mini : Adaptation Matrices Decomposition and Selective Training | 2411.15804 | AAAI Workshop 2025 | Same category as LoRA-XS (inner-only training) — detailed co | [card](papers/arxiv-2411-15804.md) |
| LoRA-SB (r x r trainable bridge between frozen SVD) | — | ICLR 2025 Workshop (SCOPE) | MUST CITE — closest to TeLoRA's bridge concept; explicit dif | [card](papers/lora-sb-r-x-r-trainable-bridge-between-frozen-svd.md) |
| LoRA-XS | — | EMNLP 2024 | claim-1 lineage (frozen-outers family) | [card](papers/lora-xs.md) |
| LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters | 2405.17604 | arXiv preprint | cited-in-paper-3; cited-in-paper-4 | [card](papers/arxiv-2405-17604.md) |
| Enhancing Low-Rank Adaptation with Structured Nonlinear Transformations | 2509.21870 | EMNLP 2024 Findings | Modifies A-to-B pathway with fixed non-linearity — detailed  | [card](papers/arxiv-2509-21870.md) |
| Mixture-of-Subspaces in Low-Rank Adaptation | 2406.11909 | EMNLP 2024 | claim-1 strong-form killer — must cite; claim 1 must be rest | [card](papers/arxiv-2406-11909.md) |
| NoRA: Efficient Fine-Tuning of Large Models via Nested Low-Rank Adaptation | — | ICCV 2025 | Bridge-matrix lineage member (nested inner LoRA, frozen oute | [card](papers/nora-efficient-fine-tuning-of-large-models-via-nested-low-ra.md) |
| OrthoGeoLoRA: Geometric Parameter-Efficient Fine-Tuning for Structured Social Sc | 2601.09185 | arXiv | Manifold-constrained LoRA — differentiated: geometry as cons | [card](papers/arxiv-2601-09185.md) |
| StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold | 2510.01938 | NeurIPS 2025 (Spotlight) | Manifold-constrained LoRA — differentiated: geometry as cons | [card](papers/arxiv-2510-01938.md) |
| TLoRA | — | arXiv (2501) | Searched but found irrelevant; transform on pretrained weigh | [card](papers/tlora.md) |
| TLoRA: Tri-Matrix Low-Rank Adaptation of Large Language Models | 2504.18735 | arXiv | MUST CITE — tri-matrix with trainable middle between fixed r | [card](papers/arxiv-2504-18735.md) |
| VeRA: Vector-based Random Matrix Adaptation | 2310.11454 | ICLR 2024 | Frozen random projections with diagonal scaling — detailed c | [card](papers/arxiv-2310-11454.md) |
| VeRA | — | ICLR 2024 | Bridge-matrix lineage member; frozen random projections with | [card](papers/vera.md) |

## Block-Diagonal Structure

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| BLAST: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Networ | 2410.21262 | arXiv Oct 2024 | Block-diagonal coupling as a general principle in efficient  | [card](papers/arxiv-2410-21262.md) |
| Emergence in non-neural models: grokking modular arithmetic via average gradient | 2407.20199 | ICML 2025 | Supports Paper 4's central claim that BD emergence reflects  | [card](papers/arxiv-2407-20199.md) |
| Block-Diagonal LoRA for Eliminating Communication Overhead in Tensor Parallel Lo | 2510.23346 | NeurIPS 2025 | Term-overlap disambiguation required: block-diagonal imposed | [card](papers/arxiv-2510-23346.md) |
| One Head Eight Arms: Block Matrix based Low Rank Adaptation for CLIP-based Few-S | 2501.16720 | arXiv | Weak relevance — partial coupling via sharing, but not topol | [card](papers/arxiv-2501-16720.md) |
| Block-Recurrent Dynamics in Vision Transformers | 2512.19941 | ICLR 2026 | Single most important external validation for Paper 4's bloc | [card](papers/arxiv-2512-19941.md) |
| DiaBlo: Diagonal Blocks Are Sufficient For Finetuning | 2506.03230 | ICLR 2026 | cited-in-paper-4 | [card](papers/arxiv-2506-03230.md) |
| Localized LoRA: A Structured Low-Rank Approximation for Efficient Fine-Tuning | 2506.00236 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2506-00236.md) |
| LoRSum | — | — | Search-result mention | [card](papers/lorsum.md) |
| MELoRA | — | ACL 2024 | Mentioned only at TERTIARY source level (search results / su | [card](papers/melora.md) |
| MELoRA: Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning | 2402.17263 | arXiv preprint | cited-in-paper-3; cited-in-paper-4 | [card](papers/arxiv-2402-17263.md) |
| Entering the overcritical regime of nonlinear Breit-Wheeler pair production in c | 2501.08790 | 2025-01-15 | Previously tracked — status unchanged | [card](papers/arxiv-2501-08790.md) |
| OFT (Orthogonal Finetuning) | — | — | Mentioned as an established PEFT method (sparse block-diagon | [card](papers/oft-orthogonal-finetuning.md) |
| SSR-Merge: Subspace Signal Routing for Training-Free LoRA Merging in Diffusion M | 2606.10617 | arXiv preprint | 2026 geometric-merging wave (subspace routing); named as the | [card](papers/arxiv-2606-10617.md) |
| Stochastic Blockmodels: First Steps | — | Social Networks 5(2), pp. 109-137 | cited-in-paper-4 | [card](papers/stochastic-blockmodels-first-steps.md) |

## Spectral Methods & Dynamics

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| Exploring Data-Free LoRA Transferability for Video Diffusion Models | 2605.01929 | arXiv preprint | Most important external result for Paper 5: near-exact warni | [card](papers/arxiv-2605-01929.md) |
| Algebraic connectivity of graphs | 10.21136/CMJ.1973.101168 | Czechoslovak Mathematical Journal 23(2), pp. 298-305 | cited-in-paper-3; cited-in-paper-4 | [card](papers/algebraic-connectivity-of-graphs.md) |
| FlexLoRA: Entropy-Guided Flexible Low-Rank Adaptation | 2601.22905 | arXiv | Identifies rank collapse via SVD — spectral ceiling parallel | [card](papers/arxiv-2601-22905.md) |
| EigenLoRAx | — | — | HF search-result mention — orthogonal | [card](papers/eigenlorax.md) |
| Spectral Gap Regularization of Neural Networks | 2304.03096 | 2023-04-06 | Closest theoretical antecedent — same mathematical object (F | [card](papers/arxiv-2304-03096.md) |
| Fiedler Regularization: Learning Neural Networks with Graph Sparsity | — | ICML 2020 (PMLR v119, pp. 9346-9355) | cited-in-paper-3; cited-in-paper-4 | [card](papers/fiedler-regularization-learning-neural-networks-with-graph-s.md) |
| Fiedler Regularization: Learning Neural Networks with Graph Sparsity | 2003.00992 | ICML 2020 | Direct prior art for differentiable Fiedler objective; appli | [card](papers/arxiv-2003-00992.md) |
| Frequency Regularization: Unveiling the Spectral Inductive Bias of Deep Neural N | 2512.22192 | 2025-12-20 | LOW — very peripheral | [card](papers/arxiv-2512-22192.md) |
| Geometry of Reason: Spectral Signatures of Valid Mathematical Reasoning | 2601.00791 | 2026-01-02 | LOW — Monitor | [card](papers/arxiv-2601-00791.md) |
| Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-o | 2502.16894 | 2025-02-24 | Tier 2 relevant background | [card](papers/arxiv-2502-16894.md) |
| Graph-Based Spectral Decomposition for Parameter Coordination in Language Model  | 2504.19583 | arXiv April 2025 | Closest spectral method: Laplacian decomposition + spectral  | [card](papers/arxiv-2504-19583.md) |
| Janus-LoRA: A Balanced Low-Rank Adaptation for Continual Learning | 2605.28495 | arXiv preprint | 2026 geometric-merging wave (orthogonality vs prior-knowledg | [card](papers/arxiv-2605-28495.md) |
| Symmetry in language statistics shapes the geometry of model representations | 2602.15029 | arXiv | Already noted for citation; theoretical foundation for topol | [card](papers/arxiv-2602-15029.md) |
| LoRA vs Full Fine-tuning: An Illusion of Equivalence | 2410.21228 | arXiv | Validates the spectral-diagnostic approach — proves spectral | [card](papers/arxiv-2410-21228.md) |
| Effective LoRA Adapter Routing using Task Representations | 2601.21795 | arXiv | Adapter routing via spectral signal — limited signal noted | [card](papers/arxiv-2601-21795.md) |
| Uniform Spectral Growth and Convergence of Muon in LoRA-Style Matrix Factorizati | 2602.06385 | 2026-02-06 | HIGH relevance — SUPPORTS Paper 4 context | [card](papers/arxiv-2602-06385.md) |
| NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks | 2603.06922 | 2026-03-06 | Tier 2 relevant background | [card](papers/arxiv-2603-06922.md) |
| Structure and Redundancy in Large Language Models: A Spectral Study via Random M | 2602.22345 | 2026-02-25 | Tier 2 relevant background | [card](papers/arxiv-2602-22345.md) |
| SeLoRA (Spectral-Encoding LoRA) | — | ACL 2025 Findings | Spectral toolkit applied to weight-update representation for | [card](papers/selora-spectral-encoding-lora.md) |
| Revisiting LoRA through the Lens of Parameter Redundancy: Spectral Encoding Help | 2506.16787 | ACL 2025 | OPTIONAL cite for landscape completeness; orthogonal, potent | [card](papers/arxiv-2506-16787.md) |
| Weight Spectra Induced Efficient Model Adaptation | 2505.23099 | arXiv | Part of the growing spectral-diagnostics-for-LoRA ecosystem; | [card](papers/arxiv-2505-23099.md) |
| Approaching Deep Learning through the Spectral Dynamics of Weights | 2408.11804 | 2024-08-21 | Tier 3 peripheral | [card](papers/arxiv-2408-11804.md) |
| Spectral Edge Dynamics of Training Trajectories: Signal--Noise Geometry Across S | 2603.15678 | 2026-03-14 | MODERATE — SUPPORTS Paper 4 Fiedler convergence dynamics | [card](papers/arxiv-2603-15678.md) |
| Spectral Surgery: Training-Free Refinement of LoRA via Gradient-Guided Singular  | 2603.03995 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2603-03995.md) |
| Stable-LoRA: Stabilizing Feature Learning of Low-Rank Adaptation | 2603.05204 | arXiv | Conflicting guidance: Update 3 says do not cite (different p | [card](papers/arxiv-2603-05204.md) |
| Tensorized Clustered LoRA Merging for Multi-Task Interference | 2508.03999 | arXiv | Closest existing method to bridge-level merging — brief ment | [card](papers/arxiv-2508-03999.md) |
| The Universal Weight Subspace Hypothesis | 2512.05117 | 2025-12-04 | Tier-3 competing prediction (universal low-dimensional weigh | [card](papers/arxiv-2512-05117.md) |
| W2T: LoRA Weights Already Know What They Can Do | 2603.15990 | arXiv | claim-5 erosion — hub-scale weight-space fingerprinting; ado | [card](papers/arxiv-2603-15990.md) |
| From SGD to Spectra: A Theory of Neural Network Weight Dynamics | 2507.12709 | arXiv | Speculative theoretical framework for Fiedler attractor conv | [card](papers/arxiv-2507-12709.md) |
| Weight space Detection of Backdoors in LoRA Adapters | 2602.15195 | arXiv | Spectral fingerprinting of LoRA matrices — brief mention in  | [card](papers/arxiv-2602-15195.md) |

## Equivariance & Physics-Matched Priors

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| Aligning Network Equivariance with Data Symmetry: A Theoretical Framework and Ad | 2605.13744 | 2026-05-13 | BM-004 theory support | [card](papers/arxiv-2605-13744.md) |
| Does equivariance matter at scale? | 2410.23179 | ICLR 2025 | BM-004 background | [card](papers/arxiv-2410-23179.md) |
| E(3)-equivariant graph neural networks for data-efficient and accurate interatom | 10.1038/s41467-022-29939-5 | Nature Communications | Existence proof that geometric priors pay off where geometry | [card](papers/e-3-equivariant-graph-neural-networks-for-data-efficient-and.md) |
| ELoRA | — | — | BM-004 background — equivariance-preserving PEFT exists as c | [card](papers/elora.md) |
| Equivariant Adaptation of Large Pretrained Models | 2310.01647 | 2023-10-02 | BM-004 background | [card](papers/arxiv-2310-01647.md) |
| Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges | 2104.13478 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2104-13478.md) |
| Highly accurate protein structure prediction with AlphaFold | — | Nature | Passing example of SE(3)-equivariance succeeding in practice | [card](papers/highly-accurate-protein-structure-prediction-with-alphafold.md) |
| A 350-MHz Green Bank Telescope Survey of Unassociated Fermi LAT Sources: Discove | 2402.09366 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2402-09366.md) |
| Learning local equivariant representations for large-scale atomistic dynamics (A | 10.1038/s41467-023-36329-y | Nature Communications | Companion existence proof to NequIP for physically-motivated | [card](papers/learning-local-equivariant-representations-for-large-scale-a.md) |
| Learning on LoRAs: GL-Equivariant Processing of Low-Rank Weight Spaces for Large | 2410.04207 | arXiv | Proves adapter structure encodes task info — detailed compar | [card](papers/arxiv-2410-04207.md) |
| Spherical CNNs | — | ICLR 2018 | cited-in-paper-4 | [card](papers/spherical-cnns.md) |
| Measuring the Symmetry--Data Exchange Rate | 2606.01090 | 2026-05-31 | BM-004 concurrent work (outside PEFT) — cite as concurrent,  | [card](papers/arxiv-2606-01090.md) |

## Representation Alignment & Cross-Modal

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| Convergence Without Understanding: When Language Models Agree on Representations | 2605.23315 | 2026-05-22 | Tier-3 protocol requirement — behavioral endpoints alongside | [card](papers/arxiv-2605-23315.md) |
| Feature Geometry of LoRA Adapters: A Sparse Autoencoder Analysis of Representati | 2605.28896 | 2026-05-27 | D1 reference point — adapter-vs-base principal angles ≈74° s | [card](papers/arxiv-2605-28896.md) |
| Platonic Representations in the Human Brain: Unsupervised Recovery of Universal  | 2605.20496 | arXiv preprint | Extends vec2vec-style unsupervised embedding translation ont | [card](papers/arxiv-2605-20496.md) |
| Time Series, Vision, and Language: Exploring the Limits of Alignment in Contrast | 2602.19367 | arXiv preprint | PRH critique: a third modality sits near-orthogonal until ex | [card](papers/arxiv-2602-19367.md) |
| Revisiting the Platonic Representation Hypothesis: An Aristotelian View | 2602.14486 | ICML 2026 | Tier-3 protocol requirement — mandates permutation/random-su | [card](papers/arxiv-2602-14486.md) |
| AuRA: Internalizing Audio Understanding into LLMs as LoRA | 2606.11033 | 2026-06-09 | adjacent — distillation route sidesteps geometry, narrows br | [card](papers/arxiv-2606-11033.md) |
| Back into Plato's Cave: Examining Cross-modal Representational Convergence at Sc | 2604.18572 | arXiv preprint | PRH critique: representation alignment degrades at scale; so | [card](papers/arxiv-2604-18572.md) |
| Crowded in B-Space: Calibrating Shared Directions for LoRA Merging | 2604.16826 | 2026-04-18 | Tier-3 asymmetry prediction (B shared, A task-specific); Tie | [card](papers/arxiv-2604-16826.md) |
| Generalizing the Geometry of Model Merging Through Frechet Averages | 2604.27155 | 2026-04-29 | Tier-3 protocol requirement — gauge invariance; measure on q | [card](papers/arxiv-2604-27155.md) |
| From "Weak" Signals to Strong Models: Preference Delta Aggregation with LoRA Mer | 2606.00357 | 2026-05-29 | Tier-3 baseline to beat (Procrustes merging) | [card](papers/arxiv-2606-00357.md) |
| Harnessing the Universal Geometry of Embeddings | 2505.12540 | arXiv preprint | Strong-form PRH: unsupervised translation between embedding  | [card](papers/arxiv-2505-12540.md) |
| Mind the Heads: Topological Representation Alignment for Multimodal LLMs | 2606.23885 | 2026-06-22 | Tier-3 adjacent (head-level MKNN alignment) | [card](papers/arxiv-2606-23885.md) |
| InfoNCE Induces Gaussian Distribution | 2602.24012 | 2026-02-27 | SUPPORTS strongly (Tier 2, Scan 10) | [card](papers/arxiv-2602-24012.md) |
| mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations | 2510.02348 | 2025-09-27 | Tier-3 baseline to beat (linear alignment) | [card](papers/arxiv-2510-02348.md) |
| Modality Gap-Driven Subspace Alignment Training Paradigm For Multimodal Large La | 2602.07026 | 2026-02-02 | Tier-3 adjacent | [card](papers/arxiv-2602-07026.md) |
| Semi-supervised Multimodal Representation Learning Through a Global Workspace | 10.1109/TNNLS.2024.3416701 | IEEE Transactions on Neural Networks and Learning Systems | Closest published analogue to the single-structured-hub (wor | [card](papers/semi-supervised-multimodal-representation-learning-through-a.md) |
| Shared LoRA Subspaces for almost Strict Continual Learning | 2602.06043 | 2026-02-05 | competitor signal — part of JHU group likeliest to run the T | [card](papers/arxiv-2602-06043.md) |
| Preference-Aligned LoRA Merging: Preserving Subspace Coverage and Addressing Dir | 2603.26299 | 2026-03-27 | Tier-3 protocol requirement — singular-value-weighted angle  | [card](papers/arxiv-2603-26299.md) |
| The Platonic Representation Hypothesis | 2405.07987 | ICML 2024 | Names the field (representation alignment) that owns the cro | [card](papers/arxiv-2405-07987.md) |

## Merging, Routing & Task Arithmetic

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| Predicting Mergeability of Parameter-Efficient Fine-Tuning Updates | 2606.19549 | 2026-06-17 | claim-5 preemption — owns training-time merge-conflict predi | [card](papers/arxiv-2606-19549.md) |
| AdaFuse | — | — | Tracked-list mention only (no assessment in source) | [card](papers/adafuse.md) |
| AdaMerging | — | ICLR 2024 | Merging baseline (Phase 2A context) — layer-wise coefficient | [card](papers/adamerging.md) |
| AdaMerging: Adaptive Model Merging for Multi-Task Learning | 2310.02575 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2310-02575.md) |
| CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging | 2603.00573 | 2026-02-28 | LOW | [card](papers/arxiv-2603-00573.md) |
| DARE | — | NeurIPS 2024 | Merging baseline (Phase 2A context) — pruning vs structured  | [card](papers/dare.md) |
| Editing Models with Task Arithmetic | 2212.04089 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2212-04089.md) |
| Language Models are Super Mario: Absorbing Abilities from Homologous Models as a | 2311.03099 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2311-03099.md) |
| LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition | 2307.13269 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2307-13269.md) |
| MoLoRA: Composable Specialization via Per-Token Adapter Routing | 2603.15965 | 2026-03-16 | MODERATE — EXTENDS toward Paper 5; later assessed ORTHOGONAL | [card](papers/arxiv-2603-15965.md) |
| Task Arithmetic | — | ICLR 2023 | Baseline for bridge-level merging comparison (Phase 2A conte | [card](papers/task-arithmetic.md) |
| TIES-Merging | — | NeurIPS 2023 | Merging baseline (Phase 2A context) — bridge merging avoids  | [card](papers/ties-merging.md) |
| TIES-Merging: Resolving Interference When Merging Models | 2306.01708 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2306-01708.md) |

## Fingerprinting, Provenance & Diagnostics

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| Spectral Geometry of LoRA Adapters Encodes Training Objective and Predicts Harmf | 2604.08844 | 2026-04-10 | claim-2 diagnostics-only neighbor; claim-5 erosion — spectra | [card](papers/arxiv-2604-08844.md) |
| Learning on Model Weights using Tree Experts | 2410.13569 | 2024-10-17 | Tier 1B prior art in kind | [card](papers/arxiv-2410-13569.md) |
| A Survey of Weight Space Learning: Understanding, Representation, and Generation | 2603.10090 | arXiv | Tier 1B survey citation | [card](papers/arxiv-2603-10090.md) |
| Interpreting the Weight Space of Customized Diffusion Models | 2406.09413 | 2024-06-13 | Tier 1B prior art in kind | [card](papers/arxiv-2406-09413.md) |

## Diffusion / Video LoRA

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| CAT (Contrastive Adapter Training for image generation) | — | — | Orthogonal search hit | [card](papers/cat-contrastive-adapter-training-for-image-generation.md) |
| CLoRA (contrastive LoRA composition) | — | ICCV 2025 | Contrastive objective between adapters (image generation), n | [card](papers/clora-contrastive-lora-composition.md) |
| DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Genera | 10.1109/cvpr52729.2023.02155 | CVPR 2023 | Anchor for the applied personalization thread (production di | [card](papers/dreambooth-fine-tuning-text-to-image-diffusion-models-for-su.md) |
| LoRA: Low-Rank Adaptation of Large Language Models | 2106.09685 | ICLR 2022 | cited-in-paper-3; cited-in-paper-4 | [card](papers/arxiv-2106-09685.md) |
| LoRAGen: Structure-Aware Weight Space (LoRA parameter generation) | — | ICLR 2026 (under review, OpenReview) | Do not cite — different problem (LoRA generation); name coll | [card](papers/loragen-structure-aware-weight-space-lora-parameter-generati.md) |
| Prompt2Effect: Training-Free Image-to-Video Model Specialization via LoRA Genera | 2606.13971 | arXiv preprint | Constructive counterpart to the negative transfer result: hy | [card](papers/arxiv-2606-13971.md) |
| T-LoRA: Single Image Diffusion Model Customization Without Overfitting | 2507.05964 | arXiv | Closest work to Phase 3A overfitting detection, but diffusio | [card](papers/arxiv-2507-05964.md) |
| T2I-Adapter | — | 2023 | Named-method anchor in the literature sweep only; no substan | [card](papers/t2i-adapter.md) |

## Engineering References (Nemotron era)

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| DAPO | — | — | briefly-cited | [card](papers/dapo.md) |
| DeepSeek-R1 | — | — | briefly-cited | [card](papers/deepseek-r1.md) |
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Mode | 2402.03300 | arXiv | algorithm-origin-reference | [card](papers/arxiv-2402-03300.md) |
| Fast-Math-R1 (AIMO-2 28th/29th place, SFT+GRPO pipeline) | — | GitHub (competition solution) | methodology-reference | [card](papers/fast-math-r1-aimo-2-28th-29th-place-sft-grpo-pipeline.md) |
| Instruction Fine-Tuning: Does Prompt Loss Matter? | 2401.13586 | arXiv | directly-actionable | [card](papers/arxiv-2401-13586.md) |
| Jamba (Mamba-Transformer hybrid — RMSNorm stabilization precedent) | — | — | architectural-precedent | [card](papers/jamba-mamba-transformer-hybrid-rmsnorm-stabilization-precede.md) |
| NEFTune (noisy embedding instruction fine-tuning — referenced as technique) | — | — | technique-adopted | [card](papers/neftune-noisy-embedding-instruction-fine-tuning-referenced-a.md) |
| Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Mod | 2512.20848 | arXiv | primary-engineering-reference | [card](papers/arxiv-2512-20848.md) |
| Open-Reasoner-Zero | — | — | briefly-cited | [card](papers/open-reasoner-zero.md) |
| AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models | 2504.16891 | arXiv | methodology-reference | [card](papers/arxiv-2504-16891.md) |
| Understanding R1-Zero | — | — | briefly-cited | [card](papers/understanding-r1-zero.md) |

## Surveys

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning  | 2606.00133 | arXiv preprint | Marks world-model integration (the program endpoint) as a ho | [card](papers/arxiv-2606-00133.md) |
| A review of modularization techniques in artificial neural networks | — | Artificial Intelligence Review 52, pp. 527-561 | cited-in-paper-3 | [card](papers/a-review-of-modularization-techniques-in-artificial-neural-n.md) |
| A Survey on LoRA of Large Language Models | 10.1007/s11704-024-40663-9 | Frontiers of Computer Science, 2025 | Chinese-led comprehensive survey; identifies no bridge-matri | [card](papers/a-survey-on-lora-of-large-language-models.md) |
| An Overview of Low-Rank Structures in the Training and Adaptation of Large Model | 2503.19859 | 2025-03-25 | HIGH — SUPPORTS Papers 3 and 4 | [card](papers/arxiv-2503-19859.md) |
| Hyperbolic Deep Neural Networks: A Survey | 10.1109/tpami.2021.3136921 | IEEE Transactions on Pattern Analysis and Machine Intelligence | Precedent that representation-manifold curvature is a design | [card](papers/hyperbolic-deep-neural-networks-a-survey.md) |
| LoRA survey (Springer 2025) | — | Springer | Consulted as background source only; no individual assessmen | [card](papers/lora-survey-springer-2025.md) |
| Parameter-efficient fine-tuning of large-scale pre-trained language models | 10.1038/s42256-023-00626-4 | Nature Machine Intelligence | Field-maturity anchor: PEFT is mature enough that claim-veri | [card](papers/parameter-efficient-fine-tuning-of-large-scale-pre-trained-l.md) |
| Topological Data Analysis for Neural Network Analysis: A Comprehensive Survey | 2312.05840 | 2023-12-10 | LOW — Monitor only | [card](papers/arxiv-2312-05840.md) |
| 《自动化学报》 flagship PEFT survey | — | 《自动化学报》 (Acta Automatica Sinica) | Chinese-venue coverage check — none of the five claims' mech | [card](papers/flagship-peft-survey.md) |

## Director's Map (Representation & Adjacent)

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| [Kepler conjecture / Hales proof — sphere packing and kissing number 12] | — | Annals of Mathematics (Hales proof, 2005) | Historical mathematical grounding for the FCC/rhombic lattic | [card](papers/kepler-conjecture-hales-proof-sphere-packing-and-kissing-num.md) |
| BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Lang | — | ACL 2022 | Named-method anchor in the literature sweep only; no substan | [card](papers/bitfit-simple-parameter-efficient-fine-tuning-for-transforme.md) |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces | — | 2023 | Named-method anchor in the literature sweep only; no substan | [card](papers/mamba-linear-time-sequence-modeling-with-selective-state-spa.md) |
| QLoRA: Efficient Finetuning of Quantized LLMs | — | NeurIPS 2023 | Named-method anchor in the literature sweep only; no substan | [card](papers/qlora-efficient-finetuning-of-quantized-llms.md) |
| UniVidX: A Unified Multimodal Framework for Versatile Video Generation via Diffu | 2605.00658 | ACM (July 2026; DOI 10.1145/3811304 unregistered at time of citation) / arXiv | Cited in the Director's literature map; only citation-hygien | [card](papers/arxiv-2605-00658.md) |

## Cited in Our Papers

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning | 2205.12410 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2205-12410.md) |
| Algebraic Geometry and Statistical Learning Theory | — | Cambridge University Press (book, cited as article in bib) | cited-in-paper-4 | [card](papers/algebraic-geometry-and-statistical-learning-theory.md) |
| An Introduction to Cybernetics | — | Chapman & Hall, London (book) | cited-in-paper-3; cited-in-paper-4 | [card](papers/an-introduction-to-cybernetics.md) |
| Asymmetry in Low-Rank Adapters of Foundation Models | 2402.16842 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2402-16842.md) |
| Brain of the Firm | — | Allen Lane, The Penguin Press, London (book, ISBN 978-0-471-94839-2) | cited-in-paper-3; cited-in-paper-4 | [card](papers/brain-of-the-firm.md) |
| Curriculum Learning | — | ICML 2009 | cited-in-paper-4 | [card](papers/curriculum-learning.md) |
| Cybernetics: Or Control and Communication in the Animal and the Machine | — | MIT Press, Cambridge, MA (book) | cited-in-paper-3; cited-in-paper-4 | [card](papers/cybernetics-or-control-and-communication-in-the-animal-and-t.md) |
| Cyclical Learning Rates for Training Neural Networks | 1506.01186 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-1506-01186.md) |
| Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Cla | — | ICCV 2015 | cited-in-paper-3 | [card](papers/delving-deep-into-rectifiers-surpassing-human-level-performa.md) |
| DoRA: Weight-Decomposed Low-Rank Adaptation | 2402.09353 | arXiv preprint | cited-in-paper-3; cited-in-paper-4 | [card](papers/arxiv-2402-09353.md) |
| DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Fr | 2210.07558 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2210-07558.md) |
| GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | 2403.03507 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2403-03507.md) |
| Gradient-based learning applied to document recognition | — | Proceedings of the IEEE 86(11), pp. 2278-2324 | cited-in-paper-4 | [card](papers/gradient-based-learning-applied-to-document-recognition.md) |
| GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | 2006.16668 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2006-16668.md) |
| HyperNetworks | 1609.09106 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-1609-09106.md) |
| Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tunin | 2012.13255 | arXiv preprint | cited-in-paper-3; cited-in-paper-4 | [card](papers/arxiv-2012-13255.md) |
| Learning Structured Sparsity in Deep Neural Networks | — | NeurIPS 2016 | cited-in-paper-4 | [card](papers/learning-structured-sparsity-in-deep-neural-networks.md) |
| Learning to learn by gradient descent by gradient descent | — | Advances in Neural Information Processing Systems 29 (NeurIPS) | cited-in-paper-3; cited-in-paper-4 | [card](papers/learning-to-learn-by-gradient-descent-by-gradient-descent.md) |
| Learning to Reweight Examples for Robust Deep Learning | — | ICML 2018 | cited-in-paper-4 | [card](papers/learning-to-reweight-examples-for-robust-deep-learning.md) |
| LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models | 2309.12307 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2309-12307.md) |
| LoRA Learns Less and Forgets Less | 2405.09673 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2405-09673.md) |
| Measuring the Intrinsic Dimension of Objective Landscapes | 1804.08838 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-1804-08838.md) |
| Mixture of LoRA Experts | 2404.13628 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2404-13628.md) |
| Modular Networks: Learning to Decompose Neural Computation | 1811.05249 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-1811-05249.md) |
| Multi-Head Adapter Routing for Cross-Task Generalization | 2211.03831 | arXiv preprint | cited-in-paper-3 | [card](papers/arxiv-2211-03831.md) |
| MultiLoRA: Democratizing LoRA for Better Multi-Task Learning | 2311.11501 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2311-11501.md) |
| Net2Net: Accelerating Learning via Knowledge Transfer | 1511.05641 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-1511-05641.md) |
| Neural Architecture Search with Reinforcement Learning | 1611.01578 | ICLR 2017 | cited-in-paper-3; cited-in-paper-4 | [card](papers/arxiv-1611-01578.md) |
| Poincare Embeddings for Learning Hierarchical Representations | — | NeurIPS 2017 | cited-in-paper-4 | [card](papers/poincare-embeddings-for-learning-hierarchical-representation.md) |
| Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for I | 2309.05444 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2309-05444.md) |
| QLoRA: Efficient Finetuning of Quantized LLMs | 2305.14314 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-2305-14314.md) |
| Quaternion Recurrent Neural Networks | 1806.04418 | arXiv preprint | cited-in-paper-4 | [card](papers/arxiv-1806-04418.md) |
| Semi-Supervised Classification with Graph Convolutional Networks | 1609.02907 | ICLR 2017 | cited-in-paper-4 | [card](papers/arxiv-1609-02907.md) |
| Sphere Packings, Lattices and Groups (3rd edition) | — | Springer-Verlag, New York (book, ISBN 978-0-387-98585-5) | cited-in-paper-3; cited-in-paper-4 | [card](papers/sphere-packings-lattices-and-groups-3rd-edition.md) |
| Switch Transformers: Scaling to Trillion Parameter Models with Simple and Effici | — | Journal of Machine Learning Research 23(120), pp. 1-39 | cited-in-paper-3 | [card](papers/switch-transformers-scaling-to-trillion-parameter-models-wit.md) |
| The evolutionary origins of modularity | — | Proceedings of the Royal Society B 280(1755), 20122863 | cited-in-paper-3; cited-in-paper-4 | [card](papers/the-evolutionary-origins-of-modularity.md) |
| The Graph Neural Network Model | — | IEEE Transactions on Neural Networks 20(1), pp. 61-80 | cited-in-paper-4 | [card](papers/the-graph-neural-network-model.md) |
| The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | 1803.03635 | arXiv preprint | cited-in-paper-3; cited-in-paper-4 | [card](papers/arxiv-1803-03635.md) |
| Visualizing the Loss Landscape of Neural Nets | — | Advances in Neural Information Processing Systems 31 (NeurIPS) | cited-in-paper-3 | [card](papers/visualizing-the-loss-landscape-of-neural-nets.md) |

## Peripheral / Monitored

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| ACE-LoRA: Graph-Attentive Context Enhancement for Parameter-Efficient Adaptation | 2603.17079 | 2026-03-17 | LOW — ORTHOGONAL | [card](papers/arxiv-2603-17079.md) |
| AdaLoRA | — | — | Named only as composable/orthogonal method — no citation det | [card](papers/adalora.md) |
| AdaLoRA (adaptive rank allocation) | — | — | Competitive landscape entry — superseded by GoRA | [card](papers/adalora-adaptive-rank-allocation.md) |
| AFLoRA | — | — | Orthogonal search hit | [card](papers/aflora.md) |
| ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws | 2410.09692 | 2024-10-13 | LOW — ORTHOGONAL | [card](papers/arxiv-2410-09692.md) |
| ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs | 2603.01792 | 2026-03-02 | LOW (Scan 7); Tier 2 relevant background (Scan 10) | [card](papers/arxiv-2603-01792.md) |
| A Backpropagation-Free Feedback-Hebbian Network for Continual Learning Dynamics | 2601.06758 | 2026-01-11 | LOW — Monitor only | [card](papers/arxiv-2601-06758.md) |
| Bio-RegNet | — | MDPI | Newly surveyed — no threat; 'homeostatic' in name only | [card](papers/bio-regnet.md) |
| BoRA | — | — | Flagged for citation in the superseded Mar 24 report; carrie | [card](papers/bora.md) |
| C-LoRA | — | — | Trend mention — LoRA subspace family | [card](papers/c-lora.md) |
| C3A | — | — | BM-003 piecewise anticipation (circulant = cyclic Cayley gra | [card](papers/c3a.md) |
| CAT | — | — | Consulted as tertiary source only (search snippets); no indi | [card](papers/cat.md) |
| CE-LoRA (computation efficiency) | — | — | Orthogonal search hit | [card](papers/ce-lora-computation-efficiency.md) |
| CoLD (contrastive decoding for LoRA knowledge) | — | — | Trend mention — contrastive+LoRA family | [card](papers/cold-contrastive-decoding-for-lora-knowledge.md) |
| Compressible Dynamics (deep linear network result) | — | ICML 2024 Oral | Cited as lineage of Emergent Low-Rank Dynamics paper | [card](papers/compressible-dynamics-deep-linear-network-result.md) |
| How connectivity structure shapes rich and lazy learning in neural circuits | 2310.08513 | 2023-10-12 | Tier 3 peripheral | [card](papers/arxiv-2310-08513.md) |
| Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework | 2505.21251 | 2025-05-27 | LOW — Monitor only | [card](papers/arxiv-2505-21251.md) |
| CorDA | — | — | Mentioned as an established structured-variant baseline in H | [card](papers/corda.md) |
| CoSO | — | — | Trend mention — LoRA subspace family | [card](papers/coso.md) |
| Cross-LoRA | — | — | Consulted as secondary source only (search snippets + summar | [card](papers/cross-lora.md) |
| DeepWeightFlow | — | — | Gap-analysis differentiation mention | [card](papers/deepweightflow.md) |
| DiaBlo | — | — | claim-4 imposed-block list | [card](papers/diablo.md) |
| DoRA | — | ICML 2024 Oral | Searched but found irrelevant; composable with TeLoRA, not c | [card](papers/dora.md) |
| DoRA: Weight-Decomposed Low-Rank Adaptation | — | ICML 2024 | Competitive landscape entry — no BD discovery, no diagnostic | [card](papers/dora-weight-decomposed-low-rank-adaptation.md) |
| DoRAN (DoRA variant) | — | Oct 2025 (Scan 6) / Feb 2026 (Scan 10) — source dates inconsistent | Competitive landscape mention | [card](papers/doran-dora-variant.md) |
| DyLoRA | — | — | HF search-result mention — orthogonal | [card](papers/dylora.md) |
| Dynamic Feedback Engines: Layer-Wise Control for Self-Regulating Continual Learn | 2512.21743 | 2025-12-25 | Tier 3 peripheral | [card](papers/arxiv-2512-21743.md) |
| Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task | 2603.12577 | 2026-03-13 | MODERATE (Scan 5) then LOW/ORTHOGONAL (Scan 6) | [card](papers/arxiv-2603-12577.md) |
| Fesser & Weber 2025 (Fiedler singular value / over-squashing in GNNs) | — | 2025 | Not a threat — diagnosis tool for GNNs, not adapter training | [card](papers/fesser-weber-2025-fiedler-singular-value-over-squashing-in-g.md) |
| Flat-LoRA | — | — | Orthogonal search hit | [card](papers/flat-lora.md) |
| FLoRA (federated LoRA) | — | — | Competitive landscape entry — federated successors noted | [card](papers/flora-federated-lora.md) |
| GaLore (gradient low-rank projection) | — | — | Competitive landscape entry — no new activity | [card](papers/galore-gradient-low-rank-projection.md) |
| GaLore 2 | — | April 2025 | Competitive landscape mention | [card](papers/galore-2.md) |
| GeLoRA | — | EMNLP 2025 | Newly surveyed — no threat | [card](papers/gelora.md) |
| GoRA (gradient-driven adaptive rank) | — | NeurIPS 2025 | Competitive landscape entry — watch for v4 | [card](papers/gora-gradient-driven-adaptive-rank.md) |
| GUM | — | October 2025 | Competitive landscape mention | [card](papers/gum.md) |
| Instruction-Tuned, but Not More Verifiable Instruction-Following: A Cross-Task D | 2603.22379 | arXiv | Not relevant — empirical analysis, not architecture | [card](papers/arxiv-2603-22379.md) |
| Interactive Training (Zhang et al.) | — | — | Gap-analysis differentiation mention | [card](papers/interactive-training-zhang-et-al.md) |
| KeepLoRA | — | — | Trend mention — LoRA subspace family | [card](papers/keeplora.md) |
| Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning | 2602.06204 | 2026-02-05 | LOW — noted for awareness | [card](papers/arxiv-2602-06204.md) |
| LoRA Done RITE | — | — | Orthogonal search hit | [card](papers/lora-done-rite.md) |
| LoRA+ (differential learning rates) | — | — | Competitive landscape entry — subsumed by LR Matters | [card](papers/lora-differential-learning-rates.md) |
| LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fin | 2507.20999 | EMNLP 2025 Findings | LOW | [card](papers/arxiv-2507-20999.md) |
| LoRA-Squeeze: Simple and Effective Post-Tuning and In-Tuning Compression of LoRA | 2602.10993 | arXiv | Not a threat — different axis (compression, not structure) | [card](papers/arxiv-2602-10993.md) |
| LoRACoE | — | EMNLP 2024 | Searched but found irrelevant; MoE with LoRA experts, differ | [card](papers/loracoe.md) |
| LoRAFusion | — | EUROSYS 2026 | Systems paper — not a threat | [card](papers/lorafusion.md) |
| LoRAN | — | — | Mentioned only at TERTIARY source level (search results / su | [card](papers/loran.md) |
| LoRAN (multi-rank, sine activation) | — | EMNLP 2024 | Competitive landscape entry — tracked competitor team, no ne | [card](papers/loran-multi-rank-sine-activation.md) |
| LoRAN / Sinter | — | — | Covered in the superseded Mar 24 report; no assessment avail | [card](papers/loran-sinter.md) |
| LoRAuter | — | — | Orthogonal search hit | [card](papers/lorauter.md) |
| LoRMA | — | ACL Findings 2025 | Searched but found irrelevant; multiplicative adaptation wit | [card](papers/lorma.md) |
| MELoRA (mini-ensemble LoRA) | — | last publication 2024 | Competitive landscape entry — no new activity | [card](papers/melora-mini-ensemble-lora.md) |
| MiSS | — | — | Mentioned as an established structured-variant baseline in H | [card](papers/miss.md) |
| MoLA | — | — | Consulted as secondary source only (search snippets + summar | [card](papers/mola.md) |
| MoRA | — | arXiv (2405) | Searched but found irrelevant; square matrix replacing A/B,  | [card](papers/mora.md) |
| Contrastive Regularization over LoRA for Multimodal Biomedical Image Incremental | 2508.11673 | ACM MM | LOW | [card](papers/arxiv-2508-11673.md) |
| NEAT | — | — | Confirmed irrelevant (tertiary check) | [card](papers/neat.md) |
| ODELoRA: Training Low-Rank Adaptation by Solving Ordinary Differential Equations | 2602.07479 | arXiv | Not a threat — theoretical, no geometry | [card](papers/arxiv-2602-07479.md) |
| On the Topology of Neural Network Superlevel Sets | 2603.02973 | 2026-03-03 | LOW — not tracked | [card](papers/arxiv-2603-02973.md) |
| Disentangling Task Conflicts in Multi-Task LoRA via Orthogonal Gradient Projecti | 2601.09684 | 2026-01-14 | MODERATE — ORTHOGONAL | [card](papers/arxiv-2601-09684.md) |
| PiSSA | — | — | Mentioned as an established structured-variant baseline in H | [card](papers/pissa.md) |
| PiSSA (principal singular values adaptation) | — | — | Competitive landscape entry — no new activity | [card](papers/pissa-principal-singular-values-adaptation.md) |
| PLoRA (federated LoRA successor) | — | February 2026 | Competitive landscape mention | [card](papers/plora-federated-lora-successor.md) |
| PolyhedronNet | — | February 2025 | Orthogonal search hit | [card](papers/polyhedronnet.md) |
| Polytopes and Machine Learning | — | 2021 | Orthogonal search hit — only polytope+ML result found | [card](papers/polytopes-and-machine-learning.md) |
| Practical and Performant Enhancements for Maximization of Algebraic Connectivity | 2511.08694 | 2025-11-11 | LOW — utility tool, not research competitor | [card](papers/arxiv-2511-08694.md) |
| Predicting GCN Performance via Fiedler Value | — | 2025 | Orthogonal search hit — Fiedler for GCN performance predicti | [card](papers/predicting-gcn-performance-via-fiedler-value.md) |
| Quantum-Inspired Fine-Tuning for Few-Shot AIGC Detection via Phase-Structured Re | 2603.02281 | 2026-03-02 | LOW — not tracked | [card](papers/arxiv-2603-02281.md) |
| R-LoRA | — | — | Consulted as secondary source only (search snippets + summar | [card](papers/r-lora.md) |
| RandLoRA: Full-rank parameter-efficient fine-tuning of large models | 2502.00987 | ICLR 2025 | Brief mention only — cite in Introduction/Background | [card](papers/arxiv-2502-00987.md) |
| RiemannLoRA | — | — | Referenced as already in Tier 1 of internal citation set | [card](papers/riemannlora.md) |
| rsLoRA (rank-stabilized LoRA) | — | — | Competitive landscape entry — no new activity | [card](papers/rslora-rank-stabilized-lora.md) |
| Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking | 2509.21519 | 2025-09-25 | Tier 3 peripheral | [card](papers/arxiv-2509-21519.md) |
| SDFLoRA (federated LoRA successor) | — | January 2026 | Competitive landscape mention | [card](papers/sdflora-federated-lora-successor.md) |
| Sparse High Rank Adapters | 2406.13175 | NeurIPS 2024 | Brief mention only — cite in Introduction/Background | [card](papers/arxiv-2406-13175.md) |
| SHiRA | — | — | Mentioned as an established PEFT method (sparse high-rank ad | [card](papers/shira.md) |
| SingLoRA | — | — | Orthogonal search hit | [card](papers/singlora.md) |
| High-Rank Structured Modulation for Parameter-Efficient Fine-Tuning | 2601.07507 | arXiv Jan 2026 | High-rank structured weight modulation; novel but operates o | [card](papers/arxiv-2601-07507.md) |
| Stiefel-LoRA | — | — | Named once as part of the manifold-aware optimization trajec | [card](papers/stiefel-lora.md) |
| Structure-aware Contrastive learning for diagrams | — | — | Orthogonal search hit | [card](papers/structure-aware-contrastive-learning-for-diagrams.md) |
| Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-b | — | March 2026 (cs.LG listings; exact arXiv ID not confirmed by source) | LOW — ORTHOGONAL | [card](papers/task-driven-subspace-decomposition-for-knowledge-sharing-and.md) |
| The Primacy of Magnitude in Low-Rank Adaptation | 2507.06558 | 2025-07-09 | LOW — worth monitoring (partially contradicts sign fingerpri | [card](papers/arxiv-2507-06558.md) |
| Tiki-Taka | — | 2022 | Gap-analysis differentiation mention | [card](papers/tiki-taka.md) |
| TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA | 2510.04682 | under review ICLR 2026 | LOW — ORTHOGONAL | [card](papers/arxiv-2510-04682.md) |
| tLoRA: Efficient Multi-LoRA Training with Elastic Shared Super-Models | 2602.07263 | 2026-02-06 | LOW — not tracked | [card](papers/arxiv-2602-07263.md) |
| Topological NN over Air | — | — | Confirmed irrelevant (tertiary check) | [card](papers/topological-nn-over-air.md) |
| Topological Regularization via Persistent Homology | — | — | LOW — Monitor only | [card](papers/topological-regularization-via-persistent-homology.md) |
| Topological Signatures of ReLU Neural Network Activation Patterns | 2510.12700 | arXiv | Not a threat — analysis, not design | [card](papers/arxiv-2510-12700.md) |
| Topology Matters: A Cautionary Case Study of Graph SSL on Neuro-Inspired Benchma | 2602.03217 | 2026-02-03 | LOW — THEMATICALLY ADJACENT | [card](papers/arxiv-2602-03217.md) |
| TopoLoRA-SAM: Topology-Aware Parameter-Efficient Adaptation of Foundation Segmen | 2601.02273 | 2026-01-05 | MEDIUM — ORTHOGONAL but NAME OVERLAP | [card](papers/arxiv-2601-02273.md) |
| Training Large Neural Networks With Low-Dimensional Error Feedback | 2502.20580 | 2025-02-27 | LOW — Monitor only | [card](papers/arxiv-2502-20580.md) |
| VasGuideNet | — | — | Orthogonal search hit | [card](papers/vasguidenet.md) |
| VB-LoRA | — | — | Consulted as tertiary source only (search snippets); no indi | [card](papers/vb-lora.md) |
| Voronoi CNN | — | — | Confirmed irrelevant (tertiary check) | [card](papers/voronoi-cnn.md) |
| Weight Space Backdoor Detection | — | — | Orthogonal search hit | [card](papers/weight-space-backdoor-detection.md) |
| Wireless Federated LoRA | — | — | Orthogonal search hit | [card](papers/wireless-federated-lora.md) |
| Wong et al. (2009) — spherical projection / face-indexed RD coordinate system (f | — | 2009 | Referenced as the method underlying the face-indexed coordin | [card](papers/wong-et-al-2009-spherical-projection-face-indexed-rd-coordin.md) |
| Zipper-LoRA: Dynamic Parameter Decoupling for Speech-LLM based Multilingual Spee | 2603.17558 | 2026-03-18 | MODERATE — ORTHOGONAL | [card](papers/arxiv-2603-17558.md) |

## Uncategorized

| Paper | ID | Venue/Date | Verdict | Card |
|---|---|---|---|---|
| Automatic Stability and Recovery for Neural Network Training | 2601.17483 | 2026-01-24 | MODERATE — SUPPORTS | [card](papers/arxiv-2601-17483.md) |
| BD-LoRA | — | — | claim-4 imposed-block list | [card](papers/bd-lora.md) |
| Block-Recurrent ViT | — | — | claim-4 emergent-BD reference outside adapters (status uncha | [card](papers/block-recurrent-vit.md) |
| BOFT | — | — | BM-003 piecewise anticipation (butterfly topology) | [card](papers/boft.md) |
| BSLoRA (Bi-Share LoRA) | — | ICML 2025 | Parameter sharing mechanism, not coupling; no bridge, no spe | [card](papers/bslora-bi-share-lora.md) |
| On Catastrophic Forgetting in Low-Rank Decomposition-Based Parameter-Efficient F | 2603.09684 | arXiv | MONITOR — worth reading for Paper 4 related-work framing | [card](papers/arxiv-2603-09684.md) |
| CLoRA (Controlled LoRA) | — | ACL 2025 | Subspace regularization against catastrophic forgetting; sub | [card](papers/clora-controlled-lora.md) |
| CoLD | — | KDD 2025 | Contrastive decoding between LoRA-adapted and base model, no | [card](papers/cold.md) |
| Defining Neural Network Architecture through Polytope Structures of Dataset | 2402.02407 | ICML 2024 | MODERATE — SUPPORTS Paper 4 | [card](papers/arxiv-2402-02407.md) |
| Defining Neural Network Architecture through Polytope Structures of Datasets | — | ICML 2024 | Not a threat — polytope as analysis tool for data, not struc | [card](papers/defining-neural-network-architecture-through-polytope-struct.md) |
| DoRAN: Stabilizing Weight-Decomposed Low-Rank Adaptation via Noise Injection and | 2510.04331 | arXiv | Should cite (brief mention) — composable context; already tr | [card](papers/arxiv-2510-04331.md) |
| Dual LoRA: Enhancing LoRA with Magnitude and Direction Updates | 2512.03402 | 2025-12-03 | MODERATE — SUPPORTS Paper 3 sign fingerprint finding | [card](papers/arxiv-2512-03402.md) |
| EEG-GraphAdapter | — | — | BM-003 piecewise anticipation (GNN-in-adapter, physical grap | [card](papers/eeg-graphadapter.md) |
| Emergent Low-Rank Training Dynamics in MLPs with Smooth Activations | 2602.06208 | 2026-02-05 | MODERATE — SUPPORTS Paper 4 | [card](papers/arxiv-2602-06208.md) |
| Evolutionarily Optimized Network Topology as Structural Prior (Topology-as-Prior | 10.64898/2026.03.12.711455 | bioRxiv | MODERATE — SUPPORTS Paper 4 | [card](papers/evolutionarily-optimized-network-topology-as-structural-prio.md) |
| A feedback control optimizer for online and hardware-aware training of Spiking N | 2602.13261 | 2026-02-03 | LOW — closest analog to Steersman in recent literature, diff | [card](papers/arxiv-2602-13261.md) |
| FourierFT | — | — | BM-003 piecewise anticipation (fixed spectral support) | [card](papers/fourierft.md) |
| GA-Net | — | — | BM-003 piecewise anticipation (GNN-in-adapter, physical grap | [card](papers/ga-net.md) |
| GoRA: Gradient-driven Adaptive Low Rank Adaptation | 2502.12171 | NeurIPS 2025 (v3 Oct 2025) | Should cite (brief mention) — establishes gradient-informed  | [card](papers/arxiv-2502-12171.md) |
| GraphLoRA | — | KDD 2025 | Structure-aware contrastive learning for cross-graph transfe | [card](papers/graphlora.md) |
| GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Trans | 2409.16670 | KDD 2025 | Conflicting guidance across scouts: Update 2 says do not cit | [card](papers/arxiv-2409-16670.md) |
| GRASP | — | — | BM-003 contrast class (dynamic/learned masks) | [card](papers/grasp.md) |
| HiP-LoRA: Budgeted Spectral Plasticity for Robust Low-Rank Adaptation | 2604.17751 | 2026-04-20 | claim-2 static-constraint neighbor | [card](papers/arxiv-2604-17751.md) |
| Hypernetwork-Driven Low-Rank Adaptation Across Attention Heads | 2510.04295 | arXiv | Coupling across heads, not within adapter — differentiated | [card](papers/arxiv-2510-04295.md) |
| HypeLoRA: Hyper-Network-Generated LoRA Adapters for Calibrated Language Model Fi | 2603.19278 | arXiv | OPTIONAL cite; if cited, distinguish inter-layer (theirs) vs | [card](papers/arxiv-2603-19278.md) |
| HyperAdapt: Simple High-Rank Adaptation | 2509.18629 | 2025-09-23 | claim-4 imposed-block list | [card](papers/arxiv-2509-18629.md) |
| IGU-LoRA: Adaptive Rank Allocation via Integrated Gradients and Uncertainty-Awar | 2603.13792 | ICLR 2026 | Should cite (brief mention) — state of the art for rank allo | [card](papers/arxiv-2603-13792.md) |
| iLoRA: Bayesian Low-Rank Adaptation with Latent Interaction Graphs for Microbiom | 2605.30179 | 2026-05-28 | BM-003 contrast class (per-input inferred graph) — cite to s | [card](papers/arxiv-2605-30179.md) |
| Kron-LoRA | — | — | claim-4 imposed-block list | [card](papers/kron-lora.md) |
| Kron-LoRA: Hybrid Kronecker-LoRA Adapters for Scalable, Sustainable Fine-tuning | 2508.01961 | arXiv Aug 2025 | Kronecker-structured block factorization; related but distin | [card](papers/arxiv-2508-01961.md) |
| L-MoE: End-to-End Training of a Lightweight Mixture of Low-Rank Adaptation Exper | 2510.17898 | arXiv | Routing between adapters, not coupling within — differentiat | [card](papers/arxiv-2510-17898.md) |
| Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning | 2602.04998 | 2026-02-04 | HIGH — CHALLENGES Paper 3 framing | [card](papers/arxiv-2602-04998.md) |
| Localized LoRA | — | — | claim-4 imposed-block list | [card](papers/localized-lora.md) |
| LoRA Dropout as a Sparsity Regularizer for Overfitting Control | 2404.09610 | arXiv | Closest work to Phase 3A overfitting detection, but uses tra | [card](papers/arxiv-2404-09610.md) |
| LoRA-Mixer: Coordinate Modular LoRA Experts Through Serial Attention Routing | 2507.00029 | arXiv | Cross-expert coordination, not within-adapter coupling — dif | [card](papers/arxiv-2507-00029.md) |
| LoRAGen: Structure-Aware Weight Space Learning for LoRA Generation | — | OpenReview | MODERATE for Paper 4 | [card](papers/loragen-structure-aware-weight-space-learning-for-lora-gener.md) |
| LoRMA: Low-Rank Multiplicative Adaptation for LLMs | 2506.07621 | ACL Findings 2025 | Multiplicative adaptation, not learnable coupling — brief me | [card](papers/arxiv-2506-07621.md) |
| MARS: Harmonizing Multimodal Convergence via Adaptive Rank Search | 2603.00720 | 2026-02-28 | MODERATE — EXTENDS | [card](papers/arxiv-2603-00720.md) |
| MLAE | — | — | BM-003 contrast class (dynamic/learned masks) | [card](papers/mlae.md) |
| MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning | 2603.24044 | arXiv | Not relevant — no bridge, no topology; not a competitor | [card](papers/arxiv-2603-24044.md) |
| MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning | 2405.12130 | arXiv | Restructures the update without factored structure — brief m | [card](papers/arxiv-2405-12130.md) |
| MoRe/Monarch | — | — | BM-003 piecewise anticipation (fixed permutations) | [card](papers/more-monarch.md) |
| MSLoRA-CR (contrastive regularization over modality-specific LoRA modules) | — | ACM MM 2025 | Different level of abstraction — contrastive loss between Lo | [card](papers/mslora-cr-contrastive-regularization-over-modality-specific-.md) |
| NeuroLoRA: Context-Aware Neuromodulation for Parameter-Efficient Multi-Task Adap | 2603.12378 | 2026-03-12 | HIGH — PARTIAL OVERLAP with Paper 3 | [card](papers/arxiv-2603-12378.md) |
| Beyond SGD, Without SVD: Proximal Subspace Iteration LoRA with Diagonal Fraction | 2602.16456 | 2026-02-18 | Tier 2 relevant background | [card](papers/arxiv-2602-16456.md) |
| R-LoRA: Randomized Multi-Head LoRA for Efficient Multi-Task Learning | 2502.15455 | NeurIPS 2025 area (per source) | MODERATE for Paper 3 | [card](papers/arxiv-2502-15455.md) |
| RadiX-Net | — | — | BM-003 piecewise anticipation (expander lineage) | [card](papers/radix-net.md) |
| Ramanujan (expander-lineage reference) | — | — | BM-003 piecewise anticipation (expander lineage) — underspec | [card](papers/ramanujan-expander-lineage-reference.md) |
| Regular Polytope Networks | 2103.15632 | IEEE TNNLS 2021 | Only known polytope use in neural networks; fixed classifier | [card](papers/arxiv-2103-15632.md) |
| ReMix: Reinforcement routing for mixtures of LoRAs in LLM finetuning | 2603.10160 | 2026-03-10 | EXTENDS (Tier 1, Scan 10) | [card](papers/arxiv-2603-10160.md) |
| Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models | 2402.02347 | arXiv / ICML 2025 (MoE extension) | Already tracked as GeoLoRA-adjacent; optimization geometry,  | [card](papers/arxiv-2402-02347.md) |
| SparseLoRA | — | — | BM-003 contrast class (dynamic/learned masks) | [card](papers/sparselora.md) |
| Spectral Imbalance Causes Forgetting in Low-Rank Continual Adaptation | 2602.00722 | 2026-01-31 | claim-2 static-constraint neighbor (Stiefel constraint) | [card](papers/arxiv-2602-00722.md) |
| Not All Directions Matter: Towards Structured and Task-Aware Low-Rank Model Adap | 2603.14228 | arXiv March 2026 | Closest architectural cousin (graph-based coordinator), but  | [card](papers/arxiv-2603-14228.md) |
| SURM | — | — | BM-003 piecewise anticipation (circulant = cyclic Cayley gra | [card](papers/surm.md) |
| SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors | 2405.19597 | 2024-05-30 | BM-003 mandatory ablation (SVFT-Random matched-params); clos | [card](papers/arxiv-2405-19597.md) |
| Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to | 2501.08727 | arXiv | Transform on pretrained weight, not between A and B — differ | [card](papers/arxiv-2501-08727.md) |
| Variance-Aware Loss Scheduling for Multimodal Alignment in Low-Data Settings | 2503.03202 | 2025-03-05 | MODERATE — SUPPORTS Paper 3 Steersman concept | [card](papers/arxiv-2503-03202.md) |
| X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Lang | 2402.07148 | arXiv | Cross-adapter routing, not within-adapter coupling — differe | [card](papers/arxiv-2402-07148.md) |

## Non-Paper Resources

| Name | Type | URL | Source |
|---|---|---|---|
| Stanford Alpaca: An Instruction-following LLaMA Model (Taori, Rohan; Gulrajani, Ishaan; Zhang, Tianyi; Dubois, Yann; Li, Xuechen; Guestrin, Carlos; Liang, Percy; Hashimoto, Tatsunori B., 2023) | github-repo / dataset | https://github.com/tatsu-lab/stanford_alpaca | `['C:\\falco\\rhombic\\paper\\rhombic-paper3.bib', 'C:\\falco\\rhombic\\paper\\refs-paper3.bib', 'C:\\falco\\rhombic\\paper\\paper4\\paper4.bib']` |
| TASUMER MAF GitHub organization | github-organization | https://github.com/tasumermaf | `C:\Users\Timothy Paul Bielec\Downloads\I'll start by reading the onboarding profile to pin down the exact….md` |
| timothybielec.com (personal website) | personal-website | http://www.timothybielec.com/ | `C:\Users\Timothy Paul Bielec\Downloads\I'll start by reading the onboarding profile to pin down the exact….md` |
| timotheospaul Substack | blog | https://timotheospaul.substack.com/ | `C:\Users\Timothy Paul Bielec\Downloads\I'll start by reading the onboarding profile to pin down the exact….md` |
| TRL Issue #4147 — Prompt-completion label bug (assistant_masks override) | github-issue | https://github.com/huggingface/trl/issues/4147 | `C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md` |
| TRL Issue #5213 — Tokenization mismatch (prompt vs prompt+completion) | github-issue | https://github.com/huggingface/trl/issues/5213 | `C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md` |
| TRL Issue #1184 — Instruction template masking failure | github-issue | https://github.com/huggingface/trl/issues/1184 | `C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md` |
| TRL Issue #3927 — assistant_only_loss silent failure with Liger kernel | github-issue | https://github.com/huggingface/trl/issues/3927 | `C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md; C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| TRL Issue #1507 — SFTTrainer without DataCollatorForCompletionOnlyLM | github-issue | https://github.com/huggingface/trl/issues/1507 | `C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md` |
| Unsloth Discussion #3810 — Trouble fine-tuning Nemotron 3 Nano | github-discussion | https://github.com/unslothai/unsloth/discussions/3810 | `C:\falco\rhombic\competition\RESEARCH_GRPO.md; C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| llama.cpp Issue #20570 — Nemotron Mamba assertion failure | github-issue | https://github.com/ggml-org/llama.cpp/issues/20570 | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| NeMo Issue #14856 — LoRA recipe request for Nemotron | github-issue | https://github.com/NVIDIA-NeMo/NeMo/issues/14856 | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| transformers Issue #38966 — Nemotron-H HF Transformers support | github-issue | https://github.com/huggingface/transformers/issues/38966 | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| SGLang rms_norm_eps AttributeError (issue referenced without link) | github-issue | None | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| TRL GRPOTrainer documentation | documentation | https://huggingface.co/docs/trl/main/en/grpo_trainer | `C:\falco\rhombic\competition\RESEARCH_GRPO.md; C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| TRL GRPOConfig source (grpo_config.py) | github-source | https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| TRL GRPOTrainer source (grpo_trainer.py) | github-source | https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| TRL SFTTrainer source (sft_trainer.py) | github-source | https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py | `C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md` |
| TRL DataCollatorForCompletionOnlyLM source (utils.py) | github-source | https://github.com/huggingface/trl/blob/15ff54790b42297d2cf569fba6d7dd44c1c269e3/trl/trainer/utils.py | `C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md` |
| mamba_ssm rmsnorm_fn source (state-spaces/mamba layernorm_gated.py) | github-source | https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py | `C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md; C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| NVIDIA Nemotron-3-Nano-30B modeling code (modeling_nemotron_h.py) | model-source | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/modeling_nemotron_h.py | `C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md` |
| Nemotron-3-Nano-30B HuggingFace model card | model-card | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | `C:\falco\rhombic\competition\RESEARCH_INTEL.md; C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| Nemotron 3 Nano HuggingFace blog post | blog-post | https://huggingface.co/blog/nvidia/nemotron-3-nano-efficient-open-intelligent-models | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| Megatron Bridge — Nemotron 3 Nano documentation | documentation | https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| NVIDIA blog — Train a Reasoning LLM in One Weekend with NeMo | blog-post | https://developer.nvidia.com/blog/train-a-reasoning-capable-llm-in-one-weekend-with-nvidia-nemo/ | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| NeMo Gym — Nemotron 3 Nano GRPO recipe | documentation | https://docs.nvidia.com/nemo/gym/latest/model-recipes/nemotron-3-nano.html | `C:\falco\rhombic\competition\RESEARCH_GRPO.md; C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| NVIDIA NeMo RL (GitHub repository) | github-repo | https://github.com/NVIDIA-NeMo/RL | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| NeMo RL GRPO guide | documentation | https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| NeMo RL grpo_math_1B.yaml example config | github-source | https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B.yaml | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| Nemotron 3 Super RL stage documentation | documentation | https://docs.nvidia.com/nemotron/nightly/nemotron/super3/rl/index.html | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| NVIDIA-NeMo/Nemotron GRPO-DAPO training cookbook | github-repo | https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super/grpo-dapo | `C:\falco\rhombic\competition\RESEARCH_INTEL.md; C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| NVIDIA NeMo RL DeepScaleR reproduction blog post | blog-post | https://developer.nvidia.com/blog/reinforcement-learning-with-nvidia-nemo-rl-reproducing-a-deepscaler-recipe-using-grpo/ | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| Fast-Math-R1 repository (analokmaus/kaggle-aimo2-fast-math-r1) | github-repo | https://github.com/analokmaus/kaggle-aimo2-fast-math-r1 | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| Cameron Wolfe — Group Relative Policy Optimization (GRPO) deep dive | blog-post | https://cameronrwolfe.substack.com/p/grpo | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| Sebastian Raschka — The State of RL for LLM Reasoning | blog-post | https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| HuggingFace Cookbook — TRL GRPO advanced reward design | documentation | https://huggingface.co/learn/cookbook/en/trl_grpo_reasoning_advanced_reward | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| Unsloth — Nemotron 3 fine-tuning guide | documentation | https://unsloth.ai/docs/models/nemotron-3 | `C:\falco\rhombic\competition\RESEARCH_INTEL.md; C:\falco\rhombic\competition\RESEARCH_TRL_BUGS.md` |
| Unsloth reinforcement learning guide | documentation | https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| Unsloth RL Environments blog post | blog-post | https://unsloth.ai/blog/rl-environments | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| nvidia/Nemotron-Post-Training-Dataset-v2 | dataset | https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2 | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| nvidia/Nemotron-Post-Training-Dataset-v1 | dataset | https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1 | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| nvidia/Nemotron-3-Nano-RL-Training-Blend | dataset | https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| nvidia/OpenMathReasoning | dataset | None | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| nvidia/Llama-Nemotron-Post-Training-Dataset | dataset | None | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| trl-lib/DeepMath-103K | dataset | None | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| Light-R1-SFT dataset | dataset | None | `C:\falco\rhombic\competition\RESEARCH_GRPO.md` |
| nvidia/OpenMath-Nemotron-14B-Kaggle | model-card | https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| AIMO-2 1st place solution writeup (NemoSkills, Kaggle) | competition-writeup | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| ARC Prize 2025 1st place (NVARC / NVIDIA) — referenced without link | competition-writeup | None | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| NVIDIA corporate blog — Math Reasoning Kaggle Win | blog-post | https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/ | `C:\falco\rhombic\competition\RESEARCH_INTEL.md` |
| WeightWatcher-PEFT | tool/blog | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| CoLorAI @ ICML 2026 | workshop | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| MERGE-PEFT | benchmark | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| NLPCC 2025 | venue | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| CCL 2025 | venue | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| Chinese ML media (机器之心 / 量子位 / PaperWeekly) | media | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| Chinese corporate labs (ByteDance Seed, DeepSeek, Zhipu, Moonshot, DAMO/Qwen, Noah's Ark) | corporate-lab-sweep | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| kexue.fm | blocked-source | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| Zhihu roundup (unspecified) | blocked-source | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| 《计算机学报》 / 《软件学报》 | venue-not-swept | None | `docs/LITERATURE_WATCH_2026-07-03.md` |
| HuggingFace PEFT library | software-library | https://huggingface.co/docs/peft | `C:\falco\rhombic\orvad-research-telora-competitive-intel.md; C:\falco\rhombic\orvad-research-telora-competitive-intel-update3.md; C:\falco\rhombic\results\research-scout-2026-03-27.md` |
| HuggingFace PEFT — GraLoRA package reference page | documentation | https://huggingface.co/docs/peft/package_reference/gralora | `C:\falco\rhombic\orvad-research-telora-competitive-intel-update3.md` |
| Semantic Scholar API (LoRA-XS citation watch) | api-service | None | `C:\falco\rhombic\results\research-scout-2026-03-27.md` |
| TLoRA GitHub repository | github-repo | None | `C:\falco\rhombic\results\research-scout-2026-03-26.md` |
| ResearchGate summary of Zhang et al. (arXiv 2504.19583) | summary-page | None | `C:\falco\rhombic\orvad-research-telora-competitive-intel.md` |
| emergentmind topic pages | topic-pages | None | `C:\falco\rhombic\results\research-scout-2026-03-26.md` |
| Stack Overflow LoRA variants discussions | forum | None | `C:\falco\rhombic\results\research-scout-2026-03-26.md` |
| dkarkada/symmetry-stats-repgeom | repo | https://github.com/dkarkada/symmetry-stats-repgeom | `C:\falco\rhombic\docs\KARKADA_EVALUATION.md` |
| selimbat/r_dodeca_vox_builder | repo | https://github.com/selimbat/r_dodeca_vox_builder | `C:\falco\rhombic\docs\EXTERNAL_REFERENCE_IMPLEMENTATIONS.md` |
| grischa/rhombic-dodecahedron | repo | https://github.com/grischa/rhombic-dodecahedron | `C:\falco\rhombic\docs\EXTERNAL_REFERENCE_IMPLEMENTATIONS.md` |
| Uspectacle/MineCraft-Rhombic-Dodecahedron | repo | https://github.com/Uspectacle/MineCraft-Rhombic-Dodecahedron | `C:\falco\rhombic\docs\EXTERNAL_REFERENCE_IMPLEMENTATIONS.md` |
| Reispfannenfresser/marching-rhombic-dodecahedrons | repo | https://github.com/Reispfannenfresser/marching-rhombic-dodecahedrons | `C:\falco\rhombic\docs\EXTERNAL_REFERENCE_IMPLEMENTATIONS.md` |
| apiotrow/Rhombic-Dodecahedral-Honeycomb | repo | https://github.com/apiotrow/Rhombic-Dodecahedral-Honeycomb | `C:\falco\rhombic\docs\EXTERNAL_REFERENCE_IMPLEMENTATIONS.md` |
| HomelikeBrick42 (GitHub user; repository name not given in source) | repo | None | `C:\falco\rhombic\docs\EXTERNAL_REFERENCE_IMPLEMENTATIONS.md` |
| NateBerglund (GitHub user; repository name not given in source) | repo | None | `C:\falco\rhombic\docs\EXTERNAL_REFERENCE_IMPLEMENTATIONS.md` |
| MaltWhiskey (GitHub user; repository name not given in source) | repo | None | `C:\falco\rhombic\docs\EXTERNAL_REFERENCE_IMPLEMENTATIONS.md` |
| Survey aggregate — 7 additional surveyed repos not named individually in source (zero extractable value) | repo-survey-aggregate | None | `C:\falco\rhombic\docs\EXTERNAL_REFERENCE_IMPLEMENTATIONS.md` |
| Negative search: "24-cell" OR "D4 lattice" AND "machine learning" | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: "topology programming" AND neural | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: "cybernetic feedback" + "LoRA" or "adapter" | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: "polytope" + "adapter" or "LoRA" | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: "Fiedler value" + "adapter" or "LoRA" | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: "contrastive loss" + "block diagonal" + "adapter" | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: "multi-channel LoRA" + "coupling" | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: Anthropic/DeepMind/OpenAI LoRA papers (March 2026) | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: "loss coefficient" + "structural formation" | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: "lattice topology" + neural network | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: "phase transition" + "training dynamics" + LoRA | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: LoRA "graph Laplacian" / "Fiedler" / "bridge matrix" + adapter coupling | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: CW-001 "speed not destination" prior-art check (contrastive coefficient as formation-rate control) | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search: HuggingFace Hub — geometric/topological/block-diagonal LoRA repos, models, spaces (Scans 6-16) | negative-search-result | https://huggingface.co | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| GitHub: skydancerosel/mini_gpt (Spectral Edge Dynamics code) | github-repo | https://github.com/skydancerosel/mini_gpt | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| GitHub: ronantakizawa/lorashare (Share implementation library) | github-repo | https://github.com/ronantakizawa/lorashare | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| GitHub: ankit-vaidya19/Share (Share official implementation) | github-repo | https://github.com/ankit-vaidya19/Share | `C:\falco\rhombic\docs\LITERATURE_WATCH.md` |
| Negative search (snapshot): "24-cell" OR "D4 lattice" AND "machine learning" | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH_2026_03_19.md` |
| Negative search (snapshot): "topology programming" AND neural | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH_2026_03_19.md` |
| Negative search (snapshot): Anthropic/DeepMind LoRA papers (March 2026) | negative-search-result | None | `C:\falco\rhombic\docs\LITERATURE_WATCH_2026_03_19.md` |
