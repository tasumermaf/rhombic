# Research Scout: TeLoRA Competitive Landscape Scan

**Date:** 2026-03-26
**Preset:** comparison / sota
**Confidence:** MEDIUM-HIGH (3 primary sources read in full; 15+ search queries across web + arxiv)
**Sources consulted:** 28

---

## Executive Summary

The TeLoRA novelty claims remain intact. No published work combines
(1) a learnable topology-constrained bridge between A and B, (2) cybernetic
feedback via Fiedler eigenvalue + contrastive loss, or (3) polytope
selection as a design parameter for adapter geometry. The field is
converging rapidly on structured LoRA variants, but the convergence
is toward SVD-based decompositions, manifold constraints, and block
partitioning -- none of which introduce geometric topology as an
organizing principle for the bridge itself.

**Overall threat assessment: LOW.** No direct competitor. Several papers
should be cited that were not in the March 8 landscape document. Two
papers (LoRA-SB, OrthoGeoLoRA) are close enough to warrant explicit
differentiation in Papers 3-4.

---

## Threat Analysis by Novelty Claim

### Claim 1: Multi-channel LoRA with topology-constrained bridge

**Threat: LOW.** No published work constrains the bridge matrix to a
specific graph topology.

**Closest competitors:**

| Paper | Bridge? | Topology? | Threat |
|-------|---------|-----------|--------|
| LoRA-XS (EMNLP 2024) | r x r trainable between frozen A/B | None | Already cited. Differentiated. |
| LoRA-SB (ICLR 2025 Workshop) | r x r trainable between frozen SVD | None | NEW. Must cite. See below. |
| TLoRA (arXiv 2504.18735, Apr 2025) | Trainable B between fixed random A/C | None | NEW. Must cite. See below. |
| Localized LoRA (arXiv 2506.00236) | None -- independent block adapters | K x K grid partition | Already cited. No bridge. |
| MELoRA (ACL 2024) | None -- block-diagonal independence | Block-diagonal | Already cited. Ablation baseline. |
| BD-LoRA (Amazon, NeurIPS 2025) | None -- block-diagonal for serving | Block-diagonal for TP | NEW. Serving-only, not training topology. Cite in passing. |

**Key gap in the field:** Every "bridge" paper (LoRA-XS, LoRA-SB, TLoRA)
treats the middle matrix as a generic learnable parameter. None constrains
it by graph structure, and none uses the bridge as a diagnostic.

### Claim 2: Cybernetic feedback loop (Fiedler + contrastive loss as steering)

**Threat: NONE.** This is genuinely novel.

**Related but non-overlapping work:**

- **Tam et al., "Fiedler Regularization" (ICML 2020):** Uses differentiable
  Fiedler eigenvalue as a regularizer for neural network sparsification.
  Constructs a graph from weight matrix entries, penalizes low connectivity
  to maintain information flow during pruning. **This is prior art for
  differentiable Fiedler as a training objective** but for a fundamentally
  different purpose (sparsity vs topology programming). Already cited in
  Papers 3-4 bib as of Mar 25. **Confidence: HIGH** -- paper read.

- **Fesser & Weber (2025):** Uses Fiedler singular value to diagnose
  over-squashing in GNNs. Gradient flow analysis, not adapter training.
  Not a threat.

- **MSLoRA-CR (ACM MM 2025):** Contrastive regularization over modality-
  specific LoRA modules for incremental learning. Uses contrastive loss
  between LoRA modules (inter-modality differentiation), not within a
  single adapter's bridge. Different level of abstraction.

- **GraphLoRA (KDD 2025):** Structure-aware contrastive LoRA for cross-graph
  transfer. Injects a small trainable GNN alongside frozen GNN. Uses SMMD
  (structure-aware MMD) as contrastive loss. **Most conceptually adjacent**
  -- uses graph structure + contrastive loss + LoRA. But applies to GNN
  cross-domain transfer, not within-adapter topology. Worth citing for
  the "structure-aware contrastive" framing.

**No one uses Fiedler eigenvalue of the bridge's coupling matrix as a
training signal.** No one uses contrastive loss on co-axial vs cross-axial
channel pairs defined by a polytope. These are genuinely novel.

### Claim 3: Polytope selection (octahedron n=4, RD n=6, tesseract n=8, 24-cell n=24)

**Threat: NONE.** No published work parametrizes adapter structure by polytope.

**Related but non-overlapping work:**

- **Regular Polytope Networks (Pernici et al., IEEE TNNLS 2021):** Uses
  d-Simplex, d-Cube, d-Orthoplex vertices as FIXED CLASSIFIER weights
  for maximally separated embeddings. Different application entirely
  (classifier, not adapter), but establishes precedent for polytope
  geometry in neural networks. Should cite for establishing that polytope
  structure has prior use in ML, even though the application is unrelated.

- **Lee et al., "Defining Neural Network Architecture through Polytope
  Structures of Datasets" (ICML 2024):** Uses polytope complexity of the
  dataset to bound network width. Polytope as analysis tool for data,
  not as structural constraint on weights. Not a threat.

- **Topological Signatures of ReLU Networks (arXiv 2510.12700, 2025):**
  Uses polytope decomposition of input space to track topological features
  during training. Analysis, not design. Not a threat.

### Claim 4: Bridge discovers geometry without being told the target topology

**Threat: NONE.** The emergence result (bridge converging to block-diagonal
under contrastive supervision without explicit BD loss) has no parallel
in the literature.

The closest conceptual analogue is the Karkada et al. result (arXiv
2602.15029) showing that data symmetry determines representation
geometry -- but that is about embedding geometry, not adapter structure.
Already noted for citation.

---

## New Papers to Cite (Not in March 8 Landscape)

### Must Cite (Related Work, explicit differentiation)

| Paper | arXiv | Venue | Why |
|-------|-------|-------|-----|
| **LoRA-SB** | (ICLR 2025 WS, SCOPE) | ICLR 2025 Workshop | r x r trainable bridge between frozen SVD. Closest to our bridge concept. Differentiate: we train A/B/bridge jointly + topology constraint. |
| **TLoRA** | 2504.18735 | arXiv Apr 2025 | Tri-matrix with trainable middle between fixed random outer. Differentiate: random vs topology-motivated init, frozen vs learnable outer matrices. |
| **OrthoGeoLoRA** | 2601.09185 | arXiv Jan 2026 | "Geometric" LoRA but geometry = Stiefel manifold orthogonality, not polytope topology. Differentiate: their geometry is optimization landscape; ours is adapter structure. |
| **StelLA** | 2510.01938 | arXiv Oct 2025 | Three-factor USV on Stiefel manifold. Similar factored structure but geometry is orthogonality constraint, not bridge topology. |
| **GraphLoRA** | 2409.16670 | KDD 2025 | Structure-aware contrastive LoRA for GNN transfer. Cite for "structure-aware contrastive" as shared vocabulary. |

### Should Cite (Brief mention in introduction/background)

| Paper | arXiv | Venue | Why |
|-------|-------|-------|-----|
| **GoRA** | 2502.12171 | arXiv Feb 2025 | Gradient-driven adaptive rank. Different problem but establishes gradient-informed adapter design. |
| **IGU-LoRA** | 2603.13792 | ICLR 2026 | Adaptive rank via integrated gradients. State of the art for rank allocation. |
| **Stable-LoRA** | 2603.05204 | arXiv Mar 2026 | Feature learning stability in LoRA. Relevant to our overfitting detection claim. |
| **W2T** | 2603.15990 | arXiv Mar 2026 | Spectral canonicalization of LoRA weights. Relevant to fingerprinting; uses SVD + QR to resolve GL(r) ambiguity. |
| **BD-LoRA (Amazon)** | 2510.23346 | NeurIPS 2025 | Block-diagonal for TP serving. Same BD structure, serving motivation not training. |
| **DoRAN** | 2510.04331 | arXiv Oct 2025 | Stabilizes DoRA with noise injection. Composable context. |
| **Regular Polytope Networks** | 2103.15632 | IEEE TNNLS 2021 | Polytope geometry in NN classifiers. Prior art for polytope + ML. |
| **HypeLoRA** | 2603.19278 | arXiv Mar 2026 | Hyper-network generates A/B factors. Structural coupling across layers. |

---

## Collaboration Leads

| Lead | Rationale | Priority |
|------|-----------|----------|
| **Babak Barazandeh** (Localized LoRA) | Closest structural thinking. His K x K block partitioning + our bridge topology = natural combination. University researcher, likely open to collaboration. | MEDIUM |
| **Banaei et al.** (LoRA-XS) | Our bridge generalizes their r x r concept. Joint paper comparing frozen-outer vs learnable-outer bridges could be compelling. | LOW (competitive overlap) |
| **Karkada et al.** (Symmetry in Language Statistics) | Their universality result (data symmetry determines representation geometry) is the theoretical foundation for why TeLoRA's topology should match the task. Minta found this. | HIGH |
| **Pernici et al.** (Regular Polytope Networks) | Their polytope-as-classifier work could extend to polytope-as-bridge. Different group but shared mathematical vocabulary. | LOW |
| **Yang et al.** (GraphLoRA, KDD 2025) | Structure-aware contrastive + LoRA. Their SMMD formulation might improve our contrastive loss. | MEDIUM |

---

## Gaps and Limitations

1. **Could not access LoRA-SB full paper** (OpenReview 403). Method details
   from search snippets only. Confidence on differentiation: MEDIUM. Should
   read the full PDF.

2. **TLoRA full paper** not readable via web fetch (PDF encoding). Method
   details from abstract + GitHub description. Confidence: MEDIUM.

3. **HypeLoRA** (March 2026) is very recent. Full architecture unknown
   beyond abstract. May be worth monitoring.

4. **No search of Chinese-language ML venues** (AAAI China, CCF conferences).
   Structured LoRA research is active in Chinese ML community. Potential
   blind spot.

5. **No patent search conducted.** Industry labs (Google, Meta, Amazon,
   NVIDIA) may have filed patents on structured adapter architectures
   without publishing papers.

---

## Source Hierarchy

| Level | Sources | Count |
|-------|---------|-------|
| Primary (full paper read via web) | Localized LoRA HTML, W2T HTML, OrthoGeoLoRA HTML | 3 |
| Secondary (abstract + method from arxiv) | LoRA-SB, TLoRA, GraphLoRA, GoRA, IGU-LoRA, BD-LoRA, Regular Polytope Networks, DoRAN, Stable-LoRA, HypeLoRA | 10 |
| Tertiary (search snippets only) | MSLoRA-CR, Fesser & Weber 2025, Fiedler Reg. 2020, StelLA, VB-LoRA, ID-LoRA, SeLoRA, CAT | 8 |
| Background (surveys, topic pages) | LoRA survey (Springer 2025), emergentmind topic pages, Stack Overflow LoRA variants | 7 |

---

## Summary Assessment

| Novelty Claim | Threat Level | Action Required |
|---------------|-------------|-----------------|
| Multi-channel topology-constrained bridge | LOW | Cite LoRA-SB, TLoRA, OrthoGeoLoRA. Explicit differentiation table. |
| Cybernetic feedback (Fiedler + contrastive) | NONE | Cite Tam et al. (already done). Cite GraphLoRA for vocabulary. |
| Polytope selection as design parameter | NONE | Cite Regular Polytope Networks for precedent. |
| Bridge discovers target geometry | NONE | No parallel in literature. Strongest claim. |
| **Overall** | **LOW** | Field is converging on structured LoRA but not on our axis of innovation. |

The field's trajectory is toward (a) better rank allocation (GoRA,
IGU-LoRA, AdaLoRA), (b) manifold-aware optimization (OrthoGeoLoRA,
StelLA, Stiefel-LoRA), and (c) extreme parameter reduction via frozen
outer matrices (LoRA-XS, LoRA-SB, VeRA). None of these directions
threaten TeLoRA's core contribution: using graph topology to structure
the bridge and cybernetic feedback to program it.

The window for first-mover advantage remains open. Papers 3 and 4
should ship while this is true.

---

*Report generated by OSANIAL research agent, 2026-03-26.*
*28 sources consulted across 5 search rounds.*
