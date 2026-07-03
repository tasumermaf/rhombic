# Literature Watch — July 3, 2026 (Fourth Expansion)

> **Sweep window:** April 6 – July 3, 2026 (post-pause currency check) + targeted backfill of missed prior art.
> **Method:** Five parallel scouts — (1) core novelty claims, (2) structural-mask adapters / BM-003+BM-004 prior art, (3) representation-alignment cluster, (4) Chinese-language venues (queries run in Chinese; NLPCC/CCL/《自动化学报》/corporate labs), (5) fingerprinting & diagnostics. ~85 distinct web queries, ~40 abstract fetches.
> **Prior sweeps:** March 19 (third expansion, 35 papers), April 5 (non-English landscape, 34 primary).
> **Headline:** Claims 3 and 4 remain fully unoccupied. Claim 1's strong form is dead (missed prior art). Claim 2 survives narrowly. Claim 5 is partially eroded and its window is measured in weeks-to-months. The Tier-3 principal-angle experiment and the BM-003/BM-004 combinations remain unclaimed but have named neighbors and concurrent work.

---

## PER-CLAIM VERDICTS (supersedes April 5 assessment)

| # | Claim | Verdict | Governing evidence |
|---|-------|---------|--------------------|
| 1 | Learnable bridge between learnable A and B | **STRONG FORM DEAD — restate** | **MoSLoRA** (arXiv:2406.11909, EMNLP 2024): learnable dense mixer between learnable A/B, frames vanilla LoRA as identity-mixer special case — missed by all four prior sweeps. **StelLA** (2510.01938, NeurIPS 2025 Spotlight): full learnable r×r matrix between learnable Stiefel-constrained factors. **AdaLoRA** (2303.10512, ICLR 2023): learnable diagonal Λ between learnable P,Q. **BoRA** (2508.06953): learnable diagonal block bridges. Surviving claim: *multi-channel bridge with programmable/structural topology + diagnostics* — the shape is lineage; what the bridge does is ours. |
| 2 | Cybernetic closed-loop spectral feedback on adapter internals | **SURVIVES NARROWLY** | **FlexLoRA** (2601.22905, ICLR 2026, SJTU/SEU): spectral-energy-entropy → live rank prune/expand — a measurement→action loop, but on rank budgets, not topology/update shaping. Static-constraint neighbors: CRMA (2606.00382, Sinkhorn-bounded mixing matrix), Spectral Imbalance (2602.00722, Stiefel constraint), HiP-LoRA (2604.17751). One-shot: Spectral Surgery (2603.03995). Diagnostics-only: 2604.08844. No closed-loop in-training feedback controller for topology found. **Crowding HIGH; publish window narrowing.** |
| 3 | Polytope selection as adapter design parameter | **SURVIVES CLEAN** | Zero hits, English and Chinese, including explicit polytope/dodecahedron/24-cell/lattice-adjacency queries. |
| 4 | Emergent block-diagonal from geometric pair specification | **SURVIVES CLEAN** | All new block-structure work is imposed: MELoRA (ACL 2024, Shandong — cite), GraLoRA (2505.20355, NeurIPS 2025 spotlight), Localized LoRA, Kron-LoRA, HyperAdapt (2509.18629), BoRA, DiaBlo, BD-LoRA. Emergent-BD elsewhere (Block-Recurrent ViT) unchanged. |
| 5 | Adapter internals as diagnostics (fingerprinting, overfit) | **PARTIALLY ERODED — reframe and accelerate** | **W2T** (2603.15990, Mar 2026): weight-space community enters PEFT — attribute classification/retrieval from LoRA checkpoints at hub scale (10k+ adapters). **Spectral Geometry** (2604.08844): linear classifiers on adapter spectra read training objective (AUC≈1.00 within-method; cross-method fails). Backdoor forensics from weights alone (2602.15195, +500-adapter benchmark). **Merge-conflict prediction preempted in training-time form** by 2606.19549 (Jun 17) — ours must be framed weight-only/post-hoc. **Overfit detection from adapter internals: still unclaimed** (cite WeightWatcher-PEFT as prior art in kind). |

## KEY DISCOVERIES BY THREAD

### A. The three-factor lineage (claim 1 restatement set)
MoSLoRA (2406.11909) · StelLA (2510.01938) · AdaLoRA (2303.10512) · BoRA (2508.06953) · Lily (2407.09946, routed A–B coupling) · CeRA (2602.22911, nonlinear) · CRMA (2606.00382) · GraphLoRA-Rec (2606.07526, trainable message-passing in the LoRA pathway, imposed data-graph) · TLoRA/ID-LoRA/LoRA-XS family (frozen outers). Taxonomy check: the Unified LoRA survey (2601.22708) still has no "bridge between learnable factors" category — naming the class remains available.

### B. BM-003 prior-art map (structural mask × learnable edge weights)
**Nothing anticipates the full combination** (learnable A/B + fixed semantically-chosen graph-adjacency mask on the bridge + LM loss only). Piecewise anticipation: **SVFT** (2405.19597 — fixed sparsity patterns [diagonal/banded/random/top-k] × learnable values between frozen SVD factors; **SVFT-Random is the mandatory matched-params ablation**) · MoSLoRA (dense-bridge upper envelope = our Config F) · BOFT (butterfly topology) · SURM/C3A (circulant = cyclic Cayley graph) · MoRe/Monarch (fixed permutations) · FourierFT (fixed spectral support) · GraLoRA (hard block prior, LM-only, NeurIPS 2025) · EEG-GraphAdapter/GA-Net (GNN-in-adapter, physical graphs) · expander-lineage (RadiX-Net, Ramanujan). Contrast class (dynamic/learned masks — cite to sharpen "no controller"): iLoRA (2605.30179, per-input inferred graph), GRASP, SparseLoRA, MLAE. Positioning: (1) graph-theoretic/polyhedral adjacency vs diagonal/banded/butterfly/circulant/block; (2) mask in the bridge with both A and B learnable; (3) geometric-semantic motivation.

### C. BM-004 concurrent work (two-factor: prior × data symmetry)
**No factorial study in adapter space found.** Concurrent outside PEFT: **Symmetry–Data Exchange Rate** (2606.01090, May 31, 2026) — wrong-group prior *actively harmful*, aligned prior pays (β=1.28 vs theory 1.0); cite as concurrent, differentiate on setting (adapters over pretrained LM) and prior type (graph adjacency vs group equivariance). Theory: Aligning Network Equivariance with Data Symmetry (2605.13744). Background: Does Equivariance Matter at Scale (2410.23179, ICLR 2025) · ELoRA (equivariance-preserving PEFT exists as category) · Equivariant Adaptation (2310.01647). **Timing risk real — run BM-004 soon.**

### D. Representation alignment / Tier-3 principal-angle experiment
**The experiment remains unclaimed** — nobody measures principal angles between two modality adapters' subspaces or validates a bridge against angle-based closability. But the protocol now has mandatory components:
1. **Permutation/random-subspace null** matched for rank and dimension — the ICML 2026 "Aristotelian" critique (2602.14486) shows uncalibrated similarity scores inflate with depth/width; convergence may be topological (neighbor relations) not geometric — which also motivates testing nonlinear/local-structure bridges against linear ones.
2. **Gauge invariance** — Fréchet-averages paper (2604.27155): LoRA's A→AG, B→G⁻¹B symmetry makes raw factor angles gauge-dependent; measure on quotient-invariant quantities (column spaces of B·A, or gauge-fixed factors).
3. **Singular-value-weighted angle spectra** (TARA-Merging, 2603.26299: coverage + anisotropy, not bare angles).
4. **Behavioral endpoints** alongside geometric ones (2605.23315: models converge representationally while diverging functionally).
5. **Baselines to beat:** Cross-LoRA's Frobenius-optimal closed-form map (2508.05232 — cross-model linear bridge prior art) · mini-vec2vec linear alignment (2510.02348) · GAM/Procrustes merging (2606.00357).
Competing predictions our measurement adjudicates: **UWSH** (2512.05117, JHU — universal low-dimensional weight subspaces ⇒ small angles) vs **near-orthogonality** (2602.19367 anchor) — and **Crowded in B-Space** (2604.16826) predicts the asymmetry: B-side crowded/shared, A-side task-specific. **The JHU group (UWSH + Share 2602.06043) is the most likely to run our experiment next.** Adjacent: HeRA (2606.23885, head-level MKNN alignment); AuRA (2606.11033, distillation route that sidesteps geometry — narrows the bridge's value prop to data-free settings); ReAlign (2602.07026); D1 reference point: adapter-vs-base principal angles ≈74° stable across ranks (2605.28896, provisional).

### E. Fingerprinting/diagnostics (Tier 1B) — reframing requirements
1. **The W2T tension is a feature:** their flattened-weight baselines collapse (2.87% macro-F1, factorization non-uniqueness at hub scale); ours reach 72.3% linearly within a controlled family (shared init/base/rank). State the regime split explicitly: *canonicalization is necessary at hub scale, unnecessary within an adapter family* — that contrast is itself a finding.
2. **Merge-conflict prediction:** reframe as weight-only, post-hoc, no-training-access (2606.19549 owns the training-time form; MERGE-PEFT is a usable benchmark).
3. **Bridge-swap corroborated independently** by Crowded in B-Space (B shared, A task-specific) — cite as convergent evidence, claim the architectural generalization.
4. Cite as prior art in kind: LoL/GL-equivariant processing (2410.04207), weights2weights (2406.09413), Tree Experts (2410.13569), WeightWatcher-PEFT (blog/tool), backdoor forensics (2602.15195); survey: 2603.10090.
5. **Sequencing implication: 1B before 1A.** 1A (audit methodology) has no visible crowding; 1B's lane filled with six directly-relevant papers Jan–Jun 2026.

### F. Chinese-language venues (blind spot CLOSED)
NLPCC 2025 (all four dblp volumes): zero structured-LoRA methods papers. CCL 2025: LoRA as tool only. 《自动化学报》 flagship PEFT survey (~263 refs): none of the five claims' mechanisms present. Chinese ML media (机器之心/量子位/PaperWeekly): secondary coverage of English arXiv work only. Corporate labs (ByteDance Seed public papers, DeepSeek, Zhipu, Moonshot, DAMO/Qwen, Noah's Ark): nothing structural on adapters. Residual gaps: kexue.fm and one Zhihu roundup 403-blocked (known content orthogonal); 《计算机学报》/《软件学报》 not individually swept.

### G. Venue watch
**CoLorAI @ ICML 2026** (first workshop on low-rank representations, Seoul) — both the likeliest surface for claim-1/4-adjacent work and a natural Stream A submission target. Re-sweep after camera-readies.

## INTER-SCOUT DISCREPANCY RESOLVED
The Chinese-venue scout reported "no dense learnable bridge between two learnable outer matrices found anywhere"; the structural-mask scout found MoSLoRA (2406.11909), which is exactly that (its A/B remain standard trainable LoRA factors; the mixer is trainable dense r×r). MoSLoRA stands; the Chinese scout's negative was a coverage gap in its query set, not a contradiction in the literature. Claim-1 language must be written against MoSLoRA.

## ACTION ITEMS
1. Read in full: MoSLoRA, StelLA, SVFT, W2T, Crowded in B-Space, 2606.19549, FlexLoRA, 2606.01090.
2. Restate claim 1 in Papers 3/4 against the three-factor lineage (MoSLoRA/StelLA/AdaLoRA/BoRA); the novelty is topology programming + diagnostics + structural masks, not bridge placement.
3. BM-003: add SVFT-Random citation/ablation framing; Config F is the MoSLoRA-equivalent arm — say so.
4. BM-004: cite 2606.01090 as concurrent; accelerate.
5. Tier-3 protocol: adopt permutation null, gauge-invariant angles, SV-weighted spectra, behavioral endpoints, Cross-LoRA + mini-vec2vec baselines.
6. 1B: adopt the W2T regime-contrast framing; reframe merge prediction weight-only; cite the prior-art-in-kind set; **propose 1B-before-1A sequencing to the Director**.
7. Add all sweep papers to the research suite (`research/`); next sweep after CoLorAI camera-readies (≈ August 2026).

*Fourth expansion compiled July 3, 2026 by Meridian from five parallel scout reports. Full scout transcripts preserved in session records; per-paper cards in `research/papers/`.*
