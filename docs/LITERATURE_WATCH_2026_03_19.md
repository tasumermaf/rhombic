# Literature Watch — March 19, 2026

> **Sweep window:** March 5–19, 2026
> **Queries:** 7 thematic searches + 1 major-labs check
> **Disposition:** Papers assessed for relevance to Papers 3, 4, and 5 of the 5-paper arc.

---

## HIGH RELEVANCE — Must Cite or Address

### 1. DiaBlo: Diagonal Blocks Are Sufficient For Finetuning
- **Authors:** Selcuk Gurses, Aozhong Zhang, Yanxia Deng, Xun Dong, Xin Li, Naigang Wang, Penghang Yin, Zi Yang (SUNY Albany + IBM Watson)
- **ID:** [arXiv:2506.03230v2](https://arxiv.org/abs/2506.03230) (revised **March 2, 2026**)
- **Thesis:** Update only the diagonal blocks of selected weight matrices — no low-rank factorization needed. Matches or exceeds LoRA on commonsense reasoning, arithmetic reasoning, code generation, and safety alignment.
- **Relevance: COMPETES with Paper 3.** DiaBlo discovers block-diagonal structure as *sufficient* for fine-tuning — the same structural motif our Steersman discovers through cybernetic feedback. Critical difference: DiaBlo **prescribes** BD structure a priori; TeLoRA's bridge **discovers** it through contrastive training. DiaBlo provides no interpretability or diagnostic capability. Their claim that BD is sufficient is actually evidence FOR our thesis that BD emergence is significant.
- **Cite in:** Paper 3 (Related Work, Discussion). Frame as: "DiaBlo prescribes what the Steersman discovers. Our contribution is showing that cybernetic feedback converges to block-diagonal structure without being told to, and that this convergence serves as a task fingerprint."
- **Action:** Read full paper. Check their block sizes vs. our 6-channel structure. Check if they test interpretability.

### 2. Stable-LoRA: Stabilizing Feature Learning of Low-Rank Adaptation
- **Authors:** (ICLR 2026 paper)
- **ID:** [arXiv:2603.05204](https://arxiv.org/abs/2603.05204) (submitted **March 5, 2026**)
- **Thesis:** Non-zero initialization of A compromises self-stability in LoRA. Proposes weight-shrinkage strategy that dynamically enhances stability by progressively shrinking A during early training. Published at ICLR 2026.
- **Relevance: SUPPORTS Paper 3.** Our sign fingerprint finding (signs frozen at 98.2% after 1200 steps) and the Fiedler convergence dynamics (overshoot-rebound-reconvergence) both describe training stability phenomena. Stable-LoRA's theoretical framework for LoRA feature learning stability is directly relevant to understanding WHY the bridge stabilizes.
- **Cite in:** Paper 3 (Section on training dynamics / Fiedler convergence). Their stability analysis may provide theoretical grounding for our empirical convergence trajectory.
- **Action:** Read. Check if their stability conditions predict the 3-phase Fiedler trajectory.

### 3. An Overview of Low-Rank Structures in the Training and Adaptation of Large Models
- **Authors:** Laura Balzano, Tianjiao Ding, Benjamin D. Haeffele et al. (8 authors)
- **ID:** [arXiv:2503.19859](https://arxiv.org/abs/2503.19859) (revised **February 3, 2026**)
- **Thesis:** Comprehensive tutorial reviewing low-rank structure emergence in deep networks — bridging mathematical foundations (optimization dynamics, implicit regularization) with practical LoRA/PEFT applications.
- **Relevance: SUPPORTS Papers 3 and 4.** Provides the theoretical scaffolding for our claim that low-rank structure emerges naturally and can be programmed. Their two perspectives (optimization dynamics vs. implicit regularization at convergence) map onto our Steersman (explicit optimization) vs. spectral attractor (implicit convergence to Fiedler ~0.09).
- **Cite in:** Paper 3 (theoretical background), Paper 4 (topology programming foundations).
- **Action:** Read sections on emergence of low-rank structure. Check for connections to our spectral attractor finding.

### 4. Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning
- **Authors:** (Microsoft Research, based on content)
- **ID:** [arXiv:2602.04998](https://arxiv.org/abs/2602.04998) (submitted **February 7, 2026**)
- **Thesis:** LoRA variants (DoRA, rsLoRA, PiSSA, LoRA+) achieve gains primarily through implicit learning rate effects. With properly tuned learning rates, vanilla LoRA matches all variants within 1-2%.
- **Relevance: CHALLENGES Paper 3 framing.** If vanilla LoRA matches variants under proper LR tuning, we need to show that TeLoRA's bridge provides value BEYOND what LR tuning captures. Our diagnostic / fingerprinting capability is not reducible to LR effects. The BD emergence is a structural finding, not a training trick. But this paper demands we address the LR confound explicitly.
- **Cite in:** Paper 3 (Limitations / Discussion). "Our claim is not that TeLoRA improves loss over vanilla LoRA, but that the bridge structure provides interpretable diagnostics unavailable to any LR schedule." (Edited 2026-07-02: an earlier draft cited the Holly Battery 3.8% improvement here — Holly Battery was retracted 2026-03-13, provenance/L-026, and must not be cited.)
- **Action:** Check their experimental setup against ours. ~~Verify Holly Battery used comparable LR sweeps.~~ (Moot — Holly Battery retracted 2026-03-13.)

---

## MODERATE RELEVANCE — Cite If Space Permits

### 5. MoLoRA: Composable Specialization via Per-Token Adapter Routing
- **Authors:** Shrey Shah, Justin Wagle
- **ID:** [arXiv:2603.15965](https://arxiv.org/abs/2603.15965) (submitted **March 16, 2026**)
- **Thesis:** Per-token routing across multiple LoRA adapters via learned gating. Qwen3-1.7B exceeds Qwen3-8B across four reasoning benchmarks.
- **Relevance: EXTENDS toward Paper 5.** MoLoRA's per-token routing between adapters parallels our vision of tessellating RD cells as MoE modules (Paper 5 / cross-modal transit). Their routing is flat; our tessellation geometry provides structured routing via shared rhombic faces.
- **Cite in:** Paper 5 (if routing becomes relevant). Note in landscape doc.

### 6. ODELoRA: Training Low-Rank Adaptation by Solving Ordinary Differential Equations
- **Authors:** Yihang Gao, Vincent Y. F. Tan
- **ID:** [arXiv:2602.07479](https://arxiv.org/abs/2602.07479) (submitted **February 7, 2026**)
- **Thesis:** Continuous-time optimization dynamics for LoRA factors via ODE that emulates gradient flow of full fine-tuning on the balanced manifold. Linear convergence under strongly convex objectives. Achieves stable feature learning.
- **Relevance: SUPPORTS Papers 3–4.** ODELoRA's continuous-time perspective on LoRA training dynamics is mathematically adjacent to our Fiedler convergence trajectory analysis. Their manifold perspective complements RiemannLoRA (already in our Tier 1). The ODE framing may provide a formal language for describing the Steersman's feedback loop as a dynamical system.
- **Cite in:** Paper 4 (theoretical framework for topology programming as dynamical system).

### 7. Localized LoRA: A Structured Low-Rank Approximation for Efficient Fine-Tuning
- **Authors:** Babak Barazandeh et al.
- **ID:** [arXiv:2506.00236v2](https://arxiv.org/abs/2506.00236v2) (revised September 2025)
- **Thesis:** Generalized framework for low-rank updates applied to structured blocks of weight matrices, extending MELoRA's diagonal-local approach to arbitrary spatial structures. Lower approximation error under matched parameter budgets.
- **Relevance: COMPETES with Paper 3 at structural level.** Localized LoRA formalizes the space of structured low-rank updates. TeLoRA's bridge is one instantiation within their framework — but with the crucial addition of learnability and diagnostic interpretability. They optimize for approximation error; we optimize for structure emergence.
- **Cite in:** Paper 3 (Related Work). Position TeLoRA as complementary: Localized LoRA asks "where to place low-rank updates"; TeLoRA asks "what does the coupling structure encode?"

### 8. SMoA: High-Rank Structured Modulation for Parameter-Efficient Fine-Tuning
- **Authors:** (authors not retrieved)
- **ID:** [arXiv:2601.07507](https://arxiv.org/abs/2601.07507) (submitted **January 12, 2026**)
- **Thesis:** Freezes pretrained weights and selectively amplifies/suppresses features across multiple subspaces. Achieves high-rank updates with fewer parameters than LoRA.
- **Relevance: EXTENDS.** SMoA's multi-subspace modulation connects to our channel structure — each FCC channel could be understood as a subspace within their framework. Peripheral relevance.
- **Cite in:** Paper 4 if subspace analysis becomes relevant.

### 9. BLAST: Block-Level Adaptive Structured Matrices for Efficient DNN Inference
- **Authors:** (authors not retrieved)
- **ID:** [arXiv:2410.21262](https://arxiv.org/abs/2410.21262) (October 2024, cited here for completeness)
- **Thesis:** BLAST matrix can construct low-rank, block-diagonal, and block low-rank matrices through learnable diagonal parameters. 70% complexity reduction on ViT, 40% on GPT-2.
- **Relevance: SUPPORTS Paper 3.** BLAST's parameterization of block-diagonal structure confirms that BD is a productive structural motif for neural networks. They use it for inference compression; we discover it during training. Different goal, same geometry.
- **Cite in:** Paper 3 (Related Work on block-diagonal structures in neural networks).

### 10. Block-Diagonal LoRA for Eliminating Communication Overhead in Tensor Parallel Serving
- **ID:** [arXiv:2510.23346](https://arxiv.org/abs/2510.23346)
- **Thesis:** Constrains LoRA factors to block-diagonal for serving efficiency — eliminates all-reduce communication in tensor-parallel inference.
- **Relevance: SUPPORTS Paper 3.** Another independent discovery that block-diagonal LoRA structure has practical value. Their motivation is communication efficiency; ours is interpretability. Convergent evidence that BD is natural for LoRA.
- **Cite in:** Paper 3 (footnote in BD discussion). "BD LoRA structure has independently been shown valuable for inference efficiency [cite], training efficiency [DiaBlo], and now for interpretable diagnostics [this work]."

---

## LOW RELEVANCE — Monitor Only

### 11. Feedback-Based Training Approaches (Cybernetic Framing)
- **Training Large Networks With Low-Dimensional Error Feedback** ([2502.20580](https://arxiv.org/abs/2502.20580)) — Feedback alignment with controlled error dimensionality. Tangential.
- **Backpropagation-Free Feedback-Hebbian Network** ([2601.06758](https://arxiv.org/abs/2601.06758)) — Local Hebbian learning with feedback pathway. Interesting for Paper 5's cybernetic circuit framing but not directly applicable to Papers 3–4.
- **Feedback Control for Spiking Neural Networks** ([2602.13261](https://arxiv.org/abs/2602.13261)) — Feedback control signals guiding weight updates in SNNs. Structurally analogous to our Steersman but in a completely different domain.

### 12. Topological Deep Learning
- **Copresheaf TNN** ([2505.21251](https://arxiv.org/abs/2505.21251)) — General framework unifying DL architectures under topological principles. Potentially relevant to Paper 4's topology programming claim but uses sheaf theory, not lattice geometry.
- **TDA for Neural Network Analysis survey** ([2312.05840v2](https://arxiv.org/abs/2312.05840)) — Updated survey. Monitor for methodological tools.

### 13. Fiedler Value / Algebraic Connectivity
- **Fiedler Regularization** ([2003.00992](https://arxiv.org/abs/2003.00992)) — Uses Fiedler value as regularization penalty. Already known. Our use of Fiedler as a diagnostic metric (not regularizer) is distinct.
- **Practical Algebraic Connectivity Maximization** ([2511.08694](https://arxiv.org/abs/2511.08694)) — Computational improvements for Fiedler computation. Utility tool, not research competitor.

---

## NOT FOUND — Searches With No Relevant Results

- **"24-cell" OR "D4 lattice" AND "machine learning":** Zero results. Our Paper 4 claim on 4D lattice topology in ML remains without prior art. The frontier is clear.
- **"topology programming" AND neural:** Only general topological deep learning surveys. No one is using the phrase "topology programming" for adapter structure control. Our Paper 4 title claim is novel.
- **Anthropic/DeepMind LoRA papers (March 2026):** No new LoRA or adapter papers from Anthropic or DeepMind in this window. Google and Meta have older MoE/LoRA work but nothing in the March 5–19 window that intersects our program.

---

## COMPETITIVE LANDSCAPE UPDATE

| Method | Bridge? | Learnable A/B? | BD Discovery? | Diagnostic? | Status |
|--------|---------|----------------|---------------|-------------|--------|
| **TeLoRA (ours)** | 6x6 geometric | Yes | Emergent (Steersman) | Yes (fingerprint, overfit) | Paper 3 submission-ready |
| **DiaBlo (NEW)** | Block-diagonal W directly | N/A (not LoRA) | Prescribed a priori | No | Revised Mar 2, 2026 |
| **Localized LoRA** | Structured blocks | Yes | Prescribed | No | Revised Sep 2025 |
| **LoRA-XS** | r x r dense | Frozen A/B | No | No | EMNLP 2024 |
| **LoRAN** | Sine activation | Yes | No | No | EMNLP 2024 |
| **DoRA** | Magnitude/direction | Yes | No | No | ICML 2024 |
| **BLAST** | Block-level adaptive | From scratch | Prescribed | No | Oct 2024 |
| **SMoA** | Multi-subspace modulation | Frozen base | No | No | Jan 2026 |
| **Stable-LoRA** | Weight shrinkage | Yes | No | No | ICLR 2026 |
| **ODELoRA** | ODE on manifold | Yes | No | No | Feb 2026 |
| **Vanilla LoRA + LR** | None | Yes | No | No | Feb 2026 |

**Key takeaway:** DiaBlo is the most important new entry. It validates block-diagonal structure for fine-tuning but from a prescriptive (engineering) direction. Our emergent discovery via Steersman remains unique. The "LR Matters" paper demands we address learning rate confounds explicitly. No one has touched topology programming, 24-cell/D4, or cybernetic bridge training.

---

## ACTION ITEMS

1. **Read DiaBlo in full** — highest priority. Check block sizes, check if they analyze why BD works, check interpretability claims. Draft a 2-paragraph Related Work addition for Paper 3.
2. **Read Stable-LoRA** — check if their stability conditions illuminate our Fiedler convergence trajectory.
3. **Read "LR Matters"** — verify our Holly Battery used comparable LR tuning. Add a paragraph to Paper 3 Discussion addressing LR confound.
4. **Add to Paper 3 bib:** DiaBlo, Stable-LoRA, LR Matters, BLAST (if not already), Block-Diagonal LoRA serving.
5. **Update competitive_landscape.md** with DiaBlo, SMoA, ODELoRA, Stable-LoRA entries.
6. **Schedule next sweep:** April 2, 2026 (biweekly cadence).

---

*Sweep conducted March 19, 2026 by Meridian. Seven thematic queries + major-labs check. 16 papers assessed, 4 high-relevance, 6 moderate, 3 low, 3 monitor. No prior art on 24-cell/D4 ML or topology programming.*
