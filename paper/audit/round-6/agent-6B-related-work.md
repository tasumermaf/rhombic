# Round 6 -- Agent 6B: Related Work Gaps

**Agent:** 6B (Related Work)
**Round:** 6 of 7
**Date:** 2026-03-15
**Files audited:** `rhombic-paper3.tex`, `rhombic-paper3.bib`, `sections/section1_introduction.tex`, `sections/section2_architecture.tex`

---

## Findings

### F-6B-01: LoRA-XS (Banaei et al. 2024) Not Cited in Final Paper
- **Severity:** CRITICAL
- **Location:** Section 2.2 (Multi-Channel and Multi-Head Parameter-Efficient Methods)
- **Finding:** LoRA-XS is the closest architectural precedent to RhombiLoRA. Both insert a learnable matrix between the A and B projections. The bib entry `banaei2024loraxs` exists but is never `\cite`d in `rhombic-paper3.tex`. A clear, well-written distinction exists in the draft (`sections/section1_introduction.tex`, lines 128-136) but was dropped during consolidation to the monolithic file. The draft text reads: "The closest architectural precedent is LoRA-XS, which places an r x r matrix between A and B but freezes the outer projections to minimize parameter count. Because A and B do not adapt to the task in LoRA-XS, the intermediate matrix cannot serve as a diagnostic of what joint A-B learning discovered -- it *is* the entirety of learned adaptation. RhombiLoRA trains all three components jointly, enabling the bridge to capture the emergent coupling pattern rather than carrying the full adaptation burden."
- **Recommendation:** Restore this paragraph into Section 2.2, after the Multi-head LoRA / Mixture-of-LoRA discussion. The distinction is structurally important on three axes: (1) bridge size (r x r vs. n x n, n << r), (2) outer projection training (frozen vs. joint), (3) bridge role (full adaptation carrier vs. emergent coupling diagnostic). A reviewer familiar with LoRA-XS will immediately ask how RhombiLoRA differs; the answer is strong but currently absent.

### F-6B-02: MELoRA (Ren et al. 2024) Not Cited in Final Paper
- **Severity:** CRITICAL
- **Location:** Section 2.2
- **Finding:** MELoRA partitions rank dimensions into independent channels -- architecturally the M = I_n special case of RhombiLoRA. The bib entry `ren2024melora` exists but is never cited in the final paper. A clear paragraph exists in the draft (`sections/section2_architecture.tex`, lines 149-157) showing RhombiLoRA's bridge spans a continuum from MELoRA (diagonal/independent) through partially coupled to fully coupled. This text was dropped during consolidation.
- **Recommendation:** Restore the MELoRA continuum framing into Section 2.2, ideally adjacent to the LoRA-XS paragraph. The positioning is structurally illuminating: MELoRA proves that channel partitioning alone helps; RhombiLoRA asks what happens when channels are coupled through a learnable bridge.

### F-6B-03: Beer (1972) Not Cited in Text
- **Severity:** MAJOR
- **Location:** Section 2.3 (Cybernetic Optimization) and Section 7.4 (Connection to Broader Literature)
- **Finding:** Beer's *Brain of the Firm* is in the bibliography (`beer1972`) but never cited in the text. The paper uses cybernetic terminology extensively (Steersman, sense-decide-actuate, homeostasis at line 1142), and the Steersman's recursive feedback architecture echoes Beer's Viable System Model. Section 2.3 currently cites only Wiener (1948) and Ashby (1956). Beer is the organizational cyberneticist who operationalized Ashby's requisite variety into practical management control -- his recursive self-organizing feedback loops are directly analogous to the Steersman's three independent control laws monitoring different state variables.
- **Recommendation:** Add Beer to the cybernetics citation cluster: `\cite{wiener1948, ashby1956, beer1972}` at line 290. A brief phrase would suffice, e.g., "Cybernetic control theory -- from foundational formalization~\cite{wiener1948, ashby1956} through organizational recursion~\cite{beer1972} -- formalizes feedback systems that maintain stability through sense-decide-actuate loops."

### F-6B-04: Biderman et al. 2024 (LoRA Learns Less and Forgets Less) Not Cited
- **Severity:** MINOR
- **Location:** Section 7 (Discussion), specifically Subsection 7.2 (Practical Implications)
- **Finding:** The bib entry `biderman2024lora` exists but is never cited. This paper provides the most thorough 2024 analysis of LoRA's learning dynamics -- what it learns vs. full fine-tuning, and the stability/plasticity tradeoff. It would be relevant context in the Discussion where Paper 3 notes that block-diagonal structure provides no measurable performance benefit (line 176: "Block-diagonal structure provides no measurable performance benefit. The bridge topology is a structural signature, not a performance feature."). Biderman et al.'s finding that LoRA preserves more pretrained knowledge could help contextualize why bridge topology is orthogonal to task loss.
- **Recommendation:** Cite in Discussion Section 7.2 or in the "Connection to Broader Literature" subsection (7.4). A single sentence would suffice: "The orthogonality of bridge topology to task performance is consistent with the observation that LoRA adapters preserve much of the pretrained model's structure~\cite{biderman2024lora}."

### F-6B-05: Putterman et al. 2024 (Group Actions on Latent Representations) Not Cited
- **Severity:** MINOR
- **Location:** Section 7.4 (Connection to Broader Literature)
- **Finding:** The bib entry `putterman2024` exists but is never cited. This paper on learning group actions in latent representations has relevance to the symmetry emergence discussed in Section 5.4 (Axis Symmetry and Polarity) where the paper observes that the bridge discovers 3-axis coordinate structure. However, the connection is tangential -- Putterman et al. focus on learning explicit group action representations, while Paper 3 observes geometric structure emerging implicitly.
- **Recommendation:** Either cite briefly in Discussion Section 7.4 as related work on geometric structure in learned representations, or remove from the .bib to avoid orphaned entries. The connection is interesting but not essential.

### F-6B-06: GaLore (Zhao et al. 2024) Missing Entirely
- **Severity:** MINOR
- **Location:** Section 2.1 (Low-Rank Adaptation)
- **Finding:** GaLore (Memory-Efficient LLM Training by Gradient Low-Rank Projection) is a significant 2024 PEFT paper that projects gradients into a low-rank subspace during training. It is not in the bibliography and not mentioned in the text. While GaLore takes a fundamentally different approach (gradient-space projection rather than adapter injection), it is part of the 2024 PEFT landscape that a reviewer would expect to see acknowledged.
- **Recommendation:** Add a brief mention in Section 2.1's list of LoRA extensions: "gradient-space projection (GaLore)~\cite{zhao2024galore}". One phrase is sufficient -- the paper's scope is adapter architecture, not gradient optimization, but the survey should acknowledge the landscape.

### F-6B-07: Four Adapter Merging References Orphaned in .bib
- **Severity:** MINOR
- **Location:** Bibliography
- **Finding:** Four adapter composition/merging references are in the .bib but never cited: Ilharco 2023 (Task Arithmetic), Yadav 2023 (TIES-Merging), Yu 2024 (DARE), Yang 2024 (AdaMerging). These were likely included for the preliminary experiment discussion (exp1a-e involve bridge interpolation, mentioned at lines 112-121), but the citations were dropped during consolidation. The paper mentions eigenspectrum preservation during interpolation but does not cite the adapter merging literature that motivates this analysis.
- **Recommendation:** Either (a) add a brief sentence in Section 2.2 or the Introduction noting that bridge interpolation connects to the adapter merging literature~\cite{ilharco2023editing, yadav2023ties}, or (b) remove these entries from the .bib. Orphaned bib entries will produce LaTeX warnings and signal incomplete editing to reviewers. Option (b) is simpler if the merging literature is not needed for the paper's argument.

### F-6B-08: VeRA (Kopiczko et al. 2024) Not Mentioned
- **Severity:** MINOR
- **Location:** Section 2.1 (Low-Rank Adaptation)
- **Finding:** VeRA (Vector-based Random Matrix Adaptation) is a 2024 PEFT method that freezes random A and B matrices and trains only small scaling vectors. Like LoRA-XS, VeRA is a parameter-minimization approach that modifies the standard LoRA decomposition. While less directly comparable to RhombiLoRA than LoRA-XS, it is part of the landscape of "structured modifications to LoRA's rank space" that Section 2 surveys. A PEFT-specialist reviewer may note its absence.
- **Recommendation:** Add a brief mention in Section 2.1 alongside the other LoRA extensions. One phrase is sufficient: "frozen-projection variants (VeRA)~\cite{kopiczko2024vera}".

### F-6B-09: Steersman Simplicity Claim Adequately Contextualized
- **Severity:** PASS (no finding)
- **Location:** Section 2.3, lines 295-301; Section 7.4, lines 1144-1147
- **Finding:** The paper claims the Steersman is "simpler than learned meta-optimizers" (Section 2.3) and "simpler than learned meta-optimizers or neural architecture search -- fixed control laws with adaptive thresholds" (Section 7.4). This is supported by concrete specification: the Steersman's state space is 3 scalar diagnostics, its actuation is limited to weight adjustments and learning rate scaling, and its control laws are fixed (not learned). The comparison to Andrychowicz et al. (2016) and Zoph & Le (2017) is appropriate. No additional contextualization needed.

### F-6B-10: Spectral Graph Theory Coverage Adequate
- **Severity:** PASS (no finding)
- **Location:** Section 3.2 (Steersman), Section 7.4
- **Finding:** Fiedler (1973) is cited correctly and used precisely (algebraic connectivity of the bridge graph Laplacian). Li et al. (2018) and Aghajanyan et al. (2021) provide the intrinsic dimensionality context. The paper's use of spectral methods (applying graph Laplacian analysis to a bridge matrix within a LoRA adapter) is novel and does not have established precedent literature to cite beyond the foundational Fiedler reference. Martin & Mahoney (2021, "Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory") could add context for spectral analysis of weight matrices generally, but is not essential.

### F-6B-11: Block-Diagonal / Modularity Literature Coverage Adequate
- **Severity:** PASS (no finding)
- **Location:** Section 2.4
- **Finding:** Clune et al. (2013), Amer & Maul (2019), Frankle & Carlin (2019), Fedus et al. (2022), and Lepikhin et al. (2021) are all cited in both Section 2.4 and the Discussion. The three mechanisms for prior block-diagonal structure (architectural imposition, pruning, gating) are clearly enumerated and distinguished from the paper's feedback-driven emergence. No significant gaps.

---

## Summary

| Category | Count |
|----------|-------|
| Missing PEFT papers (uncited but in bib) | 2 CRITICAL (LoRA-XS, MELoRA) |
| Missing PEFT papers (not in bib) | 2 MINOR (GaLore, VeRA) |
| Missing cybernetics refs (uncited but in bib) | 1 MAJOR (Beer) |
| Orphaned bib entries (merging literature) | 1 MINOR (4 entries) |
| Uncited bib entries (other) | 2 MINOR (Biderman, Putterman) |
| LoRA-XS distinction | 1 CRITICAL (exists in draft, dropped from final) |
| Adequate coverage (pass) | 3 (Steersman simplicity, spectral theory, modularity) |
| **Total findings** | **9 (2 CRITICAL, 1 MAJOR, 6 MINOR)** |

---

## Root Cause

The two CRITICAL findings and the MAJOR finding share a single root cause: **consolidation loss during assembly of the monolithic .tex file from modular `sections/` drafts.** The LoRA-XS distinction text exists at `sections/section1_introduction.tex:128-136`. The MELoRA continuum text exists at `sections/section2_architecture.tex:149-157`. Beer is in the bib. All three were prepared for the paper but dropped when the final file was assembled. This is a mechanical editing error, not a gap in the authors' awareness.

## Priority Actions

1. **CRITICAL -- Restore LoRA-XS paragraph** into Section 2.2. Use the draft text from `sections/section1_introduction.tex:128-136` as the basis. Ensure `\cite{banaei2024loraxs}` is present.
2. **CRITICAL -- Restore MELoRA paragraph** into Section 2.2. Use the draft text from `sections/section2_architecture.tex:149-157` as the basis. Ensure `\cite{ren2024melora}` is present.
3. **MAJOR -- Cite Beer** in Section 2.3 alongside Wiener and Ashby.
4. **MINOR -- Clean orphaned bib entries.** Either cite or remove: `ilharco2023editing`, `yadav2023ties`, `yu2024dare`, `yang2024adamerging`, `biderman2024lora`, `putterman2024`.
5. **MINOR -- Add GaLore and VeRA** to Section 2.1's PEFT landscape survey (one phrase each).
6. **MINOR -- QLoRA is NOT needed.** The experiments use bfloat16, not quantized training. The prior draft's mention of QLoRA was from different experiments; the final paper correctly omits it.

**Overall severity: CRITICAL** (due to LoRA-XS and MELoRA omissions). Fixable in one editing pass. The draft text exists and is well-written; the fix is mechanical restoration.
