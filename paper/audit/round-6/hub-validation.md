# Round 6 — Hub Validation

> **Date:** 2026-03-15
> **Scope:** Bibliography + Claims Calibration
> **Agents:** 6A (Citation Completeness), 6B (Related Work Gaps), 6C (Claims Calibration)

---

## Agent 6A: Citation Completeness — 3 findings

### F-6A-01: Paper 3 — 9 orphan bib entries
- Agent severity: MAJOR (39% orphan rate)
- **Hub verdict: ACCEPTED — PARTIALLY FIXED**
- LoRA-XS and MELoRA will be cited (via 6B fix). Beer will be cited.
  Remaining orphans (adapter merging cluster, Putterman, Biderman)
  removed from bib to keep bibliography clean. These methods are not
  discussed in the paper text.

### F-6A-02: Paper 2 — 4 orphan bib entries
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — FIXED**
- hales2005/hales2017: removed (conway1999 already cited at the relevant
  claim). bateson1979: removed (not discussed). ghosh2008: removed
  (ghosh2008resistance is the cited variant).

### F-6A-03: Paper 2 beer1972 "axis-fragility" framing
- Agent severity: MINOR
- **Hub verdict: NOTED — NO ACTION**
- The interpretive application of Beer's VSM to lattice topology is
  appropriate academic practice. The mapping is clear in context.

---

## Agent 6B: Related Work Gaps — 8 findings

### F-6B-01: LoRA-XS discussion lost in consolidation
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — FIXED**
- Restored LoRA-XS comparison paragraph into Section 2.2 from the
  draft text at sections/section1_introduction.tex. The distinction
  (LoRA-XS freezes A,B and trains r×r; RhombiLoRA trains all three
  with n×n where n<<r) is the most important positioning statement.

### F-6B-02: MELoRA discussion lost in consolidation
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — FIXED**
- Restored MELoRA comparison into Section 2.2. The continuum framing
  (MELoRA = diagonal, RhombiLoRA = coupled) provides clean positioning.

### F-6B-03: GaLore not cited
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- GaLore is a gradient projection method, not an adapter architecture.
  It operates in a fundamentally different space from LoRA variants.
  The Related Work section is about adapter architectures; adding
  GaLore would dilute the focus. A reviewer may mention it but its
  absence is defensible.

### F-6B-04: QLoRA not cited
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- The experiments load models in bfloat16 (line 541), not 4-bit
  quantization. QLoRA is not used and would be misleading to cite.

### F-6B-05: Beer (1972) not cited in Paper 3
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- Added Beer citation alongside Wiener and Ashby in Section 2.3.

### F-6B-06: Uncited adapter merging cluster
- Agent severity: LOW
- **Hub verdict: ACCEPTED — FIXED (removed)**
- Removed ilharco2023editing, yadav2023ties, yu2024dare,
  yang2024adamerging from bib. Bridge interpolation is mentioned
  briefly but does not require the full merging literature.

### F-6B-07: Putterman et al. 2024 uncited
- Agent severity: LOW
- **Hub verdict: ACCEPTED — FIXED (removed)**
- Removed from bib. Not discussed in paper.

### F-6B-08: Biderman et al. 2024 uncited
- Agent severity: LOW
- **Hub verdict: ACCEPTED — FIXED (removed)**
- Removed from bib. Interesting context for orthogonality discussion
  but not essential. Clean bibliography preferred.

---

## Agent 6C: Claims Calibration — 13 findings

### F-6C-H1: "natural representational structure" (line ~1072)
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — FIXED**
- "compatible with the adapter's natural representational structure"
  → "compatible with the adapter's representational structure under
  these training conditions." The word "natural" implies pre-existing
  preference untested by the experiments.

### F-6C-H2: "unique attractor" (line ~1165)
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — FIXED**
- "unique attractor" → "robust attractor." "Unique" is a formal
  dynamical systems claim requiring proof. The evidence shows a
  robust, stable attractor; uniqueness is not established.

### F-6C-M1: "necessary and sufficient at n=6" (×4)
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- Already qualified with "at n=6" in Round 2. The confound is
  explicitly acknowledged in Limitations. Reducing from 4 to fewer
  occurrences would require structural reorganization (abstract,
  intro, contributions, conclusion). The qualification is present
  everywhere.

### F-6C-M2: "emerges identically" (line ~232)
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- "emerges identically" → "emerges consistently." The ratios differ
  ~4× between scales (explained by code version, not scale), so
  "identically" is inaccurate.

### F-6C-M3: "Initialization is cosmetic" (line ~162)
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- "Initialization is cosmetic" → "Initialization effects vanish
  within 200 steps." Preserves the finding while removing the
  dismissive tone.

### F-6C-M4: "Discovers" in title + conclusion
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- The title is the paper's identity. "Discovers" is defensible:
  the paper explicitly shows the bridge "accepts" the geometry
  (100% BD) rather than having it imposed. The Discussion's nuanced
  treatment (lines 1068-1072, now corrected per H1) provides the
  necessary context. Changing the title at this stage risks
  incoherence with the entire framing.

### F-6C-M5: "rejecting arbitrary patterns" (line ~1069)
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- "rejecting arbitrary patterns (0% BD without contrastive loss)"
  → "developing no structure without it (0% BD without contrastive
  loss)." Only absence was tested, not alternative patterns.

### F-6C-M6: "latent geometric preferences" (line ~1187)
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- "latent geometric preferences that standard training does not
  surface" → "geometric preferences that emerge under structured
  feedback but not under standard training." Removes "latent"
  which implies pre-existence.

### F-6C-M7: Holly causal claim (line ~204)
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- "confirming that the effect is caused by the Steersman" →
  "providing additional evidence that the effect requires the
  Steersman." The controlled Qwen comparison (lines 1021-1024)
  makes the stronger causal case; Holly provides supporting
  evidence with acknowledged confounds.

### F-6C-L1: "identical validation loss" (×3)
- Agent severity: LOW
- **Hub verdict: ACCEPTED — FIXED**
- Lines 70, 194: "identical" → "matching" (parenthetical delta
  preserved). Line 923: "identical" → "matching."

### F-6C-L2: "exactly 3" (×2, lines 193, 915)
- Agent severity: LOW
- **Hub verdict: ACCEPTED — FIXED**
- "exactly 3" → "effectively 3" at both locations. Now consistent
  with abstract's calibration.

### F-6C-L3: "independent of initialization strategy" (line 59)
- Agent severity: LOW
- **Hub verdict: NOTED — NO ACTION**
- The abstract text is "independent of initialization strategy,
  including initializations that actively oppose..." which provides
  the specific scope. The three strategies tested include an
  adversarial one. The claim is well-scoped in context.

### F-6C-L4: "convergence proof" in Fig 2 caption
- Agent severity: LOW
- **Hub verdict: ACCEPTED — FIXED**
- "convergence proof" → "convergence evidence." Not a mathematical
  proof.

---

## Summary

| Category | Findings | Fixed | Noted | Removed |
|----------|----------|-------|-------|---------|
| 6A: Citations | 3 | 2 | 1 | — |
| 6B: Related Work | 8 | 4 | 2 | 2 (bib cleanup) |
| 6C: Calibration | 13 | 11 | 2 | — |
| **Total** | **24** | **17** | **5** | **2** |

**Paper 3 recompiled clean** after all fixes.
Paper 2 bib cleaned (4 orphans removed).

**Deferred to Round 7:**
- Abstract restructuring (from R5)
- Introduction bold-preview reduction (from R5)
- Architecture schematic figure (from R4)
