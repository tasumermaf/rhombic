# Round 5 — Hub Validation

> **Date:** 2026-03-15
> **Scope:** Writing Quality
> **Agents:** 5A (AI Tic Detection), 5B (Prose Quality), 5C (Limitations Honesty)

---

## Agent 5A: AI Tic Detection — 9 findings

### 5A-10-1: "robust" repetition (5 occurrences)
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- Reduced from 5 to 2 occurrences. Kept "Block-diagonal emergence is
  robust" (the core claim) and "The attractor is robust across all tested
  conditions" (the results statement). Replaced 3 others: "consistent"
  (line 139), "consistent" (contributions list), "holds" (conclusion).

### 5A-10-2: "the fact that" (2 occurrences)
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- Line 1062: Restructured to "The bridge accepts... indicating..."
- Line 1128: Changed to "The effective rank (3) equaling..."

### 5A-10-3: Assertive absolutes ("unambiguous")
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- Line 144: "the answer is unambiguous:" → "the results are consistent:"
- Paper 2's "answer definitively" (line 51) left unchanged — Paper 2
  is not the primary target this round, and the claim is backed by
  deterministic ratios.

### 5A-5-1/5-2: Roadmap transitions
- Agent severity: LOW
- **Hub verdict: NOTED — NO ACTION**
- Standard academic convention.

### 5A-6-1/6-2: Findings-emerge / results-suggest
- Agent severity: LOW
- **Hub verdict: NOTED — NO ACTION**
- Within normal academic range.

---

## Agent 5B: Prose Quality — 16 findings

### P3-TR-01: Missing Method→Results transition
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — FIXED**
- Added two-sentence bridge at Section 4 opening: "With these components
  in place—multi-channel bridge, contrastive and spectral losses,
  reactive feedback—we turn to the experimental evidence."

### P3-JD-01: Abstract jargon overload
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — DEFERRED**
- Valid concern. Restructuring the abstract opening is a significant
  rewrite that risks introducing errors at this stage. The abstract is
  technically correct and self-contained. Deferred to final polish
  (Round 7).

### P3-RD-01: Core finding stated 7 times
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — NOTED**
- The repetition follows standard academic structure (abstract, intro
  preview, contributions list, results, discussion, conclusion). The
  intro bold-paragraphs + contributions echo is the real issue. Reducing
  the bold previews would require structural reorganization — noted for
  Round 7.

### P3-AQ-01: Abstract has 17 numbers in 28 lines
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — DEFERRED**
- Same as P3-JD-01. Abstract reduction deferred to Round 7.

### P3-JD-03: Co-planar/cross-planar terminology without reminders
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- Added parenthetical gloss at Section 4 first body-text use: "(where ρ
  is the co-planar to cross-planar coupling ratio, i.e., same-axis vs.
  different-axis channel pairs)."

### P3-SV-01, P3-PS-01: Bold-paragraph monotony and contributions echo
- Agent severity: MEDIUM
- **Hub verdict: NOTED — DEFERRED to Round 7**
- Structural reorganization. Valid but low risk for reviewers.

### P3-VV-01: "produces" overuse (30+)
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- "Produces" is the paper's core verb for its core finding (100%
  produced vs. 0% produced). Replacing it with synonyms risks making
  the consistent finding sound inconsistent. Minor monotony is
  preferable to imprecision.

### P3-AP-01: Heavy passive in Method Section 3.1
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- Passive voice is conventional in method sections describing system
  architecture. "The rank is partitioned" is clearer than "We partition"
  in an architecture description.

### P3-SI-01: Section 4 opens with data
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED** (via P3-TR-01 transition sentence)

### P2-TR-01: Missing transition between registers (Section 4→5)
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- Paper 2 is lower priority. The reader can follow the existing text.

### P2-JD-01: "Fiedler" used without gloss in abstract
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- The Fiedler value is defined in Section 2.1. Adding a parenthetical
  to the abstract increases word count for marginal benefit. Spectral
  graph theory readers know the term.

### P2-RD-01: 6.1× stated 6 times
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- Standard repetition across abstract, results, conclusion.

### Remaining LOW items
- All NOTED — NO ACTION. Standard academic patterns.

---

## Agent 5C: Limitations Honesty — 17 findings

### L-01: Single seed (42) for all experiments — UNACKNOWLEDGED
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — FIXED**
- Added "Single seed" limitation paragraph to Paper 3 Limitations
  section. Notes that binary BD classification is robust but specific
  numerical quantities are single-seed observations.

### L-04: Channel ablation only at 1.1B — UNACKNOWLEDGED
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — FIXED**
- Added "Ablation scale" limitation paragraph. Notes the n=6
  bifurcation is untested at 7B or larger.

### L-08: 13 Steersman hyperparameters with no sensitivity analysis
- Agent severity: HIGH
- **Hub verdict: ACCEPTED — FIXED**
- Added "Steersman sensitivity" limitation paragraph. Notes hand-tuned
  parameters and absence of sensitivity analysis.

### L-02: N=2 cybernetic scales
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED**
- Expanded existing scale-range limitation to note Holly confounds
  (4 simultaneous differences) and that two cybernetic scales are
  insufficient to establish a scaling law.

### L-03: No formal statistical tests
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- The single-seed limitation (L-01) subsumes this concern. Adding
  another limitation about statistical testing would be redundant.

### L-05: Holly Battery confounds
- Agent severity: MEDIUM
- **Hub verdict: ACCEPTED — FIXED** (via L-02 expansion)

### L-09: BD detection threshold sensitivity
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- Appendix already notes thresholds are "conservative" and actual
  values (70,000:1) far exceed them. Effect too extreme for threshold
  choice to matter.

### L-10: "Scale Consistency" section title
- Agent severity: MEDIUM
- **Hub verdict: NOTED — NO ACTION**
- Round 2 already downgraded "Scale Invariance" to "Scale Consistency."
  Further softening to "Cross-Scale Comparison" loses the claim the
  section is making. The section title should state the finding.

### L-06, L-07, L-11, L-12, L-13: LOW/addressed items
- **Hub verdict: NOTED — NO ACTION**
- Scope is clear from context. No additions needed.

### L-14 through L-17: Paper 2 limitations
- **Hub verdict: PASS**
- Paper 2's limitations section is well-calibrated. No changes.

---

## Summary

| Category | Findings | Fixed | Deferred | Noted | Pass |
|----------|----------|-------|----------|-------|------|
| 5A: AI Tics | 9 | 3 | 0 | 6 | 0 |
| 5B: Prose | 16 | 3 | 3 | 10 | 0 |
| 5C: Limitations | 17 | 5 | 0 | 8 | 4 |
| **Total** | **42** | **11** | **3** | **24** | **4** |

**Paper 3 recompiled clean** after all fixes (23 pages, 0 undefined refs).
Paper 2 unchanged this round.

**Deferred to Round 7:**
- Abstract restructuring (jargon density, number count)
- Introduction bold-preview reduction
