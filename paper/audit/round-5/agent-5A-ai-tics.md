# Round 5 — Agent 5A: AI-Generated Verbal Tics Audit

**Scope:** Papers 2 and 3 (`rhombic-paper2.tex`, `rhombic-paper3.tex`)
**Auditor:** Agent 5A
**Date:** 2026-03-15

---

## Executive Summary

Both papers are remarkably clean of AI verbal tics. Neither paper contains
any of the most egregious patterns (hedging clusters, hollow intensifiers,
exclamation marks, buzzword clustering, passive voice overuse, or the
"not merely X --- it Y" escalation pattern). The prose reads as direct,
technical, and confident throughout.

**Total findings: 9** (0 HIGH, 3 MEDIUM, 6 LOW)

The papers are well below the threshold where AI-generated verbal
patterns would register with a reviewer.

---

## Category 1: Academic Filler Words

**Instances found in Paper 2:** 0
**Instances found in Paper 3:** 0

Neither "Moreover," "Furthermore," "Additionally," "Notably,"
"Importantly," "Interestingly," "It is worth noting," nor
"It should be noted" appears anywhere in either paper.

**Verdict:** CLEAN.

---

## Category 2: Hollow Intensifiers

**Instances found in Paper 2:** 0
**Instances found in Paper 3:** 0

No occurrences of "remarkably," "strikingly," "astonishingly,"
"intriguingly," or "crucially."

**Verdict:** CLEAN.

---

## Category 3: Hedging Clusters

**Instances found in Paper 2:** 0
**Instances found in Paper 3:** 0

No occurrences of "it is important to note that," "it is worth
mentioning," "one might argue," or "it could be argued."

**Verdict:** CLEAN.

---

## Category 4: AI Escalation Patterns

**Instances found in Paper 2:** 0
**Instances found in Paper 3:** 0

No "not merely X --- it Y" or "This is not X. This is Y." patterns.

**Verdict:** CLEAN.

---

## Category 5: Redundant Transitions

### Finding 5-1

**File:** `rhombic-paper3.tex`
**Line 236:** `The remainder of this paper is organized as follows.`
**Pattern:** Unnecessary meta-commentary / redundant transition
**Severity:** LOW

This is standard academic boilerplate for a paper organization paragraph.
It is conventional and expected in ML papers. Not an AI tic per se, but
it is the single most formulaic sentence in either paper. The Paper 2
equivalent (`\paragraph{Organization.}` at line 203) handles this more
elegantly by using a labeled paragraph rather than the "remainder of
this paper" formula.

### Finding 5-2

**File:** `rhombic-paper2.tex`, lines 203--215
**File:** `rhombic-paper3.tex`, lines 236--246
**Pattern:** Section-by-section roadmap paragraphs
**Severity:** LOW

Both papers include "Section X reviews... Section Y describes...
Section Z presents..." roadmaps. These are standard in ML papers and
not AI-specific, but both roadmaps are somewhat mechanical --- each
sentence has the form "Section~\ref{...} [verb]s [noun]." Paper 2
uses 7 such sentences; Paper 3 uses 7. This is acceptable but borders
on formulaic.

**Verdict:** Two LOW-severity instances. Standard academic practice,
not specifically AI-generated.

---

## Category 6: Over-Attribution to the Work Itself

### Finding 6-1

**File:** `rhombic-paper2.tex`, line 599
**Text:** `Two findings emerge.`
**Pattern:** Findings-emerge phrasing
**Severity:** LOW

Minor AI-adjacent pattern. "Two findings emerge" gives agency to the
findings rather than stating them directly. Could be: "The spectral
analysis reveals two effects." or simply cut and state the findings.

### Finding 6-2

**File:** `rhombic-paper3.tex`, line 1161--1162
**Text:** `These results suggest that multi-channel LoRA adapters have latent geometric preferences that standard training does not surface.`
**Pattern:** "These results suggest" + "does not surface" (work doing/revealing)
**Severity:** LOW

Standard academic hedging. Appropriate for a conclusion section where
the authors are extrapolating beyond the direct experimental evidence.
Not a problem.

**Verdict:** Two LOW-severity instances. Within normal academic range.

---

## Category 7: Passive Voice Overuse

**Instances found in Paper 2:** 0
**Instances found in Paper 3:** 0

No instances of "was observed," "can be seen," "is shown," "were
observed," "are shown," or "was found" anywhere in either paper. The
prose consistently uses active constructions ("the Steersman produces,"
"corpus weights suppress," "the contrastive loss drives"). This is
unusually clean for academic writing.

**Verdict:** CLEAN. This is a strength of the writing.

---

## Category 8: Unnecessary Meta-Commentary

### Finding 8-1

**File:** `rhombic-paper3.tex`, line 236
**Text:** `The remainder of this paper is organized as follows.`
**Pattern:** "The remainder of this..."
**Severity:** LOW (same as Finding 5-1; counted once)

Already flagged in Category 5. No other meta-commentary ("The reader
will recall," "As discussed in the introduction," etc.) appears.

**Verdict:** One instance only. Flagged above.

---

## Category 9: Exclamation Marks in Academic Prose

**Instances found in Paper 2:** 0
**Instances found in Paper 3:** 0

No exclamation marks in prose text. (LaTeX formatting characters like
`blue!60!black` and `8!` for factorial do not count.)

**Verdict:** CLEAN.

---

## Category 10: Buzzword Clustering

### Finding 10-1

**File:** `rhombic-paper3.tex`
**Pattern:** Repetition of "robust" (5 occurrences)
**Severity:** MEDIUM

Locations:
- Line 139: `whether the discovery is robust, fast, and mechanistically`
- Line 146: `Block-diagonal emergence is robust.`
- Line 212: `Block-diagonal emergence as a robust property of`
- Line 626: `The attractor is robust across all tested`
- Line 1144: `The finding is robust across every experimental axis tested.`

"Robust" appears 5 times across Paper 3. While each individual usage is
defensible (the word carries specific technical meaning in experimental
contexts), the repetition creates an AI-adjacent pattern --- particularly
lines 146, 212, and 1144 which all use "robust" to describe the same
finding in nearly identical constructions. Consider varying: "stable,"
"consistent," "invariant," or simply stating the evidence without the
adjective.

Paper 2 uses "robust" twice (lines 92 and 245), both in distinct
technical contexts. That frequency is fine.

### Finding 10-2

**File:** `rhombic-paper3.tex`
**Pattern:** "the fact that" (2 occurrences)
**Severity:** MEDIUM

Locations:
- Line 1062: `The fact that the bridge accepts the RD geometry`
- Line 1128: `The fact that the effective rank~(3) equals the number of`

"The fact that" is not inherently problematic, but two instances in the
Discussion section create a slight AI-essay texture. Both can be
tightened:
- Line 1062: "The bridge's acceptance of the RD geometry (100% BD)
  while rejecting arbitrary patterns..."
- Line 1128: "The effective rank (3) equaling the number of coordinate
  axes suggests..."

### Finding 10-3

**File:** `rhombic-paper2.tex`, line 51
**Text:** `answer definitively: structured weights amplify.`
**File:** `rhombic-paper3.tex`, line 144
**Text:** `the answer is unambiguous:`
**Pattern:** Assertive absolutes in framing
**Severity:** MEDIUM

Both papers use strong assertive framing ("definitively," "unambiguous")
in their introductions when summarizing results. In both cases the data
genuinely supports the claim (100% vs. 0%, deterministic ratios), so
the assertions are not overclaiming. However, reviewers may perceive
this register as promotional rather than scientific. In Paper 3
especially, "the answer is unambiguous" followed by a colon reads more
like a press release than a methods paper. Consider: "the results are
consistent across all conditions:" or simply cut and present the
enumerated findings.

---

## Summary Table

| # | File | Line | Pattern | Text (abbreviated) | Severity |
|---|------|------|---------|--------------------|----------|
| 5-1 | Paper 3 | 236 | Meta-commentary | "The remainder of this paper..." | LOW |
| 5-2 | Both | 203-215 / 236-246 | Section roadmap | "Section X reviews... Section Y describes..." | LOW |
| 6-1 | Paper 2 | 599 | Findings-emerge | "Two findings emerge." | LOW |
| 6-2 | Paper 3 | 1161 | Results-suggest | "These results suggest..." | LOW |
| 10-1 | Paper 3 | 139,146,212,626,1144 | "robust" repetition | 5 occurrences of "robust" | MEDIUM |
| 10-2 | Paper 3 | 1062, 1128 | "the fact that" | 2 occurrences in Discussion | MEDIUM |
| 10-3 | Both | P2:51 / P3:144 | Assertive absolutes | "definitively" / "unambiguous" | MEDIUM |
| 6-1b | Paper 3 | 831 | Results-separate | "The results separate cleanly along three dimensions." | LOW |
| -- | Paper 2 | 92 | "robust" (standalone) | "more robust than the baseline suggests" | LOW |

---

## Patterns NOT Found (Positive Notes)

The following common AI patterns are **entirely absent** from both papers:

- "Moreover" / "Furthermore" / "Additionally" (0 instances)
- "Notably" / "Importantly" / "Interestingly" (0 instances)
- "It is worth noting" / "It should be noted" (0 instances)
- "Remarkably" / "Strikingly" / "Astonishingly" (0 instances)
- "Not merely X --- it Y" escalation (0 instances)
- "The reader will recall" (0 instances)
- "As discussed in the introduction" (0 instances)
- Passive voice constructions like "was observed" / "can be seen" (0 instances)
- Exclamation marks in prose (0 instances)
- "Novel" / "groundbreaking" / "paradigm-shifting" (0 instances)
- "Comprehensive" / "significant" as empty modifiers (0 instances)

---

## Recommended Actions

1. **Paper 3, "robust" repetition (MEDIUM):** Reduce from 5 to 2-3
   occurrences. Lines 146 and 1144 say the same thing; keep one and
   vary the other. Line 212 (contributions list) could use "consistent"
   or simply state "100% block-diagonal structure across..."

2. **Paper 3, "the fact that" (MEDIUM):** Restructure both sentences
   to eliminate the phrase. Direct constructions are tighter.

3. **Both papers, "definitively" / "unambiguous" (MEDIUM):** Consider
   whether the assertive register is appropriate for your target venue.
   For workshop papers or independent publication, this is fine. For
   top-tier ML venues with adversarial reviewers, soften slightly.

4. **No other action required.** The LOW findings are standard academic
   conventions that do not need correction.

---

*Agent 5A audit complete. Both papers are substantially free of AI verbal
tics. The writing is direct, active-voiced, and technically precise.
The three MEDIUM findings are minor polish items, not structural problems.*
