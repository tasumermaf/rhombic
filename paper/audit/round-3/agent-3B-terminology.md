# Audit Report: Cross-Paper Terminology Consistency

**Auditor:** Agent 3B (Terminology)
**Scope:** Papers 2 and 3 (Paper 1 `.tex` not present in repo; audited via cross-references in Papers 2-3)
**Date:** 2026-03-15

---

## Summary Table

| ID | Severity | Location | Finding | Status |
|----|----------|----------|---------|--------|
| T-001 | **HIGH** | Paper 3, lines 98-103 vs. 345 | Bridge matrix symbol changes from `\mathcal{B}` to `\mathbf{M}` within Paper 3 | OPEN |
| T-002 | MEDIUM | Paper 2 line 94 vs. Paper 3 line 328 | Paper 1 metrics quoted inconsistently: "30--32%" / "2.3--2.5x" (P2) vs. "30%" / "2.4x" (P3) | OPEN |
| T-003 | LOW | Paper 2 line 271 vs. Paper 3 line 496 | "channel" terminology: quoted in P2, formal in P3 -- consistent but implicit linkage | OK |
| T-004 | **NONE** | All papers | "FCC lattice" / "rhombic dodecahedral" usage | CONSISTENT |
| T-005 | **NONE** | Papers 2-3 | Variable names: $n$ (channel count), $r$ (rank), $s$ (channel size) | CONSISTENT -- $n$ used differently in P2 (seed count, line 819) but in a non-overlapping context |
| T-006 | **NONE** | Papers 2-3 | "Fiedler eigenvalue" definition | CONSISTENT |
| T-007 | **NONE** | Paper 3 | "block-diagonal," "co-planar," "cross-planar" definitions | CONSISTENT within P3; not used in P2 |
| T-008 | **NONE** | Paper 3 | "Steersman" description | CONSISTENT (not mentioned in P2) |
| T-009 | **NONE** | Papers 2-3 | "RhombiLoRA" vs. "multi-channel LoRA" usage | CONSISTENT |
| T-010 | **NONE** | Papers 2-3 | "bridge matrix" definition | CONSISTENT across papers (see T-001 for within-P3 issue) |
| T-011 | LOW | Paper 2 line 859 vs. Paper 3 passim | Paper 2 previews "6 channels" in future work; Paper 3 uses $n=6$ as one of four tested values | OK -- no contradiction |
| T-012 | LOW | Paper 3 lines 64-67 vs. 184-187 | Abstract and intro report slightly different Fiedler numbers for non-$n=6$ cases | VERIFY |

---

## Detailed Findings

### T-001: Bridge Matrix Symbol Inconsistency Within Paper 3
**Severity:** HIGH
**Location:** Paper 3, Introduction (lines 98-103) vs. Method (line 345 onward)
**Finding:** The bridge matrix is denoted `$\mathcal{B}$` in the Introduction and `$\mathbf{M}$` in the Method section and all subsequent usage. The Introduction uses `$\mathcal{B}_{ij}$` for entry access and `$\mathcal{B} = I_n$` for the identity condition. From Section 3.1 onward, the same object is exclusively `$\mathbf{M}$`, with `$M_{ij}$` and `$M^a_{ij}$` for entries.

**Evidence:**
- Line 98: `$n \times n$ \textbf{bridge matrix} $\mathcal{B}$ between the down-`
- Line 100: `$\mathcal{B}_{ij}$ controls how strongly channel~$j$'s activations`
- Line 101: `When $\mathcal{B} = I_n$, channels are uncoupled`
- Line 103: `$\mathcal{B}$ departs from the identity`
- Line 345: `$\mathbf{M} \in \mathbb{R}^{n \times n}$`
- Line 370: `$\mathbf{M} = I_n$, channels are uncoupled`
- Lines 419-420, 518-520, 971, 1292: All use `$M_{ij}$` or `$M^a_{ij}$`

The Introduction's `$\mathcal{B}$` is never formally reconciled with the Method's `$\mathbf{M}$`. A reader will encounter two different symbols for the same object with no transition.

**Recommendation:** Unify to `$\mathbf{M}$` throughout (it is the dominant symbol, used in all equations), or add a sentence in Section 3.1 stating `$\mathbf{M}$` replaces the `$\mathcal{B}$` of the introduction.

**Status:** OPEN

---

### T-002: Paper 1 Metrics Quoted Inconsistently
**Severity:** MEDIUM
**Location:** Paper 2 (lines 83, 94, 96) vs. Paper 3 (line 328)
**Finding:** When citing Paper 1's results, the two papers use different precision:

- **Paper 2, line 83:** "a $2.3$--$2.5\times$ algebraic connectivity ratio"
- **Paper 2, line 94:** "FCC lattices achieve 30--32\% shorter average paths"
- **Paper 2, line 96:** "$2.3$--$2.5\times$ Fiedler ratio under uniform weights"
- **Paper 3, line 328:** "30\% shorter paths and $2.4\times$ higher algebraic connectivity"

Paper 2 reports ranges ("2.3--2.5x", "30--32%"), which likely reflect scale-dependent variation. Paper 3 collapses these to point estimates ("2.4x", "30%"). The "2.4x" in Paper 3 is the midpoint of Paper 2's range, and "30%" is the floor of Paper 2's "30--32%". Neither is wrong, but a reader following from Paper 2 to Paper 3 will notice the discrepancy.

**Recommendation:** Either standardize both papers on the range ("2.3--2.5x", "30--32%") or on the midpoint ("2.4x", "31%"). Since Paper 1 is LOCKED, use whatever Paper 1 actually reports.

**Status:** OPEN

---

### T-003: "Channel" Terminology -- Lattice vs. LoRA
**Severity:** LOW
**Location:** Paper 2 (line 271, 859) vs. Paper 3 (passim)
**Finding:** Paper 2 introduces "channels" in quotation marks as a metaphor for direction-pair pathways through the lattice (line 271: `creating 6 ``channels'' through the lattice`). Paper 3 uses "channel" as a formal architectural term (the $n$ parallel partitions of the LoRA rank). Paper 2's future-work section (line 859) bridges these: "a learnable bridge matrix with 6 channels following the RD's face-pair topology."

The transition from metaphorical to formal usage is deliberate and handled by the cross-reference. No action required, but the linkage is implicit -- a reader of Paper 3 alone would not know that "channel" was originally a lattice metaphor.

**Status:** OK -- intentional design

---

### T-004: FCC Lattice / Rhombic Dodecahedral Terminology
**Severity:** NONE
**Location:** All papers
**Finding:** Both papers use "FCC" consistently as the abbreviation for "face-centered cubic," first expanded in each paper (Paper 2 line 46, Paper 3 line 322). Both papers use "RD" as the abbreviation for "rhombic dodecahedron" (Paper 2 line 128, Paper 3 passim). Paper 2 focuses on FCC lattice-scale properties; Paper 3 focuses on RD geometry as a structural prior. The distinction is correct and consistently maintained.

**Status:** CONSISTENT

---

### T-005: Variable Names ($n$, $r$, $s$)
**Severity:** NONE
**Location:** Papers 2-3
**Finding:** Paper 3 uses $n$ for channel count, $r$ for LoRA rank, and $s = r/n$ for channel size. These variables do not appear with these meanings in Paper 2. Paper 2 uses $n$ once (line 819) for seed count ("$n = 100$"), but this is in an unrelated context (multi-seed validation) and creates no confusion because Paper 2 does not discuss multi-channel LoRA formally.

**Status:** CONSISTENT

---

### T-006: "Fiedler Eigenvalue" Definition
**Severity:** NONE
**Location:** Paper 2 (lines 223-234) vs. Paper 3 (lines 429-437, 964-979)
**Finding:** Paper 2 defines the Fiedler eigenvalue as the second-smallest eigenvalue of the graph Laplacian, citing Fiedler (1973), and extends to the weighted case via $L_w = D_w - A_w$. Paper 3 cites the same Fiedler (1973) and defines the Fiedler value as "the second-smallest eigenvalue of $\mathbf{L}^a$." The underlying definition is the same.

Paper 3 introduces a further distinction in Section 5.2 (lines 964-979) between **Bridge Fiedler** (within-matrix connectivity of a single bridge) and **Correlation Fiedler** (cross-layer structural consistency). These are Paper 3-specific applications of the same eigenvalue concept. Paper 2 only uses the standard graph-theory Fiedler value. The distinction is clearly labeled in Paper 3 and creates no cross-paper confusion.

**Status:** CONSISTENT

---

### T-007: "Block-Diagonal," "Co-Planar," "Cross-Planar"
**Severity:** NONE
**Location:** Paper 3 (defined in Section 3.3, lines 509-530)
**Finding:** These terms appear exclusively in Paper 3 and are defined once, consistently used throughout:
- **Co-planar:** Two face pairs sharing a coordinate axis (3 pairs). Lines 509, and used with consistent meaning in all subsequent occurrences.
- **Cross-planar:** Two face pairs on different axes ($\binom{6}{2} - 3 = 12$ pairs). Line 510-511.
- **Block-diagonal:** Classified when $\rho > 10$ AND all cross-planar entries $< 10^{-3}$. Lines 528-530, restated in Appendix A.6 (lines 1290-1295).

Paper 2 does not use any of these terms. No cross-paper inconsistency possible.

**Status:** CONSISTENT

---

### T-008: Steersman Description
**Severity:** NONE
**Location:** Paper 3 (Section 3.2, lines 390-485)
**Finding:** The Steersman is introduced and defined exclusively in Paper 3. Paper 2 does not mention it. The description is internally consistent within Paper 3: "cybernetic feedback mechanism" (abstract), "second-order cybernetic feedback mechanism" (introduction, line 124-125), defined formally in Section 3.2 with three control laws. All subsequent references are consistent with this definition.

**Status:** CONSISTENT

---

### T-009: "RhombiLoRA" vs. "Multi-Channel LoRA"
**Severity:** NONE
**Location:** Paper 2 (line 861) vs. Paper 3 (line 47 and passim)
**Finding:** Paper 2 introduces "RhombiLoRA" in quotation marks as a preview name in the future-work section (line 861: `this ``RhombiLoRA'' architecture`). Paper 3 uses both terms: "Multi-channel LoRA (RhombiLoRA)" in the abstract (line 47), establishing "RhombiLoRA" as the specific name for the multi-channel LoRA architecture with a bridge matrix. Throughout Paper 3, both terms appear contextually: "multi-channel LoRA" for the general concept, "RhombiLoRA" for the specific architecture. This is consistent.

**Status:** CONSISTENT

---

### T-010: Bridge Matrix Definition Across Papers
**Severity:** NONE (cross-paper; see T-001 for within-P3)
**Location:** Paper 2 (line 859) vs. Paper 3 (line 98, 344-345)
**Finding:** Paper 2 previews the bridge as "a learnable bridge matrix with 6 channels." Paper 3 formalizes it as "a learnable $n \times n$ bridge matrix" (line 98/345) with channels of width $s = r/n$. These are consistent -- Paper 2's informal description matches Paper 3's formal definition.

**Status:** CONSISTENT

---

### T-011: Paper 2 Future Work vs. Paper 3 Scope
**Severity:** LOW
**Location:** Paper 2 (lines 855-862) vs. Paper 3 (full paper)
**Finding:** Paper 2's future-work section previews "a learnable bridge matrix with 6 channels following the RD's face-pair topology." Paper 3 actually tests $n \in \{3, 4, 6, 8\}$, with $n=6$ being the one that produces block-diagonal structure. This is not a contradiction -- Paper 2 correctly anticipates the $n=6$ case, and Paper 3 provides the broader investigation. However, a reader might expect Paper 3 to be exclusively about $n=6$ based on Paper 2's preview.

**Status:** OK -- no factual inconsistency

---

### T-012: Minor Numerical Variation in Paper 3 Abstract vs. Body
**Severity:** LOW
**Location:** Paper 3 abstract (lines 64-67) vs. introduction (lines 184-187)
**Finding:** The abstract reports the Fiedler bifurcation as "$1{,}020\times$" (line 67) and the introduction also reports "$1{,}020\times$" (line 187-188). These are consistent. However, the abstract says "Bridge Fiedler eigenvalue collapses to 0.00009" (line 65), while the ablation table (line 807) confirms 0.00009 for H3. The co-planar/cross-planar ratio in the abstract is "82,854:1" (peak, line 65) while the introduction says "70,404:1" (H3 final, line 185). Both are correctly sourced to different experiments/timepoints (C-003 peak vs. H3 convergence), but the abstract conflates metrics from different experiments without attribution, which could confuse a careful reader.

**Status:** VERIFY -- the numbers are correct but sourced from different experiments. Consider clarifying in the abstract which experiment each number comes from.

---

## Cross-Paper Notation Summary

| Concept | Paper 2 | Paper 3 | Consistent? |
|---------|---------|---------|-------------|
| FCC lattice | "face-centered cubic (FCC)" | "face-centered cubic (FCC) lattice" | Yes |
| Rhombic dodecahedron | "RD" | "RD" | Yes |
| Fiedler eigenvalue | $\lambda_2$, $\lambda_2(L_w)$ | $\lambda_2$, $\lambda_2^a$, Bridge Fiedler, Correlation Fiedler | Yes -- P3 extends |
| Bridge matrix | "bridge matrix" (informal, future work) | $\mathcal{B}$ (intro) / $\mathbf{M}$ (method+) | **No -- within-P3 conflict** |
| Channel | "channels" (quoted metaphor for lattice) | channels (formal LoRA term) | Yes -- deliberate transition |
| RhombiLoRA | "RhombiLoRA" (quoted preview) | RhombiLoRA (formal name) | Yes |
| Steersman | Not mentioned | Defined in Section 3.2 | N/A |
| Paper 1 metrics | 2.3--2.5x, 30--32% | 2.4x, 30% | **No -- point vs. range** |
| Direction pair / face pair | "direction pair" (primary) | "face pair" (primary), "direction pair" (once, line 496) | Yes -- contextually appropriate |

---

## Actionable Items

1. **T-001 (HIGH):** Unify bridge matrix notation in Paper 3. Replace `$\mathcal{B}$` with `$\mathbf{M}$` in lines 98-103 of the Introduction.
2. **T-002 (MEDIUM):** Standardize Paper 1 metric citation. Check what Paper 1 actually reports and use the same form in both Papers 2 and 3.
3. **T-012 (LOW):** Consider attributing abstract numbers to specific experiments (e.g., "peak of 82,854:1 [C-003]" and "convergence at ~70,000:1 [H3]") for precision, or consistently report convergence-band numbers rather than mixing peak and convergence values.
