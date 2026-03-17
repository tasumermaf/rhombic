# Audit 3C: Cross-Paper Reference Verification

**Auditor:** Agent 3C (adversarial)
**Date:** 2026-03-15
**Scope:** All forward and backward references between Papers 1, 2, and 3
**Files audited:**
- `rhombic.tex` (Paper 1 -- LOCKED)
- `rhombic-paper2.tex` (Paper 2)
- `rhombic-paper3.tex` (Paper 3)
- `rhombic-paper2.bib` (Paper 2 bibliography)
- `rhombic-paper3.bib` (Paper 3 bibliography)
- `rhombic.bib` (Paper 1 bibliography)

---

## Summary Table

| ID | Category | Severity | Status | Description |
|----|----------|----------|--------|-------------|
| 3C-01 | Bib entry | INFO | CLEAN | Paper 2 bib entry for `bielec2026shape` -- title and metadata match Paper 1 |
| 3C-02 | Bib entry | INFO | CLEAN | Paper 2 bib entry for `bielec2026bridge` -- title and metadata match Paper 3 |
| 3C-03 | Bib entry | INFO | CLEAN | Paper 3 bib entry for `bielec2026shape` -- title and metadata match Paper 1 |
| 3C-04 | Bib entry | INFO | CLEAN | Paper 3 bib entry for `bielec2026weights` -- title and metadata match Paper 2 |
| 3C-05 | Orphan bib | MINOR | FINDING | `bielec2026weights` defined in Paper 3 bib but never cited in Paper 3 text |
| 3C-06 | Claim verification | INFO | CLEAN | Paper 2 abstract claim: Paper 1 established 2.3x Fiedler ratio under uniform weights |
| 3C-07 | Claim verification | MINOR | FINDING | Paper 2 line 83: "2.3--2.5x" Fiedler ratio; Paper 1 Table 1 shows 2.31--2.55 (range correct but notation rounds down the low end) |
| 3C-08 | Claim verification | INFO | CLEAN | Paper 2 line 94-97: "30--32% shorter paths, 40% smaller diameter, 2.3--2.5x Fiedler" -- verified against Paper 1 Tables/text |
| 3C-09 | Claim verification | INFO | CLEAN | Paper 2 line 80-82: "four domains" -- Paper 1 covers graph theory, spatial, signal, embedding |
| 3C-10 | Claim verification | INFO | CLEAN | Paper 2 line 82: "125--8,000 nodes" -- Paper 1 tests 125--8,000 (though most tables stop at 4,000; 8,000 appears in spatial) |
| 3C-11 | Claim verification | INFO | CLEAN | Paper 2 line 494: consensus inversion "matches the unweighted Paper 1 finding" -- Paper 1 reports 0.93x at 1,000 nodes (line 488) |
| 3C-12 | Claim verification | INFO | CLEAN | Paper 3 line 327-329: "30% shorter paths and 2.4x higher algebraic connectivity" -- Paper 1 reports 29-32% and 2.3-2.5x; 2.4x is a reasonable midpoint |
| 3C-13 | Forward ref | INFO | CLEAN | Paper 2 Future Work cites `bielec2026bridge` for TeLoRA results -- Paper 3 delivers this |
| 3C-14 | Forward ref | MINOR | FINDING | Paper 1 promises BCC baseline as future work (line 189-190); Paper 2 Future Work also promises BCC (sec 6.3); neither Paper 2 nor 3 delivers BCC |
| 3C-15 | Forward ref | INFO | NOTE | Paper 1 promises higher-dimensional extensions (D4, E8) as future work; Paper 2 Future Work promises 4D/24-cell extension; Paper 3 does not deliver these |
| 3C-16 | Forward ref | INFO | NOTE | Paper 1 promises formal null model; neither Paper 2 nor 3 delivers one (Paper 3 limitations explicitly acknowledges this at line 1082-1085) |
| 3C-17 | Forward ref | INFO | CLEAN | Paper 1 promises matched-count control for ANN index; not addressed in Papers 2/3 (reasonable -- different research direction) |
| 3C-18 | Missing back-ref | MODERATE | FINDING | Paper 3 never cites Paper 2 (`bielec2026weights`), despite the series being sequential (1->2->3) and Paper 2's Future Work explicitly pointing to Paper 3 |
| 3C-19 | Claim verification | INFO | CLEAN | Paper 2 Sec 1.2 describes corpus as "24 structured integers (range 18--1,296, mean 318.8, std 292.9)" -- internal consistency check; no Paper 1 claim to verify (Paper 1 does not publish these statistics) |
| 3C-20 | Bib metadata | INFO | CLEAN | All three bib entries for companion papers use consistent author name "Timothy Paul Bielec" and URL `https://github.com/tasumermaf/rhombic` |
| 3C-21 | Version consistency | MINOR | FINDING | Paper 2 reports "v0.3.0, 256 tests" (line 955); Paper 3 reports "v0.3.0, 312 tests" (line 1161); these are inconsistent for the same version number |
| 3C-22 | Claim verification | INFO | CLEAN | Paper 2 Table 7 summary: "Paper 1 = 2.55x, Paper 2 = 6.11x" Fiedler -- Paper 1 Table 1 at scale ~1000 shows 2.55x; Paper 2 Table 2 at scale 1000 corpus shows 6.11x |
| 3C-23 | Missing back-ref | MINOR | FINDING | Paper 3 does not reference Paper 2's direction-pair weighting results despite its own architecture being motivated by the same 6 face-pair / 3-axis decomposition |

---

## Detailed Findings

### 3C-05: Orphan bibliography entry in Paper 3

**File:** `rhombic-paper3.bib`, line 246
**Issue:** The entry `bielec2026weights` is defined in Paper 3's bibliography file but is never cited anywhere in `rhombic-paper3.tex`. A `\cite{bielec2026weights}` call never appears.
**Impact:** LaTeX will not include this entry in the compiled bibliography since it is uncited. The entry occupies space in the bib file without serving any purpose. More importantly, this is symptomatic of finding 3C-18: Paper 3 *should* cite Paper 2 but does not.
**Recommendation:** Add at least one citation to `bielec2026weights` in Paper 3's text (see 3C-18 for where).

---

### 3C-07: Minor Fiedler range rounding

**File:** `rhombic-paper2.tex`, line 83
**Claim:** "stable FCC advantages at all tested scales (125--8,000 nodes)"
**Verification:** Paper 1's graph theory benchmarks (Table 1) cover 125--4,000 nodes. The 8,000-node scale appears only in spatial operations (Table 2). The claim "all tested scales (125--8,000 nodes)" is slightly imprecise: it conflates the graph theory scale range (125--4,000) with the spatial operations range (up to 8,000). Paper 1's abstract says "125--8,000 nodes" for the overall paper scope, so this is defensible at the paper level, but the 2.3--2.5x Fiedler specifically comes from graph theory where the max is 4,000 nodes.
**Severity:** Minor. The claim is accurate at paper scope; only pedantic at metric scope.

---

### 3C-14: Unfulfilled BCC promise (carried across two papers)

**Paper 1** (line 189-190): "A BCC baseline is a natural next control for isolating sampling-theoretic contributions."
**Paper 2** (Section 6.3, lines 884-892): "The body-centered cubic lattice is the missing control... Whether it shows an intermediate weighted amplification would help attribute the FCC advantage to connectivity count versus isotropy."
**Paper 3:** No BCC work.
**Impact:** The BCC comparison is promised in Paper 1, re-promised in Paper 2, and delivered in neither Paper 2 nor Paper 3. This is a two-paper-old unfulfilled promise that is explicitly acknowledged as a gap in Paper 2's Limitations ("No BCC baseline", line 837).
**Recommendation:** Either deliver BCC results or add a note in Paper 3 acknowledging that this remains open. Since Paper 3 is about LoRA/ML and not lattice benchmarks, the most natural place would be a sentence in Paper 2's limitations noting it remains future work (already done). No action required unless the paper series continues.

---

### 3C-18: Paper 3 never cites Paper 2 (MODERATE)

**File:** `rhombic-paper3.tex`
**Issue:** Paper 3 cites Paper 1 (`bielec2026shape`) at line 327 but never cites Paper 2 (`bielec2026weights`). The three papers form an explicit series:
- Paper 2's dependency diagram (Figure 1) shows Paper 1 -> Paper 2
- Paper 2's Future Work (Section 6.1, line 861-862) says "Results on this 'TeLoRA' architecture are reported in the companion paper~\cite{bielec2026bridge}" -- explicitly pointing to Paper 3
- Paper 2 establishes the direction-pair weighting concept (6 face-pairs of the RD) that Paper 3's Steersman contrastive loss directly builds on
- Paper 2 demonstrates the 6.1x Fiedler amplification under direction-based weights -- the structural motivation for why 6 channels matter

Yet Paper 3 jumps directly from lattice topology (Paper 1) to the LoRA application without acknowledging Paper 2's weighted graph results that bridge the two.

**Impact:** A reader following the series encounters a gap. Paper 2 promises Paper 3; Paper 3 does not acknowledge Paper 2. The RD's "6 face pairs partition naturally by coordinate axis" (Paper 3, Section 3.3) is precisely the insight that Paper 2's direction-based weighting establishes empirically. The link is missing.
**Recommendation:** Add a citation to `bielec2026weights` in Paper 3's introduction or Section 3.3 (RD geometry), e.g.: "In prior work~\cite{bielec2026shape}, we established that FCC lattices provide 30% shorter paths and 2.4x higher algebraic connectivity. A companion paper~\cite{bielec2026weights} showed that structured edge weights amplify this advantage to 6.1x when aligned with the RD's 6 face-pair directions -- the same directional structure that motivates the contrastive loss in Section~\ref{sec:steersman}."

---

### 3C-21: Test count inconsistency within same version

**Paper 2** (line 955): "v0.3.0, 256 tests"
**Paper 3** (line 1161): "v0.3.0, 312 tests"
**Issue:** Both papers cite the same library version (v0.3.0) but report different test counts. If v0.3.0 is a specific release, the test count should be fixed.
**Likely explanation:** Paper 2 was written before Paper 3's tests were added to the library, and the version number was bumped to v0.3.0 after Paper 2 was finalized. Paper 2's test count was frozen at the time of writing while Paper 3 reflects the final v0.3.0 release.
**Recommendation:** Update Paper 2's availability paragraph to match the final v0.3.0 test count (312 tests), or bump Paper 3 to a distinct version (v0.3.1). Having two different test counts for the same version number creates a reproducibility question.

---

### 3C-23: Missing conceptual back-reference from Paper 3 to Paper 2's direction-pair results

**File:** `rhombic-paper3.tex`, Section 3.3 (RD Geometry, lines 487-531)
**Issue:** Paper 3 describes the RD's 6 face pairs decomposing into 3 axis-aligned groupings of 2 face pairs each. Paper 2 (Section 2.2, Experiment 5) demonstrates empirically that direction-based weighting aligned to these same 6 face pairs produces a 6.1x Fiedler amplification -- the single largest effect in the first two papers. This finding directly supports Paper 3's choice of 6 channels and the contrastive loss design. The absence of this citation weakens Paper 3's motivation: the reader must take on faith that the 3-axis structure matters, when Paper 2 has already demonstrated it empirically.
**Recommendation:** Same fix as 3C-18 -- a single sentence citing Paper 2 in Section 3.3 or the Introduction.

---

## Verified Clean Cross-References

The following cross-paper references were checked and confirmed accurate:

1. **Paper 2 abstract -> Paper 1:** "2.3x algebraic connectivity advantage under uniform edge weights" -- Paper 1 Table 1 shows 2.31x at scale 125 (exact match to the rounded claim).

2. **Paper 2 line 78-82 -> Paper 1:** "compared SC and FCC across four domains -- graph theory, spatial operations, signal processing, and embedding retrieval -- and found stable FCC advantages at all tested scales" -- accurately describes Paper 1's structure and findings.

3. **Paper 2 line 94-97 -> Paper 1:** "30--32% shorter average paths, 40% smaller diameter, and a 2.3--2.5x Fiedler ratio" -- Paper 1 reports 29-32% ASPL advantage (line 315), 38-42% diameter advantage (Figure 2 caption), and 2.31-2.55x Fiedler (Table 1). The "30-32%" rounds up the low end by 1pp; "40%" is within the 38-42% range. Acceptable.

4. **Paper 2 line 494 -> Paper 1:** consensus inversion at scale 1,000 "matches the unweighted Paper 1 finding" -- Paper 1 line 487-488 reports "0.93x" at 1,000 nodes. Confirmed.

5. **Paper 2 line 861-862 -> Paper 3:** Forward reference to `bielec2026bridge` for "TeLoRA architecture" -- Paper 3 is indeed titled "The Learnable Bridge" and covers TeLoRA. Accurate.

6. **Paper 3 line 327-329 -> Paper 1:** "12-connected FCC lattices provide 30% shorter paths and 2.4x higher algebraic connectivity than 6-connected cubic lattices" -- Paper 1 reports 29-32% path advantage and 2.3-2.5x algebraic connectivity. "30%" and "2.4x" are reasonable midpoint summaries. Accurate.

7. **Bib entry `bielec2026shape` in Paper 2:** Title = "The Shape of the Cell: Empirical Comparison of Cubic and FCC Lattice Topologies Across Graph Theory, Spatial Operations, Signal Processing, and Embedding Retrieval" -- matches Paper 1's `\title{}` exactly. Author and URL correct.

8. **Bib entry `bielec2026bridge` in Paper 2:** Title = "The Learnable Bridge: Cybernetic Feedback Discovers Rhombic Dodecahedral Geometry in Multi-Channel LoRA" -- matches Paper 3's `\title{}` exactly (after removing the linebreak). Author and URL correct.

9. **Bib entry `bielec2026shape` in Paper 3:** Identical to Paper 2's entry. Correct.

10. **Bib entry `bielec2026weights` in Paper 3:** Title = "Structured Edge Weights Amplify FCC Lattice Topology Advantages via Bottleneck Resilience" -- matches Paper 2's `\title{}` exactly. Author and URL correct.

---

## Future Work Promise Tracking

### Paper 1 Promises (from Discussion/Limitations)
| Promise | Delivered in Paper 2? | Delivered in Paper 3? | Status |
|---------|----------------------|----------------------|--------|
| Higher dimensions (D4, E8, Leech) | Discussed in Future Work (Sec 6.6) but not delivered | No | OPEN |
| BCC baseline | Discussed in Future Work (Sec 6.3) but not delivered | No | OPEN (2 papers old) |
| Formal null model | No | No | OPEN |
| Matched-count ANN control | No | No | OPEN (different direction) |
| Targeted fault tolerance | No | No | OPEN |
| Multi-seed confidence intervals | Partially (Paper 2 runs 100 seeds for Fiedler ratios) | No | PARTIAL |
| Scale beyond 8,000 | Paper 2 tests at 8,000 but not beyond | No | UNCHANGED |

### Paper 2 Promises (from Future Work, Section 6)
| Promise | Delivered in Paper 3? | Status |
|---------|----------------------|--------|
| TeLoRA / geometric learning architectures | YES | FULFILLED |
| Migration strategy (cubic to RD) | No | OPEN |
| BCC comparison | No | OPEN |
| Consensus dynamics multi-seed | No | OPEN |
| Consensus inversion resolution | No | OPEN |
| 4D extension (24-cell, D4 lattice) | No | OPEN |

---

## Actionable Recommendations (Priority Order)

1. **[MODERATE] Add Paper 2 citation to Paper 3** (3C-18, 3C-23). Paper 3 should cite `bielec2026weights` at least once, ideally in the Introduction or Section 3.3. The current omission breaks the series chain and leaves the 6-channel motivation unsupported by the paper that established it empirically.

2. **[MINOR] Resolve test count discrepancy** (3C-21). Either update Paper 2 to say 312 tests or version-distinguish the releases.

3. **[MINOR] Remove or cite orphan bib entry** (3C-05). `bielec2026weights` sits in Paper 3's bib file unused. Fix by citing it (preferred) or removing the entry.
