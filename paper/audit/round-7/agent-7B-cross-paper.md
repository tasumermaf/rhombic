# Round 7 — Agent 7B: Final Cross-Paper Coherence Check

**Auditor:** Agent 7B (Opus 4.6)
**Date:** 2026-03-15
**Scope:** Papers 1 (`rhombic.tex`, LOCKED), 2 (`rhombic-paper2.tex`), 3 (`rhombic-paper3.tex`)
**Prior rounds consulted:** FINDINGS_TRACKER.md (Rounds 1-6, 210 findings)

---

## 1. Does Paper 3's Introduction Correctly Characterize Papers 1 and 2?

### Paper 1 characterization (P3 Section 2.4, lines 341-343):

> "In prior work [bielec2026shape], we established that 12-connected FCC
> lattices provide 30% shorter paths and 2.4x higher algebraic connectivity
> than 6-connected cubic lattices at comparable spatial resolution."

**Paper 1 actual claims:**
- Paths: "29-32%" shorter (Table 1, prose line 315). The "30%" summary is a fair rounded midpoint. **CLEAN.**
- Algebraic connectivity: Paper 1 abstract says "2.4x" (line 56). Body says "2.3-2.5x" (line 316). Data shows 2.31x, 2.55x, 2.43x across three scales (Table 1). Paper 3 uses "2.4x" which matches Paper 1's own abstract. **CLEAN.**

### Paper 2 characterization (P3 Section 2.4, lines 344-346):

> "A companion paper [bielec2026weights] showed that structured edge weights
> aligned with the RD's 6 face-pair directions amplify this advantage to
> 6.1x---the same directional structure that motivates the contrastive loss
> in Section 3.2."

**Paper 2 actual claims:**
- The 6.1x figure is from direction-weighted corpus at scale 1,000 (Table 2 / Exp 5, line 441: Fiedler ratio = 6.11). Paper 3's "6.1x" is a fair rounding. **CLEAN.**
- "6 face-pair directions" is accurate: Paper 2 Section 2.2 describes 6 antipodal direction pairs. **CLEAN.**
- The amplification claim is specifically for direction-weighted corpus, not edge-cycled (which peaks at 3.18x). Paper 3's characterization is accurate---it says "aligned with the RD's 6 face-pair directions," correctly identifying the direction-weighted method. **CLEAN.**

**Verdict: Paper 3 characterizes both companions accurately.**

---

## 2. Cross-Paper Citations

### Citation keys and bib entries

| Citation | In Paper | Bib File | Title Match |
|----------|----------|----------|-------------|
| `bielec2026shape` | P2, P3 | `rhombic-paper2.bib`, `rhombic-paper3.bib` | Exact match to P1 `\title{}` |
| `bielec2026weights` | P3 | `rhombic-paper3.bib` | Exact match to P2 `\title{}` |
| `bielec2026bridge` | P2 | `rhombic-paper2.bib` | Exact match to P3 `\title{}` |

All bib entry titles verified against the actual `\title{}` commands in each paper. Metadata (author, year, URL) consistent across all instances.

**Note:** The P2 bib entry `bielec2026bridge` was fixed in Round 1 (F-2B-01) and the P3 citation to P2 was added in Round 3 (F-3A-03). Both fixes are present in the current text.

### Citation directions

- **P1 cites P2/P3:** No. Correct---Paper 1 is LOCKED and has no forward references.
- **P2 cites P1:** Yes, `bielec2026shape` at lines 45, 78, 171, 504. Correct.
- **P2 cites P3:** Yes, `bielec2026bridge` at line 872 (Future Work). Correct---forward reference to the TeLoRA architecture.
- **P3 cites P1:** Yes, `bielec2026shape` at line 341. Correct.
- **P3 cites P2:** Yes, `bielec2026weights` at line 344. Correct---added in Round 3 fix.

**Verdict: All cross-paper citations present, accurate, and bidirectional where appropriate.**

---

## 3. Three-Paper Narrative Arc

### P1 (Lattice Topology) -> P2 (Weighted Lattices) -> P3 (Learned Bridge)

**P1 establishes:** FCC lattice provides 2.3-2.5x algebraic connectivity advantage over SC under uniform weights, across four domains (graph theory, spatial ops, signal processing, embedding retrieval). Scale range 125-8,000 nodes.

**P2 extends:** Heterogeneous edge weights amplify the FCC advantage. Seven experiments across two registers (lattice-scale, single-cell). Key result: direction-weighted corpus pushes Fiedler ratio from 2.55x (uniform) to 6.11x (corpus) at scale 1,000. Mechanism identified as bottleneck resilience. Future Work section explicitly previews Paper 3's TeLoRA.

**P3 applies:** The RD's 6 face-pair direction structure (which P2 showed amplifies the lattice advantage) becomes the geometric prior for multi-channel LoRA. The Steersman encodes face-pair relationships via contrastive loss. Block-diagonal structure emerges, organized along the same 3-axis coordinate system.

**Assessment:** The arc is coherent and well-motivated. Each paper builds on its predecessor's headline finding:
- P1's connectivity advantage motivates P2's question about weight heterogeneity.
- P2's direction-pair amplification motivates P3's encoding of face-pair geometry in the contrastive loss.
- The connection from P2 to P3 is now explicit (line 344-346 of P3), thanks to the Round 3 fix.

**CLEAN.**

---

## 4. Terminology Consistency Between Papers

### Terms used across multiple papers

| Concept | Paper 1 | Paper 2 | Paper 3 | Consistent? |
|---------|---------|---------|---------|-------------|
| Lattice comparison | "SC" and "FCC" | "SC" and "FCC", also "cubic" | "FCC lattice" | Yes |
| Voronoi cell name | "rhombic dodecahedra" (line 88) | "rhombic dodecahedron" | "rhombic dodecahedron (RD)" | Yes |
| Algebraic connectivity | "Fiedler value" (line 286) | "weighted Fiedler value" | "Fiedler eigenvalue" / "Bridge Fiedler" | Yes (P3 introduces "Bridge Fiedler" as a distinct concept, clearly distinguished) |
| Face-pair structure | 12 faces, 6 pairs (implicit in Voronoi description) | "6 antipodal pairs" / "6 direction pairs" | "6 face pairs" / "6 antipodal pairs" | Yes |
| RD vertex structure | "14 vertices (8 of valence 3 and 6 of valence 4)" (line 141) | Not stated | "14 vertices (8 cubic and 6 octahedral)" (line 510) | **MINOR: Different naming convention** |
| Test count | "256 tests" (line 120) | "312 tests" (line 965) | "312 tests" (line 1212) | P1 locked at older count; P2/P3 consistent |
| Package version | "v0.1.2" (line 261) | "v0.3.0" (line 966) | "v0.3.0" (line 1212) | P1 locked at older version; P2/P3 consistent |
| Seed | "seed=42" (line 255) | "seed=42" (line 358) | "seed=42" (line 564) | Yes |
| Consensus definition | "Laplacian averaging... epsilon=0.05" (lines 464-465, 649-650) | "$\epsilon = 0.05$... $(\deg(i)+1)$" (lines 341-348) | N/A (consensus not measured) | Yes between P1/P2 |

### FINDING 7B-01 (MINOR): RD vertex terminology

Paper 1 calls the 14 RD vertices "8 of valence 3 and 6 of valence 4" (topological description). Paper 3 calls them "8 cubic and 6 octahedral" (geometric description). Both are standard and correct. The reader seeing both papers would not be confused, but the terminology shift is worth noting. A reviewer of the series might notice it.

**Severity: MINOR** (no contradiction, two valid naming conventions for the same objects)

---

## 5. Numerical Claims in P3 About P1/P2 Results

### P3 about P1

| P3 Claim | P1 Source | Match? |
|----------|-----------|--------|
| "30% shorter paths" (line 342) | "29-32%" (line 315); "30%" (abstract line 55) | Yes (P1's own abstract summary) |
| "2.4x higher algebraic connectivity" (line 342) | "2.4x" (abstract line 56); "2.3-2.5x" (line 316) | Yes (P1's own abstract) |

### P3 about P2

| P3 Claim | P2 Source | Match? |
|----------|-----------|--------|
| "amplify this advantage to 6.1x" (line 346) | 6.11x at scale 1,000 (Table 2, line 441) | Yes (fair rounding) |
| "RD's 6 face-pair directions" (line 345) | "6 antipodal pairs" (Section 2.2, line 261-263) | Yes |

### P2 about P1

| P2 Claim | P1 Source | Match? |
|----------|-----------|--------|
| "2.3x algebraic connectivity advantage" (abstract line 47) | 2.31x at scale 125 (Table 1, line 304); range 2.3-2.5x (line 316) | Yes (lower bound of range) |
| "30-32% shorter average paths" (line 94-95) | "29-32%" (line 315) | **MINOR discrepancy** |
| "40% smaller diameter" (line 95) | "38-42%" derived from diameter data (lines 339-340: "38-42%") | Yes |
| "2.3-2.5x Fiedler ratio under uniform weights" (lines 83, 96) | 2.31x, 2.55x, 2.43x (Table 1, lines 304-308) | Yes |
| "125-8,000 nodes" (lines 82, 96) | P1 tests 125-4,000 in Table 1; mentions 8,000 in spatial ops | **See below** |

### FINDING 7B-02 (MINOR): P2 path advantage range

P2 line 94 says "30-32% shorter average paths" referencing P1. P1 actually says "29-32%" (line 315) and the ASPL advantage ratios in Table 1 are 1.45x, 1.46x, 1.40x, corresponding to 31%, 31.5%, 28.6%. The "29-32%" range in P1 is the more accurate characterization. P2's "30-32%" slightly narrows the range by dropping the 28.6% lower end.

**Severity: MINOR** (the difference is trivial and P2's range is contained within reasonable rounding)

**Note:** This was previously identified as F-3A-01 in Round 3, with status ACCEPTED (both defensible). Confirmed: the finding stands but is not actionable given P1 is locked and the discrepancy is minimal.

### FINDING 7B-03 (MINOR): P2 scale range "125-8,000 nodes"

P2 lines 82 and 96 state that P1 found "stable FCC advantages at all tested scales (125-8,000 nodes)." P1's graph-theory benchmarks (Table 1) test 125, 1,000, and 4,000 nodes. P1 reaches 8,000 only in spatial operations (Table 2) and signal processing (Table 3). The "125-8,000 nodes" range is correct for P1 overall but not for the specific algebraic connectivity measurement that P2 extends. P2's own experiments then go up to 8,000, which is where the confusion might arise.

**Severity: MINOR** (technically accurate for P1 overall, slightly misleading for the specific Fiedler metric)

---

## 6. Bibliography Entries for Companion Papers

All verified in Section 2 above. Summary:

- `bielec2026shape` (Paper 1): Title, author, year, URL correct in both P2 and P3 bib files.
- `bielec2026weights` (Paper 2): Title, author, year, URL correct in P3 bib file.
- `bielec2026bridge` (Paper 3): Title, author, year, URL correct in P2 bib file.

All three entries use `@unpublished` entry type with `note` field pointing to the GitHub repository. Consistent format.

**CLEAN.**

---

## 7. Additional Cross-Paper Observations

### 7A: Consensus inversion consistency

P1 reports consensus convergence advantage of 0.93x at 1,000 nodes (line 487-488: "0.93x"). P2 reports the same effect under direction-weighted corpus: 0.73x at 1,000 nodes (Table 2, line 441) and confirms "This matches the unweighted Paper 1 finding" (lines 503-504). P2's Table 7 (summary) states "Consensus speedup: 0.93x (scale 1000)" for Paper 1. This matches P1.

**CLEAN.**

### 7B: Fiedler ratio at scale 1,000 baseline

P2's uniform Fiedler ratio at scale 1,000 is 2.55 (Table 1, line 392; Table 2, line 438). P1's algebraic connectivity at scale 1,000 is 2.55x (Table 1, line 306). These match exactly---P2 reproduces P1's baseline correctly.

**CLEAN.**

### 7C: "Direction pair" / "face pair" terminology bridge from P2 to P3

P2 uses "direction pairs" (lattice-scale language) and "face pairs" (single-cell language) interchangeably, noting they are the same geometric object viewed from different perspectives (Section 2.2, lines 260-272). P3 uses "face pairs" consistently. The transition from P2's lattice framing to P3's geometric framing is natural and internally consistent.

**CLEAN.**

### FINDING 7B-04 (MINOR): P2 "five tested 24-edge graph topologies" vs "all five" discrepancy

P2 abstract (line 61) says "all five tested 24-edge graph topologies" for Fiedler suppression. P2 Contributions item 3 (line 195) says "all five tested 24-edge graph topologies." But Table 5 (Exp 7) actually lists the RD, cuboctahedron, K_{4,6}, 3-regular, and random G(14,24)---that is indeed 5. However, the abstract says "percentiles 0.02%-5.94%" while the Contributions say the same. These are consistent with Table 5. **CLEAN** on numbers, but worth noting that the abstract and Contributions both characterize these as "graph topologies" when one of the five is a random graph, not a polytope. This was previously flagged as F-3A-09 (POLISH) and accepted.

### FINDING 7B-05 (MINOR): P1 test count in Paper 1 vs Papers 2/3

Paper 1 reports "256 tests" (line 120), referencing the `rhombic` library version at that time. Papers 2 and 3 both report "312 tests" from v0.3.0. This is not a contradiction---the library grew between papers---but a reader encountering all three would notice different test counts for the same library. Paper 1 is locked and cannot be updated. This was previously identified as F-3A-05 and fixed by updating P2 to 312. The residual discrepancy with P1 is unavoidable.

**Severity: MINOR** (acknowledged constraint of P1 being locked)

---

## Summary of New Findings

| ID | Severity | Papers | Finding |
|----|----------|--------|---------|
| 7B-01 | MINOR | P1, P3 | RD vertex terminology: "valence 3/4" (P1) vs "cubic/octahedral" (P3) |
| 7B-02 | MINOR | P1, P2 | Path advantage range: P1 says 29-32%, P2 says 30-32% |
| 7B-03 | MINOR | P1, P2 | P2 states P1 tested "125-8,000 nodes" for algebraic connectivity; P1 graph-theory table only goes to 4,000 |
| 7B-04 | MINOR | P2 | "Five graph topologies" includes a random graph (not a polytope); previously flagged |
| 7B-05 | MINOR | P1, P2, P3 | P1 says 256 tests; P2/P3 say 312 tests (same library, different versions); previously flagged |

**No CRITICAL or MAJOR findings.**

---

## Overall Cross-Paper Coherence Assessment

The three-paper series is in strong shape after six rounds of adversarial audit. The narrative arc (P1: topology baseline -> P2: weight amplification -> P3: learned bridge) is coherent and well-connected. Cross-paper citations are complete and accurate. Numerical claims about companion papers match their sources. Terminology is consistent with only minor naming variations (vertex valence vs geometric type) that are standard in the field.

The remaining findings are all MINOR and none would be caught by a reviewer as a contradiction. Most are inherent constraints of Paper 1 being locked (older test count, slightly different numerical ranges). The series reads as a unified body of work by a single author with a clear progression of ideas.

**Series-level verdict: PASS.**
