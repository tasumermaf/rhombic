# Agent 3A: Cross-Paper Narrative Coherence Audit

**Auditor:** Agent 3A (adversarial narrative coherence)
**Papers audited:**
- Paper 1: `rhombic.tex` — "The Shape of the Cell" (LOCKED)
- Paper 2: `rhombic-paper2.tex` — "Structured Edge Weights Amplify FCC Lattice Topology Advantages via Bottleneck Resilience"
- Paper 3: `rhombic-paper3.tex` — "The Learnable Bridge: Cybernetic Feedback Discovers Rhombic Dodecahedral Geometry in Multi-Channel LoRA"

**Date:** 2026-03-15

---

## Findings

### F-3A-01: Algebraic Connectivity Ratio Inconsistency Across Papers
- Severity: MINOR
- Location: Paper 1 abstract (line 56), Paper 2 abstract (line 46), Paper 3 Section 2.4 (line 328)
- Finding: The three papers cite three different values for the same headline metric (FCC/SC algebraic connectivity ratio under uniform weights).
- Evidence:
  - **Paper 1 abstract:** "2.4x algebraic connectivity" (line 56)
  - **Paper 1 data table and prose:** "2.3--2.5x" range (lines 304-308, line 316); specific values 2.31x (scale 125), 2.55x (scale 1000), 2.43x (scale 4000)
  - **Paper 2 abstract:** "$2.3\times$ algebraic connectivity advantage" (line 46)
  - **Paper 2 intro:** "$2.3$--$2.5\times$ algebraic connectivity ratio" (lines 83-84)
  - **Paper 3 Section 2.4:** "$2.4\times$ higher algebraic connectivity" (line 328)
  - Paper 1's abstract rounds to 2.4x (midpoint). Paper 2's abstract cites 2.3x (lower bound). Paper 3 matches Paper 1's abstract (2.4x). These are all defensible summaries of the same range, but a reader comparing the three will notice the discrepancy. Recommend standardizing to "2.3--2.5x" everywhere or "~2.4x" everywhere.
- Status: NEW

### F-3A-02: Paper 2 Summary Table Mixes Scales Without Annotation
- Severity: MAJOR
- Location: Paper 2, Table 7 (lines 798-812)
- Finding: The amplification summary table compares Paper 1 and Paper 2 results, but the consensus speedup row compares different scales without annotation, creating a misleading impression.
- Evidence:
  - Table caption states: "Scale noted per metric" (line 801)
  - However, no scale is noted in any row of the table.
  - The Fiedler ratio row implicitly compares scale 1,000 (Paper 1: 2.55x at scale 1000, Paper 2: 6.11x at scale 1000) -- valid.
  - The path advantage row compares scale 1,000 (Paper 1: 32% corresponds to path ratio 0.68, Paper 2: 61% corresponds to path ratio 0.39) -- valid.
  - The consensus speedup row compares Paper 1 = 0.93x (scale 1,000) with Paper 2 = 6.69x (scale 125). These are at *different scales*. The comparison is misleading because the consensus metric inverts between these scales. At scale 1,000, Paper 2's own consensus speedup is 0.73x (corpus, Table 3 line 435), which is *worse* than Paper 1's 0.93x.
  - The table makes it look like Paper 2 improved consensus from 0.93x to 6.69x, when actually consensus got worse at matched scale (0.93x to 0.73x) and better only at the smaller scale.
- Status: NEW

### F-3A-03: Paper 3 Does Not Reference Paper 2
- Severity: MAJOR
- Location: Paper 3, Section 1 (Introduction) and Section 2.4 (The Rhombic Dodecahedron)
- Finding: Paper 3 references Paper 1 (`bielec2026shape`) but never references Paper 2 (`bielec2026weights`). The narrative arc should be Paper 1 -> Paper 2 -> Paper 3, but Paper 3 skips Paper 2 entirely. Paper 2's key findings (amplification under structured weights, bottleneck resilience mechanism, direction-pair weighting as the bridge from lattice topology to channel structure) are the natural stepping stone to Paper 3's channel-based architecture.
- Evidence:
  - Paper 3 bibliography includes `bielec2026shape` (Paper 1) and `bielec2026weights` (Paper 2), so both entries exist.
  - Paper 3 cites `bielec2026shape` at line 327: "In prior work [bielec2026shape], we established that 12-connected FCC lattices provide 30% shorter paths and $2.4\times$ higher algebraic connectivity."
  - `bielec2026weights` is never cited anywhere in Paper 3's body text.
  - Paper 2's central concept -- direction-pair weighting across 6 face-pair channels of the RD -- is precisely what motivates Paper 3's 6-channel architecture. Paper 3 says channels correspond to the RD's "6 face-pair normals" (line 494-497) but never mentions that Paper 2 already demonstrated the significance of this 6-channel direction-pair structure.
  - The `bielec2026weights` entry exists in the .bib file but is unused ("orphan entry").
- Status: NEW

### F-3A-04: Paper 2 Future Work -> Paper 3 Link Is Accurate
- Severity: POLISH (positive finding)
- Location: Paper 2, Section 7.1 (lines 855-862)
- Finding: Paper 2's Future Work section describes "TeLoRA" and forward-references Paper 3 (`bielec2026bridge`) accurately. The description matches what Paper 3 actually delivers.
- Evidence:
  - Paper 2 (lines 857-862): "a learnable bridge matrix with 6 channels following the RD's face-pair topology could add cross-rank mixing at negligible parameter cost. Results on this 'TeLoRA' architecture are reported in the companion paper [bielec2026bridge]."
  - Paper 3 delivers exactly this: a 6-channel bridge matrix, RD face-pair topology, negligible parameter cost (0.04% of LoRA budget).
  - The forward reference is accurate and the delivered content matches.
- Status: NEW

### F-3A-05: Test Count Inconsistency Across Papers
- Severity: MINOR
- Location: Paper 1 (line 120), Paper 2 (lines 63, 363, 955), Paper 3 (line 1161)
- Finding: Papers 1 and 2 cite 256 tests; Paper 3 cites 312 tests. Both reference `rhombic` v0.3.0.
- Evidence:
  - Paper 1 (line 120): "256 tests" (references v0.1.2)
  - Paper 2 (lines 63, 955): "256 tests" (references v0.3.0)
  - Paper 3 (line 1161): "312 tests" (references v0.3.0)
  - If both Paper 2 and Paper 3 reference v0.3.0, the test count should be the same for that version. Either Paper 2 is stale (should be 312) or Paper 3 is wrong. Given that Paper 3 was written later and likely reflects additional test additions, Paper 2's count is probably stale.
- Status: NEW

### F-3A-06: Narrative Arc Is Coherent (Paper 1 -> Paper 2 -> Paper 3)
- Severity: POLISH (positive finding)
- Location: All three papers
- Finding: The logical progression is sound. Paper 1 establishes uniform-weight topology comparison. Paper 2 introduces heterogeneous weights and discovers amplification through direction-pair structure. Paper 3 makes the direction-pair structure learnable by inserting it into a neural network adapter. The conceptual chain (topology -> weighted topology -> learnable bridge topology) is clear and well-motivated.
- Evidence:
  - Paper 1: "What happens when you compare cubic vs FCC?" Answer: FCC wins on all propagation metrics.
  - Paper 2: "What happens when edges have different weights?" Answer: FCC advantage amplifies, especially when weights align with direction pairs.
  - Paper 3: "What happens when the direction-pair coupling is learnable?" Answer: Cybernetic feedback discovers the 3-axis coordinate structure.
  - The only gap is that Paper 3 does not explicitly trace this chain (see F-3A-03).
- Status: NEW

### F-3A-07: Paper 3 Does Not Claim Results From Papers 1/2 That Don't Exist
- Severity: POLISH (positive finding)
- Location: Paper 3, Section 2.4 (line 327-331)
- Finding: Paper 3's only backward reference to the prior work is factually accurate. It cites Paper 1 for "30% shorter paths and 2.4x higher algebraic connectivity" -- both match Paper 1's abstract. Paper 3 does not claim any specific Paper 2 results (because it never references Paper 2, per F-3A-03).
- Status: NEW

### F-3A-08: Paper 1 Has No Forward References
- Severity: POLISH (positive finding)
- Location: Paper 1 (entire document)
- Finding: Paper 1 contains no forward references to Papers 2 or 3. This is correct -- Paper 1 was published first and is LOCKED. The bibliography (rhombic.bib) contains no entries for the companion papers.
- Status: NEW

### F-3A-09: "Five Tested Polytopes" vs "Five 24-Edge Graph Topologies" Wording
- Severity: POLISH
- Location: Paper 2, Contributions item 3 (line 195) vs Experiment 7 (line 665)
- Finding: Paper 2's Contributions section says "five tested 24-edge graph topologies" but one of the five is a random G(14,24) graph, which is not a "topology" in the geometric sense. Experiment 7's text and table correctly describe them as "five graphs with 24 edges." The Contributions wording slightly overstates geometric specificity.
- Evidence:
  - Line 195: "all five tested 24-edge graph topologies"
  - Table 6 (lines 677-688): Lists RD, Cuboctahedron, K_{4,6}, 3-regular, Random G(14,24). The random graph is not a "topology" in the polytope sense.
  - Line 665: "Five graphs with 24 edges" -- correct wording.
- Status: NEW

### F-3A-10: Paper 2 Baseline Fiedler Ratio Inconsistency Between Abstract and Table
- Severity: MINOR
- Location: Paper 2, abstract (line 46) vs. intro (line 83-84) vs. Table 1 (lines 387-399)
- Finding: Paper 2's abstract says Paper 1 established a "$2.3\times$ algebraic connectivity advantage." Paper 2's intro says "$2.3$--$2.5\times$." Paper 2's own Table 1 shows the uniform Fiedler ratios are 2.31 (scale 125), 2.55 (scale 1000), and 2.28 (scale 8000). The "2.3x" in the abstract appears to use the scale-125 value while ignoring the scale-1000 value of 2.55x. This matters because Paper 2's headline amplification is "from 2.3x to 6.1x" -- using the lower baseline makes the amplification sound larger. If using the scale-1000 baseline (2.55x), the amplification is from 2.55x to 6.11x, which is still substantial but less dramatic.
- Evidence:
  - Abstract (line 46): "$2.3\times$"
  - Introduction (line 83): "$2.3$--$2.5\times$"
  - The "2.3x to 6.1x" headline (line 188) implicitly uses 2.3x as baseline.
  - At matched scale (1000), the baseline is 2.55x, making it "2.55x to 6.11x" -- a 2.4x amplification factor, not a 2.7x amplification factor.
  - The difference is not dishonest (both numbers appear in the data), but using the lower bound for the baseline and the upper bound for the result maximizes the perceived effect.
- Status: NEW

### F-3A-11: Paper 3 Preliminary Experiments Reference Without Context
- Severity: MINOR
- Location: Paper 3, Introduction (lines 113-121) and Section 2.2 (lines 277-286)
- Finding: Paper 3 references "prior work" and "preliminary experiments" (exp1a-e) establishing that untrained bridges learn task-discriminative structure (84.5% SVM accuracy), but these results are not attributed to either Paper 1 or Paper 2. They appear to be unpublished results from the same research program. The reader cannot verify these claims from the referenced papers.
- Evidence:
  - Lines 113-117: "Prior work established that untrained bridges learn task-discriminative structure: a leave-one-out SVM on 28 query-projection bridges (1,008 parameters) classifies task type at 84.5% accuracy..."
  - Lines 277-280: "Preliminary experiments (Table 1, exp1a-e) established that the bridge learns task-discriminative structure under standard (non-cybernetic) training..."
  - These experiments appear in Paper 3's Table 1 (exp1a-e, Qwen 1.5B, no Steersman) but no external citation is given.
  - The phrase "prior work established" typically implies a published reference. Here it refers to experiments within the same paper. Consider revising to "preliminary experiments (Section X) established" or similar.
- Status: NEW

### F-3A-12: Paper 3 Missing "Weighted" Context From Paper 2
- Severity: MINOR
- Location: Paper 3, Section 2.4 (lines 327-331)
- Finding: Paper 3's description of the RD's properties omits the weighted topology results from Paper 2 that are most relevant to Paper 3's design motivation. Paper 3 says the RD provides "30% shorter paths and 2.4x higher algebraic connectivity" (Paper 1 uniform-weight results) but does not mention that structured weights amplify this to 6.1x (Paper 2's central finding). This is relevant because Paper 3's Steersman uses *structured* weights (the contrastive loss) to drive direction-pair coupling -- exactly the mechanism Paper 2 identified.
- Evidence:
  - Paper 3 line 327-331 cites only Paper 1 uniform-weight results.
  - Paper 2's amplification finding (2.3x to 6.1x under direction-based weighting) directly motivates why the 6-channel direction-pair decomposition is a good prior for Paper 3's bridge matrix.
  - This is related to F-3A-03 (Paper 3 not citing Paper 2) but focuses specifically on the missing motivational context.
- Status: NEW

---

## Summary Table

| ID | Title | Severity | Paper(s) | Status |
|----|-------|----------|----------|--------|
| F-3A-01 | Algebraic connectivity ratio inconsistency | MINOR | 1, 2, 3 | NEW |
| F-3A-02 | Summary table mixes scales without annotation | MAJOR | 2 | NEW |
| F-3A-03 | Paper 3 does not reference Paper 2 | MAJOR | 3 | NEW |
| F-3A-04 | Paper 2 Future Work -> Paper 3 link accurate | POLISH | 2, 3 | NEW |
| F-3A-05 | Test count inconsistency (256 vs 312) | MINOR | 2, 3 | NEW |
| F-3A-06 | Narrative arc is coherent | POLISH | 1, 2, 3 | NEW |
| F-3A-07 | Paper 3 backward references accurate | POLISH | 3 | NEW |
| F-3A-08 | Paper 1 has no forward references (correct) | POLISH | 1 | NEW |
| F-3A-09 | "Topologies" wording overstates for random graph | POLISH | 2 | NEW |
| F-3A-10 | Baseline Fiedler ratio cherry-picking | MINOR | 2 | NEW |
| F-3A-11 | "Prior work" references unpublished experiments | MINOR | 3 | NEW |
| F-3A-12 | Paper 3 missing weighted-topology context | MINOR | 3 | NEW |

**Severity summary:** 0 CRITICAL, 2 MAJOR, 5 MINOR, 5 POLISH (positive findings or cosmetic)

**Top priority fixes:**
1. **F-3A-03 (MAJOR):** Paper 3 should cite Paper 2 and trace the narrative chain: uniform topology advantage (Paper 1) -> amplification under structured weights via direction-pair decomposition (Paper 2) -> learnable direction-pair coupling via bridge matrix (Paper 3).
2. **F-3A-02 (MAJOR):** Paper 2's summary table (Table 7) needs scale annotations, or the consensus row should compare matched scales. Currently it compares 0.93x at scale 1000 with 6.69x at scale 125, which is misleading.
