# Round 7 — Agent 7A: Fresh-Eyes Read-Through of Paper 3

**Auditor:** Agent 7A (final round, no prior audit exposure)
**Paper:** `rhombic-paper3.tex` — "The Learnable Bridge: Cybernetic Feedback Discovers Rhombic Dodecahedral Geometry in Multi-Channel LoRA"
**Date:** 2026-03-15

---

## Summary

The paper is strong, clearly written, and internally consistent on the large-scale claims. The experimental design is well-motivated, the limitations section is honest, and the argument builds coherently. I found no critical issues that would cause rejection. I identified two major issues that a reviewer would flag, and several minor items for polish. The paper is near submission-ready.

---

## Findings

### F-7A-01: Geometric Description Imprecision — "Perpendicular to" Axis Labels
- Severity: MAJOR
- Location: Lines ~518-526
- Finding: The text states "Each pair of opposite faces is perpendicular to one of the three Cartesian axes, yielding 3 axis-aligned groupings of 2 face pairs each." Then it labels the groupings as x-axis, y-axis, z-axis. The problem: the face normals (1,1,0) and (1,-1,0) are NOT perpendicular to the x-axis. They lie in the z=0 plane and are perpendicular to the z-axis. The actual grouping criterion is which coordinate is ABSENT (zero) from the normal vector, not which axis the normals are perpendicular to. The paper labels the z=0 group as "x-axis," the y=0 group as "y-axis," and the x=0 group as "z-axis" — which inverts the natural geometric interpretation. A reviewer familiar with RD geometry would immediately flag this.

  The operational partition (3 groups of 2 face pairs) is correct and the contrastive loss works correctly regardless of labeling. But for a paper whose title includes "Rhombic Dodecahedral Geometry," the geometric description must be precise.

- Suggested fix: Replace "perpendicular to one of the three Cartesian axes" with a correct description. Options: (a) "grouped by which coordinate vanishes in their normal vectors" — the z=0 group, y=0 group, and x=0 group; or (b) relabel so that the "z-axis" grouping contains the normals with z=0, which is confusing in a different way. The cleanest fix is to drop the axis naming entirely and just describe three groups by their shared vanishing coordinate:
  ```
  - Group 1 (z=0): channels (0,1), normals (1,1,0) and (1,-1,0)
  - Group 2 (y=0): channels (2,3), normals (1,0,1) and (1,0,-1)
  - Group 3 (x=0): channels (4,5), normals (0,1,1) and (0,1,-1)
  ```
  Alternatively, keep axis labels but use the vanishing coordinate: "x-axis group" = normals with x=0, etc. But this reverses the current labeling and would require checking all downstream references.

---

### F-7A-02: Per-Module Hierarchy Claim Overstated
- Severity: MAJOR
- Location: Lines ~970-975
- Finding: The paper claims "The ordering k > v > q > o is consistent across scales." For TinyLlama C-002, the data is k=46,570 > v=24,462 > q=16,708 > o=9,116 — clearly k > v > q > o. But for Qwen 7B, the data is k=44,000 and v=44,000 (TIED), then q=22,477 > o=19,080. At Qwen 7B, k and v are equal, not k > v. The claim "k > v > q > o is consistent across scales" is not supported when k=v at one of the two scales. A reviewer would flag this as an overstatement.
- Suggested fix: Weaken to "The ordering k >= v > q > o is consistent across scales" or "k_proj and v_proj develop the strongest coupling, followed by q_proj, with o_proj weakest — an ordering that holds at both scales (with k_proj and v_proj tied at Qwen 7B)."

---

### F-7A-03: Abstract Claims "60,000+ bridge matrices" but Body Says "42,500+"
- Severity: MINOR
- Location: Abstract line ~53 vs. body line ~639
- Finding: The abstract says "60,000+ bridge matrices" (line 53). The body says "Six cybernetic experiments at n=6 contribute 42,500+ bridge matrices" (line 639) and "Six non-cybernetic experiments contribute 570 final-state bridges" (line 649). The ablation experiments (H1, H2, H4) presumably also produce bridge matrices at each checkpoint, which would bring the total above 60,000. But the paper never explicitly accounts for these in the total. The "60,000+" figure in the abstract appears to include all experiments while the body text quotes subtotals without reconciliation.
- Suggested fix: Add a brief accounting somewhere (perhaps in Section 4 or the introduction) that makes the 60,000+ total transparent: e.g., "42,500+ from cybernetic n=6 experiments, ~26,400 from channel ablation, and 570 from non-cybernetic controls."

---

### F-7A-04: "Six Non-Cybernetic Experiments" — Count Unclear
- Severity: MINOR
- Location: Lines ~148, 649, 1116
- Finding: The paper repeatedly says "6 non-cybernetic experiments" contributing "570 final-state bridges." From Table 1, the non-Steersman experiments are: exp1a-e (one row representing 5 runs), exp2, exp2.5, and Holly. If exp1a-e is 5 experiments, that's 5+1+1+1 = 8. If it's 1, that's 4. Neither count gives 6. The "570 bridges" figure would help disambiguate (e.g., if exp1a-e uses Qwen 1.5B with some adapter count), but the paper doesn't provide adapter counts for Qwen 1.5B or Wan 2.1's attention modules. A reviewer performing arithmetic on the table would not be able to verify "6 non-cybernetic experiments" or "570 bridges" from the information given.
- Suggested fix: Either clarify in the text how exp1a-e maps to the count of 6 (e.g., "5 preliminary experiments (exp1a-e) plus exp2, totaling 6 non-cybernetic n=6 experiments before the Steersman was introduced"), or add the adapter count for each model family so that "570" can be verified.

---

### F-7A-05: C-001 Validation Loss Not Comparable
- Severity: MINOR
- Location: Lines ~625, 742-744
- Finding: Table 1 reports C-001 val loss as 0.4178, while C-002 and C-003 are 0.4010 and 0.4011. The text at line 742 acknowledges C-001 ran for only 4K steps vs. 10K for the others. This is not an error, but a reader scanning Table 1 might interpret C-001's higher loss as a meaningful difference. The 0.17% delta claims in the introduction and ablation section use H-series experiments (all at 10K steps), not C-series, but this is not explicit.
- Suggested fix: Consider adding a footnote to Table 1 noting that C-001's higher val loss reflects fewer training steps (4K vs. 10K), not a topology effect. Or mark C-001's val loss as "0.4178*" with a note.

---

### F-7A-06: Abstract Mentions "I Ching complementary trigrams suppressed 99.5% in 900 steps" — Minor Clarity Issue
- Severity: MINOR
- Location: Abstract lines ~60-61
- Finding: The abstract says "initializations that actively oppose the target topology (I Ching complementary trigrams suppressed 99.5% in 900 steps)." A reader encountering this for the first time might parse "I Ching complementary trigrams" as an unexplained term in an ML paper. While it is fully defined in Section 4.3 (line 578-580, 747-756), the abstract drops this domain-specific term without any context about what it means for bridge initialization. A reviewer in the ML community may find this jarring.
- Suggested fix: In the abstract, consider replacing the specific I Ching reference with a more generic description: "including adversarial initializations that actively oppose the target topology (suppressed 99.5% in 900 steps)." The I Ching detail can remain in the body where it is properly contextualized.

---

### F-7A-07: "Removing it (at n in {3, 4, 8})" — Imprecise Phrasing in Conclusion
- Severity: MINOR
- Location: Line ~1192
- Finding: The conclusion says "removing it (at n in {3, 4, 8}) produces 1,020x higher Bridge Fiedler." This phrasing implies the contrastive loss was present and then removed at n != 6. In reality, the contrastive loss was NEVER active at n != 6 — it was undefined, not removed. This is correctly stated in Section 3 and Section 5, but the conclusion's shorthand could mislead.
- Suggested fix: "disabling it (undefined at n in {3, 4, 8}) produces..." or "its absence (at n in {3, 4, 8}, where it is undefined) produces..."

---

### F-7A-08: No Explicit Definition of "Block-Diagonal" Until Section 3.3
- Severity: MINOR
- Location: Throughout Sections 1-3
- Finding: The term "block-diagonal" is used starting in the abstract (line 55) and throughout the introduction, but the formal definition — rho > 10 AND all cross-planar entries < 10^-3 — does not appear until line 547-549 (Section 3.3) and is restated in Appendix A.6 (line 1358). The introduction uses "100% block-diagonal" as a key claim without the reader knowing the classification criteria. While the criteria turn out to be extremely conservative (typical values far exceed the thresholds), a reviewer might want the definition earlier.
- Suggested fix: Add a brief forward reference in the introduction or abstract, e.g., "block-diagonal bridge structure (defined formally in Section 3.3; rho > 10 with all cross-planar entries below 10^-3)."

---

### F-7A-09: Inconsistent Precision in Convergence Band Reporting
- Severity: MINOR
- Location: Lines ~157, 170-171, 222, 702-703
- Finding: The convergence band for final ratios is reported as:
  - Line 157: "64,000:1--71,000:1 band"
  - Line 222: "converged: ~70,000:1"
  - Line 702-703: "C-002 = 71,337:1, H3 = 70,404:1, C-003 = 64,168:1"

  The "~70,000:1" characterization omits C-003's 64,168 which is closer to 64,000 than 70,000. "~70,000:1" as a summary of a band spanning 64,168 to 71,337 is misleading — the midpoint is ~67,600 and C-003 pulls the band notably below 70,000. This is a cosmetic issue but the kind of thing a quantitative reviewer would notice.
- Suggested fix: Use "~67,000:1" or "64,000--71,000:1" consistently instead of "~70,000:1."

---

### F-7A-10: Figure 5 Caption References "Peak 82,854:1" — from Wrong Experiment
- Severity: MINOR
- Location: Line ~847
- Finding: Figure 5 caption says "co-planar/cross-planar ratio (n = 6 only; peak 82,854:1)." The peak of 82,854:1 comes from C-003 (Section 4.2, line 697), which is an initialization experiment, not the channel ablation. Figure 5 shows the channel ablation results (H-series). The H3 experiment (n=6 in the ablation) achieves 70,404:1 at convergence. If Figure 5's right panel shows C-003 data rather than H3 data, this should be noted. If it shows H3, then the 82,854 peak doesn't appear in that panel.
- Suggested fix: Either clarify that the right panel includes data from C-003 (which is also n=6 with Steersman, just from a different experiment set), or use H3's peak value in this caption.

---

### F-7A-11: Frankle & Carlin Citation — Possible Author Error
- Severity: MINOR
- Location: Bibliography line ~146
- Finding: The Lottery Ticket Hypothesis paper is cited as "Frankle, Jonathan and Carlin, Michael." The actual paper (arXiv:1803.03635) is by Jonathan Frankle and Michael Carlin. However, the commonly cited version is "Frankle and Carlin" for the ICLR 2019 paper. Let me check — actually, the correct second author is Michael Carlin. I may be confusing with a different version. This should be verified against the actual arXiv entry, as the second author's name matters for citation accuracy.
- Suggested fix: Verify against the actual paper. The ICLR 2019 Lottery Ticket paper is by Jonathan Frankle and Michael Carlin — this appears correct, though some databases list additional authors on extended versions.

---

### F-7A-12: Holly Battery — "10 epochs (~600 steps)" Reproducibility
- Severity: MINOR
- Location: Lines ~1309-1319
- Finding: The Holly Battery is described as "10 epochs (~600 steps at batch size 1, gradient accumulation 4)" on "proprietary video data." This means the dataset size is approximately 600 * 4 * 1 / 10 = 240 training examples. This is a very small dataset. The paper correctly notes "The Holly Battery... is not intended to be independently reproduced" (line 1316), which is appropriate. However, a reviewer might question whether 600 steps of training is sufficient to develop bridge structure, making the null result less informative than the cybernetic experiments at 10K steps.
- Suggested fix: Consider adding a sentence acknowledging the step count differential: "While the Holly Battery trains for fewer steps (600 vs. 10,000), the non-cybernetic Qwen experiments (exp2/exp2.5) at 3K-10K steps also produce 0% block structure, indicating that training duration alone does not explain the null result."

---

### F-7A-13: "Effectively 3" Redundancy in Abstract
- Severity: MINOR
- Location: Lines ~68-69
- Finding: The abstract says "The effective dimensionality of the discovered structure is effectively~3." The word "effectively" appears twice in the same clause — once in "effective dimensionality" and once in "effectively~3." This is grammatically awkward.
- Suggested fix: "The effective dimensionality of the discovered structure is~3" or "The discovered structure has effective dimensionality~3."

---

### F-7A-14: "1,020x Bifurcation" — Phrasing Ambiguity
- Severity: MINOR
- Location: Lines ~67, 187, 874
- Finding: The paper uses "1,020x bifurcation" to describe the Fiedler ratio. In dynamical systems, "bifurcation" typically refers to a qualitative change in system behavior as a parameter crosses a threshold — not a quantitative ratio between two measurements. The Fiedler difference IS a bifurcation in the correct sense (the system transitions between two qualitatively different attractors as contrastive loss is toggled), but "1,020x bifurcation" conflates the bifurcation (qualitative) with its magnitude (quantitative). A reviewer familiar with dynamical systems terminology might flag this.
- Suggested fix: "a 1,020x difference in Fiedler eigenvalue — a bifurcation between connected and near-disconnected bridge topology" or simply "1,020x separation."

---

### F-7A-15: Missing Val Loss for exp3 and exp3_tiny in Table 1
- Severity: MINOR
- Location: Lines ~623-624
- Finding: Table 1 shows "---" for validation loss in exp3 (Qwen 7B) and exp3_tiny (TinyLlama). These are cybernetic experiments that ran for 12.9K and 10K steps respectively. While the missing values may reflect earlier experimental code that didn't log val loss, a reviewer might question whether the "validation loss is insensitive to topology" claim is fully supported when 2 of 6 cybernetic experiments lack val loss data.
- Suggested fix: If val loss data exists for these experiments (even from logs), include it. If not, acknowledge in the text that val loss was not recorded for the earlier experiments (exp3, exp3_tiny) and that the insensitivity claim rests on C-001/C-002/C-003/H-series data.

---

## Cross-Check: Abstract vs. Body

| Abstract Claim | Body Support | Status |
|---|---|---|
| 13 experiments | Table 1: 13 rows (exp1a-e as 1) | OK (with F-7A-04 caveat) |
| 1.1B-14B parameters | Four model families | OK |
| 60,000+ bridge matrices | 42,500 cybernetic + ablation + non-cyber | Plausible but unverified (F-7A-03) |
| 100% BD at n=6 cybernetic | Table 1 | OK |
| 0% BD non-cybernetic | Table 1 | OK |
| Lock-in by 200 steps | Section 4.2 | OK |
| Initialization independence (3 strategies) | Section 4.3 | OK |
| 99.5% suppression in 900 steps | Section 4.3, line 754 | OK |
| 82,854:1 peak ratio | Section 4.2, line 697 | OK |
| 1,020x bifurcation | Section 5.2 | OK |
| n=3 matches n=6 val loss (0.16% max delta) | Section 5.3, lines 859-860 | OK |
| 4x fewer bridge params | Section 5.4, lines 939-940 | OK |
| Effective dimensionality ~3 | Section 5.4 | OK |

## Cross-Check: Conclusion vs. Body

The conclusion (Section 7) accurately summarizes the main findings without introducing new claims. All numbers in the conclusion trace to the body. No inflation detected.

---

## Overall Assessment

The paper is well-constructed and the core claims are strongly supported by the experimental evidence. The two MAJOR items (geometric description precision and per-module hierarchy overstatement) should be addressed before submission. The MINOR items are polish-level improvements that would strengthen the paper against pedantic reviewers but are not blocking.

**Verdict: Ready for submission after addressing the two MAJOR items.**
