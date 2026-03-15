# Agent 2A: Abstract-Body Consistency Audit — Paper 3

**Auditor:** Agent 2A (Abstract-Body Consistency)
**Paper:** rhombic-paper3.tex
**Date:** 2026-03-15
**Scope:** Every quantitative claim in abstract (lines 46–74) and introduction (lines 81–231); contribution-to-body tracing; discussion/conclusion back-references; experiment count resolution.

---

## Findings

### F-2A-01: Experiment Count — "15 experiments" Is Unverifiable
- **Severity:** MAJOR
- **Location:** Paper 3, Abstract line 52; Section 4.1 line 573
- **Finding:** The abstract and Section 4.1 both claim "15 experiments." Table 1 contains 13 rows. Row "exp1a--e" covers sub-experiments a through e. If exp1a–e = 5 distinct experiments, the total is 17. If exp1a–e = 1 experiment with 5 sub-runs, the total is 13. Neither interpretation yields 15. No counting methodology is stated anywhere in the paper.
- **Evidence:** Table 1 rows: exp1a–e, exp2, exp2.5, exp3, exp3_tiny, C-001, C-002, C-003, H3, H1, H2, H4, Holly = 13 rows. With exp1a–e expanded: 5+12 = 17.
- **Recommendation:** Either (a) adopt 17 (if exp1a–e = 5 distinct experiments), (b) adopt 13 (if exp1a–e is 1 experiment), or (c) adopt a defensible count and state the counting rule. The current "15" matches no obvious enumeration.

### F-2A-02: exp2.5 Shows 1% BD — Contradicts "0%" Claims
- **Severity:** MAJOR
- **Location:** Paper 3, Abstract line 57; Introduction line 147; Section 4.1 lines 618–619; Conclusion line 1126–1127
- **Finding:** Four locations claim that non-cybernetic training produces 0% block-diagonal structure. Table 1 (line 593) shows exp2.5 (Qwen 7B, geometric init, No Steersman, 3K steps) with BD% = 1%. This is a direct contradiction between the table data and the text.
- **Evidence:** Abstract: "non-cybernetic training produces 0%." Table 1: exp2.5 BD% = 1%. Body line 618–619: "Six non-cybernetic experiments contribute 570 final-state bridges. None exhibits block-diagonal structure."
- **Recommendation:** Either (a) change exp2.5 BD% to 0% in Table 1 if the 1% was a rounding artifact or classification error, with a footnote explaining; or (b) revise all four text locations to say "~0%" or "≤1%" and discuss the exp2.5 anomaly. Option (a) is preferable if the raw data supports it.

### F-2A-03: "Six Non-Cybernetic Experiments" Count Is Ambiguous
- **Severity:** MAJOR
- **Location:** Paper 3, Introduction lines 146–148; Section 4.1 line 618
- **Finding:** The text claims "6 non-cybernetic experiments." Table 1 non-cybernetic rows (Steersman = No): exp1a–e, exp2, exp2.5, Holly = 4 rows. If exp1a–e = 5 sub-experiments: 5+1+1+1 = 8. Neither gives 6. If H2 and H4 (Steersman = Yes but contrastive disabled, BD = 0%) are counted as "non-cybernetic" to reach 6, this conflates "non-cybernetic" with "non-contrastive," which is a terminological error given that H2/H4 DO have the Steersman active.
- **Evidence:** Table 1 Steersman=No rows: exp1a–e, exp2, exp2.5, Holly. H1/H2/H4 have Steersman=Yes. The text says "3 model scales" for non-cybernetic, which only fits {Qwen 1.5B, Qwen 7B, Wan 14B} = the Steersman=No experiments.
- **Recommendation:** Clarify the counting. If "non-cybernetic" means Steersman=No, the count depends on how exp1a–e is enumerated. State the rule explicitly.

### F-2A-04: Abstract Assigns Wrong Delta to n=3 vs n=6 Comparison
- **Severity:** MAJOR
- **Location:** Paper 3, Abstract lines 69–70
- **Finding:** The abstract says "$n = 3$ achieves identical validation loss to $n = 6$ (0.17\% maximum delta)." The 0.17% figure is the max delta across n ∈ {3, 4, 6} (driven by n=4 vs n=6). The body (line 822–823) gives the n=3 vs n=6 specific delta as 0.058%. The abstract attributes the wrong number to the wrong comparison.
- **Evidence:** Abstract: "0.17% maximum delta" for n=3 vs n=6. Body: "maximum delta of ~0.058%" for n=3 vs n=6. Table 2: n=3 loss = 0.4020, n=6 loss = 0.4015, delta = 0.125% at final step.
- **Recommendation:** Change the abstract to say either "0.058% maximum delta" (if using the body's trajectory figure) or "0.12% maximum delta" (if using the table's final-step values). See also F-2A-05.

### F-2A-05: The 0.058% Figure Is Internally Inconsistent
- **Severity:** MAJOR
- **Location:** Paper 3, Introduction line 193; Section 5.2.1 line 823; Section 5.4 line 905
- **Finding:** Three locations claim the maximum delta between n=3 and n=6 over 100 checkpoints is 0.058%. Table 2 shows final-step values of n=3: 0.4020 and n=6: 0.4015, giving a delta of (0.4020−0.4015)/0.4015 = 0.125%. The maximum delta over all checkpoints cannot be less than the delta at the final checkpoint. Therefore 0.058% is either (a) wrong, (b) computed as an absolute difference rather than a percentage, or (c) based on different data than what appears in Table 2.
- **Evidence:** 0.058% of 0.4015 = 0.000233. Actual difference = 0.0005. 0.0005/0.4015 = 0.1245%. Even 0.0005/0.4020 = 0.1244%. The 0.058% figure is approximately half the actual percentage delta at the final step.
- **Recommendation:** Re-derive the 0.058% from source data. If the figure is wrong, correct it in all three locations plus the abstract (per F-2A-04). This number is used to support a key contribution (effective dimensionality = 3).

### F-2A-06: Introduction Conflates Experiment Sets for Val Loss Range
- **Severity:** MINOR
- **Location:** Paper 3, Introduction lines 172–173
- **Finding:** The introduction states "Validation loss at 10,000 steps is 0.4010–0.4022 across channel counts n ∈ {3, 4, 6}." The value 0.4010 is C-002's val loss (geometric init, n=6). The channel ablation uses the H-series experiments, where H3 (n=6) has val loss 0.4015. The correct range from the ablation is 0.4015–0.4022, not 0.4010–0.4022.
- **Evidence:** Table 2 (ablation): n=6 val loss = 0.4015 (H3). Table 1: C-002 val loss = 0.4010 (not part of ablation). Intro uses 0.4010 as lower bound.
- **Recommendation:** Change "0.4010" to "0.4015" on line 172, or clarify that the range spans all n=6 experiments, not just the ablation set.

### F-2A-07: "570 Bridges" for Non-Cybernetic Experiments — Unverifiable
- **Severity:** MINOR
- **Location:** Paper 3, Introduction line 146; Section 4.1 line 618; Limitations line 1076
- **Finding:** Three locations cite "570 final-state bridges" for non-cybernetic experiments. The number cannot be reconstructed from the paper's data. Qwen 7B has 112 adapters (28 layers × 4), TinyLlama has 88 (22 × 4), Wan 2.1 has 160 (40 × 4). Qwen 1.5B's layer count is not stated. No combination of the non-cybernetic experiments' adapter counts sums to exactly 570 without knowing which experiments are included and the Qwen 1.5B architecture.
- **Evidence:** exp2 (112) + exp2.5 (112) + Holly (160) = 384. Adding exp1a–e as 1 run of unknown architecture: 384 + X = 570 → X = 186 → 186/4 = 46.5 layers (not an integer). Counting exp1a–e as 5 × Y: 5Y + 384 = 570 → Y = 37.2 (not an integer).
- **Recommendation:** Either derive the 570 explicitly (state which experiments contribute and their adapter counts) or verify the number against source data. The number appears in a claim of "0% block structure" which is falsified by exp2.5 = 1% anyway.

### F-2A-08: "60,000+ Bridge Matrices" — Abstract-Only, Never Derived in Body
- **Severity:** MINOR
- **Location:** Paper 3, Abstract line 53
- **Finding:** The abstract claims "60,000+ bridge matrices." The body provides "42,500+ bridge matrices" for cybernetic n=6 experiments (line 611) and "570 final-state bridges" for non-cybernetic experiments (line 618). The remaining ~17,000 would come from ablation experiments (H1–H4 checkpoints) and/or non-cybernetic checkpoint histories, but this accounting is never provided. The number is plausible but unsupported.
- **Evidence:** 42,500 (cybernetic n=6) + 570 (non-cybernetic final) = 43,070. Ablation H1–H4 at 100 checkpoints × 88 bridges each = 35,200. Total ≈ 78,270. So 60,000+ is plausible but the derivation never appears.
- **Recommendation:** Add a sentence to Section 4.1 accounting for the total: "Across all experiments and checkpoints, the study examines [N] bridge matrices: [breakdown]."

### F-2A-09: "Three Model Architectures" — Qwen 1.5B Omitted from List
- **Severity:** MINOR
- **Location:** Paper 3, Abstract line 53
- **Finding:** The abstract says "three model architectures (TinyLlama, Qwen 7B, Wan 2.1)." Table 1 includes exp1a–e on "Qwen 1.5B," which is a fourth model variant. While Qwen 1.5B and 7B share an architecture family, listing them as the same "architecture" while separately listing TinyLlama is inconsistent — TinyLlama is also a decoder-only transformer. The paper later (line 147) says "3 model scales" for non-cybernetic experiments, which works because those span Qwen 1.5B, Qwen 7B, and Wan 14B. But the abstract's "three model architectures" omits Qwen 1.5B.
- **Evidence:** Table 1 row 1: "Qwen 1.5B." Abstract: "(TinyLlama, Qwen 7B, Wan 2.1)."
- **Recommendation:** Either say "four model scales" and list all four, or say "three model families" with a note that Qwen appears at two scales.

### F-2A-10: Introduction "0.12%" for Init-Convergence Val Loss Is Slightly Understated
- **Severity:** POLISH
- **Location:** Paper 3, Introduction line 155
- **Finding:** "validation loss within 0.12% of each other" for the three initialization strategies. The three 10K-step val losses are C-002 = 0.4010, C-003 = 0.4011, H3 = 0.4015 (C-001 at 4K steps is excluded from this comparison). Max delta = (0.4015 − 0.4010)/0.4010 = 0.1246%. "0.12%" understates this by 0.005 percentage points.
- **Evidence:** (0.4015 − 0.4010)/0.4010 = 0.001246 = 0.1246%.
- **Recommendation:** Change to "0.13%" or "~0.12%." This is cosmetic but the direction of the error (understating the gap) is the wrong direction for scientific claims.

### F-2A-11: Abstract's "82,854:1" Ratio — Sourced from Peak, Not Final
- **Severity:** POLISH
- **Location:** Paper 3, Abstract line 64
- **Finding:** The abstract says "the co-planar/cross-planar coupling ratio reaches 82,854:1." This is the peak value (C-003 at step 9,000), not the converged value. The body (line 647, 666) correctly identifies this as a peak. The abstract phrasing "reaches" is technically correct (it does reach this value) but could mislead readers into thinking this is the steady-state. The converged band is 64,000:1–71,000:1.
- **Evidence:** Abstract: "reaches 82,854:1." Body line 647: "Peak: 82,854:1 (C-003, step 9,000)." Body line 670–672: converged band 64,168:1–71,337:1.
- **Recommendation:** Consider adding "peak" before "82,854:1" in the abstract, or reporting the converged band instead. The current phrasing is defensible but potentially misleading.

### F-2A-12: Table 1 and Table 2 Inconsistency for H4 Val Loss
- **Severity:** POLISH
- **Location:** Paper 3, Table 1 line 604; Table 2 line 801
- **Finding:** Table 1 shows H4 (n=8) val loss as "---" (missing). Table 2 (ablation) shows n=8 val loss as 0.4024. The same experiment's val loss should appear consistently.
- **Evidence:** Table 1 line 604: "H4 & TinyLlama & 8 & identity & Yes & 10K & --- ". Table 2 line 801: "8 & 64 & 0.4024."
- **Recommendation:** Add 0.4024 to Table 1 for H4. If the value was unavailable at the time Table 1 was composed but recovered for Table 2, both tables should now reflect the same data.

### F-2A-13: Table 1 Missing Val Loss for exp3, exp3_tiny
- **Severity:** POLISH
- **Location:** Paper 3, Table 1 lines 595–596
- **Finding:** exp3 (Qwen 7B, cybernetic) and exp3_tiny (TinyLlama, cybernetic) both show "---" for val loss. These are key experiments (first cybernetic successes at both scales). The missing data weakens the "task performance is orthogonal" claim for the original cybernetic experiments.
- **Evidence:** Table 1: exp3 val loss = "---", exp3_tiny val loss = "---".
- **Recommendation:** Fill in the values if available. If unavailable (earlier code version didn't log val loss), add a footnote explaining the gap.

### F-2A-14: Conclusion Repeats "~0%" Claim Despite Table 1 Showing 1%
- **Severity:** MINOR (duplicate of F-2A-02 in a different location)
- **Location:** Paper 3, Conclusion line 1126–1127
- **Finding:** "six non-cybernetic controls produce ~0%." Same issue as F-2A-02. The tilde softens the claim slightly but Table 1 still shows exp2.5 = 1%.
- **Evidence:** Conclusion: "produce~0\%." Table 1: exp2.5 BD% = 1%.
- **Recommendation:** See F-2A-02 recommendation.

### F-2A-15: Contribution 3 Reports Peak Ratio Without Context
- **Severity:** POLISH
- **Location:** Paper 3, Introduction line 219
- **Finding:** Contribution 3 lists "Peak ratio: 82,854:1" without noting this is a transient peak, not the converged value. The body discusses this distinction (lines 666–668) but the contribution list does not.
- **Evidence:** Introduction line 219: "Peak ratio: 82,854:1." Body line 670–672: converged band 64,168:1–71,337:1.
- **Recommendation:** Add "(converged: ~70,000:1)" after the peak ratio, or replace the peak with the converged band.

---

## Deferred Finding Resolution: F-01-14

**Original question:** The abstract says "15 experiments" — what is the correct count?

**Analysis:** Table 1 contains 13 rows. Row "exp1a--e" aggregates 5 sub-experiments (a through e). Expanding gives 17 distinct experiment IDs: exp1a, exp1b, exp1c, exp1d, exp1e, exp2, exp2.5, exp3, exp3_tiny, C-001, C-002, C-003, H1, H2, H3, H4, Holly.

No counting rule produces exactly 15:
- 13 rows as-is = 13
- exp1a–e expanded = 17
- Excluding exp1a–e entirely + counting exp2.5 = 12
- Any selection that yields 15 requires an arbitrary and unstated exclusion

**Recommendation:** The correct count is either **13** (table rows, with exp1a–e as one entry) or **17** (all distinct experiment IDs). Given that the paper treats exp1a–e as a single row in the table and discusses it as a group, **13** is the more natural number. If the authors prefer to count sub-experiments, **17** is defensible. Neither 13 nor 17 equals 15. The abstract must be corrected.

---

## Contribution-to-Body Tracing

All 6 contributions listed in the introduction (lines 208–231) have corresponding body sections:

| # | Contribution | Body Section |
|---|-------------|-------------|
| 1 | Block-diagonal emergence | Section 4.1 (Core Finding) |
| 2 | Initialization independence | Section 4.3 |
| 3 | Coupling dynamics | Section 4.2 (Lock-in Dynamics) |
| 4 | Channel count as effective rank | Section 5.4 (Effective Rank) |
| 5 | Contrastive loss mechanism | Section 5.2.2 |
| 6 | Scale invariance | Section 6 |

**All contributions have body evidence. No orphaned claims.**

---

## Discussion/Conclusion Back-References

The Discussion (Section 7) and Conclusion (Section 8) reference:
- 6 cybernetic experiments, 42,500+ bridges → Section 4.1 ✓
- 6 non-cybernetic controls, ~0% → Section 4.1 (but see F-2A-02)
- 0.17% max delta → Section 5.2.1 ✓
- 1,020× bifurcation → Section 5.2.2 ✓
- 3 effective dimensions → Section 5.4 ✓
- 123-step half-life → Section 4.2 ✓
- Attractor dynamics (asymmetric: 3 attractive, 12 repulsive) → Novel in Discussion, supported by body data ✓

**No conclusion claims lack body support, aside from the systemic "0%" issue.**

---

## Body Findings Not in Abstract

1. **Per-module coupling hierarchy** (k > v > q > o) — consistent across scales, potentially novel. Omission is defensible; this is a secondary finding.
2. **Polarity freedom** — 50/50 positive/negative split within co-planar blocks. Interesting mechanistic detail, not abstract-level.
3. **Holly Battery performance** (3.8% lower loss, 9.15 GB less VRAM, 6% faster) — mentioned in intro but absent from abstract. Could strengthen practical relevance framing.
4. **Three-phase Fiedler trajectory at n=4** — mechanistic detail, not abstract-level.

**Recommendation:** Consider adding a sentence about the Holly Battery performance benefits to the abstract. The current abstract focuses entirely on structural findings; a brief note that multi-channel LoRA also delivers practical benefits (even without the Steersman) would strengthen the paper's relevance to practitioners.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| MAJOR | 5 (F-2A-01 through F-2A-05) |
| MINOR | 5 (F-2A-06 through F-2A-09, F-2A-14) |
| POLISH | 5 (F-2A-10 through F-2A-13, F-2A-15) |
| **Total** | **15** |

**Claims checked:** 28 distinct quantitative claims traced from abstract/introduction to body.
**Discrepancies found:** 15 (5 MAJOR, 5 MINOR, 5 POLISH).

The two most urgent issues are:
1. **F-2A-05** (0.058% figure is internally inconsistent with Table 2 data) — this undermines a key contribution claim.
2. **F-2A-02** (exp2.5 = 1% BD contradicts the paper's central "100% vs 0%" narrative) — this needs resolution before submission.

The experiment count (F-2A-01) is straightforward to fix once the counting rule is decided.
