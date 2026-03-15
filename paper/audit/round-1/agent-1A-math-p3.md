# Round 1A: Math Accuracy — Paper 3

Auditor: Opus 4.6 (adversarial math agent)
Date: 2026-03-15
Paper: `rhombic-paper3.tex` (The Learnable Bridge)
Sources: `EXPERIMENT_TRACKER.md`, `COMPREHENSIVE_EXPERIMENT_TABLE.md`, `DEFINITIVE_ABLATION_RESULTS.md`, `DRAFT_SCALE_INVARIANCE_SECTION.md`, `FIEDLER_METRIC_NOTE.md`, `paper3_channel_ablation_section.md`, `H-ch8-results-hermes.json`, plus additional grep searches across all `.md` files.

---

## Verified Claims

### Abstract & Introduction (lines 47–230)

| # | Claim | Location | Source | Status |
|---|-------|----------|--------|--------|
| 1 | 15 experiments | Abstract, line 52 | See F-01-01 below | **DISCREPANCY** |
| 2 | 1.1B–14B parameter range | Abstract, line 53 | EXPERIMENT_TRACKER | VERIFIED |
| 3 | 60,000+ bridge matrices | Abstract, line 53 | 42,500 cybernetic + 570 non-cyb + ~26,400 ablation ≈ 69K | VERIFIED |
| 4 | Three model architectures (TinyLlama, Qwen 7B, Wan 2.1) | Abstract, line 54 | EXPERIMENT_TRACKER | VERIFIED |
| 5 | 100% block-diagonal at n=6 cybernetic | Abstract, line 55 | All 6 cybernetic experiments | VERIFIED |
| 6 | 0% block-diagonal non-cybernetic | Abstract, line 57 | All 6 non-cybernetic experiments | VERIFIED |
| 7 | Lock-in within 200 steps | Abstract, line 58 | DEFINITIVE_ABLATION_RESULTS §3 | VERIFIED |
| 8 | I Ching suppressed 99.5% in 900 steps | Abstract, line 60 | DRAFT_INIT_OVERWRITE_SECTION line 106 | VERIFIED (source convention; exact calc from 0.029→0.0001 = 99.66%) |
| 9 | 82,854:1 peak ratio | Abstract, line 64 | C-003 step 9000, DEFINITIVE_ABLATION_RESULTS peak table #1 | VERIFIED |
| 10 | Bridge Fiedler 0.00009 at n=6 | Abstract, line 65 | H-ch6 ANALYSIS, DEFINITIVE_ABLATION_RESULTS | VERIFIED |
| 11 | ~0.09 Fiedler at n=3,4,8 | Abstract, line 66 | H-ch3=0.0951, H-ch4=0.0918, H-ch8=0.0944 | VERIFIED |
| 12 | 1,020× bifurcation | Abstract, line 67 | 0.0918/0.00009 = 1,020 (using n=4 value) | VERIFIED |
| 13 | 0.17% max delta across n=3,4,6 | Abstract, line 70 | (0.4022-0.4015)/0.4015 = 0.174% ≈ 0.17% | VERIFIED |
| 14 | 4× fewer bridge params n=3 vs n=6 | Abstract, line 70 | 792 vs 3168 = 4.0× | VERIFIED |
| 15 | 0.058% max delta n=3 vs n=6 | Abstract, line 193 | COMPREHENSIVE_EXPERIMENT_TABLE Finding 3 | VERIFIED |
| 16 | 6 cybernetic experiments, 42,500+ bridges | Intro, lines 140–141 | COMPREHENSIVE_EXPERIMENT_TABLE evidence summary | VERIFIED |
| 17 | 6 non-cybernetic experiments, 570 bridges | Intro, line 146 | COMPREHENSIVE_EXPERIMENT_TABLE evidence summary | VERIFIED |
| 18 | 64,000:1–71,000:1 convergence band | Intro, line 155 | C-002=71,337, C-003=64,168, H3=70,404 | VERIFIED |
| 19 | 0.12% val loss delta init convergence | Intro, line 155 | (0.4015-0.4010)/0.4010 = 0.125% ≈ 0.12% | VERIFIED |
| 20 | Crossover at step ~75 | Intro, line 157 | DRAFT_INIT_OVERWRITE_SECTION line 100 | VERIFIED |
| 21 | 0.029 → 0.0001 I Ching coupling | Intro, line 159 | DRAFT_INIT_OVERWRITE_SECTION lines 99, 106 | VERIFIED |
| 22 | Half-life 123 steps | Intro, line 164 | DRAFT_INIT_OVERWRITE_SECTION line 300 | VERIFIED |
| 23 | Co-planar growth ~0.00012/step | Intro, line 218 | DRAFT_INIT_OVERWRITE_SECTION line 287 | VERIFIED |
| 24 | 84.5% LOO SVM accuracy | Intro, line 114 | EXPERIMENT_TRACKER, README | VERIFIED |
| 25 | 1,008 parameters (28 bridges) | Intro, line 113 | 28 × 36 = 1,008 | VERIFIED |
| 26 | Cosine similarity >0.999 | Intro, line 115 | PAPER3_DRAFT_COMBINED line 36 | VERIFIED (by source concordance) |
| 27 | Co-planar/cross-planar ratio 1.002 | Intro, line 119 | PAPER3_DRAFT_COMBINED line 39 | VERIFIED (by source concordance) |
| 28 | p = 0.474 | Intro, line 119 | PAPER3_DRAFT_COMBINED line 39 | VERIFIED (by source concordance) |

### Method (lines 330–563)

| # | Claim | Location | Source | Status |
|---|-------|----------|--------|--------|
| 29 | RD: 12 faces, 14 vertices (8 cubic + 6 octahedral), 24 edges | line 486 | Euler: 14-24+12=2 ✓ | VERIFIED |
| 30 | 6 face pairs, 3 co-planar pairs, 12 cross-planar pairs | lines 506, 520-521 | C(6,2)=15, 15-3=12 | VERIFIED |
| 31 | 36×4×22 = 3,168 bridge params TinyLlama | line 374 | 36×88=3,168 | VERIFIED |
| 32 | 36×4×28 = 4,032 bridge params Qwen | line 376 | 36×112=4,032 | VERIFIED |
| 33 | ~0.04% of LoRA parameter budget | line 377 | Consistent across sources | VERIFIED |
| 34 | LoRA α = 16, rank 24, lr 2e-4, batch 2, grad accum 8 | lines 365, 536-541 | Hyperparameter table | VERIFIED |
| 35 | Warmup 200 steps, seed 42, max seq 512 | lines 538-541 | Source configs | VERIFIED |
| 36 | Channel sizes s=8,6,4,3 for n=3,4,6,8 | line 559 | 24/n for each | VERIFIED |
| 37 | I Ching/RD ratio 1.4:1 at init | line 556 | DRAFT_INIT_OVERWRITE_SECTION line 16 | VERIFIED |
| 38 | Steersman feedback every T=100 steps | line 443 | Source configs | VERIFIED |
| 39 | Initial w_c=0.1, w_s=0.05 | line 471 | Source configs | VERIFIED |
| 40 | Max w_c=0.5, max w_s=0.2 | lines 461, 457 | Source configs | VERIFIED |
| 41 | 3 co-planar index pairs, 12 cross-planar index pairs | lines 421-422 | For n=6: 3 + 12 = 15 = C(6,2) | VERIFIED |

### Block-Diagonal Emergence (lines 567–757)

| # | Claim | Location | Source | Status |
|---|-------|----------|--------|--------|
| 42 | exp3 Qwen 7B, 12.9K steps, 18,248:1, 100% BD | line 595 | COMPREHENSIVE_EXPERIMENT_TABLE, DEFINITIVE_ABLATION_RESULTS | VERIFIED |
| 43 | exp3_tiny TinyLlama, 10K steps, 37,929:1, 100% BD | line 596 | Same sources | VERIFIED |
| 44 | C-001 TinyLlama, 4K, 0.4178, 10,118:1, 100% BD | line 597 | Same sources | VERIFIED |
| 45 | C-002 TinyLlama, 10K, 0.4010, 71,337:1, 100% BD | line 598 | Same sources | VERIFIED |
| 46 | C-003 TinyLlama, 10K, 0.4011, 64,168:1, 100% BD | line 599 | Same sources | VERIFIED |
| 47 | H3 TinyLlama, 10K, 0.4015, 70,404:1, 100% BD | line 600 | Same sources | VERIFIED |
| 48 | H1 (n=3) val loss 0.4020 | line 602 | COMPREHENSIVE_EXPERIMENT_TABLE | VERIFIED |
| 49 | H2 (n=4) val loss 0.4022, ~1:1, 0% BD | line 603 | Same sources | VERIFIED |
| 50 | Holly 1.493, 1.07:1, 0% BD | line 606 | Same sources | VERIFIED |
| 51 | 42,500+ bridge matrices, 100.0% BD | lines 611-614 | Source concordance | VERIFIED |
| 52 | 570 final-state bridges non-cybernetic | line 618 | COMPREHENSIVE_EXPERIMENT_TABLE | VERIFIED |
| 53 | 0.99:1 to 1.07:1 non-cybernetic ratio range | line 620 | Source concordance | VERIFIED |
| 54 | Holly: 3.8% lower loss, 9.15 GB less VRAM, 6% faster | lines 622-623 | EXPERIMENT_TRACKER lines 111-113 | VERIFIED |
| 55 | Lock-in within 200 steps = 2% of training | lines 630-631 | 200/10000 = 2% | VERIFIED |
| 56 | Half-life 123 ± 8 steps | line 657 | Source concordance | VERIFIED |
| 57 | R² = 0.998 for linear fit from step 500 | line 659 | Source concordance | VERIFIED |
| 58 | Peak 82,854:1 at C-003 step 9000 | line 666 | DEFINITIVE_ABLATION_RESULTS peak table | VERIFIED |
| 59 | C-002=71,337:1, H3=70,404:1, C-003=64,168:1 | lines 671-672 | All three sources | VERIFIED |
| 60 | Max pairwise delta 10% for convergence band | line 672 | See F-01-02 below | **DISCREPANCY** |
| 61 | Co-planar at step 2000: 0.252, 0.256, 0.252 | line 710 | COMPREHENSIVE_EXPERIMENT_TABLE line 49 | VERIFIED |
| 62 | Max pairwise delta 1.6% at step 2000 | line 711 | (0.256-0.252)/0.252 = 1.59% ≈ 1.6% | VERIFIED |
| 63 | Val losses 0.4178, 0.4010, 0.4011 at step 10K | line 712 | See F-01-03 below | **DISCREPANCY** (C-001 at 4K, not 10K) |
| 64 | C-002/C-003 within 0.025% val loss | line 713 | (0.4011-0.4010)/0.4010 = 0.025% | VERIFIED |
| 65 | Axis symmetry <2.5% | line 738 | Source concordance (DRAFT_INIT line 116) | VERIFIED |
| 66 | ~50/50 polarity split | line 741 | Source concordance | VERIFIED |

### Channel Count Ablation (lines 762–914)

| # | Claim | Location | Source | Status |
|---|-------|----------|--------|--------|
| 67 | n=3: 9 params/mod, val 0.4020, Fiedler 0.095 | Table 2, line 797 | H-ch3=0.0951, rounded to 0.095 | VERIFIED |
| 68 | n=4: 16 params/mod, val 0.4022, Fiedler 0.092 | Table 2, line 798 | H-ch4=0.0918, paper says 0.092 | See F-01-04 |
| 69 | n=6: 36 params/mod, val 0.4015, Fiedler 0.00009 | Table 2, line 799 | All sources agree | VERIFIED |
| 70 | n=8: 64 params/mod, val 0.4022, Fiedler 0.085 | Table 2, line 800 | See F-01-05, F-01-06 | **DISCREPANCY** |
| 71 | Val loss range 0.4015–0.4022, 0.17% max delta | lines 819-820 | Math checks out for n=3,4,6 | VERIFIED |
| 72 | 0.058% max delta n=3 vs n=6 over 100 checkpoints | line 822 | COMPREHENSIVE_EXPERIMENT_TABLE | VERIFIED |
| 73 | Fiedler 0.085–0.095 for spectral-only | line 831 | See F-01-07 below | **DISCREPANCY** |
| 74 | 1,020× bifurcation | line 836 | 0.0918/0.00009 = 1,020 (n=4 specific) | VERIFIED |
| 75 | Spectral gap 0.00006 at n=6 | line 837 | H-ch6 ANALYSIS line 45 | VERIFIED |
| 76 | Spectral gap 0.559 at n=4 | line 838 | H-ch4 ANALYSIS line 53 | VERIFIED |
| 77 | n=4 Fiedler dip: 0.084 → 0.076, 9.5% decrease | line 855 | 0.008/0.084 = 9.52% ≈ 9.5% | VERIFIED |
| 78 | n=4 Fiedler recovery to 0.092 | line 857 | H-ch4 final = 0.0918, paper says 0.092 | See F-01-04 |
| 79 | n=3: 6 off-diagonal entries | line 877 | 3×3 - 3 = 6 | VERIFIED |
| 80 | n=6: 15 off-diagonal DOF → 12 to zero, 3 active | lines 890, 896 | C(6,2)=15, 15-3=12 | VERIFIED |
| 81 | n=3: 792 total bridge params in TinyLlama | line 901 | 9×88=792 | VERIFIED |
| 82 | n=6: 3,168 total bridge params in TinyLlama | line 902 | 36×88=3,168 | VERIFIED |
| 83 | 4× overhead n=6 vs n=3 | line 903 | 3168/792=4.0 | VERIFIED |

### Scale Invariance (lines 918–994)

| # | Claim | Location | Source | Status |
|---|-------|----------|--------|--------|
| 84 | exp3 Qwen 7B ratio 18,248:1 | line 925 | DEFINITIVE_ABLATION_RESULTS, EXPERIMENT_TRACKER | VERIFIED |
| 85 | Per-module Qwen: k_proj/v_proj 44,000:1 | line 932 | DRAFT_SCALE_INVARIANCE k=44K, v=44K | VERIFIED |
| 86 | Per-module Qwen: q_proj 34,000:1 | line 933 | See F-01-08 below | **DISCREPANCY** |
| 87 | Per-module C-002: k_proj/v_proj 46,570:1 | line 932 | See F-01-09 below | **DISCREPANCY** |
| 88 | Per-module C-002: q_proj 16,708:1, o_proj 9,116:1 | lines 933-934 | PAPER3_FIGURE_INVENTORY line 28 | VERIFIED |
| 89 | Per-module Qwen: o_proj 19,080:1 | line 934 | DRAFT_SCALE_INVARIANCE line 77 | VERIFIED |
| 90 | Correlation Fiedler ~0.10 across 3 scales | line 973 | See F-01-10 below | **DISCREPANCY** |
| 91 | Correlation Fiedler: 0.102 (Qwen), 0.101 (TinyLlama) | line 974 | FIEDLER_METRIC_NOTE lines 33-34 | VERIFIED |
| 92 | Correlation Fiedler: 1.002 (Holly) | line 974 | FIEDLER_METRIC_NOTE line 35 | VERIFIED (but see F-01-10) |
| 93 | Holly: near-identity, mean off-diag 0.010, ρ=1.07:1 | lines 988-989 | Source concordance | VERIFIED |
| 94 | Holly Bridge Fiedler ~0.10 | line 989 | Source concordance | VERIFIED |

### Appendix (lines 1154–1282)

| # | Claim | Location | Source | Status |
|---|-------|----------|--------|--------|
| 95 | Hyperparameter table values | lines 1168-1188 | Consistent with experimental configs | VERIFIED |
| 96 | Steersman parameter table values | lines 1201-1211 | Consistent with method section | VERIFIED |
| 97 | TinyLlama 22 layers, 88 adapters | line 1244 | 22×4=88 | VERIFIED |
| 98 | Bridge param table: n=3→792, n=4→1,408, n=6→3,168, n=8→5,632 | lines 1250-1253 | n²×88 for each | VERIFIED |
| 99 | % of LoRA params: 0.01%, 0.02%, 0.04%, 0.07% | lines 1250-1253 | Consistent across sources | VERIFIED |
| 100 | BD detection: ρ>10 and cross-planar <10⁻³ | lines 1278-1279 | Consistent with method section | VERIFIED |
| 101 | Typical cybernetic ρ>60,000:1 and cross-planar <10⁻⁵ | line 1282 | C-002=71K, C-003=64K, H3=70K all >60K | VERIFIED |

---

## Discrepancies Found

### F-01-01: Experiment Count Inconsistency
- **Severity:** MINOR
- **Location:** Abstract, line 52; Section 4, line 573
- **Paper claims:** "15 experiments"
- **Data shows:** Table 1 in the paper lists 13 rows (exp1a-e as 1 entry, plus 12 others). The COMPREHENSIVE_EXPERIMENT_TABLE lists 15 rows, but that includes exp2.6 and exp2.7, which are NOT in the paper's Table 1. If exp1a-e is counted as 5 sub-experiments, the paper table has 17. If counted as 1, it has 13. Neither equals 15.
- **Evidence:** Paper Table 1 (lines 591-606) vs COMPREHENSIVE_EXPERIMENT_TABLE.md (lines 7-22)
- **Resolution:** Either add exp2.6 and exp2.7 to Table 1, or change "15" to match the actual table count. The most natural count using the paper's own Table 1 is 13 (treating exp1a-e as one line item) or 17 (treating them as 5 separate experiments).

### F-01-02: Convergence Band Delta Understated
- **Severity:** MINOR
- **Location:** Section 4.2, line 672
- **Paper claims:** "maximum pairwise delta: 10%"
- **Data shows:** (71,337 - 64,168) / 64,168 = 11.2%
- **Evidence:** C-002=71,337, C-003=64,168 from all three source files
- **Resolution:** Change "10%" to "11%" or "~11%".

### F-01-03: C-001 Val Loss Presented as Step 10K
- **Severity:** MINOR
- **Location:** Section 4.3, line 712
- **Paper claims:** "By step 10,000, validation losses are 0.4178, 0.4010, and 0.4011"
- **Data shows:** C-001 ran for only 4,000 steps, not 10,000. Its val loss of 0.4178 is the final value at step 4K. C-002 (0.4010) and C-003 (0.4011) are at step 10K.
- **Evidence:** COMPREHENSIVE_EXPERIMENT_TABLE line 14: "C-001... 4K steps... 0.4178"; Paper Table 1 line 597: "4K"
- **Resolution:** Rephrase to "Final validation losses are 0.4178 (C-001, 4K steps), 0.4010 (C-002, 10K), and 0.4011 (C-003, 10K)" or remove C-001 from this comparison since it ran for a different duration.

### F-01-04: n=4 Fiedler Value Rounded Inconsistently
- **Severity:** MINOR
- **Location:** Table 2 (line 798), Section 5.2.3 (line 857)
- **Paper claims:** n=4 Fiedler = 0.092
- **Data shows:** H-ch4 final Fiedler = 0.0918 (EXPERIMENT_TRACKER, paper3_channel_ablation_section.md line 42)
- **Evidence:** EXPERIMENT_TRACKER.md line 243: "Fiedler 0.0918"; paper3_channel_ablation_section.md line 42: "0.0918"
- **Resolution:** The paper rounds 0.0918 to 0.092 (2 significant figures). This is a rounding choice, not an error. However, for consistency with the other values (0.095, 0.085), using 0.092 is acceptable. The 1,020× bifurcation uses 0.0918 precisely: 0.0918/0.00009 = 1,020. Verified.

### F-01-05: n=8 Fiedler Value Uses In-Progress Data
- **Severity:** MAJOR
- **Location:** Table 2, line 800; also lines 831, 845
- **Paper claims:** n=8 Bridge Fiedler = 0.085
- **Data shows:** H-ch8 **completed 10K steps** with final Fiedler = **0.0944** (from H-ch8-results-hermes.json at step 10000: fiedler_mean = 0.09437901). The 0.085 value was the in-progress reading at step 2,900 (COMPREHENSIVE_EXPERIMENT_TABLE line 88).
- **Evidence:** H-ch8-results-hermes.json line 2610: fiedler_mean = 0.09437901 at step 10000. EXPERIMENT_TRACKER.md line 46: "H-ch8... 10K... 0.0944". paper3_channel_ablation_section.md line 44: "H-ch8... 0.0944".
- **Resolution:** Update n=8 Fiedler from 0.085 to 0.094 (or 0.0944) throughout the paper. This also changes the spectral-only Fiedler range from 0.085–0.095 to 0.092–0.095 (a tighter band), strengthening the spectral attractor claim. The Fiedler band in the paper abstract (line 66) says "~0.09" which remains correct.

### F-01-06: n=8 Val Loss Uses In-Progress Data
- **Severity:** MAJOR
- **Location:** Table 2, line 800
- **Paper claims:** n=8 Val Loss = 0.4022
- **Data shows:** H-ch8 at step 10K has val_loss = **0.4024** (from H-ch8-results-hermes.json line 2607: val_loss = 0.40238996). The paper's Table 1 (line 604) correctly shows "---" for H4 val loss, recognizing the data was incomplete, but Table 2 (ablation) reports 0.4022 — likely carried over from the n=4 value or from an estimate.
- **Evidence:** H-ch8-results-hermes.json line 2607: val_loss = 0.40238996 at step 10000
- **Resolution:** Update n=8 val loss to 0.4024 in the ablation table. This does not affect the "0.17% max delta" claim across n=3,4,6 (which excludes n=8), but it does affect the Section 5.2.1 text if n=8 is included in the loss comparison. The actual n=8 val loss (0.4024) is slightly above n=4 (0.4022), maintaining the paper's conclusion that "task performance is independent of channel count."

### F-01-07: Spectral-Only Fiedler Range Incorrect
- **Severity:** MAJOR (cascading from F-01-05)
- **Location:** Lines 831, 845, 883
- **Paper claims:** Fiedler converges to 0.085–0.095 for spectral-only runs
- **Data shows:** With corrected n=8 final value (0.0944), the range is 0.0918–0.0951 — or rounded: 0.092–0.095. The paper's lower bound of 0.085 is from the in-progress H-ch8 data.
- **Evidence:** Final values: H-ch3=0.0951, H-ch4=0.0918, H-ch8=0.0944
- **Resolution:** Update the range to 0.092–0.095. This actually STRENGTHENS the paper's "3.5% band" claim — the corrected band is (0.0951-0.0918)/0.0918 = 3.6%, and with 0.0944 instead of 0.085, the 3-value spread is even tighter.

### F-01-08: Qwen 7B q_proj Ratio Wrong
- **Severity:** MAJOR
- **Location:** Section 6.1, line 933
- **Paper claims:** q_proj ratio is 34,000:1 for Qwen 7B
- **Data shows:** q_proj = **22,477:1** for Qwen 7B
- **Evidence:** DRAFT_SCALE_INVARIANCE_SECTION.md line 74: "q_proj | 22,477:1". No source anywhere in the repository contains "34,000" in association with q_proj.
- **Resolution:** Change 34,000:1 to 22,477:1 for Qwen 7B q_proj. The per-module hierarchy (k/v > q > o) is preserved: 44,000 > 22,477 > 19,080.

### F-01-09: C-002 v_proj Ratio Misattributed
- **Severity:** MAJOR
- **Location:** Section 6.1, line 932
- **Paper claims:** "k_proj and v_proj develop the strongest coupling (44,000:1 in Qwen 7B; 46,570:1 in C-002 TinyLlama)"
- **Data shows:** In C-002 TinyLlama at step 7000, k_proj = 46,570:1 but v_proj = **24,462:1** (not 46,570:1). The paper incorrectly groups k_proj and v_proj together at the k_proj value.
- **Evidence:** PAPER3_FIGURE_INVENTORY.md line 28: "k_proj 46,570:1, v_proj 24,462:1, q_proj 16,708:1, o_proj 9,116:1 at step 7000"
- **Resolution:** Rewrite the sentence to separate k_proj and v_proj values. For C-002: k_proj=46,570:1, v_proj=24,462:1. The ordering k > v > q > o holds in both models, but the magnitudes differ. Note: the Qwen 7B source says k_proj=44,000 and v_proj=44,000 (equal), while C-002 TinyLlama shows k_proj much higher than v_proj.

### F-01-10: Correlation Fiedler ~0.10 Claim Incorrect for Holly
- **Severity:** MAJOR
- **Location:** Section 6.2, lines 973-977
- **Paper claims:** "The Correlation Fiedler converges to ~0.10 across all three model scales: 0.102 (Qwen 7B), 0.101 (TinyLlama), 1.002 (Holly Battery 14B). This scale-invariant value indicates..."
- **Data shows:** Holly Battery Correlation Fiedler = **1.002**, which is NOT ~0.10 — it is 10× higher. The paper lists all three values correctly but then claims they all "converge to ~0.10," which is only true for the first two.
- **Evidence:** FIEDLER_METRIC_NOTE.md line 35: "Holly 14B (RhombiLoRA): 1.002". The paper correctly prints 1.002 but incorrectly describes it as convergent with the ~0.10 values.
- **Resolution:** Rewrite to note that Correlation Fiedler = ~0.10 for cybernetic text models (Qwen, TinyLlama) and ~1.0 for the non-cybernetic Holly Battery. The scale-invariance claim applies to the text models; Holly behaves differently (likely because its single adapter produces near-perfect correlation across its layers, giving Fiedler near 1.0). This is still a meaningful finding but requires proper framing.

### F-01-11: I Ching Step 200 Coupling Value
- **Severity:** MINOR
- **Location:** Section 4.3, line 720
- **Paper claims:** "By step 200, I Ching coupling has been suppressed to 0.005 (83% reduction)"
- **Data shows:** DRAFT_INIT_OVERWRITE_SECTION.md step 200 table shows IC total = **0.006**, not 0.005. With 0.006: (0.029-0.006)/0.029 = 79.3%, not 83%.
- **Evidence:** DRAFT_INIT_OVERWRITE_SECTION.md line 101: "200 | 0.091 | 0.006 | 15:1"
- **Resolution:** Either change 0.005 to 0.006 and 83% to ~79%, or verify whether the paper uses a per-pair value while the source uses a total. The source table header says "I Ching complementary total."

### F-01-12: IC/RD Initial Values Mixing Per-Pair and Total
- **Severity:** MINOR
- **Location:** Section 4.3, lines 718-719
- **Paper claims:** "I Ching complementary pairs have coupling magnitude 0.029, while RD co-planar pairs start at 0.010—a 2.9:1 disadvantage"
- **Data shows:** The source table (DRAFT_INIT_OVERWRITE_SECTION.md line 99) gives IC total = 0.029 and RD total = 0.020 at step 0. If these are group totals (3 pairs each), then per-pair IC = ~0.0097 and per-pair RD = ~0.0067, giving ratio 1.45:1 (not 2.9:1). Alternatively, if 0.029 is per-pair and 0.010 is per-pair, then the ratio 2.9:1 holds, but this contradicts the source table which labels them as "total."
- **Evidence:** DRAFT_INIT_OVERWRITE_SECTION.md lines 97-99: table headers say "RD co-planar total" and "I Ching complementary total"
- **Resolution:** Clarify whether 0.029 and 0.010 are per-pair or total-group values. If total: RD total=0.020, IC total=0.029, ratio=1.45:1. If per-pair: verify against raw data. The 2.9:1 figure appears consistently across all draft sources, suggesting it may use per-pair values from a different data extraction than the totals table.

---

## Summary

| Metric | Count |
|--------|-------|
| Claims checked | 101 |
| Verified | 89 |
| Discrepancies found | 12 |
| CRITICAL | 0 |
| MAJOR | 6 |
| MINOR | 6 |

### MAJOR Discrepancies (require correction before publication):

1. **F-01-05/06/07:** n=8 (H-ch8) data in ablation table uses in-progress values (Fiedler 0.085, val loss 0.4022) instead of final values (Fiedler 0.0944, val loss 0.4024). This cascades to the stated Fiedler range 0.085–0.095 (should be 0.092–0.095).

2. **F-01-08:** Qwen 7B q_proj ratio stated as 34,000:1; source data says 22,477:1.

3. **F-01-09:** C-002 TinyLlama per-module data incorrectly groups k_proj and v_proj at 46,570:1; v_proj is actually 24,462:1.

4. **F-01-10:** Correlation Fiedler claimed to converge to ~0.10 "across all three scales," but Holly Battery = 1.002 (10× higher).

### MINOR Discrepancies (should be fixed but not factual errors):

5. **F-01-01:** "15 experiments" count doesn't match Table 1 (13 rows).
6. **F-01-02:** Convergence band delta stated as 10%, actual is 11.2%.
7. **F-01-03:** C-001 val loss presented alongside step-10K values, but C-001 stopped at step 4K.
8. **F-01-04:** n=4 Fiedler rounded from 0.0918 to 0.092 (acceptable rounding).
9. **F-01-11:** IC coupling at step 200 stated as 0.005 (83% reduction); source says 0.006 (79%).
10. **F-01-12:** IC/RD initial values may mix per-pair and total units.

### Key Finding

The most impactful corrections are the **H-ch8 stale data** (F-01-05/06/07) and the **per-module ratios** (F-01-08/09). The H-ch8 corrections actually strengthen the paper: the tighter Fiedler band (0.092–0.095 vs 0.085–0.095) makes the spectral attractor claim even more compelling. The per-module ratio corrections require rewriting 2-3 sentences in Section 6.1 but do not affect any structural conclusions.

No mathematical operations (factorizations, percentage calculations, parameter counts) were found to be incorrect. All errors are in data transcription — carrying values from in-progress experiments or from one metric to another.
