# Paper Audit — Findings Tracker

> Running log across all 7 audit rounds.
> Updated after each round's validation.

## Round 1: Factual Foundation

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-01-01 | CRITICAL | P2 | 15 corpus values exposed in Exp 6 vertex-star listing | **ACCEPTED — FIXED** |
| F-01-02 | MINOR | P1 | Algebraic connectivity range 2.3-2.5x but data shows 2.55x | ACCEPTED (not fixed — P1 locked, not egregious) |
| F-01-03 | MAJOR | P3 | H-ch8 Fiedler 0.085 is in-progress; final = 0.0944 | **ACCEPTED — FIXED** |
| F-01-04 | MAJOR | P3 | H-ch8 val loss 0.4022 is in-progress; final = 0.4024 | **ACCEPTED — FIXED** |
| F-01-05 | MAJOR | P3 | Fiedler range 0.085-0.095 should be 0.092-0.095 | **ACCEPTED — FIXED** |
| F-01-06 | MAJOR | P3 | Qwen q_proj 34,000:1 should be 22,477:1 | **ACCEPTED — FIXED** |
| F-01-07 | MAJOR | P3 | C-002 v_proj grouped with k_proj at 46,570:1; v_proj = 24,462:1 | **ACCEPTED — FIXED** |
| F-01-08 | MAJOR | P3 | Correlation Fiedler ~0.10 "across all three" but Holly = 1.002 | **ACCEPTED — FIXED** |
| F-01-09 | MINOR | P3 | Convergence band delta stated 10%; actual 11.2% | **ACCEPTED — FIXED** |
| F-01-10 | MINOR | P3 | C-001 val loss presented as step 10K; actually step 4K | **ACCEPTED — FIXED** |
| F-01-11 | MINOR | P3 | n=4 Fiedler rounded 0.0918→0.092 | ACCEPTED (acceptable rounding) |
| F-01-12 | MINOR | P3 | IC step 200 coupling 0.005/83% should be 0.006/79% | **ACCEPTED — FIXED** |
| F-01-13 | MINOR | P3 | IC/RD initial values may mix per-pair and total units | **RESOLVED by F-02C-13** |
| F-01-14 | MINOR | P3 | "15 experiments" count ambiguous | **RESOLVED by F-2A-01** |
| F-01-15 | MAJOR | P2 | Permutation control results unverifiable (no saved output) | **ACCEPTED — VERIFIED** (re-run confirms paper values) |
| F-01-16 | MAJOR | P2 | SA params: paper says 20/1000, code uses 100/5000 | **ACCEPTED — FIXED** |
| F-01-17 | MINOR | P2 | Test count 256 vs actual 312 | **RESOLVED in R3** (F-3A-05) |
| F-01-18 | MINOR | P2 | Spectral gap 0.0314 should be 0.0313 | **ACCEPTED — FIXED** |
| F-01-19 | MINOR | P2 | Path advantage "31%" vs 31.6% | ACCEPTED (adequate hedging) |
| F-01-20 | MINOR | P2 | Table 1 omits 2 of 4 tested distributions | **RESOLVED in R4** (F-4C-07) |

## Round 2: Internal Consistency

### Agent 2A: Abstract-Body Consistency (Paper 3)

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-2A-01 | MAJOR | P3 | "15 experiments" matches no enumeration (13 rows or 17 expanded) | **ACCEPTED — FIXED** (→13) |
| F-2A-02 | MAJOR | P3 | exp2.5 = 1% BD in Table 1 contradicts "0%" narrative; raw data = 0% | **ACCEPTED — FIXED** (→0%) |
| F-2A-03 | MAJOR | P3 | "Six non-cybernetic experiments" count ambiguous | ACCEPTED (removed specific count from intro) |
| F-2A-04 | MAJOR | P3 | Abstract 0.17% attributed to n=3 vs n=6; correct comparison is n ∈ {3,4,6} | **ACCEPTED — FIXED** (abstract now says 0.16% for n=3 vs n=6) |
| F-2A-05 | MAJOR | P3 | 0.058% max delta is stale (40-checkpoint data); actual 100-cp max = 0.16% | **ACCEPTED — FIXED** (3 locations) |
| F-2A-06 | MINOR | P3 | Val loss range 0.4010 uses C-002, not ablation H3 (0.4015) | **ACCEPTED — FIXED** (→0.4015) |
| F-2A-07 | MINOR | P3 | "570 bridges" not derivable from paper data | ACCEPTED (kept — number is verifiable from source) |
| F-2A-08 | MINOR | P3 | "60,000+" in abstract not derived in body | ACCEPTED (plausible, minor) |
| F-2A-09 | MINOR | P3 | "three model architectures" omits Qwen 1.5B | **ACCEPTED — FIXED** (→"four model scales") |
| F-2A-10 | POLISH | P3 | Init convergence 0.12% understated (actual 0.1246%) | **ACCEPTED — FIXED** (→0.13%) |
| F-2A-11 | POLISH | P3 | Abstract 82,854:1 is peak, not converged | **ACCEPTED — FIXED** (added "peak of") |
| F-2A-12 | POLISH | P3 | H4 val loss "---" in Table 1 but 0.4024 in Table 2 | **ACCEPTED — FIXED** |
| F-2A-13 | POLISH | P3 | exp3/exp3_tiny val loss "---" in Table 1 | DEFERRED (data may not exist) |
| F-2A-14 | MINOR | P3 | Conclusion repeats "~0%" (duplicate of F-2A-02) | **ACCEPTED — FIXED** (with exp2.5 correction) |
| F-2A-15 | POLISH | P3 | Contribution 3 peak ratio without converged context | **ACCEPTED — FIXED** (added converged value) |

### Agent 2B: Cross-References + Citations

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-2B-01 | CRITICAL | P2 | bielec2026bridge bib entry has wrong Paper 3 subtitle | **ACCEPTED — FIXED** |
| F-2B-02 | CRITICAL | P3 | Paper 3 cites Paper 2 for bridge fingerprinting; Paper 2 has no such content | **ACCEPTED — FIXED** (self-references) |
| F-2B-03 | INFO | P3 | 10 orphan bib entries | NOTED |
| F-2B-04 | INFO | P2 | 4 orphan bib entries | NOTED |
| F-2B-05 | PASS | P3 | All \\cite{} resolve | — |
| F-2B-06 | PASS | P2 | All \\cite{} resolve | — |
| F-2B-07 | PASS | P3 | All \\ref{}/\\label{} consistent | — |
| F-2B-08 | PASS | P2 | All \\ref{}/\\label{} consistent | — |
| F-2B-09 | PASS | P3 | Paper 2 title in P3 bib is accurate | — |
| F-2B-10 | PASS | both | Paper 1 title consistent across bibs | — |
| F-2B-11 | PASS | both | No ?? in logs | — |
| F-2B-12 | MINOR | P3 | Fiedler (1973) used but uncited; bib entry present | **ACCEPTED — FIXED** |
| F-2B-13 | MINOR | P3 | Beer (1972) in bib but uncited | NOTED (not essential) |

### Agent 2C: Claim-Evidence Alignment

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-02C-01 | MAJOR | P3 | "Necessary and sufficient" overstates; confound acknowledged in Limitations | **ACCEPTED — FIXED** (→"at n=6") |
| F-02C-02 | MODERATE | P3 | "Universal" for 6 experiments / 2 scales / 1 task | **ACCEPTED — FIXED** (→"robust") |
| F-02C-03 | MODERATE | P3 | Holly "rules out" causal claim; Holly differs in multiple dimensions | **ACCEPTED — FIXED** (reframed) |
| F-02C-04 | MODERATE | P3 | Section header "Requires" overstates n/contrastive confound | **ACCEPTED — FIXED** (→"Emerges Only When...Active") |
| F-02C-05 | MODERATE | P3 | Circularity: contrastive loss defined to produce BD, then BD emergence is "discovered" | ACCEPTED (noted; reframing implicit in other fixes) |
| F-02C-06 | MINOR | P3 | "Scale-invariant" with N=2 cybernetic scales | **ACCEPTED — FIXED** (→"scale-consistent") |
| F-02C-07 | MINOR | P3 | "Independent of channel count" → narrow range tested | **ACCEPTED — FIXED** (→"insensitive to") |
| F-02C-08 | MINOR | P3 | "Attractor is absolute" unfalsifiable | **ACCEPTED — FIXED** (→"robust across all tested conditions") |
| F-02C-09 | LOW | P3 | Non-cybernetic 0% framing as universal negative | ACCEPTED (wording adequate with "produces~0%") |
| F-02C-10 | LOW | P3 | "Exactly 3" is empirical, not mathematical | **ACCEPTED — FIXED** (→"effectively 3") |
| F-02C-11 | MODERATE | P3 | No formal statistical tests for performance comparisons | NOTED (effect sizes make formal tests unnecessary for primary claims) |
| F-02C-12 | LOW | P3 | "Architectural invariant" from N=2 | ACCEPTED (hedged with "suggesting") |
| F-02C-13 | MODERATE | P3 | IC/RD ratio 2.9:1 uses mixed units; source = 1.45:1 group totals | **ACCEPTED — FIXED** (0.029/0.020, 1.4:1) |
| F-02C-14 | LOW | P3 | "Discovers" vs "is trained to produce" framing | NOTED (defensible; method section is clear) |
| F-02C-15 | LOW | P3 | Holly performance claims lack baseline specification | NOTED (tangential to thesis) |

## Round 3: Cross-Paper Coherence

### Agent 3A: Narrative Arc

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-3A-01 | MINOR | P1,2,3 | Algebraic connectivity ratio cited as 2.3x (P2), 2.4x (P1,P3) | ACCEPTED (both defensible) |
| F-3A-02 | MAJOR | P2 | Table 7 consensus compares 0.93x@scale1000 with 6.69x@scale125 | **ACCEPTED — FIXED** (scale per column) |
| F-3A-03 | MAJOR | P3 | Paper 3 never cites Paper 2; narrative chain broken | **ACCEPTED — FIXED** (citation added §2.4) |
| F-3A-04 | PASS | P2,3 | Paper 2 Future Work→Paper 3 link accurate | — |
| F-3A-05 | MINOR | P2,3 | Test count: P2="256 tests" v0.3.0; P3="312 tests" v0.3.0 | **ACCEPTED — FIXED** (P2→312) |
| F-3A-06 | PASS | all | Narrative arc P1→P2→P3 is coherent | — |
| F-3A-07 | PASS | P3 | Backward references to P1 are factually accurate | — |
| F-3A-08 | PASS | P1 | Paper 1 has no forward refs (correct — LOCKED) | — |
| F-3A-09 | POLISH | P2 | "Topologies" for random graph in Contributions wording | ACCEPTED (adequate) |
| F-3A-10 | MINOR | P2 | "2.3x to 6.1x" headline uses lower-bound baseline | ACCEPTED (intro gives range) |
| F-3A-11 | MINOR | P3 | "Prior work established" for P3's own preliminary experiments | **ACCEPTED — FIXED** (→"Preliminary experiments") |
| F-3A-12 | MINOR | P3 | Paper 3 missing Paper 2 weighted-topology context | **ACCEPTED — FIXED** (via F-3A-03) |

### Agent 3B: Terminology

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| T-001 | HIGH | P3 | Bridge matrix symbol changes `\mathcal{B}`→`\mathbf{M}` within P3 | **ACCEPTED — FIXED** (unified to `\mathbf{M}`) |
| T-002 | MEDIUM | P2,3 | Paper 1 metrics: ranges (P2) vs point estimates (P3) | ACCEPTED (both defensible) |
| T-003 | LOW | P2,3 | "Channel": quoted metaphor in P2, formal in P3 | PASS (deliberate) |
| T-004–T-010 | NONE | P2,3 | 7 notation items verified consistent | PASS |
| T-011 | LOW | P2,3 | P2 previews n=6; P3 tests n∈{3,4,6,8} | PASS (no contradiction) |
| T-012 | LOW | P3 | Abstract mixes peak/convergence from different experiments | ACCEPTED (R2 added "peak of") |

### Agent 3C: Cross-References

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| 3C-01–04 | INFO | P2,3 | 4 bib entries verified clean | PASS |
| 3C-05 | MINOR | P3 | `bielec2026weights` orphan bib entry | **ACCEPTED — FIXED** (via F-3A-03) |
| 3C-06–12 | INFO | P2,3 | 7 claim verifications clean | PASS |
| 3C-13 | INFO | P2,3 | P2 Future Work→P3 forward ref accurate | PASS |
| 3C-14 | MINOR | P1,2 | BCC baseline promised in P1 and P2, undelivered | NOTED (P2 limitations acknowledge) |
| 3C-15–17 | INFO | all | Open future-work items from P1 | NOTED |
| 3C-18 | MODERATE | P3 | Paper 3 never cites Paper 2 | **ACCEPTED — FIXED** (via F-3A-03) |
| 3C-19–20 | INFO | P2,3 | Internal consistency and bib metadata clean | PASS |
| 3C-21 | MINOR | P2,3 | Test count discrepancy (same version) | **ACCEPTED — FIXED** (via F-3A-05) |
| 3C-22 | INFO | P2 | Table 7 Fiedler values verified | PASS |
| 3C-23 | MINOR | P3 | Missing conceptual back-ref to P2 direction-pair results | **ACCEPTED — FIXED** (via F-3A-03) |

## Round 4: Figures + Reproducibility

### Agent 4A: Figure Audit

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-4A-01 | MAJOR | P3 | Five figures never referenced in text | **ACCEPTED — FIXED** |
| F-4A-02 | MINOR | P3 | Fig file numbering swapped for Figs 6-7 | ACCEPTED — DEFERRED |
| F-4A-03 | MAJOR | P3 | Fig 5 caption 0.22% vs text 0.17% scope mismatch | **ACCEPTED — FIXED** |
| F-4A-04 | MAJOR | P3 | Figure-mapping doc uses obsolete filenames | ACCEPTED — DEFERRED (internal doc) |
| F-4A-05 | MINOR | P3 | Mapping doc wrong peak ratio (82,154 vs 82,854) | ACCEPTED — DEFERRED (internal doc) |
| F-4A-06 | MINOR | P3 | Mapping doc wrong Fiedler range (0.085 vs 0.092) | ACCEPTED — DEFERRED (internal doc) |
| F-4A-07 | MINOR | P3 | Mapping doc conflates init and ablation val losses | ACCEPTED — DEFERRED (internal doc) |
| F-4A-08 | MINOR | P2 | Paper 2 file numbering offset from LaTeX fig numbering | ACCEPTED — NO ACTION (deliberate) |
| F-4A-09 | POLISH | P2 | Extra Paper 1 figures in shared directory | NOTED |
| F-4A-10 | MINOR | P3 | Supplementary figures listed but missing from disk | ACCEPTED — DEFERRED |

### Agent 4B: Reproducibility

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| 4B-01 | MAJOR | P3 | Holly Battery optimizer (Prodigy) unspecified | **ACCEPTED — FIXED** |
| 4B-02 | MAJOR | P3 | Holly Battery training data unspecified | **ACCEPTED — FIXED** (partially) |
| 4B-03 | MAJOR | P3 | Holly Battery adapter target modules unspecified | **ACCEPTED — FIXED** |
| 4B-04 | MINOR | P3 | Steersman gain constants not in paper | **ACCEPTED — FIXED** |
| 4B-05 | MINOR | P3 | Steersman decay dynamics not in paper | **ACCEPTED — FIXED** (via 4B-04) |
| 4B-06 | MINOR | P3 | Spectral target update rule partially specified | **ACCEPTED — FIXED** (via 4B-04) |
| 4B-07 | MINOR | P3 | AdamW betas not explicitly stated | NOTED (standard defaults) |
| 4B-08 | MINOR | P3 | Tokenizer padding strategy not stated | NOTED (standard practice) |
| 4B-09 | MINOR | P3 | Corpus-coupled init requires proprietary data | ACCEPTED — NO ACTION (cosmetic init) |
| 4B-10 | MINOR | P3 | Reproduction instructions minimal | NOTED — DEFERRED |
| 4B-11 | POLISH | P3 | Model revisions not pinned | NOTED |
| 4B-12 | POLISH | P3 | Train/val split not specified | **ACCEPTED — FIXED** |
| 4B-13 | POLISH | P3 | I Ching mapping not in paper | NOTED (cosmetic init) |
| 4B-14 | POLISH | P3 | Half-life fit model not formally specified | NOTED |

### Agent 4C: Missing Figures

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-4C-01 | MODERATE | P3 | Architecture schematic missing | ACCEPTED — DEFERRED (TikZ creation needed) |
| F-4C-02 | MINOR | P3 | Exponential decay half-life has no direct figure | ACCEPTED — DEFERRED |
| F-4C-03 | PASS | P3 | Three-phase Fiedler coverage adequate | — |
| F-4C-04 | MINOR | P3 | Correlation Fiedler has no figure | ACCEPTED — DEFERRED |
| F-4C-05 | MINOR | P3 | Linear co-planar growth has no direct figure | ACCEPTED — DEFERRED |
| F-4C-06 | MINOR | P3 | Holly Battery has no visual evidence | ACCEPTED — DEFERRED |
| F-4C-07 | MINOR | P2 | Table 1 silently omits 2 distributions at scale 8K | **ACCEPTED — FIXED** (footnote added) |
| F-4C-08 | MINOR | P3 | Filenames fig6/fig7 swapped vs compiled order | ACCEPTED — DEFERRED (dup F-4A-02) |
| F-4C-09 | POLISH | P3 | 5 of 9 figures lack explicit \\ref citations | **ACCEPTED — FIXED** (via F-4A-01) |
| F-4C-10 | PASS | P2 | Paper 2 figure references clean | — |

## Round 5: Writing Quality

### Agent 5A: AI Tic Detection

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| 5A-10-1 | MEDIUM | P3 | "robust" repeated 5 times | **ACCEPTED — FIXED** (reduced to 2) |
| 5A-10-2 | MEDIUM | P3 | "the fact that" × 2 in Discussion | **ACCEPTED — FIXED** |
| 5A-10-3 | MEDIUM | P3,P2 | Assertive absolutes ("unambiguous", "definitively") | **ACCEPTED — FIXED** (P3 only) |
| 5A-5-1 | LOW | P3 | "The remainder of this paper..." | NOTED |
| 5A-5-2 | LOW | both | Section roadmap formulaic | NOTED |
| 5A-6-1 | LOW | P2 | "Two findings emerge" | NOTED |
| 5A-6-2 | LOW | P3 | "These results suggest" | NOTED |
| 5A-6-1b | LOW | P3 | "The results separate cleanly" | NOTED |
| 5A-misc | LOW | P2 | "robust" standalone (acceptable freq) | NOTED |

### Agent 5B: Prose Quality

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| P3-TR-01 | HIGH | P3 | Missing Method→Results transition | **ACCEPTED — FIXED** |
| P3-JD-01 | HIGH | P3 | Abstract jargon overload | ACCEPTED — DEFERRED (R7) |
| P3-RD-01 | HIGH | P3 | Core 100%/0% finding stated 7 times | NOTED |
| P3-AQ-01 | HIGH | P3 | 17 numbers in 28-line abstract | ACCEPTED — DEFERRED (R7) |
| P3-SV-01 | MEDIUM | P3 | Six identical bold preview paragraphs | NOTED — DEFERRED (R7) |
| P3-JD-02 | MEDIUM | P3 | Control-theory jargon in Steersman section | NOTED |
| P3-JD-03 | MEDIUM | P3 | Co-planar/cross-planar no reminder glosses | **ACCEPTED — FIXED** |
| P3-RD-02 | MEDIUM | P3 | "0.17%" stated 5 times | NOTED |
| P3-PS-01 | MEDIUM | P3 | Contributions list duplicates bold previews | NOTED — DEFERRED (R7) |
| P3-VV-01 | MEDIUM | P3 | "produces" 30+ times | NOTED |
| P3-AP-01 | MEDIUM | P3 | Heavy passive in method section | NOTED |
| P3-SI-01 | MEDIUM | P3 | Section 4 opens with data, no framing | **ACCEPTED — FIXED** (via TR-01) |
| P2-TR-01 | MEDIUM | P2 | Missing register transition (Sec 4→5) | NOTED |
| P2-JD-01 | MEDIUM | P2 | "Fiedler" unglossed in abstract | NOTED |
| P2-RD-01 | MEDIUM | P2 | 6.1× stated 6 times | NOTED |
| P2-SI-01 | MEDIUM | P2 | Single-cell section opens without framing | NOTED |

### Agent 5C: Limitations Honesty

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| L-01 | HIGH | P3 | Single seed (42) for all experiments | **ACCEPTED — FIXED** |
| L-04 | HIGH | P3 | Channel ablation only at 1.1B | **ACCEPTED — FIXED** |
| L-08 | HIGH | P3 | 13 Steersman params, no sensitivity analysis | **ACCEPTED — FIXED** |
| L-02 | MEDIUM | P3 | N=2 cybernetic scales insufficient | **ACCEPTED — FIXED** (expanded existing) |
| L-03 | MEDIUM | P3 | No formal statistical tests | NOTED (subsumed by L-01) |
| L-05 | MEDIUM | P3 | Holly differs in 4 dimensions | **ACCEPTED — FIXED** (via L-02) |
| L-09 | MEDIUM | P3 | BD threshold sensitivity | NOTED |
| L-10 | MEDIUM | P3 | "Scale Consistency" section title | NOTED (R2 already calibrated) |
| L-13 | MEDIUM | P3 | Only causal LM tested | NOTED (already acknowledged) |
| L-06 | LOW | P3 | Only QKV/O target modules | NOTED |
| L-07 | LOW | P3 | Only LoRA, no other PEFT | NOTED |
| L-11 | LOW | P3 | "Structural invariant" from N=2 | NOTED (hedged with "suggesting") |
| L-12 | LOW | P3 | Residual causal language | NOTED (defensible) |
| L-14 | MEDIUM | P2 | Only FCC and SC compared | PASS (acknowledged) |
| L-15 | MEDIUM | P2 | Corpus values proprietary | PASS (mitigation described) |
| L-16 | LOW | P2 | SA local optimum risk | PASS |
| L-17 | LOW | P2 | Consensus inversion unresolved | PASS (acknowledged) |

## Round 6: Bibliography + Claims Calibration

### Agent 6A: Citation Completeness

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-6A-01 | MAJOR | P3 | 9 orphan bib entries (39% orphan rate) | **ACCEPTED — PARTIALLY FIXED** (LoRA-XS/MELoRA cited; Beer cited; 6 removed) |
| F-6A-02 | MINOR | P2 | 4 orphan bib entries (hales2005, hales2017, bateson1979, ghosh2008) | **ACCEPTED — FIXED** (removed) |
| F-6A-03 | MINOR | P2 | beer1972 "axis-fragility" framing | NOTED — NO ACTION |

### Agent 6B: Related Work Gaps

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-6B-01 | HIGH | P3 | LoRA-XS discussion lost in consolidation | **ACCEPTED — FIXED** (restored to §2.2) |
| F-6B-02 | HIGH | P3 | MELoRA discussion lost in consolidation | **ACCEPTED — FIXED** (restored to §2.2) |
| F-6B-03 | MEDIUM | P3 | GaLore not cited | NOTED — NO ACTION (gradient projection, not adapter) |
| F-6B-04 | MEDIUM | P3 | QLoRA not cited | NOTED — NO ACTION (experiments use bfloat16) |
| F-6B-05 | MEDIUM | P3 | Beer (1972) not cited in text | **ACCEPTED — FIXED** (added to §2.3) |
| F-6B-06 | LOW | P3 | Uncited adapter merging cluster (4 entries) | **ACCEPTED — FIXED** (removed from bib) |
| F-6B-07 | LOW | P3 | Putterman et al. 2024 uncited | **ACCEPTED — FIXED** (removed from bib) |
| F-6B-08 | LOW | P3 | Biderman et al. 2024 uncited | **ACCEPTED — FIXED** (removed from bib) |

### Agent 6C: Claims Calibration

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-6C-H1 | HIGH | P3 | "natural representational structure" (Discussion) | **ACCEPTED — FIXED** (→"under these training conditions") |
| F-6C-H2 | HIGH | P3 | "unique attractor" (Conclusion) | **ACCEPTED — FIXED** (→"robust attractor") |
| F-6C-M1 | MEDIUM | P3 | "necessary and sufficient at n=6" ×4 | NOTED — NO ACTION (already qualified in R2) |
| F-6C-M2 | MEDIUM | P3 | "emerges identically" across scales | **ACCEPTED — FIXED** (→"emerges consistently") |
| F-6C-M3 | MEDIUM | P3 | "Initialization is cosmetic" | **ACCEPTED — FIXED** (→"effects vanish within 200 steps") |
| F-6C-M4 | MEDIUM | P3 | "Discovers" in title + conclusion | NOTED — NO ACTION (defensible; Discussion now cautious) |
| F-6C-M5 | MEDIUM | P3 | "rejecting arbitrary patterns" | **ACCEPTED — FIXED** (→"developing no structure without it") |
| F-6C-M6 | MEDIUM | P3 | "latent geometric preferences" (Conclusion) | **ACCEPTED — FIXED** (→"receptive to geometric structure") |
| F-6C-M7 | MEDIUM | P3 | Holly causal claim (Intro) | **ACCEPTED — FIXED** (→"additional evidence that the effect requires") |
| F-6C-L1 | LOW | P3 | "identical validation loss" ×3 | **ACCEPTED — FIXED** (→"matching" at all 3 locations) |
| F-6C-L2 | LOW | P3 | "exactly 3" ×2 (body inconsistent with abstract) | **ACCEPTED — FIXED** (→"effectively 3" at both + §4.3) |
| F-6C-L3 | LOW | P3 | "independent of initialization strategy" (abstract) | NOTED — NO ACTION (scope clear in context) |
| F-6C-L4 | LOW | P3 | "convergence proof" in Fig 2 caption | **ACCEPTED — FIXED** (→"convergence evidence") |

## Round 7: Final Integration

### Agent 7A: Full Read-Through (Paper 3)

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-7A-01 | MAJOR | P3 | "Perpendicular to axis" incorrect; grouping is by vanishing coordinate | **ACCEPTED — FIXED** |
| F-7A-02 | MAJOR | P3 | k > v overstated (tied at Qwen 7B) | **ACCEPTED — FIXED** (→k >= v) |
| F-7A-03 | MINOR | P3 | "60,000+ bridge matrices" not reconciled in body | NOTED |
| F-7A-04 | MINOR | P3 | "Six non-cybernetic experiments" count unclear | NOTED (addressed in R2) |
| F-7A-05 | MINOR | P3 | C-001 val loss not comparable (4K vs 10K) | NOTED (already stated in text) |
| F-7A-06 | MINOR | P3 | "I Ching" term jarring in abstract | NOTED (intentional) |
| F-7A-07 | MINOR | P3 | "Removing it" at n!=6 — was never active | NOTED (Limitations covers this) |
| F-7A-08 | MINOR | P3 | "Block-diagonal" undefined until §3.3 | NOTED (standard practice) |
| F-7A-09 | MINOR | P3 | "~70,000:1" vs midpoint ~67,600 | NOTED (band stated nearby) |
| F-7A-10 | MINOR | P3 | Fig 5 caption peak from different experiment | NOTED (acceptable context) |
| F-7A-11 | MINOR | P3 | Frankle & Carlin citation author check | NOTED (correct) |
| F-7A-12 | MINOR | P3 | Holly 600 steps vs 10K differential | NOTED (exp2/2.5 also 0%) |
| F-7A-13 | MINOR | P3 | "effective...effectively" redundancy in abstract | **ACCEPTED — FIXED** |
| F-7A-14 | MINOR | P3 | "1,020x bifurcation" terminology | NOTED (colloquial use clear) |
| F-7A-15 | MINOR | P3 | Missing val loss for exp3/exp3_tiny | NOTED (earlier code limitation) |

### Agent 7B: Cross-Paper Final

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-7B-01 | MINOR | P1,P3 | RD vertex terminology differs | NOTED (both standard) |
| F-7B-02 | MINOR | P1,P2 | Path advantage range slight narrowing | NOTED (R3 flagged) |
| F-7B-03 | MINOR | P1,P2 | Scale range imprecision | NOTED (P1 locked) |
| F-7B-04 | MINOR | P2 | "Five topologies" includes random graph | NOTED (correct) |
| F-7B-05 | MINOR | P1,P2,P3 | Test count 256 vs 312 | NOTED (P1 locked) |

### Agent 7C: Submission Checklist

| ID | Severity | Paper | Finding | Status |
|----|----------|-------|---------|--------|
| F-7C-01 | FAIL | P2 | Overfull hbox 17.93pt (Availability URLs) | **ACCEPTED — FIXED** |
| F-7C-02 | FAIL | P3 | Overfull hbox 10.27pt (§2.2 citation chain) | **ACCEPTED — FIXED** |

## Statistics

| Round | Findings | Critical | Major | Minor/Other | Accepted | Fixed | Deferred | Rejected |
|-------|----------|----------|-------|-------------|----------|-------|----------|----------|
| 1 | 20 | 1 | 8 | 11 | 20 | 12 | 4 | 0 |
| 2 | 43 | 2 | 6 | 35 | 43 | 26 | 1 | 0 |
| 3 | 47 | 0 | 2 | 45 | 47 | 7 | 0 | 0 |
| 4 | 34 | 0 | 6 | 28 | 34 | 9 | 12 | 0 |
| 5 | 42 | 0 | 0 | 42 | 42 | 11 | 3 | 0 |
| 6 | 24 | 0 | 5 | 19 | 24 | 17 | 0 | 0 |
| 7 | 22 | 0 | 2 | 20 | 22 | 5 | 0 | 0 |
