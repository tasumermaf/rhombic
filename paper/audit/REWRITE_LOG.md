# Paper Audit — Rewrite Log

> Documents every change made to Papers 2 and 3 during the audit process.
> Paper 1 is LOCKED unless a glaring error is found.

## Pre-Audit Fixes (Step 0)

| Date | Paper | Location | Change | Justification |
|------|-------|----------|--------|---------------|
| 2026-03-15 | P3 | Fig 5 caption (L811) | 82,154:1 → 82,854:1 | Typo — correct peak ratio is 82,854:1 per EXPERIMENT_TRACKER |

## Round 1 Changes

| Date | Paper | Location | Change | Justification |
|------|-------|----------|--------|---------------|
| 2026-03-15 | P2 | Exp 6 vertex-star listing (L652-656) | Removed specific edge values, kept divisibility relationships | F-01-01: 15 corpus values exposed — IP boundary violation |
| 2026-03-15 | P2-IT | Same section (L692-696) | Same fix in Italian version | F-01-01: mirror fix |
| 2026-03-15 | P3 | Table 2, line 800 | n=8: Fiedler 0.085→0.094, val loss 0.4022→0.4024 | F-01-03/04: H-ch8 stale data; final values from H-ch8-results-hermes.json |
| 2026-03-15 | P3 | Lines 831, 845, 883 | Fiedler range 0.085-0.095→0.092-0.095 | F-01-05: cascading from H-ch8 correction |
| 2026-03-15 | P3 | Section 6.1, lines 932-936 | Per-module ratios: q_proj 34,000→22,477; v_proj separated from k_proj (24,462:1) | F-01-06/07: wrong q_proj value; v_proj misattributed |
| 2026-03-15 | P3 | Section 6.2, lines 973-977 | Rewritten: ~0.10 for cybernetic text models only; Holly=1.002 distinguished as different regime | F-01-08: Correlation Fiedler claim incorrect for Holly |
| 2026-03-15 | P3 | Line 672 | Convergence band delta 10%→11% | F-01-09: (71337-64168)/64168=11.2% |
| 2026-03-15 | P3 | Lines 711-713 | C-001 val loss now states "4K steps" explicitly | F-01-10: C-001 ran 4K not 10K |
| 2026-03-15 | P3 | Line 721 | IC step 200: 0.005/83%→0.006/79% | F-01-12: source table says 0.006 |
| 2026-03-15 | P2 | Line 519-520 | SA params: "20 restarts of 1,000"→"100 restarts of 5,000" | F-01-16: code uses 100/5000 |
| 2026-03-15 | P2-IT | Line 552 | Same SA param fix in Italian | F-01-16: mirror fix |
| 2026-03-15 | P2 | Table 4, line 582 | Spectral gap 0.0314→0.0313 | F-01-18: 0.0863/2.7547=0.03133 rounds to 0.0313 |
| 2026-03-15 | P3 src | sections/06_scale_invariance.md | Per-module ratios and Correlation Fiedler text updated | Source file consistency with LaTeX |

## Round 2 Changes

| Date | Paper | Location | Change | Justification |
|------|-------|----------|--------|---------------|
| 2026-03-15 | P2 bib | bielec2026bridge entry | Title updated to match Paper 3's actual title | F-2B-01: wrong subtitle in bibliography |
| 2026-03-15 | P3 | Lines ~111, ~273 | Removed \\cite{bielec2026weights} for bridge fingerprinting; replaced with self-references to Table 1 exp1a-e | F-2B-02: Paper 2 contains no bridge fingerprinting content |
| 2026-03-15 | P3 | Line ~428 | Added \\cite{fiedler1973} at first Fiedler mention | F-2B-12: foundational citation missing |
| 2026-03-15 | P3 | Abstract L52 | "15 experiments" → "13 experiments" | F-2A-01: 13 table rows |
| 2026-03-15 | P3 | Abstract L53 | "three model architectures" → "four model scales" | F-2A-09: Qwen 1.5B omitted |
| 2026-03-15 | P3 | Abstract L63 | "necessary and sufficient" → "necessary and sufficient at n=6" | F-02C-01: confound with channel count |
| 2026-03-15 | P3 | Abstract L64 | Added "peak of" before 82,854:1 | F-2A-11: peak vs converged |
| 2026-03-15 | P3 | Abstract L68 | "exactly~3" → "effectively~3" | F-02C-10: empirical, not mathematical |
| 2026-03-15 | P3 | Abstract L69-70 | "0.17% maximum delta" → "maximum delta: 0.16% across 100 checkpoints" | F-2A-04/05: wrong comparison and stale figure |
| 2026-03-15 | P3 | Intro L145 | "universal" → "robust" | F-02C-02: 6 experiments / 2 scales insufficient |
| 2026-03-15 | P3 | Intro L147 | Removed "6 experiments" from non-cybernetic count | F-2A-03: count ambiguous |
| 2026-03-15 | P3 | Intro L149 | "necessary and sufficient" → "necessary and sufficient at n=6" | F-02C-01 |
| 2026-03-15 | P3 | Intro L156 | 0.12% → 0.13% | F-2A-10: actual 0.1246% |
| 2026-03-15 | P3 | Intro L173 | 0.4010 → 0.4015 | F-2A-06: 0.4010 is C-002, not ablation |
| 2026-03-15 | P3 | Intro L193-194 | 0.058% → 0.16% | F-2A-05: stale 40-checkpoint data |
| 2026-03-15 | P3 | Intro L198 | "scale-invariant" → "scale-consistent" | F-02C-06: N=2 cybernetic |
| 2026-03-15 | P3 | Intro L204-205 | Correlation Fiedler: now specifies "cybernetic text models" | Already partially fixed in R1 |
| 2026-03-15 | P3 | Contrib L210 | "universal property" → "robust property" | F-02C-02 |
| 2026-03-15 | P3 | Contrib L221 | Peak ratio + converged value added | F-2A-15 |
| 2026-03-15 | P3 | Contrib L224-225 | "necessary and sufficient" → "necessary and sufficient at n=6" | F-02C-01 |
| 2026-03-15 | P3 | Contrib L229 | "Scale invariance" → "Scale consistency" | F-02C-06 |
| 2026-03-15 | P3 | §4.1 L576 | "15 experiments" → "13 experiments" | F-2A-01 |
| 2026-03-15 | P3 | Table 1 L596 | exp2.5 BD% 1% → 0% | F-2A-02: raw data = 0% |
| 2026-03-15 | P3 | Table 1 L607 | H4 val loss --- → 0.4024 | F-2A-12: available in Table 2 |
| 2026-03-15 | P3 | §4.1 L619 | "attractor is absolute" → "robust across all tested conditions" | F-02C-08 |
| 2026-03-15 | P3 | §4.3 L721-723 | IC/RD init: 0.029/0.010/2.9:1 → 0.029/0.020/1.4:1 (group totals) | F-02C-13: mixed-unit ratio |
| 2026-03-15 | P3 | §5.2.1 L821 | "Independent of" → "Insensitive to" in header | F-02C-07 |
| 2026-03-15 | P3 | §5.2.1 L826 | 0.058% → 0.16% | F-2A-05 |
| 2026-03-15 | P3 | §5.2.2 L830 | "Requires" → "Emerges Only When...Active" in header | F-02C-04 |
| 2026-03-15 | P3 | Fig 5 caption L816-817 | 0.17% → 0.22% (includes n=8) | F-2A-12 cascade |
| 2026-03-15 | P3 | §5.4 L908 | 0.058% → 0.16% | F-2A-05 |
| 2026-03-15 | P3 | §6 heading L922 | "Scale Invariance" → "Scale Consistency" | F-02C-06 |
| 2026-03-15 | P3 | §6.3 L1000-1001 | Holly causal claim reframed with Qwen comparison | F-02C-03 |
| 2026-03-15 | P3 | Concl L1129 | "six non-cybernetic controls" → "non-cybernetic controls" | F-2A-03 |
| 2026-03-15 | P3 | Concl L1132 | "independent of" → "insensitive to" | F-02C-07 |
| 2026-03-15 | P3 | Concl L1134-1135 | "necessary and sufficient" → "necessary and sufficient at n=6" | F-02C-01 |
| 2026-03-15 | P3 | Concl L1138-1139 | "exactly~3" → "effectively~3" | F-02C-10 |

## Round 3 Changes

| Date | Paper | Location | Change | Justification |
|------|-------|----------|--------|---------------|
| 2026-03-15 | P3 | Intro lines 98-103 | `\mathcal{B}` → `\mathbf{M}` (4 occurrences) | T-001: symbol inconsistency between intro and method |
| 2026-03-15 | P3 | §2.4 lines 330-333 | Added Paper 2 citation: "A companion paper~\cite{bielec2026weights} showed..." | F-3A-03/3C-18: Paper 3 must cite Paper 2 to complete narrative chain |
| 2026-03-15 | P3 | Intro line 111 | "Prior work" → "Preliminary experiments" | F-3A-11: these are P3's own experiments, not a separate publication |
| 2026-03-15 | P2 | Table 7 line 809 | Consensus row: added "(scale 1000)" to P1 column, "(scale 125)" to P2 column | F-3A-02: was comparing different scales without annotation |
| 2026-03-15 | P2 | Lines 363, 955 | "256 tests" → "312 tests" | F-3A-05/3C-21: stale count; current v0.3.0 has 312 |

## Round 4 Changes

| Date | Paper | Location | Change | Justification |
|------|-------|----------|--------|---------------|
| 2026-03-15 | P3 | §5.1 ~L762 | Added `Figure~\ref{fig:heatmaps}` | F-4A-01: unreferenced figure |
| 2026-03-15 | P3 | §5.2.1 ablation text | Added `(Figure~\ref{fig:ablation})` | F-4A-01: unreferenced figure |
| 2026-03-15 | P3 | §5.2.1 three-phase text | Added `(Figure~\ref{fig:h2eigen})` | F-4A-01: unreferenced figure |
| 2026-03-15 | P3 | §6.1 per-layer text | Added `(Figure~\ref{fig:perlayer})` | F-4A-01: unreferenced figure |
| 2026-03-15 | P3 | §6.1 k>v>q>o text | Added `(Figure~\ref{fig:permodule})` | F-4A-01: unreferenced figure |
| 2026-03-15 | P3 | §5.2.1 after 0.17% | Added "Including $n = 8$ (0.4024), the full range is 0.22\%." | F-4A-03: scope clarification |
| 2026-03-15 | P3 | Appendix Steersman table | Added 5 rows: spectral boost gain (10.0×), contrastive increment (0.02), stability dampen floor (0.8), spectral decay target (0.5×base), Fiedler target tracking rate (0.1) | 4B-04/05/06: gain constants missing |
| 2026-03-15 | P3 | Appendix after per-experiment table | Added Holly Battery configuration note: Prodigy optimizer, video data, attention targets, epoch count | 4B-01/02/03: Holly under-specified |
| 2026-03-15 | P3 | Appendix hyperparams table | Added "alpaca-cleaned (90/10 train/val split)" | 4B-12: split not specified |
| 2026-03-15 | P2 | Table 1 (tab:exp1) | Added footnote: random/power-law not computed at scale 8,000 | F-4C-07/F-01-20: silent omission |
| 2026-03-15 | P2 | Table 5 (tab:exp5) | Added footnote: same cost constraint as Table 1 | F-4C-07/F-01-20: same issue |

## Round 5 Changes

| Date | Paper | Location | Change | Justification |
|------|-------|----------|--------|---------------|
| 2026-03-15 | P3 | Intro L139 | "robust" → "consistent" | 5A-10-1: reduce 5→2 occurrences |
| 2026-03-15 | P3 | Contrib L212 | "robust property" → "consistent property" | 5A-10-1 |
| 2026-03-15 | P3 | Concl L1144 | "robust across" → "holds across" | 5A-10-1 |
| 2026-03-15 | P3 | Intro L144 | "the answer is unambiguous:" → "the results are consistent:" | 5A-10-3: assertive absolute |
| 2026-03-15 | P3 | Discussion L1062 | "The fact that the bridge accepts" → "The bridge accepts...indicating" | 5A-10-2 |
| 2026-03-15 | P3 | Discussion L1128 | "The fact that the effective rank" → "The effective rank...equaling" | 5A-10-2 |
| 2026-03-15 | P3 | §4.1 opening | Added 2-sentence transition bridge before Table reference | P3-TR-01/SI-01: missing Method→Results transition |
| 2026-03-15 | P3 | §4.1 after table | Added ρ definition gloss "(same-axis vs. different-axis channel pairs)" | P3-JD-03: co-planar/cross-planar reminder |
| 2026-03-15 | P3 | Limitations | Added "Single seed" paragraph (seed=42, single-seed observations) | L-01: unacknowledged |
| 2026-03-15 | P3 | Limitations | Added "Ablation scale" paragraph (1.1B only for channel ablation) | L-04: unacknowledged |
| 2026-03-15 | P3 | Limitations | Added "Steersman sensitivity" paragraph (hand-tuned, no sensitivity analysis) | L-08: unacknowledged |
| 2026-03-15 | P3 | Limitations | Expanded "Limited scale range" with Holly confounds and N=2 insufficiency | L-02/L-05: partially acknowledged |

## Round 6 Changes

| Date | Paper | Location | Change | Justification |
|------|-------|----------|--------|---------------|
| 2026-03-15 | P3 | §2.2 after L274 | Restored MELoRA comparison paragraph + \cite{ren2024melora} | F-6B-02: lost during consolidation |
| 2026-03-15 | P3 | §2.2 after MELoRA | Restored LoRA-XS comparison paragraph + \cite{banaei2024loraxs} | F-6B-01: lost during consolidation |
| 2026-03-15 | P3 | §2.3 L290 | Added Beer citation: \cite{wiener1948, ashby1956, beer1972} | F-6B-05: Beer uncited in Paper 3 |
| 2026-03-15 | P3 | Conclusion ~L1179 | "unique attractor" → "robust attractor" | F-6C-H2: formal dynamical systems claim unsupported |
| 2026-03-15 | P3 | Discussion ~L1082 | Removed "natural" from "natural representational structure"; added "under these training conditions"; rewrote "rejecting" → "developing no structure without it" | F-6C-H1/M5: implies pre-existing preference |
| 2026-03-15 | P3 | Contrib ~L232 | "emerges identically" → "emerges consistently" | F-6C-M2: 4× ratio difference between scales |
| 2026-03-15 | P3 | Intro ~L162 | "Initialization is cosmetic" → "Initialization effects vanish within 200 steps" | F-6C-M3: dismissive tone |
| 2026-03-15 | P3 | Intro ~L204 | Holly claim: "confirming caused by" → "providing additional evidence that the effect requires" | F-6C-M7: 6 confounded variables |
| 2026-03-15 | P3 | Abstract L70, Intro L194, §4.3 L937 | "identical" → "matching" (×3) | F-6C-L1: "identical" is mathematical claim; delta is 0.16% |
| 2026-03-15 | P3 | Intro L193, §4.3 L929, §4.3 L936 | "exactly 3" → "effectively 3" (×3) | F-6C-L2: suppressed entries at ~10⁻⁵, not mathematical zero |
| 2026-03-15 | P3 | Fig 2 caption L731 | "convergence proof" → "convergence evidence" | F-6C-L4: not a mathematical proof |
| 2026-03-15 | P3 | Conclusion ~L1201 | "latent geometric preferences that standard training does not surface" → "receptive to geometric structure when it is encoded in the training objective" | F-6C-M6: "latent" implies pre-existence |
| 2026-03-15 | P3 bib | rhombic-paper3.bib | Removed 6 orphan entries: biderman2024lora, ilharco2023editing, yadav2023ties, yu2024dare, yang2024adamerging, putterman2024 | F-6A-01/6B-06/07/08: uncited bib cleanup |
| 2026-03-15 | P2 bib | rhombic-paper2.bib | Removed 4 orphan entries: hales2005, hales2017, bateson1979, ghosh2008 | F-6A-02: uncited bib cleanup |

## Round 7 Changes

| Date | Paper | Location | Change | Justification |
|------|-------|----------|--------|---------------|
| 2026-03-15 | P3 | §3.1 L521-529 | "perpendicular to one of three Cartesian axes" → "grouped by coordinate that vanishes in their normal vectors"; x/y/z-axis → z=0/y=0/x=0 group labels | F-7A-01: geometric description incorrect |
| 2026-03-15 | P3 | §6.1 L976 | "k > v > q > o" → "k >= v > q > o (with k and v tied at Qwen 7B)" | F-7A-02: k/v tied at 44,000:1 in Qwen 7B |
| 2026-03-15 | P3 | §6.1 L978 | "suggesting an architectural invariant" → "suggesting a shared architectural pattern" | F-7A-02: weakened to match data |
| 2026-03-15 | P3 | Abstract L68 | "effective dimensionality...is effectively 3" → "has effective dimensionality 3" | F-7A-13: word redundancy |
| 2026-03-15 | P2 | L968 | Added \cite{hu2022lora} at LoRA mention in Future Work | Duplicate 6A finding: LoRA cited without reference |
| 2026-03-15 | P2 bib | rhombic-paper2.bib | Added hu2022lora entry | Same |
| 2026-03-15 | P2 | L963-970 | Wrapped Availability paragraph in sloppypar; shortened commit hash | F-7C-01: 17.93pt overfull hbox |
| 2026-03-15 | P3 | §2.2 L268-278 | Wrapped paragraph in sloppypar | F-7C-02: 10.27pt overfull hbox |
