# Round 5, Agent 5C: Limitations Honesty Audit

**Scope:** Are all real limitations acknowledged? Do overclaims survive Round 2 calibration?
**Papers:** Paper 3 (primary), Paper 2 (secondary)
**Date:** 2026-03-15

---

## Paper 3 Limitations Section — Current Text

The existing Limitations subsection (Section 6.3, lines 1075--1101) acknowledges five items:

1. Single task family (alpaca-cleaned instruction-following)
2. Single bridge architecture (dense n x n)
3. Contrastive loss confound (defined only for n=6)
4. No formal null model for 100% BD probability
5. Limited scale range (1.1B, 7B, 14B; 14B non-cybernetic only)

---

## Findings

### L-01: Single seed for all experiments — UNACKNOWLEDGED

**Severity: HIGH**

**Claim (line 550):** `random seed~42`

**Location:** Section 3.4 (Training Setup), Table 8 (Appendix).

**The limitation:** Every experiment uses seed=42. No multi-seed validation is reported for any Paper 3 experiment. The lock-in dynamics, coupling ratios, convergence trajectories, and final val losses are all single-seed observations. While the block-diagonal binary classification (100% vs 0%) is likely robust to seed variation given the magnitude of the effect (70,000:1 vs ~1:1), the specific numerical claims --- half-life of 123 steps, peak ratio 82,854:1, convergence band 64,168--71,337:1, the per-module hierarchy k > v > q > o --- are single-seed estimates with no confidence intervals.

**Existing coverage:** Not acknowledged anywhere in the Limitations section or elsewhere. Paper 2's Limitations section *does* acknowledge single-seed concerns for its stochastic results (and Paper 2 actually ran multi-seed validation for its deterministic Fiedler ratios). Paper 3 does neither.

**Assessment:** A reviewer will notice this immediately. The half-life claim of "123 +/- 8 steps" (line 668) provides a spread across three experiments (C-001, C-002, C-003), not across seeds --- this should not be confused with multi-seed validation. All three experiments use seed=42.

**Recommendation:** Add to Limitations: "All experiments use a single random seed (42). While the binary block-diagonal classification (100\% vs.\ 0\%) is robust given the magnitude of the effect, specific numerical quantities---coupling ratios, half-life estimates, per-module hierarchies---are single-seed observations and may vary under different initializations of the LoRA parameters and data ordering."

---

### L-02: N=2 cybernetic model scales — PARTIALLY ACKNOWLEDGED

**Severity: MEDIUM**

**Claim (line 199--207):**
> "The finding is scale-consistent. Block-diagonal structure appears at both 1.1B (TinyLlama) and 7B (Qwen) under cybernetic training."

**Claim (contribution 6, line 231--233):**
> "Scale consistency from 1.1B to 7B parameters. Block-diagonal structure emerges identically at TinyLlama~1.1B and Qwen~7B."

**Limitation line 1098--1101:**
> "Three model scales (1.1B, 7B, 14B) provide evidence, but the 14B result is non-cybernetic only. A cybernetic experiment at 14B or larger would confirm that block-diagonal emergence scales to frontier model sizes."

**Assessment:** The Limitations section acknowledges the 14B gap but does not explicitly state that N=2 cybernetic scales is a narrow basis for "scale-consistent" claims. Round 2 already downgraded "scale-invariant" to "scale-consistent" (F-02C-06), which is a significant improvement. However, the Limitations section frames the issue as "missing a cybernetic 14B datapoint" rather than as "two datapoints do not establish a scaling law."

The section title "Scale Consistency" (Section 5) and the contribution claim "Scale consistency from 1.1B to 7B" are appropriately hedged given the Round 2 calibration. The Limitations section partially covers this but could be more explicit that two scales provide insufficient evidence for scaling claims.

**Recommendation:** Consider adding one sentence to the existing scale-range limitation: "Two cybernetic scales (1.1B, 7B) are insufficient to establish a scaling law; scale-consistent behavior at these two points does not guarantee the effect persists at larger scales."

---

### L-03: No formal statistical tests for performance comparisons — PARTIALLY ACKNOWLEDGED

**Severity: MEDIUM**

**Claim (line 173--177):**
> "Validation loss at 10,000 steps is 0.4015--0.4022 across channel counts n in {3, 4, 6}, a maximum delta of 0.17%."

**Claim (line 836--837):**
> "Validation loss varies by at most 0.17% across n in {3, 4, 6}"

**Existing acknowledgment:** F-02C-11 in Round 2 noted "No formal statistical tests for performance comparisons" and the hub disposition was "NOTED (effect sizes make formal tests unnecessary for primary claims)."

**Assessment:** The Limitations section does not mention the absence of formal statistical tests. The argument that effect sizes make tests unnecessary is defensible for the block-diagonal finding (the bifurcation is extreme), but the "task performance is orthogonal" claim rests on showing that val loss differences are negligible --- and this is a null-result claim. Null results require more statistical rigor than positive results. The 0.17% delta could be meaningful or meaningless depending on the variance of the estimate, which is unknown because there is only one seed and no confidence intervals.

**Recommendation:** Either add a sentence to Limitations acknowledging that performance equivalence is demonstrated by effect size rather than by formal hypothesis testing, or note that the single-seed design precludes confidence intervals on the val loss comparison.

---

### L-04: Channel count ablation only at 1.1B scale — UNACKNOWLEDGED

**Severity: HIGH**

**Claim (Section 4, line 788--789):**
> "We vary the channel count n in {3, 4, 6, 8} while holding the total LoRA rank fixed at 24. All experiments use TinyLlama 1.1B."

**The limitation:** The entire channel-count ablation (H1-H4) is conducted at a single model scale (TinyLlama 1.1B). The paper's strongest specific finding --- that n=6 with contrastive loss produces a 1,020x Fiedler bifurcation while n in {3,4,8} produces uniform coupling --- is established at one scale only. It is unknown whether the same bifurcation, the same Fiedler convergence values, or the same performance insensitivity hold at 7B or larger.

**Existing coverage:** Not acknowledged. The Limitations section discusses scale range for the block-diagonal finding but does not note that the ablation is single-scale.

**Assessment:** A reviewer may ask: "Does the n=6 specificity hold at 7B? Could a 7B model develop block structure at n=4 or n=8 under contrastive loss?" These are unanswerable from the current data, and the limitation should be acknowledged.

**Recommendation:** Add: "The channel count ablation was conducted at 1.1B scale only; whether the n=6 bifurcation replicates at 7B or larger is untested."

---

### L-05: Holly Battery as only 14B datapoint — differences beyond Steersman absence — PARTIALLY ACKNOWLEDGED

**Severity: MEDIUM**

**Claim (line 1015--1018):**
> "The Holly result provides additional evidence consistent with the Steersman as the causal factor, complementing the more controlled Qwen-family comparison."

**Existing limitation (line 1098--1101):**
> "Three model scales (1.1B, 7B, 14B) provide evidence, but the 14B result is non-cybernetic only."

**Assessment:** Round 2 already addressed the Holly causal claim (F-02C-03), and the paper was reframed to position Holly as "additional evidence consistent with" rather than definitive. The appendix (lines 1269--1279) now documents the Holly differences: Prodigy optimizer, video diffusion model, proprietary data, no Steersman. The text appropriately positions the Qwen exp2/exp3 comparison as the stronger one.

However, the Limitations section does not explicitly note that Holly differs from the other experiments in four dimensions simultaneously (model architecture, task, dataset, optimizer), making it an uncontrolled datapoint. The reframing in the body text is good; the Limitations section should echo it.

**Recommendation:** This is partially covered by the body text reframing. Consider adding to the scale limitation: "...the 14B result differs from the causal LM experiments in model architecture, task, dataset, and optimizer, confounding its use as a scale-only comparison."

---

### L-06: Only QKV/O target modules tested — UNACKNOWLEDGED

**Severity: LOW**

**Claim (line 549):** `target modules {W_Q, W_K, W_V, W_O}`

**The limitation:** All experiments target only attention projection modules. MLP layers (up/down/gate projections) are not targeted. It is unknown whether the block-diagonal structure would emerge in MLP adapter bridges, or whether the per-module hierarchy (k > v > q > o) is specific to attention modules.

**Assessment:** This is a scope limitation that a reviewer might raise but is not critical. The paper focuses on attention modules consistently and does not claim generality beyond them.

**Recommendation:** Minor. Could add to the "Single bridge architecture" limitation: "...and all adapters target attention projections (QKV/O); the behavior in MLP or other module types is untested."

---

### L-07: Only LoRA adapter type — UNACKNOWLEDGED

**Severity: LOW**

**Claim (throughout):** The bridge is inserted into a LoRA decomposition.

**The limitation:** No other PEFT methods (prefix tuning, adapters, IA3, etc.) are tested with multi-channel bridge parameterization. The finding is specific to the LoRA BA decomposition.

**Assessment:** The paper's title and framing are explicitly about LoRA. This is not misleading --- the scope is clearly stated. A minor note in Limitations would preempt a reviewer question.

**Recommendation:** Optional. The "Single bridge architecture" limitation could be expanded: "...alternative parameterizations (sparse, symmetric, orthogonal) may produce different outcomes, and the bridge concept has not been tested with non-LoRA PEFT methods."

---

### L-08: Steersman has many hand-tuned thresholds — sensitivity analysis absent — UNACKNOWLEDGED

**Severity: HIGH**

**Claims (Table 9, Appendix, lines 1216--1242):** 13 Steersman hyperparameters are listed, including:
- Initial contrastive weight: 0.1
- Max contrastive weight: 0.5
- Initial spectral weight: 0.05
- Fiedler decline threshold: -0.001
- Stagnation threshold: 0.02
- Deviation growth threshold: 0.05
- Spectral boost gain: 10.0
- Contrastive increment: 0.02
- Stability dampen floor: 0.8
- Fiedler target tracking rate: 0.1

**The limitation:** No sensitivity analysis is reported for any of these parameters. The paper demonstrates that block-diagonal structure emerges under one specific set of Steersman hyperparameters. It is unknown whether the result is robust to (for example) doubling the contrastive weight, changing the stagnation threshold, or modifying the spectral boost gain. The Steersman is described as "simple" and "reactive," but 13 hand-tuned parameters is a substantial design space.

The initialization independence experiments (identity, geometric, corpus-coupled) test robustness to bridge initialization, not to Steersman hyperparameters. These are different axes of robustness.

**Existing coverage:** Not acknowledged anywhere.

**Assessment:** This is a significant omission. A reviewer will ask: "Is the block-diagonal result an artifact of specific threshold tuning? Would different thresholds produce different topologies, or is the result robust?" The paper cannot answer this question from its current experimental evidence.

**Recommendation:** Add to Limitations: "The Steersman's 13 control parameters (Table~\\ref{tab:steersman}) are hand-tuned; no sensitivity analysis has been conducted. The robustness of block-diagonal emergence to variations in feedback thresholds, gain constants, and weight bounds is untested."

---

### L-09: Block-diagonal detection threshold sensitivity — UNACKNOWLEDGED

**Severity: MEDIUM**

**Claim (line 533--535):**
> "We classify a bridge as block-diagonal if rho > 10 and every cross-planar entry has absolute value below 10^{-3}."

**Appendix (lines 1318--1323):** Repeats the thresholds and notes they are "conservative."

**The limitation:** The 100% BD finding is a binary classification that depends on two threshold choices (rho > 10 and max cross-planar < 10^-3). While the paper notes that typical converged values far exceed these thresholds (rho > 60,000 and max cross-planar < 10^-5), the binary metric creates a cliff edge that could be misleading if intermediate states exist in other experimental conditions. The paper does not report what happens at stricter thresholds (e.g., rho > 1000) or at earlier checkpoints.

**Assessment:** The actual effect is so extreme (70,000:1 vs ~1:1) that threshold sensitivity is unlikely to matter in practice. The paper's own data (100% at rho>10 vs the actual values being 60,000+) effectively demonstrates insensitivity. This is a minor concern, but a single sentence acknowledging the threshold choice would preempt the question.

**Recommendation:** The appendix partially addresses this by calling the thresholds "conservative." Could note in Limitations or in the detection criteria appendix: "The binary classification is insensitive to threshold choice within the tested data; applying rho > 1{,}000 or rho > 10{,}000 as thresholds yields identical classification results."

---

### L-10: "Scale-consistent" still carries weight beyond N=2 evidence — SURVIVING OVERCLAIM (partial)

**Severity: MEDIUM**

**Claim (Section 5 title):** "Scale Consistency"
**Claim (contribution 6):** "Scale consistency from 1.1B to 7B parameters."
**Claim (lines 205--207):** "Cross-layer correlation Fiedler converges to ~0.10 for cybernetic text models (0.102 at Qwen 7B, 0.101 at TinyLlama)---a scale-consistent structural metric."

**Round 2 action (F-02C-06):** "scale-invariant" downgraded to "scale-consistent."

**Assessment:** "Scale-consistent" is a significant improvement over "scale-invariant." However, two data points (0.102 and 0.101) converging to similar values is pattern-matching at N=2. The Correlation Fiedler convergence to ~0.10 *could* be coincidence, a property of the Steersman hyperparameters, or a genuine structural invariant --- but N=2 cannot distinguish these hypotheses. The phrase "scale-consistent" in a section title and contribution list implicitly claims more generality than two datapoints support.

The Limitations section acknowledges the scale range issue but frames it as "we need a cybernetic 14B experiment" rather than "N=2 is fundamentally insufficient for scale claims."

**Recommendation:** This was already addressed in Round 2. The remaining risk is modest. One option: change the Section 5 title to "Cross-Scale Comparison" (neutral description) rather than "Scale Consistency" (which implies a finding).

---

### L-11: "Architecture suggests a structural invariant" — from N=2

**Severity: LOW**

**Claim (line 954--955):**
> "suggesting an architectural invariant"

**Assessment:** Round 2 already flagged this (F-02C-12, accepted, hedged with "suggesting"). The current phrasing "suggesting" is appropriately calibrated. No further action needed.

---

### L-12: Causal language — "produces," "causes" — residual instances

**Severity: LOW**

**Claim (abstract, line 56--57):** "the Steersman at n = 6 produces 100% block-diagonal bridge structure"

**Claim (line 204):** "confirming that the effect is caused by the Steersman"

**Claim (line 1016):** "consistent with the Steersman as the causal factor"

**Assessment:** "Produces" in the abstract is defensible --- the Steersman is present, block-diagonal structure appears; the Steersman is absent, it does not. The causal direction is clear within the experimental design (the Steersman is an intervention). Line 204 uses "caused by" which is strong but defensible given the controlled comparison (same model, same data, Steersman on/off). Line 1016 uses "consistent with...causal" which is appropriately hedged. Round 2 already addressed the Holly causal framing (F-02C-03).

The remaining concern: confounds within the Steersman (is it the contrastive loss specifically, or the spectral loss, or the feedback cycle, or the combination?) are partially addressed by the ablation (contrastive vs spectral-only). The confound between n=6 and contrastive availability is acknowledged in Limitations.

No further action needed beyond what Round 2 addressed.

---

### L-13: Only causal LM fine-tuning tested (except Holly) — PARTIALLY ACKNOWLEDGED

**Severity: MEDIUM**

**Existing limitation (line 1077--1080):**
> "Single task family. All experiments use instruction-following (alpaca-cleaned). The block-diagonal structure may depend on task type, dataset, or the relationship between task complexity and channel count."

**Assessment:** This is acknowledged. However, the limitation could be more specific: the paper tests only causal language modeling. Other fine-tuning tasks (classification, sequence-to-sequence, RLHF), other datasets (beyond instruction-following), and other modalities (vision, multimodal) are all untested. The existing wording is adequate but could be slightly expanded.

**Recommendation:** The existing acknowledgment is sufficient. No change needed.

---

## Paper 2 Limitations

### L-14: Only FCC and SC lattices compared — PARTIALLY ACKNOWLEDGED

**Severity: MEDIUM**

**Existing limitation (lines 847--851):**
> "No BCC baseline. The body-centered cubic lattice (8-connected, Voronoi cell = truncated octahedron) is the natural intermediate between SC (6) and FCC (12). Its omission limits the ability to attribute effects to connectivity alone versus isotropy."

**Assessment:** This is well acknowledged. Paper 2 identifies BCC as the missing control and explains why it matters. The Future Work section (5.3) also discusses it. However, no HCP (hexagonal close-packed) or other lattice types are mentioned. Since BCC is the most important missing comparison, this is adequate.

**Recommendation:** Adequate as written.

---

### L-15: Corpus values are proprietary — reproducibility impact — PARTIALLY ACKNOWLEDGED

**Severity: MEDIUM**

**Existing limitation (lines 852--859):**
> "Corpus provenance. The corpus distribution has documented internal structure...a reader who questions the corpus's representativeness can verify all lattice-scale findings using only the random and power-law distributions, which show the same amplification gradient (though at lower magnitude)."

**Assessment:** This is handled well. The paper explicitly tells readers how to verify the core findings without the corpus values. The single-cell experiments (which depend on specific corpus values) are correctly framed as less reproducible. The limitation is honestly stated.

**Recommendation:** Adequate as written.

---

### L-16: Simulated annealing for TV optimization — local optimum risk

**Severity: LOW**

**Claim (line 528--530):** "Optimization uses simulated annealing with 100 restarts of 5,000 iterations each."

**The limitation:** Simulated annealing does not guarantee global optimality. With 100 restarts, the chance of finding the global optimum is high but not certain. The "optimal" TV value (8.12) may not be the true global minimum.

**Assessment:** The paper does not claim global optimality. The SA result is used comparatively (RD TV vs random graph TV), and the qualitative finding (RD bipartite constraint restricts rather than optimizes) would hold even with a slightly suboptimal assignment. This is a minor concern.

**Recommendation:** No action needed. The finding is robust to the SA approximation.

---

### L-17: Paper 2 consensus inversion — acknowledged but unresolved

**Severity: LOW (already acknowledged)**

**Existing limitation (lines 843--845):**
> "The transition point and its dependence on weight distribution are not resolved."

**Assessment:** Adequately acknowledged. Future work (Section 5.5) proposes intermediate-scale experiments.

---

## Summary Table

| ID | Severity | Paper | Finding | Acknowledged? |
|----|----------|-------|---------|---------------|
| L-01 | HIGH | P3 | Single seed (42) for all experiments | **NO** |
| L-02 | MEDIUM | P3 | N=2 cybernetic scales for "scale-consistent" | Partially (frames as missing 14B, not as N=2 insufficiency) |
| L-03 | MEDIUM | P3 | No formal statistical tests for val loss equivalence | **NO** (Round 2 noted, not added to Limitations) |
| L-04 | HIGH | P3 | Channel ablation only at 1.1B | **NO** |
| L-05 | MEDIUM | P3 | Holly differs in 4 dimensions from other experiments | Partially (body text reframed; Limitations section does not echo) |
| L-06 | LOW | P3 | Only QKV/O target modules | **NO** (minor) |
| L-07 | LOW | P3 | Only LoRA, no other PEFT methods | **NO** (minor, scope is clear) |
| L-08 | HIGH | P3 | 13 Steersman hyperparameters with no sensitivity analysis | **NO** |
| L-09 | MEDIUM | P3 | BD detection thresholds not tested for sensitivity | Partially (appendix notes "conservative") |
| L-10 | MEDIUM | P3 | "Scale-consistent" section title implies more than N=2 supports | Partially (Round 2 fixed "invariant"; section title still strong) |
| L-11 | LOW | P3 | "Structural invariant" from N=2 | Yes (hedged with "suggesting") |
| L-12 | LOW | P3 | Residual causal language | Yes (defensible given controlled design) |
| L-13 | MEDIUM | P3 | Only causal LM tested | Yes (acknowledged) |
| L-14 | MEDIUM | P2 | Only FCC and SC compared | Yes (BCC acknowledged as gap) |
| L-15 | MEDIUM | P2 | Corpus values proprietary | Yes (mitigation described) |
| L-16 | LOW | P2 | SA local optimum risk | Minor (no action needed) |
| L-17 | LOW | P2 | Consensus inversion unresolved | Yes (acknowledged) |

## Priority Actions

Three unacknowledged HIGH-severity limitations should be added to Paper 3's Limitations section:

1. **L-01 (single seed):** Every experiment uses seed=42. Specific numerical claims (half-life, peak ratio, per-module hierarchy) are single-seed estimates without confidence intervals.

2. **L-04 (ablation scale):** The channel-count ablation is conducted at 1.1B only. The n=6 bifurcation is untested at 7B or larger.

3. **L-08 (Steersman sensitivity):** 13 hand-tuned Steersman parameters with no sensitivity analysis. Robustness of block-diagonal emergence to threshold variation is unknown.

Two MEDIUM-severity items deserve consideration:

4. **L-03 (statistical tests):** The "task performance is orthogonal" claim is a null result without formal tests. Consider noting that performance equivalence is demonstrated by effect size rather than hypothesis testing.

5. **L-10 (section title):** Consider changing "Scale Consistency" section title to "Cross-Scale Comparison" for neutrality.

Paper 2's limitations section is well-written and honest. No additions recommended.

---

*Agent 5C, Round 5. Limitations honesty audit complete.*
