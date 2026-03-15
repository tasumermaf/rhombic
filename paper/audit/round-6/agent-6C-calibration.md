# Round 6 — Agent 6C: Claims Calibration

**Auditor:** Agent 6C (fresh-eyes pass)
**Scope:** All claims in rhombic-paper3.tex, calibrated against experimental evidence
**Date:** 2026-03-15
**Prior calibration context:** Rounds 2 and 5 already revised "universal" to "consistent," "scale-invariant" to "scale-consistent," "independent of channel count" to "insensitive to," added qualifiers to "necessary and sufficient" and "exactly 3," removed "the fact that," added single-seed/ablation-scale/Steersman-sensitivity limitations, and documented Holly confounds.

---

## Findings

### F-6C-01: "unique attractor" in Conclusion
- **Severity:** CRITICAL
- **Location:** Line ~1165
- **Current text:** "it emerges as the unique attractor of a feedback loop"
- **Issue:** "Unique" is a formal dynamical systems claim meaning no other attractors exist under these dynamics. The paper has tested exactly one geometric prior (RD face-pair) and observed one resulting topology (3-block BD). No exploration of the attractor landscape has been conducted — no alternative contrastive geometries, no bifurcation analysis, no basin-of-attraction mapping. A reviewer trained in dynamical systems will immediately flag this as unsubstantiated. The paper cannot distinguish "unique attractor" from "the only attractor we looked for."
- **Suggested revision:** "it emerges as a robust attractor of a feedback loop"

### F-6C-02: "natural representational structure" in Discussion
- **Severity:** CRITICAL
- **Location:** Line ~1071
- **Current text:** "indicating that the RD coordinate system is compatible with the adapter's natural representational structure"
- **Issue:** The word "natural" does real work here — it implies the RD geometry aligns with something intrinsic to the adapter, independent of the loss function. The evidence shows that the bridge converges to the topology encoded by the contrastive loss. Without the contrastive loss, no geometric preference emerges (ratio ~1:1). The experiment has not tested whether the bridge would equally "accept" a random 3-block partition, an alternative polyhedron's face-pair structure, or any other topology encoded via contrastive loss. "Natural" is unsupported; "compatible under these training conditions" is what the evidence shows.
- **Suggested revision:** "indicating that the RD coordinate system is compatible with the adapter's representational structure under cybernetic training"

### F-6C-03: "rejecting arbitrary patterns" in Discussion
- **Severity:** MAJOR
- **Location:** Line ~1069-1071
- **Current text:** "the bridge accepts the RD geometry (100% BD) while rejecting arbitrary patterns (0% BD without contrastive loss)"
- **Issue:** "Rejecting arbitrary patterns" implies that alternative structured patterns were offered and rejected. The actual evidence is that without any contrastive loss, the bridge develops no topological preference. This is "no structure without structured feedback," not "rejection of arbitrary patterns." The corpus-coupled initialization (I Ching) does test one opposing pattern, but the claim generalizes far beyond that single case.
- **Suggested revision:** "the bridge accepts the RD geometry (100\% BD) while developing no topological preference in its absence (0\% BD without contrastive loss)"

### F-6C-04: "latent geometric preferences" in Conclusion
- **Severity:** MAJOR
- **Location:** Line ~1187
- **Current text:** "These results suggest that multi-channel LoRA adapters have latent geometric preferences that standard training does not surface."
- **Issue:** "Latent" implies the preferences pre-exist in the adapter and merely need to be revealed. The experimental evidence shows the opposite: without the contrastive loss, zero geometric preference is observed (ratio ~1:1 across all non-cybernetic experiments). The geometric preference is created by the loss, not discovered within the adapter. "Suggest" is good hedging, but "latent" undermines it by presupposing the conclusion. A reviewer will ask: if the preference is latent, why is it exactly zero in every non-cybernetic experiment?
- **Suggested revision:** "These results suggest that multi-channel LoRA adapters are receptive to geometric structure when it is encoded in the training objective"

### F-6C-05: "confirming that the effect is caused by the Steersman" via Holly Battery
- **Severity:** MAJOR
- **Location:** Line ~203-204
- **Current text:** "confirming that the effect is caused by the Steersman, not by model scale or architecture"
- **Issue:** Holly differs from the cybernetic experiments in model architecture (video diffusion vs. causal LM), task (video generation vs. instruction following), dataset (proprietary vs. alpaca-cleaned), optimizer (Prodigy vs. AdamW), training duration (600 steps vs. 10K), and presence/absence of Steersman. With six confounded variables, Holly cannot "confirm" causality — it is *consistent with* the Steersman as the causal factor, which is the more careful phrasing already used at line ~1022 ("provides additional evidence consistent with"). The paper should use the same calibration in both locations.
- **Suggested revision:** "providing evidence consistent with the Steersman, not model scale or architecture, as the causal factor"

### F-6C-06: "emerges identically" across scales
- **Severity:** MAJOR
- **Location:** Line ~232 and line ~947
- **Current text:** (line 232) "Block-diagonal structure emerges identically at TinyLlama 1.1B and Qwen 7B" and (line 947) "Block-diagonal structure appears identically at 1.1B (TinyLlama) and 7B (Qwen2.5)"
- **Issue:** The final co-planar/cross-planar ratios differ by approximately 4x between scales: Qwen 7B achieves 18,248:1 while TinyLlama achieves 64,000-71,000:1. The paper attributes this to a code version difference (line 950-952), which may be true, but this is an uncontrolled explanation — the reader cannot verify it. "Identically" implies the same quantitative outcome. A reviewer comparing Table 1 numbers will immediately see the discrepancy.
- **Suggested revision:** "Block-diagonal structure emerges consistently at both 1.1B (TinyLlama) and 7B (Qwen2.5)" or "...emerges at both scales"

### F-6C-07: "Initialization is cosmetic"
- **Severity:** MAJOR
- **Location:** Line ~162
- **Current text:** "Initialization is cosmetic."
- **Issue:** "Cosmetic" implies zero functional consequence — purely superficial. The three initializations converge to final states with 11% ratio difference (64,168:1 vs. 71,337:1) and 0.025% validation loss difference (at 10K steps). More importantly, C-001 was only run to 4K steps with validation loss 0.4178, vs. 0.4010-0.4011 for C-002/C-003 at 10K steps — the comparison is confounded by training duration. Three initialization strategies is a reasonable test, but "cosmetic" is dismissive language that overstates the strength of a three-strategy experiment. Additionally, N=3 initialization strategies is a small sample and vastly more adversarial strategies can be imagined (e.g., large-magnitude anti-co-planar initialization, random orthogonal, scaled identity).
- **Suggested revision:** "Initialization effects are negligible" or "The final attractor is initialization-insensitive"

### F-6C-08: "identical validation loss" (3 instances)
- **Severity:** MINOR
- **Location:** Lines ~70, ~194, ~923
- **Current text:** (line 70) "n = 3 achieves identical validation loss to n = 6"; (line 194) "Both channel counts produce identical validation loss"; (line 923) "achieves identical task performance"
- **Issue:** The maximum delta is 0.16% (stated parenthetically). "Identical" is a mathematical claim (delta = 0). The parenthetical quantification corrects the text, but the word itself is technically wrong. This is a pattern — three separate instances of the same overclaim.
- **Suggested revision:** "matched" or "indistinguishable" in all three locations. The parenthetical quantification should be retained.

### F-6C-09: "exactly 3 active degrees of freedom" (inconsistency with abstract)
- **Severity:** MINOR
- **Location:** Lines ~193 and ~915
- **Current text:** (line 193) "leaving exactly 3 active degrees of freedom"; (line 915) "reduce to exactly 3 active dimensions"
- **Issue:** The abstract (line ~69) correctly says "effectively 3" — the 12 suppressed entries are at the computational floor (~10^-5), not mathematical zero. The introduction and Section 4.3 use "exactly," creating an internal inconsistency. A reviewer will notice the abstract says one thing and the body says another.
- **Suggested revision:** Change both instances to "effectively 3" to match the abstract's calibration.

### F-6C-10: "convergence proof" in figure caption
- **Severity:** MINOR
- **Location:** Line ~717
- **Current text:** "Four-panel initialization convergence proof"
- **Issue:** "Proof" in the mathematical sense requires formal demonstration. This is empirical evidence of convergence across three initializations, not a proof. In machine learning papers, "proof" is typically reserved for theoretical results. A figure caption showing experimental results should not use this term.
- **Suggested revision:** "Four-panel initialization convergence evidence" or "Four-panel initialization convergence comparison"

### F-6C-11: "suggesting an architectural invariant" for module hierarchy
- **Severity:** MINOR
- **Location:** Line ~960-961
- **Current text:** "The ordering k > v > q > o is consistent across scales (Figure 9), suggesting an architectural invariant."
- **Issue:** "Architectural invariant" is a strong claim from two model scales (and the specific ratios differ substantially between models — o_proj is 19,080:1 at Qwen 7B vs. 9,116:1 at TinyLlama, a 2x difference). "Invariant" implies a mathematical property that holds under all transformations; here it means "we saw the same ranking twice." The hedging word "suggesting" helps, but the term "invariant" is loaded.
- **Suggested revision:** "suggesting a consistent architectural pattern" or "suggesting a shared structural tendency"

### F-6C-12: "There are no exceptions, no near-misses, and no late-onset failures"
- **Severity:** MINOR
- **Location:** Line ~631-632
- **Current text:** "There are no exceptions, no near-misses, and no late-onset failures."
- **Issue:** While factually accurate for the tested conditions, the rhetorical tricolon ("no X, no Y, no Z") reads as advocacy rather than scientific reporting. This is a matter of tone — the triple negation conveys certainty beyond what a single-seed experiment across 6 runs warrants. The data is strong; the prose could let it speak for itself.
- **Suggested revision:** "No exceptions or near-misses were observed." (Simpler, same information, less rhetorical.)

### F-6C-13: "Programmable topology" extrapolation
- **Severity:** MINOR
- **Location:** Line ~1074
- **Current text:** "Programmable topology. The contrastive loss is a topology selector."
- **Issue:** "Programmable topology" as a bold heading implies a demonstrated capability to program arbitrary topologies into the bridge. The paper has tested exactly one topology (RD 3-block). The next sentence acknowledges this is an extension ("A natural extension is to test alternative topologies..."), which is appropriate. But the bold heading "Programmable topology" presents the extrapolation at the same visual weight as demonstrated findings.
- **Suggested revision:** "Toward programmable topology" or "Topology selection"

### F-6C-14: "not a scale-dependent effect" (unverifiable claim about code version)
- **Severity:** MINOR
- **Location:** Line ~950-952
- **Current text:** "The lower ratio compared to TinyLlama runs (64,000--71,000:1) reflects an earlier code version with less aggressive contrastive weighting, not a scale-dependent effect."
- **Issue:** The claim that the 4x ratio difference is due to code version and not scale is presented as fact but is unverifiable by the reader. The experiments were not re-run with the same code version at both scales to control for this variable. A skeptical reviewer might ask why the 7B experiment was not re-run with the current code. The paper should either re-run or acknowledge that this is the likely explanation but cannot be confirmed without a controlled replication.
- **Suggested revision:** "The lower ratio compared to TinyLlama runs (64,000--71,000:1) likely reflects an earlier code version with less aggressive contrastive weighting rather than a scale-dependent effect, though this has not been verified by re-running exp3 with the current code."

### F-6C-15: Necessary/sufficient repetition (4x)
- **Severity:** MINOR (individually), MAJOR (cumulative)
- **Location:** Lines ~63, ~150, ~179, ~1177-1178
- **Current text:** "necessary and sufficient at n = 6" (four instances)
- **Issue:** Each individual instance is qualified with "at n = 6." However, repeating the claim four times — abstract, introduction (twice as bold claims), and conclusion — amplifies it beyond what the experimental design supports. The confound (contrastive loss defined only for n=6, so "necessary" is tested by simultaneously changing channel count AND removing contrastive loss) is acknowledged in the Limitations section, but a reviewer who sees the claim four times before reaching the Limitations will form a strong impression. The claim is defensible but its prominence exceeds its evidential weight.
- **Suggested revision:** Keep the abstract and one introduction instance. Consider reducing the conclusion instance to "the contrastive loss is the active component" or similar, and cross-referencing the confound limitation inline rather than only in the Limitations section.

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| MAJOR | 5 |
| MINOR | 8 |
| **Total** | **15** |

## Overall Assessment

The paper is well-calibrated for approximately 85% of its claims. The core empirical claims — 100% BD at n=6, 0% without Steersman, fast lock-in, channel-count ablation quantitative results — are reported accurately with appropriate specificity.

The overclaiming concentrates in two domains:

**1. Interpretive claims in Discussion and Conclusion (F-6C-01 through F-6C-04).** The paper's core evidence shows that a specific contrastive loss produces a specific bridge topology. The Discussion and Conclusion extrapolate this to claims about the adapter's "natural" structure, "latent" preferences, and "unique" attractors — language that implies the RD geometry was pre-existing and merely revealed, rather than imposed by the loss and accepted by the optimizer. This is the paper's most significant calibration issue. A reviewer will observe that the non-cybernetic experiments show zero geometric preference, which undermines the "discovery" and "latent" framing.

**2. Precision language (F-6C-08, F-6C-09).** "Identical" where the data shows "very close" and "exactly" where the abstract correctly says "effectively." These are minor individually but create a pattern of reaching for absolute language when bounded language would be more accurate. The internal inconsistency between abstract ("effectively 3") and body ("exactly 3") is particularly easy for a reviewer to spot.

**What previous rounds got right.** The R2 and R5 calibrations were effective. "Universal" → "consistent," "scale-invariant" → "scale-consistent," and the qualified "necessary and sufficient at n=6" are all improvements that hold up. The Limitations section (lines 1082-1127) is thorough and honest. The paper would be stronger if the language in the main body matched the Limitations section's epistemic discipline throughout.

**The "Discovers" question.** The title and conclusion use "discovers" where the evidence more precisely supports "produces" or "drives." This is a judgment call — "discovers" is more evocative and arguably defensible given the corpus-coupled experiment (where the bridge demonstrably overrides an opposing topology to converge on BD). However, the framing carries interpretive weight that accumulates through the paper and reaches its strongest form in "natural representational structure" and "latent geometric preferences." If the title keeps "Discovers," the Discussion should be more cautious, not less.
