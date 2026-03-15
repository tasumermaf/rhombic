# Round 2 Hub Validation — Internal Consistency

**Date:** 2026-03-15
**Round focus:** Abstract-body consistency, cross-references, claim-evidence alignment

---

## Agent 2A: Abstract-Body Consistency — 15 findings

### Independent Validation

**F-2A-01 (15 experiments):** CONFIRMED. Table 1 has 13 rows. exp1a-e as group = 13; expanded = 17. Neither yields 15. **Fixed → 13.**

**F-2A-02 (exp2.5 = 1% BD):** CONFIRMED via raw data. BRIDGE_STRUCTURE_ANALYSIS.md and BRIDGE_BLOCK_DIAGONAL_FINDING.md both show 0% BD for exp2.5. Table 1's "1%" was incorrect. **Fixed → 0%.**

**F-2A-03 (six non-cybernetic):** CONFIRMED ambiguous. Removed specific count from intro text.

**F-2A-04 (abstract 0.17% for n=3 vs n=6):** CONFIRMED. 0.17% is the n ∈ {3,4,6} range delta, not n=3 vs n=6 specifically. **Fixed: abstract now specifies the correct comparison.**

**F-2A-05 (0.058% stale):** **INDEPENDENTLY VERIFIED.** Compared H-ch3/results.json (n=3, 100 checkpoints) against H-ch6/metrics_hermes.csv (n=6, 100 checkpoints):
- Max absolute diff: 0.000622 at step 8600
- Max relative delta: 0.155% at step 8600
- Final relative delta: 0.123%
- The 0.058% came from DRAFT_INIT_OVERWRITE_SECTION.md's early 40-checkpoint comparison at step 4000 (delta = 0.000237 = 0.057%)
- **Fixed → 0.16% (3 locations).**

**F-2A-06 (0.4010 val loss range):** CONFIRMED. 0.4010 is C-002 (init comparison), not ablation H3 (0.4015). **Fixed → 0.4015.**

**F-2A-10 (0.12% init convergence):** CONFIRMED. (0.4015 − 0.4010)/0.4010 = 0.1246%. **Fixed → 0.13%.**

**F-2A-12 (H4 val loss):** CONFIRMED from H-ch8-results-hermes.json step 10000: val_loss = 0.40239. **Fixed → 0.4024 in Table 1.**

### Dispositions

| Finding | Verdict | Action |
|---------|---------|--------|
| F-2A-01 | CONFIRMED | Fixed: 15 → 13 |
| F-2A-02 | CONFIRMED | Fixed: 1% → 0% |
| F-2A-03 | CONFIRMED | Count removed |
| F-2A-04 | CONFIRMED | Fixed: correct comparison |
| F-2A-05 | **INDEPENDENTLY VERIFIED** | Fixed: 0.058% → 0.16% |
| F-2A-06 | CONFIRMED | Fixed: 0.4010 → 0.4015 |
| F-2A-07 | ACCEPTED | Not fixed — verifiable from source |
| F-2A-08 | ACCEPTED | Not fixed — minor |
| F-2A-09 | CONFIRMED | Fixed: four model scales |
| F-2A-10 | CONFIRMED | Fixed: 0.12% → 0.13% |
| F-2A-11 | CONFIRMED | Fixed: "peak of" added |
| F-2A-12 | CONFIRMED | Fixed: H4 val loss added |
| F-2A-13 | DEFERRED | Data may not exist |
| F-2A-14 | CONFIRMED | Fixed via F-2A-02 |
| F-2A-15 | CONFIRMED | Fixed: converged value added |

---

## Agent 2B: Cross-References — 13 findings

### Dispositions

Both CRITICAL findings (F-2B-01 wrong bib title, F-2B-02 citation to nonexistent content) were fixed in the first pass. F-2B-12 (Fiedler citation) also fixed. All 7 PASS findings confirmed clean. Orphan bib entries noted but harmless.

---

## Agent 2C: Claim-Evidence Alignment — 15 findings

### Independent Validation

**F-02C-13 (IC/RD ratio):** **INDEPENDENTLY VERIFIED.** Read DRAFT_INIT_OVERWRITE_SECTION.md:
- Line 95: header says "group totals, 3 pairs each"
- Line 99: Step 0: RD co-planar total = 0.020, IC complementary total = 0.029
- Lines 121-123: Individual IC pairs sum to 0.028 ≈ 0.029 (confirming "group total")
- Paper's 0.010 for RD matches neither group total (0.020) nor per-pair avg (0.0067)
- Ratio should be 0.029/0.020 = 1.45:1, matching the 1.4:1 at line 556
- **Fixed: now uses group totals (0.029/0.020, 1.4:1).**

**F-02C-01 (necessary and sufficient):** CONFIRMED. Limitations section (lines 1069-1074) explicitly acknowledges the confound. Added "at n=6" qualifier in 5 locations.

**F-02C-02 (universal):** CONFIRMED. 6 experiments / 2 scales / 1 task ≠ universal. Changed to "robust" in 3 locations.

**F-02C-03 (Holly causal):** CONFIRMED. Holly differs in architecture, task, dataset, AND optimizer. Reframed to position Qwen exp2/exp3 as the stronger comparison.

**F-02C-04 (section header):** CONFIRMED. Softened to "Emerges Only When the Geometric Prior Is Active."

**F-02C-05 (circularity):** ACCEPTED but not explicitly addressed — the other fixes (reframing "discovers" context, softening "necessary and sufficient") implicitly address the concern.

### Dispositions

| Finding | Verdict | Action |
|---------|---------|--------|
| F-02C-01 | CONFIRMED | Fixed: "at n=6" in 5 locations |
| F-02C-02 | CONFIRMED | Fixed: "universal" → "robust" |
| F-02C-03 | CONFIRMED | Fixed: Holly reframed |
| F-02C-04 | CONFIRMED | Fixed: header softened |
| F-02C-05 | ACCEPTED | Implicit via other fixes |
| F-02C-06 | CONFIRMED | Fixed: "invariant" → "consistent" |
| F-02C-07 | CONFIRMED | Fixed: "independent" → "insensitive" |
| F-02C-08 | CONFIRMED | Fixed: "absolute" → "robust" |
| F-02C-09 | ACCEPTED | Wording adequate |
| F-02C-10 | CONFIRMED | Fixed: "exactly" → "effectively" |
| F-02C-11 | NOTED | Effect sizes make tests unnecessary |
| F-02C-12 | ACCEPTED | Hedged with "suggesting" |
| F-02C-13 | **INDEPENDENTLY VERIFIED** | Fixed: 0.029/0.020, 1.4:1 |
| F-02C-14 | NOTED | Defensible framing |
| F-02C-15 | NOTED | Tangential |

---

## Round 2 Summary

| Metric | Count |
|--------|-------|
| Total findings | 43 |
| CRITICAL | 2 (both fixed) |
| MAJOR | 6 (all fixed) |
| MODERATE | 6 (5 fixed, 1 noted) |
| MINOR | 10 (7 fixed, 3 noted) |
| LOW/POLISH/INFO/PASS | 19 |
| Fixed | 26 |
| Deferred | 1 (F-2A-13) |
| Rejected | 0 |
| Independently verified | 3 (F-2A-05, F-02C-13, F-2A-02) |

**Key independent verifications:**
1. 0.058% → 0.16%: derived from raw H-ch3/H-ch6 JSON/CSV data (100 matched checkpoints)
2. IC/RD 2.9:1 → 1.4:1: confirmed from DRAFT_INIT_OVERWRITE_SECTION.md source table headers
3. exp2.5 1% → 0%: confirmed from both BRIDGE_STRUCTURE_ANALYSIS.md and BRIDGE_BLOCK_DIAGONAL_FINDING.md

**Compilation:** Clean compile, 21 pages, no warnings.
