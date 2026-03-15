# Round 4 — Hub Validation

> **Date:** 2026-03-15
> **Scope:** Figures + Reproducibility
> **Agents:** 4A (Figure Audit), 4B (Reproducibility), 4C (Missing Figures)

---

## Validation Method

Hub independently verified each finding against the tex source files,
figure directories, and raw data. Severity assessments are hub's own.

---

## Agent 4A: Figure Audit — 10 findings

### F-4A-01: Five Paper 3 figures never referenced in text
- Agent severity: MAJOR
- **Hub verdict: ACCEPTED — FIXED**
- Verification: Grepped `rhombic-paper3.tex` for `\ref{fig:heatmaps}`,
  `\ref{fig:ablation}`, `\ref{fig:h2eigen}`, `\ref{fig:perlayer}`,
  `\ref{fig:permodule}` — zero hits confirmed. Added `Figure~\ref{}`
  citations at the appropriate prose locations for all five figures.

### F-4A-02: Figure file numbering swapped for Figs 6-7
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — DEFERRED**
- Verification: Confirmed `fig:fiedler` (LaTeX Fig 6) uses file
  `fig7-fiedler-saturation.png` and `fig:h2eigen` (Fig 7) uses
  `fig6-h2-eigenvalues.png`. Cosmetic only — LaTeX `\ref` system
  is unaffected. File rename deferred to avoid breaking any external
  references.

### F-4A-03: Figure 5 caption "0.22%" vs text "0.17%" scope mismatch
- Agent severity: MAJOR
- **Hub verdict: ACCEPTED — FIXED**
- Verification: Caption says "within 0.22%" (all four n values). Body
  text says "at most 0.17% across n ∈ {3,4,6}." Both correct for their
  stated scopes. Added clarification after the 0.17% text: "Including
  $n = 8$ (0.4024), the full range is 0.22\%."

### F-4A-04: Figure-mapping doc uses obsolete filenames
- Agent severity: MAJOR
- **Hub verdict: ACCEPTED — DEFERRED**
- `09_figures.md` is an internal planning document not included in the
  compiled paper. Stale filenames documented but not fixed this round.

### F-4A-05: Mapping doc wrong peak ratio (82,154 vs 82,854)
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — DEFERRED**
- Same as F-4A-04: internal doc, not compiled. Noted for future cleanup.

### F-4A-06: Mapping doc wrong Fiedler range (0.085 vs 0.092)
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — DEFERRED**
- Internal doc. Lower bound was already corrected in the tex (R1 fix).

### F-4A-07: Mapping doc conflates init and ablation val losses
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — DEFERRED**
- Internal doc only.

### F-4A-08: Paper 2 file numbering offset from LaTeX fig numbering
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — NO ACTION**
- Deliberate multi-paper convention. File numbers (3-6) reflect their
  position in the shared figure series across Papers 1 and 2. LaTeX
  `\ref` system handles this correctly. Not an error.

### F-4A-09: Extra Paper 1 figures in shared `figures/` directory
- Agent severity: POLISH
- **Hub verdict: NOTED — NO ACTION**
- `fig1-voronoi-cells` and `fig2-graph-benchmarks` belong to Paper 1.
  Shared directory is intentional.

### F-4A-10: Supplementary figures listed but missing from disk
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — DEFERRED**
- 6 supplementary figures (S1-S6) listed in `09_figures.md` do not
  exist on disk. If supplementary material is added later, these would
  need to be generated. Documented for future work.

---

## Agent 4B: Reproducibility — 14 gaps

### 4B-Gap-1/2/3: Holly Battery under-specification (optimizer, data, modules)
- Agent severity: MAJOR × 3
- **Hub verdict: ACCEPTED — FIXED (partially)**
- Added Holly Battery configuration note to Paper 3 appendix specifying:
  Prodigy optimizer (d0 = 10^-4, weight decay 0.01), video diffusion
  training data (proprietary), attention layer targets in Wan 2.1
  transformer blocks, 10 epochs (~600 steps at BS 1, GA 4), no Steersman.
- The training data remains proprietary and cannot be further specified
  in the paper.

### 4B-Gap-4/5/6: Steersman gain constants, decay dynamics, spectral target
- Agent severity: MINOR × 3
- **Hub verdict: ACCEPTED — FIXED**
- Added 5 rows to the Steersman parameters table in Appendix A:
  spectral boost gain (10.0×|trend|), contrastive increment (0.02),
  stability dampen floor (0.8), spectral decay target (0.5× base),
  Fiedler target tracking rate (0.1).

### 4B-Gap-7: AdamW betas not specified
- Agent severity: MINOR
- **Hub verdict: NOTED — NO ACTION**
- PyTorch defaults (0.9, 0.999) are universally assumed when not stated.
  Standard practice.

### 4B-Gap-8: Tokenizer padding strategy not specified
- Agent severity: MINOR
- **Hub verdict: NOTED — NO ACTION**
- `pad_token = eos_token` is standard for decoder-only LMs. Not worth
  a paper line.

### 4B-Gap-9: Corpus-coupled initialization proprietary
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — NO ACTION**
- Paper proves initialization is cosmetic (convergence within 200 steps).
  Identity and geometric inits are public. The proprietary dependency is
  acknowledged in the code and does not affect core claims.

### 4B-Gap-10: Reproduction instructions minimal
- Agent severity: MINOR
- **Hub verdict: NOTED — DEFERRED**
- Could add a `reproduce.sh` script. Not a paper change.

### 4B-Gap-11: Model revisions not pinned
- Agent severity: POLISH
- **Hub verdict: NOTED — NO ACTION**
- Standard practice. Models are specified by name.

### 4B-Gap-12: Train/val split not specified
- Agent severity: POLISH
- **Hub verdict: ACCEPTED — FIXED**
- Added "alpaca-cleaned (90/10 train/val split)" to hyperparameters table.

### 4B-Gap-13: I Ching mapping not in paper
- Agent severity: POLISH
- **Hub verdict: NOTED — NO ACTION**
- Initialization proved cosmetic.

### 4B-Gap-14: Half-life fit model not formally specified
- Agent severity: POLISH
- **Hub verdict: NOTED — NO ACTION**
- "Exponential decay" + "least-squares" is sufficient for the claim.

---

## Agent 4C: Missing Figures — 10 findings

### F-4C-01: Architecture schematic missing
- Agent severity: MODERATE
- **Hub verdict: ACCEPTED — DEFERRED**
- The spec exists in `section2_architecture.tex` lines 194-233.
  Creating the TikZ figure is a dedicated task. This is the highest-
  priority deferred item and a likely reviewer complaint.

### F-4C-02/05: Exponential decay and linear growth have no direct figure
- Agent severity: MINOR × 2
- **Hub verdict: ACCEPTED — DEFERRED**
- Would be a supplementary figure. The ratio plot (Fig 1) and
  dismantling plot (Fig 3) cover the qualitative story. Quantitative
  fit parameters are stated in text.

### F-4C-03: Three-phase Fiedler trajectory visual coverage
- Agent severity: PASS
- **Hub verdict: PASS**

### F-4C-04: Correlation Fiedler has no figure
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — DEFERRED**
- Low priority. Three numbers in prose.

### F-4C-06: Holly Battery has no visual evidence
- Agent severity: MINOR
- **Hub verdict: ACCEPTED — DEFERRED**
- A Holly heatmap would strengthen the 14B comparison. Deferred.

### F-4C-07: Table 1 silently omits 2 distributions at scale 8,000
- Agent severity: MINOR (confirming F-01-20)
- **Hub verdict: ACCEPTED — FIXED**
- Added footnote under both Tables 1 and 5 in Paper 2 explaining:
  "Random and power-law distributions were not computed at scale 8,000
  due to the O(n³) cost of Laplacian eigendecomposition."

### F-4C-08: Filenames fig6/fig7 swapped vs compiled order
- Agent severity: MINOR (duplicate of F-4A-02)
- **Hub verdict: ACCEPTED — DEFERRED**
- Same as F-4A-02.

### F-4C-09: 5 of 9 figures lack explicit `\ref` citations
- Agent severity: POLISH (duplicate of F-4A-01)
- **Hub verdict: ACCEPTED — FIXED (via F-4A-01)**

### F-4C-10: Paper 2 figure references clean
- Agent severity: PASS
- **Hub verdict: PASS**

---

## Summary

| Category | Findings | Fixed | Deferred | Noted | Pass |
|----------|----------|-------|----------|-------|------|
| 4A: Figures | 10 | 2 | 6 | 1 | 1 |
| 4B: Reproducibility | 14 | 5 | 1 | 6 | 2 |
| 4C: Missing Figures | 10 | 2 | 5 | 0 | 3 |
| **Total** | **34** | **9** | **12** | **7** | **6** |

**Papers recompiled:** Both Paper 2 and Paper 3 compile clean with
zero undefined references after all fixes.

**Deferred items requiring future work:**
1. F-4C-01: Architecture schematic (TikZ) — HIGHEST PRIORITY
2. F-4A-02/4C-08: Rename fig6/fig7 files — COSMETIC
3. F-4A-04-07: Update 09_figures.md — INTERNAL DOC
4. F-4A-10: Generate supplementary figures if needed
5. F-4C-02/04/05/06: Additional supplementary figures
6. 4B-Gap-10: Reproduction script
