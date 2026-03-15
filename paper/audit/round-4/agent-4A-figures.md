# Round 4 — Agent 4A: Figure Audit

**Scope:** Every figure in Paper 3 (`rhombic-paper3.tex`) and Paper 2 (`rhombic-paper2.tex`).
Checks: caption-text consistency, file existence, text references, figure-mapping doc accuracy.

**Date:** 2026-03-15

---

## Paper 3 Figures

### Inventory

Paper 3 contains 9 figures (all external PNG files in `figures3/`):

| LaTeX Fig # | Label | File | File Exists | Referenced in Text |
|-------------|-------|------|-------------|--------------------|
| 1 | `fig:cybernetic` | `figures3/fig1-cybernetic-overview` | YES | YES (line 645) |
| 2 | `fig:init` | `figures3/fig2-init-convergence` | YES | YES (line 720) |
| 3 | `fig:dismantling` | `figures3/fig3-corpus-dismantling` | YES | YES (line 728) |
| 4 | `fig:heatmaps` | `figures3/fig4-bridge-heatmaps` | YES | **NO** |
| 5 | `fig:ablation` | `figures3/fig5-channel-ablation` | YES | **NO** |
| 6 | `fig:fiedler` | `figures3/fig7-fiedler-saturation` | YES | YES (line 845) |
| 7 | `fig:h2eigen` | `figures3/fig6-h2-eigenvalues` | YES | **NO** |
| 8 | `fig:perlayer` | `figures3/fig8-per-layer-coupling` | YES | **NO** |
| 9 | `fig:permodule` | `figures3/fig9-per-module-coupling` | YES | **NO** |

---

## Findings

### F-4A-01: Five Paper 3 figures never referenced in text
- Severity: MAJOR
- Location: Paper 3, Figures 4, 5, 7, 8, 9
- Finding: Five of nine figures are included but never referenced via `\ref{}` in the body text. The affected figures are:
  - **Figure 4** (`fig:heatmaps`) — bridge heatmaps, placed after line 771
  - **Figure 5** (`fig:ablation`) — channel ablation, placed after line 826
  - **Figure 7** (`fig:h2eigen`) — H2 eigenvalue divergence, placed after line 879
  - **Figure 8** (`fig:perlayer`) — per-layer coupling, placed after line 958
  - **Figure 9** (`fig:permodule`) — per-module coupling, placed after line 967

  Standard academic practice requires every figure to be introduced in the text with an explicit reference. Unreferenced figures appear orphaned and may be missed by readers or flagged by reviewers.
- Status: NEW

### F-4A-02: Figure file numbering swapped for Figures 6 and 7
- Severity: MINOR
- Location: Paper 3, Figures 6-7 / lines 855, 874
- Finding: LaTeX Figure 6 (Fiedler saturation, `fig:fiedler`) uses file `fig7-fiedler-saturation.png`. LaTeX Figure 7 (H2 eigenvalues, `fig:h2eigen`) uses file `fig6-h2-eigenvalues.png`. The file-numbering prefix contradicts the LaTeX figure numbering. This is cosmetic (compilation works fine) but creates confusion for anyone navigating the `figures3/` directory expecting file numbers to match figure numbers.
- Status: NEW

### F-4A-03: Figure 5 caption says "0.22%" but body text says "0.17%"
- Severity: MAJOR
- Location: Paper 3, Figure 5 caption (line 822) vs. text (line 832)
- Finding: The Figure 5 caption states "all channel counts within 0.22%," which is correct when including n=8 (range 0.4015-0.4024). The body text at line 832 states "at most 0.17% across n in {3, 4, 6}," which is correct when excluding n=8. Both numbers are independently correct for their stated scope, but the discrepancy will confuse readers who compare the caption to the nearest prose. The caption should either match the text's scope (n in {3,4,6}, 0.17%) or the text should note the wider 0.22% range for all four channel counts.
- Status: NEW

### F-4A-04: Figure-mapping doc (`09_figures.md`) uses obsolete filenames
- Severity: MAJOR
- Location: `sections/09_figures.md`, all entries
- Finding: The mapping document references original analysis filenames (e.g., `fig_all_6_cybernetic_definitive.png`, `channel-ablation/fig_channel_ablation_definitive.png`) that do not exist on disk. The actual files in `figures3/` use a renamed convention (`fig1-cybernetic-overview.png`, `fig5-channel-ablation.png`, etc.). The mapping doc is stale and will mislead anyone using it to locate source figures.

  Full mapping of stale names to actual files:

  | 09_figures.md name | Actual file |
  |---|---|
  | `fig_all_6_cybernetic_definitive.png` | `fig1-cybernetic-overview.png` |
  | `fig_init_convergence_final.png` | `fig2-init-convergence.png` |
  | `fig_corpus_dismantling_full.png` | `fig3-corpus-dismantling.png` |
  | `fig_bridge_heatmap_c002_vs_exp2.png` | `fig4-bridge-heatmaps.png` |
  | `channel-ablation/fig_channel_ablation_definitive.png` | `fig5-channel-ablation.png` |
  | `channel-ablation/fig_h2_eigenvalue_divergence.png` | `fig6-h2-eigenvalues.png` |
  | `channel-ablation/fig_fiedler_saturation_comparison.png` | `fig7-fiedler-saturation.png` |
  | `fig_per_layer_coupling.png` | `fig8-per-layer-coupling.png` |
  | `fig_per_module_coupling_c002.png` | `fig9-per-module-coupling.png` |

- Status: NEW

### F-4A-05: Figure-mapping doc has wrong peak ratio for Figure 5
- Severity: MINOR
- Location: `sections/09_figures.md`, line 11
- Finding: The mapping doc describes Figure 5 as showing "peak 82,154:1" for the co/cross ratio. The tex caption (line 824) and all body text references state **82,854:1**. The mapping doc has transposed digits (154 vs. 854). The tex is internally consistent; the error is confined to `09_figures.md`.
- Status: NEW

### F-4A-06: Figure-mapping doc has wrong Fiedler range for n={3,4,8}
- Severity: MINOR
- Location: `sections/09_figures.md`, line 33
- Finding: The mapping doc states "Bridge Fiedler (n=3/4/8): 0.085-0.095." The ablation table in the tex (lines 810-813) shows 0.095, 0.092, 0.094 for n=3, 4, 8 respectively. The correct range is **0.092-0.095**. The value 0.085 appears nowhere in the paper. The mapping doc lower bound is wrong.
- Status: NEW

### F-4A-07: Figure-mapping doc conflates init-convergence and ablation val losses
- Severity: MINOR
- Location: `sections/09_figures.md`, line 35
- Finding: The mapping doc states "Val loss range: 0.4010-0.4022 (0.17% max delta)." The value 0.4010 is C-002's loss from the init-convergence experiment (Table 1), not from the channel ablation. The ablation table (Table 2) shows losses 0.4015-0.4024. The 0.17% delta applies only to n in {3,4,6} in the ablation (0.4015-0.4022). The mapping doc mixes two different experimental contexts.
- Status: NEW

---

## Paper 2 Figures

### Inventory

Paper 2 contains 5 figures:

| LaTeX Fig # | Label | Type | File | File Exists | Referenced in Text |
|-------------|-------|------|------|-------------|--------------------|
| 1 | `fig:dependency` | TikZ (inline) | N/A | N/A (inline) | YES (line 100) |
| 2 | `fig:amplification` | External | `figures/fig3-amplification-gradient` | YES (.pdf + .png) | YES (line 445) |
| 3 | `fig:consensus` | External | `figures/fig4-consensus-inversion` | YES (.pdf + .png) | YES (line 487) |
| 4 | `fig:spectra` | External | `figures/fig5-spectral-stems` | YES (.pdf + .png) | YES (line 597) |
| 5 | `fig:polytopes` | External | `figures/fig6-polytope-percentiles` | YES (.pdf + .png) | YES (line 691) |

### F-4A-08: Paper 2 figure file numbering offset from LaTeX figure numbering
- Severity: MINOR
- Location: Paper 2, Figures 2-5
- Finding: Paper 2's LaTeX Figure 1 is an inline TikZ diagram. External figure files start at `fig3-amplification-gradient` (LaTeX Figure 2) through `fig6-polytope-percentiles` (LaTeX Figure 5). The file numbering (3-6) reflects their position in a shared numbering scheme with Paper 1 (`rhombic.tex`, which uses `fig1-voronoi-cells` and `fig2-graph-benchmarks`). This is internally consistent across the paper series but means Paper 2's LaTeX figure numbers (2-5) don't match file numbers (3-6). Low severity since this is a deliberate multi-paper convention.
- Status: NEW

### F-4A-09: Extra files in `figures/` directory not used by Paper 2
- Severity: POLISH
- Location: `figures/fig1-voronoi-cells.{pdf,png}`, `figures/fig2-graph-benchmarks.{pdf,png}`
- Finding: Two figure files in the shared `figures/` directory are used by Paper 1 (`rhombic.tex`) but not by Paper 2. They coexist with Paper 2's figures. Confirmed these belong to Paper 1 via grep of `rhombic.tex`. No action needed, but a `figures1/` subdirectory would clarify provenance.
- Status: NEW

---

## Cross-Paper Checks

### F-4A-10: No supplementary figures exist on disk
- Severity: MINOR
- Location: `sections/09_figures.md`, Supplementary Figures table (lines 17-26)
- Finding: The mapping document lists 6 supplementary figures (S1-S6) with filenames like `fig_temporal_emergence.png`, `fig_early_emergence.png`, etc. None of these files exist in `figures3/` or any subdirectory. If supplementary materials are planned, the source figures need to be generated. If supplementary figures have been dropped from scope, the mapping doc should remove these entries to avoid confusion.
- Status: NEW

---

## Summary Table

| ID | Title | Severity | Paper | Status |
|----|-------|----------|-------|--------|
| F-4A-01 | Five Paper 3 figures never referenced in text | MAJOR | 3 | NEW |
| F-4A-02 | Figure file numbering swapped for Figs 6-7 | MINOR | 3 | NEW |
| F-4A-03 | Figure 5 caption "0.22%" vs text "0.17%" scope mismatch | MAJOR | 3 | NEW |
| F-4A-04 | Figure-mapping doc uses obsolete filenames | MAJOR | 3 | NEW |
| F-4A-05 | Mapping doc wrong peak ratio (82,154 vs 82,854) | MINOR | 3 | NEW |
| F-4A-06 | Mapping doc wrong Fiedler range (0.085 vs 0.092) | MINOR | 3 | NEW |
| F-4A-07 | Mapping doc conflates init and ablation val losses | MINOR | 3 | NEW |
| F-4A-08 | Paper 2 file numbering offset from LaTeX fig numbering | MINOR | 2 | NEW |
| F-4A-09 | Extra Paper 1 figures in shared `figures/` directory | POLISH | 2 | NEW |
| F-4A-10 | Supplementary figures listed but missing from disk | MINOR | 3 | NEW |

**Severity distribution:** 0 CRITICAL, 3 MAJOR, 6 MINOR, 1 POLISH

**Key takeaways:**
1. The most impactful finding is F-4A-01: five of nine Paper 3 figures are floating without text references. This is the kind of thing reviewers flag immediately.
2. The figure-mapping document (`09_figures.md`) is substantially stale (F-4A-04 through F-4A-07) and should either be updated to match current reality or removed to avoid serving as a misleading reference.
3. All figure files referenced by the tex files exist on disk. No compilation will fail due to missing figures.
4. Paper 2 figures are clean: all exist, all are referenced, captions match text.
