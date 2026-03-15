# Round 4 — Agent 4C: Missing Figures Audit

> **Scope:** Papers 2 and 3. Check for missing visual evidence, figure
> reference correctness, and claims that lack visual support.
> **Date:** 2026-03-15

---

## 1. Figure 1 (Architecture Schematic) Status

**Status: DEFERRED and absent from the final paper.**

An architecture diagram was planned in an earlier draft section file
(`sections/section2_architecture.tex`, line 192):

```
% [Figure 1: Architecture diagram — to be rendered separately]
```

A detailed specification follows (lines 194--233): dataflow from input
through A (down-projection), reshape to 6x4 grid, bridge M, flatten,
B (up-projection), with inset heatmaps at step 0 vs step 10K.

**In the final `rhombic-paper3.tex`, this figure does not exist.** The
Method section (Section 3) describes the architecture entirely in prose
and equations. Figure 1 in the final paper is `fig1-cybernetic-overview`
(the co-planar/cross-planar ratio plot), not an architecture schematic.
The section file `section2_architecture.tex` is not `\input`-ed by the
final tex — it is a draft artifact.

**Impact: MODERATE.** The paper describes a novel architecture
(RhombiLoRA with bridge matrix) entirely through text and equations.
Reviewers of ML papers universally expect an architecture diagram for
novel components. The reshape-bridge-flatten pipeline is straightforward
in equations but would be immediately comprehensible as a figure. This
is the single most impactful missing figure.

**Finding ID: F-4C-01 (MODERATE)**
Recommendation: Create the architecture schematic per the existing
specification in `section2_architecture.tex` lines 194--233. Insert as
Figure 1 (before the current Figure 1, which would become Figure 2).
All existing `\ref{fig:*}` labels are symbolic and would auto-renumber.

---

## 2. Claims Without Visual Support

### 2a. Exponential Decay with Half-Life 123 Steps

**Claim (lines 165--168, 219--220, 665--671, 1025--1028):** "Cross-planar
coupling decays exponentially with a half-life of 123 +/- 8 steps."

**Visual support:** Figure 1 (`fig:cybernetic`) shows the
co-planar/cross-planar *ratio* on a log scale. The ratio is a derived
quantity (co/cross). The exponential decay claim is about the
*cross-planar coupling magnitude* specifically, which is one component of
that ratio.

The paper states the fit parameters (half-life 123 +/- 8, R^2 from
linear fit for co-planar growth = 0.998) but no figure directly shows:
- The cross-planar coupling magnitude on its own over time
- The exponential fit overlaid on the raw data
- The co-planar coupling linear fit

Figure 3 (`fig:dismantling`) shows per-pair coupling trajectories for
the corpus-coupled experiment (C-003) specifically, which partially
addresses this — but it shows the *per-pair* decomposition rather than
the aggregate cross-planar decay with fit line.

**Finding ID: F-4C-02 (MINOR)**
The claim is quantitative (half-life with error bars) and would benefit
from a supplementary figure showing the decomposed coupling trajectories
with overlaid fits for at least one representative experiment. Not
critical — the ratio plot is persuasive and the dismantling figure
covers the qualitative story — but a reviewer asking "show me the
exponential fit" would find no figure to point to.

### 2b. Three Phases of Fiedler Trajectory at n=4

**Claim (lines 865--870):** "The Bridge Fiedler trajectory exhibits three
phases: (1) rapid growth (steps 0--1,200), (2) a dip (steps 1,200--3,000,
declining to 0.076 — a 9.5% decrease), and (3) recovery (steps 3,000--
10,000, climbing to 0.092)."

**Visual support:** Figure 6 in the compiled PDF (`fig:fiedler`, file
`fig7-fiedler-saturation`) shows Fiedler saturation across channel
counts including n=4. Figure 7 (`fig:h2eigen`, file
`fig6-h2-eigenvalues`) shows eigenvalue ratios at n=4.

**Assessment:** The Fiedler saturation figure (Fig 6) shows the n=4
trajectory where the three phases should be visible as a non-monotonic
curve. The H2 eigenvalue figure (Fig 7) provides additional detail on
the eigenvalue structure. Between these two figures, the three-phase
claim has adequate visual support, provided the Fiedler saturation plot
has sufficient temporal resolution in the 0--3000 step range to show
the dip.

**Finding ID: F-4C-03 (PASS)**
Adequate visual support exists across Figures 6 and 7.

### 2c. Cross-Layer Correlation Patterns

**Claim (lines 980--993):** The Correlation Fiedler converges to ~0.10
for cybernetic text models (0.102 Qwen 7B, 0.101 TinyLlama); Holly
Battery = 1.002.

**Visual support:** No dedicated figure for Correlation Fiedler. This
metric is described entirely in prose. Figure 8 (`fig:perlayer`) shows
per-layer co-planar coupling for the two scales, but this is a different
metric (per-layer coupling magnitude, not cross-layer correlation).

**Finding ID: F-4C-04 (MINOR)**
The Correlation Fiedler is presented as a key scale-consistency metric
(a "structural metric" per the introduction, line 207), but it has no
figure. A small panel showing Correlation Fiedler across experiments
would strengthen the scale-consistency argument. Currently, the reader
must trust three numbers in prose.

### 2d. Linear Growth of Co-Planar Coupling

**Claim (lines 668--671):** "Co-planar coupling grows linearly at
approximately 0.00012 per step (Frobenius norm per module), with no
evidence of saturation through 10,000 steps (R^2 = 0.998 for a linear
fit from step 500 onward)."

**Visual support:** Same gap as 2a — no figure directly shows the
co-planar coupling magnitude with the linear fit. The ratio plot (Fig 1)
shows the combined effect.

**Finding ID: F-4C-05 (MINOR)**
Bundled with F-4C-02. A single supplementary figure showing decomposed
co-planar (linear) and cross-planar (exponential) trajectories with
overlaid fits would satisfy both findings.

---

## 3. Scale Invariance Visual Coverage

**Claim:** Results across 1.1B, 7B, and 14B parameters.

**Figures showing multiple scales:**
- Figure 1 (`fig:cybernetic`): All 6 cybernetic experiments including
  both 1.1B and 7B. **Both cybernetic scales shown.**
- Figure 8 (`fig:perlayer`): Per-layer coupling for Qwen 7B vs
  TinyLlama. **Both cybernetic scales shown.**
- Figure 4 (`fig:heatmaps`): C-002 (TinyLlama) vs exp2 (Qwen 7B
  non-cybernetic). **Both models shown, but different regimes.**

**14B (Holly Battery) visual coverage:** The Holly Battery is described
in prose (Section 6.3) with three quantitative results (off-diagonal
magnitude 0.010, ratio 1.07:1, Bridge Fiedler ~0.10). It has
**no dedicated figure**.

**Finding ID: F-4C-06 (MINOR)**
The 14B result is the non-cybernetic control at the largest scale. It
has no visual evidence — only prose numbers. A bridge heatmap for the
Holly Battery (showing the near-identity structure) alongside one of
the cybernetic heatmaps would provide the visual contrast at 14B scale.
The existing heatmap figure (Fig 4) compares cybernetic vs non-cybernetic
at 1.1B/7B, so Holly would extend this to 14B.

---

## 4. F-01-20: Table 1 Omitting 2 of 4 Distributions (Paper 2)

**Deferred from Round 1.** The finding: Paper 2 defines four edge-weight
distributions (uniform, random, power-law, corpus) in Section 2.1, but
Table 1 (Experiment 1) at scale 8,000 shows only uniform and corpus —
omitting random and power-law.

**Current state of Table 1 (tab:exp1):**
- Scale 125: all 4 distributions present
- Scale 1,000: all 4 distributions present
- Scale 8,000: only uniform and corpus present

**Assessment:** The omission is noted but defensible — scale 8,000 is
computationally expensive, and the pattern is already established at two
smaller scales. However, the table creates an implicit gap: a reader
checking the monotonic amplification gradient (uniform < random <
power-law < corpus) cannot verify it holds at scale 8,000.

**Finding ID: F-4C-07 (MINOR — confirming F-01-20)**
Two options: (1) add a footnote to Table 1 explaining that random and
power-law were not computed at scale 8,000, or (2) run them and fill in
the table. Option 1 is sufficient for honest reporting. The current
silent omission could be read as selective reporting.

---

## 5. Figure Reference Correctness (Off-by-One Check)

### Paper 3 — Figure Numbering

LaTeX auto-numbers figures sequentially. The order of `\begin{figure}`
environments in `rhombic-paper3.tex`:

| Order | Label | Filename | Auto-Number | 09_figures.md Fig# |
|-------|-------|----------|-------------|-------------------|
| 1st | `fig:cybernetic` | `fig1-cybernetic-overview` | Fig 1 | 1 |
| 2nd | `fig:init` | `fig2-init-convergence` | Fig 2 | 2 |
| 3rd | `fig:dismantling` | `fig3-corpus-dismantling` | Fig 3 | 3 |
| 4th | `fig:heatmaps` | `fig4-bridge-heatmaps` | Fig 4 | 4 |
| 5th | `fig:ablation` | `fig5-channel-ablation` | Fig 5 | 5 |
| 6th | `fig:fiedler` | **`fig7-fiedler-saturation`** | **Fig 6** | **7** |
| 7th | `fig:h2eigen` | **`fig6-h2-eigenvalues`** | **Fig 7** | **6** |
| 8th | `fig:perlayer` | `fig8-per-layer-coupling` | Fig 8 | 8 |
| 9th | `fig:permodule` | `fig9-per-module-coupling` | Fig 9 | 9 |

**Finding ID: F-4C-08 (MINOR — cosmetic filename mismatch)**
The filenames `fig6-h2-eigenvalues` and `fig7-fiedler-saturation` are
swapped relative to their LaTeX figure order (Fiedler saturation appears
6th in the document but its filename says "fig7"; H2 eigenvalues appears
7th but filename says "fig6"). This is purely a filename issue — the
LaTeX `\ref` system uses labels, not filenames, so **no off-by-one error
exists in the compiled PDF**. All `\ref{fig:*}` references point to the
correct figure environments.

The `09_figures.md` mapping table lists the figures by their *filename*
numbering (6 and 7), not their actual compiled order. This creates a
discrepancy between the planning document and the final paper.

### Paper 3 — Reference Audit

All `Figure~\ref{fig:*}` calls verified:
- `\ref{fig:cybernetic}` at line 645 → correct (used in §4.1 text)
- `\ref{fig:init}` at line 720 → correct (used in §4.2 text)
- `\ref{fig:dismantling}` at line 728 → correct (used in §4.2 text)
- `\ref{fig:fiedler}` at line 845 → correct (used in §5.2.1 text)
- No figure is referenced before its `\label` definition.
- No orphan labels (every `\label{fig:*}` has at least one `\ref`).
  Exception: `fig:heatmaps` (Figure 4), `fig:ablation` (Figure 5),
  `fig:h2eigen` (Figure 7), `fig:perlayer` (Figure 8), and
  `fig:permodule` (Figure 9) are not referenced with `\ref` in the
  text — they are placed inline near their descriptive prose but not
  explicitly cross-referenced with "Figure~\ref{...}".

**Finding ID: F-4C-09 (POLISH)**
Five of nine figures have no explicit `\ref` citation in the running
text — they appear as float figures near their context but are never
referenced by number. This is not technically an error (the reader sees
them in context), but explicit "Figure X" references improve navigation
in printed/PDF form. Most prominent gap: Figure 5 (the channel ablation
figure, the "THE ablation figure" per 09_figures.md) is never referenced
by number in the text.

### Paper 2 — Figure Numbering

Paper 2 has 5 figures with labels: `fig:dependency` (TikZ, Fig 1),
`fig:amplification` (Fig 2), `fig:consensus` (Fig 3), `fig:spectra`
(Fig 4), `fig:polytopes` (Fig 5). All filenames align with their
position (fig3, fig4, fig5, fig6 — the offset is because Fig 1 is
TikZ inline). All `\ref` citations are correct.

**Finding ID: F-4C-10 (PASS)**
Paper 2 figure references are clean. No off-by-one errors.

---

## Summary

| ID | Severity | Paper | Finding | Recommendation |
|----|----------|-------|---------|---------------|
| F-4C-01 | MODERATE | P3 | Architecture schematic deferred and missing | Create per existing spec; insert as new Fig 1 |
| F-4C-02 | MINOR | P3 | Exponential decay half-life claim has no direct figure | Supplementary figure with decomposed fits |
| F-4C-03 | PASS | P3 | Three-phase Fiedler trajectory at n=4 | Adequate coverage in Figs 6-7 |
| F-4C-04 | MINOR | P3 | Correlation Fiedler (scale metric) has no figure | Small panel or supplementary figure |
| F-4C-05 | MINOR | P3 | Linear co-planar growth claim has no direct figure | Bundle with F-4C-02 |
| F-4C-06 | MINOR | P3 | 14B Holly Battery has no visual evidence | Holly heatmap alongside cybernetic |
| F-4C-07 | MINOR | P2 | Table 1 silently omits 2 distributions at scale 8K | Footnote explaining omission |
| F-4C-08 | MINOR | P3 | Filenames fig6/fig7 swapped vs compiled order | Rename files to match compiled numbering |
| F-4C-09 | POLISH | P3 | 5 of 9 figures lack explicit `\ref` citations | Add Figure~\ref for inline figures |
| F-4C-10 | PASS | P2 | Figure references clean | No action needed |

**Totals:** 1 MODERATE, 6 MINOR, 1 POLISH, 2 PASS.

The single MODERATE finding (F-4C-01, missing architecture schematic) is
the highest-priority item. The specification for the figure already exists
in detail. For an ML paper introducing a novel adapter component, the
absence of a visual architecture overview is a likely reviewer complaint.
