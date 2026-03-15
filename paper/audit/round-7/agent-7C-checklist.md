# Agent 7C — Submission Readiness Checklist

**Date:** 2026-03-15
**Papers:** rhombic-paper2.tex (Paper 2), rhombic-paper3.tex (Paper 3)
**Build timestamp:** Both logs dated 15 Mar 2026 14:06/14:07

---

## 1. Clean Compile

### Paper 2
- **Undefined references:** PASS — none found in log
- **Missing citations:** PASS — none found in log
- **Overfull hboxes >10pt:** FAIL — 1 instance
  - Line 964-971: 17.93pt too wide (the Availability paragraph with URLs)
- **Errors:** PASS — no LaTeX errors
- **Duplicate PDF destinations:** WARNING — 11 instances of `destination with the same identifier`. This is a known hyperref/float[H] interaction. Cosmetic; does not affect content. Fix: add `\usepackage[hypertexnames=false]{hyperref}` or use `\hypertarget`/`\phantomsection`.

### Paper 3
- **Undefined references:** PASS — none found in log
- **Missing citations:** PASS — none found in log
- **Overfull hboxes >10pt:** FAIL — 1 instance
  - Lines 268-278: 10.27pt too wide
- **Overfull hboxes 1-10pt:** WARNING — 2 additional instances
  - Lines 255-265: 2.45pt (under threshold)
  - Lines 567-572: 9.97pt (under threshold but close)
- **Errors:** PASS — no LaTeX errors
- **Duplicate PDF destinations:** WARNING — 15 instances. Same hyperref/float issue as Paper 2.
- **Note:** Paper 3 has `\usepackage{microtype}` commented out with note "MiKTeX font expansion issue". This may contribute to the overfull boxes. Investigate whether `\usepackage[activate={true,nocompatibility}]{microtype}` or a MiKTeX font update resolves it.

---

## 2. No TODOs

- **Paper 2:** PASS — no TODO, FIXME, XXX, PLACEHOLDER, or TBD found
- **Paper 3:** PASS — no TODO, FIXME, XXX, PLACEHOLDER, or TBD found

---

## 3. No Orphan Bib Entries

### Paper 2
- **Bib entries:** 15
- **Cited keys:** 15
- **Orphans:** PASS — every bib entry is cited at least once
- **Bib keys:** fiedler1973, conway1999, coxeter1973, ashby1956, beer1972, bielec2026shape, mohar1991, chung1997, kirkpatrick1983, petersen1962, spielman2004, ghosh2008resistance, fu2009rhombic, bielec2026bridge, dorogovtsev2008

### Paper 3
- **Bib entries:** 26
- **Cited keys:** 26
- **Orphans:** PASS — every bib entry is cited at least once
- **Bib keys:** hu2022lora, zhang2023adalora, liu2024dora, banaei2024loraxs, huang2023lorahub, wang2024multihead, li2024moe, ren2024melora, wiener1948, ashby1956, beer1972, andrychowicz2016, smith2017cyclical, aghajanyan2021, li2018visualizing, clune2013, amer2019modular, frankle2019lottery, fedus2022switch, lepikhin2021gshard, zoph2017neural, conway1999, fiedler1973, bielec2026shape, bielec2026weights, alpaca

---

## 4. No Phantom Citations

- **Paper 2:** PASS — all 15 `\cite{}` keys resolve to bib entries
- **Paper 3:** PASS — all 26 `\cite{}` keys resolve to bib entries

---

## 5. No ?? in Text

- **Paper 2:** PASS — no literal `??` found in tex source
- **Paper 3:** PASS — no literal `??` found in tex source

---

## 6. Figure Files Present

### Paper 2 — requires 4 figures from `figures/`
| Reference | File on disk | Status |
|-----------|-------------|--------|
| `figures/fig3-amplification-gradient` | `figures/fig3-amplification-gradient.pdf` | PASS |
| `figures/fig4-consensus-inversion` | `figures/fig4-consensus-inversion.pdf` | PASS |
| `figures/fig5-spectral-stems` | `figures/fig5-spectral-stems.pdf` | PASS |
| `figures/fig6-polytope-percentiles` | `figures/fig6-polytope-percentiles.pdf` | PASS |

All 4 figures present in both PDF and PNG formats.

### Paper 3 — requires 9 figures from `figures3/`
| Reference | File on disk | Status |
|-----------|-------------|--------|
| `figures3/fig1-cybernetic-overview` | `figures3/fig1-cybernetic-overview.png` | PASS |
| `figures3/fig2-init-convergence` | `figures3/fig2-init-convergence.png` | PASS |
| `figures3/fig3-corpus-dismantling` | `figures3/fig3-corpus-dismantling.png` | PASS |
| `figures3/fig4-bridge-heatmaps` | `figures3/fig4-bridge-heatmaps.png` | PASS |
| `figures3/fig5-channel-ablation` | `figures3/fig5-channel-ablation.png` | PASS |
| `figures3/fig6-h2-eigenvalues` | `figures3/fig6-h2-eigenvalues.png` | PASS |
| `figures3/fig7-fiedler-saturation` | `figures3/fig7-fiedler-saturation.png` | PASS |
| `figures3/fig8-per-layer-coupling` | `figures3/fig8-per-layer-coupling.png` | PASS |
| `figures3/fig9-per-module-coupling` | `figures3/fig9-per-module-coupling.png` | PASS |

All 9 figures present. Note: Paper 3 figures are PNG only (no PDF versions). Paper 2 figures have both PDF and PNG. For highest print quality, PDF vector figures are preferred; verify Paper 3 PNGs are at sufficient DPI (300+) for any print submission.

---

## 7. Author Information

### Paper 2
- **Author:** Timothy Paul Bielec — PASS
- **Affiliation:** Promptcrafted LLC, Los Angeles, CA — PASS
- **Email:** timothy@promptcrafted.com (via `\thanks`) — PASS

### Paper 3
- **Author:** Timothy Paul Bielec — PASS
- **Affiliation:** Promptcrafted LLC, Los Angeles, CA — PASS
- **Email:** timothy@promptcrafted.com (via `\thanks`) — PASS

---

## 8. Abstract Length

### Paper 2
- **Word count:** ~170 words
- **Status:** PASS — within typical range (150-250 words for CS/math)

### Paper 3
- **Word count:** ~230 words
- **Status:** PASS — within typical range

---

## 9. Page Count

### Paper 2
- **Pages:** 16 (from log: "Output written on rhombic-paper2.pdf (16 pages)")
- **Status:** PASS — reasonable for a journal or extended conference paper with 7 experiments, 7 tables, 5 figures, and 1 TikZ diagram

### Paper 3
- **Pages:** 23 (from log: "Output written on rhombic-paper3.pdf (23 pages)")
- **Status:** WARNING — on the longer side. Includes appendix with experimental details (hyperparameters, per-experiment configs). If a venue has a page limit, the appendix may need to be split to supplementary material. The main body (Sections 1-9) is ~18 pages; the appendix adds ~5.

---

## 10. Code Availability

### Paper 2
- **Source code:** `https://github.com/promptcrafted/rhombic` — PASS
- **Package:** `https://pypi.org/project/rhombic/` (v0.3.0, 312 tests) — PASS
- **Provenance:** PyPI trusted publishing binding to commit hash — PASS
- **Reproduction command:** `pip install rhombic==0.3.0 && python scripts/run_experiments.py` — PASS
- **Interactive demo:** `https://huggingface.co/spaces/timotheospaul/rhombic` — PASS
- **Seed:** 42 for all stochastic operations — PASS

### Paper 3
- **Source code:** `https://github.com/promptcrafted/rhombic` — PASS
- **Package:** `https://pypi.org/project/rhombic/` (v0.3.0, 312 tests) — PASS
- **Reproduction:** install rhombic + run training scripts with seed=42 — PASS
- **Note:** Paper 3's reproduction instruction is less specific than Paper 2's (no single `run_experiments.py` equivalent cited). Consider adding the exact script path or command.

---

## 11. Supplementary Material References

- **Paper 2:** PASS — no references to supplementary material found. Self-contained.
- **Paper 3:** PASS — no references to supplementary material found. Appendix is inline. Self-contained.

---

## Summary

| # | Check | Paper 2 | Paper 3 |
|---|-------|---------|---------|
| 1 | Clean compile | WARNING | WARNING |
| 2 | No TODOs | PASS | PASS |
| 3 | No orphan bib entries | PASS | PASS |
| 4 | No phantom citations | PASS | PASS |
| 5 | No ?? in text | PASS | PASS |
| 6 | Figure files present | PASS | PASS |
| 7 | Author information | PASS | PASS |
| 8 | Abstract length | PASS | PASS |
| 9 | Page count | PASS | WARNING |
| 10 | Code availability | PASS | PASS |
| 11 | Supplementary references | PASS | PASS |

### Items Requiring Attention (sorted by severity)

**FAIL (must fix before submission):**

1. **Paper 2, overfull hbox at 17.93pt (lines 964-971).** The Availability paragraph with long URLs overflows the text block. Fix: break the `\texttt{pip install ...}` line with `\allowbreak` or `\linebreak`, or wrap in a `\begin{sloppypar}...\end{sloppypar}`.

2. **Paper 3, overfull hbox at 10.27pt (lines 268-278).** Likely a long inline equation or citation chain. Fix: add linebreak hints or rephrase.

**WARNING (should fix, not blocking):**

3. **Both papers: duplicate PDF destination warnings (hyperref/float[H]).** Add `\hypersetup{hypertexnames=false}` to suppress. No content impact.

4. **Paper 3: microtype disabled.** Re-enabling would help with the overfull boxes. Try `\usepackage[expansion=false]{microtype}` to avoid the MiKTeX font expansion issue while retaining protrusion.

5. **Paper 3: 23 pages.** Verify target venue page limits. The 5-page appendix is a natural candidate for supplementary material if needed.

6. **Paper 3: figure format.** All figures are PNG. For print-quality submission, verify DPI >= 300. Paper 2's PDF figures are vector and preferred.

7. **Paper 3: reproduction command specificity.** Consider adding the exact training script path (e.g., `python scripts/train_rhombilora.py --seed 42`).
