# Audit 2B: Cross-Reference and Citation Integrity

**Auditor:** Agent 2B (adversarial)
**Scope:** Papers 2 and 3 -- all `\cite{}`, `\ref{}`, `\label{}` pairs, .bib consistency, cross-paper reference accuracy
**Date:** 2026-03-15

---

## FINDINGS

### F-2B-01: CRITICAL — Paper 2 bib entry for Paper 3 has wrong subtitle

**Location:** `rhombic-paper2.bib`, lines 189-196 (entry `bielec2026bridge`)

**Issue:** The bib entry title reads:
> "The Learnable Bridge: Task Fingerprinting and Adapter Composition via Structured Coupling in Low-Rank Adaptation"

The actual title of Paper 3 (`rhombic-paper3.tex`, lines 27-29) is:
> "The Learnable Bridge: Cybernetic Feedback Discovers Rhombic Dodecahedral Geometry in Multi-Channel LoRA"

The subtitle is completely different. This will render as an incorrect title in Paper 2's bibliography.

**Severity:** CRITICAL -- incorrect citation in published bibliography.

**Fix:** Update `bielec2026bridge` title in `rhombic-paper2.bib` to match Paper 3's actual title.

---

### F-2B-02: CRITICAL — Paper 3 attributes bridge fingerprinting results to Paper 2, but Paper 2 contains no such content

**Location:** `rhombic-paper3.tex`, lines 110-119 and 273-282

**Issue:** Two passages in Paper 3 cite `bielec2026weights` (Paper 2) for bridge task fingerprinting results:

Line 111: "Prior work~\cite{bielec2026weights} established that untrained bridges learn task-discriminative structure: a leave-one-out SVM on 28 query-projection bridges (1,008 parameters) classifies task type at 84.5% accuracy..."

Lines 273-278: "RhombiLoRA~\cite{bielec2026weights} introduces the bridge matrix... Paper~2 in this series established that the bridge learns task-discriminative structure: bridge fingerprints classify task type with 84.5% leave-one-out accuracy..."

**Paper 2's actual content** is entirely about weighted lattice topology benchmarks (7 experiments comparing FCC vs SC lattices under heterogeneous edge weights). It contains zero content about bridge matrices, SVM classification, task fingerprinting, or LoRA training. The word "bridge" appears only in the Future Work section (line 862) pointing forward to Paper 3.

These results appear to belong to Paper 3 itself (or to a planned but reorganized intermediate paper). As written, a reader following the citation to Paper 2 will find no supporting content.

**Severity:** CRITICAL -- readers cannot verify the cited claims by consulting the referenced paper.

**Fix:** Either (a) move the bridge fingerprinting results into Paper 2 if they belong there, (b) cite Paper 3 itself or a separate technical report, or (c) present these as new results within Paper 3 without external citation.

---

### F-2B-03: INFO — 10 orphan bib entries in Paper 3

**Location:** `rhombic-paper3.bib`

**Issue:** The following bib entries are defined but never cited in `rhombic-paper3.tex`:

| Key | Title |
|-----|-------|
| `banaei2024loraxs` | LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters |
| `biderman2024lora` | LoRA Learns Less and Forgets Less |
| `ren2024melora` | MELoRA: Mini-Ensemble Low-Rank Adapters |
| `ilharco2023editing` | Editing Models with Task Arithmetic |
| `yadav2023ties` | TIES-Merging: Resolving Interference When Merging Models |
| `yu2024dare` | Language Models are Super Mario |
| `yang2024adamerging` | AdaMerging: Adaptive Model Merging |
| `beer1972` | Brain of the Firm |
| `fiedler1973` | Algebraic connectivity of graphs |
| `putterman2024` | Learning Group Actions on Latent Representations |

**Severity:** INFO -- orphan bib entries are harmless (natbib will simply not include them in the bibliography) but indicate either removed text or entries staged for future use.

**Note:** `fiedler1973` and `beer1972` are particularly notable orphans. Fiedler (1973) is the foundational reference for the Fiedler eigenvalue, which Paper 3 discusses extensively. `beer1972` is cited in Paper 2 for the Viable System Model. Both should likely be cited in Paper 3 as well.

---

### F-2B-04: INFO — 4 orphan bib entries in Paper 2

**Location:** `rhombic-paper2.bib`

**Issue:** The following bib entries are defined but never cited in `rhombic-paper2.tex`:

| Key | Title |
|-----|-------|
| `hales2005` | A proof of the Kepler conjecture |
| `hales2017` | A formal proof of the Kepler conjecture |
| `bateson1979` | Mind and Nature: A Necessary Unity |
| `ghosh2008` | Growing well-connected graphs |

**Severity:** INFO -- `hales2005` and `hales2017` are retained from Paper 1 (sphere packing context). `bateson1979` may have been planned for the cybernetics discussion. `ghosh2008` is a separate entry from the cited `ghosh2008resistance` by the same authors.

---

### F-2B-05: PASS — All `\cite{}` commands in Paper 3 resolve to bib entries

**Verification:** 22 distinct citation keys extracted from `rhombic-paper3.tex`. All 22 match entries in `rhombic-paper3.bib`:

`hu2022lora`, `bielec2026weights`, `zhang2023adalora`, `liu2024dora`, `huang2023lorahub`, `wang2024multihead`, `li2024moe`, `wiener1948`, `ashby1956`, `smith2017cyclical`, `li2018visualizing`, `andrychowicz2016`, `clune2013`, `amer2019modular`, `frankle2019lottery`, `fedus2022switch`, `lepikhin2021gshard`, `conway1999`, `bielec2026shape`, `alpaca`, `zoph2017neural`, `aghajanyan2021`

No phantom citations.

---

### F-2B-06: PASS — All `\cite{}` commands in Paper 2 resolve to bib entries

**Verification:** 15 distinct citation keys extracted from `rhombic-paper2.tex`. All 15 match entries in `rhombic-paper2.bib`:

`bielec2026shape`, `fiedler1973`, `mohar1991`, `chung1997`, `ghosh2008resistance`, `spielman2004`, `kirkpatrick1983`, `fu2009rhombic`, `dorogovtsev2008`, `ashby1956`, `beer1972`, `bielec2026bridge`, `petersen1962`, `conway1999`, `coxeter1973`

No phantom citations.

---

### F-2B-07: PASS — All `\ref{}` commands in Paper 3 have matching `\label{}` definitions

**Verification:** 16 distinct ref targets extracted. All match labels defined in the same file:

`sec:background`, `sec:method`, `sec:emergence`, `sec:ablation`, `sec:scale`, `sec:discussion`, `sec:conclusion`, `sec:steersman`, `sec:rdgeometry`, `tab:experiments`, `fig:cybernetic`, `fig:init`, `fig:dismantling`, `sec:setup`, `fig:fiedler`

No dangling refs. No duplicate labels (29 unique labels, each defined exactly once).

---

### F-2B-08: PASS — All `\ref{}` commands in Paper 2 have matching `\label{}` definitions

**Verification:** 14 distinct ref targets extracted. All match labels defined in the same file:

`fig:dependency`, `sec:background`, `sec:method`, `sec:lattice`, `sec:cell`, `sec:discussion`, `sec:future`, `sec:conclusion`, `tab:exp1`, `tab:exp5`, `fig:amplification`, `fig:consensus`, `fig:spectra`, `fig:polytopes`

No dangling refs. No duplicate labels (23 unique labels, each defined exactly once).

---

### F-2B-09: PASS — Paper 3's bib entry for Paper 2 title is accurate

**Verification:** `bielec2026weights` in `rhombic-paper3.bib` has title "Structured Edge Weights Amplify {FCC} Lattice Topology Advantages via Bottleneck Resilience". Paper 2's actual title (line 27-28): "Structured Edge Weights Amplify FCC Lattice Topology Advantages via Bottleneck Resilience". Match confirmed.

---

### F-2B-10: PASS — Paper 3's bib entry for Paper 1 title matches Paper 2's bib entry for Paper 1

**Verification:** Both `rhombic-paper3.bib` and `rhombic-paper2.bib` contain `bielec2026shape` with identical title: "The Shape of the Cell: Empirical Comparison of Cubic and {FCC} Lattice Topologies Across Graph Theory, Spatial Operations, Signal Processing, and Embedding Retrieval". Consistent across both papers.

---

### F-2B-11: PASS — No `??` in compiled output logs

**Verification:** `rhombic-paper3.log` contains one standard "Label(s) may have changed. Rerun to get cross-references right." warning (normal for first compilation pass). No "Citation undefined" or "Reference undefined" warnings. `rhombic-paper2.log` is clean.

---

### F-2B-12: MINOR — Paper 3 discusses Fiedler eigenvalue extensively but does not cite Fiedler (1973)

**Location:** `rhombic-paper3.tex`, passim (the term "Fiedler" appears dozens of times)

**Issue:** Paper 3 uses the Fiedler eigenvalue as a core metric throughout the paper (Bridge Fiedler, Correlation Fiedler, Fiedler bifurcation, etc.) but never cites Fiedler's 1973 foundational paper. The entry `fiedler1973` exists in `rhombic-paper3.bib` but is uncited. Paper 2 correctly cites `fiedler1973` at first use (line 225).

**Severity:** MINOR -- the concept is well-established and Paper 3 could reasonably assume the reader has encountered it in Paper 2. However, a standalone reading of Paper 3 would benefit from the citation, and the bib entry is already present.

**Fix:** Add `\cite{fiedler1973}` at first mention of the Fiedler eigenvalue in Paper 3 (likely in Section 2 or Section 3.1).

---

### F-2B-13: MINOR — Paper 3 references Beer's cybernetic framework implicitly but does not cite Beer (1972)

**Location:** `rhombic-paper3.bib` contains `beer1972` (Brain of the Firm) but it is never cited in `rhombic-paper3.tex`.

**Issue:** Paper 3's Steersman concept is explicitly cybernetic and cites Wiener (1948) and Ashby (1956). Beer's Viable System Model -- cited in Paper 2's resilience discussion -- is relevant to Paper 3's cybernetic framing but goes uncited. The bib entry is present.

**Severity:** MINOR -- the existing Wiener + Ashby citations cover the cybernetic lineage adequately. Beer would strengthen the connection but is not essential.

---

## SUMMARY

| Category | Count |
|----------|-------|
| CRITICAL | 2 |
| MINOR | 2 |
| INFO | 2 |
| PASS | 7 |
| **Total findings** | **13** |

### Critical issues requiring action:
1. **F-2B-01:** Paper 2 bib has wrong title for Paper 3 (`bielec2026bridge` subtitle mismatch)
2. **F-2B-02:** Paper 3 attributes bridge fingerprinting/SVM results to Paper 2, but Paper 2 contains no such content -- citation target mismatch

### Minor issues:
3. **F-2B-12:** Paper 3 should cite Fiedler (1973) -- entry already in bib
4. **F-2B-13:** Beer (1972) in bib but uncited -- consider adding

### Clean:
- All `\cite{}` keys resolve to bib entries in both papers (no phantom citations)
- All `\ref{}`/`\label{}` pairs are consistent in both papers (no dangling refs, no duplicates)
- Cross-paper title for Paper 1 is consistent between both bibs
- Cross-paper title for Paper 2 (in Paper 3's bib) is accurate
- No `??` in compilation logs
