# Round 6 -- Agent 6A: Citation Completeness

## Findings

### F-6A-01: Orphan bib entry -- biderman2024lora (P3)
- **Severity:** MINOR
- **Paper:** P3
- **Location:** rhombic-paper3.bib line 31
- **Finding:** `@article{biderman2024lora}` ("LoRA Learns Less and Forgets Less") is present in the bibliography but never cited anywhere in rhombic-paper3.tex. The paper does not discuss forgetting behavior or LoRA training dynamics in a way that requires this reference.
- **Recommendation:** Remove from rhombic-paper3.bib, or add a brief mention in Section 2.1 (Low-Rank Adaptation) if the forgetting/learning trade-off is relevant context.

### F-6A-02: Orphan bib entry -- ilharco2023editing (P3)
- **Severity:** MINOR
- **Paper:** P3
- **Location:** rhombic-paper3.bib line 70
- **Finding:** `@article{ilharco2023editing}` ("Editing Models with Task Arithmetic") is present in the bibliography but never cited in rhombic-paper3.tex. No discussion of task arithmetic or model editing appears in the paper.
- **Recommendation:** Remove from rhombic-paper3.bib.

### F-6A-03: Orphan bib entry -- yadav2023ties (P3)
- **Severity:** MINOR
- **Paper:** P3
- **Location:** rhombic-paper3.bib line 77
- **Finding:** `@article{yadav2023ties}` ("TIES-Merging") is present in the bibliography but never cited in rhombic-paper3.tex. No discussion of model merging appears in the paper.
- **Recommendation:** Remove from rhombic-paper3.bib.

### F-6A-04: Orphan bib entry -- yu2024dare (P3)
- **Severity:** MINOR
- **Paper:** P3
- **Location:** rhombic-paper3.bib line 84
- **Finding:** `@article{yu2024dare}` ("Language Models are Super Mario" / DARE) is present in the bibliography but never cited in rhombic-paper3.tex. No discussion of homologous model absorption appears in the paper.
- **Recommendation:** Remove from rhombic-paper3.bib.

### F-6A-05: Orphan bib entry -- yang2024adamerging (P3)
- **Severity:** MINOR
- **Paper:** P3
- **Location:** rhombic-paper3.bib line 91
- **Finding:** `@article{yang2024adamerging}` ("AdaMerging") is present in the bibliography but never cited in rhombic-paper3.tex. No discussion of adaptive model merging appears in the paper.
- **Recommendation:** Remove from rhombic-paper3.bib.

### F-6A-06: Orphan bib entry -- putterman2024 (P3)
- **Severity:** MINOR
- **Paper:** P3
- **Location:** rhombic-paper3.bib line 266
- **Finding:** `@article{putterman2024}` ("Learning Group Actions on Latent Representations") is present in the bibliography but never cited in rhombic-paper3.tex. No discussion of group actions on latent spaces appears in the paper.
- **Recommendation:** Remove from rhombic-paper3.bib. Alternatively, if the learned group action concept is relevant to bridge matrix behavior (it learns a coupling structure in latent space), consider adding a brief mention in Section 7.4 (Connection to Broader Literature) with a citation.

### F-6A-07: Orphan bib entry -- hales2005 (P2)
- **Severity:** MINOR
- **Paper:** P2
- **Location:** rhombic-paper2.bib line 3
- **Finding:** `@article{hales2005}` ("A proof of the Kepler conjecture") is present in the bibliography but never cited in rhombic-paper2.tex.
- **Recommendation:** Remove, or cite when FCC packing density is implicitly invoked. The most natural location is Section 6.5 line 924 ("The $D_4$ lattice achieves the densest known lattice packing in 4D") where a parenthetical noting that FCC achieves densest packing in 3D (Hales 2005) would strengthen the 3D-to-4D analogy.

### F-6A-08: Orphan bib entry -- hales2017 (P2)
- **Severity:** MINOR
- **Paper:** P2
- **Location:** rhombic-paper2.bib line 14
- **Finding:** `@article{hales2017}` ("A formal proof of the Kepler conjecture") is present in the bibliography but never cited in rhombic-paper2.tex.
- **Recommendation:** Remove, or cite alongside hales2005 if a Kepler conjecture reference is added.

### F-6A-09: Orphan bib entry -- bateson1979 (P2)
- **Severity:** MINOR
- **Paper:** P2
- **Location:** rhombic-paper2.bib line 135
- **Finding:** `@book{bateson1979}` ("Mind and Nature") is present in the bibliography but never cited in rhombic-paper2.tex. No discussion of Bateson's work appears in the paper.
- **Recommendation:** Remove from rhombic-paper2.bib.

### F-6A-10: Orphan bib entry -- ghosh2008 (P2)
- **Severity:** MINOR
- **Paper:** P2
- **Location:** rhombic-paper2.bib line 157
- **Finding:** `@article{ghosh2008}` (Ghosh & Boyd, "Growing well-connected graphs", 2006) is present in the bibliography but never cited in rhombic-paper2.tex. Note: this is distinct from `ghosh2008resistance` (Ghosh, Boyd & Saberi, "Minimizing Effective Resistance", 2008), which IS cited at line 241. The two entries share first two authors but are different papers.
- **Recommendation:** Remove `ghosh2008` from rhombic-paper2.bib. Retain `ghosh2008resistance`.

### F-6A-11: Uncited prior-work claim -- LoRA without citation in P2
- **Severity:** MAJOR
- **Paper:** P2
- **Location:** Section 6.1, lines 868--872
- **Finding:** Paper 2 mentions "low-rank adapters (LoRA) for large language models" and "RhombiLoRA" in the Future Work section without citing hu2022lora. The term "LoRA" is a specific method introduced by Hu et al. (2022) and requires a citation on first use. Additionally, `hu2022lora` does not exist in rhombic-paper2.bib at all -- it would need to be added.
- **Recommendation:** Add the `hu2022lora` entry to rhombic-paper2.bib and cite it at line 868: "low-rank adapters (LoRA)~\cite{hu2022lora}".

### F-6A-12: Uncited claim -- quantum error correction in P2
- **Severity:** MINOR
- **Paper:** P2
- **Location:** Section 6.5, lines 932--935
- **Finding:** The claim that "lattice codes defined on higher-connectivity topologies have higher error thresholds under heterogeneous qubit coupling strengths" is stated without a citation. While framed within future work, the assertion about error thresholds under heterogeneous coupling is a substantive claim about prior work in quantum error correction that should be supported.
- **Recommendation:** Either add a citation to a relevant quantum error correction reference (e.g., Dennis et al. 2002 on topological quantum memory, or Kitaev 2003), or soften to "may have higher error thresholds" to mark it as conjecture.

### F-6A-13: Missing foundational citation -- conway1999 at first FCC/RD identification in P2
- **Severity:** MINOR
- **Paper:** P2
- **Location:** Section 2.2, line 262
- **Finding:** The claim "the Voronoi cell (the rhombic dodecahedron)" at line 262 is the first substantive assertion of the FCC-RD Voronoi relationship in the paper body. The `conway1999` entry exists in the bibliography but is not cited until line 923 (Section 6.5, Future Work). The geometric fact should carry its citation at the point of assertion.
- **Recommendation:** Add `\cite{conway1999}` at line 262: "of the Voronoi cell (the rhombic dodecahedron)~\cite{conway1999}."

### F-6A-14: No phantom citations (P3)
- **Severity:** N/A (clean)
- **Paper:** P3
- **Finding:** Every `\cite{key}` in rhombic-paper3.tex resolves to a valid `@entry` in rhombic-paper3.bib. 23 distinct cite keys verified against 29 bib entries. Zero phantoms.

### F-6A-15: No phantom citations (P2)
- **Severity:** N/A (clean)
- **Paper:** P2
- **Finding:** Every `\cite{key}` in rhombic-paper2.tex resolves to a valid `@entry` in rhombic-paper2.bib. 15 distinct cite keys verified against 19 bib entries. Zero phantoms.

### F-6A-16: Self-citation cross-references verified
- **Severity:** N/A (clean)
- **Papers:** P2, P3
- **Finding:** Cross-references between all three papers are correct and consistent:
  - P3 cites P1 as `bielec2026shape` -- title: "The Shape of the Cell: Empirical Comparison of Cubic and FCC Lattice Topologies Across Graph Theory, Spatial Operations, Signal Processing, and Embedding Retrieval". Year 2026. Matches.
  - P3 cites P2 as `bielec2026weights` -- title: "Structured Edge Weights Amplify FCC Lattice Topology Advantages via Bottleneck Resilience". Matches P2's actual document title exactly.
  - P2 cites P1 as `bielec2026shape` -- identical entry to P3's. Matches.
  - P2 cites P3 as `bielec2026bridge` -- title: "The Learnable Bridge: Cybernetic Feedback Discovers Rhombic Dodecahedral Geometry in Multi-Channel LoRA". Matches P3's actual document title exactly.
  - All four entries: year 2026, author "Timothy Paul Bielec", type `@unpublished`, GitHub URL. Consistent.

### F-6A-17: Foundational citations verified (P3)
- **Severity:** N/A (clean)
- **Paper:** P3
- **Finding:** All foundational concepts carry their originating citations:
  - **LoRA**: `hu2022lora` cited at lines 85, 255, 345 (intro, background, method). Correct.
  - **Fiedler eigenvalue**: `fiedler1973` cited at line 436 (spectral loss definition). Correct.
  - **Cybernetics**: `wiener1948`, `ashby1956`, `beer1972` cited at line 304 (cybernetic optimization subsection). Correct.
  - **RD / FCC lattice**: `conway1999` cited at line 322 (RD background). Correct.
  - **Intrinsic dimensionality**: `aghajanyan2021` cited at line 1151 (discussion). Correct.

### F-6A-18: Foundational citations verified (P2) -- with gaps
- **Severity:** N/A (informational)
- **Paper:** P2
- **Finding:** Most foundational concepts carry citations:
  - **Fiedler eigenvalue**: `fiedler1973` at line 225. Correct.
  - **Weighted Laplacian**: `mohar1991`, `chung1997` at line 233. Correct.
  - **Effective resistance**: `ghosh2008resistance` at line 241. Correct.
  - **Spectral sparsification**: `spielman2004` at line 243. Correct.
  - **Simulated annealing**: `kirkpatrick1983` at line 529. Correct.
  - **Cybernetics**: `ashby1956` at line 797, `beer1972` at line 801. Correct. (Wiener not needed -- P2 has no cybernetics framing.)
  - **RD as FCC Voronoi cell**: `conway1999` at line 923, but NOT at line 262 where the claim is first made. See F-6A-13.
  - **LoRA**: NOT cited. See F-6A-11.

### F-6A-19: Prior audit correction -- 3 false orphans
- **Severity:** N/A (correction to prior audit)
- **Paper:** P3
- **Finding:** The prior version of this audit (which this file replaces) incorrectly listed `banaei2024loraxs`, `ren2024melora`, and `beer1972` as orphan bib entries in P3. All three ARE cited:
  - `banaei2024loraxs` cited at line 280 (LoRA-XS discussion)
  - `ren2024melora` cited at line 275 (MELoRA discussion)
  - `beer1972` cited at line 304 (cybernetic optimization)

  The prior audit's grep was incomplete -- it searched only the initial set of citations found at the first pass and missed citations in the Section 2.2--2.3 range (lines 266--310) where these three references appear. The true P3 orphan count is **6**, not 9.

## Summary

| Category | P2 | P3 | Total |
|----------|----|----|-------|
| Uncited claims | 2 (F-6A-11, F-6A-12) | 0 | 2 |
| Phantom citations | 0 | 0 | 0 |
| Orphan bib entries | 4 (F-6A-07--10) | 6 (F-6A-01--06) | 10 |
| Self-citation issues | 0 | 0 | 0 |
| Missing foundational (placement) | 1 (F-6A-13) | 0 | 1 |
| **Total findings** | **7** | **6** | **13** |

### Severity breakdown

| Severity | Count | Findings |
|----------|-------|----------|
| CRITICAL | 0 | -- |
| MAJOR | 1 | F-6A-11 (LoRA uncited in P2, bib entry missing) |
| MINOR | 12 | F-6A-01--10 (orphans), F-6A-12 (QEC claim), F-6A-13 (conway1999 placement) |

### Action items (priority order)

1. **MAJOR -- F-6A-11:** Add `hu2022lora` to rhombic-paper2.bib and cite at line 868.
2. **MINOR -- F-6A-01--06:** Remove 6 orphan entries from rhombic-paper3.bib: biderman2024lora, ilharco2023editing, yadav2023ties, yu2024dare, yang2024adamerging, putterman2024.
3. **MINOR -- F-6A-07--10:** Remove 4 orphan entries from rhombic-paper2.bib: hales2005, hales2017, bateson1979, ghosh2008. (Or cite hales2005/hales2017 at line 924.)
4. **MINOR -- F-6A-13:** Add `\cite{conway1999}` at line 262 of rhombic-paper2.tex.
5. **MINOR -- F-6A-12:** Add a QEC citation at line 932 of rhombic-paper2.tex, or soften the claim.

### Verification notes

- P3 has 29 bib entries, 23 cited = 6 orphans (20.7% orphan rate).
- P2 has 19 bib entries, 15 cited = 4 orphans (21.1% orphan rate).
- Both orphan rates are high for submission; cleaning them prevents reviewer questions about unmatched references.
- All cross-paper title/year/author fields are consistent.
- All cited entries verified present in their respective bib files.

---

*Audit completed by Agent 6A. All four files (two .tex, two .bib) read in full. Every cite key cross-checked against its bib file. Every bib key cross-checked against its tex file. Prior audit corrected (3 false orphan identifications in P3).*
