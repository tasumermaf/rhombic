# Round 1C: Math Accuracy -- Paper 1 (LOCKED -- report only)

**Auditor:** Meridian (adversarial math accuracy)
**Paper:** "The Shape of the Cell" (`rhombic.tex`)
**Date:** 2026-03-15
**Source files cross-referenced:** `rhombic/lattice.py`, `rhombic/benchmark.py`, `tests/test_lattice.py`, `tests/test_benchmark.py`

---

## Verified Claims

### Node Counts (Table 1, Section 3)

| Scale | Paper SC nodes | Verified | Paper FCC nodes | Verified | Method |
|-------|---------------|----------|-----------------|----------|--------|
| ~125  | 125           | 5^3 = 125 | 108           | 4*3^3 = 108 | n_cubic=round(125^(1/3))=5, n_fcc=round(31.25^(1/3))=3 |
| ~1000 | 1000          | 10^3 = 1000 | 864         | 4*6^3 = 864 | n_cubic=round(1000^(1/3))=10, n_fcc=round(250^(1/3))=6 |
| ~4000 | 4096          | 16^3 = 4096 | 4000        | 4*10^3 = 4000 | n_cubic=round(4000^(1/3))=16, n_fcc=round(1000^(1/3))=10 |
| ~8000 | 8000          | 20^3 = 8000 | 8788        | 4*13^3 = 8788 | n_cubic=round(8000^(1/3))=20, n_fcc=round(2000^(1/3))=13 |

All node counts verified exactly. The "exactly 4n^3" claim (line 220) is correct: the four FCC basis types occupy distinct parity classes in the rounding-key space (even/even/even, odd/odd/even, odd/even/odd, even/odd/odd), so no deduplication occurs between basis types or unit cells.

### Cubic Edge Count Formula (Section 3.1, line 215)

Formula: 3n^2(n-1). Verified against code (`CubicLattice._build()`) and test (`test_edge_count_bounds`). The code generates edges in +x, +y, +z directions only (avoiding double-counting), producing exactly n^2(n-1) edges per axis, 3n^2(n-1) total.

- n=5: 3*25*4 = 300
- n=10: 3*100*9 = 2700
- n=16: 3*256*15 = 11,520
- n=20: 3*400*19 = 22,800

### Node Count Difference <15% (Section 3.1, line 236)

| Scale | |SC-FCC|/SC | Under 15%? |
|-------|-----------|------------|
| ~125  | 17/125 = 13.6% | Yes |
| ~1000 | 136/1000 = 13.6% | Yes |
| ~4000 | 96/4096 = 2.3% | Yes |
| ~8000 | 788/8000 = 9.85% | Yes |

Verified. All within claimed bound.

### FCC Interior Degree = 12 (Section 3.1, line 222)

FCC_NEIGHBOR_OFFSETS in `lattice.py` contains exactly 12 vectors: {1/2(+/-e_i +/- e_j)} for i<j in {1,2,3}. 3 axis pairs * 4 sign combinations = 12 offsets. Matches the paper's equation (line 224). Verified by test `test_interior_degree_12`.

### Packing Fraction (Section 2.2, line 162)

Paper: pi/(3*sqrt(2)) ~ 0.7405. Verified: pi/4.24264 = 0.74048, rounds to 0.7405.

### L1/L2 Geometric Argument (Section 4.2, lines 327-331)

Paper: "For a unit vector drawn uniformly on S^2, the expected L1 norm is 3/2." Verified: By symmetry, E[|x|+|y|+|z|] = 3*E[|x|]. For uniform S^2, E[|x|] = integral(0 to pi) of |cos(theta)| * sin(theta) dtheta / 2 = 1/2. So E[L1] = 3/2. With L2=1, ratio = 1.50. Predicts 33.3% reduction. Correct.

### ASPL Advantage Range (Section 4.2, line 315)

Paper: "29-32%". From table ratios: 1/1.45 - 1 = -31.0%, 1/1.46 - 1 = -31.5%, 1/1.40 - 1 = -28.6%. Range: 28.6-31.5%. Rounds to 29-32%. Correct.

### Abstract "30% shorter paths" (line 55)

Mean of 28.6%, 31.0%, 31.5% is ~30.4%. "30%" is a fair summary.

### Diameter Advantage Range (Figure 2 caption, line 339)

Paper: "38-42%". From table ratios: 1 - 1/1.71 = 41.5%, 1 - 1/1.69 = 40.8%, 1 - 1/1.61 = 37.9%. Exact range: 37.9-41.5%. "38-42%" requires rounding 37.9% up to 38%. Marginal but acceptable.

### Abstract "40% smaller diameter" (line 56)

Mean of 37.9%, 40.8%, 41.5% is ~40.1%. "40%" is accurate.

### FCC Neighbor Recall (Section 5.2, line 484)

Paper: "15-26 percentage points." From Table 4: 0.903-0.641=0.262 (26.2pp), 0.763-0.547=0.216 (21.6pp), 0.658-0.508=0.150 (15.0pp). Range: 15.0-26.2pp. "15-26" is exact.

### FCC Index Recall at 500 Nodes (Section 6.2, line 548)

Paper: "28.0% vs. 8.5%, a 3.3x relative improvement." 0.280/0.085 = 3.294x, rounds to 3.3x. Correct.

### Per-Neighbor Weight Dilution (Section 5.2, line 487)

Paper: "1/13 vs. 1/7." For Laplacian averaging: SC weight = 1/(6+1) = 1/7, FCC weight = 1/(12+1) = 1/13. Correct.

### Cost Table Byte Calculation (Table 5, line 611)

Paper: "~96 vs. 48 bytes/node." 12 pointers * 8 bytes = 96 (FCC), 6 pointers * 8 bytes = 48 (SC). Correct.

### Asymptotic Edge Ratio = 2 (lines 60, 319, 599)

Paper: "12/6 = 2." For infinite lattices, edges ~ (nodes * degree) / 2. At matched node count: (N*12/2) / (N*6/2) = 2. Correct.

### Signal Processing (Table 3)

- Paper: "4-10x lower reconstruction error" (abstract). Table MSE ratios: 4.2x, 5.3x, 10x. Range 4.2-10x. "4-10x" is correct.
- Paper: "5-20x more uniform" isotropy (line 441). Table isotropy ratios: 0.20x, 0.05x, 0.11x. Inverse: 5x, 20x, 9.1x. Range 5-20x. Correct.

### Rhombic Dodecahedron Properties (Section 2.1, line 141)

- 12 congruent rhombic faces: **Correct.**
- 14 vertices (8 of valence 3, 6 of valence 4): **Correct.** The 8 cube-type vertices have 3 edges; the 6 octahedron-type vertices have 4 edges.
- 24 edges: **Correct.** 12 faces * 4 edges per face / 2 = 24.
- Dual of cuboctahedron: **Correct.**
- Projection of tesseract: **Correct** (vertex-first projection of 4D hypercube).

### FCC Index "+7 to +20 pp" (Section 6.2, line 546)

From Table 6: 0.445-0.379=0.066 (6.6pp), 0.280-0.085=0.195 (19.5pp), 0.216-0.044=0.172 (17.2pp). Range is 6.6-19.5pp. Paper says "+7 to +20." 6.6 rounds to 7; 19.5 rounds to 20. Acceptable rounding, but the smallest gap (6.6pp) is on the boundary.

---

## Discrepancies Found (REPORT ONLY -- Paper 1 is locked)

### F-01-01: Algebraic Connectivity Range Stated as 2.3-2.5x but Data Shows 2.55x

- **Severity:** MINOR
- **Location:** Section 4.2 body text (line 316), Figure 2 caption (line 338), Discussion (line 634)
- **Paper claims:** "The algebraic connectivity ratio (2.3--2.5x)" -- stated three times
- **Table data (Table 1, line 306):** Scale ~1000 shows algebraic connectivity ratio = 2.55x
- **Correct range:** 2.31x to 2.55x, i.e., "2.3-2.6x" or more precisely "2.3-2.55x"
- **Impact:** The upper bound of the stated range (2.5x) is exceeded by the actual data (2.55x) by 2%. The abstract's "2.4x" (line 56) is a fair central summary of the data range and is not affected.
- **Note:** Paper 1 is LOCKED. This is a minor rounding/range discrepancy. The value 2.55 could be argued to "round to" 2.5, but as a range endpoint it should be 2.6 or stated as 2.55. Not egregious -- the abstract's summary figure (2.4x) is correct, and the table itself contains the accurate numbers.

---

## Summary

- **Claims checked:** 24
- **Discrepancies:** 1 (MINOR)

The paper's quantitative claims are highly accurate. Every node count, edge count formula, geometric property, percentage improvement, and ratio was verified either analytically or by cross-referencing with the source code. The single discrepancy is a minor range bound issue: the text states "2.3-2.5x" algebraic connectivity but the table data includes a value of 2.55x. The abstract's central summary of "2.4x" is correct. No CRITICAL or MAJOR errors found. Paper 1 does not require modification.
