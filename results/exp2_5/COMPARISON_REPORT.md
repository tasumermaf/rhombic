# Exp 2 vs Exp 2.5: Dataset Effect on Bridge Behavior

> Exp 2: Alpaca-cleaned (isotropic)
> Exp 2.5: Geometric dataset (directionally structured)
> Key question: Does purpose-built data strengthen co-planar signal?

> **Deep analysis enabled** — includes Karkada-informed spectral diagnostics:
> Gram alignment, eigenvalue structure, permutation test (p-value),
> perturbation test (distributed vs local structure).


### FCC RhombiLoRA (6-ch)
| Metric | Alpaca | Geometric | Δ |
|--------|---:|---:|---:|
| Val Loss | 0.2884 | 0.0194 | -0.2690 |
| Train Loss | 0.3033 | 0.0201 | -0.2832 |
| Fiedler | 0.04011 | 0.03040 | 0.76× |
| Deviation | 0.19876 | 0.05779 | 0.29× |
| **Co/Cross Ratio** | **1.0186** | **1.0017** | **-0.0168** |
| Co-planar mean | 0.01286 | 0.00824 | |
| Cross-planar mean | 0.01262 | 0.00822 | |

### Co/Cross Learning Curve
| Step | Alpaca | Geometric | Δ |
|------|--------|----------|---|
| 1000 | 0.9286 | 0.9920 | +0.0634 |
| 2000 | 1.0230 | 1.0026 | -0.0204 |
| 3000 | 1.0910 | 1.0017 | -0.0893 |
| 4000 | 1.0633 | — | — |
| 5000 | 1.0430 | — | — |
| 6000 | 1.0444 | — | — |
| 7000 | 1.0335 | — | — |
| 8000 | 1.0239 | — | — |
| 9000 | 1.0227 | — | — |
| 10000 | 1.0186 | — | — |

---
## Spectral Analysis (Karkada Framework)


### Spectral Analysis: Exp 2 — Alpaca (rhombi_fcc_r24)
*112 bridge matrices analyzed*

**Gram Matrix Alignment** (lower = more aligned with RD coupling)
- Distance: 0.9210 ± 0.0151
- Range: [0.8774, 0.9594]

**Eigenvalue Structure**
- Learned:      ['1.0220', '1.0408', '1.0582', '1.0761', '1.0972', '1.1289']
- Theoretical:  ['-0.0000', '-0.0000', '0.0000', '4.0000', '4.0000', '16.0000']
- Spectral distance: 0.8935 ± 0.0086
- Eigenvalue spread: 0.10687

**Permutation Test** (co/cross under shuffled channel labels)
- Observed ratio: 1.0186
- Null: 1.0008 ± 0.0523
- p = 0.3320 ns

**Perturbation Test** (zero co-planar entries)
- Original Fiedler: 0.04011
- Perturbed Fiedler: 0.02761
- Retention: 68.83%
- DISTRIBUTED — structure survives co-planar zeroing

### Spectral Analysis: Exp 2.5 — Geometric (geometric_rhombi_fcc_r24)
*112 bridge matrices analyzed*

**Gram Matrix Alignment** (lower = more aligned with RD coupling)
- Distance: 0.9209 ± 0.0084
- Range: [0.8968, 0.9459]

**Eigenvalue Structure**
- Learned:      ['0.9826', '0.9926', '0.9999', '1.0104', '1.0246', '1.0450']
- Theoretical:  ['-0.0000', '-0.0000', '0.0000', '4.0000', '4.0000', '16.0000']
- Spectral distance: 0.9029 ± 0.0090
- Eigenvalue spread: 0.06246

**Permutation Test** (co/cross under shuffled channel labels)
- Observed ratio: 1.0017
- Null: 1.0006 ± 0.0455
- p = 0.4744 ns

**Perturbation Test** (zero co-planar entries)
- Original Fiedler: 0.03040
- Perturbed Fiedler: 0.02118
- Retention: 69.66%
- DISTRIBUTED — structure survives co-planar zeroing

### Head-to-Head Spectral Comparison
| Metric | Alpaca | Geometric | Better |
|--------|--------|----------|--------|
| Gram alignment (↓ better) | 0.9210 | 0.9209 | Geometric |
| Permutation p-value (↓ better) | 0.3320 | 0.4744 | Alpaca |
| Co/Cross observed | 1.0186 | 1.0017 | |

### Cubic RhombiLoRA (3-ch)
*Exp 2.5 results not found.*

### Standard LoRA (frozen bridge)
*Exp 2.5 results not found.*

---
## Verdict

**Decision criteria for proceeding to Exp 3A-3C:**
1. Co/Cross ratio > 1.10 (up from Exp 2's 1.019)
2. Permutation p-value < 0.05 (statistically significant directional preference)
3. Gram alignment closer to theoretical coupling under geometric data
4. Perturbation test shows distributed structure (retention > 50%)

Meeting criteria 1+2 is sufficient to proceed. 3+4 strengthen the case.
