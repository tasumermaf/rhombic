# H-ch4 Eigenvalue Evolution Analysis

> **Experiment:** H2 channel ablation, n=4, identity init, cybernetic training
> **Model:** TinyLlama-1.1B-Chat-v1.0
> **Bridge params:** 1,408 (4x4 per layer x 22 layers)
> **Key condition:** Contrastive loss DISABLED (n!=6); only spectral regularization active
> **co_cross_ratio:** Always null (requires n=6)
> **Source:** `~/rhombic/results/channel-ablation/H-ch4/results.json` on Hermes
> **Comparison data:** `H-ch3/results.json` (n=3, 10,000 steps, 2 non-zero eigenvalues)

---

## 1. Raw Eigenvalue Table

The `eigenvalue_pattern` in `feedback_log` reports Laplacian eigenvalues of the
mean bridge matrix across all 22 layers x 4 attention projections. A 4x4
Laplacian always has one zero eigenvalue (connected graph guarantee). The three
non-zero eigenvalues are labeled lambda_2 (Fiedler), lambda_3, lambda_4.

| Step | lambda_1 (zero) | lambda_2 (Fiedler) | lambda_3 | lambda_4 | sum(lambda) |
|-----:|----------------:|-------------------:|---------:|---------:|------------:|
|    0 |            0.0  |         0.000000   | 0.000000 | 0.000000 |    0.000000 |
|  100 |           ~0.0  |         0.013523   | 0.016603 | 0.017665 |    0.047790 |
|  200 |           ~0.0  |         0.035824   | 0.041518 | 0.045954 |    0.123297 |
|  300 |           ~0.0  |         0.054683   | 0.063939 | 0.070314 |    0.188936 |
|  400 |           ~0.0  |         0.064640   | 0.075672 | 0.083037 |    0.223350 |
|  500 |           ~0.0  |         0.068809   | 0.081405 | 0.089173 |    0.239387 |
|  600 |           ~0.0  |         0.070337   | 0.084104 | 0.091756 |    0.246198 |
|  700 |           ~0.0  |         0.071305   | 0.085074 | 0.093052 |    0.249431 |

All three non-zero eigenvalues grow monotonically from zero (identity init).
Growth rate decelerates strongly: the delta from step 0-100 is an order of
magnitude larger than from step 600-700. The system is approaching a fixed point.

---

## 2. Eigenvalue Ratios (Structure Indicators)

The critical diagnostic: if eigenvalues converge toward the same value (ratio -> 1.0),
the Laplacian is approaching a degenerate spectrum, implying block/cluster structure.
If they diverge, the graph is developing a non-uniform topology.

| Step | lambda_3/lambda_2 | lambda_4/lambda_2 | lambda_4/lambda_3 | Spread (lambda_4 - lambda_2) |
|-----:|------------------:|------------------:|------------------:|-----------------------------:|
|  100 |          1.22774  |          1.30636  |          1.06400  |                    0.004142  |
|  200 |          1.15895  |          1.28280  |          1.10686  |                    0.010131  |
|  300 |          1.16929  |          1.28574  |          1.09973  |                    0.015631  |
|  400 |          1.17065  |          1.28461  |          1.09732  |                    0.018397  |
|  500 |          1.18305  |          1.29596  |          1.09544  |                    0.020364  |
|  600 |          1.19577  |          1.30455  |          1.09098  |                    0.021419  |
|  700 |          1.19312  |          1.30499  |          1.09373  |                    0.021747  |

### Key observations:

**lambda_3/lambda_2** starts at 1.228, drops to ~1.159 by step 200, then **rises back**
to ~1.193 by step 700. The ratio is NOT converging to 1.0. After an initial dip, it
stabilizes around 1.19 with slight upward drift.

**lambda_4/lambda_2** starts at 1.306, drops to ~1.283 by step 200, then **rises back**
to ~1.305 by step 700. Nearly returns to its initial value. Stable around 1.29-1.30.

**lambda_4/lambda_3** starts at 1.064, rises to 1.107 by step 200, then **slowly
decreases** to ~1.094 by step 700. The top two eigenvalues are slightly closer to each
other than they are to lambda_2, but the ratio is stable, not converging to 1.0.

**Spread (lambda_4 - lambda_2)** grows monotonically from 0.004 to 0.022. In absolute
terms the eigenvalues are spreading apart, not collapsing together.

---

## 3. Growth Rate Analysis

| Step | d(lambda_2)/100 | d(lambda_3)/100 | d(lambda_4)/100 | lambda_2 % change |
|-----:|----------------:|----------------:|----------------:|------------------:|
|  100 |       0.013523  |       0.016603  |       0.017665  |              inf  |
|  200 |       0.022301  |       0.024916  |       0.028290  |          164.9%   |
|  300 |       0.018859  |       0.022421  |       0.024360  |           52.6%   |
|  400 |       0.009957  |       0.011733  |       0.012724  |           18.2%   |
|  500 |       0.004169  |       0.005733  |       0.006136  |            6.4%   |
|  600 |       0.001528  |       0.002699  |       0.002584  |            2.2%   |
|  700 |       0.000968  |       0.000970  |       0.001296  |            1.4%   |

Lambda_2 growth decelerates from 165% per 100 steps to 1.4%. The system is rapidly
converging to a fixed point. By step 700, absolute change per step is ~0.00001.

All three eigenvalues decelerate at similar rates — they are growing in proportion,
not differentially. This means **the ratio structure is set early (by step 200-300)
and then preserved as the system scales up**.

---

## 4. Spectral Gap and Fiedler Analysis

| Step | Spectral Gap | lambda_2 / sum | lambda_4 / sum | Fiedler Mean | Fiedler Std |
|-----:|-------------:|---------------:|---------------:|-------------:|------------:|
|  100 |       0.8046 |        0.2829  |        0.3695  |      0.01383 |     0.00368 |
|  200 |       0.7803 |        0.2906  |        0.3727  |      0.03536 |     0.01427 |
|  300 |       0.7536 |        0.2894  |        0.3723  |      0.05286 |     0.02011 |
|  400 |       0.7454 |        0.2893  |        0.3718  |      0.06338 |     0.02118 |
|  500 |       0.7401 |        0.2874  |        0.3725  |      0.06986 |     0.02058 |
|  600 |       0.7355 |        0.2857  |        0.3727  |      0.07402 |     0.01971 |
|  700 |       0.7341 |        0.2859  |        0.3730  |      0.07706 |     0.01868 |

The spectral gap (1 - lambda_2/lambda_max) decreases monotonically from 0.805 to 0.734.
This means connectivity is increasing — the graph is becoming more uniformly connected.

The **fractional distribution is remarkably stable**: lambda_2 consistently accounts
for ~28.6% of the total and lambda_4 for ~37.3%. The eigenvalue spectrum scales up
self-similarly. No structural reorganization is occurring.

Fiedler standard deviation decreases from step 500 onward: the per-layer bridge
matrices are becoming more similar to each other.

---

## 5. Comparison with H-ch3

H-ch3 has n=3 channels (2 non-zero eigenvalues). It ran for 10,000 steps, providing
a long-run view of what happens under spectral-only regularization.

| Step | ch3: lambda_3/lambda_2 | ch4: lambda_3/lambda_2 | ch4: lambda_4/lambda_2 |
|-----:|-----------------------:|-----------------------:|-----------------------:|
|  100 |               1.16876  |               1.22774  |               1.30636  |
|  200 |               1.14196  |               1.15895  |               1.28280  |
|  300 |               1.14065  |               1.16929  |               1.28574  |
|  400 |               1.14144  |               1.17065  |               1.28461  |
|  500 |               1.14447  |               1.18305  |               1.29596  |
|  600 |               1.14311  |               1.19577  |               1.30455  |
|  700 |               1.14255  |               1.19312  |               1.30499  |

**ch3 eigenvalue spread at step 700:**  spread = (0.10385 - 0.09089) / 0.09089 = **14.3%**
**ch4 max eigenvalue spread at step 700:** spread = (0.09305 - 0.07130) / 0.07130 = **30.5%**

The ch4 system has **more than double** the relative eigenvalue spread of ch3. Adding
a fourth channel creates a wider spectral range, not a narrower one.

### H-ch3 Late-Stage Divergence (Long-Run Behavior)

| Step  | ch3: lambda_2 | ch3: lambda_3 | ch3: lambda_3/lambda_2 |
|------:|--------------:|--------------:|-----------------------:|
|   700 |      0.09089  |      0.10385  |               1.14255  |
| 1,000 |      0.09279  |      0.10617  |               1.14417  |
| 2,000 |      0.09019  |      0.10500  |               1.16423  |
| 3,000 |      0.09416  |      0.11176  |               1.18692  |
| 5,000 |      0.08351  |      0.11876  |               1.42208  |
| 7,000 |      0.08100  |      0.11712  |               1.44600  |
|10,000 |      0.07666  |      0.11781  |               1.53701  |

**This is the critical finding.** In ch3, the eigenvalue ratio is stable at ~1.14
through step 1000, then **slowly diverges** to 1.54 by step 10,000. Lambda_2
(Fiedler) actually **decreases** from 0.093 to 0.077 while lambda_3 **increases**
from 0.106 to 0.118. The eigenvalues move in opposite directions in the long run.

The spectral regularization (which targets a Fiedler gap of 0.1) is pulling lambda_2
upward, but the task loss gradient wants asymmetric structure, pushing eigenvalues
apart. By step 5000, the task gradient dominates and the ratio diverges rapidly.

**Prediction for ch4:** The same divergence pattern should appear after ~1000-2000
steps. The current ratio stability at step 700 (lambda_4/lambda_2 = 1.305) will
likely widen to 1.5+ by step 5000, mirroring the ch3 trajectory.

---

## 6. Degeneracy Analysis

The n=6 case (not yet run) is expected to produce **degenerate eigenvalues**
[0, lambda, lambda, lambda, mu, mu] — a 3-fold and 2-fold degeneracy implying
block structure (the contrastive loss, when active, explicitly encourages this
by clustering channels into groups).

For n=4, with contrastive loss disabled, the question is: does spectral
regularization alone produce any degeneracy?

**Answer: No.**

All three non-zero eigenvalues remain distinct at every measured step. The ratios
lambda_3/lambda_2 (~1.19) and lambda_4/lambda_3 (~1.09) are stable and clearly
separated from 1.0. There is no approach toward any degenerate pattern.

The closest pair is lambda_3 and lambda_4 (ratio ~1.09), but this ratio is
**increasing** slightly over training, not decreasing toward 1.0.

**Multiplicity at every step: [1, 1, 1, 1]** — four distinct eigenvalues, no
degeneracy. This is the complete opposite of the expected n=6 pattern.

---

## 7. Interpretation

### What spectral regularization alone does (n=4, no contrastive)

1. **Grows all eigenvalues from zero** (identity init -> connected graph)
2. **Preserves proportional structure** — the ratio between eigenvalues is
   set in the first 200 steps and then approximately maintained
3. **Does NOT induce block structure** — no eigenvalue degeneracy emerges
4. **Creates a uniformly-scaled, non-degenerate Laplacian** — each channel
   is differently coupled to each other channel, with no preferred grouping
5. **Decelerates rapidly** — the spectral loss drops from 4.5e-4 to 1.1e-5
   across 700 steps, meaning it achieves its target (Fiedler gap -> 0.1)
   quickly and then becomes passive

### The spectral weight decay amplifies this

The cybernetic controller decays spectral_weight from 0.050 to 0.025 over 700
steps (floor reached at step 700). This is correct behavior — the controller
sees connectivity IMPROVING and reduces pressure. But it means the spectral
loss has even less influence in later steps, ceding control to the task loss.

### Contrast with the expected n=6 pattern

For n=6 with contrastive loss active:
- Contrastive loss explicitly clusters channels into groups (minimizes
  intra-group distance, maximizes inter-group distance)
- This should produce degenerate eigenvalues corresponding to the group structure
- The spectral regularization ensures the groups remain connected

For n=4 with contrastive loss disabled:
- No clustering pressure exists
- Spectral regularization pushes toward connectivity but is agnostic about
  symmetry or block structure
- The result is a connected but asymmetric graph: each channel has different
  coupling strengths to each other

### The ch3 long-run warning

The ch3 data (10,000 steps) shows that after ~2000 steps, the task loss
gradient overpowers spectral regularization and the eigenvalue ratio diverges
from 1.14 to 1.54. Lambda_2 (Fiedler, the weakest connection) actually
*decreases* while lambda_3 increases — the graph becomes more asymmetric.

This strongly suggests that **spectral regularization alone is insufficient
to maintain spectral balance in the long run**. The contrastive loss in the
n=6 case may serve a dual role: (1) inducing block structure, and (2)
providing a countervailing force against the task gradient's tendency to
create asymmetric channel coupling.

---

## 8. Summary

| Property | H-ch3 (n=3) | H-ch4 (n=4) | Expected H-ch6 (n=6) |
|----------|-------------|-------------|----------------------|
| Non-zero eigenvalues | 2 | 3 | 5 |
| Eigenvalue degeneracy | None | None | [0, l, l, l, m, m] |
| Max ratio at step 700 | 1.143 | 1.305 | TBD |
| Ratio direction (early) | Stable | Stable | TBD |
| Ratio direction (late) | **Diverging** (1.54 at 10k) | TBD (predict diverging) | TBD |
| Contrastive loss | Disabled | Disabled | **Active** |
| Block structure | No | No | Expected yes |

**Bottom line:** With n=4 and spectral-only regularization, the bridge develops
a connected but non-degenerate Laplacian. Eigenvalue ratios stabilize early
(~1.19, ~1.30) and the system converges toward a fixed point by step 700. The
ch3 long-run data predicts these ratios will diverge after ~2000 steps as the
task gradient dominates. No block structure emerges without contrastive loss.
The n=6 experiment (with contrastive loss active) should show qualitatively
different behavior — degenerate eigenvalues and stable long-run ratios.

---

*Analysis performed 2026-03-13 from results.json on Hermes.*
*H-ch3 comparison data from the same channel-ablation directory.*
