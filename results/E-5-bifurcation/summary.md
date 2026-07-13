# E-5 Bifurcation Sweep — Endpoint Tally

Pre-registration: `docs/E5_BIFURCATION_PREREG_2026-07-10.md`

**This is a tally, NOT the confirmatory analysis.** It reports the endpoint co/cross ratios and the pre-registered band classification (§4) so a partial or complete sweep can be inspected. The frozen predictions (P1 bimodality, P2 controls, the graded falsifier) are evaluated in the confirmatory analysis, not here.

Runs summarized: 15 / 15.

| f | seed | co/cross_trained | co/cross_true | fiedler_mean | magnitude | band |
|---|------|------------------|---------------|--------------|-----------|------|
| 0.00 | 42 | 3.67e+04 | 2.024 | 5.879e-05 | 0.03097 | Structured |
| 0.00 | 43 | 2.078e+05 | 3.115e-05 | 1.116e-05 | 0.05339 | Structured |
| 0.00 | 44 | 4.31e+04 | 1.974 | 5.426e-05 | 0.03083 | Structured |
| 0.25 | 42 | 3.721e+04 | 1.952 | 6.028e-05 | 0.03072 | Structured |
| 0.25 | 43 | 4.191e+04 | 18.05 | 0.0001105 | 0.03098 | Structured |
| 0.25 | 44 | 3.961e+04 | 1.967 | 5.557e-05 | 0.03092 | Structured |
| 0.50 | 42 | 3.884e+04 | 5.895 | 6.016e-05 | 0.03087 | Structured |
| 0.50 | 43 | 4.299e+04 | 4.299e+04 | 5.333e-05 | 0.03077 | Structured |
| 0.50 | 44 | 4.232e+04 | 5.883 | 5.196e-05 | 0.03081 | Structured |
| 0.75 | 42 | 3.771e+04 | 18.02 | 5.864e-05 | 0.03074 | Structured |
| 0.75 | 43 | 4.203e+04 | 4.203e+04 | 5.321e-05 | 0.03077 | Structured |
| 0.75 | 44 | 4.238e+04 | 17.78 | 5.429e-05 | 0.03087 | Structured |
| 1.00 | 42 | 3.661e+04 | 3.661e+04 | 6.174e-05 | 0.03089 | Structured |
| 1.00 | 43 | 4.171e+04 | 4.171e+04 | 5.235e-05 | 0.03077 | Structured |
| 1.00 | 44 | 4.283e+04 | 4.283e+04 | 5.536e-05 | 0.03079 | Structured |

## Tally (descriptive only)

- Runs in the Intermediate band [10¹, 10³): **0** / 15 (P1 predicts ≤ 2 across the full 15; falsifier ≥ 5).
- f = 1.0 runs all Structured (P2 positive control): **YES** (3/3 present).
- f = 0.0 runs all Unstructured (P2 negative control): **NO** (3/3 present).

Bands (§4): Structured ≥ 10³; Intermediate [10¹, 10³); Unstructured < 10¹. NonFinite (NaN endpoint) runs are defective, tallied separately, and never enter P1/P2.
