# Paper 4 Experiment Trajectories — Complete Data

## Three Regimes Summary

| Regime | Experiments | Co/Cross | Fiedler | Signature |
|--------|-----------|----------|---------|-----------|
| **1: Block-Diagonal** | O-001, H-ch6, Seed-43/44, T-001r2 | 40K–470K:1 | ~10⁻⁵ | Extreme BD, near-zero Fiedler |
| **2: Spectral Attractor** | H-ch3, H-ch4, H-ch8, H-ch12 | N/A (no contrastive) | 0.084–0.102 | Connected, undirected |
| **3: Hierarchical Coherence** | E-001 | ~1.12:1 | 0.084 | Slight asymmetry, no BD |
| **4: Collapse** | WL-001, R-001 | ~10⁻⁵ | ~10⁻⁵ | Total collapse, no direction |

Note: WL-001 and R-001 are actually a FOURTH distinct outcome — they're not spectral
attractor (Fiedler too low) and not BD (co/cross ~0). They represent connectivity
COLLAPSE: the Steersman, given incoherent pair specifications, drives Fiedler to zero
(disconnecting) while deviation grows (extreme but undirected coupling). This is
different from spectral-only (which achieves stable connectivity at Fiedler ~0.09).

## WL-001 vs R-001: Nearly Identical Trajectories

These two experiments — wrong-labels (random partition) and resonance (prime-theoretic
pairs) — produce essentially the same trajectory. This means prime-theoretic structure
without geometric embedding is functionally equivalent to random noise from the
Steersman's perspective.

| Step | WL-001 Co/Cross | R-001 Co/Cross | WL-001 Fiedler | R-001 Fiedler |
|------|----------------|----------------|----------------|---------------|
| 1000 | 0.000649 | 0.000712 | 0.000163 | 0.000171 |
| 5000 | 0.000050 | 0.000051 | 0.000060 | 0.000064 |
| 10000 | 0.000009 | 0.000009 | 0.000013 | 0.000012 |

Max divergence: <10%. Functionally identical null results.

## E-001: The Third Regime (Emanation)

Emanation architecture (master bridge + per-layer offsets) produces a genuinely
different outcome: Fiedler stable at ~0.08 (near spectral attractor), co/cross
slowly rising from 1.05 to 1.12 (slight directional preference), deviation growing
from 0.09 to 0.35 (per-layer differentiation increasing).

Interpretation: The shared master bridge creates global connectivity (like
spectral-only), but the contrastive loss on the master creates slight directional
bias that propagates through offsets. Not enough to achieve BD, but enough to
break the symmetry of the spectral attractor.

## O-001: Emergence Trajectory (Strongest BD)

| Step | Co Mean | Cross Mean | Ratio |
|------|---------|------------|-------|
| 500 | 0.086 | 2.44e-5 | 3,539:1 |
| 2000 | 0.376 | 2.34e-5 | 16,051:1 |
| 5000 | 0.814 | 1.39e-5 | 58,543:1 |
| 8000 | 0.987 | 2.89e-6 | 341,287:1 |
| 10000 | 1.027 | 2.78e-6 | 369,365:1 |

Final (per-bridge): mean 473,622:1 | median 401,851:1 | min 126,782:1 | max 1,577,518:1

## Complete BD Hierarchy (Final Values)

| Experiment | n | Topology | Co/Cross (final) | Fiedler | Val Loss |
|-----------|---|----------|------------------|---------|----------|
| O-001 | 4 | octahedral | 473,622:1 (mean) | 1.1e-5 | 0.4010 |
| Seed-43 | 6 | RD contrastive | 73,309:1 | ~10⁻⁵ | 0.4015 |
| Seed-44 | 6 | RD contrastive | 70,201:1 | N/A | 0.4003 |
| H-ch6 | 6 | RD contrastive | 70,404:1 | 9e-5 | 0.4015 |
| T-001r2 | 8 | tesseract | 41,564:1 | 1.9e-4 | 0.4016 |
| E-001 | 6 | emanation | 1.12:1 | 0.084 | 0.4009 |
| WL-001 | 6 | wrong-labels | 8.7e-6 | 1.3e-5 | 0.4008 |
| R-001 | 6 | resonance | 8.9e-6 | 1.2e-5 | 0.4008 |

## Spectral Attractor (All Complete)

| Run | n | Fiedler (final) | Val Loss |
|-----|---|-----------------|----------|
| H-ch3 | 3 | 0.0951 | 0.4020 |
| H-ch4 | 4 | 0.0836 | 0.4022 |
| H-ch8 | 8 | 0.0944 | 0.4024 |
| H-ch12 | 12 | 0.1019 | 0.4025 |

Band: 0.0836–0.1019 (18% spread across 4× channel range)

*Computed March 18, 2026 from Hermes experiment data.*
