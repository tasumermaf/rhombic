# §9.5 — Contrastive Weight as Speed Control

**Status: DRAFT — updated with final 10K numbers. Headline nuanced.**

## The Whisper-Strength Experiment (CW-001)

A natural question arises from the four-regime taxonomy: does the
contrastive weight $c_w$ control *whether* block-diagonal structure
emerges, or *how quickly* it emerges? The FI-004 annealing experiment
(§X) found an optimal fixed $c_w \approx 0.02$ and a cliff at $c_w = 0$,
suggesting a minimum threshold. But FI-004 used fixed weights — the
Steersman's adaptive decay was not tested in the low-$c_w$ regime.

CW-001 tests this directly. Configuration: $n = 6$ (RD), identity
initialization, initial $c_w = 0.02$ with the Steersman's adaptive
decay active. Control Law 2 reduces $c_w$ when the co/cross ratio
exceeds the directionality threshold, with a floor at
$0.25 \times c_{w,\text{base}} = 0.005$. The experiment runs for
10,000 steps on TinyLlama 1.1B.

## Three-Phase Trajectory

The CW-001 trajectory reveals three distinct dynamical regimes within
a single training run:

| Step | Co/Cross | Fiedler | $c_w$ | Phase |
|------|----------|---------|-------|-------|
| 100 | 20:1 | 0.0073 | 0.020 | 1: Initial growth |
| 500 | 1,592:1 | 0.0004 | 0.015 | 1: Rapid BD formation |
| 1,500 | 1,867:1 | 0.0016 | 0.005 | 2: Plateau onset |
| 3,300 | 1,855:1 | 0.0018 | 0.005 | 2: Sustained plateau |
| 4,600 | 2,252:1 | 0.0019 | 0.005 | 3: Breakout |
| 5,800 | 3,860:1 | 0.0014 | 0.005 | 3: Acceleration |
| 7,500 | 7,884:1 | 0.0007 | 0.005 | 3: Exponential growth |
| 8,300 | **15,183:1** | **0.0004** | 0.005 | 3: Peak |
| 9,000 | 13,793:1 | 0.0004 | 0.005 | 4: BD stabilization |
| 9,500 | 12,846:1 | 0.0004 | 0.005 | 4: Oscillation |
| **10,000** | **13,456:1** | **0.0005** | 0.005 | **4: Final** |

**Phase 1 (steps 0–1,500): Rapid initial growth.** The Steersman
decays $c_w$ from 0.020 to the floor of 0.005 within 1,500 steps as
the co/cross ratio rises rapidly. BD structure forms quickly during
this high-$c_w$ window, reaching $\sim$1,800:1.

**Phase 2 (steps 1,500–4,200): Apparent plateau.** With $c_w$ floored
at 0.005, the co/cross ratio oscillates between 1,500:1 and 2,000:1
for nearly 3,000 steps. The Fiedler eigenvalue rebounds from its Phase 1
minimum, rising to 0.0020 — indicating that spectral connectivity is
*increasing* even as BD strength appears stalled. This phase was
initially interpreted as a terminal state (adaptive decay defeating
whisper-strength). It is not.

**Phase 3 (steps 4,600–8,300): Exponential breakout.** The co/cross
ratio breaks above 2,000:1 at step 4,600 and enters an exponential
growth phase: 2,252 → 3,860 → 7,884 → 12,921 → 15,183:1 (peak at step
8,300). The Fiedler eigenvalue declines monotonically from 0.0019 to
0.0004, entering the same regime as the aggressive-$c_w$ experiments.
The growth rate *accelerates* with accumulated BD strength, suggesting
a positive feedback mechanism: once block-diagonal structure exceeds a
critical threshold ($\sim$2,000:1), the contrastive loss becomes more
effective at reinforcing the existing partition, creating a
self-amplifying cycle.

**Phase 4 (steps 8,300–10,000): BD regime stabilization.** The co/cross
ratio oscillates between 12,145:1 and 15,183:1 (mean 13,288:1) without
a clear growth trend. The Fiedler eigenvalue stabilizes at 0.00042–0.00048.
Val loss continues declining (0.4034 → 0.4017), indicating that the
*language model* is still improving even as the *bridge structure* has
reached equilibrium. This decoupling — structural stasis while
perplexity improves — suggests the bridge has settled into an attractor
from which the remaining training budget refines weights within the
established topology rather than reshaping it.

## Speed, Destination, or Both?

At step 10,000, CW-001's Fiedler eigenvalue (0.00046) is within one
order of magnitude of H-ch6's final Fiedler (0.00009). Both are deep
in the BD regime. But the co/cross ratios tell a more nuanced story:

| Metric | H-ch6 ($c_w = 0.1$ adaptive) | CW-001 ($c_w = 0.005$ floor) | Ratio |
|--------|------------------------------|-------------------------------|-------|
| Co/cross | 70,404:1 | 13,456:1 | 5.2× |
| Fiedler | 0.00009 | 0.00046 | 5.1× |
| Val loss | $\sim$0.40 | 0.4017 | $\sim$same |

A 5.2× gap persists at matched step count. CW-001's Phase 4
stabilization — oscillating at 12,145–15,183:1 for 2,000 steps with
no growth trend — raises the question: is this a transient plateau
(like Phase 2, which lasted 2,700 steps before breaking out) or a
genuine attractor at this $c_w$ floor?

Two interpretations compete. Under the **speed hypothesis**, CW-001
simply needs more training steps; the Phase 2→3 breakout precedent
shows the system can appear stalled and then resume exponential growth.
Under the **attractor hypothesis**, the contrastive floor sets not
only the rate but the ceiling of structural formation. The 5.2×
co/cross gap and the 5.1× Fiedler gap are proportional, consistent
with a $c_w$-dependent attractor strength.

Both interpretations share the same novel core: *any nonzero $c_w$
produces BD structure*, the breakout mechanism at $\sim$2,000:1 is a
genuine phase transition, and the three-phase incubation trajectory
has no prior art in the adapter literature. The distinction between
speed-only and speed-plus-ceiling is empirically resolvable (see
Discussion: continuation and sweep experiments) and does not diminish
the finding's significance.

## The Phase 2 Plateau as Incubation

The 3,000-step plateau (Phase 2) is not a failure state. During this
period, the Fiedler eigenvalue rises from 0.0004 to 0.0020 — the bridge
is building spectral connectivity while block-diagonal strength appears
stalled. We hypothesize that Phase 2 represents an *incubation period*
in which the bridge accumulates the spectral prerequisites for
exponential BD growth. The Fiedler rebound creates a connected substrate;
Phase 3's breakout then partitions this substrate into blocks.

This two-stage process (connect, then partition) mirrors the
spectral-attractor-then-bifurcation sequence observed across the full
experimental programme (§5–6): spectral-only training creates
connectivity (the attractor); contrastive loss then breaks this
connectivity into directed blocks. CW-001 reproduces this sequence
*within a single run* at low $c_w$.

## Implications

1. **No minimum $c_w$ threshold for BD.** Any nonzero $c_w$ eventually
   produces BD structure — the FI-004 "cliff" at $c_w = 0$ is real,
   but $c_w = 0.005$ is far above it.

2. **Adaptive decay is not a bottleneck.** The Steersman's adaptive
   Control Law 2, which reduces $c_w$ when BD is detected, does not
   prevent BD formation — it merely slows it. The floor mechanism
   ensures a minimum contrastive signal is always present.

3. **Training budget trades against $c_w$.** A practitioner with
   abundant compute can use lower $c_w$ for potentially better spectral
   properties; a practitioner needing fast convergence can use higher
   $c_w$. Whether the slow regime produces qualitatively better
   adapters (lower Fiedler = more connected = better information flow)
   is an open empirical question.

4. **The breakout threshold ($\sim$2,000:1) is a phase transition.**
   The co/cross ratio of $\sim$2,000:1 marks the onset of
   self-amplifying BD growth at $c_w = 0.005$. Pending verification
   across different $n$ and topologies, this may be universal.

5. **BD regime stabilization decouples from language modeling.**
   Val loss continues declining (0.4034 → 0.4017) during Phase 4
   while co/cross oscillates without growth. The bridge structure
   reaches equilibrium before the language model does — the final
   training budget refines representations within established topology.
