# Early Channel Ablation Results

*Generated 2026-03-13 — H1 (n=3) at step 8200/10000*

## H1 (n=3) vs C-001 (n=6): Matched-Step Comparison

| Step | n=3 Val Loss | n=6 Val Loss | Delta | Delta % |
|------|-------------|-------------|-------|---------|
| 100 | 0.4766 | 0.4765 | +0.0002 | +0.04% |
| 200 | 0.4476 | 0.4476 | -0.0000 | -0.01% |
| 500 | 0.4390 | 0.4390 | +0.0000 | +0.00% |
| 1000 | 0.4344 | 0.4344 | +0.0000 | +0.00% |
| 2000 | 0.4261 | 0.4261 | +0.0001 | +0.02% |
| 4000 | 0.4181 | 0.4178 | +0.0002 | +0.06% |

**Val loss is IDENTICAL within measurement noise across all matched steps.**

## Parameter Efficiency

| Metric | n=3 | n=6 |
|--------|-----|-----|
| Bridge params per module | 9 | 36 |
| Total bridge params (88 modules) | 792 | 3,168 |
| Param ratio | 1.0x | 4.0x |
| Val loss at step 4000 | 0.4181 | 0.4178 |
| Block-diagonal % | N/A (all pairs active) | 100% |

## Bridge Structure

n=3 bridges at step 8200 (sample — L0 q_proj):
```
[1.027  0.045  0.030]
[0.048  1.012  0.024]
[0.027 -0.045  1.020]
```

All 3 off-diagonal entries are active (0.02-0.05 magnitude). No block-diagonal
possible by construction — with 3 channels, every pair is connected.

n=6 bridges at step 4000 develop strict 2×2 blocks: co-planar pairs at ~0.48
magnitude, cross-planar pairs at ~0.00009.

## Fiedler Eigenvalue

| Step | n=3 Fiedler | n=6 Fiedler |
|------|-------------|-------------|
| 4000 | 0.0907 | 0.000408 |
| 8200 | 0.0942 | — |

Fiedler is not directly comparable across different n (different graph sizes).
n=6's low Fiedler reflects the near-disconnected block structure (3 independent
2×2 blocks ≈ 3 disconnected components → Fiedler → 0).

## Interpretation

**n=3 is the effective rank of the Steersman's discovered structure.**

The Steersman at n=6 drives 12 of 15 off-diagonal entries to zero (the
cross-planar pairs), leaving only 3 active DOF — one per RD coordinate axis.
n=3 naturally has exactly 3 off-diagonal entries, all active.

**This validates the RD geometry thesis:**
- n=6 provides 15 off-diagonal DOF but the Steersman uses only 3
- n=3 provides exactly 3 off-diagonal DOF and achieves identical performance
- The redundant parameters in n=6 are structurally informative (they reveal
  which DOF the system needs) but not performatively necessary

**Prediction for H2-H5:**
- n=4: Slight overhead (10 DOF → uses ~3). Same loss.
- n=6: 15 DOF → uses 3, 100% BD. Same loss.
- n=8: 28 DOF → uses 3. Same loss. Possible 4×2 block structure.
- n=12: 66 DOF → uses 3. Same loss. Most redundant.

**Decision gate:** If n=6 shows block-diagonal at 100% while n=4 does not,
this confirms that the Steersman's structure specifically requires the
6-face RD geometry (3 opposing-face pairs).

## Theoretical Framework: Block Structure vs Channel Count

| n | Off-diag DOF | 2×2 blocks possible | Max pairs | RD axes covered | Prediction |
|---|-------------|---------------------|-----------|-----------------|------------|
| 3 | 3 | No | 0 | All pairs active | Same val loss, no structure |
| 4 | 6 | Yes (3 partitions) | 2 | 2 of 3 | 2 blocks, 1 axis missing |
| 6 | 15 | Yes (15 partitions) | 3 | **3 of 3** | 3 blocks = full RD geometry |
| 8 | 28 | Yes (105 partitions) | 4 | 3+1 redundant | 3 strong + 1 weak pair? |
| 12 | 66 | Yes (10395 partitions) | 6 | 3+3 redundant | 3 blocks of 4? 6 degenerate pairs? |

**n=6 is the minimum n that can encode all 3 RD coordinate axes in 2×2 blocks.**
n=4 can encode at most 2 axes. n=3 cannot form blocks at all.

**Critical decision gate (H2 vs H3):** If n=4 forms 2 blocks but NOT 3,
while n=6 forms exactly 3, this confirms the 3-axis RD geometry as the
Steersman's attractor — not an artifact of having an even number of channels.

## Status

- H1 (n=3): Step 8400/10000, ~2h to completion
- H2 (n=4): Queued, auto-starts after H1
- H3 (n=6): Queued
- H4 (n=8): Queued
- H5 (n=12): Queued

Total ablation runtime estimate: ~10h from H2 start.
