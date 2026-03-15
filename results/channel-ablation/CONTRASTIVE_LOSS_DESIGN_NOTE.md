# Contrastive Loss Design Note — H2-H5 Implications

Created: 2026-03-14

## The Discovery

The contrastive loss function (`contrastive_bridge_loss`) is hardcoded for
n=6 channels only (line 118 of `train_contrastive_bridge.py`):

```python
if B.shape[0] != 6:
    continue  # Only meaningful for 6-channel bridges
```

This means H2 (n=4), H4 (n=8), and H5 (n=12) all train with spectral
regularization only — no directional pressure from contrastive loss.

H3 (n=6) is unaffected and will have full Steersman functionality.

## What H2 Shows (step 800)

Without contrastive loss, no block structure forms. Fiedler grows (0.079),
but all couplings remain uniform. Eigenvalue pattern: 4 distinct values
(no degeneracy). This is qualitatively different from n=6 which achieves
100% BD by step 200.

## Implication for H4 (n=8) and H5 (n=12)

H4 and H5 will behave like H2: connected but unstructured. The spectral
loss alone does not produce blocks regardless of n.

This makes H4/H5 **redundant with H2** under the current code. They will
all show the same result: spectral regularization → connectivity,
no block structure.

## Two Options

### Option A: Run H4/H5 As-Is (Spectral Only)

**Pro:** Confirms the H2 result at larger n. Shows that spectral-only
Steersman scales predictably. Simple, no code changes needed.

**Con:** No new information beyond H2. Three experiments showing the
same thing.

### Option B: Generalize Contrastive Loss for Arbitrary n

Define co-planar/cross-planar pairs for any n based on the principle:
"Channels i and j are co-planar if they correspond to the same RD axis."

For n=6: (0,1)=±x, (2,3)=±y, (4,5)=±z — 3 co-planar, 12 cross-planar.
This is what the current code implements.

For n=8: Three options for the extra 2 channels:
  a) 4 blocks of 2: {0,1}, {2,3}, {4,5}, {6,7} — 4th axis is arbitrary
  b) 3 blocks with extras: {0,1}, {2,3}, {4,5} co-planar, channels 6,7
     are "orphan" — no co-planar partner. Cross-planar to everything.
  c) 3 blocks of 2+: {0,1,6}, {2,3,7}, {4,5} — channels assigned to
     closest RD axis.

For n=12: {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11} — but only 3 RD axes.
Could assign {0,1,6,7}, {2,3,8,9}, {4,5,10,11} (4 channels per axis).

**Pro:** Tests whether the Steersman discovers 3-axis structure from a
larger space. The prediction is: n=8 should form 3 blocks (not 4), with
2 channels left as dead weight.

**Con:** Requires design decisions about how to label pairs for arbitrary n.
Any labeling encodes assumptions about the answer. Risk of circular
reasoning.

## Recommendation

**Run H3 as priority** (n=6 replication — has full contrastive loss).

For H4/H5, propose **Option B with "orphan" scheme**: co-planar pairs
exist only for the first 6 channels (3 blocks of 2). Channels 7+ have
NO co-planar partner — they can only be cross-planar. The contrastive
loss then pushes the Steersman to:
- Amplify the 3 natural RD axes (channels 0-5)
- Suppress everything involving channels 6+
- Discover that only 6 of 8 (or 12) channels are needed

This tests a sharper prediction: **the Steersman should discover that
the extra DOF are wasted and drive them to zero.** The effective rank
at n=8 should still be 3 (same as n=6 and n=3).

## Decision Needed

Timothy should decide before H4 launches whether to:
1. Run H4/H5 with spectral-only (redundant with H2 but clean)
2. Implement generalized contrastive loss (richer but requires code changes)
3. Skip H4/H5 entirely and spend the GPU time on more valuable experiments
