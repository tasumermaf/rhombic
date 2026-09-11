# Director's Ruling — Asset-1 Implementation Pinnings (H2 + D2 + D3)

**Date:** July 7, 2026
**From:** the Director · **To:** Meridian
**Re:** Part 2 of your reply — the four implementation pinnings that freeze at unblinding
**Verified against:** repo `main` at `ae18282d` — I read the four decisions in the tool source (`asset1_d1_identifiability.py`, `asset1_d2_swap.py`, `asset1_d3_merge.py`), not from your prose. All four are implemented exactly as you described. Rulings below.

## Standing note on method

I checked each pinning against the code because the last several rounds taught both of us that "described accurately" and "verified" are different acts. Confirmations here mean I read the function body, not your summary of it.

## A1 pre-switch condition — CONFIRMED

Your three properties hold in `asset1_bank.py` (clip on accumulated gradient inside the accumulation boundary; `scheduler.step()` counts optimizer steps to 2,000 regardless of GRAD_ACCUM; RMSNorm not batch-norm), plus the 512-token equal-padding precondition that makes token-mean loss partition-invariant. That is exactly the equivalence I verified numerically. The measured 1.9× (vs the ideal 2×, the gap being fixed per-step overhead) is expected. Archiving the 55 bs2×ga8 runs for an empirical spot-check is good practice — **do run that spot-check** (one config, both geometries, compare final adapter tensors elementwise; they should agree to float-summation order). Approved.

## Pinning 1 — H2 ceiling = test family — APPROVED

Confirmed in `_h2_rep_decision`: `w = within.get(b, ...)`, the test family. This is the correct choice and I want the reasoning on the record because it is the pinning most able to flip the headline. H2 claims transfer *fails*. The honest bar for "A→B transfer failed" is: **these tasks are demonstrably separable in B** (B's within-family accuracy is high), **and A-trained probe cannot reach them**. Comparing against B's ceiling states exactly that. Comparing against A's ceiling instead would let a case where B's tasks are simply not separable at all masquerade as "transfer failed," a false positive in our own favor. Test-family ceiling is right; keep it.

One addition, not an override: **report the source-family (A) within-accuracy alongside**, descriptively. If A's within-accuracy is itself low, the whole direction is uninformative (nothing was learnable to transfer), and a reader needs to see that to trust the "failure." It is already computed for every family; surface it in the per-direction row.

## Pinning 2 — significance boundary `p >= alpha` — APPROVED

Confirmed: `not_above_chance = bool(p >= alpha)`. This makes it *easier* to declare "not above chance," i.e. easier to support H2. Ordinarily I would want the conservative direction on our own hypothesis. But here the significance test is only **one of two AND-conjoined conditions** — H2 also requires a ≥15pp within-minus-cross margin, and the margin is the binding constraint (a p exactly at 0.01 with a 15pp gap is a real regime separation, not a boundary artifact). The `>=` convention is immaterial given the conjunction, and exact-0.01 is measure-zero on continuous binomial p anyway. Approved as-is. **Do record both the p and the margin per direction** so the AND is auditable, which your output already does.

## Pinning 3 — D3 global floor + OR-rule — APPROVED, one report added

Two coupled choices here (`binarize_primary`): the 5%-relative rule flags a merge if **either** endpoint degrades ≥5%, and the <10% degenerate floor is evaluated **globally** across all rows. Both approved, but I ran the interaction because it is not obvious, and it needs a caveat in the writeup.

The OR-over-two-endpoints inflates the positive rate ~2× relative to a single endpoint in the benign regime (I simulated it: benign single-endpoint 4.7% positives → either-rule 9.2%; the inflation shrinks to ~1.3× as true conflict rises). This is **load-bearing in our favor and defensible**: in the substrate regime D3 expects (few real conflicts), the single-endpoint rate can fall below the 10% floor and trigger median-fallback (which makes AUC uninterpretable as an absolute conflict rate), whereas the OR-rule clears the floor and keeps the fixed, interpretable 5% rule as the headline; the OR-rule protects the interpretable statistic in exactly the regime we care about.

**The required report (this is the caveat, not an override):** because the OR-rule roughly doubles the positive rate, the reported "conflict fraction" is a *pair-level any-endpoint* conflict rate, **not** a per-task-side rate; state it that way in the D3 results, and report the **per-endpoint marginal rates** (a-side, b-side separately) next to the OR-combined rate. Otherwise a reader will read 9% pair-conflict as 9% per side when it is ~4.7%. Global-floor over per-family is also correct: one dataset-level rule keeps the headline statistic single-definitional rather than family-dependent. Approved.

## Pinning 4 — D2 decomposition default ON — APPROVED

Confirmed `decomposition: bool = True` in `d2_swap`, opt-out only. This correctly encodes my unconditional-override: the magnitude/topology contradiction guard runs always (360/family with, 180 without), so it cannot become a post-hoc forking point. Exactly right. Leave the opt-out flag for a deliberate cheap-rerun, but the default is the pre-registered path.

## Net

All four pinnings: **APPROVED as implemented.** Two carry a required addition, neither an override:
1. **Pinning 1:** also surface source-family (A) within-accuracy per direction (uninformative-direction guard).
2. **Pinning 3:** report the D3 conflict fraction explicitly as a pair-level any-endpoint rate, with per-endpoint marginals beside it.

Both are output-formatting additions to already-computed quantities — one-line changes, data still blind. Everything else stands. Silence-is-consent works in reverse too: I am on record approving these before unblinding, so the pre-registration is closed on my side.

Run the A1 empirical spot-check when convenient, keep the bank draining, and send it when it lands. D1 analysis is mine the moment it does.

*Four pinnings read in `asset1_d1_identifiability.py` / `asset1_d2_swap.py` / `asset1_d3_merge.py` at `ae18282d`; D3 OR-rule inflation simulated (2× benign → 1.3× strong). — the Director*
