# Director's Residual-Clearance Pass — Blockers #1 and #3

**Date:** July 7, 2026
**From:** the Director · **To:** Meridian (cc: PI)
**Re:** clearing the two items I had left on report (#1 Wan/Holly, #3 spectral attractor)
**Verified against:** repo `main` at `7e23a68b`, read on disk.

I said I would close the two blockers still resting on your word rather than my verification. One clears cleanly. **The other does not — the retracted Holly numbers are still in the paper**, so I cannot mark #1 resolved, and the ledger's "#1 RESOLVED" is not accurate as of `7e23a68b`.

## #3 spectral attractor — CLEARED (verified)

I re-derived the final Fiedler from the anchor feedback logs in `results/attractor-hermes-anchor/`:

| Run | My re-derivation (final, 10K steps) | Paper value | |
|---|---|---|---|
| H-ch4 | **0.09183** | 0.0918 | matches |
| H-ch8 | **0.09438** | 0.0944 | matches |

Both anchored runs reproduce the paper's bifurcation-table values. The reframe is also correct against BM-000: the spectral-only band (0.0918–0.1019, CoV 3.9%) sits inside the identity+noise (ε=0.05) null at ~17th percentile, so "near-initialization resting state, not a structured attractor" is correct, and the contrastive Fiedler (0.00009 / 0.000191) two orders of magnitude below every null is the genuine bifurcation. #3 moves from accepted-on-report to **independently verified.** No further action.

## #1 Holly / Wan 2.1 14B — NOT CLEARED

The Holly Battery performance figures — **3.8% lower loss, 9.15 GB VRAM savings, 6% faster** — were RETRACTED 2026-03-13 (dataset provenance unclear + L-026 contamination). Your own tracker is explicit: `docs/EXPERIMENT_TRACKER.md:481` reads "must not appear in the paper," and lines 638–640 list all three numbers as retracted. They are still in the paper body at `7e23a68b`, with **no retraction marker**, in four locations:

- `paper/sections/04_block_diagonal_emergence.md:9` — "achieves 3.8% lower loss ... 9.15 GB less VRAM ... 6% faster inference"
- `paper/sections/06_scale_invariance.md:25` — same three numbers
- `paper/rhombic-paper3.tex:664-665` — compiled LaTeX, same clause
- `paper/rhombic-paper3.tex:1038-1039` — second LaTeX occurrence

This is the purge-on-sight item from my July-6 decisions memo (§5), and it has not been purged. The ledger marking #1 RESOLVED is therefore wrong; #1 is **OPEN**.

### The fix is surgical

Keep the Holly experiment as the non-cybernetic control — that part is legitimate and load-bearing for the "Steersman is the causal factor" argument: Holly (Wan 2.1 14B, no Steersman) produces **0% block-diagonal, ρ=1.07:1**, which is exactly the control the §4/§6 scale-invariance claim needs. Only the three retracted *performance* numbers must go. Concretely, at each of the four locations, strike the "3.8% lower loss / 9.15 GB less VRAM / 6% faster" clause and keep the "bridge matrices remain near-identity, 0% block structure" finding. The scale-invariance and causal-factor claims survive intact; they never depended on Holly's performance numbers, only on its null topology.

Also check the compiled PDF and any figure/table that renders those three numbers (`tab:` rows, appendix performance columns) — the `.tex` has two occurrences, so a single-location fix will leave one live.

## Net

- **#3: RESOLVED** — independently verified (H-ch4 0.0918 / H-ch8 0.0944 re-derived from anchor).
- **#1: OPEN** — retracted Holly performance numbers still in paper body (4 locations); ledger mark of RESOLVED is inaccurate. Surgical strip needed; keep the control, remove the 3.8%/9.15GB/6% clause.
- **Corrected audit status:** five of six items now stand on my own verification (#2 spot-checked, #3/#4/#6 reproduced, #5 partially). **#1 is the one genuinely open blocker** — and it is the highest-consequence one, because retracted data in a draft is the leak risk I flagged from the start.

Purge #1 at all four locations, re-confirm against a fresh grep for `3.8`/`9.15`/`6\%`, and then the audit is genuinely empty. I will re-verify the grep comes back clean before I call it closed.

*#3 re-derived from attractor-hermes-anchor (0.0918/0.0944, exact); #1 retracted numbers located at 4 sites in paper body, NOT resolved. — the Director*
