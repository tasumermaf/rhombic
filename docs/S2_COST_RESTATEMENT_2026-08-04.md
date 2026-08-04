# §3 Cost-Table Restatement Against Measured Rates

**Filed 2026-08-04 by Meridian.** This is the artifact required by the H2-at-scale
registration's second lock condition (`REGISTRATION_H2_SCALE_2026-07-30.md`, and the
Director's Part 4 ruling promoting S2 to a precondition). It restates the draft's §3
cost table against measured per-run rates and supersedes every estimate in it.

Nothing here is estimated unless the line says PROJECTED. Every measured figure is
computed from a per-run artifact at emit time by
`scripts/s2_cost_restatement.py` (committed alongside), never hand-copied.

---

## 1. S2 timing pilots — what was measured

Nine timing-only runs, three per accessible family, excluded from every bank, computing
no H2 statistic. All nine `status = COMPLETE`; zero failures, zero OOMs. Identical
recipe across all runs: 2000 optimizer steps, bs4×ga4 (effective 16), 512 tokens,
16,384,000 token positions per run, warm HF cache, single RTX 6000 Ada (48 GB).

```
=== VERIFIED STATE: S2 MEASURED ===
gemma2-2b     runs = 3   wall_clock_min = 72.08 / 77.75 / 76.70   mean = 75.51   peak_vram_gb = 14.07
qwen2.5-3b    runs = 3   wall_clock_min = 82.21 / 82.34 / 84.03   mean = 82.86   peak_vram_gb = 11.05
qwen2.5-7b    runs = 3   wall_clock_min = 153.20 / 152.25 / 153.58 mean = 153.01  peak_vram_gb = 20.30
llama3.1-8b   runs = 0   UNMEASURED — HF license gate (GatedRepoError 403)
S2_PILOT_COST = 15.57 GPU-hours over 9 runs
=== END ===
```

VRAM headroom is comfortable everywhere: the largest measured family peaks at 20.30 GB
of 48 GB, so the S3 per-family geometry question does not arise for the measured rows —
bs4×ga4 holds, and the A1 bit-equivalence conditions are unchanged.

## 2. The restated §3 table

| Family | ~Params | Runs | Est min/run | **Measured** | Δ vs est | Est GPU-days | **Restated** |
|---|---|---|---|---|---|---|---|
| Gemma-2-2B | 2.6B | 120 | ~79 | **75.51** | **−4.42%** | ~6.6 | **6.292** |
| Qwen2.5-3B | 3.1B | 120 | ~93 | **82.86** | **−10.90%** | ~7.8 | **6.905** |
| Qwen2.5-7B | 7.6B | 60 | ~230 | **153.01** | **−33.47%** | ~9.6 | **6.375** |
| Llama-3.1-8B | 8.0B | 60 | ~243 | — *(gated)* | — | ~10.1 | **6.633 PROJECTED** |
| **Total** | | **360** | | | | **~34** | **26.206** *(3 measured + 1 projected)* |
| **3-family bank** (S1 drop option) | | **300** | | | | 24.0 | **19.573 measured** |

**The >25% overrun trigger does not fire.** Every measured family came in *under* its
estimate; the worst case is −4.42%. The Director's drop-option instruction is therefore
not activated by cost on any measured row.

## 3. Why the estimates were conservative, and what it implies for the gated row

The draft declared scaling **linear at ~30 min/B**. Measurement says the relationship is
affine with a large fixed cost and a *falling* marginal rate:

```
per-B measured:  gemma2-2b 29.04  ·  qwen2.5-3b 26.73  ·  qwen2.5-7b 20.13   min/B
least-squares (n=3, measured only):  min/run = 34.921 + 15.5352 x params_B
```

Roughly 35 minutes per run is fixed overhead independent of model size (data loading,
evaluation cadence, checkpoint writes), and only ~15.5 min/B scales. The flat 30 min/B
rule overcharges larger models, which is exactly where the −33.47% on Qwen2.5-7B comes
from.

Applying that fit to Llama-3.1-8B gives **159.20 min/run PROJECTED** (vs the draft's
~243), i.e. 6.633 GPU-days rather than 10.1. **This is a projection from three points and
is not a measurement** — it is offered only so the Director can see the shape of the
decision, and it does not satisfy S2 for that row. If the license gate is accepted, three
timing runs replace it with a measurement in ~8 GPU-hours.

## 4. S12 closed: the cost anchor exists and is now measured

The Director's Part 3 ruling found the "~42 min/run" anchor absent from the delivery
report and from `bank_manifest.json`, and held S1 on that basis. Both findings were
correct. The rate was nevertheless recoverable, from a file neither of us had parsed:
**`results/asset1-bank/campaign.log` carries `dur=` on every `RUN_END` line.**

```
=== VERIFIED STATE: ASSET-1 MEASURED PER-RUN (campaign.log, zero GPU time) ===
RUN_END lines = 539   COMPLETE = 535   indices with >1 COMPLETE (A1 restart) = 55
llama3.2-1b    n = 240   mean = 30.56 min/run   (min 28.9 / max 34.9)
qwen2.5-1.5b   n = 240   mean = 51.72 min/run   (min 48.8 / max 56.8)
blended (480)          = 41.14 min/run
=== END ===
```

Where an index has more than one COMPLETE (the A1 restart), the **last** is taken, which
is the bs4×ga4 cohort — the geometry the whole bank was re-executed under. So:

- The withdrawn "~42 min/run" figure was, as a blend, **approximately right (41.14)** —
  but it was asserted rather than measured, and the Director was correct to refuse a
  five-week commitment against an unmeasured number that two documents stated
  inconsistently. The discipline was right even though the number happened to be close.
- His gross upper bound of 50.3 min/run (campaign span ÷ 480) is consistent: the measured
  blend sits below it, the difference being the restart, the four HF-504 retries, and idle
  time.
- **The granularity design's "~30 min/run llama [ESTIMATE]" is now measured at 30.56.**

## 5. Granularity ladder — cost basis, also measured

Using the measured llama3.2-1b rate of 30.56 min/run (the granularity card is llama-only
per D1):

| Component | Runs | GPU-days |
|---|---|---|
| L1 — 12 classes × 20 seeds | 240 | 5.093 |
| L2 — 24 classes × 10 seeds | 240 | 5.093 |
| L3 — ~48 classes × 5 seeds | 240 | 5.093 |
| Arm B — squad-only deep ladder (D8, promoted) | 144 | 3.056 |
| D7 — split-pool control (mandatory) | 10 | 0.212 |
| **Total** | **874** | **18.546** |

The core ladder (L1–L3) is 15.28 GPU-days against the design's ~14–15 estimate — within
2%. Arm B and D7, both of which the Director added or promoted, account for the
remainder.

## 6. What this does and does not unlock

- **H2-at-scale, lock condition 1 (S2 rates published): SATISFIED for 3 of 4 families.**
  Llama-3.1-8B remains unmeasured on license grounds, not cost grounds.
- **H2-at-scale, lock condition 2 (§3 restated): SATISFIED by this document.**
- **Granularity, cost condition: SATISFIED** by §5 (measured, zero GPU time).
- **Granularity, D6 subsample size: STILL UNPINNED** — a value must be pinned at lock and
  this document does not presume to set it (see the accompanying brief's asks).
- **Still no bank runs**, on either card, until the Director rules on the residual items.

*Restatement computed by `scripts/s2_cost_restatement.py` from
`results/s2-timing-pilots/*/run_*/TIMING.md`, `results/asset1-bank/campaign.log`, and
`results/asset1-bank/bank_manifest.json`. Re-run it to reproduce every figure above.*
