# Q-08-1 — The Learned-Noise-Margin Question: RESULTS

```
analysis script : C:/falco/rhombic/scripts/q081_noise_margin.py
commit          : pending, filled by Meridian at commit
results         : C:/falco/rhombic/results/q081-noise-margin/
verifier        : fresh context, re-derives C1-C3 from the .npy files independently
                  of the analysis script (§9)
```

Run 2026-09-14. ZERO GPU-seconds; no gpu_guard claim taken; CPU only (card §10). Wall clock 105.8 s on a warm
filesystem cache. A first, cold run of the same script over the same tree spent
128.6 s in the Arm A load alone (83,712 `.npy` reads in total across all arms), so
a verifier reproducing this from a cold cache should expect roughly three to four
times the figure above and not treat the difference as a discrepancy.

## Status block

```
CARD                  = Q-08-1, graded 2026-09-06; additions 2026-09-11;
                        decider pins 2026-09-14
ITEM 1                = UNRULED BY THE DIRECTOR. Arm B's descent from a
                        corpus-coupled initialization has NOT been ruled on.
                        Arm B stays CONFIRMATORY as drafted, and C2 below is
                        reported PENDING THAT RULING.
ITEM 9                = OPEN. n = 8 on Arm A, n = 6 on Arms B/C/D.
ARMS C AND D          = DESCRIPTIVE ONLY. No p-value, no permutation test, no
                        inference. Confounds in the same sentence as the number.
COMPUTE               = zero GPU-seconds, no gpu_guard claim (§10)
CORPUS BOUNDARY       = no protected name or value is read, computed or reported
```

**Scope limit, carried in every claim below.** TinyLlama-1.1B (descriptively Qwen2.5-1.5B / Qwen2.5-7B), TeLoRA bridge couplings under the Steersman contrastive objective, at the configurations already written to disk; detector v2 for E-5, v1 for the FI series — no cross-detector depth comparison. No claim about learned systems in general and none about silicon (§8).

## 1. The census, as asserted at run time (§12 Item 8)

Asserted against the live filesystem **before any endpoint was computed**; the
analysis raises `CensusError` and halts on the first disagreement. A census
verified on 2026-09-11 or 2026-09-14 is not a census verified at run time.

| Arm | Directory | Runs | Step indices | Files/index | Step files | Finals | n | .npy B |
|---|---|---|---|---|---|---|---|---|
| A | `E-5-bifurcation` | 15 | 31 | 88 | 40920 | 1320 | 8 | 384 |
| B | `fi-004` | 1 | 30 | 88 | 2640 | 0 | 6 | 272 |
| B | `fi-003` | 1 | 12 | 88 | 1056 | 0 | 6 | 272 |
| C | `exp3_tinyllama` | 1 | 101 | 88 | 8888 | 88 | 6 | 272 |
| C | `fc-001` | 1 | 19 | 112 | 2128 | 112 | 6 | 272 |
| C | `fc-001-fresh` | 1 | 25 | 112 | 2800 | 112 | 6 | 272 |
| C | `exp3` | 1 | 130 | 112 | 14560 | 112 | 6 | 272 |
| D | `cw-001` | 1 | 101 | 88 | 8888 | 88 | 6 | 272 |

Arm A's 15 runs carry ONE step-index-set signature, `46370c6576ae` — the index sets are
identical, not merely equal in total, and the value equals the one pinned at §12
Item 8, which the census asserts rather than merely prints. The serialization is
`sha256(str(sorted list of distinct indices))`, first 12 hex — stated because the
digest is serialization-dependent and a verifier hashing a tuple or a joined string
gets a different value from the same, correct index set. Zero zero-byte `.npy`
anywhere in any arm.
Verdict: **CENSUS ASSERTED AT RUN TIME — every pinned count matches the live filesystem**

## 2. The Occ band, pinned and not refitted (§12 Item 2)

```
OCC_BAND         = CLOSED [0.01959, 0.11538] on |B[i,j]|
OCC_BAND_RATIO   = 5.889740
OCC_BAND_DECADES = 0.770096
PROVENANCE       = E-5 RESULTS.md:59-61, at the precision the record prints
REFIT            = FORBIDDEN for every arm, checkpoint and scale
```

The band spans **0.770 decades**, not a factor of ten, so this document writes
*"the inter-mode band (0.770 decades)"* wherever §4 says "decade". Measured on
the same pooled entries, no power-of-ten decade between the modes is empty:

```
[1e-4,1e-3]       139 entries
[1e-3,1e-2]        12 entries
[1e-2,1e-1]         2 entries
[1e-1,1e0]      5,280 entries
```

**Vacuity caveat, carried not buried.** A band whose endpoints are the data's own extrema is empty by construction. Observed Occ = 0 at E-5's final checkpoint is NOT itself a finding; C3 tests that 0 against the N2 prediction, and only the prediction side is unknown. On Arms A and B the band is fixed while the data move, so Occ there is a genuine measurement.

## 3. E-5's published statistics, re-derived from the artifacts (§9)

Loading the 15 x 88 committed `bridge_final_*.npy` and splitting each adapter's
28 off-diagonal entries by that run's `config.json` trained pair spec:

```
co entries            5,280
cross entries         31,680
pooled entries        36,960
co mean               0.226302        [RESULTS.md:57 prints 0.2263]
co min                0.115383        [:57 0.1154; :60 0.11538]
co 1st percentile     0.147287        [:57 0.1473]
cross mean            1.26185e-05     [:58 1.26e-5]
cross 95th percentile 3.7685e-05     [:58 3.77e-5]
cross max             0.01958742      [:59 0.01959]
cross above 1e-3      14 of 31,680   [:58-59]
G_min pooled          0.770164        [§4 gives log10(5.9) ~ 0.771]
```

All eight published figures reproduce from the artifacts rather than from prose.

## 4. Arm A — step. Endpoint table (15 runs pooled, 88 adapters each)

| step | G_dec median | G_dec mean | G_min median | G_min min | Occ count / total | mean\|co\| | mean\|cross\| |
|---|---|---|---|---|---|---|---|
| 0 | — | — | — | — | 0 / 36,960 | 0 | 0 |
| 100 | 0.791901 | 0.87628 | 0.356632 | -0.124962 | 0 / 36,960 | 0.0065427 | 0.00119014 |
| 200 | 2.07758 | 2.11571 | 0.994152 | 0.169267 | 4,642 / 36,960 | 0.0235098 | 0.000375937 |
| 300 | 3.12441 | 2.83332 | 2.57304 | 0.403704 | 5,266 / 36,960 | 0.0444849 | 0.000156027 |
| 400 | 3.32633 | 3.13904 | 2.79288 | 0.52264 | 5,280 / 36,960 | 0.0633544 | 9.93668e-05 |
| 500 | 3.46417 | 3.32489 | 2.92354 | 0.567984 | 5,280 / 36,960 | 0.0806148 | 7.55221e-05 |
| 600 | 3.54851 | 3.41888 | 3.00519 | 0.590172 | 5,279 / 36,960 | 0.0964992 | 7.20959e-05 |
| 700 | 3.6189 | 3.51092 | 3.07934 | 0.641917 | 3,379 / 36,960 | 0.111296 | 6.49534e-05 |
| 800 | 3.66536 | 3.58581 | 3.10699 | 0.685494 | 660 / 36,960 | 0.12501 | 5.76998e-05 |
| 900 | 3.72743 | 3.64039 | 3.16092 | 0.736443 | 305 / 36,960 | 0.137615 | 5.41466e-05 |
| 1000 | 3.74871 | 3.67772 | 3.17967 | 0.749776 | 188 / 36,960 | 0.149116 | 5.17956e-05 |
| 1100 | 3.77015 | 3.70647 | 3.17355 | 0.767341 | 106 / 36,960 | 0.159518 | 5.0115e-05 |
| 1200 | 3.78341 | 3.72201 | 3.17005 | 0.755091 | 85 / 36,960 | 0.168843 | 5.09717e-05 |
| 1300 | 3.75851 | 3.73588 | 3.16131 | 0.766931 | 66 / 36,960 | 0.177139 | 5.02679e-05 |
| 1400 | 3.76353 | 3.75284 | 3.16124 | 0.779004 | 44 / 36,960 | 0.184453 | 5.06116e-05 |
| 1500 | 3.79829 | 3.79649 | 3.17276 | 0.790324 | 30 / 36,960 | 0.190643 | 4.77243e-05 |
| 1600 | 3.79297 | 3.79878 | 3.16153 | 0.797038 | 21 / 36,960 | 0.195837 | 4.98689e-05 |
| 1700 | 3.80279 | 3.83237 | 3.18208 | 0.807512 | 14 / 36,960 | 0.200447 | 4.75075e-05 |
| 1800 | 3.85655 | 3.89558 | 3.25921 | 0.820596 | 13 / 36,960 | 0.204559 | 4.13555e-05 |
| 1900 | 3.97715 | 4.00431 | 3.37726 | 0.82989 | 7 / 36,960 | 0.208184 | 3.2454e-05 |
| 2000 | 3.98565 | 4.04552 | 3.39104 | 0.838603 | 6 / 36,960 | 0.211328 | 3.06766e-05 |
| 2100 | 4.11178 | 4.13269 | 3.49524 | 0.842706 | 6 / 36,960 | 0.214 | 2.57323e-05 |
| 2200 | 4.20486 | 4.22587 | 3.60445 | 0.852635 | 6 / 36,960 | 0.216217 | 2.17037e-05 |
| 2300 | 4.2702 | 4.31786 | 3.66472 | 0.858408 | 6 / 36,960 | 0.218006 | 1.82928e-05 |
| 2400 | 4.44002 | 4.45035 | 3.84972 | 0.870439 | 3 / 36,960 | 0.219394 | 1.46691e-05 |
| 2500 | 4.4821 | 4.52416 | 3.90977 | 0.876862 | 2 / 36,960 | 0.220514 | 1.21909e-05 |
| 2600 | 4.43544 | 4.4994 | 3.87304 | 0.885507 | 2 / 36,960 | 0.221638 | 1.2914e-05 |
| 2700 | 4.4535 | 4.50644 | 3.87044 | 0.888707 | 2 / 36,960 | 0.222779 | 1.2664e-05 |
| 2800 | 4.49719 | 4.52195 | 3.88436 | 0.895357 | 2 / 36,960 | 0.223937 | 1.21014e-05 |
| 2900 | 4.44749 | 4.49878 | 3.87343 | 0.900797 | 2 / 36,960 | 0.225111 | 1.27132e-05 |
| 3000 | 4.47539 | 4.50247 | 3.87949 | 0.911191 | 0 / 36,960 | 0.226302 | 1.26185e-05 |

Final checkpoint, pooled: G_dec median 4.47539, G_min median 3.87949, Occ 0 / 36,960.

The step-0 row carries `—` for both endpoints because all 1,320 step-0 bridges are
the exact identity: every off-diagonal entry is 0.0, both modes vanish, and the
ratio is 0/0 — UNDEFINED, not the +inf of E-5's x/0 convention. See P2 below.

## 5. Arm B — drive

### 5.1 FI-004, the anneal (30 checkpoints, c_w recorded per checkpoint)

| step | c_w (recorded) | G_dec median | G_min median | Occ / total | Occ co | Occ cross | recorded co/cross |
|---|---|---|---|---|---|---|---|
| 100 | 0.096666667 | 2.13083 | 1.65231 | 0 / 1,320 | 0 | 0 | 40.124 |
| 200 | 0.093333333 | 1.62055 | 1.16983 | 6 / 1,320 | 0 | 6 | 12.2892 |
| 300 | 0.09 | 1.45095 | 0.977659 | 243 / 1,320 | 0 | 243 | 8.18601 |
| 400 | 0.086666667 | 1.40537 | 0.888981 | 321 / 1,320 | 0 | 321 | 7.31714 |
| 500 | 0.083333333 | 1.4179 | 0.842259 | 312 / 1,320 | 0 | 312 | 7.46596 |
| 600 | 0.08 | 1.47004 | 0.81941 | 276 / 1,320 | 0 | 276 | 8.25541 |
| 700 | 0.076666667 | 1.54359 | 0.816456 | 219 / 1,320 | 0 | 219 | 9.59813 |
| 800 | 0.073333333 | 1.62463 | 0.827773 | 171 / 1,320 | 0 | 171 | 11.6906 |
| 900 | 0.07 | 1.72233 | 0.851174 | 131 / 1,320 | 0 | 131 | 15.2209 |
| 1000 | 0.066666667 | 1.82114 | 0.873918 | 117 / 1,320 | 0 | 117 | 38.9189 |
| 1100 | 0.063333333 | 1.91217 | 0.913497 | 107 / 1,320 | 0 | 107 | 83.6227 |
| 1200 | 0.06 | 1.98939 | 0.959338 | 100 / 1,320 | 0 | 100 | 74.6965 |
| 1300 | 0.056666667 | 2.07011 | 1.00916 | 90 / 1,320 | 0 | 90 | 279.087 |
| 1400 | 0.053333333 | 2.14424 | 1.06774 | 81 / 1,320 | 0 | 81 | 228.39 |
| 1500 | 0.05 | 2.20335 | 1.13021 | 71 / 1,320 | 0 | 71 | 625.271 |
| 1600 | 0.046666667 | 2.27669 | 1.19834 | 65 / 1,320 | 0 | 65 | 684.52 |
| 1700 | 0.043333333 | 2.3547 | 1.26737 | 54 / 1,320 | 0 | 54 | 1399.62 |
| 1800 | 0.04 | 2.42689 | 1.34469 | 46 / 1,320 | 0 | 46 | 2325.62 |
| 1900 | 0.036666667 | 2.50564 | 1.42385 | 35 / 1,320 | 0 | 35 | 3768.74 |
| 2000 | 0.033333333 | 2.58203 | 1.49293 | 29 / 1,320 | 0 | 29 | 4610.98 |
| 2100 | 0.03 | 2.64854 | 1.55982 | 19 / 1,320 | 0 | 19 | 6307.03 |
| 2200 | 0.026666667 | 2.70855 | 1.63611 | 16 / 1,320 | 0 | 16 | 7379.77 |
| 2300 | 0.023333333 | 2.75763 | 1.69292 | 14 / 1,320 | 0 | 14 | 10824.6 |
| 2400 | 0.02 | 2.79602 | 1.7339 | 14 / 1,320 | 0 | 14 | 16768 |
| 2500 | 0.016666667 | 2.8263 | 1.76615 | 13 / 1,320 | 0 | 13 | 18670.5 |
| 2600 | 0.013333333 | 2.86908 | 1.79684 | 11 / 1,320 | 0 | 11 | 17059.5 |
| 2700 | 0.01 | 2.90671 | 1.82113 | 10 / 1,320 | 0 | 10 | 15737.3 |
| 2800 | 0.0066666667 | 2.93347 | 1.84246 | 10 / 1,320 | 0 | 10 | 14740.8 |
| 2900 | 0.0033333333 | 2.94437 | 1.85712 | 9 / 1,320 | 0 | 9 | 12466.9 |
| 3000 | 0 | 2.94212 | 1.86712 | 9 / 1,320 | 0 | 9 | 2941.67 |

Across all 30 anneal checkpoints the band is occupied by cross entries only: the
`Occ co` column sums to 0 over the whole table. The band fills from
below — suppressed couplings climbing into it — not from above.

### 5.2 FI-003, zero drive (12 checkpoints, c_w = 0 throughout)

| step | G_dec median | G_min median | Occ count / total | recorded co/cross |
|---|---|---|---|---|
| 100 | 2.82203 | 2.45995 | 0 / 1,320 | 178.66 |
| 200 | 2.35786 | 1.94129 | 0 / 1,320 | 59.972 |
| 300 | 2.05989 | 1.69011 | 0 / 1,320 | 29.9365 |
| 400 | 1.91018 | 1.53753 | 4 / 1,320 | 21.8389 |
| 500 | 1.8363 | 1.4331 | 6 / 1,320 | 18.0733 |
| 600 | 1.7877 | 1.3748 | 10 / 1,320 | 15.5884 |
| 700 | 1.72069 | 1.34125 | 14 / 1,320 | 13.732 |
| 800 | 1.69433 | 1.2848 | 18 / 1,320 | 12.4864 |
| 900 | 1.63179 | 1.24221 | 27 / 1,320 | 11.5249 |
| 1000 | 1.62824 | 1.23323 | 43 / 1,320 | 10.8484 |
| 1100 | 1.57805 | 1.21428 | 47 / 1,320 | 10.2183 |
| 1200 | 1.55792 | 1.17141 | 50 / 1,320 | 9.79651 |

Occ on FI-003 and FI-004 is **descriptive**: `OCC_BAND` is an absolute magnitude
band pinned from E-5 (TinyLlama-1.1B, n = 8, detector v2, pooled magnitude
0.0307-0.0310), while the FI series is n = 6, detector v1, pooled magnitude
0.0277-0.0280 — the transfer is named here in the same sentence as the number.

## 6. Arms C and D — DESCRIPTIVE ONLY, confounds attached to every number

No p-value, permutation test or inference of any kind is computed on Arm C or
Arm D (§7; §14, Director 2026-09-06: *"Keep it that way regardless of how clean
the trend looks"*). §14 bars any later promotion to confirmatory.

| Directory | model | n | adapters | checkpoints | G_dec @ first post-init | G_dec last | G_min last | Occ last |
|---|---|---|---|---|---|---|---|---|
| `exp3_tinyllama` | TinyLlama-1.1B | 6 | 88 | 101 (0..10000) | 2.00168 | 4.67337 | 4.16985 | 0/1,320 |
| `fc-001` | Qwen2.5-1.5B | 6 | 112 | 19 (0..1800) | 0.212239 | 3.85243 | 3.29457 | 9/1,680 |
| `fc-001-fresh` | Qwen2.5-1.5B | 6 | 112 | 25 (0..2400) | 0.210521 | 3.87582 | 3.37793 | 1/1,680 |
| `exp3` | Qwen2.5-7B | 6 | 112 | 130 (0..12900) | 1.42316 | 4.31245 | 3.81695 | 0/1,680 |
| `cw-001` | TinyLlama-1.1B | 6 | 88 | 101 (0..10000) | 0.489037 | 3.78144 | 3.253 | 0/1,320 |

The step-0 layer of each of these directories is reported separately rather than in
the column above, because at step 0 the bridge is the identity in every arm that
has a step-0 snapshot: all off-diagonal entries are exactly 0, both modes vanish,
and G_dec and G_min are 0/0 and UNDEFINED — not infinite. Step-0 rows appear in the
JSON with `zero_offdiag = 1` and null endpoints.

**The confounds, in the same sentence as the trend.** These five directories differ
in `c_w` regime (`exp3_tinyllama` / `fc-001` / `fc-001-fresh` / `exp3` all start at
`initial_contrastive` 0.1, `cw-001` at 0.02), in step count (1,800 to 12,900), in
adapter count (88 vs 112), and in model family (TinyLlama-1.1B, Qwen2.5-1.5B,
Qwen2.5-7B) — so any scale-shaped trend across them is a trend across four
unmatched variables, not across scale, and every figure above is read that way.
All five are n = 6, detector v1, against E-5's n = 8, detector v2; the Occ column
carries the same absolute-band transfer flag as §5.

## 7. The confirmatory set — exactly three (§7)

### C1 — Arm A: sign and monotonicity of the G_dec-vs-log(step) slope

```
TEST               = C1 (confirmatory)
ARM                = A (E-5, 15 runs x 88 adapters)
REGRESSOR          = log10(step), steps 100..3000 (30 points)
STEP 0             = EXCLUDED (log10(0) undefined; step 0 is P2's control)
PAIRING            = (run, adapter) — 15 x 88 = 1320
STATISTIC          = mean OLS slope of G_dec on log10(step) over all (run, adapter) pairs
SLOPE mean         = 2.01545
SLOPE median       = 1.9403
SLOPE sd           = 0.446585
FRAC slopes > 0    = 1.0000
KENDALL tau mean   = 0.8054   (median 0.8207)
FRAC tau > 0       = 1.0000
PERMUTATION        = run-level: one independent permutation of the 30 step labels per run per draw, applied to all 88 adapters of that run
PERM unit / draws  = run / 10000
PERM seed          = 20260914
NULL mean / sd     = 0.000448 / 0.101
p TWO-SIDED        = 9.999e-05
p ONE-SIDED (+)    = 9.999e-05
```

### C2 — Arm B: sign and monotonicity of the G_dec-vs-c_w slope

```
TEST               = C2 (confirmatory) — REPORTED PENDING THE DIRECTOR'S RULING
                     on item 1 (Arm B's corpus-init descent), which is UNRULED.
ARM                = B (FI-004 anneal, 30 checkpoints x 88 adapters)
REGRESSOR          = recorded contrastive_weight from fi-004/results.json checkpoints (30 points, 0.09666666666666668 -> 0.0)
PAIRING            = adapter — 88
SLOPE mean         = -24.9654
SLOPE median       = -17.0979
FRAC slopes < 0    = 1.0000
KENDALL tau mean   = -0.8446   (median -0.8851)
PERMUTATION        = run-level: one permutation of the 30 c_w labels per draw, applied to all 88 adapters (Arm B has one annealing run)
PERM unit / draws  = run / 10000
PERM seed          = 20260914
p TWO-SIDED        = 9.999e-05
p ONE-SIDED (-)    = 9.999e-05
c_w vs step r      = -1.000000
FI-003 c_w=0 mean G_dec, step 100  = 2.82663
FI-003 c_w=0 mean G_dec, step 1200 = 1.54204
FI-004 c_w=0 row (step 3000) mean G_dec = 3.26432
```

**The collinearity, stated rather than buried.** Within FI-004, `c_w(s) = 0.1 x (1 - s/3000)`
exactly, so the drive axis and the training-step axis are the same axis up to an
affine map (Pearson r = -1.0000). C2 measures a slope along
that single axis and **cannot separate less drive from more training**. FI-003 is the
only off-axis observation: a separate run at c_w = 0 throughout, n = 6, detector v1,
12 step indices — and it is a different run, not a 31st row of the same trajectory.

### C3 — N2: observed vs two-fit predicted band occupancy, E-5 final, pooled

```
TEST               = C3 (confirmatory)
NULL               = N2 (forbidden-zone null)
FITS               = 2,640 (1,320 adapters x 2 modes),
                     log-normal per mode per adapter, fitted INDEPENDENTLY
OBSERVED in band   = 0 of 36,960
OBSERVED occupancy = 0
PREDICTED count    = 2.4141
PREDICTED occupancy= 6.53165e-05
PREDICTED +-2sd    = [-0.648667, 5.47686]   (Poisson-binomial)
PREDICTED boot CI95= [1.35404, 3.6419]   (2000 adapter resamples, seed 20260914)
DECISION STATISTIC P(observe 0 | N2) = 0.0863316   (log10 -1.064)
                     tested against the REALIZATION distribution, not against
                     an interval on the predicted mean (see P6)
mean P_co / P_cross= 0.000405743 / 8.57873e-06
sigma_co median    = 0.0194195  (5th 0.003536, 95th 0.1278)
sigma_cross median = 1.15803
SENSITIVITY ddof=1 predicted count = 4.38633   P(observe 0) = 0.0112921
```

**The thin fit, flagged rather than silently changed.** A 2-parameter log-normal
fitted to 4 co entries has 2 residual degrees of freedom. The card specifies
per-adapter fits and they are kept; the sigma_co quantiles above let a reader see
how thin the co fit is, and the ddof = 1 line is a sensitivity, not a substitution.

**What C3 is and is not testing.** Observed Occ = 0 at E-5's final checkpoint is
not itself a finding — the band's endpoints are that data's own extrema, so it is
empty by construction. Only the prediction side is unknown.

## 8. Predictions P1-P7, each with its OUTCOME under the card's both-outcomes wording

### P1 — Arm A: G_dec rises with log(step) and saturates before step 3000

**OUTCOME: RISE. G_dec rises with log(step); the card's first branch holds and 'widening with training' is licensed. The second clause does NOT hold as written: the late-window slope is 0.86x the early-window slope, so the rise has not flattened by step 3000 and 'saturates before step 3000' is not supported here.**

```
G_dec median at step 100   = 0.791901
G_dec median at step 3000  = 4.47539
rise, step 100 -> 3000     = 3.68349 decades
fraction of final gap already open at step 100 = 0.176946
mean slope vs log10(step)  = 2.01545
permutation p (two-sided)  = 9.999e-05
early-window slope (steps 100-1000)  = 2.69703
late-window slope  (steps 2100-3000) = 2.32686
late / early               = 0.862749
frac pairs late < early    = 0.6636
```

*Descriptive companion to C1, not a fourth confirmatory test: the confirmatory claim is the sign and monotonicity of the whole-grid slope, and these two window slopes only say whether the rise flattens, which is P1's second clause.*

### P2 — Arm A control: at bridge_step0 (initialization) there is no gap

**OUTCOME: NO GAP AT INITIALIZATION, in the strongest available form. Every one of the 1,320 step-0 bridges is the exact identity matrix: all off-diagonal entries are 0.0, so the co mode and the cross mode both vanish, the ratio is 0/0, and G_dec and G_min are UNDEFINED rather than infinite. The endpoint cannot be measuring the initializer, because the initializer writes nothing off-diagonal for it to measure. ARM A EXISTS. Occ at step 0 is 0 of 36,960 because every entry sits below the band, not because a gap is holding it open — a distinction the band cannot make by itself and which is therefore stated here.**

```
CONTROL            = bridge_step0 (initialization), 15 runs x 88 adapters = 1,320 adapters
step 0 adapters with ALL off-diagonal entries exactly 0 = 1,320 of 1,320
step 0    G_dec median = —   G_min median = —   G_min<=0 0/1320
step 100  G_dec median = 0.791901   G_min median = 0.356632   G_min<=0 8/1320
step 3000 G_dec median = 4.47539   G_min median = 3.87949   G_min<=0 0/1320
step 0    Occ = 0 / 36,960   (fraction 0)
G_dec(step0)/G_dec(step3000) medians = —
GAP PRESENT AT INIT? = False
ARM A VOID?        = False
```

*IMPLEMENTER OPERATIONALISATION, dated 2026-09-14, flagged as such and not a registered criterion: a gap is called PRESENT at initialization iff the step-0 layer has off-diagonal structure at all AND the median G_min over the 1,320 step-0 adapters exceeds 0 (a strict forbidden zone exists at init). Where the step-0 bridge carries NO off-diagonal structure, both modes vanish, the ratio is 0/0 and both endpoints are UNDEFINED — which is reported as undefined, never as the +inf of E-5's x/0 convention, because conflating them would report a gap where there is no structure.*

### P3 — Arm B: G_dec tracks the run-level co/cross ratio monotonically

**OUTCOME: TRACKING. G_dec tracks the run-level co/cross ratio monotonically across the anneal — gap and ratio read as one object seen at two levels.**

```
Kendall tau, median G_dec vs recorded co/cross ratio (30 checkpoints) = 0.8667
Spearman rho, same pair                                              = 0.9546
recorded co/cross at step 100 / 1500 / 3000 = 40.124 / 625.271 / 2941.67
```

### P4 — Arm B, zero drive: under FI-003 the band fills

**OUTCOME: THE BAND FILLS, AND THE FILL IS SLOW ENOUGH TO BE RESOLVED ON THIS GRID. The first three checkpoints are empty; occupancy opens between step 300 and step 400, then rises monotonically at every subsequent checkpoint to 50 of 1,320 entries (3.788%) at step 1200 — still far from full when the run early-stops. The timing branch remains UNDERPOWERED as pinned, and no half-life is claimed for the fill. What the grid does support is the one-sided statement it can carry: the fill had not begun by step 300, so it is not a process that was over before the first snapshot [ANALYTICAL CONTRIBUTION: the reading; the observation is the Occ column below]. That is a bound, not a measurement, and it is not offered as a null against the record's fast component.**

```
POWER VERDICT (timing branch) = UNDERPOWERED — the 12-checkpoint grid cannot separate a 15-step from a 230-step half-life (card §12 Item 3). The timing branch is reported as UNDERPOWERED, NEVER as a null.
FILL vs NO FILL              = FULLY POWERED at all 12 points — reported as a result either way
OCCUPANCY ONSET              = between step 300 and 400
GAP half-life, endpoint fit  = 261.9 steps
GAP half-life, OLS fit       = 333.3 steps   (R2 0.7578)
Occ and gap endpoints across the 12 FI-003 checkpoints:
  step   100  Occ   0 / 1,320 (co 0, cross 0)   G_dec med 2.82203   G_min med 2.45995   co/cross 178.66
  step   200  Occ   0 / 1,320 (co 0, cross 0)   G_dec med 2.35786   G_min med 1.94129   co/cross 59.972
  step   300  Occ   0 / 1,320 (co 0, cross 0)   G_dec med 2.05989   G_min med 1.69011   co/cross 29.9365
  step   400  Occ   4 / 1,320 (co 0, cross 4)   G_dec med 1.91018   G_min med 1.53753   co/cross 21.8389
  step   500  Occ   6 / 1,320 (co 0, cross 6)   G_dec med 1.8363   G_min med 1.4331   co/cross 18.0733
  step   600  Occ  10 / 1,320 (co 0, cross 10)   G_dec med 1.7877   G_min med 1.3748   co/cross 15.5884
  step   700  Occ  14 / 1,320 (co 1, cross 13)   G_dec med 1.72069   G_min med 1.34125   co/cross 13.732
  step   800  Occ  18 / 1,320 (co 1, cross 17)   G_dec med 1.69433   G_min med 1.2848   co/cross 12.4864
  step   900  Occ  27 / 1,320 (co 1, cross 26)   G_dec med 1.63179   G_min med 1.24221   co/cross 11.5249
  step  1000  Occ  43 / 1,320 (co 1, cross 42)   G_dec med 1.62824   G_min med 1.23323   co/cross 10.8484
  step  1100  Occ  47 / 1,320 (co 1, cross 46)   G_dec med 1.57805   G_min med 1.21428   co/cross 10.2183
  step  1200  Occ  50 / 1,320 (co 2, cross 48)   G_dec med 1.55792   G_min med 1.17141   co/cross 9.79651
half-lives elapsed at s=100:  tau=15 -> 6.667   tau=230 -> 0.435
surviving fraction at s=100:  tau=15 -> 0.0098431   tau=230 -> 0.73981
smallest tau constrainable, TWO points at >=10% retention = 60.21 steps
smallest tau constrainable, ONE point                     = 30.10 steps
```

**Which side of the band fills.** The band fills FROM BELOW: the occupied entries are cross couplings rising into it, not co couplings falling into it. The co mode stays above the band at every FI-003 checkpoint but two. Removing the drive does not collapse the trained couplings into the gap; it lets the suppressed ones climb. At the terminal checkpoint the occupied entries split 2 co / 48 cross. The FI-004 table in §5.1 carries the same split per checkpoint and sums it.

**The measured timescale, and what it is allowed to be compared with.** Half-life of the GAP (median G_dec, i.e. log10 of the co/cross mean ratio) across the 12 FI-003 checkpoints. The endpoint figure is assumption-light (first and last points); the OLS figure assumes G_dec falls linearly in step and its R2 shows how well that holds. Both are far above the 60.21-step two-point power floor, so they lie inside what this grid can constrain; 15 steps lies below even the one-point floor and is not measurable here. The record's slow component is 230 steps; both figures above are of the same order, and this document does not convert that into an identity claim — a two-point and a 12-point fit of one quantity on one run cannot confirm a four-parameter fit of a different quantity.

**Transfer flag, in the same sentence as the number.** OCC_BAND is an ABSOLUTE magnitude band pinned from E-5 (TinyLlama-1.1B, n=8, detector v2, pooled magnitude 0.0307-0.0310); FI-003 is n=6, detector v1, pooled magnitude 0.0277-0.0280 — so Occ here is DESCRIPTIVE and the transfer is named in the same sentence.

**How the 15-step figure is cited.** The record's double-exponential fit (R2 = 0.9999998, fast half-life 15 steps, slow 230) has as its only available trajectory a step-0 row plus these same 12 checkpoints; a double exponential carries four free parameters and there is no observation anywhere between step 0 and step 100, so the whole factor-70 drop from step 0 to step 100 is the only evidence the fast timescale has. The 15-step figure is an extrapolation across one interval, not a resolved decay [ANALYTICAL CONTRIBUTION: the reading of the fit's support; the fit and its R2 are the record's].

### P5 — Arm C

**OUTCOME: NO PREDICTION WAS REGISTERED, AND NONE IS CLAIMED.** The confounds
forbid one (§6 item 5). Arm C's figures are in §6 above, each with its confounds
attached; no inference is drawn from them.

### P6 — N2: observed band occupancy is below the two-fit prediction

**OUTCOME: THE PREDICTED COUNT IS ABOVE THE OBSERVED ZERO, BUT NOT BY ENOUGH TO REFUTE N2: P(observe 0 | N2) = 0.08633, so a band that is empty by chance under two independently fitted log-normals is an ordinary outcome, not a rare one. Under the card's second branch the 'forbidden zone' reading of Station 8 is, on this evidence, a description of two well-separated humps, and should be written as such.**

```
observed  0 / 36,960   occupancy 0
predicted 2.4141 / 36,960   occupancy 6.53165e-05
predicted bootstrap CI95 (on the MEAN) = [1.35404, 3.6419]
DECISION STATISTIC  P(observe 0 | N2) = 0.08633   (log10 -1.064)
same, ddof=1 sensitivity              = 0.01129
```

**Why the decision rests on P(observe 0), not on the interval.** The observed 0 is a single REALIZATION; the bootstrap interval above is an interval on the predicted MEAN. Comparing the one to the other is a category error that would manufacture an exclusion out of a mean of roughly two expected entries, so the test is run against the realization distribution instead: under the fitted two-mode model the entries are independent, P(none lands in the band) is the product of their misses, and that is the number above.

**The comparison is reported, not a bare ratio**, and the vacuity caveat binds: the band's endpoints are this data's own extrema, so the observed 0 is empty by construction and only the prediction side is unknown.

**Which branch of §6 item 6 this is — IMPLEMENTER READING, dated 2026-09-14, flagged as such and not a registered criterion.** §6 item 6 writes the branches as *"below -> a genuine exclusion; at or above -> the 'forbidden zone' reading of Station 8 is a description of two separated humps"*, and the observed 0 IS literally below the predicted 2.414. The second branch is selected above anyway, on §5's criterion rather than on the bare inequality: §5 defines N2 as *"fit each mode independently ... predict band occupancy, compare to observed. Under N2 they agree"*, and whether a single realization agrees with a predicted mean is a statistical question, so item 6's *below* is read here as *significantly below*. Both halves are reported so a reader may apply either: the literal below-condition is SATISFIED (0 < 2.414); the statistical criterion is NOT met (P(observe 0 | N2) = 0.08633, not significant at any conventional level), and N2 is therefore NOT REFUTED. Reported under L-006 as a dated reading of the registered wording, not a revision of it.

### P7 — G_min moves with G_dec (bar: Kendall tau >= 0.6)

**OUTCOME: CO-MOVEMENT on both arms — the registered bar is met for 100.0% of Arm A pairs and 100.0% of Arm B adapters, so the margin is one quantity seen through two statistics. The card's second branch does NOT hold at scale, but it is not empty either: G_min falls to <= 0 in 8 of 39,600 Arm A post-init adapter-checkpoints (0.0202%), confined to these Arm A steps: {'100': 8}, and in 0 of 2,640 on Arm B (0.0000%). The silicon engineer's failure case — means a decade apart while the tails touch, a margin with no guaranteed zone — is therefore real here but transient: it belongs to the first hundred steps of training and is closed thereafter.**

```
Arm A  tau mean 0.8589  median 0.8575  frac tau>=0.6 1.0000  (n=1,320 run-adapter pairs)
Arm A  G_min <= 0 at 8 of 39,600 adapter-checkpoints (0.00020202)
Arm A  G_min <= 0 by step: {'100': 8}
Arm B  tau mean 0.8406  median 0.8483  frac tau>=0.6 1.0000  (n=88 adapters)
Arm B  G_min <= 0 at 0 of 2,640 adapter-checkpoints (0)
```

G_min at step 0 is subject to the same control as P2; see the P2 block.

## 9. The two nulls, with verdicts

```
N1 (lawfulness null) = G_dec, G_min and Occ are CONSTANT across step, drive and
                       scale; the trainer has one gap and no response surface.
N1 VERDICT           = REFUTED on Arm A (C1) and on Arm B (C2). The gap is not constant: it has a response surface, within the scope limit. Two qualifications travel with this verdict and are not separable from it. (i) Arm B's regressor is collinear with training step by construction (r = -1.000), so C2 refutes constancy along a combined drive-and-duration axis and does not by itself establish a DRIVE response; the register's null named three dials and this run refutes it on one clean dial (step) and one compound dial. (ii) The scale dial is untested — Arm C is descriptive and contributes nothing to this verdict, by §14.
N2 (forbidden-zone)  = the empty band is only the tails of two well-separated
                       humps; observed and predicted occupancy agree.
N2 VERDICT           = NOT REFUTED (P(observe 0 | N2) = 0.08633; predicted mean 2.414 of 36,960, observed 0). The two-fit model expects only a couple of entries in the band to begin with, and seeing none is an ordinary outcome under it. On this evidence the empty band is the tails of two well-separated humps, and the 'forbidden zone' reading of Station 8 is a description of two separated humps and should be written as such. Note what this does NOT say: it does not show the band is occupiable, only that E-5's single final checkpoint has too little occupancy mass at stake to tell the two readings apart.
```

Both verdicts inherit the scope limit verbatim: TinyLlama-1.1B (descriptively Qwen2.5-1.5B / Qwen2.5-7B), TeLoRA bridge couplings under the Steersman contrastive objective, at the configurations already written to disk; detector v2 for E-5, v1 for the FI series — no cross-detector depth comparison. No claim about learned systems in general and none about silicon (§8).

## 10. Standing constraints, restated at the point of filing

- **Item 1 UNRULED.** The Director has NOT ruled on whether Arm B's descent from a corpus-coupled initialization may stand inside a confirmatory arm. Arm B stays CONFIRMATORY as drafted and C2 is reported PENDING THAT RULING.
- **Item 4.** DESCRIPTIVE ONLY. No p-value, permutation test or inference of any kind is computed on Arm C or Arm D (§7, §14). Every figure carries its confounds in the same sentence.
- **Item 9 OPEN.** The channel count is not constant across arms: Arm A is n=8 (28 pairs = 4 co + 24 cross), Arms B/C/D are n=6 (15 pairs = 3 co + 12 cross). Pooled entry counts therefore differ by arm and no cross-arm pooled count may be quoted as if it were E-5's 36,960.
- **§10.** ZERO GPU-seconds; no gpu_guard claim taken; CPU only (card §10). The card queues behind L1 and takes no place in
  the GPU queue.
- **§11.** `C:/falco/tools/corpus_guard.py` runs over this document and every
  output table before filing; nothing is filed on a guard run reporting no
  protected set loaded.
- **§9 maker-grader.** A fresh-context verifier re-derives C1-C3 from the `.npy`
  files independently of this script.

---

*Filed 2026-09-14. Zero GPU-seconds. No artifact read by this analysis was
modified. Every number above was computed at run time from the `.npy` artifacts
named in the census, under the pinned `OCC_BAND` and the pinned trained pair
specs; none was copied from prose.*
