# E-5 — Coherence-Interpolation Bifurcation Sweep: Results

> **Run:** 2026-07-10T19:12Z → 2026-07-13T11:27Z, Hermes RTX 4090 Laptop,
> 15/15 runs COMPLETE, zero endpoint errors (after the dated 2026-07-13
> rtol edit — a float-summation-order artifact at the 7th significant digit,
> not a data issue). chain.log's leading all-15-FAILED block is the aborted
> first launch (stale editable-install bug, fixed at commit b229237 before
> any training); the manifest and all endpoints derive from the clean second
> pass. Run 1 took 8h21m vs 3.5–4.7h for the rest — an AC-power loss was
> diagnosed live on Jul 11 (battery-capped GPU clocks; integrity
> unaffected), on top of first-run model-load overhead.
> Pre-registration: `docs/E5_BIFURCATION_PREREG_2026-07-10.md` (frozen
> before launch, incl. amendments). All numbers below re-computed from the
> fetched artifacts in this directory (`endpoints.json`, per-run
> `config.json` + `bridge_final_*.npy`).

## Headline

**There is no bifurcation on the f-axis — because there is no second basin.
All 15 runs, at every pair-correctness f from 1.0 down to 0.0, crystallized
full block-diagonal-class coherence on whatever spec they were given**
(co/cross_trained 3.66×10⁴ – 2.08×10⁵; every run Structured; the
pre-registered Intermediate band [10¹, 10³) is empty, 0/15). The
all-or-none coherence claim survives — but it lives at the **pair level**,
with exact combinatorial structure, not at the run level.

## The 15-run table

| run | nominal f | realized agreement | overlap k | co/cross_trained | co/cross_true | ladder 6k/(4−k) |
|---|---|---|---|---|---|---|
| f0.00_s42 | 0.00 | 0.786 | 1 | 3.67e4 | 2.024 | 2.0 |
| f0.00_s43 | 0.00 | 0.714 | 0 | 2.08e5 | 3.11e-5 | ~0 |
| f0.00_s44 | 0.00 | 0.786 | 1 | 4.31e4 | 1.974 | 2.0 |
| f0.25_s42 | 0.25 | 0.786 | 1 | 3.72e4 | 1.952 | 2.0 |
| f0.25_s43 | 0.25 | 0.929 | 3 | 4.19e4 | 18.05 | 18.0 |
| f0.25_s44 | 0.25 | 0.786 | 1 | 3.96e4 | 1.967 | 2.0 |
| f0.50_s42 | 0.50 | 0.857 | 2 | 3.88e4 | 5.895 | 6.0 |
| f0.50_s43 | 0.50 | **1.000** | 4 | 4.30e4 | 4.30e4 | = trained |
| f0.50_s44 | 0.50 | 0.857 | 2 | 4.23e4 | 5.883 | 6.0 |
| f0.75_s42 | 0.75 | 0.929 | 3 | 3.77e4 | 18.02 | 18.0 |
| f0.75_s43 | 0.75 | **1.000** | 4 | 4.20e4 | 4.20e4 | = trained |
| f0.75_s44 | 0.75 | 0.929 | 3 | 4.24e4 | 17.78 | 18.0 |
| f1.00_s42 | 1.00 | 1.000 | 4 | 3.66e4 | 3.66e4 | = trained |
| f1.00_s43 | 1.00 | 1.000 | 4 | 4.17e4 | 4.17e4 | = trained |
| f1.00_s44 | 1.00 | 1.000 | 4 | 4.28e4 | 4.28e4 | = trained |

k = |trained co-set ∩ true co-set| (from each run's recorded
`co_pairs_trained`). Bands: all 15 **Structured**; Intermediate 0;
Unstructured 0; NonFinite 0. No magnitude flags (pooled coupling magnitude
0.0307–0.0310 in 14/15 runs; the k=0 frustrated run sits at 0.0534 — all
far above the amendment-1 collapse floor).

## The two confirmed structures

**1. Per-pair all-or-none crystallization.** Pooled over all 15 runs × 88
adapters (36,960 off-diagonal pair entries): trained-co entries |B[i,j]|
have mean 0.2263, min 0.1154 (1st percentile 0.1473); trained-cross entries
have mean 1.26e-5, 95th percentile 3.77e-5, with a thin tail — 14 of 31,680
above 1e-3, the largest 0.01959. **Between the largest cross entry (0.01959)
and the smallest co entry (0.11538) lies a completely empty factor-5.9
gap**; the modes sit ~4.2 decades apart mean-to-mean (0.2263 vs 1.26e-5).
Every pair either crystallizes (~0.23) or vanishes (~1e-5, tail ≤ 0.02);
nothing lands between. This is the bimodality P1 was hunting — one level
down from where it was pre-registered.

**2. The coherence ladder is exact.** If every trained-co pair carries
magnitude B and everything else ~0, then co/cross_true must equal
(k/4)/((4−k)/24) = **6k/(4−k)**. Observed vs predicted: k=1 → 1.95–2.02
(pred 2.0); k=2 → 5.88–5.90 (pred 6.0); k=3 → 17.78–18.05 (pred 18.0);
k=0 → 3e-5; k=4 → equals trained. Every run within ~2.5% (worst −2.42%,
f0.25_s42). Alignment with the
true geometry is pure spec combinatorics — the optimizer contributes no
partial credit.

## Verdicts on the pre-registered predictions

- **P1 (bimodality):** the no-intermediates core is CONFIRMED (0/15 in
  [10¹,10³), vs ≤2 predicted) — but the "both basins occupied" clause
  FAILS: the Unstructured basin is never visited. P1's two-basin premise
  was wrong; its empty-middle prediction was right, and holds more strongly
  at pair level (0 of 36,960 entries in the inter-mode decade).
- **P2 positive control:** CONFIRMED — all three f=1.0 runs Structured
  (3.66–4.28e4), plus two bonus fixed-point draws (f0.50_s43, f0.75_s43,
  realized agreement 1.000) landing identically.
- **P2 negative control:** REFUTED, in exactly the pre-flagged
  "striking non-predicted outcome" form — all three f=0 runs crystallized
  on their corrupted specs, including f0.00_s43, whose spec attracts
  channel 0 to three mutually-repelled partners (maximally frustrated) and
  which produced the HIGHEST trained-spec ratio of the sweep (2.08e5).
  Frustration did not weaken crystallization.
- **Graded falsifier:** NOT triggered — nothing intermediate, no monotone
  grading. The transition is not graded; on the trained metric there is no
  transition at all.
- **P3 (critical f, seed splitting):** moot — no second basin to split
  into.

## Reading, and what it changes

At this depth and config (TinyLlama, n=8, Steersman default, 3000 steps,
detector v2), **the Steersman + contrastive machinery crystallizes ANY
count-preserving pair spec — coherent, incoherent, or frustrated — with
equal totality.** "Coherence of the spec" controls only *which* pairs
crystallize, never *whether*. Two consequences:

1. **Paper 4's regime taxonomy needs a reframe for the wrong-label class:**
   WL-001/R-001's "collapse" endpoints (co/cross_true ≈ 8.7e-6) are
   re-read as crystallization-on-the-wrong-spec — their 3+3 eigensplits
   said so, and E-5's k=0 run reproduces the reading (true 3.1e-5, trained
   2.1e5). The all-or-none coherence claim survives and sharpens: basins
   live in pair-magnitude space, not run-ratio space. The Regime-4 "collapse"
   label conflates two distinct outcomes (wrong-spec crystallization vs
   genuine magnitude collapse) that only the trained-spec metric separates.
2. **The bifurcation figure is still a figure no PEFT paper has** — it is
   just a different figure than pre-registered: trained-spec coherence flat
   at ~4×10⁴ across the whole f-axis, true-spec alignment quantized on the
   exact 6k/(4−k) ladder, and a per-pair magnitude histogram with an empty
   middle. Bimodal, combinatorial, and fully predicted by a two-parameter
   story (B, ~0).

**Realized-agreement caveat (per prereg amendment 2):** under the
count-preserving shuffle at n=8's 4/24 label imbalance, nominal f=0 still
leaves ~75% of labels fixed (observed 0.714–0.786), and two mid-f draws
were full fixed points (agreement 1.000). The f-axis as operationalized
spans realized coherence ≈0.71–1.0 — a narrower range than the nominal
labels suggest. A future sweep wanting genuinely lower coherence needs a
different corruption operator (e.g., partition-count-breaking relabels).
This does not weaken the headline: even the most incoherent achievable
spec (k=0, frustrated triangles) crystallized totally.

**Detector-version note (per prereg amendment 3):** these f=1 endpoints
(~4×10⁴ at 3000 steps) are much deeper than r3's detector-v1 trajectory at
the same step (5.14×10³) — the adaptive controller differs across detector
versions, as the prereg cautioned. E-5's internal controls anchor all
comparisons; no cross-version depth claims are made.

## Artifacts

`endpoints.json` (15 runs, 0 errors) · `summary.md` (tally) ·
`sweep_manifest.json` · per-run `config.json` (corruption records) +
`bridge_final_*.npy` (88 × 15, the pair-level evidence, force-added past
the `*.npy` ignore) + `results.json` (checkpoint trajectories). Per-step
bridge snapshots (`bridge_step{100..3000}_*.npy`, 30 × 88 per run) exist
on local disk and Hermes but are untracked — raw material for a future
crystallization-onset timing analysis. Logs untracked (`*.log`). Provenance: rhombic
`bae4079` shipped as `~/rhombic-e5` on Hermes; runner commits through
`b229237`; endpoint rtol edit dated 2026-07-13.
