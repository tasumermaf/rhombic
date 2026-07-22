# Label-Granularity Bridge Study — Experiment Design

**STATUS: DESIGN DRAFT — for Director review; not registered; no data collected.**

**Date:** 2026-07-21 · **Author:** Meridian (hub), for Timothy Paul Bielec — TASUMER MAF
**Grounding:** `C:\falco\docs\ASSET1_BANK_DELIVERY_2026-07-20.md` · `paper/ASSET1_PAPER_DRAFT_v0.md` §2/§4.1/§6/§6.1 · `docs/DIRECTOR_DECISIONS_2026-07-06.md` (A2)

---

## 0. Question and regime axis

Asset-1's H1 sits at one end of a regime axis the paper deliberately does not cross. Stated correctly —
the A2 ruling's "10k+ fine-grained attribute classes" is the known erratum; **10k+ counts adapters, not
classes** — W2T (arXiv:2603.15990) classifies **fine-grained attribute labels, up to 312 classes, over
10k+-adapter collections** (Stable Diffusion vision; same-base, same-rank families, their Table 6), with
headroom even for its best models. Asset-1 classifies **6 coarse language tasks over 240
adapters/family**, where a linear probe on GL(r)-canonical features hits 1.0000. The paper claims
nothing about the middle, in either direction (§6.1). This study walks the axis *within the Asset-1
methodology* — same recipe, same H1 machinery — to find where the linear-probe ceiling breaks, and
whether the raw-vs-canonical gap grows or shrinks as labels sharpen.

## 1. Anchor (already measured — reused, not re-run)

```
L0.classes        = 6            [alpaca code math xsum squad agnews]
L0.acc.raw        = 0.0792 qwen / 0.1375 llama   [FAIL 1.5x lock; chance 0.1667]
L0.acc.canonical  = 1.0000 both  [PASS; perm p 0.000999]; vocab_sig 1.0000 both kv modes
L0.source         = ASSET1_BANK_DELIVERY_2026-07-20.md §2; bank 480/480
```

## 2. Design overview

**One family** (cost): **llama3.2-1b** — the cheaper cohort (1B, 64 modules), higher raw baseline
(0.1375); qwen replication at one level is optional (sign-off 9). Adapters are trained per *subtask*
class — existing L0 adapters cannot be relabeled; a finer label requires a finer training distribution.

| level | classes (target) | seeds/class | new runs | chance | 1.5x lock bar |
|---|---|---|---|---|---|
| L0 | 6 | 40 (existing) | 0 | 0.1667 | 0.2500 |
| L1 | 12 | 20 | 240 | 0.0833 | 0.1250 |
| L2 | 24 | 10 | 240 | 0.0417 | 0.0625 |
| L3 | 40–48 (ragged; frozen at registration) | 5 | 200–240 | 1/K | 1.5/K |

Constant ~240 adapters/level: equal GPU cost and classifier N at every level; only n-per-class varies
(20 → 10 → 5). Classes are **balanced in adapter count** within each level, so chance = 1/K exactly and
the permutation null is unchanged.

## 3. Label derivation — what the datasets actually yield

Sources (`scripts/asset1_datasets.py`; pools = post-val, POOL_CAP 40,000, VAL_SEED 777 design kept):
alpaca = yahma/alpaca-cleaned (40k pool) · code = sahil2801/CodeAlpaca-20k (~19.5k) · math =
openai/gsm8k main (~7.0k) · xsum = EdinburghNLP/xsum (40k) · squad = rajpurkar/squad v1.1 (40k; **442
native titles**) · agnews = fancyzhx/ag_news (40k; **4 native classes**). Counts Stage-0-verified.

**Label tiers** (every cell tagged in the frozen manifest): **T1 native** (dataset field) · **T2
deterministic** (frozen pure-function rule on the example) · **T3 model-annotated** (external classifier;
label noise unavoidable). Primary curves computed on all classes AND on the T1+T2 "clean core."

| task | L1 (2) | L2 (4) | L3 (per-task) |
|---|---|---|---|
| alpaca | with_input / no_input [T1 — already `TASK_TEMPLATE_KEYS`] | 4 instruction-verb families [T2 keyword rules] | 8 verb families [T2] |
| code | with_input / no_input [T1] | output language {Python / non-Python / …} [T2; skew audit] | 4 only (skew) |
| math | solution steps ≤3 / ≥4 [T2 answer line count] | 4 step bins [T2] | 4 only (pool floor) |
| xsum | doc-length median [T2, weak] or 2 topics [T3] | 4 topics [T3] | 8–12 topics [T3 — weakest cells] |
| squad | 2 frozen title groups [T1 titles + T2 grouping] | 4 title groups | 8–16 title groups [deepest clean axis] |
| agnews | {World,Sports} / {Business,Sci/Tech} [T1 grouped] | 4 native classes [T1 — cleanest cell] | 4 only (no native depth) |

**Pool floor:** class eligible only if training pool ≥ **1,000 examples** (≤ 32 epochs under the fixed
32,000-sequence recipe). Hence the ragged L3: gsm8k (~7.0k/8 ≈ 872) and agnews (4 native) cannot go
deeper; depth comes from squad titles (40k/16 = 2.5k/class) and xsum topics. **Pool survival (approx.):**
L1 ~3.5k (math) – 24k (alpaca); L2 ~1.7k – ~10k; L3 1.0–5k. Realized K frozen pre-training; target ≥ 40.

## 4. Training — recipe held identical

```
recipe = rank-24 q/k/v/o LoRA + 6-channel identity bridge; 2000 opt steps; eff. batch 16
         (bs4xga4, A1 cohort tag); seq 512; labels=input_ids.clone(), no -100 mask
splits = per-CLASS fixed 500-example val (VAL_SEED 777 canonical shuffle of the class pool);
         pools subsampled to common per-class n within each level (discards logged)
seeds  = seed_base/data_seed_base offsets extended per (level, class, seed); pure function of index
```

Held fixed deliberately: 32,000 training sequences per run at every level. Epochs-per-pool therefore
inflates as pools shrink (L3 worst ~32 vs L0's ~1–4.6) — inherent to the axis; named confound (§6.2, §9.2).

## 5. Analysis — H1 machinery, identical at each level

Per level, per representation (**raw**, **canonical** [QR→SVD, bridge absorbed, frozen tooling],
**vocab_signature** [both kv modes]): LOO linear SVM (C = 1.0), 1,000-shuffle permutation null, lock =
**acc > 1.5 × chance(K) AND perm p < 0.01**, Wilson CIs (standing LOO caveat), variance-heterogeneity
guard (trigger 3.7) in the exact per-level feature space.

**Pre-declared readouts:**
1. **Accuracy-vs-granularity curves** — raw / canonical / vocab_signature × {6, 12, 24, K}, with
   **Cohen's kappa** alongside (accuracy is not comparable across K; the 1.5× bar ≈ 0.031 at K = 48
   becomes a floor, not the question).
2. **Ceiling departure** — "ceiling" := LOO acc ≥ 0.99 (≤ 2 errors at N = 240). Confirmatory endpoint:
   first level where canonical drops below 0.99.
3. **Gap trend** — Δ(K) = acc_canonical − acc_raw (and kappa form): grows, shrinks, or non-monotonic.
4. **Parent-collapsed accuracy** — fine confusions collapsed to the 6 parent tasks at every level:
   "loses fine structure, keeps coarse" vs "loses everything."
5. Per-class recalls at L3 (n = 5): **descriptive only** (§8).

## 6. Controls (mandatory)

1. **Data-space separability reference.** Per level: TF-IDF + linear SVM (frozen config, subsampled
   pools) on the training *documents* under the same labels — bounds attainable weight-space accuracy.
   A class whose own text is inseparable cannot be read from weights; failure there is label noise.
2. **Split-pool data-identity control.** One L2 class: adapters on two disjoint halves of the same pool
   (5 + 5 seeds, ~10 runs). If the probe separates the halves, fine-granularity accuracy is partly
   **data fingerprinting** (memorization forensics), not task structure.

## 7. Pre-committed outcome readings

- **(A) Canonical ≥ 0.99 through L3 (~48):** the gauge story generalizes an order of magnitude past
  the coarse regime; the W2T contrast then lives at still-finer granularity, collection heterogeneity,
  or modality — motivates a 96+ extension.
- **(B) Departure at L1–L2 (12–24):** the coarse-task ceiling was easy; the boundary sits within one
  order of magnitude of 6 and the paper's non-claim about the middle was load-bearing.
- **(C) Departure coincides with the data-space reference's:** the constraint is label realizability,
  not the gauge — reported as a bound on the label space, not on canonicalization.
- **(D) Raw accuracy *rises* with K:** read only through control 6.2 — if split-pool halves separate,
  raw is reading data identity; report as fingerprinting, not legibility.
- **(E) Δ(K) grows** → canonicalization increasingly necessary as labels sharpen (converges with W2T's
  raw-vs-symmetry-aware gap); **shrinks** → the gauge obstacle is specific to coarse label spaces.
  Sparse results are data: a monotone, unremarkable decay curve is a publishable bridge measurement.

## 8. Budget honesty

```
runs.new     = 690-730 (L1 240 + L2 240 + L3 200-240 + ~10 control)   [llama only]
rate.basis   = Asset-1 measured: 480 mixed runs Jul 3 21:52Z -> Jul 20 16:21Z (~16.8 d incl. A1
               restart at ~1.9x); pure-bs4xga4 equivalent ~12-13 d -> ~36-39 min/run mixed
rate.llama   = ~30 min/run [ESTIMATE — Stage 0 replaces with measured per-run wall-clock from
               bank_manifest.json timestamps, llama cohort]
gpu.days     = ~14-15 primary | fallback 16/8/4 seeds -> 576 runs, ~12 d
gpu.optional = +~7 d (qwen L2 spot check, 240 runs @ ~42 min);  cpu = negligible (as Asset-1)
```

**Seeds-vs-classes tradeoff, stated:** 40 seeds/class at L3 = 1,920+ runs (~6 weeks) — the honest
answer is thinner seeding. At n = 5/class, per-class recall CIs are wide (±~0.4) and class-level claims
impossible; level-wise accuracy at N = 240 stays well-estimated (Wilson ±~0.06 mid-range) and the
permutation null fully valid. The design buys the **curve**, not per-class structure; if L3 departs
ceiling, a deeper-seed follow-up at the departure level is the next registration.

## 9. Threats to validity

1. **T3 label noise** (xsum topics): bounded by control 6.1; clean-core (T1+T2) curves reported alongside.
2. **Pool shrinkage ↔ granularity confound:** smaller pools → more epochs → memorization-driven
   separability. Epochs/pool reported per class; control 6.2 is the direct test. The confound *helps*
   accuracy: a break despite it is conservative; a hold is not fully clean, and says so.
3. **Semanticity heterogeneity:** format (with/no-input) ≠ topic ≠ entity (titles); made visible via
   parent-collapsed and per-task-block confusions.
4. **Class skew** (code languages): balanced subsampling + pool floor; discarded mass logged.
5. **One family:** curve shape unmeasured on qwen unless item 9 funds it; scope stated in every claim.
6. **Multiple testing:** confirmatory set small and named (§5.2 departure, §5.3 trend, per-level locks);
   all else descriptive; no per-class permutation tests (Asset-1 D1 precedent).
7. **Recipe-vs-exposure:** fixed 32,000 sequences (not fixed epochs) is declared; the alternative
   measures a different question and is out of scope.

## 10. Pre-registration mechanics + Stage 0

Same three layers as Asset-1: locked card (hypotheses + ladder frozen before any training), Director
pins on every open choice, completeness interlock (`require_complete_bank` generalized to the ladder
manifest; levels unlock independently — L1 analyzable while L3 trains — each fired exactly once); dated
amendments only (L-006). **Stage 0 (pre-freeze, CPU-only):** verify dataset counts; build + freeze
subdivision manifests (rules, tiers, realized K, per-class pools); extract measured llama per-run
wall-clock; extend D1 tooling to K-way labels with synthetic selftests; fresh-context adversarial
verification of every label-derivation script before it touches data (D3-labels precedent).

## 11. Director sign-off items

1. **Family:** llama3.2-1b only (cost) — approve, or require both at a reduced ladder.
2. **Ladder + seeds:** 12/24/~48 at 20/10/5 (240/level) — approve, or fallback 16/8/4.
3. **T3 admissibility:** are model-annotated cells in, or does xsum stop at L1? (Gates L3 reaching 40+.)
4. **Pool floor** 1,000/class and the resulting ragged L3 — approve the number.
5. **Ceiling definition** (≥ 0.99 at N = 240) + confirmatory contrast set (§5.1–5.4) — pin.
6. **Data-space reference config** (control 6.1) — pin method and subsample size.
7. **Split-pool control** (6.2) — include (+~10 runs) or drop.
8. **Arm B** (squad-only deep ladder, 2→16 native title groups — the cleanest W2T-analog attribute
   axis; ~144 runs) — include, defer, or reject.
9. **Qwen L2 spot check** (+240 runs, ~7 GPU-days) — fund or defer.
10. **Lock form at high K:** keep 1.5×-chance (very low bar at K = 48) or add a kappa lock — pin wording.

---

*Design draft only. Nothing registered; no manifest frozen; no training launched. Every approximate count (§3) and rate (§8) is replaced by a measured value in Stage 0 before the card locks.*
