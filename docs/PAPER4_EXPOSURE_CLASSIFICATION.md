# Paper 4 Controller-Exposure Classification

**Charter:** First task of the Paper 4 audit per the Director's F4 response and
`C:\falco\docs\MERIDIAN_ROUND2_2026-07-03.md` §2: classify every number reported
in Paper 4 by controller exposure, so re-measurement after the C6b
stability-detector fix is scoped to what the defect could actually have touched.
**Date:** 2026-07-04. **Method:** read-only. Every classification cites a config
field, feedback-log value, code line, or tracker quote. Inferences are marked
INFERRED and listed in §8. No numbers were softened; discrepancies found during
enumeration are reported in §9 as facts.

**Sources enumerated:** `paper/paper4/paper4-main.tex` + its six `\input`
section files (the canonical draft), cross-checked against
`paper/PAPER4_OUTLINE.md` results tables, `docs/EXPERIMENT_TRACKER.md`,
`results/*/config.json`, `results/*/results.json` feedback logs, and
`scripts/train_cybernetic.py`.

---

## 1. Exposure classes

| Class | Definition | Defect exposure |
|---|---|---|
| **ADAPTIVE-GOVERNED** | Run used adaptive Control Law 2 (c_w adaptation) — dynamics governed by the defective adaptive path | Full — re-measure after C6b fix |
| **FIXED-WEIGHT** | `--fixed-contrastive` set, weights scripted, or CL2 inoperative (24C-001 accident; spectral-only runs with no pair spec) | Partial — CL2 out of loop, but CL1 (spectral weight) and CL3 (bridge LR) remain adaptive and carry the defective STABLE-declaration logic (§2) |
| **CONTROLLER-FREE** | No Steersman in the training loop at all | None — immune |
| **ANALYSIS-ONLY** | Number produced by post-hoc analysis of stored artifacts | Inherited from the run(s) analyzed (named per row) |

---

## 2. Defect surface (code-level, `scripts/train_cybernetic.py`)

The C6b defect: *"the adaptive controller can declare STABLE while its
controlled metric is still moving; related to the Control Law 2 telemetry bug
documented in the 24C-001 recovery"* — `docs/BM_BATTERY_PLAN.md` lines 135–140
(Known issues #1). Origin: internal review April 2026; public references:
`DIRECTOR_DIALOGUE_2026-07-03.md` Part C item 6(b); README Paper 4 provisional
block (frozen 2026-07-03).

### 2.1 Which control laws run under fixed contrastive weight

`Steersman.observe_and_decide()`:

- **Control Law 2 (DIRECTIONALITY) is the only law disabled.** Lines 365–373:

  ```python
  # Control Law 2: DIRECTIONALITY
  # If co/cross ratio is stagnant near 1.0, increase contrastive pressure
  # SKIP when fixed_contrastive is set — weight stays constant
  if self.fixed_contrastive is not None:
      signals["directionality"] = (
          f"FIXED (c_w={self.fixed_contrastive:.4f}, ..."
  ```

  CL2 is also silently inoperative whenever `co_cross is None` (line 374 guard
  `elif co_cross is not None and len(co_cross_history) >= 2:`). `co_cross` is
  `None` whenever `coplanar_crossplanar_ratio()`
  (`scripts/train_exp2_scale.py:155`, via `_coplanar_crossplanar_indices`,
  lines 99–152) has no handler for the run's `n` at process-load time. This is
  the 24C-001 accident (tracker, 24C-001 CRITICAL FINDING, Mar 20 2026) and the
  structural reason all spectral-only runs (n∈{3,12} always; n∈{4,8} at the
  time those runs launched) had constant c_w.

- **Control Law 1 (CONNECTIVITY) always runs.** Lines 335–356: adapts
  `_spectral_weight` on `fiedler_trend`; declares
  `signals["connectivity"] = f"STABLE (trend=...)"` (line 356) whenever
  −0.001 ≤ trend ≤ +0.001. Spectral-target adaptation (lines 358–363) also
  always runs. **Not gated by `fixed_contrastive`.**

- **Control Law 3 (STABILITY) always runs.** Lines 400–424: adapts
  `_bridge_lr_scale` on `deviation_trend`; declares
  `signals["stability"] = f"STABLE (trend=...)"` (line 424) whenever
  −0.01 ≤ trend ≤ +0.05. **Not gated by `fixed_contrastive`.**

The STABLE declarations live in **CL1 and CL3** (the trend deadbands, slope
computed over a 5-sample window at 100-step feedback intervals,
`window_size=5`, line 210). A monotonic drift below the per-sample slope
threshold is declared STABLE indefinitely — over a 10K-step run (100 samples)
CL1's deadband alone admits cumulative Fiedler drift up to ~0.1, the full
magnitude of the spectral attractor. **Consequence: FIXED-WEIGHT runs are
partially exposed** — their c_w (the dominant driver of co/cross headline
ratios) is outside the defective loop, but spectral weight and bridge LR
remain adaptively governed. Empirical demonstrations on disk:

- 24C-001 (CL2 inoperative): CL1 drove s_w 0.05 → 0.036 → 0.181 → 0.2 (cap)
  and CL3 damped bridge_lr_scale 1.0 → 0.832 → 0.708 → 0.672
  (`results/channel-ablation/24C-001/results.json` feedback_log, steps 0–3000).
- WL-001 (adaptive): deviation grew to 2.12 while `bridge_lr_scale` stayed
  pinned at 1.0 for the whole run — CL3 never intervened because the
  per-sample deviation slope stayed inside the STABLE deadband. This is a
  live instance of "declares STABLE while the controlled metric still moves."

### 2.2 What the adaptive CL2 path did in the Paper 4 runs (disk evidence)

| Run | c_w trajectory (feedback_log) | CL2 verdict |
|---|---|---|
| T-001r1/r2 | 0.1 → 0.025 floor (`base×0.25`, line 390) by ~step 2400 | ACTIVE (throttled) |
| Seed-43 / Seed-44 | 0.1 → 0.025 floor | ACTIVE (throttled) |
| CW-001 | 0.02 → 0.005 floor by step 1500 | ACTIVE (throttled) |
| WL-001 | 0.1 → 0.24 → 0.48 → **0.5 (max cap, line 199)** from ~step 3600 | ACTIVE (ramped to max) |
| FI-002 P-000/P-001/P-002/P-CTRL | 0.1 → 0.025 floor | ACTIVE (throttled) |
| H-ch3/H-ch12 (and H-ch4/H-ch8 per local logs) | constant 0.1, `co_cross=None`, contrastive loss inactive (`contrastive_active=False`, train_cybernetic.py:602) | INOPERATIVE |
| 24C-001 | constant 0.1, `co_cross=None` entire run | INOPERATIVE (accident) |

Every successful adaptive BD headline was achieved under CL2-throttled c_w
(cut 4× within ~2K steps); both collapse controls ran with CL2 ramped to the
0.5 cap. A corrected detector plausibly changes both trajectories. That is the
concrete mechanism by which the defect could have touched the headline ratios.

---

## 3. Master classification table — `paper4-main.tex` (canonical draft)

Priorities: **P1** = headline, defect-plausibly-material, re-measure before any
publication. **P2** = supporting; re-measure/re-derive in the same campaign.
**P3** = insensitive or secondary; verify opportunistically. **N** = no
re-measurement required.

| # | Claim | Value | Source exp. | Exposure class | Evidence | Re-measure |
|---|---|---|---|---|---|---|
| 1 | Tesseract co/cross @10K | 41,564:1 | T-001r2 | ADAPTIVE-GOVERNED | `results/T-001-full-r2/config.json` `initial_contrastive: 0.1`, no `fixed_contrastive`; feedback_log final co_cross 41,563.66, c_w 0.1→0.025 | **Y P1** |
| 2 | Tesseract Fiedler | 0.000191 | T-001r2 | ADAPTIVE-GOVERNED | feedback_log final fiedler_mean 1.9115e-4 | **Y P1** |
| 3 | 4+4 eigenvalue split (4 @ O(1e-4), 4 @ ~0.34) | 4+4 | T-001r2 | ADAPTIVE-GOVERNED | same run | Y P2 |
| 4 | Tesseract val loss; "within 0.17% of baseline" | 0.4016 | T-001r2 | ADAPTIVE-GOVERNED (insensitive) | checkpoints final val 0.40163 | P3 |
| 5 | Co/cross trajectory 1,921 (400) → 5,009 (2,800) → 41,564 (10K) | traj. | T-001r2 | ADAPTIVE-GOVERNED | feedback_log | Y P2 |
| 6 | T-001r1 partial: 2,700 steps, Fiedler 0.00070, ρ 5,395:1 | 5,395:1 | T-001r1 | ADAPTIVE-GOVERNED | `results/T-001-full/results.json`: identical adaptive pattern (c_w 0.1→0.025) | Y P2 |
| 7 | Reproducibility r = 1.0000, max deviation 3.5%, 6-checkpoint table (§4.3) | r=1.0000 | T-001r1 vs r2 | ANALYSIS-ONLY (inherits ADAPTIVE ×2) | comparison of two adaptive runs; `results/channel-ablation/compare_t001_runs.py` | Y P2 |
| 8 | lm-eval mean Δ −0.75% across six 0-shot tasks | −0.75% | T-001r1 merged | ANALYSIS-ONLY (inherits ADAPTIVE) | `results/lm-eval/t001-adapted/` | P3 |
| 9 | Octahedral per-bridge mean co/cross | **473,622:1** | O-001 | ADAPTIVE-GOVERNED | Tracker FO-001 row: "vs **O-001 adaptive** 473,622:1"; run dir Hermes `results/octahedral/O-001/` (INFERRED from tracker — §8) | **Y P1** |
| 10 | O-001 pooled 369,365:1; median 401,851:1; range 126,782–1,577,518:1 (88 bridges) | pooled/median/range | O-001 | ANALYSIS-ONLY (inherits ADAPTIVE) | per-bridge stats over saved bridges | **Y P1** |
| 11 | O-001 Fiedler | 1.1e-5 | O-001 | ADAPTIVE-GOVERNED | as #9 | **Y P1** |
| 12 | O-001 2+2 split [0, 0, 2.06, 2.08] | 2+2 | O-001 | ADAPTIVE-GOVERNED | as #9 | Y P2 |
| 13 | O-001 val loss | 0.401 | O-001 | ADAPTIVE-GOVERNED (insensitive) | tracker | P3 |
| 14 | O-001 co-planar magnitude 0.086 (500) → 1.027 (10K), linear, no saturation | growth | O-001 | ANALYSIS-ONLY (inherits ADAPTIVE) | checkpoint analysis | Y P2 |
| 15 | RD co/cross @10K | **70,404:1** | H-ch6 (Paper 3 H3) | ADAPTIVE-GOVERNED | `results/channel-ablation/H-ch6/metrics_hermes.csv` final row 70,404.01; tracker FC-001 row: "comparable to **H-ch6 adaptive**" | **Y P1** |
| 16 | RD Fiedler | 0.00009 | H-ch6 | ADAPTIVE-GOVERNED | metrics_hermes.csv final 8.93e-5 | **Y P1** |
| 17 | RD val loss | 0.4015 | H-ch6 | ADAPTIVE-GOVERNED (insensitive) | metrics_hermes.csv 0.40148 | P3 |
| 18 | "H-ch6 exceeded 200:1 by step 200" | 200:1 | H-ch6 | ADAPTIVE-GOVERNED | metrics_hermes.csv | P3 |
| 19 | Seed replication ρ (hierarchy table, conclusion) | 73,309:1 | Seed-43 | ADAPTIVE-GOVERNED | `results/Seed-43/results.json`: c_w 0.1→0.025; final 73,308.5, Fiedler 8.55e-5 | Y P2 |
| 20 | Spectral-only Fiedler n=3 | 0.0951 | H-ch3 | FIXED-WEIGHT (CL2 inoperative; contrastive OFF; CL1 active) | `results/channel-ablation/H-ch3/results.json`: co_cross=None throughout, c_w const 0.1, s_w 0.05→0.025; final fiedler 0.09513 | Y P2 |
| 21 | Spectral-only Fiedler n=4 | 0.0836 (tex) | H-ch4 | FIXED-WEIGHT (as #20) | local partial log to step 1100 shows 0.0836 *at step 1100*; tracker final = **0.0918** — DISCREPANCY §9.1 | Y P2 |
| 22 | Spectral-only Fiedler n=8 | 0.0944 (tex) | H-ch8 | FIXED-WEIGHT (as #20) | tracker channel-ablation table = **0.0889**; summary table = 0.0944 — DISCREPANCY §9.2 | Y P2 |
| 23 | Spectral-only Fiedler n=12 | 0.1019 | H-ch12 | FIXED-WEIGHT (as #20) | `results/channel-ablation/H-ch12/results.json` final 0.10195; co_cross=None, c_w const | Y P2 |
| 24 | Attractor band 0.0836–0.1019; "10.4% band/CV across 4× range" | band | H-ch3/4/8/12 | ANALYSIS-ONLY (inherits FIXED-WEIGHT; CL1-exposed) | derived from #20–23; lower edge rests on the disputed H-ch4 value | Y P2 |
| 25 | Spectral-only ρ ≈ 1:1; val ≈ 0.40 | ~1:1 | H-ch3/4/8/12 | FIXED-WEIGHT | feedback logs (no pair spec ⇒ ratio undefined/isotropic) | P3 |
| 26 | Bifurcation factors 1,020× (n=6), ~470× (n=8); BD classification 0% → 100% | 1,020× / 470× | H-ch6, T-001r2 vs spectral-only | ANALYSIS-ONLY (numerator ADAPTIVE, denominator FIXED-WEIGHT) | ratio of #15/#16 and #1/#2 to #20–23 | Y P2 |
| 27 | Wrong-labels ρ @10K | 8.7×10⁻⁶:1 | WL-001 | ADAPTIVE-GOVERNED | `results/channel-ablation/WL-001/results.json`: CL2 drove c_w 0.1→**0.5 cap**; final co_cross 8.709e-6 | **Y P1** |
| 28 | WL-001 Fiedler | 1.26×10⁻⁵ | WL-001 | ADAPTIVE-GOVERNED | final fiedler_mean 1.2552e-5 | **Y P1** |
| 29 | WL-001 eigenvalues [0,0,0,2.44,2.44,2.44] (3+3 wrong direction) | 3+3 | WL-001 | ADAPTIVE-GOVERNED | same run | Y P2 |
| 30 | WL-001 deviation from identity | 2.12 | WL-001 | ADAPTIVE-GOVERNED — **CL3 non-intervention is a live defect instance** (§2.1) | feedback_log: deviation 2.12 with bridge_lr_scale 1.0 throughout | **Y P1** |
| 31 | WL-001 val loss | 0.4008 | WL-001 | ADAPTIVE-GOVERNED (insensitive) | final val 0.40079 | P3 |
| 32 | Resonance ρ @10K | 8.9×10⁻⁶:1 | R-001 | ADAPTIVE-GOVERNED (INFERRED — §8) | tracker: default Steersman, n=6 (CL2 operable); Hermes `results/resonance/R-001/`; WL-001 twin (<10% divergence) ran c_w→0.5 | **Y P1** |
| 33 | R-001 Fiedler | 1.2×10⁻⁵ | R-001 | ADAPTIVE-GOVERNED (INFERRED) | as #32 | **Y P1** |
| 34 | R-001 chain eigenvalues [0, 0, 1.22, 2.45, 3.67] | chain | R-001 | ADAPTIVE-GOVERNED (INFERRED) | as #32 | Y P2 |
| 35 | R-001 decay: 0.257 (100) → 6.5e-4 (1K) → 8.9e-6 (10K); Fiedler 1.7e-4 (1K) | traj. | R-001 | ADAPTIVE-GOVERNED (INFERRED) | as #32 | Y P2 |
| 36 | R-001 val 0.4008; deviation 2.12 | 0.4008 / 2.12 | R-001 | ADAPTIVE-GOVERNED | as #32; deviation shares #30's CL3 exposure | val P3 / dev **P1** |
| 37 | WL≡R functional equivalence, max divergence <10% at all checkpoints (matched table 1K/5K/10K) | <10% | WL-001 + R-001 | ANALYSIS-ONLY (inherits ADAPTIVE ×2) | trajectory comparison of two adaptive runs | Y P2 |
| 38 | Emanation Fiedler | 0.084 | E-001 | ADAPTIVE-GOVERNED (INFERRED — §8) | tracker: default Steersman, contrastive via master bridge; Hermes `results/emanation/E-001/` | Y P2 |
| 39 | Emanation co/cross | 1.12:1 | E-001 | ADAPTIVE-GOVERNED (INFERRED) | as #38 | Y P2 |
| 40 | Emanation deviation 0.35 (≈1/6 of collapse 2.12) | 0.35 | E-001 | ADAPTIVE-GOVERNED (INFERRED) | as #38 | P3 |
| 41 | Emanation val loss; eigenvalues smooth 0.09→0.25 | 0.4009 | E-001 | ADAPTIVE-GOVERNED (insensitive) | as #38 | P3 |
| 42 | Four-regime boundaries (ρ>40,000:1 / ≈1:1 / ≈10⁻⁵:1; Fiedler bands; "no intermediate outcomes"; "Regime 1 spans 42K–474K, 4 orders separation") | taxonomy | all runs | ANALYSIS-ONLY (inherits mostly ADAPTIVE) | synthesis of #1–41 | Y P2 (follows constituents) |
| 43 | Inverse-n relationship 474K (n=4) > 70K (n=6) > 42K (n=8); pair-count ratios 0.50/0.25/0.167 | ordering | O-001, H-ch6, T-001r2 | ANALYSIS-ONLY (inherits ADAPTIVE ×3); pair-count ratios are MATHEMATICAL FACT | ordering preserved by fixed-weight replications FO-001 262,920:1 > FC-001 67,501:1 (tracker) — supportive but n=8 fixed point missing | Y P2 |
| 44 | Paper 3 recap (§1–2, §8): 100% BD across six experiments / 0% controls; ratios >82,154:1 (§2) and >22,000:1 (§1); 42,500+ bridges (§1) vs 60,000+ (§2); "three model scales (TinyLlama, Qwen 7B, **Wan 2.1 14B**)"; init convergence within 200 steps; I-Ching init suppressed 99.5% in 900 steps; n=3 matches n=6 at 4× fewer bridge params | recap set | Paper 3 corpus (exp3, exp3-TL, C-001/2/3, ablations) | ADAPTIVE-GOVERNED (all Paper 3 cybernetic runs) | Paper 3 scope; **Wan 2.1 14B = Holly Battery, RETRACTED 2026-03-13** (tracker Holly section) — §9.4/§9.5 | Y — owned by Paper 3 audit; Wan-14B reference must be removed regardless |
| 45 | P0 loss parity 0.1763 vs 0.1762 (0.01%) | 0.1762/0.1763 | P0 (Qwen 2.5-1.5B) | **CONTROLLER-FREE** | trainer is `scripts/train_comparison.py` (LM loss only; no Steersman import, no spectral/contrastive loss — grep clean); `results/p0-proof/*/results.json` has **no feedback_log**; config schema lacks all Steersman fields | **N** |
| 46 | P0 params: 3,268,608 / 3,270,624; bridge 2,016 (0.06%); 36 per bridge × 56 projections | params | P0 | CONTROLLER-FREE (artifact counts) | results.json `trainable_params: 3270624` | **N** |
| 47 | P0 time 42.8 vs 58.3 min (+36.2%) | 36.2% | P0 | CONTROLLER-FREE | wall-clock from run logs; **text attribution to "Steersman per-step … control law evaluation" contradicts the trainer on disk — §9.6** | **N** (text fix required) |
| 48 | P0 bridge metrics: Fiedler 0.038; co/cross ≈1:1; deviation 0.07 | 0.038 | P0 | CONTROLLER-FREE | passive measurement; **"evolves under spectral pressure alone" is wrong — no spectral loss exists in this trainer — §9.6** | **N** (text fix required) |
| 49 | Supp. fig: 24-cell co/cross 35,808:1; 12 independent 2×2 blocks | 35,808:1 | 24C-001 (+PC-001) | FIXED-WEIGHT (accidental c_w=0.1; CL2 inoperative) + ANALYSIS-ONLY (PC-001 post-hoc recovery from 100 saved checkpoints) | tracker 24C-001 CRITICAL FINDING; local feedback_log co_cross=None / c_w=0.1 const; **CL1 hit s_w=0.2 cap and CL3 damped blr→0.672 — partially exposed** (§2.1) | Y P2 |
| 50 | Supp. fig FI-002: 100% identical sign patterns from Hamming 9/12 inits; Frobenius ~10⁻⁴ | 100% / 1e-4 | FI-002 P-000/1/2 (+P-CTRL) | ADAPTIVE-GOVERNED + ANALYSIS | `results/fi-002/P-000/results.json`: c_w 0.1→0.025, s_w→0.2 cap; `fi-002-results.json` | Y P2 |
| 51 | Supp. fig FI-003: BD dissolves 12,586:1 → ~10:1 within 1,200 steps when Steersman disabled | 12,586→10 | FI-003 | CONTROLLER-FREE (decay phase; Steersman disabled) — initial 12,586:1 inherited from ADAPTIVE FI-002 P-000 | `scripts/fi_003_*` + tracker: "Steersman DISABLED (no spectral or contrastive loss)" | **N** for the decay dynamics; P3 for the initial magnitude |
| 52 | Supp. fig FI-004: peak coupling 18,671:1 at fixed c_w = 0.017 | 18,671:1 | FI-004 | FIXED-WEIGHT (scheduled linear anneal; adaptive laws not in the loop) | `scripts/fi_004_steersman_annealing.py` lines 246–249: weights computed directly from schedule, `anneal_mode: linear_to_zero`; start checkpoint = adaptive P-000 | P3 |

Appendix configuration constants (Table `tab:config`, channel sizes s = r/n,
hardware list, experiment index pair specs) are configuration, not findings —
no exposure class; excluded from counts. Note: the appendix experiment index
lists 24C-001 nowhere and describes all runs as "identical Steersman
configuration," which is false for 24C-001 (accidental fixed c_w) — §9.7.

---

## 4. Outline-only numbers (`paper/PAPER4_OUTLINE.md` results tables; slated for the final draft but not yet in the tex body)

| # | Claim | Value | Source | Exposure | Evidence | Re-measure |
|---|---|---|---|---|---|---|
| B1 | CW-001 whisper-strength: final 13,456:1; peak 15,183:1 (step 8300); Fiedler 0.00046; val 0.4017; plateau ~1,700:1 (steps 1500–4200); breakout threshold ~2,000:1; four-phase trajectory | full set | CW-001 | ADAPTIVE-GOVERNED — **maximal exposure: the entire trajectory is a direct readout of adaptive c_w decay (0.02→0.005 floor)** | `results/cw-001/config.json` `initial_contrastive: 0.02`; feedback_log c_w 0.02→0.005 by step 1500; final 13,456.1 / 4.64e-4 / 0.40167 | **Y P1** if promoted into the tex |
| B2 | 24C-001 detail: Fiedler 0.000555; stabilization bands 0.000535–0.000588 (2,100 steps) and 34,600–37,600:1 (from step 8000); step trajectory table; val 0.4022; "co/cross grew 6.3× in the CL2 logging blind spot (5,673:1 @4300 → 35,808:1 @10K)" | detail set | 24C-001 + PC-001 | FIXED-WEIGHT + ANALYSIS-ONLY | as row 49; PC-001 recovery from bridge .npy checkpoints | Y P2 |
| B3 | H-ch6 vs CW-001 comparison: 5.2× co/cross gap, 5.1× Fiedler gap at matched steps; "two operating regimes" | 5.2× | H-ch6, CW-001 | ANALYSIS-ONLY (inherits ADAPTIVE ×2) | derived | Y P2 |
| B4 | Step-1000 convergence-rate scaling: 7,224:1 (n=4) / 7,246:1 (n=6) / 4,611:1 (n=8) / 1,832:1 (n=24) vs suppression load | scaling | O-001, H-ch6, T-001, 24C-001 | ANALYSIS-ONLY (mixed inheritance) — **confounded: n=4/6/8 points are adaptive-c_w runs, the n=24 point is fixed c_w=0.1** | tracker signal-density comparison | Y P2 + confound must be stated |
| B5 | FI-002 co/cross 50,344 / 51,677 / 50,382; P-CTRL 10,654:1 @1300 | 50–52K | FI-002 | ADAPTIVE-GOVERNED | `results/fi-002/*/results.json` | Y P2 |

Post-outline tracker experiments not yet cited by either document (FC-001
67,501:1; FO-001 262,920:1; 24C-002 5,983:1; FC-002 null at n=12) are
FIXED-WEIGHT for c_w by config (`fc-001/config.json` `fixed_contrastive: 0.02`)
but retain adaptive CL1/CL3 — if promoted into Paper 4 they are partially
exposed, not clean.

---

## 5. Summary counts

Main-tex rows (§3, 52 rows) by primary class:

| Class | Rows | Row numbers |
|---|---|---|
| ADAPTIVE-GOVERNED | **30** | 1–6, 9, 11–13, 15–19, 27–36, 38–41, 44 |
| FIXED-WEIGHT (partial exposure via CL1/CL3) | **7** | 20–23, 25, 49, 52 |
| CONTROLLER-FREE (immune) | **5** | 45–48, 51 |
| ANALYSIS-ONLY (exposure inherited) | **10** | 7, 8, 10, 14, 24, 26, 37, 42, 43, 50 — of these, 8 inherit adaptive exposure, 1 (row 24) inherits fixed-weight/CL1 exposure, 1 (row 26) mixed |

Outline-only rows (§4): 2 ADAPTIVE (B1, B5), 1 FIXED-WEIGHT+ANALYSIS (B2),
2 ANALYSIS-ONLY inheriting adaptive/mixed (B3, B4).

Re-measurement requirement across all 57 rows: **P1 = 11 rows, P2 = 25 rows,
P3 = 12 rows, N = 5 rows, deferred-to-Paper-3 = 1 row (44), text-fix-only
flagged on 2 of the N rows (47, 48).**

---

## 6. Headline values requiring re-measurement after the C6b fix (P1)

1. **473,622:1** (+ pooled/median/range) and **Fiedler 1.1e-5** — O-001, adaptive (rows 9–11).
2. **70,404:1** and **Fiedler 0.00009** — H-ch6, adaptive (rows 15–16). Fixed-weight bracket already exists: FC-001 = 67,501:1 (c_w fixed 0.02) — 96% of the adaptive value, so the *qualitative* claim is near-certainly safe; the *published number* still requires post-fix re-measurement under the freeze.
3. **41,564:1** and **Fiedler 0.000191** — T-001r2, adaptive (rows 1–2). No fixed-weight tesseract bracket exists (CW-002 was planned, never run).
4. **8.7×10⁻⁶:1 / 1.26×10⁻⁵** (WL-001) and **8.9×10⁻⁶:1 / 1.2×10⁻⁵** (R-001) — the collapse regime numbers (rows 27–28, 32–33). Both collapses were produced with CL2 ramped to the 0.5 cap and CL3 inert while deviation grew to 2.12 (rows 30, 36) — the regime's *depth* is a joint product of the defective adaptive path. The four-regime taxonomy's Regime 4 boundary re-derives from these.
5. **CW-001 13,456:1 / 15,183:1 / 0.00046** (B1) — if promoted into the tex. The entire "two operating regimes / c_w is the speed dial" argument is a readout of the adaptive path and is the single most defect-entangled result in the programme.

## 7. Certifiably clean (no re-measurement)

- **All P0 numbers** (rows 45–48): loss parity 0.1762/0.1763, parameter counts,
  +36.2% time, Fiedler 0.038, co/cross ≈1, deviation 0.07. Trainer verified
  controller-free in code and artifacts. Two *sentences* about them need
  correction (§9.6), but the numbers stand.
- **FI-003 decay dynamics** (row 51): the homeostasis claim (topology dissolves
  in ~100–1,200 steps without the controller) is measured with the controller
  removed. Only the *starting magnitude* (12,586:1) is adaptive-inherited.
- Configuration constants and pair-specification set definitions (mathematical
  facts).
- The `rd_graph` / BM-series results are controller-free by construction but
  are not reported anywhere in Paper 4, so nothing in the draft rides on them.

Qualitative structural claims (existence of the BD regime, the attractor band's
existence, eigenvalue-split shapes, geometric-coherence selectivity) are
supported across both adaptive and fixed-c_w regimes (24C-001, FC-001, FO-001)
and are unlikely to invert — but under the freeze they carry the provisional
marker until the post-fix campaign confirms the magnitudes.

## 8. Could not be determined from disk (flagged, not guessed)

1. **O-001, R-001, E-001 configs and feedback logs** live only on Hermes
   (`/home/timm156/rhombic/results/{octahedral,resonance,emanation}/`). O-001's
   adaptive status is explicit in the tracker (FO-001 row: "O-001 adaptive").
   R-001 and E-001 adaptive status is INFERRED from the tracker's "default
   Steersman" channel-ablation protocol and (for R-001) its <10%-divergence
   twin WL-001, whose local log shows c_w→0.5. Config-level confirmation
   requires a Hermes pull; until then rows 32–41 are inference-backed.
2. **H-ch4 and H-ch8 final Fiedler values** cannot be adjudicated locally: the
   local H-ch4 log is a 1,100-step partial (0.0836 at step 1100); the tracker
   holds two conflicting values for each run (§9.1–9.2). Full-run logs are on
   Hermes.
3. **Paper 3 recap magnitudes** (82,154:1; 60,000+/42,500+ bridges; 22,477:1;
   99.5%/900 steps) were not re-derived here — they are Paper 3 audit scope.
   Their exposure class (adaptive) is certain; their values are not re-verified
   by this document.
4. **WL-001's actual random partition** is reproducible from code
   (`_compute_wrong_label_pairs(seed=42)`) but the concrete pair identities
   were not re-executed here (read-only task).

## 9. Number-consistency flags found during enumeration (facts, cited)

1. **tab:attractor H-ch4 = 0.0836** (`section5_attractor.tex` line 27) vs
   tracker/outline final **0.0918**. 0.0836 matches the *step-1100 partial*
   local log and coincidentally equals E-001's final Fiedler. The same section's
   §5.3 text uses 0.0918 and labels it "H-ch4" — the tex is internally
   inconsistent, and the attractor band's lower edge (0.0836) depends on which
   is right.
2. **H-ch8 = 0.0944** (tex, outline, tracker summary) vs **0.0889** (tracker
   channel-ablation results table and Key Numbers table). Unresolved on local disk.
3. **Intro: "r = 1.0000 across 34 matching checkpoints"** vs **§4.3: "6
   matching steps"** (and a 6-row table). The outline says 6.
4. **Intro "42,500+ bridges" vs §2 "82,154:1 across 60,000+ bridges"** — the
   two Paper 3 recap counts disagree.
5. **§2 cites "three model scales (TinyLlama 1.1B, Qwen 7B, Wan 2.1 14B)".**
   The Wan 2.1 14B evidence is the Holly Battery, **RETRACTED 2026-03-13**
   (tracker: "must not be cited as findings"). The recap must drop or reword
   the third scale independent of any C6b re-measurement.
6. **§8 P0 text misattributes mechanism:** "the 36.2% training time overhead
   reflects the Steersman's per-step Fiedler eigenvalue computation and control
   law evaluation" and "the bridge evolves under spectral pressure alone" —
   the P0 trainer (`train_comparison.py`) contains no Steersman and no spectral
   loss; overhead comes from bridge metric collection, and the bridge evolves
   under LM loss only. Numbers unaffected; sentences false as written.
7. **Appendix claims "identical Steersman configuration" across all
   experiments** — false for 24C-001 (accidental fixed c_w=0.1, CL2
   inoperative), which the appendix experiment index omits entirely while the
   supplementary figure reports its 35,808:1.
8. **Octahedral pair spec inconsistency in the tex:** Eq. (3) and
   `tab:topology_specs` give pairs(4) = {(0,1),(2,3)} (matching
   `_compute_pair_indices(4)` in code), but §3.3 text says
   "pairs_oct(4) = {(0,3),(1,2)}".
9. **Attractor band width:** tex says "10.4% band/CV"; outline says "19.4%
   band" for the same 0.0836–0.1019 interval. (0.1019/0.0836 − 1 = 21.9%;
   (max−min)/mean ≈ 19.4%; neither obviously yields 10.4% — the tex figure
   caption repeats 10.4%.)
10. **Tracker's channel-ablation table lists T-001r2 val loss 0.439** — stale;
    `results/T-001-full-r2/results.json` final val = 0.40163, matching the
    paper's 0.4016.

---

*Classification executed 2026-07-04 per the audit charter
(MERIDIAN_ROUND2_2026-07-03.md §2; AMENDMENTS_REQUEST_2026-07-04.md §4).
Read-only except this file. Re-measurement scoping: fix the CL1/CL3 STABLE
deadband logic and the CL2 telemetry gate first (BM_BATTERY_PLAN.md Known
Issues #1), then run the P1 list, then P2 within the same campaign.*
