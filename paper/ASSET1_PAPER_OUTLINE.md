# Asset-1 Paper — Complete Outline

**Working name:** rhombic-asset1 (house of rhombic-xr001)
**Status:** OUTLINE — for section writers. Every number below is copied from a
grounding document; none is computed fresh. LaTeX conversion comes after the
markdown draft.

**Grounding documents (the only permitted number sources):**

| tag | document |
|---|---|
| [DELIVERY] | `C:\falco\docs\ASSET1_BANK_DELIVERY_2026-07-20.md` (verified campaign report — NOTE: lives in falco-root `docs/`, not `rhombic/docs/`) |
| [SIGNOFF] | `rhombic/docs/DIRECTOR_SIGNOFF_ASSET1_2026-07-21.md` (independent regrade at commit `638f4a8`) |
| [DECISIONS] | `rhombic/docs/DIRECTOR_DECISIONS_2026-07-06.md` (pinned pre-registration decisions) |
| [PIPELINE] | `rhombic/docs/ASSET1_ANALYSIS_PIPELINE.md` (methods detail: interlock, fire sequence, decision rules) |
| [VERIFY-README] | `rhombic/results/asset1-delivery-verify/README.md` (per-item data map; exact D3 feature-set definition) |
| [PREDECL] | `rhombic/results/asset1-delivery-verify/D3_PAIR_DESIGN_PREDECLARATION_2026-07-20.md` (dated amendment) |
| [d1.json] | `results/asset1-delivery-verify/d1_results.json` |
| [d2.json] | `results/asset1-delivery-verify/d2_results_{qwen2.5-1.5b,llama3.2-1b}.json` |
| [d3.json] | `results/asset1-delivery-verify/d3_report.json` (+ `d3_labels.json`, `d3_pairs.json`) |
| [daux.json] | `results/asset1-delivery-verify/daux_report.json` (+ `daux_run_table.csv`) |
| [manifest] | `results/asset1-delivery-verify/bank_manifest.json` |
| [XR001-STYLE] | `rhombic/paper/rhombic-xr001.tex` (voice/structure reference only — no numbers) |
| [THESIS] | `rhombic/docs/THESIS.md` (project framing only) |

---

## 1. Title candidates

1. **Weight-Only Diagnostics for LoRA Adapters: A Pre-Registered 480-Run Bank Where Canonicalization Is the Whole Story**
2. **The Gauge Is the Obstacle: Task Identity, Cross-Family Transfer, and Merge-Conflict Prediction from Adapter Weights Alone**
3. **Refuted by Our Own Control: Pre-Registered Weight-Space Diagnostics on a 480-Adapter LoRA Bank**

(House convention per [XR001-STYLE]: declarative claim-first title, colon,
plain descriptive subtitle. Candidate 3 leads with the honesty register;
candidate 1 leads with the artifact; candidate 2 leads with the mechanism.)

---

## 2. Abstract skeleton

One paragraph, five results in delivery order, matching the register of the
XR-001 abstract (claim, design, numbers, implication — no hedging tics).
Skeleton with pinned numbers and their sources:

> We train a pre-registered bank of 480 LoRA adapters (2 model families ×
> 6 tasks × 40 seeds; Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct)
> [DELIVERY §1] and run five locked weight-only analyses against it exactly
> once, under a completeness interlock, with every open analytical choice
> pinned by an independent Director before the bank completed [DECISIONS,
> PIPELINE §3–4].
> **(1) H1 — canonicalization ceiling:** raw adapter weights do not reveal
> their training task (LOO accuracy 0.0792 qwen / 0.1375 llama, both failing
> the 1.5×-chance lock at chance 0.1667), while the GL(r)-gauge-canonical
> and vocab-signature representations identify all six tasks at 1.0000, LOO,
> permutation p = 0.000999 [DELIVERY §2].
> **(2) H2 — refutation by our own control:** the pre-registered prediction
> that task structure would NOT transfer across families is refuted, and
> refuted specifically by the triviality control added in review: raw
> transfer-at-chance was a family-scale artifact (family-identity probe
> 1.0000 raw → 0.1521/0.0000 standardized), and shift-controlled transfer
> runs 0.7375–0.7833 (binomial p ≤ 1.20e-84) [DELIVERY §3].
> **(3) D2 — the backbone is the load-bearing structure:** swapping trained
> bridges across tasks costs ~0.0000 val-loss; only full-entry permutation
> that destroys the identity backbone costs +2.8086 (qwen) / +3.8365 (llama)
> nats [DELIVERY §4].
> **(4) D3 — post-hoc merge prediction:** weight-only features predict
> midpoint-merge conflict at group-aware AUC 0.995 [0.983, 1.000] (qwen) /
> 0.962 [0.898, 0.999] (llama), a +0.320/+0.249 margin over a distance-only
> baseline with both margin CIs excluding 0 [DELIVERY §5] — in the weight-only
> post-hoc regime, distinct from the training-time form of arXiv:2606.19549.
> **(5) D-aux — honest shrink:** the pilot bridge-deviation↔generalization-gap
> correlation r = 0.888 shrinks to r = 0.300 [0.175, 0.415] pooled at bank
> scale, real but modest and task-heterogeneous [DELIVERY §6].
> Every headline was independently re-derived from per-item data by the
> Director [SIGNOFF]. Costs and limitations are reported alongside benefits.

---

## 3. Section list with headline claims and number carriers

Structure mirrors [XR001-STYLE]: Introduction → Related work → Experimental
design → Results → Discussion → Reproducibility.

### §1 Introduction
- **Claim:** weight-only, post-hoc diagnostics on LoRA adapters are
  measurable at scale, and pre-registration with pinned decisions is what
  makes the answers trustworthy — including the one that refuted us.
- Contributions list (house pattern): the bank artifact; the H1
  canonicalization ceiling; the H2 refutation-by-control; the D2 backbone
  result; the D3 post-hoc merge predictor; the D-aux honest re-verification.
- Numbers: headline five, copied from [DELIVERY §0] one-paragraph result.
- Frame from [THESIS] only at the "examine every default" register — do not
  import lattice numbers into this paper.

### §2 Related work
- **Claim:** the regime contrast with W2T runs on the LABEL-GRANULARITY axis;
  the merge-prediction claim is scoped weight-only/post-hoc against
  arXiv:2606.19549. (See §6 coverage list below; rules 4–5 bind.)
- No bank numbers in this section except the 10k+-classes vs 6-coarse-tasks
  contrast, whose wording comes from [DECISIONS A2] / [d1.json
  `h2_cross_family.regime_axis`].

### §3 The bank (experimental design, part 1)
- **Claim:** a 480/480-complete, cohort-tagged, outage-audited adapter bank.
- Carriers: [DELIVERY §1] typed core (480/480 COMPLETE at
  2026-07-20T16:21:19Z; bs4×ga4 geometry per A1; 4 HF-Hub-504 retries idx
  329–332 all COMPLETE; ~17 days Jul 3→Jul 20; interlock HELD; 171/171
  tests; chance 0.1667). Family/task/seed grid, max_steps 2000, val_seed
  777, val_size 500, tasks alpaca/code/math/xsum/squad/agnews: [manifest].
  Module counts 112 (qwen) / 64 (llama) discovered per adapter: [PIPELINE
  header]. A1 bit-equivalence argument if summarized: [DECISIONS A1].

### §4 Pre-registration and analysis pipeline (design, part 2)
- **Claim:** every classifier/correlation ran against the real bank exactly
  once; every open choice was pinned before completion; deviations are dated
  amendments, never silent revisions.
- Carriers: [PIPELINE §1–4] (interlock semantics, fire sequence, pinned
  decision rules: H2 α = 0.01 / ≥15pp both directions; D2
  permuted_deviation as THE H3 reference, decomposition unconditional; D3
  fixed 5% relative-degradation label rule, degenerate floor 10%);
  [DECISIONS] for provenance of each pin; [PREDECL] for the D3 pair design
  and its dated amendment (report per rule 6 below).

### §5 Results — H1: task identity is legible only after canonicalization
- **Claim:** raw fails the lock; canonical and vocab_signature sit at ceiling.
- Table carrier: [DELIVERY §2] (8-row representation table: raw
  0.0792/0.1375 FAIL at dims 6,541,248/5,114,112; canonical 1.0000 PASS at
  88,704/50,688; vocab_signature and kv_exclude 1.0000 at
  15,232/7,616/8,704/4,352; perm p 0.000999 throughout).
- Guards: heterogeneity ratios 1.00–1.46 vs trigger 3.7 [DELIVERY §2,
  SIGNOFF H1]; raw's weak recoverable signal (code recall 0.40–0.45) and
  why the 1.5× lock rejects it [DELIVERY §2].
- Independent re-derivation sentence: canonical LOO re-run from the feature
  matrices from scratch, 1.0000 both families [SIGNOFF H1].
- Per-item source for any recomputed table cell: [d1.json
  `families.<fam>.representations.<rep>`].

### §6 Results — H2: the pre-registered prediction, refuted by its own control
- **Claim (rule 2, non-negotiable):** H2 predicted transfer FAILS; the
  verdict is NOT SUPPORTED; the triviality control added in round-1 review
  is what converted a would-be false confirmation into a refutation. This is
  pre-registration working as designed. Never spin as a predicted success.
- Carriers: [DELIVERY §3] — family-identity probe raw 1.0000 → standardized
  0.1521 (spectrum) / 0.0000 (probe); transfer raw 0.1667/0.1667 (spectrum)
  and 0.1750/0.2167 (probe) vs standardized 0.7833/0.7375 (spectrum) and
  0.7792/0.7792 (probe); binomial p 7.70e-98 / 1.20e-84 / 1.37e-96; margins
  21.67–26.25pp; spectrum and probe agree; verdict NOT supported.
- Control integrity sentence: `familywise_standardize` confirmed unsupervised
  (per-family z-scoring, task labels never touched) [SIGNOFF H2].
- Decision-rule constants and verdict fields: [d1.json `h2_cross_family`].
- Regime sentence(s) keyed ONLY to label granularity [DECISIONS A2] (rule 5).

### §7 Results — D2: the identity backbone is the sole load-bearing structure
- **Claim:** trained bridges are nearly free to swap; only destroying the
  backbone costs.
- Table carrier: [DELIVERY §4] 7-kind × 2-family penalty table (cross_seed
  +0.0000/+0.0002; cross_task +0.0000/+0.0003; magnitude +0.0000/+0.0000;
  topology +0.0000/+0.0002; identity +0.0000/+0.0007; permuted_deviation
  +0.0000/+0.0007; permuted full +2.8086/+3.8365).
- 360 evals/family, all assembly SHA-verified [DELIVERY §4]; all 14 per-kind
  means reproduce from per-eval rows [SIGNOFF D2]. Per-eval source:
  [d2.json `penalties`].

### §8 Results — D3: weight-only post-hoc merge-conflict prediction
- **Claim:** two adapters' weights alone predict midpoint-merge degradation
  near ceiling; the gauge-invariant block adds a CI-separated margin over
  raw distance.
- Table carrier: [DELIVERY §5] — full AUC 0.995 [0.983, 1.000] qwen / 0.962
  [0.898, 0.999] llama; distance-only 0.675 [0.484, 0.848] / 0.713
  [0.458, 0.923]; diff +0.320 [0.150, 0.511] / +0.249 [0.039, 0.490].
  Pooled OOF (descriptive): full 0.9890 [0.9730, 0.9992], distance 0.7082,
  diff 0.2808 [0.1565, 0.4205] [d3.json `pooled_oof`].
- Design: N = 120 vertex-disjoint pairs/family, α = 0.5 midpoint merges,
  conflict rate 85.8% under the primary 5% rule (above the 10% degenerate
  floor — no median fallback) [DELIVERY §5; d3.json `binarization`
  frac_positive 0.8583].
- Group-aware vs naive: vertex-disjoint design gave 120 single-pair
  components, so group-aware and naive agree within fold-reshuffle noise
  (llama 0.962 group-aware vs 0.952 naive) [DELIVERY §5; d3.json].
- **Baseline text (rule 3, Director-binding):** state the pinned CV fold
  seed — `seed = 0`, `n_splits = 5`, logistic model, n_boot = 1000
  [d3.json top-level fields] — AND report the seed-sensitivity: the
  Director's independent re-run of the 2-feature distance-only baseline
  gave 0.686/0.667 vs the reported 0.675/0.713, "a few points of
  CV-seed sensitivity on a 2-feature model over 120 points"; the
  margin-over-distance CI lower end depends on this baseline, so the seed
  is pinned and the sensitivity reported [SIGNOFF, Net item].
- **Feature-set definition (copy exactly):** `distance` =
  `[cos_distance, l2_distance]`; `full` = those 2 + 4 aggregates
  `[angle_mean_weighted, angle_mean_unweighted, chordal_rms_weighted,
  chordal_rms_unweighted]` + per-module `module_l2` (len 112 qwen / 64
  llama) + per-module `module_angle_mean` (same len, NaN→0.0);
  `module_chordal_rms` and `module_weight` carried but NOT in `full`
  [VERIFY-README, d3_pairs.json row].
- **Scope sentence (rule 4):** arXiv:2606.19549 owns the training-time
  form of merge-conflict prediction; this result is weight-only, post-hoc,
  no training access [DELIVERY §5; LITERATURE_WATCH_2026-07-03 framing].
- **Amendment reporting (rule 6):** the pair design was pre-declared
  2026-07-20 ~16:50Z with per-cell stratification, then amended ~19:4xZ —
  before Step 5, zero pairs/merges/labels in existence — to the uniform
  sampler the frozen tool actually implements, with realized (task_i,
  task_j) cell coverage reported descriptively (expected ~16% same-task
  under uniformity) [PREDECL]. Director ruling: "clean dated amendment
  (L-006 / R10); no objection" [SIGNOFF §7 items]. Report this in the
  paper body or a design box, not a footnote.
- Interpretive caveat available if needed: principal-angle features are
  provably insensitive to bridge numerics (bridge info reaches the
  classifier via magnitude weights and raw distances) [PIPELINE D-auxd].

### §9 Results — D-aux: the pilot correlation shrinks honestly
- **Claim (rule 6):** pilot r = 0.888 was small-n/task-mixture inflated;
  at bank scale the association is real, positive, modest, and
  heterogeneous within task. Reported as the pooled claim with
  Simpson's-guard cells shown, not hidden.
- Table carrier: [DELIVERY §6] — pooled r = 0.300 [0.175, 0.415] (n=480);
  qwen 0.418 [0.323, 0.522]; llama 0.337 [0.201, 0.466]; within-math
  llama 0.549 [0.324, 0.736], qwen 0.336 [0.030, 0.652]; within-xsum qwen
  −0.301 [−0.502, −0.066]; step-0 identity control = 0.0 exactly.
- Per-cell source: [daux.json `primary.cells`]. Pilot 0.888 provenance:
  [DECISIONS D-aux] ("Re-verify r=0.888 … on the bank").

### §10 Discussion
- **Claims:** (a) the gauge, not the information, was the obstacle — one
  mechanism (GL(r) ambiguity / family scale) explains H1-raw failure and
  the H2 raw artifact; (b) D2 and D-aux together say the trained bridge
  deviation is real (D-aux sees it) but negligible to in-distribution loss
  (D2) [DELIVERY §4 reading]; (c) what pre-registration bought: the H2
  refutation and the D-aux shrink are the credibility of the other three
  results; (d) costs and limitations (rule 7): ~17 days of a single local
  GPU [DELIVERY §1], two model families of 1–1.5B only, 6 coarse tasks
  (the label-granularity axis cuts both ways — no claim about 10k-class
  regimes), midpoint merges only, val-loss-based labels, D3 baseline
  seed-sensitivity [SIGNOFF], pilot-scale D-aux heterogeneity.
- Run-log honesty (house register): D1 first launch died on the
  default-HF-cache trap (environment-level, nothing unblinded); the D3
  labels runner was written post-card, adversarially verified fresh-context
  BEFORE its GPU run, two blocking defects caught; natives taken from
  trainer metrics after 0.00000% divergence check vs 36 fresh D2 evals
  [DELIVERY §8; SIGNOFF §7 items].

### §11 Reproducibility
- **Claim:** every number re-derivable from the per-item verification
  bundle; independently re-derived by the Director.
- Carriers: bundle contents and recompute recipes [VERIFY-README]; commit
  `638f4a8`, archive sha256 `c1891d50…`, all 16 files matching SHA256SUMS
  [SIGNOFF Integrity]; interlock description [PIPELINE §3]; deterministic
  feature extraction, fixed seeds [VERIFY-README Reproduction environment].
- Release scope note for Timothy's decision (bank/features are currently
  local-only; two-stream IP rules apply) — placeholder, not a claim.

---

## 4. Figure and table list

| # | item | content | generate from |
|---|---|---|---|
| T1 | Bank design table | 2 families × 6 tasks × 40 seeds; steps, val split, geometry | [manifest] campaign block (copy fields) |
| T2 | H1 representation table | 8 rows: family × {raw, canonical, vocab_signature, vocab_sig_kv_exclude}: dim, LOO acc, perm p, lock | copy [DELIVERY §2]; per-item check vs [d1.json] |
| F1 | H1 confusion matrices (raw vs canonical, per family) | 6×6 heatmaps showing raw scatter vs canonical diagonal | [d1.json] `representations.<rep>.confusion_matrix.rows_true_cols_pred` |
| T3 | H2 triviality-control table | probe acc raw 1.0000 → std 0.1521/0.0000 | copy [DELIVERY §3]; source [d1.json] `h2_cross_family.<rep>.family_probe` |
| T4 | H2 transfer table | 4 direction-rows: raw vs standardized acc, binom p, margin | copy [DELIVERY §3]; source [d1.json] `h2_cross_family.<rep>.pairs` + `.decision` |
| F2 | H2 bar figure | raw vs standardized transfer per direction/representation, chance line 0.1667 | [d1.json] same paths as T4 |
| T5 | D2 penalty matrix | 7 kinds × 2 families mean penalty | copy [DELIVERY §4]; recompute check = mean(val_loss − native) grouped by kind from [d2.json] `penalties` |
| F3 | D2 penalty strip/log plot | per-eval penalties by kind (~0 cluster vs +2.8/+3.8 permuted) | [d2.json] `penalties` per-eval rows |
| T6 | D3 headline table | per-family group-aware full / distance / diff AUC with CIs + pooled OOF row | copy [DELIVERY §5] + [d3.json] `pooled_oof`; state seed=0, n_splits=5 in caption + Director 0.686/0.667 sensitivity [SIGNOFF] |
| F4 | D3 ROC or OOF-score separation figure | full vs distance-only, per family | [d3.json] per-fold OOF scores |
| F5 | D3 realized cell-coverage figure (amendment transparency) | (task_i, task_j) pair counts under the uniform sampler; same-task fraction | [d3_pairs.json] pair task fields; expectation ~16% same-task [PREDECL] |
| T7 | D-aux correlation table | pooled + per-family + named within-task cells, Pearson r with CIs; step-0 control | copy [DELIVERY §6]; source [daux.json] `primary.cells`, `step0_control` |
| F6 | D-aux scatter | dev_mean vs final_gap, 480 points, colored by family; pooled r annotated | [daux_run_table.csv] |
| F7 | (optional) shrink figure | pilot r=0.888 vs bank pooled/per-cell forest plot | pilot value [DECISIONS D-aux]; bank values [daux.json] |

Figure style: house palette rules live in `rhombic/CLAUDE.md`; XR-001 figure
conventions in `paper/figures-xr001/`.

---

## 5. Related-work coverage list

1. **W2T — arXiv:2603.15990** ("LoRA Weights Already Know What They Can
   Do," Mar 2026). Weight-space attribute classification/retrieval over
   10k+-adapter collections; QR→SVD canonicalization. **Contrast on the
   LABEL-GRANULARITY / task-structure axis ONLY** (their 10k+ fine-grained
   attribute classes vs our 6 coarse tasks). W2T's own collections are
   same-base/same-rank families (their Table 6), so the hub-scale-vs-family
   framing mis-attributes the split and is RETIRED [DECISIONS A2]. Our
   canonical-vs-raw contrast within a controlled family is the complement,
   not a contradiction.
2. **arXiv:2606.19549** (Jun 2026). Owns the **training-time** form of
   merge-conflict prediction. D3 is scoped exactly as the card scopes it:
   weight-only, post-hoc, no training access. State the boundary plainly;
   claim only our regime. MERGE-PEFT is a usable benchmark citation in the
   same territory (per `docs/LITERATURE_WATCH_2026-07-03.md`).
3. **LoRA canonicalization / weight-space symmetry literature:** the QR→SVD
   canonicalization lineage (W2T), spectral readouts of adapters —
   Spectral Geometry (arXiv:2604.08844: linear classifiers on adapter
   spectra, AUC≈1.00 within-method, cross-method fails), and the
   symmetry-aware-encoder line W2T reports against raw-flattened baselines.
   Position H1 as: within-family, a *linear* probe reaches ceiling once the
   GL(r) gauge is removed — canonicalization, not model capacity, is the
   binding constraint at coarse label granularity.
4. **Adjacent weight-only diagnostics (prior art in kind):** backdoor
   forensics from adapter weights (arXiv:2602.15195); WeightWatcher-PEFT
   for overfit-adjacent weight diagnostics (D-aux's neighborhood). Cite as
   kind-precedent per `docs/LITERATURE_WATCH_2026-07-03.md`.
5. **House lineage:** rhombic-xr001 (pre-registration + typed-state
   discipline as the operational method that produced this report's
   numbers); the experiment card + Director-decision protocol [DECISIONS]
   as the pre-registration instrument. Cite for method provenance, not for
   numbers.

Verification note for the section writer: items 1–4 must be re-checked
against the papers themselves before submission; the arXiv IDs above are
copied from `docs/LITERATURE_WATCH_2026-07-03.md` and `docs/LITERATURE_WATCH.md`,
not re-derived.

---

## 6. Constraints appendix — binding rules for section writers

Restated from the paper-architect brief; rules 2–6 verbatim in intent.
Rule 1 (numbers copied, never computed or remembered; missing number →
`[NUMBER: <where it lives>]`), rule 7 (cost alongside benefit), and rule 8
(XR-001 voice: plain, precise, confident; no "remarkably" / "moreover" /
"it is worth noting") apply globally.

- **Rule 2 — H2 is a refutation, and we say so.** The pre-registered
  prediction was that task structure would NOT transfer across families.
  The verdict is NOT SUPPORTED [d1.json `h2_verdict.headline_supported:
  false`]. The refutation was produced by the triviality control added in
  round-1 review (family-identity probe: raw 1.0000 → standardized
  0.1521/0.0000). Frame: *pre-registration working as designed* — the
  control prevented a false confirmation on a covariate-shift artifact
  [DELIVERY §3; SIGNOFF H2]. Prohibited: any wording that presents H2 as a
  predicted or hoped-for success, or that buries the refutation.
- **Rule 3 — D3 distance-only baseline text MUST state the pinned CV fold
  seed and its seed-sensitivity.** Pinned: `seed = 0`, `n_splits = 5`
  [d3.json]. Sensitivity: Director's independent re-run of the baseline
  gave 0.686/0.667 vs the reported 0.675/0.713 — a few points of CV-seed
  sensitivity on a 2-feature model over 120 points; the full-model result
  and the existence of the margin are not in question, but the
  margin-over-distance CI lower end depends on the baseline [SIGNOFF].
  This is a Director write-up requirement, not optional color.
- **Rule 4 — Scope D3 against arXiv:2606.19549 exactly as the card does.**
  That work owns the training-time form of merge-conflict prediction; ours
  is weight-only and post-hoc, with no training access. No claim of
  priority over the training-time form; no blurring of the two regimes.
- **Rule 5 — The D1 regime contrast uses the LABEL-GRANULARITY axis
  only.** W2T's 10k+ fine-grained attribute classes vs our 6 coarse tasks.
  The "canonicalization necessary at hub scale, unnecessary within a
  family" framing is RETIRED — W2T's collections are themselves
  same-base/same-rank families [DECISIONS A2; d1.json `regime_axis`].
  Prohibited phrases: "hub scale vs family," "hub-scale framing," or any
  sentence attributing the raw/canonical split to collection size.
- **Rule 6 — Report the dated D3 amendment and the D-aux shrink openly.**
  (a) The D3 pair-design pre-declaration (2026-07-20 ~16:50Z) was amended
  (~19:4xZ, before Step 5, zero labels in existence) from per-cell
  stratification to the uniform sampler; realized cell coverage is reported
  descriptively [PREDECL]; the Director approved it as a clean dated
  amendment [SIGNOFF]. It appears in the paper body, dated, not hidden.
  (b) The pilot r = 0.888 → bank r = 0.300 [0.175, 0.415] shrink is
  reported as a headline property of the result, attributed to small-n and
  task mixture, with the within-task heterogeneity cells (including the
  negative xsum/qwen cell, −0.301) shown [DELIVERY §6]. Honesty is the
  house register: the shrink and the refutation are selling points of the
  method, and they are written that way — plainly, without apology and
  without spin.

---

*Outline assembled 2026-07-21 by the paper architect. Every number above was
copied from the tagged grounding document; the delivery report's true path
(falco-root `docs/`) is flagged because the brief's `rhombic/docs/` path does
not exist.*
