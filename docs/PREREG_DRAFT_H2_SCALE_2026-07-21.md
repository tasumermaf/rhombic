# Pre-Registration Proposal — H2 Follow-Up: Cross-Family Universality at Scale + the Vocab-Signature Mapping

**STATUS: DRAFT — for Director review and ruling; NOT registered; no data
collected; hypotheses NOT locked until the Director locks them.**

Date: 2026-07-21 · From: Meridian · To: the Director (cc: PI)
Grounding: `DIRECTOR_SIGNOFF_ASSET1_2026-07-21_v2.md` (H2 numbers re-derived
end-to-end), `DIRECTOR_DECISIONS_2026-07-06.md` (§6/H2 pinned machinery),
`DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md` (A3 §2: cross-family
vocab_signature = DESIGN ONLY, "a separate H2 decision" — this document
proposes that decision), `ASSET1_ANALYSIS_PIPELINE.md` (§4 sign-off style,
interlock), `scripts/asset1_vocab_signature.py` (`CROSS_FAMILY_DESIGN`).
Provenance flag: `docs/ASSET1_BANK_DELIVERY_2026-07-20.md` is cited by the
verify-bundle README but absent from the repo tree; every number below comes
from the v2 sign-off or the campaign record, not the missing file (S12).

---

## 1. The result being followed up

Asset-1 H2 pre-registered "cross-family transfer fails." The triviality
control refuted it: under per-family standardization (verified unsupervised),
task structure transfers between Qwen2.5-1.5B and Llama-3.2-1B at
**spectrum 0.7833 / 0.7375, probe 0.7792 / 0.7792** (chance 0.1667; raw
baselines 0.1667–0.2167; family-identity probe 1.0000 raw → 0.1521/0.0000
standardized; binomial p down to 7.70e-98). Re-derived end-to-end by the
Director. Two open questions follow: **Axis 1 (scale/breadth)** — does
standardized transfer strengthen, plateau, or break as family distance and
parameter scale grow? One pair at 1–1.5B was measured; the paper's own
limitations section names the gap. **Axis 2 (representation)** — does the
transfer survive in an **output-referenced** coordinate system, the
shared-token vocab-signature mapping the Director left DESIGN ONLY pending a
separate H2 decision? Both existing H2 representations are internal-geometry;
the workspace reading predicts the output-referenced one carries the same
structure.

## 2. Design overview — two phases

- **Phase V (CPU, no new training):** vocab_signature-shared as a THIRD H2
  representation on the existing 480-run bank (anchor pair Qwen2.5-1.5B ↔
  Llama-3.2-1B). Honest disclosure: the anchor pair's spectrum/probe results
  are already unblinded; the H2-V prediction pair (§5) is written knowing
  them. What is pre-registered is the vocab-signature number, which does not
  exist anywhere yet — nothing is computed until the Director locks this
  document.
- **Phase S (GPU campaign):** a new adapter bank over added families (§3),
  analyzed with the identical pinned H2 machinery on completion.

Scope: **H2 only**. No new H1 claims (per-family H1 LOO is reported only as
the margin-term ingredient), no D2/D3 extension; D-aux rides along free on
metrics.json as descriptive-only replication — include or strike (S11).

## 3. Axis 1 — family set, run counts, cost (ALL cost figures are ESTIMATES)

Training recipe identical to Asset-1: 6 tasks × N reps, 2000 optimizer steps,
effective batch 16, 512 tokens, val_seed=777 splits, same LoRA/bridge config
(sigma_slots = rank = 24), module counts discovered from adapters, instruct
checkpoints. Cost model: Asset-1 measured **~42 min/run at bs4×ga4** across
the 1.54B/1.24B anchors (delivery report; consistent with ~78 min/run at
bs2×ga8 × ~1.9 throughput gain, 480 runs, Jul 3 21:52Z → Jul 20 16:21Z).
Scaling declared **linear in parameter count** (~30 min/B at this fixed
recipe) — calibrated by 3 timing-only pilots per family (S2) before
commitment; param counts are public-card approximations.

| Family (instruct) | ~Params | Reps | Runs | Est min/run | Est GPU-days | Role |
|---|---|---|---|---|---|---|
| Gemma-2-2B | 2.6B | 20 | 120 | ~79 | ~6.6 | third lineage near anchor scale |
| Qwen2.5-3B | 3.1B | 20 | 120 | ~93 | ~7.8 | within-lineage scale step |
| Qwen2.5-7B | 7.6B | 10 | 60 | ~230 | ~9.6 | within-lineage scale endpoint |
| Llama-3.1-8B | 8.0B | 10 | 60 | ~243 | ~10.1 | cross-lineage at 7–8B |
| **Total** | | | **360** | | **~34** | ~5 weeks with pilot/restart margin |

Considered and excluded: **Gemma-2-9B** (~9.2B; costliest per run, lineage
covered at 2B), **Mistral-7B** (~7.2B; third new lineage at 7B, +~10 est
GPU-days; ALTERNATE if Llama-3.1-8B is license-blocked — G1 reduced-bank path
applies). Drop options: Llama-3.1-8B (−10.1 d, loses cross-lineage-at-scale)
or Qwen2.5-7B (−9.6 d, loses the scale endpoint). VRAM: all four fit 48 GB in
bf16 (est. weights 5–16 GB + activations); 7–8B families may need bs2×ga8 —
any geometry change requires the A1 bit-equivalence conditions re-verified
per family (no batch-norm; clip at accumulation boundary; LR counts optimizer
steps) and a `batch_geometry` cohort tag (S3).

Power at reduced reps: transfer INTO a 60-run family tests n=60 (exact
binomial vs chance 1/6: α=0.01 one-sided needs ≥ ~0.30 observed; at the
Asset-1 band power ≈ 1). Margin precision at n=60: SE ≈ 5.7 pp against the
15 pp margin — readable but noisier; per-direction n (60–240) reported with
every test.

## 4. Axis 2 — vocab_signature-shared as the THIRD H2 representation

Per `CROSS_FAMILY_DESIGN` (`asset1_vocab_signature.py`, NOT BUILT): intersect
literal token strings across the two tokenizers (identical byte-level
normalization), sort lexicographically, restrict each family's W_eff to
shared-token rows in that order, draw the vocab sketch on the shared axis
([seed, 72, |T_shared|, sketch_dim]) so sketch and top-k blocks live in one
output-referenced coordinate system. Residual family scale remains — the
pinned shift control (per-family z-scoring) still applies. |T_shared| per
pair is computed and reported, never assumed. The A3 kv_mode condition
carries over: zero_pad primary, exclude secondary, approximation surfaced in
every artifact; Level B (J-lens) remains the arbiter for any output-null
reading. Role: **third H2 representation, corroborating** — spectrum stays
PRIMARY, probe corroborating, per pinned H2a; three-way disagreement is
itself reportable (S7d offers elevating H2-V to co-primary on the anchor
pair, where it is the entire point of Phase V). The design's three open
sub-decisions are S7a–c.

## 5. Hypotheses — prediction pairs, informative under either outcome

The Asset-1 lesson is written into the format: the boldest prediction was
wrong, and the control that caught it was the pre-registration working. Each
hypothesis states both directional outcomes; neither is privileged.

- **H2-S (scale, within-lineage).** Qwen2.5-1.5B ↔ 3B ↔ 7B, standardized.
  (A) Transfer holds at or above the Asset-1 band (~0.74–0.78) across all
  scale pairs → standardized task geometry is scale-stable within a lineage.
  (B) Transfer decays with scale ratio, the pinned transfer-fails rule firing
  at 1.5B↔7B → standardization removes family location but not
  scale-dependent shape; the universality claim is scale-local.
- **H2-D (lineage distance, matched scale).** Gemma-2-2B ↔ {Qwen2.5-1.5B,
  Llama-3.2-1B}; Qwen2.5-7B ↔ Llama-3.1-8B. (A) Transfer at the band across
  all primary lineage pairs → universality across three-plus lineages.
  (B) Transfer breaks on Gemma pairs while Qwen↔Llama replicates → the
  Asset-1 result is lineage-pair-specific (shared recipe ancestry), demoted
  honestly.
- **H2-V (output-referenced carrier).** Anchor pair, vocab_signature-shared,
  standardized. (A) Transfer at or above the pair's spectrum result
  (0.7375/0.7833) → task structure is output-referenced; the workspace
  reading gains its cross-family leg. (B) Vocab-signature fails while
  spectrum stands → transferable structure lives in internal geometry and
  its vocabulary expression is family-specific — the A3 question answered in
  the negative, equally reportable.

## 6. Decision rules — pinned machinery reused; changes flagged

Unchanged (per DIRECTOR_DECISIONS §6/H2, baked into the D1 tool): PRIMARY =
depth-binned SV spectra; probe corroborating; **shift-controlled headline,
raw descriptive**; per directed pair, exact binomial one-sided **α = 0.01**
vs chance 1/6 plus the **≥ 15 pp** within-minus-cross margin; per-pair
verdict requires BOTH directions (the pinned AND); constants recorded in
output. No constant changes proposed. Flagged deviations requiring ruling:
(i) Holm multiplicity across the primary pair set (S5); (ii) per-direction n
varies 60–240, reported per test (S1); (iii) family-identity probe
generalizes per-pair (S6); (iv) vocab_signature-shared as arm #3 (S7).

## 7. Controls

Per family pair: **family-identity probe** on raw (expected ≈ 1.0) and on the
standardized representation (expected collapse toward its 0.5 chance) — the
control that caught the Asset-1 covariate-shift artifact, now mandatory per
pair; `familywise_standardize` reused verbatim (Director-verified
unsupervised). **Shift-controlled variant is the headline** for every pair
and representation; raw descriptive. Heterogeneity guard per family at the
Asset-1 trigger. For H2-V the same probe runs in the shared-token space, plus
dual kv_mode reporting per A3. An F-way identity probe over all families is
descriptive color only.

## 8. Multiplicity

Primary confirmatory set = **four undirected pairs** (8 directed tests):
P1 Qwen1.5B↔Qwen3B, P2 Qwen3B↔Qwen7B, P3 Qwen7B↔Llama-3.1-8B,
P4 Gemma-2-2B↔Llama-3.2-1B; plus H2-V on the anchor pair as its own single
pre-registered test. Holm–Bonferroni at family-wise α = 0.01 across the
primary directed tests; every other pair (the full 30-directed-pair grid is
computed — nearly free on extracted features) is **descriptive**, plotted as
the transfer-vs-(scale gap, lineage distance) surface with no per-pair
claims. Alternative for ruling: unadjusted α = 0.01 per pair, all pairs
confirmatory — rejected in this draft (30 tests at 0.01 ≈ 26% family-wise
false-positive exposure), but the choice is S5.

## 9. Interlock and bank design

New bank root `results/asset1h2-bank/` (never inside `asset1-bank/`), same
manifest-as-source-of-truth design, `require_complete_bank` reused with
`expected_total` = the S1-ruled count. Phase V has its own gate: **no
cross-family vocab-signature number is computed until this document is
locked** (the builder refuses without a `--prereg-locked` acknowledgement
naming the registered doc). Interlock options for ruling (S9): **T1** single
gate, all analysis waits for the full ~5-week bank (Asset-1 precedent);
**T2** per-family-tier gates with staged unblinding — all hypotheses locked
at registration for all tiers, each tier analyzed only when its families are
COMPLETE, any post-unblinding change a dated amendment restricted to
not-yet-unblinded tiers (L-006). Draft recommends T2, tier order = table
order (cheapest first). `--allow-partial-bank` semantics, loud warning, and
the no-quiet-middle-ground rule carry over verbatim (G1).

## 10. Director sign-off items (nothing proceeds until each is ruled)

- **S1 — Family set + reps.** The §3 table (360 runs, ~34 est GPU-days), a
  drop option, or substitutions; per-direction n disclosure rule included.
- **S2 — Timing pilots.** 3 timing-only runs per family, excluded from the
  bank; confirm timing-only runs are not unblinding.
- **S3 — Batch geometry.** Per-family geometry; A1 bit-equivalence conditions
  re-verified per family; `batch_geometry` cohort tags mandatory.
- **S4 — Decision rule reuse.** Confirm α=0.01 / ≥15pp / both-directions /
  shift-controlled-headline per pair with NO constant changes.
- **S5 — Multiplicity.** Primary set P1–P4 + Holm at FW α=0.01; all other
  pairs descriptive. Or the flagged alternative.
- **S6 — Family-identity probe extension.** Per-pair binary probe as the
  gating control; F-way probe descriptive-only.
- **S7 — Vocab-signature cross-family mapping** (the A3 deferred decision):
  (a) token-string intersection as THE alignment; (b) top-k over shared ids
  only, or full-vocab top-k with shared-axis sketch only; (c) treatment of
  string-equal tokens with differing merge contexts (draft: include, count,
  report the fraction); (d) role — arm #3 corroborating everywhere, or
  co-primary for H2-V on the anchor pair.
- **S8 — H2-V triviality control.** Family-identity probe + per-family
  z-scoring in shared-token space; shift-controlled headline; dual kv_mode
  per A3. Confirm.
- **S9 — Interlock topology.** T1 single-gate vs T2 tiered (draft: T2).
- **S10 — Hypothesis lock.** Lock H2-S / H2-D / H2-V as written in §5 or as
  amended. Until ruled, nothing here is registered.
- **S11 — Scope.** Confirm H2-only; D-aux descriptive replication in or out.
- **S12 — Provenance repair.** Restore/locate the missing delivery report;
  confirm the ~42 min/run anchor against it before S1 cost figures are
  treated as calibrated.

---

*Draft by Meridian, 2026-07-21, following the Asset-1 §4 sign-off-section
convention. Every choice above is a proposal; the Director's rulings bind and
deviations after lock are dated amendments, never silent revisions (L-006).*
