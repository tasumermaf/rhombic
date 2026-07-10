# BM-004 Pre-Registration v2 — rd_graph Structural Mask × Paired-Transit Data

> **Date:** 2026-07-07
> **Status:** PROPOSED — awaiting Director sign-off. No GPU run occurs before
> (a) the asset-1 bank completes (~Jul 20), (b) this document is approved as a
> dated pre-registration amendment, and (c) the positive-control gate (F2)
> passes. Amendments after approval are dated edits, not silent revisions
> (LEARNINGS L-006).
> **Status update (dated annotation, 2026-07-07):** APPROVED as
> pre-registered the same day —
> `docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md` (A5), with two
> conditions, both encoded (revision log, end of document). Gates (a) and
> (c) above remain in force; the original Status line is preserved
> unmodified per L-006.
> **Supersedes:** the BM-004 section of `docs/BM_BATTERY_PLAN.md` (2026-07-03),
> which remains the design-intent record; its three hard fixes are carried
> forward verbatim as §8.
> **Lineage:** `docs/MERIDIAN_RESEARCH_POSITION_2026-07-02.md` §3.2 (BM-004 as
> the never-run fair test of the original conception);
> `docs/GLOBAL_WORKSPACE_MAPPING_2026-07-07.md` §3 item 3 (the upgrade package
> this v2 pre-registers); `docs/LITERATURE_WATCH_2026-07-03.md` §C (2606.01090
> concurrency; "run BM-004 soon"); `results/BM-003/PROTOCOL.md` (architecture).
> **Data/tooling:** `scripts/bm004_transit_data.py` (+ 29 property tests in
> `tests/test_bm004_transit_data.py`). Pure polytope geometry, public math —
> no corpus-derived values anywhere in data, endpoints, or analysis
> (Stream A/B discipline; IP boundary holds by construction).

**Workspace-paper import marking.** Every element imported from the Anthropic
global-workspace paper (transformer-circuits.pub/2026/workspace, as digested in
`docs/GLOBAL_WORKSPACE_MAPPING_2026-07-07.md` §1) carries a source number:

| Import | Source figure/finding |
|--------|----------------------|
| **[WS-1]** | Coordinate-swap causal concentration: **88% inside J-space vs 5%** in the orthogonal complement |
| **[WS-2]** | Two-hop swap transfer **54–70%**; cross-function swaps **76/192 (α=1) → 101/192 (α=2)** |
| **[WS-3]** | Broadcast hub: one concept serves multiple functions ({capital-of, speaks, located-in}) with flexible re-routing — reusable relation, not per-example lookup |
| **[WS-4]** | Counterfactual reflection training: articulate-if-interrupted training implants representations that improve uninterrupted behavior; **ablating the implanted representations reverts the improvement** |
| **[WS-5]** | Selective recruitment: a count representation present **only when needed** |

The anti-claim discipline of the mapping doc §4 is binding: no "workspace,"
"J-space," or access-consciousness language in any public artifact from this
experiment. Neutral vocabulary throughout: *swap transportability*,
*relational reuse*, *articulation arm*, *selective recruitment*.

---

## 1. Design

**Question.** Does a structural geometric prior (the rd_graph fixed-adjacency
mask) help exactly when the training data carries the relation the geometry
encodes — and is the learned relation *reusable* rather than memorized?
The interaction term is the entire question (BM_BATTERY_PLAN). v2 adds: even
if the interaction is positive, is the relation LEARNED-AS-REUSABLE (§4)?

**Architecture.** Exactly BM-003's controller-free configuration
(`results/BM-003/PROTOCOL.md`): fixed RD adjacency mask × learnable edge
weights (`rd_adjacency_mask()` in `rhombic/nn/topology.py`; 6 channels,
co-planar 1.0 / cross-planar 0.5), learnable A/B, **LM loss only, no
Steersman, no auxiliary losses**. Control parameterization: frozen-identity
bridge (exact standard LoRA).

**Scale and budget (pinned).**

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Base model | Qwen2.5-1.5B (primary); TinyLlama-1.1B fallback if VRAM contention | mapping doc §3 item 3 ("~4 runs × 4–6h at 1.5B") — **invented default: which 1.5B** |
| Steps | 4,000 optimizer steps | **invented default** (2× the asset-1 bank's 2,000; transit corpus is smaller-vocabulary) |
| Rank / channels / seed | 24 / 6 / 42 | BM-003 protocol |
| LR / batch geometry | 2e-4 / bs4×ga4 (effective 16) | BM-003 protocol + Director decision A1 (2026-07-06) |
| Target modules | q,k,v,o_proj, all layers | BM-003 protocol |
| Runs | **6 unconditional** (4 core + E4 arm + 6a; 6b = conditional 7th run added by dated amendment iff the E1 interaction is non-null, §6) | **invented default** |

**Core 2×2 (4 runs):**

| | Paired-transit data | Generic data (alpaca-class) |
|---|---|---|
| **rd_graph mask** | Run 1 — the thesis cell | Run 2 — mask-without-data control |
| **Identity bridge (std LoRA)** | Run 3 — data-without-mask control | Run 4 — double control |

**Timing.** Data build + eval harness: now (CPU). Positive-control gate (F2):
first GPU slot post-bank. Core runs: post-~Jul 20. No run before Director
approval of this document.

## 2. Data

Built by `scripts/bm004_transit_data.py` (grammar `bm004-v2`, seed 20260707;
all builder defaults recorded in `BM004_TRANSIT_DATA_MANIFEST.json` — runs
self-document). Geometry: FCC cells (integer triples, even coordinate sum),
12 face-neighbors at the (±1,±1,0)-type offsets, face labels F00–F11 mapped
to the mask's 6 channels by the documented antipodal-pair table
(`face_channel_map`).

**Production sizes (pinned; builder defaults are smaller on purpose):**
n_walks = n_pairs = **40,000 per training arm** — i.e. 80,000 walk+pair
records per arm ("40,000 sequences per arm" in the earlier draft meant *per
record kind*; pinned here to remove the ambiguity); walk_len 24; box_radius
8. Articulation admixture (E4 arm only): **articulation_fraction = 0.075 of
PAIRED sequences** — the denominator the mapping doc states its band in
(§3 item 3: "~5–10% of paired sequences"; 0.075 is the midpoint) and the
denominator the builder implements (n_articulation = round(f × n_pairs) =
3,000 records). At the pinned 1:1 pairs:walks mixture the E4 arm is
therefore 40,000 + 40,000 + 3,000 = 83,000 records, of which articulation
is **3.61% of total arm sequences**. (**Invented default** — value and
denominator both flagged for sign-off. The rejected alternative, 7.5% of
*total* arm records, would put articulation at 16.2% of paired sequences,
outside the memo band; verified by `test_production_mixture_arithmetic` and
`test_articulation_denominator_is_paired_sequences`. This paragraph is a
2026-07-07 review fix: the earlier draft's "92.5% walks+pairs / 7.5%
articulation records" phrasing implied the total-records denominator, which
neither the memo nor the builder uses.) Pairs:walks ratio 1:1 (**invented
default**). Held-out eval: 2,000 fresh walks per condition (disjoint
seeds), plus the cross-domain sets of §3-E2.

**The pair signal.** A paired sample is the same walk in two frames —
`PAIR EGO START x y z F.. F.. END => ABS CELL ... END`. Translating EGO→ABS
requires applying transit(cell, face) → (cell′, entry_face) step by step:
the frame mapping *is* the relation, nothing else (verified by property
test `test_pair_frame_consistency`). This is the paired control-data
pattern from production video-LoRA practice, as specified in the March 8
DETERMINED note and never previously executed.

**Shuffled-adjacency twin (the data negative control).** Per-cell scrambled
face→offset assignment: preserves the neighbor set, degree 12, symmetry,
token frequencies, and walk-length statistics (property-tested) while
destroying translation invariance of the face→displacement rule — the
relation exists only as per-cell lookup. A global label permutation was
considered and **rejected** as gauge-equivalent for a token-embedding LM
(design decision D2 in the builder; **invented default flagged for
sign-off**).

## 3. Endpoints

**E1 (primary) — transit completion.** Held-out next-cell prediction:
given `... CELL x y z F##`, score the model's continuation `CELL x′ y′ z′`.
Metric: exact-match accuracy (chance = 1/12 against the geometric neighbor
set) and per-token LM loss on held-out walks. The pre-registered quantity is
the **interaction term**: [E1(rd_graph|paired) − E1(identity|paired)] −
[E1(rd_graph|generic) − E1(identity|generic)].

**E2 (NEW) — cross-domain relational transfer.** Does the learned relation
apply beyond its training surface [WS-3]? Three held-out generalization sets,
built by the same generator with disjoint seeds:
 (a) **fresh regions** — start cells drawn from a disjoint coordinate shell
     (box_radius 33–41, beyond the maximum training-reachable radius
     8 + 24 = 32, guaranteeing zero cell contamination — training walks
     start in box 8 and can drift at most walk_len = 24 beyond it);
 (b) **renamed entities** — an unseen coordinate-token surface (offset all
     coordinates by a constant vector, producing cell tokens never seen in
     training, same relation);
 (c) **held-out compositions** — two-hop queries: `FROM CELL c EXIT F## THEN
     F## ARRIVE ?` (never trained as a single-step pattern) [WS-2 analog].
Metric: E1-style accuracy per set. Prediction: matched-arm rd_graph transfers;
shuffled-data arms collapse toward chance (lookup cannot transfer).

**E3 (NEW) — swap-transportability success criterion** (imported from
[WS-1]/[WS-2]; the analog of the 88%-vs-5% dissociation at our scale).
Definition and protocol in §4.

**E4 (NEW) — articulation-if-interrupted arm** [WS-4]. One additional
data-variant condition per §5, with probe-and-ablate reversal as the causal
check.

**E5 — selective-recruitment analysis** [WS-5]. For rd_graph runs: per-token
norm of the bridge-path contribution (h→A→bridge→B minus the same with
bridge≡I, measured at injection sites) on (a) transit-token positions vs
(b) matched-length generic text. Recruitment ratio = mean(a)/mean(b).
Pre-registered reading: ratio ≥ 5 in the thesis cell = selective recruitment
(**threshold is an invented default**); ratio ≈ 1 in the mask-without-data
cell = the channel idles when the relation is absent. Descriptive companion:
edge-weight trajectory divergence between the two cells.

## 4. LEARNED-AS-REUSABLE — the swap-transportability criterion (E3)

**Definition (pre-registered).** A trained transit relation counts as
**LEARNED-AS-REUSABLE** if and only if a representation-level swap transports
across contexts *and* across at least two functions at high rate, while
matched non-transit controls do not — the scale-appropriate analog of the
paper's 88%-vs-5% causal concentration [WS-1] and cross-function reuse
[WS-2]/[WS-3].

**Measurement protocol (probe + patching on the adapter-injected model):**

1. **Probe.** Linear probe on the residual stream at face-token positions,
   layers in the middle-third band of the model (Qwen2.5-1.5B: 28 layers →
   L10–L18; per-layer probes, best layer selected on a probe-validation
   split that is disjoint from all patching evaluation prompts —
   **band and selection rule are invented defaults**), trained to decode
   the exit-face identity (12-way). The probe's 12 class directions span
   the candidate relation subspace (≤ 12 dims).
2. **Patch (swap).** In a frozen forward pass, replace the relation-subspace
   component of the activation at the face-token position for face *f* with
   the class-mean component for face *f′* (mean-difference patch, α = 1;
   α = 2 recorded as secondary, mirroring the paper's 76/192 → 101/192
   α-sensitivity [WS-2]).
3. **Functions (≥ 2, pre-registered).**
   F-A: next-cell prediction — success iff the model now predicts
   cell + offset(f′).
   F-B: entry-face naming (`... ARRIVE CELL x′ y′ z′ ENTER ?`) — success iff
   the model now names entry(f′).
4. **Controls (matched).**
   C-1: same-norm patch along random directions of equal dimension in the
   orthogonal complement of the relation subspace.
   C-2: identical patch applied at non-face token positions.
5. **Sample:** ≥ 192 patch trials per function per run (matching the paper's
   cross-function denominator [WS-2]), over ≥ 20 distinct contexts.

**Success criterion (pinned):** transport rate **≥ 70%** in both functions
for the thesis cell, with both controls **≤ 15%** (**thresholds are invented
defaults** — scaled down from 88/5 to respect the scale gap the mapping doc
§2.2 forbids importing numbers across; the *dissociation*, not the exact
figures, is the imported prediction). A relation that passes E1 but fails E3
is declared LEARNED-AS-LOOKUP, not reusable.

**Threshold justification against our own chance and control distribution
(Director condition A5-1, dated edit 2026-07-07 —
`docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md`).** The workspace paper's
88%-vs-5% arises in an open-vocabulary next-token regime where the chance
rate of emitting the specific transported target is ≈ 0. Our two functions
are 12-way closed-form decisions (pinned default #15: chance = 1/12 ≈
8.33%), so both bounds are re-derived against that floor rather than
inherited:

- **Control ceiling ≤ 15% is a null-consistency band.** Under the null
  (the patch carries no directional relation content) a control's success
  count over the pinned ≥ 192 trials is Binomial(192, 1/12): mean 16
  counts (8.33%), SD 3.83 counts (2.0pp). 15% = 29/192 counts = chance
  + 3.4σ, which a chance-level control exceeds with one-sided exact
  binomial p ≈ 1.3 × 10⁻³; the chance-null's 99th percentile is 25/192 =
  13.0%. The ceiling is therefore the tightest round bound that a genuinely
  chance-level control passes with high probability — and a control that
  EXCEEDS it is itself evidence of patch leakage: that outcome invalidates
  the E3 trial set for that run and is reported as a control failure (no
  threshold re-rolling, no criterion rescue).
- **Transport floor ≥ 70% is placed for dissociation magnitude, not
  statistical detectability.** 70% = 8.40× the 1/12 chance and 4.67× the
  control ceiling; an observed rate ≥ 70% is significant beyond any α
  against either reference (exact binomial p = 4.9 × 10⁻⁹⁹ vs chance,
  2.0 × 10⁻⁶⁶ vs a control sitting exactly at the 15% ceiling). The bar's
  content is substantive: the swap must move the model's answer to the
  transported target in a large majority of trials in BOTH functions —
  which is the reusable-mechanism claim itself, not a proxy for it.
- **Why not import 88% directly:** the paper's figure comes from mature
  circuits in a fully trained LM; demanding it of a rank-24 adapter
  relation trained 4,000 steps from scratch would confound "not trained to
  saturation" with "not reusable." Because E3's failure is a *declared
  negative* (LEARNED-AS-LOOKUP, H3), an inflated bar would manufacture
  negatives; 70/15 preserves the imported prediction's structure
  (transport ≫ control, control ≈ chance) at our chance floor while
  keeping failure interpretable.

All justification figures are exact binomial computations at the pinned
n = 192, p₀ = 1/12 (scipy.stats.binom; reproducible one-liners).

## 5. Articulation-if-interrupted arm (E4)

**Data.** The `articulation` corpus kind: walks interrupted at a uniformly
random transition with an `INTERRUPT STATE FROM ... EXIT ... OFFSET ...
ARRIVE ... ENTER ... RESUME` clause stating the local adjacency relation in
pure-geometry vocabulary (counterfactual-reflection template [WS-4]; the
relation is stated abstractly — nothing corpus-derived can appear).

**Runs.** One additional rd_graph × paired-transit run with the
articulation admixture (7.5% of paired sequences = 3.61% of the arm's
records; §2) — Run 5 in §6's table — vs the thesis cell (Run 1,
no articulation).

**Predictions and causal check (pre-registered):**
1. Run 5 ≥ Run 1 on *uninterrupted* E1/E2 (the implanted-representation
   benefit [WS-4]).
2. **Probe-and-ablate reversal:** probe for the articulation representation
   (linear probe distinguishing articulation-trained vs plain model
   activations at face tokens; ablation = projecting out the top-k probe
   directions, k ≤ 4 — **invented default**), then re-run uninterrupted E1.
   The E4 improvement must **revert by ≥ 50%** of the Run5−Run1 gap under
   ablation (**threshold invented default**) while a matched random-subspace
   ablation reverts < 20%. If the improvement exists but does not revert,
   it is not carried by the probed representation — report as such.

## 6. Negative controls — the control 2×2

{mask: rd_graph | **shuffled mask**} × {data: paired-transit | **shuffled-
adjacency data**}. Two cells are already core runs; two are new:

| Run | Mask | Data | Role |
|-----|------|------|------|
| 1 | rd_graph | paired-transit | thesis cell (core) |
| 2 | rd_graph | generic | mask-without-data (core) |
| 3 | identity | paired-transit | data-without-mask (core) |
| 4 | identity | generic | double control (core) |
| 5 | rd_graph | paired-transit + articulation (7.5% of pairs; §2) | E4 arm |
| 6a | **shuffled mask** | paired-transit | wrong-symmetry arm (hard fix F3) |
| 6b | rd_graph | **shuffled-adjacency data** | shuffled-data twin |

Runs 6a/6b are the two control cells; with Run 5 the table configures 7
runs — the pinned budget is **6 unconditional GPU runs** (runs 1–5 plus
6a; F3 is mandatory). **6b is a conditional SEVENTH run**, added by dated
amendment iff the interaction term in E1 is non-null (pre-registered
conditional, declared now, so it is not a data-dependent forking choice:
6b's purpose is to decompose a positive interaction, which does not
exist if E1 nulls — and the budget extension is declared here, not
improvised later).
Shuffled mask = a seeded permutation of the rd adjacency mask's off-diagonal
pattern at equal edge count and weight multiset, rejected against relation
automorphisms (`label_permutation_is_geometric` provides the rejection test
at the label level; the mask-level permutation reuses BM-000's rewire-null
discipline). Every metric reported carries a BM-000 percentile.

## 7. Locked hypotheses (directional; failure meanings declared now)

**H1 (interaction).** E1: rd_graph − identity > 0 on paired-transit data;
≈ 0 on generic data (BM-003 already bounds the generic cost). Test: paired
bootstrap over 2,000 held-out items, 10,000 resamples, one-sided α = 0.01
(house rule per Director sign-offs 2026-07-06). *If null:* the original
conception — geometry in weight space paying off on geometry-matched data —
is **closed honestly and finally**, by the right experiment rather than a
proxy (research-position memo §3.2). The result publishes either way inside
the BM battery framing.

**H2 (relational transfer).** E2(a,b): matched thesis cell above chance at
α = 0.01 and within 15 points of in-domain accuracy; 6b (shuffled data)
at/near chance on (a,b). E2(c) two-hop: thesis cell > identity cell.
*If H2 fails while H1 holds:* the prior speeds fitting but the relation is
surface-bound — report as "prior-as-optimizer, not prior-as-representation."

**H3 (swap transportability).** Thesis cell meets the LEARNED-AS-REUSABLE
criterion (§4); Run 3 (data-without-mask) may pass E1 but is predicted to
show *lower* transport; 6b fails the criterion outright. *If H3 fails
everywhere while E1/E2 pass:* relational reuse at 1.5B adapter scale is not
measurable by linear probe + patching — the WS-1 dissociation does not
descend to our scale; report as a boundary result, do not iterate probes
post-hoc (that would be an unregistered fishing expedition).

**H4 (articulation).** Run 5 > Run 1 on uninterrupted E1/E2, and the gain
reverts under targeted ablation (§5) but not random ablation. *If the gain
is absent:* the counterfactual-reflection template does not transfer from
content-installation to geometry-installation — the Pillar 4 question
(mapping doc §2.4) answered in the negative for weight-space priors; a
publishable null.

**H5 (selective recruitment).** Bridge-path recruitment ratio ≥ 5 in the
thesis cell; ≈ 1 in mask-without-data. *If recruitment is uniform:* the
mask acts as static reparameterization, not a recruited channel — sharpens
Paper 4's limitations honestly (mapping doc §4, claimable-ground lane 3).

**Global failure meaning.** If H1–H3 all null, BM-004 closes Track 1's
form-first conjecture for adapter weight space; surviving lanes are exactly
those the mapping doc §5 names — substrates with real locality costs.

## 8. The three hard fixes (carried verbatim from BM_BATTERY_PLAN, operationalized)

**F1 — Operational pre-specification.** The rd_graph mask encodes the RD
face-pair coupling: 6 channels = 6 antipodal face pairs; co-planar coupling
1.0, cross-planar 0.5 (`rhombic/nn/topology.py`). A dataset is **invariant
under the mask's structure** iff its generating process is equivariant under
the **joint action** of the offsets' relation-matrix automorphism group:
formally, for every label permutation π with R[π,π] = R (R = the 12×12
offset Gram matrix; exactly the 48 permutations faithfully induced by the
octahedral symmetries of the offset set — verified by
`test_all_48_octahedral_symmetries_induce_geometric_permutations`), the
corpus distribution is invariant under *simultaneously* relabeling all face
tokens by π **and** transforming all cell coordinates (start cells included)
by the induced signed-permutation matrix g_π — the unique orthogonal map
with g_π·OFFSETS[f] = OFFSETS[π(f)], constructed and verified by
`induced_cell_map` (which rejects non-automorphisms).

*(2026-07-07 review fix.)* The earlier draft stated the criterion as
"applying π to all face tokens maps the corpus distribution to itself" —
that token-only statement is formally wrong in both directions: a
non-identity π applied to face tokens alone maps ABS/PAIR records **off the
support** (the walk arithmetic no longer holds; verified for all 47
non-identity automorphisms by
`test_face_relabeling_alone_breaks_every_nonidentity_automorphism`), while
EGO face sequences are i.i.d.-uniform and therefore invariant under *any*
of the 12! permutations, geometric or not — so the token-only statement
discriminates nothing. The joint statement is the one with content, and the
paired-transit generator satisfies it **by construction**: transit steps
are equivariant (g_π(c + OFFSETS[f]) = g_π·c + OFFSETS[π(f)]), face
sampling is uniform over the full orbit, and the start-cell box and lattice
parity are g_π-invariant (`test_joint_action_preserves_walk_validity`,
`test_joint_action_preserves_start_box_and_parity`). Which π qualify
remains checkable via `label_permutation_is_geometric` +
`test_geometric_permutation_criterion`. "Not matched enough" is therefore
not available as a post-hoc escape: the criterion is a property of the
generator, verified before any run.

**F2 — Positive-control manipulation check.** Before any natural-data or
full-scale arm: one short run pair (rd_graph vs identity, ≤ 1,000 steps,
TinyLlama-1.1B, transit corpus only) must show rd_graph ≥ identity on E1
held-out loss. The transit corpus is provably invariant under the mask's
structure (F1), so **if the prior cannot help here, the mechanism is broken
and no other result is interpretable** — halt and debug before spending the
budget. Gate outcome is recorded either way.

*Wired as a hard interlock (Director condition A5-2, dated edit
2026-07-07):* `scripts/bm004_runner.py` is the single authorized launch
path for every BM-004 arm; `require_f2_gate()` runs as a PRECONDITION of
every launch (first statement on the launch path, before any command is
built or any process started), refusing with SystemExit unless
`results/BM-004/F2-gate/gate_result.json` records a PASS under the pinned
criteria (rd_graph held-out E1 loss ≤ identity; ≤ 1,000 steps;
TinyLlama-1.1B; transit corpus only). No bypass flag exists. A FAIL is
recorded and launching stays blocked — halt and debug, per the paragraph
above. Run 6b additionally requires an explicit dated-amendment path
(pre-registered conditional, §6).

**F3 — Wrong-symmetry arm.** Run 6a (shuffled mask × matched data) is
mandatory and unconditional. Methodology per arXiv:2606.01090 (Symmetry–Data
Exchange Rate): misaligned priors are *actively harmful*, so the
pre-registered prediction for 6a is **E1 ≤ identity-bridge level** (not
merely ≤ rd_graph). 2606.01090 is cited as concurrent work; differentiation:
adapters over a pretrained LM, graph-adjacency prior vs group equivariance
(LITERATURE_WATCH §C).

## 9. Analysis discipline

- All statistics computed by scripts checked into `scripts/`, validated on
  synthetic fixtures before any real run exists (this repo's standing
  pre-registration hygiene).
- Exact binomial / bootstrap tests at one-sided α = 0.01; Wilson 95% CIs
  descriptive (Director house rules, 2026-07-06).
- Every topology/recruitment metric reports a BM-000 null percentile.
- No peeking: E1–E5 computed once, after all runs in a stage complete.
- Deviations from this document: dated amendments only (L-006).
- **No contact with the asset-1 bank or the archived bs2×8 cohort**: BM-004
  uses fresh runs and synthetic-geometry data exclusively.

## 10. Pinned defaults requiring Director sign-off

| # | Default | Value | Rationale |
|---|---------|-------|-----------|
| 1 | Base model | Qwen2.5-1.5B (TinyLlama fallback) | memo said "TinyLlama-or-1.5B"; picking the stronger default |
| 2 | Steps / LR / batch | 4,000 / 2e-4 / bs4×ga4 | BM-003 + A1; step count doubled for from-scratch relation |
| 3 | Run budget | 6 GPU runs (§6 conditional rule for 6b) | mapping doc "~4 runs" + mandatory F3 + E4 arm |
| 4 | Corpus sizes | n_walks = n_pairs = 40,000/arm (80,000 records; 83,000 in the E4 arm); walk_len 24; box_radius 8 | ~½ asset-1 pool scale; token budget fits 4k steps |
| 5 | Articulation fraction | 0.075 **of paired sequences** (= 3,000 records = 3.61% of E4-arm total) | midpoint of memo's 5–10%-of-paired-sequences band; denominator pinned to the memo's (§2) |
| 6 | Pairs:walks mixture | 1:1 | simplest pre-specifiable |
| 7 | Shuffled twin construction | per-cell face→offset scramble | global permutation is gauge-equivalent (builder D2) |
| 8 | Text grammar | bm004-v2 (builder D3) | exact grammar pinned before runs |
| 9 | Walk policy | uniform faces, backtracking allowed | any shaping = unregistered forking point (builder D4) |
| 10 | E3 thresholds | ≥ 70% transport / ≤ 15% controls; ≥ 192 trials × 2 functions | justified against our 1/12 chance + control null in §4 (Director condition A5-1, 2026-07-07) |
| 11 | E3 probe protocol | linear probe, middle-third layers, α=1 patch (α=2 secondary) | WS import at our scale |
| 12 | E4 ablation | top-k ≤ 4 probe directions; ≥ 50% reversion vs < 20% random | WS-4 reversal check |
| 13 | E5 threshold | recruitment ratio ≥ 5 | order-of-magnitude selectivity |
| 14 | Builder seed | 20260707 | date-derived, fixed |
| 15 | Chance level for E1/E2 | 1/12 (uniform over geometric neighbors) | grammar-aware chance |

---

## 11. Revision log (dated edits, L-006; all pre-approval)

**2026-07-07 (review fixes, pre-approval — document was and remains
PROPOSED):**
1. **§2 mixture denominator pinned.** The draft's "92.5% walks+pairs / 7.5%
   articulation records" implied a total-records denominator; the mapping
   memo's band and the builder both use the paired-sequences denominator.
   Pinned: articulation_fraction = 0.075 of paired sequences
   (n_articulation = round(f × n_pairs) = 3,000 = 3.61% of the 83,000-record
   E4 arm; inside the memo band, where the total-records reading is not).
   Builder changed from round(f × n_walks) to round(f × n_pairs) so the
   semantics survive any future change to the 1:1 mixture; manifest now
   records `mixture_accounting` (denominator + both shares).
2. **§2 arm-size ambiguity resolved.** "40,000 sequences per arm" pinned as
   n_walks = n_pairs = 40,000 (80,000 records/arm; 83,000 in the E4 arm).
3. **§8 F1 restated as joint equivariance.** The token-only invariance
   statement was formally wrong (maps ABS/PAIR off the support; vacuous on
   EGO); replaced with the joint (π on face tokens, g_π on cell coordinates)
   statement, with `induced_cell_map` and four new property tests as the
   operational check. `label_permutation_is_geometric` is unchanged and
   remains the test for which π qualify.
4. Property-test count 22 → 29.

**2026-07-07 (Director conditions, post-approval — APPROVED as
pre-registered per `docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md`, A5):**
1. **§4 E3 threshold justification added** (condition A5-1): the imported
   ≥70%/≤15% bar is now derived against our own chance level (1/12) and
   control null (Binomial(192, 1/12)) with exact binomial figures — no
   longer merely inherited from the workspace paper's regime.
2. **§8 F2 wired as a hard interlock** (condition A5-2):
   `scripts/bm004_runner.py` is the single authorized launch path for
   BM-004 arms; its launch path calls `require_f2_gate()` BEFORE any
   full-scale arm and refuses to proceed unless a PASSED gate artifact
   (`results/BM-004/F2-gate/gate_result.json`) exists and matches the
   pinned gate criteria. There is NO bypass flag. The gate outcome is
   recorded either way (a FAIL is written and launching stays blocked).
   Enforced by `tests/test_bm004_runner.py`.
3. Disclosure: the first draft of this dated entry inadvertently
   replaced the document's original closing attestation; it is restored
   below, unmodified (caught by the fresh-context verification pass,
   2026-07-07).

**2026-07-10 (trainer wiring landed, pre-GPU — no run occurred):**
1. The two trainer capabilities the runner refuses to launch without
   (`scripts/bm004_runner.py`, honest fail-fast) are now implemented in
   `scripts/train_cybernetic.py`, ahead of the GPU phase and by dated
   edit as anticipated:
   - **`--transit-corpus <dir>` + `--transit-arm {matched,
     matched+articulation,shuffled}`** — loads the paired-transit corpus
     (`scripts/bm004_transit_data.py` output) for the chosen arm (walks +
     pairs, plus articulation records for the E4 arm), mutually exclusive
     with `--dataset`; generic-dataset behavior is byte-identical when the
     flag is absent. Corpus→text is a pure, tokenizer-free stage
     (`load_transit_texts`), tokenized with the trainer's existing
     conventions (`TransitCorpusDataset`).
   - **`--bridge-mode shuffled_rd`** — the §6 wrong-symmetry twin (hard fix
     F3): a seeded relabeling of the RD adjacency mask's off-diagonal
     pattern at equal edge count and weight multiset, rejected against
     relation automorphisms (mask-level mirror of
     `bm004_transit_data.label_permutation_is_geometric`), seeded from
     `--seed`. Same fixed-mask × learnable-edge-weights mechanism as
     `rd_graph` (`rhombic/nn/topology.shuffled_rd_adjacency_mask`,
     consumed by `RhombiLoRALinear`).
2. `trainer_supports('transit_corpus')` and `trainer_supports('shuffled_rd')`
   in the runner now both report True; the runner's refuse-don't-improvise
   guards therefore pass naturally, and its interlock test's
   until-wired refusal case skips as designed. Tests:
   `tests/test_bm004_trainer_wiring.py` (CPU-only, no model).
3. **Runs remain gated.** This is code + tests only. No GPU run occurred;
   every launch is still blocked by the F2 positive-control interlock
   (§8, condition A5-2) and by bank completion (Status gate (a)). Landing
   the wiring early does not open any gate — it only removes the
   "capability not yet implemented" refusal so the F2 gate is the sole
   remaining launch precondition.

---

*Pre-registered July 7, 2026 by Meridian (Lane E-3). Data builder and 29
property tests land with this document; no GPU work, no bank contact, no
corpus values. The interaction term is the entire question — and this time,
whether the answer is reusable is part of the question.*
