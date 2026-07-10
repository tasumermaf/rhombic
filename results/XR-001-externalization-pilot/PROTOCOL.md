# XR-001 — Externalization-Robustness Pilot (Protocol, pre-registered)

> **Date:** 2026-07-10 (protocol written and committed BEFORE any run)
> **Status:** PRE-REGISTERED — predictions below are frozen before data collection
> **Hardware:** Hermes (RTX 4090 Laptop 16GB), local Ollama models, $0 marginal cost
> **Lane:** Operational deliverable (agent-memory engineering). Not part of the
> BM battery or the asset-1 analysis chain; no Director gate applies. Feeds the
> typed-adjacency compaction prototype (Face-11 / plan item I5), which must beat
> the baseline table this pilot produces.

## 1. Question

When an agent's context is compacted between steps of a multi-step numeric
task, does the **format** of the compacted state affect numeric fidelity?
Three memory regimes are compared at matched token budgets:

- **R0 — no-compaction**: the full running transcript is carried forward
  (ceiling condition).
- **R1 — prose summary**: at each checkpoint the model replaces its transcript
  with a flowing prose summary (budget-capped).
- **R2 — structured typed blocks**: at each checkpoint the model replaces its
  transcript with a typed state block (`values:` / `combined:` / `relations:`
  fields), same budget cap.

Compaction **cascades**: checkpoint k's state is produced from
[state k−1 + segment k], never from full history — the iterated lossy
re-encode is the phenomenon under test.

**Motivation.** (a) A documented incident in our own research workflow
(internal ops log, Feb 2026): an orchestrating agent's prose summary reported
a computed value off by one and conflated three related quantities — a base
value, a neighboring value one unit away, and a combined sum — into a single
incorrect claim, while every computed artifact on disk was correct. The
codified fix (verbatim-quotable structured results blocks) has never been
measured against the prose alternative at matched budgets. (b) External
support: Anthropic's global-workspace interpretability report (2026) finds
chain-of-thought externalization substantially more robust to internal-state
ablation than direct answers — externalized tokens survive what internal
carry loses. This pilot tests the agent-scaffolding analog. No claim about
model internals is made or implied here; this is an engineering measurement
on agent memory formats.

## 2. Task construction (ground truth by construction)

Each **episode** is a synthetic analytical session, generated from a seeded
RNG (episode seed = base_seed + episode_index; base_seed = 77001):

- **4 segments**, each introducing **3 probed entities** and **2 distractor
  entities**. Entities are pseudoword names (CV-pattern, 6 letters, globally
  unique within episode). Each entity has an integer value in [101, 987];
  all values globally distinct within an episode except designed collisions.
- Per segment, the designed structure mirrors the incident class:
  - base entity A = v; partner entity B = w;
  - the **combined value** v+w stated explicitly in the segment text;
  - a **neighbor entity** C with value v+1 (the conflation trap);
  - 2 distractors with unrelated values (never probed — budget pressure).
- Segment text embeds the facts in ~120–180 words of narrative filler so that
  4 segments strictly exceed any single compaction budget.

**8 questions per episode**, asked one per call after the final checkpoint,
all with integer answers, ground truth known by construction:

| # | Type | Probes |
|---|------|--------|
| 1–3 | direct recall | one base/partner value from segments 1, 2, 3 |
| 4–5 | conflation probe | neighbor entities (truth = base+1) from two segments |
| 6 | combined recall | a stated sum from segment 1 or 2 |
| 7–8 | multi-hop | computed sum of two entities from different segments |

In R1/R2, answers are produced from the final compacted state **only** (a
checkpoint runs after segment 4 as well); in R0 from the full transcript.
The same episodes (identical seeds, identical text) are used across all three
regimes and all models — a fully paired design.

## 3. Regimes — exact mechanics

- Compaction budget: instructed ≤ 120 words in both R1 and R2; hard cap
  `num_predict = 256` tokens in both. Realized token counts recorded from the
  Ollama API (`eval_count`) and reported per regime; the matched-budget claim
  is checked, not assumed.
- Compaction prompts are format-symmetric: both state that the transcript
  will be REPLACED by the output and anything omitted is lost; both instruct
  merging previous state with new information; neither reveals the questions.
- R2 block format:

  ```
  === STATE ===
  values:
    NAME = N
  combined:
    NAME1 + NAME2 = N
  relations:
    NAMEC = NAMEA + 1
  notes: <free text>
  === END STATE ===
  ```

- Answer calls: temperature 0, one question per call, context = [system
  preamble + state (or transcript) + question], required to end with
  `ANSWER: <integer>`.
- All generation at temperature 0, fixed Ollama options, thinking disabled
  where the model supports a think toggle.

## 4. Models

Three local Ollama models spanning size (as available on Hermes 2026-07-10):
`gemma3:4b`, `qwen3:14b` (think disabled), `qwen3-coder:30b`. If the 30B
model proves too slow in the smoke run (>25 s/call sustained), it is dropped
and the drop is reported — a dated amendment, not a silent one.

**N = 15 episodes** per model (same 15 across regimes/models) →
15 × 8 = **120 probes per (model × regime) cell**.

## 5. Metrics and scoring

Primary metric — **numeric-corruption rate**: fraction of probes whose parsed
answer ≠ ground truth, classified:

- `correct`
- `off_by_one` — answer = truth ± 1
- `conflation` — answer equals a *different* ground-truth quantity of the
  episode (any base/partner/neighbor/distractor value or stated sum) — the
  incident-class signature
- `other_wrong`
- `omission` — no parseable `ANSWER: <integer>`

Secondary — **multi-hop completion**: correctness on questions 7–8.

Exploratory (stage decomposition) — **compaction-stage corruption**: R2 state
blocks are parsed and each retained (name, value) pair scored against ground
truth; R1 prose scored by nearest-number-to-name heuristic. Flagged
exploratory; heuristic scoring of prose is not exact.

## 6. Pre-registered predictions (frozen 2026-07-10, before any run)

- **P1 (H-compaction):** corruption rate R2 < R1 at matched realized budgets;
  expected ordering R0 ≤ R2 < R1.
- **P2:** the `conflation` class specifically is elevated in R1 relative to
  R2 (the incident-class signature).
- **P3 (exploratory):** compaction-stage corruption is lower in R2 than R1 —
  i.e., part of the R1 deficit enters at write time, not only at read time.
- **Falsifier:** if R1 ≈ R2 on the primary metric (paired McNemar per probe
  across the pooled cells, α = 0.05, plus per-model direction checks), then
  format does not matter at matched budgets and H-compaction is falsified —
  the Face-11 prototype loses its measured motivation and the result is
  published as the null.

Analysis: per-cell rates with 95% Wilson intervals; paired McNemar (R1 vs R2)
on identical probes; regime × class breakdown table. No hypothesis test other
than the pre-registered McNemar is treated as confirmatory.

## 7. Artifacts

```
results/XR-001-externalization-pilot/
├── PROTOCOL.md            this file (committed before first run)
├── tasks.json             generated episodes + ground truth (seeded, committed after generation, before runs)
├── manifest.json          run ledger — one entry per (model, regime, episode), resumable
├── raw/*.jsonl            every LLM call: full prompt, response, token counts, timing
├── results.json           scored probe-level records
└── RESULTS.md             analysis + the baseline table for the Face-11 prototype
```

Runner: `scripts/xr001_externalization_pilot.py` (stdlib-only; Ollama HTTP
API at localhost:11434; bounded cycles — manifest updated after every
episode; safe to kill and resume at any point).

*Amendments to this protocol after the first run are dated edits below this
line, never silent revisions.*

---

**Amendment 2026-07-10 (pre-launch, before any run; findings from the
four-lens fresh-context adversarial verification of the implementation):**

1. **P2 class reading pinned.** Under the documented precedence (off_by_one
   checked before conflation), the designed conflation-trap answer — the
   base value on a neighbor probe, which is truth−1 AND a conflation-set
   member — always classifies as `off_by_one`. P2 is therefore read on the
   **union class `off_by_one` ∪ `conflation`** ("incident-signature
   errors"); the full five-class breakdown is still reported. P1 and the
   confirmatory McNemar (correct vs incorrect) are unaffected.
2. **Uniform generation caps pinned.** Answer calls carry
   `num_predict = 192` (the protocol had pinned only the 256 compaction
   cap); Ollama `num_ctx` is pinned to 4096 on every call so R0's full
   transcript is never silently head-truncated by a model-default context
   window. Both uniform across regimes and models — no asymmetry.
3. **Scoring integrity.** Scoring filters to manifest-COMPLETE cells only
   and dedups retried compaction calls, so FAILED/partial cells and retry
   duplicates cannot contaminate probes, the exploratory stage metric, or
   the matched-budget token means. `analyze` refuses to run on an
   incomplete bank unless `--allow-partial` is passed, and such output is
   marked NON-CONFIRMATORY. The run manifest also records the tasks.json
   sha256 and refuses to resume against a different task set; a corrupted
   manifest halts the runner instead of silently restarting the sweep.
