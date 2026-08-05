# DATED AMENDMENT — G-5: XSum Topic Cells via Native Editorial Section (T3 → T2)

**Filed 2026-08-05 by Meridian (decider)**, under L-006, resolving ambiguity
G-5 of the granularity registration (`REGISTRATION_GRANULARITY_2026-07-30.md`)
per the Director's grade condition of 2026-08-04 ("the amendment comes to me
as its own document"). **L2/L3 remain STOPPED until the Director rules on
this document.** Consult: LAORA (2026-08-05); every load-bearing measurement
below was **re-run by the hub** from the consult's script and reproduced
exactly before filing.

## 1. The proposal: retire the annotator — the labels already exist

The registered card declares xsum L2 (4 topics) and L3 (12 topics) as
[T3 model-annotated] with no annotator pinned. **No annotator is needed.**
The HF dataset's `id` field is the BBC article id, and the XSum authors
publish `XSum-WebArxiveUrls-BBCids.txt` (226,711 rows, URL ↔ bbc_id; sha256
`c882b869644e29fd76cb54f1f65c113423895faa914f4ee0def5e67aa4635725`) whose
URL paths carry the BBC **editorial section** (sport/football,
news/business, news/science-environment, …).

**Measured over the LOCKED 40,000-row pool** (derived through
`asset1_datasets.split_ids` exactly as `granularity_labels.py` derives it;
hub-reproduced):

- Coverage **40,000/40,000 = 100.0000%** — zero misses, zero malformed.
- Content spot-check **10/10** (e.g., a tarantula-colouration study under
  science-environment; British Gas price cuts under business).
- **13 authored section families, all clearing the D4 floor** (post-val
  G-9 reading, ≥1,500 rows): sport 9,750 · uk-england 6,574 · uk-scotland
  3,249 · uk-wales 2,590 · uk-national-politics 2,499 ·
  sci-health-tech-edu 2,304 · culture-media 2,202 · business 2,192 ·
  world-europe 1,846 · uk-n-ireland 1,803 · world-asia-pacific 1,744 ·
  world-americas 1,697 · world-africa-mideast 1,550. Σ = 40,000.
- **K lands exactly as locked**: merge the two smallest world buckets →
  12 classes at L3; L2 = frozen 4-way strict coarsening (uk 16,715 /
  world 6,837 / sport 9,750 / business-science-culture 6,698), mirroring
  `ALPACA_VERB_L2_MERGE`. L2 K=24, L3 K=48, 240 adapters per level — the
  locked cost table is not restated.
- Class-size spread 5.7× (1,697–9,750) — narrower than the imbalance L2
  already carries under the ratified `--balance none` (math:steps_4 1,026
  vs squad:tg4_0 9,503).

**Consequently the cells are [T2]** — a deterministic pure function of a
native field plus a frozen external table, byte-for-byte reproducible by
committed script with the existing `row_ids_sha256` machinery. Structurally
the same hybrid as squad's T1-titles + T2-grouping.

## 2. Why this beats any annotator (the decisive argument)

**A clustering annotator would have made the card circular.** Control 6.1 —
the card's designated bound on T3 label noise (§9.1) — IS TF-IDF + linear
SVM. Any annotator whose assignment objective is TF-IDF-based (LDA, TF-IDF
k-means, NMF) would have D6 scoring labels manufactured in D6's own feature
space, rendering the label-noise bound vacuous by construction. URL
metadata is independent of both D6's text features and the llama3.2-1b
weights the ladder probes: **zero model lineage, zero leakage, zero cost,
zero GPU.**

Further consequences: the **entire ladder becomes clean-core** (L2 24/24,
L3 48/48 at T1+T2) — D3's RESTRICTED clause has no T3 cells to bite on
anywhere, which per the Director's G-4 principle is reported as **"no T3
cells — not testable," never "passed."**

## 3. Honest scope caveat (named, not buried)

The axis is **publication section**, not semantic topic — geography
cross-cuts subject on the BBC's desks (measured example: a footballer's
driving ban filed under uk-scotland-glasgow-west, not sport/football). The
amendment names the axis "editorial section" in all outputs. It remains
substantially more semantic than doc-length and different in kind from
squad's entity groups — preserving the card's §9.3 format ≠ topic ≠ entity
heterogeneity design. (The rejected alternative — greedy bin-packing over
raw sections — reproduces cleanly but mixes 19–22 unrelated sections per
bin and is brittle to URL query-string artifacts; measured, and declined
as semantically arbitrary.)

## 4. Freeze artifacts (on acceptance)

1. Pool-restricted `bbc_id → section` map (1.17 MB) — **committed in-tree**;
2. upstream file sha256 (above);
3. the URL→section normalizer (strips trailing `[-/]\d+`, query strings,
   fragments);
4. the 12-family inventory as a frozen dict; 5. the L2 4-way merge dict;
6. per-class `row_ids_sha256` via the existing machinery.
Implementation note: `granularity_labels.py` `_TASK_COLUMNS` must add `id`
for xsum (currently pulls `document` only).

## 5. Both outcomes

- **Accepted:** xsum L2/L3 labels re-emit under the section rule; L2 and
  L3 become launchable on the frozen tier order with K as locked; the
  ladder is fully clean-core; this document plus the freeze artifacts are
  the record.
- **Rejected (the Director rules the registered T3 axis must remain
  model-annotated):** the fallback is pinned here — a non-llama annotator
  (llama-3.x excluded absolutely: the adapted family must not define the
  labels whose weight-space separability the ladder measures; qwen2.5
  would contaminate a revived D9), run as **batched logit-scoring over the
  12 fixed label continuations** (no autoregressive decoding, no
  sampling), with the **committed assignment file + sha256 as the frozen
  artifact** — temperature 0 is NOT sufficient for reproducibility
  (batched matmul reduction order and kernel selection flip near-tie
  argmaxes across drivers/batch sizes); the model is provenance, not a
  regeneration path, and T3 pools are committed in-tree (the T1/T2
  gitignore policy does not transfer). Cost [PROJECTED]: ~14M prefill
  tokens ≈ 3–6 GPU-h local at a tier boundary, or ≈$7–14 via an unrelated-
  lab API.

## 6. Provenance

LAORA consult 2026-08-05 (structured block in the session record; its
verified-vs-taken-on-report line included first-hand measurement of
coverage, counts, floors, spot-check, and the upstream file hash). Hub
independently re-ran the measurement script before filing: all 13 family
counts, floor clearance, Σ=40,000, and the bin-packing alternative's loads
reproduced exactly. Schedule: Arm B completes ~Aug 14; a ruling by then
keeps the ladder gapless.
