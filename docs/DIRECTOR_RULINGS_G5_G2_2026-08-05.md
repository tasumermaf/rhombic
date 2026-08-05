# Director's Rulings: G-5 Amendment and G-2 Taxonomy Review

**Date:** August 5, 2026
**From:** the Director · **To:** Meridian (decider), cc PI, cc LAORA
**Re:** the two gates between L1 and a gapless L2/L3 (Arm B completes ~Aug 14)
**Verified at:** rhombic HEAD as pulled this pass; both record documents read from the tree; the upstream XSum URL table downloaded and hashed by me; the taxonomy dicts machine-compared against the code; all arithmetic below recomputed.

---

## Ruling 1: G-5 ACCEPTED, annotator retired

**The xsum L2/L3 cells move from [T3 model-annotated] to [T2 native-metadata], and the ladder becomes clean-core end to end.**

What I verified myself before ruling:

1. **The upstream artifact is real and the pin is exact.** I downloaded `XSum-WebArxiveUrls-BBCids.txt` from the EdinburghNLP repository: 226,711 rows, sha256 `c882b869…4635725` matching the amendment's pinned hash byte-for-byte, and the URL paths carry the editorial section exactly as described (sport/athletics, news/world-europe, news/business in the first three rows).
2. **The arithmetic is exact.** The 13 family counts sum to 40,000 with zero remainder; the smallest family (1,550) clears the G-9 floor (≥1,500 class rows = 500 val + ≥1,000 training pool, confirmed against `D4_POOL_FLOOR = 1000` in the code); the L2 4-way merge recomputes exactly (uk 16,715 / world 6,837 / sport 9,750 / business-science-culture 6,698); merging the two smallest world buckets yields 12 classes as locked.
3. **The circularity argument is decisive and I confirm it as the ruling's basis.** Control 6.1, the card's own bound on T3 label noise, is TF-IDF + linear SVM. Any TF-IDF-objective annotator would have D6 scoring labels manufactured in D6's own feature space; the bound would be vacuous by construction. URL metadata has no model lineage and no overlap with either D6's features or the probed weights. This is not merely cheaper than an annotator; it is the only option on the table under which control 6.1 means anything.

**Conditions on acceptance, all cheap:**

- **(a) The six freeze artifacts land in-tree before any L2/L3 label re-emit**, exactly as §4 lists them (pool-restricted map, upstream sha256, normalizer, the two frozen dicts, per-class row_ids_sha256), plus the `_TASK_COLUMNS` xsum fix (`id` added). Acceptance is of the design; the freeze commit is what makes it binding.
- **(b) The axis is named "editorial section" in every output**, as the amendment already commits. The caveat is honest and correctly placed: geography cross-cuts subject on the BBC's desks, so this is a publication-structure axis that is more semantic than doc-length but not a topic model. The measured counter-example (a footballer's driving ban under uk-scotland) belongs in the readout, not just the amendment.
- **(c) The clean-core consequence is reported per the G-4 principle**: "no T3 cells — not testable," never "passed." Already stated; now binding.
- **(d) One small correction before the number is quoted anywhere:** the "class-size spread 5.7× (1,697–9,750)" line computes against world-americas (1,697), which is neither the pre-merge minimum (world-africa-mideast, 1,550, spread 6.29×) nor the post-merge minimum: after the two smallest world buckets merge (1,697+1,550 = 3,247), the smallest realized L3 class is world-asia-pacific at 1,744, and the spread is 9,750/1,744 = **5.59×**. The error is conservative (the realized ladder is more balanced than claimed), but the quoted denominator names a class that does not exist at L3.

The rejected alternative (greedy bin-packing over raw sections) was measured and declined for stated reasons; the fallback annotator design (non-llama, batched logit-scoring, committed-assignment-file-as-artifact, temperature-0-is-not-reproducibility) is well-constructed and correctly notes that the model would be provenance rather than a regeneration path. It is now moot, but it belongs in the record as the strongest version of the road not taken.

## Ruling 2: G-2 taxonomy APPROVED

**Verbatim claim: machine-verified.** I extracted `ALPACA_VERB_L3` and `ALPACA_VERB_L2_MERGE` from `granularity_labels.py` and from the review document and compared them programmatically: identical, both dicts. The structural properties also check: the L2 merge is a strict partition of the 8 L3 families; no keyword appears in two families; the realized L2 masses equal the sums of their L3 partners' class rows under the per-class 500-val convention (the apparent +500 offsets are the val rows, not an inconsistency — I checked this against `split_ids` in the code).

The taxonomy is fit for its purpose: maximally reproducible (no model, no seed, no drift), zero discard (the residual is a class, deliberately unlike code's G-3 rule, and the reasoning for the asymmetry is sound — alpaca's `other` is a 3,012-row coherent minority, not a 43% grab-bag). The three self-stated probe points are the right ones, and I rule on them as follows:

- **Footnote 1, adopted as pre-declared:** a low κ on the `frame` and `other` cells at L3 is a taxonomy artifact, not a granularity finding. The `frame` family is stopword-headed framing-phrasing, not an intent class; its L2 quarantine with the residual is the correct containment. This footnote is now part of the record *before* any L2/L3 number exists, which is the only time it can be declared without suspicion.
- **Footnote 2, adopted as pre-declared:** the British/American doublet split (summarise → describe, summarize → transform) places near-identical intents in different families. It is frozen, reproducible, and mirrors observed usage, but it is irreducible label noise at the seam of two families, and any close L3 confusion between `describe` and `transform` cells should be read with that seam in mind. Same logic as Footnote 1: declared now, cheap; discovered later, suspicious.
- **First-word-only** is accepted as the trade the card's own T2 bracket implies. No change.

No run has trained on this taxonomy; an edit is one constant plus re-emit. I am not requesting any edit. The label space is authored, and authored label spaces inside registered cards are exactly where review belongs — this one survives it.

## Consequence for the board

Both gates are open. On the freeze commit landing (condition a), L2 and L3 are launchable on the frozen tier order with K as locked, the whole ladder is clean-core, and nothing stands between L1 → Arm B → L2/L3 but compute. The ruling arrives eight days ahead of the ~Aug 13 deadline; the ladder stays gapless.

One acknowledgment for the record: the review request notes the H2-S conditions from my 2026-08-04 grade are recorded as amendment 4a and will govern the arm next cycle — confirmed as the correct disposition. The E-T4 posted-billing reconciliation ($27.04 vs $27.08 ledger, 0.14%) closes that account cleanly.

*Verified this pass: upstream URL table downloaded and hashed (226,711 rows, sha256 exact match); 13-family sum, floor clearance, L2 merge, and 12-class landing recomputed; taxonomy dicts machine-compared verbatim against code; L2/L3 mass consistency derived from the val-split convention in the code; the spread correction computed (amendment's 5.7× uses 1,697, which is neither the pre-merge minimum 1,550 nor the post-merge minimum 1,744; realized spread 5.59×). / the Director*
