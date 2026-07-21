# Director's Review — Asset-1 Paper Draft v0

> Filed verbatim by Meridian, 2026-07-21, per the standing rule. Received via
> Timothy (PI). Verdict: ACCEPT with one required correction (the D3
> distance-baseline diagnosis — the Director's own earlier framing, corrected
> by him; all four paper sites fixed same day). Companion document: the
> re-versioned sign-off, filed as DIRECTOR_SIGNOFF_ASSET1_2026-07-21_v2.md.

---

Date: July 21, 2026 From: the Director · To: Meridian (cc: PI) Re: ASSET1_PAPER_DRAFT_v0.md (sha256 `7923b58…`, rhombic `7e747b7`), "The Gauge Is the Obstacle" Method: I re-checked every quantitative claim in the draft against the per-item verification bundle I regraded on Jul 21 (`638f4a8`, manifest-verified). Fourteen of fifteen load-bearing numbers reproduce exactly. One framing is wrong, and it is mine; the paper faithfully propagated an imprecise characterization from my own sign-off. That one needs a fix; the rest is accept.

## The one correction (my error, not yours)

The D3 distance-only baseline is not "CV-seed-sensitive." My re-run reproduced the report's naive-CV block exactly, and the gap is the fold scheme, not the seed. The draft says this in four places (§3.3.3, §5.2, Abstract-adjacent, Limitations), quoting my own earlier language: "the Director's independent re-run gave 0.686/0.667 against the reported 0.675/0.713, a few points of CV-seed sensitivity."

When I re-checked against `d3_report.json`, my 0.686/0.667 matches the naive block (`auc_distance` naive = 0.6859 qwen / 0.6671 llama) essentially to the digit, while the headline is the group-aware block (0.6753 / 0.7132). My from-scratch re-run used a plain StratifiedKFold, which is the naive scheme, so I reproduced the naive number exactly and mislabeled the difference as seed noise. The same holds for the full model: my 0.995/0.952 are the naive-block values (0.9952 / 0.9518), not a seed variant.

The correct statement is: the Director's independent (naive-CV) re-run reproduced the report's naive block exactly (distance 0.686/0.667, full 0.995/0.952); the headline uses group-aware StratifiedGroupKFold, which differs from naive by the fold-assignment scheme, not the seed. Because the vertex-disjoint design yields 120 single-pair components, group-aware and naive should agree to within fold-reshuffle noise, and they nearly do — the ~0.02-0.05 distance-baseline gap is fold-scheme variance on a 2-feature model over 120 points. That is still a real reason to pin the fold configuration and report the baseline's fragility, so the conclusion (pin it, report it) is unchanged; only the diagnosis (seed vs scheme) was wrong.

Please replace "CV-seed sensitivity / CV-seed-sensitive" with "fold-scheme (naive vs group-aware) sensitivity" at all four sites, and correct the parenthetical to say my re-run reproduced the naive block. Keep the pinned-seed requirement — pinning is still right — but do not attribute the baseline gap to the seed. And drop the erratum-style credit to me for catching a seed effect; I did not catch a seed effect, I ran the naive scheme and misnamed the result. The honest version is less flattering to the regrade and more accurate, which is the trade this whole loop is built on.

## Everything else reproduces exactly

I verified each of these against the bundle, not against the delivery report:

* H1 (Table 1): all eight LOO accuracies exact from the confusion matrices; the canonical LOO re-run from scratch on the feature matrices is 1.0000 both families; all eight dimensions match; permutation p 0.000999 throughout.
* H1 heterogeneity guard: the "1.00–1.46 across cells" claim is correct — I confirmed the 1.46 is the vocab_signature_kv_exclude ratio (1.4575 qwen), and no cell triggers the 3.7 threshold. Good; I had only re-derived raw/canonical earlier and this closes it.
* H1 raw code-recall: the "0.40–0.45 drives the null rejection" claim holds — qwen code recall 0.45, llama code 0.40, every other raw per-class recall at or near 0.
* H2 (Tables 2–3): family probe 1.0000 → 0.1521 (spectrum) / 0.0000 (probe); standardized transfer 0.7833 / 0.7375 / 0.7792 / 0.7792; raw baselines 0.1667 / 0.1667 / 0.1750 / 0.2167; binomial p 7.70e-98 / 1.20e-84 / 1.37e-96 / 1.37e-96 — all exact, re-run end-to-end with my own standardization.
* D2 (Table 4): all 14 per-kind means exact from the 360 per-eval rows.
* D3 (Table 5): group-aware AUC 0.995 [0.983,1.000] / 0.962 [0.898,0.999], margins +0.320 [0.150,0.511] / +0.249 [0.039,0.490], pooled 0.9890 — all match `d3_report.json` exactly; conflict rate 0.8583 at both endpoints; realized same-task coverage 14/120 qwen (11.7%) / 22/120 llama (18.3%) exact.
* D-aux (Table 6): pooled 0.300 [0.175,0.415], qwen 0.418, llama 0.337, llama-math 0.549, qwen-math 0.336, qwen-xsum −0.301 — all exact from the 480 per-run pairs.

## The two points you flagged

* W2T "10k+" = adapter count, not class count. The draft states it correctly in both §2 and the H1 discussion ("up to 312 classes … over collections of 10k+ adapters"), and the regime-contrast argument (label granularity, not collection scale, is the distinguishing variable) is the right one. Fixed cleanly. I did not re-verify W2T against its source in this pass; I take your R4 re-fetch on report for the three citation-wording fixes.
* The erratum on my pinned language. Noted and correct to flag rather than silently diverge. The historical DIRECTOR_DECISIONS_2026-07-06.md carries the adapters-vs-classes conflation; it stays as written under the dated-amendment doctrine, and the paper states it right. That is exactly the protocol. For the record: I accept the erratum; my pinned text was wrong on that noun, the paper is correct, and no historical document should be silently edited to match.

## Smaller notes, non-blocking

1. Title. "The Gauge Is the Obstacle" is the right unifying claim and it earns the mechanism framing across all five results. The rejected candidate ("canonicalization is the whole story") would indeed have overclaimed against D2/D3; good call, and keep the §2 sentence "canonicalization is the whole story at this operating point" scoped with that qualifier, since it is true only for H1.
2. Abstract H2 framing. "Two of the five outcomes went against us, and they are the credibility of the other three" is the strongest sentence in the paper and it is honest. Keep it. It is the sentence a skeptical referee will believe the rest of the numbers because of.
3. §5.2 naive-vs-group-aware sentence ("llama full 0.962 group-aware vs 0.952 naive; qwen 0.995 under both") is exactly right and now doubles as the correct frame for the baseline-gap fix above — the full model shows the same naive/group-aware split, it is just tiny there. Make the distance-baseline paragraph consistent with this one, since right now §5.2 correctly calls it a scheme difference for the full model two paragraphs before mislabeling it a seed difference for the baseline.
4. Figures F1–F7 are all generatable from the bundle files named in §8; the generate-from column is correct against the JSON structure I read. No concern.

## Net

* Accept, with one required correction: replace the "CV-seed sensitivity" diagnosis of the D3 distance baseline with the accurate "naive-vs-group-aware fold-scheme" one at all four sites, and correct the credit to me. My earlier sign-off introduced the error; the paper inherited it faithfully; the fix is the same in both. I will re-version my own Jul 21 sign-off to carry the correction so the record is consistent on both sides.
* All fifteen headline quantities re-checked against per-item data; fourteen exact, the fifteenth (the baseline gap) real but misdiagnosed, not miscomputed.
* Both flagged points and the erratum handled correctly. Nothing else blocks.

This is a publishable draft. The numbers are sound, the H2 refutation is told as a refutation, the limitations are complete, and the one correction makes the reproducibility section more accurate about what the regrade actually did. Send the LaTeX/BibTeX version and the figures when they land and I will check the figure numbers against the same bundle.

Asset-1 draft v0 regraded against the per-item bundle: 14/15 headline quantities exact; the D3 distance-baseline "CV-seed sensitivity" framing corrected to a naive-vs-group-aware fold-scheme difference (the error was in my own sign-off, propagated faithfully); W2T adapters-vs-classes fix and the pinned-language erratum both correct; accept with that one correction. / the Director
