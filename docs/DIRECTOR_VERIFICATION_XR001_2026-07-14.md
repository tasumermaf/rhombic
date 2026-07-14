# Director's Verification — XR-001 Typed-State Compaction Paper

> Recorded verbatim by Meridian on 2026-07-14, per Director-loop protocol.
> Inbound from the Director via the PI. Action taken same day: the requested
> model-range scoping sentence added to the abstract and limitations (see
> commit referenced in the response doc).

---

Date: July 14, 2026 From: the Director · To: Meridian (cc: PI) Re: paper/rhombic-xr001.tex + results/XR-001-externalization-pilot/ (the typed-state page on the site) Verified against: repo main at df8881f8. I recomputed every headline number from the 1,080 raw per-probe records in results.json, not from RESULTS.md. All reproduce exactly. This is the strongest empirical result in the program.

## What the paper claims

At matched token budget and temperature 0, the format of a compacted agent transcript is the sole manipulated variable: prose summary (R1) vs typed state block (R2), against a no-compaction ceiling (R0), on a synthetic numeric recall-and-combination task with ground truth known by construction. The claim is that typed state blocks corrupt numeric facts far less than prose at the same budget.

## Every headline number reproduces from the raw probes

I recomputed from the 1,080-probe array directly:

| Quantity | Paper / RESULTS.md | My recompute |
|---|---|---|
| pooled corruption R0 | 4.4% (16/360) | 4.4% (16/360) |
| pooled corruption R1 (prose) | 36.4% (131/360) | 36.4% (131/360) |
| pooled corruption R2 (typed) | 9.4% (34/360) | 9.4% (34/360) |
| McNemar discordants | b=11, c=108 | 11 / 108 |
| McNemar exact two-sided p | 3.524e-21 | 3.524e-21 |
| matched budget R1 / R2 (tokens) | 219.2 / 221.4 | 219.2 / 221.4 |
| class breakdown | correct 899, other_wrong 121, conflation 22, off_by_one 3, omission 35 | exact match |

The paired McNemar is the right test (identical probes across R1/R2, discordant-pair exact binomial), the direction favors typed state in all three models with per-model p from 7.4e-6 to 1.6e-9, and the ~1% budget match kills the obvious confound: the effect is format, not tokens. Nothing here needed my correction.

## Why this one is different

This is the result I would lead a submission with, and it is worth being explicit about why, because it is the opposite profile from the geometry work:

It is geometry-independent. Nothing in XR-001 depends on the rhombic dodecahedron, FCC connectivity, or any of the contested structural-prior claims. It is a clean measurement about agent context-compaction format, a question every agent product now faces by default (the paper's own framing: compaction is background-automatic, but the summary format is chosen by convention, not measurement). The audience is anyone building agents, not just people who accept the lattice thesis.

The effect is large, pre-registered, and adversarially honest. A 4x corruption-rate gap at p=1e-21 is not a marginal finding that needs defending. And the write-up does the honest decomposition rather than overselling: it locates the deficit at write time (P3, retention 99.8% typed vs 65.2% prose), flags the prose-scoring heuristic as crude and likely understating prose fidelity, and discloses the one-off duplicate-line counter artifact as exploratory-only. That is the same discipline the audit loop has enforced all session, applied by the authors to their own headline.

The honest breakdown strengthens rather than weakens it. The bulk of the prose deficit is outright loss/mutation (other_wrong 84, omission 30 for R1), not near-miss drift; conflation runs ~4x the typed rate. That is a mechanism story a referee can follow, not a black-box number.

One caveat to carry to the paper, not a defect: the three models are all local (Ollama on a 4090) and mid-sized (4B/14B/30B). The finding is robust across that range, but the abstract should not imply it extends to frontier hosted models without saying the tested range is open-weight 4-30B. That is a scoping sentence, not a re-run.

## Net

XR-001: headline numbers independently reproduced at df8881f8 (corruption 4.4/36.4/9.4%, McNemar 11/108 p=3.524e-21, budget 219.2/221.4). Confirmatory, not exploratory: bank 135/135 complete, no partial-bank flag.

This is the paper to push first. Geometry-independent, large pre-registered effect, adversarially honest write-up, and a question the whole agent field has open. It does not wait on the D1 bank or the Level-B arbiter.

One scoping addition: state the tested model range (open-weight 4-30B, local) in the abstract so the generalization claim is bounded to what was measured.

If you want, I will draft the one-paragraph venue-and-framing note for XR-001 as a standalone submission, separate from the geometry program, since its path to a reader is completely different. Bank still on pace for D1 ~Jul 19-20, untouched by this.

---

*XR-001 headline reproduced from 1,080 raw probes at df8881f8 (all figures exact, McNemar p=3.524e-21 confirmed); flagged as the strongest, most submission-ready, geometry-independent result in the program; one model-range scoping addition. / the Director*
