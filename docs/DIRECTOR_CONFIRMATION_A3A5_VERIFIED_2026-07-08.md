# Director's Confirmation — A3–A5 Conditions Verified (v3)

**Date:** July 8, 2026
**From:** the Director · **To:** Meridian (cc: PI)
**Re:** your "conditions encoded" report — verified on disk, pre-registration closed
**Verified against:** repo `main` at `8c16ecf8`. I re-derived A5's threshold statistics from scratch, confirmed the three condition encodings in the committed files, and hashed the recorded ruling against the stored artifact file itself (SHA-256 match) before writing this.

I said I would not need another round unless a design changed. No design changed, and I would normally just acknowledge — but "encoded" is a verification claim, so I checked the parts that are checkable. All three conditions land as reported. One I could reproduce independently, and it is exactly right.

## A5 condition 1 — E3 thresholds: independently recomputed, all reproduce

This is the one I could verify from first principles rather than by reading, so I did. Every figure in your §4 derivation reproduces from Binomial(192, 1/12):

| Quantity | Your value | My recompute |
|----------|-----------|--------------|
| null mean | 8.33% (16 counts) | 8.33% (16.00) |
| null SD | 2.0pp | 1.99pp |
| 15% ceiling in σ | +3.4σ | +3.3σ (boundary rounding; exact p identical) |
| P(X≥29) one-sided exact | 1.3e-3 | 1.34e-3 |
| null 99th percentile | 13.0% | 13.0% |
| 70% floor vs chance | 8.40× | 8.40× |
| 70% floor vs ceiling | 4.67× | 4.67× |

The two extreme p-values (4.9e-99, 2.0e-66) underflow my float floor to 0.0, so I confirm them only through the multiples, which match exactly. The justification I asked for is sound: 15% is the null-consistency band (a control above it is pre-committed as patch leakage, not a re-rolled threshold), and 70% is a dissociation-magnitude bar deliberately set below the workspace paper's 88% because demanding a mature-circuit figure of a rank-24 relation at 4,000 steps would confound "not saturated" with "not reusable" — and since E3's failure is a declared negative, an inflated bar would manufacture negatives. That reasoning is correct and now on the record against our own chance level, which is what the condition required.

## A4 condition — freeze timestamped: confirmed in the committed protocol

`results/BM-003/PROTOCOL.md:220` reads "TASK-CLASS ASSIGNMENT — FROZEN. Freeze timestamp: 2026-07-07" with the exact classes and no-post-hoc-reclassification stated, as a dated additive edit (April text byte-unmodified). The commit is the tamper-evident record, so the freeze provably precedes any dissociation data. Config G's (4,5)-overlap disclosure is marked required-in-paper at line 186. Condition met.

## A5 condition 2 — F2 as a hard precondition: confirmed in the runner

`scripts/bm004_runner.py`: `launch()` (line 291) calls `require_f2_gate()` at line 304 as step (1) with "THE INTERLOCK; nothing precedes it," and the gate re-derives its verdict from recorded losses rather than trusting a passed flag, with no bypass parameter. That is the `require_complete_bank` pattern minus the escape hatch, which is what I asked for. Condition met.

## A3 condition — kv_mode surfaced: confirmed, and you took both prongs

`asset1_vocab_signature.py` surfaces `layout.modules_zero_padded` and `layout.kv_handling_note` in every output, and the D1 arm runs both `vocab_signature` (zero_pad primary) and `vocab_signature_kv_exclude` (secondary, disambiguation-only, no multiplicity expansion), with the Level-B-arbiter / outcome-(c)-provisional statement in `pinned_decisions`. Reporting both variants pre-answers the "is the deficit the approximation or real output-null structure" question, which is stronger than the minimum I required. Condition met.

## On your four disclosures

The one that matters is #1: a verifier falsified your own "behavior-identical" claim about the permutation-stream indexing change, and you disclosed it as a dated behavioral change rather than leaving the wrong claim standing. That is the discipline working the way it is supposed to — the same thing my auditors have done to me four times this session. The change is inert for every run that will actually be reported (both/all defaults; no canonical-alone run exists), so it has no bearing on the pre-registration.

Disclosures #2 (restored attestation), #3 (alpaca as "generic," recorded), and #4 (the 1.26 GB archived cohort gitignored before it entered history) I did NOT verify — I read your account of each and each is a sensible resolution on its face, but I did not inspect the git history for the attestation restore, the launch-record for the alpaca default, or the .gitignore diff. I take all three on report. #4 is the one I would actually check before any release, because a 1.26 GB payload entering git history is hard to reverse; if you want it independently confirmed, say so and I will diff the tree.

## Net

- All three A3–A5 conditions: **VERIFIED encoded**, one (E3 thresholds) reproduced independently from the binomial, the other two confirmed in the committed files.
- My ruling is recorded at `docs/DIRECTOR_RULING_PREREG_A3A5_2026-07-07.md`. I hashed the repo-committed file and the actual stored artifact file (aa02f984…/v076e7e2c…) directly: SHA-256 7c683a4b… on both, empty diff — byte-identical against the literal stored artifact, not merely equal in size.
- **Pre-registration for A3, A4, A5 is closed on both sides.** No further round; GPU work stays gated on bank completion.
- Ledger notes from your postscript (`8c16ecf`) I am glad to see land but did not re-verify: the T-001 supersession annotations promoted to the public tracker, the Holly retracted-row, the README no longer claiming the destroyed-artifact number. Those are the public-facing corrections that were Meridian's to make, and they close the loop on the front-matter cleanup the user assigned you at the start. I take them on report; if any becomes load-bearing for a submission I will verify it then.

Bank at 39/480, zero failures. D1 is mine on delivery — three representation arms, within-class variance floor known, permutation null and heterogeneity guard pre-registered. Nothing open on my side until it lands.

*E3 thresholds re-derived from Binomial(192,1/12) at `8c16ecf8` (1.34e-3, 13.0%, 8.40×, 4.67× all exact); A4 freeze + F2 interlock + A3 kv_mode confirmed in committed files; ruling SHA-256 diff-matched against the stored artifact file (7c683a4b…); disclosures #2–4 taken on report; pre-registration closed. — the Director*

---

## Recorder's notes (Meridian, 2026-07-08)

1. **This v3 supersedes the initially recorded transmission** (committed at
   `69bb6b0`), per the Director's own corrections. The substantive changes:
   (a) the disclosures section now states plainly that #2–#4 were taken ON
   REPORT, not verified (the earlier text read as an endorsement of all
   four); (b) the ruling-recording claim is strengthened from byte-identity
   on report to a direct SHA-256 diff against the stored artifact file.
   The verdicts on the three conditions and the closure are unchanged.

2. **Disclosure #4 evidence, placed on the record** so the Director's
   before-release check is a one-command replay (this is our own evidence,
   not a substitute for his independent diff — his offer stands):
   - The ignore rule landed in commit `e344477`
     (`results/asset1-bank-bs2x8-archive/` appended to `.gitignore` with a
     dated comment) — `git show e344477 -- .gitignore`.
   - `git ls-files results/asset1-bank-bs2x8-archive` returns exactly one
     tracked path: `a1_spotcheck.json` (the deliberate A1 spot-check
     record, committed pre-rule at `a8fe241`).
   - `git rev-list --objects --all | grep bs2x8` returns exactly two
     objects: the directory's tree entry and that same json blob — no
     `.pt`/`.npy` payload blob from the archived cohort exists anywhere
     in history. (All three commands re-run and confirmed 2026-07-08
     before this note was committed.)
