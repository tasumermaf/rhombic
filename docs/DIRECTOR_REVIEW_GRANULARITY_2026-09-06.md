# Director's Review: Granularity Ladder, 2026-09-05

**Date:** September 6, 2026
**From:** the Director · **To:** Meridian (decider), cc PI
**Re:** the six items in §8 of the 2026-09-05 review request; the period 2026-08-05 to 2026-09-05
**Verified at:** rhombic `d46c082` pulled this pass. Packet sha256 `5dfa7867…867a60` (13,031 bytes) matches the transmittal. Design §5, the registration's operative rulings, `granularity_queue.py` (interlock body), `granularity_analysis.py` (`require_tier_order`, `record_gate`, `allow_partial`), `QUEUE_STATE.md`, `RATE_RECONCILIATION.md`, and the committed L0 dry-run JSON all read from the tree; every number below recomputed.

**The month reads as a program that paused cleanly and re-audited itself before touching anything.** Zero GPU spent, zero runs launched, three self-reported process defects with fixes recorded, and a dry run that surfaced a real tooling fault before the one-way run could hit it. The six items are ruled below; four are confirmations, one is a conditional authorization, one is a hold.

---

## Item 1: L0 gate run, AUTHORIZED

**The dry run does what a dry run is for, and it earns the one-way run.** I re-derived every headline from the committed `L0_results.json` rather than the report: raw 0.1375 / κ −0.0350 / p 0.001998 / D10 FAIL (accuracy bar and κ floor both fail, p passes); canonical, vocab_signature, and vocab_signature_kv_exclude all 1.0000 / κ 1.0000 / p 0.000999 / D10 PASS; dims 5,114,112 / 50,688 / 8,704 / 4,352; D6 cv 0.9745 / κ 0.9694 at 1,000 per class realized for all six. All exact. `exploratory_only = true`, `allow_partial = true`, and `TIER_GATES.json` absent, as stated.

The anchor reproduces the released Asset-1 llama H1 numbers **to the digit** against my own July 21 regrade of the verification bundle (raw 0.1375, canonical 1.0000, vocab 1.0000 both kv modes). That is the strongest possible readiness evidence: the ladder's own script, on the ladder's own path, recovers the published result on the same 240 adapters. The tokenizer fault it caught (`d46c082`) is exactly the class of defect a dry run exists to find, and the fix is a path change with no numeric consequence.

**Condition:** the gate-recording run is launched from a tree at or after `d46c082` with **no further change to `granularity_analysis.py`** between the dry run and the one-way run. If anything in that file changes first, one more `--allow-partial` pass precedes the gate. The dry run certifies the code that ran, not the file by name.

One reading to hold onto for the paper: the raw arm's κ is **negative** at the anchor. The D10 form I set at registration was built to catch "statistically distinguishable, not usefully legible," and this is that case in the cleanest form it can take.

## Item 2: Third representation, COMPLIANCE

Design §5 reads, verbatim from the tree: *"Per level, per representation (**raw**, **canonical** [QR→SVD, bridge absorbed, frozen tooling], **vocab_signature** [both kv modes])"*, and readout 1 lists *"raw / canonical / vocab_signature × {6, 12, 24, K}"*. The registration's "Operative rulings (deltas from the design)" touches D1 through D10 and does not name representations, so §5 stands as written. **The script implemented two of a registered three; `53be598` is the code catching up with the registration.** No L-006 amendment is required.

Two notes. The default flip (`--representation all`) is correct, since the registration never contemplated a two-representation run as a reportable configuration; `both` should be labelled a *reproduction* mode in the docstring, not an alternative. And the D1 arm #3 provenance (one `VocabReadout` per family, null-stream indices 2 and 3) means the vocab arm inherits the A3 pre-registration's kv_mode disclosure requirement: `zero_pad` primary, `exclude` secondary, both surfaced in every output. The dry run already does this.

## Item 3: Interlock, CONFIRMED

I read `training_interlock()` in the tree. It computes `TIER_ORDER[1:idx]`, which starts at index 1: **L0 is structurally never a training-side requirement**, and the verification triple (L1 → [], ARMB → [L1], L2 → [L1, ARMB]) recomputes exactly. The design's §2 table gives L0 "new runs = 0"; L0 is analysis on the existing cohort; its gate governs L1's *analysis* through `require_tier_order`. Nothing in the registration, the lock, or the design forbids L1 training while the L0 analysis is unrecorded, and the training-side condition for L1 (`check_l0_rebaseline`: the 240-adapter cohort intact) is the right one.

**Ruling: the 176 pending L1 runs may complete before the L0 gate is recorded.** Two things travel with it. First, the interlock is now load-bearing and should be under test: a unit test that asserts the three-triple above, plus `L3 → [L1, ARMB, L2]`, belongs in the suite before the next tier is enqueued. Second, "may complete" is not "may be analyzed": no L1 analysis, and no L1 gate, until L0's gate is in the ledger. The queue already enforces this; the ruling makes it explicit.

## Item 4: Rate basis, CONFIRMED

`wall_clock_min` is the budget-relevant field (process end to end is what the card is paying for), and the reconciliation is sound: mean gap 0.3774 min, median 0.0025, one outlier (run_002, 17.02 min, cache warm-up at cohort start) that explains the mean-median split. All recomputed: overrun (31.9206/30.56 − 1) = **+4.45%**; L1 remaining 176 × 31.9206 / 1440 = **3.901 GPU-days**; ladder remaining 810 × 31.9206 / 1440 = **17.955 GPU-days**.

The card's return-to-Director trigger is >25%. **4.45% does not trip it; the card stays in force.** The projection method change (measured rate over the pending remainder, replacing a literal 30.56 over the full plan) is a correction, not an amendment: the earlier "5.093 GPU-days" for a quarter-done tier was a display error, and the ladder's cost table was never restated on it.

## Item 5: Cards, HELD

**The four cards and the questions register are not in the public tree.** `docs/cards/` and `docs/CALCULEMICHA_QUESTIONS_REGISTER.md` are cited as falco (private bench) paths, and the packet's public artifact list does not include them. I grade documents I can read; I do not grade summaries of documents. Send the four card files and the register, and I will pin the κ bars and the Q-02-1 sample frame and rule on the corpus-handling procedure in the same turn.

Two things I can say now, so the resend is not a round-trip lost. On the **κ bars**: a maker-set 0.70 is the conventional "substantial agreement" boundary and is defensible as a *reporting* threshold, but as a *lock* it should be argued against the card's own chance level and the coder count, not imported. Show me K, the number of coders, and the expected marginal distribution, and I will set the bar the same way D10's 0.40 was set: before any data. On **Q-02-1's handling rule**: reducing the protected stratum to transport features for an uncleared coder is the right shape, and the fact that the PI reviews it is correct. What the card must state is whether the reduction is *symmetric*, meaning whether the cleared coder also codes from transport features on that stratum so the two coders are comparable, or whether the stratum is excluded from the inter-coder κ entirely. Those give different κ and the card should choose before a coder sees a claim.

## Item 6: A7b, RUN IT bounded

The 64 completed L1 adapters are a zero-GPU held-out replication of D1 on the same base model, and there is no reason to wait for 240. But the note must state what n = 60 can and cannot establish, because it cannot establish what the ladder's confirmatory endpoint asks.

At n = 60, K = 6, balanced 10 per class: a perfect LOO gives a Jeffreys 95% interval of **[0.959, 1.000]**, so the 0.99 ceiling is inside the interval and **A7b cannot distinguish "at ceiling" from "one error below it."** One error at n = 60 is 0.983, already under the ceiling definition (≤2 errors at N = 240). What A7b *can* establish is the **direction and rough size of the raw-vs-canonical gap** on adapters trained under the L1 taxonomy rather than the Asset-1 taxonomy: if canonical is near 1.0 and raw is near chance on held-out adapters that share nothing with the Asset-1 bank but the base model, D1's central finding replicates out of sample. That is worth having and worth having now.

**Ruling: one paragraph suffices, with three pins.** (a) It computes LOO accuracy, κ, and the 1,000-permutation null for raw / canonical / vocab_signature, and reports Jeffreys intervals; it does **not** apply the D10 lock, the D5 ceiling, or any registered granularity metric, because it is not a level. (b) It is labelled *"D1 replication on L1-taxonomy adapters, exploratory, n = 60"* in every output. (c) The 60 adapters used are named by run index in the note so that, when 240 are complete, the same analysis can be re-run and the two compared. If the direction of the gap at n = 60 disagrees with D1, that is a finding the ladder needs before L1 analysis, not after.

## Two things outside the six items

**The history rewrite.** Nothing is asked of me, and I am not objecting to it. One request for the record: every commit SHA in my review documents from July 3 to today will break. The commit map should carry, for each of my memos, the old-SHA → new-SHA pairs for the commits that memo cites, so that a reader of the memo can resolve them without reconstructing the map themselves. The tag `asset1-bundle-anchor` covers the two release commits; my memos cite roughly thirty others.

**The redacted audit records.** `round-1/agent-1B-math-p2.md` and `agent-1D-ip-boundary.md` are described as mine and redacted in place, marked and dated. I have not read the redactions; I am noting that a redaction of a reviewer's record should be reviewable by the reviewer. If the redacted content is protected-corpus material, a one-line description of what was removed (not the content) is sufficient for me to confirm the record still says what I wrote it to say.

## Where this stands

Items 1, 2, 3, 4, and 6 are cleared. The queue may resume L1 on the PI's lifting the pause; the L0 gate-recording run may proceed from `d46c082` or an unchanged successor; A7b may run on the 64 with the three pins. Item 5 waits on the four card files reaching me.

*Verified this pass: packet hash; design §5 and registration deltas read verbatim from the tree; `training_interlock()` body read and the triple recomputed; all four dry-run representations plus D6 re-derived from the committed JSON and matched to the released Asset-1 llama H1 to the digit; rate arithmetic (4.45%, 3.901, 17.955, 17.02, 0.3774) recomputed; A7b Jeffreys intervals at n = 60 computed. Cards not verified: absent from the public tree. / the Director*
