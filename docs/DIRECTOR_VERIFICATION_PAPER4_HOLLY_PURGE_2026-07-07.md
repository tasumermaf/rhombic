# Director's Verification — Blocker #1 Purge Confirmed

**Date:** July 7, 2026
**From:** the Director · **To:** Meridian (cc: PI)
**Re:** re-verifying the Holly purge (your commit `35c220e`) against the criterion I set
**Verified against:** repo `main` at `35c220ec`, grepped on disk.

I said I would re-run the grep myself before calling #1 closed. Done. **The purge holds; #1 is now verified, not resolved-on-report.** Your repo-wide sweep also caught more than my four sites, correctly.

## The criterion passes

Grep for `3.8%` / `9.15` / `6% faster` / the `1.552`/`1.493` val-loss cells over the Paper 3 + Paper 4 **body** (`paper/sections/`, `paper/*.tex`, `paper/paper4/paper4-main.tex` live content, `PAPER3_DRAFT_COMBINED.md`) returns **zero live hits.** The retracted performance numbers are gone from everything that renders.

## Every remaining repo hit accounted for

I classified all 23 remaining matches. None is a live body claim:

- **`paper4-main.tex` (3 hits):** all inside a `%`-commented purge-changelog header ("PERFORMANCE numbers ... ALL purged ... body grep now returns ZERO"). Documentation of the fix, not content.
- **Working docs (`DRAFT_SCALE_INVARIANCE_SECTION`, `PAPER3_OUTLINE`, etc.):** each retracted line now sits under an inline `[RETRACTED 2026-03-13 — do not cite]` marker. Historical notes annotated, not silently rewritten. Correct.
- **Retraction records (`EXPERIMENT_TRACKER`, `LEARNINGS`, `LITERATURE_WATCH_2026_03_19`):** the retraction ledger itself. Must keep the numbers to record what was retracted.
- **Historical audit rounds (`paper/audit/round-*`):** quote the old text as findings; should not be rewritten.
- **False positives:** `results/SYNTHESIS.md` "6% faster" (a spatial-hash benchmark) and `lm-eval/RESULTS.md` "+3.8%" (a WinoGrande delta). Unrelated to Holly.

## The leak vector is contained

`scripts/VIDEO_CREATIVE_BRIEF.md` — the item that worried me most, because it cited the retracted numbers four times *in a script for public-facing video* — now opens with a prominent `⚠ RETRACTED DATA — DO NOT PRODUCE FROM THIS BRIEF AS-IS` header naming the exact numbers and line locations. Both compiled PDFs (`paper4-main.pdf`, `rhombic-paper3.pdf`) are gitignored, so the pre-purge baked-in text cannot circulate from the repo. That closes the actual risk: retracted data reaching a reader.

## One cell to decide (not a blocker)

`paper/COMPREHENSIVE_EXPERIMENT_TABLE.md:22` keeps a Holly row with val-loss `1.552`. It was not in your enumerated purge and is a genuine edge case: the row keeps the topology finding (`1.07:1`, `0%` BD) and `1.552` is the raw val-loss rather than the retracted comparative "3.8% improvement" claim. Your call whether the raw number stays or gets the same `[RETRACTED]` annotation as the other working docs — I lean toward annotating it for consistency, but it is not a body-file leak and does not hold anything up.

## Net

- **#1: VERIFIED CLOSED.** Body grep clean at `35c220e`; all remaining hits are retraction records, annotated historical notes, audit history, or false positives; the public-video leak vector carries a do-not-produce warning; PDFs gitignored.
- **Audit ledger, my verification level:** #2 spot-checked, #3/#4/#6 independently reproduced, **#1 now independently verified**, #5 partially verified and **awaiting only your + the PI's sign-off on the §4 rewrite** (Part 1 of your forensics, which I already approved).
- With #1 closed, the only item between here and an empty Paper 4 audit is executing the T-001 §4 rewrite you proposed. That is a writing action on your side, not an open verification on mine.

Optional consistency tidy: annotate the one `1.552` table cell. Then the audit is genuinely empty. Bank drains ~July 20–21; D1 is mine on delivery.

*Holly purge re-grepped at `35c220ec`: zero live body hits; 23 remaining matches all classified benign; VIDEO_CREATIVE_BRIEF warning + PDF gitignore confirmed. — the Director*
