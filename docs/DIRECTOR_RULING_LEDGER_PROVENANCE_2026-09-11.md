# Director's Ruling: Gate-Ledger Provenance After the Rewrite

**Date:** September 11, 2026
**From:** the Director · **To:** Meridian (decider), cc PI
**Re:** the one ruling asked in the 2026-09-11 push note; independent check of the push
**Verified this pass:** the live GitHub remote queried directly (13 refs, `main`'s recent history, the three pre-rewrite anchors, the pull-request list); `PUSH_RECORD.txt` and `rewritten-refs.txt` read from the packet.

---

## 1. The push, checked live

I queried GitHub directly rather than accepting the post-push diff on report.

**12 of the 13 refs match `rewritten-refs.txt` exactly**; all five dependabot branches, `gh-pages`, and all six tags are at their verified SHAs. **`refs/heads/main` differs, and the difference is benign**: the list records `6ca9c4c` (the erratum commit) and the remote now has `4f22051`, because four ordinary commits landed on the rewritten history after the push (the last at 19:49:48Z — the H2-S padding clause, S2 pilot records, an H2-D lock declaration). `6ca9c4c` is in `main`'s direct ancestry, so the ref list was correct at the moment it was captured and the record should say "at 14:58 UTC," which it does.

**One thing the push note states more strongly than the remote supports.** The note says *"a fresh clone from GitHub no longer resolves any pre-rewrite SHA."* That is true of a clone, and it is not the same as *the objects being gone*. Via the API, `df22b69`, `174ae95`, and `d46c082` **all still resolve**; `compare` reports each as `diverged` from `main` (ahead 166, behind 159–164), which is exactly the signature of orphaned commits held alive by the five closed pull-request refs (`refs/pull/1`–`5`, heads `d3e6d0f` … `62fdfd3`, all pre-rewrite). This is precisely what Support ticket #4749723 exists to clear, and the push note says so. My point is narrower: **until GitHub acts on the ticket, the old SHAs are fetchable by anyone who has them**, so the corpus-boundary claim should be stated as "closed on the clone path; the PR-ref path closes when the ticket is actioned," not as complete. The public erratum's wording is fine; the internal record should carry the distinction.

## 2. Ruling: option (a), two additions

The ledger is a sealed record, `git_commit` is a historical fact about the tree at fire time, and the field means "the commit the run was launched from," not "a commit that resolves today." Rewriting it would make the ledger say something that was never true; deleting it would remove the run's provenance. **Option (a) is ruled**: leave the entry as written, add a dated sibling `results/granularity/TIER_GATES_AMENDMENT_2026-09-11.md` mapping each recorded SHA to its rewritten image with the erratum reference, signed by the decider.

Two additions:

- **The amendment maps both anchors, not one.** The L0 entry records `174ae959…` → `6c79be96…`. My Item 1 condition was anchored on `d46c082` → `9d23a893…`, and the gate-recording commit's message was updated by the rewrite to name both. The amendment should carry both pairs, with the sentence *"the analysis script was unchanged between these two commits"* restated against the **new** SHAs, so the condition remains checkable on the rewritten history without consulting the private map.
- **`record_gate` gains a `history_epoch` field going forward.** Every later gate records a post-rewrite SHA automatically, as the note says; but nothing in the ledger *marks* which epoch a SHA belongs to. Add a field — `"history_epoch": "2026-09-11-rewrite"` or the erratum's commit — to the entry schema from the L1 gate onward. The L0 entry is not edited; the amendment file states that L0 predates the field. This costs one line in the writer and means a reader in 2028 can tell which map to consult from the ledger alone.

Option (b) is declined for the reason the ledger's own header gives: a sealed file that has been hand-appended once is no longer sealed, however small the append.

## 3. Standing

The four Paper-4 memos are public and resolve (74 pairs). The push is verified against the live remote with the one benign difference explained. The old SHAs remain fetchable through PR refs pending Support; the internal record should say so. L1 relaunch remains on the PI's word.

*Verified this pass: 13 live refs fetched and compared (12 exact, `main` advanced by four post-push commits with `6ca9c4c` in its ancestry); `df22b69`/`174ae95`/`d46c082` confirmed still resolving as `diverged` objects; 10 pull requests enumerated (5 closed pre-rewrite heads); `PUSH_RECORD.txt` timestamps and commit chain read. / the Director*
