# DATED AMENDMENT — Gate-Ledger Provenance After the History Rewrite

**Filed 2026-09-11 by the decider (Meridian)**, on the Director's ruling of
2026-09-11 (`docs/DIRECTOR_RULING_LEDGER_PROVENANCE_2026-09-11.md`, filed
verbatim in this repository). This file is the dated sibling of
`results/granularity/TIER_GATES.json`. **It does not edit that ledger.**

The ledger's `git_commit` field means "the commit the run was launched from,"
not "a commit that resolves today." The history of this repository was
rewritten and published on 2026-09-11, so the SHA the L0 gate recorded no
longer resolves here. This amendment carries the map, so the gate's provenance
stays checkable on the rewritten history without consulting the private map.

---

## 1. Authority

Option (a), ruled
[`docs/DIRECTOR_RULING_LEDGER_PROVENANCE_2026-09-11.md:20`]:

> **Option (a) is ruled**: leave the entry as written, add a dated sibling
> `results/granularity/TIER_GATES_AMENDMENT_2026-09-11.md` mapping each
> recorded SHA to its rewritten image with the erratum reference, signed by
> the decider.

Addition 1 [`:24`]:

> **The amendment maps both anchors, not one.** The L0 entry records
> `174ae959…` → `6c79be96…`. My Item 1 condition was anchored on `d46c082` →
> `9d23a893…`, and the gate-recording commit's message was updated by the
> rewrite to name both. The amendment should carry both pairs, with the
> sentence *"the analysis script was unchanged between these two commits"*
> restated against the **new** SHAs, so the condition remains checkable on the
> rewritten history without consulting the private map.

Addition 2 [`:25`]:

> **`record_gate` gains a `history_epoch` field going forward.** Every later
> gate records a post-rewrite SHA automatically, as the note says; but nothing
> in the ledger *marks* which epoch a SHA belongs to. Add a field —
> `"history_epoch": "2026-09-11-rewrite"` or the erratum's commit — to the
> entry schema from the L1 gate onward. The L0 entry is not edited; the
> amendment file states that L0 predates the field. This costs one line in the
> writer and means a reader in 2028 can tell which map to consult from the
> ledger alone.

Option (b) — appending the map into the ledger itself — declined [`:27`]:

> Option (b) is declined for the reason the ledger's own header gives: a
> sealed file that has been hand-appended once is no longer sealed, however
> small the append.

## 2. The ledger entry as written — not edited

The single entry in `results/granularity/TIER_GATES.json` records, verbatim:

```
"git_commit": "174ae959f2ac42d9ea2230e3ddc0f15e57ce6632"
```

alongside `tier` `L0`, `level` `L0`, `fired_at`
`2026-09-06T04:48:57.413066+00:00`, `tiers_already_unblinded` `[]`, `k` 6,
`n_runs` 240.

**The file is unchanged by this amendment**, measured immediately before and
immediately after it was filed:

| quantity | value |
|---|---|
| sha256 | `919f811cfe4cc56f1cb34d7b223c75a1514db0e4d4da261429ef8c7d1dc5b5a6` |
| git blob id | `0a1cdf832b4e19f0bffa7b50597b7ab0bd47a9ba` |
| ledger entries | 1 |

`174ae959f2ac42d9ea2230e3ddc0f15e57ce6632` does not resolve in this
repository: `git cat-file -t` returns *"could not get object info"* for it, as
it does for `d46c0822aa8504b42a76894f2313a85f5a54dd95`. That is the rewrite,
not a corrupted record — hence this map.

## 3. The map: recorded SHA → rewritten image

| recorded SHA | rewritten image | role | source line | type / ancestry |
|---|---|---|---|---|
| `174ae959f2ac42d9ea2230e3ddc0f15e57ce6632` | `6c79be96b372ceb51979d46ea3ab4958587373c2` | the gate's tree — the commit the registered L0 run was launched from, as recorded in the ledger | `docs/history-rewrite/cited-commit-map.txt:8` | commit; ancestor of `main` |
| `d46c0822aa8504b42a76894f2313a85f5a54dd95` | `9d23a893fa7f8883b4e36be0bb815380b8fba262` | the Director's Item 1 condition anchor — the commit since which the analysis script was required to be unchanged | `docs/history-rewrite/cited-commit-map.txt:62` | commit; ancestor of `main` |

Both images were opened in git this pass: `git cat-file -t` returns `commit`
for each, and `git merge-base --is-ancestor <image> HEAD` succeeds for each
(HEAD `4f22051355118584d6aabf9b56395e4b2cfeca82`, `== origin/main`).

**Erratum reference.** The rewrite is disclosed in
`docs/HISTORY_REWRITE_2026-09.md`, committed as
`6ca9c4c910b904ab465208cbb7ef97ab2beec2a8` (an ancestor of `main`). The maps
it publishes are `docs/history-rewrite/cited-commit-map.txt` (old → new, one
pair per line, 75 lines) and `docs/history-rewrite/COMMIT_MAP_BY_DOCUMENT.md`
(the same map indexed by the document that cites each SHA).

For orientation in the rewritten history:

| image | date (author) | subject |
|---|---|---|
| `9d23a893…` | 2026-09-05T14:23:05-07:00 | granularity: tokenizer from the local snapshot in D6; the queue requires the HF token (offline is not a substitute) |
| `6c79be96…` | 2026-09-05T15:35:35-07:00 | L0 re-baseline: exploratory CPU dry run, all three representations (no gate recorded) |

## 4. The Item 1 condition, restated on the new SHAs

> **`scripts/granularity_analysis.py` was unchanged between
> `9d23a893fa7f8883b4e36be0bb815380b8fba262` and
> `6c79be96b372ceb51979d46ea3ab4958587373c2`.**

Measured, not asserted — the path resolves to one and the same blob at both
commits:

```
git rev-parse 9d23a893:scripts/granularity_analysis.py
  -> 446fa8b7dc788f730f6b407311c595c66c817ed3
git rev-parse 6c79be96:scripts/granularity_analysis.py
  -> 446fa8b7dc788f730f6b407311c595c66c817ed3
```

The gate-recording commit `2bddf531f191b84b07f4991beb7f81023627051d` names
both anchors in its own message, in their pre-rewrite abbreviations:

> L0 gate RECORDED 2026-09-06 04:48:57Z (Director review Item 1): registered
> run from 6c79be9, analysis script unchanged since 9d23a89; TIER_GATES.json
> ledger entry 1; results identical to the dry run; Director review filed;
> tracker rows

Read that message in the rewritten history and the two abbreviations are
already the new SHAs — the rewrite updated quoted SHAs in messages (erratum,
`docs/HISTORY_REWRITE_2026-09.md`: 19 `main` commits whose message quoted
another commit's SHA were updated). `2bddf53…` is an ancestor of `main` and
carries the same blob `446fa8b7…` at that path.

## 5. The `history_epoch` field

| key | value |
|---|---|
| field name | `history_epoch` |
| value written | `2026-09-11-rewrite` |
| meaning | the `git_commit` in this entry belongs to the history published 2026-09-11, disclosed by erratum commit `6ca9c4c910b904ab465208cbb7ef97ab2beec2a8` |
| maps to consult for that epoch | `docs/history-rewrite/cited-commit-map.txt`, `docs/history-rewrite/COMMIT_MAP_BY_DOCUMENT.md` |
| written by | `record_gate`, `scripts/granularity_analysis.py:264-279` (constant at `:126`) |
| in force from | the L1 gate onward |

**The L0 entry predates the field and is not edited.** A ledger entry with no
`history_epoch` key is a pre-2026-09-11 record: its `git_commit` belongs to
the history that ended at the rewrite, and §3 above is its map. A reader in
2028 can tell the two apart from the ledger alone, which is the point of the
field.

Consumers are unaffected. `granularity_queue.fired_gates()`
(`scripts/granularity_queue.py:181`) reads only each entry's `tier` key, and
`require_tier_order()` (`scripts/granularity_analysis.py:242`) reads the same;
neither enumerates the entry schema, so the added key changes no gating
decision. Verified by test — see §7.

## 6. Disclosure: the analysis script at HEAD is not the gate-time blob

The ledger's `git_commit` pins the tree at fire time; the working script has
moved since. Stated plainly so no reader infers otherwise:

| blob of `scripts/granularity_analysis.py` | where |
|---|---|
| `446fa8b7dc788f730f6b407311c595c66c817ed3` | at `9d23a893…`, at `6c79be96…`, and at the gate-recording commit `2bddf53…` — the gate-time blob |
| `e2ad0ddf59ed696d4f7c231e1605e4494789218b` | at HEAD `4f22051…`, before this amendment |
| `166b4480d5cd4638f0c8073cc2000c5843f03a91` | with this amendment's writer change applied |

**Exactly one commit changed that path after the gate was recorded**, by
`git log 6c79be96..HEAD -- scripts/granularity_analysis.py`:

| commit | date | blob after | reason, from its own message |
|---|---|---|---|
| `00af9309b62a8aff35f5cf622aa131304fff7836` | 2026-09-05T21:52:23-07:00 | `e2ad0ddf…` | "Director review Items 2+3: `--representation both` documented as REPRODUCTION mode; interlock unit test (12) in the suite; `fired_gates()` parses the analysis side's ledger (it read the top-level keys and could never see a fired gate)" |

Its parent's blob at that path is `446fa8b7…`, so `00af930` is the whole of
the difference between the gate-time script and HEAD's.

**This amendment's own change to the writer adds the field and nothing else**:
`+6 / -0` lines — a constant with its four-line citation comment at `:122-126`
and one dict entry `"history_epoch": HISTORY_EPOCH,` at `:274`. No analysis
logic, no classifier, no representation, no gating path is touched. The
recorded diff is the audit of that claim.

## 7. Verification filed with this amendment

`tests/test_tier_gates_epoch.py` asserts, without touching
`results/granularity/TIER_GATES.json`:

1. `record_gate` on a redirected tmp ledger writes `history_epoch ==
   "2026-09-11-rewrite"` and a 40-hex `git_commit`, and
   `granularity_queue.fired_gates()` still reports the fired tier from that
   same ledger (the added key breaks no consumer);
2. the real ledger's entry 0 is the L0 entry with `git_commit`
   `174ae959f2ac42d9ea2230e3ddc0f15e57ce6632` and **no** `history_epoch` key —
   L0 predates the field. The test pins the entry, not the file's hash: the
   file changes by design when L1 fires;
3. this amendment exists and names both mapped pairs.

## 8. Typed block

```
RULING                  = docs/DIRECTOR_RULING_LEDGER_PROVENANCE_2026-09-11.md [filed verbatim, this repo]
RULING_OPTION           = (a) [ruling:20]
LEDGER_FILE             = results/granularity/TIER_GATES.json [not edited by this amendment]
LEDGER_SHA256           = 919f811cfe4cc56f1cb34d7b223c75a1514db0e4d4da261429ef8c7d1dc5b5a6 [sha256sum, 2026-09-11]
LEDGER_BLOB             = 0a1cdf832b4e19f0bffa7b50597b7ab0bd47a9ba [git hash-object, 2026-09-11]
LEDGER_ENTRIES          = 1 [TIER_GATES.json]
L0_RECORDED_SHA         = 174ae959f2ac42d9ea2230e3ddc0f15e57ce6632 [TIER_GATES.json ledger[0].git_commit]
L0_FIRED_AT             = 2026-09-06T04:48:57.413066+00:00 [TIER_GATES.json ledger[0].fired_at]
MAP_GATE_TREE           = 174ae959f2ac42d9ea2230e3ddc0f15e57ce6632 -> 6c79be96b372ceb51979d46ea3ab4958587373c2 [docs/history-rewrite/cited-commit-map.txt:8]
MAP_ITEM1_ANCHOR        = d46c0822aa8504b42a76894f2313a85f5a54dd95 -> 9d23a893fa7f8883b4e36be0bb815380b8fba262 [docs/history-rewrite/cited-commit-map.txt:62]
IMAGES_ARE_COMMITS      = both [git cat-file -t]
IMAGES_ANCESTOR_OF_MAIN = both [git merge-base --is-ancestor <image> HEAD]
SCRIPT_BLOB_AT_ANCHORS  = 446fa8b7dc788f730f6b407311c595c66c817ed3 at 9d23a893 AND at 6c79be96 [git rev-parse <c>:scripts/granularity_analysis.py]
CONDITION_ITEM1         = scripts/granularity_analysis.py unchanged between 9d23a893 and 6c79be96 [equal blob, above]
GATE_RECORDING_COMMIT   = 2bddf531f191b84b07f4991beb7f81023627051d [names both anchors in its message]
SCRIPT_BLOB_AT_HEAD     = e2ad0ddf59ed696d4f7c231e1605e4494789218b [git rev-parse HEAD:scripts/granularity_analysis.py, before this amendment]
POST_GATE_COMMITS       = 1 -> 00af9309b62a8aff35f5cf622aa131304fff7836 [git log 6c79be96..HEAD -- scripts/granularity_analysis.py]
ERRATUM                 = docs/HISTORY_REWRITE_2026-09.md @ 6ca9c4c910b904ab465208cbb7ef97ab2beec2a8 [git log -1 -- <path>]
HISTORY_EPOCH           = 2026-09-11-rewrite [scripts/granularity_analysis.py:126]
EPOCH_IN_FORCE_FROM     = the L1 gate [ruling:25]
L0_HAS_EPOCH_FIELD      = NO — predates the field, entry not edited [ruling:25]
WRITER_DIFF             = +6 / -0 lines, scripts/granularity_analysis.py [git diff --numstat]
REPO_HEAD               = 4f22051355118584d6aabf9b56395e4b2cfeca82 == origin/main [git rev-parse]
SIGNED                  = the decider (Meridian), 2026-09-11
```

---

*Filed under L-006 by the decider (Meridian), 2026-09-11. Every SHA in this
file was opened in git or read from the published map at the cited line; no
value is restated from memory.*
