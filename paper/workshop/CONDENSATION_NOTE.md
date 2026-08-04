# CONDENSATION_NOTE — Asset-1 workshop condensation (final assembly)

Assembled: 2026-08-04. Output: `rhombic-asset1-condensed.tex` (this directory).
Both lens drafts (`draft_finding-first.tex`, `draft_discipline-first.tex`) are
retained unmodified. Nothing is committed by this assembly.

## Provenance (typed)

```
PARENT_PAPER_PATH        = C:/falco/rhombic/paper/rhombic-asset1.tex
PARENT_PAPER_SHA256      = 856eab22ab4138ca1f168816ff7d74c7a10c1b0e6913bb86c7eb8ff7fc964b85 [sha256sum, 2026-08-04; working tree clean vs git]
PARENT_LAST_COMMIT       = 42b0e097c91413cdf69ebe29a67c25546c411577 (2026-07-29) [git log -- paper/rhombic-asset1.tex]
REPO_HEAD_AT_ASSEMBLY    = 55d7bb209b5512bf58af300ebc9827d025b3d90e [git log -1]
WINNER_LENS              = finding-first (base draft)
RUNNER_UP_LENS           = discipline-first (source of 10 judge-selected grafts)
JUDGE_SCORES             = not provided to this assembly step — held in the orchestrator's judge record; only the graft list was passed down [omitted rather than approximated]
TYPED_TABLE_KEYS_CARRIED = 57 (NUMBERS USED trailer of rhombic-asset1-condensed.tex)
NUMBER_AUDIT_DISCREPANCIES = 0 found, 0 to fix [orchestrator number-audit report]
NUMBER_AUDIT_ORPHANS     = 1 — \date{August 2026} (document metadata, not a data claim); resolved with a dated comment: assembly date 2026-08-04, parent dated July 2026
SCOPE_FINDINGS           = 9 (3 major, 3 moderate, 2 minor, 1 note) — ALL FIXED (see below)
GRAFTS_APPLIED           = 10 of 10 (2 mandatory fidelity fixes + 8 judge selections)
```

## Grafts applied (from draft_discipline-first.tex)

1. MANDATORY — experiment-card provenance gap disclosed in Methods layer (1).
2. MANDATORY — Acknowledgments: Director "isolated from the analysis process, not blinded to results."
3. H2 closer: cost-of-the-control sentence appended.
4. D-aux framing: "shrink is a success of the protocol" sentence.
5. D3 third certification: invertible bridges are a gauge on the update column space (parent-attested lines 1201-1206).
6. Interlock `--allow-partial-bank` escape-hatch disclosure.
7. D3 amendment rationale: "lesser deviation versus modifying frozen analysis code."
8. Trailer conventions: absolute-path SOURCE locators; EXTRA-marking rule; 84.5% guardrail line.
9. Training-config alpha restored: "rank-24 adapter ($\alpha = 16.0$)".
10. Repro pointers: explicit HF dataset URL + `paper/audit/` ledger pointer.

## Scope-audit findings fixed (9/9)

| # | Severity | Finding | Fix in condensed.tex |
|---|----------|---------|----------------------|
| 1 | major | "every headline independently re-derived" broader than parent (F005/F036 regression) | Abstract, hook (iii), and Methods now read "re-derived or verified"; parent's scope-of-re-derivation qualifier mirrored at all three sites; D3 group-aware-verified-against-pinned-report stated in Methods |
| 2 | major | Card provenance gap dropped | Graft 1 sentence inserted in Methods layer (1) |
| 3 | major | Director not-blinded disclosure dropped (F018) | Graft 2 sentence in Acknowledgments, plus three citation checks taken-on-report |
| 4 | moderate | proj_seed 0 / H1-ceiling seed-sensitivity sentence dropped (F016) | Parent sentence appended to one-recipe limitation |
| 5 | moderate | Unscoped "canonicalize before you conclude anything" directive | Hook (i) scoped to "regimes like ours---coarse task labels within controlled families" with pointer to Limitations |
| 6 | moderate | Per-run rates falsely claimed as parent-bundle numbers, no SOURCE | Header amended ("or a cited measured artifact"); % SOURCE: RATE_EXTRACT.md lines 12-14 at use site + EXTRA-marked trailer entries |
| 7 | minor | "every finding resolved" overstates ledger | Now "every finding was dispositioned---fixed or noted---and the audit stopped at its round hard-cap, not at exhaustion"; 1 plausible finding disclosed |
| 8 | minor | D3 uniform-pair-population scope sentence dropped | Appended to Limitations fragilities block |
| 9 | note | D3 declaration timeline asserted without self-dated-provenance caveat | Parenthetical added in Methods (single file, document's own dated entries, sole commit 2026-07-21); parent lines 694-699 cited |

## Submission gate

**A the-adversary v2.0.2 adversarial ladder pass on `rhombic-asset1-condensed.tex`
is REQUIRED before submission and has NOT run yet.** This file is assembled,
not audited. Do not submit, post, or circulate until that pass completes and
its findings are dispositioned.

Also pending before submission: venue CFP class refit (layout is generic
10pt article), venue author block, and the \date field.

## Program guardrails honored

- 84.5% (retracted fingerprinting figure) appears nowhere; guardrail line embedded in the trailer. The verified P3 figure (72.3%) belongs to Paper 3 and also does not appear.
- No Sacred Language / isopsephy / corpus content.
- No claims broader than the parent paper; all nine scope regressions narrowed to or below parent scope.
- Rates cited from results/asset1-bank/RATE_EXTRACT.md and labeled [measured].

=== VERIFIED STATE ===
H1_RAW_LOO             = 0.0792 (qwen) / 0.1375 (llama), chance 0.1667 [parent rhombic-asset1.tex Table tab:h1-results; verification bundle @ 638f4a8]
H1_CANONICAL_LOO       = 1.0000 both families, perm p = 0.000999; lock = acc > 0.2500 AND p < 0.01 [same]
H2_FAMILY_PROBE_RAW    = 1.0000 (spectrum and probe, chance 0.5000) [parent H2 tables; bundle]
H2_TRANSFER_STD        = 0.7375-0.7833 (spectrum 0.7833/0.7375; probe 0.7792/0.7792); binom p <= 1.20e-84 [parent lines 954-957; bundle]
D2_BACKBONE_PRESERVING = all swap kinds < +0.001 nats [parent D2 penalty matrix; 360 per-eval rows/family in bundle]
D2_FULL_PERMUTATION    = +2.8086 (qwen) / +3.8365 (llama) nats [same]
D3_AUC_GROUPAWARE      = 0.995 [0.983, 1.000] (qwen) / 0.962 [0.898, 0.999] (llama); margin over distance +0.320 / +0.249, CIs exclude 0 [parent D3 tables; 240 per-pair dicts in bundle]
DAUX_SHRINK            = pilot r 0.888 -> pooled r 0.300 [0.175, 0.415], n = 480 [parent D-aux table; 480 per-run pairs in bundle]
BANK                   = 480 = 2 families x 6 tasks x 40 seeds; manifest closed 480/480 COMPLETE 2026-07-20T16:21:19Z [parent; bank manifest]
RATES_MEASURED         = llama3.2-1b 30.56 min/run (n=240); qwen2.5-1.5b 51.72 (n=240); blended 41.14 (n=480) [C:/falco/rhombic/results/asset1-bank/RATE_EXTRACT.md lines 12-14 — NOT in parent bundle]
AUDIT_LADDER           = 117 confirmed (1 blocker / 30 major / 53 minor / 33 note) + 1 plausible; 7 rounds, 19 lenses; stop = hard-cap [C:/falco/rhombic/paper/audit/round-2/ADVERSARY_LADDER_LEDGER_ASSET1.md lines 5-6]
BUNDLE_ANCHOR          = rhombic commit 638f4a8; archive SHA-256 c1891d50fc7c48c27d6fb606d667f1f1120105ed55cb7414c8e4b3d874acfadd; 16/16 files match SHA256SUMS [parent Reproducibility]
PARENT_FILE            = paper/rhombic-asset1.tex; SHA-256 856eab22ab4138ca1f168816ff7d74c7a10c1b0e6913bb86c7eb8ff7fc964b85; last commit 42b0e097 (2026-07-29); clean at assembly [sha256sum + git, 2026-08-04]
V2.0.2_LADDER_PASS     = REQUIRED before submission — NOT RUN as of 2026-08-04
=== END VERIFIED STATE ===
