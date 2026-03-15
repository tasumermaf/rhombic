# Round 1: Hub Validation

> Hub independently confirms or rejects each agent finding against raw data.

## Agent 1C (Paper 1 Math) — VALIDATED

- **24 claims checked, 1 discrepancy (MINOR)**
- F-01-02: Algebraic connectivity range 2.3-2.5x stated in text, but Table 1 shows 2.55x
  - **Hub verdict: ACCEPTED.** The table data is authoritative; the text range should technically be 2.3-2.6x. However, Paper 1 is LOCKED and this is not egregious (abstract's "2.4x" is correct, table has exact values).

## Agent 1D (IP Boundary) — VALIDATED

- **1 CRITICAL finding, 2 informational**
- F-01-01: Paper 2 Exp 6 lists 15 of 24 corpus values as edge weights
  - **Hub verdict: ACCEPTED.** Independently verified at P2 lines 652-656. Confirmed each value matches the Continental Attribution Table. **FIX APPLIED:** Removed specific edge-value triplets, kept divisibility relationships. Both EN and IT versions fixed.
- F-01-02 (IP LOW): Statistical characterization within policy — no action.
- F-01-03 (IP INFO): "transliteration protocol" phrasing acceptable — no action.

## Agent 1A (Paper 3 Math) — VALIDATED

- **101 claims checked, 12 discrepancies (6 MAJOR, 6 MINOR)**

### MAJOR findings — all independently verified against raw data:

- **F-01-05/06/07 (H-ch8 stale data):** Hub verified H-ch8-results-hermes.json step 10000: fiedler_mean=0.0944, val_loss=0.4024. Paper had 0.085 and 0.4022 from in-progress readings. **ACCEPTED — FIXED.** Table 2 updated (0.085→0.094, 0.4022→0.4024). Fiedler range updated throughout (0.085-0.095→0.092-0.095). Strengthens the spectral attractor claim — tighter band.

- **F-01-08 (Qwen q_proj ratio):** Hub verified DRAFT_SCALE_INVARIANCE_SECTION.md line 74: q_proj=22,477:1. Paper had 34,000:1. No source anywhere contains 34,000 for q_proj. **ACCEPTED — FIXED.** Section 6.1 rewritten with correct values.

- **F-01-09 (C-002 v_proj misattributed):** Hub verified PAPER3_FIGURE_INVENTORY.md line 28: k_proj=46,570:1, v_proj=24,462:1. Paper grouped both at 46,570:1. **ACCEPTED — FIXED.** Section 6.1 now separates k and v values correctly. Hierarchy k>v>q>o preserved.

- **F-01-10 (Correlation Fiedler Holly):** Hub verified FIEDLER_METRIC_NOTE.md line 35: Holly=1.002. Paper claimed "converges to ~0.10 across all three scales" but Holly is 10× higher. **ACCEPTED — FIXED.** Rewritten to distinguish cybernetic (~0.10) from non-cybernetic (1.002) regimes.

### MINOR findings — disposition:

- **F-01-01 (experiment count):** "15 experiments" vs Table 1 count. **DEFERRED to Round 2** (internal consistency). The count depends on how exp1a-e are tallied.
- **F-01-02 (convergence band delta):** 10%→11.2%. **ACCEPTED — FIXED.**
- **F-01-03 (C-001 at 4K):** Step mixing. **ACCEPTED — FIXED.** Now reads "Final validation losses are 0.4178 (C-001, 4K steps)..."
- **F-01-04 (n=4 Fiedler rounding):** 0.0918→0.092. **ACCEPTED — no fix needed.** Acceptable 2-sig-fig rounding. 1,020× uses precise value.
- **F-01-11 (IC step 200):** 0.005/83%→0.006/79%. **ACCEPTED — FIXED.**
- **F-01-12 (IC/RD per-pair vs total):** Ambiguous units. **DEFERRED to Round 2** for deeper investigation. The 2.9:1 figure appears consistently and may use per-pair values from a different extraction.

## Agent 1B (Paper 2 Math) — VALIDATED

- **97 claims checked, 6 discrepancies (2 MAJOR, 4 MINOR)**

### MAJOR findings:

- **MAJOR-1 (Permutation control):** Paper claims 2.68±0.68, p≤0.001. No saved output existed. Hub re-ran `scripts/permutation_control.py` — output: **mean=2.68, std=0.68, p=0.001.** Paper's values exactly match. **ACCEPTED — VERIFIED.** Output saved to `results/paper2/permutation_control_results.txt`.

- **MAJOR-2 (SA parameters):** Code uses n_restarts=100, max_iters=5000. Paper said "20 restarts of 1,000 iterations." **ACCEPTED — FIXED.** Both EN and IT versions updated to "100 restarts of 5,000 iterations."

### MINOR findings — disposition:

- **MINOR-1 (test count 256→312):** Stale number. **DEFERRED to Round 5** (writing quality). Higher count strengthens claims.
- **MINOR-2 (spectral gap 0.0314→0.0313):** Off by 1 in last digit. **ACCEPTED — FIXED** in Paper 2 EN. IT version did not contain the value.
- **MINOR-3 (path advantage 31%):** Adequate with "approximately" hedge. **ACCEPTED — no fix needed.**
- **MINOR-4 (Table 1 omits 2 distributions):** Editorial choice. **DEFERRED to Round 4** (completeness review).

## Round 1 Summary

| Metric | P1 (1C) | P2 (1B) | P2 (1D) | P3 (1A) | Total |
|--------|---------|---------|---------|---------|-------|
| Claims checked | 24 | 97 | 3 | 101 | 225 |
| CRITICAL | 0 | 0 | 1 | 0 | 1 |
| MAJOR | 0 | 2 | 0 | 6 | 8 |
| MINOR | 1 | 4 | 0 | 6 | 11 |
| Fixed this round | 0 | 3 | 1 | 8 | 12 |
| Deferred | 0 | 2 | 0 | 2 | 4 |
| No action needed | 1 | 1 | 2 | 2 | 6 |

**No CRITICAL findings remain open.** All MAJOR findings resolved (7 fixed, 1 verified correct). Paper 1 remains LOCKED — its single MINOR finding does not warrant unlocking.
