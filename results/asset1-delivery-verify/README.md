# Asset-1 Delivery Verification Bundle

**For:** the Director's maker–grader regrade of the 480/480 delivery packet
(`docs/ASSET1_BANK_DELIVERY_2026-07-20.md`).
**Purpose:** the per-item data behind every headline number, so each is
re-derivable from raw per-run / per-eval / per-pair values — not from the
report's summary tables. Assembled 2026-07-20; verify integrity with
`SHA256SUMS`.

Nothing here is an adapter or a raw (6.5M-dim) feature vector. Tier 1 is
kilobytes-to-MB of result JSON/CSV; Tier 2 (`features/`) is the reduced
feature matrices (tens of MB, float32).

## Tier 1 — per-item result data

| file | what the Director re-derives from it |
|------|--------------------------------------|
| `d1_results.json` | H1: per-family 6×6 confusion matrices (`representations.<rep>.confusion_matrix.rows_true_cols_pred`) → recompute LOO accuracy + per-class recall; permutation summary (`.permutation`: n=1000, null_mean/max/p99/p_value). H2: `h2_cross_family` transfer + family-identity-probe tables (raw & family_standardized), binomial p, margins, verdict. Per-run feature norms where emitted. **Note:** the per-run *feature matrices* for a from-scratch SVM/permutation re-run are in Tier 2, not here — this file carries the confusion counts and the null summary, not the 1000-shuffle array. |
| `d2_results_qwen2.5-1.5b.json`, `d2_results_llama3.2-1b.json` | 360 per-eval rows each: `kind`, `task_recipient`, `recipient_run_index`, `assembled_sha256`, `val_loss`. Recompute per-kind mean penalty = mean(val_loss − native) grouped by kind; confirm `permuted_deviation` ~0 vs `permuted` ~3–4 nats. |
| `d2_swap_plan.json` | `assembled_sha256` per eval — the bank-integrity anchor Stage B re-verified. |
| `d3_labels.json` | 240 per-pair labels: `degradation`, `merged_ppl_a/b`, `native_ppl_a/b`. Primary conflict = degradation ≥ 0.05 (either endpoint). |
| `d3_pairs.json` | 240 per-pair feature dicts. **Exact feature sets (from `asset1_d3_merge.assemble_matrix`, so you bit-match the AUCs):** `distance` = `[cos_distance, l2_distance]`; `full` = `[cos_distance, l2_distance]` + the 4 aggregates `[angle_mean_weighted, angle_mean_unweighted, chordal_rms_weighted, chordal_rms_unweighted]` + the per-module vector `module_l2` (len 112 qwen / 64 llama) + the per-module vector `module_angle_mean` (same len, NaN→0.0). `module_chordal_rms` and `module_weight` are carried but NOT in the `full` matrix. |
| `d3_report.json` | per-fold OOF scores, group-aware & naive AUC, bootstrap CIs, `binarization` block (rule_used, degenerate=false, frac_positive). |
| `daux_run_table.csv` | 480 per-run rows: `dev_mean`, `final_gap` (+ dev_max, gap_auc, update_mag). The pairs behind r; recompute pooled + per-cell Pearson r. |
| `daux_report.json` | pooled + per-cell Pearson/Spearman r with bootstrap CIs; `step0_control` = 0.0. |
| `bank_manifest.json` | 480-COMPLETE manifest (status_counts + per-run ids/status). |

## Tier 2 — reduced feature matrices (`features/`)

`features_<family>.npz` per family, reusing the FROZEN D1 feature functions
(`_canonical_feature`, `h2_features_for_run` — no reimplementation):

- `X_canonical` (n, 88704 qwen / 50688 llama) — H1 representation at ceiling.
- `X_spectrum` (n, 384), `X_probe` (n, 12672) — the H2 representations; apply
  per-family z-standardization and re-run transfer + the family-identity probe
  end-to-end (the strongest check that the H2 refutation is not a
  standardization artifact).
- `y_task` / `y_int` / `run_index` / `tasks` / `params` (proj_dim=16,
  proj_seed=0, sigma_slots=24, n_depth_bins=4, svm_c=1.0).

Sizes: `features_qwen2.5-1.5b.npz` 90.6 MB, `features_llama3.2-1b.npz` 56.7 MB
(float32, compressed). Larger than git should carry — transferred out-of-band
(HF or direct), not committed to the repo tree.

`vocab_signature` (A3 arm #3) is **not** exported here — it needs the base-model
unembedding, and it only corroborates canonical's ceiling; available on request.

## §7 record items (for the Director's ruling)

- `D3_PAIR_DESIGN_PREDECLARATION_2026-07-20.md` — the amended-to-uniform sampler
  declaration. The dated amendment is timestamped before any label existed
  (temporal integrity is the question).
- `asset1_d3_labels.py` — the "external" Step-6 labels runner (D3 AUC rests on
  its labels).
- `asset1_d3_labels_verifier_note.md` — the fresh-context adversarial verifier's
  report on that runner: two blocking defects caught and fixed before the GPU run.

## Reproduction environment

falco conda env; D1/D3/D-aux selftests PASS at the tooling commit. Feature
extraction and all Tier-1 outputs are deterministic (no wall-clock, fixed
seeds). Recompute recipe per analysis is in the table above.
