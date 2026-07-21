# Director's Sign-Off — Asset-1 Bank, Independently Re-Derived (v2, re-versioned)

> Filed verbatim by Meridian, 2026-07-21. This is the Director's own
> re-versioning of his Jul 21 sign-off, issued alongside his paper-v0 review:
> the D3 distance-baseline note is corrected from "CV-seed sensitivity" to a
> naive-vs-group-aware fold-scheme difference (his from-scratch re-run used
> the naive scheme and reproduced the report's naive block exactly). The
> original remains on file untouched as DIRECTOR_SIGNOFF_ASSET1_2026-07-21.md
> per the dated-amendment doctrine.
>
> Filing note (verbatim fidelity): the closing one-line summary below still
> reads "one CV-seed note on the D3 distance baseline" — residual v1 language
> the body's dated correction supersedes. Filed as received; flagged, not
> edited.

---

Date: July 21, 2026 From: the Director · To: Meridian (cc: PI) Re: the 480/480 delivery packet, regraded from the per-item verification bundle Verified against: ASSET1_VERIFY_BUNDLE_2026-07-21.zip (sha256 `c1891d5…`, matches the cover note), timestamped by commit `638f4a8`. All 16 files match the SHA256SUMS manifest. I recomputed every headline from the per-item data myself; where I could, I re-ran the analysis from the feature matrices rather than from the saved predictions. Every headline reproduces. The delivery is sound.

## Integrity

The archive's own sha256 matches the cover note, and all 14 Tier-1 files plus both Tier-2 `.npz` matrices match SHA256SUMS byte-for-byte. So what I regraded is provably the bytes commit `638f4a8` pins. This is the anchor everything below rests on.

## H1 — task identity, re-run from features

Re-derived two ways. From the per-family confusion matrices, all eight LOO accuracies reproduce exactly. Raw qwen 0.0792 / llama 0.1375 both FAIL the 1.5x-chance lock; canonical and both vocab_signature variants read 1.0000 (PASS); permutation p is 0.000999 throughout; and the lock logic is correct (raw's tiny p with sub-threshold accuracy is exactly the weak-signal case the 1.5x bar is designed to reject). Then the stronger check: I re-ran the canonical LOO-SVM from scratch on the 88,704/50,688-dim feature matrix (precomputed-Gram LOO, C=1.0) and got 1.0000 for both families — the ceiling result is real (not a summary artifact), and the heterogeneity guard (ratios 1.00-1.46, trigger 3.7) confirms it is not a variance artifact. The reading holds: task identity is in the weights, but only legible once the GL(r) gauge is removed.

## H2 — cross-family transfer (re-run end-to-end)

A pre-registration that flips via a control added in review is the highest motivated-reasoning risk in the set, so I re-ran the entire H2 pipeline from the spectrum and probe feature matrices, applying the standardization myself:

* Triviality control reproduces: the family-identity probe reads 1.0000 on raw (families are perfectly separable by scale) and collapses to 0.1521 (spectrum) / 0.0000 (probe) under per-family standardization. I confirmed in source that `familywise_standardize` is unsupervised (per-family z-scoring, task labels never touched), so it removes family scale without any capacity to manufacture task structure.
* Standardized transfer reproduces to the digit: spectrum 0.7833 (qwen->llama) / 0.7375 (llama->qwen), probe 0.7792 / 0.7792, raw baselines 0.1667 / 0.1667 / 0.1750 / 0.2167, and the binomial p-values (7.70e-98, 1.20e-84, 1.37e-96, 1.37e-96) all match.

The refutation is genuine: the pre-registered "transfer fails" prediction was a covariate-shift artifact, and once the family-scale shift is removed, task structure transfers across a 1.5B and a 1B model of different lineages at ~74-78%. The control added in review is exactly what prevented a false confirmation, which is the pre-registration working as designed rather than failing.

## D2 / D3 / D-aux (re-derived)

* D2: all 14 per-kind mean penalties reproduce exactly from the 360 per-eval rows/family against the native reference. Cross-task bridge swap ~0.0000, full-permuted +2.8086 (qwen) / +3.8365 (llama). The backbone is the sole load-bearing structure; the trained bridge is nearly free to swap.
* D3: conflict rate 0.858 reproduces from the 240 raw labels; the weight-only full AUC reproduces (qwen 0.995 exact; llama 0.952, which matches the report's own naive-CV number, vs its 0.962 group-aware). The distance-only baseline came out 0.686/0.667 vs the reported 0.675/0.713, a few points of CV-seed sensitivity on a 2-feature model over 120 points, not a discrepancy in the load-bearing claim (full >> distance, full near ceiling, both hold).
* D-aux: pooled r = 0.300, qwen 0.418, llama 0.337 all reproduce from the 480 (dev_mean, final_gap) pairs; the honest 0.888 -> 0.300 shrink stands.

## The two §7 record items — both approved

* D3 pre-declaration + amendment: temporal integrity holds. The declaration (~16:50Z) and the amendment to uniform-no-stratification (~19:4xZ) both explicitly precede Step 5 with zero labels in existence, and the amendment is the conservative choice: amending the declaration rather than modifying frozen, adversarially-reviewed analysis code after the bank exists, with realized cell coverage reported descriptively. Approved as a clean dated amendment (L-006 / R10); no objection.
* D3 labels runner + verifier note: approved. The fresh-context verifier caught two real pre-launch defects (the flat-vs-nested loader bug, reproduced to an actual crash, and the machine-absolute manifest path), and the native-loss shortcut was verified at 0.00000% divergence against 36 fresh D2 evaluations. This is the maker-grader separation applied to the one component that had no shipped runner, which is exactly where it was most needed.

## Net

* Every headline independently re-derived and confirmed. H1 and H2 re-run from the feature matrices (not saved predictions); D2/D3/D-aux recomputed from per-item rows; integrity anchored on a matching manifest.
* One minor, non-load-bearing note (corrected 2026-07-21): my from-scratch D3 re-run used a plain StratifiedKFold, which is the report's NAIVE scheme, and reproduced its naive block exactly (distance 0.686/0.667, full 0.995/0.952). The headline uses group-aware StratifiedGroupKFold (distance 0.675/0.713). So the baseline gap is a naive-vs-group-aware fold-scheme difference, not the CV-seed sensitivity I first called it. The conclusion is unchanged (pin the fold configuration and report the 2-feature baseline's fragility, since the margin-over-distance CI's lower end depends on it), but the diagnosis was misnamed. The full-model result and the margin's existence are not in question.
* Both §7 record items approved. Nothing outstanding on my side.

This is the result the program should be judged on: canonicalization makes task identity perfectly legible (H1), the boldest pre-registered prediction was refuted by its own control (H2), the backbone is the load-bearing structure (D2), weights predict merge conflict post-hoc (D3), and the pilot correlation shrank honestly at scale (D-aux). Verified from the per-item data, not the summary. It is ready to be written up.

Asset-1 bank regraded from the per-item bundle at `638f4a8` (manifest-verified): H1 canonical LOO 1.0000 re-run from features; H2 standardized transfer 0.74-0.78 re-run end-to-end with the triviality control confirmed unsupervised; D2/D3/D-aux recomputed exact from per-item rows; both §7 items approved; one CV-seed note on the D3 distance baseline. / the Director
