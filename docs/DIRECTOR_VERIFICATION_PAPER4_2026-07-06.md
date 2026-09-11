# Director's Verification Pass — Paper 4 Ratios + BM-001 Parity

**Date:** July 6, 2026
**From:** the Director · **To:** Meridian
**Charter:** the verification pass I owed since the F4 ruling — independently re-derive Paper 4's load-bearing ratios and BM-001 parity from raw artifacts on local disk, not from summary docs or from your verification pass.
**Verified against:** repo `main` at `8194b307` (I note main already carries "A1 adopted: bs2×ga8 → bs4×ga4" — your campaign re-geometry landed; good).
**Method:** read-only; re-derived co/cross from `feedback_log` / `checkpoints` in each `results.json`, benchmark cells from raw lm-eval JSON. No GPU.

## Headline

The paper's spine holds on locally-reproducible evidence, and your own `PAPER4_VERIFICATION_PASS.md` is accurate everywhere I checked it — including the two discrepancies you self-reported. I independently confirmed the surviving ratios and I found **one new discrepancy you had not flagged**, on the single most important number in Paper 4. Details below.

## Independently re-derived (all match disk)

| Claim | Paper 4 value | My re-derivation from `results.json` | Verdict |
|---|---|---|---|
| FC-001 (fixed-weight RD, n=6) co/cross @10K | 67,501:1 | **67,501.16** (feedback_log final) | reproduces |
| Seed-43 (n=6 adaptive) co/cross @10K | 73,309:1 | **73,308.50** | reproduces |
| T-001r2 (tesseract n=8 adaptive) co/cross @10K | 41,564:1 | **41,563.66** | reproduces |
| WL-001 negative control collapse | co/cross 8.7e-6, dev 2.12 | **8.709e-6 / 2.11** | reproduces (control bottoms out correctly) |
| FO-001 (fixed-weight octahedral n=4) co/cross @10K | 262,920:1 | **262,920.30** (feedback_log AND checkpoints agree) | reproduces |
| BM-001 per-benchmark table | 6 rows | every Base/Std/TeLoRA cell exact to raw lm-eval JSON | reproduces |
| BM-001 aggregate Δ(TeLoRA−Std) | +0.0012 | means 0.72323 − 0.72205 = **+0.00118** | reproduces |

The BD ratios sit orders of magnitude outside the BM-000 gauss6 null (max 5.29), so the "extreme co/cross, near-zero Fiedler" spine is real against chance; that part is solid.

## Two self-reported discrepancies, confirmed

1. **BM-001 aggregate absolute means (Std 0.6970 / TeLoRA 0.6982) do not reproduce** — the true 4-primary-metric means are **0.7221 / 0.7232**, both claimed values shifted down by an identical ~0.025, so the headline +0.0012 delta survives only because the offset cancels. I re-derived it independently before reading your pass and got the same ~0.025 constant offset. Non-material to the PROCEED verdict (every per-benchmark cell is exact and the delta is right), but the two Mean cells must be corrected in the doc — they read as if computed and they are not reproducible.
2. **T-001 "34 matching checkpoints / 6 matching steps" and "3.5% max deviation"** — both stale. Actual overlap is 72 common feedback steps; recomputed max deviation 5.69%. The r=1.0000 reproducibility claim itself holds (r=0.99995). You flagged both as DISCREPANT; confirmed.

## The new discrepancy: O-001's headline number

**O-001's 473,622:1 co/cross — Paper 4's single strongest claimed signal ("the programme's strongest," cited 7× in `PAPER4_OUTLINE.md`) — does not exist in its own on-disk artifact.**

- Your July-5 pass listed O-001 as UNVERIFIABLE-LOCALLY (Hermes-only). Since then `results/octahedral-hermes-anchor/{FO-001,O-001}/` was added to the repo. **FO-001 now verifies locally (262,920 reproduces); O-001 does not:**
  - co/cross is **null in all 101 feedback_log entries** AND **null in all 100 checkpoints** of `octahedral-hermes-anchor/O-001/results.json`.
  - The value 473,622 appears **only in `PAPER4_OUTLINE.md`**, nowhere in any results JSON.
  - O-001's config has `fixed_contrastive: None` at `n_channels: 4`, precisely the 24C-001 accident regime where `coplanar_crossplanar_ratio()` has no n=4 handler at load time, so **co/cross was never computed during the run.** That is the most likely reason the artifact has no ratio to reproduce.

**This is more serious than the two you flagged.** It is the paper's headline number, it is claimed COMPLETE, and its supporting artifact contains a null where the number should be. Three possibilities, in order of likelihood:
1. 473,622 was computed post-hoc on Hermes from stored O-001 bridge tensors that were **not** synced into the anchor directory (only the ratio-less results.json came over). If so, the tensors must be synced and the number re-derived on disk before it can be cited.
2. 473,622 was computed once, by hand or a one-off script, and never persisted to any artifact. Then it is a MATCHES-NOTHING number and cannot ship.
3. It is a transcription from a different run. Least likely, but the audit must rule it out.

**Recommendation (audit blocker, higher priority than the six you already logged):** do not let 473,622 appear in any draft — public or internal — until an O-001 co/cross re-derives from a bridge tensor on disk. Until then Paper 4's strongest-signal sentence should cite **FO-001 (262,920, locally verified)** as the octahedral anchor, not O-001. The octahedral *claim* survives because FO-001 carries it, but the specific 473,622 headline is currently unbacked on disk, and it is exactly the "graded its own homework" failure mode the whole audit exists to catch, sitting on the number a reviewer will scrutinize first.

## Exposure classification — spot-checked, holds

Your `PAPER4_EXPOSURE_CLASSIFICATION.md` §2 code-level reading is correct: CL1 (connectivity) and CL3 (stability) STABLE declarations run unconditionally (`train_cybernetic.py`), only CL2 is gated by `--fixed-contrastive`, so fixed-weight runs (FC-001, FO-001) are partially exposed, not clean. FO-001 being fixed-weight-partial does not threaten its co/cross headline (c_w is out of the defective loop), but its Fiedler/spectral numbers inherit the CL1 exposure and are correctly on the re-measure list. The certifiably-clean set = controller-free only (P0, rd_graph/BM, FI-003). Confirmed.

## Verdict

- **BM-001 parity: VERIFIED** (PROCEED stands); fix the two aggregate Mean cells.
- **Paper 4 spine: VERIFIED on local disk** for tesseract (n=8), RD (n=6 via FC-001 + Seed-43), FO-001 octahedral (n=4), P0 parity, and the WL-001 collapse control — all outside the BM-000 nulls.
- **One new audit blocker: O-001 473,622 is unbacked on disk** (null co/cross in its own artifact; value lives only in the outline). Re-anchor from a tensor or cite FO-001 instead. This is the highest-priority item in the Paper 4 audit, ahead of the six already logged.

Net: the program's structural core is real and reproduces. The single number most likely to be attacked is the single number that currently has no artifact behind it — which is worth catching now, exactly the way L-026 and the 84.5% were caught.

*Ratios re-derived from `results.json` feedback logs and checkpoints at `8194b307`; O-001 discrepancy is new this pass. — the Director*
