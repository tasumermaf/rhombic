#!/usr/bin/env python
"""Export reduced feature matrices for the Director's Tier-2 verification.

Reuses the FROZEN D1 feature functions (no reimplementation, no divergence):
- canonical 'full' (_canonical_feature) — the H1 representation that hits ceiling
- H2 spectrum + probe (h2_features_for_run) — the representations behind the
  cross-family transfer refutation the Director scrutinizes hardest

All three are adapter-only, deterministic linear algebra (no base model, no HF
cache). Per family: one .npz with X_canonical (n, d_can), X_spectrum (n, 384),
X_probe (n, 12672), y task labels (int + string), and run_index — everything
needed to re-run H1's SVM/permutation and H2's standardization+transfer from
features. vocab_signature (arm #3) is NOT exported here: it needs the model
unembedding and only corroborates canonical's ceiling; available on request.

Interlock: refuses unless the bank is 480/480 COMPLETE.

Usage:
    python scripts/asset1_export_features.py --out results/asset1-delivery-verify/features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asset1_analysis_io as aio
import asset1_d1_identifiability as d1

PROJ_DIM = 16
PROJ_SEED = 0
SIGMA_SLOTS = 24
N_DEPTH_BINS = 4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank-root", default="results/asset1-bank")
    ap.add_argument("--out", default="results/asset1-delivery-verify/features")
    args = ap.parse_args()

    bank_root = Path(args.bank_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = aio.require_complete_bank(bank_root)  # 480/480 gate
    families = sorted({r["family_short"] for r in manifest["runs"]})

    for fam in families:
        runs = [r for r in aio.iter_runs(bank_root, family=fam, only_complete=True)]
        runs.sort(key=lambda r: r["run_index"])
        tasks = sorted({r["task"] for r in runs})
        task_to_int = {t: i for i, t in enumerate(tasks)}

        Xc, Xs, Xp, y, yi, idx = [], [], [], [], [], []
        for i, r in enumerate(runs):
            rd = r["run_dir"]
            Xc.append(d1._canonical_feature(rd, PROJ_DIM, PROJ_SEED))
            spec, probe = d1.h2_features_for_run(rd, SIGMA_SLOTS, N_DEPTH_BINS,
                                                 PROJ_DIM, PROJ_SEED)
            Xs.append(spec)
            Xp.append(probe)
            y.append(r["task"])
            yi.append(task_to_int[r["task"]])
            idx.append(r["run_index"])
            if (i + 1) % 40 == 0:
                print(f"  {fam}: {i + 1}/{len(runs)}", flush=True)

        path = out / f"features_{fam}.npz"
        np.savez_compressed(
            path,
            X_canonical=np.asarray(Xc, dtype=np.float32),
            X_spectrum=np.asarray(Xs, dtype=np.float32),
            X_probe=np.asarray(Xp, dtype=np.float32),
            y_task=np.asarray(y),
            y_int=np.asarray(yi, dtype=np.int64),
            run_index=np.asarray(idx, dtype=np.int64),
            tasks=np.asarray(tasks),
            params=np.asarray([f"proj_dim={PROJ_DIM}", f"proj_seed={PROJ_SEED}",
                               f"sigma_slots={SIGMA_SLOTS}",
                               f"n_depth_bins={N_DEPTH_BINS}", "svm_c=1.0"]),
        )
        print(f"  wrote {path} ({path.stat().st_size:,} bytes; "
              f"n={len(runs)} can={len(Xc[0])} spec={len(Xs[0])} probe={len(Xp[0])})",
              flush=True)

    print("[export-features] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
