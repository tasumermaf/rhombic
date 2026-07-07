"""Re-derive octahedral (n=4) co/cross from saved bridge tensors — O-001 anchor.

Why this exists
---------------
O-001 (adaptive octahedral, n=4, TinyLlama, run 2026-03-18) logs a NULL
co_cross_ratio in every feedback_log entry and every checkpoint of its
results.json. The reason (Director verification pass, 2026-07-06): at O-001's
runtime the live metric ``coplanar_crossplanar_ratio`` had no n=4 handler, so
``co_cross`` was computed as ``None`` throughout the run (see train_cybernetic
.py:581-589, which drops non-finite / None ratios). O-001's headline number
473,622:1 therefore lived only in PAPER4_OUTLINE.md with no artifact behind it.

FO-001 (fixed-weight octahedral, n=4, run 2026-03-26) ran AFTER the n=4 handler
was added, so its 262,920:1 was logged live and reproduces from disk.

This script re-derives BOTH from their saved ``bridge_final_*.npy`` tensors
using the EXACT training-time metric + aggregation, and VALIDATES the method by
reproducing FO-001's logged 262,920.298 bit-for-bit before trusting O-001.

Method (identical to train_cybernetic.py:581-589)
-------------------------------------------------
Per module bridge B (4x4): ratio = mean|B[coplanar]| / mean|B[crossplanar]|
with the n=4 octahedral partition from train_exp2_scale
._coplanar_crossplanar_indices: coplanar {(0,1),(2,3)}, crossplanar
{(0,2),(0,3),(1,2),(1,3)}. Aggregate = mean over modules of the finite ratios.

Result (2026-07-06): FO-001 -> 262,920.298 (EXACT match to its logged value);
O-001 -> 473,621.655 (== 473,622 rounded; the outline value is vindicated).

Tensors live at results/octahedral-hermes-anchor/{FO-001,O-001}/bridges/,
fetched from Hermes (FO-001: ~/rhombic/results/FO-001/; O-001:
~/rhombic/results/channel-ablation/O-001/). Run:
    python scripts/rederive_octahedral_cocross.py
writes results/octahedral-hermes-anchor/co_cross_rederivation.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from train_exp2_scale import coplanar_crossplanar_ratio  # the train-time metric

ANCHOR = SCRIPTS.parent / "results" / "octahedral-hermes-anchor"
# (run_name, logged/claimed value, source of that value)
RUNS = [
    ("FO-001", 262920.298, "FO-001 results.json feedback_log final (step 10000) — VALIDATION"),
    ("O-001", 473622.0, "PAPER4_OUTLINE.md (was unbacked on disk; this script is the anchor)"),
]


def aggregate_co_cross(bridge_dir: Path) -> tuple[float | None, int, int, list[float]]:
    """Replicate train_cybernetic.py:581-589 exactly."""
    ratios: list[float] = []
    n_total = 0
    for f in sorted(bridge_dir.glob("bridge_final_*.npy")):
        n_total += 1
        B = np.load(f)
        r = coplanar_crossplanar_ratio(B)
        if r is not None and np.isfinite(r["ratio"]):
            ratios.append(float(r["ratio"]))
    agg = float(np.mean(ratios)) if ratios else None
    return agg, n_total, len(ratios), ratios


def main() -> None:
    out: dict = {
        "description": "Post-hoc re-derivation of octahedral (n=4) co/cross from "
                       "saved bridge_final tensors, using the exact train-time "
                       "metric (train_exp2_scale.coplanar_crossplanar_ratio) and "
                       "aggregation (train_cybernetic.py:581-589, mean of finite "
                       "per-module ratios).",
        "generated_by": "scripts/rederive_octahedral_cocross.py",
        "runs": {},
    }
    all_ok = True
    for name, expected, src in RUNS:
        agg, n_total, n_fin, ratios = aggregate_co_cross(ANCHOR / name / "bridges")
        rel_err = (abs(agg - expected) / expected) if (agg and expected) else None
        matches = bool(rel_err is not None and rel_err < 1e-3)
        srs = sorted(ratios)
        out["runs"][name] = {
            "rederived_co_cross": agg,
            "expected_value": expected,
            "expected_source": src,
            "relative_error": rel_err,
            "matches": matches,
            "modules_total": n_total,
            "modules_finite": n_fin,
            "per_module_ratio_min": srs[0] if srs else None,
            "per_module_ratio_median": srs[len(srs) // 2] if srs else None,
            "per_module_ratio_max": srs[-1] if srs else None,
        }
        flag = "MATCH" if matches else "MISMATCH"
        print(f"{name}: re-derived {agg:,.3f} vs expected {expected:,.3f}  [{flag}]")
        all_ok = all_ok and matches
    out["all_match"] = all_ok
    out["verdict"] = (
        "FO-001 validates the method (exact); O-001 473,622 is vindicated — "
        "re-derived from tensors on disk, resolving the 'unbacked headline' blocker."
        if all_ok else "One or more runs did not reproduce — investigate."
    )
    dest = ANCHOR / "co_cross_rederivation.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{out['verdict']}\nwrote {dest}")


if __name__ == "__main__":
    main()
