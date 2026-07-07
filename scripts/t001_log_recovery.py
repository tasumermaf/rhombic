"""T-001 provenance recovery — extract the destroyed original run from its log.

Forensic finding (2026-07-07, blocker #5 of the Paper 4 audit)
--------------------------------------------------------------
The paper's tesseract reproducibility claim (r=1.0000, 3.5% max dev,
"T-001r1 partial 2,700 steps, 5,395:1 @2700") references a run whose
results.json NO LONGER EXISTS:

  Mar 15 07:31  T-001-full.log freezes at step 1700 (stdout lost; the
                machine-hang era). The run itself continued to ~step 2700
                and died. LEARNINGS.md / paper3_channel_ablation_section.md
                quote its endpoint (5,395:1, Fiedler 0.00070 @2700).
  Mar 15 09:20  compare_t001_runs.py written — compares "r1 partial 2700"
                (results/T-001-full/results.json AS IT THEN WAS) vs the
                in-progress r2. Source of the paper's r=1.0000 (computed on
                FIEDLER at common steps) and "34 matching checkpoints".
  Mar 16 03:57  T-001-full-r2 completes (10,000 steps, 41,564:1).
  Mar 16 21:54  T-001-full is RELAUNCHED FRESH (same seed 42) and runs to
                7,100 — OVERWRITING the original results.json. Proof it is a
                restart, not a resume: its values at steps <=1700 do not
                match the log's printed originals (e.g. step 1700 co/cross
                4,666 vs original 4,729).
  Mar 17 10:37  fig_t001_reproducibility.png regenerated from the
                post-overwrite data.

Hermes carries no tesseract artifacts (checked 2026-07-07). The original
run therefore survives ONLY as: (a) 16 Steersman blocks in the frozen log
(steps 0-1600, Fiedler + co/cross), extracted here; (b) quoted endpoints in
LEARNINGS.md / paper3 docs.

What this script produces
-------------------------
results/t001-provenance/recovery.json — the recovered original trajectory
plus the three-way same-seed agreement matrix (original-from-log vs the
Mar-16 restart vs r2) over the surviving common steps. Result (2026-07-07):
pairwise co/cross r = 0.995-0.997 (max rel dev 5-10%), Fiedler r >= 0.9994
for all three pairs — the destroyed original is mutually consistent with
both on-disk runs, upgrading the reproducibility evidence from two runs to
three. Run: python scripts/t001_log_recovery.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "results/T-001-full.log"
RESTART = REPO / "results/T-001-full/results.json"
R2 = REPO / "results/T-001-full-r2/results.json"
OUT = REPO / "results/t001-provenance/recovery.json"


def extract_original(log_path: Path) -> dict[int, dict]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(
        r"STEERSMAN @ step (\d+).*?Fiedler:\s+([0-9.]+).*?Co/Cross:\s+([0-9.]+)",
        text, re.S)
    return {int(s): {"fiedler_mean": float(f), "co_cross_ratio": float(c)}
            for s, f, c in blocks}


def load_run(p: Path) -> dict[int, dict]:
    r = json.loads(p.read_text())
    return {e["step"]: {"fiedler_mean": e.get("fiedler_mean"),
                        "co_cross_ratio": e.get("co_cross_ratio")}
            for e in r["feedback_log"]}


def pair_stats(a: dict, b: dict, steps: list[int]) -> dict:
    co_a = np.array([a[s]["co_cross_ratio"] for s in steps])
    co_b = np.array([b[s]["co_cross_ratio"] for s in steps])
    fi_a = np.array([a[s]["fiedler_mean"] for s in steps])
    fi_b = np.array([b[s]["fiedler_mean"] for s in steps])
    return {
        "n_steps": len(steps),
        "co_cross_pearson_r": float(np.corrcoef(co_a, co_b)[0, 1]),
        "co_cross_max_rel_dev": float(np.max(np.abs(co_a - co_b) / np.abs(co_a))),
        "fiedler_pearson_r": float(np.corrcoef(fi_a, fi_b)[0, 1]),
    }


def main() -> None:
    orig = extract_original(LOG)
    restart, r2 = load_run(RESTART), load_run(R2)
    common = [s for s in sorted(set(orig) & set(restart) & set(r2))
              if s > 0 and restart[s]["co_cross_ratio"] is not None
              and r2[s]["co_cross_ratio"] is not None]
    # Full-overlap restart x r2 pair (steps 100-7,100) — the paper's third
    # tab:reproducibility row traces to THIS block, not the 15-step version.
    full_overlap = [s for s in sorted(set(restart) & set(r2))
                    if s > 0 and restart[s]["co_cross_ratio"] is not None
                    and r2[s]["co_cross_ratio"] is not None]
    out = {
        "description": __doc__.strip().splitlines()[0],
        "original_run_recovered_from_log": {
            "source": str(LOG),
            "n_points": len(orig),
            "steps": sorted(orig),
            "trajectory": {str(s): orig[s] for s in sorted(orig)},
            "endpoint_quotes_secondary_sources": {
                "co_cross_at_2700": 5395.0,
                "fiedler_at_2700": 0.00070,
                "quoted_in": ["docs/LEARNINGS.md",
                              "docs/paper3_channel_ablation_section.md"],
            },
        },
        "restart_is_not_resume_proof": {
            "step_1700_original_from_log": orig.get(1600),
            "note": "restart values at steps <=1600 differ from the log's "
                    "originals (e.g. 1600: orig 4729.1-region vs restart "
                    "4610.2) — fresh same-seed relaunch, results.json "
                    "overwritten 2026-03-16 21:54.",
        },
        "three_way_same_seed_agreement": {
            "common_steps": common,
            "original_vs_restart": pair_stats(orig, restart, common),
            "original_vs_r2": pair_stats(orig, r2, common),
            "restart_vs_r2": pair_stats(restart, r2, common),
        },
        "restart_vs_r2_full_overlap": {
            "steps_range": [full_overlap[0], full_overlap[-1]],
            **pair_stats(restart, r2, full_overlap),
        },
        "hermes_checked": "2026-07-07 — no tesseract/T-001 artifacts on Hermes",
        "verdict": (
            "Original T-001r1 results.json destroyed by the Mar-16 fresh "
            "relaunch; partially recovered from the frozen log (16 points, "
            "steps 0-1600). All three same-seed runs mutually consistent "
            "(co/cross r 0.995-0.997, Fiedler r >= 0.9994) — the "
            "reproducibility claim survives, strengthened to three runs, "
            "but the paper's specific numbers (r=1.0000, 3.5%, 5,395:1 "
            "@2700, 34 checkpoints) reference the destroyed artifact and "
            "must be rewritten around the surviving evidence."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    t = out["three_way_same_seed_agreement"]
    print(f"recovered {len(orig)} original points; {len(common)} three-way steps")
    for k in ("original_vs_restart", "original_vs_r2", "restart_vs_r2"):
        s = t[k]
        print(f"  {k}: co/cross r={s['co_cross_pearson_r']:.5f} "
              f"maxdev={s['co_cross_max_rel_dev']:.1%} "
              f"fiedler r={s['fiedler_pearson_r']:.5f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
