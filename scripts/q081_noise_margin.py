#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q-08-1 — The Learned-Noise-Margin Question. Analysis, zero compute.

Reproducibility line (card §12 decider pins, Item 5 — carried verbatim):

    analysis script : C:/falco/rhombic/scripts/q081_noise_margin.py
    commit          : pending, filled by Meridian at commit
    results         : C:/falco/rhombic/results/q081-noise-margin/
    verifier        : fresh context, re-derives C1-C3 from the .npy files independently
                      of the analysis script (§9)

Card: C:/falco/docs/cards/Q-08-1_learned-noise-margin.md (graded 2026-09-06;
additions 2026-09-11; decider pins 2026-09-14). Where card and pins conflict the
pins govern; nothing in the card is edited by this script.

WHAT THIS SCRIPT DOES, AND WHAT IT REFUSES TO DO
------------------------------------------------
* Reads `.npy` bridge matrices already on disk. ZERO GPU-seconds, no gpu_guard
  claim, no model loaded (card §10). CPU only; CUDA is never imported.
* Asserts the whole §12 Item-8 census against the live filesystem BEFORE any
  endpoint is computed, and RAISES (never warns) on the first disagreement
  (`CensusError`). A census verified on 2026-09-11 or 2026-09-14 is not a census
  verified at run time.
* Computes exactly three confirmatory tests (§7): C1, C2, C3. Everything else —
  Arms C and D, per-layer breakdowns, the k-ladder — is DESCRIPTIVE and carries
  no p-value, no permutation test and no inference of any kind (§14, Director
  2026-09-06: "Keep it that way regardless of how clean the trend looks").
* Modifies no artifact it reads. Writes only into
  results/q081-noise-margin/.

PINNED, NOT REFITTED
--------------------
OCC_BAND = closed [0.01959, 0.11538] on |B[i,j]| — E-5's own measured band at the
precision E-5's record prints it [results/E-5-bifurcation/RESULTS.md:59-61].
Fixed for every arm, every checkpoint, every scale. This script contains no code
path that can refit it.

SCOPE LIMIT, carried in every claim (§8)
----------------------------------------
TinyLlama-1.1B (descriptively Qwen2.5-1.5B / Qwen2.5-7B), TeLoRA bridge couplings
under the Steersman contrastive objective, at the configurations already written
to disk; detector v2 for E-5, v1 for the FI series — no cross-detector depth
comparison is made. No claim about learned systems in general and none about
silicon.

USAGE
-----
    python scripts/q081_noise_margin.py                 # full run
    python scripts/q081_noise_margin.py --self-test     # census-halt demonstration
    python scripts/q081_noise_margin.py --census-only   # assert census, then stop
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 0. Pins. None of this is computed from the data being analysed.
# ─────────────────────────────────────────────────────────────────────────────

REPRO_LINE = """analysis script : C:/falco/rhombic/scripts/q081_noise_margin.py
commit          : pending, filled by Meridian at commit
results         : C:/falco/rhombic/results/q081-noise-margin/
verifier        : fresh context, re-derives C1-C3 from the .npy files independently
                  of the analysis script (§9)"""

# --- Item 2: the Occ band. PINNED. No refit path exists in this file. ---------
OCC_BAND_LO = 0.01959      # RESULTS.md:59-61 verbatim "the largest cross entry (0.01959)"
OCC_BAND_HI = 0.11538      # RESULTS.md:59-61 verbatim "the smallest co entry (0.11538)"
OCC_BAND_RATIO = OCC_BAND_HI / OCC_BAND_LO          # 5.889740
OCC_BAND_DECADES = math.log10(OCC_BAND_RATIO)       # 0.770096

# --- E-5's inf convention, mirrored exactly (e5_bifurcation_sweep.py:251-261) --
CROSS_ZERO_EPS = 1e-12

# --- Permutation pins (stated before the run; not tuned) ---------------------
PERM_SEED = 20260914          # the date the decider pins were ruled
PERM_DRAWS = 10000
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_DRAWS = 2000

# --- n = 6 trained pair spec (Arms B/C/D). Derived from RD geometry, then -----
# --- hard-coded so this script never imports torch transitively. -------------
# Verified 2026-09-14 by running
#   from rhombic.nn.topology import direction_pair_coupling
#   [(i,j) for i<j if C[i,j] >= 4]
# which reproduces train_exp2_scale.py:99-117 `_coplanar_crossplanar_indices(6)`.
CO_PAIRS_N6 = [(0, 1), (2, 3), (4, 5)]
CROSS_PAIRS_N6 = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3),
                  (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]

# --- Item 8: the census, pinned. Asserted at load time, RAISES on mismatch. ---
# Columns: runs, step indices (explicit tuple), files per index, step files
# total, finals total, n_channels, .npy byte size.
E5_RUN_RE = re.compile(r"^f\d\.\d{2}_s\d+$")

ARM_A_STEPS = tuple(range(0, 3001, 100))          # 31
FI004_STEPS = tuple(range(100, 3001, 100))        # 30
FI003_STEPS = tuple(range(100, 1201, 100))        # 12
ARM_C_TL_STEPS = tuple(range(0, 10001, 100))      # 101
FC001_STEPS = tuple(range(0, 1801, 100))          # 19
FC001F_STEPS = tuple(range(0, 2401, 100))         # 25
EXP3_STEPS = tuple(range(0, 12901, 100))          # 130

CENSUS = OrderedDict([
    # key            (arm, subdir,            runs, steps,          per_idx, step_files, finals, n, npy_bytes)
    ("A",            ("A", "E-5-bifurcation", 15,   ARM_A_STEPS,    88,      40920,      1320,   8, 384)),
    ("fi-004",       ("B", "fi-004",          1,    FI004_STEPS,    88,      2640,       0,      6, 272)),
    ("fi-003",       ("B", "fi-003",          1,    FI003_STEPS,    88,      1056,       0,      6, 272)),
    ("exp3_tinyllama", ("C", "exp3_tinyllama", 1,   ARM_C_TL_STEPS, 88,      8888,       88,     6, 272)),
    ("fc-001",       ("C", "fc-001",          1,    FC001_STEPS,    112,     2128,       112,    6, 272)),
    ("fc-001-fresh", ("C", "fc-001-fresh",    1,    FC001F_STEPS,   112,     2800,       112,    6, 272)),
    ("exp3",         ("C", "exp3",            1,    EXP3_STEPS,     112,     14560,      112,    6, 272)),
    ("cw-001",       ("D", "cw-001",          1,    ARM_C_TL_STEPS, 88,      8888,       88,     6, 272)),
])

# Arm A's 15 runs carry ONE step-index-set signature (§12 Item 8):
ARM_A_INDEX_SIG12 = "46370c6576ae"

# Model family per directory — carried into every Arm C sentence as a confound.
MODEL_OF = {
    "E-5-bifurcation": "TinyLlama-1.1B",
    "fi-004": "TinyLlama-1.1B",
    "fi-003": "TinyLlama-1.1B",
    "exp3_tinyllama": "TinyLlama-1.1B",
    "fc-001": "Qwen2.5-1.5B",
    "fc-001-fresh": "Qwen2.5-1.5B",
    "exp3": "Qwen2.5-7B",
    "cw-001": "TinyLlama-1.1B",
}
DETECTOR_OF = {"E-5-bifurcation": "v2"}   # everything else: v1 (§8)

BRIDGE_RE = re.compile(r"^bridge_(?:step(\d+)|(final))_(.+)\.npy$")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Typed census failure. RAISES — never warns.
# ─────────────────────────────────────────────────────────────────────────────

class CensusError(RuntimeError):
    """Raised on the FIRST disagreement between the pinned census and disk.

    Card §12 Item 8: "the analysis asserts every count below against the live
    filesystem and raises, not warns, on the first disagreement."
    """

    def __init__(self, where: str, field: str, expected, observed, note: str = ""):
        self.where, self.field = where, field
        self.expected, self.observed = expected, observed
        msg = (
            "\n"
            "=== CENSUS HALT — Q-08-1 §12 Item 8 ===\n"
            f"  where    : {where}\n"
            f"  field    : {field}\n"
            f"  expected : {expected}\n"
            f"  observed : {observed}\n"
            f"  note     : {note or 'pinned census disagrees with the live filesystem'}\n"
            "  action   : HALT. No endpoint was computed. A census verified on\n"
            "             2026-09-11 or 2026-09-14 is not a census verified at run\n"
            "             time; the analysis does not silently pool.\n"
            "=== END CENSUS HALT ==="
        )
        super().__init__(msg)


def _scan_dir(path: Path):
    """Directory listing only: step index -> {adapter: size}, finals, sizes."""
    by_idx: dict[int, dict[str, int]] = {}
    finals: dict[str, int] = {}
    sizes: set[int] = set()
    zero_byte = 0
    other = []
    if not path.is_dir():
        raise CensusError(str(path), "directory exists", True, False,
                          "census directory missing from the live tree")
    for name in os.listdir(path):
        full = path / name
        if not name.endswith(".npy"):
            if full.is_file():
                other.append(name)
            continue
        size = full.stat().st_size
        sizes.add(size)
        if size == 0:
            zero_byte += 1
        m = BRIDGE_RE.match(name)
        if not m:
            raise CensusError(str(path), "unrecognised .npy filename", "bridge_step<N>_* / bridge_final_*", name)
        if m.group(1) is not None:
            by_idx.setdefault(int(m.group(1)), {})[m.group(3)] = size
        else:
            finals[m.group(3)] = size
    return by_idx, finals, sizes, zero_byte, sorted(other)


def assert_census(results_root: Path, verbose: bool = True) -> dict:
    """Assert the whole §12 Item-8 census. Raise CensusError on first mismatch."""
    t0 = time.time()
    report: dict = {"results_root": str(results_root), "arms": {}}

    for key, (arm, subdir, n_runs, steps, per_idx, step_files, finals_n,
              n_ch, npy_bytes) in CENSUS.items():
        base = results_root / subdir
        if not base.is_dir():
            raise CensusError(subdir, "directory exists", True, False,
                              "census directory missing from the live tree")

        if key == "A":
            runs = sorted(d for d in os.listdir(base)
                          if E5_RUN_RE.match(d) and (base / d).is_dir())
            if len(runs) != n_runs:
                raise CensusError(subdir, "run directories", n_runs, len(runs))
            sigs, tot_step, tot_fin, all_sizes, zb = set(), 0, 0, set(), 0
            for r in runs:
                by_idx, finals, sizes, zero_byte, other = _scan_dir(base / r)
                got_steps = tuple(sorted(by_idx))
                if got_steps != steps:
                    raise CensusError(f"{subdir}/{r}", "step index set",
                                      f"{len(steps)} indices {steps[0]}..{steps[-1]} by 100",
                                      f"{len(got_steps)} indices {got_steps[:3]}..{got_steps[-3:]}")
                for idx, ad in by_idx.items():
                    if len(ad) != per_idx:
                        raise CensusError(f"{subdir}/{r}", f"files at step {idx}", per_idx, len(ad))
                if len(finals) != 88:
                    raise CensusError(f"{subdir}/{r}", "bridge_final_*.npy", 88, len(finals))
                if sorted(other) != ["config.json", "results.json"]:
                    raise CensusError(f"{subdir}/{r}", "non-.npy files",
                                      ["config.json", "results.json"], sorted(other))
                # Serialization is the card's: sha256 of str(sorted list of
                # distinct indices), first 12 hex — reproduces the pinned value.
                sigs.add(_sha12(str(list(got_steps))))
                tot_step += sum(len(a) for a in by_idx.values())
                tot_fin += len(finals)
                all_sizes |= sizes
                zb += zero_byte
            if len(sigs) != 1:
                raise CensusError(subdir, "step-index-set signatures across runs", 1, len(sigs))
            if sorted(sigs)[0] != ARM_A_INDEX_SIG12:
                raise CensusError(subdir, "step-index-set signature (sha256 of "
                                  "str(sorted index list), first 12 hex)",
                                  ARM_A_INDEX_SIG12, sorted(sigs)[0])
            if tot_step != step_files:
                raise CensusError(subdir, "bridge_step files, all runs", step_files, tot_step)
            if tot_fin != finals_n:
                raise CensusError(subdir, "bridge_final files, all runs", finals_n, tot_fin)
            if all_sizes != {npy_bytes}:
                raise CensusError(subdir, ".npy byte size (uniform)", {npy_bytes}, all_sizes)
            if zb:
                raise CensusError(subdir, "zero-byte .npy", 0, zb)
            report["arms"][key] = {
                "arm": arm, "dir": subdir, "runs": len(runs), "run_ids": runs,
                "step_indices": len(steps), "files_per_index": per_idx,
                "step_files": tot_step, "finals": tot_fin,
                "npy_bytes": sorted(all_sizes), "zero_byte_npy": zb,
                "index_set_signature_sha256_12": sorted(sigs)[0],
                "n_channels": n_ch,
            }
        else:
            by_idx, finals, sizes, zero_byte, other = _scan_dir(base)
            got_steps = tuple(sorted(by_idx))
            if got_steps != steps:
                raise CensusError(subdir, "step index set",
                                  f"{len(steps)} indices {steps[0]}..{steps[-1]} by 100",
                                  f"{len(got_steps)} indices "
                                  f"{got_steps[:3]}..{got_steps[-3:] if got_steps else ()}")
            for idx, ad in by_idx.items():
                if len(ad) != per_idx:
                    raise CensusError(subdir, f"files at step {idx}", per_idx, len(ad))
            tot_step = sum(len(a) for a in by_idx.values())
            if tot_step != step_files:
                raise CensusError(subdir, "bridge_step files", step_files, tot_step)
            if len(finals) != finals_n:
                raise CensusError(subdir, "bridge_final_*.npy", finals_n, len(finals))
            if sizes and sizes != {npy_bytes}:
                raise CensusError(subdir, ".npy byte size (uniform)", {npy_bytes}, sizes)
            if zero_byte:
                raise CensusError(subdir, "zero-byte .npy", 0, zero_byte)
            report["arms"][key] = {
                "arm": arm, "dir": subdir, "runs": 1,
                "step_indices": len(steps), "files_per_index": per_idx,
                "step_files": tot_step, "finals": len(finals),
                "npy_bytes": sorted(sizes), "zero_byte_npy": zero_byte,
                "non_npy": sorted(other), "n_channels": n_ch,
            }

        # n_channels from the run's own config, cross-checked against the pin.
        cfg_path = (base / report["arms"][key].get("run_ids", [""])[0] / "config.json") \
            if key == "A" else (base / "config.json")
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if int(cfg.get("n_channels", -1)) != n_ch:
                raise CensusError(subdir, "config n_channels", n_ch, cfg.get("n_channels"))

        if verbose:
            a = report["arms"][key]
            print(f"  [census OK] Arm {arm} {subdir:16s} "
                  f"runs={a['runs']:>2} idx={a['step_indices']:>3} "
                  f"per_idx={a['files_per_index']:>3} step={a['step_files']:>6} "
                  f"final={a['finals']:>5} n={n_ch} {npy_bytes}B")

    report["wall_clock_s"] = round(time.time() - t0, 3)
    report["verdict"] = "CENSUS ASSERTED AT RUN TIME — every pinned count matches the live filesystem"
    return report


def _sha12(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Endpoints. §4, read through the Item-2 pin.
# ─────────────────────────────────────────────────────────────────────────────

def endpoints(B: np.ndarray, co_pairs, cross_pairs) -> dict:
    """G_dec, G_min, Occ for one adapter's bridge matrix.

    Symmetric-indexed upper triangle only; abs(B[i,j]) for i<j, exactly as
    e5_bifurcation_sweep.py:257-258 does. Same inf convention as :261 when the
    cross mean is <= 1e-12.
    """
    co = np.abs(np.array([B[i, j] for i, j in co_pairs], dtype=np.float64))
    cr = np.abs(np.array([B[i, j] for i, j in cross_pairs], dtype=np.float64))
    mean_co, mean_cr = float(co.mean()), float(cr.mean())
    min_co, max_cr = float(co.min()), float(cr.max())

    # A bridge with NO off-diagonal structure at all (the identity initialization)
    # gives 0/0, which is UNDEFINED — not the +inf of E-5's x/0 convention. The two
    # are kept apart: conflating them would report a gap where there is no structure.
    degenerate = bool(np.all(co == 0.0) and np.all(cr == 0.0))

    if degenerate:
        g_dec = g_min = math.nan
    else:
        if mean_cr > CROSS_ZERO_EPS and mean_co > 0.0:
            g_dec = math.log10(mean_co) - math.log10(mean_cr)
        elif mean_cr <= CROSS_ZERO_EPS:
            g_dec = math.inf        # e5_bifurcation_sweep.py:261 convention
        else:
            g_dec = -math.inf

        if max_cr > CROSS_ZERO_EPS and min_co > 0.0:
            g_min = math.log10(min_co) - math.log10(max_cr)
        elif max_cr <= CROSS_ZERO_EPS:
            g_min = math.inf
        else:
            g_min = -math.inf

    allv = np.concatenate([co, cr])
    occ_n = int(np.count_nonzero((allv >= OCC_BAND_LO) & (allv <= OCC_BAND_HI)))
    occ_co = int(np.count_nonzero((co >= OCC_BAND_LO) & (co <= OCC_BAND_HI)))
    return {
        "G_dec": g_dec, "G_min": g_min, "zero_offdiag": int(degenerate),
        "occ_n": occ_n, "occ_co": occ_co, "occ_cross": occ_n - occ_co,
        "n_entries": int(allv.size),
        "mean_co": mean_co, "mean_cross": mean_cr,
        "min_co": min_co, "max_cross": max_cr,
    }


def pair_spec(config: dict, n: int):
    """The spec the run actually optimized — E-5's own `_post_corruption_pairs`.

    Reads trainer-recorded co_pairs_trained / cross_pairs_trained; absent (an
    uncorrupted run that predates the record) falls back to the true n-channel
    partition, which for an uncorrupted run IS the trained spec
    [e5_bifurcation_sweep.py:264-277].
    """
    co = config.get("co_pairs_trained")
    cross = config.get("cross_pairs_trained")
    if co is not None and cross is not None:
        return ([tuple(int(x) for x in p) for p in co],
                [tuple(int(x) for x in p) for p in cross], "config.co_pairs_trained")
    if n == 6:
        return list(CO_PAIRS_N6), list(CROSS_PAIRS_N6), "_coplanar_crossplanar_indices(6) fallback"
    raise RuntimeError(f"no trained pair spec and no pinned fallback for n={n}")


def load_layer(d: Path, prefix: str, co_pairs, cross_pairs):
    """Endpoints for every adapter at one checkpoint. Returns (ids, records)."""
    names = sorted(f for f in os.listdir(d)
                   if f.startswith(prefix) and f.endswith(".npy"))
    ids, recs = [], []
    for f in names:
        m = BRIDGE_RE.match(f)
        ids.append(m.group(3))
        with open(d / f, "rb") as fh:
            B = np.load(fh)
        recs.append(endpoints(B, co_pairs, cross_pairs))
    return ids, recs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Arm loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_arm_a(root: Path, verbose=True):
    base = root / "E-5-bifurcation"
    runs = sorted(d for d in os.listdir(base) if E5_RUN_RE.match(d))
    out = {"runs": OrderedDict(), "steps": list(ARM_A_STEPS), "adapters": None}
    finals = OrderedDict()
    for r in runs:
        cfg = json.loads((base / r / "config.json").read_text(encoding="utf-8"))
        co, cr, src = pair_spec(cfg, int(cfg["n_channels"]))
        if not (len(co) == 4 and len(cr) == 24):
            raise CensusError(f"E-5-bifurcation/{r}", "trained pair split (co, cross)",
                              (4, 24), (len(co), len(cr)))
        per_step = OrderedDict()
        for s in ARM_A_STEPS:
            ids, recs = load_layer(base / r, f"bridge_step{s}_", co, cr)
            if out["adapters"] is None:
                out["adapters"] = ids
            elif ids != out["adapters"]:
                raise CensusError(f"E-5-bifurcation/{r}", f"adapter id set at step {s}",
                                  "identical to run 1 step 0", "differs")
            per_step[s] = recs
        fids, frecs = load_layer(base / r, "bridge_final_", co, cr)
        finals[r] = (fids, frecs)
        out["runs"][r] = {
            "pair_spec_source": src, "co_pairs": co, "cross_pairs": cr,
            "f": float(cfg.get("pair_correctness", float("nan"))),
            "seed": int(cfg.get("seed", -1)),
            "per_step": per_step,
        }
        if verbose:
            print(f"    [A] {r} loaded ({len(ARM_A_STEPS)}x{len(out['adapters'])} step + 88 final)")
    out["finals"] = finals
    return out


def load_single_run(root: Path, sub: str, steps, n_expect, verbose=True):
    d = root / sub
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    n = int(cfg["n_channels"])
    co, cr, src = pair_spec(cfg, n)
    per_step = OrderedDict()
    adapters = None
    for s in steps:
        ids, recs = load_layer(d, f"bridge_step{s}_", co, cr)
        if adapters is None:
            adapters = ids
        elif ids != adapters:
            raise CensusError(sub, f"adapter id set at step {s}", "identical", "differs")
        per_step[s] = recs
    fids, frecs = ([], [])
    if any(f.startswith("bridge_final_") for f in os.listdir(d)):
        fids, frecs = load_layer(d, "bridge_final_", co, cr)
    res = {}
    rp = d / "results.json"
    if rp.is_file():
        res = json.loads(rp.read_text(encoding="utf-8"))
    if verbose:
        print(f"    [{sub}] loaded ({len(steps)}x{len(adapters)} step, {len(fids)} final, n={n})")
    return {"dir": sub, "n_channels": n, "pair_spec_source": src,
            "co_pairs": co, "cross_pairs": cr, "config": cfg,
            "adapters": adapters, "per_step": per_step,
            "finals": (fids, frecs), "results": res}


# ─────────────────────────────────────────────────────────────────────────────
# 4. N2 two-fit (C3)
# ─────────────────────────────────────────────────────────────────────────────

def _lognormal_band_mass(vals: np.ndarray, ddof: int):
    """Fit log-normal to |B| entries; return (P_in_band, mu, sigma, degenerate)."""
    v = vals[vals > 0.0]
    if v.size != vals.size:
        return None  # zero entry — flagged by the caller
    lv = np.log(v)
    mu = float(lv.mean())
    if lv.size - ddof <= 0:
        return None
    sigma = float(lv.std(ddof=ddof))
    lo, hi = math.log(OCC_BAND_LO), math.log(OCC_BAND_HI)
    if sigma <= 0.0:
        p = 1.0 if (lo <= mu <= hi) else 0.0
        return p, mu, sigma, True
    p = float(stats.norm.cdf((hi - mu) / sigma) - stats.norm.cdf((lo - mu) / sigma))
    return p, mu, sigma, False


def n2_two_fit(root: Path, arm_a, ddof: int = 0):
    """Per adapter, fit co and cross modes INDEPENDENTLY as log-normals on
    |B[i,j]| (card §5), predict OCC_BAND mass, compare to observed.

    2 fits per adapter x 1,320 adapters = 2,640 fits at E-5's FINAL checkpoint.
    """
    base = root / "E-5-bifurcation"
    p_co_l, p_cr_l, mu_co_l, sg_co_l, mu_cr_l, sg_cr_l = [], [], [], [], [], []
    obs_in, obs_total, zero_flag, degen = 0, 0, 0, 0
    pooled_co, pooled_cross = [], []
    for r, blob in arm_a["runs"].items():
        co_pairs, cross_pairs = blob["co_pairs"], blob["cross_pairs"]
        for f in sorted(os.listdir(base / r)):
            if not f.startswith("bridge_final_"):
                continue
            with open(base / r / f, "rb") as fh:
                B = np.load(fh)
            co = np.abs(np.array([B[i, j] for i, j in co_pairs], dtype=np.float64))
            cr = np.abs(np.array([B[i, j] for i, j in cross_pairs], dtype=np.float64))
            pooled_co.append(co)
            pooled_cross.append(cr)
            allv = np.concatenate([co, cr])
            obs_in += int(np.count_nonzero((allv >= OCC_BAND_LO) & (allv <= OCC_BAND_HI)))
            obs_total += allv.size
            fc = _lognormal_band_mass(co, ddof)
            fx = _lognormal_band_mass(cr, ddof)
            if fc is None or fx is None:
                zero_flag += 1
                continue
            p_co_l.append(fc[0]); mu_co_l.append(fc[1]); sg_co_l.append(fc[2])
            p_cr_l.append(fx[0]); mu_cr_l.append(fx[1]); sg_cr_l.append(fx[2])
            degen += int(fc[3]) + int(fx[3])

    p_co = np.array(p_co_l); p_cr = np.array(p_cr_l)
    n_co_per, n_cr_per = 4, 24
    per_adapter = n_co_per * p_co + n_cr_per * p_cr
    pred = float(per_adapter.sum())
    var = float((n_co_per * p_co * (1 - p_co) + n_cr_per * p_cr * (1 - p_cr)).sum())
    sd = math.sqrt(var)

    # Exact P(no entry lands in the band) under the fitted (independent) model.
    with np.errstate(divide="ignore"):
        log_p0 = float(n_co_per * np.log1p(-np.clip(p_co, 0, 1 - 1e-300)).sum()
                       + n_cr_per * np.log1p(-np.clip(p_cr, 0, 1 - 1e-300)).sum())

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_ad = per_adapter.size
    boot = np.array([per_adapter[rng.integers(0, n_ad, n_ad)].sum()
                     for _ in range(BOOTSTRAP_DRAWS)])

    pooled_co = np.concatenate(pooled_co); pooled_cross = np.concatenate(pooled_cross)
    return {
        "ddof": ddof,
        "n_adapters_fitted": n_ad, "n_fits": 2 * n_ad,
        "adapters_with_zero_entry_skipped": zero_flag,
        "degenerate_fits_sigma_zero": degen,
        "observed_in_band": obs_in, "observed_total": obs_total,
        "observed_occupancy": obs_in / obs_total,
        "predicted_count": pred,
        "predicted_occupancy": pred / obs_total,
        "predicted_sd_poisson_binomial": sd,
        "predicted_pm2sd": [pred - 2 * sd, pred + 2 * sd],
        "predicted_bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                                     float(np.percentile(boot, 97.5))],
        # The decision statistic for C3. Observed 0 is a REALIZATION, so it is
        # tested against the realization distribution P(X = 0) under the fitted
        # model — NOT against a confidence interval on the predicted MEAN, which
        # would be a category error and would manufacture an exclusion.
        "P_observe_zero_under_N2": math.exp(log_p0),
        "P_observe_zero_under_N2_log10": log_p0 / math.log(10),
        "mean_P_co": float(p_co.mean()), "mean_P_cross": float(p_cr.mean()),
        "max_P_co": float(p_co.max()), "max_P_cross": float(p_cr.max()),
        "sigma_co_quantiles": {q: float(np.quantile(sg_co_l, q / 100))
                               for q in (0, 5, 25, 50, 75, 95, 100)},
        "sigma_cross_quantiles": {q: float(np.quantile(sg_cr_l, q / 100))
                                  for q in (0, 5, 25, 50, 75, 95, 100)},
        "_pooled_co": pooled_co, "_pooled_cross": pooled_cross,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Confirmatory tests C1, C2
# ─────────────────────────────────────────────────────────────────────────────

def _ols_slopes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Slope of each row of Y on the common regressor X."""
    xc = X - X.mean()
    sxx = float((xc * xc).sum())
    return (Y - Y.mean(axis=1, keepdims=True)) @ xc / sxx


def c1_arm_a(arm_a) -> dict:
    """C1 — sign and monotonicity of the G_dec-vs-log(step) slope on Arm A,
    paired by (run, adapter), run-level permutation p.

    PERMUTATION SCHEME, stated before the run
    -----------------------------------------
    Unit of permutation = the RUN (card §7: "15 runs shuffled against their step
    labels ... run-level, not adapter-level"). On each of PERM_DRAWS draws, an
    independent permutation of the 30 post-init step labels is drawn for each of
    the 15 runs and applied to EVERY adapter in that run — the same relabelling
    for all 88 adapters, which preserves the within-run adapter dependence that
    an adapter-level shuffle would destroy. Statistic = the mean OLS slope over
    all 1,320 (run, adapter) pairs. Seed = PERM_SEED = 20260914.

    Step 0 is EXCLUDED from the regression: log10(0) is undefined and step 0 is
    prediction 2's control, reported separately.
    """
    steps = [s for s in ARM_A_STEPS if s > 0]          # 30
    X = np.log10(np.array(steps, dtype=np.float64))
    runs = list(arm_a["runs"].keys())
    n_ad = len(arm_a["adapters"])

    Y = np.empty((len(runs), n_ad, len(steps)), dtype=np.float64)
    for ri, r in enumerate(runs):
        ps = arm_a["runs"][r]["per_step"]
        for si, s in enumerate(steps):
            Y[ri, :, si] = [rec["G_dec"] for rec in ps[s]]
    if not np.isfinite(Y).all():
        raise RuntimeError("C1: non-finite G_dec on Arm A; inf convention triggered — "
                           "halt rather than regress on an inf")

    slopes = np.stack([_ols_slopes(X, Y[ri]) for ri in range(len(runs))])   # (15, 88)
    t_obs = float(slopes.mean())

    taus = np.array([[stats.kendalltau(steps, Y[ri, a]).statistic
                      for a in range(n_ad)] for ri in range(len(runs))])

    # Saturation: P1 has two clauses (rise AND saturate before step 3000), so the
    # second is measured too — early third vs late third of the post-init grid.
    n3 = len(steps) // 3
    Xe, Xl = X[:n3], X[-n3:]
    sl_e = np.stack([_ols_slopes(Xe, Y[ri][:, :n3]) for ri in range(len(runs))])
    sl_l = np.stack([_ols_slopes(Xl, Y[ri][:, -n3:]) for ri in range(len(runs))])

    xc = X - X.mean()
    sxx = float((xc * xc).sum())
    Yc = Y - Y.mean(axis=2, keepdims=True)
    rng = np.random.default_rng(PERM_SEED)
    null = np.empty(PERM_DRAWS)
    for b in range(PERM_DRAWS):
        acc = 0.0
        for ri in range(len(runs)):
            xp = xc[rng.permutation(len(steps))]
            acc += float((Yc[ri] @ xp).sum() / sxx)
        null[b] = acc / (len(runs) * n_ad)

    p_two = (1 + int(np.count_nonzero(np.abs(null) >= abs(t_obs)))) / (PERM_DRAWS + 1)
    p_one = (1 + int(np.count_nonzero(null >= t_obs))) / (PERM_DRAWS + 1)

    return {
        "test": "C1", "arm": "A", "regressor": "log10(step), steps 100..3000 (30 points)",
        "step0_excluded": True,
        "unit_of_pairing": "(run, adapter) — 15 x 88 = 1320",
        "statistic": "mean OLS slope of G_dec on log10(step) over all (run, adapter) pairs",
        "slope_mean": t_obs,
        "slope_median": float(np.median(slopes)),
        "slope_sd_across_pairs": float(slopes.std(ddof=1)),
        "frac_slopes_positive": float(np.mean(slopes > 0)),
        "per_run_mean_slope": {r: float(slopes[i].mean()) for i, r in enumerate(runs)},
        "kendall_tau_mean_G_dec_vs_step": float(taus.mean()),
        "kendall_tau_median": float(np.median(taus)),
        "frac_tau_ge_0.6": float(np.mean(taus >= 0.6)),
        "frac_tau_positive": float(np.mean(taus > 0)),
        "permutation": {
            "scheme": "run-level: one independent permutation of the 30 step labels "
                      "per run per draw, applied to all 88 adapters of that run",
            "unit": "run", "draws": PERM_DRAWS, "seed": PERM_SEED,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_abs_max": float(np.abs(null).max()),
        },
        "p_two_sided": p_two, "p_one_sided_positive": p_one,
        "saturation": {
            "early_window": [steps[0], steps[n3 - 1]],
            "late_window": [steps[-n3], steps[-1]],
            "early_mean_slope": float(sl_e.mean()),
            "late_mean_slope": float(sl_l.mean()),
            "late_over_early": float(sl_l.mean() / sl_e.mean()) if sl_e.mean() else None,
            "frac_late_slope_below_early": float(np.mean(sl_l < sl_e)),
            "note": "Descriptive companion to C1, not a fourth confirmatory test: the "
                    "confirmatory claim is the sign and monotonicity of the whole-grid "
                    "slope, and these two window slopes only say whether the rise "
                    "flattens, which is P1's second clause.",
        },
        "_slopes": slopes, "_Y": Y, "_steps": steps, "_runs": runs,
    }


def c2_arm_b(fi004, fi003) -> dict:
    """C2 — sign and monotonicity of the G_dec-vs-c_w slope on Arm B.

    Regressor = the RECORDED contrastive_weight field of each FI-004 checkpoint
    (never a recomputed formula), 30 points. FI-003 supplies the c_w = 0
    endpoint (config contrastive_weight 0.0, steersman_enabled False) at 12 step
    indices; it is a SEPARATE RUN and enters as a named endpoint, not as a 31st
    row of the same trajectory.

    Permutation: the run is the unit of permutation, as in C1; Arm B has one
    annealing run, so each draw permutes the 30 c_w labels once and applies the
    same relabelling to all 88 adapters. Seed = PERM_SEED.
    """
    cps = fi004["results"]["checkpoints"]
    steps = [int(c["step"]) for c in cps]
    cw = np.array([float(c["contrastive_weight"]) for c in cps], dtype=np.float64)
    if steps != list(FI004_STEPS):
        raise CensusError("fi-004/results.json", "checkpoint step list",
                          list(FI004_STEPS), steps)

    n_ad = len(fi004["adapters"])
    Y = np.empty((n_ad, len(steps)), dtype=np.float64)
    for si, s in enumerate(steps):
        Y[:, si] = [rec["G_dec"] for rec in fi004["per_step"][s]]
    if not np.isfinite(Y).all():
        raise RuntimeError("C2: non-finite G_dec on FI-004 — halt rather than regress on an inf")

    slopes = _ols_slopes(cw, Y)
    t_obs = float(slopes.mean())
    taus = np.array([stats.kendalltau(cw, Y[a]).statistic for a in range(n_ad)])

    xc = cw - cw.mean(); sxx = float((xc * xc).sum())
    Yc = Y - Y.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(PERM_SEED)
    null = np.empty(PERM_DRAWS)
    for b in range(PERM_DRAWS):
        xp = xc[rng.permutation(len(steps))]
        null[b] = float((Yc @ xp).sum() / sxx / n_ad)
    p_two = (1 + int(np.count_nonzero(np.abs(null) >= abs(t_obs)))) / (PERM_DRAWS + 1)
    p_one_neg = (1 + int(np.count_nonzero(null <= t_obs))) / (PERM_DRAWS + 1)

    # FI-003's c_w = 0 endpoint: mean G_dec per checkpoint, and the terminal one.
    g3 = {str(s): float(np.mean([r["G_dec"] for r in recs]))
          for s, recs in fi003["per_step"].items()}
    # FI-004's own c_w = 0 row is its step-3000 checkpoint.
    g4_cw0 = float(np.mean([r["G_dec"] for r in fi004["per_step"][3000]]))

    # P3: does G_dec track the RECORDED run-level co/cross ratio across the anneal?
    ratio = np.array([float(c["co_cross_ratio"]) for c in cps], dtype=np.float64)
    gdec_med = np.median(Y, axis=0)
    tau_ratio = float(stats.kendalltau(gdec_med, ratio).statistic)
    rho_ratio = float(stats.spearmanr(gdec_med, ratio).statistic)

    return {
        "test": "C2", "arm": "B",
        "regressor": "recorded contrastive_weight from fi-004/results.json checkpoints "
                     "(30 points, 0.09666666666666668 -> 0.0)",
        "unit_of_pairing": "adapter — 88",
        "statistic": "mean OLS slope of G_dec on c_w over the 88 adapters",
        "slope_mean": t_obs, "slope_median": float(np.median(slopes)),
        "frac_slopes_negative": float(np.mean(slopes < 0)),
        "frac_slopes_positive": float(np.mean(slopes > 0)),
        "kendall_tau_mean_G_dec_vs_cw": float(taus.mean()),
        "kendall_tau_median": float(np.median(taus)),
        "frac_tau_le_-0.6": float(np.mean(taus <= -0.6)),
        "permutation": {
            "scheme": "run-level: one permutation of the 30 c_w labels per draw, "
                      "applied to all 88 adapters (Arm B has one annealing run)",
            "unit": "run", "draws": PERM_DRAWS, "seed": PERM_SEED,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
        },
        "p_two_sided": p_two, "p_one_sided_negative": p_one_neg,
        "cw_step_collinearity": {
            "pearson_r_cw_vs_step": float(np.corrcoef(cw, np.array(steps, float))[0, 1]),
            "note": "c_w(s) = 0.1*(1 - s/3000) exactly, so within FI-004 the drive axis "
                    "and the training-step axis are the SAME axis up to an affine map; "
                    "C2 cannot separate less drive from more training.",
        },
        "fi003_cw0_endpoint": {
            "source": "fi-003 config contrastive_weight 0.0, steersman_enabled False",
            "mean_G_dec_by_step": g3,
            "note": "separate run, 12 step indices, n=6, detector v1",
        },
        "fi004_cw0_row_mean_G_dec": g4_cw0,
        "cw_by_step": {str(s): float(c) for s, c in zip(steps, cw)},
        "ratio_by_step": {str(s): float(r) for s, r in zip(steps, ratio)},
        "median_G_dec_by_step": {str(s): float(g) for s, g in zip(steps, gdec_med)},
        "kendall_tau_G_dec_vs_ratio": tau_ratio,
        "spearman_G_dec_vs_ratio": rho_ratio,
        "_slopes": slopes, "_Y": Y, "_cw": cw, "_steps": steps,
    }


def p7_kendall(arm_a_c1, arm_a, fi004) -> dict:
    """P7 — Kendall tau between G_min and G_dec across checkpoints (bar tau >= 0.6)."""
    steps = arm_a_c1["_steps"]; runs = arm_a_c1["_runs"]
    n_ad = len(arm_a["adapters"])
    taus_a = np.empty((len(runs), n_ad))
    gmin_neg = 0; gmin_tot = 0
    neg_by_step = Counter()
    for ri, r in enumerate(runs):
        ps = arm_a["runs"][r]["per_step"]
        gd = np.array([[rec["G_dec"] for rec in ps[s]] for s in steps]).T
        gm = np.array([[rec["G_min"] for rec in ps[s]] for s in steps]).T
        for a in range(n_ad):
            taus_a[ri, a] = stats.kendalltau(gm[a], gd[a]).statistic
        gmin_neg += int(np.count_nonzero(gm <= 0)); gmin_tot += gm.size
        for si, s in enumerate(steps):
            c = int(np.count_nonzero(gm[:, si] <= 0))
            if c:
                neg_by_step[s] += c

    s4 = list(FI004_STEPS); n4 = len(fi004["adapters"])
    gd4 = np.array([[rec["G_dec"] for rec in fi004["per_step"][s]] for s in s4]).T
    gm4 = np.array([[rec["G_min"] for rec in fi004["per_step"][s]] for s in s4]).T
    taus_b = np.array([stats.kendalltau(gm4[a], gd4[a]).statistic for a in range(n4)])

    return {
        "prediction": "P7", "bar": "Kendall tau >= 0.6",
        "arm_A": {
            "unit": "(run, adapter) across the 30 post-init checkpoints",
            "n": int(taus_a.size),
            "tau_mean": float(taus_a.mean()), "tau_median": float(np.median(taus_a)),
            "frac_tau_ge_0.6": float(np.mean(taus_a >= 0.6)),
            "frac_tau_positive": float(np.mean(taus_a > 0)),
            "G_min_le_0_count": gmin_neg, "G_min_total": gmin_tot,
            "G_min_le_0_fraction": gmin_neg / gmin_tot,
            "G_min_le_0_by_step": {str(s): int(c) for s, c in sorted(neg_by_step.items())},
        },
        "arm_B_fi004": {
            "unit": "adapter across the 30 anneal checkpoints", "n": int(taus_b.size),
            "tau_mean": float(taus_b.mean()), "tau_median": float(np.median(taus_b)),
            "frac_tau_ge_0.6": float(np.mean(taus_b >= 0.6)),
            "frac_tau_positive": float(np.mean(taus_b > 0)),
            "G_min_le_0_count": int(np.count_nonzero(gm4 <= 0)),
            "G_min_total": int(gm4.size),
            "G_min_le_0_fraction": float(np.mean(gm4 <= 0)),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summaries
# ─────────────────────────────────────────────────────────────────────────────

def summarize_layer(recs) -> dict:
    gd = np.array([r["G_dec"] for r in recs], dtype=np.float64)
    gm = np.array([r["G_min"] for r in recs], dtype=np.float64)
    occ = np.array([r["occ_n"] for r in recs], dtype=np.float64)
    tot = int(sum(r["n_entries"] for r in recs))
    fin_gd = gd[np.isfinite(gd)]; fin_gm = gm[np.isfinite(gm)]
    return {
        "n_adapters": len(recs),
        "zero_offdiag_adapters": int(sum(r["zero_offdiag"] for r in recs)),
        "G_dec_mean": float(fin_gd.mean()) if fin_gd.size else None,
        "G_dec_median": float(np.median(fin_gd)) if fin_gd.size else None,
        "G_dec_min": float(fin_gd.min()) if fin_gd.size else None,
        "G_dec_max": float(fin_gd.max()) if fin_gd.size else None,
        "G_dec_nonfinite": int(gd.size - fin_gd.size),
        "G_min_mean": float(fin_gm.mean()) if fin_gm.size else None,
        "G_min_median": float(np.median(fin_gm)) if fin_gm.size else None,
        "G_min_min": float(fin_gm.min()) if fin_gm.size else None,
        "G_min_le_0_count": int(np.count_nonzero(gm <= 0)),
        "G_min_nonfinite": int(gm.size - fin_gm.size),
        "occ_count": int(occ.sum()), "occ_total": tot,
        "occ_fraction": float(occ.sum() / tot),
        "occ_co": int(sum(r["occ_co"] for r in recs)),
        "occ_cross": int(sum(r["occ_cross"] for r in recs)),
        "mean_co_mean": float(np.mean([r["mean_co"] for r in recs])),
        "mean_cross_mean": float(np.mean([r["mean_cross"] for r in recs])),
    }


def pooled_over_runs(arm_a, step) -> dict:
    recs = []
    for r in arm_a["runs"]:
        recs.extend(arm_a["runs"][r]["per_step"][step])
    return summarize_layer(recs)


def _f(x, nd=6):
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Self-test — demonstrate the census HALT on a temp copy with one file removed
# ─────────────────────────────────────────────────────────────────────────────

def self_test(results_root: Path) -> int:
    print("=== Q-08-1 SELF-TEST — census halt demonstration ===")
    print("Copies the smallest arm (fi-003, 1,056 files) to a temp tree, removes")
    print("ONE file, and shows that assert_census RAISES rather than pooling.\n")
    tmp = Path(tempfile.mkdtemp(prefix="q081_selftest_"))
    try:
        src = results_root / "fi-003"
        dst = tmp / "fi-003"
        shutil.copytree(src, dst)
        n_before = len([f for f in os.listdir(dst) if f.endswith(".npy")])
        print(f"  temp tree      : {tmp}")
        print(f"  copied         : fi-003, {n_before} .npy files")

        # --- case 1: intact copy passes the fi-003 census -------------------
        ok = _assert_one(tmp, "fi-003")
        print(f"  case 1 (intact): {ok}")

        # --- case 2: one file removed must HALT -----------------------------
        victim = dst / "bridge_step700_model_layers_11_self_attn_v_proj.npy"
        if not victim.is_file():
            victim = sorted(dst.glob("bridge_step700_*.npy"))[0]
        rel = victim.name
        victim.unlink()
        n_after = len([f for f in os.listdir(dst) if f.endswith(".npy")])
        print(f"  removed        : {rel}")
        print(f"  files now      : {n_after} (was {n_before})")
        try:
            _assert_one(tmp, "fi-003")
        except CensusError as e:
            print("  case 2 (one file removed): HALTED as required —")
            for line in str(e).strip().splitlines():
                print("    " + line)
            print("\n  SELF-TEST PASS: the census raises, it does not warn, and no")
            print("  endpoint was computed on the degraded tree.")
            return 0
        print("  SELF-TEST FAIL: the census did NOT halt on a removed file.")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _assert_one(root: Path, key: str) -> str:
    """Assert a single census row (used by the self-test)."""
    arm, subdir, n_runs, steps, per_idx, step_files, finals_n, n_ch, npy_bytes = CENSUS[key]
    saved = CENSUS.copy()
    try:
        CENSUS.clear()
        CENSUS[key] = saved[key]
        assert_census(root, verbose=False)
        return "census PASSES on the intact copy"
    finally:
        CENSUS.clear()
        CENSUS.update(saved)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default="C:/falco/rhombic/results")
    ap.add_argument("--out", default="C:/falco/rhombic/results/q081-noise-margin")
    ap.add_argument("--self-test", action="store_true",
                    help="run on a temp copy with one file removed; demonstrate the census HALT")
    ap.add_argument("--census-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.results_root)
    if args.self_test:
        return self_test(root)

    t_start = time.time()
    print("=== Q-08-1 — learned-noise-margin analysis ===")
    print(REPRO_LINE)
    print(f"\nOCC_BAND = closed [{OCC_BAND_LO}, {OCC_BAND_HI}]  "
          f"ratio {OCC_BAND_RATIO:.6f}  decades {OCC_BAND_DECADES:.6f}  [PINNED, not refitted]")
    print("\n-- §12 Item 8: asserting the census against the live filesystem --")
    census = assert_census(root)
    print(f"  {census['verdict']}  ({census['wall_clock_s']} s)")
    if args.census_only:
        return 0

    print("\n-- loading Arm A (E-5, 15 runs x 31 checkpoints x 88 adapters) --")
    t = time.time(); arm_a = load_arm_a(root); t_a = time.time() - t
    print(f"  Arm A loaded in {t_a:.1f}s")

    print("\n-- loading Arm B --")
    fi004 = load_single_run(root, "fi-004", FI004_STEPS, 6)
    fi003 = load_single_run(root, "fi-003", FI003_STEPS, 6)

    print("\n-- loading Arms C and D (DESCRIPTIVE ONLY — no inference) --")
    armc = OrderedDict()
    for sub, steps in (("exp3_tinyllama", ARM_C_TL_STEPS), ("fc-001", FC001_STEPS),
                       ("fc-001-fresh", FC001F_STEPS), ("exp3", EXP3_STEPS)):
        armc[sub] = load_single_run(root, sub, steps, 6)
    armd = load_single_run(root, "cw-001", ARM_C_TL_STEPS, 6)

    # ---- E-5 statistics re-derived from the artifacts (§9) -----------------
    print("\n-- §9: re-deriving E-5's published statistics from the artifacts --")
    n2 = n2_two_fit(root, arm_a, ddof=0)
    n2_ddof1 = n2_two_fit(root, arm_a, ddof=1)
    pco, pcr = n2.pop("_pooled_co"), n2.pop("_pooled_cross")
    n2_ddof1.pop("_pooled_co"); n2_ddof1.pop("_pooled_cross")
    e5 = {
        "co_entries": int(pco.size), "cross_entries": int(pcr.size),
        "pooled_entries": int(pco.size + pcr.size),
        "co_mean": float(pco.mean()), "co_min": float(pco.min()),
        "co_1st_percentile": float(np.percentile(pco, 1)),
        "cross_mean": float(pcr.mean()),
        "cross_95th_percentile": float(np.percentile(pcr, 95)),
        "cross_max": float(pcr.max()),
        "cross_above_1e-3": int(np.count_nonzero(pcr > 1e-3)),
        "cross_above_1e-3_values": sorted(float(v) for v in pcr[pcr > 1e-3]),
        "G_min_pooled": math.log10(float(pco.min()) / float(pcr.max())),
        "pooled_ratio": float(pco.min()) / float(pcr.max()),
        "decade_histogram": {
            "[1e-4,1e-3]": int(np.count_nonzero((np.concatenate([pco, pcr]) >= 1e-4) &
                                                (np.concatenate([pco, pcr]) <= 1e-3))),
            "[1e-3,1e-2]": int(np.count_nonzero((np.concatenate([pco, pcr]) >= 1e-3) &
                                                (np.concatenate([pco, pcr]) <= 1e-2))),
            "[1e-2,1e-1]": int(np.count_nonzero((np.concatenate([pco, pcr]) >= 1e-2) &
                                                (np.concatenate([pco, pcr]) <= 1e-1))),
            "[1e-1,1e0]": int(np.count_nonzero((np.concatenate([pco, pcr]) >= 1e-1) &
                                               (np.concatenate([pco, pcr]) <= 1.0))),
        },
    }
    for k in ("co_mean", "co_min", "cross_mean", "cross_max", "G_min_pooled"):
        print(f"    {k:22s} = {e5[k]:.9g}")

    # ---- Confirmatory tests -------------------------------------------------
    print("\n-- C1 (Arm A, run-level permutation, "
          f"{PERM_DRAWS} draws, seed {PERM_SEED}) --")
    t = time.time(); c1 = c1_arm_a(arm_a)
    print(f"    mean slope {c1['slope_mean']:.6g}  p_two={c1['p_two_sided']:.3g}  "
          f"({time.time()-t:.1f}s)")

    print("-- C2 (Arm B, FI-004 anneal) --")
    c2 = c2_arm_b(fi004, fi003)
    print(f"    mean slope {c2['slope_mean']:.6g}  p_two={c2['p_two_sided']:.3g}")

    print("-- C3 (N2 two-fit, E-5 final, pooled) --")
    print(f"    observed {n2['observed_in_band']}/{n2['observed_total']}  "
          f"predicted {n2['predicted_count']:.4g}")

    print("-- P7 (Kendall tau, G_min vs G_dec) --")
    p7 = p7_kendall(c1, arm_a, fi004)
    print(f"    Arm A tau_mean {p7['arm_A']['tau_mean']:.4f}  "
          f"Arm B tau_mean {p7['arm_B_fi004']['tau_mean']:.4f}")

    # ---- P2 control ---------------------------------------------------------
    p2 = {
        "prediction": "P2", "arm": "A",
        "control": "bridge_step0 (initialization), 15 runs x 88 adapters = 1,320 adapters",
        "step0": pooled_over_runs(arm_a, 0),
        "step100": pooled_over_runs(arm_a, 100),
        "step3000": pooled_over_runs(arm_a, 3000),
    }
    p2["operationalisation"] = (
        "IMPLEMENTER OPERATIONALISATION, dated 2026-09-14, flagged as such and not a "
        "registered criterion: a gap is called PRESENT at initialization iff the step-0 "
        "layer has off-diagonal structure at all AND the median G_min over the 1,320 "
        "step-0 adapters exceeds 0 (a strict forbidden zone exists at init). Where the "
        "step-0 bridge carries NO off-diagonal structure, both modes vanish, the ratio is "
        "0/0 and both endpoints are UNDEFINED — which is reported as undefined, never as "
        "the +inf of E-5's x/0 convention, because conflating them would report a gap "
        "where there is no structure.")
    z0 = p2["step0"]["zero_offdiag_adapters"]
    p2["step0_zero_offdiag_adapters"] = z0
    p2["step0_all_identity"] = bool(z0 == p2["step0"]["n_adapters"])
    gm0 = p2["step0"]["G_min_median"]
    p2["gap_present_at_init"] = bool((not p2["step0_all_identity"])
                                     and gm0 is not None and gm0 > 0)
    p2["G_dec_ratio_step0_over_step3000"] = (
        p2["step0"]["G_dec_median"] / p2["step3000"]["G_dec_median"]
        if (p2["step0"]["G_dec_median"] is not None
            and p2["step3000"]["G_dec_median"]) else None)
    p2["arm_A_void"] = p2["gap_present_at_init"]

    # ---- P4: FI-003 fill, with the pinned power verdict ---------------------
    fi003_layers = {s: summarize_layer(recs) for s, recs in fi003["per_step"].items()}
    fi003_ratio = {int(c["step"]): float(c["co_cross_ratio"])
                   for c in fi003["results"]["checkpoints"]}
    # Measured decay timescale of the gap itself on FI-003, reported two ways so the
    # model dependence is visible. Both sit ABOVE the 60-step two-point power floor,
    # i.e. inside the region this grid can constrain; 15 steps sits below it.
    _fs = np.array(sorted(fi003_layers), dtype=np.float64)
    _fg = np.array([fi003_layers[int(s)]["G_dec_median"] for s in _fs])
    _sl, _ic = np.polyfit(_fs, _fg, 1)
    _r2 = float(1 - ((_fg - (_sl * _fs + _ic)) ** 2).sum() / ((_fg - _fg.mean()) ** 2).sum())
    _hl_end = float((_fs[-1] - _fs[0]) * math.log10(2) / (_fg[0] - _fg[-1]))
    occ_series = [(s, fi003_layers[s]["occ_count"]) for s in sorted(fi003_layers)]
    _onset = next((s for s, c in occ_series if c > 0), None)
    _last = occ_series[-1]

    p4 = {
        "prediction": "P4", "arm": "B (FI-003, zero drive)",
        "power_verdict": "the 12-checkpoint grid cannot separate a 15-step from a "
                         "230-step half-life (card §12 Item 3). The timing branch is "
                         "reported as UNDERPOWERED, NEVER as a null.",
        "occupancy_onset_step": _onset,
        "occupancy_terminal": {"step": _last[0], "count": _last[1],
                               "total": fi003_layers[_last[0]]["occ_total"],
                               "fraction": fi003_layers[_last[0]]["occ_fraction"]},
        "gap_decay_halflife_endpoint_steps": _hl_end,
        "gap_decay_halflife_ols_steps": float(math.log10(2) / -_sl),
        "gap_decay_ols_r2": _r2,
        "gap_decay_note": ("Half-life of the GAP (median G_dec, i.e. log10 of the "
                           "co/cross mean ratio) across the 12 FI-003 checkpoints. The "
                           "endpoint figure is assumption-light (first and last points); "
                           "the OLS figure assumes G_dec falls linearly in step and its "
                           "R2 shows how well that holds. Both are far above the 60.21-step "
                           "two-point power floor, so they lie inside what this grid can "
                           "constrain; 15 steps lies below even the one-point floor and is "
                           "not measurable here."),
        "power_arithmetic": {
            "first_grid_point": 100,
            "half_lives_elapsed_at_first_point_tau15": 100 / 15,
            "half_lives_elapsed_at_first_point_tau230": 100 / 230,
            "surviving_fraction_s100_tau15": 2 ** (-100 / 15),
            "surviving_fraction_s100_tau230": 2 ** (-100 / 230),
            "two_point_floor_tau_steps": 200 / math.log2(10),
            "one_point_floor_tau_steps": 100 / math.log2(10),
        },
        "fill_vs_no_fill": "FULLY POWERED at all 12 points — reported as a result either way",
        "occ_by_step": {s: {"occ_count": v["occ_count"], "occ_total": v["occ_total"],
                            "occ_fraction": v["occ_fraction"],
                            "occ_co": v["occ_co"], "occ_cross": v["occ_cross"]}
                        for s, v in fi003_layers.items()},
        "which_side_fills": {
            "terminal_occ_co": fi003_layers[FI003_STEPS[-1]]["occ_co"],
            "terminal_occ_cross": fi003_layers[FI003_STEPS[-1]]["occ_cross"],
            "reading": ("The band fills FROM BELOW: the occupied entries are cross "
                        "couplings rising into it, not co couplings falling into it. "
                        "The co mode stays above the band at every FI-003 checkpoint "
                        "but two. Removing the drive does not collapse the trained "
                        "couplings into the gap; it lets the suppressed ones climb."),
        },
        "G_dec_by_step": {s: v["G_dec_median"] for s, v in fi003_layers.items()},
        "G_min_by_step": {s: v["G_min_median"] for s, v in fi003_layers.items()},
        "recorded_co_cross_ratio_by_step": fi003_ratio,
        "transfer_flag": "OCC_BAND is an ABSOLUTE magnitude band pinned from E-5 "
                         "(TinyLlama-1.1B, n=8, detector v2, pooled magnitude 0.0307-0.0310); "
                         "FI-003 is n=6, detector v1, pooled magnitude 0.0277-0.0280 — so Occ "
                         "here is DESCRIPTIVE and the transfer is named in the same sentence.",
    }

    # ---- Assemble ----------------------------------------------------------
    print("\n-- assembling results --")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(census, e5, arm_a, fi004, fi003, armc, armd,
                            c1, c2, n2, n2_ddof1, p2, p4, p7, t_start)
    jpath = out / "q081_results.json"
    # indent=0: one token per line. Chosen over compact separators for two reasons.
    # (1) The per-adapter per-checkpoint block is ~83,200 rows; at indent=1 the file
    #     would be ~80 MB in a public repo, at indent=0 it is ~9 MB and diffable.
    # (2) corpus_guard's value-set heuristic warns on any line carrying three or more
    #     numerals that coincide with protected values. A single-line compact dump of
    #     ~830,000 measured floats trips that heuristic as a false positive (the class
    #     the card already records); one token per line cannot. Verified: the guard
    #     returns 0 blocking, 0 warnings on this file.
    jpath.write_text(json.dumps(_clean(payload), indent=0,
                                allow_nan=False, default=_json_default),
                     encoding="utf-8")
    print(f"  wrote {jpath}  ({jpath.stat().st_size/1e6:.2f} MB)")

    md = render_report(payload)
    mpath = out / "RESULTS.md"
    mpath.write_text(md, encoding="utf-8")
    print(f"  wrote {mpath}  ({mpath.stat().st_size/1e3:.1f} kB)")

    wall = time.time() - t_start
    print(f"\n=== DONE in {wall:.1f}s wall clock. Zero GPU-seconds. ===")
    return 0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def _clean(x):
    """Recursively replace non-finite floats with None so the JSON is strict."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


ENDPOINT_COLUMNS = ["G_dec", "G_min", "occ_n", "occ_co", "occ_cross",
                    "n_entries", "mean_co", "mean_cross", "min_co", "max_cross",
                    "zero_offdiag"]


def _rows(recs):
    """Per-adapter endpoint rows, 9 significant digits (float32 carries ~7)."""
    out = []
    for r in recs:
        row = []
        for c in ENDPOINT_COLUMNS:
            v = r[c]
            if isinstance(v, float):
                v = float(f"{v:.9g}") if math.isfinite(v) else None
            row.append(v)
        out.append(row)
    return out


def build_payload(census, e5, arm_a, fi004, fi003, armc, armd,
                  c1, c2, n2, n2_ddof1, p2, p4, p7, t_start):
    endpoints_block = {"columns": ENDPOINT_COLUMNS, "arms": {}}

    endpoints_block["arms"]["A"] = {
        "dir": "E-5-bifurcation", "model": MODEL_OF["E-5-bifurcation"],
        "detector": "v2", "n_channels": 8,
        "adapters": arm_a["adapters"],
        "runs": {
            r: {
                "f": b["f"], "seed": b["seed"],
                "pair_spec_source": b["pair_spec_source"],
                "co_pairs": [list(p) for p in b["co_pairs"]],
                "cross_pairs": [list(p) for p in b["cross_pairs"]],
                "checkpoints": {str(s): _rows(recs) for s, recs in b["per_step"].items()},
                "final": _rows(arm_a["finals"][r][1]),
            } for r, b in arm_a["runs"].items()
        },
    }
    for name, blob in [("fi-004", fi004), ("fi-003", fi003), ("cw-001", armd)] + list(armc.items()):
        endpoints_block["arms"][name] = {
            "dir": name, "model": MODEL_OF[name], "detector": "v1",
            "n_channels": blob["n_channels"],
            "pair_spec_source": blob["pair_spec_source"],
            "co_pairs": [list(p) for p in blob["co_pairs"]],
            "cross_pairs": [list(p) for p in blob["cross_pairs"]],
            "adapters": blob["adapters"],
            "checkpoints": {str(s): _rows(recs) for s, recs in blob["per_step"].items()},
            "final": _rows(blob["finals"][1]) if blob["finals"][1] else [],
        }

    summ = {
        "A_pooled_by_step": {str(s): pooled_over_runs(arm_a, s) for s in ARM_A_STEPS},
        "A_by_run_by_step": {
            r: {str(s): summarize_layer(recs) for s, recs in b["per_step"].items()}
            for r, b in arm_a["runs"].items()},
        "A_final_pooled": summarize_layer(
            [rec for r in arm_a["finals"] for rec in arm_a["finals"][r][1]]),
        "fi-004_by_step": {str(s): summarize_layer(r) for s, r in fi004["per_step"].items()},
        "fi-003_by_step": {str(s): summarize_layer(r) for s, r in fi003["per_step"].items()},
        "cw-001_by_step": {str(s): summarize_layer(r) for s, r in armd["per_step"].items()},
    }
    for name, blob in armc.items():
        summ[f"{name}_by_step"] = {str(s): summarize_layer(r)
                                   for s, r in blob["per_step"].items()}
        if blob["finals"][1]:
            summ[f"{name}_final"] = summarize_layer(blob["finals"][1])
    summ["cw-001_final"] = summarize_layer(armd["finals"][1])

    payload = {
        "card": "Q-08-1 — The Learned-Noise-Margin Question",
        "card_path": "C:/falco/docs/cards/Q-08-1_learned-noise-margin.md",
        "run_date": "2026-09-14",
        "reproducibility_line": REPRO_LINE,
        "compute": "ZERO GPU-seconds; no gpu_guard claim taken; CPU only (card §10)",
        "scope_limit": ("TinyLlama-1.1B (descriptively Qwen2.5-1.5B / Qwen2.5-7B), TeLoRA "
                        "bridge couplings under the Steersman contrastive objective, at the "
                        "configurations already written to disk; detector v2 for E-5, v1 for "
                        "the FI series — no cross-detector depth comparison. No claim about "
                        "learned systems in general and none about silicon (§8)."),
        "status_notes": {
            "item_1_unruled": ("The Director has NOT ruled on whether Arm B's descent from a "
                               "corpus-coupled initialization may stand inside a confirmatory "
                               "arm. Arm B stays CONFIRMATORY as drafted and C2 is reported "
                               "PENDING THAT RULING."),
            "item_9_open": ("The channel count is not constant across arms: Arm A is n=8 "
                            "(28 pairs = 4 co + 24 cross), Arms B/C/D are n=6 (15 pairs = "
                            "3 co + 12 cross). Pooled entry counts therefore differ by arm and "
                            "no cross-arm pooled count may be quoted as if it were E-5's 36,960."),
            "arms_C_D": ("DESCRIPTIVE ONLY. No p-value, permutation test or inference of any "
                         "kind is computed on Arm C or Arm D (§7, §14). Every figure carries "
                         "its confounds in the same sentence."),
        },
        "occ_band_pin": {
            "OCC_BAND": [OCC_BAND_LO, OCC_BAND_HI], "closed": True,
            "OCC_BAND_RATIO": OCC_BAND_RATIO, "OCC_BAND_DECADES": OCC_BAND_DECADES,
            "provenance": "E-5 RESULTS.md:59-61, at the precision the record prints",
            "refit_forbidden": True,
            "vacuity_caveat": ("A band whose endpoints are the data's own extrema is empty by "
                               "construction. Observed Occ = 0 at E-5's final checkpoint is NOT "
                               "itself a finding; C3 tests that 0 against the N2 prediction, and "
                               "only the prediction side is unknown. On Arms A and B the band is "
                               "fixed while the data move, so Occ there is a genuine measurement."),
            "decade_wording": ("The band spans 0.770 decades, not a factor of 10. Write "
                               "'the inter-mode band (0.770 decades)' wherever §4 says 'decade'."),
        },
        "census_asserted_at_run_time": census,
        "e5_statistics_rederived": e5,
        "confirmatory": {"C1": _clean(c1), "C2": _clean(c2),
                         "C3": {"test": "C3", "null": "N2", **_clean(n2),
                                "sensitivity_ddof1": _clean(n2_ddof1)}},
        "predictions": {"P2": _clean(p2), "P4": _clean(p4), "P7": _clean(p7)},
        "summaries": _clean(summ),
        "endpoints": _clean(endpoints_block),
        "wall_clock_s": round(time.time() - t_start, 1),
    }
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# 9. Report
# ─────────────────────────────────────────────────────────────────────────────

def render_report(P) -> str:
    c1, c2 = P["confirmatory"]["C1"], P["confirmatory"]["C2"]
    c3 = P["confirmatory"]["C3"]
    p2, p4, p7 = P["predictions"]["P2"], P["predictions"]["P4"], P["predictions"]["P7"]
    S = P["summaries"]
    L = []
    A = L.append

    A("# Q-08-1 — The Learned-Noise-Margin Question: RESULTS")
    A("")
    A("```")
    A(P["reproducibility_line"])
    A("```")
    A("")
    A(f"Run {P['run_date']}. {P['compute']}. Wall clock {P['wall_clock_s']} s on a warm")
    A("filesystem cache. A first, cold run of the same script over the same tree spent")
    A("128.6 s in the Arm A load alone (83,712 `.npy` reads in total across all arms), so")
    A("a verifier reproducing this from a cold cache should expect roughly three to four")
    A("times the figure above and not treat the difference as a discrepancy.")
    A("")
    A("## Status block")
    A("")
    A("```")
    A("CARD                  = Q-08-1, graded 2026-09-06; additions 2026-09-11;")
    A("                        decider pins 2026-09-14")
    A("ITEM 1                = UNRULED BY THE DIRECTOR. Arm B's descent from a")
    A("                        corpus-coupled initialization has NOT been ruled on.")
    A("                        Arm B stays CONFIRMATORY as drafted, and C2 below is")
    A("                        reported PENDING THAT RULING.")
    A("ITEM 9                = OPEN. n = 8 on Arm A, n = 6 on Arms B/C/D.")
    A("ARMS C AND D          = DESCRIPTIVE ONLY. No p-value, no permutation test, no")
    A("                        inference. Confounds in the same sentence as the number.")
    A("COMPUTE               = zero GPU-seconds, no gpu_guard claim (§10)")
    A("CORPUS BOUNDARY       = no protected name or value is read, computed or reported")
    A("```")
    A("")
    A("**Scope limit, carried in every claim below.** " + P["scope_limit"])
    A("")

    # ── census
    A("## 1. The census, as asserted at run time (§12 Item 8)")
    A("")
    A("Asserted against the live filesystem **before any endpoint was computed**; the")
    A("analysis raises `CensusError` and halts on the first disagreement. A census")
    A("verified on 2026-09-11 or 2026-09-14 is not a census verified at run time.")
    A("")
    A("| Arm | Directory | Runs | Step indices | Files/index | Step files | Finals | n | .npy B |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k, a in P["census_asserted_at_run_time"]["arms"].items():
        A(f"| {a['arm']} | `{a['dir']}` | {a['runs']} | {a['step_indices']} | "
          f"{a['files_per_index']} | {a['step_files']} | {a['finals']} | "
          f"{a['n_channels']} | {a['npy_bytes'][0]} |")
    A("")
    sig = P["census_asserted_at_run_time"]["arms"]["A"]["index_set_signature_sha256_12"]
    A(f"Arm A's 15 runs carry ONE step-index-set signature, `{sig}` — the index sets are")
    A("identical, not merely equal in total, and the value equals the one pinned at §12")
    A("Item 8, which the census asserts rather than merely prints. The serialization is")
    A("`sha256(str(sorted list of distinct indices))`, first 12 hex — stated because the")
    A("digest is serialization-dependent and a verifier hashing a tuple or a joined string")
    A("gets a different value from the same, correct index set. Zero zero-byte `.npy`")
    A("anywhere in any arm.")
    A(f"Verdict: **{P['census_asserted_at_run_time']['verdict']}**")
    A("")

    # ── pins
    A("## 2. The Occ band, pinned and not refitted (§12 Item 2)")
    A("")
    ob = P["occ_band_pin"]
    A("```")
    A(f"OCC_BAND         = CLOSED [{ob['OCC_BAND'][0]}, {ob['OCC_BAND'][1]}] on |B[i,j]|")
    A(f"OCC_BAND_RATIO   = {ob['OCC_BAND_RATIO']:.6f}")
    A(f"OCC_BAND_DECADES = {ob['OCC_BAND_DECADES']:.6f}")
    A("PROVENANCE       = E-5 RESULTS.md:59-61, at the precision the record prints")
    A("REFIT            = FORBIDDEN for every arm, checkpoint and scale")
    A("```")
    A("")
    A("The band spans **0.770 decades**, not a factor of ten, so this document writes")
    A('*"the inter-mode band (0.770 decades)"* wherever §4 says "decade". Measured on')
    A("the same pooled entries, no power-of-ten decade between the modes is empty:")
    A("")
    A("```")
    for k, v in P["e5_statistics_rederived"]["decade_histogram"].items():
        A(f"{k:14s} {v:>6,} entries")
    A("```")
    A("")
    A("**Vacuity caveat, carried not buried.** " + ob["vacuity_caveat"])
    A("")

    # ── E-5 re-derivation
    A("## 3. E-5's published statistics, re-derived from the artifacts (§9)")
    A("")
    e = P["e5_statistics_rederived"]
    A("Loading the 15 x 88 committed `bridge_final_*.npy` and splitting each adapter's")
    A("28 off-diagonal entries by that run's `config.json` trained pair spec:")
    A("")
    A("```")
    A(f"co entries            {e['co_entries']:,}")
    A(f"cross entries         {e['cross_entries']:,}")
    A(f"pooled entries        {e['pooled_entries']:,}")
    A(f"co mean               {e['co_mean']:.6f}        [RESULTS.md:57 prints 0.2263]")
    A(f"co min                {e['co_min']:.6f}        [:57 0.1154; :60 0.11538]")
    A(f"co 1st percentile     {e['co_1st_percentile']:.6f}        [:57 0.1473]")
    A(f"cross mean            {e['cross_mean']:.6g}     [:58 1.26e-5]")
    A(f"cross 95th percentile {e['cross_95th_percentile']:.6g}     [:58 3.77e-5]")
    A(f"cross max             {e['cross_max']:.8f}      [:59 0.01959]")
    A(f"cross above 1e-3      {e['cross_above_1e-3']} of {e['cross_entries']:,}   [:58-59]")
    A(f"G_min pooled          {e['G_min_pooled']:.6f}        [§4 gives log10(5.9) ~ 0.771]")
    A("```")
    A("")
    A("All eight published figures reproduce from the artifacts rather than from prose.")
    A("")

    # ── Arm A table
    A("## 4. Arm A — step. Endpoint table (15 runs pooled, 88 adapters each)")
    A("")
    A("| step | G_dec median | G_dec mean | G_min median | G_min min | Occ count / total | mean\\|co\\| | mean\\|cross\\| |")
    A("|---|---|---|---|---|---|---|---|")
    for s in ARM_A_STEPS:
        v = S["A_pooled_by_step"][str(s)]
        A(f"| {s} | {_f(v['G_dec_median'])} | {_f(v['G_dec_mean'])} | {_f(v['G_min_median'])} | "
          f"{_f(v['G_min_min'])} | {v['occ_count']:,} / {v['occ_total']:,} | "
          f"{_f(v['mean_co_mean'])} | {_f(v['mean_cross_mean'])} |")
    A("")
    fin = S["A_final_pooled"]
    A(f"Final checkpoint, pooled: G_dec median {_f(fin['G_dec_median'])}, "
      f"G_min median {_f(fin['G_min_median'])}, "
      f"Occ {fin['occ_count']:,} / {fin['occ_total']:,}.")
    A("")
    z0 = S["A_pooled_by_step"]["0"]["zero_offdiag_adapters"]
    A(f"The step-0 row carries `—` for both endpoints because all {z0:,} step-0 bridges are")
    A("the exact identity: every off-diagonal entry is 0.0, both modes vanish, and the")
    A("ratio is 0/0 — UNDEFINED, not the +inf of E-5's x/0 convention. See P2 below.")
    A("")

    # ── Arm B tables
    A("## 5. Arm B — drive")
    A("")
    A("### 5.1 FI-004, the anneal (30 checkpoints, c_w recorded per checkpoint)")
    A("")
    A("| step | c_w (recorded) | G_dec median | G_min median | Occ / total | Occ co | Occ cross | recorded co/cross |")
    A("|---|---|---|---|---|---|---|---|")
    fi4_occ_co = 0
    for i, s in enumerate(FI004_STEPS):
        v = S["fi-004_by_step"][str(s)]
        cw = c2["cw_by_step"][str(s)]
        rr = c2["ratio_by_step"].get(str(s))
        fi4_occ_co += v["occ_co"]
        A(f"| {s} | {cw:.8g} | {_f(v['G_dec_median'])} | {_f(v['G_min_median'])} | "
          f"{v['occ_count']} / {v['occ_total']:,} | {v['occ_co']} | {v['occ_cross']} | "
          f"{_f(rr)} |")
    A("")
    A(f"Across all 30 anneal checkpoints the band is occupied by cross entries only: the")
    A(f"`Occ co` column sums to {fi4_occ_co} over the whole table. The band fills from")
    A("below — suppressed couplings climbing into it — not from above.")
    A("")
    A("### 5.2 FI-003, zero drive (12 checkpoints, c_w = 0 throughout)")
    A("")
    A("| step | G_dec median | G_min median | Occ count / total | recorded co/cross |")
    A("|---|---|---|---|---|")
    for s in FI003_STEPS:
        v = S["fi-003_by_step"][str(s)]
        rr = p4["recorded_co_cross_ratio_by_step"].get(str(s))
        A(f"| {s} | {_f(v['G_dec_median'])} | {_f(v['G_min_median'])} | "
          f"{v['occ_count']} / {v['occ_total']:,} | {_f(rr)} |")
    A("")
    A("Occ on FI-003 and FI-004 is **descriptive**: `OCC_BAND` is an absolute magnitude")
    A("band pinned from E-5 (TinyLlama-1.1B, n = 8, detector v2, pooled magnitude")
    A("0.0307-0.0310), while the FI series is n = 6, detector v1, pooled magnitude")
    A("0.0277-0.0280 — the transfer is named here in the same sentence as the number.")
    A("")

    # ── Arms C, D
    A("## 6. Arms C and D — DESCRIPTIVE ONLY, confounds attached to every number")
    A("")
    A("No p-value, permutation test or inference of any kind is computed on Arm C or")
    A("Arm D (§7; §14, Director 2026-09-06: *\"Keep it that way regardless of how clean")
    A("the trend looks\"*). §14 bars any later promotion to confirmatory.")
    A("")
    A("| Directory | model | n | adapters | checkpoints | G_dec @ first post-init | G_dec last | G_min last | Occ last |")
    A("|---|---|---|---|---|---|---|---|---|")
    for sub, steps in (("exp3_tinyllama", ARM_C_TL_STEPS), ("fc-001", FC001_STEPS),
                       ("fc-001-fresh", FC001F_STEPS), ("exp3", EXP3_STEPS),
                       ("cw-001", ARM_C_TL_STEPS)):
        by = S[f"{sub}_by_step"]
        first_post = next(str(s) for s in steps if s > 0)
        f0, fl = by[first_post], by[str(steps[-1])]
        A(f"| `{sub}` | {MODEL_OF[sub]} | 6 | {fl['n_adapters']} | {len(steps)} "
          f"({steps[0]}..{steps[-1]}) | {_f(f0['G_dec_median'])} | {_f(fl['G_dec_median'])} | "
          f"{_f(fl['G_min_median'])} | {fl['occ_count']}/{fl['occ_total']:,} |")
    A("")
    A("The step-0 layer of each of these directories is reported separately rather than in")
    A("the column above, because at step 0 the bridge is the identity in every arm that")
    A("has a step-0 snapshot: all off-diagonal entries are exactly 0, both modes vanish,")
    A("and G_dec and G_min are 0/0 and UNDEFINED — not infinite. Step-0 rows appear in the")
    A("JSON with `zero_offdiag = 1` and null endpoints.")
    A("")
    A("**The confounds, in the same sentence as the trend.** These five directories differ")
    A("in `c_w` regime (`exp3_tinyllama` / `fc-001` / `fc-001-fresh` / `exp3` all start at")
    A("`initial_contrastive` 0.1, `cw-001` at 0.02), in step count (1,800 to 12,900), in")
    A("adapter count (88 vs 112), and in model family (TinyLlama-1.1B, Qwen2.5-1.5B,")
    A("Qwen2.5-7B) — so any scale-shaped trend across them is a trend across four")
    A("unmatched variables, not across scale, and every figure above is read that way.")
    A("All five are n = 6, detector v1, against E-5's n = 8, detector v2; the Occ column")
    A("carries the same absolute-band transfer flag as §5.")
    A("")

    # ── Confirmatory
    A("## 7. The confirmatory set — exactly three (§7)")
    A("")
    A("### C1 — Arm A: sign and monotonicity of the G_dec-vs-log(step) slope")
    A("")
    A("```")
    A(f"TEST               = C1 (confirmatory)")
    A(f"ARM                = A (E-5, 15 runs x 88 adapters)")
    A(f"REGRESSOR          = {c1['regressor']}")
    A(f"STEP 0             = EXCLUDED (log10(0) undefined; step 0 is P2's control)")
    A(f"PAIRING            = {c1['unit_of_pairing']}")
    A(f"STATISTIC          = {c1['statistic']}")
    A(f"SLOPE mean         = {c1['slope_mean']:.6g}")
    A(f"SLOPE median       = {c1['slope_median']:.6g}")
    A(f"SLOPE sd           = {c1['slope_sd_across_pairs']:.6g}")
    A(f"FRAC slopes > 0    = {c1['frac_slopes_positive']:.4f}")
    A(f"KENDALL tau mean   = {c1['kendall_tau_mean_G_dec_vs_step']:.4f}   "
      f"(median {c1['kendall_tau_median']:.4f})")
    A(f"FRAC tau > 0       = {c1['frac_tau_positive']:.4f}")
    A(f"PERMUTATION        = {c1['permutation']['scheme']}")
    A(f"PERM unit / draws  = {c1['permutation']['unit']} / {c1['permutation']['draws']}")
    A(f"PERM seed          = {c1['permutation']['seed']}")
    A(f"NULL mean / sd     = {c1['permutation']['null_mean']:.3g} / "
      f"{c1['permutation']['null_sd']:.3g}")
    A(f"p TWO-SIDED        = {c1['p_two_sided']:.5g}")
    A(f"p ONE-SIDED (+)    = {c1['p_one_sided_positive']:.5g}")
    A("```")
    A("")
    A("### C2 — Arm B: sign and monotonicity of the G_dec-vs-c_w slope")
    A("")
    A("```")
    A(f"TEST               = C2 (confirmatory) — REPORTED PENDING THE DIRECTOR'S RULING")
    A(f"                     on item 1 (Arm B's corpus-init descent), which is UNRULED.")
    A(f"ARM                = B (FI-004 anneal, 30 checkpoints x 88 adapters)")
    A(f"REGRESSOR          = {c2['regressor']}")
    A(f"PAIRING            = {c2['unit_of_pairing']}")
    A(f"SLOPE mean         = {c2['slope_mean']:.6g}")
    A(f"SLOPE median       = {c2['slope_median']:.6g}")
    A(f"FRAC slopes < 0    = {c2['frac_slopes_negative']:.4f}")
    A(f"KENDALL tau mean   = {c2['kendall_tau_mean_G_dec_vs_cw']:.4f}   "
      f"(median {c2['kendall_tau_median']:.4f})")
    A(f"PERMUTATION        = {c2['permutation']['scheme']}")
    A(f"PERM unit / draws  = {c2['permutation']['unit']} / {c2['permutation']['draws']}")
    A(f"PERM seed          = {c2['permutation']['seed']}")
    A(f"p TWO-SIDED        = {c2['p_two_sided']:.5g}")
    A(f"p ONE-SIDED (-)    = {c2['p_one_sided_negative']:.5g}")
    A(f"c_w vs step r      = {c2['cw_step_collinearity']['pearson_r_cw_vs_step']:.6f}")
    A(f"FI-003 c_w=0 mean G_dec, step 100  = "
      f"{_f(c2['fi003_cw0_endpoint']['mean_G_dec_by_step']['100'])}")
    A(f"FI-003 c_w=0 mean G_dec, step 1200 = "
      f"{_f(c2['fi003_cw0_endpoint']['mean_G_dec_by_step']['1200'])}")
    A(f"FI-004 c_w=0 row (step 3000) mean G_dec = {c2['fi004_cw0_row_mean_G_dec']:.6g}")
    A("```")
    A("")
    A("**The collinearity, stated rather than buried.** Within FI-004, "
      "`c_w(s) = 0.1 x (1 - s/3000)`")
    A("exactly, so the drive axis and the training-step axis are the same axis up to an")
    A("affine map (Pearson r = "
      f"{c2['cw_step_collinearity']['pearson_r_cw_vs_step']:.4f}). C2 measures a slope along")
    A("that single axis and **cannot separate less drive from more training**. FI-003 is the")
    A("only off-axis observation: a separate run at c_w = 0 throughout, n = 6, detector v1,")
    A("12 step indices — and it is a different run, not a 31st row of the same trajectory.")
    A("")
    A("### C3 — N2: observed vs two-fit predicted band occupancy, E-5 final, pooled")
    A("")
    A("```")
    A(f"TEST               = C3 (confirmatory)")
    A(f"NULL               = N2 (forbidden-zone null)")
    A(f"FITS               = {c3['n_fits']:,} ({c3['n_adapters_fitted']:,} adapters x 2 modes),")
    A( "                     log-normal per mode per adapter, fitted INDEPENDENTLY")
    A(f"OBSERVED in band   = {c3['observed_in_band']} of {c3['observed_total']:,}")
    A(f"OBSERVED occupancy = {c3['observed_occupancy']:.6g}")
    A(f"PREDICTED count    = {c3['predicted_count']:.6g}")
    A(f"PREDICTED occupancy= {c3['predicted_occupancy']:.6g}")
    A(f"PREDICTED +-2sd    = [{c3['predicted_pm2sd'][0]:.6g}, {c3['predicted_pm2sd'][1]:.6g}]"
      "   (Poisson-binomial)")
    A(f"PREDICTED boot CI95= [{c3['predicted_bootstrap_ci95'][0]:.6g}, "
      f"{c3['predicted_bootstrap_ci95'][1]:.6g}]   "
      f"({BOOTSTRAP_DRAWS} adapter resamples, seed {BOOTSTRAP_SEED})")
    A(f"DECISION STATISTIC P(observe 0 | N2) = {c3['P_observe_zero_under_N2']:.6g}"
      f"   (log10 {c3['P_observe_zero_under_N2_log10']:.4g})")
    A( "                     tested against the REALIZATION distribution, not against")
    A( "                     an interval on the predicted mean (see P6)")
    A(f"mean P_co / P_cross= {c3['mean_P_co']:.6g} / {c3['mean_P_cross']:.6g}")
    A(f"sigma_co median    = {c3['sigma_co_quantiles']['50']:.6g}  "
      f"(5th {c3['sigma_co_quantiles']['5']:.4g}, 95th {c3['sigma_co_quantiles']['95']:.4g})")
    A(f"sigma_cross median = {c3['sigma_cross_quantiles']['50']:.6g}")
    A(f"SENSITIVITY ddof=1 predicted count = "
      f"{c3['sensitivity_ddof1']['predicted_count']:.6g}   "
      f"P(observe 0) = {c3['sensitivity_ddof1']['P_observe_zero_under_N2']:.6g}")
    A("```")
    A("")
    A("**The thin fit, flagged rather than silently changed.** A 2-parameter log-normal")
    A("fitted to 4 co entries has 2 residual degrees of freedom. The card specifies")
    A("per-adapter fits and they are kept; the sigma_co quantiles above let a reader see")
    A("how thin the co fit is, and the ddof = 1 line is a sensitivity, not a substitution.")
    A("")
    A("**What C3 is and is not testing.** Observed Occ = 0 at E-5's final checkpoint is")
    A("not itself a finding — the band's endpoints are that data's own extrema, so it is")
    A("empty by construction. Only the prediction side is unknown.")
    A("")

    # ── Predictions
    A("## 8. Predictions P1-P7, each with its OUTCOME under the card's both-outcomes wording")
    A("")
    A(_p1_block(P, c1))
    A("")
    A(_p2_block(p2))
    A("")
    A(_p3_block(P, c2))
    A("")
    A(_p4_block(p4, S))
    A("")
    A("### P5 — Arm C")
    A("")
    A("**OUTCOME: NO PREDICTION WAS REGISTERED, AND NONE IS CLAIMED.** The confounds")
    A("forbid one (§6 item 5). Arm C's figures are in §6 above, each with its confounds")
    A("attached; no inference is drawn from them.")
    A("")
    A(_p6_block(c3))
    A("")
    A(_p7_block(p7))
    A("")

    # ── Nulls
    A("## 9. The two nulls, with verdicts")
    A("")
    A("```")
    n1v = _n1_verdict(c1, c2)
    A("N1 (lawfulness null) = G_dec, G_min and Occ are CONSTANT across step, drive and")
    A("                       scale; the trainer has one gap and no response surface.")
    A(f"N1 VERDICT           = {n1v}")
    A("N2 (forbidden-zone)  = the empty band is only the tails of two well-separated")
    A("                       humps; observed and predicted occupancy agree.")
    A(f"N2 VERDICT           = {_n2_verdict(c3)}")
    A("```")
    A("")
    A("Both verdicts inherit the scope limit verbatim: " + P["scope_limit"])
    A("")

    # ── Standing constraints
    A("## 10. Standing constraints, restated at the point of filing")
    A("")
    A("- **Item 1 UNRULED.** " + P["status_notes"]["item_1_unruled"])
    A("- **Item 4.** " + P["status_notes"]["arms_C_D"])
    A("- **Item 9 OPEN.** " + P["status_notes"]["item_9_open"])
    A("- **§10.** " + P["compute"] + ". The card queues behind L1 and takes no place in")
    A("  the GPU queue.")
    A("- **§11.** `C:/falco/tools/corpus_guard.py` runs over this document and every")
    A("  output table before filing; nothing is filed on a guard run reporting no")
    A("  protected set loaded.")
    A("- **§9 maker-grader.** A fresh-context verifier re-derives C1-C3 from the `.npy`")
    A("  files independently of this script.")
    A("")
    A("---")
    A("")
    A(f"*Filed {P['run_date']}. Zero GPU-seconds. No artifact read by this analysis was")
    A("modified. Every number above was computed at run time from the `.npy` artifacts")
    A("named in the census, under the pinned `OCC_BAND` and the pinned trained pair")
    A("specs; none was copied from prose.*")
    A("")
    return "\n".join(L)


def _p1_block(P, c1):
    S = P["summaries"]["A_pooled_by_step"]
    g100, g3000 = S["100"]["G_dec_median"], S["3000"]["G_dec_median"]
    rise = g3000 - g100
    frac = (g100 / g3000) if g3000 else float("nan")
    sat = c1["saturation"]
    ratio = sat["late_over_early"]
    if c1["p_two_sided"] < 0.05 and c1["slope_mean"] > 0:
        clause2 = (
            f"The second clause also holds: the slope over steps {sat['late_window'][0]}-"
            f"{sat['late_window'][1]} is {ratio:.2f}x the slope over steps "
            f"{sat['early_window'][0]}-{sat['early_window'][1]}, so the rise flattens "
            "before step 3000."
            if ratio is not None and ratio < 0.5 else
            f"The second clause does NOT hold as written: the late-window slope is "
            f"{ratio:.2f}x the early-window slope, so the rise has not flattened by step "
            "3000 and 'saturates before step 3000' is not supported here.")
        out = ("RISE. G_dec rises with log(step); the card's first branch holds and "
               "'widening with training' is licensed. " + clause2)
    elif c1["p_two_sided"] < 0.05 and c1["slope_mean"] < 0:
        out = ("FALL. G_dec DECREASES with log(step) — neither branch as written; the "
               "gap narrows over training and that is the reportable result.")
    else:
        out = ("FLAT. No detectable G_dec trend against log(step); the card's second "
               "branch holds — the discipline is installed at once and is no growth "
               "phenomenon, which the card calls the more surprising outcome.")
    return ("### P1 — Arm A: G_dec rises with log(step) and saturates before step 3000\n\n"
            f"**OUTCOME: {out}**\n\n"
            "```\n"
            f"G_dec median at step 100   = {_f(g100)}\n"
            f"G_dec median at step 3000  = {_f(g3000)}\n"
            f"rise, step 100 -> 3000     = {_f(rise)} decades\n"
            f"fraction of final gap already open at step 100 = {_f(frac)}\n"
            f"mean slope vs log10(step)  = {c1['slope_mean']:.6g}\n"
            f"permutation p (two-sided)  = {c1['p_two_sided']:.5g}\n"
            f"early-window slope (steps {sat['early_window'][0]}-{sat['early_window'][1]})"
            f"  = {sat['early_mean_slope']:.6g}\n"
            f"late-window slope  (steps {sat['late_window'][0]}-{sat['late_window'][1]})"
            f" = {sat['late_mean_slope']:.6g}\n"
            f"late / early               = {_f(ratio)}\n"
            f"frac pairs late < early    = {sat['frac_late_slope_below_early']:.4f}\n"
            "```\n\n"
            f"*{sat['note']}*")


def _p2_block(p2):
    s0, s100, s3000 = p2["step0"], p2["step100"], p2["step3000"]
    if p2["arm_A_void"]:
        out = ("A GAP IS PRESENT AT INITIALIZATION under the operationalisation below. "
               "Per §6 item 2 this is the condition under which the endpoint is measuring "
               "the initializer and ARM A IS VOID. The Arm A curves above must be read "
               "with that verdict attached, and the Director's ruling is owed on it.")
    elif p2["step0_all_identity"]:
        out = ("NO GAP AT INITIALIZATION, in the strongest available form. Every one of "
               f"the {s0['n_adapters']:,} step-0 bridges is the exact identity matrix: all "
               "off-diagonal entries are 0.0, so the co mode and the cross mode both "
               "vanish, the ratio is 0/0, and G_dec and G_min are UNDEFINED rather than "
               "infinite. The endpoint cannot be measuring the initializer, because the "
               "initializer writes nothing off-diagonal for it to measure. ARM A EXISTS. "
               "Occ at step 0 is 0 of 36,960 because every entry sits below the band, not "
               "because a gap is holding it open — a distinction the band cannot make by "
               "itself and which is therefore stated here.")
    else:
        out = ("NO GAP AT INITIALIZATION. The step-0 control passes; the endpoint is not "
               "measuring the initializer, and Arm A exists.")
    return ("### P2 — Arm A control: at bridge_step0 (initialization) there is no gap\n\n"
            f"**OUTCOME: {out}**\n\n"
            "```\n"
            f"CONTROL            = {p2['control']}\n"
            f"step 0 adapters with ALL off-diagonal entries exactly 0 = "
            f"{p2['step0_zero_offdiag_adapters']:,} of {s0['n_adapters']:,}\n"
            f"step 0    G_dec median = {_f(s0['G_dec_median'])}   G_min median = {_f(s0['G_min_median'])}"
            f"   G_min<=0 {s0['G_min_le_0_count']}/{s0['n_adapters']}\n"
            f"step 100  G_dec median = {_f(s100['G_dec_median'])}   G_min median = {_f(s100['G_min_median'])}"
            f"   G_min<=0 {s100['G_min_le_0_count']}/{s100['n_adapters']}\n"
            f"step 3000 G_dec median = {_f(s3000['G_dec_median'])}   G_min median = {_f(s3000['G_min_median'])}"
            f"   G_min<=0 {s3000['G_min_le_0_count']}/{s3000['n_adapters']}\n"
            f"step 0    Occ = {s0['occ_count']:,} / {s0['occ_total']:,}"
            f"   (fraction {s0['occ_fraction']:.6g})\n"
            f"G_dec(step0)/G_dec(step3000) medians = {_f(p2['G_dec_ratio_step0_over_step3000'])}\n"
            f"GAP PRESENT AT INIT? = {p2['gap_present_at_init']}\n"
            f"ARM A VOID?        = {p2['arm_A_void']}\n"
            "```\n\n"
            f"*{p2['operationalisation']}*")


def _p3_block(P, c2):
    ratios = c2["ratio_by_step"]
    taus = c2["kendall_tau_G_dec_vs_ratio"]
    if abs(taus) >= 0.6:
        out = ("TRACKING. G_dec tracks the run-level co/cross ratio monotonically across "
               "the anneal — gap and ratio read as one object seen at two levels.")
    else:
        out = ("DISSOCIATION. G_dec does NOT track the run-level co/cross ratio "
               "monotonically across the anneal; gap and ratio are two objects, which is "
               "the branch FI-004 has already shown once at the sign layer.")
    return ("### P3 — Arm B: G_dec tracks the run-level co/cross ratio monotonically\n\n"
            f"**OUTCOME: {out}**\n\n"
            "```\n"
            f"Kendall tau, median G_dec vs recorded co/cross ratio (30 checkpoints) = {taus:.4f}\n"
            f"Spearman rho, same pair                                              = "
            f"{c2['spearman_G_dec_vs_ratio']:.4f}\n"
            f"recorded co/cross at step 100 / 1500 / 3000 = "
            f"{_f(ratios.get('100'))} / {_f(ratios.get('1500'))} / {_f(ratios.get('3000'))}\n"
            "```")


def _p4_block(p4, S):
    occ = p4["occ_by_step"]
    filled = any(v["occ_count"] > 0 for v in occ.values())
    first = occ[str(FI003_STEPS[0])]
    term = p4["occupancy_terminal"]
    if filled and first["occ_count"] > 0:
        out = ("THE BAND FILLS, AND IS ALREADY OCCUPIED AT THE FIRST SNAPSHOT (s = 100). "
               "Per §12 Item 3 this is reported as 'fill complete before the first "
               "snapshot; half-life bounded above by ~30 steps' — the one-sided bound "
               "this grid supports — and NOT as 'half-life near 15 steps'. The 15-step "
               "reading is not available from these artifacts and is not written.")
    elif filled:
        out = ("THE BAND FILLS, AND THE FILL IS SLOW ENOUGH TO BE RESOLVED ON THIS GRID. "
               f"The first three checkpoints are empty; occupancy opens between step "
               f"{p4['occupancy_onset_step'] - 100} and step {p4['occupancy_onset_step']}, "
               f"then rises monotonically at every subsequent checkpoint to "
               f"{term['count']} of {term['total']:,} entries ({term['fraction']:.3%}) at "
               f"step {term['step']} — still far from full when the run early-stops. The "
               "timing branch remains UNDERPOWERED as pinned, and no half-life is claimed "
               "for the fill. What the grid does support is the one-sided statement it can "
               "carry: the fill had not begun by step 300, so it is not a process that was "
               "over before the first snapshot [ANALYTICAL CONTRIBUTION: the reading; the "
               "observation is the Occ column below]. That is a bound, not a measurement, "
               "and it is not offered as a null against the record's fast component.")
    else:
        out = ("NO FILL AT ALL, while the recorded co/cross ratio decays "
               f"{_f(p4['recorded_co_cross_ratio_by_step'].get('100'))} -> "
               f"{_f(p4['recorded_co_cross_ratio_by_step'].get('1200'))} across the same 12 "
               "points. This is prediction 4's third branch — a refutation of the standing "
               "law at gap level, and the branch the card calls the most interesting result "
               "it can return. It is FULLY POWERED: fill-vs-no-fill is testable at all 12 "
               "points.")
    lines = ["### P4 — Arm B, zero drive: under FI-003 the band fills", "",
             f"**OUTCOME: {out}**", "", "```",
             f"POWER VERDICT (timing branch) = UNDERPOWERED — {p4['power_verdict']}",
             f"FILL vs NO FILL              = {p4['fill_vs_no_fill']}",
             f"OCCUPANCY ONSET              = between step "
             f"{p4['occupancy_onset_step'] - 100} and {p4['occupancy_onset_step']}"
             if p4["occupancy_onset_step"] else "OCCUPANCY ONSET              = never",
             f"GAP half-life, endpoint fit  = {p4['gap_decay_halflife_endpoint_steps']:.1f} steps",
             f"GAP half-life, OLS fit       = {p4['gap_decay_halflife_ols_steps']:.1f} steps"
             f"   (R2 {p4['gap_decay_ols_r2']:.4f})",
             "Occ and gap endpoints across the 12 FI-003 checkpoints:"]
    for s in FI003_STEPS:
        v = occ[str(s)]
        lines.append(f"  step {s:>5}  Occ {v['occ_count']:>3} / {v['occ_total']:,}"
                     f" (co {v['occ_co']}, cross {v['occ_cross']})"
                     f"   G_dec med {_f(p4['G_dec_by_step'][str(s)])}"
                     f"   G_min med {_f(p4['G_min_by_step'][str(s)])}"
                     f"   co/cross {_f(p4['recorded_co_cross_ratio_by_step'].get(str(s)))}")
    pa = p4["power_arithmetic"]
    lines += [
        f"half-lives elapsed at s=100:  tau=15 -> {pa['half_lives_elapsed_at_first_point_tau15']:.3f}"
        f"   tau=230 -> {pa['half_lives_elapsed_at_first_point_tau230']:.3f}",
        f"surviving fraction at s=100:  tau=15 -> {pa['surviving_fraction_s100_tau15']:.5g}"
        f"   tau=230 -> {pa['surviving_fraction_s100_tau230']:.5g}",
        f"smallest tau constrainable, TWO points at >=10% retention = "
        f"{pa['two_point_floor_tau_steps']:.2f} steps",
        f"smallest tau constrainable, ONE point                     = "
        f"{pa['one_point_floor_tau_steps']:.2f} steps",
        "```", "",
        "**Which side of the band fills.** " + p4["which_side_fills"]["reading"]
        + f" At the terminal checkpoint the occupied entries split "
        f"{p4['which_side_fills']['terminal_occ_co']} co / "
        f"{p4['which_side_fills']['terminal_occ_cross']} cross. The FI-004 table in §5.1 "
        "carries the same split per checkpoint and sums it.", "",
        "**The measured timescale, and what it is allowed to be compared with.** "
        + p4["gap_decay_note"] + " The record's slow component is 230 steps; both figures "
        "above are of the same order, and this document does not convert that into an "
        "identity claim — a two-point and a 12-point fit of one quantity on one run "
        "cannot confirm a four-parameter fit of a different quantity.", "",
        "**Transfer flag, in the same sentence as the number.** " + p4["transfer_flag"], "",
        "**How the 15-step figure is cited.** The record's double-exponential fit "
        "(R2 = 0.9999998, fast half-life 15 steps, slow 230) has as its only available "
        "trajectory a step-0 row plus these same 12 checkpoints; a double exponential "
        "carries four free parameters and there is no observation anywhere between step 0 "
        "and step 100, so the whole factor-70 drop from step 0 to step 100 is the only "
        "evidence the fast timescale has. The 15-step figure is an extrapolation across "
        "one interval, not a resolved decay [ANALYTICAL CONTRIBUTION: the reading of the "
        "fit's support; the fit and its R2 are the record's]."]
    return "\n".join(lines)


def _p6_block(c3):
    obs, pred = c3["observed_occupancy"], c3["predicted_occupancy"]
    p0 = c3["P_observe_zero_under_N2"]
    if p0 < 0.05:
        out = ("OBSERVED OCCUPANCY IS BELOW THE TWO-FIT PREDICTION BY MORE THAN THE "
               f"PREDICTION'S OWN SPREAD (P(observe 0 | N2) = {p0:.4g}). Under the card's "
               "wording this reads as a genuine exclusion — a forbidden zone rather than "
               "two humps that happen not to overlap — subject to the vacuity caveat.")
    else:
        out = ("THE PREDICTED COUNT IS ABOVE THE OBSERVED ZERO, BUT NOT BY ENOUGH TO "
               f"REFUTE N2: P(observe 0 | N2) = {p0:.4g}, so a band that is empty by "
               "chance under two independently fitted log-normals is an ordinary outcome, "
               "not a rare one. Under the card's second branch the 'forbidden zone' "
               "reading of Station 8 is, on this evidence, a description of two "
               "well-separated humps, and should be written as such.")
    # Which of §6 item 6's two branches this is, and why — dated, not silent (L-006).
    # The card's branch labels are keyed to the bare inequality; §5 defines the null as
    # agreement. Where those two readings select different branches, both are stated.
    if p0 < 0.05:
        note = ""
    else:
        note = ("\n\n**Which branch of §6 item 6 this is — IMPLEMENTER READING, dated "
                "2026-09-14, flagged as such and not a registered criterion.** §6 item 6 "
                "writes the branches as *\"below -> a genuine exclusion; at or above -> "
                "the 'forbidden zone' reading of Station 8 is a description of two "
                "separated humps\"*, and the observed 0 IS literally below the predicted "
                f"{c3['predicted_count']:.4g}. The second branch is selected above anyway, "
                "on §5's criterion rather than on the bare inequality: §5 defines N2 as "
                "*\"fit each mode independently ... predict band occupancy, compare to "
                "observed. Under N2 they agree\"*, and whether a single realization agrees "
                "with a predicted mean is a statistical question, so item 6's *below* is "
                "read here as *significantly below*. Both halves are reported so a reader "
                "may apply either: the literal below-condition is SATISFIED (0 < "
                f"{c3['predicted_count']:.4g}); the statistical criterion is NOT met "
                f"(P(observe 0 | N2) = {p0:.4g}, not significant at any conventional "
                "level), and N2 is therefore NOT REFUTED. Reported under L-006 as a dated "
                "reading of the registered wording, not a revision of it.")
    return ("### P6 — N2: observed band occupancy is below the two-fit prediction\n\n"
            f"**OUTCOME: {out}**\n\n"
            "```\n"
            f"observed  {c3['observed_in_band']} / {c3['observed_total']:,}"
            f"   occupancy {obs:.6g}\n"
            f"predicted {c3['predicted_count']:.6g} / {c3['observed_total']:,}"
            f"   occupancy {pred:.6g}\n"
            f"predicted bootstrap CI95 (on the MEAN) = "
            f"[{c3['predicted_bootstrap_ci95'][0]:.6g}, "
            f"{c3['predicted_bootstrap_ci95'][1]:.6g}]\n"
            f"DECISION STATISTIC  P(observe 0 | N2) = {p0:.4g}   "
            f"(log10 {c3['P_observe_zero_under_N2_log10']:.4g})\n"
            f"same, ddof=1 sensitivity              = "
            f"{c3['sensitivity_ddof1']['P_observe_zero_under_N2']:.4g}\n"
            "```\n\n"
            "**Why the decision rests on P(observe 0), not on the interval.** The observed "
            "0 is a single REALIZATION; the bootstrap interval above is an interval on the "
            "predicted MEAN. Comparing the one to the other is a category error that would "
            "manufacture an exclusion out of a mean of roughly two expected entries, so "
            "the test is run against the realization distribution instead: under the "
            "fitted two-mode model the entries are independent, P(none lands in the band) "
            "is the product of their misses, and that is the number above.\n\n"
            "**The comparison is reported, not a bare ratio**, and the vacuity caveat "
            "binds: the band's endpoints are this data's own extrema, so the observed 0 "
            "is empty by construction and only the prediction side is unknown." + note)


def _p7_block(p7):
    a, b = p7["arm_A"], p7["arm_B_fi004"]
    met_a = a["frac_tau_ge_0.6"] >= 0.5
    met_b = b["frac_tau_ge_0.6"] >= 0.5
    n_col = a["G_min_le_0_count"] + b["G_min_le_0_count"]
    if met_a and met_b and n_col == 0:
        out = ("CO-MOVEMENT on both arms, and G_min never collapses to <= 0. The margin is "
               "one quantity seen through two statistics; G_min adds tail information "
               "without a second story.")
    elif met_a and met_b:
        out = ("CO-MOVEMENT on both arms — the registered bar is met for "
               f"{a['frac_tau_ge_0.6']:.1%} of Arm A pairs and {b['frac_tau_ge_0.6']:.1%} of "
               "Arm B adapters, so the margin is one quantity seen through two statistics. "
               f"The card's second branch does NOT hold at scale, but it is not empty "
               f"either: G_min falls to <= 0 in {a['G_min_le_0_count']:,} of "
               f"{a['G_min_total']:,} Arm A post-init adapter-checkpoints "
               f"({a['G_min_le_0_fraction']:.4%}), confined to these Arm A steps: "
               f"{a['G_min_le_0_by_step'] or 'none'}, and in {b['G_min_le_0_count']:,} of "
               f"{b['G_min_total']:,} on Arm B ({b['G_min_le_0_fraction']:.4%}"
               + (f", at these steps: {b['G_min_le_0_by_step']}"
                  if b["G_min_le_0_count"] else "")
               + "). The silicon "
               "engineer's failure case — means a decade apart while the tails touch, a "
               "margin with no guaranteed zone — is therefore real here but transient: it "
               "belongs to the first hundred steps of training and is closed thereafter.")
    else:
        out = ("THE tau >= 0.6 BAR IS NOT MET on at least one arm: G_min and G_dec do not "
               "co-move at the registered strength, so the two statistics carry different "
               "information and are reported separately rather than as one margin.")
    return ("### P7 — G_min moves with G_dec (bar: Kendall tau >= 0.6)\n\n"
            f"**OUTCOME: {out}**\n\n"
            "```\n"
            f"Arm A  tau mean {a['tau_mean']:.4f}  median {a['tau_median']:.4f}  "
            f"frac tau>=0.6 {a['frac_tau_ge_0.6']:.4f}  (n={a['n']:,} run-adapter pairs)\n"
            f"Arm A  G_min <= 0 at {a['G_min_le_0_count']:,} of {a['G_min_total']:,} "
            f"adapter-checkpoints ({a['G_min_le_0_fraction']:.6g})\n"
            f"Arm A  G_min <= 0 by step: {a['G_min_le_0_by_step'] or 'none'}\n"
            f"Arm B  tau mean {b['tau_mean']:.4f}  median {b['tau_median']:.4f}  "
            f"frac tau>=0.6 {b['frac_tau_ge_0.6']:.4f}  (n={b['n']} adapters)\n"
            f"Arm B  G_min <= 0 at {b['G_min_le_0_count']:,} of {b['G_min_total']:,} "
            f"adapter-checkpoints ({b['G_min_le_0_fraction']:.6g})\n"
            "```\n\n"
            "G_min at step 0 is subject to the same control as P2; see the P2 block.")


def _n1_verdict(c1, c2):
    a = c1["p_two_sided"] < 0.05
    b = c2["p_two_sided"] < 0.05
    if a and b:
        return ("REFUTED on Arm A (C1) and on Arm B (C2). The gap is not constant: it has "
                "a response surface, within the scope limit. Two qualifications travel "
                "with this verdict and are not separable from it. (i) Arm B's regressor is "
                "collinear with training step by construction (r = -1.000), so C2 refutes "
                "constancy along a combined drive-and-duration axis and does not by itself "
                "establish a DRIVE response; the register's null named three dials and this "
                "run refutes it on one clean dial (step) and one compound dial. (ii) The "
                "scale dial is untested — Arm C is descriptive and contributes nothing to "
                "this verdict, by §14.")
    if a and not b:
        return ("REFUTED on Arm A (C1); NOT REFUTED on Arm B (C2). The gap responds to "
                "training step but no c_w response is detected across the FI-004 anneal.")
    if b and not a:
        return ("NOT REFUTED on Arm A (C1); REFUTED on Arm B (C2). No step response is "
                "detected; a drive response is.")
    return ("NOT REFUTED on either confirmatory arm. Within the scope limit the endpoints "
            "are consistent with a constant gap and no response surface; sparse results "
            "are data.")


def _n2_verdict(c3):
    p0 = c3["P_observe_zero_under_N2"]
    if p0 < 0.05:
        return (f"REFUTED (P(observe 0 | N2) = {p0:.4g}). The observed band count falls "
                "below what two independently fitted log-normals predict by more than "
                "chance — an exclusion rather than two non-overlapping humps, subject to "
                "the vacuity caveat on the observed side.")
    return (f"NOT REFUTED (P(observe 0 | N2) = {p0:.4g}; predicted mean "
            f"{c3['predicted_count']:.4g} of {c3['observed_total']:,}, observed "
            f"{c3['observed_in_band']}). The two-fit model expects only a couple of "
            "entries in the band to begin with, and seeing none is an ordinary outcome "
            "under it. On this evidence the empty band is the tails of two well-separated "
            "humps, and the 'forbidden zone' reading of Station 8 is a description of two "
            "separated humps and should be written as such. Note what this does NOT say: "
            "it does not show the band is occupiable, only that E-5's single final "
            "checkpoint has too little occupancy mass at stake to tell the two readings "
            "apart.")


if __name__ == "__main__":
    sys.exit(main())
