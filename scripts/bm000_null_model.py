"""BM-000: Null-model calibration for the program's four bridge metrics.

Pre-registered protocol: results/BM-000/PROTOCOL.md (read it first).
Plan: docs/BM_BATTERY_PLAN.md section "BM-000".

Builds empirical null distributions for:
  M1  co/cross ratio        (canonical: train_exp2_scale.coplanar_crossplanar_ratio,
                             RD face-pair definition, upper triangle)
  M1b co/cross full-matrix  (analyze_bridge_structure.py:155-187 convention,
                             both triangles, computed per-bridge)
  M2  bridge Fiedler        (canonical telemetry definition:
                             rhombic.spectral.fiedler_value on upper-tri |B|,
                             as in rhombi_lora.bridge_fiedler / train_cybernetic)
  M2b Fiedler variant       (analyze_task_bridges.py:397-405, verbatim incl.
                             the eigvalsh lower-triangle quirk)
  M3a block-diagonal score  (detect_blocks.detect_blocks, gap+union-find)
  M3b block-diagonal top-3  (analyze_bridge_structure.py:168-172 criterion)
  M4  asymmetry ratio       (ar001_asymmetry_analysis_v2.py:133-141:
                             ||(B-B^T)/2||_F / ||B||_F)

CPU-ONLY by construction: no model is instantiated; torch is imported only as
a side effect of reusing train_exp2_scale definitions. CUDA is masked before
any import that could touch it.

Usage:
  python scripts/bm000_null_model.py            # full run, writes results/BM-000/
  python scripts/bm000_null_model.py --n 500    # quick pass (files still written)

Seed: 20260704 (protocol-fixed).
"""

from __future__ import annotations

import os

# GPU is running a training campaign — hard-mask CUDA before torch can load it.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ── REUSED metric implementations (imported, not reimplemented) ──────
from rhombic.spectral import fiedler_value                      # noqa: E402
from rhombic.nn.topology import rd_adjacency_mask               # noqa: E402
from detect_blocks import detect_blocks                         # noqa: E402
from analyze_bridge_structure import CO_PLANAR_SET              # noqa: E402
from train_exp2_scale import (                                  # noqa: E402
    coplanar_crossplanar_ratio,
    _coplanar_crossplanar_indices,
)
from train_contrastive_bridge import _compute_pair_indices      # noqa: E402

SEED = 20260704
DEFAULT_N = 10_000
OUT_DIR = REPO / "results" / "BM-000"

PCT_GRID = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
REPORT_PCTS = [50, 90, 99, 99.9]


def _verify_pair_definitions() -> None:
    """Cross-check pair definitions across the two modules that define them.

    Protocol section 1 (M1). Abort if the codebase definitions ever diverge.
    """
    for n in (4, 6, 8):
        a_co, a_cross = _coplanar_crossplanar_indices(n)
        b_co, b_cross = _compute_pair_indices(n)
        assert list(a_co) == list(b_co), f"co-pair mismatch at n={n}"
        assert list(a_cross) == list(b_cross), f"cross-pair mismatch at n={n}"
    co6, _ = _coplanar_crossplanar_indices(6)
    assert co6 == [(0, 1), (2, 3), (4, 5)], "RD face-pair set changed"


# ── Metric wrappers (verbatim variants carry file:line citations) ────


def metric_co_cross(B: np.ndarray) -> float:
    """M1 canonical — train_exp2_scale.py:155-181 (imported)."""
    r = coplanar_crossplanar_ratio(B)
    if r is None:
        return float("nan")
    return float(r["ratio"])


def metric_co_cross_fullmatrix(B: np.ndarray) -> float:
    """M1b — analyze_bridge_structure.py:155-187 convention, per-bridge.

    Both triangles (i != j); co iff (i, j) in CO_PLANAR_SET (n=6 only).
    """
    n = B.shape[0]
    if n != 6:
        return float("nan")
    co_vals, cross_vals = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = abs(B[i, j])
            if (i, j) in CO_PLANAR_SET:
                co_vals.append(v)
            else:
                cross_vals.append(v)
    co_mean = float(np.mean(co_vals))
    cross_mean = float(np.mean(cross_vals))
    return co_mean / cross_mean if cross_mean > 1e-10 else float("inf")


def metric_fiedler(B: np.ndarray) -> float:
    """M2 canonical — rhombi_lora.py:284-295 / train_cybernetic.py:286-290.

    Upper-triangle |B[i,j]| as complete-graph edge weights; lambda_2 of the
    weighted Laplacian via rhombic.spectral.fiedler_value (imported).
    """
    n = B.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = [float(abs(B[i, j])) for i, j in edges]
    return float(fiedler_value(n, edges, weights))


def metric_fiedler_atb(B: np.ndarray) -> float:
    """M2b — analyze_task_bridges.py:397-405, copied VERBATIM.

    W = |B| zero-diag; L = diag(rowsum(W)) - W; np.linalg.eigvalsh(L)[1].
    For asymmetric B, L is not symmetric and eigvalsh reads the lower
    triangle (UPLO='L'). The quirk is intentional: this null calibrates the
    code as it ran, not a corrected version.
    """
    n = B.shape[0]
    W = np.abs(B)
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1]) if len(eigs) > 1 else 0.0


def metric_bd_top3(B: np.ndarray) -> float:
    """M3b — analyze_bridge_structure.py:168-172, copied VERBATIM (n=6 only).

    1.0 iff the top-3 off-diagonal magnitudes (30 directed entries) are all
    co-planar. Analytic chance rate C(6,3)/C(30,3) = 20/4060 ~ 0.4926%.
    """
    n = B.shape[0]
    if n != 6:
        return float("nan")
    off_diag = {}
    for i in range(6):
        for j in range(6):
            if i != j:
                off_diag[(i, j)] = abs(B[i, j])
    sorted_od = sorted(off_diag.items(), key=lambda x: x[1], reverse=True)
    top3 = set(sorted_od[i][0] for i in range(min(3, len(sorted_od))))
    return 1.0 if top3.issubset(CO_PLANAR_SET) else 0.0


def metric_asymmetry(B: np.ndarray) -> float:
    """M4 — ar001_asymmetry_analysis_v2.py:133-141, copied VERBATIM.

    (That file is a top-level script that executes its analysis on import,
    so the formula is copied with citation rather than imported.)
    """
    A = (B - B.T) / 2
    norm_B = np.linalg.norm(B, "fro")
    norm_A = np.linalg.norm(A, "fro")
    return float(norm_A / max(norm_B, 1e-10))


def compute_metrics(bridges: list[np.ndarray]) -> dict[str, np.ndarray]:
    """All BM-000 metrics for a list of bridge matrices."""
    out: dict[str, list[float]] = {
        "co_cross": [], "co_cross_fullmatrix": [],
        "fiedler": [], "fiedler_atb": [],
        "bd_flag": [], "bd_ratio": [], "bd_max_gap_ratio": [], "bd_n_blocks": [],
        "bd_top3": [], "asymmetry": [],
    }
    for B in bridges:
        out["co_cross"].append(metric_co_cross(B))
        out["co_cross_fullmatrix"].append(metric_co_cross_fullmatrix(B))
        out["fiedler"].append(metric_fiedler(B))
        out["fiedler_atb"].append(metric_fiedler_atb(B))
        db = detect_blocks(B)  # M3a — detect_blocks.py:16-132 (imported)
        out["bd_flag"].append(1.0 if db["is_block_diagonal"] else 0.0)
        ratio = db["ratio"]
        out["bd_ratio"].append(float(ratio) if np.isfinite(ratio) else np.nan)
        out["bd_max_gap_ratio"].append(float(db.get("max_gap_ratio", 0.0)))
        out["bd_n_blocks"].append(float(db["n_blocks"]))
        out["bd_top3"].append(metric_bd_top3(B))
        out["asymmetry"].append(metric_asymmetry(B))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ── Empirical bridge loading ─────────────────────────────────────────


def load_bridge_dir(directory: Path) -> list[np.ndarray]:
    """Load all bridge_final_*.npy in a directory (sorted, deterministic)."""
    return [np.load(f) for f in sorted(directory.glob("bridge_final_*.npy"))]


def load_smoke_bank() -> dict[str, list[np.ndarray]]:
    """Load smoke bridges per family x task run (protocol section 2/3)."""
    root = REPO / "results" / "asset1-smoke"
    sets: dict[str, list[np.ndarray]] = {}
    for fam in sorted(p for p in root.iterdir() if p.is_dir()):
        for task in sorted(p for p in fam.iterdir() if p.is_dir()):
            for run in sorted(p for p in task.iterdir()
                              if p.is_dir() and p.name.startswith("run_")):
                bridges = load_bridge_dir(run)
                if bridges:
                    sets[f"smoke-{fam.name}-{task.name}-{run.name}"] = bridges
    return sets


def match_moments(bridges: list[np.ndarray]) -> dict[str, float]:
    """Pooled diagonal / off-diagonal moments of a bridge set (protocol s.2)."""
    diag_vals, off_vals = [], []
    for B in bridges:
        n = B.shape[0]
        mask = np.eye(n, dtype=bool)
        diag_vals.append(B[mask])
        off_vals.append(B[~mask])
    d = np.concatenate(diag_vals)
    o = np.concatenate(off_vals)
    return {
        "diag_mean": float(d.mean()), "diag_std": float(d.std()),
        "offdiag_mean": float(o.mean()), "offdiag_std": float(o.std()),
        "n_matrices": len(bridges),
    }


# ── Null ensemble generators (single RNG stream, protocol order) ─────


def gen_gauss(rng: np.random.Generator, n: int, count: int,
              m: dict[str, float]) -> list[np.ndarray]:
    """N-A / N-D: moment-matched Gaussian bridges."""
    out = []
    for _ in range(count):
        B = rng.normal(m["offdiag_mean"], m["offdiag_std"], size=(n, n))
        di = rng.normal(m["diag_mean"], m["diag_std"], size=n)
        B[np.arange(n), np.arange(n)] = di
        out.append(B)
    return out


def gen_rdmask(rng: np.random.Generator, count: int,
               m: dict[str, float]) -> list[np.ndarray]:
    """N-B: random {0.5,1.0} mask matched to rd_adjacency_mask(6) multiset.

    Symmetric mask, diagonal 1.0; a uniformly random 3 of the 15 unordered
    off-diagonal pairs get 1.0 (both directions), the other 12 get 0.5 —
    the bridge-scale analogue of a degree-preserving rewire. Weights are the
    same moment-matched Gaussian entries as N-A, multiplied elementwise.
    """
    ref = rd_adjacency_mask(6)
    ref_off = ref[~np.eye(6, dtype=bool)]
    n_strong_pairs = int(round(float((ref_off == 1.0).sum())) // 2)  # = 3
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    out = []
    for _ in range(count):
        mask = np.full((6, 6), 0.5)
        np.fill_diagonal(mask, 1.0)
        idx = rng.choice(len(pairs), size=n_strong_pairs, replace=False)
        for k in idx:
            i, j = pairs[k]
            mask[i, j] = 1.0
            mask[j, i] = 1.0
        G = rng.normal(m["offdiag_mean"], m["offdiag_std"], size=(6, 6))
        di = rng.normal(m["diag_mean"], m["diag_std"], size=6)
        G[np.arange(6), np.arange(6)] = di
        out.append(mask * G)
    return out


def gen_identity_noise(rng: np.random.Generator, count: int,
                       eps: float) -> list[np.ndarray]:
    """N-C: I_6 + N(0, eps^2) iid on all entries (near-init family)."""
    return [np.eye(6) + rng.normal(0.0, eps, size=(6, 6)) for _ in range(count)]


# ── Statistics ───────────────────────────────────────────────────────


def summarize(arr: np.ndarray) -> dict:
    """Distribution summary over finite values (protocol section 5)."""
    finite = arr[np.isfinite(arr)]
    n_nonfinite = int(arr.size - finite.size)
    if finite.size == 0:
        return {"n": 0, "n_nonfinite": n_nonfinite}
    s = {
        "n": int(finite.size),
        "n_nonfinite": n_nonfinite,
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "percentiles": {str(p): float(np.percentile(finite, p)) for p in PCT_GRID},
    }
    return s


def percentile_of(value: float, arr: np.ndarray) -> float:
    """pct(v) = 100 * #{null < v} / N over finite null values."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(100.0 * np.sum(finite < value) / finite.size)


def build_ensembles(seed: int, n_samples: int,
                    moments: dict[str, float]) -> dict[str, list[np.ndarray]]:
    """All null ensembles from ONE rng stream in protocol order."""
    rng = np.random.default_rng(seed)
    ensembles = {
        "gauss6": gen_gauss(rng, 6, n_samples, moments),
        "rdmask6": gen_rdmask(rng, n_samples, moments),
        "identity+eps0.01": gen_identity_noise(rng, n_samples, 0.01),
        "identity+eps0.05": gen_identity_noise(rng, n_samples, 0.05),
        "identity+eps0.1": gen_identity_noise(rng, n_samples, 0.1),
        "gauss4": gen_gauss(rng, 4, n_samples, moments),
        "gauss8": gen_gauss(rng, 8, n_samples, moments),
    }
    return ensembles


def run_nulls(seed: int, n_samples: int,
              moments: dict[str, float]) -> dict[str, dict[str, dict]]:
    """Generate ensembles, compute metrics, summarize. Returns nested dict."""
    ensembles = build_ensembles(seed, n_samples, moments)
    tables: dict[str, dict[str, dict]] = {}
    raw: dict[str, dict[str, np.ndarray]] = {}
    for name, bridges in ensembles.items():
        metrics = compute_metrics(bridges)
        raw[name] = metrics
        tables[name] = {m: summarize(v) for m, v in metrics.items()}
        # BD rates as explicit fields
        tables[name]["bd_flag"]["rate"] = float(np.nanmean(metrics["bd_flag"]))
        # Derived from reused outputs: detect_blocks also flags single-block
        # unions as BD; the program's "100% BD" claims are multi-block, so the
        # multi-block chance rate is the comparable null figure.
        tables[name]["bd_flag"]["rate_multiblock"] = float(np.mean(
            (metrics["bd_flag"] == 1.0) & (metrics["bd_n_blocks"] >= 2)))
        top3 = metrics["bd_top3"]
        top3f = top3[np.isfinite(top3)]
        tables[name]["bd_top3"]["rate"] = (
            float(top3f.mean()) if top3f.size else float("nan"))
        # bd_ratio conditional on detection
        flagged = metrics["bd_ratio"][(metrics["bd_flag"] == 1.0)
                                      & np.isfinite(metrics["bd_ratio"])]
        tables[name]["bd_ratio_given_bd"] = summarize(flagged)
    run_nulls.raw = raw  # stashed for report generation in main()
    return tables


# ── Headline values (protocol section 4) ─────────────────────────────


def read_headlines() -> dict[str, dict]:
    """Read the program's headline values from their source artifacts."""
    heads: dict[str, dict] = {}

    csv_path = REPO / "results" / "channel-ablation" / "H-ch6" / "metrics_hermes.csv"
    if csv_path.exists():
        last = csv_path.read_text().strip().splitlines()[-1].split(",")
        # columns: step, val_loss, fiedler, co_cross, ..., deviation, ...
        heads["H-ch6 co/cross (final, step %s)" % last[0]] = {
            "metric": "co_cross", "value": float(last[3]),
            "source": "results/channel-ablation/H-ch6/metrics_hermes.csv last row",
        }
        heads["H-ch6 bridge Fiedler (final)"] = {
            "metric": "fiedler", "value": float(last[2]),
            "source": "results/channel-ablation/H-ch6/metrics_hermes.csv last row",
        }

    heads["Spectral-attractor Fiedler (representative)"] = {
        "metric": "fiedler", "value": 0.09,
        "source": "docs/EXPERIMENT_TRACKER.md (band 0.084-0.102: E-001, H-ch12)",
    }

    ar_path = REPO / "results" / "AR-001-raw-results.json"
    if ar_path.exists():
        ar = json.loads(ar_path.read_text())
        for exp_name, r in ar.items():
            heads[f"AR-001 {exp_name} asymmetry (n={r['n']})"] = {
                "metric": "asymmetry", "value": float(r["asym_ratio_mean"]),
                "n": int(r["n"]),
                "source": "results/AR-001-raw-results.json",
            }
            if r.get("cc_B_mean") is not None:
                heads[f"AR-001 {exp_name} co/cross (n={r['n']})"] = {
                    "metric": "co_cross", "value": float(r["cc_B_mean"]),
                    "n": int(r["n"]),
                    "source": "results/AR-001-raw-results.json",
                }
    return heads


HEADLINE_NULL_FOR_N = {4: "gauss4", 6: "gauss6", 8: "gauss8"}
N6_ENSEMBLES = ["gauss6", "rdmask6", "identity+eps0.01",
                "identity+eps0.05", "identity+eps0.1"]


def headline_ensembles(h: dict) -> list[str]:
    """Which null ensembles a headline is placed against.

    Default: the moment-matched Gaussian at the headline's n (n=3 has no
    ensemble; the nearest, gauss6, is used and labeled as such). Fiedler
    headlines at n=6 are additionally placed against every n=6 ensemble —
    the protocol pre-registered that the attractor's placement must include
    the identity+noise (near-init) family.
    """
    n = h.get("n", 6)
    base = HEADLINE_NULL_FOR_N.get(n, "gauss6")
    if h["metric"] == "fiedler" and n == 6:
        return N6_ENSEMBLES
    return [base]


# ── Report generation ────────────────────────────────────────────────

METRIC_LABELS = {
    "co_cross": "M1 co/cross (canonical, upper-tri, RD/octa/tess pairs)",
    "co_cross_fullmatrix": "M1b co/cross (full-matrix, per-bridge)",
    "fiedler": "M2 bridge Fiedler (canonical telemetry def.)",
    "fiedler_atb": "M2b Fiedler (analyze_task_bridges variant)",
    "bd_flag": "M3a BD detected (detect_blocks, rate)",
    "bd_ratio_given_bd": "M3a within/between ratio | BD detected",
    "bd_max_gap_ratio": "M3a max gap ratio",
    "bd_top3": "M3b BD top-3 criterion (rate)",
    "asymmetry": "M4 asymmetry ratio ||A||F/||B||F",
}


def fmt(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    if abs(x) >= 10000 or (abs(x) < 0.001):
        return f"{x:.3e}"
    return f"{x:.4f}"


def write_results_md(path: Path, moments: dict, tables: dict, raw: dict,
                     empirical: dict, heads: dict, n_samples: int) -> None:
    L: list[str] = []
    L.append("# BM-000 — Null-Model Calibration: RESULTS")
    L.append("")
    L.append(f"> Generated by `scripts/bm000_null_model.py` (seed {SEED}, "
             f"N = {n_samples:,} per ensemble). Protocol: `PROTOCOL.md` "
             "(pre-registered before this run). Every number below is "
             "computed output; nothing is hand-edited.")
    L.append("")
    L.append("## Matched moments (from the trained smoke bank)")
    L.append("")
    L.append(f"Measured over {moments['n_matrices']} bridge matrices "
             "(`results/asset1-smoke/*/*/run_*/bridge_final_*.npy`):")
    L.append("")
    L.append("| pooled diagonal mean | diagonal std | off-diagonal mean | off-diagonal std |")
    L.append("|---|---|---|---|")
    L.append(f"| {moments['diag_mean']:.6f} | {moments['diag_std']:.6f} "
             f"| {moments['offdiag_mean']:.6e} | {moments['offdiag_std']:.6e} |")
    L.append("")

    L.append("## Null distributions (per metric × ensemble)")
    L.append("")
    for metric_key, label in METRIC_LABELS.items():
        L.append(f"### {label}")
        L.append("")
        if metric_key == "bd_flag":
            L.append("| Ensemble | rate (as coded) | rate multi-block (derived) | n |")
            L.append("|---|---|---|---|")
            for ens, t in tables.items():
                s = t.get(metric_key, {})
                if s.get("n", 0) == 0:
                    L.append(f"| {ens} | — | — | 0 |")
                else:
                    L.append(f"| {ens} | {s['rate']*100:.3f}% | "
                             f"{s['rate_multiblock']*100:.3f}% | {s['n']} |")
        elif metric_key == "bd_top3":
            L.append("| Ensemble | rate | n |")
            L.append("|---|---|---|")
            for ens, t in tables.items():
                s = t.get(metric_key, {})
                if s.get("n", 0) == 0:
                    L.append(f"| {ens} | — | 0 |")
                else:
                    L.append(f"| {ens} | {s['rate']*100:.3f}% | {s['n']} |")
        else:
            L.append("| Ensemble | mean | std | p50 | p90 | p99 | p99.9 | max | n |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for ens, t in tables.items():
                s = t.get(metric_key, {})
                if s.get("n", 0) == 0:
                    L.append(f"| {ens} | — | — | — | — | — | — | — | 0 |")
                    continue
                p = s["percentiles"] if "percentiles" in s else {}
                L.append(
                    f"| {ens} | {fmt(s['mean'])} | {fmt(s['std'])} | "
                    f"{fmt(p.get('50'))} | {fmt(p.get('90'))} | "
                    f"{fmt(p.get('99'))} | {fmt(p.get('99.9'))} | "
                    f"{fmt(s['max'])} | {s['n']} |")
        L.append("")

    L.append("## Empirical comparison sets (points, NOT nulls)")
    L.append("")
    L.append("Per-bridge metric means (n=6 sets; percentile vs `gauss6` null):")
    L.append("")
    L.append("| Set | bridges | co/cross mean | pct | Fiedler mean | pct "
             "| asym mean | pct | BD(detect) | BD(top3) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    g6 = raw["gauss6"]
    for name, m in empirical.items():
        cc = m["co_cross"][np.isfinite(m["co_cross"])]
        fv = m["fiedler"][np.isfinite(m["fiedler"])]
        ay = m["asymmetry"][np.isfinite(m["asymmetry"])]
        bd = float(np.nanmean(m["bd_flag"])) * 100
        t3 = m["bd_top3"][np.isfinite(m["bd_top3"])]
        t3r = float(t3.mean()) * 100 if t3.size else float("nan")
        L.append(
            f"| {name} | {len(m['co_cross'])} | {fmt(cc.mean())} | "
            f"{percentile_of(float(cc.mean()), g6['co_cross']):.1f} | "
            f"{fmt(fv.mean())} | "
            f"{percentile_of(float(fv.mean()), g6['fiedler']):.1f} | "
            f"{fmt(ay.mean())} | "
            f"{percentile_of(float(ay.mean()), g6['asymmetry']):.1f} | "
            f"{bd:.1f}% | {t3r:.1f}% |")
    L.append("")

    L.append("## Headline values vs. nulls (the key deliverable)")
    L.append("")
    L.append("Percentile rule: pct(v) = 100 · #{null < v}/N. `>max` = beyond "
             "every null sample.")
    L.append("")
    L.append("| Headline value | metric | value | null (matched n) | "
             "null p99.9 | null max | percentile | verdict |")
    L.append("|---|---|---|---|---|---|---|---|")
    verdict_flags: list[str] = []
    for hname, h in heads.items():
        metric = h["metric"]
        n = h.get("n", 6)
        for ens in headline_ensembles(h):
            arr = raw[ens][metric]
            finite = arr[np.isfinite(arr)]
            p999 = float(np.percentile(finite, 99.9))
            mx = float(finite.max())
            mn = float(finite.min())
            pct = percentile_of(h["value"], arr)
            beyond = h["value"] > mx
            below = h["value"] < mn
            # Verdict: clearly outside the null band?
            if beyond:
                verdict = "OUTSIDE null (beyond max)"
            elif below:
                verdict = "OUTSIDE null (below min)"
            elif pct > 99.9:
                verdict = "outside 99.9th pct"
            elif pct < 0.1:
                verdict = "outside null band (< 0.1st pct)"
            else:
                verdict = f"**INSIDE null band** (pct {pct:.1f})"
                verdict_flags.append(
                    f"- **{hname}** = {fmt(h['value'])} sits at percentile "
                    f"{pct:.1f} of `{ens}` ({metric}) — NOT distinguishable "
                    "from chance under this null.")
            ens_label = ens if n in HEADLINE_NULL_FOR_N else (
                f"{ens} (nearest; no n={n} ensemble)")
            pct_str = ">max" if beyond else ("<min" if below else f"{pct:.2f}")
            L.append(
                f"| {hname} | {metric} | {fmt(h['value'])} | {ens_label} | "
                f"{fmt(p999)} | {fmt(mx)} | {pct_str} | {verdict} |")
    L.append("")

    L.append("### Flags (metrics whose trained values are NOT clearly outside the null band)")
    L.append("")
    if verdict_flags:
        L.extend(verdict_flags)
    else:
        L.append("None — every headline value fell outside its null band.")
    L.append("")
    L.append("### Notes fixed by protocol")
    L.append("")
    L.append("- The 'spectral-attractor' Fiedler is scale-confounded by "
             "construction: null Fiedler scales with off-diagonal magnitude. "
             "Its placement is against the smoke-moment-matched and "
             "identity+noise families; see flags above for the outcome.")
    L.append("- M3b analytic chance rate is C(6,3)/C(30,3) = 20/4060 = "
             "0.4926%; compare the `bd_top3` rate row for `gauss6`.")
    g6_mb = tables["gauss6"]["bd_flag"]["rate_multiblock"] * 100
    g6_t3 = tables["gauss6"]["bd_top3"]["rate"] * 100
    L.append(f"- H-ch6 reported **100% BD** "
             "(paper/COMPREHENSIVE_EXPERIMENT_TABLE.md, 3-axis blocks). "
             f"Chance multi-block detection under `gauss6`: {g6_mb:.2f}% "
             f"(detect_blocks) / {g6_t3:.2f}% (top-3 criterion). "
             "P(all 88 bridges BD by chance) is effectively zero at these "
             "rates.")
    L.append("- Frozen-identity bridges (BM plan family (c) proper) are "
             "degenerate: co/cross = 0/0, Fiedler = 0, asymmetry = 0. The "
             "identity+eps ensembles bracket the near-init neighbourhood.")
    L.append("")
    L.append("*Raw percentile grids: `nulls.json` (same directory).*")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="BM-000 null-model calibration")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="samples per ensemble (protocol: 10000)")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _verify_pair_definitions()

    print("Loading smoke bank for moment matching ...")
    smoke_sets = load_smoke_bank()
    all_smoke = [b for bridges in smoke_sets.values() for b in bridges]
    if not all_smoke:
        raise SystemExit("STOP: no smoke bridges found under "
                         "results/asset1-smoke — cannot moment-match "
                         "(protocol section 2). A dead path is a silent "
                         "failure; not proceeding.")
    moments = match_moments(all_smoke)
    print(f"  {moments['n_matrices']} matrices; diag {moments['diag_mean']:.4f}"
          f"±{moments['diag_std']:.4f}, offdiag {moments['offdiag_mean']:.2e}"
          f"±{moments['offdiag_std']:.2e}")

    print(f"Generating null ensembles (seed {SEED}, N={args.n:,}) ...")
    tables = run_nulls(SEED, args.n, moments)
    raw = run_nulls.raw

    print("Computing empirical comparison sets ...")
    empirical: dict[str, dict[str, np.ndarray]] = {}
    pilots = {
        "pilot-code (fingerprints/code)": REPO / "results" / "fingerprints" / "code",
        "pilot-math (fingerprints/math)": REPO / "results" / "fingerprints" / "math",
        "pilot-alpaca (exp2/rhombi_fcc_r24)": REPO / "results" / "exp2" / "rhombi_fcc_r24",
    }
    for name, d in pilots.items():
        bridges = load_bridge_dir(d)
        if bridges:
            empirical[name] = compute_metrics(bridges)
        else:
            print(f"  WARNING: no bridges at {d}")
    for name, bridges in smoke_sets.items():
        empirical[name] = compute_metrics(bridges)

    heads = read_headlines()

    print("Writing outputs ...")
    nulls_payload = {
        "protocol": "results/BM-000/PROTOCOL.md",
        "seed": SEED,
        "n_samples": args.n,
        "matched_moments": moments,
        "percentile_grid": PCT_GRID,
        "ensembles": tables,
        "headlines": {
            k: {**v, "percentile_vs_null": {
                ens: percentile_of(v["value"], raw[ens][v["metric"]])
                for ens in headline_ensembles(v)
            }}
            for k, v in heads.items()
        },
        "empirical_sets": {
            name: {m: summarize(vals) for m, vals in mets.items()}
            for name, mets in empirical.items()
        },
    }
    (out_dir / "nulls.json").write_text(
        json.dumps(nulls_payload, indent=2), encoding="utf-8")
    write_results_md(out_dir / "RESULTS.md", moments, tables, raw,
                     empirical, heads, args.n)
    print(f"Done: {out_dir / 'RESULTS.md'}, {out_dir / 'nulls.json'}")


if __name__ == "__main__":
    main()
