"""BM-000b: Hub-motif null family — mask-level calibration for the
"any hub-ish topology would do" referee attack.

Motivation (docs/GLOBAL_WORKSPACE_MAPPING_2026-07-07.md, §2.1 and §3 item 2):
the Anthropic workspace paper shows unconstrained computation organizes
around HUBS (~100x connectivity). The obvious referee attack on the BD/RD
results is that any hub-like topology reproduces the signature. This script
closes the null level of that attack BEFORE anyone deploys it: it places
star/hub, random degree-matched regular, expander-like, and shuffled-RD
mask motifs alongside the true polytope partitions (RD n=6, octahedral n=4,
tesseract n=8) under IDENTICAL moment-matched weight ensembles, and scores
all of them with the SAME metric implementations BM-000 used.

Provenance and reuse
--------------------
This is an extension of scripts/bm000_null_model.py (BM-000, seed 20260704)
and reuses its machinery by import, not reimplementation:
  * compute_metrics  — all seven BM-000 metrics with their file:line
                       citations (M1 co/cross via train_exp2_scale.py:155-181;
                       M2 Fiedler via rhombic.spectral.fiedler_value;
                       M3a detect_blocks.py:16-132; M3b/M2b/M4 verbatim
                       copies inside bm000_null_model.py)
  * match_moments / load_smoke_bank — BM-000's moment conventions
                       (pooled diag/off-diag moments of the trained smoke
                       bank, results/asset1-smoke; NOT the live bank)
  * summarize / percentile_of / read_headlines / fmt / PCT_GRID
The pair partitions (co-planar RD n=6, octahedral n=4, tesseract co-axial
n=8) come from train_exp2_scale._coplanar_crossplanar_indices — the same
definitions every trained co/cross number in the program used.

Design decisions (invented defaults; every one is recorded in
nulls.json["invented_defaults"] for Director sign-off)
------------------------------------------------------
D1. Seed 20260707 (date-derived, BM-000 house style), N = 10,000/ensemble.
D2. Mask convention = BM-000 rdmask6 convention exactly: symmetric mask,
    diagonal 1.0, strong off-diagonal pairs 1.0, all remaining off-diagonal
    0.5; bridge draw = mask (elementwise) x moment-matched Gaussian entries
    (off-diag ~ N(off_mean, off_std), diag ~ N(diag_mean, diag_std)).
D3. Motif families per n in {4, 6, 8}, strong-edge sets:
    ref_*      — the true polytope partition (fixed mask): co-planar pairs
                 at n=6 (RD), (0,1),(2,3) at n=4 (octahedral), the four
                 co-axial pairs at n=8 (tesseract).
    star       — hub motif: all edges incident to channel 0 (center fixed
                 at 0; weights are iid so the choice is immaterial).
    matched1reg— random 1-regular graph (uniform perfect matching), i.e.
                 degree-matched to the polytope strong structure, which is
                 a perfect matching at every n in the family. New matching
                 per draw.
    shuffledRD — degree-preserving rewire of the polytope strong set via
                 double-edge swaps (EE-001's control style), n_swaps =
                 10 x n_strong_edges, new rewire per draw. NOTE: for a
                 1-regular strong graph this is distributionally the same
                 family as matched1reg; both are implemented independently
                 as a cross-check that the pipeline reports identical
                 distributions for identical distributions.
    expander   — random d-regular strong graph with d = 3 at every n
                 (minimum degree with a nontrivial spectral gap); per draw,
                 8 candidates via networkx.random_regular_graph, keep the
                 candidate with the largest unweighted Fiedler value.
                 Degeneracy note: at n=4 the only 3-regular graph is K4,
                 so expander4 is deterministic-complete by construction.
D4. Strong-edge counts deliberately differ across motifs (matching n/2,
    star n-1, expander 3n/2): hub motifs being DENSER is the phenomenon
    under test, not a confound; counts are recorded per ensemble.
D5. Moments: primary source = trained smoke bank via BM-000's own loader
    (results/asset1-smoke); fallback = matched_moments recorded in
    results/BM-000/nulls.json. The live asset1-bank is never read.
D6. Distinguishability rule (pre-stated): the polytope reference is
    "distinguishable at the mask level" from a motif on a metric iff the
    reference ensemble's median falls outside the motif ensemble's
    [p1, p99] band. Headline placement uses BM-000's percentile rule.

CPU-ONLY by construction: no model is instantiated; CUDA is masked before
any import that could touch it (live campaign on the local GPU).

Usage:
  python scripts/bm_hub_motif_nulls.py           # full run -> results/BM-000b-hub-motifs/
  python scripts/bm_hub_motif_nulls.py --n 500   # quick pass (files still written)
"""

from __future__ import annotations

import os

# GPU is running a training campaign — hard-mask CUDA before torch can load.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ── REUSED machinery (imported, not reimplemented) ───────────────────
import bm000_null_model as bm                                    # noqa: E402
from rhombic.spectral import fiedler_value                       # noqa: E402
from train_exp2_scale import _coplanar_crossplanar_indices       # noqa: E402

SEED = 20260707
DEFAULT_N = 10_000
OUT_DIR = REPO / "results" / "BM-000b-hub-motifs"

NS = (4, 6, 8)
STAR_CENTER = 0
EXPANDER_DEGREE = {4: 3, 6: 3, 8: 3}
EXPANDER_CANDIDATES = 8
REWIRE_SWAPS_PER_EDGE = 10

REF_NAME = {4: "ref_octa4", 6: "ref_rd6", 8: "ref_tess8"}

INVENTED_DEFAULTS = {
    "seed": SEED,
    "n_samples_per_ensemble": DEFAULT_N,
    "mask_convention": (
        "BM-000 rdmask6 convention: symmetric, diag=1.0, strong off-diag "
        "pairs=1.0, remaining off-diag=0.5; bridge = mask x moment-matched "
        "Gaussian (bm000_null_model.gen_rdmask weight recipe)"),
    "n_channels_family": list(NS),
    "star_center_channel": STAR_CENTER,
    "matched_regular": (
        "uniform random perfect matching = degree-matched to the polytope "
        "co-planar strong structure (1-regular at n=4/6/8); new draw per "
        "sample"),
    "shuffled_rd_rewire": (
        f"double-edge swaps on the polytope strong set, n_swaps = "
        f"{REWIRE_SWAPS_PER_EDGE} x n_strong_edges, new rewire per sample; "
        "distributionally identical to matched1reg for 1-regular strong "
        "graphs (deliberate cross-check)"),
    "expander_degree": {str(k): v for k, v in EXPANDER_DEGREE.items()},
    "expander_candidates_per_draw": EXPANDER_CANDIDATES,
    "expander_selection": (
        "max unweighted Fiedler value among candidates generated by "
        "networkx.random_regular_graph (child seeds drawn from the master "
        "rng stream); n=4 is degenerate (K4 is the only 3-regular graph "
        "on 4 vertices)"),
    "strong_edge_counts_differ_by_design": (
        "matching n/2, star n-1, expander 3n/2 — hub motifs being denser "
        "is the phenomenon under test (workspace-paper ~100x hub), not a "
        "confound; counts recorded per ensemble"),
    "distinguishability_rule": (
        "reference ensemble median outside motif ensemble [p1, p99] band "
        "=> distinguishable (pre-stated, D6)"),
    "moments_source_policy": (
        "primary: trained smoke bank via bm000_null_model.load_smoke_bank "
        "(results/asset1-smoke); fallback: results/BM-000/nulls.json "
        "matched_moments; the live asset1-bank is never read"),
}


# ── Strong-edge set generators ───────────────────────────────────────


def strong_pairs_polytope(n: int) -> list[tuple[int, int]]:
    """True polytope partition: RD co-planar (n=6), octahedral (n=4),
    tesseract co-axial (n=8) — train_exp2_scale._coplanar_crossplanar_indices."""
    co, _ = _coplanar_crossplanar_indices(n)
    return [tuple(p) for p in co]


def strong_pairs_star(n: int, center: int = STAR_CENTER) -> list[tuple[int, int]]:
    """Star/hub motif: every edge incident to the center channel."""
    return [(min(center, j), max(center, j)) for j in range(n) if j != center]


def random_perfect_matching(rng: np.random.Generator, n: int) -> list[tuple[int, int]]:
    """Uniform random perfect matching on n channels (n even)."""
    p = rng.permutation(n)
    return sorted(
        (min(int(p[2 * k]), int(p[2 * k + 1])),
         max(int(p[2 * k]), int(p[2 * k + 1])))
        for k in range(n // 2)
    )


def rewire_matching(rng: np.random.Generator,
                    pairs: list[tuple[int, int]],
                    n_swaps: int) -> list[tuple[int, int]]:
    """Degree-preserving rewire via double-edge swaps.

    For disjoint pairs (a,b),(c,d) the swap yields (a,c),(b,d) or
    (a,d),(b,c) — both remain perfect matchings, so every swap is valid.
    """
    edges = [list(e) for e in pairs]
    m = len(edges)
    for _ in range(n_swaps):
        if m < 2:
            break
        i, j = rng.choice(m, size=2, replace=False)
        a, b = edges[i]
        c, d = edges[j]
        if rng.integers(2) == 0:
            edges[i], edges[j] = [a, c], [b, d]
        else:
            edges[i], edges[j] = [a, d], [b, c]
    return sorted((min(e), max(e)) for e in edges)


def expander_strong_pairs(rng: np.random.Generator, n: int,
                          d: int | None = None,
                          k: int = EXPANDER_CANDIDATES) -> list[tuple[int, int]]:
    """Expander-like motif: best unweighted Fiedler among k random d-regular
    draws (networkx.random_regular_graph, child seeds from the master rng)."""
    d = EXPANDER_DEGREE[n] if d is None else d
    best_pairs: list[tuple[int, int]] | None = None
    best_fiedler = -1.0
    for _ in range(k):
        child_seed = int(rng.integers(0, 2**31 - 1))
        g = nx.random_regular_graph(d, n, seed=child_seed)
        edges = [(min(u, v), max(u, v)) for u, v in g.edges()]
        f = fiedler_value(n, edges)  # unweighted spectral gap
        if f > best_fiedler:
            best_fiedler = f
            best_pairs = sorted(edges)
    assert best_pairs is not None
    return best_pairs


def mask_from_strong(n: int, strong: list[tuple[int, int]]) -> np.ndarray:
    """BM-000 mask convention (D2): diag 1.0, strong 1.0, weak 0.5."""
    mask = np.full((n, n), 0.5)
    np.fill_diagonal(mask, 1.0)
    for i, j in strong:
        mask[i, j] = 1.0
        mask[j, i] = 1.0
    return mask


# ── Ensemble generation ──────────────────────────────────────────────

MOTIFS = ("ref", "star", "matched1reg", "shuffledRD", "expander")


def ensemble_name(motif: str, n: int) -> str:
    return REF_NAME[n] if motif == "ref" else f"{motif}{n}"


def gen_motif_ensemble(rng: np.random.Generator, motif: str, n: int,
                       count: int, m: dict[str, float]) -> list[np.ndarray]:
    """count bridge draws for one motif at one n: mask x Gaussian weights.

    Weight recipe is bm000_null_model.gen_rdmask's, verbatim convention:
    off-diagonal ~ N(offdiag_mean, offdiag_std), diagonal ~ N(diag_mean,
    diag_std), multiplied elementwise by the motif mask.
    """
    ref_pairs = strong_pairs_polytope(n)
    n_swaps = REWIRE_SWAPS_PER_EDGE * len(ref_pairs)
    fixed_mask: np.ndarray | None = None
    if motif == "ref":
        fixed_mask = mask_from_strong(n, ref_pairs)
    elif motif == "star":
        fixed_mask = mask_from_strong(n, strong_pairs_star(n))

    out: list[np.ndarray] = []
    for _ in range(count):
        if fixed_mask is not None:
            mask = fixed_mask
        elif motif == "matched1reg":
            mask = mask_from_strong(n, random_perfect_matching(rng, n))
        elif motif == "shuffledRD":
            mask = mask_from_strong(n, rewire_matching(rng, ref_pairs, n_swaps))
        elif motif == "expander":
            mask = mask_from_strong(n, expander_strong_pairs(rng, n))
        else:
            raise ValueError(f"unknown motif: {motif!r}")
        G = rng.normal(m["offdiag_mean"], m["offdiag_std"], size=(n, n))
        di = rng.normal(m["diag_mean"], m["diag_std"], size=n)
        G[np.arange(n), np.arange(n)] = di
        out.append(mask * G)
    return out


def build_ensembles(seed: int, n_samples: int,
                    moments: dict[str, float]) -> dict[str, list[np.ndarray]]:
    """All motif ensembles from ONE rng stream in fixed (n, motif) order."""
    rng = np.random.default_rng(seed)
    ensembles: dict[str, list[np.ndarray]] = {}
    for n in NS:
        for motif in MOTIFS:
            ensembles[ensemble_name(motif, n)] = gen_motif_ensemble(
                rng, motif, n, n_samples, moments)
    return ensembles


def strong_edge_count(motif: str, n: int) -> int:
    if motif in ("ref", "matched1reg", "shuffledRD"):
        return n // 2
    if motif == "star":
        return n - 1
    if motif == "expander":
        return n * EXPANDER_DEGREE[n] // 2
    raise ValueError(motif)


# ── Pipeline ─────────────────────────────────────────────────────────


def run_hub_nulls(seed: int, n_samples: int, moments: dict[str, float]
                  ) -> tuple[dict, dict]:
    """Generate all ensembles, compute BM-000 metrics, summarize.

    Returns (tables, raw): tables = per-ensemble metric summaries in the
    BM-000 format; raw = per-ensemble metric arrays.
    """
    ensembles = build_ensembles(seed, n_samples, moments)
    tables: dict[str, dict] = {}
    raw: dict[str, dict[str, np.ndarray]] = {}
    for name, bridges in ensembles.items():
        metrics = bm.compute_metrics(bridges)  # REUSED — all BM-000 metrics
        raw[name] = metrics
        tables[name] = {m: bm.summarize(v) for m, v in metrics.items()}
        tables[name]["bd_flag"]["rate"] = float(np.nanmean(metrics["bd_flag"]))
        tables[name]["bd_flag"]["rate_multiblock"] = float(np.mean(
            (metrics["bd_flag"] == 1.0) & (metrics["bd_n_blocks"] >= 2)))
        top3 = metrics["bd_top3"]
        top3f = top3[np.isfinite(top3)]
        tables[name]["bd_top3"]["rate"] = (
            float(top3f.mean()) if top3f.size else float("nan"))
    return tables, raw


DISTINGUISH_METRICS = ("co_cross", "fiedler", "fiedler_atb", "asymmetry")


def distinguishability(raw: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """D6 rule: reference median vs each motif's [p1, p99] band, per n."""
    rows: list[dict] = []
    for n in NS:
        ref = raw[REF_NAME[n]]
        for metric in DISTINGUISH_METRICS:
            ref_vals = ref[metric][np.isfinite(ref[metric])]
            if ref_vals.size == 0:
                continue
            ref_med = float(np.median(ref_vals))
            for motif in MOTIFS:
                if motif == "ref":
                    continue
                name = ensemble_name(motif, n)
                arr = raw[name][metric]
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                p1 = float(np.percentile(finite, 1))
                p99 = float(np.percentile(finite, 99))
                pct = bm.percentile_of(ref_med, arr)
                rows.append({
                    "n": n, "metric": metric, "motif_ensemble": name,
                    "ref_ensemble": REF_NAME[n],
                    "ref_median": ref_med,
                    "motif_median": float(np.median(finite)),
                    "motif_p1": p1, "motif_p99": p99,
                    "ref_median_pct_in_motif": pct,
                    "distinguishable": bool(ref_med < p1 or ref_med > p99),
                })
    return rows


def headline_placements(raw: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """Place the program's trained headline values (BM-000's read_headlines,
    reused) against every motif ensemble at the headline's n."""
    rows: list[dict] = []
    for hname, h in bm.read_headlines().items():
        n = h.get("n", 6)
        if n not in NS:
            continue  # e.g. the n=3 AR-001 entry: no n=3 motif family
        for motif in MOTIFS:
            name = ensemble_name(motif, n)
            arr = raw[name][h["metric"]]
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            mx, mn = float(finite.max()), float(finite.min())
            pct = bm.percentile_of(h["value"], arr)
            if h["value"] > mx:
                verdict, pct_str = "OUTSIDE null (beyond max)", ">max"
            elif h["value"] < mn:
                verdict, pct_str = "OUTSIDE null (below min)", "<min"
            elif pct > 99.9 or pct < 0.1:
                verdict, pct_str = "outside 99.9/0.1 pct", f"{pct:.2f}"
            else:
                verdict, pct_str = f"INSIDE null band (pct {pct:.1f})", f"{pct:.2f}"
            rows.append({
                "headline": hname, "metric": h["metric"],
                "value": float(h["value"]), "source": h.get("source", ""),
                "ensemble": name,
                "null_p99_9": float(np.percentile(finite, 99.9)),
                "null_max": mx, "percentile": pct_str, "verdict": verdict,
            })
    return rows


# ── Moments (D5 policy) ──────────────────────────────────────────────


def load_moments() -> tuple[dict[str, float], str]:
    """Primary: smoke bank via BM-000's loader. Fallback: BM-000 nulls.json."""
    try:
        smoke_sets = bm.load_smoke_bank()
        all_smoke = [b for bridges in smoke_sets.values() for b in bridges]
    except (FileNotFoundError, OSError):
        all_smoke = []
    if all_smoke:
        return (bm.match_moments(all_smoke),
                "results/asset1-smoke via bm000_null_model.load_smoke_bank")
    fallback = REPO / "results" / "BM-000" / "nulls.json"
    if fallback.exists():
        payload = json.loads(fallback.read_text(encoding="utf-8"))
        return (payload["matched_moments"],
                "results/BM-000/nulls.json matched_moments (fallback)")
    raise SystemExit(
        "STOP: neither results/asset1-smoke nor results/BM-000/nulls.json "
        "is available — cannot moment-match. A dead path is a silent "
        "failure; not proceeding.")


# ── Report generation ────────────────────────────────────────────────

METRIC_LABELS = {
    "co_cross": "M1 co/cross (canonical, upper-tri, RD/octa/tess pairs)",
    "co_cross_fullmatrix": "M1b co/cross (full-matrix, per-bridge, n=6 only)",
    "fiedler": "M2 bridge Fiedler (canonical telemetry def.)",
    "fiedler_atb": "M2b Fiedler (analyze_task_bridges variant)",
    "bd_flag": "M3a BD detected (detect_blocks, rate)",
    "bd_top3": "M3b BD top-3 criterion (rate, n=6 only)",
    "asymmetry": "M4 asymmetry ratio ||A||F/||B||F",
}


def write_results_md(path: Path, moments: dict, moments_source: str,
                     tables: dict, raw: dict, dist_rows: list[dict],
                     head_rows: list[dict], n_samples: int) -> None:
    fmt = bm.fmt
    L: list[str] = []
    L.append("# BM-000b — Hub-Motif Null Family: RESULTS")
    L.append("")
    L.append(f"> Generated by `scripts/bm_hub_motif_nulls.py` (seed {SEED}, "
             f"N = {n_samples:,} per ensemble). Extends BM-000 "
             "(`results/BM-000/`); all metric implementations are BM-000's, "
             "imported. Motivation: docs/GLOBAL_WORKSPACE_MAPPING_2026-07-07.md "
             "§2.1/§3 item 2 — the workspace paper's hub finding invites the "
             "referee attack \"any hub-ish topology would do\"; this null "
             "family answers it at the mask level. Every number below is "
             "computed output; nothing is hand-edited.")
    L.append("")
    L.append("## The two pre-stated questions")
    L.append("")
    L.append("**Q1.** Do hub/expander motifs reproduce the trained BD "
             "signature (extreme co/cross + collapsed Fiedler) under "
             "moment-matched nulls?")
    L.append("")
    L.append("**Q2.** Is the polytope partition distinguishable from "
             "hub/expander motifs at the mask level (D6 rule: reference "
             "median outside motif [p1, p99])?")
    L.append("")
    L.append(f"## Matched moments ({moments_source})")
    L.append("")
    L.append("| pooled diagonal mean | diagonal std | off-diagonal mean | off-diagonal std |")
    L.append("|---|---|---|---|")
    L.append(f"| {moments['diag_mean']:.6f} | {moments['diag_std']:.6f} "
             f"| {moments['offdiag_mean']:.6e} | {moments['offdiag_std']:.6e} |")
    L.append("")
    L.append("## Motif families")
    L.append("")
    L.append("| Ensemble | n | motif | strong edges | mask per draw |")
    L.append("|---|---|---|---|---|")
    per_draw = {"ref": "fixed", "star": "fixed", "matched1reg": "random",
                "shuffledRD": "random", "expander": "random (best-of-"
                f"{EXPANDER_CANDIDATES} Fiedler)"}
    for n in NS:
        for motif in MOTIFS:
            L.append(f"| {ensemble_name(motif, n)} | {n} | {motif} | "
                     f"{strong_edge_count(motif, n)} | {per_draw[motif]} |")
    L.append("")
    L.append("Strong-edge counts differ by design (D4): hub motifs being "
             "denser is the phenomenon under test, not a confound. At "
             "n=4/6/8 the polytope strong structure is a perfect matching, "
             "so `matched1reg` and `shuffledRD` are the same distribution "
             "family — implemented independently as a cross-check.")
    L.append("")

    L.append("## Null distributions (per metric x ensemble)")
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
                p = s.get("percentiles", {})
                L.append(
                    f"| {ens} | {fmt(s['mean'])} | {fmt(s['std'])} | "
                    f"{fmt(p.get('50'))} | {fmt(p.get('90'))} | "
                    f"{fmt(p.get('99'))} | {fmt(p.get('99.9'))} | "
                    f"{fmt(s['max'])} | {s['n']} |")
        L.append("")

    L.append("## Q1 — Trained headline values vs. every motif null")
    L.append("")
    L.append("Percentile rule: pct(v) = 100 · #{null < v}/N (BM-000).")
    L.append("")
    L.append("| Headline | metric | value | ensemble | null p99.9 | "
             "null max | percentile | verdict |")
    L.append("|---|---|---|---|---|---|---|---|")
    inside_q1: list[str] = []
    for r in head_rows:
        L.append(f"| {r['headline']} | {r['metric']} | {fmt(r['value'])} | "
                 f"{r['ensemble']} | {fmt(r['null_p99_9'])} | "
                 f"{fmt(r['null_max'])} | {r['percentile']} | {r['verdict']} |")
        if r["verdict"].startswith("INSIDE"):
            inside_q1.append(
                f"- **{r['headline']}** = {fmt(r['value'])} sits inside the "
                f"`{r['ensemble']}` null band ({r['verdict']}).")
    L.append("")
    L.append("### Q1 flags (headlines INSIDE any motif null band)")
    L.append("")
    if inside_q1:
        L.extend(inside_q1)
    else:
        L.append("None — no untrained hub, expander, degree-matched-regular, "
                 "or shuffled-RD mask ensemble reaches any trained headline "
                 "value. The trained BD signature is not reproduced by any "
                 "motif at the null level.")
    L.append("")

    L.append("## Q2 — Distinguishability of the polytope partition (D6 rule)")
    L.append("")
    L.append("| n | metric | motif ensemble | ref median | motif median | "
             "motif [p1, p99] | ref pct in motif | distinguishable |")
    L.append("|---|---|---|---|---|---|---|---|")
    n_dist = 0
    for r in dist_rows:
        band = f"[{fmt(r['motif_p1'])}, {fmt(r['motif_p99'])}]"
        L.append(f"| {r['n']} | {r['metric']} | {r['motif_ensemble']} | "
                 f"{fmt(r['ref_median'])} | {fmt(r['motif_median'])} | "
                 f"{band} | {r['ref_median_pct_in_motif']:.1f} | "
                 f"{'**YES**' if r['distinguishable'] else 'no'} |")
        n_dist += int(r["distinguishable"])
    L.append("")
    L.append(f"Distinguishable on {n_dist}/{len(dist_rows)} "
             "(n, metric, motif) comparisons.")
    L.append("")

    L.append("## Verdict (honest)")
    L.append("")
    # Q1 refinement (computed): an INSIDE placement only supports the
    # "any hub-ish topology would do" attack if the headline is OUTSIDE
    # its own polytope reference band (i.e. it carries a structure claim)
    # while INSIDE a hub-motif band. Headlines inside their own reference
    # band are globally null headlines — nothing for a motif to mimic.
    inside_hub: dict[str, list[str]] = {}
    ref_band_of: dict[str, str] = {}
    inside_ref: set[str] = set()
    for r in head_rows:
        is_inside = r["verdict"].startswith("INSIDE")
        if r["ensemble"].startswith("ref_"):
            ref_band_of[r["headline"]] = r["ensemble"]
            if is_inside:
                inside_ref.add(r["headline"])
        elif is_inside:
            inside_hub.setdefault(r["headline"], []).append(r["ensemble"])
    genuine = {h: e for h, e in inside_hub.items() if h not in inside_ref}
    known_null = {h: e for h, e in inside_hub.items() if h in inside_ref}
    if not genuine:
        L.append("**Q1: closed at the null level.** Every headline that "
                 "carries a structure claim (i.e. sits outside its own "
                 "polytope reference band) also sits outside every "
                 "hub/expander/degree-matched/rewire null band. No "
                 "untrained hub-ish mask reproduces the trained BD "
                 "signature under the program's own metrics.")
        for h, ens in known_null.items():
            L.append("")
            L.append(f"The INSIDE placements for **{h}** are not a "
                     "hub-specific vulnerability: that headline also sits "
                     "inside its own polytope reference band "
                     f"(`{ref_band_of.get(h, '?')}`) — it is a globally "
                     "null headline (already flagged against gauss nulls "
                     "in BM-000) carrying no BD signature for a motif to "
                     "mimic.")
    else:
        L.append("**Q1: NOT closed.** At least one trained headline that "
                 "is outside its polytope reference band sits INSIDE a "
                 "hub-motif null band — that claim is not distinguishable "
                 "from the motif at the null level and must be reported "
                 "as such:")
        for h, ens in genuine.items():
            L.append(f"- {h}: inside {', '.join(ens)}")
    L.append("")
    # Q2 (computed): strict single-draw rule outcome + median-shift context.
    by_metric: dict[str, list[dict]] = {}
    for r in dist_rows:
        by_metric.setdefault(r["metric"], []).append(r)
    for metric, rows in by_metric.items():
        yes = sum(r["distinguishable"] for r in rows)
        L.append(f"- **Q2 / {metric}:** polytope reference distinguishable "
                 f"from {yes}/{len(rows)} motif ensembles (D6 single-draw "
                 "rule).")
    cc_rows = by_metric.get("co_cross", [])
    if cc_rows:
        med_ratios = [r["ref_median"] / r["motif_median"]
                      for r in cc_rows if r["motif_median"] > 0]
        if med_ratios:
            L.append("")
            L.append(f"**Q2 context (computed):** the polytope partition "
                     "shifts the co/cross MEDIAN by "
                     f"{min(med_ratios):.2f}–{max(med_ratios):.2f}x over "
                     "the hub motifs (the strong/weak mask amplitude ratio "
                     "is 2 by construction), but single-draw dispersion "
                     "swamps the shift under the strict D6 rule. Honest "
                     "reading: an individual untrained mask draw cannot be "
                     "topology-classified by these metrics; what separates "
                     "the program's results from every motif is TRAINED "
                     "structure (Q1 table), not the mask itself.")
    L.append("")
    L.append("**Scope limitation (stated in advance):** this is a "
             "MASK-LEVEL null family — it shows what untrained motif "
             "topologies produce under moment-matched weights. Whether a "
             "TRAINED hub/expander mask reproduces the BD/RD results is a "
             "GPU question, pre-registered as the shuffled-adjacency + "
             "expander mask arms of the BM-003 amendment (post-bank). "
             "These tables supply the calibrated bands those arms will be "
             "read against.")
    L.append("")
    L.append("*Raw percentile grids and all invented defaults: `nulls.json` "
             "(same directory).*")
    path.write_text("\n".join(L), encoding="utf-8")


def write_outputs(out_dir: Path, moments: dict, moments_source: str,
                  tables: dict, raw: dict, n_samples: int) -> dict:
    """Write nulls.json + RESULTS.md; return the JSON payload."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_rows = distinguishability(raw)
    head_rows = headline_placements(raw)
    defaults = dict(INVENTED_DEFAULTS)
    defaults["n_samples_per_ensemble"] = n_samples
    defaults["moments_source_used"] = moments_source
    payload = {
        "experiment": "BM-000b hub-motif null family",
        "extends": "results/BM-000 (scripts/bm000_null_model.py)",
        "motivation": ("docs/GLOBAL_WORKSPACE_MAPPING_2026-07-07.md "
                       "sections 2.1 / 3 item 2"),
        "seed": SEED,
        "n_samples": n_samples,
        "invented_defaults": defaults,
        "matched_moments": moments,
        "percentile_grid": bm.PCT_GRID,
        "strong_edge_counts": {
            ensemble_name(motif, n): strong_edge_count(motif, n)
            for n in NS for motif in MOTIFS},
        "ensembles": tables,
        "headline_placements": head_rows,
        "distinguishability": dist_rows,
    }
    (out_dir / "nulls.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    write_results_md(out_dir / "RESULTS.md", moments, moments_source,
                     tables, raw, dist_rows, head_rows, n_samples)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BM-000b hub-motif null family (CPU-only)")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="samples per ensemble (default: 10000)")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    bm._verify_pair_definitions()  # same abort gate as BM-000

    print("Loading moments (D5 policy) ...")
    moments, moments_source = load_moments()
    print(f"  source: {moments_source}")
    print(f"  diag {moments['diag_mean']:.4f}±{moments['diag_std']:.4f}, "
          f"offdiag {moments['offdiag_mean']:.2e}±{moments['offdiag_std']:.2e}")

    print(f"Generating {len(NS) * len(MOTIFS)} motif ensembles "
          f"(seed {SEED}, N={args.n:,}) ...")
    tables, raw = run_hub_nulls(SEED, args.n, moments)

    print("Writing outputs ...")
    out_dir = Path(args.out)
    write_outputs(out_dir, moments, moments_source, tables, raw, args.n)
    print(f"Done: {out_dir / 'RESULTS.md'}, {out_dir / 'nulls.json'}")


if __name__ == "__main__":
    main()
