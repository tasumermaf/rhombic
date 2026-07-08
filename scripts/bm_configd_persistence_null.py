"""BM-000c: Config-D persistence null — calibration for the "init does not
persist" prediction on the internal corpus-modulated edge-weight arm.

Motivation (docs/GLOBAL_WORKSPACE_MAPPING_2026-07-07.md, §2.2 line 58 and
§3 item 2 line 92):
BM-003 Config D is an internal initialization-variant arm (corpus-modulated
edge-weight init; results NOT published — docs/BM_BATTERY_PLAN.md:81, "Labels
D and E are reserved: D is an internal initialization-variant arm"). The
workspace paper's structure/content split plus FI-002 (topology is
pair-specification-determined, not init-determined) jointly predict, BEFORE
launch, that the corpus init pattern DRIFTS INTO BM-000's rewire-null
percentile band by end of training, with no performance delta. If it nulls,
the workspace paper supplies the citable general principle for the internal
tombstone.

This script calibrates the null distribution of the PERSISTENCE METRIC — the
correlation between a trained bridge's off-diagonal edge weights and the
Config-D corpus-modulated init template — so that the end-of-training number
can be read against a pre-registered band rather than an eyeballed threshold.
It is the CPU-only `bm000_null_model.py` extension the memo mandates.

Provenance and reuse
--------------------
Extension of scripts/bm000_null_model.py (BM-000, seed 20260704) and
scripts/bm_hub_motif_nulls.py (BM-000b, seed 20260707). Reuses BM-000's null
machinery by import, not reimplementation:
  * gen_rdmask       — BM-000's N-B family: degree-preserving RD-mask rewire
                       x moment-matched Gaussian. THIS is "BM-000's rewire
                       null" that the memo's prediction names.
  * gen_gauss        — BM-000's N-A/N-D family: moment-matched Gaussian.
  * match_moments /
    load_smoke_bank  — BM-000's moment conventions (results/asset1-smoke;
                       NOT the live bank).
  * summarize /
    percentile_of /
    fmt / PCT_GRID    — BM-000's summary + percentile conventions.

The Config-D template is loaded at RUNTIME via rhombic.corpus (bridge_init
mode 'corpus_coupled' = corpus.corpus_coupled_matrix(edge_values())). Its raw
values are proprietary Stream-B IP and appear NOWHERE in this file, the tests,
the results JSON, or the prereg note — only a SHA-256 of the template array
bytes is recorded for identity.

Design decisions (invented defaults; every one recorded in
nulls.json["invented_defaults"] for Director sign-off)
------------------------------------------------------
D1. Seed 20260708 (date-derived, BM-000 house style), N = 10,000/ensemble.
D2. PERSISTENCE METRIC (PINNED): Pearson correlation r between a bridge's 30
    OFF-DIAGONAL directed entries (row-major order over i != j) and the
    Config-D template's 30 off-diagonal directed entries. The DIAGONAL is
    EXCLUDED: the corpus_coupled template's diagonal is a constant identity
    (all 1.0) carrying no corpus signal, so including it would inject a
    degenerate constant block and bias r. r is scale- and shift-invariant,
    which is a feature here (see D8). Degenerate input (zero variance on
    either side) -> NaN, filtered by summarize (BM-000 convention).
D3. TEMPLATE (PINNED): bridge_init(mode='corpus_coupled') =
    rhombic.corpus.corpus_coupled_matrix(edge_values()). Config D is the
    "corpus-modulated EDGE-WEIGHT init"; corpus_coupled is the L-026-corrected
    corpus init that places corpus-derived weights on the OFF-DIAGONAL edge
    weights (identity diagonal). The pre-L-026 'corpus' mode (diagonal
    scaling) is NOT used; the choice is flagged as an open question for the
    reviewer.
D4. NULLS (both from BM-000 machinery; PRIMARY = rewire):
    rewire            — bm.gen_rdmask (BM-000 N-B). PRIMARY: the memo predicts
                        drift into "BM-000's rewire-null percentile band," and
                        BM-000's rewire null IS gen_rdmask.
    matched_moments   — bm.gen_gauss (BM-000 N-A/N-D). SECONDARY cross-check.
    Each null bridge is scored by the persistence metric against the FIXED
    template (bridges vary; the template is held fixed — this is the
    distribution of template-correlation a bridge with NO memory of the
    init would produce).
D5. Two-sided band: "init does not persist" = trained r INSIDE [p2.5, p97.5];
    "init persisted / falsified" = trained r ABOVE p99 (one-sided upper).
    Percentile grid = BM-000's PCT_GRID augmented with 2.5 and 97.5 for the
    two-sided band; p99 is the falsification threshold.
D6. Moments source policy = BM-000b D5 exactly: primary = trained smoke bank
    via bm.load_smoke_bank (results/asset1-smoke); fallback =
    results/BM-000/nulls.json matched_moments. The live asset1-bank is never
    read. (See D8: moments barely affect this metric.)
D7. Single RNG stream (seed D1), ensembles drawn in fixed order
    (rewire before matched_moments) so the run is byte-reproducible.
D8. SCALE-INVARIANCE (recorded honestly): Pearson r is invariant to positive
    affine transforms of the bridge, so the calibrated bands are essentially
    INDEPENDENT of the trained-bridge moment values — they depend only on the
    off-diagonal dimensionality (30 directed / 15 symmetric) and the template
    structure (through the rewire mask). This is a strength: the calibration
    is valid before the real Config-D trained moments exist. Moments are
    still loaded (D6) for faithfulness to BM-000's rewire recipe; they enter
    only through the off-diagonal mean/std ratio of the mask's Gaussian
    weights, which is ~0 in practice.

CPU-ONLY by construction: no model is instantiated; CUDA is masked before any
import that could touch it (a training campaign is live on the local GPU).
No bank data is read; --out-dir refusing any path containing 'asset1-bank'.

Usage:
  python scripts/bm_configd_persistence_null.py            # full run
  python scripts/bm_configd_persistence_null.py --n 500    # quick pass
"""

from __future__ import annotations

import os

# GPU is running a training campaign — hard-mask CUDA before torch can load.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import hashlib
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

# ── REUSED machinery (imported, not reimplemented) ───────────────────
import bm000_null_model as bm                                     # noqa: E402

SEED = 20260708
DEFAULT_N = 10_000
N_CHANNELS = 6
OUT_DIR = REPO / "results" / "BM-000c-configd-persistence"

# Two-sided persistence band: BM-000's grid augmented with 2.5 / 97.5.
PERSIST_PCT_GRID = sorted(set(bm.PCT_GRID) | {2.5, 97.5})

# Null family names. Primary first (fixed draw order, D7).
NULLS = ("rewire", "matched_moments")
PRIMARY_NULL = "rewire"

INVENTED_DEFAULTS = {
    "seed": SEED,
    "n_samples_per_null": DEFAULT_N,
    "n_channels": N_CHANNELS,
    "persistence_metric": (
        "Pearson r between a bridge's 30 off-diagonal directed entries "
        "(row-major over i != j) and the Config-D template's 30 off-diagonal "
        "directed entries; diagonal EXCLUDED (template diagonal is constant "
        "identity, no corpus signal); degenerate (zero-variance) -> NaN"),
    "template_source": (
        "rhombic.corpus.corpus_coupled_matrix(rhombic.corpus.edge_values()) "
        "== bridge_init(mode='corpus_coupled'); loaded at RUNTIME; raw values "
        "are proprietary Stream-B IP and are recorded only as a SHA-256"),
    "template_mode_choice": (
        "'corpus_coupled' (L-026-corrected corpus EDGE-WEIGHT init, weights on "
        "off-diagonal). The pre-L-026 'corpus' mode (diagonal scaling) is NOT "
        "used; flagged as an open question for the reviewer"),
    "primary_null": (
        "rewire = bm.gen_rdmask (BM-000 N-B, degree-preserving RD-mask rewire "
        "x moment-matched Gaussian) — the family the memo's prediction names "
        "('BM-000's rewire-null percentile band')"),
    "secondary_null": (
        "matched_moments = bm.gen_gauss (BM-000 N-A/N-D, moment-matched "
        "Gaussian) — cross-check"),
    "null_scoring": (
        "each null bridge's off-diagonal directed entries correlated against "
        "the FIXED template's; bridges vary, template held fixed"),
    "band_rule": (
        "init-does-not-persist = trained r INSIDE [p2.5, p97.5]; "
        "init-persisted/falsified = trained r ABOVE p99 (one-sided upper); "
        "intermediate reported with exact percentile, no threshold re-rolling"),
    "percentile_grid": (
        "BM-000 PCT_GRID augmented with 2.5 and 97.5 for the two-sided band"),
    "moments_source_policy": (
        "primary: trained smoke bank via bm000_null_model.load_smoke_bank "
        "(results/asset1-smoke); fallback: results/BM-000/nulls.json "
        "matched_moments; the live asset1-bank is never read"),
    "scale_invariance": (
        "Pearson r is invariant to positive affine transforms of the bridge, "
        "so the bands are essentially independent of the trained-bridge moment "
        "values; valid before real Config-D trained moments exist"),
}


# ── Persistence metric ───────────────────────────────────────────────


def offdiag_index(n: int = N_CHANNELS) -> list[tuple[int, int]]:
    """Fixed row-major order over off-diagonal (i != j) positions."""
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def offdiag_vector(M: np.ndarray) -> np.ndarray:
    """The 30 off-diagonal directed entries of an n x n matrix, row-major."""
    n = M.shape[0]
    idx = offdiag_index(n)
    return np.array([M[i, j] for i, j in idx], dtype=np.float64)


def persistence_metric(B: np.ndarray, template_off: np.ndarray) -> float:
    """Pearson r between a bridge's off-diagonal entries and the template's.

    D2. Diagonal excluded. Scale/shift-invariant. Degenerate -> NaN.
    """
    b = offdiag_vector(B)
    if b.size != template_off.size:
        raise ValueError(
            f"off-diag size mismatch: bridge {b.size} vs template "
            f"{template_off.size}")
    # Pearson r is undefined when either vector is constant. Peak-to-peak is
    # exactly 0 for a truly constant vector (unlike std, which leaves ~1e-17
    # summation residue), so it detects the degenerate case without a
    # scale-dependent tolerance.
    if np.ptp(b) == 0.0 or np.ptp(template_off) == 0.0:
        return float("nan")
    return float(np.corrcoef(b, template_off)[0, 1])


# ── Template loading (runtime, IP-safe) ──────────────────────────────


def load_template() -> tuple[np.ndarray, str]:
    """Load the Config-D corpus_coupled template and its SHA-256 identity.

    Returns (template_matrix, sha256_hex). Raises CorpusUnavailable-derived
    error via rhombic.corpus if the proprietary data file is absent — the
    caller (main) turns that into a STOP rather than fabricating.
    """
    from rhombic.corpus import corpus_available, edge_values, corpus_coupled_matrix
    if not corpus_available():
        raise SystemExit(
            "STOP: rhombic corpus (corpus_private.json) is unavailable — the "
            "Config-D template cannot be loaded. Not fabricating a template; "
            "reporting instead. (A dead path is a silent failure.)")
    M = corpus_coupled_matrix(edge_values())
    sha = hashlib.sha256(np.ascontiguousarray(M, dtype=np.float64).tobytes()
                         ).hexdigest()
    return M, sha


# ── Null ensembles (single RNG stream, D7) ───────────────────────────


def build_nulls(seed: int, n_samples: int, moments: dict[str, float]
                ) -> dict[str, list[np.ndarray]]:
    """rewire (gen_rdmask) then matched_moments (gen_gauss), one rng stream."""
    rng = np.random.default_rng(seed)
    return {
        "rewire": bm.gen_rdmask(rng, n_samples, moments),
        "matched_moments": bm.gen_gauss(rng, N_CHANNELS, n_samples, moments),
    }


def band_edges(arr: np.ndarray) -> dict[str, float]:
    """Two-sided persistence band + one-sided falsification threshold (D5)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"p2.5": float("nan"), "p97.5": float("nan"),
                "p99": float("nan")}
    return {
        "p2.5": float(np.percentile(finite, 2.5)),
        "p97.5": float(np.percentile(finite, 97.5)),
        "p99": float(np.percentile(finite, 99)),
    }


def summarize_persist(arr: np.ndarray) -> dict:
    """BM-000 summarize (mean/std/min/max + BM-000 grid) augmented with the
    2.5/97.5/99 band edges (D5). Reuses bm.summarize for the shared fields."""
    s = bm.summarize(arr)
    # bm.summarize uses BM-000's PCT_GRID (no 2.5/97.5); add the two-sided band.
    finite = arr[np.isfinite(arr)]
    if finite.size:
        for p in (2.5, 97.5):
            s.setdefault("percentiles", {})[str(p)] = float(
                np.percentile(finite, p))
    s["band"] = band_edges(arr)
    return s


def run_persistence_nulls(seed: int, n_samples: int,
                          moments: dict[str, float],
                          template_off: np.ndarray
                          ) -> tuple[dict, dict]:
    """Generate both null families, score with the persistence metric.

    Returns (tables, raw): tables = per-null summaries; raw = metric arrays.
    """
    nulls = build_nulls(seed, n_samples, moments)
    tables: dict[str, dict] = {}
    raw: dict[str, np.ndarray] = {}
    for name in NULLS:  # fixed order
        vals = np.array([persistence_metric(B, template_off)
                         for B in nulls[name]], dtype=np.float64)
        raw[name] = vals
        tables[name] = summarize_persist(vals)
    return tables, raw


# ── Output-shape IP guard ────────────────────────────────────────────


# Keys whose numeric lists are schema-defined and safe by construction. The
# percentile grid legitimately has 15 entries (BM-000's 13 + 2.5 + 97.5),
# which collides with a 6x6 upper-triangle off-diagonal length — so it is
# whitelisted by name rather than by length.
_IP_GUARD_WHITELIST_KEYS = frozenset({"percentile_grid"})


def assert_no_raw_matrix(payload: dict) -> None:
    """Structural IP guard: the payload must not smuggle a bridge-shaped array.

    Refuses any numeric list of length in {6, 15, 30, 36} (the sizes a 6x6
    template / its off-diagonal / half-off-diagonal / flattened matrix could
    take), and any nested 6x6. Correlations, percentiles, counts, and the
    SHA-256 string are all scalars or short/whitelisted grids and pass. Called
    before writing, and re-checked by the test suite.
    """
    suspect = {6, 15, 30, 36}

    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, (list, tuple)):
            if (key not in _IP_GUARD_WHITELIST_KEYS
                    and x and all(isinstance(e, (int, float)) for e in x)
                    and len(x) in suspect):
                raise AssertionError(
                    f"IP guard: numeric array of length {len(x)} under key "
                    f"{key!r} — possible template/matrix leak")
            for e in x:
                walk(e)

    walk(payload)


# ── Report generation ────────────────────────────────────────────────


def _null_label(name: str) -> str:
    return {
        "rewire": "rewire (bm.gen_rdmask — BM-000 N-B, PRIMARY)",
        "matched_moments": "matched-moments (bm.gen_gauss — BM-000 N-A/N-D)",
    }[name]


def write_results_md(path: Path, tables: dict, raw: dict, moments: dict,
                     moments_source: str, template_sha: str,
                     n_samples: int) -> None:
    fmt = bm.fmt
    L: list[str] = []
    L.append("# BM-000c — Config-D Persistence Null: RESULTS")
    L.append("")
    L.append("> **INTERNAL-ONLY.** Config D is an internal "
             "initialization-variant arm; its results are not published "
             "(docs/BM_BATTERY_PLAN.md:81). This file calibrates a null "
             "band; it reports no trained result.")
    L.append("")
    L.append(f"> Generated by `scripts/bm_configd_persistence_null.py` "
             f"(seed {SEED}, N = {n_samples:,} per null). Extends BM-000 "
             "(`results/BM-000/`) and BM-000b (`results/BM-000b-hub-motifs/`); "
             "all null generators are BM-000's, imported. Motivation: "
             "docs/GLOBAL_WORKSPACE_MAPPING_2026-07-07.md §2.2 (line 58) / §3 "
             "item 2 (line 92). Every number below is computed output; "
             "nothing is hand-edited.")
    L.append("")
    L.append("## The pre-registered prediction")
    L.append("")
    L.append("> The corpus init pattern **drifts into BM-000's rewire-null "
             "percentile band by end of training, with no performance "
             "delta** — jointly predicted by FI-002 (topology is "
             "pair-specification-determined, not init-determined) and the "
             "workspace paper's structure/content split.")
    L.append("")
    L.append("## The persistence metric (pinned)")
    L.append("")
    L.append("Pearson **r** between a trained bridge's 30 off-diagonal "
             "directed edge-weight entries (row-major, i != j) and the "
             "Config-D corpus_coupled init template's 30 off-diagonal "
             "entries. The diagonal is excluded (the template diagonal is a "
             "constant identity, no corpus signal). r is scale/shift-"
             "invariant, so these bands do not depend on the trained-bridge "
             "moment values — a strength, since the real Config-D trained "
             "moments do not exist yet.")
    L.append("")
    L.append(f"**Template identity (IP-safe):** corpus_coupled(edge_values); "
             f"SHA-256 `{template_sha}`. Raw template values are proprietary "
             "Stream-B IP and appear nowhere in this run.")
    L.append("")
    L.append(f"## Matched moments ({moments_source})")
    L.append("")
    L.append("| pooled diagonal mean | diagonal std | off-diagonal mean | off-diagonal std |")
    L.append("|---|---|---|---|")
    L.append(f"| {moments['diag_mean']:.6f} | {moments['diag_std']:.6f} "
             f"| {moments['offdiag_mean']:.6e} | {moments['offdiag_std']:.6e} |")
    L.append("")
    L.append("*(Moments enter only the rewire mask's Gaussian weights; the "
             "Pearson metric is scale-invariant so the bands are effectively "
             "moment-independent — see the metric note above.)*")
    L.append("")
    L.append("## Calibrated null distributions (persistence r)")
    L.append("")
    L.append("| Null | mean | std | p2.5 | p50 | p97.5 | **p99** | p99.9 | "
             "max | n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in NULLS:
        s = tables[name]
        p = s.get("percentiles", {})
        b = s.get("band", {})
        L.append(
            f"| {_null_label(name)} | {fmt(s.get('mean'))} | "
            f"{fmt(s.get('std'))} | {fmt(b.get('p2.5'))} | "
            f"{fmt(p.get('50'))} | {fmt(b.get('p97.5'))} | "
            f"**{fmt(b.get('p99'))}** | {fmt(p.get('99.9'))} | "
            f"{fmt(s.get('max'))} | {s.get('n')} |")
    L.append("")
    prim = tables[PRIMARY_NULL]["band"]
    L.append("## Pre-registered three-outcome reading (PRIMARY = rewire null)")
    L.append("")
    L.append(f"Primary (rewire) band: **[{fmt(prim['p2.5'])}, "
             f"{fmt(prim['p97.5'])}]** two-sided; falsification threshold "
             f"**p99 = {fmt(prim['p99'])}**.")
    L.append("")
    L.append("| End-of-training persistence r | Reading |")
    L.append("|---|---|")
    L.append(f"| INSIDE [{fmt(prim['p2.5'])}, {fmt(prim['p97.5'])}] | "
             "**Prediction CONFIRMED** — init does not persist; the "
             "workspace paper's structure/content split supplies the citable "
             "principle for the internal tombstone. |")
    L.append(f"| ABOVE p99 = {fmt(prim['p99'])} | **Init PERSISTED, "
             "prediction falsified** — reported as such. |")
    L.append("| intermediate / between p97.5 and p99 | **Ambiguous** — "
             "reported with the exact percentile; no threshold re-rolling. |")
    L.append("")
    L.append("**Scope:** this calibrates the null band ONLY. No trained "
             "Config-D bridge is read here (the arm is internal and, per the "
             "memo, may never launch); when/if a trained bridge exists its r "
             "is scored by this same metric and read against the band above. "
             "Config-D results are not published.")
    L.append("")
    L.append("*Raw percentile grids and all invented defaults: `nulls.json` "
             "(same directory).*")
    path.write_text("\n".join(L), encoding="utf-8")


def build_payload(tables: dict, moments: dict, moments_source: str,
                  template_sha: str, n_samples: int) -> dict:
    defaults = dict(INVENTED_DEFAULTS)
    defaults["n_samples_per_null"] = n_samples
    defaults["moments_source_used"] = moments_source
    prim = tables[PRIMARY_NULL]["band"]
    return {
        "experiment": "BM-000c Config-D persistence null",
        "internal_only": True,
        "publication_status": ("Config D is an internal initialization-variant "
                               "arm; results not published "
                               "(docs/BM_BATTERY_PLAN.md:81)"),
        "extends": ("results/BM-000 (scripts/bm000_null_model.py), "
                    "results/BM-000b-hub-motifs (scripts/bm_hub_motif_nulls.py)"),
        "motivation": ("docs/GLOBAL_WORKSPACE_MAPPING_2026-07-07.md "
                       "section 2.2 line 58 / section 3 item 2 line 92"),
        "prediction": ("the corpus init pattern drifts into BM-000's "
                       "rewire-null percentile band by end of training, with "
                       "no performance delta (FI-002 + structure/content "
                       "split)"),
        "seed": SEED,
        "n_samples": n_samples,
        "invented_defaults": defaults,
        "template_identity": {
            "source": ("rhombic.corpus.corpus_coupled_matrix("
                       "rhombic.corpus.edge_values())"),
            "bridge_init_mode": "corpus_coupled",
            "shape": [N_CHANNELS, N_CHANNELS],
            "sha256": template_sha,
            "note": ("Config-D corpus-modulated edge-weight init template; "
                     "raw values are proprietary Stream-B IP and are NOT "
                     "recorded — only this content hash for identity"),
        },
        "matched_moments": moments,
        "percentile_grid": PERSIST_PCT_GRID,
        "primary_null": PRIMARY_NULL,
        "nulls": tables,
        "reading": {
            "primary_band_p2.5": prim["p2.5"],
            "primary_band_p97.5": prim["p97.5"],
            "falsification_threshold_p99": prim["p99"],
            "confirmed": ("end-of-training r INSIDE [p2.5, p97.5] -> init "
                          "does not persist; prediction CONFIRMED"),
            "falsified": ("end-of-training r ABOVE p99 -> init persisted; "
                          "prediction falsified"),
            "ambiguous": ("intermediate -> reported with exact percentile; "
                          "no threshold re-rolling"),
        },
    }


def write_outputs(out_dir: Path, tables: dict, moments: dict,
                  moments_source: str, template_sha: str,
                  n_samples: int, raw: dict) -> dict:
    """Write nulls.json + RESULTS.md; return the JSON payload."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(tables, moments, moments_source, template_sha,
                            n_samples)
    assert_no_raw_matrix(payload)  # IP guard before anything hits disk
    (out_dir / "nulls.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    write_results_md(out_dir / "RESULTS.md", tables, raw, moments,
                     moments_source, template_sha, n_samples)
    return payload


# ── Moments (D6 policy, reused verbatim from BM-000b) ────────────────


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
        "STOP: neither results/asset1-smoke nor results/BM-000/nulls.json is "
        "available — cannot moment-match. A dead path is a silent failure; "
        "not proceeding.")


# ── Bank-path guard ──────────────────────────────────────────────────


def assert_safe_out_dir(out_dir: str | os.PathLike) -> Path:
    """Refuse any --out-dir whose path contains 'asset1-bank' (live campaign;
    null-model work never touches the bank)."""
    p = Path(out_dir)
    if "asset1-bank" in str(p).replace("\\", "/").lower():
        raise SystemExit(
            f"STOP: refusing out-dir containing 'asset1-bank' ({p}); the live "
            "adapter bank is off-limits to null-model work.")
    return p


# ── Entry point ──────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BM-000c Config-D persistence null (CPU-only)")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="samples per null (default: 10000)")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = assert_safe_out_dir(args.out_dir)

    bm._verify_pair_definitions()  # same abort gate as BM-000

    print("Loading Config-D template (runtime, IP-safe) ...")
    template, template_sha = load_template()
    template_off = offdiag_vector(template)
    print(f"  template corpus_coupled, sha256 {template_sha[:16]}... "
          f"(raw values not shown — proprietary)")

    print("Loading moments (D6 policy) ...")
    moments, moments_source = load_moments()
    print(f"  source: {moments_source}")

    print(f"Calibrating {len(NULLS)} null families "
          f"(seed {SEED}, N={args.n:,}) ...")
    tables, raw = run_persistence_nulls(SEED, args.n, moments, template_off)
    for name in NULLS:
        b = tables[name]["band"]
        print(f"  {name:16s} band [p2.5={b['p2.5']:+.4f}, "
              f"p97.5={b['p97.5']:+.4f}]  p99={b['p99']:+.4f}")

    print("Writing outputs ...")
    write_outputs(out_dir, tables, moments, moments_source, template_sha,
                  args.n, raw)
    print(f"Done: {out_dir / 'RESULTS.md'}, {out_dir / 'nulls.json'}")


if __name__ == "__main__":
    main()
