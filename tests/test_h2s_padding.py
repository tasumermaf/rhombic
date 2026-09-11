"""Tests for the H2-S rank-padding and padded-slot-occupancy clause.

The clause is Director-pinned and binding on the H2-S arm:

  AMENDMENT_H2S_RANK_FRACTION_v2_2026-08-04.md:91-101 (section 3,
  "Mandatory analysis-side clause"), restated as pin condition 3 at
  :131-134; pin GRANTED at :117 --- pad both families to
  sigma_slots = max(rank) on the trailing-zero convention, report the
  registered contrast on BOTH the padded spectrum and the top-24 slice,
  guard zero-variance padded slots before standardization, and
  "curing by top-24 truncation alone is prohibited".

  DIRECTOR_GRADES_2026-08-04.md:29 item (c) --- "approved as written,
  with one addition": report the padded-slot occupancy, so a null on the
  padded contrast can be distinguished from "the extra slots were never
  used."

Five required cases: padding-to-max with the guard; the contrast computed
for BOTH views; truncation-only raises; occupancy on a synthetic case with
a hand-computed answer; and a regression that equal-rank inputs are
unchanged.

A sixth case covers the RECORD WRITE. "Report" is what pin condition 3
requires, and the only place rank_padding and padded_slot_occupancy enter
the results record is the unequal-rank block of ``analyze_bank`` --- so
that block is exercised end to end on synthetic banks whose two families
carry DIFFERENT ranks (4 and 6), with the equal-rank control (4 and 4)
alongside it.

Pre-registration hygiene, same as tests/test_asset1_d1.py: cases 1-5 run
on pure-array fixtures and case 6 on synthetic banks written into
``tmp_path`` by asset1_synth's own writers. Nothing reads or writes
results/asset1-bank/, no HF downloads, no network, no GPU. The H2-S arm is
UNRUN and no rank-54 bank exists, which is the point --- the clause has to
exist in code BEFORE the GPU is spent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("torch")

import asset1_d1_identifiability as d1  # noqa: E402
import asset1_synth as synth  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────


def _spectrum_matrix(n_runs, n_blocks, sigma_slots, native_rank, rng):
    """Feature matrix laid out exactly as h2_features_for_run builds it:
    n_blocks concatenated blocks of sigma_slots log1p(sigma) values, with
    slots >= native_rank left at the trailing zero of the pad."""
    X = np.zeros((n_runs, n_blocks * sigma_slots), dtype=np.float64)
    for b in range(n_blocks):
        lo = b * sigma_slots
        X[:, lo:lo + native_rank] = np.abs(
            rng.standard_normal((n_runs, native_rank))) + 0.5
    return X


def _probe_matrix(n_runs, n_blocks, sigma_slots, proj_dim, native_rank, rng):
    """Probe layout: per block [sig; u.ravel(); v.ravel()], each segment
    slot-major, padded slots zero in every segment."""
    rows = []
    for _ in range(n_runs):
        blocks = []
        for _ in range(n_blocks):
            sig = np.zeros(sigma_slots)
            u = np.zeros((sigma_slots, proj_dim))
            v = np.zeros((sigma_slots, proj_dim))
            sig[:native_rank] = np.abs(
                rng.standard_normal(native_rank)) + 0.5
            u[:native_rank] = rng.standard_normal((native_rank, proj_dim))
            v[:native_rank] = rng.standard_normal((native_rank, proj_dim))
            blocks.append(np.concatenate([sig, u.ravel(), v.ravel()]))
        rows.append(np.concatenate(blocks))
    return np.stack(rows)


def _canon(obj):
    """Exact, order-sensitive comparison key for a results dict."""
    return json.dumps(obj, sort_keys=False, default=d1._json_default)


def _mixed_rank_bank(root, ranks, *, n_tasks=3, n_reps=2, n_layers=2,
                     d_model=16, n_channels=2, task_effect=1.0, seed=11):
    """Write a miniature synthetic bank whose families carry DIFFERENT ranks.

    ``asset1_synth.make_synthetic_bank`` takes ONE rank for the whole bank
    --- which is precisely the case the H2-S clause does NOT govern --- so
    the per-family loop is unrolled here over asset1_synth's OWN writers:
    ``_family_geometry``, ``_write_run`` and ``RunSpec.to_manifest_entry``.
    Every file on disk therefore has the exact schema
    ``asset1_analysis_io`` reads (adapter_state.pt key naming, config.json
    with its ``rank`` field, metrics.json, COMPLETE markers, bridge npys,
    bank_manifest.json), which is what makes this a test OF analyze_bank
    rather than of a mock.

    Fixture only: writes under tmp_path, reads nothing from
    results/asset1-bank/, no HF download, no network, no GPU.
    Returns (root, n_runs).
    """
    root = Path(root)
    if "asset1-bank" in root.resolve().parts:
        raise AssertionError(f"fixture refuses to write into {root}")
    root.mkdir(parents=True, exist_ok=True)
    families = [{"model": f"synthetic/family-{f}", "short": f"synthfam{f}"}
                for f in range(len(ranks))]
    tasks = [f"task{t:02d}" for t in range(n_tasks)]
    geoms = [synth._family_geometry(f, d_model, n_layers)
             for f in range(len(ranks))]
    specs, idx = [], 0
    for rep in range(n_reps):
        for f in range(len(ranks)):
            for t in range(n_tasks):
                spec = synth.RunSpec(
                    run_index=idx,
                    family=families[f]["model"],
                    family_short=families[f]["short"],
                    task=tasks[t],
                    replicate=rep,
                    seed=synth.SEED_BASE + idx,
                    data_seed=synth.DATA_SEED_BASE + idx,
                    run_dir=(root / families[f]["short"] / tasks[t]
                             / f"run_{idx:03d}"))
                synth._write_run(spec, f, t, geoms[f], ranks[f], n_channels,
                                 task_effect, seed)
                specs.append(spec)
                idx += 1
    manifest = {
        "updated_at": synth._PLACEHOLDER_END,
        "campaign": {
            "tag": "synthetic-mixed-rank",
            "bank_root": str(root),
            "families": families,
            "tasks": tasks,
            "n_replicates": n_reps,
            "n_runs": len(specs),
            "max_steps": synth.METRIC_STEPS[-1],
        },
        "status_counts": {"COMPLETE": len(specs)},
        "runs": [s.to_manifest_entry("COMPLETE") for s in specs],
    }
    (root / "bank_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return root, len(specs)


# ── Case 1 — padding to max rank, with the zero-variance guard ──────


def test_resolve_sigma_slots_pads_to_max_rank():
    """sigma_slots = max(rank), top_slots = min(rank), pad widths per
    family, and the unequal flag --- the case that used to be a hard
    raise ('sigma_slots aggregation undefined') and is now the case the
    clause governs."""
    slots = d1.resolve_sigma_slots({"short": 24, "long": 54})
    assert slots["sigma_slots"] == 54
    assert slots["top_slots"] == 24
    assert slots["native_ranks"] == {"short": 24, "long": 54}
    assert slots["pad_width"] == {"short": 30, "long": 0}
    assert slots["unequal"] is True
    # The binding text travels with the resolution, not just the numbers.
    assert "Curing by top-24 truncation alone is prohibited" in slots["clause"]


def test_pad_to_max_rank_checks_layout_and_guard():
    """The checked pad step accepts a correctly padded short leg, counts
    the padded band, and reports it as identically zero."""
    rng = np.random.default_rng(0)
    sigma_slots, top_slots, n_blocks = 54, 24, 3
    feats = {
        "short": _spectrum_matrix(6, n_blocks, sigma_slots, top_slots, rng),
        "long": _spectrum_matrix(6, n_blocks, sigma_slots, sigma_slots, rng),
    }
    rep = d1.pad_to_max_rank(feats, {"short": 24, "long": 54}, sigma_slots,
                             n_blocks)
    assert rep["truncation_occurred"] is False
    assert rep["block_dim"] == sigma_slots
    assert rep["families"]["short"]["pad_width"] == 30
    assert rep["families"]["short"]["padded_columns"] == 30 * n_blocks
    assert rep["families"]["short"]["padded_region_all_zero"] is True
    assert rep["families"]["short"]["max_abs_in_padded_region"] == 0.0
    assert rep["families"]["long"]["pad_width"] == 0
    assert rep["families"]["long"]["padded_columns"] == 0


def test_pad_to_max_rank_rejects_nonzero_pad_and_overlong_rank():
    """Two refusals: a padded band that is not identically zero (the
    trailing-zero convention the clause cites does not hold), and a native
    rank above sigma_slots (which would mean truncation already happened
    upstream)."""
    rng = np.random.default_rng(1)
    X = _spectrum_matrix(4, 2, 10, 6, rng)
    X[0, 8] = 0.75                              # dirty the padded band
    with pytest.raises(ValueError, match="NOT identically zero"):
        d1.pad_to_max_rank({"f": X}, {"f": 6}, 10, 2)
    clean = _spectrum_matrix(4, 2, 10, 6, rng)
    with pytest.raises(ValueError, match="cannot truncate"):
        d1.pad_to_max_rank({"f": clean}, {"f": 12}, 10, 2)


def test_zero_variance_guard_counts_padded_band_and_survives_standardize():
    """The guard names and COUNTS what familywise_standardize already
    does silently, and the paired invariant holds: padded columns leave
    standardization as exactly 0.0, not as a divide-by-zero blow-up."""
    rng = np.random.default_rng(2)
    sigma_slots, top_slots, n_blocks = 54, 24, 3
    short = _spectrum_matrix(8, n_blocks, sigma_slots, top_slots, rng)
    guard = d1.zero_variance_guard(
        short, sigma_slots=sigma_slots, top_slots=top_slots,
        n_blocks=n_blocks)
    assert guard["tol"] == d1.ZERO_VAR_TOL
    assert guard["n_features"] == n_blocks * sigma_slots
    # exactly 30 padded slots per block, all zero-variance, and nothing else
    assert guard["padded_region"]["n_features"] == 30 * n_blocks
    assert guard["padded_region"]["n_zero_variance"] == 30 * n_blocks
    assert guard["padded_region"]["all_zero_variance"] is True
    assert guard["padded_region"]["slot_band"] == [24, 54]
    assert guard["native_region"]["n_zero_variance"] == 0
    assert guard["n_zero_variance"] == 30 * n_blocks

    std = d1.familywise_standardize({"short": {"spectrum": short}})
    Z = std["short"]["spectrum"]
    pad_cols = d1._block_columns(n_blocks, sigma_slots, sigma_slots, (1,),
                                 top_slots, sigma_slots)
    assert np.array_equal(Z[:, pad_cols], np.zeros_like(Z[:, pad_cols]))


# ── Case 2 — the contrast is computed on BOTH views ─────────────────


def test_slice_top_slots_is_segment_aware_on_the_probe_layout():
    """The probe block is [sig; u.ravel(); v.ravel()] --- three slot-major
    segments, NOT an interleaved (slots x width) reshape. The top-k slice
    must therefore be taken per segment. Checked against an explicitly
    constructed expectation."""
    sigma_slots, proj_dim, k = 6, 3, 2
    layout = d1.h2_block_layout("probe", sigma_slots, proj_dim)
    assert layout["segment_widths"] == (1, proj_dim, proj_dim)
    # the same arithmetic as h2_features_for_run's probe_block_dim
    assert layout["block_dim"] == sigma_slots * (1 + 2 * proj_dim)

    rng = np.random.default_rng(3)
    sig = rng.standard_normal(sigma_slots)
    u = rng.standard_normal((sigma_slots, proj_dim))
    v = rng.standard_normal((sigma_slots, proj_dim))
    X = np.concatenate([sig, u.ravel(), v.ravel()])[None, :]
    got = d1.slice_top_slots(X, sigma_slots=sigma_slots, top_slots=k,
                             n_blocks=1, segment_widths=(1, proj_dim,
                                                         proj_dim))
    expect = np.concatenate([sig[:k], u[:k].ravel(), v[:k].ravel()])[None, :]
    assert np.allclose(got, expect)
    assert got.shape[1] == k * (1 + 2 * proj_dim)


def test_h2_transfer_reports_contrast_on_both_views():
    """Both views carry an independent decision; the flat keys remain the
    PADDED view, which is the one the clause makes mandatory."""
    rng = np.random.default_rng(4)
    sigma_slots, top_slots, n_blocks, proj_dim = 8, 5, 2, 3
    fams = ["short", "long"]
    features = {
        "short": {
            "spectrum": _spectrum_matrix(12, n_blocks, sigma_slots,
                                         top_slots, rng),
            "probe": _probe_matrix(12, n_blocks, sigma_slots, proj_dim,
                                   top_slots, rng)},
        "long": {
            "spectrum": _spectrum_matrix(12, n_blocks, sigma_slots,
                                         sigma_slots, rng),
            "probe": _probe_matrix(12, n_blocks, sigma_slots, proj_dim,
                                   sigma_slots, rng)},
    }
    labels = {f: np.repeat(np.arange(3), 4) for f in fams}
    layout = {"sigma_slots": sigma_slots, "top_slots": top_slots,
              "n_blocks": n_blocks,
              "segment_widths": {"spectrum": (1,),
                                 "probe": (1, proj_dim, proj_dim)}}
    out = d1.h2_transfer(features, labels, fams, chance=1 / 3, seed=0,
                         views=d1.H2_VIEWS, slot_layout=layout)

    widths = {"spectrum": 1, "probe": 1 + 2 * proj_dim}
    for rep in ("spectrum", "probe"):
        assert out[rep]["headline_view"] == "padded"
        assert set(out[rep]["views"]) == {"padded", "top_slots"}
        for view in ("padded", "top_slots"):
            v = out[rep]["views"][view]
            assert set(v["pairs"]) == {"short->long", "long->short"}
            assert set(v["family_probe"]) == {"raw", "family_standardized"}
            assert v["decision"]["variant"] == d1.H2_HEADLINE_VARIANT
            assert isinstance(v["decision"]["supported"], bool)
        assert (out[rep]["views"]["padded"]["dim"]
                == n_blocks * sigma_slots * widths[rep])
        assert (out[rep]["views"]["top_slots"]["dim"]
                == n_blocks * top_slots * widths[rep])
        # the flat keys ARE the padded view, so the headline never silently
        # becomes the truncated one
        assert out[rep]["dim"] == out[rep]["views"]["padded"]["dim"]
        assert (_canon(out[rep]["decision"])
                == _canon(out[rep]["views"]["padded"]["decision"]))


# ── Case 3 — truncation-only is refused ─────────────────────────────


def test_truncation_only_raises():
    """'Curing by top-24 truncation alone is prohibited' is enforced, not
    merely documented."""
    with pytest.raises(ValueError, match="TRUNCATION-ONLY REFUSED"):
        d1.assert_not_truncation_only(("top_slots",))
    with pytest.raises(ValueError, match="TRUNCATION-ONLY REFUSED"):
        d1.assert_not_truncation_only(())
    with pytest.raises(ValueError, match="unknown H2 view"):
        d1.assert_not_truncation_only(("padded", "head"))
    assert d1.assert_not_truncation_only(("padded",)) == ("padded",)
    assert d1.assert_not_truncation_only(d1.H2_VIEWS) == ("padded",
                                                          "top_slots")


def test_h2_transfer_refuses_truncation_only_views():
    rng = np.random.default_rng(5)
    fams = ["a", "b"]
    features = {f: {"spectrum": rng.standard_normal((12, 8)),
                    "probe": rng.standard_normal((12, 20))} for f in fams}
    labels = {f: np.repeat(np.arange(3), 4) for f in fams}
    with pytest.raises(ValueError) as exc:
        d1.h2_transfer(features, labels, fams, chance=1 / 3,
                       views=("top_slots",))
    assert "truncation alone is prohibited" in str(exc.value)
    with pytest.raises(ValueError, match="slot_layout is None"):
        d1.h2_transfer(features, labels, fams, chance=1 / 3,
                       views=d1.H2_VIEWS)


def test_sigma_slots_below_max_rank_is_truncation():
    """Asking for sigma_slots = 24 while a family is rank 54 is exactly
    the prohibited cure, and is refused at the pad step."""
    rng = np.random.default_rng(6)
    X = _spectrum_matrix(4, 2, 24, 24, rng)
    with pytest.raises(ValueError) as exc:
        d1.pad_to_max_rank({"long": X}, {"long": 54}, 24, 2)
    assert "TRUNCATION-ONLY REFUSED" in str(exc.value)


# ── Case 4 — occupancy, synthetic, hand-computed ────────────────────


def test_padded_slot_occupancy_known_answer():
    """Per block: 3 native slots of column-mass 1.0 and 2 padded slots of
    column-mass 0.5 each, one of which varies across runs.

      nonzero_mass_fraction     = (0.5 + 0.5) / (3*1.0 + 1.0) = 0.25
      nonzero_variance_fraction = 1 occupied / 2 padded slots = 0.5
    """
    sigma_slots, top_slots, n_blocks = 5, 3, 2
    block = np.array([
        [0.5, 0.5, 0.5, 0.25, 0.00],     # run 0
        [0.5, 0.5, 0.5, 0.25, 0.50],     # run 1
    ])
    X = np.concatenate([block] * n_blocks, axis=1)
    occ = d1.padded_slot_occupancy(X, sigma_slots=sigma_slots,
                                   top_slots=top_slots, n_blocks=n_blocks)
    assert occ["applicable"] is True
    assert occ["slot_band"] == [3, 5]
    assert occ["n_padded_slots_per_block"] == 2
    assert occ["overall"]["nonzero_mass_fraction"] == pytest.approx(0.25)
    assert occ["overall"]["nonzero_variance_fraction"] == pytest.approx(0.5)
    assert occ["overall"]["padded_mass"] == pytest.approx(2.0)
    assert occ["overall"]["total_mass"] == pytest.approx(8.0)
    assert len(occ["per_block"]) == n_blocks
    for blk in occ["per_block"]:
        assert blk["nonzero_mass_fraction"] == pytest.approx(0.25)
        assert blk["nonzero_variance_fraction"] == pytest.approx(0.5)
        assert blk["n_padded_slots_nonzero_variance"] == 1
    # both statistics are present and separately sourced (the Director's
    # mass fraction is primary; the variance fraction is its companion)
    assert "PRIMARY" in occ["definitions"]["nonzero_mass_fraction"]
    assert "COMPANION" in occ["definitions"]["nonzero_variance_fraction"]
    assert "slots 25-54" in occ["addition"]


def test_padded_slot_occupancy_at_the_pinned_h2s_geometry():
    """The Director's own shape: slots 25-54 of a rank-54 leg. 24 native
    slots at column-mass 1.0, 30 padded slots at 0.5 --->
    15 / 39 = 0.3846...  (both slot bands vary across runs, so the
    variance-occupancy is 1.0)."""
    sigma_slots, top_slots = 54, 24
    X = np.zeros((2, sigma_slots))
    X[0, :top_slots] = 1.0
    X[0, top_slots:] = 0.5
    occ = d1.padded_slot_occupancy(X, sigma_slots=sigma_slots,
                                   top_slots=top_slots, n_blocks=1)
    assert occ["slot_band_1_indexed"] == [25, 54]
    assert occ["overall"]["nonzero_mass_fraction"] == pytest.approx(15 / 39)
    assert occ["overall"]["nonzero_variance_fraction"] == pytest.approx(1.0)


def test_padded_slot_occupancy_detects_unused_slots():
    """The reading the addition exists to make reachable: an all-zero
    padded band reports 0.0 on both statistics --- 'the extra slots were
    never used', distinguishable from a null on the padded contrast."""
    rng = np.random.default_rng(7)
    sigma_slots, top_slots, n_blocks = 54, 24, 3
    short = _spectrum_matrix(8, n_blocks, sigma_slots, top_slots, rng)
    occ = d1.padded_slot_occupancy(short, sigma_slots=sigma_slots,
                                   top_slots=top_slots, n_blocks=n_blocks)
    assert occ["applicable"] is True        # the slots EXIST — they are 0.0,
    assert occ["overall"]["nonzero_mass_fraction"] == 0.0        # not None
    assert occ["overall"]["nonzero_variance_fraction"] == 0.0
    assert occ["overall"]["n_padded_slots"] == 30 * n_blocks
    assert occ["overall"]["n_padded_slots_nonzero_variance"] == 0


def test_padded_slot_occupancy_inapplicable_when_ranks_equal():
    """An EMPTY padded band reports None on BOTH fractions, never 0.0.

    The two readings are different claims and the addition exists to keep
    them apart: 0.0 means 'the extra slots exist and were never used'
    (the applicable case, above), None means 'there are no extra slots'.
    A 0.0 here would assert the first about a record that cannot support
    it. The symmetry is the point --- one fraction None and the other 0.0
    is the failure mode this pins shut."""
    rng = np.random.default_rng(8)
    X = _spectrum_matrix(6, 2, 24, 24, rng)
    occ = d1.padded_slot_occupancy(X, sigma_slots=24, top_slots=24,
                                   n_blocks=2)
    assert occ["applicable"] is False
    assert occ["n_padded_slots_per_block"] == 0
    assert occ["slot_band_1_indexed"] is None
    assert occ["overall"]["nonzero_variance_fraction"] is None
    assert occ["overall"]["nonzero_mass_fraction"] is None
    for blk in occ["per_block"]:
        assert blk["nonzero_mass_fraction"] is None
        assert blk["nonzero_variance_fraction"] is None
        assert blk["n_padded_slots"] == 0
    # the convention travels with the record, not just with the code
    assert "never 0.0" in occ["definitions"]["inapplicable"]
    # and it still serializes (None -> null), which is how it reaches the
    # results JSON analyze_bank writes
    assert json.loads(json.dumps(occ, default=d1._json_default))[
        "overall"]["nonzero_mass_fraction"] is None


# ── Case 5 — regression: equal-rank inputs are unchanged ────────────


def test_equal_ranks_reproduce_the_pre_clause_path():
    slots = d1.resolve_sigma_slots({"a": 24, "b": 24})
    assert slots["sigma_slots"] == 24 == slots["top_slots"]
    assert slots["pad_width"] == {"a": 0, "b": 0}
    assert slots["unequal"] is False


def test_slice_top_slots_identity_when_no_padding():
    rng = np.random.default_rng(9)
    X = _spectrum_matrix(5, 3, 24, 24, rng)
    got = d1.slice_top_slots(X, sigma_slots=24, top_slots=24, n_blocks=3)
    assert np.array_equal(got, X)
    assert got is not X                      # a copy, never a view
    Xp = _probe_matrix(4, 2, 6, 3, 6, rng)
    gotp = d1.slice_top_slots(Xp, sigma_slots=6, top_slots=6, n_blocks=2,
                              segment_widths=(1, 3, 3))
    assert np.array_equal(gotp, Xp)


def test_h2_transfer_default_output_is_byte_for_byte_unchanged():
    """The default views=("padded",) must reproduce the pre-clause output
    exactly --- same keys, same order, same values, and no new keys. This
    is what keeps the released Asset-1 d1_results.json H2 block identical.
    """
    rng = np.random.default_rng(10)
    fams = ["famA", "famB"]
    features = {f: {"spectrum": rng.standard_normal((12, 8)),
                    "probe": rng.standard_normal((12, 20))} for f in fams}
    labels = {f: np.repeat(np.arange(3), 4) for f in fams}
    base = d1.h2_transfer(features, labels, fams, chance=1 / 3, seed=0)
    explicit = d1.h2_transfer(features, labels, fams, chance=1 / 3, seed=0,
                              views=("padded",))
    assert _canon(base) == _canon(explicit)
    for rep in ("spectrum", "probe"):
        assert set(base[rep]) == {"dim", "role", "family_probe",
                                  "within_family_accuracy", "pairs",
                                  "decision"}
        assert "views" not in base[rep]
    assert base["spectrum"]["role"] == "PRIMARY"
    assert base["probe"]["role"] == "corroborating"


# ── Case 6 — the RECORD WRITE: analyze_bank's unequal-rank block ─────
#
# Pin condition 3 says "report". The pure functions above compute the
# statistics; the only place they are written INTO the results record is
# the unequal-rank block of analyze_bank, and that block is what a reader
# of d1_results.json actually sees. These two tests run the real
# analyze_bank end to end on a synthetic bank built in tmp_path --- an
# unequal-rank bank (ranks 4 and 6) and its equal-rank control (4 and 4)
# --- and assert the record, not the return value of a helper.


def _analyze(root, out, n_runs, tag):
    """analyze_bank at fixture scale: 'canonical' arm only, tiny
    permutation count, 2 depth bins, proj_dim 2. expected_total is the
    fixture's own run count (the override the docstring reserves for
    synthetic fixtures; the CLI never exposes it)."""
    return d1.analyze_bank(root, out, n_permutations=5, seed=0,
                           representation="canonical",
                           expected_total=n_runs, chunk_rows=4,
                           n_depth_bins=2, proj_dim=2, tag=tag)


def test_analyze_bank_writes_the_padding_record_on_unequal_ranks(tmp_path):
    """Ranks 4 and 6 --- the clause's own case --- carried all the way into
    the results record: sigma_slots = max = 6, top_slots = min = 4, BOTH
    views reported with the padded one as headline, no truncation, and the
    occupancy statistic present and in range for every (representation,
    family) cell."""
    root, n_runs = _mixed_rank_bank(tmp_path / "bank_unequal", (4, 6))
    out = tmp_path / "out_unequal"
    res = _analyze(root, out, n_runs, "_mixed")

    h2 = res["h2_cross_family"]
    assert h2["sigma_slots"] == 6
    rp = h2["rank_padding"]
    assert rp["sigma_slots"] == 6
    assert rp["top_slots"] == 4
    assert rp["native_ranks"] == {"synthfam0": 4, "synthfam1": 6}
    assert rp["pad_width"] == {"synthfam0": 2, "synthfam1": 0}
    assert rp["unequal"] is True
    assert rp["views_reported"] == ["padded", "top_slots"]
    assert rp["n_blocks"] == 4 * 2                 # 4 projections x 2 bins

    for rep in ("spectrum", "probe"):
        pc = rp["pad_check"][rep]
        assert pc["truncation_occurred"] is False
        assert pc["families"]["synthfam0"]["pad_width"] == 2
        assert pc["families"]["synthfam0"]["padded_region_all_zero"] is True
        assert pc["families"]["synthfam1"]["pad_width"] == 0
        zg = rp["zero_variance_guard"][rep]["synthfam0"]
        assert zg["padded_region"]["all_zero_variance"] is True
        # both views reported, padded is the headline, and the flat keys
        # remain the padded view (never silently the truncated one)
        assert h2[rep]["headline_view"] == "padded"
        assert set(h2[rep]["views"]) == {"padded", "top_slots"}
        assert h2[rep]["dim"] == h2[rep]["views"]["padded"]["dim"]
        assert (h2[rep]["views"]["top_slots"]["dim"]
                < h2[rep]["views"]["padded"]["dim"])

    occ = h2["padded_slot_occupancy"]
    assert occ["applicable"] is True
    for rep in ("spectrum", "probe"):
        for fam in ("synthfam0", "synthfam1"):
            o = occ[rep][fam]
            assert o["applicable"] is True
            assert o["slot_band"] == [4, 6]
            assert o["n_padded_slots_per_block"] == 2
            ov = o["overall"]
            assert 0.0 <= ov["nonzero_mass_fraction"] <= 1.0
            assert 0.0 <= ov["nonzero_variance_fraction"] <= 1.0
    # the reading the addition exists for, on a real record: the rank-4 leg
    # contributes an empty (0.0) padded band, the rank-6 leg a used one
    assert occ["spectrum"]["synthfam0"]["overall"][
        "nonzero_mass_fraction"] == 0.0
    assert occ["spectrum"]["synthfam1"]["overall"][
        "nonzero_mass_fraction"] > 0.0

    # and it survives the JSON write --- the artifact, not the return value
    written = json.loads(
        (out / "d1_results_mixed.json").read_text(encoding="utf-8"))
    wh2 = written["h2_cross_family"]
    assert wh2["rank_padding"]["views_reported"] == ["padded", "top_slots"]
    assert wh2["padded_slot_occupancy"]["applicable"] is True
    assert wh2["spectrum"]["headline_view"] == "padded"
    report = (out / "D1_REPORT_mixed.md").read_text(encoding="utf-8")
    assert "Rank padding and padded-slot occupancy" in report
    assert "nonzero mass fraction" in report


def test_analyze_bank_equal_ranks_keep_the_pre_clause_record(tmp_path):
    """The control: equal ranks (4 and 4) leave the record in its
    pre-clause shape --- one view, no 'views' key --- and the occupancy
    cells report applicable False with BOTH fractions None, so an
    equal-rank record never says 'the extra slots were never used'."""
    root, n_runs = _mixed_rank_bank(tmp_path / "bank_equal", (4, 4))
    out = tmp_path / "out_equal"
    res = _analyze(root, out, n_runs, "_equal")

    h2 = res["h2_cross_family"]
    assert h2["sigma_slots"] == 4
    rp = h2["rank_padding"]
    assert rp["sigma_slots"] == 4 == rp["top_slots"]
    assert rp["unequal"] is False
    assert rp["views_reported"] == ["padded"]
    for rep in ("spectrum", "probe"):
        assert "views" not in h2[rep]
        assert "headline_view" not in h2[rep]
        assert rp["pad_check"][rep]["truncation_occurred"] is False
        for fam in ("synthfam0", "synthfam1"):
            assert rp["pad_check"][rep]["families"][fam]["pad_width"] == 0

    occ = h2["padded_slot_occupancy"]
    assert occ["applicable"] is False
    for rep in ("spectrum", "probe"):
        for fam in ("synthfam0", "synthfam1"):
            o = occ[rep][fam]
            assert o["applicable"] is False
            assert o["n_padded_slots_per_block"] == 0
            assert o["overall"]["nonzero_mass_fraction"] is None
            assert o["overall"]["nonzero_variance_fraction"] is None
            assert all(b["nonzero_mass_fraction"] is None
                       and b["nonzero_variance_fraction"] is None
                       for b in o["per_block"])

    written = json.loads(
        (out / "d1_results_equal.json").read_text(encoding="utf-8"))
    wocc = written["h2_cross_family"]["padded_slot_occupancy"]
    assert wocc["spectrum"]["synthfam0"]["overall"][
        "nonzero_mass_fraction"] is None
    report = (out / "D1_REPORT_equal.md").read_text(encoding="utf-8")
    assert "occupancy statistic is inapplicable" in report
    assert "NOT 0.0" in report


def test_no_hardcoded_rank_literal_entered_the_module():
    """The pin comment claims 'NOTHING below hardcodes 54 or 24'
    (scripts/asset1_d1_identifiability.py, the H2-S constants block).
    Enforced by an AST walk over integer constants: 24 and 54 may appear
    in prose (the Director's 'top-24', 'slots 25-54') but never as a
    literal the code computes with. Every slot count is read at runtime
    from r["config"]["rank"]."""
    import ast
    src = Path(d1.__file__).read_text(encoding="utf-8")
    hits = [(n.lineno, n.value) for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Constant) and isinstance(n.value, int)
            and not isinstance(n.value, bool) and n.value in (24, 54)]
    assert hits == [], f"hard-coded rank literal(s) in the module: {hits}"
