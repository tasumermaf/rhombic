"""
Deterministic figure generation for the XR-001 externalization-robustness note.

Reads probe-level records from
  results/XR-001-externalization-pilot/results.json
and re-derives every plotted number (per-cell corruption rates, Wilson
intervals, regime x class composition) from the raw probes, so the figures
cannot drift from the paper's tables.

Run:
  C:\\miniconda3\\envs\\falco\\python.exe paper/figures-xr001/make_figures.py

Outputs (PDF + PNG) into paper/figures-xr001/:
  fig1-corruption-by-regime
  fig2-regime-class-breakdown

No randomness, no network, no seeds. matplotlib + stdlib only.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# House style (rhombic/visualize.py, 8-Law Weave palette)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

CUBIC_COLOR = "#3D3D6B"    # Fall of Neutral Events (11) — the unexamined default
FCC_COLOR = "#B34444"      # Geometric Essence (67) — the revealed structure
SYNC_ORANGE = "#FF8C42"    # Synchronicity (89)
ARROW_AZURE = "#4A90E2"    # Arrow of Complexity (17)
JUSTICE_GREEN = "#52A352"  # Sole Atom (19)
KAOS_GOLD = "#FFD700"      # Kaos (23)
GRID_SILVER = "#C0C0C0"    # Time Matrix (29)

# Regime encoding. Blue / red / gray is colorblind-safe; markers add a
# redundant (non-color) channel. Order = ceiling -> structured -> prose.
REGIME_ORDER = ["R0", "R2", "R1"]
REGIME_LABEL = {
    "R0": "R0  no compaction (ceiling)",
    "R2": "R2  structured typed blocks",
    "R1": "R1  prose summary",
}
REGIME_COLOR = {"R0": GRID_SILVER, "R2": FCC_COLOR, "R1": CUBIC_COLOR}
REGIME_MARKER = {"R0": "o", "R2": "s", "R1": "D"}

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "results" / "XR-001-externalization-pilot" / "results.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval as (low, high) percentages."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100.0 * (center - half), 100.0 * (center + half))


def load_probes() -> list[dict]:
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)["probes"]


def corruption_by_cell(probes: list[dict]) -> dict[tuple[str, str], tuple[int, int]]:
    cell: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for pr in probes:
        key = (pr["model"], pr["regime"])
        cell[key][1] += 1
        if pr["class"] != "correct":
            cell[key][0] += 1
    return {k: (v[0], v[1]) for k, v in cell.items()}


def class_by_regime(probes: list[dict]) -> dict[str, dict[str, int]]:
    tally: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pr in probes:
        tally[pr["regime"]][pr["class"]] += 1
    return tally


# ---------------------------------------------------------------------------
# Figure 1 — per-model numeric-corruption rate by regime, 95% Wilson CI
# ---------------------------------------------------------------------------
def make_fig1(probes: list[dict]) -> None:
    cells = corruption_by_cell(probes)
    models = ["gemma3:4b", "qwen3:14b", "qwen3-coder:30b"]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = list(range(len(models)))
    span = 0.62
    offsets = {r: (i - (len(REGIME_ORDER) - 1) / 2) * (span / (len(REGIME_ORDER) - 1))
               for i, r in enumerate(REGIME_ORDER)}

    for regime in REGIME_ORDER:
        xs, ys, lo_err, hi_err = [], [], [], []
        for xi, model in zip(x, models):
            k, n = cells[(model, regime)]
            rate = 100.0 * k / n
            lo, hi = wilson(k, n)
            xs.append(xi + offsets[regime])
            ys.append(rate)
            lo_err.append(max(0.0, rate - lo))
            hi_err.append(max(0.0, hi - rate))
        ax.errorbar(
            xs, ys, yerr=[lo_err, hi_err], fmt=REGIME_MARKER[regime],
            color=REGIME_COLOR[regime], markersize=8, markeredgecolor="black",
            markeredgewidth=0.6, capsize=4, elinewidth=1.4, linestyle="none",
            label=REGIME_LABEL[regime], zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Numeric-corruption rate (%)")
    ax.set_xlabel("Model")
    ax.set_title("Numeric corruption under cascaded compaction (n = 120 probes / cell)")
    ax.set_ylim(-2, 55)
    ax.yaxis.grid(True, color=GRID_SILVER, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = HERE / f"fig1-corruption-by-regime.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — composition of the corrupt probes, by regime (pooled)
# ---------------------------------------------------------------------------
def make_fig2(probes: list[dict]) -> None:
    tally = class_by_regime(probes)
    # Error classes only (correct excluded): the shape of the corruption.
    classes = ["off_by_one", "conflation", "other_wrong", "omission"]
    class_label = {
        "off_by_one": "off-by-one (signature)",
        "conflation": "conflation (signature)",
        "other_wrong": "other wrong",
        "omission": "omission",
    }
    class_color = {
        "off_by_one": SYNC_ORANGE,
        "conflation": KAOS_GOLD,
        "other_wrong": ARROW_AZURE,
        "omission": GRID_SILVER,
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    order = ["R0", "R2", "R1"]
    x = list(range(len(order)))
    bottoms = [0.0] * len(order)
    for cls in classes:
        heights = [tally[r].get(cls, 0) for r in order]
        ax.bar(x, heights, bottom=bottoms, width=0.6, color=class_color[cls],
               edgecolor="black", linewidth=0.5, label=class_label[cls], zorder=3)
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    for xi, r in zip(x, order):
        total = sum(tally[r].get(c, 0) for c in classes)
        sig = tally[r].get("off_by_one", 0) + tally[r].get("conflation", 0)
        ax.annotate(f"{total} corrupt\n(sig {sig})", xy=(xi, bottoms[xi]),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL[r] for r in order], fontsize=9)
    ax.set_ylabel("Corrupt probes (of 360 / regime)")
    ax.set_title("Composition of corruption by regime (pooled across models)")
    ax.set_ylim(0, max(bottoms) * 1.25)
    ax.yaxis.grid(True, color=GRID_SILVER, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = HERE / f"fig2-regime-class-breakdown.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    probes = load_probes()
    assert len(probes) == 1080, f"expected 1080 probes, got {len(probes)}"
    make_fig1(probes)
    make_fig2(probes)
    print("done.")


if __name__ == "__main__":
    main()
