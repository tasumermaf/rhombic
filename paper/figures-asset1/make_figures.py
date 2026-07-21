"""
Deterministic figure generation for the Asset-1 paper ("The Gauge Is the
Obstacle"). Reads ONLY the per-item verification bundle at
  results/asset1-delivery-verify/
— the same files the Director regraded — and re-derives every plotted number
from raw records (confusion counts, per-eval rows, per-pair labels, per-run
pairs), so the figures cannot drift from the paper's tables and the Director
can check figure numbers against the same bundle.

Run:
  C:\\miniconda3\\envs\\falco\\python.exe paper/figures-asset1/make_figures.py

Outputs (PDF + PNG) into paper/figures-asset1/:
  F1-h1-confusion            raw vs canonical confusion matrices, both families
  F2-h2-transfer             raw -> standardized transfer dumbbells + probe collapse
  F3-d2-penalties            per-eval swap-penalty strips by kind
  F4-d3-auc                  D3 AUC dot-interval plot, full vs distance, + margins
  F5-d3-coverage             realized task-pair cell coverage per family
  F6-daux-scatter            dev_mean vs final_gap, 480 runs
  F7-daux-forest             pilot -> bank correlation forest

No randomness, no network, no seeds. matplotlib + stdlib only.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# House style (rhombic/visualize.py, 8-Law Weave palette; conventions per
# figures-xr001/make_figures.py). CUBIC indigo = "the unexamined default" ->
# raw representations; FCC red = "the revealed structure" -> canonical /
# standardized. Family identity carries marker shape as a redundant channel.
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

CUBIC_COLOR = "#3D3D6B"    # raw / unexamined default
FCC_COLOR = "#B34444"      # canonical, standardized / revealed structure
GRID_SILVER = "#C0C0C0"
ARROW_AZURE = "#4A90E2"

FAM_ORDER = ["qwen2.5-1.5b", "llama3.2-1b"]
FAM_LABEL = {"qwen2.5-1.5b": "Qwen2.5-1.5B", "llama3.2-1b": "Llama-3.2-1B"}
FAM_COLOR = {"qwen2.5-1.5b": CUBIC_COLOR, "llama3.2-1b": FCC_COLOR}
FAM_MARKER = {"qwen2.5-1.5b": "o", "llama3.2-1b": "^"}

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "results" / "asset1-delivery-verify"
OUT = Path(__file__).resolve().parent

# Pilot deviation<->gap correlation, quoted from the pre-registration record
# (DIRECTOR_DECISIONS_2026-07-06.md / the experiment card). A documented
# constant, not computed here.
PILOT_R = 0.888


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# ---------------------------------------------------------------------------
# F1 — H1 confusion matrices, raw vs canonical, both families
# ---------------------------------------------------------------------------
def f1(d1: dict) -> None:
    tasks = d1["tasks"]
    ramp = LinearSegmentedColormap.from_list("indigo", ["#FFFFFF", CUBIC_COLOR])
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.8))
    for row, fam in enumerate(FAM_ORDER):
        reps = d1["families"][fam]["representations"]
        for col, rep in enumerate(("raw", "canonical")):
            cm = reps[rep]["confusion_matrix"]["rows_true_cols_pred"]
            acc = reps[rep]["loo_accuracy"]
            ax = axes[row][col]
            ax.imshow(cm, cmap=ramp, vmin=0, vmax=40, aspect="equal")
            for i in range(6):
                for j in range(6):
                    v = cm[i][j]
                    if v:
                        dark = v > 20
                        ax.text(j, i, str(v), ha="center", va="center",
                                fontsize=8.5,
                                color="white" if dark else "#333333")
            ax.set_xticks(range(6), tasks, rotation=45, ha="right", fontsize=8.5)
            ax.set_yticks(range(6), tasks, fontsize=8.5)
            ax.set_title(f"{FAM_LABEL[fam]} — {rep}   (LOO acc {acc:.4f})",
                         fontsize=11)
            if col == 0:
                ax.set_ylabel("true task")
            if row == 1:
                ax.set_xlabel("predicted task")
    fig.suptitle("H1: task identity is illegible raw, perfect once the "
                 "GL(r) gauge is removed", y=0.995, fontsize=13)
    fig.tight_layout()
    save(fig, "F1-h1-confusion")


# ---------------------------------------------------------------------------
# F2 — H2 transfer dumbbells (raw -> standardized) + family-probe collapse
# ---------------------------------------------------------------------------
def f2(d1: dict) -> None:
    h2 = d1["h2_cross_family"]
    chance = h2["spectrum"]["decision"]["chance"]
    rows = []
    for rep in ("spectrum", "probe"):
        for direction, p in h2[rep]["pairs"].items():
            arrow = direction.replace("qwen2.5-1.5b", "Qwen").replace(
                "llama3.2-1b", "Llama").replace("->", " → ")
            rows.append((f"{rep}: {arrow}",
                         p["raw"]["accuracy"],
                         p["family_standardized"]["accuracy"]))

    fig, (ax, axp) = plt.subplots(
        1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [3, 1.5]})
    ys = range(len(rows))[::-1]
    for y, (label, raw, std) in zip(ys, rows):
        ax.plot([raw, std], [y, y], color=GRID_SILVER, lw=2, zorder=1)
        ax.scatter([raw], [y], s=70, color=GRID_SILVER, edgecolor="#666666",
                   zorder=2, label=None)
        ax.scatter([std], [y], s=80, color=FCC_COLOR, edgecolor="#5A1F1F",
                   zorder=3)
        ax.text(std + 0.018, y, f"{std:.4f}", va="center", fontsize=9,
                color="#333333")
    ax.axvline(chance, color="#888888", ls="--", lw=1)
    ax.text(chance + 0.008, -0.48, f"chance {chance:.4f}",
            fontsize=8.5, color="#666666", va="top")
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.9, len(rows) - 0.4)
    ax.set_xlabel("cross-family task-transfer accuracy")
    ax.set_title("Transfer: raw (gray) → family-standardized (red)")
    ax.spines[["top", "right"]].set_visible(False)

    probes = []
    for rep in ("spectrum", "probe"):
        fp = h2[rep]["family_probe"]
        probes.append((rep, fp["raw"]["accuracy"],
                       fp["family_standardized"]["accuracy"]))
    ysp = range(len(probes))[::-1]
    for y, (rep, raw, std) in zip(ysp, probes):
        axp.plot([std, raw], [y, y], color=GRID_SILVER, lw=2, zorder=1)
        axp.scatter([raw], [y], s=70, color=CUBIC_COLOR, edgecolor="#22223F",
                    zorder=3)
        axp.scatter([std], [y], s=70, color=FCC_COLOR, edgecolor="#5A1F1F",
                    zorder=3)
        axp.text(raw, y + 0.22, f"{raw:.2f}", va="bottom", ha="center",
                 fontsize=9, color="#333333")
        axp.text(std, y + 0.22, f"{std:.4f}", va="bottom", ha="center",
                 fontsize=9, color="#333333")
    axp.axvline(0.5, color="#888888", ls="--", lw=1)
    axp.text(0.508, -0.48, "chance 0.5", fontsize=8.5, color="#666666",
             va="top")
    axp.set_yticks(list(ysp), [p[0] for p in probes], fontsize=9.5)
    axp.set_xlim(-0.09, 1.09)
    axp.set_ylim(-0.9, len(probes) - 0.25)
    axp.set_xlabel("family-identity probe accuracy")
    axp.set_title("Triviality control: probe collapses")
    axp.spines[["top", "right"]].set_visible(False)
    fig.suptitle("H2: raw transfer-at-chance was a family-scale artifact; "
                 "standardization reveals 0.74–0.78 transfer", y=1.04)
    fig.tight_layout()
    save(fig, "F2-h2-transfer")


# ---------------------------------------------------------------------------
# F3 — D2 per-eval swap penalties by kind (strip plot, re-derived from rows)
# ---------------------------------------------------------------------------
KIND_ORDER = ["cross_seed", "cross_task", "cross_task_magnitude",
              "cross_task_topology", "identity", "permuted_deviation",
              "permuted"]


def f3() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for fi, fam in enumerate(FAM_ORDER):
        d2 = json.loads((BUNDLE / f"d2_results_{fam}.json").read_text(
            encoding="utf-8"))
        rows = d2["rows"]
        native = {(r["task_recipient"], r["recipient_run_index"],
                   r["pair_slot"]): r["val_loss"]
                  for r in rows if r["kind"] == "native"}
        pens = defaultdict(list)
        for r in rows:
            if r["kind"] == "native":
                continue
            key = (r["task_recipient"], r["recipient_run_index"],
                   r["pair_slot"])
            pens[r["kind"]].append(r["val_loss"] - native[key])
        for ki, kind in enumerate(KIND_ORDER):
            vals = pens[kind]
            # deterministic horizontal spread (no RNG): rank-based offsets
            n = len(vals)
            xs = [ki + (fi - 0.5) * 0.36 + ((i / max(n - 1, 1)) - 0.5) * 0.22
                  for i in range(n)]
            ax.scatter(xs, sorted(vals), s=14, alpha=0.55,
                       color=FAM_COLOR[fam], marker=FAM_MARKER[fam],
                       linewidths=0, zorder=2)
    ax.axhline(0, color="#888888", lw=0.8, zorder=1)
    ax.set_xticks(range(len(KIND_ORDER)),
                  [k.replace("_", "\n") for k in KIND_ORDER], fontsize=9)
    ax.set_ylabel("val-loss penalty vs native (nats)")
    ax.set_title("D2: every backbone-preserving swap costs ~0; only full "
                 "permutation costs")
    ax.legend(handles=[
        Line2D([], [], color=FAM_COLOR[f], marker=FAM_MARKER[f], ls="",
               markersize=7, label=FAM_LABEL[f]) for f in FAM_ORDER],
        frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "F3-d2-penalties")


# ---------------------------------------------------------------------------
# F4 — D3 AUC dot-interval plot (group-aware headline) + margin CIs
# ---------------------------------------------------------------------------
def f4() -> None:
    d3 = json.loads((BUNDLE / "d3_report.json").read_text(encoding="utf-8"))
    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(9.6, 3.2), gridspec_kw={"width_ratios": [2.4, 1.6]})
    rows, labels = [], []
    for fam in FAM_ORDER:
        g = d3["per_family"][fam]
        rows.append((g["auc_full"], g["auc_full_ci"], FAM_COLOR[fam], "full"))
        labels.append(f"{FAM_LABEL[fam]} — full (weight-only)")
        rows.append((g["auc_distance"], g["auc_distance_ci"], GRID_SILVER,
                     "distance"))
        labels.append(f"{FAM_LABEL[fam]} — distance-only")
    ys = range(len(rows))[::-1]
    for y, (auc, ci, color, _kind) in zip(ys, rows):
        ax.plot(ci, [y, y], color=color, lw=2.5, solid_capstyle="round")
        ax.scatter([auc], [y], s=60, color=color, edgecolor="#444444",
                   zorder=3)
        ax.text(ci[1] + 0.012, y, f"{auc:.3f}", va="center", fontsize=9,
                color="#333333")
    ax.axvline(0.5, color="#888888", ls="--", lw=1)
    ax.text(0.503, 0.0, "chance", fontsize=8.5, color="#666666")
    ax.set_yticks(list(ys), labels, fontsize=9.5)
    ax.set_xlim(0.4, 1.09)
    ax.set_xlabel("merge-conflict AUC (group-aware CV, 95% CI)")
    ax.set_title("D3: weight-only features vs distance baseline")
    ax.spines[["top", "right"]].set_visible(False)

    ysm = range(len(FAM_ORDER))[::-1]
    for y, fam in zip(ysm, FAM_ORDER):
        g = d3["per_family"][fam]
        ci = g["auc_diff_ci"]
        axm.plot(ci, [y, y], color=FAM_COLOR[fam], lw=2.5,
                 solid_capstyle="round")
        axm.scatter([g["auc_diff"]], [y], s=60, color=FAM_COLOR[fam],
                    edgecolor="#444444", zorder=3)
        axm.text(ci[1] + 0.012, y, f"+{g['auc_diff']:.3f}", va="center",
                 fontsize=9, color="#333333")
    axm.axvline(0, color="#888888", ls="--", lw=1)
    axm.set_yticks(list(ysm), [FAM_LABEL[f] for f in FAM_ORDER], fontsize=9.5)
    axm.set_xlim(-0.05, 0.62)
    axm.set_xlabel("full − distance margin (95% CI)")
    axm.set_title("Margin: both CIs exclude 0")
    axm.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "F4-d3-auc")


# ---------------------------------------------------------------------------
# F5 — D3 realized task-pair cell coverage (uniform sampler, amendment)
# ---------------------------------------------------------------------------
def f5(tasks: list[str]) -> None:
    pairs = json.loads((BUNDLE / "d3_pairs.json").read_text(
        encoding="utf-8"))["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, fam in zip(axes, FAM_ORDER):
        counts = defaultdict(int)
        for p in (q for q in pairs if q["family_short"] == fam):
            a, b = sorted((p["task_a"], p["task_b"]))
            counts[(a, b)] += 1
        ramp = LinearSegmentedColormap.from_list(
            "fam", ["#FFFFFF", FAM_COLOR[fam]])
        grid = [[None] * 6 for _ in range(6)]
        vmax = max(counts.values())
        for i, a in enumerate(tasks):
            for j, b in enumerate(tasks):
                if j < i:
                    continue
                grid[i][j] = counts.get((a, b), 0)
        import numpy as np  # stdlib-adjacent; already a rhombic dependency

        arr = np.array([[v if v is not None else np.nan for v in row]
                        for row in grid], dtype=float)
        masked = np.ma.masked_invalid(arr)
        ax.imshow(masked, cmap=ramp, vmin=0, vmax=vmax, aspect="equal")
        same = sum(v for (a, b), v in counts.items() if a == b)
        total = sum(counts.values())
        for i in range(6):
            for j in range(6):
                if grid[i][j] is not None:
                    v = grid[i][j]
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=9,
                            color="white" if v > 0.6 * vmax else "#333333")
        ax.set_xticks(range(6), tasks, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(range(6), tasks, fontsize=8.5)
        ax.set_title(f"{FAM_LABEL[fam]} — {total} pairs, "
                     f"{same} same-task ({100 * same / total:.1f}%)",
                     fontsize=11)
    fig.suptitle("D3 realized pair coverage (uniform vertex-disjoint sampler; "
                 "expected ~16% same-task)", y=0.99)
    fig.tight_layout()
    save(fig, "F5-d3-coverage")


# ---------------------------------------------------------------------------
# F6 — D-aux scatter: dev_mean vs final_gap, 480 runs
# ---------------------------------------------------------------------------
def f6(daux_report: dict) -> None:
    rows = list(csv.DictReader(
        (BUNDLE / "daux_run_table.csv").open(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for fam in FAM_ORDER:
        xs = [float(r["dev_mean"]) for r in rows if r["family_short"] == fam]
        ys = [float(r["final_gap"]) for r in rows if r["family_short"] == fam]
        ax.scatter(xs, ys, s=18, alpha=0.55, color=FAM_COLOR[fam],
                   marker=FAM_MARKER[fam], linewidths=0,
                   label=f"{FAM_LABEL[fam]} (n={len(xs)})")
    pooled = daux_report["primary"]["cells"]["pooled"]
    ci = pooled["pearson_ci"]
    ax.set_xlabel("bridge deviation  dev_mean = meanₘ ‖bridge − I‖_F")
    ax.set_ylabel("generalization gap  (val − train, step 2000)")
    ax.set_title(f"D-aux: pooled r = {pooled['pearson_r']:.3f} "
                 f"[{ci[0]:.3f}, {ci[1]:.3f}]  (pilot r = {PILOT_R})")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "F6-daux-scatter")


# ---------------------------------------------------------------------------
# F7 — D-aux forest: pilot -> bank shrink, with guard cells
# ---------------------------------------------------------------------------
def f7(daux_report: dict) -> None:
    cells = daux_report["primary"]["cells"]
    spec = [
        ("pilot (pre-registration record)", PILOT_R, None, GRID_SILVER),
        ("bank pooled (n=480)", None, "pooled", "#333333"),
        ("Qwen2.5-1.5B (n=240)", None, "family:qwen2.5-1.5b",
         FAM_COLOR["qwen2.5-1.5b"]),
        ("Llama-3.2-1B (n=240)", None, "family:llama3.2-1b",
         FAM_COLOR["llama3.2-1b"]),
        ("within math — Llama (n=40)", None, "task:llama3.2-1b/math",
         FAM_COLOR["llama3.2-1b"]),
        ("within math — Qwen (n=40)", None, "task:qwen2.5-1.5b/math",
         FAM_COLOR["qwen2.5-1.5b"]),
        ("within xsum — Qwen (n=40)", None, "task:qwen2.5-1.5b/xsum",
         FAM_COLOR["qwen2.5-1.5b"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ys = range(len(spec))[::-1]
    for y, (label, fixed, key, color) in zip(ys, spec):
        if fixed is not None:
            ax.scatter([fixed], [y], s=70, color=color, edgecolor="#444444",
                       marker="D", zorder=3)
            ax.text(fixed + 0.02, y, f"{fixed:.3f} (no CI on record)",
                    va="center", fontsize=8.5, color="#666666")
        else:
            c = cells[key]
            ci = c["pearson_ci"]
            ax.plot(ci, [y, y], color=color, lw=2.5, solid_capstyle="round")
            ax.scatter([c["pearson_r"]], [y], s=55, color=color,
                       edgecolor="#444444", zorder=3)
            ax.text(ci[1] + 0.02, y, f"{c['pearson_r']:.3f}", va="center",
                    fontsize=9, color="#333333")
    ax.axvline(0, color="#888888", ls="--", lw=1)
    ax.set_yticks(list(ys), [s[0] for s in spec], fontsize=9.5)
    ax.set_xlim(-0.65, 1.05)
    ax.set_xlabel("Pearson r, dev_mean vs final_gap (95% CI)")
    ax.set_title("D-aux: the pilot correlation shrinks honestly at bank scale")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "F7-daux-forest")


def main() -> None:
    d1 = json.loads((BUNDLE / "d1_results.json").read_text(encoding="utf-8"))
    daux = json.loads((BUNDLE / "daux_report.json").read_text(
        encoding="utf-8"))
    print("[figures-asset1] generating from", BUNDLE)
    f1(d1)
    f2(d1)
    f3()
    f4()
    f5(d1["tasks"])
    f6(daux)
    f7(daux)
    print("[figures-asset1] done")


if __name__ == "__main__":
    main()
