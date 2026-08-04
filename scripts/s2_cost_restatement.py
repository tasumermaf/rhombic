# -*- coding: utf-8 -*-
"""Restate both registered cards' cost tables against MEASURED rates.

H2-at-scale: from results/s2-timing-pilots/*/run_*/TIMING.md (9 runs).
Granularity: from results/asset1-bank/campaign.log RUN_END dur= values,
             which DO carry per-run durations even though bank_manifest.json
             does not (the Director's Part 3 established the manifest gap;
             the campaign log closes it with zero GPU time).

Every number printed here is computed from an artifact at run time.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

R = Path(r"C:\falco\rhombic")
S2 = R / "results" / "s2-timing-pilots"

# ---------- H2: measured per-family means ----------
runs = defaultdict(list)
for t in sorted(S2.glob("*/run_*/TIMING.md")):
    fam = t.parent.parent.name
    txt = t.read_text(encoding="utf-8")
    m = re.search(r"^wall_clock_min\s*=\s*([\d.]+)", txt, re.M)
    v = re.search(r"^peak_vram_gb\s*=\s*([\d.]+)", txt, re.M)
    if m:
        runs[fam].append((float(m.group(1)), float(v.group(1)) if v else None))

# Draft §3 table: (params_B, reps, runs, est_min_per_run, est_gpu_days)
DRAFT = {
    "gemma2-2b":   (2.6, 20, 120, 79,  6.6),
    "qwen2.5-3b":  (3.1, 20, 120, 93,  7.8),
    "qwen2.5-7b":  (7.6, 10,  60, 230, 9.6),
    "llama3.1-8b": (8.0, 10,  60, 243, 10.1),
}

print("=== H2 §3 RESTATEMENT (measured) ===")
meas = {}
tot_meas_days = tot_est_days_measfam = 0.0
for fam in ["gemma2-2b", "qwen2.5-3b", "qwen2.5-7b", "llama3.1-8b"]:
    pB, reps, nruns, est_min, est_days = DRAFT[fam]
    rs = runs.get(fam, [])
    if rs:
        mean = sum(x[0] for x in rs) / len(rs)
        vram = rs[0][1]
        days = nruns * mean / 1440.0
        meas[fam] = (pB, mean, days)
        tot_meas_days += days
        tot_est_days_measfam += est_days
        print(f"{fam:12s} n={len(rs)} mean={mean:7.2f} min/run (est {est_min})  "
              f"delta={100*(mean-est_min)/est_min:+6.2f}%  "
              f"{nruns} runs -> {days:6.3f} GPU-days (est {est_days})  VRAM {vram} GB")
    else:
        print(f"{fam:12s} UNMEASURED (license-gated); draft est {est_min} min/run, {est_days} d")

# Least-squares fit over measured families -> projection for the gated one
xs = [meas[f][0] for f in meas]
ys = [meas[f][1] for f in meas]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
sxx = sum((x - mx) ** 2 for x in xs)
sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
b = sxy / sxx
a = my - b * mx
llama_proj = a + b * 8.0
llama_days = 60 * llama_proj / 1440.0
print()
print(f"FIT (measured only, n={n}): min/run = {a:.3f} + {b:.4f} x params_B")
print(f"  per-B at each point: " + " · ".join(f"{f}={meas[f][1]/meas[f][0]:.2f}" for f in meas)
      + "   (falls with scale -> draft's flat ~30 min/B was conservative)")
print(f"LLAMA-3.1-8B PROJECTION (NOT measured) = {llama_proj:.2f} min/run -> {llama_days:.3f} GPU-days")
print()
print(f"TOTAL 3 MEASURED FAMILIES = {tot_meas_days:.3f} GPU-days  (draft est for same 3 = {tot_est_days_measfam:.1f}, "
      f"{100*(tot_meas_days-tot_est_days_measfam)/tot_est_days_measfam:+.2f}%)")
print(f"TOTAL 4 FAMILIES (3 measured + 1 projected) = {tot_meas_days + llama_days:.3f} GPU-days  (draft est 34)")
print(f"TOTAL 3-FAMILY BANK (drop Llama per S1 option) = {tot_meas_days:.3f} GPU-days = {tot_meas_days*24:.1f} GPU-hours")
print(f"S2 PILOT COST ITSELF = {sum(x[0] for rs in runs.values() for x in rs)/60:.2f} GPU-hours over 9 runs")

# ---------- Granularity: measured llama3.2-1b rate from campaign.log ----------
print()
print("=== GRANULARITY COST BASIS (measured, zero GPU time) ===")
log = (R / "results" / "asset1-bank" / "campaign.log").read_text(encoding="utf-8", errors="ignore")
man = json.loads((R / "results" / "asset1-bank" / "bank_manifest.json").read_text(encoding="utf-8"))
fam_of = {}
for run in man["runs"]:
    fam_of[run["run_index"]] = run.get("family_short") or run.get("family")

ends = re.findall(r"RUN_END idx=(\d+) status=(\w+) rc=(-?\d+) dur=(\d+)s", log)
by_fam = defaultdict(list)
dupes = defaultdict(list)
for idx, status, rc, dur in ends:
    if status != "COMPLETE":
        continue
    i = int(idx)
    dupes[i].append(int(dur))
    by_fam[fam_of.get(i, "?")].append((i, int(dur)))

# For duplicated indices (A1 restart), the LAST COMPLETE is the bs4xga4 cohort run.
final = defaultdict(dict)
for fam, lst in by_fam.items():
    for i, d in lst:
        final[fam][i] = d  # later overwrites earlier -> final cohort

print(f"RUN_END lines total = {len(ends)}; COMPLETE = {sum(1 for e in ends if e[1]=='COMPLETE')}; "
      f"indices with >1 COMPLETE (A1 restart) = {sum(1 for v in dupes.values() if len(v)>1)}")
for fam in sorted(final):
    ds = list(final[fam].values())
    mean_min = sum(ds) / len(ds) / 60.0
    print(f"{fam:16s} n={len(ds):3d}  mean={mean_min:6.2f} min/run  "
          f"(min {min(ds)/60:.1f} / max {max(ds)/60:.1f})")

llama_key = [f for f in final if "llama" in f.lower()]
if llama_key:
    k = llama_key[0]
    ds = list(final[k].values())
    mm = sum(ds) / len(ds) / 60.0
    print()
    print(f"GRANULARITY ANCHOR: {k} measured mean = {mm:.2f} min/run "
          f"(design carried '~30 min/run llama [ESTIMATE]')")
    for label, nruns in [("L1 12-class (20 seeds x 12)", 240), ("L2 24-class (10 x 24)", 240),
                         ("L3 ~48-class (5 x 48)", 240), ("Arm B squad ladder", 144),
                         ("D7 split-pool control", 10)]:
        print(f"  {label:32s} {nruns:4d} runs -> {nruns*mm/1440.0:6.3f} GPU-days")
    total = (240 * 3 + 144 + 10) * mm / 1440.0
    print(f"  {'TOTAL (L1+L2+L3+ArmB+D7)':32s} {240*3+144+10:4d} runs -> {total:6.3f} GPU-days "
          f"(design est ~14-15 + ArmB)")
