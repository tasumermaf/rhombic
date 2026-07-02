#!/usr/bin/env python3
"""Produce 4K hackathon video — Nous Research Hermes Agent Hackathon.

Renders ~2700 frames at 3840x2160 from real training data + benchmarks,
encodes via ffmpeg (h264_nvenc → libx264 fallback), deploys to website/.
"""

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Resolution ──────────────────────────────────────────────────────
W, H = 3840, 2160
FPS = 30

# ── Palette (Executive Register) ───────────────────────────────────
NUIT = (8, 6, 32)
GOLD = (220, 201, 100)
GOLD_DIM = (110, 100, 50)
AVORIO = (240, 232, 208)
FCC_RED = (179, 68, 68)
CUBIC_BLUE = (61, 61, 107)
AZURE = (92, 143, 175)
GREEN = (74, 140, 92)
VIOLA = (123, 63, 140)

NUIT_HEX = '#%02x%02x%02x' % NUIT

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEBSITE = ROOT / "website"

# ── Font cache ─────────────────────────────────────────────────────
_fc = {}

def get_font(size, style="sans"):
    key = (size, style)
    if key in _fc:
        return _fc[key]
    cands = {
        "sans":  ["C:/Windows/Fonts/bahnschrift.ttf", "C:/Windows/Fonts/segoeui.ttf"],
        "serif": ["C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/times.ttf"],
        "mono":  ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"],
        "bold":  ["C:/Windows/Fonts/bahnschrift.ttf"],
    }
    for p in cands.get(style, cands["sans"]):
        try:
            f = ImageFont.truetype(p, size)
            _fc[key] = f
            return f
        except OSError:
            continue
    f = ImageFont.load_default()
    _fc[key] = f
    return f


# ── Utilities ──────────────────────────────────────────────────────

def sc(v):
    """Scale value for 3840-wide reference."""
    return int(v * W / 3840)

def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3

def fade(frame, start, fi, hold, fo):
    """Opacity 0-1: fade_in → hold → fade_out."""
    if frame < start:
        return 0.0
    f = frame - start
    if f < fi:
        return f / fi if fi else 1.0
    f -= fi
    if f < hold:
        return 1.0
    f -= hold
    if f < fo:
        return 1.0 - f / fo if fo else 0.0
    return 0.0

def blend(color, alpha):
    """Blend color toward NUIT by alpha."""
    return tuple(int(c * alpha + NUIT[i] * (1 - alpha)) for i, c in enumerate(color))

def centered(draw, text, y, font, color):
    bb = draw.textbbox((0, 0), text, font=font)
    x = (W - (bb[2] - bb[0])) // 2
    draw.text((x, y), text, fill=color, font=font)

def rule(draw, y, w=0.25):
    span = int(W * w)
    x1 = (W - span) // 2
    draw.line([(x1, y), (x1 + span, y)], fill=GOLD_DIM, width=max(1, sc(3)))

def particles(img, frame, n=40, seed=777):
    rng = np.random.RandomState(seed)
    draw = ImageDraw.Draw(img)
    r = max(1, sc(2))
    for i in range(n):
        bx, by = rng.random() * W, rng.random() * H
        dx = math.sin(frame * 0.01 + i * 1.7) * sc(30)
        dy = math.cos(frame * 0.013 + i * 2.3) * sc(20)
        px, py = int((bx + dx) % W), int((by + dy) % H)
        br = (math.sin(frame * 0.05 + i * 3.1) * 0.5 + 0.5) ** 2 * 0.4
        c = blend(GOLD, br)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=c)

def mpl_to_pil(fig):
    """Convert matplotlib figure to PIL Image and close."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer('RGBA', fig.canvas.get_width_height(), buf)
    img = img.convert('RGB')
    plt.close(fig)
    return img

def make_fig(w_frac=1.0, h_frac=0.78):
    """Create a styled matplotlib figure on NUIT background."""
    fw, fh = W * w_frac / 200, H * h_frac / 200
    fig, ax = plt.subplots(1, 1, figsize=(fw, fh), dpi=200)
    fig.patch.set_facecolor(NUIT_HEX)
    ax.set_facecolor(NUIT_HEX)
    return fig, ax

def style_ax(ax, title, ylabel, ylim=None, log=False):
    """Common axis styling."""
    ax.set_title(title, color='#DCC964', fontsize=22, pad=20)
    ax.set_ylabel(ylabel, color='white', fontsize=14)
    ax.tick_params(colors='white', labelsize=12)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('bottom', 'left'):
        ax.spines[s].set_color('#555')
    if log:
        ax.set_yscale('log')
    if ylim:
        ax.set_ylim(*ylim)


# ── Data Loading ───────────────────────────────────────────────────

def load_json(name):
    for p in [RESULTS / name, RESULTS / "exp3a-overfit" / name]:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"Cannot find {name} in results/")

def load_temporal():
    """Load C-001, C-002, C-003 → dict of name: [(step, ratio), ...]"""
    ds = {}

    c001 = load_json("C-001_temporal_emergence.json")
    ds["C-001"] = [(d["step"], d["ratio"]) for d in c001]

    for tag in ("C-002_temporal_emergence_extended.json",
                "C-003_temporal_emergence.json"):
        raw = load_json(tag)
        entries = raw.get("checkpoints", raw) if isinstance(raw, dict) else raw
        name = tag.split("_")[0]
        ds[name] = [(d["step"], d.get("co_cross_ratio", d.get("ratio", 0)))
                     for d in entries]
    return ds


# ── Scene Renderers ────────────────────────────────────────────────
# Each: render_xxx(f, n, **kw) → PIL.Image
# f = frame within scene, n = total frames in scene

def render_title(f, n):
    img = Image.new("RGB", (W, H), NUIT)
    particles(img, f)
    draw = ImageDraw.Draw(img)
    a = fade(f, 0, 30, n - 60, 30)
    centered(draw, "rhombic-agent", sc(700), get_font(sc(140), "sans"), blend(GOLD, a))
    rule(draw, sc(880), 0.2)
    centered(draw, "Keep Your Cube, Add Six Bridges.", sc(940),
             get_font(sc(56), "sans"), blend(AVORIO, a))
    centered(draw, "@NousResearch  #HermesAgentHackathon", sc(1300),
             get_font(sc(36), "mono"), blend(GOLD_DIM, a))
    return img


def render_default(f, n, bench=None):
    img = Image.new("RGB", (W, H), NUIT)
    prog = ease(min(1.0, f / (n * 0.6)))

    cf, ff = bench.cubic_fiedler, bench.fcc_fiedler
    ratio = ff / cf if cf else 0

    fig, ax = make_fig()
    bars = ax.bar(
        ["Cubic\n(6 neighbors)", "FCC / Rhombic\n(12 neighbors)"],
        [cf * prog, ff * prog],
        color=['#3D3D6B', '#B34444'], edgecolor='black', linewidth=0.5, width=0.5)
    style_ax(ax, "Lattice Connectivity Comparison",
             "Algebraic Connectivity (Fiedler)", ylim=(0, ff * 1.3))

    if prog > 0.3:
        ra = min(1.0, (prog - 0.3) / 0.3)
        rv = ratio * min(1.0, (prog - 0.3) / 0.4)
        ax.annotate(f"{rv:.2f}×", xy=(1, ff * prog), xytext=(1.35, ff * 0.8),
                    fontsize=28, color='#DCC964', alpha=ra, ha='center',
                    arrowprops=dict(arrowstyle='->', color='#DCC964', alpha=ra))
    fig.tight_layout(pad=2)
    plot = mpl_to_pil(fig).resize((W, int(H * 0.75)), Image.LANCZOS)
    img.paste(plot, (0, int(H * 0.05)))

    draw = ImageDraw.Draw(img)
    ta = fade(f, int(n * 0.15), 30, n, 0)
    centered(draw, "Every computation defaults to 6 neighbors.",
             int(H * 0.88), get_font(sc(48), "sans"), blend(AVORIO, ta))
    return img


def render_amplification(f, n):
    img = Image.new("RGB", (W, H), NUIT)
    draw = ImageDraw.Draw(img)
    prog = ease(min(1.0, f / (n * 0.7)))

    labels = ["Uniform", "Random", "Power-law", "Corpus"]
    ratios = [2.31, 3.23, 4.10, 6.11]
    colors = [CUBIC_BLUE, AZURE, VIOLA, FCC_RED]
    a = fade(f, 0, 20, n, 0)

    centered(draw, "Weight Distribution Amplifies Advantage",
             sc(120), get_font(sc(64), "sans"), blend(GOLD, a))

    bar_x, bar_mw, bar_h, gap = sc(500), sc(2400), sc(100), sc(160)
    y0 = sc(400)
    mx = max(ratios) * 1.15

    for i, (lbl, rat, col) in enumerate(zip(labels, ratios, colors)):
        y = y0 + i * (bar_h + gap)
        delay = i * 0.15
        bp = ease(max(0, min(1, (prog - delay) / max(0.01, 1 - delay))))

        draw.text((sc(80), y + sc(20)), lbl, fill=blend(AVORIO, a),
                  font=get_font(sc(40), "sans"))
        draw.rectangle([bar_x, y, bar_x + bar_mw, y + bar_h], fill=(20, 18, 45))
        fw = int(bar_mw * (rat / mx) * bp)
        if fw > 0:
            draw.rectangle([bar_x, y, bar_x + fw, y + bar_h], fill=col)
        draw.text((bar_x + fw + sc(20), y + sc(15)),
                  f"{rat * bp:.2f}×", fill=blend(GOLD, a), font=get_font(sc(48), "mono"))

    ca = fade(f, int(n * 0.4), 30, n, 0)
    cp = ease(max(0, (f / n - 0.4) / 0.5))
    cv = 2.31 + (6.11 - 2.31) * cp
    centered(draw, f"FCC / Cubic Ratio:  {cv:.1f}×",
             y0 + 4 * (bar_h + gap) + sc(60),
             get_font(sc(72), "mono"), blend(GOLD, ca))
    centered(draw, "Heterogeneous weights amplify the advantage.",
             int(H * 0.88), get_font(sc(44), "sans"), blend(AVORIO, a))
    return img


def render_bridge(f, n, temporal=None):
    """THE HERO SHOT: animated temporal emergence time-series."""
    img = Image.new("RGB", (W, H), NUIT)
    prog = ease(min(1.0, f / (n * 0.85)))

    all_steps = [s for data in temporal.values() for s, _ in data]
    max_step = max(all_steps) if all_steps else 7000
    cur_max = max(100, int(max_step * prog))

    fig, ax = make_fig(h_frac=0.78)
    lc = {"C-001": ('#4A8FB5', "TinyLlama 1.1B"),
          "C-002": ('#B34444', "Qwen 7B (geometric)"),
          "C-003": ('#7B3F8C', "Qwen 7B (corpus)")}

    for name, data in temporal.items():
        steps = [s for s, r in data if s <= cur_max and s > 0]
        rats =  [r for s, r in data if s <= cur_max and s > 0]
        if steps:
            col, lbl = lc.get(name, ('white', name))
            ax.plot(steps, rats, color=col, linewidth=2.5, label=lbl, alpha=0.9)

    style_ax(ax, "Temporal Emergence: Axis Alignment",
             "Co-planar / Cross-planar Ratio", ylim=(0.5, 100000), log=True)
    ax.set_xlabel("Training Step", color='white', fontsize=14)
    ax.set_xlim(0, max_step * 1.05)
    ax.legend(loc='upper left', fontsize=11, facecolor=NUIT_HEX,
              edgecolor='#555', labelcolor='white')

    if cur_max >= 200:
        la = min(1.0, (cur_max - 200) / 500)
        ax.axvline(x=200, color='#FFD700', ls='--', alpha=la * 0.7, lw=1.5)
        if la > 0.5:
            ax.text(280, 50000, "Lock-in", color='#FFD700', fontsize=13,
                    alpha=la, style='italic')
    ax.axhline(y=1, color='#555', ls=':', alpha=0.3, lw=1)

    fig.tight_layout(pad=2)
    plot = mpl_to_pil(fig).resize((W, int(H * 0.78)), Image.LANCZOS)
    img.paste(plot, (0, 0))

    draw = ImageDraw.Draw(img)
    ta = fade(f, int(n * 0.5), 30, n, 0)
    centered(draw, "100% block-diagonal.  0% in controls.",
             int(H * 0.84), get_font(sc(48), "sans"), blend(AVORIO, ta))
    if prog > 0.6:
        cp = ease((prog - 0.6) / 0.4)
        centered(draw, f"Peak ratio: {int(70404 * cp):,}:1",
                 int(H * 0.92), get_font(sc(56), "mono"), blend(GOLD, ta))
    return img


def render_bifurcation(f, n):
    img = Image.new("RGB", (W, H), NUIT)
    prog = ease(min(1.0, f / (n * 0.65)))

    channels = [3, 4, 6, 8]
    fvals = [0.092, 0.088, 0.000089, 0.085]
    bcolors = ['#4A8FB5', '#4A8FB5', '#B34444', '#4A8FB5']

    fig, ax = make_fig()
    ax.bar([f"n={c}" for c in channels], [v * prog for v in fvals],
           color=bcolors, edgecolor='black', linewidth=0.5, width=0.5)
    style_ax(ax, "Channel Ablation: The n=6 Bifurcation",
             "Bridge Fiedler Value", ylim=(1e-5, 0.5), log=True)

    if prog > 0.5:
        aa = min(1.0, (prog - 0.5) / 0.3)
        rv = (0.09 / 0.000089) * min(1.0, (prog - 0.5) / 0.3)
        target_y = max(fvals[2] * prog, 1e-5)
        ax.annotate(f"{int(rv):,}× lower", xy=(2, target_y), xytext=(2.6, 0.01),
                    fontsize=20, color='#FFD700', alpha=aa, ha='center',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#FFD700', alpha=aa, lw=2))
    fig.tight_layout(pad=2)
    plot = mpl_to_pil(fig).resize((W, int(H * 0.78)), Image.LANCZOS)
    img.paste(plot, (0, int(H * 0.02)))

    draw = ImageDraw.Draw(img)
    ta = fade(f, int(n * 0.3), 30, n, 0)
    centered(draw, "1,020\u00d7 bifurcation at the rhombic channel count.",
             int(H * 0.88), get_font(sc(48), "sans"), blend(AVORIO, ta))
    return img


def render_holly(f, n):
    img = Image.new("RGB", (W, H), NUIT)
    particles(img, f, n=25)
    draw = ImageDraw.Draw(img)
    prog = ease(min(1.0, f / (n * 0.6)))
    a = fade(f, 0, 20, n - 30, 30)

    centered(draw, "Holly Battery: CogVideoX-5B (14B params)",
             sc(140), get_font(sc(64), "sans"), blend(GOLD, a))
    rule(draw, sc(260), 0.35)

    cols = [
        ("Standard LoRA", CUBIC_BLUE,
         {"Loss": 1.6137, "VRAM (GB)": 75.75, "Time (min)": 1625}),
        ("TeLoRA", GREEN,
         {"Loss": 1.5517, "VRAM (GB)": 66.60, "Time (min)": 1527}),
        ("Corpus", VIOLA,
         {"Loss": 1.6453, "VRAM (GB)": 66.60, "Time (min)": 1528}),
    ]
    metrics = ["Loss", "VRAM (GB)", "Time (min)"]
    col_w = W // 3
    y0 = sc(400)
    hf, vf, lf, uf = (get_font(sc(44), "sans"), get_font(sc(56), "mono"),
                       get_font(sc(36), "sans"), get_font(sc(32), "mono"))

    for ci, (name, color, vals) in enumerate(cols):
        cx = col_w * ci + col_w // 2
        # Header
        bb = draw.textbbox((0, 0), name, font=hf)
        draw.text((cx - (bb[2] - bb[0]) // 2, y0), name,
                  fill=blend(color, a), font=hf)
        # Winner highlight
        if ci == 1:
            rx1, rx2 = col_w * ci + sc(40), col_w * (ci + 1) - sc(40)
            draw.rectangle([rx1, y0 - sc(20), rx2, y0 + sc(600)],
                           outline=GREEN, width=sc(3))

        for mi, met in enumerate(metrics):
            y = y0 + sc(120) + mi * sc(160)
            target = vals[met]
            cur = target * prog
            # Label
            bb = draw.textbbox((0, 0), met, font=lf)
            draw.text((cx - (bb[2] - bb[0]) // 2, y), met,
                      fill=blend(AVORIO, a * 0.6), font=lf)
            # Value
            vs = f"{int(cur)}" if "min" in met else f"{cur:.4f}" if "Loss" in met else f"{cur:.2f}"
            vc = blend(GOLD if ci == 1 else AVORIO, a)
            bb = draw.textbbox((0, 0), vs, font=vf)
            draw.text((cx - (bb[2] - bb[0]) // 2, y + sc(45)), vs, fill=vc, font=vf)

    sa = fade(f, int(n * 0.5), 30, n, 0)
    centered(draw, "3.8% better loss  \u00b7  9.15 GB less VRAM  \u00b7  6% faster",
             int(H * 0.88), get_font(sc(48), "sans"), blend(GOLD, sa))
    return img


def render_punchline(f, n):
    img = Image.new("RGB", (W, H), NUIT)
    particles(img, f, n=30)
    draw = ImageDraw.Draw(img)
    big, med = get_font(sc(120), "sans"), get_font(sc(72), "sans")

    beats = [(0, 90, "100% vs 0%."),
             (90, 190, "70,404 : 1"),
             (190, n, "The bridge tells us what it learned.")]
    for s, e, txt in beats:
        if s <= f < e:
            lf = f - s
            a = fade(lf, 0, 15, e - s - 30, 15)
            font = big if len(txt) < 20 else med
            centered(draw, txt, int(H * 0.45), font, blend(GOLD, a))
            break
    return img


def render_close(f, n):
    img = Image.new("RGB", (W, H), NUIT)
    particles(img, f, n=30)
    draw = ImageDraw.Draw(img)
    a = fade(f, 0, 30, n - 60, 30)
    centered(draw, "pip install rhombic", sc(650),
             get_font(sc(64), "mono"), blend(GREEN, a))
    rule(draw, sc(800), 0.2)
    centered(draw, "255 tests  \u00b7  3 papers  \u00b7  MPL-2.0", sc(860),
             get_font(sc(44), "sans"), blend(AVORIO, a))
    centered(draw, "github.com/tasumermaf/rhombic", sc(1050),
             get_font(sc(36), "mono"), blend(AZURE, a))
    centered(draw, "Timothy Bielec  \u00b7  Promptcrafted LLC", sc(1200),
             get_font(sc(44), "sans"), blend(GOLD_DIM, a))
    centered(draw, "@NousResearch  #HermesAgentHackathon", sc(1350),
             get_font(sc(36), "mono"), blend(GOLD_DIM, a))
    return img


# ── Encoding ───────────────────────────────────────────────────────

def encode(frame_dir, out):
    pat = str(frame_dir / "frame_%05d.png").replace("\\", "/")
    base = ["ffmpeg", "-y", "-framerate", str(FPS), "-start_number", "1",
            "-i", pat, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    nvenc = base + ["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "16",
                    "-profile:v", "high", str(out)]
    x264 = base + ["-c:v", "libx264", "-crf", "18", "-preset", "slow",
                   "-profile:v", "high", str(out)]

    print(f"  Encoding -> {out.name}")
    r = subprocess.run(nvenc, capture_output=True, text=True)
    if r.returncode != 0:
        print("  NVENC failed, trying libx264...")
        r = subprocess.run(x264, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ENCODE FAILED:\n{r.stderr[-500:]}")
        return False
    return True


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HACKATHON FINAL — 4K Video Production")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/4] Loading data...")
    temporal = load_temporal()
    for k, v in temporal.items():
        print(f"  {k}: {len(v)} points, max ratio {max(r for _, r in v):.0f}")

    print("  Running benchmark (n=125)...")
    sys.path.insert(0, str(ROOT))
    from rhombic.benchmark import run_benchmark
    bench = run_benchmark(125)
    ratio = bench.fcc_fiedler / bench.cubic_fiedler if bench.cubic_fiedler else 0
    print(f"  Fiedler ratio: {ratio:.2f}×")

    # ── Scene table ──
    scenes = [
        ("title",         render_title,         4,   {}),
        ("default",       render_default,       14,  {"bench": bench}),
        ("amplification", render_amplification, 14,  {}),
        ("bridge",        render_bridge,        20,  {"temporal": temporal}),
        ("bifurcation",   render_bifurcation,   10,  {}),
        ("holly",         render_holly,         10,  {}),
        ("punchline",     render_punchline,     10,  {}),
        ("close",         render_close,         8,   {}),
    ]
    total = sum(d * FPS for _, _, d, _ in scenes)
    print(f"  {total} frames, {sum(d for _, _, d, _ in scenes)}s")

    # ── Render frames ──
    fdir = ROOT / "assets" / "video" / "final_frames"
    fdir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/4] Rendering {total} frames at {W}×{H}...")
    t0 = time.time()
    fnum = 1

    for sname, renderer, dur, kw in scenes:
        nf = dur * FPS
        print(f"  {sname} ({nf}f)...", end="", flush=True)
        st = time.time()
        for i in range(nf):
            img = renderer(i, nf, **kw)
            img.save(fdir / f"frame_{fnum:05d}.png")
            fnum += 1
            if (i + 1) % 60 == 0:
                print(f" {i+1}", end="", flush=True)
        print(f" [{time.time()-st:.0f}s]")

    print(f"  Total render: {time.time()-t0:.0f}s ({total/(time.time()-t0):.1f} fps)")

    # ── Encode ──
    out_mp4 = ROOT / "assets" / "video" / "hackathon_final.mp4"
    print(f"\n[3/4] Encoding...")
    if not encode(fdir, out_mp4):
        sys.exit(1)
    sz = out_mp4.stat().st_size / (1024 * 1024)
    print(f"  {sz:.1f} MB")

    # ── Deploy ──
    demo = WEBSITE / "demo.mp4"
    print(f"\n[4/4] Copying to {demo}...")
    shutil.copy2(out_mp4, demo)
    print(f"  Done ({demo.stat().st_size / (1024*1024):.1f} MB)")
    if sz > 95:
        print("  ⚠ File > 95 MB — may exceed GitHub Pages limit")

    print("\n" + "=" * 60)
    print("COMPLETE.")
    print(f"  Play:   {out_mp4}")
    print(f"  Deploy: cd rhombic && git add website/demo.mp4 && git commit && git push")
    print("=" * 60)


if __name__ == "__main__":
    main()
