#!/usr/bin/env python3
"""
TASUMER MAF — Production Demo Video (4K Motion Graphics).

Replaces the terminal-simulation demo with clean, cinematic motion graphics.
Uses the 8-Law Weave palette on NUIT background. All 3840x2160.

Reads pre-captured data from assets/video/captures/.

Output: assets/video/demo_frames/frame_NNNNN.png

Usage:
    python scripts/render_demo_production.py
    python scripts/render_demo_production.py --preview   # 1280x720
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

if sys.platform == "win32" and not hasattr(sys.stdout, '_rhombic_wrapped'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stdout._rhombic_wrapped = True
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ===================================================================
# Configuration
# ===================================================================

PREVIEW = "--preview" in sys.argv
W, H = (1280, 720) if PREVIEW else (3840, 2160)
FPS = 30

BASE = Path(__file__).resolve().parent.parent
CAPTURES = BASE / "assets" / "video" / "captures"
FRAME_DIR = BASE / "assets" / "video" / "demo_frames"

# ===================================================================
# Palette
# ===================================================================

NUIT = (8, 6, 32)
NUIT_F = np.array([8, 6, 32], dtype=np.float32)
GOLD = (220, 201, 100)
GOLD_DIM = (160, 145, 70)
AVORIO = (240, 232, 208)
DIM = (130, 126, 138)
FCC_RED = (179, 68, 68)
CUBIC_BLUE = (61, 61, 107)
AZURE = (92, 143, 175)
VIOLA = (123, 63, 140)
GREEN = (74, 140, 92)


# ===================================================================
# Font System
# ===================================================================

_font_cache = {}

def get_font(size: int, style: str = "sans") -> ImageFont.FreeTypeFont:
    key = (size, style)
    if key in _font_cache:
        return _font_cache[key]

    candidates = {
        "sans": [
            "C:/Windows/Fonts/bahnschrift.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ],
        "serif": [
            "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/times.ttf",
        ],
        "mono": [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
        ],
        "bold": [
            "C:/Windows/Fonts/bahnschrift.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ],
    }

    for path in candidates.get(style, candidates["sans"]):
        if os.path.exists(path):
            try:
                f = ImageFont.truetype(path, size)
                _font_cache[key] = f
                return f
            except Exception:
                continue
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


# ===================================================================
# Drawing Primitives
# ===================================================================

def scale(v: int) -> int:
    """Scale a design value from 3840-base to current resolution."""
    return int(v * W / 3840)


def make_frame() -> Image.Image:
    """Create a blank NUIT frame."""
    return Image.new("RGB", (W, H), NUIT)


def draw_centered_text(draw, text, y, font, color, img_w=None):
    """Draw horizontally centered text. Returns (x, y, text_w, text_h)."""
    if img_w is None:
        img_w = W
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img_w - tw) // 2
    draw.text((x, y), text, fill=color, font=font)
    return x, y, tw, th


def draw_gold_rule(draw, y, width_frac=0.25):
    """Draw a centered horizontal gold line."""
    span = int(W * width_frac)
    x1 = (W - span) // 2
    x2 = x1 + span
    lw = max(1, scale(3))
    draw.line([(x1, y), (x2, y)], fill=GOLD_DIM, width=lw)


def draw_progress_bar(draw, x, y, w, h, progress, fill_color, bg_color=(30, 28, 55)):
    """Animated progress/data bar."""
    draw.rectangle([x, y, x + w, y + h], fill=bg_color)
    fill_w = int(w * min(1.0, max(0, progress)))
    if fill_w > 0:
        draw.rectangle([x, y, x + fill_w, y + h], fill=fill_color)


def apply_fade(img_arr, opacity):
    """Fade image to NUIT background. img_arr is float32 [H,W,3]."""
    bg = NUIT_F[None, None, :]
    return (img_arr * opacity + bg * (1 - opacity)).astype(np.uint8)


def particle_field(img, frame, n=80, seed=123):
    """Subtle floating particle dots for visual texture."""
    rng = np.random.RandomState(seed)
    draw = ImageDraw.Draw(img)
    for i in range(n):
        # Slow floating motion
        base_x = rng.random() * W
        base_y = rng.random() * H
        drift_x = math.sin(frame * 0.01 + i * 1.7) * scale(30)
        drift_y = math.cos(frame * 0.013 + i * 2.3) * scale(20)
        px = int((base_x + drift_x) % W)
        py = int((base_y + drift_y) % H)
        # Twinkle
        brightness = (math.sin(frame * 0.05 + i * 3.1) * 0.5 + 0.5) ** 2
        alpha = int(40 * brightness)
        r = max(1, scale(2))
        color = (GOLD[0], GOLD[1], GOLD[2], alpha)
        # Direct pixel set for speed
        if 0 <= px < W and 0 <= py < H:
            draw.ellipse([px - r, py - r, px + r, py + r],
                         fill=(GOLD[0], GOLD[1], GOLD[2]))
    return img


# ===================================================================
# Animation Curves
# ===================================================================

def ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3

def ease_in_out(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

def fade_in_out(frame, start, fade_in_dur, hold_dur, fade_out_dur):
    """Returns opacity 0-1 for a fade-in, hold, fade-out cycle."""
    f = frame - start
    total = fade_in_dur + hold_dur + fade_out_dur
    if f < 0 or f >= total:
        return 0.0
    if f < fade_in_dur:
        return ease_out_cubic(f / max(1, fade_in_dur))
    if f < fade_in_dur + hold_dur:
        return 1.0
    remaining = f - fade_in_dur - hold_dur
    return 1.0 - ease_out_cubic(remaining / max(1, fade_out_dur))


def count_up(t, target, duration=2.0):
    """Animated number counter with ease-out."""
    progress = min(1.0, max(0, t / duration))
    eased = 1 - (1 - progress) ** 3
    return target * eased


# ===================================================================
# Scene Renderers
# ===================================================================

def render_scene_thesis(frame_in_scene, total_frames):
    """Scene 1: The thesis statement."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, frame_in_scene)

    title_font = get_font(scale(110), "serif")
    sub_font = get_font(scale(52), "sans")
    body_font = get_font(scale(38), "sans")

    # Phase timing within scene
    t = frame_in_scene / FPS

    # Main statement fades in
    op1 = fade_in_out(frame_in_scene, 0, 20, total_frames - 35, 15)
    if op1 > 0:
        alpha = int(255 * op1)
        color = (AVORIO[0], AVORIO[1], AVORIO[2])

        # "Every computation"
        draw_centered_text(draw, "Every computation", int(H * 0.30),
                           title_font, color)
        draw_centered_text(draw, "defaults to cubic.", int(H * 0.44),
                           title_font, color)

    # Supporting detail fades in later
    op2 = fade_in_out(frame_in_scene, 35, 20, total_frames - 70, 15)
    if op2 > 0:
        alpha2 = int(200 * op2)
        detail_color = (DIM[0], DIM[1], DIM[2])

        draw_centered_text(draw, "6 neighbors.  3 directions.  1 assumption.",
                           int(H * 0.62), sub_font, detail_color)

        draw_gold_rule(draw, int(H * 0.72), 0.15)

        draw_centered_text(draw, "What if the default is leaving performance on the table?",
                           int(H * 0.77), body_font, GOLD)

    return img


def render_scene_numbers(frame_in_scene, total_frames, data):
    """Scene 2: Key numbers with animated counters."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, frame_in_scene + 500)

    t = frame_in_scene / FPS

    big_font = get_font(scale(160), "bold")
    unit_font = get_font(scale(48), "sans")
    label_font = get_font(scale(36), "sans")
    header_font = get_font(scale(60), "serif")

    # Section header
    op_header = fade_in_out(frame_in_scene, 0, 15, total_frames - 25, 10)
    if op_header > 0:
        draw_centered_text(draw, "The Numbers", int(H * 0.08),
                           header_font, GOLD)
        draw_gold_rule(draw, int(H * 0.15), 0.12)

    # Three columns of data
    col_positions = [W * 0.18, W * 0.50, W * 0.82]

    # Column 1: 2.3x (Paper 1)
    op1 = fade_in_out(frame_in_scene, 15, 20, total_frames - 50, 15)
    if op1 > 0:
        val = count_up(max(0, t - 0.5), 2.3, 1.5)
        cx = int(col_positions[0])
        text = f"{val:.1f}x"
        bbox = draw.textbbox((0, 0), text, font=big_font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, int(H * 0.30)), text, fill=AVORIO, font=big_font)
        draw.text((cx - scale(120), int(H * 0.52)), "algebraic", fill=DIM, font=unit_font)
        draw.text((cx - scale(120), int(H * 0.58)), "connectivity", fill=DIM, font=unit_font)

        draw_progress_bar(draw, cx - scale(120), int(H * 0.66),
                          scale(240), scale(8), op1 * 0.38, CUBIC_BLUE)

        draw.text((cx - scale(80), int(H * 0.72)), "Paper 1 — uniform",
                   fill=GOLD_DIM, font=label_font)

    # Column 2: 6.1x (Paper 2)
    op2 = fade_in_out(frame_in_scene, 45, 20, total_frames - 80, 15)
    if op2 > 0:
        val = count_up(max(0, t - 1.5), 6.1, 2.0)
        cx = int(col_positions[1])
        text = f"{val:.1f}x"
        bbox = draw.textbbox((0, 0), text, font=big_font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, int(H * 0.30)), text, fill=FCC_RED, font=big_font)
        draw.text((cx - scale(140), int(H * 0.52)), "under structured", fill=DIM, font=unit_font)
        draw.text((cx - scale(140), int(H * 0.58)), "weights", fill=DIM, font=unit_font)

        draw_progress_bar(draw, cx - scale(140), int(H * 0.66),
                          scale(280), scale(8), op2 * 1.0, FCC_RED)

        draw.text((cx - scale(130), int(H * 0.72)), "Paper 2 — direction-weighted",
                   fill=GOLD_DIM, font=label_font)

    # Column 3: p = 0.000025
    op3 = fade_in_out(frame_in_scene, 75, 20, total_frames - 110, 15)
    if op3 > 0:
        cx = int(col_positions[2])
        # Show the p-value formatted
        p_display = "p < 0.001"
        bbox = draw.textbbox((0, 0), p_display, font=big_font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, int(H * 0.32)), p_display, fill=GREEN, font=get_font(scale(120), "bold"))
        draw.text((cx - scale(140), int(H * 0.52)), "prime-vertex", fill=DIM, font=unit_font)
        draw.text((cx - scale(140), int(H * 0.58)), "mapping", fill=DIM, font=unit_font)
        draw.text((cx - scale(130), int(H * 0.72)), "256 tests — not noise",
                   fill=GOLD_DIM, font=label_font)

    # Bottom bar: "The advantage amplifies under stress."
    op_bottom = fade_in_out(frame_in_scene, 100, 20, total_frames - 135, 15)
    if op_bottom > 0:
        draw_gold_rule(draw, int(H * 0.84), 0.35)
        draw_centered_text(draw, "The advantage amplifies under stress. Not noise. Structure.",
                           int(H * 0.88), get_font(scale(42), "serif"), GOLD)

    return img


def render_scene_agent(frame_in_scene, total_frames, data):
    """Scene 3: The agent — 9 tools, live experiments."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, frame_in_scene + 1000)

    header_font = get_font(scale(60), "serif")
    tool_font = get_font(scale(32), "mono")
    desc_font = get_font(scale(30), "sans")
    stat_font = get_font(scale(90), "bold")

    op_header = fade_in_out(frame_in_scene, 0, 15, total_frames - 25, 10)
    if op_header > 0:
        draw_centered_text(draw, "The Agent", int(H * 0.06), header_font, GOLD)
        draw_gold_rule(draw, int(H * 0.13), 0.10)

    # Tool grid (3x3)
    tools = [
        ("lattice_compare", "Compare topologies"),
        ("fiedler_ratio", "Algebraic connectivity"),
        ("direction_weights", "Weight distributions"),
        ("spectral_analysis", "Laplacian spectrum"),
        ("prime_vertex_map", "Prime-to-vertex search"),
        ("permutation_control", "Statistical control"),
        ("explain_mechanism", "Natural language"),
        ("visualize_rd", "3D visualization"),
        ("visualize_amplification", "Chart generation"),
    ]

    for idx, (name, desc) in enumerate(tools):
        row, col = idx // 3, idx % 3
        delay = 10 + idx * 6
        op = fade_in_out(frame_in_scene, delay, 12, total_frames - delay - 20, 10)
        if op <= 0:
            continue

        bx = int(W * 0.08 + col * W * 0.30)
        by = int(H * 0.20 + row * H * 0.22)
        bw = int(W * 0.26)
        bh = int(H * 0.17)

        # Card background
        alpha = int(op * 40)
        draw.rectangle([bx, by, bx + bw, by + bh], fill=(20, 18, 48))
        draw.rectangle([bx, by, bx + bw, by + scale(4)], fill=GOLD_DIM)

        # Tool name
        draw.text((bx + scale(16), by + scale(16)), name,
                  fill=AZURE, font=tool_font)
        # Description
        draw.text((bx + scale(16), by + scale(56)), desc,
                  fill=DIM, font=desc_font)

    # Stats row at bottom
    op_stats = fade_in_out(frame_in_scene, 80, 20, total_frames - 115, 15)
    if op_stats > 0:
        stats = [("9", "tools"), ("3", "skills"), ("256", "tests"), ("3", "papers")]
        for i, (num, label) in enumerate(stats):
            cx = int(W * (0.15 + i * 0.23))
            cy = int(H * 0.88)
            bbox = draw.textbbox((0, 0), num, font=stat_font)
            tw = bbox[2] - bbox[0]
            draw.text((cx - tw // 2, cy - scale(50)), num, fill=AVORIO, font=stat_font)
            bbox2 = draw.textbbox((0, 0), label, font=desc_font)
            tw2 = bbox2[2] - bbox2[0]
            draw.text((cx - tw2 // 2, cy + scale(40)), label, fill=DIM, font=desc_font)

    return img


def render_scene_vision(frame_in_scene, total_frames):
    """Scene 4: The vision — keep your cube, add six bridges."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, frame_in_scene + 1500)

    title_font = get_font(scale(90), "serif")
    body_font = get_font(scale(48), "sans")
    tagline_font = get_font(scale(72), "serif")

    # RhombiLoRA title
    op1 = fade_in_out(frame_in_scene, 0, 20, total_frames - 35, 15)
    if op1 > 0:
        draw_centered_text(draw, "RhombiLoRA", int(H * 0.18), title_font, VIOLA)
        draw_centered_text(draw, "Geometric LoRA adapters that add diagonal bridges",
                           int(H * 0.30), body_font, DIM)

    # Three-paper arc
    papers = [
        ("Paper 1", "What happens when you replace the cube?", CUBIC_BLUE),
        ("Paper 2", "Where does the advantage come from?", FCC_RED),
        ("Paper 3", "What happens when you embrace it?", GOLD),
    ]
    for i, (label, question, color) in enumerate(papers):
        delay = 30 + i * 25
        op = fade_in_out(frame_in_scene, delay, 15, total_frames - delay - 25, 10)
        if op <= 0:
            continue
        y = int(H * (0.42 + i * 0.10))
        draw.text((int(W * 0.25), y), label, fill=color, font=get_font(scale(36), "bold"))
        draw.text((int(W * 0.35), y), question, fill=AVORIO, font=get_font(scale(36), "sans"))

    # Tagline
    op_tag = fade_in_out(frame_in_scene, 90, 25, total_frames - 130, 15)
    if op_tag > 0:
        draw_gold_rule(draw, int(H * 0.76), 0.30)
        draw_centered_text(draw, "Keep your cube, add six bridges.",
                           int(H * 0.82), tagline_font, GOLD)

    return img


def render_scene_cta(frame_in_scene, total_frames):
    """Scene 5: Call to action — install, links, 0 competitors."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, frame_in_scene + 2000)

    big_font = get_font(scale(56), "mono")
    url_font = get_font(scale(40), "sans")
    tag_font = get_font(scale(80), "serif")
    small_font = get_font(scale(32), "sans")

    # pip install
    op1 = fade_in_out(frame_in_scene, 0, 15, total_frames - 25, 10)
    if op1 > 0:
        draw_centered_text(draw, "pip install rhombic", int(H * 0.22), big_font, GREEN)

    # Links
    op2 = fade_in_out(frame_in_scene, 20, 15, total_frames - 45, 10)
    if op2 > 0:
        links = [
            "promptcrafted.github.io/rhombic",
            "github.com/promptcrafted/rhombic",
            "pypi.org/project/rhombic",
        ]
        for i, link in enumerate(links):
            y = int(H * 0.38 + i * H * 0.07)
            draw_centered_text(draw, link, y, url_font, AZURE)

    # Tagline
    op3 = fade_in_out(frame_in_scene, 45, 20, total_frames - 75, 10)
    if op3 > 0:
        draw_gold_rule(draw, int(H * 0.66), 0.25)
        draw_centered_text(draw, "0 competitors.", int(H * 0.73), tag_font, GOLD)

    # Attribution
    op4 = fade_in_out(frame_in_scene, 60, 15, total_frames - 85, 10)
    if op4 > 0:
        draw_centered_text(draw, "Promptcrafted  |  @NousResearch  |  #HermesAgentHackathon",
                           int(H * 0.90), small_font, DIM)

    return img


# ===================================================================
# Scene Sequencer
# ===================================================================

def build_demo(captures_dir: Path = CAPTURES):
    """Build all demo frames as a scene sequence."""
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    act1 = json.loads((captures_dir / "act1_lattice_compare.json").read_text())
    act3 = json.loads((captures_dir / "act3_explain_mechanism.json").read_text())

    # Scene schedule: (name, duration_seconds, renderer, extra_args)
    scenes = [
        ("Thesis",   5.5,  render_scene_thesis,  {}),
        ("Numbers",  10.0, render_scene_numbers,  {"data": act1}),
        ("Agent",    8.0,  render_scene_agent,    {"data": act3}),
        ("Vision",   7.5,  render_scene_vision,   {}),
        ("CTA",      6.0,  render_scene_cta,      {}),
    ]

    # Cross-fade duration (frames)
    XFADE = 12  # 0.4s crossfade between scenes

    total_frames = 0
    scene_frame_counts = []
    for name, dur, _, _ in scenes:
        n = int(dur * FPS)
        scene_frame_counts.append(n)
        total_frames += n

    print(f"Demo: {len(scenes)} scenes, {total_frames} frames, {total_frames / FPS:.1f}s")

    frame_num = 0
    prev_last_frame = None

    for si, (name, dur, renderer, kwargs) in enumerate(scenes):
        n_frames = scene_frame_counts[si]
        print(f"  {name} ({dur}s, {n_frames} frames)...", end=" ", flush=True)

        for f in range(n_frames):
            img = renderer(f, n_frames, **kwargs)

            # Cross-fade with previous scene
            if prev_last_frame is not None and f < XFADE:
                blend = ease_in_out(f / XFADE)
                arr_new = np.array(img, dtype=np.float32)
                arr_old = np.array(prev_last_frame, dtype=np.float32)
                blended = arr_old * (1 - blend) + arr_new * blend
                img = Image.fromarray(blended.astype(np.uint8))

            # Global fade in (first scene) and fade out (last scene)
            if si == 0 and f < 15:
                opacity = ease_out_cubic(f / 15)
                arr = np.array(img, dtype=np.float32)
                img = Image.fromarray(apply_fade(arr, opacity))
            if si == len(scenes) - 1 and f >= n_frames - 20:
                remaining = n_frames - 1 - f
                opacity = ease_out_cubic(remaining / 20)
                arr = np.array(img, dtype=np.float32)
                img = Image.fromarray(apply_fade(arr, opacity))

            frame_num += 1
            path = FRAME_DIR / f"frame_{frame_num:05d}.png"
            img.save(str(path))

            if f == n_frames - 1:
                prev_last_frame = img.copy()

        print("done")

    print(f"\n  Total: {frame_num} frames ({frame_num / FPS:.1f}s)")
    print(f"  Output: {FRAME_DIR}")
    return frame_num


if __name__ == "__main__":
    import shutil
    import time

    if FRAME_DIR.exists():
        shutil.rmtree(str(FRAME_DIR))

    t0 = time.time()
    n = build_demo()
    elapsed = time.time() - t0
    print(f"\nRendered in {elapsed:.1f}s ({elapsed / max(1, n) * 1000:.0f}ms/frame)")
