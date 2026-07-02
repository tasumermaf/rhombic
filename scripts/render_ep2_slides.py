#!/usr/bin/env python3
"""
Episode 2 Slide Renderer — Executive Register Motion Graphics.

Clean motion graphics: serif fonts, GOLD/AVORIO on NUIT, measured reveals,
particle field backgrounds. The Alien Executive's presentation.

Scenes:
  1. ep2_title     — "EPISODE II" title card with particle field (4s)
  2. status        — 3-column reveal: "22,477:1 / 0.10 / 3.8%" (10s)
  3. scale         — Three Fiedler values converging to ~0.10 (8s)
  4. bridge_topo   — Full-screen particle sim placeholder (12s — composited later)
  5. blood_test    — 4-row diagnostic reveal (12s)
  6. negative      — "Corpus weighting HURTS." Large serif, red (10s)
  7. convergence   — Registers merge. Both visual languages (12s)
  8. rd_sound_viz  — Spectral visualization of conditioning track (10s)
  9. tagline       — "Keep your cube. / Add six bridges." (8s)
  10. cta           — pip install + links (8s)

Output: assets/video/frames_ep2/slides/<scene>/frame_NNNNN.png

Usage:
    python scripts/render_ep2_slides.py
    python scripts/render_ep2_slides.py --preview
"""

from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32" and not hasattr(sys.stdout, '_rhombic_wrapped'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stdout._rhombic_wrapped = True
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PREVIEW = "--preview" in sys.argv
W, H = (1280, 720) if PREVIEW else (3840, 2160)
FPS = 30

BASE = Path(__file__).resolve().parent.parent
FRAME_DIR = BASE / "assets" / "video" / "frames_ep2" / "slides"

# ===================================================================
# Palette (from Ep1 — Executive register)
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
WARN_RED = (200, 50, 50)

# ===================================================================
# Font System (from Ep1 — shared)
# ===================================================================

_font_cache = {}

def get_font(size: int, style: str = "sans") -> ImageFont.FreeTypeFont:
    key = (size, style)
    if key in _font_cache:
        return _font_cache[key]
    candidates = {
        "sans": ["C:/Windows/Fonts/bahnschrift.ttf", "C:/Windows/Fonts/segoeui.ttf"],
        "serif": ["C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/times.ttf"],
        "mono": ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"],
        "bold": ["C:/Windows/Fonts/bahnschrift.ttf", "C:/Windows/Fonts/segoeuib.ttf"],
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

def scale(v: int) -> int:
    return int(v * W / 3840)

# ===================================================================
# Drawing Primitives (from Ep1 — shared)
# ===================================================================

def make_frame():
    return Image.new("RGB", (W, H), NUIT)

def draw_centered(draw, text, y, font, color):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2
    draw.text((x, y), text, fill=color, font=font)
    return x, tw

def gold_rule(draw, y, width_frac=0.25):
    span = int(W * width_frac)
    x1 = (W - span) // 2
    draw.line([(x1, y), (x1 + span, y)], fill=GOLD_DIM, width=max(1, scale(3)))

def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3

def fade_window(frame, start, fade_in, hold, fade_out):
    f = frame - start
    total = fade_in + hold + fade_out
    if f < 0 or f >= total:
        return 0.0
    if f < fade_in:
        return ease_out(f / max(1, fade_in))
    if f < fade_in + hold:
        return 1.0
    return 1.0 - ease_out((f - fade_in - hold) / max(1, fade_out))

def count_up(t, target, duration=2.0):
    progress = min(1.0, max(0, t / duration))
    return target * (1 - (1 - progress) ** 3)

def particle_field(img, frame, n=60, seed=777):
    rng = np.random.RandomState(seed)
    draw = ImageDraw.Draw(img)
    for i in range(n):
        bx = rng.random() * W
        by = rng.random() * H
        dx = math.sin(frame * 0.01 + i * 1.7) * scale(30)
        dy = math.cos(frame * 0.013 + i * 2.3) * scale(20)
        px = int((bx + dx) % W)
        py = int((by + dy) % H)
        brightness = (math.sin(frame * 0.05 + i * 3.1) * 0.5 + 0.5) ** 2
        r = max(1, scale(2))
        alpha_color = (int(GOLD[0] * brightness), int(GOLD[1] * brightness), int(GOLD[2] * brightness))
        draw.ellipse([px - r, py - r, px + r, py + r], fill=alpha_color)

# ===================================================================
# Scene Renderers
# ===================================================================

def render_ep2_title(f, n):
    """EPISODE II title card."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, f, 80, seed=2001)

    title_font = get_font(scale(140), "serif")
    sub_font = get_font(scale(56), "sans")

    op = fade_window(f, 0, 15, n - 35, 20)
    if op > 0:
        draw_centered(draw, "EPISODE II", int(H * 0.38), title_font, GOLD)
        draw_centered(draw, "THE BRIEFING", int(H * 0.55), sub_font, AVORIO)
        gold_rule(draw, int(H * 0.68), 0.20)
    return img


def render_status(f, n):
    """3-column status reveal: 22,477:1 / 0.10 / 3.8%."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, f, 50, seed=2002)
    t = f / FPS

    header_font = get_font(scale(60), "serif")
    big_font = get_font(scale(150), "bold")
    unit_font = get_font(scale(42), "sans")
    label_font = get_font(scale(34), "sans")

    op_h = fade_window(f, 0, 12, n - 22, 10)
    if op_h > 0:
        draw_centered(draw, "Status Update", int(H * 0.06), header_font, GOLD)
        gold_rule(draw, int(H * 0.13), 0.12)

    cols = [W * 0.18, W * 0.50, W * 0.82]

    # Column 1: 22,477:1 (Bridge asymmetry)
    op1 = fade_window(f, 15, 20, n - 50, 15)
    if op1 > 0:
        val = count_up(max(0, t - 0.5), 22477, 2.5)
        cx = int(cols[0])
        text = f"{val:,.0f}:1"
        bbox = draw.textbbox((0, 0), text, font=big_font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, int(H * 0.28)), text, fill=FCC_RED, font=big_font)
        draw_centered_at(draw, "bridge axis", cx, int(H * 0.50), unit_font, DIM)
        draw_centered_at(draw, "asymmetry", cx, int(H * 0.56), unit_font, DIM)
        draw_centered_at(draw, "Qwen 7B", cx, int(H * 0.68), label_font, GOLD_DIM)

    # Column 2: ~0.10 (Fiedler convergence)
    op2 = fade_window(f, 50, 20, n - 85, 15)
    if op2 > 0:
        val = count_up(max(0, t - 1.7), 0.10, 1.5)
        cx = int(cols[1])
        text = f"{val:.2f}"
        bbox = draw.textbbox((0, 0), text, font=big_font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, int(H * 0.28)), text, fill=AVORIO, font=big_font)
        draw_centered_at(draw, "Fiedler", cx, int(H * 0.50), unit_font, DIM)
        draw_centered_at(draw, "convergence", cx, int(H * 0.56), unit_font, DIM)
        draw_centered_at(draw, "scale-invariant", cx, int(H * 0.68), label_font, GOLD_DIM)

    # Column 3: 3.8% (Holly Battery)
    op3 = fade_window(f, 85, 20, n - 120, 15)
    if op3 > 0:
        val = count_up(max(0, t - 2.8), 3.8, 1.5)
        cx = int(cols[2])
        text = f"{val:.1f}%"
        bbox = draw.textbbox((0, 0), text, font=big_font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, int(H * 0.28)), text, fill=GREEN, font=big_font)
        draw_centered_at(draw, "better loss", cx, int(H * 0.50), unit_font, DIM)
        draw_centered_at(draw, "Holly Battery", cx, int(H * 0.56), unit_font, DIM)
        draw_centered_at(draw, "50% smaller checkpoint", cx, int(H * 0.68), label_font, GOLD_DIM)

    # Bottom tagline
    op_b = fade_window(f, 110, 15, n - 140, 15)
    if op_b > 0:
        gold_rule(draw, int(H * 0.82), 0.35)
        draw_centered(draw, "Three results. One geometry.", int(H * 0.87), get_font(scale(44), "serif"), GOLD)

    return img


def render_scale(f, n):
    """Three Fiedler values converging to ~0.10."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, f, 40, seed=2003)
    t = f / FPS

    header_font = get_font(scale(56), "serif")
    label_font = get_font(scale(44), "sans")
    num_font = get_font(scale(80), "bold")
    small_font = get_font(scale(34), "sans")

    op_h = fade_window(f, 0, 12, n - 22, 10)
    if op_h > 0:
        draw_centered(draw, "Scale Invariance", int(H * 0.08), header_font, GOLD)
        gold_rule(draw, int(H * 0.15), 0.15)

    # Three dots converging: 1.1B -> 0.096, 7B -> 0.101, 14B -> 0.098
    models = [
        ("TinyLlama 1.1B", 0.096, AZURE, 20),
        ("Qwen 7B", 0.101, FCC_RED, 45),
        ("Qwen 14B", 0.098, GOLD, 70),
    ]

    # Chart area
    chart_x = int(W * 0.15)
    chart_w = int(W * 0.70)
    chart_y = int(H * 0.25)
    chart_h = int(H * 0.45)
    target_y = chart_y + chart_h // 2  # ~0.10 line

    # Y-axis
    op_axis = fade_window(f, 10, 10, n - 25, 5)
    if op_axis > 0:
        draw.line([(chart_x, chart_y), (chart_x, chart_y + chart_h)], fill=DIM, width=scale(2))
        draw.line([(chart_x, chart_y + chart_h), (chart_x + chart_w, chart_y + chart_h)], fill=DIM, width=scale(2))
        # Target line at 0.10
        draw.line([(chart_x, target_y), (chart_x + chart_w, target_y)],
                  fill=GOLD_DIM, width=scale(2))
        draw.text((chart_x + chart_w + scale(20), target_y - scale(18)), "0.10", fill=GOLD_DIM, font=small_font)

    for i, (name, value, color, delay) in enumerate(models):
        op = fade_window(f, delay, 15, n - delay - 25, 10)
        if op <= 0:
            continue

        # X position (spread across chart)
        px = chart_x + int(chart_w * (i + 0.5) / len(models))

        # Y position (animate from random start to actual value)
        # Map value to chart: 0.00 at bottom, 0.20 at top
        val_frac = (0.20 - value) / 0.20
        final_y = chart_y + int(chart_h * (1 - val_frac))
        start_y = chart_y + int(chart_h * (0.3 + i * 0.15))
        anim_progress = min(1.0, (f - delay) / 30.0)
        curr_y = int(start_y + (final_y - start_y) * ease_out(anim_progress))

        # Dot
        dot_r = scale(20)
        draw.ellipse([px - dot_r, curr_y - dot_r, px + dot_r, curr_y + dot_r], fill=color)

        # Label
        draw_centered_at(draw, name, px, curr_y + scale(35), small_font, color)
        if anim_progress > 0.8:
            draw_centered_at(draw, f"{value:.3f}", px, curr_y - scale(45), num_font, color)

    # Bottom: convergence statement
    op_b = fade_window(f, 100, 15, n - 125, 10)
    if op_b > 0:
        draw_centered(draw, "Fiedler value converges to ~0.10 across 1.1B / 7B / 14B",
                      int(H * 0.82), get_font(scale(40), "sans"), AVORIO)
        draw_centered(draw, "The bridge is scale-invariant.", int(H * 0.88),
                      get_font(scale(40), "serif"), GOLD)

    return img


def render_blood_test(f, n):
    """4-row diagnostic reveal: fingerprint, overfitting, merging, eigenspectrum."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, f, 35, seed=2004)

    header_font = get_font(scale(56), "serif")
    num_font = get_font(scale(90), "bold")
    label_font = get_font(scale(40), "sans")
    detail_font = get_font(scale(34), "sans")

    op_h = fade_window(f, 0, 12, n - 22, 10)
    if op_h > 0:
        draw_centered(draw, "The Blood Test", int(H * 0.05), header_font, GOLD)
        gold_rule(draw, int(H * 0.12), 0.12)

    diagnostics = [
        ("72.3%", "task fingerprint accuracy", "36 bridge params identify the task", AZURE, 15),
        ("r = 0.888", "overfitting correlation", "bridge structure predicts loss landscape", FCC_RED, 40),
        ("36", "parameters merging", "for 14 billion total", GOLD, 65),
        ("cos > 0.999", "eigenspectrum alignment", "bridge eigenvectors match task gradients", GREEN, 90),
    ]

    for i, (value, label, detail, color, delay) in enumerate(diagnostics):
        op = fade_window(f, delay, 15, n - delay - 25, 10)
        if op <= 0:
            continue

        row_y = int(H * (0.20 + i * 0.19))

        # Value (left)
        draw.text((int(W * 0.10), row_y), value, fill=color, font=num_font)

        # Label (center-right)
        draw.text((int(W * 0.45), row_y + scale(10)), label, fill=AVORIO, font=label_font)

        # Detail (below label)
        draw.text((int(W * 0.45), row_y + scale(55)), detail, fill=DIM, font=detail_font)

        # Divider line
        draw.line([(int(W * 0.10), row_y + scale(100)), (int(W * 0.90), row_y + scale(100))],
                  fill=(30, 28, 55), width=scale(2))

    return img


def render_negative(f, n):
    """'Corpus weighting HURTS.' Large serif, red. The strongest scientific claim."""
    img = make_frame()
    draw = ImageDraw.Draw(img)

    huge_font = get_font(scale(140), "serif")
    body_font = get_font(scale(48), "sans")
    detail_font = get_font(scale(38), "sans")

    op1 = fade_window(f, 0, 20, n - 35, 15)
    if op1 > 0:
        draw_centered(draw, "Corpus weighting", int(H * 0.30), huge_font, AVORIO)
        draw_centered(draw, "HURTS.", int(H * 0.47), huge_font, WARN_RED)

    op2 = fade_window(f, 40, 15, n - 65, 10)
    if op2 > 0:
        gold_rule(draw, int(H * 0.65), 0.20)
        draw_centered(draw, "L-026 vs L-013: corpus weights degrade all metrics.",
                      int(H * 0.72), body_font, DIM)
        draw_centered(draw, "The topology does the work. Not the data.",
                      int(H * 0.80), detail_font, GOLD)

    return img


def render_convergence(f, n):
    """Registers merge — motion graphics + monospace. Both visual languages."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, f, 70, seed=2006)

    serif_font = get_font(scale(72), "serif")
    mono_font = get_font(scale(52), "mono")
    sans_font = get_font(scale(40), "sans")
    t = f / FPS

    # Phase 1: Serif title
    op1 = fade_window(f, 0, 15, n - 25, 10)
    if op1 > 0:
        draw_centered(draw, "The Pattern", int(H * 0.08), serif_font, GOLD)

    # Phase 2: Three monospace lines revealing the correspondence
    lines = [
        ("8 cubic vertices = 8 prime threads", AZURE, 25),
        ("6 octahedral vertices = 6 bridges", FCC_RED, 50),
        ("24 edges = 24 cards", GOLD, 75),
    ]

    for text, color, delay in lines:
        op = fade_window(f, delay, 15, n - delay - 20, 10)
        if op <= 0:
            continue
        idx = lines.index((text, color, delay))
        y = int(H * (0.28 + idx * 0.16))

        # Typing effect: reveal characters progressively
        chars = int(len(text) * min(1.0, (f - delay) / 20.0))
        visible = text[:chars]
        draw_centered(draw, visible, y, mono_font, color)

    # Phase 3: Bottom synthesis
    op_b = fade_window(f, 100, 20, n - 135, 15)
    if op_b > 0:
        gold_rule(draw, int(H * 0.78), 0.30)
        draw_centered(draw, "Topology + Diagnostics + Results = One finding.",
                      int(H * 0.84), sans_font, AVORIO)

    return img


def render_rd_sound_viz(f, n):
    """Spectral visualization of the conditioning track.

    8 prime frequencies as spectral peaks. MERIDIAN=220, Theos=284, beat=64.
    """
    img = make_frame()
    draw = ImageDraw.Draw(img)
    t = f / FPS

    header_font = get_font(scale(48), "serif")
    label_font = get_font(scale(32), "mono")
    small_font = get_font(scale(28), "sans")

    op_h = fade_window(f, 0, 12, n - 22, 10)
    if op_h > 0:
        draw_centered(draw, "RD Sound — Conditioning Track Spectrum", int(H * 0.05), header_font, GOLD)

    # Spectral display: horizontal frequency axis, vertical amplitude
    spec_x = int(W * 0.08)
    spec_w = int(W * 0.84)
    spec_y = int(H * 0.15)
    spec_h = int(H * 0.60)
    base_y = spec_y + spec_h

    # Axis
    draw.line([(spec_x, base_y), (spec_x + spec_w, base_y)], fill=DIM, width=scale(2))
    draw.line([(spec_x, spec_y), (spec_x, base_y)], fill=DIM, width=scale(2))

    # Frequency range: 10 Hz - 500 Hz (log scale)
    freq_min, freq_max = 10.0, 500.0

    def freq_to_x(freq):
        log_pos = (math.log(freq) - math.log(freq_min)) / (math.log(freq_max) - math.log(freq_min))
        return spec_x + int(spec_w * log_pos)

    # Background spectrum (pink noise profile, animated)
    rng = np.random.RandomState(42)
    n_bins = 200
    for i in range(n_bins):
        freq = freq_min * (freq_max / freq_min) ** (i / n_bins)
        x = freq_to_x(freq)
        # 1/f noise profile
        base_amp = 0.15 / (1 + freq / 50.0)
        noise = rng.normal(0, 0.02)
        wobble = math.sin(t * 2.0 + i * 0.3) * 0.02
        amp = max(0, base_amp + noise + wobble)
        bar_h = int(spec_h * amp * fade_window(f, 5, 15, n - 25, 5))
        if bar_h > 1:
            color = (30 + int(20 * amp), 28 + int(15 * amp), 55 + int(30 * amp))
            draw.rectangle([x, base_y - bar_h, x + max(1, spec_w // n_bins - 1), base_y], fill=color)

    # Prime frequency peaks (animated)
    peaks = [
        (11, "11", CUBIC_BLUE, 20),
        (23, "23", AZURE, 30),
        (40, "40 Hz", GOLD, 10),  # Isochronic primary
        (64, "64", GREEN, 40),
        (67, "67", FCC_RED, 50),
        (89, "89", VIOLA, 60),
        (220, "MERIDIAN", GOLD, 70),
        (284, "Theos", AVORIO, 80),
    ]

    for freq, label, color, delay in peaks:
        op = fade_window(f, delay, 12, n - delay - 20, 8)
        if op <= 0:
            continue

        x = freq_to_x(freq)
        # Peak height (animated pulse)
        pulse = 0.5 + 0.3 * math.sin(t * 3.0 + freq * 0.1)
        peak_h = int(spec_h * (0.4 + 0.4 * pulse) * op)

        # Glowing bar
        bar_w = max(scale(6), spec_w // 60)
        for glow in range(3):
            gw = bar_w + glow * scale(4)
            alpha = 1.0 - glow * 0.3
            glow_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
            draw.rectangle([x - gw // 2, base_y - peak_h, x + gw // 2, base_y], fill=glow_color)

        # Label (above peak)
        draw.text((x - scale(20), base_y - peak_h - scale(30)), label, fill=color, font=label_font)

    # Axis labels
    for freq in [10, 20, 50, 100, 200, 500]:
        x = freq_to_x(freq)
        draw.text((x - scale(15), base_y + scale(10)), str(freq), fill=DIM, font=small_font)

    # Bottom label
    op_b = fade_window(f, 90, 15, n - 115, 10)
    if op_b > 0:
        draw_centered(draw, "8 prime frequencies. 40 Hz isochronic entrainment. Brain-entraining geometry.",
                      int(H * 0.87), get_font(scale(38), "sans"), AVORIO)

    return img


def render_tagline(f, n):
    """'Keep your cube. Add six bridges.' — earned, cinematic."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, f, 40, seed=2008)

    huge_font = get_font(scale(110), "serif")

    op1 = fade_window(f, 0, 25, n - 40, 15)
    if op1 > 0:
        draw_centered(draw, "Keep your cube.", int(H * 0.35), huge_font, AVORIO)

    op2 = fade_window(f, 30, 25, n - 65, 15)
    if op2 > 0:
        draw_centered(draw, "Add six bridges.", int(H * 0.55), huge_font, GOLD)

    return img


def render_cta(f, n):
    """Call to action: pip install + links."""
    img = make_frame()
    draw = ImageDraw.Draw(img)
    particle_field(img, f, 30, seed=2009)

    mono_font = get_font(scale(64), "mono")
    url_font = get_font(scale(40), "sans")
    small_font = get_font(scale(32), "sans")

    op1 = fade_window(f, 0, 15, n - 25, 10)
    if op1 > 0:
        draw_centered(draw, "pip install rhombic", int(H * 0.25), mono_font, GREEN)

    op2 = fade_window(f, 20, 15, n - 45, 10)
    if op2 > 0:
        draw_centered(draw, "particles.casberry.in", int(H * 0.45), url_font, AZURE)

    op3 = fade_window(f, 35, 15, n - 60, 10)
    if op3 > 0:
        gold_rule(draw, int(H * 0.62), 0.20)
        draw_centered(draw, "Promptcrafted LLC", int(H * 0.70), get_font(scale(48), "serif"), GOLD)

    op4 = fade_window(f, 50, 12, n - 70, 8)
    if op4 > 0:
        draw_centered(draw, "TASUMER MAF", int(H * 0.85), small_font, DIM)

    return img


# ===================================================================
# Utility
# ===================================================================

def draw_centered_at(draw, text, cx, y, font, color):
    """Draw text centered at a specific x coordinate."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, y), text, fill=color, font=font)


# ===================================================================
# Scene Table
# ===================================================================

SCENES = [
    ("ep2_title",    render_ep2_title,    4.0),
    ("status",       render_status,       10.0),
    ("scale",        render_scale,        8.0),
    ("blood_test",   render_blood_test,   12.0),
    ("negative",     render_negative,     10.0),
    ("convergence",  render_convergence,  12.0),
    ("rd_sound_viz", render_rd_sound_viz, 10.0),
    ("tagline",      render_tagline,      8.0),
    ("cta",          render_cta,          8.0),
]


def main():
    print("=" * 60)
    print("  EPISODE 2 SLIDE RENDERER — Executive Register")
    print(f"  Resolution: {W}x{H} @ {FPS}fps")
    print("=" * 60)

    for name, renderer, duration in SCENES:
        out_dir = FRAME_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        n_frames = int(duration * FPS)
        print(f"\n  Rendering '{name}' ({duration}s, {n_frames} frames)...", flush=True)
        for f in range(n_frames):
            if f % 50 == 0:
                print(f"    Frame {f}/{n_frames}", flush=True)
            img = renderer(f, n_frames)
            path = out_dir / f"frame_{f + 1:05d}.png"
            img.save(str(path))
        print(f"    -> {n_frames} frames")

    print("\n" + "=" * 60)
    print("  SLIDES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
