#!/usr/bin/env python3
"""
Terminal Frame Renderer — Strongbad-style cybernetic circuit visualization.

Renders simulated terminal interactions between an observer and rhombic-agent
as PNG frame sequences. Typing animation for prompts, streaming for responses,
cursor blink, terminal chrome, subtle VHS texture. Inline visualizations
appear after each interaction as if the agent generated them.

Output: assets/video/terminal_frames/frame_NNNNN.png

Usage:
    python scripts/render_terminal.py
    python scripts/render_terminal.py --preview   # 1280x720
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import textwrap
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
FRAME_DIR = BASE / "assets" / "video" / "terminal_frames"

# ===================================================================
# Palette — terminal on NUIT
# ===================================================================

NUIT = (8, 6, 32)
TERM_BG = (12, 10, 28)
CHROME_BG = (22, 18, 42)
CHROME_DOT_R = (179, 68, 68)
CHROME_DOT_Y = (220, 201, 100)
CHROME_DOT_G = (74, 140, 92)
PROMPT_COLOR = (74, 140, 92)
USER_COLOR = (240, 232, 208)
AGENT_COLOR = (180, 175, 190)
TOOL_COLOR = (92, 143, 175)
HEADER_COLOR = (220, 201, 100)
DIM_COLOR = (100, 96, 110)
NUM_COLOR = (179, 68, 68)
CURSOR_COLOR = (240, 232, 208)
CUBIC_RGB = (61, 61, 107)
FCC_RGB = (179, 68, 68)
GOLD_RGB = (220, 201, 100)
SCANLINE_ALPHA = 0.03

# ===================================================================
# Font System
# ===================================================================

_font_cache = {}


def get_font(size: int, style: str = "mono") -> ImageFont.FreeTypeFont:
    key = (size, style)
    if key in _font_cache:
        return _font_cache[key]
    candidates = {
        "mono": [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
        ],
        "mono_bold": [
            "C:/Windows/Fonts/consolab.ttf",
            "C:/Windows/Fonts/courbd.ttf",
        ],
        "sans": [
            "C:/Windows/Fonts/bahnschrift.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ],
    }
    for path in candidates.get(style, candidates["mono"]):
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
# Terminal Layout
# ===================================================================

MARGIN = 0.06
CHROME_H_FRAC = 0.04


def term_rect():
    mx = int(W * MARGIN)
    my = int(H * MARGIN)
    ch = int(H * CHROME_H_FRAC)
    return mx, my + ch, W - 2 * mx, H - 2 * my - ch


def chrome_rect():
    mx = int(W * MARGIN)
    my = int(H * MARGIN)
    ch = int(H * CHROME_H_FRAC)
    return mx, my, W - 2 * mx, ch


# ===================================================================
# Text Metrics
# ===================================================================

def char_size(font):
    bbox = font.getbbox("M")
    cw = bbox[2] - bbox[0]
    ch = bbox[3] - bbox[1]
    line_h = int(ch * 1.45)
    return cw, line_h


# ===================================================================
# Interaction Script
# ===================================================================

INTERACTIONS = [
    {
        "prompt": "Compare cubic and FCC lattices at scale 125.",
        "tool": "lattice_compare(n=5)",
        "response": [
            ("header", "Lattice Comparison - Scale 125"),
            ("blank", ""),
            ("row", "              Cubic     FCC"),
            ("row", "Nodes         125       125"),
            ("row", "Edges         300       750"),
            ("row", "Avg path      4.00      2.83"),
            ("row", "Diameter      8         5"),
            ("num", "Fiedler       0.382     0.877"),
            ("num", "Ratio                   2.30x"),
            ("blank", ""),
            ("text", "FCC: 2.30x algebraic connectivity."),
            ("text", "30% shorter paths. 40% smaller diameter."),
            ("dim", "Paper 1, Table 1"),
        ],
        "viz": "comparison",
    },
    {
        "prompt": "What happens with heterogeneous edge weights?",
        "tool": "direction_weights(distribution='corpus')",
        "response": [
            ("header", "Fiedler Ratio - Amplification"),
            ("blank", ""),
            ("row", "Distribution   Edge-cycled  Dir-weighted"),
            ("row", "Uniform        2.55x        2.55x"),
            ("row", "Random         2.64x        3.65x"),
            ("row", "Power-law      3.06x        3.37x"),
            ("num", "Corpus         3.11x        6.11x"),
            ("blank", ""),
            ("text", "Heterogeneous weights create bottlenecks."),
            ("text", "Cubic: 5 alternatives around a block."),
            ("num", "FCC: 11 alternatives. Advantage amplifies."),
            ("blank", ""),
            ("num", "The 6.1x finding: structure, not noise."),
            ("dim", "Paper 2, Figure 3"),
        ],
        "viz": "amplification",
    },
    {
        "prompt": "Run the permutation test.",
        "tool": "permutation_control(n_perms=10000)",
        "response": [
            ("header", "Permutation Test - Prime-Vertex Mapping"),
            ("blank", ""),
            ("text", "Null hypothesis: prime positions are"),
            ("text", "arbitrary w.r.t. RD geometry."),
            ("blank", ""),
            ("row", "Sorted (geometric):  score = 0.847"),
            ("row", "Shuffled (random):   score = 0.412 +/- 0.09"),
            ("num", "p = 0.000025"),
            ("blank", ""),
            ("text", "Primes cluster at high-connectivity"),
            ("text", "vertices of the rhombic dodecahedron."),
            ("num", "Not accidental. Geometric affinity."),
            ("dim", "Paper 2, Section 5.3"),
        ],
        "viz": "permutation",
    },
    {
        "prompt": "Explain the mechanism. Why does FCC beat cubic?",
        "tool": "explain_mechanism(depth='intuitive')",
        "response": [
            ("header", "Mechanism - Bottleneck Resilience"),
            ("blank", ""),
            ("text", "Think of it as road networks."),
            ("blank", ""),
            ("text", "Cubic: each intersection has 6 roads."),
            ("text", "Block one, you have 5 alternatives."),
            ("text", "If several are weak, you're stuck."),
            ("blank", ""),
            ("text", "FCC: each intersection has 12 roads."),
            ("text", "Block one, you have 11 alternatives."),
            ("num", "Traffic flows even when roads are unequal."),
            ("blank", ""),
            ("text", "The rhombic dodecahedron contains the cube."),
            ("text", "8 trivalent vertices = cube corners."),
            ("num", "6 tetravalent vertices = the bridges."),
            ("blank", ""),
            ("num", "Keep your cube. Add six bridges."),
        ],
        "viz": "tagline",
    },
]


# ===================================================================
# Visualization Renderers
# ===================================================================

def _centered_text(draw, text, cx, y, font, fill):
    """Draw text centered horizontally at cx."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, y), text, fill=fill, font=font)


def _bar(draw, x, y, w, h, fill, progress=1.0):
    """Draw an animated horizontal bar."""
    bw = max(1, int(w * min(1.0, progress)))
    draw.rectangle([x, y, x + bw, y + h], fill=fill)


def render_viz_comparison(draw, x, y, w, h, progress):
    """Viz 1: Cubic vs FCC algebraic connectivity bars."""
    font_huge = get_font(scale(100), "sans")
    font_title = get_font(scale(52), "sans")
    font_label = get_font(scale(44), "mono")

    cx = x + w // 2
    bar_left = x + scale(260)
    bar_max = int(w * 0.50)
    bar_h = scale(64)

    # Title
    if progress > 0.02:
        _centered_text(draw, "Algebraic Connectivity", cx, y + scale(50), font_title, HEADER_COLOR)

    # Cubic bar
    if progress > 0.12:
        by = y + scale(220)
        bp = min(1.0, (progress - 0.12) / 0.25)
        draw.text((x + scale(60), by + scale(6)), "Cubic", fill=AGENT_COLOR, font=font_label)
        cubic_w = int(bar_max * 0.382 / 0.877)
        _bar(draw, bar_left, by, cubic_w, bar_h, CUBIC_RGB, bp)
        if progress > 0.45:
            draw.text((bar_left + cubic_w + scale(24), by + scale(6)), "0.382", fill=AGENT_COLOR, font=font_label)

    # FCC bar
    if progress > 0.22:
        by = y + scale(340)
        bp = min(1.0, (progress - 0.22) / 0.30)
        draw.text((x + scale(60), by + scale(6)), "FCC", fill=NUM_COLOR, font=font_label)
        _bar(draw, bar_left, by, bar_max, bar_h, FCC_RGB, bp)
        if progress > 0.58:
            draw.text((bar_left + bar_max + scale(24), by + scale(6)), "0.877", fill=NUM_COLOR, font=font_label)

    # Path length comparison
    if progress > 0.50:
        by = y + scale(490)
        draw.text((x + scale(60), by), "Avg path", fill=DIM_COLOR, font=font_label)
        path_cubic_w = int(bar_max * 0.71)  # 4.00/5.65
        path_fcc_w = int(bar_max * 0.50)    # 2.83/5.65
        _bar(draw, bar_left, by, path_cubic_w, scale(36), CUBIC_RGB, min(1.0, (progress - 0.50) / 0.2))
        by2 = by + scale(52)
        _bar(draw, bar_left, by2, path_fcc_w, scale(36), FCC_RGB, min(1.0, (progress - 0.55) / 0.2))
        if progress > 0.72:
            draw.text((bar_left + path_cubic_w + scale(16), by - scale(2)), "4.00", fill=DIM_COLOR, font=get_font(scale(34), "mono"))
            draw.text((bar_left + path_fcc_w + scale(16), by2 - scale(2)), "2.83", fill=DIM_COLOR, font=get_font(scale(34), "mono"))

    # Big ratio
    if progress > 0.70:
        _centered_text(draw, "2.30x", cx, y + scale(670), font_huge, NUM_COLOR)
    if progress > 0.82:
        _centered_text(draw, "algebraic connectivity advantage", cx, y + scale(800), font_label, DIM_COLOR)


def render_viz_amplification(draw, x, y, w, h, progress):
    """Viz 2: Fiedler ratio amplification gradient (4 distributions)."""
    font_huge = get_font(scale(100), "sans")
    font_title = get_font(scale(52), "sans")
    font_label = get_font(scale(40), "mono")
    font_sm = get_font(scale(34), "mono")

    cx = x + w // 2
    bar_left = x + scale(320)
    bar_max = int(w * 0.48)
    bar_h = scale(40)
    row_gap = scale(110)

    data = [
        ("Uniform",   2.55, 2.55),
        ("Random",    2.64, 3.65),
        ("Power-law", 3.06, 3.37),
        ("Corpus",    3.11, 6.11),
    ]
    max_val = 6.11

    if progress > 0.02:
        _centered_text(draw, "Fiedler Ratio Amplification at Scale 1,000", cx, y + scale(40), font_title, HEADER_COLOR)

    # Legend
    if progress > 0.08:
        ly = y + scale(120)
        draw.rectangle([x + scale(280), ly, x + scale(310), ly + scale(24)], fill=TOOL_COLOR)
        draw.text((x + scale(320), ly - scale(4)), "Edge-cycled", fill=DIM_COLOR, font=font_sm)
        draw.rectangle([x + scale(560), ly, x + scale(590), ly + scale(24)], fill=NUM_COLOR)
        draw.text((x + scale(600), ly - scale(4)), "Direction-weighted", fill=DIM_COLOR, font=font_sm)

    for i, (label, ec, dw) in enumerate(data):
        row_start = 0.10 + i * 0.15
        if progress < row_start:
            continue

        by = y + scale(180) + i * row_gap
        bp = min(1.0, (progress - row_start) / 0.20)

        # Label
        draw.text((x + scale(50), by + scale(4)), label, fill=AGENT_COLOR if i < 3 else NUM_COLOR, font=font_label)

        # EC bar
        ec_w = int(bar_max * ec / max_val)
        _bar(draw, bar_left, by, ec_w, bar_h, TOOL_COLOR, bp)

        # DW bar (below)
        dw_w = int(bar_max * dw / max_val)
        _bar(draw, bar_left, by + bar_h + scale(8), dw_w, bar_h, FCC_RGB if i == 3 else NUM_COLOR, bp)

        # Values
        if progress > row_start + 0.18:
            draw.text((bar_left + ec_w + scale(12), by + scale(2)), f"{ec:.2f}x", fill=DIM_COLOR, font=font_sm)
            draw.text((bar_left + dw_w + scale(12), by + bar_h + scale(10)), f"{dw:.2f}x",
                       fill=NUM_COLOR if i == 3 else DIM_COLOR, font=font_sm)

    # Big headline number
    if progress > 0.78:
        _centered_text(draw, "6.11x", cx, y + scale(680), font_huge, NUM_COLOR)
    if progress > 0.88:
        _centered_text(draw, "structure amplifies the advantage", cx, y + scale(810), font_label, DIM_COLOR)


def render_viz_permutation(draw, x, y, w, h, progress):
    """Viz 3: Permutation test — null distribution with observed score."""
    font_huge = get_font(scale(100), "sans")
    font_title = get_font(scale(52), "sans")
    font_label = get_font(scale(44), "mono")
    font_sm = get_font(scale(36), "mono")

    cx = x + w // 2

    if progress > 0.02:
        _centered_text(draw, "Prime-Vertex Geometry Affinity", cx, y + scale(40), font_title, HEADER_COLOR)

    # Draw a bell curve approximation (null distribution)
    if progress > 0.10:
        chart_x = x + scale(200)
        chart_y = y + scale(180)
        chart_w = int(w * 0.65)
        chart_h = scale(350)
        base_y = chart_y + chart_h

        # Axis line
        draw.line([(chart_x, base_y), (chart_x + chart_w, base_y)], fill=DIM_COLOR, width=scale(3))

        # Bell curve bars (centered at 0.412, std 0.09)
        n_bars = 50
        bar_w = chart_w // n_bars
        mean, std = 0.412, 0.09
        x_min, x_max = 0.15, 0.90

        bp = min(1.0, (progress - 0.10) / 0.30)
        for i in range(n_bars):
            bx_val = x_min + (x_max - x_min) * i / n_bars
            # Gaussian
            g = math.exp(-0.5 * ((bx_val - mean) / std) ** 2)
            bar_height = int(chart_h * 0.85 * g * bp)
            if bar_height < 2:
                continue
            bx = chart_x + i * bar_w
            # Color: blue for null, red if near observed
            color = TOOL_COLOR if bx_val < 0.75 else (140, 100, 100)
            draw.rectangle([bx, base_y - bar_height, bx + bar_w - scale(2), base_y], fill=color)

        # Observed marker
        if progress > 0.45:
            obs_x = chart_x + int(chart_w * (0.847 - x_min) / (x_max - x_min))
            marker_h = int(chart_h * 0.7)
            draw.line([(obs_x, base_y - marker_h), (obs_x, base_y)], fill=NUM_COLOR, width=scale(5))
            draw.text((obs_x + scale(12), base_y - marker_h - scale(8)), "observed", fill=NUM_COLOR, font=font_sm)
            draw.text((obs_x + scale(12), base_y - marker_h + scale(30)), "0.847", fill=NUM_COLOR, font=font_sm)

        # Axis labels
        if progress > 0.35:
            draw.text((chart_x, base_y + scale(12)), "0.15", fill=DIM_COLOR, font=font_sm)
            mean_x = chart_x + int(chart_w * (mean - x_min) / (x_max - x_min))
            draw.text((mean_x - scale(20), base_y + scale(12)), "0.41", fill=DIM_COLOR, font=font_sm)
            draw.text((chart_x + chart_w - scale(40), base_y + scale(12)), "0.90", fill=DIM_COLOR, font=font_sm)
            _centered_text(draw, "10,000 random permutations", cx, base_y + scale(55), font_sm, DIM_COLOR)

    # Big p-value
    if progress > 0.65:
        _centered_text(draw, "p = 0.000025", cx, y + scale(650), font_huge, NUM_COLOR)
    if progress > 0.80:
        _centered_text(draw, "not accidental", cx, y + scale(785), font_label, GOLD_RGB)


def render_viz_tagline(draw, x, y, w, h, progress):
    """Viz 4: The tagline — cinematic closing."""
    font_huge = get_font(scale(96), "sans")
    font_big = get_font(scale(64), "sans")
    font_med = get_font(scale(48), "mono")
    font_sm = get_font(scale(38), "mono")

    cx = x + w // 2

    # Cube: 6 neighbors
    if progress > 0.05:
        by = y + scale(120)
        _centered_text(draw, "Cubic lattice", cx, by, font_big, CUBIC_RGB)
        if progress > 0.12:
            _centered_text(draw, "6 neighbors   3 directions   1 assumption", cx, by + scale(80), font_sm, DIM_COLOR)

    # RD: 12 neighbors
    if progress > 0.25:
        by = y + scale(320)
        _centered_text(draw, "FCC lattice", cx, by, font_big, FCC_RGB)
        if progress > 0.32:
            _centered_text(draw, "12 neighbors   6 directions   0 assumptions", cx, by + scale(80), font_sm, DIM_COLOR)

    # Divider
    if progress > 0.45:
        line_w = scale(400)
        ly = y + scale(500)
        draw.line([(cx - line_w, ly), (cx + line_w, ly)], fill=GOLD_RGB, width=scale(3))

    # The tagline
    if progress > 0.55:
        _centered_text(draw, "Keep your cube.", cx, y + scale(560), font_huge, USER_COLOR)
    if progress > 0.68:
        _centered_text(draw, "Add six bridges.", cx, y + scale(680), font_huge, GOLD_RGB)

    # Install line
    if progress > 0.85:
        _centered_text(draw, "pip install rhombic", cx, y + scale(840), font_med, PROMPT_COLOR)


VIZ_RENDERERS = {
    "comparison": render_viz_comparison,
    "amplification": render_viz_amplification,
    "permutation": render_viz_permutation,
    "tagline": render_viz_tagline,
}


# ===================================================================
# Frame Renderer
# ===================================================================

def draw_chrome(img):
    draw = ImageDraw.Draw(img)
    cx, cy, cw, ch = chrome_rect()
    draw.rectangle([cx, cy, cx + cw, cy + ch], fill=CHROME_BG)

    dot_r = max(3, scale(16))
    dot_y = cy + ch // 2
    dot_gap = scale(44)
    dot_start = cx + scale(32)
    for i, color in enumerate([CHROME_DOT_R, CHROME_DOT_Y, CHROME_DOT_G]):
        dx = dot_start + i * dot_gap
        draw.ellipse([dx - dot_r, dot_y - dot_r, dx + dot_r, dot_y + dot_r], fill=color)

    title_font = get_font(scale(36), "sans")
    title = "rhombic-agent"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    tx = cx + cw // 2 - tw // 2
    draw.text((tx, dot_y - scale(14)), title, fill=DIM_COLOR, font=title_font)


def draw_scanlines(img):
    arr = np.array(img, dtype=np.float32)
    for y in range(0, H, 3 if PREVIEW else 4):
        arr[y, :, :] *= (1.0 - SCANLINE_ALPHA)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def render_terminal_frame(
    visible_chars: int,
    lines: list,
    cursor_on: bool,
    frame_idx: int,
) -> Image.Image:
    img = Image.new("RGB", (W, H), NUIT)
    draw_chrome(img)
    draw = ImageDraw.Draw(img)

    tx, ty, tw, th = term_rect()
    draw.rectangle([tx, ty, tx + tw, ty + th], fill=TERM_BG)

    # ── Viz mode: lines[0] == ("__VIZ__", type, progress) ──
    if lines and len(lines) == 1 and lines[0][0] == "__VIZ__":
        viz_type = lines[0][1]
        viz_progress = lines[0][2]
        renderer = VIZ_RENDERERS.get(viz_type)
        if renderer:
            renderer(draw, tx, ty, tw, th, viz_progress)
        img = draw_scanlines(img)
        # Vignette
        arr = np.array(img, dtype=np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        cx_f, cy_f = W / 2, H / 2
        dist = np.sqrt((xx - cx_f) ** 2 + (yy - cy_f) ** 2)
        max_dist = math.sqrt(cx_f ** 2 + cy_f ** 2)
        vignette = 1.0 - 0.25 * (dist / max_dist) ** 1.5
        arr *= vignette[:, :, None]
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    # ── Normal text mode ──
    FONT_SIZE = scale(56)
    font = get_font(FONT_SIZE, "mono")
    font_bold = get_font(FONT_SIZE, "mono_bold")
    cw, lh = char_size(font)

    pad_x = scale(36)
    pad_y = scale(28)
    max_chars_per_line = max(20, (tw - 2 * pad_x) // cw)

    # Build display lines with wrapping
    display_lines = []
    for style, text in lines:
        if style == "blank":
            display_lines.append((style, "", False))
        elif style == "prompt":
            display_lines.append((style, text, False))
        else:
            wrapped = textwrap.wrap(text, width=max_chars_per_line) if len(text) > max_chars_per_line else [text]
            for i, wl in enumerate(wrapped):
                display_lines.append((style, wl, i > 0))

    # Render visible characters
    char_count = 0
    last_drawn_x = tx + pad_x
    last_drawn_y = ty + pad_y

    for dl_idx, (style, text, is_cont) in enumerate(display_lines):
        line_y = ty + pad_y + dl_idx * lh

        if line_y + lh > ty + th - pad_y:
            break

        if style == "blank":
            char_count += 1
            if char_count > visible_chars:
                last_drawn_x = tx + pad_x
                last_drawn_y = line_y
                break
            continue

        if style == "prompt":
            prompt_str = "> "
            draw.text((tx + pad_x, line_y), prompt_str, fill=PROMPT_COLOR, font=font_bold)
            text_start_x = tx + pad_x + len(prompt_str) * cw
            color = USER_COLOR
            use_font = font_bold
        elif style == "tool":
            text_start_x = tx + pad_x + scale(12)
            color = TOOL_COLOR
            use_font = font
        elif style == "header":
            text_start_x = tx + pad_x
            color = HEADER_COLOR
            use_font = font_bold
        elif style == "num":
            text_start_x = tx + pad_x + (scale(20) if is_cont else 0)
            color = NUM_COLOR
            use_font = font_bold
        elif style == "dim":
            text_start_x = tx + pad_x + (scale(20) if is_cont else 0)
            color = DIM_COLOR
            use_font = font
        else:
            text_start_x = tx + pad_x + (scale(20) if is_cont else 0)
            color = AGENT_COLOR
            use_font = font

        for ci, ch in enumerate(text):
            char_count += 1
            if char_count > visible_chars:
                last_drawn_x = text_start_x + ci * cw
                last_drawn_y = line_y
                break
            xp = text_start_x + ci * cw
            draw.text((xp, line_y), ch, fill=color, font=use_font)
            last_drawn_x = xp + cw
            last_drawn_y = line_y
        else:
            char_count += 1
            if char_count > visible_chars:
                last_drawn_x = text_start_x + len(text) * cw
                last_drawn_y = line_y
                break
            continue
        if char_count > visible_chars:
            break

    # Cursor
    if cursor_on:
        cursor_x = min(last_drawn_x, tx + tw - pad_x - cw)
        cursor_y = last_drawn_y
        draw.rectangle([cursor_x, cursor_y, cursor_x + cw, cursor_y + lh - scale(4)], fill=CURSOR_COLOR)

    # Scanlines
    img = draw_scanlines(img)

    # Vignette
    arr = np.array(img, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    cx_f, cy_f = W / 2, H / 2
    dist = np.sqrt((xx - cx_f) ** 2 + (yy - cy_f) ** 2)
    max_dist = math.sqrt(cx_f ** 2 + cy_f ** 2)
    vignette = 1.0 - 0.25 * (dist / max_dist) ** 1.5
    arr *= vignette[:, :, None]
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    return img


# ===================================================================
# Sequence Builder
# ===================================================================

def build_interaction_frames(interaction: dict, start_frame: int = 0) -> list:
    prompt = interaction["prompt"]
    tool = interaction["tool"]
    response = interaction["response"]
    viz_type = interaction.get("viz")

    frames = []

    # Phase 1: Typing the prompt
    prompt_line = ("prompt", prompt)
    typing_speed = 1
    prompt_chars = len(prompt)
    typing_frames = prompt_chars * typing_speed

    for f in range(typing_frames):
        chars_shown = (f // typing_speed) + 1
        cursor_blink = (f % 20) < 14
        frames.append(([prompt_line], chars_shown + 2, cursor_blink))

    # Phase 2: Brief pause
    pause_frames = int(FPS * 0.4)
    full_prompt_chars = len(prompt) + 2
    for f in range(pause_frames):
        cursor_blink = (f % 15) < 8
        frames.append(([prompt_line], full_prompt_chars, cursor_blink))

    # Phase 3: Tool invocation
    tool_line = ("tool", f"[calling {tool}]")
    tool_display_frames = int(FPS * 0.8)
    lines_so_far = [prompt_line, ("blank", ""), tool_line]
    total_chars_prompt = full_prompt_chars + 1 + len(tool_line[1]) + 1
    for f in range(tool_display_frames):
        cursor_blink = (f % 20) < 10
        frames.append((lines_so_far, total_chars_prompt, cursor_blink))

    # Phase 4: Response streams in
    response_lines = [prompt_line, ("blank", ""), tool_line, ("blank", "")]
    base_chars = total_chars_prompt + 1

    stream_speed = 1
    for resp_style, resp_text in response:
        response_lines.append((resp_style, resp_text))
        line_chars = max(1, len(resp_text)) + 1
        line_frames = line_chars * stream_speed

        for f in range(line_frames):
            chars_shown = base_chars + (f // stream_speed) + 1
            frames.append((list(response_lines), chars_shown, False))

        base_chars += line_chars

    # Phase 5: Hold on completed response
    hold_frames = int(FPS * 1.5)
    for f in range(hold_frames):
        cursor_blink = (f % 24) < 12
        frames.append((list(response_lines), base_chars + 999, cursor_blink))

    # Phase 6: Visualization (if present)
    if viz_type:
        # 6a: "[generating visualization...]" tool line (0.6s)
        viz_tool_line = ("tool", f"[calling visualize_{viz_type}()]")
        viz_announce_lines = list(response_lines) + [("blank", ""), viz_tool_line]
        viz_announce_chars = base_chars + 999
        for f in range(int(FPS * 0.6)):
            frames.append((viz_announce_lines, viz_announce_chars, (f % 20) < 10))

        # 6b: Transition — text fades out (0.4s, rendered as rapid blank)
        for f in range(int(FPS * 0.4)):
            # Empty terminal with just chrome during transition
            frames.append(([], 0, False))

        # 6c: Visualization animates in (2.5s)
        viz_anim_frames = int(FPS * 2.5)
        for f in range(viz_anim_frames):
            progress = f / max(1, viz_anim_frames - 1)
            frames.append(([("__VIZ__", viz_type, progress)], 0, False))

        # 6d: Viz hold at full (2.0s)
        viz_hold_frames = int(FPS * 2.0)
        for f in range(viz_hold_frames):
            frames.append(([("__VIZ__", viz_type, 1.0)], 0, False))

    return frames


def build_all_terminal_frames():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []

    intro_frames = int(FPS * 1.5)
    for f in range(intro_frames):
        cursor_blink = (f % 24) < 12
        all_frames.append(([], 0, cursor_blink))

    for idx, interaction in enumerate(INTERACTIONS):
        print(f"  Building interaction {idx + 1}/{len(INTERACTIONS)}: "
              f"{interaction['prompt'][:40]}...")
        interaction_frames = build_interaction_frames(interaction)
        all_frames.extend(interaction_frames)

    if all_frames:
        last_lines, last_chars, _ = all_frames[-1]
        for f in range(int(FPS * 1.0)):
            all_frames.append((last_lines, last_chars, False))

    total = len(all_frames)
    print(f"\n  Total terminal frames: {total} ({total / FPS:.1f}s)")

    print(f"  Rendering {total} frames at {W}x{H}...")
    for i, (lines, visible_chars, cursor_on) in enumerate(all_frames):
        if i % 100 == 0:
            print(f"    Frame {i}/{total} ({i * 100 // total}%)...", flush=True)

        img = render_terminal_frame(visible_chars, lines, cursor_on, i)

        if i < 20:
            opacity = i / 20.0
            arr = np.array(img, dtype=np.float32)
            bg = np.array(NUIT, dtype=np.float32)[None, None, :]
            img = Image.fromarray((arr * opacity + bg * (1 - opacity)).clip(0, 255).astype(np.uint8))
        if i >= total - 30:
            remaining = total - 1 - i
            opacity = remaining / 30.0
            arr = np.array(img, dtype=np.float32)
            bg = np.array(NUIT, dtype=np.float32)[None, None, :]
            img = Image.fromarray((arr * opacity + bg * (1 - opacity)).clip(0, 255).astype(np.uint8))

        path = FRAME_DIR / f"frame_{i + 1:05d}.png"
        img.save(str(path))

    print(f"  Done. Output: {FRAME_DIR}")
    return total


if __name__ == "__main__":
    import shutil
    import time

    if FRAME_DIR.exists():
        shutil.rmtree(str(FRAME_DIR))

    t0 = time.time()
    n = build_all_terminal_frames()
    elapsed = time.time() - t0
    print(f"\nRendered {n} frames in {elapsed:.1f}s ({elapsed / max(1, n) * 1000:.0f}ms/frame)")
