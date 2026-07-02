#!/usr/bin/env python3
"""
Episode 2 Terminal Renderer — Nerd Register Interactions.

Terminal window, Consolas mono, green prompt, ivory text, typing animation,
inline visualizations. The Hermes Agent Nerd's sessions.

Terminals:
  1. "Show me the bridge anatomy for Qwen-7B" — 6x6 heatmap, co/cross breakdown (20s)
  2. "What does the bridge look like at 14B?" — Holly Battery reveal (22s)
  3. "Run task fingerprint on Holly bridge" — spectral barcode (20s)
  4. "Compare 6-channel FCC vs 3-channel cubic" — 4.6x Fiedler advantage (16s)
  5. "What is TASUMER MAF?" — kybernetes = 1093 = steersman. The Nerd gets it (12s)

Output: assets/video/frames_ep2/terminals/<terminal_N>/frame_NNNNN.png

Usage:
    python scripts/render_ep2_terminals.py
    python scripts/render_ep2_terminals.py --preview
"""

from __future__ import annotations

import io
import math
import os
import sys
import textwrap
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
FRAME_DIR = BASE / "assets" / "video" / "frames_ep2" / "terminals"

# ===================================================================
# Palette (Nerd register — from Ep1 terminal)
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
GOLD_RGB = (220, 201, 100)
SCANLINE_ALPHA = 0.03

# Heatmap colors for bridge anatomy
HEAT_COLD = (30, 30, 80)   # Cross-planar (near-zero)
HEAT_HOT = (200, 60, 60)   # Co-planar (dominant)
HEAT_MED = (100, 60, 120)  # Moderate

# ===================================================================
# Font System (from Ep1 terminal)
# ===================================================================

_font_cache = {}

def get_font(size, style="mono"):
    key = (size, style)
    if key in _font_cache:
        return _font_cache[key]
    candidates = {
        "mono": ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"],
        "mono_bold": ["C:/Windows/Fonts/consolab.ttf", "C:/Windows/Fonts/courbd.ttf"],
        "sans": ["C:/Windows/Fonts/bahnschrift.ttf", "C:/Windows/Fonts/segoeui.ttf"],
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

def scale(v):
    return int(v * W / 3840)

# ===================================================================
# Terminal Layout (from Ep1)
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

def char_size(font):
    bbox = font.getbbox("M")
    cw = bbox[2] - bbox[0]
    ch = bbox[3] - bbox[1]
    return cw, int(ch * 1.45)

# ===================================================================
# Terminal Chrome + Effects (from Ep1)
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
    title = "hermes-agent — episode 2"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    tx = cx + cw // 2 - tw // 2
    draw.text((tx, dot_y - scale(14)), title, fill=DIM_COLOR, font=title_font)

def draw_scanlines(img):
    arr = np.array(img, dtype=np.float32)
    step = 3 if PREVIEW else 4
    for y in range(0, H, step):
        arr[y, :, :] *= (1.0 - SCANLINE_ALPHA)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def apply_vignette(img):
    arr = np.array(img, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    cx_f, cy_f = W / 2, H / 2
    dist = np.sqrt((xx - cx_f) ** 2 + (yy - cy_f) ** 2)
    max_dist = math.sqrt(cx_f ** 2 + cy_f ** 2)
    vignette = 1.0 - 0.25 * (dist / max_dist) ** 1.5
    arr *= vignette[:, :, None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

# ===================================================================
# Inline Visualizations
# ===================================================================

def render_heatmap_6x6(draw, x, y, w, h, progress):
    """Bridge anatomy heatmap: 6x6, co-planar hot / cross-planar cold.

    Shows the massive asymmetry between co-planar and cross-planar bridge components.
    """
    font = get_font(scale(32), "mono")
    title_font = get_font(scale(40), "mono_bold")

    # Title
    draw.text((x + scale(20), y + scale(10)), "Bridge Anatomy — Qwen 7B", fill=HEADER_COLOR, font=title_font)

    # 6x6 grid representing 6 FCC directions
    labels = ["[100]", "[010]", "[001]", "[110]", "[101]", "[011]"]
    cell_size = min(scale(120), (w - scale(200)) // 6)
    grid_x = x + scale(180)
    grid_y = y + scale(80)

    # Asymmetry data (normalized): diagonal=co-planar=hot, off-diagonal=cross-planar=cold
    rng = np.random.RandomState(42)
    for row in range(6):
        # Row label
        if progress > row * 0.1:
            draw.text((x + scale(20), grid_y + row * cell_size + cell_size // 3),
                      labels[row], fill=DIM_COLOR, font=font)

        for col in range(6):
            cell_progress = min(1.0, (progress - (row + col) * 0.04) / 0.3)
            if cell_progress <= 0:
                continue

            cx = grid_x + col * cell_size
            cy = grid_y + row * cell_size

            # Co-planar (diagonal): very hot. Cross-planar: near zero
            if row == col:
                heat = 0.85 + rng.uniform(0, 0.15)  # Very hot
            elif abs(row - col) == 3:
                heat = 0.001 + rng.uniform(0, 0.005)  # Essentially zero
            else:
                heat = 0.01 + rng.uniform(0, 0.03)  # Near zero

            heat *= cell_progress

            # Color interpolation
            r = int(HEAT_COLD[0] + (HEAT_HOT[0] - HEAT_COLD[0]) * heat)
            g = int(HEAT_COLD[1] + (HEAT_HOT[1] - HEAT_COLD[1]) * heat)
            b = int(HEAT_COLD[2] + (HEAT_HOT[2] - HEAT_COLD[2]) * heat)

            draw.rectangle([cx, cy, cx + cell_size - 2, cy + cell_size - 2], fill=(r, g, b))

    # Column labels
    if progress > 0.3:
        for col in range(6):
            cx = grid_x + col * cell_size
            draw.text((cx, grid_y - scale(30)), labels[col], fill=DIM_COLOR, font=get_font(scale(26), "mono"))

    # Annotation
    if progress > 0.7:
        ann_y = grid_y + 6 * cell_size + scale(20)
        draw.text((x + scale(20), ann_y), "Co-planar: 99.996%", fill=HEAT_HOT, font=font)
        draw.text((x + scale(20), ann_y + scale(40)), "Cross-planar: 0.004%", fill=TOOL_COLOR, font=font)
        draw.text((x + scale(20), ann_y + scale(80)), "Ratio: 22,477 : 1", fill=NUM_COLOR, font=get_font(scale(44), "mono_bold"))


def render_fingerprint(draw, x, y, w, h, progress):
    """Task fingerprint spectral barcode — 36 bridge eigenvalues as vertical bars."""
    font = get_font(scale(32), "mono")
    title_font = get_font(scale(40), "mono_bold")

    draw.text((x + scale(20), y + scale(10)), "Bridge Fingerprint — Holly Battery", fill=HEADER_COLOR, font=title_font)

    # 36 eigenvalue bars
    bar_area_x = x + scale(80)
    bar_area_w = w - scale(160)
    bar_area_y = y + scale(80)
    bar_area_h = h - scale(160)
    n_bars = 36

    rng = np.random.RandomState(84)
    # Eigenvalue profile: dominant first few, rapid decay
    eigenvalues = [0.95 * (0.85 ** i) + rng.uniform(0, 0.05) for i in range(n_bars)]

    for i in range(n_bars):
        bar_progress = min(1.0, (progress - i * 0.02) / 0.3)
        if bar_progress <= 0:
            continue

        bar_x = bar_area_x + int(bar_area_w * i / n_bars)
        bar_w = max(scale(4), bar_area_w // n_bars - scale(4))
        bar_h = int(bar_area_h * eigenvalues[i] * bar_progress)

        # Color gradient: first few dominant bars in red, rest in blue
        if i < 5:
            color = NUM_COLOR
        elif i < 15:
            color = TOOL_COLOR
        else:
            color = (50, 48, 80)

        draw.rectangle([bar_x, bar_area_y + bar_area_h - bar_h,
                        bar_x + bar_w, bar_area_y + bar_area_h], fill=color)

    # Labels
    if progress > 0.5:
        draw.text((bar_area_x, bar_area_y + bar_area_h + scale(15)),
                  "36 parameters for 14 billion", fill=NUM_COLOR, font=font)
    if progress > 0.7:
        draw.text((bar_area_x, bar_area_y + bar_area_h + scale(55)),
                  "72.3% task identification accuracy", fill=GOLD_RGB, font=font)


def render_fcc_vs_cubic(draw, x, y, w, h, progress):
    """FCC vs Cubic connectivity comparison: 12 vs 6 neighbors."""
    font = get_font(scale(36), "mono")
    title_font = get_font(scale(40), "mono_bold")
    num_font = get_font(scale(60), "mono_bold")

    draw.text((x + scale(20), y + scale(10)), "6-Channel FCC vs 3-Channel Cubic",
              fill=HEADER_COLOR, font=title_font)

    cx_left = x + w // 4
    cx_right = x + 3 * w // 4
    node_y = y + h // 2

    # Cubic: 6 connections (3 axes × 2 directions)
    if progress > 0.1:
        node_r = scale(30)
        draw.ellipse([cx_left - node_r, node_y - node_r, cx_left + node_r, node_y + node_r],
                     fill=TOOL_COLOR)
        draw.text((cx_left - scale(40), y + scale(70)), "Cubic", fill=TOOL_COLOR, font=font)
        draw.text((cx_left - scale(60), y + scale(110)), "6 neighbors", fill=DIM_COLOR, font=font)

        # 6 connection lines
        for i in range(6):
            angle = i * math.pi / 3
            line_progress = min(1.0, (progress - 0.1 - i * 0.03) / 0.15)
            if line_progress <= 0:
                continue
            length = scale(120) * line_progress
            ex = cx_left + int(math.cos(angle) * length)
            ey = node_y + int(math.sin(angle) * length)
            draw.line([(cx_left, node_y), (ex, ey)], fill=TOOL_COLOR, width=scale(3))

    # FCC: 12 connections (6 axes × 2 directions)
    if progress > 0.35:
        node_r = scale(30)
        draw.ellipse([cx_right - node_r, node_y - node_r, cx_right + node_r, node_y + node_r],
                     fill=NUM_COLOR)
        draw.text((cx_right - scale(20), y + scale(70)), "FCC", fill=NUM_COLOR, font=font)
        draw.text((cx_right - scale(70), y + scale(110)), "12 neighbors", fill=DIM_COLOR, font=font)

        # 12 connection lines
        for i in range(12):
            angle = i * math.pi / 6
            line_progress = min(1.0, (progress - 0.35 - i * 0.02) / 0.15)
            if line_progress <= 0:
                continue
            length = scale(120) * line_progress
            ex = cx_right + int(math.cos(angle) * length)
            ey = node_y + int(math.sin(angle) * length)
            draw.line([(cx_right, node_y), (ex, ey)], fill=NUM_COLOR, width=scale(3))

    # Result
    if progress > 0.7:
        result_y = y + h - scale(100)
        draw.text((x + scale(20), result_y), "Fiedler advantage: 4.6x", fill=NUM_COLOR, font=num_font)
    if progress > 0.85:
        draw.text((x + scale(20), y + h - scale(40)), "More roads. Same land.", fill=GOLD_RGB, font=font)


# ===================================================================
# Interaction Scripts
# ===================================================================

INTERACTIONS = [
    {
        "prompt": "Show me the bridge anatomy for Qwen-7B.",
        "tool": "bridge_anatomy(model='qwen-7b')",
        "response": [
            ("header", "Bridge Anatomy - Qwen 7B"),
            ("blank", ""),
            ("row", "Total bridge parameters:   2,048"),
            ("row", "Co-planar component:       2,047.91"),
            ("row", "Cross-planar component:    0.09"),
            ("blank", ""),
            ("num", "Axis asymmetry ratio: 22,477 : 1"),
            ("blank", ""),
            ("text", "The bridge is almost entirely co-planar."),
            ("text", "Cross-planar flow: effectively zero."),
            ("num", "Rivers vs trickles."),
            ("dim", "Paper 3, Figure 2"),
        ],
        "viz": "heatmap",
        "viz_renderer": render_heatmap_6x6,
        "duration": 20.0,
    },
    {
        "prompt": "What does the bridge look like at 14B?",
        "tool": "bridge_profile(model='qwen-14b', variant='holly')",
        "response": [
            ("header", "Holly Battery - Qwen 14B"),
            ("blank", ""),
            ("row", "Training loss:    3.8% better"),
            ("row", "VRAM usage:       9.15 GB less"),
            ("row", "Training speed:   6% faster"),
            ("row", "Checkpoint size:  50% smaller"),
            ("blank", ""),
            ("num", "Axis ratio at 14B: 47,145 : 1"),
            ("blank", ""),
            ("text", "The asymmetry INCREASES with scale."),
            ("text", "Larger models have sharper bridges."),
            ("num", "First genuine surprise."),
            ("dim", "Paper 3, Table 3"),
        ],
        "viz": None,
        "duration": 22.0,
    },
    {
        "prompt": "Run task fingerprint on Holly bridge.",
        "tool": "task_fingerprint(bridge='holly', method='eigendecomp')",
        "response": [
            ("header", "Task Fingerprint - Holly Bridge"),
            ("blank", ""),
            ("row", "Bridge rank:         36"),
            ("row", "Model parameters:    14,167,000,000"),
            ("row", "Identification acc:  72.3%"),
            ("blank", ""),
            ("text", "36 bridge eigenvalues uniquely identify"),
            ("text", "the downstream task from the LoRA merge."),
            ("blank", ""),
            ("num", "36 parameters for 14 billion."),
            ("text", "The bridge carries the task signature."),
            ("dim", "Paper 3, Section 4.2"),
        ],
        "viz": "fingerprint",
        "viz_renderer": render_fingerprint,
        "duration": 20.0,
    },
    {
        "prompt": "Compare 6-channel FCC vs 3-channel cubic.",
        "tool": "channel_compare(fcc_channels=6, cubic_channels=3)",
        "response": [
            ("header", "Channel Comparison"),
            ("blank", ""),
            ("row", "Cubic (3-channel):  Fiedler = 0.096"),
            ("row", "FCC (6-channel):    Fiedler = 0.441"),
            ("blank", ""),
            ("num", "Advantage: 4.6x"),
            ("blank", ""),
            ("text", "Zero additional parameters."),
            ("text", "Same weight budget. More paths."),
            ("num", "More roads. Same land."),
            ("dim", "Paper 3, Section 5.1"),
        ],
        "viz": "fcc_vs_cubic",
        "viz_renderer": render_fcc_vs_cubic,
        "duration": 16.0,
    },
    {
        "prompt": "What is TASUMER MAF?",
        "tool": "isopsephy(text='TASUMER MAF')",
        "response": [
            ("header", "TASUMER MAF"),
            ("blank", ""),
            ("tool", "Computing isopsephic value..."),
            ("blank", ""),
            ("row", "T(300) A(1) S(200) U(400) M(40) E(5) R(100)"),
            ("row", "M(40) A(1) F(6)"),
            ("blank", ""),
            ("num", "TASUMER MAF = 1093"),
            ("blank", ""),
            ("text", "1093 = prime (the 183rd)"),
            ("blank", ""),
            ("row", "Greek correspondence:"),
            ("num", "kybernetes = 1093"),
            ("blank", ""),
            ("num", "The steersman."),
        ],
        "viz": None,
        "duration": 12.0,
    },
]

# ===================================================================
# Frame Builder (adapted from Ep1)
# ===================================================================

def build_interaction_frames(interaction):
    """Build frame sequence for a terminal interaction."""
    prompt = interaction["prompt"]
    tool = interaction["tool"]
    response = interaction["response"]
    viz_type = interaction.get("viz")
    viz_renderer = interaction.get("viz_renderer")

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

    # Phase 5: Hold
    hold_frames = int(FPS * 1.5)
    for f in range(hold_frames):
        cursor_blink = (f % 24) < 12
        frames.append((list(response_lines), base_chars + 999, cursor_blink))

    # Phase 6: Visualization
    if viz_type and viz_renderer:
        viz_tool_line = ("tool", f"[calling visualize_{viz_type}()]")
        viz_announce_lines = list(response_lines) + [("blank", ""), viz_tool_line]
        for f in range(int(FPS * 0.6)):
            frames.append((viz_announce_lines, base_chars + 999, (f % 20) < 10))

        # Transition
        for f in range(int(FPS * 0.4)):
            frames.append(([], 0, False))

        # Viz animation (3.0s)
        viz_frames = int(FPS * 3.0)
        for f in range(viz_frames):
            progress = f / max(1, viz_frames - 1)
            frames.append(([("__VIZ__", viz_type, progress, viz_renderer)], 0, False))

        # Viz hold (2.0s)
        for f in range(int(FPS * 2.0)):
            frames.append(([("__VIZ__", viz_type, 1.0, viz_renderer)], 0, False))

    return frames


def render_terminal_frame(visible_chars, lines, cursor_on, frame_idx):
    """Render a single terminal frame."""
    img = Image.new("RGB", (W, H), NUIT)
    draw_chrome(img)
    draw = ImageDraw.Draw(img)

    tx, ty, tw, th = term_rect()
    draw.rectangle([tx, ty, tx + tw, ty + th], fill=TERM_BG)

    # Viz mode
    if lines and len(lines) == 1 and lines[0][0] == "__VIZ__":
        viz_type = lines[0][1]
        viz_progress = lines[0][2]
        viz_renderer = lines[0][3]
        if viz_renderer:
            viz_renderer(draw, tx, ty, tw, th, viz_progress)
        img = draw_scanlines(img)
        return apply_vignette(img)

    # Normal text mode
    FONT_SIZE = scale(56)
    font = get_font(FONT_SIZE, "mono")
    font_bold = get_font(FONT_SIZE, "mono_bold")
    cw, lh = char_size(font)

    pad_x = scale(36)
    pad_y = scale(28)
    max_cpl = max(20, (tw - 2 * pad_x) // cw)

    display_lines = []
    for style, text in lines:
        if style == "blank":
            display_lines.append((style, "", False))
        elif style == "prompt":
            display_lines.append((style, text, False))
        else:
            wrapped = textwrap.wrap(text, width=max_cpl) if len(text) > max_cpl else [text]
            for i, wl in enumerate(wrapped):
                display_lines.append((style, wl, i > 0))

    char_count = 0
    last_x = tx + pad_x
    last_y = ty + pad_y

    style_colors = {
        "prompt": PROMPT_COLOR,
        "tool": TOOL_COLOR,
        "header": HEADER_COLOR,
        "num": NUM_COLOR,
        "dim": DIM_COLOR,
        "row": AGENT_COLOR,
        "text": AGENT_COLOR,
    }

    for dl_idx, (style, text, is_cont) in enumerate(display_lines):
        line_y = ty + pad_y + dl_idx * lh
        if line_y + lh > ty + th - pad_y:
            break

        if style == "blank":
            char_count += 1
            if char_count > visible_chars:
                last_x = tx + pad_x
                last_y = line_y
                break
            continue

        if style == "prompt":
            prompt_str = "> "
            draw.text((tx + pad_x, line_y), prompt_str, fill=PROMPT_COLOR, font=font_bold)
            text_start_x = tx + pad_x + len(prompt_str) * cw
            color = USER_COLOR
            use_font = font_bold
        else:
            text_start_x = tx + pad_x + (scale(20) if is_cont else 0)
            color = style_colors.get(style, AGENT_COLOR)
            use_font = font_bold if style in ("header", "num") else font

        for ci, ch_char in enumerate(text):
            char_count += 1
            if char_count > visible_chars:
                last_x = text_start_x + ci * cw
                last_y = line_y
                break
            xp = text_start_x + ci * cw
            draw.text((xp, line_y), ch_char, fill=color, font=use_font)
            last_x = xp + cw
            last_y = line_y
        else:
            char_count += 1
            if char_count > visible_chars:
                last_x = text_start_x + len(text) * cw
                last_y = line_y
                break
            continue
        if char_count > visible_chars:
            break

    # Cursor
    if cursor_on:
        cursor_x = min(last_x, tx + tw - pad_x - cw)
        draw.rectangle([cursor_x, last_y, cursor_x + cw, last_y + lh - scale(4)], fill=CURSOR_COLOR)

    img = draw_scanlines(img)
    return apply_vignette(img)


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("  EPISODE 2 TERMINAL RENDERER — Nerd Register")
    print(f"  Resolution: {W}x{H} @ {FPS}fps")
    print("=" * 60)

    bg = np.array(NUIT, dtype=np.float32)[None, None, :]

    for idx, interaction in enumerate(INTERACTIONS):
        name = f"terminal_{idx + 1}"
        out_dir = FRAME_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt_short = interaction["prompt"][:45]
        print(f"\n  Terminal {idx + 1}: '{prompt_short}'...", flush=True)

        seq_frames = build_interaction_frames(interaction)

        # Intro + outro padding
        intro = [([], 0, (f % 24) < 12) for f in range(int(FPS * 0.5))]
        outro = []
        if seq_frames:
            last = seq_frames[-1]
            outro = [(last[0], last[1], (f % 24) < 12) for f in range(int(FPS * 1.0))]

        all_frames = intro + seq_frames + outro
        total = len(all_frames)
        print(f"    {total} frames ({total/FPS:.1f}s)")

        for i, (lines, vis_chars, cursor_on) in enumerate(all_frames):
            if i % 100 == 0:
                print(f"    Frame {i}/{total} ({i*100//total}%)", flush=True)

            img = render_terminal_frame(vis_chars, lines, cursor_on, i)

            # Fade in/out
            if i < 15:
                arr = np.array(img, dtype=np.float32)
                img = Image.fromarray((arr * (i / 15.0) + bg * (1 - i / 15.0)).clip(0, 255).astype(np.uint8))
            if i >= total - 15:
                r = max(0, (total - 1 - i) / 15.0)
                arr = np.array(img, dtype=np.float32)
                img = Image.fromarray((arr * r + bg * (1 - r)).clip(0, 255).astype(np.uint8))

            path = out_dir / f"frame_{i + 1:05d}.png"
            img.save(str(path))

        print(f"    -> {total} frames ({total/FPS:.1f}s)")

    print("\n" + "=" * 60)
    print("  TERMINALS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
