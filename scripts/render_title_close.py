#!/usr/bin/env python3
"""
Render title card and closing card for the definitive hackathon video.

Output:
  assets/video/hackathon_title.png   (1920x1080)
  assets/video/hackathon_close.png   (1920x1080)

Usage:
    python scripts/render_title_close.py
"""

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "assets" / "video"
OUT.mkdir(parents=True, exist_ok=True)

# Palette
BG = (8, 6, 32)          # NUIT
GOLD = (220, 201, 100)
WHITE = (240, 232, 208)   # AVORIO
DIM = (130, 126, 138)
FCC_RED = (179, 68, 68)


def get_font(size, style="sans"):
    candidates = {
        "sans": ["C:/Windows/Fonts/bahnschrift.ttf", "C:/Windows/Fonts/segoeui.ttf"],
        "serif": ["C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/times.ttf"],
        "mono": ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"],
    }
    for path in candidates.get(style, candidates["sans"]):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def text_center(draw, text, y, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2
    draw.text((x, y), text, font=font, fill=fill)


def render_title():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Main title
    f_main = get_font(72, "sans")
    f_sub = get_font(36, "sans")
    f_tag = get_font(24, "mono")

    text_center(draw, "rhombic-agent", 340, f_main, GOLD)
    text_center(draw, "Keep Your Cube, Add Six Bridges.", 440, f_sub, WHITE)

    # Bottom tag
    text_center(draw, "@NousResearch  #HermesAgentHackathon", 680, f_tag, DIM)

    path = OUT / "hackathon_title.png"
    img.save(str(path), "PNG")
    print(f"  Title card: {path}")
    return path


def render_close():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    f_big = get_font(48, "sans")
    f_med = get_font(32, "sans")
    f_mono = get_font(28, "mono")
    f_small = get_font(24, "sans")

    # pip install
    text_center(draw, "pip install rhombic", 200, f_mono, GOLD)

    # Stats line
    text_center(draw, "v0.3.0  ·  255 tests  ·  3 papers  ·  MPL-2.0", 280, f_small, DIM)

    # Links
    y = 380
    for link in [
        "tasumermaf.github.io/rhombic",
        "github.com/tasumermaf/rhombic",
        "pypi.org/project/rhombic",
    ]:
        text_center(draw, link, y, f_mono, WHITE)
        y += 50

    # Credit
    text_center(draw, "Built with Hermes Agent", 580, f_med, FCC_RED)
    text_center(draw, "@NousResearch  #HermesAgentHackathon", 630, f_small, DIM)

    # Team
    text_center(draw, "Timothy Paul Bielec  ×  Minta Carlson", 730, f_med, WHITE)
    text_center(draw, "TASUMER MAF", 780, f_small, DIM)

    path = OUT / "hackathon_close.png"
    img.save(str(path), "PNG")
    print(f"  Close card: {path}")
    return path


if __name__ == "__main__":
    print("Rendering hackathon cards...")
    render_title()
    render_close()
    print("Done.")
