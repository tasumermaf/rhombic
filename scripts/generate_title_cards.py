#!/usr/bin/env python3
"""Generate title card and overlay assets for the hackathon demo video."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ── Palette ──
NUIT = (8, 6, 32)
FCC = (179, 68, 68)
CUBIC = (61, 61, 107)
GOLD = (220, 201, 100)
TEXT = (232, 228, 223)
DIM = (153, 149, 160)

W, H = 1920, 1080
OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'video')
os.makedirs(OUT, exist_ok=True)


def get_font(size, bold=False):
    """Try to load a good font, fall back to default."""
    candidates = [
        'C:/Windows/Fonts/georgia.ttf' if not bold else 'C:/Windows/Fonts/georgiab.ttf',
        'C:/Windows/Fonts/times.ttf' if not bold else 'C:/Windows/Fonts/timesbd.ttf',
        'C:/Windows/Fonts/arial.ttf' if not bold else 'C:/Windows/Fonts/arialbd.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_centered(draw, text, y, font, fill=TEXT):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, y), text, font=font, fill=fill)


def title_card():
    """Opening title card."""
    img = Image.new('RGB', (W, H), NUIT)
    draw = ImageDraw.Draw(img)

    title_font = get_font(72, bold=True)
    sub_font = get_font(36)
    tag_font = get_font(28)

    draw_centered(draw, "rhombic-agent", 340, title_font, FCC)
    draw_centered(draw, "Keep Your Cube, Add Six Bridges.", 440, sub_font, TEXT)
    draw_centered(draw, "9 tools  \u00b7  3 papers  \u00b7  256 tests  \u00b7  0 competitors", 520, tag_font, DIM)

    # Gold accent line
    draw.line([(W//2 - 200, 500), (W//2 + 200, 500)], fill=GOLD, width=2)

    img.save(os.path.join(OUT, 'title_card.png'), quality=95)
    print(f"Saved: title_card.png ({W}x{H})")


def closing_card():
    """Closing card with links."""
    img = Image.new('RGB', (W, H), NUIT)
    draw = ImageDraw.Draw(img)

    big_font = get_font(48, bold=True)
    cmd_font = get_font(40)
    link_font = get_font(28)
    tag_font = get_font(24)

    draw_centered(draw, "pip install rhombic", 300, cmd_font, FCC)

    # Gold line
    draw.line([(W//2 - 300, 370), (W//2 + 300, 370)], fill=GOLD, width=2)

    draw_centered(draw, "promptcrafted.github.io/rhombic", 400, link_font, TEXT)
    draw_centered(draw, "github.com/promptcrafted/rhombic", 445, link_font, DIM)
    draw_centered(draw, "huggingface.co/spaces/timotheospaul/rhombic", 490, link_font, DIM)

    # Tagline
    draw_centered(draw, "256 tests. 3 papers. 0 competitors.", 580, big_font, GOLD)

    # Attribution
    draw_centered(draw, "Built by Promptcrafted  \u00b7  @NousResearch #HermesAgentHackathon", 700, tag_font, DIM)

    img.save(os.path.join(OUT, 'closing_card.png'), quality=95)
    print(f"Saved: closing_card.png ({W}x{H})")


def overlay_text(text, filename, color=TEXT, size=36):
    """Generate a text overlay PNG with transparent background."""
    font = get_font(size, bold=True)
    # Measure text
    tmp = Image.new('RGBA', (1, 1))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0] + 40, bbox[3] - bbox[1] + 20

    img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((20, 10), text, font=font, fill=color + (255,))

    img.save(os.path.join(OUT, filename))
    print(f"Saved: {filename} ({tw}x{th})")


def section_divider(text, filename):
    """Full-width section divider."""
    img = Image.new('RGB', (W, 120), NUIT)
    draw = ImageDraw.Draw(img)
    font = get_font(32, bold=True)

    # Gold lines
    draw.line([(100, 60), (W//2 - 200, 60)], fill=GOLD, width=1)
    draw.line([(W//2 + 200, 60), (W - 100, 60)], fill=GOLD, width=1)

    draw_centered(draw, text, 40, font, GOLD)
    img.save(os.path.join(OUT, filename), quality=95)
    print(f"Saved: {filename} ({W}x120)")


if __name__ == '__main__':
    print("Generating video assets...")

    title_card()
    closing_card()

    # Text overlays for key moments
    overlay_text("Every computation defaults to cubic.", "overlay_default.png", TEXT, 32)
    overlay_text("The FCC lattice: 12 neighbors. 6 directions.", "overlay_fcc.png", FCC, 32)
    overlay_text("Not noise. Structure.", "overlay_structure.png", GOLD, 40)
    overlay_text("The advantage amplifies under stress.", "overlay_amplifies.png", FCC, 32)
    overlay_text("Keep your cube, add six bridges.", "overlay_tagline.png", TEXT, 36)

    # Section dividers
    section_divider("THE COMPARISON", "divider_comparison.png")
    section_divider("THE PROOF", "divider_proof.png")
    section_divider("THE VISION", "divider_vision.png")

    print("\nAll video assets generated in assets/video/")
