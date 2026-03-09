#!/usr/bin/env python3
"""Render captured Hermes tool outputs as terminal-style video frames.

Takes JSON captures from capture_hermes.py and produces numbered PNG frames
simulating a dark terminal session. Reuses the 8-Law Weave palette from
generate_title_cards.py for visual consistency.

Output: assets/video/frames/frame_NNNNN.png (30fps, 1920x1080)
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Palette (matches title cards) ──
NUIT = (8, 6, 32)
FCC = (179, 68, 68)
CUBIC = (61, 61, 107)
GOLD = (220, 201, 100)
TEXT = (232, 228, 223)
DIM = (153, 149, 160)
GREEN = (100, 200, 120)
CYAN = (120, 200, 220)

W, H = 1920, 1080
FPS = 30

# ── Paths ──
BASE = Path(__file__).resolve().parent.parent
CAPTURES = BASE / "assets" / "video" / "captures"
FRAMES = BASE / "assets" / "video" / "frames"
ASSETS = BASE / "assets" / "video"
OVERLAYS = BASE / "assets" / "video"


# ── Fonts ──
def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/consola.ttf" if not bold else "C:/Windows/Fonts/consolab.ttf",
        "C:/Windows/Fonts/CascadiaCode.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_MAIN = get_font(20)
FONT_BOLD = get_font(20, bold=True)
FONT_PROMPT = get_font(22, bold=True)
FONT_TITLE = get_font(16)
FONT_SMALL = get_font(16)


class FrameRenderer:
    """Renders sequential frames for the terminal video."""

    def __init__(self):
        self.frame_num = 0
        FRAMES.mkdir(parents=True, exist_ok=True)

    def save_frame(self, img: Image.Image) -> int:
        """Save frame and return its number."""
        self.frame_num += 1
        path = FRAMES / f"frame_{self.frame_num:05d}.png"
        img.save(str(path))
        return self.frame_num

    def make_terminal_base(self) -> Image.Image:
        """Create a blank terminal frame with chrome."""
        img = Image.new("RGB", (W, H), NUIT)
        draw = ImageDraw.Draw(img)

        # Terminal title bar
        draw.rectangle([0, 0, W, 36], fill=(20, 18, 48))
        # Window dots
        for i, color in enumerate([(255, 95, 87), (255, 189, 46), (39, 201, 63)]):
            draw.ellipse([16 + i * 24, 10, 30 + i * 24, 24], fill=color)
        # Title text
        draw.text((W // 2 - 100, 8), "hermes — rhombic-agent", font=FONT_TITLE, fill=DIM)

        # Subtle border at bottom of title bar
        draw.line([(0, 36), (W, 36)], fill=(40, 38, 68), width=1)

        return img

    def draw_text_block(
        self,
        draw: ImageDraw.ImageDraw,
        lines: list[str],
        start_y: int = 56,
        line_height: int = 28,
        colors: Optional[list[tuple]] = None,
        fonts: Optional[list] = None,
    ) -> int:
        """Draw text lines and return the y position after the last line."""
        y = start_y
        left_margin = 32
        for i, line in enumerate(lines):
            color = colors[i] if colors and i < len(colors) else TEXT
            font = fonts[i] if fonts and i < len(fonts) else FONT_MAIN
            draw.text((left_margin, y), line, font=font, fill=color)
            y += line_height
        return y

    # ── Scene builders ──

    def render_static(self, img: Image.Image, duration_s: float) -> None:
        """Render a static image for the given duration."""
        n_frames = int(duration_s * FPS)
        for _ in range(n_frames):
            self.save_frame(img)

    def render_image_file(self, path: str | Path, duration_s: float) -> None:
        """Render an image file scaled to 1920x1080 for the given duration."""
        img = Image.open(str(path)).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), Image.Resampling.LANCZOS)
        self.render_static(img, duration_s)

    def render_typing(
        self,
        base_img: Image.Image,
        prompt_text: str,
        y_pos: int,
        chars_per_frame: int = 3,
        prefix: str = "hermes> ",
    ) -> Image.Image:
        """Render typing animation for a prompt. Returns final frame."""
        left = 32
        full = prefix + prompt_text
        final_img = None
        for i in range(0, len(full), chars_per_frame):
            img = base_img.copy()
            draw = ImageDraw.Draw(img)
            partial = full[: i + chars_per_frame]
            # Prefix in green, rest in gold
            draw.text((left, y_pos), prefix, font=FONT_PROMPT, fill=GREEN)
            prefix_w = draw.textlength(prefix, font=FONT_PROMPT)
            typed = partial[len(prefix) :]
            draw.text((left + prefix_w, y_pos), typed, font=FONT_PROMPT, fill=GOLD)
            # Cursor
            cursor_x = left + draw.textlength(partial, font=FONT_PROMPT)
            draw.rectangle([cursor_x, y_pos, cursor_x + 12, y_pos + 24], fill=TEXT)
            final_img = img
            self.save_frame(img)
        return final_img

    def render_tool_call(
        self,
        base_img: Image.Image,
        tool_name: str,
        y_pos: int,
        duration_s: float = 0.8,
    ) -> Image.Image:
        """Render a tool call indicator with brief spinner."""
        n_frames = int(duration_s * FPS)
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        final_img = None
        for f in range(n_frames):
            img = base_img.copy()
            draw = ImageDraw.Draw(img)
            s = spinners[f % len(spinners)]
            draw.text(
                (32, y_pos),
                f"  {s} calling {tool_name}...",
                font=FONT_MAIN,
                fill=CYAN,
            )
            final_img = img
            self.save_frame(img)
        return final_img

    def render_response_lines(
        self,
        base_img: Image.Image,
        lines: list[str],
        colors: list[tuple],
        y_start: int,
        line_delay_frames: int = 3,
        line_height: int = 26,
    ) -> Image.Image:
        """Render response text appearing line by line. Returns final frame."""
        left = 32
        current_img = base_img.copy()
        for i, (line, color) in enumerate(zip(lines, colors)):
            draw = ImageDraw.Draw(current_img)
            draw.text((left, y_start + i * line_height), line, font=FONT_MAIN, fill=color)
            for _ in range(line_delay_frames):
                self.save_frame(current_img)
        # Hold final state
        for _ in range(FPS):  # 1 second hold
            self.save_frame(current_img)
        return current_img

    def render_overlay(
        self,
        base_img: Image.Image,
        text: str,
        duration_s: float = 2.0,
        color: tuple = TEXT,
    ) -> None:
        """Render text overlay at bottom of frame."""
        n_frames = int(duration_s * FPS)
        font = get_font(28, bold=True)
        for f in range(n_frames):
            img = base_img.copy()
            draw = ImageDraw.Draw(img)
            # Semi-transparent bar at bottom
            draw.rectangle([0, H - 80, W, H], fill=(8, 6, 32))
            draw.rectangle([0, H - 80, W, H - 78], fill=GOLD)
            # Center text
            tw = draw.textlength(text, font=font)
            draw.text(((W - tw) // 2, H - 65), text, font=font, fill=color)
            # Fade in/out
            alpha = 1.0
            if f < 10:
                alpha = f / 10.0
            elif f > n_frames - 10:
                alpha = (n_frames - f) / 10.0
            if alpha < 1.0:
                arr = np.array(img, dtype=np.float32)
                base_arr = np.array(base_img, dtype=np.float32)
                blended = base_arr * (1 - alpha) + arr * alpha
                img = Image.fromarray(blended.astype(np.uint8))
            self.save_frame(img)


def format_lattice_compare(data: dict) -> tuple[list[str], list[tuple]]:
    """Format lattice compare results as terminal output lines."""
    c = data["cubic"]
    f = data["fcc"]
    lines = [
        "",
        "  ┌─────────────────────────────────────────────────────┐",
        "  │              Lattice Comparison (scale 5)            │",
        "  ├─────────────────┬──────────────┬─────────────────────┤",
        "  │ Metric          │    Cubic     │        FCC          │",
        "  ├─────────────────┼──────────────┼─────────────────────┤",
        f"  │ Nodes           │   {c['nodes']:>6}     │      {f['nodes']:>6}           │",
        f"  │ Edges           │   {c['edges']:>6}     │      {f['edges']:>6}           │",
        f"  │ Connectivity    │   {c['connectivity']:>6}     │      {f['connectivity']:>6}           │",
        f"  │ Avg Degree      │   {c['avg_degree']:>6.1f}     │      {f['avg_degree']:>6.2f}           │",
        f"  │ Avg Path Length │   {c['avg_path_length']:>6.3f}     │      {f['avg_path_length']:>6.4f}           │",
        "  └─────────────────┴──────────────┴─────────────────────┘",
        "",
        f"  FCC: {data['ratios']['edge_ratio']:.1f}× more edges, {int(f['connectivity']/c['connectivity'])}× connectivity",
    ]
    colors = [DIM] + [CUBIC] * 11 + [DIM, FCC]
    return lines, colors


def format_published_findings(data: dict) -> tuple[list[str], list[tuple]]:
    """Format published research findings from explain_mechanism data."""
    lines = [
        "",
        "  Published Results (scale 1,000)",
        "  ──────────────────────────────────",
        "",
        "  ┌─────────────────────────────────────────────────────┐",
        "  │ Fiedler Ratio (FCC / Cubic algebraic connectivity)  │",
        "  ├──────────────────────────┬──────────────────────────┤",
        "  │ Uniform weights          │   2.31×  (Paper 1)       │",
        "  │ Edge-cycled corpus       │   3.11×                  │",
        "  │ Direction-weighted corpus │   6.11×  (Paper 2)       │",
        "  └──────────────────────────┴──────────────────────────┘",
        "",
    ]
    colors = [
        DIM, GOLD, GOLD, DIM,
        CUBIC, CUBIC, CUBIC,
        TEXT, FCC, FCC,
        CUBIC,
        DIM,
    ]

    # Add paper2 findings from the mechanism data
    for finding in data.get("paper2_findings", [])[3:5]:
        lines.append(f"  • {finding[:75]}")
        colors.append(FCC)

    lines += [
        "",
        "  The advantage AMPLIFIES under heterogeneous weights.",
        "  Not noise. Structure. Bottleneck resilience.",
    ]
    colors += [DIM, GOLD, GOLD]
    return lines, colors


def format_mechanism(data: dict) -> tuple[list[str], list[tuple]]:
    """Format mechanism explanation."""
    lines = [
        "",
        "  Complete Mechanism — Paper 1 → Paper 2 → Paper 3",
        "  ═══════════════════════════════════════════════════",
        "",
        f"  Paper 1: {data.get('paper1', '')[:70]}",
        "",
    ]
    colors = [DIM, GOLD, GOLD, DIM, TEXT, DIM]

    for finding in data.get("paper2_findings", [])[:4]:
        lines.append(f"  • {finding[:75]}")
        colors.append(FCC)

    lines += [
        "",
        f"  {data.get('paper3_vision', 'RhombiLoRA: keep your cube, add six bridges.')[:75]}",
        "",
        "  Every computation defaults to cubic.",
        "  The rhombic dodecahedron has 12 faces where the cube has 6.",
        "  Keep your cube, add six bridges.",
    ]
    colors += [DIM, CYAN, DIM, TEXT, TEXT, GOLD]
    return lines, colors


def build_video(captures_dir: Path = CAPTURES):
    """Build all video frames from captured data."""
    r = FrameRenderer()

    # Load captures
    act1 = json.loads((captures_dir / "act1_lattice_compare.json").read_text())
    act2a = json.loads((captures_dir / "act2a_direction_weights.json").read_text())
    act2b = json.loads((captures_dir / "act2b_permutation_control.json").read_text())
    act3 = json.loads((captures_dir / "act3_explain_mechanism.json").read_text())

    print(f"Building video frames at {FPS} fps...")

    # ── Title Card (5s) ──
    title_path = ASSETS / "title_card.png"
    if title_path.exists():
        print("  Title card (5s)...")
        r.render_image_file(title_path, 5.0)

    # ── Divider: THE COMPARISON (1.5s) ──
    div1 = ASSETS / "divider_comparison.png"
    if div1.exists():
        # Scale divider to full frame with NUIT background
        div_img = Image.open(str(div1)).convert("RGB")
        frame = Image.new("RGB", (W, H), NUIT)
        # Center the divider vertically
        y_off = (H - div_img.height) // 2
        frame.paste(div_img, (0, y_off))
        r.render_static(frame, 1.5)

    # ── Act 1: Lattice Compare (12s) ──
    print("  Act 1: The Comparison...")
    base = r.make_terminal_base()
    prompt1 = "Compare cubic and FCC lattices at scale 5"
    typed = r.render_typing(base, prompt1, y_pos=56)
    called = r.render_tool_call(typed, "lattice_compare", y_pos=90)

    lines, colors = format_lattice_compare(act1)
    final1 = r.render_response_lines(called, lines, colors, y_start=130)

    # Overlay
    r.render_overlay(
        final1,
        "Every computation defaults to cubic. 6 neighbors. 3 directions.",
        duration_s=3.0,
        color=TEXT,
    )

    # ── Divider: THE PROOF (1.5s) ──
    div2 = ASSETS / "divider_proof.png"
    if div2.exists():
        frame = Image.new("RGB", (W, H), NUIT)
        div_img = Image.open(str(div2)).convert("RGB")
        frame.paste(div_img, (0, (H - div_img.height) // 2))
        r.render_static(frame, 1.5)

    # ── Act 2: Published Findings + Mechanism (15s) ──
    print("  Act 2: The Proof...")
    base2 = r.make_terminal_base()
    prompt2 = "Show me the published results"
    typed2 = r.render_typing(base2, prompt2, y_pos=56)
    called2 = r.render_tool_call(typed2, "explain_mechanism", y_pos=90)

    lines2, colors2 = format_published_findings(act3)
    final2 = r.render_response_lines(called2, lines2, colors2, y_start=130)

    # Overlay
    r.render_overlay(
        final2,
        "Not noise. Structure. The advantage amplifies under stress.",
        duration_s=3.0,
        color=FCC,
    )

    # ── Amplification chart (3s) ──
    amp_path = BASE / "assets" / "amplification_demo.png"
    if amp_path.exists():
        print("  Amplification chart (3s)...")
        r.render_image_file(amp_path, 3.0)

    # ── Divider: THE VISION (1.5s) ──
    div3 = ASSETS / "divider_vision.png"
    if div3.exists():
        frame = Image.new("RGB", (W, H), NUIT)
        div_img = Image.open(str(div3)).convert("RGB")
        frame.paste(div_img, (0, (H - div_img.height) // 2))
        r.render_static(frame, 1.5)

    # ── Act 3: The Vision (12s) ──
    print("  Act 3: The Vision...")
    base3 = r.make_terminal_base()
    prompt3 = "What does this mean for neural networks?"
    typed3 = r.render_typing(base3, prompt3, y_pos=56)
    called3 = r.render_tool_call(typed3, "explain_mechanism", y_pos=90)

    lines3, colors3 = format_mechanism(act3)
    final3 = r.render_response_lines(called3, lines3, colors3, y_start=130)

    # Final overlay
    r.render_overlay(
        final3,
        "Keep your cube, add six bridges.",
        duration_s=3.0,
        color=GOLD,
    )

    # ── Closing Card (8s) ──
    close_path = ASSETS / "closing_card.png"
    if close_path.exists():
        print("  Closing card (8s)...")
        r.render_image_file(close_path, 8.0)

    print(f"\n  Total frames: {r.frame_num}")
    print(f"  Duration: {r.frame_num / FPS:.1f}s at {FPS}fps")
    print(f"  Output: {FRAMES}")
    return r.frame_num


if __name__ == "__main__":
    build_video()
