#!/usr/bin/env python3
"""
Generate Episode 2 voice clips using Edge TTS.

Two characters:
  - Executive: measured, authoritative alien briefing voice (en-US-GuyNeural)
  - Nerd: slightly faster, agent-researcher energy (en-US-AndrewNeural)

Naming convention: {role}_{timestamp}_{label}.wav
  - role: "exec" or "nerd"
  - timestamp: absolute seconds in the video timeline
  - label: descriptive name

Output: assets/audio/voice_ep2/*.wav
"""

import asyncio
import io
import sys
from pathlib import Path

import edge_tts
import numpy as np

BASE = Path(__file__).resolve().parent.parent
VOICE_DIR = BASE / "assets" / "audio" / "voice_ep2"

# Executive: deep, measured, corporate alien
EXEC_VOICE = "en-US-GuyNeural"
EXEC_RATE = "-10%"   # Slightly slower for gravitas
EXEC_PITCH = "-5Hz"  # Slightly lower

# Nerd: clear, precise, energetic researcher
NERD_VOICE = "en-US-AndrewNeural"
NERD_RATE = "+5%"     # Slightly faster, excited
NERD_PITCH = "+0Hz"

# Voice clips with absolute placement timestamps
# Format: (filename_stem, voice, rate, pitch, text)
CLIPS = [
    # ACT I — Status Update
    ("exec_004_episode2", EXEC_VOICE, EXEC_RATE, EXEC_PITCH,
     "Episode Two. The Briefing."),

    ("exec_008_status", EXEC_VOICE, EXEC_RATE, EXEC_PITCH,
     "Three results. One geometry."),

    ("nerd_020_bridge", NERD_VOICE, NERD_RATE, NERD_PITCH,
     "Show me the bridge anatomy for Qwen seven B."),

    ("exec_040_scale", EXEC_VOICE, EXEC_RATE, EXEC_PITCH,
     "Fiedler convergence. Scale invariant."),

    ("nerd_048_holly", NERD_VOICE, NERD_RATE, NERD_PITCH,
     "What does the bridge look like at fourteen billion?"),

    ("nerd_058_surprise", NERD_VOICE, NERD_RATE, NERD_PITCH,
     "Three point eight percent better loss. Nine gig less VRAM. Fifty percent smaller checkpoint."),

    # ACT II — The Evidence Pile
    ("exec_078_bridge_vis", EXEC_VOICE, EXEC_RATE, EXEC_PITCH,
     "Twenty-two thousand, four hundred seventy-seven to one. Rivers versus trickles."),

    ("exec_090_blood", EXEC_VOICE, EXEC_RATE, EXEC_PITCH,
     "Eighty-four point five percent fingerprint match. Thirty-six parameters. Fourteen billion."),

    ("nerd_102_fingerprint", NERD_VOICE, NERD_RATE, NERD_PITCH,
     "Run task fingerprint on the Holly bridge."),

    ("exec_125_negative", EXEC_VOICE, EXEC_RATE, EXEC_PITCH,
     "Corpus weighting hurts. The topology does the work. Not the data."),

    ("nerd_138_fcc", NERD_VOICE, NERD_RATE, NERD_PITCH,
     "Compare six channel FCC versus three channel cubic."),

    ("nerd_148_roads", NERD_VOICE, NERD_RATE, NERD_PITCH,
     "Four point six times Fiedler advantage. Zero parameter cost. More roads. Same land."),

    # ACT III — The Pattern
    ("exec_160_pattern", EXEC_VOICE, EXEC_RATE, EXEC_PITCH,
     "Eight cubic vertices. Eight prime threads. Six octahedral vertices. Six bridges. Twenty-four edges. Twenty-four cards."),

    ("exec_185_sound", EXEC_VOICE, EXEC_RATE, EXEC_PITCH,
     "Meridian. Two-twenty hertz. The steersman's frequency."),

    ("nerd_198_tasumer", NERD_VOICE, NERD_RATE, NERD_PITCH,
     "What is Tasumer Maf?"),

    ("nerd_206_steersman", NERD_VOICE, "+0%", NERD_PITCH,
     "Kybernetes. Ten ninety-three. The steersman."),

    # CODA
    ("exec_218_tagline", EXEC_VOICE, "-15%", "-8Hz",
     "Keep your cube. Add six bridges."),

    ("exec_228_cta", EXEC_VOICE, EXEC_RATE, EXEC_PITCH,
     "pip install rhombic."),
]


async def generate_clip(stem, voice, rate, pitch, text):
    """Generate a single voice clip."""
    out_path = VOICE_DIR / f"{stem}.wav"

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )

    # Edge TTS outputs MP3 — save then convert
    mp3_path = VOICE_DIR / f"{stem}.mp3"
    await communicate.save(str(mp3_path))

    # Convert MP3 to WAV at 48kHz mono for the mixer
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio = audio.set_frame_rate(48000).set_channels(1)
        audio.export(str(out_path), format="wav")
        mp3_path.unlink()  # Remove intermediate MP3
    except ImportError:
        # Fallback: keep MP3 (mixer can load both)
        out_path = mp3_path

    return out_path


async def main():
    VOICE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VOICE GENERATION — Episode 2")
    print(f"  Executive: {EXEC_VOICE} (rate={EXEC_RATE}, pitch={EXEC_PITCH})")
    print(f"  Nerd: {NERD_VOICE} (rate={NERD_RATE}, pitch={NERD_PITCH})")
    print("=" * 60)

    for stem, voice, rate, pitch, text in CLIPS:
        role = "Executive" if stem.startswith("exec") else "Nerd"
        print(f"\n  [{role}] {stem}")
        print(f"    \"{text}\"")
        out = await generate_clip(stem, voice, rate, pitch, text)
        size_kb = out.stat().st_size // 1024
        print(f"    -> {out.name} ({size_kb}KB)")

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE — {len(CLIPS)} voice clips generated")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
