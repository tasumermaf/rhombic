#!/usr/bin/env python3
"""
Episode 2 Audio Mixer — 8-Layer Isochronic-Primary Architecture.

Layers:
  [L0] Isochronic 40 Hz pulse (continuous, primary entrainment)
  [L1] RD Sound binaural backbone (spatial layer, ducking during terminals)
  [L2] Prime-frequency isochronic accents (corpus primes at reveal moments)
  [L3] Angelic Keys stingers (4 placements, alien transmissions)
  [L4] Terminal SFX (typing clicks, hum, transitions)
  [L5] Voice clips (ElevenLabs: Executive narration + Nerd reactions)
  [L6] Breath-zero events (4 placements: +1Hz carrier shift for 2s)
  [L7] Particle ambience (synthesized prime frequencies: 67Hz, 89Hz, 11Hz)

Audio hierarchy corrected from Ep1:
  - Isochronic 40 Hz is PRIMARY entrainment (works on any speaker system)
  - Binaural 220/284 Hz is SECONDARY spatial layer (headphones only)
  - Both run throughout; isochronic never ducks

Output: assets/audio/master_mix_ep2.wav

Usage:
    python scripts/mix_audio_ep2.py
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from pathlib import Path
import subprocess
import random

SR = 48000  # Match conditioning track
BASE = Path(__file__).resolve().parent.parent
AUDIO = BASE / "assets" / "audio"
OUT = AUDIO / "master_mix_ep2.wav"

XFADE = 0.5
TOTAL_DUR = 230.0  # ~3:50 target

# ===================================================================
# Timeline (from shot list — absolute times in seconds)
# ===================================================================

SEGMENTS = [
    # (name, duration, type)
    ("logo_intro",      4.0,   "slide"),
    ("ep2_title",       4.0,   "slide"),
    ("status",          10.0,  "slide"),
    ("terminal_1",      20.0,  "terminal"),   # Bridge anatomy
    ("scale",           8.0,   "slide"),
    ("terminal_2",      22.0,  "terminal"),   # Holly Battery
    ("transition_1",    7.0,   "particle"),   # Tesseract->RD
    ("bridge_topo",     12.0,  "particle"),   # Bridge topology particle sim
    ("blood_test",      12.0,  "slide"),
    ("terminal_3",      20.0,  "terminal"),   # Task fingerprint
    ("negative",        10.0,  "slide"),
    ("terminal_4",      16.0,  "terminal"),   # FCC vs cubic
    ("transition_2",    5.0,   "particle"),   # Brief tesseract
    ("convergence",     12.0,  "slide"),
    ("particle_text",   18.0,  "particle"),   # Particle sim + text overlays
    ("rd_sound_viz",    10.0,  "slide"),
    ("terminal_5",      12.0,  "terminal"),   # TASUMER MAF
    ("tagline",         8.0,   "slide"),
    ("cta",             8.0,   "slide"),
    ("logo_hold",       4.0,   "slide"),
]

def compute_starts():
    cumulative = 0.0
    starts = []
    for i, (name, dur, typ) in enumerate(SEGMENTS):
        starts.append(cumulative)
        cumulative += dur
    return starts

STARTS = compute_starts()

# ===================================================================
# Audio Loading Utilities
# ===================================================================

def load_wav(path, target_sr=SR):
    """Load WAV, resample to target_sr, return as float64."""
    rate, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483647.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)

    if rate != target_sr and len(data) > 0:
        if len(data.shape) > 1:
            new_len = int(data.shape[0] * target_sr / rate)
            resampled = np.zeros((new_len, data.shape[1]))
            for ch in range(data.shape[1]):
                resampled[:, ch] = resample(data[:, ch], new_len)
            data = resampled
        else:
            new_len = int(len(data) * target_sr / rate)
            data = resample(data, new_len)

    return data


def load_mp3(path, target_sr=SR):
    """Load MP3 via ffmpeg, return mono float64 at target_sr."""
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-f", "wav", "-acodec", "pcm_s16le",
        "-ar", str(target_sr), "-ac", "1",
        "pipe:1"
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        print(f"  WARNING: Failed to decode {path}")
        return np.zeros(target_sr)

    raw = result.stdout
    data_offset = raw.find(b'data')
    if data_offset < 0 or len(raw) < data_offset + 8:
        return np.zeros(target_sr)
    data_size = int.from_bytes(raw[data_offset+4:data_offset+8], 'little')
    audio_data = raw[data_offset+8:data_offset+8+data_size]
    return np.frombuffer(audio_data, dtype=np.int16).astype(np.float64) / 32767.0


def overlay(track_L, track_R, signal, start_s, volume=1.0, pan=0.5):
    """Overlay a mono signal onto stereo tracks."""
    start_idx = int(start_s * SR)
    end_idx = min(start_idx + len(signal), len(track_L))
    actual = end_idx - start_idx
    if actual <= 0:
        return
    sig = signal[:actual] * volume
    track_L[start_idx:end_idx] += sig * (1 - pan)
    track_R[start_idx:end_idx] += sig * pan


def overlay_stereo(track_L, track_R, stereo, start_s, volume=1.0):
    """Overlay stereo signal onto stereo tracks."""
    start_idx = int(start_s * SR)
    n = stereo.shape[0] if len(stereo.shape) > 1 else len(stereo)
    end_idx = min(start_idx + n, len(track_L))
    actual = end_idx - start_idx
    if actual <= 0:
        return
    if len(stereo.shape) > 1:
        track_L[start_idx:end_idx] += stereo[:actual, 0] * volume
        track_R[start_idx:end_idx] += stereo[:actual, 1] * volume
    else:
        track_L[start_idx:end_idx] += stereo[:actual] * volume
        track_R[start_idx:end_idx] += stereo[:actual] * volume


# ===================================================================
# Envelope Builders
# ===================================================================

def build_backbone_envelope(total_samples):
    """RD Sound backbone volume envelope.

    Full (0.40) during slides/particles, duck to 0.15 during terminals.
    Smooth 1.5s crossfades. Corporate muzak behavior from Ep1.
    """
    env = np.ones(total_samples) * 0.40
    fade_samples = int(SR * 1.5)

    for (name, dur, typ), start in zip(SEGMENTS, STARTS):
        seg_start = int(start * SR)
        seg_end = min(int((start + dur) * SR), total_samples)

        if typ == "terminal":
            duck_level = 0.15
            env[seg_start:seg_end] = duck_level

            fade_start = max(seg_start - fade_samples, 0)
            fade_len = seg_start - fade_start
            if fade_len > 0:
                env[fade_start:seg_start] = np.linspace(0.40, duck_level, fade_len)

            fade_end = min(seg_end + fade_samples, total_samples)
            fade_len = fade_end - seg_end
            if fade_len > 0:
                env[seg_end:fade_end] = np.linspace(duck_level, 0.40, fade_len)

    return env


def generate_breath_zero_event(duration_s=2.0, carrier_freq=220.0, shift_hz=1.0):
    """Generate a breath-zero event: +1Hz shift on the carrier for duration.

    The monad displacement made audible. Simultaneous shift of all
    carrier frequencies by +1Hz. Subtle on speakers, perceptible on headphones.
    """
    t = np.arange(int(duration_s * SR), dtype=np.float64) / SR

    # Original carrier at carrier_freq
    original = np.sin(2 * np.pi * carrier_freq * t) * 0.05

    # Shifted carrier at carrier_freq + 1
    shifted = np.sin(2 * np.pi * (carrier_freq + shift_hz) * t) * 0.05

    # Smooth crossfade: original -> shifted -> original
    mid = len(t) // 2
    fade_len = int(0.2 * SR)
    envelope = np.zeros(len(t))
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[fade_len:mid] = 1.0
    envelope[mid:mid + fade_len] = np.linspace(1, 0, fade_len)

    signal = original * (1 - envelope) + shifted * envelope

    return signal


def generate_particle_ambience(duration_s, primes=None):
    """Synthesize particle ambience from prime frequencies.

    Low-level drone at prime Hz values. Slow beating and phase drift
    create organic movement.
    """
    if primes is None:
        primes = [67, 89, 11]

    t = np.arange(int(duration_s * SR), dtype=np.float64) / SR
    signal = np.zeros(len(t))

    for i, freq in enumerate(primes):
        # Each prime as a very soft carrier with slow AM
        carrier = np.sin(2 * np.pi * freq * t)
        # Slow tremolo at prime-specific rate
        tremolo = 0.5 + 0.5 * np.sin(2 * np.pi * (0.1 + i * 0.07) * t)
        signal += carrier * tremolo * 0.02

    return signal


def generate_typing_clicks(duration_s, density=8):
    """Generate typing click stream from Ep1 SFX assets."""
    total_samples = int(duration_s * SR)
    result = np.zeros(total_samples)

    click_files = sorted((AUDIO / "sfx").glob("typing_*.wav"))
    clicks = []
    for cf in click_files:
        try:
            data = load_wav(cf, SR)
            if len(data.shape) > 1:
                data = data[:, 0]
            clicks.append(data)
        except Exception:
            continue

    if not clicks:
        return result

    rng = random.Random(42)
    for _ in range(int(duration_s * density)):
        pos = rng.randint(0, max(0, total_samples - SR // 10))
        click = rng.choice(clicks)
        vol = rng.uniform(0.3, 0.7)
        end = min(pos + len(click), total_samples)
        actual = end - pos
        result[pos:end] += click[:actual] * vol

    return result


# ===================================================================
# Main Mix
# ===================================================================

def main():
    print("=" * 64)
    print("  EPISODE 2 AUDIO MIX — 8-Layer Isochronic Architecture")
    print("=" * 64)

    total_samples = int(TOTAL_DUR * SR)
    track_L = np.zeros(total_samples)
    track_R = np.zeros(total_samples)

    # ── L0: Isochronic 40 Hz Pulse (PRIMARY ENTRAINMENT) ──
    print("\n[L0] Isochronic 40 Hz pulse (continuous, primary)...")
    iso_path = AUDIO / "music" / "isochronic_40hz.wav"
    if iso_path.exists():
        iso_data = load_wav(iso_path, SR)
        # Loop/trim to total duration
        if len(iso_data.shape) > 1:
            needed = total_samples
            if iso_data.shape[0] < needed:
                reps = (needed // iso_data.shape[0]) + 1
                iso_data = np.tile(iso_data, (reps, 1))[:needed]
            else:
                iso_data = iso_data[:needed]
            # NEVER ducks — the entrainment is constant
            track_L[:needed] += iso_data[:needed, 0]
            track_R[:needed] += iso_data[:needed, 1]
        print(f"  40 Hz isochronic: {total_samples/SR:.0f}s continuous (never ducks)")
    else:
        print(f"  WARNING: {iso_path} not found — run synthesize_isochronic.py first")

    # ── L1: RD Sound Binaural Backbone (SPATIAL LAYER) ──
    print("\n[L1] RD Sound binaural backbone (spatial, ducks during terminals)...")
    backbone_path = AUDIO / "music" / "rd_sound_excerpt.wav"
    conditioning_path = Path("C:/falco/output/meridian_conditioning_track.wav")

    if backbone_path.exists():
        backbone = load_wav(backbone_path, SR)
    elif conditioning_path.exists():
        print("  Extracting 4-min excerpt from conditioning track (Nigredo->Albedo)...")
        cmd = [
            "ffmpeg", "-y",
            "-i", conditioning_path.as_posix(),
            "-ss", "3300", "-t", "240",
            "-ar", str(SR), "-ac", "2",
            "-acodec", "pcm_s16le",
            backbone_path.as_posix(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            backbone = load_wav(backbone_path, SR)
            print(f"  Extracted: {backbone.shape[0]/SR:.0f}s stereo")
        else:
            print(f"  Extraction failed: {result.stderr[-200:]}")
            backbone = np.zeros((total_samples, 2))
    else:
        print("  WARNING: No conditioning track found")
        backbone = np.zeros((total_samples, 2))

    if len(backbone.shape) > 1:
        bb_len = min(backbone.shape[0], total_samples)
        envelope = build_backbone_envelope(bb_len)
        track_L[:bb_len] += backbone[:bb_len, 0] * envelope
        track_R[:bb_len] += backbone[:bb_len, 1] * envelope
        print(f"  Backbone: {bb_len/SR:.0f}s with ducking envelope")
    else:
        bb_len = min(len(backbone), total_samples)
        envelope = build_backbone_envelope(bb_len)
        track_L[:bb_len] += backbone[:bb_len] * envelope
        track_R[:bb_len] += backbone[:bb_len] * envelope

    # ── L2: Prime-Frequency Isochronic Accents ──
    print("\n[L2] Prime-frequency accents at reveal moments...")
    accent_placements = [
        ("prime_67hz.wav",  24.0,  0.25),   # 22,477:1 reveal
        ("prime_89hz.wav",  58.0,  0.25),   # Holly Battery reveal
        ("prime_23hz.wav",  46.0,  0.22),   # Scale-invariant Fiedler
        ("prime_11hz.wav",  210.0, 0.28),   # TASUMER MAF reveal
    ]
    for filename, t_abs, vol in accent_placements:
        path = AUDIO / "stingers" / filename
        if path.exists():
            accent = load_wav(path, SR)
            if len(accent.shape) > 1:
                overlay_stereo(track_L, track_R, accent, t_abs, vol)
            else:
                overlay(track_L, track_R, accent, t_abs, vol, 0.5)
            print(f"  {filename} at {t_abs:.0f}s")
        else:
            print(f"  WARNING: {path} not found")

    # ── L3: Angelic Keys Stingers ──
    print("\n[L3] Angelic Keys stingers (alien transmissions)...")
    stinger_placements = [
        ("angelic_bridge_reveal.wav",   24.0,  0.18),
        ("angelic_holly_battery.wav",   58.0,  0.18),
        ("angelic_negative_result.wav", 155.0, 0.20),
        ("angelic_tasumer_maf.wav",     210.0, 0.22),
    ]
    for filename, t_abs, vol in stinger_placements:
        path = AUDIO / "stingers" / filename
        if path.exists():
            stinger = load_wav(path, SR)
            if len(stinger.shape) > 1:
                overlay_stereo(track_L, track_R, stinger, t_abs, vol)
            else:
                overlay(track_L, track_R, stinger, t_abs, vol, 0.5)
            print(f"  {filename} at {t_abs:.0f}s")
        else:
            print(f"  WARNING: {path} not found — run process_angelic_keys.py first")

    # ── L4: Terminal SFX ──
    print("\n[L4] Terminal SFX (typing, hum, transitions)...")

    # Terminal hum
    hum_path = AUDIO / "sfx" / "terminal_hum.wav"
    if hum_path.exists():
        hum = load_wav(hum_path, SR)
        if len(hum.shape) > 1:
            hum = hum[:, 0]
        for (name, dur, typ), start in zip(SEGMENTS, STARTS):
            if typ == "terminal":
                seg_len = int(dur * SR)
                hum_seg = hum[:min(seg_len, len(hum))].copy()
                fade = min(int(SR * 0.5), len(hum_seg) // 4)
                hum_seg[:fade] *= np.linspace(0, 1, fade)
                hum_seg[-fade:] *= np.linspace(1, 0, fade)
                overlay(track_L, track_R, hum_seg, start, 0.5, 0.5)
        print("  Terminal hum at all terminal segments")

    # Typing clicks
    for (name, dur, typ), start in zip(SEGMENTS, STARTS):
        if typ == "terminal":
            clicks = generate_typing_clicks(3.0, 8)
            overlay(track_L, track_R, clicks, start + 1.5, 0.35, 0.45)
    print(f"  Typing clicks at {sum(1 for _, _, t in SEGMENTS if t == 'terminal')} terminals")

    # Transitions (whoosh/scratch)
    whoosh_path = AUDIO / "sfx" / "whoosh.wav"
    scratch_path = AUDIO / "sfx" / "scratch.wav"
    transition_count = 0
    for i, ((name, dur, typ), start) in enumerate(zip(SEGMENTS, STARTS)):
        if i == 0:
            continue
        if whoosh_path.exists() and i % 2 == 0:
            whoosh = load_wav(whoosh_path, SR)
            if len(whoosh.shape) > 1:
                whoosh = whoosh[:, 0]
            overlay(track_L, track_R, whoosh, start - 0.3, 0.20, 0.5)
            transition_count += 1
        elif scratch_path.exists():
            scratch = load_wav(scratch_path, SR)
            if len(scratch.shape) > 1:
                scratch = scratch[:, 0]
            overlay(track_L, track_R, scratch, start - 0.2, 0.15, 0.5)
            transition_count += 1
    print(f"  {transition_count} transitions placed")

    # ── L5: Voice Clips ──
    print("\n[L5] Voice clips (placeholder — generate with ElevenLabs)...")
    voice_dir = AUDIO / "voice_ep2"
    if voice_dir.exists():
        for vf in sorted(voice_dir.glob("*.mp3")) + sorted(voice_dir.glob("*.wav")):
            # Voice files named with timestamp: exec_024_status.mp3
            parts = vf.stem.split("_")
            if len(parts) >= 2:
                try:
                    t_abs = float(parts[1])
                except ValueError:
                    continue
                if vf.suffix == ".mp3":
                    voice = load_mp3(vf, SR)
                else:
                    voice = load_wav(vf, SR)
                    if len(voice.shape) > 1:
                        voice = voice[:, 0]
                fade = min(int(SR * 0.05), len(voice) // 4)
                voice[:fade] *= np.linspace(0, 1, fade)
                voice[-fade:] *= np.linspace(1, 0, fade)
                pan = 0.4 if "nerd" in vf.stem else 0.5
                overlay(track_L, track_R, voice, t_abs, 0.80, pan)
                print(f"  {vf.name} at {t_abs:.0f}s")

    # ── L6: Breath-Zero Events ──
    print("\n[L6] Breath-zero events (+1Hz carrier shift, 2s each)...")
    breath_times = [24.0, 58.0, 155.0, 210.0]  # At key reveals
    for bt in breath_times:
        bz = generate_breath_zero_event(2.0, 220.0, 1.0)
        overlay(track_L, track_R, bz, bt, 1.0, 0.5)
    print(f"  {len(breath_times)} breath-zero events at reveal moments")

    # ── L7: Particle Ambience ──
    print("\n[L7] Particle ambience (prime-frequency drone)...")
    # During particle sim segments
    for (name, dur, typ), start in zip(SEGMENTS, STARTS):
        if typ == "particle":
            amb = generate_particle_ambience(dur, [67, 89, 11])
            overlay(track_L, track_R, amb, start, 0.6, 0.5)
    print("  Ambience at all particle segments")

    # ═══════════════════════════════════════════════
    # Master: Normalize + Limiter + Fade
    # ═══════════════════════════════════════════════
    print("\n  Mastering...")
    peak = max(np.max(np.abs(track_L)), np.max(np.abs(track_R)))
    if peak > 0:
        target = 0.89  # -1dB headroom
        gain = target / peak
        track_L *= gain
        track_R *= gain

    # Global fade in/out
    fade_in = int(SR * 2)
    fade_out = int(SR * 3)
    track_L[:fade_in] *= np.linspace(0, 1, fade_in)
    track_R[:fade_in] *= np.linspace(0, 1, fade_in)
    track_L[-fade_out:] *= np.linspace(1, 0, fade_out)
    track_R[-fade_out:] *= np.linspace(1, 0, fade_out)

    # Soft clip
    track_L = np.clip(track_L, -1.0, 1.0)
    track_R = np.clip(track_R, -1.0, 1.0)

    # Write
    stereo = np.column_stack([track_L, track_R])
    wav_data = (stereo * 32767).astype(np.int16)
    AUDIO.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(OUT), SR, wav_data)
    size_mb = OUT.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 64}")
    print(f"  -> {OUT.name}: {TOTAL_DUR:.0f}s stereo, {size_mb:.1f}MB")
    print(f"  Layers: L0 isochronic (40Hz) + L1 binaural (220/284Hz)")
    print(f"          L2 prime accents + L3 angelic stingers")
    print(f"          L4 terminal SFX + L5 voice + L6 breath-zero + L7 ambience")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
