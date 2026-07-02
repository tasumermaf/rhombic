#!/usr/bin/env python3
"""
Angelic Keys Stinger Processor — Episode 2 Alien Transmission Layer.

Takes Timothy's 2013 96kHz Angelic Keys recordings and processes them
into 3-5 second stingers for placement at key reveal moments:
  - Downsample to 48kHz
  - Pitch-shift -5 semitones (deeper, more alien)
  - Time-stretch 2x (slower, more expansive)
  - Convolution reverb (large hall impulse, synthesized)
  - Trim to target duration

Output: assets/audio/stingers/angelic_*.wav

Usage:
    python scripts/process_angelic_keys.py
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, fftconvolve
from pathlib import Path

TARGET_SR = 48000
BASE = Path(__file__).resolve().parent.parent
ANGELIC_DIR = BASE.parent / "Angelic Keys"
STINGER_DIR = BASE / "assets" / "audio" / "stingers"


def load_and_downsample(path, target_sr=TARGET_SR):
    """Load WAV (possibly 96kHz), downsample to target_sr, return mono float64."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483647.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)

    # Mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Downsample
    if sr != target_sr:
        # Use rational resampling
        from math import gcd
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down)

    return data


def pitch_shift_semitones(data, semitones, sr=TARGET_SR):
    """Pitch-shift by resampling (changes duration — we time-stretch after).

    -5 semitones = multiply frequency by 2^(-5/12) ≈ 0.7491.
    To shift DOWN, we resample UP then play at original rate.
    """
    ratio = 2.0 ** (semitones / 12.0)
    # Resample: to shift down, we need MORE samples (lower pitch = longer)
    new_len = int(len(data) / ratio)
    indices = np.linspace(0, len(data) - 1, new_len)
    shifted = np.interp(indices, np.arange(len(data)), data)
    return shifted


def time_stretch_simple(data, factor):
    """Simple time-stretch via resampling (preserves pitch approximately).

    factor=2.0 makes it twice as long.
    For a stinger effect, the slight pitch artifact is actually desirable.
    """
    new_len = int(len(data) * factor)
    indices = np.linspace(0, len(data) - 1, new_len)
    return np.interp(indices, np.arange(len(data)), data)


def synth_reverb_ir(duration_s=2.5, sr=TARGET_SR, decay_rate=3.0):
    """Synthesize a large-hall reverb impulse response.

    Exponentially decaying noise with frequency-dependent absorption
    (high frequencies decay faster).
    """
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr

    # Broadband noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, n)

    # Exponential decay
    envelope = np.exp(-decay_rate * t)

    # Early reflections (a few discrete echoes)
    ir = noise * envelope * 0.3
    for delay_ms, amp in [(23, 0.6), (47, 0.4), (71, 0.3), (113, 0.2)]:
        d = int(delay_ms * sr / 1000)
        if d < n:
            ir[d] += amp

    # High-frequency absorption: apply progressive LPF via moving average
    # Crude but effective for synthesis
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 4000 / (sr / 2), btype='low')
    late = filtfilt(b, a, noise * envelope * 0.15)
    ir += late

    # Normalize
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir /= peak

    return ir


def process_excerpt(data, excerpt_start_s, excerpt_dur_s, sr=TARGET_SR):
    """Extract, pitch-shift, time-stretch, and reverb a section."""
    start = int(excerpt_start_s * sr)
    dur = int(excerpt_dur_s * sr)
    end = min(start + dur, len(data))
    excerpt = data[start:end]

    if len(excerpt) < sr:
        print(f"    WARNING: excerpt too short ({len(excerpt)} samples)")
        return excerpt

    # Pitch-shift -5 semitones
    shifted = pitch_shift_semitones(excerpt, -5, sr)

    # Time-stretch 2x
    stretched = time_stretch_simple(shifted, 2.0)

    # Apply convolution reverb
    ir = synth_reverb_ir(2.5, sr)
    wet = fftconvolve(stretched, ir, mode='full')[:len(stretched) + int(2.0 * sr)]

    # Mix dry/wet (60% wet for spacious alien sound)
    dry_len = len(stretched)
    mixed = np.zeros(len(wet))
    mixed[:dry_len] += stretched * 0.4
    mixed[:len(wet)] += wet * 0.6

    # Trim to target duration (3-5s)
    target_len = int(4.0 * sr)
    if len(mixed) > target_len:
        mixed = mixed[:target_len]

    # Fade in/out
    fade_in = int(0.05 * sr)
    fade_out = int(0.8 * sr)
    mixed[:fade_in] *= np.linspace(0, 1, fade_in)
    mixed[-fade_out:] *= np.linspace(1, 0, fade_out)

    return mixed


def write_wav(path, data, sr=TARGET_SR):
    """Write mono float64 data as stereo 16-bit WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.90
    stereo = np.column_stack([data, data])
    wav_data = (stereo * 32767).astype(np.int16)
    wavfile.write(str(path), sr, wav_data)
    size_kb = path.stat().st_size // 1024
    print(f"  -> {path.name}: {len(data)/sr:.1f}s, {size_kb}KB")


# Stinger assignments: (angelic key number, excerpt start in seconds, name)
STINGERS = [
    (1, 15.0, 2.0, "angelic_bridge_reveal"),     # 22,477:1 reveal
    (3, 20.0, 1.5, "angelic_holly_battery"),      # Holly Battery
    (7, 10.0, 2.0, "angelic_negative_result"),    # Negative result
    (9, 25.0, 1.5, "angelic_tasumer_maf"),         # TASUMER MAF reveal
]


def main():
    print("=" * 60)
    print("  ANGELIC KEYS STINGER PROCESSOR — Episode 2")
    print("=" * 60)

    if not ANGELIC_DIR.exists():
        print(f"  ERROR: Angelic Keys directory not found: {ANGELIC_DIR}")
        return

    keys_available = sorted(ANGELIC_DIR.glob("*.wav"))
    print(f"  Found {len(keys_available)} Angelic Keys recordings")

    for key_num, start_s, dur_s, name in STINGERS:
        # Find the matching file
        pattern = f"*{key_num:02d}*"
        matches = list(ANGELIC_DIR.glob(f"*- {key_num:02d} *.wav"))
        if not matches:
            matches = list(ANGELIC_DIR.glob(f"*{key_num:02d}*.wav"))
        if not matches:
            print(f"\n  WARNING: No Angelic Key {key_num} found, skipping {name}")
            continue

        source = matches[0]
        print(f"\n  Processing Key {key_num}: {source.name}")
        print(f"    Excerpt: {start_s}s + {dur_s}s -> pitch -5, stretch 2x, reverb")

        data = load_and_downsample(source)
        stinger = process_excerpt(data, start_s, dur_s)
        write_wav(STINGER_DIR / f"{name}.wav", stinger)

    print("\n" + "=" * 60)
    print("  COMPLETE — Angelic Keys stingers generated")
    print("=" * 60)


if __name__ == "__main__":
    main()
