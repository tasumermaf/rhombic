#!/usr/bin/env python3
"""
TASUMER MAF — Production Ray-Traced Rhombic Dodecahedron Logo.

GPU-accelerated ray tracer on RTX 6000 Ada 48GB.
2x supersampled anti-aliasing, anamorphic bloom, chromatic aberration,
film grain, cinematic post-processing.

12 faces. 8 Laws. The geometry of transit.

Usage:
    python scripts/raytrace_rd_logo.py              # Full 4K render (SSAA 2x)
    python scripts/raytrace_rd_logo.py --preview     # 720p preview (no SSAA)
    python scripts/raytrace_rd_logo.py --frame 150   # Single frame test
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import subprocess
import sys
import os
import io
import time
import math

# Force UTF-8 on Windows
if sys.platform == "win32" and not hasattr(sys.stdout, '_rhombic_wrapped'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stdout._rhombic_wrapped = True
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ===================================================================
# Configuration
# ===================================================================

PREVIEW = "--preview" in sys.argv
SINGLE_FRAME = None
for _i, _a in enumerate(sys.argv):
    if _a == "--frame" and _i + 1 < len(sys.argv):
        SINGLE_FRAME = int(sys.argv[_i + 1])

W, H = (1280, 720) if PREVIEW else (3840, 2160)
SS = 1 if PREVIEW else 2          # Supersampling factor
RENDER_W, RENDER_H = W * SS, H * SS
FPS = 30
DURATION = 10.0
NFRAMES = int(FPS * DURATION)
MAX_BOUNCES = 4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float32

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "video"
FRAME_DIR = OUT_DIR / "logo_frames"
OUTPUT_FILE = OUT_DIR / "tasumer_maf_logo.mp4"


# ===================================================================
# 8-Law Weave Palette
# ===================================================================

def _rgb(r, g, b):
    return [r / 255.0, g / 255.0, b / 255.0]


# Ordered by LAW_PRIMES: [11, 23, 67, 17, 29, 19, 89, 31]
LAW_PALETTE = [
    _rgb(0x3D, 0x3D, 0x6B),  # 0: indaco  -- Fall of Neutral Events (11)
    _rgb(0x7B, 0x3F, 0x8C),  # 1: viola   -- Kaos (23)
    _rgb(0xB3, 0x44, 0x44),  # 2: mattone -- Geometric Essence (67)
    _rgb(0xE8, 0x85, 0x3D),  # 3: arancio -- Time Matrix (17)
    _rgb(0xDC, 0xC9, 0x64),  # 4: oro     -- Synchronicity (29)
    _rgb(0x4A, 0x8C, 0x5C),  # 5: verde   -- Arrow of Complexity (19)
    _rgb(0x5C, 0x8F, 0xAF),  # 6: azzurro -- (89)
    _rgb(0xF0, 0xE8, 0xD0),  # 7: avorio  -- (31)
]

NUIT = torch.tensor(_rgb(0x08, 0x06, 0x20), device=DEV, dtype=DT)

# 12 RD faces -> 8 Law colors
_FIDX = [0, 4, 1, 5, 2, 6, 3, 7, 2, 0, 4, 6]
FACE_COLORS = torch.tensor([LAW_PALETTE[i] for i in _FIDX], device=DEV, dtype=DT)


# ===================================================================
# RD Geometry — slab-based intersection
# ===================================================================

SLABS = torch.tensor([
    [1, 1, 0], [1, -1, 0], [1, 0, 1],
    [1, 0, -1], [0, 1, 1], [0, 1, -1],
], device=DEV, dtype=DT)

RD_SIZE = 1.0

FACE_NORMALS = torch.zeros(12, 3, device=DEV, dtype=DT)
for _i in range(6):
    _n = SLABS[_i] / SLABS[_i].norm()
    FACE_NORMALS[2 * _i] = _n
    FACE_NORMALS[2 * _i + 1] = -_n

ALL_VERTS = torch.tensor([
    [0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
    [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
], device=DEV, dtype=DT)


def _build_edges():
    edges = []
    for ci in range(8):
        for oi in range(8, 14):
            d2 = (ALL_VERTS[ci] - ALL_VERTS[oi]).pow(2).sum().item()
            if d2 < 0.76:
                edges.append((ci, oi))
    return edges


RD_EDGES = _build_edges()
assert len(RD_EDGES) == 24


# ===================================================================
# Utilities
# ===================================================================

def smooth(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def ease_out_quart(x):
    x = float(np.clip(x, 0.0, 1.0))
    return 1 - (1 - x) ** 4


def vnorm(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def roty(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=DEV, dtype=DT)


def rotx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=DEV, dtype=DT)


def rotz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=DEV, dtype=DT)


# ===================================================================
# Star Field Environment Map (Production)
# ===================================================================

def create_environment(eh=1024, ew=2048):
    """Procedural deep-space environment with galaxy features."""
    env = NUIT[None, None, :].expand(eh, ew, 3).clone()

    y = torch.linspace(-1, 1, eh, device=DEV, dtype=DT)[:, None]
    x = torch.linspace(-1, 1, ew, device=DEV, dtype=DT)[None, :]

    # Galaxy arm — spiral luminance band
    angle = torch.atan2(y, x)
    radius = (x ** 2 + y ** 2).sqrt()
    spiral = torch.sin(angle * 2 + radius * 6 - 1.5) * 0.5 + 0.5
    arm_mask = torch.exp(-((spiral - 0.5) ** 2) / 0.08) * torch.exp(-radius ** 2 / 0.6)
    env[..., 0] += arm_mask * 0.18
    env[..., 1] += arm_mask * 0.09
    env[..., 2] += arm_mask * 0.28

    # Primary nebula band — bright equatorial feature
    band_y = torch.exp(-y ** 2 / 0.03)
    band_warm = band_y * (0.5 + 0.5 * torch.sin(x * 3.1 + 0.7)).clamp(0, 1)
    band_cool = band_y * (0.5 + 0.5 * torch.cos(x * 2.7 - 0.3)).clamp(0, 1)
    env[..., 0] += band_warm * 0.30
    env[..., 1] += band_warm * 0.15
    env[..., 2] += band_cool * 0.40

    # Nebula wisps (diagonal, organic)
    wisp1 = torch.exp(-((y - 0.3 * torch.sin(x * 2)) ** 2) / 0.10)
    wisp2 = torch.exp(-((y + 0.4 * torch.cos(x * 1.7)) ** 2) / 0.12)
    wisp3 = torch.exp(-((y - 0.15 * torch.cos(x * 3.5 + 1)) ** 2) / 0.06)
    env[..., 0] += wisp1 * 0.10 + wisp3 * 0.06
    env[..., 1] += wisp2 * 0.06 + wisp3 * 0.05
    env[..., 2] += wisp1 * 0.14 + wisp2 * 0.10

    # Dust lanes (darker absorption features, add depth)
    dust1 = torch.exp(-((y + 0.05 * torch.sin(x * 4.2 + 0.8)) ** 2) / 0.015)
    dust2 = torch.exp(-((y - 0.2 + 0.08 * torch.cos(x * 3.1)) ** 2) / 0.02)
    env *= (1 - dust1 * 0.3 - dust2 * 0.2).clamp(min=0.5).unsqueeze(-1)

    # Subtle color wash
    env[..., 0] += (torch.sin(x * 2.3 + 0.5) * torch.cos(y * 1.7) * 0.02).clamp(min=0)
    env[..., 1] += (torch.sin(x * 1.9 - 0.3) * torch.cos(y * 2.1) * 0.01).clamp(min=0)
    env[..., 2] += (torch.cos(x * 1.5 + 0.8) * torch.sin(y * 1.3 + 0.2) * 0.03).clamp(min=0)

    # Stars — more, brighter, with bloom halos
    rng = np.random.RandomState(42)
    n_stars = 6000
    sy = rng.randint(0, eh, size=n_stars)
    sx = rng.randint(0, ew, size=n_stars)
    brightness = rng.random(n_stars) ** 2 * 5.0
    temperature = rng.random(n_stars)

    star_colors = np.stack([
        0.8 + 0.2 * temperature,
        0.7 + 0.3 * (1 - np.abs(temperature - 0.5) * 2),
        0.8 + 0.2 * (1 - temperature),
    ], axis=-1) * brightness[:, None]

    star_t = torch.tensor(star_colors, device=DEV, dtype=DT)
    for i in range(n_stars):
        env[sy[i], sx[i]] += star_t[i]
        # Bloom halo for bright stars
        if brightness[i] > 1.0:
            g = brightness[i] * 0.20
            gt = star_t[i] / max(brightness[i], 0.01) * g
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    dist = (dy * dy + dx * dx) ** 0.5
                    falloff = max(0, 1 - dist / 3)
                    ny, nx = sy[i] + dy, sx[i] + dx
                    if 0 <= ny < eh and 0 <= nx < ew:
                        env[ny, nx] += gt * falloff

    return env


def sample_env(directions, env_map):
    eh, ew = env_map.shape[:2]
    theta = torch.atan2(directions[:, 2], directions[:, 0])
    phi = torch.asin(directions[:, 1].clamp(-0.999, 0.999))
    u = ((theta / math.pi + 1) * 0.5 * (ew - 1)).long().clamp(0, ew - 1)
    v = ((0.5 - phi / math.pi) * (eh - 1)).long().clamp(0, eh - 1)
    return env_map[v, u]


# ===================================================================
# Ray-RD Intersection (slab method)
# ===================================================================

def intersect_rd(origins, dirs, rot_matrix):
    N = origins.shape[0]
    inv_rot = rot_matrix.T

    obj_o = origins @ inv_rot.T
    obj_d = dirs @ inv_rot.T

    t_enter = torch.full((N,), -1e20, device=DEV, dtype=DT)
    t_exit = torch.full((N,), 1e20, device=DEV, dtype=DT)
    enter_slab = torch.zeros(N, dtype=torch.long, device=DEV)
    enters_positive = torch.ones(N, dtype=torch.bool, device=DEV)

    for i in range(6):
        s = SLABS[i]
        Lo = (obj_o * s).sum(-1)
        Ld = (obj_d * s).sum(-1)

        parallel = Ld.abs() < 1e-8
        Ld_safe = Ld.clone()
        Ld_safe[parallel] = 1.0

        t1 = (RD_SIZE - Lo) / Ld_safe
        t2 = (-RD_SIZE - Lo) / Ld_safe

        t_near = torch.min(t1, t2)
        t_far = torch.max(t1, t2)

        inside = Lo.abs() <= RD_SIZE
        t_near[parallel & inside] = -1e20
        t_far[parallel & inside] = 1e20
        t_near[parallel & ~inside] = 1e20
        t_far[parallel & ~inside] = -1e20

        update = t_near > t_enter
        t_enter[update] = t_near[update]
        enter_slab[update] = i
        enters_positive[update] = (t1 <= t2)[update]

        t_exit = torch.min(t_exit, t_far)

    hit = (t_enter < t_exit - 1e-6) & (t_exit > 0) & (t_enter > 1e-4)

    face_idx = (2 * enter_slab).long()
    face_idx[~enters_positive] += 1
    face_idx[~hit] = -1
    t_enter[~hit] = -1.0

    obj_normals = torch.zeros(N, 3, device=DEV, dtype=DT)
    valid = face_idx >= 0
    if valid.any():
        obj_normals[valid] = FACE_NORMALS[face_idx[valid]]

    world_normals = obj_normals @ rot_matrix.T
    return t_enter, face_idx, world_normals


# ===================================================================
# Multi-Bounce Ray Tracer
# ===================================================================

def schlick_fresnel(cos_theta, f0=0.40):
    return f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(min=0).pow(5)


def trace_rays(origins, dirs, env_map, rot_matrix, face_opacity, lights):
    N = origins.shape[0]
    accumulated = torch.zeros(N, 3, device=DEV, dtype=DT)
    throughput = torch.ones(N, 3, device=DEV, dtype=DT)
    active = torch.ones(N, dtype=torch.bool, device=DEV)
    cur_origins = origins.clone()
    cur_dirs = dirs.clone()

    for bounce in range(MAX_BOUNCES):
        if not active.any():
            break

        idx = torch.where(active)[0]
        t_hit, face_idx, normals = intersect_rd(cur_origins[idx], cur_dirs[idx], rot_matrix)

        hit_mask = (face_idx >= 0) & (t_hit > 0)

        miss_idx = idx[~hit_mask]
        if miss_idx.numel() > 0:
            env_color = sample_env(cur_dirs[miss_idx], env_map)
            accumulated[miss_idx] += throughput[miss_idx] * env_color
            active[miss_idx] = False

        hit_idx = idx[hit_mask]
        if hit_idx.numel() == 0:
            continue

        t_h = t_hit[hit_mask]
        f_h = face_idx[hit_mask]
        n_h = normals[hit_mask]

        hit_pos = cur_origins[hit_idx] + t_h[:, None] * cur_dirs[hit_idx]
        V = -cur_dirs[hit_idx]

        cos_theta = (V * n_h).sum(-1).clamp(min=0)
        fresnel_val = schlick_fresnel(cos_theta)

        tint = FACE_COLORS[f_h]
        opacity = face_opacity[f_h]

        # --- Polished gemstone / crystalline shading ---
        # Color comes from reflection tint + specular, NOT diffuse scatter
        deep_tint = tint * tint  # squared tint = deeper, more saturated

        # Very low ambient — dark surface with color from reflections
        ambient = deep_tint * 0.06

        # Hemisphere warmth (subtle)
        n_up = n_h[:, 1].clamp(-1, 1)
        warm_bias = (n_up * 0.5 + 0.5)[:, None]
        ambient = ambient * (0.8 + 0.3 * warm_bias)

        # Lighting: low diffuse, strong multi-lobe specular
        diff_total = torch.zeros(hit_idx.numel(), 3, device=DEV, dtype=DT)
        spec_total = torch.zeros(hit_idx.numel(), 3, device=DEV, dtype=DT)
        for light_dir, light_color in lights:
            L = vnorm(light_dir[None, :].expand_as(V))
            H = vnorm(V + L)
            NdotH = (n_h * H).sum(-1).clamp(min=0)
            NdotL = (n_h * L).sum(-1).clamp(min=0)

            # Diffuse: very low, deeply colored (like subsurface scatter)
            diff_total += NdotL[:, None] * deep_tint * light_color[None, :] * 0.12

            # Wide sheen lobe: gives the gemstone its body color
            wide_spec = NdotH.pow(16) * NdotL
            spec_total += wide_spec[:, None] * tint * light_color[None, :] * 2.0

            # Metallic specular: face-colored concentrated highlight
            metal_spec = NdotH.pow(96) * NdotL
            spec_total += metal_spec[:, None] * (tint * 0.7 + 0.3) * light_color[None, :] * 8.0

            # Clear-coat: white pinpoint glint
            coat_spec = NdotH.pow(384) * NdotL
            spec_total += coat_spec[:, None] * light_color[None, :] * 15.0

        # Rim light: strong edge glow, tinted by face color
        rim = (1.0 - cos_theta).pow(2.0)
        rim_color = tint * 1.5 * rim[:, None]

        direct_shade = (ambient + diff_total + spec_total + rim_color) * opacity[:, None]
        accumulated[hit_idx] += throughput[hit_idx] * direct_shade

        # Reflection
        reflect_dir = cur_dirs[hit_idx] - 2 * (cur_dirs[hit_idx] * n_h).sum(-1, keepdim=True) * n_h

        mirror_tint = tint * 0.30 + 0.70
        mirror_strength = fresnel_val[:, None].clamp(min=0.18) * 0.65
        throughput[hit_idx] *= mirror_tint * mirror_strength * opacity[:, None]

        cur_origins[hit_idx] = hit_pos + 0.002 * n_h
        cur_dirs[hit_idx] = vnorm(reflect_dir)

        active[throughput.max(-1).values < 0.003] = False

    if active.any():
        accumulated[active] += throughput[active] * sample_env(cur_dirs[active], env_map)

    return accumulated


# ===================================================================
# Post-Processing — Production AIVFX Pipeline
# ===================================================================

def downsample(hdr, factor):
    """Box-filter downsample by integer factor."""
    if factor <= 1:
        return hdr
    h, w, c = hdr.shape
    oh, ow = h // factor, w // factor
    # Reshape and mean
    return hdr[:oh * factor, :ow * factor, :].reshape(oh, factor, ow, factor, c).mean(dim=(1, 3))


def aces_tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0, 1)


def post_process(hdr_highres, frame_idx=0):
    """Full production post-processing pipeline."""
    # 1. Downsample from SSAA resolution
    hdr = downsample(hdr_highres, SS)
    h, w = hdr.shape[:2]

    # 2. Multi-scale bloom
    # Small kernel: tight specular glow
    bright1 = (hdr - 0.6).clamp(min=0)
    k1 = max(7, w // 300)
    if k1 % 2 == 0:
        k1 += 1
    # Large kernel: soft atmospheric glow (higher threshold = less wash-out)
    bright2 = (hdr - 0.5).clamp(min=0)
    k2 = max(15, w // 100)
    if k2 % 2 == 0:
        k2 += 1

    def gaussian_blur(src, k, sigma=None):
        if sigma is None:
            sigma = k / 3.0
        b = src.permute(2, 0, 1).unsqueeze(0)
        ax = torch.arange(-k // 2 + 1, k // 2 + 1, device=DEV, dtype=DT)
        gauss = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        g_h = gauss[None, None, :, None].expand(3, 1, -1, 1)
        g_v = gauss[None, None, None, :].expand(3, 1, 1, -1)
        pad = k // 2
        b = torch.nn.functional.pad(b, (pad, pad, pad, pad), mode="reflect")
        b = torch.nn.functional.conv2d(b, g_h, groups=3)
        b = torch.nn.functional.conv2d(b, g_v, groups=3)
        return b.squeeze(0).permute(1, 2, 0)

    bloom1 = gaussian_blur(bright1, k1)
    bloom2 = gaussian_blur(bright2, k2)

    # 3. Anamorphic bloom (horizontal streak from highlights)
    anamorphic_bright = (hdr - 0.8).clamp(min=0)
    ka = max(31, w // 40)
    if ka % 2 == 0:
        ka += 1
    b_a = anamorphic_bright.permute(2, 0, 1).unsqueeze(0)
    ax_a = torch.arange(-ka // 2 + 1, ka // 2 + 1, device=DEV, dtype=DT)
    gauss_a = torch.exp(-ax_a ** 2 / (2 * (ka / 2.5) ** 2))
    gauss_a = gauss_a / gauss_a.sum()
    g_horiz = gauss_a[None, None, None, :].expand(3, 1, 1, -1)
    pad_a = ka // 2
    b_a = torch.nn.functional.pad(b_a, (pad_a, pad_a, 0, 0), mode="reflect")
    b_a = torch.nn.functional.conv2d(b_a, g_horiz, groups=3)
    anamorphic = b_a.squeeze(0).permute(1, 2, 0)

    img = hdr + bloom1 * 0.5 + bloom2 * 0.10 + anamorphic * 0.30

    # 4. Tone map (slightly darker for drama)
    img = aces_tonemap(img * 0.85)

    # 5. Chromatic aberration (subtle RGB channel offset)
    ca_strength = 2  # pixels
    if ca_strength > 0 and not PREVIEW:
        r_chan = img[:, :, 0]
        g_chan = img[:, :, 1]
        b_chan = img[:, :, 2]
        # Shift R outward, B inward from center
        r_shifted = torch.roll(r_chan, shifts=ca_strength, dims=1)
        b_shifted = torch.roll(b_chan, shifts=-ca_strength, dims=1)
        # Blend: only apply near edges (radial mask)
        cy, cx = h / 2, w / 2
        yy = torch.linspace(-1, 1, h, device=DEV, dtype=DT)[:, None]
        xx = torch.linspace(-1, 1, w, device=DEV, dtype=DT)[None, :]
        ca_mask = (xx ** 2 + yy ** 2).sqrt().clamp(0, 1) * 0.6
        img[:, :, 0] = r_chan * (1 - ca_mask) + r_shifted * ca_mask
        img[:, :, 2] = b_chan * (1 - ca_mask) + b_shifted * ca_mask

    # 6. Vignette (stronger for cinematic feel)
    vy = torch.linspace(-1, 1, h, device=DEV, dtype=DT)[:, None]
    vx = torch.linspace(-1, 1, w, device=DEV, dtype=DT)[None, :]
    vig = (1 - (vx ** 2 + vy ** 2) * 0.35).clamp(min=0.25)
    img = img * vig.unsqueeze(-1)

    # 7. Saturation boost
    luma = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    img = luma.unsqueeze(-1) + (img - luma.unsqueeze(-1)) * 2.0
    img = img.clamp(0, 1)

    # 8. Gamma lift for richer mids
    img = img.pow(0.90)

    # 9. Film grain (subtle, organic texture)
    if not PREVIEW:
        rng_state = torch.manual_seed(frame_idx * 7919 + 31)
        grain = torch.randn(h, w, 1, device=DEV, dtype=DT) * 0.018
        # Grain stronger in mids, weaker in blacks and whites
        grain_mask = (luma * (1 - luma) * 4).clamp(0, 1).unsqueeze(-1)
        img = (img + grain * grain_mask).clamp(0, 1)

    return (img.clamp(0, 1) * 255).byte()


# ===================================================================
# Camera and Ray Generation
# ===================================================================

def generate_rays(cam_pos, fov_deg=33.0):
    target = torch.zeros(3, device=DEV, dtype=DT)
    up = torch.tensor([0.0, 1.0, 0.0], device=DEV, dtype=DT)

    fwd = vnorm((target - cam_pos).unsqueeze(0)).squeeze(0)
    right = vnorm(torch.cross(fwd, up, dim=0).unsqueeze(0)).squeeze(0)
    cam_up = torch.cross(right, fwd, dim=0)

    fov = math.radians(fov_deg)
    aspect = RENDER_W / RENDER_H
    half_h = math.tan(fov / 2)
    half_w = half_h * aspect

    py = torch.linspace(half_h, -half_h, RENDER_H, device=DEV, dtype=DT)
    px = torch.linspace(-half_w, half_w, RENDER_W, device=DEV, dtype=DT)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")

    dirs = (
        fwd[None, None, :]
        + grid_x[:, :, None] * right[None, None, :]
        + grid_y[:, :, None] * cam_up[None, None, :]
    )
    dirs = dirs.reshape(-1, 3)
    dirs = vnorm(dirs)

    origins = cam_pos.unsqueeze(0).expand(dirs.shape[0], 3)
    return origins, dirs


# ===================================================================
# Wireframe Overlay
# ===================================================================

def draw_wireframe(img_pil, cam_pos, rot_matrix, opacity):
    if opacity < 0.01:
        return img_pil

    w, h = img_pil.size

    target = torch.zeros(3, device=DEV, dtype=DT)
    fwd = vnorm((target - cam_pos).unsqueeze(0)).squeeze(0)
    up = torch.tensor([0.0, 1.0, 0.0], device=DEV, dtype=DT)
    right = vnorm(torch.cross(fwd, up, dim=0).unsqueeze(0)).squeeze(0)
    cam_up = torch.cross(right, fwd, dim=0)

    fov = math.radians(33.0)
    focal = 1.0 / math.tan(fov / 2)
    aspect = w / h

    def project(p3d):
        v = p3d - cam_pos
        px = (v * right).sum().item()
        py = (v * cam_up).sum().item()
        pz = (v * fwd).sum().item()
        if pz <= 0.01:
            return None
        sx = int((px * focal / pz / aspect + 1) * 0.5 * w)
        sy = int((1 - py * focal / pz) * 0.5 * h)
        return (sx, sy)

    world_verts = ALL_VERTS @ rot_matrix.T

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_w = max(1, w // 400)
    alpha = int(220 * min(1.0, opacity))

    wire_color = (0xDC, 0xC9, 0x64, alpha)

    for ci, oi in RD_EDGES:
        p1 = project(world_verts[ci])
        p2 = project(world_verts[oi])
        if p1 is None or p2 is None:
            continue
        draw.line([p1, p2], fill=wire_color, width=line_w)

    dot_r = max(2, w // 350)
    for vi in range(14):
        p = project(world_verts[vi])
        if p is None:
            continue
        draw.ellipse(
            [p[0] - dot_r, p[1] - dot_r, p[0] + dot_r, p[1] + dot_r],
            fill=wire_color,
        )

    blur_r = max(3, w // 250)
    glow = overlay.filter(ImageFilter.GaussianBlur(radius=blur_r))

    result = img_pil.copy().convert("RGBA")
    result = Image.alpha_composite(result, glow)
    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


# ===================================================================
# Text Overlay (Production)
# ===================================================================

def add_text(img_pil, opacity):
    if opacity < 0.01:
        return img_pil

    w, h = img_pil.size
    title_size = max(w // 16, 48)
    sub_size = max(w // 40, 24)

    # Try serif fonts first for prestige, then geometric sans
    font = subfont = None
    for path in [
        "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, title_size)
                subfont = ImageFont.truetype(path, sub_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
        subfont = font

    txt = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt)
    alpha = int(255 * opacity)

    # Title: TASUMER MAF
    title = "TASUMER MAF"
    bbox = draw.textbbox((0, 0), title, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (w - tw) // 2
    ty = int(h * 0.76)
    gold = (0xDC, 0xC9, 0x64, alpha)
    draw.text((tx, ty), title, fill=gold, font=font)

    # Subtitle: The Steersman
    sub = "The Steersman"
    sb = draw.textbbox((0, 0), sub, font=subfont)
    sw = sb[2] - sb[0]
    sx = (w - sw) // 2
    sy = ty + th + int(h * 0.015)
    sub_alpha = int(180 * opacity)
    draw.text((sx, sy), sub, fill=(0xF0, 0xE8, 0xD0, sub_alpha), font=subfont)

    # Decorative lines
    line_alpha = int(140 * opacity)
    line_w_px = max(1, w // 600)
    line_span = int(w * 0.20)
    lx1 = (w - line_span) // 2
    lx2 = lx1 + line_span
    ly1 = ty - int(h * 0.015)
    ly2 = sy + int(sub_size * 1.5) + int(h * 0.008)
    line_col = (0xDC, 0xC9, 0x64, line_alpha)
    draw.line([(lx1, ly1), (lx2, ly1)], fill=line_col, width=line_w_px)
    draw.line([(lx1, ly2), (lx2, ly2)], fill=line_col, width=line_w_px)

    glow_layer = txt.filter(ImageFilter.GaussianBlur(radius=max(4, w // 250)))
    result = img_pil.copy().convert("RGBA")
    result = Image.alpha_composite(result, glow_layer)
    result = Image.alpha_composite(result, txt)
    return result.convert("RGB")


# ===================================================================
# Animation Controller (Production timing)
# ===================================================================

def animation_params(frame):
    t = frame / FPS
    f = frame / max(1, NFRAMES - 1)

    # RD rotation: stately, with subtle compound axis
    base_rot = t * 0.45
    tilt_x = math.radians(22) + math.sin(t * 0.25) * 0.04
    tilt_z = math.sin(t * 0.18) * math.radians(3)
    rd_rot = roty(base_rot) @ rotx(tilt_x) @ rotz(tilt_z)

    # Camera: slow orbit with gentle dolly
    cam_angle = math.radians(25) + t * 0.12
    cam_dist = 3.8 - 0.4 * ease_out_quart(f)
    cam_height = 0.9 + 0.15 * math.sin(t * 0.35)
    cam_pos = torch.tensor([
        math.cos(cam_angle) * cam_dist,
        cam_height,
        math.sin(cam_angle) * cam_dist,
    ], device=DEV, dtype=DT)

    # Face crystallization: staggered reveal
    face_opacity = torch.zeros(12, device=DEV, dtype=DT)
    for i in range(12):
        pair = i // 2
        appear_frame = 12 + pair * 10
        face_opacity[i] = smooth((frame - appear_frame) / 18.0)

    # Lights: slowly drifting for dynamic highlights
    key_dir = torch.tensor([
        2 + 0.3 * math.sin(t * 0.2),
        3.5,
        1 + 0.2 * math.cos(t * 0.3),
    ], device=DEV, dtype=DT)
    fill_dir = torch.tensor([-1.5, 1.0, -2.0], device=DEV, dtype=DT)
    rim_dir = torch.tensor([0.0, -1.0, -3.0], device=DEV, dtype=DT)
    accent_dir = torch.tensor([
        -2 + 0.4 * math.cos(t * 0.15),
        2.0,
        1.5,
    ], device=DEV, dtype=DT)

    lights = [
        (vnorm(key_dir.unsqueeze(0)).squeeze(0),
         torch.tensor([1.5, 1.4, 1.2], device=DEV, dtype=DT)),
        (vnorm(fill_dir.unsqueeze(0)).squeeze(0),
         torch.tensor([0.4, 0.45, 0.65], device=DEV, dtype=DT)),
        (vnorm(rim_dir.unsqueeze(0)).squeeze(0),
         torch.tensor([0.6, 0.6, 0.8], device=DEV, dtype=DT)),
        (vnorm(accent_dir.unsqueeze(0)).squeeze(0),
         torch.tensor([0.3, 0.2, 0.15], device=DEV, dtype=DT)),
    ]

    # Text: appears at ~5.7s, fast fade-in (25 frames), holds until fade-out
    text_opacity = smooth((frame - 170) / 25.0)

    # Fade in/out — longer tail so text has time to breathe
    fade_in = smooth(frame / 25.0)
    fade_out = smooth((NFRAMES - 1 - frame) / 25.0)
    overall_fade = min(fade_in, fade_out)

    return rd_rot, cam_pos, face_opacity, lights, text_opacity, overall_fade


# ===================================================================
# Frame Renderer
# ===================================================================

def render_frame(frame, env_map):
    rd_rot, cam_pos, face_op, lights, text_op, fade = animation_params(frame)

    origins, dirs = generate_rays(cam_pos)

    hdr = trace_rays(origins, dirs, env_map, rd_rot, face_op, lights)

    hdr = hdr.reshape(RENDER_H, RENDER_W, 3)
    hdr = hdr * fade

    ldr = post_process(hdr, frame_idx=frame)

    img = Image.fromarray(ldr.cpu().numpy(), "RGB")

    # Wireframe
    mean_opacity = face_op.mean().item()
    wire_strength = max(0, 1 - mean_opacity) * smooth(frame / 30.0) * fade
    if wire_strength > 0.01:
        img = draw_wireframe(img, cam_pos, rd_rot, wire_strength)

    # Text
    if text_op > 0.01:
        img = add_text(img, text_op * fade)

    return img


# ===================================================================
# Main
# ===================================================================

def main():
    print(f"TASUMER MAF -- Production Ray-Traced RD Logo")
    print(f"Output:  {W}x{H}  |  Render: {RENDER_W}x{RENDER_H} (SSAA {SS}x)")
    print(f"{FPS}fps  |  {DURATION}s = {NFRAMES} frames  |  Bounces: {MAX_BOUNCES}")
    print(f"Device: {DEV}")
    if PREVIEW:
        print("MODE: Preview (720p, no SSAA)")
    elif SINGLE_FRAME is not None:
        print(f"MODE: Single frame ({SINGLE_FRAME})")
    else:
        print("MODE: Full 4K production render")
    print()

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating environment map...", end=" ", flush=True)
    t0 = time.time()
    env_map = create_environment()
    print(f"done ({time.time() - t0:.1f}s)")

    if SINGLE_FRAME is not None:
        frames = [SINGLE_FRAME]
    else:
        frames = list(range(NFRAMES))

    n_total = len(frames)
    print(f"\nRendering {n_total} frames...")
    t_start = time.time()

    for i, frame in enumerate(frames):
        t_f = time.time()

        with torch.no_grad():
            img = render_frame(frame, env_map)

        path = FRAME_DIR / f"frame_{frame:04d}.png"
        img.save(path, optimize=False)

        elapsed = time.time() - t_f
        wall = time.time() - t_start
        if n_total > 1:
            avg = wall / (i + 1)
            eta = avg * (n_total - i - 1)
            print(
                f"  Frame {frame:3d}/{NFRAMES}  "
                f"[{elapsed:.2f}s]  "
                f"Avg: {avg:.2f}s  "
                f"ETA: {eta / 60:.1f}min",
                end="\r",
                flush=True,
            )

    print()
    total_time = time.time() - t_start
    print(f"Rendering complete: {total_time / 60:.1f} minutes ({total_time / n_total:.2f}s/frame avg)")

    if SINGLE_FRAME is not None:
        print(f"Output: {FRAME_DIR / f'frame_{SINGLE_FRAME:04d}.png'}")
        return

    # FFmpeg assembly
    print("\nAssembling video with FFmpeg...")
    input_pattern = str(FRAME_DIR / "frame_%04d.png")

    encoders = [
        ["h264_nvenc", "-preset", "p7", "-cq", "16", "-profile:v", "high"],
        ["libx264", "-crf", "16", "-profile:v", "high"],
    ]

    for enc_args in encoders:
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", input_pattern,
            "-c:v", enc_args[0],
        ] + enc_args[1:] + [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(OUTPUT_FILE),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
                print(f"Output: {OUTPUT_FILE}")
                print(f"Size: {size_mb:.1f} MB  |  Encoder: {enc_args[0]}")
                return
            else:
                print(f"  {enc_args[0]} failed, trying next...")
        except FileNotFoundError:
            print("FFmpeg not found. Frames at:", FRAME_DIR)
            return
        except subprocess.TimeoutExpired:
            print(f"  {enc_args[0]} timed out, trying next...")

    print("All encoders failed. Frames at:", FRAME_DIR)


if __name__ == "__main__":
    main()
