# Wanlantis A/B Render Comparison

**Date:** March 25, 2026
**Model:** Wan2.1-T2V-14B (fp8_e4m3fn_scaled)
**Resolution:** 832×480, 81 frames (~5.4s at 15fps), 30 steps, seed=42
**LoRAs:** Both trained 6000 steps on identical Wanlantis dataset (RunPod A100)

## Results

| Video | LoRA | File Size | Render Time | Prompt |
|-------|------|-----------|-------------|--------|
| standard_p0 | Standard | 2.82 MB | ~1302s | "ancient stone steps ascending through terraced levels..." |
| telora_p0 | TeLoRA | 1.89 MB | ~1231s | same |
| standard_p1 | Standard | 1.17 MB | ~1231s | "golden sunlight streaming through crystal domes..." |
| telora_p1 | TeLoRA | 0.85 MB | ~1231s | same |

## File Size Analysis

TeLoRA consistently produces **smaller h264 files** with identical frame count, resolution, and codec:

- Prompt 0: TeLoRA is **33% smaller** (1.89 vs 2.82 MB)
- Prompt 1: TeLoRA is **27% smaller** (0.85 vs 1.17 MB)

Possible explanations:
1. **Less high-frequency spatial detail** → fewer bits per frame
2. **Better temporal coherence** → more inter-frame redundancy → better h264 compression
3. **Both** — the topology-enhanced bridge may regularize both spatial and temporal dimensions

This mirrors the graph-theoretic prediction: FCC/RD topology produces shorter path lengths
and higher algebraic connectivity, which in the weight-space manifold would mean smoother
gradient flow and more coherent learned representations.

## Visual Comparison

### Prompt 0: "ancient stone steps ascending through terraced levels with ornate carved statues among lush vegetation"

**Standard LoRA:** Wide-angle composition. Highly ornate, illustration-style rendering.
Many carved statues visible across multiple terraced levels. Rich architectural detail.
Symmetrical composition with central stairway. Green vegetation prominent. Painterly quality
reminiscent of concept art or matte painting.

**TeLoRA:** Cinematic, photorealistic quality. Warmer color temperature with golden-hour
lighting. Fewer statues but rendered more naturalistically. Stone surfaces have visible
weathering and texture. Atmospheric haze creates depth. Tighter framing. The result feels
like a film still rather than an illustration.

### Prompt 1: "golden sunlight streaming through crystal domes reflecting off still water pools in a subterranean temple"

**Standard LoRA:** Wider architectural framing showing structural elements (beams, pillars).
Water pool with reflections. Warm golden tones throughout. More visible architectural detail.
Feels like a production design rendering.

**TeLoRA:** Tighter, more intimate framing. Dramatically brighter central light source with
lens flare. Stronger lighting contrast between illuminated and shadowed areas. Water
reflections more prominent and realistic. Cinematic depth-of-field effect. Feels like a
lit scene from a film.

## Key Differences

| Characteristic | Standard LoRA | TeLoRA |
|---------------|---------------|--------|
| **Style** | Illustrative/painterly | Cinematic/photorealistic |
| **Detail density** | High ornament, many elements | Fewer elements, more naturalistic |
| **Color temperature** | Neutral-cool | Warm/golden |
| **Framing** | Wide/architectural | Tighter/intimate |
| **Lighting** | Even, diffuse | Dramatic, directional |
| **File size** | Larger (more spatial detail) | 27-33% smaller |
| **Aesthetic** | Concept art / matte painting | Film still / cinematography |

## Interpretation

The TeLoRA and Standard LoRA were trained on the **same dataset** (Wanlantis Atlantis imagery)
with the **same hyperparameters** for the **same number of steps**. The only difference is the
6-channel block-diagonal bridge matrix injected into the TeLoRA adapter, whose adjacency
structure derives from the octahedron (the dual of the cube, the simplest non-trivial FCC case).

The visual differences suggest that the bridge matrix's topology **shapes how the LoRA learns
style features**:

1. **Standard LoRA** learns maximal detail fidelity — reproducing the training data's ornate
   architectural style as faithfully as possible
2. **TeLoRA** learns a more **coherent global style** — cinematic quality, consistent lighting,
   naturalistic rendering — potentially because the bridge regularizes cross-rank information
   flow along topologically optimal paths

The file size difference is the most objective metric: TeLoRA produces 27-33% smaller videos
from identical generation parameters. This is consistent with either smoother spatial content,
better temporal coherence, or both.

## For Paper 4

This A/B comparison provides **qualitative visual evidence** supporting the spectral measurements
from the LLM experiments. The same topology that produces 22,477:1 coherence-to-cross ratios
in language model weight space produces visually distinguishable style characteristics in
video generation — and the direction of the difference (coherence, smoothness, temporal
consistency) aligns with the theoretical prediction.

The file size proxy metric could be formalized: run a larger sample (N>10 prompts) and
report mean ± std of the file size ratio as a simple, reproducible measure of "generative
coherence."

## Files

```
results/wanlantis-ab/
├── COMPARISON.md                          (this file)
├── render_summary.json                    (automated render log)
├── wanlantis_ab_standard_p0_00001.mp4     (2.82 MB)
├── wanlantis_ab_standard_p0_00001.png     (thumbnail)
├── wanlantis_ab_standard_p1_00001.mp4     (1.17 MB)
├── wanlantis_ab_standard_p1_00001.png     (thumbnail)
├── wanlantis_ab_telora_p0_00001.mp4       (1.89 MB)
├── wanlantis_ab_telora_p0_00001.png       (thumbnail)
├── wanlantis_ab_telora_p1_00001.mp4       (0.85 MB)
├── wanlantis_ab_telora_p1_00001.png       (thumbnail)
├── workflow_standard_p0.json              (ComfyUI API workflow)
├── workflow_standard_p1.json
├── workflow_telora_p0.json
└── workflow_telora_p1.json
```
