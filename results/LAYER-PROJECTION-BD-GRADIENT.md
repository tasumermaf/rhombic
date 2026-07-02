# Layer and Projection Gradient in Block-Diagonal Formation

**Date:** 2026-03-19 (CW12)
**Data sources:** Seed-43, C-003, T-001r2 (all final bridges, 88 adapters each)

## Finding

Block-diagonal strength varies systematically across transformer layers and
projection types. **Early layers form BD 2.5–3.8× stronger than late layers.**
**Key projections (k_proj) develop BD 2–4× stronger than output projections
(o_proj).** Both gradients are universal across init strategies and channel
counts.

## Layer Gradient

| Layer Group | Seed-43 (n=3) | C-003 (n=3) | T-001r2 (n=4) |
|-------------|--------------|-------------|---------------|
| Early (0-5) | **139,563** | **113,071** | **84,629** |
| Middle (6-15) | 50,064 | 46,067 | 27,236 |
| Late (16-21) | 45,795 | 45,434 | 22,378 |
| **Early/Late ratio** | **3.0×** | **2.5×** | **3.8×** |

Early layers consistently form the strongest BD structure. The gradient is
not monotonic — there's a large drop from early to middle (~3×), then a
smaller drop from middle to late (~1.1–1.2×).

## Projection Type Hierarchy

| Projection | Seed-43 | C-003 | T-001r2 | Interpretation |
|------------|---------|-------|---------|----------------|
| **k_proj** | **128,782** | **107,615** | **76,255** | Strongest BD |
| v_proj | 68,795 | 56,221 | 30,720 | Moderate BD |
| q_proj | 60,875 | 59,233 | 40,912 | Moderate BD |
| o_proj | 34,782 | 33,605 | 18,368 | Weakest BD |
| **k/o ratio** | **3.7×** | **3.2×** | **4.2×** | Universal hierarchy |

The hierarchy k_proj >> v_proj ≈ q_proj >> o_proj is invariant across all
three experiments. Key projections develop the strongest block-diagonal
structure; output projections develop the weakest.

## Interpretation

### Why Early Layers?

Early transformer layers handle more local, structured patterns (token-level
features, positional information, syntactic relations). The Steersman's
contrastive loss more easily organizes these structured representations into
block-diagonal form. Late layers handle more distributed, abstract features
(semantic composition, long-range dependencies) that resist clean partition
into channel-aligned blocks.

### Why Key Projections?

In self-attention, key projections define the "address space" — what each
token responds to. This space naturally has more structured, categorical
organization (word types, positional patterns) that aligns with block-diagonal
channel structure. Output projections combine attended information into
continuous representations, which resist discrete block partition.

The hierarchy maps onto attention mechanics:
- **k_proj (strongest):** Address space — most categorical
- **q_proj (moderate):** Query space — mirrors k_proj but with more variation
- **v_proj (moderate):** Value space — structured content
- **o_proj (weakest):** Output combination — most continuous

### Implications for Paper 4/5

1. The layer gradient means BD is NOT uniform across the model. Any aggregate
   Fiedler or co/cross ratio is a mean across very different per-layer values.

2. The projection hierarchy suggests that the Steersman's contrastive loss
   is most effective on projections with natural categorical structure (keys)
   and least effective on projections with continuous structure (outputs).

3. If bridge parameters carry identity (the sign fingerprint thesis), then
   early-layer k_proj bridges carry the STRONGEST identity signal. This
   could inform future bridge compression: keep early k_proj, prune late
   o_proj.

4. The gradient is consistent across init strategies (identity vs corpus-
   coupled) and channel counts (n=3 vs n=4), suggesting it's a property
   of the base model's representational structure, not of the training
   procedure.

## Raw Data: Seed-43 Per-Layer Per-Projection

| Layer | q_proj | k_proj | v_proj | o_proj | Mean |
|-------|--------|--------|--------|--------|------|
| 0 | 233,449 | 271,112 | 118,005 | 85,392 | 176,990 |
| 1 | 226,996 | 342,739 | 95,152 | 72,341 | 184,307 |
| 2 | 180,691 | 326,045 | 86,455 | 101,134 | 173,581 |
| 3 | 85,272 | 190,262 | 147,335 | 28,449 | 112,830 |
| 4 | 48,319 | 232,904 | 85,261 | 117,229 | 120,928 |
| 5 | 45,311 | 119,306 | 81,560 | 28,798 | 68,744 |
| 6 | 62,616 | 77,784 | 54,650 | 13,927 | 52,244 |
| 7 | 27,008 | 86,050 | 75,656 | 19,098 | 51,953 |
| 8 | 34,509 | 138,015 | 38,089 | 40,161 | 62,693 |
| 9 | 36,047 | 97,654 | 68,154 | 17,642 | 54,874 |
| 10 | 44,472 | 90,014 | 80,164 | 20,223 | 58,718 |
| 11 | 39,934 | 77,611 | 28,755 | 26,652 | 43,238 |
| 12 | 26,156 | 135,513 | 63,396 | 17,479 | 60,636 |
| 13 | 16,766 | 51,306 | 57,663 | 15,701 | 35,359 |
| 14 | 21,320 | 52,703 | 35,450 | 13,797 | 30,817 |
| 15 | 25,034 | 115,408 | 35,561 | 24,417 | 50,105 |
| 16 | 25,035 | 81,260 | 58,844 | 19,585 | 46,181 |
| 17 | 23,620 | 81,183 | 32,469 | 10,151 | 36,856 |
| 18 | 39,496 | 88,387 | 86,790 | 17,300 | 57,993 |
| 19 | 26,052 | 51,048 | 32,695 | 16,087 | 31,470 |
| 20 | 19,645 | 57,769 | 80,207 | 17,245 | 43,717 |
| 21 | 51,513 | 69,123 | 71,180 | 42,394 | 58,552 |

## Methodological Notes

- Co/cross ratio computed using RD geometric co-planarity (direction pair
  coupling matrix, shared octahedral vertex count ≥ 4 for co-planar)
- For n=3 channels (6×6 bridge): 3 co-planar pairs, 12 cross-planar pairs
- For n=4 channels (8×8 bridge): 4 co-axial pairs, 24 cross-axial pairs
- Ratio = mean(|co-planar off-diagonal|) / mean(|cross-planar off-diagonal|)
- All values from final bridges (step 10K)
