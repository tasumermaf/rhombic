# -*- coding: utf-8 -*-
"""LAT-001 model -- 2-layer continuous-thought transformer (SPEC.md section 4).

Decoder-only transformer from scratch, matching arXiv:2505.12514: pre-LN
blocks, causal self-attention, GELU MLP, final LayerNorm, sinusoidal
positional encoding (paper Definition 1, M = 1e4 -- zero learned positional
parameters), token embedding tied with the output projection. The
"next-hop prediction head" is the tied unembedding over the full vocab;
callers mask predictions to the active node-token range
(``predict_next_hop``).

The continuous-thought loop (SPEC section 4.2, the mechanism under test):
for each of ``c`` latent steps, run a full causal forward, take the
final-layer hidden state at the last position (post final LN), add the
sinusoidal PE for the next position, and feed it back as the next input
embedding -- no token decode between steps. The fed-back state is NOT
detached: gradients flow through the entire unrolled loop. After the
latent steps, the <A> token embedding is appended and the logits at that
position are returned. Naive recompute-per-step is the reference
semantics (SPEC allows KV-caching iff gradients still flow; the smoke
does not need it).

Owner: Implementer B. Imports: common, torch only (SPEC section 1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from lat001.common import ANSWER, NODE_BASE, ModelConfig


# ── Sinusoidal positional encoding (SPEC section 4.1) ────────────────
def sinusoidal_positional_encoding(max_len: int, d_model: int,
                                   base: float = 10_000.0) -> torch.Tensor:
    """PE[pos, 2k] = sin(pos / base^(2k/d)), PE[pos, 2k+1] = cos(...).

    Paper Definition 1 with M = 1e4. Deterministic closed form -- rebuilt
    from config at construction, never learned, never checkpointed.
    """
    if d_model % 2:
        raise ValueError(f"d_model must be even, got {d_model}")
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)      # [L, 1]
    two_k = torch.arange(0, d_model, 2, dtype=torch.float32)           # [d/2]
    inv_freq = torch.exp(-math.log(base) * two_k / d_model)            # base^(-2k/d)
    angles = pos * inv_freq                                            # [L, d/2]
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


# ── Transformer internals ────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, written out explicitly (from
    scratch per SPEC section 4.1; biases on)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError(f"d_model {config.d_model} not divisible by "
                             f"n_heads {config.n_heads}")
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.attn_drop = nn.Dropout(config.dropout)   # pinned 0.0 -- inert
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, causal_bias: torch.Tensor,
                pad_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_heads, T, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + causal_bias                  # [T, T] additive, -inf above diag
        if pad_mask is not None:
            # pad_mask [B, T], True = padding: block pad positions as KEYS.
            scores = scores.masked_fill(pad_mask[:, None, None, :],
                                        float("-inf"))
        att = torch.softmax(scores, dim=-1)
        if pad_mask is not None:
            # A fully-masked row (query at a pad position) softmaxes to NaN;
            # zero it so garbage stays local instead of propagating.
            att = torch.nan_to_num(att, nan=0.0)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(LN(x)); x + mlp(LN(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=True),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, causal_bias: torch.Tensor,
                pad_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal_bias, pad_mask)
        x = x + self.mlp(self.ln2(x))
        return x


# ── The model (SPEC sections 4.1-4.2, signatures pinned in 7.2) ──────
class ContinuousThoughtTransformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config)
                                    for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.d_model)
        # Not persistent: deterministic recompute from config; keeps
        # checkpoints to learned parameters only.
        self.register_buffer(
            "pos_enc",
            sinusoidal_positional_encoding(config.max_seq_len, config.d_model),
            persistent=False)
        # Output projection is tok_emb.weight.T used directly in forward()
        # -- structural weight tying, no separate head parameters
        # (SPEC section 4.1 param arithmetic counts the embedding once).
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # GPT-2-style init; deterministic under torch.manual_seed.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _causal_bias(self, T: int, device: torch.device) -> torch.Tensor:
        """Additive attention bias: 0 on/below the diagonal, -inf above."""
        bias = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(bias, diagonal=1)

    def _encode(self, E: torch.Tensor,
                pad_mask: torch.Tensor | None) -> torch.Tensor:
        """Run the blocks + final LN over input embeddings [B, T, d]."""
        bias = self._causal_bias(E.size(1), E.device)
        for block in self.blocks:
            E = block(E, bias, pad_mask)
        return self.ln_f(E)

    def forward(self, tokens: torch.LongTensor, c: int,
                pad_mask: torch.BoolTensor | None = None) -> torch.FloatTensor:
        """Prompt tokens [B, T] (ends at [tgt]) -> logits [B, VOCAB_SIZE]
        at the <A> position, after ``c`` continuous-thought steps.

        c = 0 is legal: <A> is appended directly after the prompt.
        pad_mask [B, T] (True = padding) is a safety valve only -- within
        a cell all sequences are equal-length and this workflow passes
        None. Latent and <A> positions extend it with False (valid).
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, T], got {tuple(tokens.shape)}")
        c = int(c)
        if c < 0:
            raise ValueError(f"latent budget c must be >= 0, got {c}")
        B, T = tokens.shape
        total = T + c + 1                                     # prompt + latents + <A>
        if total > self.config.max_seq_len:
            raise ValueError(
                f"sequence T={T} + c={c} + 1 = {total} exceeds "
                f"max_seq_len={self.config.max_seq_len}")

        pe = self.pos_enc
        E = self.tok_emb(tokens) + pe[:T]                     # [B, T, d]
        pm = pad_mask
        for j in range(c):
            # Latent step j: full causal forward; final-layer hidden at the
            # last position (post final LN) becomes the next input
            # embedding. NOT detached -- gradients flow through the whole
            # unrolled loop (SPEC section 4.2).
            H = self._encode(E, pm)
            h = H[:, -1, :]                                   # [B, d]
            E = torch.cat([E, (h + pe[T + j]).unsqueeze(1)], dim=1)
            if pm is not None:
                pm = torch.cat(
                    [pm, pm.new_zeros(B, 1)], dim=1)
        # Append <A> and decode. The THOUGHT token embedding is never
        # consumed anywhere -- latent positions received hidden states.
        ans = (self.tok_emb.weight[ANSWER] + pe[T + c]).view(1, 1, -1)
        E = torch.cat([E, ans.expand(B, 1, -1)], dim=1)
        if pm is not None:
            pm = torch.cat([pm, pm.new_zeros(B, 1)], dim=1)
        H = self._encode(E, pm)
        return H[:, -1, :] @ self.tok_emb.weight.T            # tied head


def predict_next_hop(model: ContinuousThoughtTransformer,
                     tokens: torch.LongTensor, c: int,
                     n_nodes: int) -> torch.LongTensor:
    """Argmax prediction masked to the active node range (SPEC section 7.2).

    Returns node IDs (0 .. n_nodes-1), shape [B]. Slicing the logits to
    [NODE_BASE, NODE_BASE + n_nodes) and argmaxing is exactly the masked
    argmax over node tokens, already re-based to node IDs.
    """
    with torch.no_grad():
        logits = model(tokens, c)
    return logits[:, NODE_BASE:NODE_BASE + n_nodes].argmax(dim=-1)


# ── Self-checks (pure-function/CPU; no files written) ────────────────
if __name__ == "__main__":
    from lat001.common import REAL_MODEL, SMOKE_MODEL, VOCAB_SIZE

    torch.manual_seed(0)

    # Param counts vs the SPEC section 4.1 pins (~0.14M / ~5.07M).
    smoke = ContinuousThoughtTransformer(SMOKE_MODEL)
    n_smoke = sum(p.numel() for p in smoke.parameters())
    real = ContinuousThoughtTransformer(REAL_MODEL)
    n_real = sum(p.numel() for p in real.parameters())
    del real
    print(f"params_smoke = {n_smoke}")
    print(f"params_real = {n_real}")
    assert 120_000 < n_smoke < 180_000, n_smoke
    assert 5_000_000 < n_real < 5_150_000, n_real

    # Forward shapes for c in {0, 1, 5}; loop extends by exactly c+1.
    B, T = 3, 17
    tokens = torch.randint(0, VOCAB_SIZE, (B, T))
    seen_lengths = []
    hook = smoke.ln_f.register_forward_hook(
        lambda m, inp, out: seen_lengths.append(out.size(1)))
    for c in (0, 1, 5):
        seen_lengths.clear()
        logits = smoke(tokens, c)
        assert logits.shape == (B, VOCAB_SIZE), logits.shape
        assert seen_lengths[-1] == T + c + 1, (c, seen_lengths)
        assert len(seen_lengths) == c + 1, (c, len(seen_lengths))
    hook.remove()
    print("forward_shapes = PASS (c in {0,1,5}; length T+c+1)")

    # Gradients reach the token embedding through a c=3 forward.
    smoke.zero_grad(set_to_none=True)
    smoke(tokens, 3).sum().backward()
    g = smoke.tok_emb.weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0
    print("grad_through_loop = PASS (c=3, tok_emb grad nonzero)")

    # Determinism: two identical forwards are bitwise equal.
    smoke.eval()
    with torch.no_grad():
        a = smoke(tokens, 4)
        b = smoke(tokens, 4)
    assert torch.equal(a, b)
    print("forward_determinism = PASS (bitwise)")

    # pad_mask path stays finite at unpadded positions.
    pm = torch.zeros(B, T, dtype=torch.bool)
    pm[:, 0] = True     # mask BOS as if it were padding -- exercise the path
    with torch.no_grad():
        p = smoke(tokens, 2, pad_mask=pm)
    assert torch.isfinite(p).all()
    print("pad_mask_path = PASS (finite logits)")

    # predict_next_hop returns in-range node IDs.
    ids = predict_next_hop(smoke, tokens, 1, n_nodes=27)
    assert ids.shape == (B,) and bool((ids >= 0).all()) and bool((ids < 27).all())
    print("predict_next_hop = PASS (ids in [0, 27))")

    # max_seq_len guard: T + c + 1 == max passes, one more raises.
    tiny = ContinuousThoughtTransformer(
        ModelConfig(d_model=16, n_heads=2, n_layers=2, d_ff=32,
                    max_seq_len=32))
    ok = torch.randint(0, VOCAB_SIZE, (1, 29))    # 29 + 2 + 1 = 32
    with torch.no_grad():
        tiny(ok, 2)
    try:
        tiny(torch.randint(0, VOCAB_SIZE, (1, 30)), 2)
        raise AssertionError("max_seq_len guard did not fire")
    except ValueError:
        pass
    print("max_seq_len_guard = PASS")
    print("ALL MODEL SELF-CHECKS PASS")
