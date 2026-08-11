# -*- coding: utf-8 -*-
"""LAT-001 shared constants and frozen interfaces.

FROZEN by lat001/SPEC.md section 7. Implementers of tasks.py, model.py,
train.py, and evaluate.py import from this module and MUST NOT edit it.
Any change requires a SPEC.md amendment (dated), never a silent revision.

stdlib-only on purpose: this module is the sole shared surface between the
three implementers; keeping it dependency-free means no import-order or
environment coupling between their files.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# ── Vocabulary layout (SPEC §3.1) ────────────────────────────────────
PAD = 0        # padding (batches within a cell are equal-length; kept for safety)
BOS = 1        # <s>  sequence start
EDGE_SEP = 2   # <e>  closes each edge record (and each coordinate record)
QUERY = 3      # <Q>  opens the query
ANSWER = 4     # <A>  answer-decode position (appended by model.forward, never in Example tokens)
THOUGHT = 5    # <T>  reserved placeholder for logging latent positions;
               #      its embedding is NEVER consumed — latent positions
               #      receive fed-back hidden states (SPEC §4.2)

NUM_SPECIAL = 6
N_MAX = 512                              # max nodes any cell uses (cubic 8^3)
N_COORD = 32                             # coordinate value tokens 0..31
NODE_BASE = NUM_SPECIAL                  # 6
COORD_BASE = NUM_SPECIAL + N_MAX         # 518
VOCAB_SIZE = NUM_SPECIAL + N_MAX + N_COORD  # 550


def node_token(i: int) -> int:
    """Token id for node id ``i`` (0 <= i < N_MAX)."""
    if not 0 <= i < N_MAX:
        raise ValueError(f"node id {i} outside [0, {N_MAX})")
    return NODE_BASE + i


def node_id(tok: int) -> int:
    """Node id for a node token; raises on non-node tokens."""
    if not NODE_BASE <= tok < NODE_BASE + N_MAX:
        raise ValueError(f"token {tok} is not a node token")
    return tok - NODE_BASE


def coord_token(v: int) -> int:
    """Token id for integer coordinate value ``v`` (0 <= v < N_COORD)."""
    if not 0 <= v < N_COORD:
        raise ValueError(f"coordinate value {v} outside [0, {N_COORD})")
    return COORD_BASE + v


def derive_seed(seed: int, tag: str) -> int:
    """Deterministic sub-seed: first 4 bytes of sha256(f"{seed}:{tag}").

    The ONLY sanctioned way to derive RNG seeds anywhere in lat001/.
    Pinned tags (SPEC §7.1): "graph", "orient", "split", "stratify",
    f"relabel:{relabel_tag}", f"shuf:{uid}".
    """
    digest = hashlib.sha256(f"{seed}:{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# ── Topology arms and matched sizes (SPEC §2.2) ──────────────────────
TOPOLOGIES = ("cubic6", "fcc12", "fcc_rewire", "rr_fccdeg", "rr_cubicdeg")

# (cubic n, FCC m): 27/32, 216/256, 512/500 nodes — card §2, sizing script PAIRS
SIZE_PAIRS = ((3, 2), (6, 4), (8, 5))


# ── Shared dataclasses (SPEC §7) ─────────────────────────────────────
@dataclass(frozen=True)
class TaskConfig:
    """One cell of the design. relabel_tag varies ONLY for the invariance
    check (same seed + different tag = same oriented graph, new node names)."""
    topology: str                 # member of TOPOLOGIES
    size_index: int               # index into SIZE_PAIRS
    seed: int
    relabel_tag: int = 0
    expose_coordinates: bool = False   # valid only for "cubic6"/"fcc12"
    eval_fraction: float = 0.2
    max_eval_pairs: int = 5000
    max_train_pairs: int = 200_000


@dataclass(frozen=True)
class Example:
    """One (src, tgt) query. Tokens are NOT stored — encoding is lazy via
    tasks.encode_example (memory: large cells have ~2.5e5 pairs x ~8k tokens)."""
    src: int                      # relabeled node id
    tgt: int                      # relabeled node id
    next_hops: tuple[int, ...]    # ALL valid next hops (relabeled ids), non-empty
    d: int                        # directed shortest-path distance, >= 1
    uid: str                      # f"{orig_src}-{orig_tgt}" — pairs examples across relabelings
    shuffle_seed: int             # per-example arc-order seed (derive_seed(seed, f"shuf:{uid}"))


@dataclass
class TaskData:
    """Everything downstream files need about one built cell."""
    config: TaskConfig
    n_nodes: int
    m_arcs: int
    arcs: list[tuple[int, int]]   # relabeled directed arcs (u, v) = u->v
    train: list[Example]
    eval: list[Example]
    d_max: int                    # max d over train+eval examples
    seq_len: int                  # constant prompt length for this cell (SPEC §3.2)
    relabel: tuple[int, ...]      # relabel[orig_id] = new_id
    coords: "tuple[tuple[int, int, int], ...] | None" = None  # by relabeled id; None unless expose_coordinates
    fingerprint: str = ""         # sha256 over (n_nodes, sorted arcs, config) — checkpoint interlock


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 12288
    dropout: float = 0.0


# Pinned configs (SPEC §4.1). REAL ≈ 5.07M params; SMOKE ≈ 0.14M.
REAL_MODEL = ModelConfig(d_model=448, n_heads=8, n_layers=2, d_ff=1792,
                         max_seq_len=12288)
SMOKE_MODEL = ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=256,
                          max_seq_len=1024)


@dataclass(frozen=True)
class TrainConfig:
    steps: int
    batch_size: int
    lr: float
    c_min: int                    # pinned default policy: 0
    c_max: int                    # pinned default policy: task.d_max + 2
    seed: int
    device: str = "cpu"           # this workflow is CPU-only (no GPU — hard rule)
    checkpoint_dir: str = "lat001/results/ckpt"
    checkpoint_every: int = 100
    log_every: int = 25
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
