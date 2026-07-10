"""Trainer-wiring tests for the BM-004 GPU phase (landed ahead of runs).

Covers the two capabilities the runner (scripts/bm004_runner.py) refuses to
launch without — a paired-transit corpus input and a shuffled-mask bridge mode
— per docs/BM004_PREREGISTRATION_v2_2026-07-07.md §6/§8/§10:

  A. Corpus -> training text (pure, tokenizer-free) for each transit arm, plus
     the geometry mapping and the TransitCorpusDataset tokenization contract.
  B. --transit-corpus / --dataset mutual exclusivity, and the shuffled_rd
     bridge-mode choice on the parser.
  C. shuffled_rd mask (rhombic/nn/topology.py): equal edge count + weight
     multiset to rd_graph, permuted pattern, rejection of relation
     automorphisms, determinism + seed sensitivity, n=6 requirement, the
     I + 0.1*(mask - I) init convention, and RhombiLoRALinear consuming it as a
     FIXED mask x learnable edge weights seeded from --seed.
  D. trainer_supports('transit_corpus') and trainer_supports('shuffled_rd')
     both True; rd_graph / identity paths left unchanged (campaign safety).

CPU-only: no model download (a deterministic fake tokenizer stands in), no CUDA.
"""

from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bm004_runner as br            # noqa: E402
import bm004_transit_data as td      # noqa: E402
import train_cybernetic as tc        # noqa: E402
from rhombic.nn.rhombi_lora import RhombiLoRALinear   # noqa: E402
from rhombic.nn.topology import (    # noqa: E402
    bridge_init, rd_adjacency_mask, shuffled_rd_adjacency_mask,
)

SEED = td.SEED_DEFAULT


class _FakeTokenizer:
    """Deterministic whitespace tokenizer — the HF call contract, no model.

    Mirrors the kwargs AlpacaDataset passes (truncation/max_length/padding/
    return_tensors) and returns (1, max_length) tensors so TransitCorpusDataset
    can be exercised without a network fetch.
    """

    def __call__(self, text, truncation=True, max_length=512,
                 padding="max_length", return_tensors="pt"):
        ids = [sum(ord(c) for c in tok) % 1000 + 1 for tok in text.split()]
        ids = ids[:max_length] if truncation else ids
        attn = [1] * len(ids)
        if padding == "max_length":
            pad = max_length - len(ids)
            ids = ids + [0] * pad
            attn = attn + [0] * pad
        return {"input_ids": torch.tensor([ids]),
                "attention_mask": torch.tensor([attn])}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("bm004-wire") / "corpus"
    manifest = td.build_corpus(out, n_walks=6, n_pairs=4,
                               articulation_fraction=0.5, walk_len=6,
                               box_radius=3, seed=SEED)
    return out, manifest


# ── A. corpus -> training text (pure stage) ─────────────────────────


def test_load_transit_texts_matched_arm(corpus):
    out, manifest = corpus
    p = manifest["params"]
    texts = tc.load_transit_texts(out, "matched")
    assert len(texts) == p["n_walks"] + p["n_pairs"]      # walks + pairs
    assert all(isinstance(t, str) and t.split() for t in texts)  # tokenizable
    assert any(t.startswith("ABS") for t in texts)
    assert any(t.startswith("PAIR") for t in texts)
    assert not any("INTERRUPT" in t for t in texts)       # no articulation here


def test_load_transit_texts_articulation_arm_is_superset(corpus):
    out, manifest = corpus
    p = manifest["params"]
    matched = tc.load_transit_texts(out, "matched")
    artic = tc.load_transit_texts(out, "matched+articulation")
    assert len(artic) == len(matched) + p["n_articulation"]
    assert sum("INTERRUPT" in t for t in artic) == p["n_articulation"]


def test_load_transit_texts_shuffled_reads_shuffled_geometry(corpus):
    out, manifest = corpus
    p = manifest["params"]
    texts = tc.load_transit_texts(out, "shuffled")
    assert len(texts) == p["n_walks"] + p["n_pairs"]
    # It genuinely reads the shuffled_* files, not the matched ones:
    assert (out / "shuffled_walks.jsonl").exists()
    assert texts != tc.load_transit_texts(out, "matched")


def test_load_transit_texts_deterministic_order(corpus):
    out, _ = corpus
    assert tc.load_transit_texts(out, "matched") == \
        tc.load_transit_texts(out, "matched")


def test_load_transit_texts_unknown_arm(corpus):
    out, _ = corpus
    with pytest.raises(ValueError, match="unknown transit arm"):
        tc.load_transit_texts(out, "bogus")


def test_load_transit_texts_missing_file(tmp_path):
    # A matched-only corpus has no shuffled_* files: the shuffled arm must fail
    # loudly rather than silently yield nothing.
    td.build_corpus(tmp_path / "m", n_walks=3, n_pairs=2, walk_len=4,
                    box_radius=2, seed=SEED, arms=("matched",))
    with pytest.raises(FileNotFoundError, match="shuffled_walks"):
        tc.load_transit_texts(tmp_path / "m", "shuffled")


def test_transit_corpus_dataset_tokenizes(corpus):
    out, manifest = corpus
    ds = tc.TransitCorpusDataset(out, "matched", _FakeTokenizer(), max_len=32)
    p = manifest["params"]
    assert len(ds) == p["n_walks"] + p["n_pairs"]
    ex = ds[0]
    assert ex["input_ids"].shape == (32,)
    assert ex["attention_mask"].shape == (32,)
    assert torch.equal(ex["labels"], ex["input_ids"])     # labels == input_ids
    assert int(ex["attention_mask"].sum()) > 0            # real (non-pad) tokens


# ── B. dataset-source exclusivity + parser choices ──────────────────


def test_resolve_dataset_source_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        tc.resolve_dataset_source("some/corpus", "alpaca")


def test_resolve_dataset_source_transit_only():
    assert tc.resolve_dataset_source("some/corpus", None) == \
        ("transit", "some/corpus")


def test_resolve_dataset_source_default_is_alpaca():
    assert tc.resolve_dataset_source(None, None) == ("dataset", "alpaca")
    assert tc.resolve_dataset_source(None, "code") == ("dataset", "code")


def test_parser_bridge_mode_includes_shuffled_rd():
    parser = tc.build_parser()
    ns = parser.parse_args(["--bridge-mode", "shuffled_rd"])
    assert ns.bridge_mode == "shuffled_rd"


def test_parser_transit_args_parse():
    parser = tc.build_parser()
    ns = parser.parse_args(["--transit-corpus", "d",
                            "--transit-arm", "matched+articulation"])
    assert ns.transit_corpus == "d"
    assert ns.transit_arm == "matched+articulation"
    # dataset default is the None sentinel so exclusivity is detectable
    assert ns.dataset is None


def test_parser_dataset_and_transit_both_error(capsys):
    parser = tc.build_parser()
    # main()'s exclusivity check surfaces via parser.error -> SystemExit(2).
    # Reproduce the resolve step the way main() calls it.
    ns = parser.parse_args(["--transit-corpus", "d", "--dataset", "alpaca"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        tc.resolve_dataset_source(ns.transit_corpus, ns.dataset)


# ── C. shuffled_rd mask (topology) ──────────────────────────────────


def _offdiag(m):
    return m[~np.eye(m.shape[0], dtype=bool)]


def test_shuffled_rd_matches_rd_edge_count_and_multiset():
    rd = rd_adjacency_mask(6)
    sh = shuffled_rd_adjacency_mask(6, seed=42)
    assert sh.shape == (6, 6)
    np.testing.assert_allclose(np.diag(sh), np.ones(6))        # diag 1.0
    np.testing.assert_allclose(sh, sh.T)                       # symmetric
    assert (_offdiag(sh) > 0).sum() == (_offdiag(rd) > 0).sum()  # edge count
    assert sorted(_offdiag(sh)) == sorted(_offdiag(rd))         # weight multiset


def test_shuffled_rd_pattern_differs_and_is_a_conjugation():
    rd = rd_adjacency_mask(6)
    sh = shuffled_rd_adjacency_mask(6, seed=42)
    # Rejection of relation automorphisms => the returned mask is NOT the RD
    # mask (an automorphism would have left it unchanged and been rejected):
    assert not np.array_equal(sh, rd)
    # ...but it IS a relabeling (conjugation) of the RD mask, not an arbitrary
    # matrix — some channel permutation maps rd onto it.
    assert any(np.array_equal(rd[np.ix_(p, p)], sh)
               for p in permutations(range(6)))


def test_shuffled_rd_deterministic_and_seed_sensitive():
    a = shuffled_rd_adjacency_mask(6, seed=42)
    b = shuffled_rd_adjacency_mask(6, seed=42)
    np.testing.assert_array_equal(a, b)                        # deterministic
    distinct = {shuffled_rd_adjacency_mask(6, seed=s).tobytes()
                for s in range(12)}
    assert len(distinct) > 1                                   # seed matters


def test_shuffled_rd_requires_six_channels():
    with pytest.raises(ValueError, match="n_channels=6"):
        shuffled_rd_adjacency_mask(4, seed=42)


def test_bridge_init_shuffled_rd_convention():
    sh = shuffled_rd_adjacency_mask(6, seed=42)
    init = bridge_init(6, "shuffled_rd", seed=42)
    expected = np.eye(6) + 0.1 * (sh - np.eye(6))
    np.testing.assert_allclose(init, expected)
    np.testing.assert_allclose(np.diag(init), np.ones(6))
    assert bridge_init(6, "shuffled_rd", seed=7).tobytes() != init.tobytes()


def test_bridge_init_still_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        bridge_init(6, "bogus")


# ── C(cont). RhombiLoRALinear consuming shuffled_rd ─────────────────


def test_rhombiloralinear_shuffled_rd_is_fixed_mask_bridge():
    m = RhombiLoRALinear(64, 128, rank=24, bridge_mode="shuffled_rd",
                         bridge_seed=42)
    assert m.shuffled_rd and m._fixed_mask_bridge and not m.rd_graph
    assert "rd_mask" in dict(m.named_buffers())
    assert "edge_weights" in dict(m.named_parameters())
    np.testing.assert_allclose(
        m.rd_mask.numpy(), shuffled_rd_adjacency_mask(6, seed=42))
    torch.testing.assert_close(m.effective_bridge, m.rd_mask * m.edge_weights)
    assert m.bridge_param is m.edge_weights


def test_rhombiloralinear_shuffled_rd_seed_threads():
    m42 = RhombiLoRALinear(64, 128, rank=24, bridge_mode="shuffled_rd",
                           bridge_seed=42)
    m7 = RhombiLoRALinear(64, 128, rank=24, bridge_mode="shuffled_rd",
                          bridge_seed=7)
    assert not np.allclose(m42.rd_mask.numpy(), m7.rd_mask.numpy())


def test_rhombiloralinear_shuffled_rd_gradient_isolates_mask():
    m = RhombiLoRALinear(64, 128, rank=24, bridge_mode="shuffled_rd",
                         bridge_seed=42)
    torch.nn.init.normal_(m.lora_B, std=0.01)
    m(torch.randn(4, 64)).sum().backward()
    assert m.edge_weights.grad is not None            # weights learn
    assert not m.rd_mask.requires_grad                # mask is fixed


def test_rd_graph_and_identity_paths_unchanged():
    # Campaign-safety guard: the shared-predicate refactor must not perturb the
    # existing modes the live bank depends on.
    g = RhombiLoRALinear(64, 128, rank=24, bridge_mode="rd_graph")
    assert g.rd_graph and not g.shuffled_rd
    torch.testing.assert_close(g.effective_bridge, g.rd_mask * g.edge_weights)
    i = RhombiLoRALinear(64, 128, rank=24, bridge_mode="identity")
    assert not i._fixed_mask_bridge
    torch.testing.assert_close(i.bridge, torch.eye(6))


# ── D. trainer capability flags ─────────────────────────────────────


def test_trainer_supports_transit_and_shuffled():
    assert br.trainer_supports("transit_corpus")
    assert br.trainer_supports("shuffled_rd")
