"""
E5QueryEmbedder — the missing 'query: ' prefix on E5 embeddings.

E5 models (intfloat/e5-base-v2 and family) require every input to be
prefixed with 'query: ' or 'passage: ' to produce well-calibrated
embeddings (per the model card); without it, cosine similarity between
essentially any two unrelated English sentences collapses into a narrow,
artificially-high range. rag/src/belief/obs_extractor.py's E5Embedder
(legacy, kept for ablation per CLAUDE.md) never adds this prefix, and ACEC's
own Embedder protocol is intentionally E5-agnostic (action_labeler.py,
offline_fit.py's EmpiricalKPredictor, build_acec_calibration.py's
match_slots_to_sf all just call .encode(texts)) — this is where the
E5-specific convention actually lives, so a future non-E5 embedder swap
doesn't require touching any call-site code.

Diagnosed from the Week-1 pilot: with the un-prefixed embedder, every one of
500 replayed records spawned *exactly* one slot (avg slots spawned = 1.00,
zero variance) despite K_true = 2 always — ActionLabeler's tau_new routing
almost never saw a query dissimilar enough from the existing slot to trigger
DECOMPOSE, because unprefixed E5 cosine similarities were uninformative.

ACEC only ever uses E5 in a symmetric similarity role (query vs query, query
vs slot hypothesis, query vs SF title) — never asymmetric query-vs-passage
retrieval — so a uniform 'query: ' prefix on every input is the correct
choice (E5's own usage examples do this for symmetric STS-style tasks).
"""

from __future__ import annotations

from typing import List, Protocol

import numpy as np


class _RawEncoder(Protocol):
    def encode(self, texts: List[str], **kwargs) -> np.ndarray: ...


class E5QueryEmbedder:
    """Wraps any raw encoder (e.g. obs_extractor.E5Embedder) and prefixes
    every input with 'query: ' before delegating."""

    def __init__(self, raw_encoder: _RawEncoder):
        self._raw = raw_encoder

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        return self._raw.encode([f"query: {t}" for t in texts], **kwargs)
