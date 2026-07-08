"""
NLI scorer implementations for ACEC's observation model (design doc Section
1.2). Kept in the core package (not duplicated in run_scripts scripts) since
`inference_new.py`'s live rollout and `build_acec_calibration.py`'s offline
replay both need the same scorer — a second copy is exactly the kind of
duplication that let the entailment-index bug go unnoticed once already this
session.
"""

from __future__ import annotations

from typing import Dict, Protocol, Tuple

import numpy as np


class _RawEncoder(Protocol):
    def encode(self, texts, **kwargs) -> np.ndarray: ...


class CosineNLIScorer:
    """E5-cosine stand-in for a calibrated NLI cross-encoder. Use only to
    sanity-check the replay pipeline — the real go/no-go gate needs a real
    cross-encoder (--nli_model), since cosine cannot separate topically-similar
    non-entailing docs from true hits (the exact distractor-mode failure mode
    the design doc's observation model is meant to fix)."""

    def __init__(self, embedder: _RawEncoder):
        self.embedder = embedder
        self._cache: Dict[Tuple[str, str], float] = {}

    def score(self, premise: str, hypothesis: str) -> float:
        key = (premise[:200], hypothesis)
        if key not in self._cache:
            embs = self.embedder.encode([premise[:200], hypothesis], show_progress_bar=False)
            sim = float(
                embs[0] @ embs[1] / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9)
            )
            self._cache[key] = (sim + 1.0) / 2.0  # cosine [-1,1] -> [0,1]
        return self._cache[key]


class CrossEncoderNLIScorer:
    """Real calibrated NLI observation model (design doc Section 1.2)."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        # sentence-transformers' own convention for its cross-encoder/nli-*
        # model family (see their CrossEncoder usage docs):
        #   label_mapping = ['contradiction', 'entailment', 'neutral']
        # i.e. entailment is index 1, NOT the last index — an earlier version
        # of this scorer used index -1 (neutral) instead of entailment, which
        # explained a per-slot hit AUC stuck *below* 0.5 (not just weak,
        # systematically wrong-direction) across several unrelated fixes.
        self._entail_idx = 1

    def score(self, premise: str, hypothesis: str) -> float:
        probs = self.model.predict([(premise, hypothesis)], apply_softmax=True)[0]
        return float(probs[self._entail_idx])
