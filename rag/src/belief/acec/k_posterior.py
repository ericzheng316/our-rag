"""
K posterior — kappa_t(k) = P(K* = k | q, history), design doc Section 1.1/1.3.

K* is the unknown number of evidence pieces required to answer the question
(HotpotQA: typically 2; MuSiQue: up to 4). kappa_0 is initialized from an
amortized predictor p_psi(K|q); a DECOMPOSE that spawns slot K_t+1 truncates
and renormalizes kappa (the question needs at least as many slots as have
already been identified).
"""

from __future__ import annotations

from typing import List, Protocol

import numpy as np


class KPredictor(Protocol):
    def predict(self, question: str) -> List[float]: ...


class UniformKPredictor:
    """Maximum-entropy default prior before any offline calibration exists."""

    def __init__(self, k_max: int = 4):
        self.k_max = k_max

    def predict(self, question: str) -> List[float]:
        return [1.0 / self.k_max] * self.k_max


class SplitCountKPredictor:
    """
    Cheap, no-training default: peaks kappa_0 around the number of sub-queries
    the split server already produced for this question, decaying geometrically
    outward. Meant as a pragmatic stand-in for the amortized MLP p_psi(K|q) of
    Section 2.2 until offline_fit.EmpiricalKPredictor has been fitted on
    logged trajectories.
    """

    def __init__(self, k_max: int = 4, decay: float = 0.5):
        self.k_max = k_max
        self.decay = decay
        self._hint = 2

    def set_hint(self, num_sub_queries: int) -> None:
        self._hint = max(1, num_sub_queries)

    def predict(self, question: str) -> List[float]:
        hint = min(self._hint, self.k_max)
        weights = [self.decay ** abs((k + 1) - hint) for k in range(self.k_max)]
        total = sum(weights)
        return [w / total for w in weights]


class KPosterior:
    """Categorical posterior over {1, ..., k_max}, 1-indexed in the math,
    stored 0-indexed internally (kappa[k-1] == kappa_t(k))."""

    def __init__(self, k_max: int, prior: List[float]):
        if len(prior) != k_max:
            raise ValueError(f"prior has length {len(prior)}, expected k_max={k_max}")
        self.k_max = k_max
        self.kappa = np.array(prior, dtype=float)
        total = self.kappa.sum()
        self.kappa = self.kappa / total if total > 0 else np.full(k_max, 1.0 / k_max)

    def truncate_ge(self, k_min: int) -> None:
        """kappa(k) propto kappa(k) * 1[k >= k_min] — a DECOMPOSE spawning the
        k_min-th slot rules out K < k_min."""
        idx = np.arange(1, self.k_max + 1)
        mask = (idx >= k_min).astype(float)
        self.kappa = self.kappa * mask
        total = self.kappa.sum()
        if total > 1e-12:
            self.kappa = self.kappa / total
        else:
            self.kappa[:] = 0.0
            self.kappa[-1] = 1.0

    def expectation(self) -> float:
        idx = np.arange(1, self.k_max + 1)
        return float((self.kappa * idx).sum())

    def entropy(self) -> float:
        p = self.kappa[self.kappa > 1e-12]
        if p.size == 0:
            return 0.0
        return float(-(p * np.log(p)).sum())

    def weight_slot_le_k(self, j: int) -> float:
        """omega_j = E_kappa[(1/K) * 1[j <= K]] — Section 2.1's VOI stopping rule."""
        idx = np.arange(1, self.k_max + 1)
        w = np.where(idx >= j, self.kappa / idx, 0.0)
        return float(w.sum())
