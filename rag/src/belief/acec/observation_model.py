"""
Action-indexed observation model — design doc Section 1.2/1.3.

Per turn, a slot either receives a hit (a new doc resolves it) or not; the
NLI entailment score m_j is emitted from class-conditional densities f1/f0
that depend on the routing role (target vs incidental) and binding status.
This module implements those densities plus the closed-form Bayes update
g_a(m) that turns a raw NLI score into a hit posterior.

Densities are 1-D Beta distributions (design doc: "histogram or two-parameter
logistic" — a Beta on [0,1] is the two-parameter choice used here), fit by
method of moments on labeled scores via offline_fit.py (Section 2.2). Until
fit, weakly-informative defaults separate "hit" (skewed high) from "miss"
(skewed low).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _beta_pdf(x: float, a: float, b: float) -> float:
    x = min(max(x, 1e-6), 1.0 - 1e-6)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return math.exp(log_norm + (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x))


@dataclass
class BetaDensity:
    a: float
    b: float

    def pdf(self, x: float) -> float:
        return _beta_pdf(x, self.a, self.b)

    @classmethod
    def fit_moments(cls, samples: List[float]) -> "BetaDensity":
        """Method-of-moments fit; falls back to a weak default on too little data."""
        if len(samples) < 2:
            return cls(2.0, 2.0)
        arr = np.clip(np.array(samples, dtype=float), 1e-3, 1.0 - 1e-3)
        mean = float(arr.mean())
        var = float(arr.var())
        max_var = mean * (1.0 - mean) - 1e-4
        if max_var <= 1e-4:
            return cls(2.0, 2.0)
        var = min(var, max_var) if var > 0 else max_var / 2
        common = mean * (1.0 - mean) / var - 1.0
        a = max(mean * common, 0.5)
        b = max((1.0 - mean) * common, 0.5)
        return cls(a, b)

    def to_dict(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BetaDensity":
        return cls(d["a"], d["b"])


# Weakly-informative defaults used before offline_fit.py has calibrated
# anything: hits skew the NLI score high, misses skew it low.
_DEFAULT_HIT = BetaDensity(4.0, 1.5)
_DEFAULT_MISS = BetaDensity(1.5, 4.0)


class HitRateBeta:
    """Action-indexed Beta(alpha_a, beta_a) over hit rate pi_a — Section 1.3.

    `ess_cap` bounds the within-episode effective sample size so a single
    episode's evidence cannot dominate the offline-fitted prior (avoiding the
    posterior-feeding-itself circularity flagged in the design doc)."""

    def __init__(self, alpha0: float, beta0: float, ess_cap: Optional[float] = None):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alpha = alpha0
        self.beta = beta0
        self.ess_cap = ess_cap

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, g: float) -> None:
        self.alpha += g
        self.beta += 1.0 - g
        if self.ess_cap is not None:
            total = self.alpha + self.beta
            cap_total = self.ess_cap + self.alpha0 + self.beta0
            if total > cap_total:
                scale = cap_total / total
                self.alpha *= scale
                self.beta *= scale

    def reset(self) -> None:
        self.alpha, self.beta = self.alpha0, self.beta0


class ObservationModel:
    """Holds f1/f0 class-conditional densities keyed by (role, bound) and
    computes the per-turn hit posterior g_a(m) (Section 1.3, eq. boxed)."""

    def __init__(self) -> None:
        self._f1: Dict[Tuple[str, bool], BetaDensity] = {}
        self._f0: Dict[Tuple[str, bool], BetaDensity] = {}

    def f1(self, role: str, bound: bool) -> BetaDensity:
        return self._f1.get((role, bound), _DEFAULT_HIT)

    def f0(self, role: str, bound: bool) -> BetaDensity:
        return self._f0.get((role, bound), _DEFAULT_MISS)

    def fit(self, role: str, bound: bool, hit_scores: List[float], miss_scores: List[float]) -> None:
        self._f1[(role, bound)] = BetaDensity.fit_moments(hit_scores)
        self._f0[(role, bound)] = BetaDensity.fit_moments(miss_scores)

    def hit_posterior(self, m: float, pi_a: float, role: str, bound: bool) -> float:
        """
        g_a(m) = pi_a * f1(m) / (pi_a * f1(m) + (1 - pi_a) * f0(m))

        Bayes update of the per-turn hit probability given the observed NLI
        entailment score m and the current action's hit-rate estimate pi_a.
        """
        f1_val = self.f1(role, bound).pdf(m)
        f0_val = self.f0(role, bound).pdf(m)
        numerator = pi_a * f1_val
        denominator = numerator + (1.0 - pi_a) * f0_val
        if denominator < 1e-12:
            return pi_a
        return numerator / denominator

    def to_dict(self) -> Dict[str, dict]:
        return {
            "f1": {f"{role}|{bound}": d.to_dict() for (role, bound), d in self._f1.items()},
            "f0": {f"{role}|{bound}": d.to_dict() for (role, bound), d in self._f0.items()},
        }
