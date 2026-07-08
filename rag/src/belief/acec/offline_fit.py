"""
Offline calibration of the ACEC observation model — design doc Section 2.2
and Section 9 (Week 1, entirely on logged data, no new GPU inference).

Gold hit labels come from `supporting_facts`-based distant supervision
(reusing the existing SF-title-matching machinery); this module only consumes
the resulting per-(slot, turn) table — it does not depend on any particular
records.jsonl schema, so it stays decoupled from the R3-RAG inference code.

At test time the computation is identical with these parameters frozen: the
only train/test asymmetry is that f1/f0, pi_a and the K predictor were
estimated on labeled trajectories (Section 2.2's closing argument).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .observation_model import ObservationModel


@dataclass
class HitExample:
    """One (slot, turn) observation with its gold hit label — the Week-1
    D1-2 deliverable: 'per-turn (slot, action-mode, NLI score, gold-hit) table'."""

    action_mode: str  # "EXPAND" | "REWRITE" | "DECOMPOSE"
    role: str  # "tgt" | "inc"
    bound: bool
    nli_score: float  # m_{j,t} in [0,1]
    is_hit: bool  # gold hit label from SF-title Hungarian matching


@dataclass
class KExample:
    question_embedding: np.ndarray
    k_true: int  # |supporting_facts| for this question


def auc_score(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Mann-Whitney-U based AUC (no sklearn dependency), with tie-averaged ranks."""
    labels_arr = np.asarray(labels, dtype=bool)
    scores_arr = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(labels_arr.sum()), int((~labels_arr).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores_arr)
    ranks = np.empty(len(scores_arr))
    ranks[order] = np.arange(1, len(scores_arr) + 1)
    sorted_scores = scores_arr[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1

    rank_sum_pos = ranks[labels_arr].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def fit_observation_model(examples: Sequence[HitExample]) -> Tuple[ObservationModel, Dict[str, float]]:
    """Fit f1/f0 per (role, bound) and empirical hit rates pi_a per action mode
    (Section 2.2, D3-4)."""
    model = ObservationModel()
    buckets: Dict[Tuple[str, bool], Tuple[List[float], List[float]]] = {}
    for ex in examples:
        key = (ex.role, ex.bound)
        hit_scores, miss_scores = buckets.setdefault(key, ([], []))
        (hit_scores if ex.is_hit else miss_scores).append(ex.nli_score)
    for (role, bound), (hit_scores, miss_scores) in buckets.items():
        model.fit(role, bound, hit_scores, miss_scores)

    by_mode: Dict[str, List[bool]] = {}
    for ex in examples:
        by_mode.setdefault(ex.action_mode, []).append(ex.is_hit)
    hit_rates = {mode: (sum(hits) / len(hits) if hits else 0.5) for mode, hits in by_mode.items()}

    return model, hit_rates


def load_calibrated_observation_model(path: str) -> Tuple[ObservationModel, Dict[str, float]]:
    """Reload the observation_model.json written by
    run_scripts/build_acec_calibration.py, so a live rollout uses the same
    frozen, offline-fitted parameters the Week-1 gate was checked against
    (design doc Section 2.2: "at test time the computation is identical with
    frozen parameters")."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    model = ObservationModel.from_dict(payload["observation_model"])
    hit_rates = payload["hit_rates"]
    return model, hit_rates


def hit_rates_to_beta_priors(
    hit_rates: Dict[str, float], ess: float = 10.0
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Convert calibrated empirical hit rates (pi_a per action mode) into
    ACECConfig's Beta pseudo-count priors (hit_prior_alpha0/beta0), at a
    fixed effective sample size. This is a scoped simplification, not a
    re-derivation of the "correct" ESS from the calibration data's actual
    per-mode sample counts — revisit if live results are sensitive to it."""
    alpha0 = {mode: rate * ess for mode, rate in hit_rates.items()}
    beta0 = {mode: (1.0 - rate) * ess for mode, rate in hit_rates.items()}
    return alpha0, beta0


def evaluate_hit_auc(examples: Sequence[HitExample]) -> float:
    """Per-slot hit AUC — go/no-go gate: proceed only if >= 0.80 on held-out
    dev (Section 9). Below that, the observation model can't support the
    belief and the design should stop here, cheaply."""
    labels = [ex.is_hit for ex in examples]
    scores = [ex.nli_score for ex in examples]
    return auc_score(labels, scores)


class EmpiricalKPredictor:
    """
    Amortized p_psi(K|q) via nearest-neighbor vote over frozen question
    embeddings paired with logged K* = |supporting_facts| labels. Serves the
    same role as the 2-layer MLP in Section 2.2 without requiring a
    gradient-based training loop.
    """

    def __init__(self, k_max: int, examples: Sequence[KExample], neighbors: int = 15):
        self.k_max = k_max
        self.neighbors = neighbors
        self._embs = np.stack([e.question_embedding for e in examples]) if examples else np.zeros((0, 1))
        self._ks = np.array([min(e.k_true, k_max) for e in examples], dtype=int)

    def predict_from_embedding(self, question_embedding: np.ndarray) -> List[float]:
        if len(self._ks) == 0:
            return [1.0 / self.k_max] * self.k_max
        norms = np.linalg.norm(self._embs, axis=1) * (np.linalg.norm(question_embedding) + 1e-9) + 1e-9
        sims = (self._embs @ question_embedding) / norms
        top = np.argsort(-sims)[: self.neighbors]
        counts = np.zeros(self.k_max)
        for idx in top:
            counts[max(int(self._ks[idx]) - 1, 0)] += 1
        counts += 0.5  # Laplace smoothing
        return list(counts / counts.sum())

    def evaluate_k_accuracy(self, examples: Sequence[KExample]) -> float:
        """K-accuracy gate — go/no-go threshold 75% on held-out dev (Section 9)."""
        if not examples:
            return float("nan")
        correct = 0
        for ex in examples:
            probs = self.predict_from_embedding(ex.question_embedding)
            pred_k = int(np.argmax(probs)) + 1
            if pred_k == min(ex.k_true, self.k_max):
                correct += 1
        return correct / len(examples)

    def as_k_predictor(self, embedder: Any) -> "_BoundKPredictor":
        """Wrap into the `KPredictor` protocol (`.predict(question: str)`)
        expected by `CoverageBelief`, given an embedder with `.encode(...)`."""
        return _BoundKPredictor(self, embedder)


@dataclass
class _BoundKPredictor:
    inner: EmpiricalKPredictor
    embedder: Any

    def predict(self, question: str) -> List[float]:
        emb = self.embedder.encode([question])[0]
        return self.inner.predict_from_embedding(emb)
