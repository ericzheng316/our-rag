"""Versioned ACEC calibration artifact and explicit K-posterior modes.

The legacy ``offline_fit.py`` and ``build_acec_calibration.py`` remain
unchanged for reproducibility.  New training runs should use this v2 schema,
which validates the artifact before constructing a runtime belief state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .k_posterior import KPredictor, UniformKPredictor
from .observation_model import ObservationModel
from .offline_fit import HitExample, KExample, auc_score


ARTIFACT_TYPE = "acec_calibration"
ARTIFACT_VERSION = 2


@dataclass
class CalibrationArtifactV2:
    observation_model: ObservationModel
    hit_rates: Dict[str, float]
    k_predictor_payload: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]


class FixedKPredictor:
    """One-hot K posterior for datasets with a fixed evidence count."""

    def __init__(self, k_max: int, fixed_k: int):
        if not 1 <= fixed_k <= k_max:
            raise ValueError(f"fixed_k must be in [1, {k_max}], got {fixed_k}")
        self.k_max = k_max
        self.fixed_k = fixed_k

    def predict(self, question: str) -> List[float]:
        probs = [0.0] * self.k_max
        probs[self.fixed_k - 1] = 1.0
        return probs


class SerializedKNNKPredictor:
    """Runtime form of the fitted nearest-neighbor K predictor."""

    def __init__(
        self,
        k_max: int,
        embeddings: Sequence[Sequence[float]],
        labels: Sequence[int],
        neighbors: int,
        embedder: Any,
    ):
        self.k_max = k_max
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.neighbors = int(neighbors)
        self.embedder = embedder
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.labels):
            raise ValueError("invalid serialized K predictor shapes")
        if len(self.labels) == 0:
            raise ValueError("serialized K predictor contains no fit examples")

    def predict(self, question: str) -> List[float]:
        q = np.asarray(self.embedder.encode([question])[0], dtype=np.float32)
        denom = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9
        similarities = (self.embeddings @ q) / denom
        top = np.argsort(-similarities)[: self.neighbors]
        counts = np.full(self.k_max, 0.5, dtype=np.float64)
        for idx in top:
            k = min(max(int(self.labels[idx]), 1), self.k_max)
            counts[k - 1] += 1.0
        return list(counts / counts.sum())


def build_k_predictor(
    mode: str,
    k_max: int,
    embedder: Any,
    fixed_k: int = 2,
    artifact: Optional[CalibrationArtifactV2] = None,
) -> KPredictor:
    if mode == "fixed":
        return FixedKPredictor(k_max, fixed_k)
    if mode == "uniform":
        return UniformKPredictor(k_max)
    if mode != "predictor":
        raise ValueError(f"unknown K mode: {mode}")
    if artifact is None or artifact.k_predictor_payload is None:
        raise ValueError("K mode 'predictor' requires a v2 artifact with k_predictor data")
    payload = artifact.k_predictor_payload
    if payload.get("kind") != "knn_e5":
        raise ValueError(f"unsupported serialized K predictor kind: {payload.get('kind')}")
    if int(payload.get("k_max", -1)) != k_max:
        raise ValueError(
            f"K predictor k_max={payload.get('k_max')} does not match runtime k_max={k_max}"
        )
    return SerializedKNNKPredictor(
        k_max=k_max,
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        neighbors=payload["neighbors"],
        embedder=embedder,
    )


def fit_observation_model_v2(
    examples: Sequence[HitExample],
    action_modes: Sequence[str] = ("EXPAND", "REWRITE", "DECOMPOSE"),
) -> Tuple[ObservationModel, Dict[str, float], Dict[str, Dict[str, int]]]:
    """Fit role/bound densities and *target-only* action hit priors.

    Live ``CoverageBelief`` updates each action Beta with the targeted slot
    only.  Fitting priors from incidental slots as v1 did mixes two different
    events and makes the offline and online models inconsistent.
    """

    model = ObservationModel()
    buckets: Dict[Tuple[str, bool], Tuple[List[float], List[float]]] = {}
    for ex in examples:
        hit_scores, miss_scores = buckets.setdefault((ex.role, ex.bound), ([], []))
        (hit_scores if ex.is_hit else miss_scores).append(ex.nli_score)
    for (role, bound), (hit_scores, miss_scores) in buckets.items():
        model.fit(role, bound, hit_scores, miss_scores)

    hit_rates: Dict[str, float] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for mode in action_modes:
        targeted = [ex for ex in examples if ex.role == "tgt" and ex.action_mode == mode]
        n = len(targeted)
        hits = sum(int(ex.is_hit) for ex in targeted)
        # Beta(1,1) posterior mean avoids zero pseudo-counts at runtime.
        hit_rates[mode] = (hits + 1.0) / (n + 2.0)
        counts[mode] = {"target_examples": n, "target_hits": hits}
    return model, hit_rates, counts


def posterior_quality_metrics(
    examples: Sequence[HitExample],
    model: ObservationModel,
    hit_rates: Dict[str, float],
    ece_bins: int = 10,
) -> Dict[str, float]:
    if not examples:
        return {"n": 0, "raw_nli_auc": float("nan"), "posterior_auc": float("nan"),
                "brier": float("nan"), "ece": float("nan")}
    labels = np.asarray([float(ex.is_hit) for ex in examples], dtype=np.float64)
    raw_scores = [ex.nli_score for ex in examples]
    posteriors = np.asarray(
        [
            model.hit_posterior(
                ex.nli_score,
                hit_rates.get(ex.action_mode, 0.5),
                role=ex.role,
                bound=ex.bound,
            )
            for ex in examples
        ],
        dtype=np.float64,
    )
    brier = float(np.mean((posteriors - labels) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    for idx in range(ece_bins):
        lower, upper = edges[idx], edges[idx + 1]
        mask = (posteriors >= lower) & (posteriors < upper if idx < ece_bins - 1 else posteriors <= upper)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(posteriors[mask].mean() - labels[mask].mean()))
    return {
        "n": int(len(examples)),
        "raw_nli_auc": auc_score([bool(v) for v in labels], raw_scores),
        "posterior_auc": auc_score([bool(v) for v in labels], list(posteriors)),
        "brier": brier,
        "ece": float(ece),
    }


def k_predictor_payload(
    examples: Sequence[KExample],
    k_max: int,
    neighbors: int = 15,
) -> Dict[str, Any]:
    return {
        "kind": "knn_e5",
        "k_max": int(k_max),
        "neighbors": int(neighbors),
        "embeddings": [np.asarray(ex.question_embedding, dtype=np.float32).tolist() for ex in examples],
        "labels": [min(max(int(ex.k_true), 1), k_max) for ex in examples],
    }


def save_calibration_artifact_v2(
    path: str,
    observation_model: ObservationModel,
    hit_rates: Dict[str, float],
    k_payload: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "observation_model": observation_model.to_dict(),
        "hit_rates": hit_rates,
        "k_predictor": k_payload,
        "metadata": metadata,
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_calibration_artifact_v2(path: str) -> CalibrationArtifactV2:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"not an ACEC calibration artifact: {path}")
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported ACEC artifact version {payload.get('artifact_version')}; "
            f"expected {ARTIFACT_VERSION}"
        )
    required = ("observation_model", "hit_rates", "metadata", "metrics")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"v2 calibration artifact missing fields: {missing}")
    return CalibrationArtifactV2(
        observation_model=ObservationModel.from_dict(payload["observation_model"]),
        hit_rates={key: float(value) for key, value in payload["hit_rates"].items()},
        k_predictor_payload=payload.get("k_predictor"),
        metadata=payload["metadata"],
        metrics=payload["metrics"],
    )
