"""Strict ACEC calibration artifact with monotonic NLI posteriors.

This module intentionally lives beside, rather than replacing, the legacy and
v2 calibration paths.  Version 3 fixes three statistical failure modes found
by held-out replay:

* unconstrained Beta-density ratios could reverse a useful NLI score;
* a target-action prior was also applied to incidental slots;
* unseen action modes received an arbitrary 0.5 hit rate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration_v2 import FixedKPredictor, SerializedKNNKPredictor
from .k_posterior import KPredictor, UniformKPredictor
from .offline_fit import HitExample, KExample, auc_score


ARTIFACT_TYPE = "acec_calibration"
ARTIFACT_VERSION = 3
MODEL_KIND = "monotonic_platt_role_v1"


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-6), 1.0 - 1e-6)


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value) - math.log1p(-value)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(max(value, -60.0))
    return exp_pos / (1.0 + exp_pos)


@dataclass
class MonotonicPlattCalibrator:
    """Logistic calibration constrained to preserve NLI score ordering."""

    slope: float
    intercept: float
    base_rate: float
    n_examples: int

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        labels: Sequence[bool],
        l2: float = 1e-2,
        max_iterations: int = 100,
    ) -> "MonotonicPlattCalibrator":
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have equal length")
        if not scores:
            return cls(0.0, 0.0, 0.5, 0)

        y = np.asarray(labels, dtype=np.float64)
        base_rate = float((y.sum() + 1.0) / (len(y) + 2.0))
        if len(y) < 4 or np.all(y == y[0]):
            return cls(0.0, _logit(base_rate), base_rate, len(y))

        raw = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        x = np.clip(np.log(raw) - np.log1p(-raw), -12.0, 12.0)
        design = np.column_stack((np.ones_like(x), x))
        beta = np.asarray([_logit(base_rate), 1.0], dtype=np.float64)

        # Penalized IRLS.  The intercept is left unpenalized; the slope is
        # projected to >= 0 after every step so calibration cannot invert NLI.
        for _ in range(max_iterations):
            eta = np.clip(design @ beta, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-eta))
            weights = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
            gradient = design.T @ (y - probabilities)
            gradient[1] -= l2 * beta[1]
            information = design.T @ (weights[:, None] * design)
            information[1, 1] += l2
            information += np.eye(2) * 1e-8
            try:
                step = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                break
            step = np.clip(step, -5.0, 5.0)
            candidate = beta + step
            candidate[1] = max(float(candidate[1]), 0.0)
            if float(np.max(np.abs(candidate - beta))) < 1e-8:
                beta = candidate
                break
            beta = candidate

        if not np.all(np.isfinite(beta)):
            return cls(0.0, _logit(base_rate), base_rate, len(y))
        if beta[1] <= 1e-10:
            beta = np.asarray([_logit(base_rate), 0.0], dtype=np.float64)
        return cls(float(beta[1]), float(beta[0]), base_rate, len(y))

    def predict(self, score: float) -> float:
        return _sigmoid(self.intercept + self.slope * _logit(score))

    def log_likelihood_ratio(self, score: float) -> float:
        # Replace the offline role base rate with the live target-action prior.
        return _logit(self.predict(score)) - _logit(self.base_rate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "base_rate": self.base_rate,
            "n_examples": self.n_examples,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MonotonicPlattCalibrator":
        return cls(
            slope=max(float(payload["slope"]), 0.0),
            intercept=float(payload["intercept"]),
            base_rate=_clip_probability(payload["base_rate"]),
            n_examples=int(payload["n_examples"]),
        )


class MonotonicObservationModel:
    """Role-aware posterior model with no target/incidental prior leakage."""

    def __init__(self, calibrators: Dict[str, MonotonicPlattCalibrator]):
        if "global" not in calibrators:
            raise ValueError("monotonic observation model requires a global calibrator")
        self.calibrators = calibrators

    @classmethod
    def fit(
        cls,
        examples: Sequence[HitExample],
        min_role_examples: int = 20,
    ) -> "MonotonicObservationModel":
        global_calibrator = MonotonicPlattCalibrator.fit(
            [example.nli_score for example in examples],
            [example.is_hit for example in examples],
        )
        calibrators = {"global": global_calibrator}
        for role in ("tgt", "inc"):
            role_examples = [example for example in examples if example.role == role]
            labels = {bool(example.is_hit) for example in role_examples}
            if len(role_examples) >= min_role_examples and len(labels) == 2:
                calibrators[role] = MonotonicPlattCalibrator.fit(
                    [example.nli_score for example in role_examples],
                    [example.is_hit for example in role_examples],
                )
            else:
                calibrators[role] = global_calibrator
        return cls(calibrators)

    def _calibrator(self, role: str) -> MonotonicPlattCalibrator:
        return self.calibrators.get(role, self.calibrators["global"])

    def hit_posterior(self, m: float, pi_a: float, role: str, bound: bool) -> float:
        del bound  # Bound-specific buckets were too sparse for stable calibration.
        calibrator = self._calibrator(role)
        if role == "tgt":
            return _sigmoid(calibrator.log_likelihood_ratio(m) + _logit(pi_a))
        # pi_a describes success of the routed target action.  It is not the
        # prior probability that the same document incidentally covers slot j.
        return calibrator.predict(m)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": MODEL_KIND,
            "calibrators": {
                role: calibrator.to_dict()
                for role, calibrator in self.calibrators.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MonotonicObservationModel":
        if payload.get("kind") != MODEL_KIND:
            raise ValueError(f"unsupported v3 observation model kind: {payload.get('kind')}")
        return cls(
            {
                role: MonotonicPlattCalibrator.from_dict(calibrator)
                for role, calibrator in payload["calibrators"].items()
            }
        )


@dataclass
class CalibrationArtifactV3:
    observation_model: MonotonicObservationModel
    hit_rates: Dict[str, float]
    k_predictor_payload: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]


def build_k_predictor(
    mode: str,
    k_max: int,
    embedder: Any,
    fixed_k: int = 2,
    artifact: Optional[CalibrationArtifactV3] = None,
) -> KPredictor:
    if mode == "fixed":
        return FixedKPredictor(k_max, fixed_k)
    if mode == "uniform":
        return UniformKPredictor(k_max)
    if mode != "predictor":
        raise ValueError(f"unknown K mode: {mode}")
    if artifact is None or artifact.k_predictor_payload is None:
        raise ValueError("K mode 'predictor' requires a v3 artifact with non-constant K labels")
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


def fit_observation_model_v3(
    examples: Sequence[HitExample],
    action_modes: Sequence[str] = ("EXPAND", "REWRITE", "DECOMPOSE"),
    action_prior_strength: float = 5.0,
) -> Tuple[MonotonicObservationModel, Dict[str, float], Dict[str, Dict[str, Any]]]:
    if not examples:
        raise ValueError("cannot fit ACEC calibration without hit examples")
    model = MonotonicObservationModel.fit(examples)

    target_examples = [example for example in examples if example.role == "tgt"]
    pooled_hits = sum(int(example.is_hit) for example in target_examples)
    pooled_rate = (pooled_hits + 1.0) / (len(target_examples) + 2.0)
    hit_rates: Dict[str, float] = {}
    counts: Dict[str, Dict[str, Any]] = {}
    for mode in action_modes:
        selected = [example for example in target_examples if example.action_mode == mode]
        hits = sum(int(example.is_hit) for example in selected)
        rate = (hits + action_prior_strength * pooled_rate) / (
            len(selected) + action_prior_strength
        )
        hit_rates[mode] = float(rate)
        counts[mode] = {
            "target_examples": len(selected),
            "target_hits": hits,
            "hierarchical_rate": float(rate),
            "fallback": "pooled_target" if not selected else "action_plus_pooled_target",
        }
    counts["_pooled_target"] = {
        "target_examples": len(target_examples),
        "target_hits": pooled_hits,
        "hierarchical_rate": float(pooled_rate),
        "fallback": "beta_1_1",
    }
    return model, hit_rates, counts


def _quality_metrics(labels: np.ndarray, raw: np.ndarray, posterior: np.ndarray) -> Dict[str, float]:
    if len(labels) == 0:
        return {
            "n": 0,
            "raw_nli_auc": float("nan"),
            "posterior_auc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        lower, upper = edges[index], edges[index + 1]
        mask = (posterior >= lower) & (
            posterior < upper if index < 9 else posterior <= upper
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(posterior[mask].mean() - labels[mask].mean())
            )
    return {
        "n": int(len(labels)),
        "raw_nli_auc": auc_score([bool(value) for value in labels], list(raw)),
        "posterior_auc": auc_score([bool(value) for value in labels], list(posterior)),
        "brier": float(np.mean((posterior - labels) ** 2)),
        "ece": float(ece),
    }


def posterior_quality_metrics(
    examples: Sequence[HitExample],
    model: MonotonicObservationModel,
    hit_rates: Dict[str, float],
) -> Dict[str, Any]:
    labels = np.asarray([float(example.is_hit) for example in examples], dtype=np.float64)
    raw = np.asarray([example.nli_score for example in examples], dtype=np.float64)
    posterior = np.asarray(
        [
            model.hit_posterior(
                example.nli_score,
                hit_rates.get(example.action_mode, hit_rates.get("DECOMPOSE", 0.5)),
                role=example.role,
                bound=example.bound,
            )
            for example in examples
        ],
        dtype=np.float64,
    )
    metrics: Dict[str, Any] = _quality_metrics(labels, raw, posterior)
    metrics["by_role"] = {}
    for role in ("tgt", "inc"):
        indices = np.asarray([example.role == role for example in examples], dtype=bool)
        metrics["by_role"][role] = _quality_metrics(
            labels[indices], raw[indices], posterior[indices]
        )
    return metrics


def k_predictor_payload(
    examples: Sequence[KExample],
    k_max: int,
    neighbors: int = 15,
) -> Dict[str, Any]:
    return {
        "kind": "knn_e5",
        "k_max": int(k_max),
        "neighbors": int(neighbors),
        "embeddings": [
            np.asarray(example.question_embedding, dtype=np.float32).tolist()
            for example in examples
        ],
        "labels": [min(max(int(example.k_true), 1), k_max) for example in examples],
    }


def save_calibration_artifact_v3(
    path: str,
    observation_model: MonotonicObservationModel,
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


def load_calibration_artifact_v3(path: str) -> CalibrationArtifactV3:
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
        raise ValueError(f"v3 calibration artifact missing fields: {missing}")
    return CalibrationArtifactV3(
        observation_model=MonotonicObservationModel.from_dict(payload["observation_model"]),
        hit_rates={key: float(value) for key, value in payload["hit_rates"].items()},
        k_predictor_payload=payload.get("k_predictor"),
        metadata=payload["metadata"],
        metrics=payload["metrics"],
    )
