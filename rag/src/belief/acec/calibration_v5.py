"""ACEC v5 calibration for conditional marginal evidence utility.

V5 does not calibrate a HotpotQA title-hit label.  It calibrates the event
that the exact runtime-selected document adds previously missing evidence,
conditioned on both semantic support and document novelty.  The evidence
standard stored in the artifact defines how the utility delta was measured;
dataset-specific ids remain audit metadata only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, IO, Optional, Sequence, Tuple

import numpy as np

from .evidence_standard_v5 import standard_payload
from .k_strategy_v5 import KStrategyV5
from .offline_fit import auc_score


ARTIFACT_TYPE = "acec_calibration"
ARTIFACT_VERSION = 5
MODEL_KIND = "monotonic_platt_marginal_support_novelty_v1"


def strict_json_value(value: Any) -> Any:
    """Return a JSON-standard representation with no NaN/Infinity values.

    Python's ``json`` module emits non-standard ``NaN`` tokens by default.
    Calibration metrics legitimately contain undefined values when a split has
    one class or K is constant, so persist those values as JSON ``null`` rather
    than producing an artifact that strict consumers cannot parse.
    """

    if isinstance(value, dict):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [strict_json_value(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump_strict_json(value: Any, handle: IO[str], **kwargs: Any) -> None:
    """Serialize ``value`` as standards-compliant JSON."""

    json.dump(strict_json_value(value), handle, allow_nan=False, **kwargs)


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-standard numeric token in ACEC artifact: {token}")


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-6), 1.0 - 1e-6)


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value) - math.log1p(-value)


def _sigmoid(value: float) -> float:
    value = min(max(float(value), -60.0), 60.0)
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(frozen=True)
class MarginalUtilityExample:
    action_mode: str
    role: str
    bound: bool
    support_score: float
    novelty_score: float
    is_gain: bool
    marginal_utility: float


@dataclass
class MonotonicMarginalCalibrator:
    intercept: float
    support_slope: float
    novelty_slope: float
    base_rate: float
    n_examples: int

    @classmethod
    def fit(
        cls,
        examples: Sequence[MarginalUtilityExample],
        l2: float = 1e-2,
        max_iterations: int = 100,
    ) -> "MonotonicMarginalCalibrator":
        if not examples:
            return cls(0.0, 0.0, 0.0, 0.5, 0)
        y = np.asarray([float(example.is_gain) for example in examples], dtype=np.float64)
        base_rate = float((y.sum() + 1.0) / (len(y) + 2.0))
        if len(y) < 4 or np.all(y == y[0]):
            return cls(_logit(base_rate), 0.0, 0.0, base_rate, len(y))

        support = np.asarray(
            [_logit(example.support_score) for example in examples], dtype=np.float64
        )
        novelty = np.asarray(
            [_logit(example.novelty_score) for example in examples], dtype=np.float64
        )
        design = np.column_stack(
            (np.ones_like(support), np.clip(support, -12.0, 12.0), np.clip(novelty, -12.0, 12.0))
        )
        beta = np.asarray([_logit(base_rate), 1.0, 1.0], dtype=np.float64)
        for _ in range(max_iterations):
            eta = np.clip(design @ beta, -30.0, 30.0)
            probabilities = 1.0 / (1.0 + np.exp(-eta))
            weights = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
            gradient = design.T @ (y - probabilities)
            gradient[1:] -= l2 * beta[1:]
            information = design.T @ (weights[:, None] * design)
            information[1, 1] += l2
            information[2, 2] += l2
            information += np.eye(3) * 1e-8
            try:
                step = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                break
            candidate = beta + np.clip(step, -5.0, 5.0)
            candidate[1:] = np.maximum(candidate[1:], 0.0)
            if float(np.max(np.abs(candidate - beta))) < 1e-8:
                beta = candidate
                break
            beta = candidate
        if not np.all(np.isfinite(beta)):
            return cls(_logit(base_rate), 0.0, 0.0, base_rate, len(y))
        return cls(
            intercept=float(beta[0]),
            support_slope=float(max(beta[1], 0.0)),
            novelty_slope=float(max(beta[2], 0.0)),
            base_rate=base_rate,
            n_examples=len(y),
        )

    def predict(self, support_score: float, novelty_score: float) -> float:
        return _sigmoid(
            self.intercept
            + self.support_slope * _logit(support_score)
            + self.novelty_slope * _logit(novelty_score)
        )

    def log_likelihood_ratio(self, support_score: float, novelty_score: float) -> float:
        return _logit(self.predict(support_score, novelty_score)) - _logit(self.base_rate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intercept": self.intercept,
            "support_slope": self.support_slope,
            "novelty_slope": self.novelty_slope,
            "base_rate": self.base_rate,
            "n_examples": self.n_examples,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MonotonicMarginalCalibrator":
        return cls(
            intercept=float(payload["intercept"]),
            support_slope=max(float(payload["support_slope"]), 0.0),
            novelty_slope=max(float(payload["novelty_slope"]), 0.0),
            base_rate=_clip_probability(payload["base_rate"]),
            n_examples=int(payload["n_examples"]),
        )


class MarginalUtilityObservationModel:
    def __init__(self, calibrators: Dict[str, MonotonicMarginalCalibrator]):
        if "global" not in calibrators:
            raise ValueError("v5 marginal observation model requires a global calibrator")
        self.calibrators = calibrators

    @classmethod
    def fit(
        cls,
        examples: Sequence[MarginalUtilityExample],
        min_role_examples: int = 20,
    ) -> "MarginalUtilityObservationModel":
        calibrators = {"global": MonotonicMarginalCalibrator.fit(examples)}
        for role in ("tgt", "inc"):
            selected = [example for example in examples if example.role == role]
            labels = {example.is_gain for example in selected}
            calibrators[role] = (
                MonotonicMarginalCalibrator.fit(selected)
                if len(selected) >= min_role_examples and len(labels) == 2
                else calibrators["global"]
            )
        return cls(calibrators)

    def _calibrator(self, role: str) -> MonotonicMarginalCalibrator:
        return self.calibrators.get(role, self.calibrators["global"])

    def gain_posterior(
        self,
        support_score: float,
        novelty_score: float,
        pi_a: float,
        role: str,
        bound: bool,
    ) -> float:
        del bound
        calibrator = self._calibrator(role)
        if role == "tgt":
            return _sigmoid(
                calibrator.log_likelihood_ratio(support_score, novelty_score)
                + _logit(pi_a)
            )
        return calibrator.predict(support_score, novelty_score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": MODEL_KIND,
            "features": ["nli_support", "document_novelty"],
            "calibrators": {
                role: calibrator.to_dict() for role, calibrator in self.calibrators.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MarginalUtilityObservationModel":
        if payload.get("kind") != MODEL_KIND:
            raise ValueError(f"unsupported v5 observation model kind: {payload.get('kind')}")
        return cls(
            {
                role: MonotonicMarginalCalibrator.from_dict(value)
                for role, value in payload["calibrators"].items()
            }
        )


@dataclass
class CalibrationArtifactV5:
    observation_model: MarginalUtilityObservationModel
    gain_rates: Dict[str, float]
    k_strategy: KStrategyV5
    binding_threshold: float
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]


def fit_observation_model_v5(
    examples: Sequence[MarginalUtilityExample],
    action_modes: Sequence[str] = ("EXPAND", "REWRITE", "DECOMPOSE"),
    action_prior_strength: float = 5.0,
) -> Tuple[MarginalUtilityObservationModel, Dict[str, float], Dict[str, Dict[str, Any]]]:
    if not examples:
        raise ValueError("cannot fit ACEC v5 without marginal utility examples")
    model = MarginalUtilityObservationModel.fit(examples)
    target_examples = [example for example in examples if example.role == "tgt"]
    pooled_gains = sum(int(example.is_gain) for example in target_examples)
    pooled_rate = (pooled_gains + 1.0) / (len(target_examples) + 2.0)
    gain_rates: Dict[str, float] = {}
    counts: Dict[str, Dict[str, Any]] = {}
    for mode in action_modes:
        selected = [example for example in target_examples if example.action_mode == mode]
        gains = sum(int(example.is_gain) for example in selected)
        rate = (gains + action_prior_strength * pooled_rate) / (
            len(selected) + action_prior_strength
        )
        gain_rates[mode] = float(rate)
        counts[mode] = {
            "target_examples": len(selected),
            "target_gains": gains,
            "hierarchical_rate": float(rate),
        }
    counts["_pooled_target"] = {
        "target_examples": len(target_examples),
        "target_gains": pooled_gains,
        "hierarchical_rate": float(pooled_rate),
    }
    return model, gain_rates, counts


def predict_examples_v5(
    examples: Sequence[MarginalUtilityExample],
    model: MarginalUtilityObservationModel,
    gain_rates: Dict[str, float],
) -> np.ndarray:
    fallback = gain_rates.get("DECOMPOSE", 0.5)
    return np.asarray(
        [
            model.gain_posterior(
                example.support_score,
                example.novelty_score,
                gain_rates.get(example.action_mode, fallback),
                example.role,
                example.bound,
            )
            for example in examples
        ],
        dtype=np.float64,
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def operating_point_metrics(
    examples: Sequence[MarginalUtilityExample],
    probabilities: Sequence[float],
    threshold: float,
) -> Dict[str, Any]:
    labels = np.asarray([example.is_gain for example in examples], dtype=bool)
    predictions = np.asarray(probabilities, dtype=float) >= threshold
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    tn = int(np.sum(~predictions & ~labels))
    return {
        "threshold": float(threshold),
        "predicted_gain": int(predictions.sum()),
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def choose_binding_threshold_v5(
    examples: Sequence[MarginalUtilityExample],
    probabilities: Sequence[float],
    min_precision: float = 0.80,
    min_predicted_gain: int = 5,
) -> Tuple[Optional[float], Dict[str, Any]]:
    if not examples:
        return None, {"reason": "no_validation_examples"}
    candidates = sorted({float(value) for value in probabilities})
    eligible = []
    for threshold in candidates:
        metrics = operating_point_metrics(examples, probabilities, threshold)
        if (
            metrics["predicted_gain"] >= min_predicted_gain
            and np.isfinite(metrics["precision"])
            and metrics["precision"] >= min_precision
        ):
            eligible.append(metrics)
    if not eligible:
        return None, {
            "reason": "no_threshold_meets_precision_and_support",
            "min_precision": min_precision,
            "min_predicted_gain": min_predicted_gain,
        }
    chosen = max(eligible, key=lambda item: (item["recall"], item["precision"], item["threshold"]))
    return float(chosen["threshold"]), chosen


def posterior_quality_metrics_v5(
    examples: Sequence[MarginalUtilityExample],
    model: MarginalUtilityObservationModel,
    gain_rates: Dict[str, float],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    if not examples:
        return {
            "n": 0,
            "gain_rate": float("nan"),
            "raw_support_auc": float("nan"),
            "novelty_only_auc": float("nan"),
            "support_novelty_auc": float("nan"),
            "posterior_auc": float("nan"),
            "posterior_auc_gain_over_novelty": float("nan"),
            "average_precision": float("nan"),
            "novelty_only_average_precision": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "nonrepeat": {
                "n": 0,
                "gain_rate": float("nan"),
                "raw_support_auc": float("nan"),
                "posterior_auc": float("nan"),
                "average_precision": float("nan"),
            },
        }
    labels = np.asarray([float(example.is_gain) for example in examples], dtype=np.float64)
    support = np.asarray([example.support_score for example in examples], dtype=np.float64)
    novelty = np.asarray([example.novelty_score for example in examples], dtype=np.float64)
    posterior = predict_examples_v5(examples, model, gain_rates)
    label_bools = [bool(value) for value in labels]
    novelty_auc = auc_score(label_bools, list(novelty))
    posterior_auc = auc_score(label_bools, list(posterior))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        lower, upper = edges[index], edges[index + 1]
        mask = (posterior >= lower) & (posterior < upper if index < 9 else posterior <= upper)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(posterior[mask].mean() - labels[mask].mean()))
    metrics: Dict[str, Any] = {
        "n": len(examples),
        "gain_rate": float(labels.mean()),
        "raw_support_auc": auc_score(label_bools, list(support)),
        "novelty_only_auc": novelty_auc,
        "support_novelty_auc": auc_score(
            label_bools, list(support * novelty)
        ),
        "posterior_auc": posterior_auc,
        "posterior_auc_gain_over_novelty": (
            float(posterior_auc - novelty_auc)
            if np.isfinite(posterior_auc) and np.isfinite(novelty_auc)
            else float("nan")
        ),
        "average_precision": _average_precision(labels, posterior),
        "novelty_only_average_precision": _average_precision(labels, novelty),
        "brier": float(np.mean((posterior - labels) ** 2)),
        "ece": float(ece),
    }
    # Exact repeats are often deterministic zero-gain cases.  Report the
    # semantic discrimination on non-repeated documents separately so a model
    # cannot pass the validity gate merely by learning the novelty shortcut.
    nonrepeat_mask = novelty > 1e-6
    nonrepeat_labels = labels[nonrepeat_mask]
    nonrepeat_support = support[nonrepeat_mask]
    nonrepeat_posterior = posterior[nonrepeat_mask]
    metrics["nonrepeat"] = {
        "n": int(nonrepeat_mask.sum()),
        "gain_rate": (
            float(nonrepeat_labels.mean()) if len(nonrepeat_labels) else float("nan")
        ),
        "raw_support_auc": auc_score(
            [bool(value) for value in nonrepeat_labels], list(nonrepeat_support)
        ),
        "posterior_auc": auc_score(
            [bool(value) for value in nonrepeat_labels], list(nonrepeat_posterior)
        ),
        "average_precision": _average_precision(nonrepeat_labels, nonrepeat_posterior),
    }
    if threshold is not None:
        metrics["operating_point"] = operating_point_metrics(examples, posterior, threshold)
    return metrics


def save_calibration_artifact_v5(
    path: str,
    observation_model: MarginalUtilityObservationModel,
    gain_rates: Dict[str, float],
    k_strategy: KStrategyV5,
    binding_threshold: float,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "supervision_standard": standard_payload(),
        "observation_model": observation_model.to_dict(),
        "gain_rates": gain_rates,
        "k_strategy": k_strategy.to_dict(),
        "binding_threshold": float(binding_threshold),
        "metadata": metadata,
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as handle:
        dump_strict_json(payload, handle, indent=2)


def load_calibration_artifact_v5(path: str) -> CalibrationArtifactV5:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_nonfinite_json)
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"not an ACEC calibration artifact: {path}")
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported ACEC artifact version {payload.get('artifact_version')}; expected 5"
        )
    expected_standard = standard_payload()
    actual_standard = payload.get("supervision_standard") or {}
    if (
        actual_standard.get("name") != expected_standard["name"]
        or actual_standard.get("version") != expected_standard["version"]
        or actual_standard.get("target") != expected_standard["target"]
    ):
        raise ValueError("unsupported or missing ACEC v5 marginal-utility standard")
    required = (
        "observation_model",
        "gain_rates",
        "k_strategy",
        "binding_threshold",
        "metadata",
        "metrics",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"v5 calibration artifact missing fields: {missing}")
    return CalibrationArtifactV5(
        observation_model=MarginalUtilityObservationModel.from_dict(payload["observation_model"]),
        gain_rates={key: float(value) for key, value in payload["gain_rates"].items()},
        k_strategy=KStrategyV5.from_dict(payload["k_strategy"]),
        binding_threshold=float(payload["binding_threshold"]),
        metadata=dict(payload["metadata"]),
        metrics=dict(payload["metrics"]),
    )


__all__ = [
    "ARTIFACT_TYPE",
    "ARTIFACT_VERSION",
    "MODEL_KIND",
    "CalibrationArtifactV5",
    "MarginalUtilityExample",
    "MarginalUtilityObservationModel",
    "MonotonicMarginalCalibrator",
    "choose_binding_threshold_v5",
    "dump_strict_json",
    "fit_observation_model_v5",
    "load_calibration_artifact_v5",
    "operating_point_metrics",
    "posterior_quality_metrics_v5",
    "predict_examples_v5",
    "save_calibration_artifact_v5",
    "strict_json_value",
]
