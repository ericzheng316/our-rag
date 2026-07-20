"""Conservative, dataset-agnostic K strategy selection for ACEC v5.

``K`` is the number of minimal evidence requirements supplied by the v5
evidence adapter, not the number of titles or hops hard-coded by one dataset.
Auto mode only serializes a learned predictor when it beats a modal fixed-K
baseline on question-disjoint validation data by a configured margin.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .calibration_v2 import FixedKPredictor, SerializedKNNKPredictor
from .k_posterior import KPredictor, UniformKPredictor
from .offline_fit import EmpiricalKPredictor, KExample


@dataclass(frozen=True)
class KStrategyV5:
    mode: str
    k_max: int
    fixed_k: Optional[int]
    predictor_payload: Optional[Dict[str, Any]]
    selection_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "k_max": self.k_max,
            "fixed_k": self.fixed_k,
            "predictor": self.predictor_payload,
            "selection_metrics": self.selection_metrics,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "KStrategyV5":
        mode = str(payload.get("mode"))
        if mode not in {"fixed", "uniform", "predictor"}:
            raise ValueError(f"unsupported v5 K mode: {mode}")
        return cls(
            mode=mode,
            k_max=int(payload["k_max"]),
            fixed_k=(int(payload["fixed_k"]) if payload.get("fixed_k") is not None else None),
            predictor_payload=payload.get("predictor"),
            selection_metrics=dict(payload.get("selection_metrics") or {}),
        )


def _clipped_k(example: KExample, k_max: int) -> int:
    return min(max(int(example.k_true), 1), k_max)


def _accuracy_fixed(examples: Sequence[KExample], fixed_k: int, k_max: int) -> float:
    if not examples:
        return float("nan")
    return float(np.mean([_clipped_k(example, k_max) == fixed_k for example in examples]))


def _predictor_payload(
    examples: Sequence[KExample], k_max: int, neighbors: int
) -> Dict[str, Any]:
    return {
        "kind": "knn_e5",
        "k_max": int(k_max),
        "neighbors": int(neighbors),
        "embeddings": [
            np.asarray(example.question_embedding, dtype=np.float32).tolist()
            for example in examples
        ],
        "labels": [_clipped_k(example, k_max) for example in examples],
    }


def select_k_strategy_v5(
    requested_mode: str,
    fit_examples: Sequence[KExample],
    validation_examples: Sequence[KExample],
    k_max: int,
    fixed_k: int = 2,
    neighbors: int = 15,
    min_predictor_examples: int = 100,
    min_examples_per_k: int = 10,
    min_accuracy_gain: float = 0.03,
) -> KStrategyV5:
    if requested_mode not in {"auto", "fixed", "uniform", "predictor"}:
        raise ValueError(f"unknown v5 K mode: {requested_mode}")
    if not 1 <= fixed_k <= k_max:
        raise ValueError("fixed_k must be within [1, k_max]")

    fit_labels = [_clipped_k(example, k_max) for example in fit_examples]
    counts = Counter(fit_labels)
    modal_k = counts.most_common(1)[0][0] if counts else fixed_k
    baseline_k = fixed_k if requested_mode == "fixed" else modal_k
    baseline_accuracy = _accuracy_fixed(validation_examples, baseline_k, k_max)
    metrics: Dict[str, Any] = {
        "fit_examples": len(fit_examples),
        "validation_examples": len(validation_examples),
        "fit_k_counts": {str(key): value for key, value in sorted(counts.items())},
        "modal_fixed_k": modal_k,
        "fixed_validation_accuracy": baseline_accuracy,
        "predictor_validation_accuracy": None,
        "predictor_accuracy_gain": None,
        "selection_reason": None,
    }

    if requested_mode == "uniform":
        metrics["selection_reason"] = "explicit_uniform"
        return KStrategyV5("uniform", k_max, None, None, metrics)
    if requested_mode == "fixed":
        metrics["selection_reason"] = "explicit_fixed"
        return KStrategyV5("fixed", k_max, fixed_k, None, metrics)

    diverse = len(counts) >= 2
    enough_examples = len(fit_examples) >= min_predictor_examples
    enough_per_class = bool(counts) and min(counts.values()) >= min_examples_per_k
    can_fit_predictor = diverse and bool(fit_examples)
    predictor = (
        EmpiricalKPredictor(k_max, fit_examples, neighbors=neighbors)
        if can_fit_predictor
        else None
    )
    predictor_accuracy = (
        predictor.evaluate_k_accuracy(validation_examples)
        if predictor is not None and validation_examples
        else float("nan")
    )
    accuracy_gain = (
        predictor_accuracy - baseline_accuracy
        if np.isfinite(predictor_accuracy) and np.isfinite(baseline_accuracy)
        else float("nan")
    )
    metrics["predictor_validation_accuracy"] = predictor_accuracy
    metrics["predictor_accuracy_gain"] = accuracy_gain

    if requested_mode == "predictor":
        if predictor is None:
            raise ValueError("predictor K mode requires at least two observed K values")
        metrics["selection_reason"] = "explicit_predictor"
        return KStrategyV5(
            "predictor",
            k_max,
            None,
            _predictor_payload(fit_examples, k_max, neighbors),
            metrics,
        )

    use_predictor = bool(
        predictor is not None
        and enough_examples
        and enough_per_class
        and np.isfinite(accuracy_gain)
        and accuracy_gain >= min_accuracy_gain
    )
    if use_predictor:
        metrics["selection_reason"] = "validated_predictor_gain"
        return KStrategyV5(
            "predictor",
            k_max,
            None,
            _predictor_payload(fit_examples, k_max, neighbors),
            metrics,
        )

    failed = []
    if not diverse:
        failed.append("constant_k")
    if not enough_examples:
        failed.append("too_few_examples")
    if not enough_per_class:
        failed.append("sparse_k_class")
    if not np.isfinite(accuracy_gain) or accuracy_gain < min_accuracy_gain:
        failed.append("no_validated_accuracy_gain")
    metrics["selection_reason"] = "fixed_fallback:" + ",".join(failed)
    return KStrategyV5("fixed", k_max, modal_k, None, metrics)


def build_k_predictor_v5(
    strategy: KStrategyV5,
    embedder: Any,
) -> KPredictor:
    if strategy.mode == "fixed":
        if strategy.fixed_k is None:
            raise ValueError("fixed v5 K strategy is missing fixed_k")
        return FixedKPredictor(strategy.k_max, strategy.fixed_k)
    if strategy.mode == "uniform":
        return UniformKPredictor(strategy.k_max)
    payload = strategy.predictor_payload
    if not payload or payload.get("kind") != "knn_e5":
        raise ValueError("predictor v5 K strategy is missing a knn_e5 payload")
    if int(payload.get("k_max", -1)) != strategy.k_max:
        raise ValueError("serialized v5 K predictor k_max mismatch")
    return SerializedKNNKPredictor(
        k_max=strategy.k_max,
        embeddings=payload["embeddings"],
        labels=payload["labels"],
        neighbors=int(payload["neighbors"]),
        embedder=embedder,
    )


__all__ = [
    "KStrategyV5",
    "build_k_predictor_v5",
    "select_k_strategy_v5",
]
