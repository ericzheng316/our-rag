"""Frozen signed answer-value artifact for ACEC v7.

The artifact is intentionally independent from the coverage belief.  It is
fitted on answer-supervised, question-disjoint examples whose target is a
teacher-forced gold-answer log-likelihood delta (or another signed downstream
answer utility).  Supporting-fact labels are rejected from feature payloads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .contracts_v7 import assert_sf_label_free, stable_id_hash


ANSWER_VALUE_SCHEMA_V7 = "acec_signed_answer_value"
ANSWER_VALUE_VERSION_V7 = 70
ANSWER_VALUE_CROSSFIT_SCHEMA_V7 = "acec_signed_answer_value_crossfit"


@dataclass(frozen=True)
class AnswerValueExampleV7:
    question_id: str
    features: Mapping[str, float]
    target_delta: float
    state_id: str = ""

    def __post_init__(self) -> None:
        if not self.question_id:
            raise ValueError("answer-value example requires a question id")
        if not self.features:
            raise ValueError("answer-value example requires features")
        assert_sf_label_free(self.features, path="answer_value.features")
        for name, value in self.features.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"answer-value feature {name!r} is not finite")
        if not math.isfinite(float(self.target_delta)):
            raise ValueError("answer-value target must be finite")


@dataclass(frozen=True)
class AnswerValueArtifactV7:
    feature_names: Tuple[str, ...]
    center: Tuple[float, ...]
    scale: Tuple[float, ...]
    weights: Tuple[float, ...]
    intercept: float
    fit_question_id_sha256: str
    validation_question_id_sha256: str
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ANSWER_VALUE_SCHEMA_V7
    version: int = ANSWER_VALUE_VERSION_V7

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if not size or len(set(self.feature_names)) != size:
            raise ValueError("answer-value feature names must be unique/non-empty")
        if not (
            len(self.center) == len(self.scale) == len(self.weights) == size
        ):
            raise ValueError("answer-value vector sizes do not match")
        if any(float(value) <= 0.0 for value in self.scale):
            raise ValueError("answer-value scales must be positive")
        if not all(
            math.isfinite(float(value))
            for value in (
                *self.center,
                *self.scale,
                *self.weights,
                self.intercept,
            )
        ):
            raise ValueError("answer-value artifact contains non-finite parameters")
        if not all(
            math.isfinite(float(value)) for value in self.metrics.values()
        ):
            raise ValueError("answer-value artifact contains non-finite metrics")
        if self.fit_question_id_sha256 == self.validation_question_id_sha256:
            raise ValueError("fit and validation question hashes must differ")
        assert_sf_label_free(self.metadata, path="answer_value.metadata")

    def score(self, features: Mapping[str, float]) -> float:
        if set(features) != set(self.feature_names):
            raise ValueError(
                "answer-value features differ from artifact contract: "
                f"expected={self.feature_names}, got={tuple(features)}"
            )
        vector = np.asarray(
            [float(features[name]) for name in self.feature_names], dtype=np.float64
        )
        normalized = (vector - np.asarray(self.center)) / np.asarray(self.scale)
        return float(self.intercept + normalized @ np.asarray(self.weights))

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "metrics": dict(self.metrics), "metadata": dict(self.metadata)}

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnswerValueArtifactV7":
        if payload.get("schema") != ANSWER_VALUE_SCHEMA_V7:
            raise ValueError("not an ACEC v7 answer-value artifact")
        return cls(
            feature_names=tuple(str(v) for v in payload["feature_names"]),
            center=tuple(float(v) for v in payload["center"]),
            scale=tuple(float(v) for v in payload["scale"]),
            weights=tuple(float(v) for v in payload["weights"]),
            intercept=float(payload["intercept"]),
            fit_question_id_sha256=str(payload["fit_question_id_sha256"]),
            validation_question_id_sha256=str(
                payload["validation_question_id_sha256"]
            ),
            metrics=dict(payload.get("metrics") or {}),
            metadata=dict(payload.get("metadata") or {}),
            schema=str(payload["schema"]),
            version=int(payload.get("version", ANSWER_VALUE_VERSION_V7)),
        )


@dataclass(frozen=True)
class AnswerValueCrossFitBundleV7:
    full_artifact: AnswerValueArtifactV7
    fold_artifacts: Tuple[AnswerValueArtifactV7, ...]
    question_to_fold: Mapping[str, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ANSWER_VALUE_CROSSFIT_SCHEMA_V7
    version: int = ANSWER_VALUE_VERSION_V7

    def __post_init__(self) -> None:
        if len(self.fold_artifacts) < 2:
            raise ValueError("answer-value cross-fit bundle requires >=2 folds")
        for question_id, fold in self.question_to_fold.items():
            if not question_id or not 0 <= int(fold) < len(self.fold_artifacts):
                raise ValueError("invalid answer-value question-to-fold mapping")
        assert_sf_label_free(self.metadata, path="answer_value_crossfit.metadata")

    def artifact_for_question(self, question_id: str) -> AnswerValueArtifactV7:
        fold = self.question_to_fold.get(str(question_id))
        return (
            self.full_artifact
            if fold is None
            else self.fold_artifacts[int(fold)]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "full_artifact": self.full_artifact.to_dict(),
            "fold_artifacts": [
                artifact.to_dict() for artifact in self.fold_artifacts
            ],
            "question_to_fold": {
                str(key): int(value)
                for key, value in self.question_to_fold.items()
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "AnswerValueCrossFitBundleV7":
        if payload.get("schema") != ANSWER_VALUE_CROSSFIT_SCHEMA_V7:
            raise ValueError("not an ACEC v7 answer-value cross-fit bundle")
        return cls(
            full_artifact=AnswerValueArtifactV7.from_dict(
                payload["full_artifact"]
            ),
            fold_artifacts=tuple(
                AnswerValueArtifactV7.from_dict(value)
                for value in payload["fold_artifacts"]
            ),
            question_to_fold={
                str(key): int(value)
                for key, value in payload["question_to_fold"].items()
            },
            metadata=dict(payload.get("metadata") or {}),
            schema=str(payload["schema"]),
            version=int(payload.get("version", ANSWER_VALUE_VERSION_V7)),
        )


def _matrix(
    examples: Sequence[AnswerValueExampleV7], feature_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    expected = set(feature_names)
    for example in examples:
        if set(example.features) != expected:
            raise ValueError("answer-value examples use inconsistent features")
    x = np.asarray(
        [[float(example.features[name]) for name in feature_names] for example in examples],
        dtype=np.float64,
    )
    y = np.asarray([float(example.target_delta) for example in examples], dtype=np.float64)
    return x, y


def _metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    examples: Sequence[AnswerValueExampleV7],
) -> Dict[str, float]:
    error = prediction - target
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    if len(target) >= 2 and float(np.std(target)) > 0 and float(np.std(prediction)) > 0:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    else:
        correlation = 0.0
    if len(target) >= 2:
        def rank(values: np.ndarray) -> np.ndarray:
            order = np.argsort(values, kind="mergesort")
            ranks = np.empty(len(values), dtype=np.float64)
            start = 0
            while start < len(values):
                end = start + 1
                while (
                    end < len(values)
                    and values[order[end]] == values[order[start]]
                ):
                    end += 1
                average = (start + end - 1) / 2.0
                ranks[order[start:end]] = average
                start = end
            return ranks

        prediction_rank = rank(prediction)
        target_rank = rank(target)
        if float(np.std(prediction_rank)) > 0 and float(np.std(target_rank)) > 0:
            spearman = float(
                np.corrcoef(prediction_rank, target_rank)[0, 1]
            )
        else:
            spearman = 0.0
    else:
        spearman = 0.0
    pair_correct = 0
    pair_total = 0
    by_state: Dict[str, List[int]] = {}
    for index, example in enumerate(examples):
        group = str(example.state_id or example.question_id)
        by_state.setdefault(group, []).append(index)
    for indices in by_state.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                gold = float(target[left] - target[right])
                if gold == 0.0:
                    continue
                pair_total += 1
                pair_correct += int(
                    (prediction[left] - prediction[right]) * gold > 0.0
                )
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson": correlation,
        "spearman": spearman,
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
        "pairwise_pairs": float(pair_total),
        "examples": float(len(target)),
    }


def fit_answer_value_v7(
    fit_examples: Sequence[AnswerValueExampleV7],
    validation_examples: Sequence[AnswerValueExampleV7],
    *,
    l2: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> AnswerValueArtifactV7:
    if not fit_examples or not validation_examples:
        raise ValueError("answer-value fit/validation examples must be non-empty")
    fit_ids = {example.question_id for example in fit_examples}
    validation_ids = {example.question_id for example in validation_examples}
    overlap = fit_ids & validation_ids
    if overlap:
        raise ValueError(
            f"answer-value question split leaks {len(overlap)} ids"
        )
    feature_names = tuple(sorted(fit_examples[0].features))
    x_fit, y_fit = _matrix(fit_examples, feature_names)
    x_validation, y_validation = _matrix(validation_examples, feature_names)
    center = np.median(x_fit, axis=0)
    q75 = np.percentile(x_fit, 75, axis=0)
    q25 = np.percentile(x_fit, 25, axis=0)
    scale = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
    z_fit = (x_fit - center) / scale
    z_validation = (x_validation - center) / scale
    design = np.concatenate([np.ones((len(z_fit), 1)), z_fit], axis=1)
    penalty = np.eye(design.shape[1]) * float(l2)
    penalty[0, 0] = 0.0
    solution = np.linalg.solve(
        design.T @ design + penalty, design.T @ y_fit
    )
    fit_prediction = design @ solution
    validation_prediction = np.concatenate(
        [np.ones((len(z_validation), 1)), z_validation], axis=1
    ) @ solution
    metrics = {
        **{
            f"fit_{key}": value
            for key, value in _metrics(
                fit_prediction, y_fit, fit_examples
            ).items()
        },
        **{
            f"validation_{key}": value
            for key, value in _metrics(
                validation_prediction,
                y_validation,
                validation_examples,
            ).items()
        },
    }
    return AnswerValueArtifactV7(
        feature_names=feature_names,
        center=tuple(float(v) for v in center),
        scale=tuple(float(v) for v in scale),
        weights=tuple(float(v) for v in solution[1:]),
        intercept=float(solution[0]),
        fit_question_id_sha256=stable_id_hash(tuple(fit_ids)),
        validation_question_id_sha256=stable_id_hash(tuple(validation_ids)),
        metrics=metrics,
        metadata=dict(metadata or {}),
    )


def fit_answer_value_crossfit_v7(
    fit_examples: Sequence[AnswerValueExampleV7],
    validation_examples: Sequence[AnswerValueExampleV7],
    *,
    folds: int = 5,
    l2: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> AnswerValueCrossFitBundleV7:
    if folds < 2:
        raise ValueError("cross-fit folds must be >=2")
    fit_ids = sorted({example.question_id for example in fit_examples})
    if len(fit_ids) < folds:
        raise ValueError("cross-fit needs at least one question per fold")
    question_to_fold = {
        question_id: int(
            int.from_bytes(
                hashlib.sha256(question_id.encode("utf-8")).digest()[:8],
                "big",
            )
            % folds
        )
        for question_id in fit_ids
    }
    if len(set(question_to_fold.values())) != folds:
        raise ValueError("hashed cross-fit assignment produced an empty fold")
    fold_artifacts = []
    for fold in range(folds):
        fold_fit = [
            example
            for example in fit_examples
            if question_to_fold[example.question_id] != fold
        ]
        fold_validation = [
            example
            for example in fit_examples
            if question_to_fold[example.question_id] == fold
        ]
        fold_artifacts.append(
            fit_answer_value_v7(
                fold_fit,
                fold_validation,
                l2=l2,
                metadata={
                    **dict(metadata or {}),
                    "crossfit_fold": fold,
                    "crossfit_folds": folds,
                },
            )
        )
    full = fit_answer_value_v7(
        fit_examples,
        validation_examples,
        l2=l2,
        metadata={
            **dict(metadata or {}),
            "runtime_full_fit": True,
            "crossfit_folds": folds,
        },
    )
    return AnswerValueCrossFitBundleV7(
        full_artifact=full,
        fold_artifacts=tuple(fold_artifacts),
        question_to_fold=question_to_fold,
        metadata={
            **dict(metadata or {}),
            "crossfit_folds": folds,
            "fit_question_id_sha256": stable_id_hash(tuple(fit_ids)),
        },
    )


def save_answer_value_artifact_v7(
    path: Path, artifact: AnswerValueArtifactV7
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite answer-value artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_answer_value_artifact_v7(path: Path) -> AnswerValueArtifactV7:
    return AnswerValueArtifactV7.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def save_answer_value_crossfit_bundle_v7(
    path: Path, bundle: AnswerValueCrossFitBundleV7
) -> None:
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite answer-value cross-fit bundle: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_answer_value_crossfit_bundle_v7(
    path: Path,
) -> AnswerValueCrossFitBundleV7:
    return AnswerValueCrossFitBundleV7.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


__all__ = [
    "ANSWER_VALUE_SCHEMA_V7",
    "AnswerValueArtifactV7",
    "AnswerValueCrossFitBundleV7",
    "AnswerValueExampleV7",
    "fit_answer_value_crossfit_v7",
    "fit_answer_value_v7",
    "load_answer_value_artifact_v7",
    "load_answer_value_crossfit_bundle_v7",
    "save_answer_value_artifact_v7",
    "save_answer_value_crossfit_bundle_v7",
]
