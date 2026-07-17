"""ACEC calibration artifact with selected-evidence label semantics.

Version 4 intentionally reuses v3's monotonic, role-aware observation model
while changing the supervised event recorded in the artifact contract:

    the document that produced the runtime max-NLI score supports the slot.

Earlier builders paired a max score with a positive label whenever *any*
document in the retrieval batch had the assigned gold title.  That can label
a distractor's high score as a hit when a different, lower-scoring document is
gold evidence.  V4 artifacts reject that batch-level label schema explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .calibration_v3 import (
    MODEL_KIND,
    MonotonicObservationModel,
    build_k_predictor,
    fit_observation_model_v3,
    k_predictor_payload,
    posterior_quality_metrics,
)
from .offline_fit import HitExample


ARTIFACT_TYPE = "acec_calibration"
ARTIFACT_VERSION = 4
LABEL_SCHEMA = "selected_evidence_title_v1"


@dataclass
class CalibrationArtifactV4:
    observation_model: MonotonicObservationModel
    hit_rates: Dict[str, float]
    k_predictor_payload: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    metrics: Dict[str, Any]


def fit_observation_model_v4(
    examples: Sequence[HitExample],
    action_modes: Sequence[str] = ("EXPAND", "REWRITE", "DECOMPOSE"),
    action_prior_strength: float = 5.0,
) -> Tuple[MonotonicObservationModel, Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Fit v3's monotonic model to v4 selected-evidence labels."""

    return fit_observation_model_v3(
        examples,
        action_modes=action_modes,
        action_prior_strength=action_prior_strength,
    )


def save_calibration_artifact_v4(
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
        "label_schema": LABEL_SCHEMA,
        "observation_model": observation_model.to_dict(),
        "hit_rates": hit_rates,
        "k_predictor": k_payload,
        "metadata": metadata,
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_calibration_artifact_v4(path: str) -> CalibrationArtifactV4:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError(f"not an ACEC calibration artifact: {path}")
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported ACEC artifact version {payload.get('artifact_version')}; "
            f"expected {ARTIFACT_VERSION}"
        )
    if payload.get("label_schema") != LABEL_SCHEMA:
        raise ValueError(
            f"unsupported ACEC v4 label schema {payload.get('label_schema')}; "
            f"expected {LABEL_SCHEMA}"
        )
    required = ("observation_model", "hit_rates", "metadata", "metrics")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"v4 calibration artifact missing fields: {missing}")
    return CalibrationArtifactV4(
        observation_model=MonotonicObservationModel.from_dict(payload["observation_model"]),
        hit_rates={key: float(value) for key, value in payload["hit_rates"].items()},
        k_predictor_payload=payload.get("k_predictor"),
        metadata=payload["metadata"],
        metrics=payload["metrics"],
    )


__all__ = [
    "ARTIFACT_TYPE",
    "ARTIFACT_VERSION",
    "LABEL_SCHEMA",
    "MODEL_KIND",
    "CalibrationArtifactV4",
    "build_k_predictor",
    "fit_observation_model_v4",
    "k_predictor_payload",
    "load_calibration_artifact_v4",
    "posterior_quality_metrics",
    "save_calibration_artifact_v4",
]
