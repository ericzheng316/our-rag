"""Adapter for the canonical, dataset-neutral ACEC v5 evidence manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..evidence_standard_v5 import (
    AnnotationScope,
    EvidenceRequirement,
    EvidenceSpecification,
    EvidenceUnit,
)


@dataclass
class CanonicalEvidenceAdapterV5:
    """Read a pre-normalized ``evidence_specification`` JSON object.

    This is the preferred boundary for new datasets.  Dataset conversion is a
    separate preprocessing concern; the v5 builder itself only needs this
    common requirement/unit representation.
    """

    name: str = "canonical_evidence_manifest_v1"

    def evidence_specification(self, record: Dict[str, Any]) -> EvidenceSpecification:
        annotation = record.get("_annotation_record") or record
        payload = annotation.get("evidence_specification") or record.get(
            "evidence_specification"
        )
        if not isinstance(payload, dict):
            raise ValueError("record is missing canonical evidence_specification")
        requirements = []
        for requirement_payload in payload.get("requirements", []):
            units = tuple(
                EvidenceUnit(
                    unit_id=str(unit["unit_id"]),
                    text=str(unit["text"]),
                    source_id=(str(unit["source_id"]) if unit.get("source_id") is not None else None),
                    metadata=dict(unit.get("metadata") or {}),
                )
                for unit in requirement_payload.get("evidence_units", [])
            )
            requirements.append(
                EvidenceRequirement(
                    requirement_id=str(requirement_payload["requirement_id"]),
                    description=str(requirement_payload.get("description", "")),
                    evidence_units=units,
                    satisfaction_rule=str(requirement_payload.get("satisfaction_rule", "all")),
                    metadata=dict(requirement_payload.get("metadata") or {}),
                )
            )
        question_id = payload.get(
            "question_id",
            record.get("id", record.get("_id", record.get("problem", ""))),
        )
        return EvidenceSpecification(
            question_id=str(question_id),
            requirements=tuple(requirements),
            annotation_scope=AnnotationScope(
                payload.get("annotation_scope", AnnotationScope.SUPPORT_ONLY.value)
            ),
            adapter_name=str(payload.get("adapter_name", self.name)),
            annotation_version=str(payload.get("annotation_version", "1")),
            metadata=dict(payload.get("metadata") or {}),
        )


__all__ = ["CanonicalEvidenceAdapterV5"]
