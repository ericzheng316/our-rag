"""MuSiQue-Ans adapter for the dataset-agnostic ACEC v5 evidence contract.

One requirement per decomposition step; the evidence unit is the step's
supporting paragraph (paragraph-level granularity, mirroring the HotpotQA
adapter).  The 20-paragraph candidate pool makes the annotation scope
CLOSED_CANDIDATE_POOL — the same closed-world property that licenses
confident zero-gain labels during calibration.

MuSiQue decomposition questions contain back-references ("#1", "#2") to
earlier step answers; descriptions resolve them so slot-requirement E5
assignment sees natural text instead of placeholders.

Source data: dgslibisey/MuSiQue (musique_ans_v1.0_{train,dev}.jsonl),
linked at ~/acec/datasets/musique/.  The -Ans release is answerable-only;
the unanswerable twins of MuSiQue-Full are NOT in this mirror.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

from ..evidence_standard_v5 import (
    AnnotationScope,
    EvidenceRequirement,
    EvidenceSpecification,
    EvidenceUnit,
)

_BACKREF = re.compile(r"#(\d+)")


def _resolve_backrefs(question: str, step_answers: Dict[int, str]) -> str:
    return _BACKREF.sub(
        lambda m: step_answers.get(int(m.group(1)), m.group(0)), question
    )


@dataclass
class MuSiQueEvidenceAdapterV5:
    name: str = "musique_decomposition_v1"

    def evidence_specification(self, record: Dict[str, Any]) -> EvidenceSpecification:
        annotation = record.get("_annotation_record") or record
        paragraphs = {int(p["idx"]): p for p in annotation.get("paragraphs", [])}
        decomposition = annotation.get("question_decomposition") or []
        if not decomposition:
            raise ValueError("MuSiQue record has no question_decomposition")

        # "#N" back-references are POSITIONAL (1-indexed step order), not the
        # global step ids MuSiQue stores in step["id"].
        step_answers = {
            position + 1: str(step.get("answer", ""))
            for position, step in enumerate(decomposition)
        }
        requirements = []
        for step in decomposition:
            step_id = int(step["id"])
            support_idx = step.get("paragraph_support_idx")
            paragraph = paragraphs.get(int(support_idx)) if support_idx is not None else None
            if paragraph is None:
                continue
            unit = EvidenceUnit(
                unit_id=f"step{step_id}:para{paragraph['idx']}",
                text=str(paragraph["paragraph_text"]),
                source_id=str(paragraph["title"]),
                metadata={
                    "paragraph_idx": int(paragraph["idx"]),
                    "step_question": step["question"],
                    "step_answer": step.get("answer"),
                },
            )
            requirements.append(
                EvidenceRequirement(
                    requirement_id=f"step:{step_id}",
                    description=(
                        f"{paragraph['title']}: "
                        f"{_resolve_backrefs(str(step['question']), step_answers)}"
                        f" — {step.get('answer', '')}"
                    ),
                    evidence_units=(unit,),
                    satisfaction_rule="all",
                    metadata={"step_id": step_id, "hop_family": str(annotation.get("id", "")).split("__")[0]},
                )
            )
        if not requirements:
            raise ValueError("MuSiQue record has no usable supporting paragraphs")

        return EvidenceSpecification(
            question_id=str(annotation.get("id", record.get("id", ""))),
            requirements=tuple(requirements),
            annotation_scope=AnnotationScope.CLOSED_CANDIDATE_POOL,
            adapter_name=self.name,
            annotation_version="musique_ans_v1.0",
            metadata={
                "answerable": bool(annotation.get("answerable", True)),
                "answer": annotation.get("answer"),
                "answer_aliases": list(annotation.get("answer_aliases") or []),
            },
        )


__all__ = ["MuSiQueEvidenceAdapterV5"]
