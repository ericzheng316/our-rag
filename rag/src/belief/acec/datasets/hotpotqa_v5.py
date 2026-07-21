"""HotpotQA adapter for the dataset-agnostic ACEC v5 evidence contract.

The adapter groups supporting sentences by source paragraph, producing one
minimal evidence requirement per annotated source/hop.  Titles and sentence
ids are retained only as provenance.  The requirement target itself is the
supporting sentence text, which prevents a same-title but non-supporting
paragraph from becoming a hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from ..evidence_standard_v5 import (
    AnnotationScope,
    EvidenceRequirement,
    EvidenceSpecification,
    EvidenceUnit,
)


def _context_pairs(context: Any) -> List[Tuple[str, Sequence[str]]]:
    if isinstance(context, dict):
        titles = context.get("title", context.get("titles", []))
        sentences = context.get("sentences", [])
        return [(str(title), list(parts)) for title, parts in zip(titles, sentences)]
    if isinstance(context, list):
        pairs = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), list(item[1])))
            elif isinstance(item, dict):
                title = item.get("title", item.get("source", ""))
                parts = item.get("sentences", item.get("text", []))
                if isinstance(parts, str):
                    parts = [parts]
                pairs.append((str(title), list(parts)))
        return pairs
    return []


@dataclass
class HotpotQAEvidenceAdapterV5:
    name: str = "hotpotqa_supporting_sentences_v1"

    def evidence_specification(self, record: Dict[str, Any]) -> EvidenceSpecification:
        annotation = record.get("_annotation_record") or record
        supporting_facts = annotation.get("supporting_facts") or record.get("supporting_facts") or {}
        sf_titles = list(supporting_facts.get("title") or [])
        sf_sent_ids = list(supporting_facts.get("sent_id") or [])
        context_by_title = {
            title: list(sentences)
            for title, sentences in _context_pairs(annotation.get("context"))
        }

        grouped: Dict[str, List[int]] = {}
        title_order: List[str] = []
        for title, sent_id in zip(sf_titles, sf_sent_ids):
            title = str(title)
            if title not in grouped:
                grouped[title] = []
                title_order.append(title)
            grouped[title].append(int(sent_id))

        requirements = []
        for requirement_index, title in enumerate(title_order):
            units = []
            source_sentences = context_by_title.get(title, [])
            for sent_id in grouped[title]:
                if sent_id < 0 or sent_id >= len(source_sentences):
                    continue
                text = str(source_sentences[sent_id]).strip()
                if not text:
                    continue
                units.append(
                    EvidenceUnit(
                        unit_id=f"{requirement_index}:{sent_id}",
                        text=text,
                        source_id=title,
                        metadata={"title": title, "sent_id": sent_id},
                    )
                )
            if not units:
                continue
            requirements.append(
                EvidenceRequirement(
                    requirement_id=f"hop:{requirement_index}",
                    description=f"{title}: " + " ".join(unit.text for unit in units),
                    evidence_units=tuple(units),
                    satisfaction_rule="all",
                    metadata={"source_title": title},
                )
            )

        question_id = record.get("_canonical_question_id")
        if question_id is None:
            question_id = annotation.get(
                "id",
                annotation.get(
                    "_id",
                    record.get("id", record.get("_id", record.get("question", record.get("problem", "")))),
                ),
            )
        return EvidenceSpecification(
            question_id=str(question_id),
            requirements=tuple(requirements),
            annotation_scope=AnnotationScope.CLOSED_CANDIDATE_POOL,
            adapter_name=self.name,
            annotation_version="hotpotqa_supporting_facts_context_v1",
            metadata={
                "question_type": annotation.get(
                    "question_type", annotation.get("type", record.get("question_type"))
                )
            },
        )


__all__ = ["HotpotQAEvidenceAdapterV5"]
