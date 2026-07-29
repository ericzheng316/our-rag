"""Native-provenance HotpotQA distractor adapter for ACEC v6.3.

Unlike the legacy distractor preprocessor, this module never joins sentences
and never reconstructs sentence ids.  Official ``(title, sent_id)`` identity
comes directly from HotpotQA's ``context`` arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from ..evidence_contract_v63 import EvidenceUnitV63


ADAPTER_NAME = "hotpotqa_distractor_native"
ADAPTER_VERSION = "63.1"


def question_id(record: Mapping[str, Any]) -> str:
    value = record.get("_id", record.get("id"))
    if value is None or not str(value).strip():
        raise ValueError("HotpotQA record lacks _id/id")
    return str(value)


def context_pairs(context: Any) -> List[Tuple[str, List[str]]]:
    """Normalize HF dict and original list encodings without re-segmenting."""

    pairs: List[Tuple[str, List[str]]] = []
    if isinstance(context, Mapping):
        titles = context.get("title", context.get("titles", []))
        sentences = context.get("sentences", [])
        if len(titles) != len(sentences):
            raise ValueError("HotpotQA context title/sentences lengths differ")
        raw_pairs: Iterable[Any] = zip(titles, sentences)
    elif isinstance(context, list):
        raw_pairs = context
    else:
        raise ValueError("HotpotQA context must be a mapping or list")

    for item in raw_pairs:
        if isinstance(item, Mapping):
            title = item.get("title")
            parts = item.get("sentences")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            title, parts = item
        else:
            raise ValueError(f"unsupported HotpotQA context item: {item!r}")
        if not str(title or "").strip():
            raise ValueError("HotpotQA context contains an empty title")
        if not isinstance(parts, (list, tuple)):
            raise ValueError(f"HotpotQA context for {title!r} is not sentence-indexed")
        pairs.append((str(title), [str(sentence) for sentence in parts]))
    return pairs


def supporting_fact_pairs(record: Mapping[str, Any]) -> Tuple[Tuple[str, int], ...]:
    facts = record.get("supporting_facts", [])
    if isinstance(facts, Mapping):
        titles = facts.get("title", [])
        sent_ids = facts.get("sent_id", [])
        if len(titles) != len(sent_ids):
            raise ValueError("supporting_facts title/sent_id lengths differ")
        raw = zip(titles, sent_ids)
    elif isinstance(facts, list):
        raw = facts
    else:
        raise ValueError("unsupported HotpotQA supporting_facts encoding")
    pairs = {(str(title), int(sent_id)) for title, sent_id in raw}
    return tuple(sorted(pairs, key=lambda value: (value[0], value[1])))


@dataclass(frozen=True)
class HotpotQADistractorAdapterV63:
    name: str = ADAPTER_NAME
    version: str = ADAPTER_VERSION

    def evidence_units(
        self,
        record: Mapping[str, Any],
        *,
        retrieval_turn: int = 0,
    ) -> Tuple[EvidenceUnitV63, ...]:
        qid = question_id(record)
        units: List[EvidenceUnitV63] = []
        for document_rank, (title, sentences) in enumerate(
            context_pairs(record.get("context"))
        ):
            document_id = f"{qid}:doc:{document_rank}"
            for sent_id, sentence in enumerate(sentences):
                if not sentence.strip():
                    raise ValueError(
                        f"HotpotQA native sentence is empty: {qid} {title!r} {sent_id}"
                    )
                units.append(
                    EvidenceUnitV63(
                        evidence_id=f"{qid}:doc:{document_rank}:sent:{sent_id}",
                        source_id=title,
                        unit_index=sent_id,
                        text=sentence,
                        document_id=document_id,
                        retrieval_turn=retrieval_turn,
                        retrieval_rank=document_rank,
                        provenance_status="native",
                        metadata={"document_rank": document_rank},
                    )
                )
        return tuple(units)

    def runtime_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Return candidate data safe for the selector process."""

        qid = question_id(record)
        question = record.get("question")
        if not str(question or "").strip():
            raise ValueError(f"HotpotQA record {qid} lacks question")
        pairs = context_pairs(record.get("context"))
        documents = []
        for rank, (title, sentences) in enumerate(pairs):
            documents.append(
                {
                    "document_id": f"{qid}:doc:{rank}",
                    "title": title,
                    "sentences": list(sentences),
                    "sentence_ids": list(range(len(sentences))),
                    "contents": f"{title}: " + " ".join(sentences),
                    "provenance_status": "native",
                    "retrieval_rank": rank,
                }
            )
        return {
            "schema": "hotpotqa_distractor_runtime",
            "schema_version": 63,
            "question_id": qid,
            "question": str(question),
            "documents": documents,
            "candidate_evidence": [
                unit.to_dict() for unit in self.evidence_units(record)
            ],
            "metadata": {
                "adapter": self.name,
                "adapter_version": self.version,
                "question_type": record.get("type", record.get("question_type")),
                "level": record.get("level"),
            },
        }

    def evaluator_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the minimal record accepted by the unchanged official script."""

        qid = question_id(record)
        answer = record.get("answer")
        if answer is None:
            golden = record.get("golden_answers") or []
            answer = golden[0] if golden else None
        if answer is None:
            raise ValueError(f"HotpotQA record {qid} lacks an evaluator answer")
        return {
            "_id": qid,
            "answer": str(answer),
            "supporting_facts": [list(pair) for pair in supporting_fact_pairs(record)],
            "type": record.get("type", record.get("question_type")),
            "level": record.get("level"),
        }

    def validate_round_trip(self, record: Mapping[str, Any]) -> None:
        context = context_pairs(record.get("context"))
        units = self.evidence_units(record)
        expected_count = sum(len(sentences) for _, sentences in context)
        if len(units) != expected_count:
            raise AssertionError("native sentence count changed during adaptation")
        for unit in units:
            rank = int(unit.metadata["document_rank"])
            title, sentences = context[rank]
            if unit.source_id != title or unit.text != sentences[int(unit.unit_index)]:
                raise AssertionError("native HotpotQA provenance failed round trip")
        candidates = {
            unit.official_hotpot_pair
            for unit in units
            if unit.official_hotpot_pair is not None
        }
        missing_gold = set(supporting_fact_pairs(record)) - candidates
        if missing_gold:
            raise ValueError(
                f"HotpotQA supporting facts do not map to native context: {missing_gold}"
            )


__all__ = [
    "ADAPTER_NAME",
    "ADAPTER_VERSION",
    "HotpotQADistractorAdapterV63",
    "context_pairs",
    "question_id",
    "supporting_fact_pairs",
]
