"""Build an ACEC v5 artifact under the marginal-evidence-utility standard.

The builder replays logged trajectories in time order.  For every slot it
audits the exact document selected by runtime max NLI, measures whether that
document satisfies an assigned minimal evidence requirement, and converts the
static assessment into a conditional delta relative to evidence already seen.

HotpotQA is one adapter, not the target schema.  New datasets should preferably
provide the canonical evidence manifest consumed by ``CanonicalEvidenceAdapterV5``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rag" / "src"))

from belief.acec import (  # noqa: E402
    ACECBeliefState,
    ACECConfig,
    CosineNLIScorer,
    CrossEncoderNLIScorer,
    E5QueryEmbedder,
)
from belief.acec.calibration_v5 import (  # noqa: E402
    MODEL_KIND,
    MarginalUtilityExample,
    choose_binding_threshold_v5,
    fit_observation_model_v5,
    posterior_quality_metrics_v5,
    predict_examples_v5,
    save_calibration_artifact_v5,
    strict_json_value,
    dump_strict_json,
)
from belief.acec.datasets.canonical_v5 import CanonicalEvidenceAdapterV5  # noqa: E402
from belief.acec.datasets.hotpotqa_v5 import HotpotQAEvidenceAdapterV5  # noqa: E402
from belief.acec.evidence_standard_v5 import (  # noqa: E402
    EvidenceSpecificationAdapter,
    RequirementCoverageHistory,
    assess_selected_document,
    assign_slots_to_requirements,
    document_novelty_score,
    normalize_text,
    standard_payload,
)
from belief.acec.k_strategy_v5 import select_k_strategy_v5  # noqa: E402
from belief.acec.offline_fit import KExample  # noqa: E402


@dataclass
class MarginalUtilityRow:
    record_id: str
    question_id_source: str
    question: str
    turn_index: int
    slot_index: int
    slot_hypothesis: str
    action_mode: str
    role: str
    bound: bool
    support_score: float
    novelty_score: float
    selected_document: str
    selected_document_id: str
    selected_document_id_source: str
    selected_source_id: str
    evidence_adapter: str
    annotation_version: str
    annotation_scope: str
    utility_basis: str
    requirement_id: Optional[str]
    evidence_unit_id: Optional[str]
    assignment_similarity: Optional[float]
    assignment_margin: Optional[float]
    assignment_reason: str
    static_support: Optional[float]
    static_assessment_basis: str
    assessment_confidence: float
    utility_before: Optional[float]
    utility_after: Optional[float]
    marginal_utility: Optional[float]
    new_requirement_coverage: Optional[bool]
    repeated_document: bool
    fit_eligible: bool
    rationale: str

    def fit_example(self) -> Optional[MarginalUtilityExample]:
        if not self.fit_eligible or self.new_requirement_coverage is None:
            return None
        return MarginalUtilityExample(
            action_mode=self.action_mode,
            role=self.role,
            bound=self.bound,
            support_score=self.support_score,
            novelty_score=self.novelty_score,
            is_gain=self.new_requirement_coverage,
            marginal_utility=float(self.marginal_utility or 0.0),
        )


class NoUsableEvidenceError(ValueError):
    """The record lacks the annotation required by its selected adapter."""


def _nonempty_id(payload: Dict[str, Any]) -> str:
    for key in ("id", "_id", "question_id"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _record_id(record: Dict[str, Any]) -> str:
    value = _nonempty_id(record)
    if value:
        return value
    canonical = record.get("_canonical_question_id")
    if canonical is not None and str(canonical).strip():
        return str(canonical).strip()
    annotation = record.get("_annotation_record")
    if isinstance(annotation, dict):
        value = _nonempty_id(annotation)
        if value:
            return value
    question = _question(record)
    if question:
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        return f"question_sha256:{digest}"
    return ""


def _question_id_source(record: Dict[str, Any]) -> str:
    return str(record.get("_question_id_source") or "unknown")


def _question(record: Dict[str, Any]) -> str:
    value = record.get("problem", record.get("question", ""))
    return str(value)


def _selected_document(selected_doc: Any) -> Tuple[str, str, str, str]:
    if isinstance(selected_doc, str):
        text = selected_doc
        source_id = text.split(":", 1)[0].strip() if ":" in text else ""
        digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
        return text, source_id, f"content_sha256:{digest}", "content_sha256"
    if isinstance(selected_doc, dict):
        text = str(selected_doc.get("contents", selected_doc.get("text", "")))
        source_id = str(
            selected_doc.get(
                "title",
                selected_doc.get("source_id", text.split(":", 1)[0].strip() if ":" in text else ""),
            )
        )
        document_id = ""
        for key in ("document_id", "passage_id", "doc_id", "id"):
            value = selected_doc.get(key)
            if value is not None and str(value).strip():
                document_id = str(value).strip()
                break
        id_source = str(selected_doc.get("document_id_source") or "native")
        if not document_id:
            digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
            document_id = f"content_sha256:{digest}"
            id_source = "content_sha256"
        return text, source_id, document_id, id_source
    raise TypeError(f"unsupported selected evidence type: {type(selected_doc).__name__}")


def _annotation_context_entries(record: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Index annotated candidate paragraphs by normalized full text.

    Legacy R3 logs store retrieved documents as strings.  HotpotQA annotations
    still provide an auditable question-scoped paragraph index, so recover a
    stable ID when the logged text exactly matches that closed candidate pool.
    """

    annotation = record.get("_annotation_record")
    if not isinstance(annotation, dict):
        return {}
    context = annotation.get("context")
    pairs: List[Tuple[str, Sequence[str]]] = []
    if isinstance(context, dict):
        titles = context.get("title", context.get("titles", []))
        sentences = context.get("sentences", [])
        pairs = [(str(title), list(parts)) for title, parts in zip(titles, sentences)]
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), list(item[1])))
            elif isinstance(item, dict):
                title = str(item.get("title", item.get("source", "")))
                parts = item.get("sentences", item.get("text", []))
                pairs.append((title, [parts] if isinstance(parts, str) else list(parts)))
    question_id = _record_id(record)
    entries: Dict[str, Dict[str, str]] = {}
    for index, (title, sentences) in enumerate(pairs):
        text = f"{title}: " + " ".join(str(sentence) for sentence in sentences)
        entries[normalize_text(text)] = {
            "document_id": f"{question_id}:context:{index}",
            "document_id_source": "annotation_context",
            "source_id": title,
        }
    return entries


def _enrich_document(
    document: Any, context_entries: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    payload = dict(document) if isinstance(document, dict) else {"contents": str(document)}
    text = str(payload.get("contents", payload.get("text", "")))
    payload["contents"] = text
    has_native_id = any(
        payload.get(key) is not None and str(payload.get(key)).strip()
        for key in ("document_id", "passage_id", "doc_id", "id")
    )
    if has_native_id:
        payload.setdefault("document_id_source", "native")
        return payload
    matched = context_entries.get(normalize_text(text))
    if matched:
        payload.update(matched)
        return payload
    digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
    payload["document_id"] = f"content_sha256:{digest}"
    payload["document_id_source"] = "content_sha256"
    return payload


def replay_record(
    record: Dict[str, Any],
    belief: ACECBeliefState,
    adapter: EvidenceSpecificationAdapter,
    min_fit_confidence: float = 0.90,
    assignment_min_similarity: float = 0.30,
    assignment_min_margin: float = 0.05,
    span_support_threshold: float = 0.90,
    span_absence_threshold: float = 0.35,
) -> Tuple[List[MarginalUtilityRow], KExample]:
    specification = adapter.evidence_specification(record)
    if not specification.requirements:
        raise NoUsableEvidenceError("record has no auditable evidence requirements")

    question = _question(record)
    belief.reset(question)
    split_queries = record.get("split_querys", [])
    docs_per_turn = record.get("docs", [])
    context_entries = _annotation_context_entries(record)
    per_turn = []
    for turn_index, docs in enumerate(docs_per_turn):
        query = (
            split_queries[turn_index][0]
            if turn_index < len(split_queries) and split_queries[turn_index]
            else None
        )
        bound_before = {
            index: slot.bound for index, slot in enumerate(belief.coverage_belief.slots)
        }
        result = belief.turn(
            query=query,
            new_docs=[_enrich_document(doc, context_entries) for doc in docs],
            is_answer=False,
        )
        bound_at_score = {
            slot_index: bound_before.get(slot_index, False)
            for slot_index in result.slot_scores
        }
        per_turn.append((turn_index, result, bound_at_score))

    slot_hypotheses = [slot.hypothesis for slot in belief.coverage_belief.slots]
    assignments = assign_slots_to_requirements(
        slot_hypotheses,
        specification.requirements,
        belief.labeler.embedder,
        min_similarity=assignment_min_similarity,
        min_margin=assignment_min_margin,
    )
    history = RequirementCoverageHistory(len(specification.requirements))
    prior_documents: List[str] = []
    rows: List[MarginalUtilityRow] = []

    for turn_index, result, bound_at_score in per_turn:
        selected_docs = getattr(result, "slot_best_docs", None)
        if selected_docs is None:
            raise RuntimeError(
                "calibration v5 requires TurnResult.slot_best_docs for score/document alignment"
            )
        turn_documents: List[str] = []
        prior_fingerprints = set(history.selected_document_fingerprints)
        covered_before_turn = set(history.covered_requirement_ids)
        newly_covered_this_turn = set()
        for slot_index, support_score in result.slot_scores.items():
            if slot_index not in selected_docs:
                raise RuntimeError(
                    f"slot {slot_index} has a support score but no selected document"
                )
            (
                selected_text,
                selected_source_id,
                selected_document_id,
                selected_document_id_source,
            ) = _selected_document(selected_docs[slot_index])
            turn_documents.append(selected_text)
            assignment = assignments[slot_index]
            requirement = (
                specification.requirements[assignment.requirement_index]
                if assignment.requirement_index is not None
                else None
            )
            static_assessment = assess_selected_document(
                requirement,
                selected_text,
                specification.annotation_scope,
                support_threshold=span_support_threshold,
                absence_threshold=span_absence_threshold,
            )
            fingerprint = normalize_text(selected_text)
            repeated_before_turn = bool(fingerprint and fingerprint in prior_fingerprints)
            marginal = history.observe(
                static_assessment,
                selected_text,
                document_seen_before=repeated_before_turn,
                record_document=False,
                covered_requirement_ids_before=covered_before_turn,
                commit_coverage=False,
            )
            if marginal.new_requirement_coverage and marginal.requirement_id:
                newly_covered_this_turn.add(marginal.requirement_id)
            binary_target = marginal.binary_gain_target(min_fit_confidence)
            rows.append(
                MarginalUtilityRow(
                    record_id=_record_id(record),
                    question_id_source=_question_id_source(record),
                    question=question,
                    turn_index=turn_index,
                    slot_index=slot_index,
                    slot_hypothesis=slot_hypotheses[slot_index],
                    action_mode=result.action.mode.value,
                    role="tgt" if slot_index == result.action.target_slot else "inc",
                    bound=bound_at_score[slot_index],
                    support_score=float(support_score),
                    novelty_score=document_novelty_score(selected_text, prior_documents),
                    selected_document=selected_text,
                    selected_document_id=selected_document_id,
                    selected_document_id_source=selected_document_id_source,
                    selected_source_id=selected_source_id,
                    evidence_adapter=specification.adapter_name,
                    annotation_version=specification.annotation_version,
                    annotation_scope=specification.annotation_scope.value,
                    utility_basis=marginal.basis.value,
                    requirement_id=marginal.requirement_id,
                    evidence_unit_id=marginal.evidence_unit_id,
                    assignment_similarity=assignment.similarity,
                    assignment_margin=assignment.margin,
                    assignment_reason=assignment.reason,
                    static_support=marginal.static_support,
                    static_assessment_basis=static_assessment.basis.value,
                    assessment_confidence=marginal.confidence,
                    utility_before=marginal.utility_before,
                    utility_after=marginal.utility_after,
                    marginal_utility=marginal.marginal_utility,
                    new_requirement_coverage=marginal.new_requirement_coverage,
                    repeated_document=marginal.repeated_document,
                    fit_eligible=binary_target is not None,
                    rationale=marginal.rationale,
                )
            )
        history.covered_requirement_ids.update(newly_covered_this_turn)
        for text in turn_documents:
            fingerprint = normalize_text(text)
            if fingerprint:
                history.selected_document_fingerprints.add(fingerprint)
            if text and text not in prior_documents:
                prior_documents.append(text)

    question_embedding = belief.labeler.embedder.encode(
        [question], show_progress_bar=False
    )[0]
    return rows, KExample(question_embedding, len(specification.requirements))


def collect_split(
    records: Sequence[Dict[str, Any]],
    belief: ACECBeliefState,
    adapter: EvidenceSpecificationAdapter,
    **replay_kwargs: Any,
) -> Tuple[List[MarginalUtilityExample], List[KExample], List[MarginalUtilityRow], Dict[str, int]]:
    rows: List[MarginalUtilityRow] = []
    k_examples: List[KExample] = []
    skipped = 0
    for record in records:
        if not record.get("docs"):
            skipped += 1
            continue
        try:
            record_rows, k_example = replay_record(
                record, belief, adapter, **replay_kwargs
            )
        except NoUsableEvidenceError:
            skipped += 1
            continue
        rows.extend(record_rows)
        k_examples.append(k_example)
    examples = [example for row in rows if (example := row.fit_example()) is not None]
    return examples, k_examples, rows, {
        "records_seen": len(records),
        "records_usable": len(k_examples),
        "records_skipped": skipped,
    }


def summarize_rows(rows: Sequence[MarginalUtilityRow]) -> Dict[str, Any]:
    eligible = [row for row in rows if row.fit_eligible]
    gains = [row for row in eligible if row.new_requirement_coverage]
    zero_gain = [row for row in eligible if row.new_requirement_coverage is False]
    redundant_support = [
        row
        for row in zero_gain
        if row.static_support == 1.0
    ]
    return {
        "rows": len(rows),
        "fit_eligible_rows": len(eligible),
        "fit_eligible_fraction": float(len(eligible) / len(rows)) if rows else 0.0,
        "unassessable_rows": len(rows) - len(eligible),
        "new_coverage_rows": len(gains),
        "zero_marginal_gain_rows": len(zero_gain),
        "redundant_support_rows": len(redundant_support),
        "repeated_document_rows": sum(row.repeated_document for row in rows),
        "action_mode_counts": dict(sorted(Counter(row.action_mode for row in rows).items())),
        "fit_eligible_action_mode_counts": dict(
            sorted(Counter(row.action_mode for row in eligible).items())
        ),
        "mean_marginal_utility": (
            float(np.mean([row.marginal_utility for row in eligible]))
            if eligible
            else float("nan")
        ),
        "gain_rate": float(len(gains) / len(eligible)) if eligible else float("nan"),
    }


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _join_annotations(
    records: Sequence[Dict[str, Any]], annotations: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotation_id = _nonempty_id(annotation)
        if annotation_id:
            if annotation_id in by_id:
                raise ValueError(f"duplicate annotation id: {annotation_id}")
            by_id[annotation_id] = annotation
        question = _question(annotation)
        if question:
            by_question[question].append(annotation)
    joined = []
    for record in records:
        combined = dict(record)
        native_id = _nonempty_id(record)
        annotation = by_id.get(native_id) if native_id else None
        join_mode = "id" if annotation is not None else "none"
        if annotation is None:
            question_matches = by_question.get(_question(record), [])
            if len(question_matches) > 1:
                raise ValueError(
                    "question-text annotation join is ambiguous for: " + _question(record)
                )
            if question_matches:
                annotation = question_matches[0]
                join_mode = "question"
        if annotation is not None:
            combined["_annotation_record"] = annotation
            annotation_id = _nonempty_id(annotation)
            if annotation_id:
                combined["_canonical_question_id"] = annotation_id
                combined["_question_id_source"] = (
                    "annotation_id_join" if join_mode == "id" else "annotation_question_join"
                )
            elif native_id:
                combined["_canonical_question_id"] = native_id
                combined["_question_id_source"] = "native_record_id"
            else:
                annotation_question = _question(annotation)
                digest = hashlib.sha256(annotation_question.encode("utf-8")).hexdigest()
                combined["_canonical_question_id"] = (
                    f"annotation_question_sha256:{digest}"
                )
                combined["_question_id_source"] = (
                    "annotation_question_sha256_join"
                )
        elif native_id:
            combined["_canonical_question_id"] = native_id
            combined["_question_id_source"] = "native_record_id"
        else:
            question = _question(record)
            digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
            combined["_canonical_question_id"] = f"question_sha256:{digest}"
            combined["_question_id_source"] = "question_sha256"
        combined["_annotation_join_mode"] = join_mode
        joined.append(combined)
    return joined


def _split_by_question_id(
    records: Sequence[Dict[str, Any]],
    validation_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Question-disjoint split that keeps repeated trajectories together."""

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        question_id = _record_id(record)
        if not question_id:
            raise ValueError("record is missing a stable question identity")
        groups[question_id].append(record)
    question_ids = list(groups)
    random.Random(seed).shuffle(question_ids)
    n_test = max(1, int(len(question_ids) * test_frac))
    n_validation = max(1, int(len(question_ids) * validation_frac))
    if n_test + n_validation >= len(question_ids):
        raise ValueError("split configuration leaves no fitting question groups")
    test_ids = set(question_ids[:n_test])
    validation_ids = set(question_ids[n_test : n_test + n_validation])
    fit_ids = set(question_ids[n_test + n_validation :])
    if test_ids & validation_ids or test_ids & fit_ids or validation_ids & fit_ids:
        raise AssertionError("question-id split overlap")

    def flatten(ids: set[str]) -> List[Dict[str, Any]]:
        return [
            record
            for question_id in question_ids
            if question_id in ids
            for record in groups[question_id]
        ]

    return flatten(fit_ids), flatten(validation_ids), flatten(test_ids)


def summarize_provenance(
    records: Sequence[Dict[str, Any]], rows: Sequence[MarginalUtilityRow]
) -> Dict[str, Any]:
    question_sources = Counter(_question_id_source(record) for record in records)
    join_modes = Counter(str(record.get("_annotation_join_mode") or "none") for record in records)
    document_sources = Counter(row.selected_document_id_source for row in rows)
    dataset_id_sources = {
        "native_record_id",
        "annotation_id_join",
        "annotation_question_join",
        "annotation_question_sha256_join",
    }
    corpus_id_sources = {"native", "annotation_context"}
    return {
        "records": len(records),
        "unique_question_ids": len({_record_id(record) for record in records}),
        "question_id_source_counts": dict(sorted(question_sources.items())),
        "annotation_join_mode_counts": dict(sorted(join_modes.items())),
        "dataset_question_id_fraction": (
            sum(count for source, count in question_sources.items() if source in dataset_id_sources)
            / len(records)
            if records
            else 0.0
        ),
        "selected_document_rows": len(rows),
        "selected_document_id_source_counts": dict(sorted(document_sources.items())),
        "corpus_document_id_fraction": (
            sum(count for source, count in document_sources.items() if source in corpus_id_sources)
            / len(rows)
            if rows
            else 0.0
        ),
    }


def evaluate_validity_gate(
    validation_examples: Sequence[MarginalUtilityExample],
    validation_rows: Sequence[MarginalUtilityRow],
    validation_metrics: Dict[str, Any],
    action_counts: Dict[str, Dict[str, Any]],
    provenance: Dict[str, Any],
    binding_threshold: Optional[float],
    required_action_modes: Sequence[str],
    *,
    min_validation_examples: int,
    min_validation_fit_eligible_fraction: float,
    min_validation_posterior_auc: float,
    min_validation_average_precision: float,
    min_validation_posterior_gain_over_novelty: float,
    min_validation_nonrepeat_examples: int,
    min_validation_nonrepeat_raw_support_auc: float,
    min_fit_target_examples_per_action: int,
    min_dataset_question_id_fraction: float,
    min_corpus_document_id_fraction: float,
) -> Dict[str, Any]:
    """Evaluate evidence validity, not merely artifact build success."""

    checks: Dict[str, Dict[str, Any]] = {}

    def minimum_check(name: str, value: Any, minimum: float) -> None:
        numeric = float(value) if value is not None else float("nan")
        checks[name] = {
            "value": numeric,
            "minimum": minimum,
            "pass": bool(np.isfinite(numeric) and numeric >= minimum),
        }

    minimum_check("validation_examples", len(validation_examples), min_validation_examples)
    eligible_fraction = (
        len(validation_examples) / len(validation_rows) if validation_rows else 0.0
    )
    minimum_check(
        "validation_fit_eligible_fraction",
        eligible_fraction,
        min_validation_fit_eligible_fraction,
    )
    minimum_check(
        "posterior_auc",
        validation_metrics.get("posterior_auc"),
        min_validation_posterior_auc,
    )
    minimum_check(
        "average_precision",
        validation_metrics.get("average_precision"),
        min_validation_average_precision,
    )
    minimum_check(
        "posterior_auc_gain_over_novelty",
        validation_metrics.get("posterior_auc_gain_over_novelty"),
        min_validation_posterior_gain_over_novelty,
    )
    nonrepeat = dict(validation_metrics.get("nonrepeat") or {})
    minimum_check(
        "validation_nonrepeat_examples",
        nonrepeat.get("n", 0),
        min_validation_nonrepeat_examples,
    )
    minimum_check(
        "validation_nonrepeat_raw_support_auc",
        nonrepeat.get("raw_support_auc"),
        min_validation_nonrepeat_raw_support_auc,
    )
    for mode in required_action_modes:
        minimum_check(
            f"fit_target_examples_{mode}",
            (action_counts.get(mode) or {}).get("target_examples", 0),
            min_fit_target_examples_per_action,
        )
    minimum_check(
        "dataset_question_id_fraction",
        provenance.get("dataset_question_id_fraction"),
        min_dataset_question_id_fraction,
    )
    minimum_check(
        "corpus_document_id_fraction",
        provenance.get("corpus_document_id_fraction"),
        min_corpus_document_id_fraction,
    )
    checks["binding_operating_point"] = {
        "value": binding_threshold,
        "required": True,
        "pass": binding_threshold is not None and np.isfinite(binding_threshold),
    }
    failed = [name for name, check in checks.items() if not check["pass"]]
    return {"pass": not failed, "failed_checks": failed, "checks": checks}


def _write_audit(path: str, split_rows: Sequence[Tuple[str, Sequence[MarginalUtilityRow]]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for split, rows in split_rows:
            for row in rows:
                dump_strict_json(
                    {"split": split, **asdict(row)}, handle, ensure_ascii=False
                )
                handle.write("\n")


def _adapter(name: str) -> EvidenceSpecificationAdapter:
    if name == "hotpotqa":
        return HotpotQAEvidenceAdapterV5()
    if name == "canonical":
        return CanonicalEvidenceAdapterV5()
    raise ValueError(f"unknown v5 evidence adapter: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument(
        "--annotations",
        default=None,
        help="Optional question-level annotation JSONL joined by id or question.",
    )
    parser.add_argument("--evidence_adapter", choices=("hotpotqa", "canonical"), default="hotpotqa")
    parser.add_argument("--e5_model_path", required=True)
    parser.add_argument("--nli_model", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--k_mode", choices=("auto", "fixed", "uniform", "predictor"), default="auto")
    parser.add_argument("--fixed_k", type=int, default=2)
    parser.add_argument("--k_neighbors", type=int, default=15)
    parser.add_argument("--tau_new", type=float, default=ACECConfig().tau_new)
    parser.add_argument("--assignment_min_similarity", type=float, default=0.30)
    parser.add_argument("--assignment_min_margin", type=float, default=0.05)
    parser.add_argument("--span_support_threshold", type=float, default=0.90)
    parser.add_argument("--span_absence_threshold", type=float, default=0.35)
    parser.add_argument("--min_fit_confidence", type=float, default=0.90)
    parser.add_argument("--validation_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_validation_examples", type=int, default=50)
    parser.add_argument("--min_validation_fit_eligible_fraction", type=float, default=0.15)
    parser.add_argument("--min_validation_posterior_auc", type=float, default=0.75)
    parser.add_argument("--min_validation_average_precision", type=float, default=0.30)
    parser.add_argument(
        "--min_validation_posterior_gain_over_novelty", type=float, default=0.03
    )
    parser.add_argument("--min_validation_nonrepeat_examples", type=int, default=20)
    parser.add_argument(
        "--min_validation_nonrepeat_raw_support_auc", type=float, default=0.70
    )
    parser.add_argument("--min_fit_target_examples_per_action", type=int, default=10)
    parser.add_argument(
        "--required_action_modes", default="EXPAND,REWRITE,DECOMPOSE"
    )
    parser.add_argument("--min_dataset_question_id_fraction", type=float, default=1.0)
    parser.add_argument("--min_corpus_document_id_fraction", type=float, default=0.95)
    parser.add_argument("--min_binding_precision", type=float, default=0.80)
    parser.add_argument("--min_binding_predictions", type=int, default=5)
    args = parser.parse_args()

    if args.validation_frac <= 0 or args.test_frac <= 0:
        raise ValueError("validation_frac and test_frac must be positive")
    if args.validation_frac + args.test_frac >= 1:
        raise ValueError("validation_frac + test_frac must be < 1")
    for name in (
        "assignment_min_similarity",
        "assignment_min_margin",
        "span_support_threshold",
        "span_absence_threshold",
        "min_fit_confidence",
        "min_validation_fit_eligible_fraction",
        "min_validation_posterior_auc",
        "min_validation_average_precision",
        "min_validation_posterior_gain_over_novelty",
        "min_validation_nonrepeat_raw_support_auc",
        "min_dataset_question_id_fraction",
        "min_corpus_document_id_fraction",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    if args.span_absence_threshold >= args.span_support_threshold:
        raise ValueError("span_absence_threshold must be below span_support_threshold")

    required_action_modes = tuple(
        mode.strip() for mode in args.required_action_modes.split(",") if mode.strip()
    )
    unknown_action_modes = set(required_action_modes) - {"EXPAND", "REWRITE", "DECOMPOSE"}
    if unknown_action_modes:
        raise ValueError(f"unknown required action modes: {sorted(unknown_action_modes)}")

    records = _load_jsonl(args.records)
    annotations = _load_jsonl(args.annotations) if args.annotations else []
    records = _join_annotations(records, annotations)
    fit_records, validation_records, test_records = _split_by_question_id(
        records, args.validation_frac, args.test_frac, args.seed
    )

    from belief.obs_extractor import E5Embedder

    embedder = E5QueryEmbedder(E5Embedder(args.e5_model_path))
    nli_scorer = (
        CrossEncoderNLIScorer(args.nli_model)
        if args.nli_model
        else CosineNLIScorer(embedder)
    )
    belief = ACECBeliefState(
        embedder,
        nli_scorer,
        config=ACECConfig(k_max=args.k_max, tau_new=args.tau_new),
    )
    adapter = _adapter(args.evidence_adapter)
    replay_kwargs = {
        "min_fit_confidence": args.min_fit_confidence,
        "assignment_min_similarity": args.assignment_min_similarity,
        "assignment_min_margin": args.assignment_min_margin,
        "span_support_threshold": args.span_support_threshold,
        "span_absence_threshold": args.span_absence_threshold,
    }
    fit_examples, fit_k, fit_rows, fit_record_stats = collect_split(
        fit_records, belief, adapter, **replay_kwargs
    )
    validation_examples, validation_k, validation_rows, validation_record_stats = collect_split(
        validation_records, belief, adapter, **replay_kwargs
    )
    test_examples, test_k, test_rows, test_record_stats = collect_split(
        test_records, belief, adapter, **replay_kwargs
    )

    observation_model, gain_rates, action_counts = fit_observation_model_v5(fit_examples)
    validation_probabilities = predict_examples_v5(
        validation_examples, observation_model, gain_rates
    )
    binding_threshold, threshold_selection = choose_binding_threshold_v5(
        validation_examples,
        validation_probabilities,
        min_precision=args.min_binding_precision,
        min_predicted_gain=args.min_binding_predictions,
    )
    validation_metrics = posterior_quality_metrics_v5(
        validation_examples,
        observation_model,
        gain_rates,
        threshold=binding_threshold,
    )
    test_metrics = posterior_quality_metrics_v5(
        test_examples,
        observation_model,
        gain_rates,
        threshold=binding_threshold,
    )
    k_strategy = select_k_strategy_v5(
        args.k_mode,
        fit_k,
        validation_k,
        args.k_max,
        fixed_k=args.fixed_k,
        neighbors=args.k_neighbors,
    )

    all_rows = [*fit_rows, *validation_rows, *test_rows]
    provenance = summarize_provenance(records, all_rows)
    gate = evaluate_validity_gate(
        validation_examples,
        validation_rows,
        validation_metrics,
        action_counts,
        provenance,
        binding_threshold,
        required_action_modes,
        min_validation_examples=args.min_validation_examples,
        min_validation_fit_eligible_fraction=args.min_validation_fit_eligible_fraction,
        min_validation_posterior_auc=args.min_validation_posterior_auc,
        min_validation_average_precision=args.min_validation_average_precision,
        min_validation_posterior_gain_over_novelty=(
            args.min_validation_posterior_gain_over_novelty
        ),
        min_validation_nonrepeat_examples=args.min_validation_nonrepeat_examples,
        min_validation_nonrepeat_raw_support_auc=(
            args.min_validation_nonrepeat_raw_support_auc
        ),
        min_fit_target_examples_per_action=args.min_fit_target_examples_per_action,
        min_dataset_question_id_fraction=args.min_dataset_question_id_fraction,
        min_corpus_document_id_fraction=args.min_corpus_document_id_fraction,
    )
    gate_pass = bool(gate["pass"])
    gate_reasons = list(gate["failed_checks"])

    metrics = {
        "fit": {
            **fit_record_stats,
            "examples": len(fit_examples),
            "k_examples": len(fit_k),
            "marginal_utility": summarize_rows(fit_rows),
        },
        "validation": {
            **validation_record_stats,
            **validation_metrics,
            "marginal_utility": summarize_rows(validation_rows),
        },
        "test": {
            **test_record_stats,
            **test_metrics,
            "marginal_utility": summarize_rows(test_rows),
        },
        "target_action_counts": action_counts,
        "provenance": provenance,
        "binding_threshold_selection": threshold_selection,
        "k_strategy": k_strategy.selection_metrics,
        "gate": gate,
    }
    metadata = {
        "evidence_adapter": adapter.name,
        "records_path": os.path.abspath(args.records),
        "annotations_path": os.path.abspath(args.annotations) if args.annotations else None,
        "e5_model_path": args.e5_model_path,
        "nli_model": args.nli_model or "cosine-sanity-only",
        "observation_model_kind": MODEL_KIND,
        "supervision_standard": standard_payload(),
        "target_is_conditional_marginal_utility": True,
        "utility_basis": "requirement_coverage",
        "source_ids_used_as_targets": False,
        "k_definition": "number_of_minimal_evidence_requirements",
        "tau_new": args.tau_new,
        "assignment_min_similarity": args.assignment_min_similarity,
        "assignment_min_margin": args.assignment_min_margin,
        "span_support_threshold": args.span_support_threshold,
        "span_absence_threshold": args.span_absence_threshold,
        "min_fit_confidence": args.min_fit_confidence,
        "seed": args.seed,
        "validation_frac": args.validation_frac,
        "test_frac": args.test_frac,
        "validity_gate_config": {
            "min_validation_examples": args.min_validation_examples,
            "min_validation_fit_eligible_fraction": (
                args.min_validation_fit_eligible_fraction
            ),
            "min_validation_posterior_auc": args.min_validation_posterior_auc,
            "min_validation_average_precision": args.min_validation_average_precision,
            "min_validation_posterior_gain_over_novelty": (
                args.min_validation_posterior_gain_over_novelty
            ),
            "min_validation_nonrepeat_examples": (
                args.min_validation_nonrepeat_examples
            ),
            "min_validation_nonrepeat_raw_support_auc": (
                args.min_validation_nonrepeat_raw_support_auc
            ),
            "min_fit_target_examples_per_action": (
                args.min_fit_target_examples_per_action
            ),
            "required_action_modes": list(required_action_modes),
            "min_dataset_question_id_fraction": (
                args.min_dataset_question_id_fraction
            ),
            "min_corpus_document_id_fraction": (
                args.min_corpus_document_id_fraction
            ),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    audit_path = os.path.join(args.out_dir, "acec_calibration_v5_marginal_audit.jsonl")
    metrics_path = os.path.join(args.out_dir, "acec_calibration_v5_metrics.json")
    _write_audit(
        audit_path,
        (("fit", fit_rows), ("validation", validation_rows), ("test", test_rows)),
    )
    with open(metrics_path, "w", encoding="utf-8") as handle:
        dump_strict_json({"metadata": metadata, "metrics": metrics}, handle, indent=2)

    print("=== ACEC calibration v5: conditional marginal evidence utility ===")
    print(json.dumps(strict_json_value(metrics), indent=2, allow_nan=False))
    print(f"Wrote marginal audit {audit_path}")
    print(f"Wrote metrics {metrics_path}")
    if not gate_pass:
        raise RuntimeError(
            "ACEC v5 calibration gate failed; artifact was not written: "
            + ", ".join(gate_reasons)
        )

    artifact_path = os.path.join(args.out_dir, "acec_calibration_v5.json")
    save_calibration_artifact_v5(
        artifact_path,
        observation_model,
        gain_rates,
        k_strategy,
        float(binding_threshold),
        metadata,
        metrics,
    )
    print(f"Wrote {artifact_path}")


if __name__ == "__main__":
    main()
