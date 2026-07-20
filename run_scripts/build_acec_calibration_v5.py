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
import json
import os
import random
import sys
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


def _record_id(record: Dict[str, Any]) -> str:
    value = record.get("id", record.get("_id", ""))
    return str(value) if value is not None else ""


def _question(record: Dict[str, Any]) -> str:
    value = record.get("problem", record.get("question", ""))
    return str(value)


def _selected_document(selected_doc: Any) -> Tuple[str, str]:
    if isinstance(selected_doc, str):
        text = selected_doc
        source_id = text.split(":", 1)[0].strip() if ":" in text else ""
        return text, source_id
    if isinstance(selected_doc, dict):
        text = str(selected_doc.get("contents", selected_doc.get("text", "")))
        source_id = str(
            selected_doc.get(
                "title",
                selected_doc.get("source_id", text.split(":", 1)[0].strip() if ":" in text else ""),
            )
        )
        return text, source_id
    raise TypeError(f"unsupported selected evidence type: {type(selected_doc).__name__}")


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
            new_docs=[{"contents": doc} if isinstance(doc, str) else doc for doc in docs],
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
            selected_text, selected_source_id = _selected_document(selected_docs[slot_index])
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
        "unassessable_rows": len(rows) - len(eligible),
        "new_coverage_rows": len(gains),
        "zero_marginal_gain_rows": len(zero_gain),
        "redundant_support_rows": len(redundant_support),
        "repeated_document_rows": sum(row.repeated_document for row in rows),
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
    by_question: Dict[str, Dict[str, Any]] = {}
    for annotation in annotations:
        annotation_id = str(annotation.get("id", annotation.get("_id", "")))
        if annotation_id:
            by_id[annotation_id] = annotation
        question = str(annotation.get("question", annotation.get("problem", "")))
        if question:
            by_question[question] = annotation
    joined = []
    for record in records:
        annotation = by_id.get(_record_id(record)) or by_question.get(_question(record))
        combined = dict(record)
        if annotation is not None:
            combined["_annotation_record"] = annotation
        joined.append(combined)
    return joined


def _write_audit(path: str, split_rows: Sequence[Tuple[str, Sequence[MarginalUtilityRow]]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for split, rows in split_rows:
            for row in rows:
                json.dump({"split": split, **asdict(row)}, handle, ensure_ascii=False)
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
    parser.add_argument("--min_validation_examples", type=int, default=20)
    parser.add_argument("--min_validation_posterior_auc", type=float, default=0.75)
    parser.add_argument("--min_validation_average_precision", type=float, default=0.30)
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
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    if args.span_absence_threshold >= args.span_support_threshold:
        raise ValueError("span_absence_threshold must be below span_support_threshold")

    records = _load_jsonl(args.records)
    if args.annotations:
        records = _join_annotations(records, _load_jsonl(args.annotations))
    random.Random(args.seed).shuffle(records)
    n_test = max(1, int(len(records) * args.test_frac))
    n_validation = max(1, int(len(records) * args.validation_frac))
    test_records = records[:n_test]
    validation_records = records[n_test : n_test + n_validation]
    fit_records = records[n_test + n_validation :]
    if not fit_records:
        raise ValueError("split configuration leaves no fitting records")

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

    gate_reasons = []
    if len(validation_examples) < args.min_validation_examples:
        gate_reasons.append("validation_examples")
    posterior_auc = float(validation_metrics["posterior_auc"])
    average_precision = float(validation_metrics["average_precision"])
    if not np.isfinite(posterior_auc) or posterior_auc < args.min_validation_posterior_auc:
        gate_reasons.append("posterior_auc")
    if not np.isfinite(average_precision) or average_precision < args.min_validation_average_precision:
        gate_reasons.append("average_precision")
    if binding_threshold is None:
        gate_reasons.append("binding_operating_point")
    gate_pass = not gate_reasons

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
        "binding_threshold_selection": threshold_selection,
        "k_strategy": k_strategy.selection_metrics,
        "gate": {"pass": gate_pass, "failed_checks": gate_reasons},
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
    }

    os.makedirs(args.out_dir, exist_ok=True)
    audit_path = os.path.join(args.out_dir, "acec_calibration_v5_marginal_audit.jsonl")
    metrics_path = os.path.join(args.out_dir, "acec_calibration_v5_metrics.json")
    _write_audit(
        audit_path,
        (("fit", fit_rows), ("validation", validation_rows), ("test", test_rows)),
    )
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "metrics": metrics}, handle, indent=2)

    print("=== ACEC calibration v5: conditional marginal evidence utility ===")
    print(json.dumps(metrics, indent=2))
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
