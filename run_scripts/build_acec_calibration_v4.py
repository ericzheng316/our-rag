"""Build ACEC calibration-v4 with runtime-selected evidence labels.

V4 keeps the v3 monotonic calibrator and split/gating strategy, but changes
the supervised event from a batch-level title hit to the exact document that
produced the runtime max-NLI score for a slot.  It also writes a compact label
audit so the score/document/label triples can be inspected before training.
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
RUN_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "rag" / "src"))
sys.path.insert(0, str(RUN_SCRIPTS))

from build_acec_calibration_v3 import (  # noqa: E402
    evaluate_k_accuracy,
    match_slots_to_sf,
    normalize_title,
    select_k_strategy,
)
from belief.acec import (  # noqa: E402
    ACECBeliefState,
    ACECConfig,
    CosineNLIScorer,
    CrossEncoderNLIScorer,
    E5QueryEmbedder,
    GoldEvidenceAdapter,
    doc_title,
    get_adapter,
)
from belief.acec.calibration_v4 import (  # noqa: E402
    LABEL_SCHEMA,
    MODEL_KIND,
    fit_observation_model_v4,
    k_predictor_payload,
    posterior_quality_metrics,
    save_calibration_artifact_v4,
)
from belief.acec.offline_fit import EmpiricalKPredictor, HitExample, KExample  # noqa: E402


@dataclass
class CalibrationRow:
    """Auditable score/document/label tuple for one runtime slot update."""

    record_id: str
    question: str
    turn_index: int
    slot_index: int
    slot_hypothesis: str
    action_mode: str
    role: str
    bound: bool
    nli_score: float
    selected_doc_title: str
    assigned_sf_title: Optional[str]
    label_status: str  # "positive" | "negative" | "unknown"
    label_reason: str

    def hit_example(self) -> Optional[HitExample]:
        if self.label_status == "unknown":
            return None
        return HitExample(
            action_mode=self.action_mode,
            role=self.role,
            bound=self.bound,
            nli_score=self.nli_score,
            is_hit=self.label_status == "positive",
        )


def _record_id(record: Dict[str, Any]) -> str:
    value = record.get("id", record.get("_id", ""))
    return str(value) if value is not None else ""


def _selected_title(selected_doc: Any) -> str:
    if isinstance(selected_doc, dict):
        return str(doc_title(selected_doc))
    if isinstance(selected_doc, str):
        return str(doc_title({"contents": selected_doc}))
    raise TypeError(f"unsupported selected evidence type: {type(selected_doc).__name__}")


def replay_record(
    record: Dict[str, Any],
    belief: ACECBeliefState,
    adapter: GoldEvidenceAdapter,
    unmatched_slot_policy: str = "unknown",
    slot_match_threshold: float = 0.30,
) -> Tuple[List[CalibrationRow], KExample]:
    """Replay one trajectory and label the exact argmax evidence per slot.

    Matched slots use HotpotQA title supervision: the selected document is a
    positive only when its title matches the supporting-fact title assigned to
    that slot.  Unmatched slots are unknown by default because official gold
    supporting facts need not exhaust every potentially useful document.  A
    distractor-only calibration run may opt into treating them as negatives.
    """

    if unmatched_slot_policy not in {"unknown", "negative"}:
        raise ValueError("unmatched_slot_policy must be 'unknown' or 'negative'")

    sf_titles = adapter.gold_titles(record)
    question = record["problem"]
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
            index: slot.bound
            for index, slot in enumerate(belief.coverage_belief.slots)
        }
        result = belief.turn(
            query=query,
            new_docs=[{"contents": doc} for doc in docs],
            is_answer=False,
        )
        bound_at_score = {
            slot_index: bound_before.get(slot_index, False)
            for slot_index in result.slot_scores
        }
        per_turn.append((turn_index, result, bound_at_score))

    slot_hypotheses = [slot.hypothesis for slot in belief.coverage_belief.slots]
    slot_to_sf = match_slots_to_sf(
        slot_hypotheses,
        sf_titles,
        belief.labeler.embedder,
        threshold=slot_match_threshold,
    )

    rows: List[CalibrationRow] = []
    for turn_index, result, bound_at_score in per_turn:
        selected_docs = getattr(result, "slot_best_docs", None)
        if selected_docs is None:
            raise RuntimeError(
                "calibration v4 requires TurnResult.slot_best_docs; "
                "the score/document argmax contract is unavailable"
            )
        for slot_index, nli_score in result.slot_scores.items():
            if slot_index not in selected_docs:
                raise RuntimeError(
                    f"slot {slot_index} has a max-NLI score but no selected evidence"
                )
            selected_doc_title = _selected_title(selected_docs[slot_index])
            sf_title = slot_to_sf.get(slot_index)
            if sf_title is None:
                if unmatched_slot_policy == "negative":
                    label_status = "negative"
                    label_reason = "unmatched_slot_assumed_negative"
                else:
                    label_status = "unknown"
                    label_reason = "unmatched_slot"
            elif normalize_title(selected_doc_title) == normalize_title(sf_title):
                label_status = "positive"
                label_reason = "selected_title_matches_assigned_sf"
            else:
                label_status = "negative"
                label_reason = "selected_title_differs_from_assigned_sf"

            rows.append(
                CalibrationRow(
                    record_id=_record_id(record),
                    question=question,
                    turn_index=turn_index,
                    slot_index=slot_index,
                    slot_hypothesis=slot_hypotheses[slot_index],
                    action_mode=result.action.mode.value,
                    role="tgt" if slot_index == result.action.target_slot else "inc",
                    bound=bound_at_score[slot_index],
                    nli_score=float(nli_score),
                    selected_doc_title=selected_doc_title,
                    assigned_sf_title=sf_title,
                    label_status=label_status,
                    label_reason=label_reason,
                )
            )

    question_embedding = belief.labeler.embedder.encode(
        [question], show_progress_bar=False
    )[0]
    return rows, KExample(question_embedding, len(sf_titles))


def collect_split(
    records: Sequence[Dict[str, Any]],
    belief: ACECBeliefState,
    adapter: GoldEvidenceAdapter,
    unmatched_slot_policy: str,
    slot_match_threshold: float,
) -> Tuple[List[HitExample], List[KExample], List[CalibrationRow]]:
    rows: List[CalibrationRow] = []
    k_examples: List[KExample] = []
    for record in records:
        if not adapter.gold_titles(record) or not record.get("docs"):
            continue
        record_rows, k_example = replay_record(
            record,
            belief,
            adapter,
            unmatched_slot_policy=unmatched_slot_policy,
            slot_match_threshold=slot_match_threshold,
        )
        rows.extend(record_rows)
        k_examples.append(k_example)
    hit_examples = [example for row in rows if (example := row.hit_example()) is not None]
    return hit_examples, k_examples, rows


def summarize_rows(rows: Sequence[CalibrationRow]) -> Dict[str, Any]:
    by_status = {status: 0 for status in ("positive", "negative", "unknown")}
    by_reason: Dict[str, int] = {}
    for row in rows:
        by_status[row.label_status] = by_status.get(row.label_status, 0) + 1
        by_reason[row.label_reason] = by_reason.get(row.label_reason, 0) + 1
    labeled = by_status["positive"] + by_status["negative"]
    return {
        "rows": len(rows),
        "labeled_rows": labeled,
        "positive_rows": by_status["positive"],
        "negative_rows": by_status["negative"],
        "unknown_rows": by_status["unknown"],
        "positive_rate": (
            float(by_status["positive"] / labeled) if labeled else float("nan")
        ),
        "by_reason": by_reason,
    }


def _write_audit(
    path: str,
    split_rows: Sequence[Tuple[str, Sequence[CalibrationRow]]],
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for split, rows in split_rows:
            for row in rows:
                payload = {"split": split, **asdict(row)}
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")


def main() -> None:
    from belief.obs_extractor import E5Embedder

    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--e5_model_path", required=True)
    parser.add_argument("--nli_model", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--k_mode", choices=("auto", "fixed", "predictor"), default="auto")
    parser.add_argument("--fixed_k", type=int, default=2)
    parser.add_argument("--k_neighbors", type=int, default=15)
    parser.add_argument("--tau_new", type=float, default=ACECConfig().tau_new)
    parser.add_argument("--slot_match_threshold", type=float, default=0.30)
    parser.add_argument(
        "--unmatched_slot_policy",
        choices=("unknown", "negative"),
        default="unknown",
        help="Unknown excludes unmatched slots from fitting. Use negative only "
        "when the logged candidate pool is known to contain annotated distractors.",
    )
    parser.add_argument("--validation_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_validation_labeled_examples", type=int, default=20)
    parser.add_argument("--min_validation_raw_auc", type=float, default=0.75)
    parser.add_argument("--min_validation_posterior_auc", type=float, default=0.75)
    parser.add_argument("--max_validation_auc_drop", type=float, default=0.03)
    args = parser.parse_args()

    if args.validation_frac <= 0 or args.test_frac <= 0:
        raise ValueError("validation_frac and test_frac must both be positive")
    if args.validation_frac + args.test_frac >= 1:
        raise ValueError("validation_frac + test_frac must be < 1")
    if not 1 <= args.fixed_k <= args.k_max:
        raise ValueError("fixed_k must be within [1, k_max]")
    if not 0.0 <= args.slot_match_threshold <= 1.0:
        raise ValueError("slot_match_threshold must be within [0, 1]")

    adapter = get_adapter(args.dataset)
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

    with open(args.records, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    random.Random(args.seed).shuffle(records)

    n_test = max(1, int(len(records) * args.test_frac))
    n_validation = max(1, int(len(records) * args.validation_frac))
    test_records = records[:n_test]
    validation_records = records[n_test : n_test + n_validation]
    fit_records = records[n_test + n_validation :]
    if not fit_records:
        raise ValueError("split configuration leaves no fitting records")

    fit_hits, fit_k, fit_rows = collect_split(
        fit_records,
        belief,
        adapter,
        args.unmatched_slot_policy,
        args.slot_match_threshold,
    )
    validation_hits, validation_k, validation_rows = collect_split(
        validation_records,
        belief,
        adapter,
        args.unmatched_slot_policy,
        args.slot_match_threshold,
    )
    test_hits, test_k, test_rows = collect_split(
        test_records,
        belief,
        adapter,
        args.unmatched_slot_policy,
        args.slot_match_threshold,
    )

    observation_model, hit_rates, action_counts = fit_observation_model_v4(fit_hits)
    validation_metrics = posterior_quality_metrics(
        validation_hits, observation_model, hit_rates
    )
    test_metrics = posterior_quality_metrics(test_hits, observation_model, hit_rates)

    k_mode, fixed_k, k_predictor, k_payload = select_k_strategy(
        args.k_mode, fit_k, args.k_max, args.fixed_k, args.k_neighbors
    )
    validation_k_accuracy = evaluate_k_accuracy(
        validation_k, k_mode, fixed_k, k_predictor, args.k_max
    )
    test_k_accuracy = evaluate_k_accuracy(
        test_k, k_mode, fixed_k, k_predictor, args.k_max
    )

    raw_auc = float(validation_metrics["raw_nli_auc"])
    posterior_auc = float(validation_metrics["posterior_auc"])
    gate_reasons = []
    if len(validation_hits) < args.min_validation_labeled_examples:
        gate_reasons.append("validation_labeled_examples")
    if not np.isfinite(raw_auc) or raw_auc < args.min_validation_raw_auc:
        gate_reasons.append("raw_nli_auc")
    if not np.isfinite(posterior_auc) or posterior_auc < args.min_validation_posterior_auc:
        gate_reasons.append("posterior_auc")
    if np.isfinite(raw_auc) and np.isfinite(posterior_auc):
        if raw_auc - posterior_auc > args.max_validation_auc_drop:
            gate_reasons.append("posterior_auc_drop")
    gate_pass = not gate_reasons

    metrics = {
        "fit": {
            "records": len(fit_records),
            "hit_examples": len(fit_hits),
            "k_examples": len(fit_k),
            "labels": summarize_rows(fit_rows),
        },
        "validation": {
            **validation_metrics,
            "k_accuracy": validation_k_accuracy,
            "labels": summarize_rows(validation_rows),
        },
        "test": {
            **test_metrics,
            "k_accuracy": test_k_accuracy,
            "labels": summarize_rows(test_rows),
        },
        "target_action_counts": action_counts,
        "gate": {"pass": gate_pass, "failed_checks": gate_reasons},
    }
    metadata = {
        "dataset": args.dataset,
        "records_path": os.path.abspath(args.records),
        "e5_model_path": args.e5_model_path,
        "nli_model": args.nli_model or "cosine-sanity-only",
        "observation_model_kind": MODEL_KIND,
        "label_schema": LABEL_SCHEMA,
        "label_target": "selected_max_nli_document_supports_assigned_slot",
        "unmatched_slot_policy": args.unmatched_slot_policy,
        "slot_match_threshold": args.slot_match_threshold,
        "k_max": args.k_max,
        "recommended_k_mode": k_mode,
        "fixed_k": fixed_k if k_mode == "fixed" else None,
        "tau_new": args.tau_new,
        "seed": args.seed,
        "validation_frac": args.validation_frac,
        "test_frac": args.test_frac,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    audit_path = os.path.join(args.out_dir, "acec_calibration_v4_label_audit.jsonl")
    metrics_path = os.path.join(args.out_dir, "acec_calibration_v4_metrics.json")
    _write_audit(
        audit_path,
        (
            ("fit", fit_rows),
            ("validation", validation_rows),
            ("test", test_rows),
        ),
    )
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "metrics": metrics}, handle, indent=2)

    print("=== ACEC calibration v4 ===")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote label audit {audit_path}")
    print(f"Wrote metrics {metrics_path}")
    if not gate_pass:
        raise RuntimeError(
            "ACEC calibration gate failed; artifact was not written: "
            + ", ".join(gate_reasons)
        )

    artifact_path = os.path.join(args.out_dir, "acec_calibration_v4.json")
    save_calibration_artifact_v4(
        artifact_path,
        observation_model,
        hit_rates,
        k_payload,
        metadata,
        metrics,
    )
    print(f"Wrote {artifact_path}")


if __name__ == "__main__":
    main()
