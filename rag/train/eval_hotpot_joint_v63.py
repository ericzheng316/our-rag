"""Fixed-trace ACEC-vs-NLI evidence selection and official HotpotQA scoring.

Headline metrics are produced by an unchanged copy of HotpotQA's official
``hotpot_evaluate_v1.py`` passed with ``--official_evaluator``.  The metric
implementation in this module is explicitly a diagnostic reference used for
parity checks, failure buckets, and paired bootstrap deltas.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.evidence_contract_v63 import (  # noqa: E402
    EvidenceTraceV63,
    EvidenceUnitV63,
    native_hotpot_pairs,
)
from belief.acec.evidence_selector_minimal_v63 import (  # noqa: E402
    ACEC_ACCEPT_SCORE_KEY,
    ANSWER_HYPOTHESIS_VERSION,
    RAW_NLI_SCORE_KEY,
    SELECTOR_VERSION,
    SharedStoppingRuleV63,
    answer_hypothesis,
    select_acec_existing_v63,
    select_generic_nli_v63,
)
from short_answer_contract_v63 import ANSWER_CONTRACT_VERSION  # noqa: E402


REFERENCE_METRIC_VERSION = "hotpot_official_reference_v63.1"
TRACE_REQUIRED_METADATA = (
    "acec_artifact_sha256",
    "acec_artifact_version",
    "acec_binding_threshold",
    "acec_k_mode",
    "acec_replay_version",
    "provenance_adapter",
    "provenance_adapter_version",
    "trace_generator",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(ids))).encode("utf-8")).hexdigest()


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction: str, gold: str) -> Dict[str, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    em = float(normalized_prediction == normalized_gold)
    special = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special or normalized_gold in special
    ) and normalized_prediction != normalized_gold:
        return {"em": em, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return {"em": em, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return {"em": em, "f1": f1, "prec": precision, "recall": recall}


def supporting_fact_scores(
    prediction: Iterable[Tuple[str, int]], gold: Iterable[Tuple[str, int]]
) -> Dict[str, float]:
    prediction_set = set(prediction)
    gold_set = set(gold)
    true_positives = len(prediction_set & gold_set)
    false_positives = len(prediction_set - gold_set)
    false_negatives = len(gold_set - prediction_set)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sp_em": float(prediction_set == gold_set),
        "sp_f1": f1,
        "sp_prec": precision,
        "sp_recall": recall,
    }


def reference_question_metrics(
    answer_prediction: str,
    sp_prediction: Iterable[Tuple[str, int]],
    gold_answer: str,
    gold_sp: Iterable[Tuple[str, int]],
) -> Dict[str, float]:
    answer = answer_scores(answer_prediction, gold_answer)
    support = supporting_fact_scores(sp_prediction, gold_sp)
    joint_precision = answer["prec"] * support["sp_prec"]
    joint_recall = answer["recall"] * support["sp_recall"]
    joint_f1 = (
        2.0 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall
        else 0.0
    )
    return {
        **answer,
        **support,
        "joint_em": answer["em"] * support["sp_em"],
        "joint_f1": joint_f1,
        "joint_prec": joint_precision,
        "joint_recall": joint_recall,
    }


def aggregate_reference_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
    }


def export_official_submission(
    answers: Mapping[str, str],
    supporting_facts: Mapping[str, Iterable[Tuple[str, int]]],
) -> Dict[str, Any]:
    if set(answers) != set(supporting_facts):
        raise ValueError("official answer and sp prediction ids must match")
    return {
        "answer": {qid: str(answers[qid]) for qid in sorted(answers)},
        "sp": {
            qid: [
                [title, sent_id]
                for title, sent_id in sorted(
                    {(str(title), int(sent_id)) for title, sent_id in supporting_facts[qid]},
                    key=lambda value: (value[0], value[1]),
                )
            ]
            for qid in sorted(supporting_facts)
        },
    }


def run_official_evaluator(
    evaluator_path: Path, prediction_path: Path, gold_path: Path
) -> Dict[str, float]:
    command = [sys.executable, str(evaluator_path), str(prediction_path), str(gold_path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "official HotpotQA evaluator failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    parsed: Any = None
    for line in reversed([line.strip() for line in completed.stdout.splitlines()]):
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
        if isinstance(parsed, dict):
            break
    if not isinstance(parsed, dict):
        raise ValueError(
            f"could not parse official evaluator output: {completed.stdout!r}"
        )
    return {str(key): float(value) for key, value in parsed.items()}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file contains no records: {path}")
    return rows


def load_traces(path: Path) -> List[EvidenceTraceV63]:
    traces = [EvidenceTraceV63.from_dict(row) for row in _read_jsonl(path)]
    ids = [trace.question_id for trace in traces]
    if len(ids) != len(set(ids)):
        raise ValueError("fixed trace contains duplicate question ids")
    for trace in traces:
        if trace.answer_contract_version != ANSWER_CONTRACT_VERSION:
            raise ValueError(
                f"trace {trace.question_id} uses {trace.answer_contract_version!r}; "
                f"expected {ANSWER_CONTRACT_VERSION!r}"
            )
        missing = [key for key in TRACE_REQUIRED_METADATA if key not in trace.metadata]
        if missing:
            raise ValueError(
                f"trace {trace.question_id} lacks reproducibility metadata: {missing}"
            )
        threshold = float(trace.metadata["acec_binding_threshold"])
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"trace {trace.question_id} has invalid ACEC binding threshold"
            )
    fixed_keys = (
        "acec_artifact_sha256",
        "acec_artifact_version",
        "acec_binding_threshold",
        "acec_k_mode",
        "acec_replay_version",
        "provenance_adapter",
        "provenance_adapter_version",
        "trace_generator",
    )
    for key in fixed_keys:
        values = {str(trace.metadata[key]) for trace in traces}
        if len(values) != 1:
            raise ValueError(f"fixed trace mixes {key}: {sorted(values)}")
    return traces


def load_gold(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("official evaluator manifest must be a non-empty JSON array")
    ids = [str(row.get("_id")) for row in payload]
    if any(value == "None" for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("official evaluator manifest has missing/duplicate ids")
    return payload


def _gold_sp(row: Mapping[str, Any]) -> set[Tuple[str, int]]:
    facts = row.get("supporting_facts") or []
    if isinstance(facts, Mapping):
        facts = zip(facts.get("title", []), facts.get("sent_id", []))
    return {(str(title), int(sent_id)) for title, sent_id in facts}


def _score_missing_raw_nli(
    traces: Sequence[EvidenceTraceV63],
    *,
    model_name: Optional[str],
    model_version: str,
    entailment_index: int,
    batch_size: int,
) -> List[EvidenceTraceV63]:
    all_candidates = [
        (trace, unit)
        for trace in traces
        for unit in trace.candidates
    ]
    precomputed = [
        (trace, unit)
        for trace, unit in all_candidates
        if RAW_NLI_SCORE_KEY in unit.scores
    ]
    if len(precomputed) == len(all_candidates):
        for trace in traces:
            recorded_version = trace.metadata.get("raw_nli_model_version")
            recorded_index = trace.metadata.get("raw_nli_entailment_index")
            if str(recorded_version) != str(model_version):
                raise ValueError(
                    f"trace {trace.question_id} raw NLI model metadata differs from "
                    "--nli_model_version"
                )
            if recorded_index is None or int(recorded_index) != int(entailment_index):
                raise ValueError(
                    f"trace {trace.question_id} raw NLI entailment index differs from CLI"
                )
        return list(traces)
    if not model_name:
        raise ValueError(
            f"{len(all_candidates) - len(precomputed)} candidates lack "
            f"{RAW_NLI_SCORE_KEY!r}; pass --nli_model to rescore the whole trace"
        )
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    pairs = [
        (unit.text, answer_hypothesis(trace.question, trace.draft_answer))
        for trace, unit in all_candidates
    ]
    probabilities = model.predict(
        pairs, apply_softmax=True, batch_size=batch_size, show_progress_bar=True
    )
    scores_by_id: Dict[Tuple[str, str], float] = {}
    for (trace, unit), probability in zip(all_candidates, probabilities):
        try:
            score = float(probability[entailment_index])
        except (IndexError, TypeError) as error:
            raise ValueError(
                "NLI model output is not a multi-class probability vector; "
                "set the correct model/entailment index"
            ) from error
        scores_by_id[(trace.question_id, unit.evidence_id)] = score

    rescored = []
    for trace in traces:
        units = []
        for unit in trace.candidates:
            key = (trace.question_id, unit.evidence_id)
            units.append(unit.with_scores({RAW_NLI_SCORE_KEY: scores_by_id[key]}))
        rescored.append(trace.with_candidates(units))
    return rescored


def _belief_pairs(trace: EvidenceTraceV63) -> set[Tuple[str, int]]:
    accepted_documents = {
        unit.document_id
        for unit in trace.candidates
        if float(unit.scores.get(ACEC_ACCEPT_SCORE_KEY, 0.0)) >= 0.5
    }
    return native_hotpot_pairs(
        unit for unit in trace.candidates if unit.document_id in accepted_documents
    )


def _recall(prediction: set[Tuple[str, int]], gold: set[Tuple[str, int]]) -> float:
    return len(prediction & gold) / len(gold) if gold else 1.0


def _failure_bucket(
    *,
    candidate: set[Tuple[str, int]],
    belief: set[Tuple[str, int]],
    selected: set[Tuple[str, int]],
    gold: set[Tuple[str, int]],
    answer_correct: bool,
    use_belief_filter: bool,
) -> str:
    if not gold.issubset(candidate):
        return "retrieval_missing_gold"
    if use_belief_filter and not gold.issubset(belief):
        return "belief_dropped_gold"
    missing = gold - selected
    extra = selected - gold
    if missing and extra:
        return "selector_missing_and_extra"
    if missing:
        return "selector_missing_gold"
    if extra:
        return "selector_added_extra"
    if not answer_correct:
        return "answer_wrong_with_exact_evidence"
    return "joint_exact"


def evaluate_variant(
    traces: Sequence[EvidenceTraceV63],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    *,
    selector: str,
    stop_rule: SharedStoppingRuleV63,
) -> Dict[str, Any]:
    answers: Dict[str, str] = {}
    sp_predictions: Dict[str, set[Tuple[str, int]]] = {}
    audit_rows = []
    diagnostic_rows = []
    question_metrics = []
    failure_counts: Counter[str] = Counter()
    answer_evidence_table: Counter[str] = Counter()
    candidate_recalls = []
    belief_recalls = []
    emitted_counts = []
    native_count = 0
    candidate_count = 0

    for trace in traces:
        gold_row = gold_by_id[trace.question_id]
        gold_sp = _gold_sp(gold_row)
        gold_cardinality = len(gold_sp)
        if selector == "nli":
            selection = select_generic_nli_v63(
                trace, stop_rule, gold_cardinality=gold_cardinality
            )
        elif selector == "acec":
            selection = select_acec_existing_v63(
                trace, stop_rule, gold_cardinality=gold_cardinality
            )
        else:
            raise ValueError(f"unknown selector {selector!r}")

        selected_sp = set(selection.official_hotpot_sp(trace))
        candidate_sp = native_hotpot_pairs(trace.candidates)
        belief_sp = _belief_pairs(trace)
        metrics = reference_question_metrics(
            trace.draft_answer,
            selected_sp,
            str(gold_row["answer"]),
            gold_sp,
        )
        bucket = _failure_bucket(
            candidate=candidate_sp,
            belief=belief_sp,
            selected=selected_sp,
            gold=gold_sp,
            answer_correct=bool(metrics["em"]),
            use_belief_filter=selector == "acec",
        )
        evidence_exact = bool(metrics["sp_em"])
        table_key = (
            ("answer_correct" if metrics["em"] else "answer_wrong")
            + "__"
            + ("evidence_exact" if evidence_exact else "evidence_inexact")
        )
        failure_counts[bucket] += 1
        answer_evidence_table[table_key] += 1
        candidate_recalls.append(_recall(candidate_sp, gold_sp))
        belief_recalls.append(_recall(belief_sp, gold_sp))
        emitted_counts.append(len(selected_sp))
        candidate_count += len(trace.candidates)
        native_count += sum(
            unit.official_hotpot_pair is not None for unit in trace.candidates
        )
        answers[trace.question_id] = trace.draft_answer
        sp_predictions[trace.question_id] = selected_sp
        question_metrics.append(metrics)
        audit_rows.append(
            {
                "question_id": trace.question_id,
                "draft_answer": trace.draft_answer,
                "selector_name": selection.selector_name,
                "selector_version": selection.selector_version,
                "stop_mode": selection.stop_mode,
                "stop_value": selection.stop_value,
                "candidate_evidence": [unit.to_dict() for unit in trace.candidates],
                "selected_evidence": [
                    {
                        **trace.candidate_by_id[evidence_id].to_dict(),
                        "selection_reason": selection.reasons[evidence_id],
                    }
                    for evidence_id in selection.selected_evidence_ids
                ],
                "official_sp": [list(pair) for pair in sorted(selected_sp)],
                "unmappable_candidate_count": sum(
                    unit.official_hotpot_pair is None for unit in trace.candidates
                ),
            }
        )
        diagnostic_rows.append(
            {
                "question_id": trace.question_id,
                "metrics": metrics,
                "candidate_gold_sp_recall": candidate_recalls[-1],
                "belief_gold_sp_recall": belief_recalls[-1],
                "failure_bucket": bucket,
                "gold_sp_count": gold_cardinality,
                "predicted_sp_count": len(selected_sp),
            }
        )

    correct_subset = [
        row for row in question_metrics if float(row["em"]) == 1.0
    ]
    diagnostics = {
        "reference_metric_version": REFERENCE_METRIC_VERSION,
        "reference_metrics": aggregate_reference_metrics(question_metrics),
        "correct_answer_subset_count": len(correct_subset),
        "correct_answer_subset_sp_em": (
            sum(row["sp_em"] for row in correct_subset) / len(correct_subset)
            if correct_subset
            else None
        ),
        "correct_answer_subset_sp_f1": (
            sum(row["sp_f1"] for row in correct_subset) / len(correct_subset)
            if correct_subset
            else None
        ),
        "candidate_gold_sp_mean_recall": sum(candidate_recalls) / len(candidate_recalls),
        "candidate_full_gold_sp_recall_rate": sum(value == 1.0 for value in candidate_recalls)
        / len(candidate_recalls),
        "belief_gold_sp_mean_recall": sum(belief_recalls) / len(belief_recalls),
        "belief_full_gold_sp_recall_rate": sum(value == 1.0 for value in belief_recalls)
        / len(belief_recalls),
        "provenance_mapping_coverage": native_count / candidate_count,
        "mean_emitted_evidence_count": sum(emitted_counts) / len(emitted_counts),
        "failure_buckets": dict(sorted(failure_counts.items())),
        "answer_by_evidence_exactness": dict(sorted(answer_evidence_table.items())),
    }
    return {
        "answers": answers,
        "sp": sp_predictions,
        "audit_rows": audit_rows,
        "diagnostic_rows": diagnostic_rows,
        "question_metrics": question_metrics,
        "diagnostics": diagnostics,
    }


def paired_bootstrap_delta(
    left: Sequence[Mapping[str, float]],
    right: Sequence[Mapping[str, float]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap needs two equally sized non-empty samples")
    deltas = [float(a[metric]) - float(b[metric]) for a, b in zip(left, right)]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(
            sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return {"delta": observed, "ci95_low": low, "ci95_high": high}


def _git_commit() -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--official_evaluator", required=True)
    parser.add_argument("--acec_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shared_threshold", type=float, required=True)
    parser.add_argument("--max_evidence", type=int, default=None)
    parser.add_argument("--threshold_provenance", required=True, choices=[
        "frozen_unsupervised", "hotpot_threshold_supervised"
    ])
    parser.add_argument("--threshold_fit_id_hash", default=None)
    parser.add_argument("--threshold_validation_id_hash", default=None)
    parser.add_argument("--nli_model", default=None)
    parser.add_argument("--nli_model_version", required=True)
    parser.add_argument("--nli_entailment_index", type=int, default=1)
    parser.add_argument("--nli_batch_size", type=int, default=128)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace_path = Path(args.trace).expanduser()
    gold_path = Path(args.gold).expanduser()
    evaluator_path = Path(args.official_evaluator).expanduser()
    acec_artifact_path = Path(args.acec_artifact).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    for path in (trace_path, gold_path, evaluator_path, acec_artifact_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required v6.3 input is missing/empty: {path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v6.3 evaluation: {output_dir}")
    if not 0.0 <= args.shared_threshold <= 1.0:
        raise ValueError("--shared_threshold must be in [0, 1]")
    if args.bootstrap_iterations <= 0 or args.nli_batch_size <= 0:
        raise ValueError("bootstrap iterations and NLI batch size must be positive")
    if args.threshold_provenance == "hotpot_threshold_supervised":
        if not args.threshold_fit_id_hash or not args.threshold_validation_id_hash:
            raise ValueError(
                "Hotpot-supervised threshold requires fit and validation id hashes"
            )
        if args.threshold_fit_id_hash == args.threshold_validation_id_hash:
            raise ValueError("threshold fit and validation id hashes must differ")

    traces = load_traces(trace_path)
    gold_rows = load_gold(gold_path)
    gold_by_id = {str(row["_id"]): row for row in gold_rows}
    trace_ids = [trace.question_id for trace in traces]
    missing_gold = sorted(set(trace_ids) - set(gold_by_id))
    if missing_gold:
        raise ValueError(f"fixed trace ids missing from evaluator manifest: {missing_gold[:10]}")
    expected_artifact_hash = _sha256(acec_artifact_path)
    mismatched = [
        trace.question_id
        for trace in traces
        if str(trace.metadata["acec_artifact_sha256"]) != expected_artifact_hash
    ]
    if mismatched:
        raise ValueError(
            "fixed trace ACEC artifact hash differs from --acec_artifact for ids: "
            + ", ".join(mismatched[:10])
        )
    traces = _score_missing_raw_nli(
        traces,
        model_name=args.nli_model,
        model_version=args.nli_model_version,
        entailment_index=args.nli_entailment_index,
        batch_size=args.nli_batch_size,
    )

    threshold_rule = SharedStoppingRuleV63(
        mode="shared_threshold",
        threshold=args.shared_threshold,
        max_units=args.max_evidence,
    )
    oracle_rule = SharedStoppingRuleV63(
        mode="gold_cardinality_oracle"
    )
    variants = {
        "nli_shared_threshold": ("nli", threshold_rule),
        "acec_shared_threshold": ("acec", threshold_rule),
        "nli_gold_cardinality_oracle": ("nli", oracle_rule),
        "acec_gold_cardinality_oracle": ("acec", oracle_rule),
    }
    evaluated = {
        name: evaluate_variant(traces, gold_by_id, selector=selector, stop_rule=rule)
        for name, (selector, rule) in variants.items()
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    gold_subset_path = output_dir / "official_gold_subset.json"
    with gold_subset_path.open("x", encoding="utf-8") as handle:
        json.dump([gold_by_id[qid] for qid in trace_ids], handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    official_results: Dict[str, Dict[str, float]] = {}
    for name, result in evaluated.items():
        submission = export_official_submission(result["answers"], result["sp"])
        prediction_path = output_dir / f"{name}_official_predictions.json"
        with prediction_path.open("x", encoding="utf-8") as handle:
            json.dump(submission, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        official_results[name] = run_official_evaluator(
            evaluator_path, prediction_path, gold_subset_path
        )
        with (output_dir / f"{name}_selection_audit.jsonl").open(
            "x", encoding="utf-8"
        ) as handle:
            for row in result["audit_rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (output_dir / f"{name}_evaluator_diagnostics.jsonl").open(
            "x", encoding="utf-8"
        ) as handle:
            for row in result["diagnostic_rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    parity = {}
    for name, result in evaluated.items():
        reference = result["diagnostics"]["reference_metrics"]
        official = official_results[name]
        shared_keys = sorted(set(reference) & set(official))
        maximum_error = max(
            (abs(float(reference[key]) - float(official[key])) for key in shared_keys),
            default=float("inf"),
        )
        parity[name] = {"shared_keys": shared_keys, "max_abs_error": maximum_error}
        if maximum_error > 1e-12:
            raise AssertionError(
                f"official/reference metric parity failed for {name}: {maximum_error}"
            )

    bootstrap = {}
    for mode in ("shared_threshold", "gold_cardinality_oracle"):
        acec_rows = evaluated[f"acec_{mode}"]["question_metrics"]
        nli_rows = evaluated[f"nli_{mode}"]["question_metrics"]
        bootstrap[mode] = {
            metric: paired_bootstrap_delta(
                acec_rows,
                nli_rows,
                metric,
                iterations=args.bootstrap_iterations,
                seed=args.seed + index,
            )
            for index, metric in enumerate(("sp_em", "sp_f1", "joint_em", "joint_f1"))
        }

    metadata = {
        "schema": "acec_v63_selector_evaluation",
        "schema_version": 63,
        "git_commit": _git_commit(),
        "question_count": len(traces),
        "evaluation_question_id_sha256": _id_hash(trace_ids),
        "trace_sha256": _sha256(trace_path),
        "gold_source_sha256": _sha256(gold_path),
        "acec_artifact_sha256": expected_artifact_hash,
        "utility_provider": "raw_answer_conditioned_nli",
        "answer_contract_version": ANSWER_CONTRACT_VERSION,
        "answer_hypothesis_version": ANSWER_HYPOTHESIS_VERSION,
        "selector_version": SELECTOR_VERSION,
        "selector_modes": ["raw_nli", "acec_existing"],
        "shared_threshold": args.shared_threshold,
        "max_evidence": args.max_evidence,
        "threshold_provenance": args.threshold_provenance,
        "threshold_fit_id_hash": args.threshold_fit_id_hash,
        "threshold_validation_id_hash": args.threshold_validation_id_hash,
        "nli_model_version": args.nli_model_version,
        "nli_entailment_index": args.nli_entailment_index,
        "feature_model_versions": {
            "raw_answer_nli": args.nli_model_version,
            "acec_document_filter_artifact_sha256": expected_artifact_hash,
        },
        "acec_artifact_versions": sorted(
            {str(trace.metadata["acec_artifact_version"]) for trace in traces}
        ),
        "acec_k_modes": sorted(
            {str(trace.metadata["acec_k_mode"]) for trace in traces}
        ),
        "acec_binding_thresholds": sorted(
            {float(trace.metadata["acec_binding_threshold"]) for trace in traces}
        ),
        "acec_replay_versions": sorted(
            {str(trace.metadata["acec_replay_version"]) for trace in traces}
        ),
        "trace_generators": sorted(
            {str(trace.metadata["trace_generator"]) for trace in traces}
        ),
        "provenance_adapters": sorted(
            {str(trace.metadata["provenance_adapter"]) for trace in traces}
        ),
        "provenance_adapter_versions": sorted(
            {str(trace.metadata["provenance_adapter_version"]) for trace in traces}
        ),
        "official_evaluator_sha256": _sha256(evaluator_path),
        "official_metrics": official_results,
        "diagnostics": {
            name: result["diagnostics"] for name, result in evaluated.items()
        },
        "paired_bootstrap_acec_minus_nli": bootstrap,
        "official_reference_parity": parity,
    }
    with (output_dir / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
