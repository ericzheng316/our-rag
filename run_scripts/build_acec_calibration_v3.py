"""Build a gated ACEC calibration-v3 artifact without touching v1/v2 paths."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
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
    GoldEvidenceAdapter,
    get_adapter,
    title_from_content,
)
from belief.acec.calibration_v3 import (  # noqa: E402
    MODEL_KIND,
    fit_observation_model_v3,
    k_predictor_payload,
    posterior_quality_metrics,
    save_calibration_artifact_v3,
)
from belief.acec.offline_fit import EmpiricalKPredictor, HitExample, KExample  # noqa: E402


def normalize_title(title: str) -> str:
    """Canonical comparison form for distant-supervision title labels."""
    value = unicodedata.normalize("NFKC", str(title)).casefold().replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


def match_slots_to_sf(
    slot_hypotheses: Sequence[str],
    sf_titles: Sequence[str],
    embedder: Any,
    threshold: float = 0.30,
) -> Dict[int, str]:
    if not slot_hypotheses or not sf_titles:
        return {}
    embeddings = embedder.encode(
        list(slot_hypotheses) + list(sf_titles), show_progress_bar=False
    )
    slot_embeddings = embeddings[: len(slot_hypotheses)]
    sf_embeddings = embeddings[len(slot_hypotheses) :]
    similarities = slot_embeddings @ sf_embeddings.T
    pairs = sorted(
        (
            (float(similarities[i, j]), i, j)
            for i in range(len(slot_hypotheses))
            for j in range(len(sf_titles))
        ),
        reverse=True,
    )
    assignment: Dict[int, str] = {}
    claimed = set()
    for similarity, slot_index, sf_index in pairs:
        if similarity < threshold:
            break
        if slot_index in assignment or sf_index in claimed:
            continue
        assignment[slot_index] = sf_titles[sf_index]
        claimed.add(sf_index)
    return assignment


def replay_record(
    record: Dict[str, Any],
    belief: ACECBeliefState,
    adapter: GoldEvidenceAdapter,
) -> Tuple[List[HitExample], KExample]:
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
    slot_to_sf = match_slots_to_sf(slot_hypotheses, sf_titles, belief.labeler.embedder)

    examples: List[HitExample] = []
    for turn_index, result, bound_at_score in per_turn:
        doc_titles = {
            normalize_title(title_from_content(doc))
            for doc in docs_per_turn[turn_index]
        }
        for slot_index, nli_score in result.slot_scores.items():
            sf_title = slot_to_sf.get(slot_index)
            if sf_title is None:
                continue
            role = "tgt" if slot_index == result.action.target_slot else "inc"
            examples.append(
                HitExample(
                    action_mode=result.action.mode.value,
                    role=role,
                    bound=bound_at_score[slot_index],
                    nli_score=nli_score,
                    is_hit=normalize_title(sf_title) in doc_titles,
                )
            )

    question_embedding = belief.labeler.embedder.encode(
        [question], show_progress_bar=False
    )[0]
    return examples, KExample(question_embedding, len(sf_titles))


def collect_split(
    records: Sequence[Dict[str, Any]],
    belief: ACECBeliefState,
    adapter: GoldEvidenceAdapter,
) -> Tuple[List[HitExample], List[KExample]]:
    hit_examples: List[HitExample] = []
    k_examples: List[KExample] = []
    for record in records:
        if not adapter.gold_titles(record) or not record.get("docs"):
            continue
        hits, k_example = replay_record(record, belief, adapter)
        hit_examples.extend(hits)
        k_examples.append(k_example)
    return hit_examples, k_examples


def select_k_strategy(
    requested_mode: str,
    fit_examples: Sequence[KExample],
    k_max: int,
    fixed_k: int,
    neighbors: int,
) -> Tuple[str, int, Optional[EmpiricalKPredictor], Optional[Dict[str, Any]]]:
    labels = [min(max(int(example.k_true), 1), k_max) for example in fit_examples]
    unique_labels = sorted(set(labels))
    if requested_mode == "auto":
        mode = "fixed" if len(unique_labels) == 1 else "predictor"
    else:
        mode = requested_mode

    if mode == "fixed":
        selected_k = unique_labels[0] if requested_mode == "auto" and unique_labels else fixed_k
        return mode, selected_k, None, None
    predictor = EmpiricalKPredictor(k_max, fit_examples, neighbors=neighbors)
    return (
        mode,
        fixed_k,
        predictor,
        k_predictor_payload(fit_examples, k_max, neighbors),
    )


def evaluate_k_accuracy(
    examples: Sequence[KExample],
    mode: str,
    fixed_k: int,
    predictor: Optional[EmpiricalKPredictor],
    k_max: int,
) -> float:
    if not examples:
        return float("nan")
    if mode == "predictor":
        if predictor is None:
            raise ValueError("predictor mode requires a fitted K predictor")
        return predictor.evaluate_k_accuracy(examples)
    return float(
        np.mean(
            [min(max(int(example.k_true), 1), k_max) == fixed_k for example in examples]
        )
    )


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
    parser.add_argument("--validation_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
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

    fit_hits, fit_k = collect_split(fit_records, belief, adapter)
    validation_hits, validation_k = collect_split(validation_records, belief, adapter)
    test_hits, test_k = collect_split(test_records, belief, adapter)

    observation_model, hit_rates, action_counts = fit_observation_model_v3(fit_hits)
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
        },
        "validation": {**validation_metrics, "k_accuracy": validation_k_accuracy},
        "test": {**test_metrics, "k_accuracy": test_k_accuracy},
        "target_action_counts": action_counts,
        "gate": {"pass": gate_pass, "failed_checks": gate_reasons},
    }
    metadata = {
        "dataset": args.dataset,
        "records_path": os.path.abspath(args.records),
        "e5_model_path": args.e5_model_path,
        "nli_model": args.nli_model or "cosine-sanity-only",
        "observation_model_kind": MODEL_KIND,
        "k_max": args.k_max,
        "recommended_k_mode": k_mode,
        "fixed_k": fixed_k if k_mode == "fixed" else None,
        "tau_new": args.tau_new,
        "seed": args.seed,
        "validation_frac": args.validation_frac,
        "test_frac": args.test_frac,
    }

    print("=== ACEC calibration v3 ===")
    print(json.dumps(metrics, indent=2))
    if not gate_pass:
        raise RuntimeError(
            "ACEC calibration gate failed; artifact was not written: "
            + ", ".join(gate_reasons)
        )

    os.makedirs(args.out_dir, exist_ok=True)
    artifact_path = os.path.join(args.out_dir, "acec_calibration_v3.json")
    save_calibration_artifact_v3(
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
