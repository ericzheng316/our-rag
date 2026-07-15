"""Build the versioned ACEC calibration-v2 artifact.

Differences from the preserved v1 builder:
  * records each slot's binding state at the turn it was scored;
  * fits action hit-rate priors from targeted slots only;
  * reports posterior AUC, Brier score, and ECE on validation and untouched
    test splits;
  * serializes the optional K predictor and artifact metadata under a strict
    versioned schema.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from belief.acec.calibration_v2 import (  # noqa: E402
    fit_observation_model_v2,
    k_predictor_payload,
    posterior_quality_metrics,
    save_calibration_artifact_v2,
)
from belief.acec.offline_fit import EmpiricalKPredictor, HitExample, KExample  # noqa: E402


def match_slots_to_sf(
    slot_hypotheses: Sequence[str],
    sf_titles: Sequence[str],
    embedder: Any,
    threshold: float = 0.30,
) -> Dict[int, str]:
    """Greedy one-to-one slot/title matching, preserved from the v1 gate."""
    if not slot_hypotheses or not sf_titles:
        return {}
    embeddings = embedder.encode(list(slot_hypotheses) + list(sf_titles), show_progress_bar=False)
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
    for similarity, slot_idx, sf_idx in pairs:
        if similarity < threshold:
            break
        if slot_idx in assignment or sf_idx in claimed:
            continue
        assignment[slot_idx] = sf_titles[sf_idx]
        claimed.add(sf_idx)
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

    # Each entry stores the binding status used by the observation model on
    # that exact turn.  Reading slot.bound after the full replay (v1) leaks a
    # future state backward into historical calibration rows.
    per_turn = []
    for turn_idx, docs in enumerate(docs_per_turn):
        query = split_queries[turn_idx][0] if turn_idx < len(split_queries) and split_queries[turn_idx] else None
        bound_before = {idx: slot.bound for idx, slot in enumerate(belief.coverage_belief.slots)}
        result = belief.turn(
            query=query,
            new_docs=[{"contents": doc} for doc in docs],
            is_answer=False,
        )
        bound_at_score = {slot_idx: bound_before.get(slot_idx, False) for slot_idx in result.slot_scores}
        per_turn.append((turn_idx, result, bound_at_score))

    slot_hypotheses = [slot.hypothesis for slot in belief.coverage_belief.slots]
    slot_to_sf = match_slots_to_sf(slot_hypotheses, sf_titles, belief.labeler.embedder)

    examples: List[HitExample] = []
    for turn_idx, result, bound_at_score in per_turn:
        doc_titles = {title_from_content(doc) for doc in docs_per_turn[turn_idx]}
        for slot_idx, nli_score in result.slot_scores.items():
            sf_title = slot_to_sf.get(slot_idx)
            if sf_title is None:
                continue
            role = "tgt" if slot_idx == result.action.target_slot else "inc"
            examples.append(
                HitExample(
                    action_mode=result.action.mode.value,
                    role=role,
                    bound=bound_at_score[slot_idx],
                    nli_score=nli_score,
                    is_hit=sf_title in doc_titles,
                )
            )

    question_embedding = belief.labeler.embedder.encode([question], show_progress_bar=False)[0]
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


def main() -> None:
    # Keep the heavyweight torch/transformers dependency out of module import
    # so the replay-label logic can be unit-tested with lightweight fakes.
    from belief.obs_extractor import E5Embedder

    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--e5_model_path", required=True)
    parser.add_argument("--nli_model", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--k_neighbors", type=int, default=15)
    parser.add_argument("--tau_new", type=float, default=ACECConfig().tau_new)
    parser.add_argument("--validation_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.validation_frac <= 0 or args.test_frac <= 0:
        raise ValueError("validation_frac and test_frac must both be positive")
    if args.validation_frac + args.test_frac >= 1:
        raise ValueError("validation_frac + test_frac must be < 1")

    os.makedirs(args.out_dir, exist_ok=True)
    adapter = get_adapter(args.dataset)
    embedder = E5QueryEmbedder(E5Embedder(args.e5_model_path))
    nli_scorer = CrossEncoderNLIScorer(args.nli_model) if args.nli_model else CosineNLIScorer(embedder)
    config = ACECConfig(k_max=args.k_max, tau_new=args.tau_new)
    belief = ACECBeliefState(embedder, nli_scorer, config=config)

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

    observation_model, hit_rates, action_counts = fit_observation_model_v2(fit_hits)
    validation_metrics = posterior_quality_metrics(validation_hits, observation_model, hit_rates)
    test_metrics = posterior_quality_metrics(test_hits, observation_model, hit_rates)

    k_predictor = EmpiricalKPredictor(args.k_max, fit_k, neighbors=args.k_neighbors)
    validation_k_accuracy = k_predictor.evaluate_k_accuracy(validation_k)
    test_k_accuracy = k_predictor.evaluate_k_accuracy(test_k)

    metrics = {
        "fit": {"records": len(fit_records), "hit_examples": len(fit_hits), "k_examples": len(fit_k)},
        "validation": {**validation_metrics, "k_accuracy": validation_k_accuracy},
        "test": {**test_metrics, "k_accuracy": test_k_accuracy},
        "target_action_counts": action_counts,
    }
    metadata = {
        "dataset": args.dataset,
        "records_path": os.path.abspath(args.records),
        "e5_model_path": args.e5_model_path,
        "nli_model": args.nli_model or "cosine-sanity-only",
        "k_max": args.k_max,
        "tau_new": args.tau_new,
        "seed": args.seed,
        "validation_frac": args.validation_frac,
        "test_frac": args.test_frac,
    }
    artifact_path = os.path.join(args.out_dir, "acec_calibration_v2.json")
    save_calibration_artifact_v2(
        artifact_path,
        observation_model,
        hit_rates,
        k_predictor_payload(fit_k, args.k_max, args.k_neighbors),
        metadata,
        metrics,
    )

    print("=== ACEC calibration v2 ===")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {artifact_path}")


if __name__ == "__main__":
    main()
