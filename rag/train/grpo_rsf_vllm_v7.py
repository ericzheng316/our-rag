"""ACEC v7 training with frozen belief-conditioned set selection.

V6 files remain unchanged.  V7 retrieves a wide pool, scores every candidate
against the current frozen belief, sequentially selects K documents, updates
the live belief exactly once with only those selected documents, and assigns
the resulting potential change to the current query turn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import torch
from vllm import SamplingParams

import grpo_rsf_simple as simple_trainer
import grpo_rsf_vllm as base_trainer
import grpo_rsf_vllm_v5 as v5_entry
from acec_v6_contract import AdaptiveKLController, EvidenceAnswerGuard
from grpo_estimator_v7 import (
    CoverageRewardModeV7,
    grpo_loss_v7,
    terminal_potential_correction_v7,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.answer_value_v7 import (  # noqa: E402
    load_answer_value_artifact_v7,
)
from belief.acec.runtime_v7 import BeliefSelectorRuntimeV7  # noqa: E402
from belief.acec.set_selector_v7 import (  # noqa: E402
    SequentialSetSelectorV7,
    load_selector_artifact_v7,
)


def _document_label(document: Dict[str, Any]) -> str:
    title = document.get("title")
    if title:
        return str(title)[:200]
    contents = str(document.get("contents") or "")
    return contents.splitlines()[0][:200] if contents else ""


def _document_id(document: Dict[str, Any], rank: int = 0) -> str:
    value = document.get("id")
    if value is not None and str(value).strip():
        return str(value)
    digest = hashlib.sha256(
        str(document.get("contents") or "").encode("utf-8")
    ).hexdigest()[:20]
    return f"content:{digest}:{rank}"


def _terminal_correction(state: Dict[str, Any], reward_config: Any, mode: str) -> float:
    coverage = float(state["belief"].coverage_belief.coverage())
    return terminal_potential_correction_v7(
        coverage, float(reward_config.coverage), mode
    )


class MetricsWriterV7:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.episode = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite v7 metrics: {self.path}")

    def write(self, metrics: Dict[str, float]) -> None:
        self.episode += 1
        payload = {
            "schema": "acec_online_metrics_v7",
            "episode": self.episode,
            **{key: float(value) for key, value in metrics.items()},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class TraceWriterV7:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.episode = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite v7 trace: {self.path}")

    def write_episode(self, states: List[Dict[str, Any]]) -> None:
        self.episode += 1
        with self.path.open("a", encoding="utf-8") as handle:
            for state in states:
                payload = {
                    "schema": "acec_rollout_trace_v7",
                    "version": 70,
                    "episode": self.episode,
                    "question_id": state["question_id"],
                    "question_index": state["qi"],
                    "sample_index": state["sample_index"],
                    "question": state["question"],
                    "answer": state.get("answer"),
                    "answer_correct": bool(state["answer_correct"]),
                    "answered": bool(state["answered"]),
                    "format_error": bool(state["format_error"]),
                    "retrieval_calls": int(state["retrieval_calls"]),
                    "candidate_documents": int(state["candidate_documents"]),
                    "selected_documents": int(state["selected_documents"]),
                    "total_reward": sum(
                        float(turn[2]) for turn in state["turns"]
                    ),
                    "timing": {
                        "generation_seconds_share": float(
                            state["generation_seconds"]
                        ),
                        "retrieval_seconds_share": float(
                            state["retrieval_seconds"]
                        ),
                        "candidate_scoring_seconds": float(
                            state["candidate_scoring_seconds"]
                        ),
                        "selection_seconds": float(
                            state["selector_seconds"]
                        ),
                    },
                    "events": state["trace"],
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def vllm_rollout_batch_v7(
    llm,
    lora_request,
    questions: List[Dict[str, Any]],
    n_samples: int,
    n_turns: int = 5,
    belief_factory=None,
    rollout_temperature: float = 0.7,
    reward_config=None,
    collect_metrics: bool = False,
    *,
    answer_guard: EvidenceAnswerGuard,
    trace_writer: TraceWriterV7,
    selector_runtime: BeliefSelectorRuntimeV7,
    coverage_reward_mode: str,
    metrics_writer: MetricsWriterV7 | None = None,
):
    if belief_factory is None or reward_config is None:
        raise ValueError("ACEC v7 requires belief_factory and RewardConfig")
    reward_mode = CoverageRewardModeV7(coverage_reward_mode)
    states: List[Dict[str, Any]] = []
    for question_index, item in enumerate(questions):
        for sample_index in range(n_samples):
            states.append(
                {
                    "qi": question_index,
                    "sample_index": sample_index,
                    "question": item["question"],
                    "question_id": str(
                        item.get("id")
                        or item.get("_id")
                        or hashlib.sha256(
                            str(item["question"]).strip().encode("utf-8")
                        ).hexdigest()[:24]
                    ),
                    "context": f"The question: {item['question']}",
                    "retrieved_ids": [],
                    "found_sf": set(),
                    "belief": belief_factory(item["question"]),
                    "temperature": rollout_temperature,
                    "turns": [],
                    "trace": [],
                    "done": False,
                    "answered": False,
                    "answer": None,
                    "answer_correct": False,
                    "format_error": False,
                    "retrieval_calls": 0,
                    "empty_retrievals": 0,
                    "answer_coverage": None,
                    "guard_applied": False,
                    "generation_seconds": 0.0,
                    "retrieval_seconds": 0.0,
                    "candidate_scoring_seconds": 0.0,
                    "selector_seconds": 0.0,
                    "candidate_documents": 0,
                    "selected_documents": 0,
                    "selector_calls": 0,
                    "selector_traces": 0,
                }
            )

    for turn_index in range(n_turns):
        active = [index for index, state in enumerate(states) if not state["done"]]
        if not active:
            break
        prompts = [
            base_trainer.apply_chat_template(states[index]["context"])
            for index in active
        ]
        sampling_params = [
            SamplingParams(
                temperature=states[index]["temperature"],
                max_tokens=512,
                stop_token_ids=[base_trainer.STOP_TOKEN],
                logprobs=1,
            )
            for index in active
        ]
        generation_started = time.perf_counter()
        outputs = llm.generate(
            prompts, sampling_params, lora_request=lora_request, use_tqdm=False
        )
        generation_share = (
            (time.perf_counter() - generation_started) / max(len(active), 1)
        )
        for state_index in active:
            states[state_index]["generation_seconds"] += generation_share
            states[state_index]["_last_generation_seconds"] = generation_share

        pending_retrieval: List[int] = []
        for output_index, state_index in enumerate(active):
            state = states[state_index]
            completion = outputs[output_index].outputs[0]
            input_ids = torch.tensor(
                outputs[output_index].prompt_token_ids,
                dtype=torch.long,
                device="cuda",
            )
            output_ids = torch.tensor(
                completion.token_ids, dtype=torch.long, device="cuda"
            )
            engine_logprobs = base_trainer._vllm_sampled_token_logprobs(
                completion, device=input_ids.device
            )
            parsed = base_trainer.parse_step(
                f"Step {turn_index + 1}:\n{completion.text}"
            )
            if not parsed.get("analysis"):
                reward = -reward_config.format_error + _terminal_correction(
                    state, reward_config, reward_mode.value
                )
                state["turns"].append(
                    (
                        input_ids,
                        output_ids,
                        reward,
                        engine_logprobs,
                        state["temperature"],
                        "format_error",
                    )
                )
                state["trace"].append(
                    {
                        "turn": turn_index + 1,
                        "kind": "format_error",
                        "reward": reward,
                        "coverage_reward_mode": reward_mode.value,
                        "timing": {
                            "generation_seconds_share": state.pop(
                                "_last_generation_seconds", 0.0
                            )
                        },
                    }
                )
                state["format_error"] = True
                state["done"] = True
                continue

            if parsed.get("answer"):
                golds = questions[state["qi"]]["golden_answers"]
                correct = base_trainer.exact_match(parsed["answer"], golds)
                coverage = float(state["belief"].coverage_belief.coverage())
                adjustment = answer_guard.reward_adjustment(correct, coverage)
                terminal = _terminal_correction(
                    state, reward_config, reward_mode.value
                )
                reward = (
                    reward_config.answer_reward(correct) + adjustment + terminal
                )
                state["belief"].turn(query=None, new_docs=[], is_answer=True)
                state["turns"].append(
                    (
                        input_ids,
                        output_ids,
                        reward,
                        engine_logprobs,
                        state["temperature"],
                        "answer",
                    )
                )
                state["trace"].append(
                    {
                        "turn": turn_index + 1,
                        "kind": "answer",
                        "answer": parsed["answer"],
                        "correct": correct,
                        "coverage": coverage,
                        "guard_adjustment": adjustment,
                        "terminal_potential_correction": terminal,
                        "coverage_reward_mode": reward_mode.value,
                        "reward": reward,
                        "timing": {
                            "generation_seconds_share": state.pop(
                                "_last_generation_seconds", 0.0
                            )
                        },
                    }
                )
                state["answer"] = parsed["answer"]
                state["answered"] = True
                state["answer_correct"] = correct
                state["answer_coverage"] = coverage
                state["guard_applied"] = adjustment < 0.0
                state["done"] = True
                continue

            if parsed.get("query"):
                state["_pending"] = (
                    input_ids,
                    output_ids,
                    engine_logprobs,
                    parsed,
                )
                pending_retrieval.append(state_index)
            else:
                reward = -reward_config.format_error + _terminal_correction(
                    state, reward_config, reward_mode.value
                )
                state["turns"].append(
                    (
                        input_ids,
                        output_ids,
                        reward,
                        engine_logprobs,
                        state["temperature"],
                        "format_error",
                    )
                )
                state["trace"].append(
                    {
                        "turn": turn_index + 1,
                        "kind": "format_error",
                        "reward": reward,
                        "coverage_reward_mode": reward_mode.value,
                        "timing": {
                            "generation_seconds_share": state.pop(
                                "_last_generation_seconds", 0.0
                            )
                        },
                    }
                )
                state["format_error"] = True
                state["done"] = True

        if pending_retrieval:
            queries = [
                states[index]["_pending"][3]["query"]
                for index in pending_retrieval
            ]
            retrieval_started = time.perf_counter()
            documents_by_state = base_trainer._retrieve_batch(queries)
            retrieval_share = (
                (time.perf_counter() - retrieval_started)
                / max(len(pending_retrieval), 1)
            )
            for state_index, candidate_documents in zip(
                pending_retrieval, documents_by_state
            ):
                state = states[state_index]
                state["retrieval_seconds"] += retrieval_share
                input_ids, output_ids, engine_logprobs, parsed = state.pop(
                    "_pending"
                )
                state["retrieval_calls"] += 1
                seen = set(state["retrieved_ids"])
                candidate_documents = [
                    document
                    for rank, document in enumerate(candidate_documents)
                    if _document_id(document, rank) not in seen
                ]
                state["candidate_documents"] += len(candidate_documents)
                if not candidate_documents:
                    state["empty_retrievals"] += 1

                remaining_capacity = max(
                    int(base_trainer.MAX_DOCS) - len(state["retrieved_ids"]), 0
                )
                if remaining_capacity == 0:
                    belief_result = state["belief"].turn(
                        query=parsed["query"], new_docs=[]
                    )
                    reward = reward_config.retrieval_reward(
                        belief_result.delta_coverage
                    )
                    selected_documents: List[Dict[str, Any]] = []
                    selection_payload: Dict[str, Any] = {
                        "selected_ids": [],
                        "reason": "cumulative_context_capacity_exhausted",
                    }
                    candidate_scoring_seconds = 0.0
                    selection_seconds = 0.0
                else:
                    state["selector_calls"] += 1
                    v7_result = selector_runtime.turn(
                        question_id=state["question_id"],
                        belief=state["belief"],
                        query=parsed["query"],
                        candidate_documents=candidate_documents,
                        retrieval_budget_remaining=n_turns - turn_index,
                        sample_selector=False,
                        selector_seed=(
                            state["sample_index"] * 1000 + turn_index
                        ),
                        selected_k=min(
                            selector_runtime.selected_k, remaining_capacity
                        ),
                    )
                    candidate_scoring_seconds = (
                        v7_result.candidate_scoring_seconds
                    )
                    selection_seconds = v7_result.selection_seconds
                    state["candidate_scoring_seconds"] += (
                        candidate_scoring_seconds
                    )
                    state["selector_seconds"] += selection_seconds
                    state["selector_traces"] += 1
                    selected_documents = [
                        dict(document)
                        for document in v7_result.selected_documents
                    ]
                    belief_result = v7_result.belief_result
                    reward = reward_config.retrieval_reward(
                        belief_result.delta_coverage
                    )
                    selection_payload = v7_result.selection_trace.to_dict()
                    selection_payload["candidate_features"] = [
                        {
                            "candidate_id": candidate.candidate_id,
                            "retrieval_rank": candidate.retrieval_rank,
                            "retrieval_score": candidate.retrieval_score,
                            "slot_entailment": list(candidate.slot_entailment),
                            "slot_hit_probabilities": list(
                                candidate.slot_hit_probabilities
                            ),
                            "slot_contradiction": list(
                                candidate.slot_contradiction
                            ),
                            "slot_neutral": list(candidate.slot_neutral),
                            "signed_answer_value": candidate.signed_answer_value,
                            "source_quality": candidate.source_quality,
                            "document_length": candidate.document_length,
                        }
                        for candidate in v7_result.candidate_features
                    ]
                    selection_payload["simulator_live_coverage_error"] = (
                        v7_result.simulator_live_coverage_error
                    )

                selected_trace_ids = list(
                    selection_payload.get("selected_ids") or ()
                )
                for selected_index, document in enumerate(selected_documents):
                    identifier = (
                        selected_trace_ids[selected_index]
                        if selected_index < len(selected_trace_ids)
                        else _document_id(document, selected_index)
                    )
                    if identifier not in state["retrieved_ids"]:
                        state["retrieved_ids"].append(identifier)
                state["selected_documents"] += len(selected_documents)
                document_text = "\n".join(
                    str(document.get("contents") or "")
                    for document in selected_documents
                )
                found_before = set(state["found_sf"])
                gold_delta, state["found_sf"] = base_trainer.rsf_marginal(
                    document_text,
                    questions[state["qi"]]["sf_titles"],
                    state["found_sf"],
                )

                state["turns"].append(
                    (
                        input_ids,
                        output_ids,
                        reward,
                        engine_logprobs,
                        state["temperature"],
                        "retrieval",
                    )
                )
                trace_event = {
                    "turn": turn_index + 1,
                    "kind": "retrieval",
                    "analysis": parsed["analysis"][:512],
                    "query": parsed["query"],
                    "candidate_pool": [
                        {
                            "id": _document_id(document, rank),
                            "title": _document_label(document),
                            "rank": rank,
                            "retrieval_score": document.get(
                                "retrieval_score",
                                document.get("score", document.get("distance")),
                            ),
                        }
                        for rank, document in enumerate(candidate_documents)
                    ],
                    "selected_documents": [
                        {
                            "id": (
                                selected_trace_ids[selected_index]
                                if selected_index < len(selected_trace_ids)
                                else _document_id(document, selected_index)
                            ),
                            "title": _document_label(document),
                        }
                        for selected_index, document in enumerate(
                            selected_documents
                        )
                    ],
                    "selection": selection_payload,
                    "new_gold_sf_titles": sorted(
                        set(state["found_sf"]) - found_before
                    ),
                    "gold_sf_delta": gold_delta,
                    "coverage_reward_mode": reward_mode.value,
                    "reward": reward,
                    "timing": {
                        "generation_seconds_share": state.pop(
                            "_last_generation_seconds", 0.0
                        ),
                        "retrieval_seconds_share": retrieval_share,
                        "candidate_scoring_seconds": candidate_scoring_seconds,
                        "selection_seconds": selection_seconds,
                    },
                }
                if belief_result is not None:
                    trace_event.update(
                        {
                            "action": belief_result.action.mode.value,
                            "target_slot": belief_result.action.target_slot,
                            "coverage_before": belief_result.coverage_before,
                            "coverage_after": belief_result.coverage_after,
                            "delta_coverage": belief_result.delta_coverage,
                            "stop_voi": belief_result.stop_voi,
                            "slot_p": list(belief_result.features.slot_p),
                            "slot_bound": list(
                                belief_result.features.slot_bound
                            ),
                        }
                    )
                state["trace"].append(trace_event)
                state["context"] += (
                    f"\nStep {turn_index + 1}:\n"
                    f"The problem analysis: {parsed['analysis']}\n"
                    f"The retrieval query: {parsed['query']}\n"
                    f"The retrieval documents: {document_text[:512]}"
                )
                if turn_index == n_turns - 1:
                    terminal = _terminal_correction(
                        state, reward_config, reward_mode.value
                    )
                    last = state["turns"][-1]
                    adjusted = (
                        last[2] - reward_config.format_error + terminal
                    )
                    state["turns"][-1] = (
                        *last[:2],
                        adjusted,
                        *last[3:],
                    )
                    state["trace"][-1]["reward"] = adjusted
                    state["trace"][-1]["max_turn_penalty"] = (
                        reward_config.format_error
                    )
                    state["trace"][-1][
                        "terminal_potential_correction"
                    ] = terminal
                    state["done"] = True

    trajectories: List[List[List]] = [[] for _ in questions]
    for state in states:
        trajectories[state["qi"]].append(state["turns"])
    trace_writer.write_episode(states)
    if not collect_metrics:
        return trajectories
    rollout_count = max(len(states), 1)
    total_retrievals = sum(state["retrieval_calls"] for state in states)
    sf_recalls = []
    for state in states:
        titles = set(questions[state["qi"]]["sf_titles"])
        sf_recalls.append(
            len(state["found_sf"]) / len(titles) if titles else 0.0
        )
    answer_coverages = [
        float(state["answer_coverage"])
        for state in states
        if state["answer_coverage"] is not None
    ]
    selector_pipeline_seconds = [
        float(state["candidate_scoring_seconds"])
        + float(state["selector_seconds"])
        for state in states
    ]
    selector_overhead_fractions = [
        pipeline
        / max(
            pipeline
            + float(state["generation_seconds"])
            + float(state["retrieval_seconds"]),
            1e-9,
        )
        for state, pipeline in zip(states, selector_pipeline_seconds)
    ]

    def percentile(values: List[float], probability: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = probability * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    metrics = {
        "answer_em": sum(bool(state["answer_correct"]) for state in states)
        / rollout_count,
        "answer_rate": sum(bool(state["answered"]) for state in states)
        / rollout_count,
        "gold_sf_recall": sum(sf_recalls) / rollout_count,
        "retrieval_calls": total_retrievals / rollout_count,
        "format_error_rate": sum(bool(state["format_error"]) for state in states)
        / rollout_count,
        "empty_retrieval_rate": sum(state["empty_retrievals"] for state in states)
        / max(total_retrievals, 1),
        "premature_guard_rate": sum(bool(state["guard_applied"]) for state in states)
        / rollout_count,
        "mean_answer_coverage": sum(answer_coverages)
        / max(len(answer_coverages), 1),
        "candidate_documents_per_rollout": sum(
            state["candidate_documents"] for state in states
        )
        / rollout_count,
        "selected_documents_per_rollout": sum(
            state["selected_documents"] for state in states
        )
        / rollout_count,
        "selector_seconds_per_rollout": sum(
            state["selector_seconds"] for state in states
        )
        / rollout_count,
        "candidate_scoring_seconds_per_rollout": sum(
            state["candidate_scoring_seconds"] for state in states
        )
        / rollout_count,
        "selector_pipeline_seconds_per_rollout": sum(
            selector_pipeline_seconds
        )
        / rollout_count,
        "selector_overhead_fraction_p95": percentile(
            selector_overhead_fractions, 0.95
        ),
        "selection_trace_rate": sum(
            state["selector_traces"] for state in states
        )
        / max(sum(state["selector_calls"] for state in states), 1),
        "selector_calls_per_rollout": sum(
            state["selector_calls"] for state in states
        )
        / rollout_count,
    }
    if metrics_writer is not None:
        metrics_writer.write(metrics)
    base_trainer.log.info(
        "[ACEC-v7] selector_pipeline_s=%.3f overhead_p95=%.3f "
        "candidates=%.2f selected=%.2f mode=%s",
        metrics["selector_pipeline_seconds_per_rollout"],
        metrics["selector_overhead_fraction_p95"],
        metrics["candidate_documents_per_rollout"],
        metrics["selected_documents_per_rollout"],
        reward_mode.value,
    )
    return trajectories, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = v5_entry.build_parser()
    parser.description = "ACEC-v7 frozen learnable set-selector training"
    parser.add_argument(
        "--selector_strategy_v7",
        choices=("learned", "relevance", "fixed_rel_plus_coverage"),
        default="learned",
    )
    parser.add_argument("--selector_artifact_v7")
    parser.add_argument("--answer_value_artifact_v7")
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--selected_k", type=int, default=5)
    parser.add_argument("--fixed_coverage_weight", type=float, default=1.0)
    parser.add_argument(
        "--coverage_reward_mode",
        choices=tuple(mode.value for mode in CoverageRewardModeV7),
        default=CoverageRewardModeV7.LEGACY_COVERAGE_AUX.value,
    )
    parser.add_argument("--answer_guard_coverage", type=float, default=0.50)
    parser.add_argument("--premature_answer_penalty", type=float, default=0.10)
    parser.add_argument("--answer_kl_multiplier", type=float, default=2.0)
    parser.add_argument("--kl_high_watermark", type=float, default=0.08)
    parser.add_argument("--kl_recovery_watermark", type=float, default=0.05)
    parser.add_argument("--kl_max_coef", type=float, default=0.05)
    parser.add_argument("--trace_path", type=str)
    return parser


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    if args.candidate_pool_size < args.selected_k:
        raise ValueError("candidate pool size must be >= selected K")
    if args.selected_k > int(base_trainer.MAX_DOCS):
        raise ValueError("selected K cannot exceed cumulative MAX_DOCS")
    if args.selector_strategy_v7 == "learned" and (
        not args.selector_artifact_v7
        or not args.answer_value_artifact_v7
    ):
        raise ValueError(
            "learned v7 selection requires selector and answer-value artifacts"
        )
    args.use_acec = True
    v5_entry._load_checked_artifact(args.acec_artifact_v5)

    selector_artifact = (
        load_selector_artifact_v7(Path(args.selector_artifact_v7))
        if args.selector_artifact_v7
        else None
    )
    answer_value_artifact = (
        load_answer_value_artifact_v7(Path(args.answer_value_artifact_v7))
        if args.answer_value_artifact_v7
        else None
    )
    selector = SequentialSetSelectorV7(
        artifact=selector_artifact,
        answer_value_artifact=answer_value_artifact,
        strategy=args.selector_strategy_v7,
        fixed_coverage_weight=args.fixed_coverage_weight,
    )
    selector_runtime = BeliefSelectorRuntimeV7(
        selector, selected_k=args.selected_k
    )
    # `_retrieve_batch` was imported from grpo_rsf_simple; its globals remain
    # owned by that module, so widening the pool is an explicit v7-only change.
    simple_trainer.RETRIEVE_K = int(args.candidate_pool_size)

    guard = EvidenceAnswerGuard(
        min_coverage=args.answer_guard_coverage,
        wrong_answer_penalty=args.premature_answer_penalty,
    )
    controller = AdaptiveKLController(
        base_coef=args.kl_coef,
        high_watermark=args.kl_high_watermark,
        recovery_watermark=args.kl_recovery_watermark,
        max_coef=args.kl_max_coef,
    )
    trace_path = (
        Path(args.trace_path)
        if args.trace_path
        else Path(args.save_path).resolve().parent / "trajectory_v7.jsonl"
    )
    trace_writer = TraceWriterV7(trace_path)
    metrics_writer = MetricsWriterV7(
        trace_path.with_name("online_metrics_v7.jsonl")
    )
    contract_path = Path(args.save_path).resolve().parent / "v7_contract.json"
    if contract_path.exists():
        raise FileExistsError(f"refusing to overwrite {contract_path}")
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": "acec_training_v7",
        "version": 70,
        "git_commit": _git_commit(),
        "selector": {
            "strategy": args.selector_strategy_v7,
            "artifact": args.selector_artifact_v7,
            "artifact_sha256": (
                selector_artifact.sha256 if selector_artifact else "baseline"
            ),
            "answer_value_artifact": args.answer_value_artifact_v7,
            "answer_value_artifact_sha256": (
                answer_value_artifact.sha256
                if answer_value_artifact is not None
                else "none"
            ),
            "candidate_pool_size": args.candidate_pool_size,
            "selected_k": args.selected_k,
            "semantic_hard_filters": False,
        },
        "belief": {
            "artifact": args.acec_artifact_v5,
            "artifact_sha256": _file_sha256(args.acec_artifact_v5),
            "nli_model": args.acec_nli_model,
            "action_embedder": args.e5_model_path,
        },
        "reward": {
            "mode": args.coverage_reward_mode,
            "predicted_gain_bonus": 0.0,
        },
        "training": vars(args),
    }
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[GRPO-RSF-vLLM-v7] contract={contract_path}")

    base_trainer.make_belief_factory = v5_entry.make_belief_factory_v5
    base_trainer.vllm_rollout_batch = partial(
        vllm_rollout_batch_v7,
        answer_guard=guard,
        trace_writer=trace_writer,
        selector_runtime=selector_runtime,
        coverage_reward_mode=args.coverage_reward_mode,
        metrics_writer=metrics_writer,
    )
    base_trainer.grpo_loss_fn = partial(
        grpo_loss_v7,
        controller=controller,
        answer_kl_multiplier=args.answer_kl_multiplier,
    )
    base_trainer.train(args)


if __name__ == "__main__":
    main()
