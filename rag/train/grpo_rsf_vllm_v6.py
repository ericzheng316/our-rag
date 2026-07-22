"""ACEC-v6 training entry point with guardrails and rollout traces.

V1-v5 files remain unchanged.  V6 adds three safeguards diagnosed from the
fixed Episode-50/75/100 evaluation:

1. wrong answers made below an evidence-coverage floor receive a small
   terminal penalty; correct zero-hop answers are never penalized;
2. final-answer turns use a stronger KL anchor and the global KL coefficient
   increases when measured KL crosses a high-water mark;
3. every rollout writes a compact question-level action/evidence trace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from vllm import SamplingParams

import grpo_rsf_vllm as base_trainer
import grpo_rsf_vllm_v5 as v5_entry
from acec_v6_contract import AdaptiveKLController, EvidenceAnswerGuard
from grpo_estimator_v6 import grpo_loss_v6


def _document_label(document: Dict[str, Any]) -> str:
    title = document.get("title")
    if title:
        return str(title)[:200]
    contents = str(document.get("contents") or "")
    return contents.splitlines()[0][:200] if contents else ""


class TraceWriterV6:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.episode = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite v6 trace: {self.path}")

    def write_episode(self, states: List[Dict[str, Any]]) -> None:
        self.episode += 1
        with self.path.open("a", encoding="utf-8") as handle:
            for state in states:
                payload = {
                    "episode": self.episode,
                    "question_index": state["qi"],
                    "sample_index": state["sample_index"],
                    "question": state["question"],
                    "answer": state.get("answer"),
                    "answer_correct": bool(state["answer_correct"]),
                    "answered": bool(state["answered"]),
                    "format_error": bool(state["format_error"]),
                    "retrieval_calls": int(state["retrieval_calls"]),
                    "total_reward": sum(float(turn[2]) for turn in state["turns"]),
                    "events": state["trace"],
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def vllm_rollout_batch_v6(
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
    trace_writer: TraceWriterV6,
):
    if belief_factory is None:
        raise ValueError("ACEC-v6 requires the gate-passing v5 belief runtime")
    if reward_config is None:
        raise ValueError("ACEC-v6 requires an explicit RewardConfig")

    states: List[Dict[str, Any]] = []
    for question_index, item in enumerate(questions):
        for sample_index in range(n_samples):
            states.append(
                {
                    "qi": question_index,
                    "sample_index": sample_index,
                    "question": item["question"],
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
                }
            )

    for turn_index in range(n_turns):
        active = [index for index, state in enumerate(states) if not state["done"]]
        if not active:
            break
        prompts = [base_trainer.apply_chat_template(states[index]["context"]) for index in active]
        sampling_params = [
            SamplingParams(
                temperature=states[index]["temperature"],
                max_tokens=512,
                stop_token_ids=[base_trainer.STOP_TOKEN],
                logprobs=1,
            )
            for index in active
        ]
        outputs = llm.generate(
            prompts, sampling_params, lora_request=lora_request, use_tqdm=False
        )

        pending_retrieval: List[int] = []
        for output_index, state_index in enumerate(active):
            state = states[state_index]
            completion = outputs[output_index].outputs[0]
            input_ids = torch.tensor(
                outputs[output_index].prompt_token_ids, dtype=torch.long, device="cuda"
            )
            output_ids = torch.tensor(
                completion.token_ids, dtype=torch.long, device="cuda"
            )
            engine_lp = base_trainer._vllm_sampled_token_logprobs(
                completion, device=input_ids.device
            )
            parsed = base_trainer.parse_step(
                f"Step {turn_index + 1}:\n{completion.text}"
            )

            if not parsed.get("analysis"):
                reward = -reward_config.format_error
                state["turns"].append(
                    (
                        input_ids,
                        output_ids,
                        reward,
                        engine_lp,
                        state["temperature"],
                        "format_error",
                    )
                )
                state["trace"].append(
                    {"turn": turn_index + 1, "kind": "format_error", "reward": reward}
                )
                state["format_error"] = True
                state["done"] = True
                continue

            if parsed.get("answer"):
                golds = questions[state["qi"]]["golden_answers"]
                correct = base_trainer.exact_match(parsed["answer"], golds)
                coverage = float(state["belief"].coverage_belief.coverage())
                adjustment = answer_guard.reward_adjustment(correct, coverage)
                reward = reward_config.answer_reward(correct) + adjustment
                state["belief"].turn(query=None, new_docs=[], is_answer=True)
                state["turns"].append(
                    (
                        input_ids,
                        output_ids,
                        reward,
                        engine_lp,
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
                        "reward": reward,
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
                state["_pending"] = (input_ids, output_ids, engine_lp, parsed)
                pending_retrieval.append(state_index)
            else:
                reward = -reward_config.format_error
                state["turns"].append(
                    (
                        input_ids,
                        output_ids,
                        reward,
                        engine_lp,
                        state["temperature"],
                        "format_error",
                    )
                )
                state["trace"].append(
                    {"turn": turn_index + 1, "kind": "format_error", "reward": reward}
                )
                state["format_error"] = True
                state["done"] = True

        if pending_retrieval:
            queries = [states[index]["_pending"][3]["query"] for index in pending_retrieval]
            documents_by_state = base_trainer._retrieve_batch(queries)
            for state_index, documents in zip(pending_retrieval, documents_by_state):
                state = states[state_index]
                input_ids, output_ids, engine_lp, parsed = state.pop("_pending")
                state["retrieval_calls"] += 1
                if not documents:
                    state["empty_retrievals"] += 1

                new_documents = []
                for document in documents:
                    if (
                        document.get("id") not in state["retrieved_ids"]
                        and len(state["retrieved_ids"]) < base_trainer.MAX_DOCS
                    ):
                        state["retrieved_ids"].append(document.get("id"))
                        new_documents.append(document)
                document_text = "\n".join(
                    str(document.get("contents") or "") for document in new_documents
                )

                found_before = set(state["found_sf"])
                gold_delta, state["found_sf"] = base_trainer.rsf_marginal(
                    document_text,
                    questions[state["qi"]]["sf_titles"],
                    state["found_sf"],
                )
                result = state["belief"].turn(
                    query=parsed["query"], new_docs=new_documents
                )
                reward = reward_config.retrieval_reward(result.delta_coverage)
                state["turns"].append(
                    (
                        input_ids,
                        output_ids,
                        reward,
                        engine_lp,
                        state["temperature"],
                        "retrieval",
                    )
                )
                state["trace"].append(
                    {
                        "turn": turn_index + 1,
                        "kind": "retrieval",
                        "analysis": parsed["analysis"][:512],
                        "query": parsed["query"],
                        "documents": [
                            {"id": document.get("id"), "title": _document_label(document)}
                            for document in new_documents
                        ],
                        "new_gold_sf_titles": sorted(set(state["found_sf"]) - found_before),
                        "gold_sf_delta": gold_delta,
                        "action": result.action.mode.value,
                        "target_slot": result.action.target_slot,
                        "coverage_before": result.coverage_before,
                        "coverage_after": result.coverage_after,
                        "delta_coverage": result.delta_coverage,
                        "stop_voi": result.stop_voi,
                        "slot_p": list(result.features.slot_p),
                        "slot_bound": list(result.features.slot_bound),
                        "reward": reward,
                    }
                )
                state["context"] += (
                    f"\nStep {turn_index + 1}:\n"
                    f"The problem analysis: {parsed['analysis']}\n"
                    f"The retrieval query: {parsed['query']}\n"
                    f"The retrieval documents: {document_text[:512]}"
                )
                if turn_index == n_turns - 1:
                    last = state["turns"][-1]
                    adjusted = last[2] - reward_config.format_error
                    state["turns"][-1] = (*last[:2], adjusted, *last[3:])
                    state["trace"][-1]["reward"] = adjusted
                    state["trace"][-1]["max_turn_penalty"] = reward_config.format_error
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
        sf_recalls.append(len(state["found_sf"]) / len(titles) if titles else 0.0)
    answer_coverages = [
        float(state["answer_coverage"])
        for state in states
        if state["answer_coverage"] is not None
    ]
    metrics = {
        "answer_em": sum(bool(state["answer_correct"]) for state in states) / rollout_count,
        "answer_rate": sum(bool(state["answered"]) for state in states) / rollout_count,
        "gold_sf_recall": sum(sf_recalls) / rollout_count,
        "retrieval_calls": total_retrievals / rollout_count,
        "format_error_rate": sum(bool(state["format_error"]) for state in states) / rollout_count,
        "empty_retrieval_rate": sum(state["empty_retrievals"] for state in states)
        / max(total_retrievals, 1),
        "premature_guard_rate": sum(bool(state["guard_applied"]) for state in states)
        / rollout_count,
        "mean_answer_coverage": sum(answer_coverages) / max(len(answer_coverages), 1),
    }
    base_trainer.log.info(
        "[ACEC-v6] guard_rate=%.3f mean_answer_coverage=%.3f trace_episode=%d",
        metrics["premature_guard_rate"],
        metrics["mean_answer_coverage"],
        trace_writer.episode,
    )
    return trajectories, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = v5_entry.build_parser()
    parser.description = "ACEC-v6 guarded training with traces and adaptive KL"
    parser.add_argument("--answer_guard_coverage", type=float, default=0.50)
    parser.add_argument("--premature_answer_penalty", type=float, default=0.10)
    parser.add_argument("--answer_kl_multiplier", type=float, default=2.0)
    parser.add_argument("--kl_high_watermark", type=float, default=0.08)
    parser.add_argument("--kl_recovery_watermark", type=float, default=0.05)
    parser.add_argument("--kl_max_coef", type=float, default=0.05)
    parser.add_argument("--trace_path", type=str, default=None)
    return parser


def _write_contract(
    args: argparse.Namespace,
    guard: EvidenceAnswerGuard,
    controller: AdaptiveKLController,
    trace_path: Path,
) -> Path:
    output_root = Path(args.save_path).resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "v6_contract.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v6 contract: {path}")
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    payload = {
        "schema": "acec_training_v6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "guard": {
            "min_coverage": guard.min_coverage,
            "wrong_answer_penalty": guard.wrong_answer_penalty,
            "correct_answers_are_exempt": True,
        },
        "adaptive_kl": controller.snapshot(),
        "answer_kl_multiplier": args.answer_kl_multiplier,
        "trace_path": str(trace_path),
        "training": vars(args),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = build_parser().parse_args()
    args.use_acec = True
    v5_entry._load_checked_artifact(args.acec_artifact_v5)
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
    trace_path = Path(args.trace_path) if args.trace_path else Path(args.save_path).resolve().parent / "trajectory_v6.jsonl"
    trace_writer = TraceWriterV6(trace_path)
    contract_path = _write_contract(args, guard, controller, trace_path)
    print(f"[GRPO-RSF-vLLM-v6] contract={contract_path}")

    from belief.acec.runtime_v5 import RUNTIME_DIAGNOSTICS_V5

    RUNTIME_DIAGNOSTICS_V5.reset()
    base_trainer.make_belief_factory = v5_entry.make_belief_factory_v5
    base_trainer.vllm_rollout_batch = partial(
        vllm_rollout_batch_v6,
        answer_guard=guard,
        trace_writer=trace_writer,
    )
    base_trainer.grpo_loss_fn = partial(
        grpo_loss_v6,
        controller=controller,
        answer_kl_multiplier=args.answer_kl_multiplier,
    )
    base_trainer.train(args)
    print(
        "[GRPO-RSF-vLLM-v6] summary="
        + json.dumps(
            {
                "trace": str(trace_path),
                "guard": {
                    "min_coverage": guard.min_coverage,
                    "wrong_answer_penalty": guard.wrong_answer_penalty,
                },
                "adaptive_kl": controller.snapshot(),
                "runtime": RUNTIME_DIAGNOSTICS_V5.summary(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
