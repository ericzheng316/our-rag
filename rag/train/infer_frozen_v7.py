"""Frozen ACEC v7 inference for Gate-4 answer and retrieval diagnostics.

The trajectory contains selected document ids/titles. Official HotpotQA
sentence-level SP/Joint scoring still requires the native distractor
provenance adapter from the separate v6.3 evaluation path; this file does not
invent sentence ids for wiki chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from datasets import load_dataset
from vllm import LLM
from vllm.lora.request import LoRARequest


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
RAG_TRAIN = REPO_ROOT / "rag" / "train"
for path in (RAG_SRC, RAG_TRAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import grpo_rsf_simple as simple_trainer  # noqa: E402
from acec_v6_contract import EvidenceAnswerGuard  # noqa: E402
from grpo_estimator_v2 import RewardConfig  # noqa: E402
from grpo_rsf_vllm_v5 import make_belief_factory_v5  # noqa: E402
from grpo_rsf_vllm_v7 import TraceWriterV7, vllm_rollout_batch_v7  # noqa: E402
from belief.acec.answer_value_v7 import (  # noqa: E402
    load_answer_value_artifact_v7,
)
from belief.acec.runtime_v7 import BeliefSelectorRuntimeV7  # noqa: E402
from belief.acec.set_selector_v7 import (  # noqa: E402
    SequentialSetSelectorV7,
    load_selector_artifact_v7,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_questions", type=int, default=256)
    parser.add_argument("--question_batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vllm_gpu_mem_frac", type=float, default=0.55)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--selected_k", type=int, default=5)
    parser.add_argument(
        "--selector_strategy",
        choices=("learned", "relevance", "fixed_rel_plus_coverage"),
        default="learned",
    )
    parser.add_argument("--selector_artifact")
    parser.add_argument("--answer_value_artifact")
    parser.add_argument("--fixed_coverage_weight", type=float, default=1.0)
    parser.add_argument(
        "--coverage_reward_mode",
        choices=("legacy_coverage_aux", "strict_pbrs"),
        default="legacy_coverage_aux",
    )
    parser.add_argument("--acec_artifact_v5", required=True)
    parser.add_argument("--e5_model_path", required=True)
    parser.add_argument(
        "--acec_nli_model", default="cross-encoder/nli-deberta-v3-base"
    )
    parser.add_argument("--acec_tau_new", type=float)
    return parser


def _question_payload(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    supporting = item.get("supporting_facts") or {}
    answer_values = item.get("golden_answers")
    if answer_values is None:
        answer = item.get("answer")
        answer_values = [answer] if answer is not None else []
    return {
        "id": str(item.get("id") or item.get("_id") or index),
        "question": str(item["question"]),
        "sf_titles": (
            list(supporting.get("title") or ())
            if isinstance(supporting, dict)
            else []
        ),
        "golden_answers": [str(value) for value in answer_values],
    }


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
    if args.selector_strategy == "learned" and (
        not args.selector_artifact or not args.answer_value_artifact
    ):
        raise ValueError(
            "learned v7 inference requires selector and answer-value artifacts"
        )
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    trace_path = output_dir / "trajectory_v7.jsonl"
    trace_writer = TraceWriterV7(trace_path)

    selector_artifact = (
        load_selector_artifact_v7(Path(args.selector_artifact).expanduser())
        if args.selector_artifact
        else None
    )
    answer_value = (
        load_answer_value_artifact_v7(
            Path(args.answer_value_artifact).expanduser()
        )
        if args.answer_value_artifact
        else None
    )
    selector = SequentialSetSelectorV7(
        artifact=selector_artifact,
        answer_value_artifact=answer_value,
        strategy=args.selector_strategy,
        fixed_coverage_weight=args.fixed_coverage_weight,
    )
    runtime = BeliefSelectorRuntimeV7(selector, selected_k=args.selected_k)
    belief_factory = make_belief_factory_v5(args)
    simple_trainer.RETRIEVE_K = int(args.candidate_pool_size)

    llm = LLM(
        model=args.model_path,
        enable_lora=bool(args.adapter),
        max_lora_rank=args.lora_rank,
        max_loras=2,
        gpu_memory_utilization=args.vllm_gpu_mem_frac,
        max_model_len=4096,
        dtype="bfloat16",
        logprobs_mode="raw_logprobs",
    )
    lora_request = (
        LoRARequest("frozen-v7", 1, str(Path(args.adapter).expanduser()))
        if args.adapter
        else None
    )
    dataset = load_dataset("json", data_files=args.data_path)["train"]
    if args.max_questions:
        dataset = dataset.select(
            range(min(int(args.max_questions), len(dataset)))
        )
    questions = [
        _question_payload(dict(item), index)
        for index, item in enumerate(dataset)
    ]
    guard = EvidenceAnswerGuard(
        min_coverage=0.50, wrong_answer_penalty=0.10
    )
    reward = RewardConfig()
    weighted_metrics: Dict[str, float] = {}
    total = 0
    for offset in range(0, len(questions), args.question_batch_size):
        batch = questions[offset : offset + args.question_batch_size]
        _, metrics = vllm_rollout_batch_v7(
            llm,
            lora_request,
            batch,
            args.n_samples,
            args.n_turns,
            belief_factory=belief_factory,
            rollout_temperature=args.temperature,
            reward_config=reward,
            collect_metrics=True,
            answer_guard=guard,
            trace_writer=trace_writer,
            selector_runtime=runtime,
            coverage_reward_mode=args.coverage_reward_mode,
        )
        weight = len(batch) * args.n_samples
        total += weight
        for key, value in metrics.items():
            weighted_metrics[key] = weighted_metrics.get(key, 0.0) + float(
                value
            ) * weight
    metrics = {
        key: value / max(total, 1) for key, value in weighted_metrics.items()
    }
    trace_rows = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                trace_rows.append(json.loads(line))
    question_lookup = {item["id"]: item for item in questions}
    predictions = []
    for row in trace_rows:
        question = str(row["question"])
        source = question_lookup[str(row["question_id"])]
        predictions.append(
            {
                "id": source["id"],
                "question": question,
                "sample_index": int(row["sample_index"]),
                "prediction": row.get("answer"),
                "golden_answers": source["golden_answers"],
                "events": row["events"],
            }
        )
    (output_dir / "predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
        ),
        encoding="utf-8",
    )
    result = {
        "schema": "acec_frozen_inference_v7",
        "questions": len(questions),
        "samples_per_question": args.n_samples,
        "selector_strategy": args.selector_strategy,
        "selector_artifact_sha256": (
            selector_artifact.sha256 if selector_artifact else "baseline"
        ),
        "answer_value_artifact_sha256": (
            answer_value.sha256 if answer_value else "none"
        ),
        "belief_artifact_sha256": _file_sha256(args.acec_artifact_v5),
        "candidate_pool_size": args.candidate_pool_size,
        "selected_k": args.selected_k,
        "metrics": metrics,
        "trace": str(trace_path),
        "predictions": str(output_dir / "predictions.jsonl"),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
