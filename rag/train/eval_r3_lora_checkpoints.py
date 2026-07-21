"""Matched held-out evaluation for frozen R3 and GRPO LoRA checkpoints.

The evaluator reuses the validated batched multi-turn rollout path but does
not construct ACEC/NLI modules: belief rewards are unnecessary at inference.
One vLLM base engine serves the frozen model and every requested adapter so
all variants see the same questions, retriever, prompt format, and decoder.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import random
import re
import string
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple


logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("eval_r3_lora_checkpoints")


def normalize_answer(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match_score(prediction: str, golds: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gold) for gold in golds))


def substring_match_score(prediction: str, golds: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(
        any(
            (gold_norm := normalize_answer(gold)) and gold_norm in pred
            for gold in golds
        )
    )


def _single_f1(prediction: str, gold: str) -> float:
    pred_normalized = normalize_answer(prediction)
    gold_normalized = normalize_answer(gold)
    pred_tokens = pred_normalized.split()
    gold_tokens = gold_normalized.split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    special = {"yes", "no", "noanswer"}
    if (
        pred_normalized in special or gold_normalized in special
    ) and pred_normalized != gold_normalized:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def f1_score(prediction: str, golds: Sequence[str]) -> float:
    return max((_single_f1(prediction, gold) for gold in golds), default=0.0)


class TaggedAnswers(list):
    """List-compatible gold answers carrying a stable evaluation id."""

    def __init__(self, values: Iterable[str], question_id: str):
        super().__init__(values)
        self.question_id = question_id


class AnswerCollector:
    """Drop-in replacement for the trainer's exact_match function."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def __call__(self, prediction: str, golds: Sequence[str]) -> bool:
        record = {
            "id": str(getattr(golds, "question_id", "unknown")),
            "prediction": prediction,
            "golden_answers": list(golds),
            "em": exact_match_score(prediction, golds),
            "f1": f1_score(prediction, golds),
            "sub_em": substring_match_score(prediction, golds),
            "answered": True,
        }
        self.records.append(record)
        return bool(record["em"])


def _supporting_fact_titles(row: Dict[str, Any]) -> List[str]:
    sf = row.get("supporting_facts")
    if not sf:
        sf = (row.get("metadata") or {}).get("supporting_facts")
    if isinstance(sf, dict):
        return [str(value) for value in sf.get("title", [])]
    return []


def load_heldout_questions(path: str, max_questions: int, seed: int) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"held-out dataset is empty: {path}")
    if max_questions <= 0 or max_questions > len(rows):
        max_questions = len(rows)
    rng = random.Random(seed)
    selected = rng.sample(rows, max_questions)
    questions = []
    for index, row in enumerate(selected):
        qid = str(row.get("id", f"heldout_{index}"))
        question = row.get("question") or row.get("problem")
        golds = row.get("golden_answers") or []
        if not question or not golds:
            raise ValueError(f"held-out row {qid} lacks question or golden_answers")
        questions.append(
            {
                "id": qid,
                "question": str(question),
                "golden_answers": TaggedAnswers(golds, qid),
                "sf_titles": _supporting_fact_titles(row),
            }
        )
    return questions


def parse_adapter(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must be NAME=/absolute/path")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("adapter must be NAME=/absolute/path")
    return name, path


def _weighted_merge(accumulator: Dict[str, float], metrics: Dict[str, float], weight: int) -> None:
    for key, value in metrics.items():
        accumulator[key] = accumulator.get(key, 0.0) + float(value) * weight


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter", action="append", type=parse_adapter, default=[])
    parser.add_argument("--max_questions", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--vllm_gpu_mem_frac", type=float, default=0.60)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    parser.add_argument("--retrieve_url", default=os.environ.get("RETRIEVE_URL", ""))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.retrieve_url:
        raise ValueError("--retrieve_url or RETRIEVE_URL is required")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("--temperature must be in [0, 2]")

    adapters = list(args.adapter)
    for name, path in adapters:
        adapter_path = Path(path)
        for required in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter_path / required).is_file():
                raise FileNotFoundError(f"{name} missing {required}: {adapter_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    if results_path.exists():
        raise FileExistsError(f"refusing to overwrite prior evaluation: {results_path}")

    questions = load_heldout_questions(args.data_path, args.max_questions, args.seed)
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in questions:
            handle.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "golden_answers": list(item["golden_answers"]),
                        "sf_titles": item["sf_titles"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Heavy imports stay inside main so metric helpers remain unit-testable on
    # machines without the remote vLLM/CUDA runtime.
    import gc
    import torch
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    import grpo_rsf_simple
    import grpo_rsf_vllm as rollout_impl

    grpo_rsf_simple.RETRIEVE_URL = args.retrieve_url
    variants: List[Tuple[str, Any]] = [("frozen_r3", None)]
    for adapter_id, (name, path) in enumerate(adapters, start=1):
        variants.append((name, LoRARequest(name, adapter_id, path)))

    log.info(
        "[eval] questions=%d batch=%d temperature=%.3f variants=%s",
        len(questions),
        args.batch_size,
        args.temperature,
        [name for name, _ in variants],
    )
    llm = LLM(
        model=args.model_path,
        enable_lora=bool(adapters),
        max_lora_rank=args.max_lora_rank,
        max_loras=2,
        gpu_memory_utilization=args.vllm_gpu_mem_frac,
        max_model_len=4096,
        dtype="bfloat16",
        logprobs_mode="raw_logprobs",
        seed=args.seed,
        disable_log_stats=True,
    )

    original_exact_match = rollout_impl.exact_match
    result_models = []
    per_model_records: Dict[str, Dict[str, Dict[str, Any]]] = {}
    try:
        for variant_name, lora_request in variants:
            collector = AnswerCollector()
            rollout_impl.exact_match = collector
            aggregate: Dict[str, float] = {}
            total_rollouts = 0
            started = time.monotonic()
            for offset in range(0, len(questions), args.batch_size):
                batch = questions[offset : offset + args.batch_size]
                trajectories, metrics = rollout_impl.vllm_rollout_batch(
                    llm,
                    lora_request,
                    batch,
                    n_samples=1,
                    n_turns=args.n_turns,
                    belief_factory=None,
                    rollout_temperature=args.temperature,
                    collect_metrics=True,
                )
                weight = len(batch)
                _weighted_merge(aggregate, metrics, weight)
                total_rollouts += weight
                del trajectories
                gc.collect()
                torch.cuda.empty_cache()
                log.info(
                    "[eval] model=%s done=%d/%d raw_EM=%.4f SF=%.4f retrievals=%.3f",
                    variant_name,
                    min(offset + len(batch), len(questions)),
                    len(questions),
                    metrics["answer_em"],
                    metrics["gold_sf_recall"],
                    metrics["retrieval_calls"],
                )

            answered_by_id = {record["id"]: record for record in collector.records}
            complete_records = []
            for item in questions:
                record = answered_by_id.get(item["id"])
                if record is None:
                    record = {
                        "id": item["id"],
                        "prediction": None,
                        "golden_answers": list(item["golden_answers"]),
                        "em": 0.0,
                        "f1": 0.0,
                        "sub_em": 0.0,
                        "answered": False,
                    }
                complete_records.append(record)

            predictions_path = output_dir / f"predictions_{_safe_name(variant_name)}.jsonl"
            with predictions_path.open("w", encoding="utf-8") as handle:
                for record in complete_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            indexed_records = {record["id"]: record for record in complete_records}
            per_model_records[variant_name] = indexed_records

            averaged = {key: value / total_rollouts for key, value in aggregate.items()}
            averaged.update(
                {
                    "raw_f1": sum(record["f1"] for record in complete_records) / total_rollouts,
                    "sub_em": sum(record["sub_em"] for record in complete_records) / total_rollouts,
                    "elapsed_seconds": time.monotonic() - started,
                    "n_questions": total_rollouts,
                    "predictions": str(predictions_path),
                }
            )
            result_models.append({"name": variant_name, "metrics": averaged})
            log.info("[eval] COMPLETE model=%s metrics=%s", variant_name, averaged)
    finally:
        rollout_impl.exact_match = original_exact_match

    base_records = per_model_records["frozen_r3"]
    paired = {}
    for variant_name, _ in variants[1:]:
        records = per_model_records[variant_name]
        em_diffs = [records[qid]["em"] - base_records[qid]["em"] for qid in base_records]
        f1_diffs = [records[qid]["f1"] - base_records[qid]["f1"] for qid in base_records]
        paired[variant_name] = {
            "raw_em_delta": sum(em_diffs) / len(em_diffs),
            "raw_f1_delta": sum(f1_diffs) / len(f1_diffs),
            "em_wins": sum(diff > 0 for diff in em_diffs),
            "em_ties": sum(diff == 0 for diff in em_diffs),
            "em_losses": sum(diff < 0 for diff in em_diffs),
        }

    payload = {
        "config": {
            "model_path": args.model_path,
            "data_path": args.data_path,
            "max_questions": len(questions),
            "batch_size": args.batch_size,
            "n_turns": args.n_turns,
            "temperature": args.temperature,
            "seed": args.seed,
            "retrieve_url": args.retrieve_url,
            "manifest": str(manifest_path),
            "adapters": dict(adapters),
        },
        "models": result_models,
        "paired_vs_frozen_r3": paired,
    }
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    log.info("[eval] RESULTS %s", results_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
