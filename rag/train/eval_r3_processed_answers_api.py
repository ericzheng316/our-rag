"""Post-process saved R3/ACEC answers with the R3-RAG processed-EM protocol.

This is an inference-free evaluator: it reads the manifest and prediction
JSONL files produced by ``eval_r3_lora_checkpoints.py`` and calls an
OpenAI-compatible Qwen2.5-72B-Instruct endpoint only for short-answer
extraction. API credentials are read exclusively from an environment
variable and are never persisted.

The extraction prompt, parsing rule, raw-EM short circuit, one-retry fallback,
normalization, and candidate scoring mirror the official R3-RAG
``benchmark/R3-RAG/src/cal_metric.py`` implementation. In addition, this
script keeps resumable, auditable API outputs and paired bootstrap intervals.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import string
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("eval_r3_processed_answers_api")

R3_METRIC_SOURCE = (
    "https://github.com/Yuan-Li-FNLP/R3-RAG/blob/main/"
    "benchmark/R3-RAG/src/cal_metric.py"
)


def r3_em_prompt(question: str, answer: str) -> str:
    """Return the official R3-RAG candidate-answer extraction prompt."""

    return f"""
Your task is to extract a list of short candidate answers from a detailed sentence answer for a given question.

### Instructions:
- Provide multiple candidate answers whenever possible.
- All candidate answers must have exactly the same meaning and precisely answer the original question.
- For dates or numbers, include multiple common formats.
- For people's names or places, include common variations, abbreviations, or full names.
- For yes/no questions, include ["Yes"] or ["No"] and other synonymous expressions.

### Important formatting guidelines (must strictly follow):
- Your output must be a Python-style list of strings, for example: ["Answer1", "Answer2", "Answer3"]
- Ensure your output can be directly parsed by Python's `ast.literal_eval()` function.
- Do NOT mix quotation marks improperly. Maintain consistent and proper quotation usage to ensure your list is valid JSON format.

### Examples:

Example 1 (Date Question):
Question: When was the People's Republic of China founded?
Answer: The founding ceremony of the People's Republic of China was officially held on October 1st, 1949.
Candidate Answers: ["October 1st, 1949", "1949-10-01", "1949.10.01", "Oct 1, 1949"]

Example 2 (Person Name):
Question: Who wrote the novel "1984"?
Answer: "1984" is a novel authored by the British writer George Orwell, whose real name was Eric Arthur Blair.
Candidate Answers: ["George Orwell", "Eric Arthur Blair"]

Example 3 (Place Name):
Question: What is the capital of the United States?
Answer: The capital city of the United States is Washington, D.C., commonly known as Washington DC or simply Washington.
Candidate Answers: ["Washington, D.C.", "Washington DC", "Washington"]

Example 4 (Yes/No Question - positive):
Question: Did Albert Einstein develop the theory of relativity?
Answer: Yes, Albert Einstein is credited with developing the theory of relativity.
Candidate Answers: ["Yes"]

Example 5 (Yes/No Question - negative):
Question: Is Mercury the farthest planet from the Sun?
Answer: No, Mercury is actually the closest planet to the Sun, not the farthest.
Candidate Answers: ["No"]

### Now, based on the following question and answer, carefully provide the candidate answers as instructed (strictly following the JSON and quotation guidelines above):

Question: {question}
Answer: {answer}
Candidate Answers:
"""


def extract_candidate_answers_strict(llm_output: Any) -> List[str] | None:
    """Parse the first Python-style list exactly as the R3 evaluator does."""

    if not isinstance(llm_output, str) or not llm_output.strip():
        return None
    match = re.search(r"\[.*?\]", llm_output, re.DOTALL)
    if not match:
        return None
    try:
        candidate_list = ast.literal_eval(match.group())
    except (ValueError, SyntaxError):
        return None
    if not isinstance(candidate_list, list):
        return None
    parsed: List[str] = []
    for item in candidate_list:
        if not isinstance(item, (str, int, float)):
            return None
        parsed.append(str(item).strip())
    return parsed


def normalize_answer(value: Any) -> str:
    """Apply the R3/HotpotQA lower/punctuation/article/space normalization."""

    text = str(value or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _r3_f1(normalized_golds: Sequence[str], normalized_prediction: str) -> float:
    gold_counters = [Counter(gold.split()) for gold in normalized_golds]
    prediction_counter = Counter(normalized_prediction.split())
    best = 0.0
    for gold_counter in gold_counters:
        common = sum((gold_counter & prediction_counter).values())
        if common == 0:
            continue
        precision = common / sum(prediction_counter.values())
        recall = common / sum(gold_counter.values())
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def compute_r3_em_f1(golds: Sequence[str], predictions: Sequence[str]) -> Tuple[float, float]:
    """Compute R3 processed EM and max candidate F1."""

    normalized_golds = [normalize_answer(gold) for gold in golds]
    normalized_predictions = [normalize_answer(prediction) for prediction in predictions]
    em = float(
        any(prediction == gold for prediction in normalized_predictions for gold in normalized_golds)
    )
    f1 = max(
        (_r3_f1(normalized_golds, prediction) for prediction in normalized_predictions),
        default=0.0,
    )
    return em, f1


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be NAME=/absolute/path.jsonl")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("prediction must be NAME=/absolute/path.jsonl")
    return name, Path(path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _index_unique(rows: Iterable[Dict[str, Any]], path: Path) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("id", ""))
        if not qid:
            raise ValueError(f"row without id in {path}")
        if qid in indexed:
            raise ValueError(f"duplicate id {qid!r} in {path}")
        indexed[qid] = row
    return indexed


def load_inputs(
    manifest_path: Path,
    prediction_paths: Sequence[Tuple[str, Path]],
) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Load and validate matched manifest/prediction files."""

    manifest_rows = _read_jsonl(manifest_path)
    manifest = _index_unique(manifest_rows, manifest_path)
    order = [str(row["id"]) for row in manifest_rows]
    if not order:
        raise ValueError(f"empty manifest: {manifest_path}")

    all_predictions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, path in prediction_paths:
        if name in all_predictions:
            raise ValueError(f"duplicate prediction name: {name}")
        indexed = _index_unique(_read_jsonl(path), path)
        if set(indexed) != set(manifest):
            missing = sorted(set(manifest) - set(indexed))[:5]
            extra = sorted(set(indexed) - set(manifest))[:5]
            raise ValueError(f"{name} id mismatch; missing={missing} extra={extra}")
        for qid in order:
            expected_golds = [str(value) for value in manifest[qid].get("golden_answers", [])]
            observed_golds = [str(value) for value in indexed[qid].get("golden_answers", [])]
            if expected_golds != observed_golds:
                raise ValueError(f"{name}/{qid} golden_answers differ from manifest")
        all_predictions[name] = indexed
    return order, manifest, all_predictions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        return "".join(parts)
    return ""


def request_candidate_answers(
    client: Any,
    *,
    model: str,
    question: str,
    answer: str,
    max_tokens: int,
    retries: int,
    provider_routing: bool,
) -> Dict[str, Any]:
    """Call the extraction endpoint, retrying parse/API failures."""

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": r3_em_prompt(question, answer)},
    ]
    extra_body = None
    if provider_routing:
        extra_body = {
            "provider": {
                "only": [],
                "order": [],
                "sort": None,
                "input_price_range": [],
                "output_price_range": [],
                "input_length_range": [],
                "output_length_range": [],
                "throughput_range": [],
                "latency_range": [],
            }
        }

    last_output = ""
    last_error_type = ""
    for attempt in range(1, retries + 2):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**kwargs)
            if not getattr(response, "choices", None):
                raise ValueError("response has no choices")
            last_output = _message_text(response.choices[0].message)
            candidates = extract_candidate_answers_strict(last_output)
            usage = getattr(response, "usage", None)
            usage_payload = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            if candidates is not None:
                return {
                    "ok": True,
                    "candidates": candidates,
                    "raw_output": last_output,
                    "attempts": attempt,
                    "error_type": "",
                    "usage": usage_payload,
                }
            last_error_type = "CandidateParseError"
        except Exception as exc:  # API SDK exceptions differ by provider.
            last_error_type = type(exc).__name__
        if attempt <= retries:
            time.sleep(min(30.0, 2.0 ** (attempt - 1) + random.random()))

    return {
        "ok": False,
        "candidates": [],
        "raw_output": last_output,
        "attempts": retries + 1,
        "error_type": last_error_type,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def score_prediction(
    *,
    variant: str,
    qid: str,
    manifest_row: Mapping[str, Any],
    prediction_row: Mapping[str, Any],
    extraction: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Apply the R3 raw-EM shortcut or processed candidate scoring."""

    prediction_value = prediction_row.get("prediction")
    answered = bool(prediction_row.get("answered", prediction_value is not None))
    prediction = str(prediction_value or "")
    golds = [str(value) for value in manifest_row.get("golden_answers", [])]
    direct_em, direct_f1 = compute_r3_em_f1(golds, [prediction] if answered else [])

    if not answered:
        status = "no_answer"
        candidates: List[str] = []
        processed_em = 0.0
        processed_f1 = 0.0
        extraction_payload: Mapping[str, Any] = {}
    elif direct_em == 1.0:
        status = "raw_exact_short_circuit"
        candidates = []
        processed_em = 1.0
        processed_f1 = 1.0
        extraction_payload = {}
    else:
        if extraction is None:
            raise ValueError(f"missing extraction for {variant}/{qid}")
        extraction_payload = extraction
        candidates = [str(value) for value in extraction.get("candidates", [])]
        status = "extracted" if extraction.get("ok") else "extraction_failed_fallback"
        # Official R3 appends the normalized original answer to extracted
        # candidates, and falls back to the original answer after final failure.
        processed_em, processed_f1 = compute_r3_em_f1(golds, candidates + [prediction])

    return {
        "variant": variant,
        "id": qid,
        "question": str(manifest_row.get("question", "")),
        "prediction": prediction_value,
        "golden_answers": golds,
        "answered": answered,
        "input_hotpot_em": float(prediction_row.get("em", 0.0) or 0.0),
        "input_hotpot_f1": float(prediction_row.get("f1", 0.0) or 0.0),
        "r3_direct_em": direct_em,
        "r3_direct_f1": direct_f1,
        "r3_processed_em": processed_em,
        "r3_processed_f1": processed_f1,
        "status": status,
        "candidate_answers": candidates,
        "original_answer_included_in_scoring": answered,
        "extractor_output": extraction_payload.get("raw_output", ""),
        "extractor_attempts": int(extraction_payload.get("attempts", 0) or 0),
        "extractor_error_type": str(extraction_payload.get("error_type", "")),
        "usage": dict(extraction_payload.get("usage", {})),
    }


def summarize_variant(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize zero records")
    n = len(records)

    def mean(key: str) -> float:
        return sum(float(record[key]) for record in records) / n

    api_records = [record for record in records if int(record["extractor_attempts"]) > 0]
    extracted = [record for record in records if record["status"] == "extracted"]
    failed = [record for record in records if record["status"] == "extraction_failed_fallback"]
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage = {
        key: sum(int((record.get("usage") or {}).get(key, 0) or 0) for record in records)
        for key in usage_keys
    }
    return {
        "n": n,
        "answer_rate": sum(bool(record["answered"]) for record in records) / n,
        "input_hotpot_em": mean("input_hotpot_em"),
        "input_hotpot_f1": mean("input_hotpot_f1"),
        "r3_direct_em": mean("r3_direct_em"),
        "r3_direct_f1": mean("r3_direct_f1"),
        "r3_processed_em": mean("r3_processed_em"),
        "r3_processed_f1": mean("r3_processed_f1"),
        "raw_exact_short_circuits": sum(
            record["status"] == "raw_exact_short_circuit" for record in records
        ),
        "api_requests_completed": len(api_records),
        "successful_extractions": len(extracted),
        "failed_extractions": len(failed),
        "extraction_failure_rate": len(failed) / len(api_records) if api_records else 0.0,
        "mean_candidate_answers_on_success": (
            sum(len(record["candidate_answers"]) for record in extracted) / len(extracted)
            if extracted
            else 0.0
        ),
        "usage": usage,
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take percentile of empty values")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_interval(
    differences: Sequence[float], *, samples: int, seed: int
) -> Dict[str, Any]:
    if not differences:
        raise ValueError("paired differences are empty")
    observed = sum(differences) / len(differences)
    if samples <= 0:
        return {"delta": observed, "ci95": None, "bootstrap_samples": 0}
    rng = random.Random(seed)
    n = len(differences)
    boot = []
    for _ in range(samples):
        boot.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    return {
        "delta": observed,
        "ci95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "bootstrap_samples": samples,
    }


def paired_comparison(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    baseline_by_id = {str(record["id"]): record for record in baseline}
    candidate_by_id = {str(record["id"]): record for record in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired variants have different ids")
    result: Dict[str, Any] = {}
    for offset, metric in enumerate(
        ("input_hotpot_em", "input_hotpot_f1", "r3_processed_em", "r3_processed_f1")
    ):
        differences = [
            float(candidate_by_id[qid][metric]) - float(baseline_by_id[qid][metric])
            for qid in baseline_by_id
        ]
        metric_result = paired_bootstrap_interval(
            differences,
            samples=bootstrap_samples,
            seed=seed + offset,
        )
        metric_result.update(
            {
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
            }
        )
        result[metric] = metric_result
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_named_path,
        required=True,
        help="Repeat NAME=/absolute/path.jsonl; the first variant is the paired baseline.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key_env", default="AIPING_API_KEY")
    parser.add_argument("--model", default="Qwen2.5-72B-Instruct")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--provider_routing", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.base_url:
        raise ValueError("--base_url or OPENAI_BASE_URL is required")
    if args.workers <= 0 or args.max_tokens <= 0 or args.retries < 0:
        raise ValueError("workers/max_tokens must be positive and retries non-negative")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap_samples must be non-negative")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    for _, path in args.prediction:
        if not path.is_file():
            raise FileNotFoundError(path)

    order, manifest, predictions = load_inputs(args.manifest, args.prediction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    progress_path = args.output_dir / "api_progress.jsonl"
    metadata_path = args.output_dir / "request_metadata.json"
    if results_path.exists():
        raise FileExistsError(f"refusing to overwrite completed evaluation: {results_path}")

    prompt_hash = hashlib.sha256(r3_em_prompt("{question}", "{answer}").encode()).hexdigest()
    metadata = {
        "metric_source": R3_METRIC_SOURCE,
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "predictions": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in args.prediction
        },
        "base_url": args.base_url,
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "provider_routing": args.provider_routing,
        "prompt_sha256": prompt_hash,
        "seed": args.seed,
    }
    if metadata_path.exists():
        prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prior_metadata != metadata:
            raise ValueError(f"resume metadata mismatch: {metadata_path}")
    else:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    completed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if progress_path.exists():
        for row in _read_jsonl(progress_path):
            key = (str(row["variant"]), str(row["id"]))
            completed[key] = row
        log.info("[resume] loaded=%d from %s", len(completed), progress_path)

    api_jobs: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    immediate: List[Dict[str, Any]] = []
    for variant, _ in args.prediction:
        for qid in order:
            if (variant, qid) in completed:
                continue
            manifest_row = manifest[qid]
            prediction_row = predictions[variant][qid]
            prediction_value = prediction_row.get("prediction")
            answered = bool(prediction_row.get("answered", prediction_value is not None))
            direct_em, _ = compute_r3_em_f1(
                [str(value) for value in manifest_row.get("golden_answers", [])],
                [str(prediction_value or "")] if answered else [],
            )
            if not answered or direct_em == 1.0:
                immediate.append(
                    score_prediction(
                        variant=variant,
                        qid=qid,
                        manifest_row=manifest_row,
                        prediction_row=prediction_row,
                        extraction=None,
                    )
                )
            else:
                api_jobs.append((variant, qid, manifest_row, prediction_row))

    log.info(
        "[preflight] variants=%s questions=%d completed=%d immediate=%d api_jobs=%d",
        [name for name, _ in args.prediction],
        len(order),
        len(completed),
        len(immediate),
        len(api_jobs),
    )

    with progress_path.open("a", encoding="utf-8") as progress_handle:
        for row in immediate:
            progress_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed[(str(row["variant"]), str(row["id"]))] = row
        progress_handle.flush()

        if api_jobs:
            api_key = os.environ.get(args.api_key_env, "")
            if not api_key:
                raise RuntimeError(
                    f"API credential environment variable {args.api_key_env!r} is not set"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai package is required: pip install openai") from exc

            client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_jobs = {
                    executor.submit(
                        request_candidate_answers,
                        client,
                        model=args.model,
                        question=str(manifest_row.get("question", "")),
                        answer=str(prediction_row.get("prediction") or ""),
                        max_tokens=args.max_tokens,
                        retries=args.retries,
                        provider_routing=args.provider_routing,
                    ): (variant, qid, manifest_row, prediction_row)
                    for variant, qid, manifest_row, prediction_row in api_jobs
                }
                done = 0
                for future in as_completed(future_jobs):
                    variant, qid, manifest_row, prediction_row = future_jobs[future]
                    extraction = future.result()
                    row = score_prediction(
                        variant=variant,
                        qid=qid,
                        manifest_row=manifest_row,
                        prediction_row=prediction_row,
                        extraction=extraction,
                    )
                    progress_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress_handle.flush()
                    completed[(variant, qid)] = row
                    done += 1
                    if done == 1 or done % 25 == 0 or done == len(api_jobs):
                        log.info(
                            "[api] done=%d/%d latest=%s/%s status=%s attempts=%d",
                            done,
                            len(api_jobs),
                            variant,
                            qid,
                            row["status"],
                            row["extractor_attempts"],
                        )

    expected = len(order) * len(args.prediction)
    if len(completed) != expected:
        raise RuntimeError(f"incomplete processed records: {len(completed)}/{expected}")

    ordered_by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for variant, _ in args.prediction:
        records = [completed[(variant, qid)] for qid in order]
        ordered_by_variant[variant] = records
        output_path = args.output_dir / f"processed_predictions_{_safe_name(variant)}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    baseline_name = args.prediction[0][0]
    summaries = {
        variant: summarize_variant(ordered_by_variant[variant])
        for variant, _ in args.prediction
    }
    paired = {
        variant: paired_comparison(
            ordered_by_variant[baseline_name],
            ordered_by_variant[variant],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + index * 100,
        )
        for index, (variant, _) in enumerate(args.prediction[1:], start=1)
    }
    payload = {
        "config": metadata,
        "baseline": baseline_name,
        "summaries": summaries,
        "paired_vs_baseline": paired,
    }
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log.info("[complete] results=%s", results_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
