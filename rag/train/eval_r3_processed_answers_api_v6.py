"""Cached R3 processed-EM evaluator for multi-arm ACEC experiments.

This is an isolated wrapper around ``eval_r3_processed_answers_api.py``.  It
preserves that evaluator's scoring and output schema while ensuring identical
``(question, prediction)`` pairs share one extractor request, even when several
checkpoint variants emit the same answer.  API usage is charged only to the
first record; cache-hit records retain the same candidates with zero usage.
"""

from __future__ import annotations

from concurrent.futures import Future
import copy
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Mapping

import eval_r3_processed_answers_api as legacy


def extraction_cache_key(question: str, prediction: str) -> str:
    payload = json.dumps(
        [str(question), str(prediction)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extraction_from_scored_record(record: Mapping[str, Any]) -> Dict[str, Any] | None:
    status = str(record.get("status", ""))
    if status not in {"extracted", "extraction_failed_fallback"}:
        return None
    return {
        "ok": status == "extracted",
        "candidates": [str(value) for value in record.get("candidate_answers", [])],
        "raw_output": str(record.get("extractor_output", "")),
        "attempts": int(record.get("extractor_attempts", 0) or 0),
        "error_type": str(record.get("extractor_error_type", "")),
        "usage": dict(record.get("usage") or {}),
    }


class SingleFlightExtractionCache:
    """Coalesce concurrent duplicate extractor requests into one request."""

    def __init__(self, request: Callable[..., Dict[str, Any]]) -> None:
        self._request = request
        self._lock = threading.Lock()
        self._futures: Dict[str, Future] = {}

    def seed(self, question: str, prediction: str, extraction: Mapping[str, Any]) -> None:
        key = extraction_cache_key(question, prediction)
        completed = Future()
        completed.set_result(copy.deepcopy(dict(extraction)))
        with self._lock:
            self._futures.setdefault(key, completed)

    def __call__(
        self,
        client: Any,
        *,
        model: str,
        question: str,
        answer: str,
        max_tokens: int,
        retries: int,
        provider_routing: bool,
    ) -> Dict[str, Any]:
        key = extraction_cache_key(question, answer)
        with self._lock:
            future = self._futures.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._futures[key] = future

        assert future is not None
        if owner:
            try:
                result = self._request(
                    client,
                    model=model,
                    question=question,
                    answer=answer,
                    max_tokens=max_tokens,
                    retries=retries,
                    provider_routing=provider_routing,
                )
            except BaseException as exc:
                future.set_exception(exc)
                raise
            future.set_result(copy.deepcopy(result))
        else:
            result = future.result()

        payload = copy.deepcopy(result)
        payload["cache_key"] = key
        payload["cache_hit"] = not owner
        if not owner:
            payload["attempts"] = 0
            payload["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        return payload


def _seed_from_progress(cache: SingleFlightExtractionCache, progress_path: Path) -> int:
    if not progress_path.exists():
        return 0
    seeded = 0
    for record in legacy._read_jsonl(progress_path):
        extraction = extraction_from_scored_record(record)
        if extraction is None:
            continue
        cache.seed(
            str(record.get("question", "")),
            str(record.get("prediction") or ""),
            extraction,
        )
        seeded += 1
    return seeded


def main() -> None:
    args = legacy.build_parser().parse_args()

    original_request = legacy.request_candidate_answers
    original_score = legacy.score_prediction
    cache = SingleFlightExtractionCache(original_request)
    seeded = _seed_from_progress(cache, args.output_dir / "api_progress.jsonl")
    if seeded:
        legacy.log.info("[cache] seeded=%d completed extractions", seeded)

    def score_with_cache_metadata(**kwargs: Any) -> Dict[str, Any]:
        extraction = kwargs.get("extraction")
        row = original_score(**kwargs)
        if extraction is None:
            row["extractor_cache_key"] = ""
            row["extractor_cache_hit"] = False
        else:
            row["extractor_cache_key"] = str(extraction.get("cache_key", ""))
            row["extractor_cache_hit"] = bool(extraction.get("cache_hit", False))
        return row

    legacy.request_candidate_answers = cache
    legacy.score_prediction = score_with_cache_metadata
    legacy.main()


if __name__ == "__main__":
    main()
