"""Compare serial and batched retriever calls for correctness and throughput."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import requests


def load_queries(path: str, limit: int) -> List[str]:
    queries = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            query = item.get("question") or item.get("problem")
            if query:
                queries.append(query)
            if len(queries) >= limit:
                break
    if not queries:
        raise ValueError(f"no question/problem strings found in {path}")
    return queries


def request_batch(url: str, queries: Sequence[str], timeout: float) -> Tuple[List[List[Dict]], float]:
    start = time.perf_counter()
    response = requests.post(
        url,
        json={"querys": list(queries)},
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) != len(queries):
        raise RuntimeError(
            f"retriever returned {len(payload) if isinstance(payload, list) else type(payload)} "
            f"results for {len(queries)} queries"
        )
    return payload, elapsed


def compare_results(
    serial: Sequence[Sequence[Dict]],
    batched: Sequence[Sequence[Dict]],
    topk: int,
    score_tolerance: float,
) -> None:
    if len(serial) != len(batched):
        raise AssertionError(f"serial/batch result count mismatch: {len(serial)} vs {len(batched)}")
    for query_idx, (serial_docs, batch_docs) in enumerate(zip(serial, batched)):
        serial_top = list(serial_docs[:topk])
        batch_top = list(batch_docs[:topk])
        serial_ids = [doc.get("id") for doc in serial_top]
        batch_ids = [doc.get("id") for doc in batch_top]
        if serial_ids != batch_ids:
            raise AssertionError(
                f"query {query_idx}: top-{topk} ids differ\nserial={serial_ids}\nbatch={batch_ids}"
            )
        for rank, (serial_doc, batch_doc) in enumerate(zip(serial_top, batch_top)):
            serial_score = float(serial_doc.get("score", 0.0))
            batch_score = float(batch_doc.get("score", 0.0))
            if not math.isclose(serial_score, batch_score, abs_tol=score_tolerance, rel_tol=1e-6):
                raise AssertionError(
                    f"query {query_idx} rank {rank}: score differs "
                    f"{serial_score} vs {batch_score}"
                )


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(int(math.ceil(fraction * len(ordered))) - 1, len(ordered) - 1)
    return ordered[max(idx, 0)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--num_queries", type=int, default=32)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--score_tolerance", type=float, default=1e-5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    queries = load_queries(args.data_path, args.num_queries)
    request_batch(args.url, queries[: min(4, len(queries))], args.timeout)  # warm-up

    serial_results = []
    serial_latencies = []
    for query in queries:
        result, elapsed = request_batch(args.url, [query], args.timeout)
        serial_results.extend(result)
        serial_latencies.append(elapsed)

    report = {
        "num_queries": len(queries),
        "serial": {
            "wall_seconds": sum(serial_latencies),
            "queries_per_second": len(queries) / max(sum(serial_latencies), 1e-9),
            "request_p50_seconds": statistics.median(serial_latencies),
            "request_p95_seconds": percentile(serial_latencies, 0.95),
        },
        "batches": {},
    }

    for batch_size in args.batch_sizes:
        results = []
        latencies = []
        for start in range(0, len(queries), batch_size):
            chunk = queries[start : start + batch_size]
            chunk_results, elapsed = request_batch(args.url, chunk, args.timeout)
            results.extend(chunk_results)
            latencies.append(elapsed)
        compare_results(serial_results, results, args.topk, args.score_tolerance)
        wall = sum(latencies)
        report["batches"][str(batch_size)] = {
            "wall_seconds": wall,
            "queries_per_second": len(queries) / max(wall, 1e-9),
            "speedup_vs_serial": sum(serial_latencies) / max(wall, 1e-9),
            "request_p50_seconds": statistics.median(latencies),
            "request_p95_seconds": percentile(latencies, 0.95),
            "topk_matches_serial": True,
        }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
