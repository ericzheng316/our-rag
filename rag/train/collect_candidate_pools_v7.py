"""Replay frozen V6 trajectories to collect ACEC v7 wide candidate pools.

The replay uses the frozen v5 belief artifact and a relevance top-K selector
to preserve the previous evidence path while logging all L candidates before
selection.  Gold answers are joined for the separate answer-value scorer;
supporting-fact labels are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
RAG_TRAIN = REPO_ROOT / "rag" / "train"
for path in (RAG_SRC, RAG_TRAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from belief.acec.runtime_v7 import BeliefSelectorRuntimeV7  # noqa: E402
from belief.acec.contracts_v7 import assert_sf_label_free  # noqa: E402
from belief.acec.set_selector_v7 import SequentialSetSelectorV7  # noqa: E402
from grpo_rsf_vllm_v5 import make_belief_factory_v5  # noqa: E402


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _question_id(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:24]


def _document_id(document: Dict[str, Any], rank: int) -> str:
    value = document.get("id")
    if value is not None and str(value).strip():
        return str(value)
    digest = hashlib.sha256(
        str(document.get("contents") or "").encode("utf-8")
    ).hexdigest()[:20]
    return f"content:{digest}:{rank}"


def _dataset_info(path: Path) -> Dict[str, Dict[str, List[str]]]:
    result: Dict[str, Dict[str, List[str]]] = {}
    for row in _read_jsonl(path):
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        values = row.get("golden_answers")
        if values is None:
            answer = row.get("answer")
            values = [answer] if answer is not None else []
        supporting = row.get("supporting_facts") or {}
        sf_titles = (
            supporting.get("title", [])
            if isinstance(supporting, dict)
            else []
        )
        result[question] = {
            "gold_answers": [str(value) for value in values],
            "sf_titles": [str(value) for value in sf_titles],
        }
    return result


def _retrieve_batches(
    url: str,
    queries: Sequence[str],
    *,
    batch_size: int,
    pool_size: int,
    timeout: float,
) -> List[List[Dict[str, Any]]]:
    output: List[List[Dict[str, Any]]] = []
    for offset in range(0, len(queries), batch_size):
        batch = list(queries[offset : offset + batch_size])
        response = requests.post(
            url,
            json={"querys": batch},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) != len(batch):
            raise ValueError("retriever returned a malformed batch")
        output.extend([list(documents[:pool_size]) for documents in payload])
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--oracle_output",
        help="Optional evaluator-only SF sidecar; never feed this file to training.",
    )
    parser.add_argument("--retrieve_url", default="http://127.0.0.1:8001/search")
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--selected_k", type=int, default=5)
    parser.add_argument("--max_context_documents", type=int, default=10)
    parser.add_argument("--retrieval_batch_size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--acec_artifact_v5", required=True)
    parser.add_argument("--e5_model_path", required=True)
    parser.add_argument(
        "--acec_nli_model", default="cross-encoder/nli-deberta-v3-base"
    )
    parser.add_argument("--acec_tau_new", type=float)
    parser.add_argument("--max_trajectories", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selected_k <= 0 or args.candidate_pool_size < args.selected_k:
        raise ValueError("candidate pool size must be >= positive selected K")
    if args.max_context_documents <= 0:
        raise ValueError("max context documents must be positive")
    if args.retrieval_batch_size <= 0:
        raise ValueError("retrieval batch size must be positive")
    trajectory_path = Path(args.trajectory).expanduser()
    dataset_path = Path(args.dataset).expanduser()
    output_path = Path(args.output).expanduser()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    trajectories = list(_read_jsonl(trajectory_path))
    if args.max_trajectories:
        trajectories = trajectories[: args.max_trajectories]
    if not trajectories:
        raise ValueError("trajectory input is empty")
    dataset_by_question = _dataset_info(dataset_path)
    queries = [
        str(event["query"])
        for trajectory in trajectories
        for event in trajectory.get("events") or ()
        if event.get("kind") == "retrieval" and event.get("query")
    ]
    retrieved = iter(
        _retrieve_batches(
            args.retrieve_url,
            queries,
            batch_size=args.retrieval_batch_size,
            pool_size=args.candidate_pool_size,
            timeout=args.timeout,
        )
    )
    belief_factory = make_belief_factory_v5(args)
    runtime = BeliefSelectorRuntimeV7(
        SequentialSetSelectorV7(artifact=None, strategy="relevance"),
        selected_k=args.selected_k,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    states_written = 0
    oracle_path = (
        Path(args.oracle_output).expanduser() if args.oracle_output else None
    )
    if oracle_path is not None and oracle_path.exists():
        raise FileExistsError(f"refusing to overwrite {oracle_path}")
    if oracle_path is not None:
        oracle_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_handle = (
        oracle_path.open("w", encoding="utf-8")
        if oracle_path is not None
        else None
    )
    with output_path.open("w", encoding="utf-8") as output:
        for trajectory_index, trajectory in enumerate(trajectories):
            question = str(trajectory["question"])
            question_id = str(
                trajectory.get("question_id") or _question_id(question)
            )
            belief = belief_factory(question)
            history_parts: List[str] = []
            selected_document_ids: List[str] = []
            retrieval_index = 0
            for event in trajectory.get("events") or ():
                if event.get("kind") != "retrieval" or not event.get("query"):
                    continue
                selected_set = set(selected_document_ids)
                documents = [
                    document
                    for rank, document in enumerate(next(retrieved))
                    if _document_id(document, rank) not in selected_set
                ]
                remaining_capacity = max(
                    int(args.max_context_documents)
                    - len(selected_document_ids),
                    0,
                )
                if remaining_capacity == 0:
                    belief.turn(query=str(event["query"]), new_docs=[])
                    history_parts.append(
                        f"Query: {event['query']}\nEvidence:\n"
                    )
                    retrieval_index += 1
                    continue
                effective_k = min(int(args.selected_k), remaining_capacity)
                result = runtime.turn(
                    question_id=question_id,
                    belief=belief,
                    query=str(event["query"]),
                    candidate_documents=documents,
                    retrieval_budget_remaining=max(5 - retrieval_index, 0),
                    selector_seed=trajectory_index * 100 + retrieval_index,
                    selected_k=effective_k,
                )
                dataset_info = dataset_by_question.get(
                    question, {"gold_answers": [], "sf_titles": []}
                )
                record = {
                    "schema": "acec_candidate_pool_v7",
                    "version": 70,
                    "state_id": (
                        f"{trajectory.get('episode', 0)}:"
                        f"{trajectory.get('question_index', trajectory_index)}:"
                        f"{trajectory.get('sample_index', 0)}:{retrieval_index}"
                    ),
                    "question": question,
                    "history": "\n".join(history_parts),
                    "gold_answers": dataset_info["gold_answers"],
                    "state": result.selector_state.to_dict(),
                    "candidates": [
                        candidate.to_dict()
                        for candidate in result.candidate_features
                    ],
                    "replay_selected_ids": list(
                        result.selection_trace.selected_ids
                    ),
                    "evidence_membership_supervision_used": False,
                }
                assert_sf_label_free(record, path="candidate_pool")
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                if oracle_handle is not None:
                    sf_titles = {
                        value.casefold() for value in dataset_info["sf_titles"]
                    }
                    gold_candidate_ids = [
                        candidate.candidate_id
                        for candidate in result.candidate_features
                        if str(candidate.metadata.get("title") or "").casefold()
                        in sf_titles
                    ]
                    oracle_handle.write(
                        json.dumps(
                            {
                                "state_id": record["state_id"],
                                "candidates": [
                                    {
                                        "candidate_id": candidate.candidate_id,
                                        "metadata": {
                                            "title": candidate.metadata.get(
                                                "title", ""
                                            )
                                        },
                                    }
                                    for candidate in result.candidate_features
                                ],
                                "gold_candidate_ids": gold_candidate_ids,
                                "sf_titles": dataset_info["sf_titles"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                states_written += 1
                selected_document_ids.extend(
                    candidate_id
                    for candidate_id in result.selection_trace.selected_ids
                    if candidate_id not in selected_document_ids
                )
                selected_text = "\n".join(
                    str(document.get("contents") or "")
                    for document in result.selected_documents
                )
                history_parts.append(
                    f"Query: {event['query']}\nEvidence:\n{selected_text[:2048]}"
                )
                retrieval_index += 1
    if oracle_handle is not None:
        oracle_handle.close()
    try:
        next(retrieved)
    except StopIteration:
        pass
    else:
        raise AssertionError("retriever replay produced unused batches")
    if states_written == 0:
        raise ValueError("trajectory replay contained no retrieval states")
    print(
        json.dumps(
            {"states": states_written, "output": str(output_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
