"""Evaluate ACEC v7 Gate-1 headroom without feeding SF labels to training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _candidate_gold_ids(row: Dict[str, Any]) -> Set[str]:
    explicit = row.get("gold_candidate_ids")
    if explicit is not None:
        return {str(value) for value in explicit}
    sf_titles = {str(value).casefold() for value in row.get("sf_titles") or ()}
    result = set()
    for candidate in row["candidates"]:
        title = str(
            (candidate.get("metadata") or {}).get("title")
            or candidate.get("title")
            or ""
        ).casefold()
        if title and title in sf_titles:
            result.add(str(candidate["candidate_id"]))
    return result


def _recall(ids: Sequence[str], gold: Set[str], total_gold: int) -> float:
    return len(set(ids) & gold) / total_gold if total_gold > 0 else 0.0


def _title_recall(
    candidates: Sequence[Dict[str, Any]], gold_titles: Set[str]
) -> float:
    retrieved_titles = {
        str(
            (candidate.get("metadata") or {}).get("title")
            or candidate.get("title")
            or ""
        ).casefold()
        for candidate in candidates
    }
    retrieved_titles.discard("")
    return (
        len(retrieved_titles & gold_titles) / len(gold_titles)
        if gold_titles
        else 0.0
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--answer_input",
        help="Optional scored counterfactual JSONL for answer-utility headroom.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=20)
    parser.add_argument("--min_recall_headroom", type=float, default=0.05)
    parser.add_argument("--min_answer_utility_headroom", type=float, default=1e-6)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    recalls_k: List[float] = []
    recalls_l: List[float] = []
    utility_headroom: List[float] = []
    rows = 0
    for row in _read_jsonl(Path(args.input).expanduser()):
        rows += 1
        candidates = row["candidates"][: args.pool_size]
        ids = [str(candidate["candidate_id"]) for candidate in candidates]
        gold = _candidate_gold_ids(row)
        gold_titles = {
            str(value).casefold() for value in row.get("sf_titles") or ()
        }
        gold_titles.discard("")
        if gold_titles:
            recalls_k.append(_title_recall(candidates[: args.k], gold_titles))
            recalls_l.append(_title_recall(candidates, gold_titles))
        elif gold:
            recalls_k.append(_recall(ids[: args.k], gold, len(gold)))
            recalls_l.append(_recall(ids, gold, len(gold)))
        if not args.answer_input:
            slates = row.get("slates") or ()
            utilities = {
                str(slate["slate_id"]): float(slate["answer_utility"])
                for slate in slates
                if "answer_utility" in slate
            }
            if utilities and "relevance" in utilities:
                utility_headroom.append(
                    max(utilities.values()) - utilities["relevance"]
                )
    if args.answer_input:
        for row in _read_jsonl(Path(args.answer_input).expanduser()):
            utilities = {
                str(slate["slate_id"]): float(slate["answer_utility"])
                for slate in row.get("slates") or ()
                if "answer_utility" in slate
            }
            if utilities and "relevance" in utilities:
                utility_headroom.append(
                    max(utilities.values()) - utilities["relevance"]
                )
    if rows == 0:
        raise ValueError("headroom input is empty")
    recall_k = sum(recalls_k) / max(len(recalls_k), 1)
    recall_l = sum(recalls_l) / max(len(recalls_l), 1)
    answer_headroom = sum(utility_headroom) / max(len(utility_headroom), 1)
    result = {
        "schema": "acec_candidate_headroom_v7",
        "rows": rows,
        "rows_with_gold_evidence": len(recalls_k),
        "recall_at_k": recall_k,
        "recall_at_l": recall_l,
        "recall_headroom": recall_l - recall_k,
        "mean_counterfactual_answer_utility_headroom": answer_headroom,
        "answer_utility_available": bool(utility_headroom),
        "go_recall": recall_l - recall_k >= args.min_recall_headroom,
        "go_answer_utility": (
            bool(utility_headroom)
            and answer_headroom >= args.min_answer_utility_headroom
        ),
    }
    result["go"] = bool(result["go_recall"] and result["go_answer_utility"])
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser()
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
