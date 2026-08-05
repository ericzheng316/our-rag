#!/usr/bin/env python3
"""Prepare and compare the auditable v6.4 HotpotQA bridge-entity oracle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.answer_verifier_v64 import (  # noqa: E402
    answer_exact_match,
    answer_f1,
    paired_bootstrap_delta_v64,
)
from belief.acec.inference_controller_v64 import (  # noqa: E402
    BRIDGE_ENTITY_SCHEMA,
    derive_gold_bridge_entity_v64,
)


def _read_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(4096).lstrip()[:1]
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"JSON source is not an array: {path}")
            return [dict(row) for row in payload]
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qid(row: Mapping[str, Any]) -> str:
    value = row.get("question_id", row.get("_id", row.get("id")))
    if value is None:
        raise ValueError("record lacks question id")
    return str(value)


def _golds(row: Mapping[str, Any]) -> List[str]:
    answers = row.get("golden_answers")
    if answers:
        return [str(answer) for answer in answers]
    if row.get("answer") is not None:
        return [str(row["answer"])]
    raise ValueError(f"record {_qid(row)} lacks gold answer")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def prepare(args: argparse.Namespace) -> None:
    source = Path(args.data).expanduser()
    output = Path(args.output_manifest).expanduser()
    ids_output = Path(args.output_ids).expanduser()
    metadata_output = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or ids_output.exists() or metadata_output.exists():
        raise FileExistsError("refusing to overwrite bridge-oracle outputs")
    # 类型字段与 infer_belief_controller_v64.load_questions 同款三级回退：
    # 仓库自备的 dev_distractor.jsonl 把 type 放在 metadata.type，只读顶层
    # 会得到 "no bridge questions"（HotpotQA 实际约八成是 bridge）。
    labels = [
        derive_gold_bridge_entity_v64(row)
        for row in _read_records(source)
        if str(
            row.get(
                "type",
                row.get("question_type", (row.get("metadata") or {}).get("type", "")),
            )
        )
        == "bridge"
    ]
    if not labels:
        raise ValueError("bridge-oracle source contains no bridge questions")
    rows = [label.to_dict() for label in labels]
    _write_jsonl(output, rows)
    assessable = [label.question_id for label in labels if label.assessable]
    _write_jsonl(ids_output, [{"question_id": qid} for qid in assessable])
    reasons = Counter(label.basis for label in labels)
    metadata = {
        "schema": "acec_bridge_entity_manifest",
        "version": 64,
        "source": str(source),
        "source_sha256": _sha256(source),
        "bridge_questions": len(labels),
        "assessable": len(assessable),
        "assessable_rate": len(assessable) / max(len(labels), 1),
        "basis_counts": dict(sorted(reasons.items())),
        "manifest": str(output),
        "question_ids": str(ids_output),
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )


def _prediction_map(path: Path, field: str) -> Dict[str, str]:
    result = {}
    for row in _read_records(path):
        qid = _qid(row)
        if qid in result:
            raise ValueError(f"duplicate prediction id {qid}: {path}")
        if field not in row:
            raise ValueError(f"prediction row {qid} lacks {field!r}: {path}")
        result[qid] = str(row[field])
    return result


def _manifest_records(path: Path) -> List[Dict[str, Any]]:
    rows = _read_records(path)
    for row in rows:
        if (
            row.get("schema") != BRIDGE_ENTITY_SCHEMA
            or int(row.get("version", -1)) != 64
        ):
            raise ValueError(f"bridge manifest contains a non-v6.4 row: {path}")
    return rows


def _retrieved_title_map(path: Optional[str], sample_index: int) -> Dict[str, set[str]]:
    if not path:
        return {}
    result: Dict[str, set[str]] = {}
    for row in _read_records(Path(path).expanduser()):
        if int(row.get("sample_index", -1)) != sample_index:
            continue
        titles = {
            str(document.get("title", "")).casefold()
            for event in row.get("events", [])
            if event.get("kind") == "retrieval"
            for document in event.get("documents", [])
            if document.get("title")
        }
        result[_qid(row)] = titles
    return result


def _supporting_titles(row: Mapping[str, Any]) -> set[str]:
    supporting = row.get("supporting_facts") or {}
    if isinstance(supporting, Mapping):
        return {str(value).casefold() for value in supporting.get("title", [])}
    return {str(value[0]).casefold() for value in supporting}


def compare(args: argparse.Namespace) -> None:
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite bridge comparison: {output}")
    data_by_id = {_qid(row): row for row in _read_records(Path(args.data).expanduser())}
    assessable_ids = {
        _qid(row)
        for row in _manifest_records(Path(args.manifest).expanduser())
        if bool(row.get("assessable")) and row.get("entity")
    }
    baseline = _prediction_map(Path(args.baseline_predictions).expanduser(), args.answer_field)
    oracle = _prediction_map(Path(args.oracle_predictions).expanduser(), args.answer_field)
    ids = sorted(assessable_ids & set(baseline) & set(oracle) & set(data_by_id))
    if not ids:
        raise ValueError("bridge comparison has no aligned assessable questions")

    baseline_em = {
        qid: answer_exact_match(baseline[qid], _golds(data_by_id[qid]))
        for qid in ids
    }
    oracle_em = {
        qid: answer_exact_match(oracle[qid], _golds(data_by_id[qid]))
        for qid in ids
    }
    baseline_f1 = {
        qid: answer_f1(baseline[qid], _golds(data_by_id[qid])) for qid in ids
    }
    oracle_f1 = {
        qid: answer_f1(oracle[qid], _golds(data_by_id[qid])) for qid in ids
    }
    baseline_titles = _retrieved_title_map(args.baseline_runtime, args.sample_index)
    oracle_titles = _retrieved_title_map(args.oracle_runtime, args.sample_index)

    payload: Dict[str, Any] = {
        "questions": len(ids),
        "answer_field": args.answer_field,
        "baseline_em": sum(baseline_em.values()) / len(ids),
        "oracle_em": sum(oracle_em.values()) / len(ids),
        "baseline_f1": sum(baseline_f1.values()) / len(ids),
        "oracle_f1": sum(oracle_f1.values()) / len(ids),
        "paired_oracle_minus_baseline_em": paired_bootstrap_delta_v64(
            oracle_em, baseline_em, iterations=args.bootstrap_iterations, seed=args.seed
        ),
        "paired_oracle_minus_baseline_f1": paired_bootstrap_delta_v64(
            oracle_f1,
            baseline_f1,
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        ),
        "em_wins": sum(oracle_em[qid] > baseline_em[qid] for qid in ids),
        "em_ties": sum(oracle_em[qid] == baseline_em[qid] for qid in ids),
        "em_losses": sum(oracle_em[qid] < baseline_em[qid] for qid in ids),
    }
    if baseline_titles and oracle_titles:
        baseline_recall = {}
        oracle_recall = {}
        for qid in ids:
            gold = _supporting_titles(data_by_id[qid])
            baseline_recall[qid] = len(baseline_titles.get(qid, set()) & gold) / max(
                len(gold), 1
            )
            oracle_recall[qid] = len(oracle_titles.get(qid, set()) & gold) / max(
                len(gold), 1
            )
        payload.update(
            {
                "baseline_sf_title_recall": sum(baseline_recall.values()) / len(ids),
                "oracle_sf_title_recall": sum(oracle_recall.values()) / len(ids),
                "paired_oracle_minus_baseline_sf_title_recall": paired_bootstrap_delta_v64(
                    oracle_recall,
                    baseline_recall,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + 2,
                ),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--data", required=True)
    prepare_parser.add_argument("--output_manifest", required=True)
    prepare_parser.add_argument("--output_ids", required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--data", required=True)
    compare_parser.add_argument("--manifest", required=True)
    compare_parser.add_argument("--baseline_predictions", required=True)
    compare_parser.add_argument("--oracle_predictions", required=True)
    compare_parser.add_argument("--baseline_runtime", default=None)
    compare_parser.add_argument("--oracle_runtime", default=None)
    compare_parser.add_argument("--answer_field", default="answer")
    compare_parser.add_argument("--sample_index", type=int, default=0)
    compare_parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    compare_parser.add_argument("--seed", type=int, default=20260723)
    compare_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "compare":
        compare(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
