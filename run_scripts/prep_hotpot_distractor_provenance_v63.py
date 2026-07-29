#!/usr/bin/env python3
"""Build separate runtime and evaluator manifests from HotpotQA distractor.

The runtime JSONL preserves native sentence boundaries but contains no gold
answer or supporting-fact labels.  The evaluator JSON is kept separate and is
compatible with HotpotQA's unchanged official evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.datasets.hotpotqa_v63 import (  # noqa: E402
    ADAPTER_NAME,
    ADAPTER_VERSION,
    HotpotQADistractorAdapterV63,
    question_id,
)
from belief.acec.evidence_contract_v63 import (  # noqa: E402
    assert_runtime_payload_has_no_gold,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_hash(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(4096).lstrip()[:1]
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"JSON input is not an array: {path}")
            records = payload
        else:
            records = [json.loads(line) for line in handle if line.strip()]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"HotpotQA input is empty or malformed: {path}")
    return records


def _parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SPLIT=/path/to/data.json[l]")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected SPLIT=/path/to/data.json[l]")
    return name.strip(), Path(path).expanduser()


def _ids_from_path(path: Path) -> set[str]:
    result = set()
    for record in _read_records(path):
        value = record.get("question_id", record.get("_id", record.get("id")))
        if value is None:
            raise ValueError(f"cannot find question id in disjointness file {path}")
        result.add(str(value))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runtime_output", required=True)
    parser.add_argument("--evaluator_output", required=True)
    parser.add_argument("--manifest_output", required=True)
    parser.add_argument("--split_name", required=True)
    parser.add_argument(
        "--disjoint_against",
        action="append",
        default=[],
        type=_parse_named_path,
        help="Repeatable SPLIT=/path check against another JSON/JSONL split.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser()
    outputs = [
        Path(args.runtime_output).expanduser(),
        Path(args.evaluator_output).expanduser(),
        Path(args.manifest_output).expanduser(),
    ]
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing/empty HotpotQA input: {input_path}")
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite v6.3 artifacts: {existing}")
    if len(set(outputs)) != len(outputs):
        raise ValueError("runtime, evaluator, and manifest outputs must differ")

    adapter = HotpotQADistractorAdapterV63()
    source = _read_records(input_path)
    runtime_rows: List[Dict[str, Any]] = []
    evaluator_rows: List[Dict[str, Any]] = []
    ids: List[str] = []
    seen_ids = set()
    sentence_count = 0
    for record in source:
        adapter.validate_round_trip(record)
        runtime = adapter.runtime_record(record)
        assert_runtime_payload_has_no_gold(runtime)
        evaluator = adapter.evaluator_record(record)
        qid = question_id(record)
        if qid in seen_ids:
            raise ValueError(f"duplicate HotpotQA question id: {qid}")
        seen_ids.add(qid)
        ids.append(qid)
        sentence_count += len(runtime["candidate_evidence"])
        runtime_rows.append(runtime)
        evaluator_rows.append(evaluator)

    own_ids = set(ids)
    disjointness: Dict[str, Any] = {}
    for split_name, path in args.disjoint_against:
        if not path.is_file():
            raise FileNotFoundError(f"disjointness input does not exist: {path}")
        other_ids = _ids_from_path(path)
        overlap = sorted(own_ids & other_ids)
        disjointness[split_name] = {
            "path": str(path),
            "id_count": len(other_ids),
            "id_sha256": _id_hash(other_ids),
            "overlap_count": len(overlap),
        }
        if overlap:
            raise ValueError(
                f"split {args.split_name!r} overlaps {split_name!r}: {overlap[:10]}"
            )

    manifest = {
        "schema": "hotpotqa_distractor_provenance_manifest",
        "schema_version": 63,
        "split_name": args.split_name,
        "source_path": str(input_path),
        "source_sha256": _sha256(input_path),
        "question_count": len(ids),
        "question_id_sha256": _id_hash(ids),
        "native_sentence_count": sentence_count,
        "native_provenance_coverage": 1.0,
        "runtime_contains_gold_labels": False,
        "adapter": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "disjointness": disjointness,
    }

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x", encoding="utf-8") as handle:
        for row in runtime_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with outputs[1].open("x", encoding="utf-8") as handle:
        json.dump(evaluator_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with outputs[2].open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
