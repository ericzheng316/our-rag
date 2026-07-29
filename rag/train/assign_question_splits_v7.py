"""Assign deterministic question-disjoint ACEC v7 fit/calibration/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _bucket(question_id: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{question_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", default="acec-v7")
    parser.add_argument("--fit_fraction", type=float, default=0.70)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.fit_fraction < 1.0:
        raise ValueError("fit_fraction must be in (0, 1)")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if args.fit_fraction + args.validation_fraction >= 1.0:
        raise ValueError("fit+validation fractions must leave an internal test split")
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    manifest_path = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    counts = {"fit": 0, "validation": 0, "test": 0}
    seen_question_split: Dict[str, str] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in _read_jsonl(input_path):
            state = row.get("state") or {}
            question_id = str(
                state.get("question_id") or row.get("question_id") or ""
            )
            if not question_id:
                raise ValueError("every row requires a question id")
            value = _bucket(question_id, args.seed)
            if value < args.fit_fraction:
                split = "fit"
            elif value < args.fit_fraction + args.validation_fraction:
                split = "validation"
            else:
                split = "test"
            previous = seen_question_split.setdefault(question_id, split)
            if previous != split:
                raise AssertionError("one question was assigned to multiple splits")
            row["split"] = split
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    if not all(counts.values()):
        raise ValueError(
            f"deterministic split produced an empty partition: {counts}"
        )
    manifest = {
        "schema": "acec_question_split_v7",
        "seed": args.seed,
        "fractions": {
            "fit": args.fit_fraction,
            "validation": args.validation_fraction,
            "test": 1.0 - args.fit_fraction - args.validation_fraction,
        },
        "question_ids": {
            split: sorted(
                question_id
                for question_id, assigned in seen_question_split.items()
                if assigned == split
            )
            for split in counts
        },
        "rows": counts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "rows": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
