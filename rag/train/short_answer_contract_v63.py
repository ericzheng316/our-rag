"""Deterministic short-answer contract used by every v6.3 policy arm."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping


ANSWER_CONTRACT_VERSION = "tagged_short_answer_v63.1"
SHORT_ANSWER_INSTRUCTION = (
    "Return the final answer exactly once inside <answer> and </answer>. "
    "The tagged text must be only the shortest answer span, with no sentence "
    "wrapper or explanation. Example: <answer>swimming</answer>."
)
_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_ANY_ANSWER_TAG = re.compile(r"</?answer\b", re.IGNORECASE)


@dataclass(frozen=True)
class ShortAnswerParseV63:
    answer: str
    valid: bool
    status: str
    contract_version: str = ANSWER_CONTRACT_VERSION


def parse_tagged_short_answer(text: Any) -> ShortAnswerParseV63:
    """Extract one unambiguous tag span without an LLM or heuristic cleanup."""

    raw = str(text or "")
    matches = list(_ANSWER_TAG.finditer(raw))
    if not matches:
        return ShortAnswerParseV63("", False, "missing_answer_tag")
    if len(matches) != 1:
        return ShortAnswerParseV63("", False, "multiple_answer_tags")
    match = matches[0]
    answer = " ".join(match.group(1).split())
    if not answer:
        return ShortAnswerParseV63("", False, "empty_answer")
    if "<" in answer or ">" in answer or _ANY_ANSWER_TAG.search(answer):
        return ShortAnswerParseV63("", False, "nested_or_malformed_tag")
    # A stray opening/closing answer tag outside the matched pair is ambiguous.
    outside = raw[: match.start()] + raw[match.end() :]
    if _ANY_ANSWER_TAG.search(outside):
        return ShortAnswerParseV63("", False, "multiple_or_malformed_answer_tags")
    return ShortAnswerParseV63(answer, True, "ok")


def attach_parsed_answer(
    row: Mapping[str, Any], *, input_field: str, id_field: str
) -> Dict[str, Any]:
    if input_field not in row:
        raise KeyError(f"prediction row lacks {input_field!r}")
    if id_field not in row:
        raise KeyError(f"prediction row lacks {id_field!r}")
    parsed = parse_tagged_short_answer(row[input_field])
    return {
        "question_id": str(row[id_field]),
        "draft_answer": parsed.answer,
        "parse_valid": parsed.valid,
        "parse_status": parsed.status,
        "answer_contract_version": parsed.contract_version,
    }


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input_field", default="generation")
    parser.add_argument("--id_field", default="question_id")
    parser.add_argument("--min_coverage", type=float, default=0.99)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.min_coverage <= 1.0:
        raise ValueError("--min_coverage must be in [0, 1]")
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing/empty prediction input: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite parsed answers: {output_path}")

    parsed_rows = [
        attach_parsed_answer(row, input_field=args.input_field, id_field=args.id_field)
        for row in _read_jsonl(input_path)
    ]
    if not parsed_rows:
        raise ValueError("prediction input contains no rows")
    coverage = sum(bool(row["parse_valid"]) for row in parsed_rows) / len(parsed_rows)
    if coverage < args.min_coverage:
        counts: Dict[str, int] = {}
        for row in parsed_rows:
            counts[row["parse_status"]] = counts.get(row["parse_status"], 0) + 1
        raise ValueError(
            f"short-answer parse coverage {coverage:.4f} is below "
            f"{args.min_coverage:.4f}; statuses={counts}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in parsed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "contract_version": ANSWER_CONTRACT_VERSION,
                "rows": len(parsed_rows),
                "parse_coverage": coverage,
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
