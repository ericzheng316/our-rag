"""Append frozen ACEC v7 selector slates for external answer/SF evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.answer_value_v7 import load_answer_value_artifact_v7  # noqa: E402
from belief.acec.contracts_v7 import CandidateV7, SelectorStateV7  # noqa: E402
from belief.acec.set_selector_v7 import (  # noqa: E402
    SequentialSetSelectorV7,
    load_selector_artifact_v7,
)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--selector_artifact", required=True)
    parser.add_argument("--answer_value_artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected_k", type=int, default=5)
    parser.add_argument(
        "--split",
        default="test",
        help="Only evaluate this question-disjoint split; use an empty value for all.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selected_k <= 0:
        raise ValueError("selected K must be positive")
    output_path = Path(args.output).expanduser()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    selector = SequentialSetSelectorV7(
        artifact=load_selector_artifact_v7(
            Path(args.selector_artifact).expanduser()
        ),
        answer_value_artifact=load_answer_value_artifact_v7(
            Path(args.answer_value_artifact).expanduser()
        ),
        strategy="learned",
    )
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in _read_jsonl(Path(args.input).expanduser()):
            if args.split and str(row.get("split") or "") != args.split:
                continue
            state = SelectorStateV7.from_dict(row["state"])
            candidates = tuple(
                CandidateV7.from_dict(value) for value in row["candidates"]
            )
            effective_k = min(
                int(args.selected_k),
                int(
                    state.metadata.get(
                        "selection_capacity", args.selected_k
                    )
                ),
            )
            if effective_k <= 0:
                raise ValueError("selector evaluation state has no selection capacity")
            _, trace = selector.select(state, candidates, k=effective_k)
            row["slates"] = [
                slate
                for slate in row.get("slates") or ()
                if str(slate.get("slate_id")) != "learned_v7"
            ]
            row["slates"].append(
                {
                    "slate_id": "learned_v7",
                    "selected_ids": list(trace.selected_ids),
                    "step_propensities": [
                        float(step.candidate_probabilities[step.selected_id])
                        for step in trace.steps
                    ],
                    "selection_trace": trace.to_dict(),
                }
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("selector evaluation input is empty")
    print(json.dumps({"rows": count, "output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
