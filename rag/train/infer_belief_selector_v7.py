"""Run a frozen ACEC v7 selector over serialized wide candidate-pool states."""

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

from belief.acec.answer_value_v7 import (  # noqa: E402
    load_answer_value_artifact_v7,
)
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
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--strategy",
        choices=("learned", "relevance", "fixed_rel_plus_coverage"),
        default="learned",
    )
    parser.add_argument("--selector_artifact")
    parser.add_argument("--answer_value_artifact")
    parser.add_argument("--selected_k", type=int, default=5)
    parser.add_argument("--fixed_coverage_weight", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selected_k <= 0:
        raise ValueError("selected K must be positive")
    output_path = Path(args.output).expanduser()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    selector_artifact = (
        load_selector_artifact_v7(Path(args.selector_artifact).expanduser())
        if args.selector_artifact
        else None
    )
    answer_value = (
        load_answer_value_artifact_v7(
            Path(args.answer_value_artifact).expanduser()
        )
        if args.answer_value_artifact
        else None
    )
    selector = SequentialSetSelectorV7(
        artifact=selector_artifact,
        answer_value_artifact=answer_value,
        strategy=args.strategy,
        fixed_coverage_weight=args.fixed_coverage_weight,
    )
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row_index, row in enumerate(
            _read_jsonl(Path(args.input).expanduser())
        ):
            state = SelectorStateV7.from_dict(row["state"])
            candidates = tuple(
                CandidateV7.from_dict(payload) for payload in row["candidates"]
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
                raise ValueError("selector inference state has no selection capacity")
            selected, trace = selector.select(
                state,
                candidates,
                k=effective_k,
                sample=args.sample,
                seed=args.seed + row_index,
                temperature=args.temperature,
            )
            output.write(
                json.dumps(
                    {
                        "schema": "acec_selector_inference_v7",
                        "version": 70,
                        "state_id": row.get("state_id"),
                        "question": row.get("question"),
                        "selected_documents": [
                            {
                                "candidate_id": candidate.candidate_id,
                                "contents": candidate.contents,
                                "retrieval_rank": candidate.retrieval_rank,
                            }
                            for candidate in selected
                        ],
                        "selection_trace": trace.to_dict(),
                        "evidence_membership_supervision_used": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    if count == 0:
        raise ValueError("selector inference input is empty")
    print(json.dumps({"rows": count, "output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
