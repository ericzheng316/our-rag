"""Build SF-label-free counterfactual slates for ACEC v7.

Input JSONL rows must contain ``state`` (SelectorStateV7), ``candidates``
(CandidateV7 rows), and may contain question/history/gold_answers for the
separate answer-utility scorer.  No supporting-fact field is copied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.contracts_v7 import (  # noqa: E402
    CandidateV7,
    SelectorStateV7,
    assert_sf_label_free,
)
from belief.acec.set_selector_v7 import (  # noqa: E402
    SequentialSetSelectorV7,
)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield payload


def _slate_payload(name: str, trace: Any) -> Dict[str, Any]:
    return {
        "slate_id": name,
        "selected_ids": list(trace.selected_ids),
        "step_propensities": [
            float(step.candidate_probabilities[step.selected_id])
            for step in trace.steps
        ],
        "step_log_probabilities": [
            float(step.log_probability) for step in trace.steps
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected_k", type=int, default=5)
    parser.add_argument("--candidate_pool_size", type=int, default=20)
    parser.add_argument("--fixed_coverage_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.selected_k <= 0 or args.candidate_pool_size < args.selected_k:
        raise ValueError("candidate pool size must be >= positive selected K")
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    relevance = SequentialSetSelectorV7(artifact=None, strategy="relevance")
    fixed = SequentialSetSelectorV7(
        artifact=None,
        strategy="fixed_rel_plus_coverage",
        fixed_coverage_weight=args.fixed_coverage_weight,
    )
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for row_index, payload in enumerate(_read_jsonl(input_path)):
            state = SelectorStateV7.from_dict(payload["state"])
            candidates = tuple(
                CandidateV7.from_dict(value)
                for value in payload["candidates"][: args.candidate_pool_size]
            )
            if not candidates:
                continue
            effective_k = min(
                int(args.selected_k),
                int(
                    state.metadata.get(
                        "selection_capacity", args.selected_k
                    )
                ),
            )
            if effective_k <= 0:
                continue
            _, relevance_trace = relevance.select(
                state, candidates, k=effective_k
            )
            _, fixed_trace = fixed.select(state, candidates, k=effective_k)
            _, stochastic_a = fixed.select(
                state,
                candidates,
                k=effective_k,
                sample=True,
                seed=args.seed + row_index * 2 + 1,
                temperature=1.0,
            )
            _, stochastic_b = fixed.select(
                state,
                candidates,
                k=effective_k,
                sample=True,
                seed=args.seed + row_index * 2 + 2,
                temperature=1.5,
            )
            safe_top_level = {
                key: payload[key]
                for key in (
                    "question",
                    "history",
                    "gold_answers",
                    "split",
                    "state_id",
                )
                if key in payload
            }
            assert_sf_label_free(safe_top_level, path="counterfactual.top_level")
            record = {
                **safe_top_level,
                "schema": "acec_counterfactual_slates_v7",
                "version": 70,
                "state": state.to_dict(),
                "candidates": [candidate.to_dict() for candidate in candidates],
                "slates": [
                    _slate_payload("relevance", relevance_trace),
                    _slate_payload("fixed_rel_plus_coverage", fixed_trace),
                    _slate_payload("stochastic_a", stochastic_a),
                    _slate_payload("stochastic_b", stochastic_b),
                ],
                "evidence_membership_supervision_used": False,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("no counterfactual states were written")
    print(json.dumps({"states": count, "output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
