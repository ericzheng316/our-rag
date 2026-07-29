"""Fit the frozen ACEC v7 signed answer-value head from scored slate prefixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.answer_value_v7 import (  # noqa: E402
    AnswerValueExampleV7,
    fit_answer_value_crossfit_v7,
    save_answer_value_artifact_v7,
    save_answer_value_crossfit_bundle_v7,
)
from belief.acec.belief_simulator_v7 import BeliefSimulatorV7  # noqa: E402
from belief.acec.contracts_v7 import CandidateV7, SelectorStateV7  # noqa: E402
from belief.acec.set_selector_v7 import (  # noqa: E402
    base_answer_value_features_v7,
)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _examples(row: Dict[str, Any]) -> List[AnswerValueExampleV7]:
    state = SelectorStateV7.from_dict(row["state"])
    candidates = tuple(CandidateV7.from_dict(value) for value in row["candidates"])
    simulator = BeliefSimulatorV7(state, candidates)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    retrieval = np.asarray(
        [candidate.retrieval_score for candidate in candidates], dtype=np.float64
    )
    median = float(np.median(retrieval))
    q75, q25 = np.percentile(retrieval, [75, 25])
    scale = float(q75 - q25)
    if scale <= 1e-9:
        scale = float(np.std(retrieval)) or 1.0
    score_z = {
        candidate.candidate_id: float(
            (candidate.retrieval_score - median) / scale
        )
        for candidate in candidates
    }
    result = []
    for slate in row["slates"]:
        marginals = slate.get("prefix_marginal_utilities")
        if marginals is None or len(marginals) != len(slate["selected_ids"]):
            raise ValueError("slate lacks aligned prefix marginal utilities")
        selected: List[str] = []
        for candidate_id, target in zip(slate["selected_ids"], marginals):
            candidate = by_id[str(candidate_id)]
            features = base_answer_value_features_v7(
                state,
                candidate,
                simulator,
                selected,
                retrieval_score_z=score_z[candidate.candidate_id],
            )
            result.append(
                AnswerValueExampleV7(
                    question_id=state.question_id,
                    features=features,
                    target_delta=float(target),
                    state_id=str(
                        row.get("state_id")
                        or f"{state.question_id}:{state.turn_index}"
                    ),
                )
            )
            selected.append(candidate.candidate_id)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crossfit_output")
    parser.add_argument("--fit_split", default="fit")
    parser.add_argument("--validation_split", default="validation")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser()
    fit_examples: List[AnswerValueExampleV7] = []
    validation_examples: List[AnswerValueExampleV7] = []
    for row in _read_jsonl(input_path):
        split = str(row.get("split") or "")
        if split == args.fit_split:
            fit_examples.extend(_examples(row))
        elif split == args.validation_split:
            validation_examples.extend(_examples(row))
    bundle = fit_answer_value_crossfit_v7(
        fit_examples,
        validation_examples,
        folds=args.folds,
        l2=args.l2,
        metadata={
            "source": str(input_path),
            "supervision": "gold_answer_teacher_forced_logprob_delta",
            "evidence_membership_supervision_used": False,
        },
    )
    artifact = bundle.full_artifact
    output_path = Path(args.output).expanduser()
    save_answer_value_artifact_v7(output_path, artifact)
    crossfit_path = (
        Path(args.crossfit_output).expanduser()
        if args.crossfit_output
        else output_path.with_suffix(".crossfit.json")
    )
    save_answer_value_crossfit_bundle_v7(crossfit_path, bundle)
    print(
        json.dumps(
            {
                "runtime_artifact": artifact.to_dict(),
                "crossfit_bundle": str(crossfit_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
