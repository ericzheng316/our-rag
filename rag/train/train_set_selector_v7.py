"""Train the ACEC v7 state/set-conditioned selector from scored slates."""

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
    load_answer_value_artifact_v7,
    load_answer_value_crossfit_bundle_v7,
)
from belief.acec.belief_simulator_v7 import BeliefSimulatorV7  # noqa: E402
from belief.acec.contracts_v7 import CandidateV7, SelectorStateV7  # noqa: E402
from belief.acec.selector_loss_v7 import (  # noqa: E402
    SelectorFitConfigV7,
    SelectorSlateExampleV7,
    SelectorSlateStepV7,
    fit_selector_from_slates_v7,
)
from belief.acec.set_selector_v7 import (  # noqa: E402
    SELECTOR_FEATURE_NAMES_V7,
    save_selector_artifact_v7,
    selector_features_v7,
)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _slates(
    row: Dict[str, Any], answer_value_artifact: Any
) -> List[SelectorSlateExampleV7]:
    state = SelectorStateV7.from_dict(row["state"])
    candidates = tuple(CandidateV7.from_dict(value) for value in row["candidates"])
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    simulator = BeliefSimulatorV7(state, candidates)
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
    output = []
    for slate in row["slates"]:
        if "answer_utility" not in slate:
            raise ValueError("selector training slate lacks answer_utility")
        selected: List[str] = []
        steps = []
        for selected_id in slate["selected_ids"]:
            available = [
                candidate
                for candidate in candidates
                if candidate.candidate_id not in selected
            ]
            features = tuple(
                selector_features_v7(
                    state,
                    candidate,
                    simulator,
                    selected,
                    retrieval_score_z=score_z[candidate.candidate_id],
                    answer_value_artifact=answer_value_artifact,
                )
                for candidate in available
            )
            steps.append(
                SelectorSlateStepV7(
                    candidate_ids=tuple(
                        candidate.candidate_id for candidate in available
                    ),
                    features=features,
                    retriever_logits=tuple(
                        score_z[candidate.candidate_id] for candidate in available
                    ),
                    selected_id=str(selected_id),
                )
            )
            selected.append(str(selected_id))
        output.append(
            SelectorSlateExampleV7(
                question_id=state.question_id,
                state_id=str(row.get("state_id") or f"{state.question_id}:{state.turn_index}"),
                slate_id=str(slate["slate_id"]),
                steps=tuple(steps),
                answer_utility=float(slate["answer_utility"]),
            )
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--answer_value_artifact", required=True)
    parser.add_argument("--answer_value_crossfit_bundle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit_split", default="fit")
    parser.add_argument("--validation_split", default="validation")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--selector_temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable_feature",
        action="append",
        choices=SELECTOR_FEATURE_NAMES_V7,
        default=[],
        help="Train and freeze an explicit feature ablation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser()
    answer_value = load_answer_value_artifact_v7(
        Path(args.answer_value_artifact).expanduser()
    )
    crossfit = (
        load_answer_value_crossfit_bundle_v7(
            Path(args.answer_value_crossfit_bundle).expanduser()
        )
        if args.answer_value_crossfit_bundle
        else None
    )
    fit_slates: List[SelectorSlateExampleV7] = []
    validation_slates: List[SelectorSlateExampleV7] = []
    for row in _read_jsonl(input_path):
        split = str(row.get("split") or "")
        state = SelectorStateV7.from_dict(row["state"])
        row_answer_value = (
            crossfit.artifact_for_question(state.question_id)
            if crossfit is not None and split == args.fit_split
            else answer_value
        )
        if split == args.fit_split:
            fit_slates.extend(_slates(row, row_answer_value))
        elif split == args.validation_split:
            validation_slates.extend(_slates(row, row_answer_value))
    config = SelectorFitConfigV7(
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        selector_temperature=args.selector_temperature,
        seed=args.seed,
        disabled_features=tuple(args.disable_feature),
    )
    artifact = fit_selector_from_slates_v7(
        fit_slates,
        validation_slates,
        config=config,
        metadata={
            "source": str(input_path),
            "answer_value_artifact": str(args.answer_value_artifact),
            "answer_value_crossfit_bundle": args.answer_value_crossfit_bundle,
            "evidence_membership_supervision_used": False,
        },
    )
    output_path = Path(args.output).expanduser()
    save_selector_artifact_v7(output_path, artifact)
    print(json.dumps(artifact.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
