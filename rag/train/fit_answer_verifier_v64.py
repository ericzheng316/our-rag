"""Fit the question-disjoint ACEC v6.4 answer-correctness verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.answer_verifier_v64 import (  # noqa: E402
    TrajectoryCandidateV64,
    VerifierExampleV64,
    fit_answer_verifier_v64,
    save_answer_verifier_artifact_v64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_examples(path: Path) -> List[VerifierExampleV64]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload: Dict[str, Any] = json.loads(line)
            if "answer_correct" not in payload:
                raise ValueError(
                    f"{path}:{line_number} lacks evaluator-side answer_correct"
                )
            candidate = TrajectoryCandidateV64.from_dict(payload)
            examples.append(
                VerifierExampleV64.from_candidate(
                    candidate, correct=bool(payload["answer_correct"])
                )
            )
    if not examples:
        raise ValueError(f"verifier example file is empty: {path}")
    return examples


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--l2", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fit_path = Path(args.fit).expanduser()
    validation_path = Path(args.validation).expanduser()
    output_path = Path(args.output).expanduser()
    for path in (fit_path, validation_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing/empty verifier input: {path}")
    if fit_path.resolve() == validation_path.resolve():
        raise ValueError("verifier fit and validation files must differ")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite verifier artifact: {output_path}")

    fit_examples = _read_examples(fit_path)
    validation_examples = _read_examples(validation_path)
    artifact = fit_answer_verifier_v64(
        fit_examples,
        validation_examples,
        learning_rate=args.learning_rate,
        steps=args.steps,
        l2=args.l2,
        metadata={
            "git_commit": _git_commit(),
            "fit_source_sha256": _sha256(fit_path),
            "validation_source_sha256": _sha256(validation_path),
            "supervision": "answer_correctness_only",
            "supporting_fact_membership_used": False,
            "fit_question_ids": sorted(
                {example.question_id for example in fit_examples}
            ),
            "validation_question_ids": sorted(
                {example.question_id for example in validation_examples}
            ),
        },
    )
    save_answer_verifier_artifact_v64(output_path, artifact)
    print(json.dumps(artifact.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
