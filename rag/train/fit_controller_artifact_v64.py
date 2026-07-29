"""Freeze a cost-matched v6.4 controller from disjoint validation sweeps.

Each input is a completed ``infer_belief_controller_v64.py`` results.json made
on the same calibration question ids but with a different frozen threshold
configuration.  Selection minimizes the absolute mean-retrieval cost gap to a
declared reference; answer EM is used only as a tie-breaker on this disjoint
calibration split.  The held-out evaluator accepts only the resulting immutable
gate-passing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.inference_controller_v64 import (  # noqa: E402
    ControllerArtifactV64,
    ControllerConfigV64,
    ControllerMode,
    save_controller_artifact_v64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _id_hash(ids: List[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(set(ids))).encode("utf-8")
    ).hexdigest()


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "acec_v64_factorial_run":
        raise ValueError(f"not a v6.4 factorial result: {path}")
    return dict(payload)


def _controller_config(payload: Mapping[str, Any], arm: str) -> ControllerConfigV64:
    if "__" not in arm:
        raise ValueError("arm must be POLICY__CONTROLLER")
    mode_name = arm.rsplit("__", 1)[1]
    config = dict(payload["controller_configs"][mode_name])
    config["mode"] = ControllerMode(config["mode"])
    return ControllerConfigV64(**config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_result", action="append", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--selector", default="first@1")
    parser.add_argument("--target_mean_retrievals", type=float, required=True)
    parser.add_argument("--max_mean_cost_gap", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.target_mean_retrievals < 0.0 or args.max_mean_cost_gap < 0.0:
        raise ValueError("cost target and tolerance must be non-negative")
    output = Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite controller artifact: {output}")

    candidates: List[Dict[str, Any]] = []
    calibration_hash = None
    data_hash = None
    calibration_question_ids = None
    for source_value in args.sweep_result:
        source = Path(source_value).expanduser()
        payload = _load(source)
        source_question_ids = sorted(
            {str(value) for value in payload.get("question_ids", [])}
        )
        if not source_question_ids:
            raise ValueError(f"{source} lacks explicit calibration question ids")
        if _id_hash(source_question_ids) != str(payload.get("question_id_sha256")):
            raise ValueError(f"{source} question ids do not match their hash")
        if (
            calibration_question_ids is not None
            and source_question_ids != calibration_question_ids
        ):
            raise ValueError("controller sweeps do not share identical question ids")
        calibration_question_ids = source_question_ids
        for key, expected, label in (
            ("question_id_sha256", calibration_hash, "question-id split"),
            ("data_sha256", data_hash, "data source"),
        ):
            value = str(payload.get(key) or "")
            if not value:
                raise ValueError(f"{source} lacks {key}")
            if expected is not None and value != expected:
                raise ValueError(f"controller sweeps do not share the same {label}")
            if key == "question_id_sha256":
                calibration_hash = value
            else:
                data_hash = value
        if args.arm not in payload.get("runtime_diagnostics", {}):
            raise ValueError(f"{source} lacks runtime arm {args.arm!r}")
        if args.selector not in (payload.get("factorial", {}).get(args.arm) or {}):
            raise ValueError(
                f"{source} lacks selector {args.selector!r} for {args.arm!r}"
            )
        mean_cost = float(
            payload["runtime_diagnostics"][args.arm]["mean_retrieval_calls"]
        )
        em = float(payload["factorial"][args.arm][args.selector]["em"])
        if not math.isfinite(mean_cost) or not math.isfinite(em):
            raise ValueError(f"{source} contains non-finite calibration metrics")
        candidates.append(
            {
                "source": str(source),
                "source_sha256": _sha256(source),
                "config": _controller_config(payload, args.arm),
                "mean_retrievals": mean_cost,
                "answer_em": em,
                "absolute_cost_gap": abs(mean_cost - args.target_mean_retrievals),
            }
        )

    selected = min(
        candidates,
        key=lambda row: (
            row["absolute_cost_gap"],
            -row["answer_em"],
            row["source"],
        ),
    )
    if selected["absolute_cost_gap"] > args.max_mean_cost_gap:
        raise ValueError(
            "no controller threshold sweep matched the reference cost: "
            f"best_gap={selected['absolute_cost_gap']:.6f} "
            f"tolerance={args.max_mean_cost_gap:.6f}"
        )
    artifact = ControllerArtifactV64(
        config=selected["config"],
        calibration_split_id_hash=calibration_hash,
        expected_mean_retrievals=selected["mean_retrievals"],
        metadata={
            "git_commit": _git_commit(),
            "data_sha256": data_hash,
            "arm": args.arm,
            "selector": args.selector,
            "target_mean_retrievals": args.target_mean_retrievals,
            "max_mean_cost_gap": args.max_mean_cost_gap,
            "selected_source": selected["source"],
            "selected_source_sha256": selected["source_sha256"],
            "selection_rule": "min_abs_cost_gap_then_max_answer_em",
            "answer_labels_used_only_on_disjoint_calibration": True,
            "calibration_question_ids": calibration_question_ids,
            "candidate_sweeps": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "config"
                }
                for row in candidates
            ],
            "gate": {
                "pass": True,
                "absolute_mean_retrieval_gap": selected["absolute_cost_gap"],
                "maximum_absolute_mean_retrieval_gap": args.max_mean_cost_gap,
            },
        },
    )
    save_controller_artifact_v64(output, artifact)
    print(json.dumps(artifact.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
