"""Validate ACEC v7 G2/G3 artifact gates before inference or GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.answer_value_v7 import load_answer_value_artifact_v7  # noqa: E402
from belief.acec.set_selector_v7 import load_selector_artifact_v7  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer_value_artifact", required=True)
    parser.add_argument("--selector_artifact", required=True)
    parser.add_argument(
        "--selector_eval",
        help="Independent test-split selector_heldout_eval.json.",
    )
    parser.add_argument(
        "--selected_calibration",
        help="Evaluator-only selected_calibration.json.",
    )
    parser.add_argument(
        "--delta_c_ablation",
        help="Independent full-vs-no-delta-C ablation report.",
    )
    parser.add_argument("--min_value_pairwise", type=float, default=0.58)
    parser.add_argument("--min_value_spearman", type=float, default=0.20)
    parser.add_argument("--min_selector_win_rate", type=float, default=0.55)
    parser.add_argument("--allow_failed", action="store_true")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    value = load_answer_value_artifact_v7(
        Path(args.answer_value_artifact).expanduser()
    )
    selector = load_selector_artifact_v7(
        Path(args.selector_artifact).expanduser()
    )
    checks = {
        "value_pairwise": float(
            value.metrics.get("validation_pairwise_accuracy", 0.0)
        )
        >= args.min_value_pairwise,
        "value_spearman": float(
            value.metrics.get("validation_spearman", 0.0)
        )
        >= args.min_value_spearman,
        "selector_win_rate": float(
            selector.metrics.get("validation_state_win_rate_vs_relevance", 0.0)
        )
        >= args.min_selector_win_rate,
        "selector_mean_utility": float(
            selector.metrics.get(
                "validation_mean_utility_gain_vs_relevance", 0.0
            )
        )
        > 0.0,
    }
    selector_eval = None
    if args.selector_eval:
        selector_eval = json.loads(
            Path(args.selector_eval).expanduser().read_text(encoding="utf-8")
        )
        if selector_eval.get("schema") != "acec_selector_heldout_eval_v7":
            raise ValueError("selector_eval has the wrong schema")
        checks["selector_heldout"] = bool(selector_eval.get("go"))
        checks["selector_eval_artifact_match"] = (
            str(selector_eval.get("selector_artifact_sha256"))
            == selector.sha256
            and str(selector_eval.get("answer_value_artifact_sha256"))
            == value.sha256
        )
    selected_calibration = None
    if args.selected_calibration:
        selected_calibration = json.loads(
            Path(args.selected_calibration)
            .expanduser()
            .read_text(encoding="utf-8")
        )
        if selected_calibration.get("schema") != "acec_selected_calibration_v7":
            raise ValueError("selected_calibration has the wrong schema")
        checks["selected_calibration"] = bool(selected_calibration.get("go"))
        checks["calibration_artifact_match"] = (
            str(selected_calibration.get("selector_artifact_sha256"))
            == selector.sha256
            and str(selected_calibration.get("answer_value_artifact_sha256"))
            == value.sha256
        )
    delta_c_ablation = None
    if args.delta_c_ablation:
        delta_c_ablation = json.loads(
            Path(args.delta_c_ablation)
            .expanduser()
            .read_text(encoding="utf-8")
        )
        if delta_c_ablation.get("schema") != "acec_selector_delta_c_ablation_v7":
            raise ValueError("delta_c_ablation has the wrong schema")
        checks["delta_c_ablation"] = bool(delta_c_ablation.get("go"))
        checks["delta_c_full_artifact_match"] = (
            str(delta_c_ablation.get("full_selector_artifact_sha256"))
            == selector.sha256
            and str(delta_c_ablation.get("answer_value_artifact_sha256"))
            == value.sha256
        )
    result = {
        "schema": "acec_artifact_gate_v7",
        "checks": checks,
        "pass": all(checks.values()),
        "value_metrics": dict(value.metrics),
        "selector_metrics": dict(selector.metrics),
        "selector_heldout_eval": selector_eval,
        "selected_calibration": selected_calibration,
        "delta_c_ablation": delta_c_ablation,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser()
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["pass"] and not args.allow_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
