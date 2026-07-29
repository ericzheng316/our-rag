"""Audit post-selection hit calibration and winner's curse for ACEC v7.

The calibrated quantity is the frozen observation model's maximum
action-conditioned slot-hit probability for a candidate.  Conditional
coverage gain is a state-weighted utility, not a Bernoulli probability, so it
must not be compared directly with binary supporting-document membership.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.contracts_v7 import CandidateV7  # noqa: E402


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _ece(rows: Sequence[Tuple[float, int]], bins: int) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    value = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            (score, label)
            for score, label in rows
            if lower <= score < upper
            or (bin_index == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        confidence = sum(score for score, _ in members) / len(members)
        accuracy = sum(label for _, label in members) / len(members)
        value += len(members) / total * abs(confidence - accuracy)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max_selected_ece", type=float, default=0.12)
    parser.add_argument("--max_ece_ratio", type=float, default=1.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    oracle = {
        str(row["state_id"]): {
            str(value) for value in row.get("gold_candidate_ids") or ()
        }
        for row in _read_jsonl(Path(args.oracle).expanduser())
    }
    selected_rows: List[Tuple[float, int]] = []
    pool_rows: List[Tuple[float, int]] = []
    states = 0
    selector_hashes = set()
    answer_value_hashes = set()
    for row in _read_jsonl(Path(args.selection).expanduser()):
        state_id = str(row.get("state_id") or "")
        if state_id not in oracle:
            continue
        gold = oracle[state_id]
        candidates = tuple(
            CandidateV7.from_dict(value) for value in row["candidates"]
        )
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        for candidate in candidates:
            hit_probability = max(candidate.slot_hit_probabilities, default=0.0)
            pool_rows.append(
                (
                    min(max(float(hit_probability), 0.0), 1.0),
                    int(candidate.candidate_id in gold),
                )
            )
        learned = next(
            (
                slate
                for slate in row.get("slates") or ()
                if slate.get("slate_id") == "learned_v7"
            ),
            None,
        )
        if learned is None:
            continue
        trace = learned.get("selection_trace") or {}
        selector_hashes.add(str(trace.get("selector_artifact_sha256") or ""))
        answer_value_hashes.add(
            str(trace.get("answer_value_artifact_sha256") or "")
        )
        for step in trace.get("steps") or ():
            candidate_id = str(step["selected_id"])
            if candidate_id not in by_id:
                raise ValueError(
                    f"selected candidate {candidate_id!r} is outside state {state_id}"
                )
            hit_probability = max(
                by_id[candidate_id].slot_hit_probabilities, default=0.0
            )
            selected_rows.append(
                (
                    min(max(float(hit_probability), 0.0), 1.0),
                    int(candidate_id in gold),
                )
            )
        states += 1
    if states == 0:
        raise ValueError("no selection rows joined the oracle sidecar")
    if (
        len(selector_hashes) != 1
        or "" in selector_hashes
        or len(answer_value_hashes) != 1
        or "" in answer_value_hashes
    ):
        raise ValueError("selected calibration mixes or omits artifact hashes")
    selected_ece = _ece(selected_rows, args.bins)
    pool_ece = _ece(pool_rows, args.bins)
    ratio = selected_ece / max(pool_ece, 1e-6)
    result = {
        "schema": "acec_selected_calibration_v7",
        "states": states,
        "selected_rows": len(selected_rows),
        "pool_rows": len(pool_rows),
        "selected_ece": selected_ece,
        "pool_ece": pool_ece,
        "selected_to_pool_ece_ratio": ratio,
        "calibrated_quantity": "max_action_conditioned_slot_hit_probability",
        "label": "candidate_title_matches_any_gold_supporting_title",
        "selector_artifact_sha256": next(iter(selector_hashes)),
        "answer_value_artifact_sha256": next(iter(answer_value_hashes)),
        "go_selected_ece": selected_ece <= args.max_selected_ece,
        "go_winners_curse": ratio <= args.max_ece_ratio,
    }
    result["go"] = bool(result["go_selected_ece"] and result["go_winners_curse"])
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output).expanduser()
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
