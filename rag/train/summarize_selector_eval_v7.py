"""Summarize question-disjoint ACEC v7 selector utility with paired bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(
    deltas: Sequence[float], *, iterations: int, seed: int
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not deltas:
        return (0.0, 0.0), (0.0, 0.0)
    rng = random.Random(seed)
    means: List[float] = []
    wins: List[float] = []
    size = len(deltas)
    for _ in range(iterations):
        sample = [deltas[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
        wins.append(sum(value > 0.0 for value in sample) / size)
    return (
        (_percentile(means, 0.025), _percentile(means, 0.975)),
        (_percentile(wins, 0.025), _percentile(wins, 0.975)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--learned_slate", default="learned_v7")
    parser.add_argument("--baseline_slate", default="relevance")
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--min_win_rate", type=float, default=0.55)
    parser.add_argument("--require_positive_ci", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    deltas: List[float] = []
    question_ids = set()
    selector_hashes = set()
    answer_value_hashes = set()
    for row in _read_jsonl(Path(args.input).expanduser()):
        learned = next(
            (
                slate
                for slate in row.get("slates") or ()
                if str(slate.get("slate_id")) == args.learned_slate
            ),
            None,
        )
        if learned is None or not learned.get("selection_trace"):
            raise ValueError("learned selector slate lacks its frozen trace")
        trace = learned["selection_trace"]
        selector_hashes.add(str(trace.get("selector_artifact_sha256") or ""))
        answer_value_hashes.add(
            str(trace.get("answer_value_artifact_sha256") or "")
        )
        utilities = {
            str(slate["slate_id"]): float(slate["answer_utility"])
            for slate in row.get("slates") or ()
            if "answer_utility" in slate
        }
        if (
            args.learned_slate not in utilities
            or args.baseline_slate not in utilities
        ):
            raise ValueError(
                "scored selector row lacks learned or baseline answer utility"
            )
        deltas.append(
            utilities[args.learned_slate] - utilities[args.baseline_slate]
        )
        state = row.get("state") or {}
        question_ids.add(str(state.get("question_id") or row.get("question") or ""))
    if not deltas:
        raise ValueError("selector evaluation input is empty")
    if (
        len(selector_hashes) != 1
        or "" in selector_hashes
        or len(answer_value_hashes) != 1
        or "" in answer_value_hashes
    ):
        raise ValueError("selector evaluation mixes or omits frozen artifact hashes")
    mean_delta = sum(deltas) / len(deltas)
    win_rate = sum(value > 0.0 for value in deltas) / len(deltas)
    tie_rate = sum(value == 0.0 for value in deltas) / len(deltas)
    mean_ci, win_ci = _bootstrap(
        deltas, iterations=args.bootstrap_iterations, seed=args.seed
    )
    go_ci = mean_ci[0] > 0.0 if args.require_positive_ci else win_ci[0] > 0.5
    result = {
        "schema": "acec_selector_heldout_eval_v7",
        "rows": len(deltas),
        "questions": len(question_ids),
        "learned_slate": args.learned_slate,
        "baseline_slate": args.baseline_slate,
        "mean_answer_utility_gain": mean_delta,
        "answer_utility_gain_ci95": list(mean_ci),
        "win_rate": win_rate,
        "win_rate_ci95": list(win_ci),
        "tie_rate": tie_rate,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "selector_artifact_sha256": next(iter(selector_hashes)),
        "answer_value_artifact_sha256": next(iter(answer_value_hashes)),
        "go_mean_utility": mean_delta > 0.0,
        "go_win_rate": win_rate >= args.min_win_rate,
        "go_paired_bootstrap": go_ci,
    }
    result["go"] = bool(
        result["go_mean_utility"]
        and result["go_win_rate"]
        and result["go_paired_bootstrap"]
    )
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
