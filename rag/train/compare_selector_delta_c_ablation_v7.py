"""Compare full and no-delta-C ACEC v7 selectors on identical test states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _state_id(row: Mapping[str, Any]) -> str:
    value = row.get("state_id")
    if value is None:
        state = row.get("state") or {}
        value = f"{state.get('question_id')}:{state.get('turn_index')}"
    return str(value)


def _utilities(
    path: Path, *, learned_slate: str, baseline_slate: str
) -> Tuple[Dict[str, Tuple[float, float]], str, str]:
    result: Dict[str, Tuple[float, float]] = {}
    selector_hashes = set()
    answer_value_hashes = set()
    for row in _read_jsonl(path):
        state_id = _state_id(row)
        if state_id in result:
            raise ValueError(f"duplicate selector evaluation state {state_id!r}")
        values = {
            str(slate["slate_id"]): float(slate["answer_utility"])
            for slate in row.get("slates") or ()
            if "answer_utility" in slate
        }
        if learned_slate not in values or baseline_slate not in values:
            raise ValueError(
                f"state {state_id!r} lacks learned/baseline answer utility"
            )
        learned = next(
            slate
            for slate in row.get("slates") or ()
            if str(slate.get("slate_id")) == learned_slate
        )
        trace = learned.get("selection_trace") or {}
        selector_hashes.add(str(trace.get("selector_artifact_sha256") or ""))
        answer_value_hashes.add(
            str(trace.get("answer_value_artifact_sha256") or "")
        )
        result[state_id] = (values[learned_slate], values[baseline_slate])
    if not result:
        raise ValueError(f"selector evaluation is empty: {path}")
    if (
        len(selector_hashes) != 1
        or "" in selector_hashes
        or len(answer_value_hashes) != 1
        or "" in answer_value_hashes
    ):
        raise ValueError(f"selector evaluation mixes artifact hashes: {path}")
    return (
        result,
        next(iter(selector_hashes)),
        next(iter(answer_value_hashes)),
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True)
    parser.add_argument("--no_delta_c", required=True)
    parser.add_argument("--output")
    parser.add_argument("--learned_slate", default="learned_v7")
    parser.add_argument("--baseline_slate", default="relevance")
    parser.add_argument("--min_win_rate_gap", type=float, default=0.02)
    parser.add_argument("--bootstrap_iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--require_positive_ci", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    full, full_selector_hash, full_answer_value_hash = _utilities(
        Path(args.full).expanduser(),
        learned_slate=args.learned_slate,
        baseline_slate=args.baseline_slate,
    )
    no_delta_c, ablated_selector_hash, ablated_answer_value_hash = _utilities(
        Path(args.no_delta_c).expanduser(),
        learned_slate=args.learned_slate,
        baseline_slate=args.baseline_slate,
    )
    if set(full) != set(no_delta_c):
        raise ValueError("full and no-delta-C evaluations use different states")
    if full_answer_value_hash != ablated_answer_value_hash:
        raise ValueError("full and no-delta-C selectors use different value artifacts")
    if full_selector_hash == ablated_selector_hash:
        raise ValueError("full and no-delta-C evaluations use the same selector artifact")

    rows = []
    for state_id in sorted(full):
        if abs(full[state_id][1] - no_delta_c[state_id][1]) > 1e-5:
            raise ValueError(
                f"baseline answer utility differs for state {state_id!r}; "
                "the ablation is not compute-matched"
            )
        full_gain = full[state_id][0] - full[state_id][1]
        ablated_gain = no_delta_c[state_id][0] - no_delta_c[state_id][1]
        rows.append(
            (
                full_gain,
                ablated_gain,
                float(full_gain > 0.0) - float(ablated_gain > 0.0),
                full_gain - ablated_gain,
            )
        )
    size = len(rows)
    full_win_rate = sum(row[0] > 0.0 for row in rows) / size
    ablated_win_rate = sum(row[1] > 0.0 for row in rows) / size
    win_rate_gap = full_win_rate - ablated_win_rate
    mean_gain_gap = sum(row[3] for row in rows) / size

    rng = random.Random(args.seed)
    boot_win_gaps: List[float] = []
    boot_mean_gaps: List[float] = []
    for _ in range(args.bootstrap_iterations):
        sample = [rows[rng.randrange(size)] for _ in range(size)]
        boot_win_gaps.append(sum(row[2] for row in sample) / size)
        boot_mean_gaps.append(sum(row[3] for row in sample) / size)
    win_ci = (
        _percentile(boot_win_gaps, 0.025),
        _percentile(boot_win_gaps, 0.975),
    )
    mean_ci = (
        _percentile(boot_mean_gaps, 0.025),
        _percentile(boot_mean_gaps, 0.975),
    )
    go_ci = (
        win_ci[0] > 0.0 and mean_ci[0] > 0.0
        if args.require_positive_ci
        else True
    )
    result = {
        "schema": "acec_selector_delta_c_ablation_v7",
        "states": size,
        "full_win_rate_vs_relevance": full_win_rate,
        "no_delta_c_win_rate_vs_relevance": ablated_win_rate,
        "win_rate_gap": win_rate_gap,
        "win_rate_gap_ci95": list(win_ci),
        "mean_answer_utility_gap": mean_gain_gap,
        "mean_answer_utility_gap_ci95": list(mean_ci),
        "min_win_rate_gap": args.min_win_rate_gap,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "full_selector_artifact_sha256": full_selector_hash,
        "no_delta_c_selector_artifact_sha256": ablated_selector_hash,
        "answer_value_artifact_sha256": full_answer_value_hash,
        "go_win_rate_gap": win_rate_gap >= args.min_win_rate_gap,
        "go_mean_utility_gap": mean_gain_gap > 0.0,
        "go_paired_bootstrap": go_ci,
    }
    result["go"] = bool(
        result["go_win_rate_gap"]
        and result["go_mean_utility_gap"]
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
