"""Validate ACEC v7 one-episode GPU smoke metrics before longer training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max_format_error_rate", type=float, default=0.01)
    parser.add_argument("--max_selector_overhead_p95", type=float, default=0.50)
    parser.add_argument("--min_selection_trace_rate", type=float, default=1.0)
    parser.add_argument("--allow_failed", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows: List[Dict[str, Any]] = []
    with Path(args.input).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("v7 smoke metrics are empty")
    latest = rows[-1]
    numeric = {
        key: float(value)
        for key, value in latest.items()
        if key not in {"schema", "episode"}
    }
    finite = all(math.isfinite(value) for value in numeric.values())
    checks = {
        "finite_metrics": finite,
        "format_error_rate": float(latest.get("format_error_rate", 1.0))
        < args.max_format_error_rate,
        "selector_overhead_p95": float(
            latest.get("selector_overhead_fraction_p95", math.inf)
        )
        <= args.max_selector_overhead_p95,
        "selection_trace_rate": float(
            latest.get("selection_trace_rate", 0.0)
        )
        >= args.min_selection_trace_rate,
        "selector_exercised": float(
            latest.get("selector_calls_per_rollout", 0.0)
        )
        > 0.0,
    }
    result = {
        "schema": "acec_smoke_gate_v7",
        "episodes_recorded": len(rows),
        "latest_episode": int(latest.get("episode", len(rows))),
        "checks": checks,
        "pass": all(checks.values()),
        "metrics": latest,
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
