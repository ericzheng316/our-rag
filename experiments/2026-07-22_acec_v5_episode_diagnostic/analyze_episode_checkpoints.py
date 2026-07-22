#!/usr/bin/env python3
"""Diagnose ACEC-v5 checkpoint movement from saved evaluation artifacts.

The script is inference-free. It aligns the fixed 256-question held-out
manifest, R3 processed predictions, 16-question evaluation-batch aggregates,
and the 100-episode training log. It writes bounded CSV/JSON artifacts for a
reader-facing report and preserves all comparisons at their available grain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


VARIANTS = ("step50", "step75", "step100")
COMPARISONS = (("step50", "step75"), ("step50", "step100"), ("step75", "step100"))
TRAINING_COLUMNS = (
    "episode",
    "loss",
    "mean_R",
    "avg_turns",
    "online_em",
    "answer_rate",
    "sf_recall",
    "retrievals",
    "fmt",
    "empty",
    "kl",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    if x_ss <= 0.0 or y_ss <= 0.0:
        return None
    return numerator / math.sqrt(x_ss * y_ss)


def paired_bootstrap(
    deltas: Sequence[float], *, samples: int = 10_000, seed: int = 20260722
) -> List[float]:
    rng = random.Random(seed)
    n = len(deltas)
    estimates = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)
    )
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]]


def question_type(question: str) -> str:
    q = question.lower().strip()
    if q.startswith("who"):
        return "who"
    if q.startswith("when") or "what year" in q or "what date" in q:
        return "when/year"
    if q.startswith("where"):
        return "where"
    if q.startswith(("how many", "how much", "how old")):
        return "numeric"
    if q.startswith("which"):
        return "which"
    if q.startswith(("are ", "is ", "was ", "were ", "do ", "did ")):
        return "yes/no"
    if q.startswith("what"):
        return "what"
    return "other"


def word_count(text: str | None) -> int:
    return len(re.findall(r"\w+", text or ""))


def transition(before: float, after: float) -> str:
    if after > before:
        return "win"
    if after < before:
        return "loss"
    if before == 1.0:
        return "both_correct"
    return "both_wrong"


def parse_training_log(path: Path) -> List[Dict[str, float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"Episode\s+(\d+)\s+\|\s+loss=([^|]+)\|\s+mean_R=([^|]+)"
        r"\|\s+avg_turns=([^|]+)\|\s+online_EM=([^|]+)"
        r"\|\s+answer_rate=([^|]+)\|\s+SF_recall=([^|]+)"
        r"\|\s+retrievals=([^|]+)\|\s+fmt=([^|]+)"
        r"\|\s+empty=([^|]+)\|\s+KL=([^|]+)"
    )
    rows: List[Dict[str, float]] = []
    for match in pattern.finditer(text):
        values = [int(match.group(1))] + [float(value.strip()) for value in match.groups()[1:]]
        rows.append(dict(zip(TRAINING_COLUMNS, values)))
    if len(rows) != 100:
        raise ValueError(f"expected 100 episode rows, found {len(rows)} in {path}")
    for index, row in enumerate(rows):
        window = rows[max(0, index - 9) : index + 1]
        for field in ("mean_R", "online_em", "sf_recall", "retrievals", "kl"):
            row[f"rolling10_{field}"] = mean(item[field] for item in window)
    return rows


def parse_eval_batches(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"\[eval\] model=(\w+) done=(\d+)/256 raw_EM=([0-9.]+) "
        r"SF=([0-9.]+) retrievals=([0-9.]+)"
    )
    rows = []
    for match in pattern.finditer(text):
        rows.append(
            {
                "variant": match.group(1),
                "batch": int(match.group(2)) // 16,
                "question_count": 16,
                "raw_em": float(match.group(3)),
                "gold_sf_recall": float(match.group(4)),
                "retrieval_calls": float(match.group(5)),
            }
        )
    focal = [row for row in rows if row["variant"] in VARIANTS]
    if len(focal) != len(VARIANTS) * 16:
        raise ValueError(f"expected 48 focal batch rows, found {len(focal)}")
    return focal


def build_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    heldout_dir = Path(args.heldout_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = heldout_dir / "manifest.jsonl"
    manifest = read_jsonl(manifest_path)
    if len(manifest) != 256 or len({row["id"] for row in manifest}) != 256:
        raise ValueError("held-out manifest must contain 256 unique questions")

    processed_paths = {
        variant: processed_dir / f"processed_predictions_{variant}.jsonl"
        for variant in VARIANTS
    }
    processed = {
        variant: {row["id"]: row for row in read_jsonl(path)}
        for variant, path in processed_paths.items()
    }
    expected_ids = {row["id"] for row in manifest}
    for variant in VARIANTS:
        if set(processed[variant]) != expected_ids:
            raise ValueError(f"{variant} ids do not match the manifest")

    per_question: List[Dict[str, Any]] = []
    for index, item in enumerate(manifest):
        qid = item["id"]
        row: Dict[str, Any] = {
            "id": qid,
            "index": index,
            "batch": index // 16 + 1,
            "question_type": question_type(item["question"]),
            "question": item["question"],
            "golden_answers": " | ".join(item["golden_answers"]),
            "gold_sf_titles": " | ".join(item.get("sf_titles") or []),
        }
        for variant in VARIANTS:
            record = processed[variant][qid]
            row.update(
                {
                    f"{variant}_prediction": record.get("prediction") or "",
                    f"{variant}_candidates": " | ".join(record.get("candidate_answers") or []),
                    f"{variant}_prediction_words": word_count(record.get("prediction")),
                    f"{variant}_direct_em": float(record["r3_direct_em"]),
                    f"{variant}_direct_f1": float(record["r3_direct_f1"]),
                    f"{variant}_processed_em": float(record["r3_processed_em"]),
                    f"{variant}_processed_f1": float(record["r3_processed_f1"]),
                }
            )
        for before, after in COMPARISONS:
            row[f"{before}_to_{after}_transition"] = transition(
                row[f"{before}_processed_em"], row[f"{after}_processed_em"]
            )
            row[f"{before}_to_{after}_f1_delta"] = (
                row[f"{after}_processed_f1"] - row[f"{before}_processed_f1"]
            )
            row[f"{before}_to_{after}_prediction_changed"] = int(
                row[f"{before}_prediction"] != row[f"{after}_prediction"]
            )
        row["correctness_pattern_50_75_100"] = "".join(
            str(int(row[f"{variant}_processed_em"])) for variant in VARIANTS
        )
        per_question.append(row)

    overall: Dict[str, Any] = {}
    for variant in VARIANTS:
        overall[variant] = {
            metric: mean(row[f"{variant}_{metric}"] for row in per_question)
            for metric in ("direct_em", "direct_f1", "processed_em", "processed_f1")
        }
        overall[variant]["mean_prediction_words"] = mean(
            row[f"{variant}_prediction_words"] for row in per_question
        )

    comparisons: Dict[str, Any] = {}
    for before, after in COMPARISONS:
        name = f"{before}_to_{after}"
        em_deltas = [
            row[f"{after}_processed_em"] - row[f"{before}_processed_em"]
            for row in per_question
        ]
        f1_deltas = [row[f"{before}_to_{after}_f1_delta"] for row in per_question]
        statuses = Counter(row[f"{before}_to_{after}_transition"] for row in per_question)
        changed = [
            row for row in per_question if row[f"{before}_to_{after}_prediction_changed"]
        ]
        same = [
            row for row in per_question if not row[f"{before}_to_{after}_prediction_changed"]
        ]
        lost = [row for row in per_question if row[f"{before}_to_{after}_transition"] == "loss"]
        comparisons[name] = {
            "processed_em_delta": mean(em_deltas),
            "processed_em_ci95": paired_bootstrap(em_deltas),
            "processed_f1_delta": mean(f1_deltas),
            "processed_f1_ci95": paired_bootstrap(f1_deltas),
            "wins": statuses["win"],
            "losses": statuses["loss"],
            "both_correct": statuses["both_correct"],
            "both_wrong": statuses["both_wrong"],
            "prediction_changed": len(changed),
            "prediction_unchanged": len(same),
            "changed_f1_improved": sum(row[f"{before}_to_{after}_f1_delta"] > 1e-12 for row in changed),
            "changed_f1_worse": sum(row[f"{before}_to_{after}_f1_delta"] < -1e-12 for row in changed),
            "lost_exact_with_zero_after_f1": sum(row[f"{after}_processed_f1"] == 0.0 for row in lost),
            "same_prediction_nonzero_f1_delta": sum(
                abs(row[f"{before}_to_{after}_f1_delta"]) > 1e-12 for row in same
            ),
        }

    pattern_counts = Counter(row["correctness_pattern_50_75_100"] for row in per_question)
    stability = {
        "same_prediction_all_three": sum(
            row["step50_prediction"] == row["step75_prediction"] == row["step100_prediction"]
            for row in per_question
        ),
        "processed_correct_all_three": pattern_counts["111"],
        "processed_wrong_all_three": pattern_counts["000"],
        "processed_correct_any": len(per_question) - pattern_counts["000"],
        "correctness_pattern_counts": dict(sorted(pattern_counts.items())),
    }

    driver_rows: List[Dict[str, Any]] = []
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_question:
        by_type[row["question_type"]].append(row)
    for q_type, group in sorted(by_type.items()):
        driver: Dict[str, Any] = {"question_type": q_type, "n": len(group)}
        for variant in VARIANTS:
            driver[f"{variant}_processed_em"] = mean(
                row[f"{variant}_processed_em"] for row in group
            )
            driver[f"{variant}_processed_f1"] = mean(
                row[f"{variant}_processed_f1"] for row in group
            )
        for after in ("step75", "step100"):
            em_sum = sum(
                row[f"{after}_processed_em"] - row["step50_processed_em"] for row in group
            )
            f1_sum = sum(
                row[f"{after}_processed_f1"] - row["step50_processed_f1"] for row in group
            )
            driver[f"{after}_vs_step50_em_delta"] = em_sum / len(group)
            driver[f"{after}_vs_step50_f1_delta"] = f1_sum / len(group)
            driver[f"{after}_vs_step50_em_contribution"] = em_sum / len(per_question)
            driver[f"{after}_vs_step50_f1_contribution"] = f1_sum / len(per_question)
        driver_rows.append(driver)

    batch_rows = parse_eval_batches(heldout_dir / "eval.log")
    per_batch_questions: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_question:
        per_batch_questions[int(row["batch"])].append(row)
    for row in batch_rows:
        group = per_batch_questions[int(row["batch"])]
        variant = row["variant"]
        row["processed_em"] = mean(item[f"{variant}_processed_em"] for item in group)
        row["processed_f1"] = mean(item[f"{variant}_processed_f1"] for item in group)
    batch_lookup = {(row["variant"], row["batch"]): row for row in batch_rows}
    batch_diagnostics: Dict[str, Any] = {}
    for after in ("step75", "step100"):
        changes = []
        for batch in range(1, 17):
            before_row = batch_lookup[("step50", batch)]
            after_row = batch_lookup[(after, batch)]
            changes.append(
                {
                    "batch": batch,
                    "retrieval_delta": after_row["retrieval_calls"] - before_row["retrieval_calls"],
                    "sf_delta": after_row["gold_sf_recall"] - before_row["gold_sf_recall"],
                    "processed_em_delta": after_row["processed_em"] - before_row["processed_em"],
                    "processed_f1_delta": after_row["processed_f1"] - before_row["processed_f1"],
                }
            )
        name = f"{after}_vs_step50"
        batch_diagnostics[name] = {
            "batches": 16,
            "retrieval_down_batches": sum(row["retrieval_delta"] < 0 for row in changes),
            "sf_down_batches": sum(row["sf_delta"] < 0 for row in changes),
            "retrieval_and_sf_down_batches": sum(
                row["retrieval_delta"] < 0 and row["sf_delta"] < 0 for row in changes
            ),
            "corr_retrieval_delta_sf_delta": pearson(
                [row["retrieval_delta"] for row in changes],
                [row["sf_delta"] for row in changes],
            ),
            "corr_sf_delta_processed_f1_delta": pearson(
                [row["sf_delta"] for row in changes],
                [row["processed_f1_delta"] for row in changes],
            ),
            "corr_retrieval_delta_processed_f1_delta": pearson(
                [row["retrieval_delta"] for row in changes],
                [row["processed_f1_delta"] for row in changes],
            ),
        }

    training_rows = parse_training_log(Path(args.training_log))
    training_windows: Dict[str, Any] = {}
    for checkpoint in (25, 50, 75, 100):
        window = [
            row for row in training_rows if checkpoint - 9 <= int(row["episode"]) <= checkpoint
        ]
        training_windows[f"last10_to_step{checkpoint}"] = {
            field: mean(row[field] for row in window)
            for field in ("mean_R", "online_em", "sf_recall", "retrievals", "kl")
        }

    progress_path = processed_dir / "api_progress.jsonl"
    extracted = [row for row in read_jsonl(progress_path) if row["status"] == "extracted"]
    extraction_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in extracted:
        extraction_groups[(row["question"], row["prediction"])].append(row)
    duplicate_groups = [group for group in extraction_groups.values() if len(group) > 1]
    extractor_reuse = {
        "extracted_api_calls": len(extracted),
        "unique_question_prediction_pairs": len(extraction_groups),
        "reusable_calls_if_cached": len(extracted) - len(extraction_groups),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_groups_with_processed_em_disagreement": sum(
            len({row["r3_processed_em"] for row in group}) > 1 for group in duplicate_groups
        ),
        "duplicate_groups_with_processed_f1_disagreement": sum(
            len({round(row["r3_processed_f1"], 12) for row in group}) > 1
            for group in duplicate_groups
        ),
    }

    transition_examples = [
        {
            "id": row["id"],
            "question_type": row["question_type"],
            "question": row["question"],
            "golden_answers": row["golden_answers"],
            "pattern_50_75_100": row["correctness_pattern_50_75_100"],
            "step50_prediction": row["step50_prediction"],
            "step75_prediction": row["step75_prediction"],
            "step100_prediction": row["step100_prediction"],
            "step50_processed_f1": row["step50_processed_f1"],
            "step75_processed_f1": row["step75_processed_f1"],
            "step100_processed_f1": row["step100_processed_f1"],
        }
        for row in per_question
        if row["correctness_pattern_50_75_100"] not in {"000", "111"}
    ]

    heldout_results = json.loads((heldout_dir / "results.json").read_text(encoding="utf-8"))
    heldout_model_metrics = {
        row["name"]: row["metrics"] for row in heldout_results["models"] if row["name"] in VARIANTS
    }
    for variant in VARIANTS:
        overall[variant]["gold_sf_recall"] = float(
            heldout_model_metrics[variant]["gold_sf_recall"]
        )
        overall[variant]["retrieval_calls"] = float(
            heldout_model_metrics[variant]["retrieval_calls"]
        )
        overall[variant]["substring_em"] = float(heldout_model_metrics[variant]["sub_em"])

    summary = {
        "scope": {
            "heldout_questions": len(per_question),
            "batch_size": 16,
            "evaluation_temperature": 0.0,
            "variants": list(VARIANTS),
            "available_retrieval_grain": "16-question evaluation batch",
        },
        "overall": overall,
        "comparisons": comparisons,
        "stability": stability,
        "question_type_drivers": driver_rows,
        "batch_diagnostics": batch_diagnostics,
        "training_last10_windows": training_windows,
        "extractor_reuse_audit": extractor_reuse,
        "source_hashes": {
            "manifest.jsonl": sha256(manifest_path),
            "eval.log": sha256(heldout_dir / "eval.log"),
            "train.log": sha256(Path(args.training_log)),
            **{
                f"processed_predictions_{variant}.jsonl": sha256(path)
                for variant, path in processed_paths.items()
            },
        },
    }

    write_csv(output_dir / "per_question_diagnostic.csv", per_question)
    write_csv(output_dir / "question_type_drivers.csv", driver_rows)
    write_csv(output_dir / "batch_diagnostic.csv", batch_rows)
    write_csv(output_dir / "training_episode_metrics.csv", training_rows)
    write_csv(output_dir / "transition_examples.csv", transition_examples)
    write_json(output_dir / "analysis_results.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout_dir", required=True)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--training_log", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


if __name__ == "__main__":
    result = build_diagnostic(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
