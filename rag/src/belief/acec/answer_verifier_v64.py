"""Answer-verifier and best-of-N contracts for ACEC v6.4.

The module keeps three claims separate:

* ``oracle@N`` measures whether sampling produces a correct candidate at all;
* generic NLI selects by answer grounding alone;
* ACEC adds belief coverage/uncertainty, either with a fixed zero-shot score or
  a small correctness calibrator fitted on question-disjoint answer labels.

Gold answers are consumed only by evaluation and calibrator fitting.  Runtime
selection receives candidate features, never gold membership.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


VERIFIER_SCHEMA = "acec_answer_verifier"
VERIFIER_VERSION = 64
VERIFIER_FEATURES = (
    "grounding_score",
    "coverage",
    "coverage_confidence",
)


def _clamp_probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("verifier probability must be finite")
    return min(max(value, 0.0), 1.0)


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def answer_exact_match(prediction: str, golds: Sequence[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(gold) for gold in golds))


def _single_f1(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if (pred in special or target in special) and pred != target:
        return 0.0
    pred_tokens = pred.split()
    target_tokens = target.split()
    common = Counter(pred_tokens) & Counter(target_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2.0 * precision * recall / (precision + recall)


def answer_f1(prediction: str, golds: Sequence[str]) -> float:
    return max((_single_f1(prediction, gold) for gold in golds), default=0.0)


@dataclass(frozen=True)
class TrajectoryCandidateV64:
    question_id: str
    sample_index: int
    answer: str
    grounding_score: float
    coverage: float
    coverage_std: float
    retrieval_calls: int
    generation_calls: int = 0
    grounding_pairs: int = 0
    answered: bool = True
    format_valid: bool = True
    processed_answer: Optional[str] = None
    # 策略输出的原始自由文本，未经 v6.3 标签契约裁剪。`answer` 在缺 <answer>
    # 标签时会是空串，但模型往往已经答对了（实测 96.5–98.8% 的轨迹如此），
    # 所以原文必须单独留存，否则 processed-EM 的抽取器无米可炊。
    raw_answer: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question_id or self.sample_index < 0:
            raise ValueError("candidate requires question id and non-negative sample index")
        # `answered` 表示"策略产出了答案文本"，与是否满足 v6.3 标签契约无关。
        # 契约不合规时 `answer` 为空但 `raw_answer` 有内容，此时仍算已作答，
        # 契约合规性由 `format_valid` 单独承载。
        if self.answered and not (
            str(self.answer).strip() or str(self.raw_answer).strip()
        ):
            raise ValueError("answered candidate has no answer text")
        for name in ("grounding_score", "coverage"):
            _clamp_probability(getattr(self, name))
        if not math.isfinite(float(self.coverage_std)) or self.coverage_std < 0.0:
            raise ValueError("coverage_std must be finite and non-negative")
        if self.retrieval_calls < 0:
            raise ValueError("retrieval_calls must be non-negative")
        if self.generation_calls < 0:
            raise ValueError("generation_calls must be non-negative")
        if self.grounding_pairs < 0:
            raise ValueError("grounding_pairs must be non-negative")

    @property
    def coverage_confidence(self) -> float:
        return _clamp_probability(self.coverage * (1.0 - min(self.coverage_std, 1.0)))

    @property
    def zero_shot_acec_score(self) -> float:
        return _clamp_probability(self.grounding_score * self.coverage_confidence)

    def scoring_answer(self, processed: bool = False) -> str:
        """用于计分的答案：严格遵循 v6.3 契约，契约不合规时为空串。

        **不要**在这里回退到 raw_answer —— 本方法同时供 EM/F1 计算使用，
        回退会把严格 Answer EM 悄悄变成归一化 EM，破坏两种口径的分离。
        选择（best-of-N）请用 selection_answer()。
        """
        if processed and self.processed_answer:
            return str(self.processed_answer)
        return str(self.answer)

    def selection_answer(self, processed: bool = False) -> str:
        """用于候选间比较/分组的答案文本，契约不合规时回退到原始输出。

        majority 之类的选择器需要看到真实答案才能分组；若沿用 scoring_answer，
        契约不合规时全部候选都是空串，会退化成单一分组并总是选中 sample 0。
        """
        return self.scoring_answer(processed) or str(self.raw_answer or "")

    def feature_dict(self) -> Dict[str, float]:
        return {
            "grounding_score": float(self.grounding_score),
            "coverage": float(self.coverage),
            "coverage_confidence": float(self.coverage_confidence),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "metadata": dict(self.metadata),
            "coverage_confidence": self.coverage_confidence,
            "zero_shot_acec_score": self.zero_shot_acec_score,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryCandidateV64":
        return cls(
            question_id=str(payload["question_id"]),
            sample_index=int(payload["sample_index"]),
            answer=str(payload.get("answer") or ""),
            grounding_score=float(payload.get("grounding_score", 0.0)),
            coverage=float(payload.get("coverage", 0.0)),
            coverage_std=float(payload.get("coverage_std", 0.0)),
            retrieval_calls=int(payload.get("retrieval_calls", 0)),
            generation_calls=int(payload.get("generation_calls", 0)),
            grounding_pairs=int(payload.get("grounding_pairs", 0)),
            answered=bool(payload.get("answered", True)),
            format_valid=bool(payload.get("format_valid", True)),
            raw_answer=str(payload.get("raw_answer") or ""),
            processed_answer=(
                None
                if payload.get("processed_answer") is None
                else str(payload["processed_answer"])
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class VerifierExampleV64:
    question_id: str
    sample_index: int
    features: Mapping[str, float]
    correct: bool

    def __post_init__(self) -> None:
        if not self.question_id or self.sample_index < 0:
            raise ValueError(
                "verifier example requires a question id and non-negative sample index"
            )
        if len(self.features) != len(VERIFIER_FEATURES) or set(
            self.features
        ) != set(VERIFIER_FEATURES):
            raise ValueError(
                f"verifier example features must be exactly {VERIFIER_FEATURES}"
            )
        for name in VERIFIER_FEATURES:
            _clamp_probability(float(self.features[name]))

    @classmethod
    def from_candidate(
        cls, candidate: TrajectoryCandidateV64, *, correct: bool
    ) -> "VerifierExampleV64":
        return cls(
            question_id=candidate.question_id,
            sample_index=candidate.sample_index,
            features=candidate.feature_dict(),
            correct=bool(correct),
        )


def _sigmoid(value: float) -> float:
    if value >= 0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


@dataclass(frozen=True)
class AnswerVerifierArtifactV64:
    weights: Mapping[str, float]
    intercept: float
    fit_question_id_sha256: str
    validation_question_id_sha256: str
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = VERIFIER_SCHEMA
    version: int = VERIFIER_VERSION

    def __post_init__(self) -> None:
        if len(self.weights) != len(VERIFIER_FEATURES) or set(
            self.weights
        ) != set(VERIFIER_FEATURES):
            raise ValueError(
                f"verifier features must be exactly {VERIFIER_FEATURES}, "
                f"got {tuple(self.weights)}"
            )
        if self.fit_question_id_sha256 == self.validation_question_id_sha256:
            raise ValueError("verifier fit and validation id hashes must differ")
        if not all(math.isfinite(float(value)) for value in self.weights.values()):
            raise ValueError("verifier weights must be finite")
        if not math.isfinite(float(self.intercept)):
            raise ValueError("verifier intercept must be finite")
        for metadata_key, expected_hash in (
            ("fit_question_ids", self.fit_question_id_sha256),
            ("validation_question_ids", self.validation_question_id_sha256),
        ):
            ids = self.metadata.get(metadata_key)
            if ids and _id_hash(str(value) for value in ids) != expected_hash:
                raise ValueError(
                    f"verifier {metadata_key} do not match the recorded hash"
                )

    def score_features(self, features: Mapping[str, float]) -> float:
        if len(features) != len(VERIFIER_FEATURES) or set(features) != set(
            VERIFIER_FEATURES
        ):
            raise ValueError(
                f"verifier scoring features must be exactly {VERIFIER_FEATURES}"
            )
        checked = {
            name: _clamp_probability(float(features[name]))
            for name in VERIFIER_FEATURES
        }
        logit = float(self.intercept) + sum(
            float(self.weights[name]) * checked[name]
            for name in VERIFIER_FEATURES
        )
        return _sigmoid(logit)

    def score_candidate(self, candidate: TrajectoryCandidateV64) -> float:
        return self.score_features(candidate.feature_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "features": list(VERIFIER_FEATURES),
            "weights": {name: float(self.weights[name]) for name in VERIFIER_FEATURES},
            "intercept": float(self.intercept),
            "fit_question_id_sha256": self.fit_question_id_sha256,
            "validation_question_id_sha256": self.validation_question_id_sha256,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnswerVerifierArtifactV64":
        if payload.get("schema") != VERIFIER_SCHEMA:
            raise ValueError("not an ACEC v6.4 answer-verifier artifact")
        if int(payload.get("version", -1)) != VERIFIER_VERSION:
            raise ValueError("unsupported answer-verifier artifact version")
        if tuple(payload.get("features") or ()) != VERIFIER_FEATURES:
            raise ValueError("answer-verifier feature schema differs from runtime")
        return cls(
            weights={
                name: float(payload["weights"][name]) for name in VERIFIER_FEATURES
            },
            intercept=float(payload["intercept"]),
            fit_question_id_sha256=str(payload["fit_question_id_sha256"]),
            validation_question_id_sha256=str(
                payload["validation_question_id_sha256"]
            ),
            metrics=dict(payload.get("metrics") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )


def _id_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(ids))).encode("utf-8")).hexdigest()


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _ece(labels: Sequence[bool], scores: Sequence[float], bins: int = 10) -> float:
    total = len(labels)
    if total == 0:
        return float("nan")
    error = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        selected = [
            index
            for index, score in enumerate(scores)
            if low <= score < high or (bin_index == bins - 1 and score == 1.0)
        ]
        if not selected:
            continue
        confidence = sum(scores[index] for index in selected) / len(selected)
        accuracy = sum(labels[index] for index in selected) / len(selected)
        error += len(selected) / total * abs(confidence - accuracy)
    return error


def verifier_quality_metrics(
    examples: Sequence[VerifierExampleV64], scores: Sequence[float]
) -> Dict[str, Any]:
    if len(examples) != len(scores) or not examples:
        raise ValueError("verifier metrics require aligned non-empty rows")
    labels = [bool(example.correct) for example in examples]
    brier = sum(
        (float(score) - float(label)) ** 2 for score, label in zip(scores, labels)
    ) / len(labels)
    by_question: Dict[str, List[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_question[example.question_id].append(index)
    selectable = 0
    recovered = 0
    for indices in by_question.values():
        if not any(labels[index] for index in indices):
            continue
        selectable += 1
        selected = max(indices, key=lambda index: (scores[index], -examples[index].sample_index))
        recovered += int(labels[selected])
    return {
        "rows": len(examples),
        "questions": len(by_question),
        "positive_rate": sum(labels) / len(labels),
        "auc": _auc(labels, scores),
        "brier": brier,
        "ece_10": _ece(labels, scores),
        "oracle_recoverable_questions": selectable,
        "top1_oracle_recovery": recovered / selectable if selectable else float("nan"),
    }


def fit_answer_verifier_v64(
    fit_examples: Sequence[VerifierExampleV64],
    validation_examples: Sequence[VerifierExampleV64],
    *,
    learning_rate: float = 0.1,
    steps: int = 2000,
    l2: float = 0.01,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AnswerVerifierArtifactV64:
    if not fit_examples or not validation_examples:
        raise ValueError("verifier fit and validation examples must be non-empty")
    fit_ids = {example.question_id for example in fit_examples}
    validation_ids = {example.question_id for example in validation_examples}
    overlap = fit_ids & validation_ids
    if overlap:
        raise ValueError(f"verifier fit/validation ids overlap: {sorted(overlap)[:5]}")
    if steps <= 0 or learning_rate <= 0.0 or l2 < 0.0:
        raise ValueError("invalid verifier optimizer configuration")

    prevalence = (
        sum(bool(example.correct) for example in fit_examples) + 0.5
    ) / (len(fit_examples) + 1.0)
    intercept = math.log(prevalence / (1.0 - prevalence))
    weights = {name: 0.0 for name in VERIFIER_FEATURES}
    count = len(fit_examples)
    for _ in range(steps):
        gradient_intercept = 0.0
        gradients = {name: 0.0 for name in VERIFIER_FEATURES}
        for example in fit_examples:
            logit = intercept + sum(
                weights[name] * float(example.features[name])
                for name in VERIFIER_FEATURES
            )
            error = _sigmoid(logit) - float(example.correct)
            gradient_intercept += error
            for name in VERIFIER_FEATURES:
                gradients[name] += error * float(example.features[name])
        intercept -= learning_rate * gradient_intercept / count
        for name in VERIFIER_FEATURES:
            gradient = gradients[name] / count + l2 * weights[name]
            weights[name] -= learning_rate * gradient

    provisional = AnswerVerifierArtifactV64(
        weights=weights,
        intercept=intercept,
        fit_question_id_sha256=_id_hash(fit_ids),
        validation_question_id_sha256=_id_hash(validation_ids),
        metrics={},
        metadata=dict(metadata or {}),
    )
    fit_scores = [
        provisional.score_features(example.features) for example in fit_examples
    ]
    validation_scores = [
        provisional.score_features(example.features)
        for example in validation_examples
    ]
    fit_metrics = verifier_quality_metrics(fit_examples, fit_scores)
    validation_metrics = verifier_quality_metrics(
        validation_examples, validation_scores
    )
    validation_auc = float(validation_metrics["auc"])
    recovery = float(validation_metrics["top1_oracle_recovery"])
    gate_pass = bool(
        math.isfinite(validation_auc)
        and validation_auc >= 0.55
        and math.isfinite(recovery)
        and recovery >= 0.50
    )
    return AnswerVerifierArtifactV64(
        weights=weights,
        intercept=intercept,
        fit_question_id_sha256=provisional.fit_question_id_sha256,
        validation_question_id_sha256=provisional.validation_question_id_sha256,
        metrics={
            "fit": fit_metrics,
            "validation": validation_metrics,
            "gate": {
                "pass": gate_pass,
                "minimum_auc": 0.55,
                "minimum_top1_oracle_recovery": 0.50,
            },
        },
        metadata=dict(metadata or {}),
    )


def save_answer_verifier_artifact_v64(
    path: str | Path, artifact: AnswerVerifierArtifactV64
) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verifier artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_answer_verifier_artifact_v64(
    path: str | Path, *, require_gate: bool = True
) -> AnswerVerifierArtifactV64:
    with Path(path).open("r", encoding="utf-8") as handle:
        artifact = AnswerVerifierArtifactV64.from_dict(json.load(handle))
    if require_gate and (artifact.metrics.get("gate") or {}).get("pass") is not True:
        raise ValueError("runtime refuses an answer-verifier artifact that failed its gate")
    return artifact


def select_candidate_v64(
    candidates: Sequence[TrajectoryCandidateV64],
    *,
    mode: str,
    artifact: Optional[AnswerVerifierArtifactV64] = None,
    processed: bool = False,
) -> TrajectoryCandidateV64:
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    ordered = sorted(candidates, key=lambda candidate: candidate.sample_index)
    if mode == "first":
        return ordered[0]
    if mode == "majority":
        valid = [
            candidate
            for candidate in ordered
            if candidate.answered
            and candidate.format_valid
            and normalize_answer(candidate.scoring_answer(processed))
        ]
        if valid:
            ordered = valid
        groups: Dict[str, List[TrajectoryCandidateV64]] = defaultdict(list)
        for candidate in ordered:
            groups[normalize_answer(candidate.selection_answer(processed))].append(candidate)
        winning_key = min(
            groups,
            key=lambda key: (
                -len(groups[key]),
                min(candidate.sample_index for candidate in groups[key]),
            ),
        )
        return min(groups[winning_key], key=lambda candidate: candidate.sample_index)
    if mode == "nli":
        return max(
            ordered,
            key=lambda candidate: (
                int(candidate.answered and candidate.format_valid),
                candidate.grounding_score,
                -candidate.sample_index,
            ),
        )
    if mode == "acec_zero_shot":
        return max(
            ordered,
            key=lambda candidate: (
                int(candidate.answered and candidate.format_valid),
                candidate.zero_shot_acec_score,
                -candidate.sample_index,
            ),
        )
    if mode == "acec_calibrated":
        if artifact is None:
            raise ValueError("acec_calibrated selection requires a verifier artifact")
        return max(
            ordered,
            key=lambda candidate: (
                int(candidate.answered and candidate.format_valid),
                artifact.score_candidate(candidate),
                -candidate.sample_index,
            ),
        )
    raise ValueError(f"unsupported answer verifier mode {mode!r}")


@dataclass(frozen=True)
class BestOfNSelectionV64:
    question_id: str
    k: int
    mode: str
    selected_sample_index: int
    selected_answer: str
    em: float
    f1: float
    oracle_em: float
    oracle_f1: float
    unique_answer_count: int
    total_retrieval_calls: int
    total_generation_calls: int
    total_grounding_pairs: int
    # 被选中那条轨迹的原始自由文本；严格 EM 仍由 selected_answer 计算，
    # 这一项只供 processed-EM 的外部抽取器使用。
    selected_raw_answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_best_of_n_v64(
    candidates: Sequence[TrajectoryCandidateV64],
    gold_answers: Mapping[str, Sequence[str]],
    *,
    k_values: Sequence[int],
    modes: Sequence[str],
    artifact: Optional[AnswerVerifierArtifactV64] = None,
    processed: bool = False,
) -> Dict[str, Any]:
    by_question: Dict[str, List[TrajectoryCandidateV64]] = defaultdict(list)
    for candidate in candidates:
        by_question[candidate.question_id].append(candidate)
    if set(by_question) != set(gold_answers):
        raise ValueError("candidate and gold question-id sets must match exactly")
    for qid, question_candidates in by_question.items():
        sample_indices = [
            candidate.sample_index for candidate in question_candidates
        ]
        if len(sample_indices) != len(set(sample_indices)):
            raise ValueError(f"duplicate sample indices for question {qid}")
    selections: Dict[str, List[BestOfNSelectionV64]] = {}
    summaries: Dict[str, Dict[str, float]] = {}
    for k in sorted(set(int(value) for value in k_values)):
        if k <= 0:
            raise ValueError("best-of-N K values must be positive")
        for mode in modes:
            key = f"{mode}@{k}"
            rows = []
            for qid in sorted(by_question):
                available = sorted(
                    by_question[qid], key=lambda candidate: candidate.sample_index
                )
                if len(available) < k:
                    raise ValueError(
                        f"question {qid} has {len(available)} candidates, needs K={k}"
                    )
                subset = available[:k]
                selected = select_candidate_v64(
                    subset, mode=mode, artifact=artifact, processed=processed
                )
                golds = list(gold_answers[qid])
                answer = selected.scoring_answer(processed)
                candidate_ems = [
                    answer_exact_match(candidate.scoring_answer(processed), golds)
                    for candidate in subset
                ]
                candidate_f1s = [
                    answer_f1(candidate.scoring_answer(processed), golds)
                    for candidate in subset
                ]
                rows.append(
                    BestOfNSelectionV64(
                        question_id=qid,
                        k=k,
                        mode=mode,
                        selected_sample_index=selected.sample_index,
                        selected_answer=answer,
                        selected_raw_answer=str(selected.raw_answer or ""),
                        em=answer_exact_match(answer, golds),
                        f1=answer_f1(answer, golds),
                        oracle_em=max(candidate_ems),
                        oracle_f1=max(candidate_f1s),
                        # 用 selection_answer：契约不合规时 scoring_answer 全为空串，
                        # 该计数会恒等于 1，掩盖真实的候选多样性。
                        unique_answer_count=len(
                            {
                                normalize_answer(candidate.selection_answer(processed))
                                for candidate in subset
                            }
                        ),
                        total_retrieval_calls=sum(
                            candidate.retrieval_calls for candidate in subset
                        ),
                        total_generation_calls=sum(
                            candidate.generation_calls for candidate in subset
                        ),
                        total_grounding_pairs=sum(
                            candidate.grounding_pairs for candidate in subset
                        )
                        if mode
                        in {"nli", "acec_zero_shot", "acec_calibrated"}
                        else 0,
                    )
                )
            selections[key] = rows
            summaries[key] = {
                "questions": len(rows),
                "em": sum(row.em for row in rows) / len(rows),
                "f1": sum(row.f1 for row in rows) / len(rows),
                "oracle_em": sum(row.oracle_em for row in rows) / len(rows),
                "oracle_f1": sum(row.oracle_f1 for row in rows) / len(rows),
                "selection_regret_em": sum(
                    row.oracle_em - row.em for row in rows
                )
                / len(rows),
                "mean_unique_answers": sum(
                    row.unique_answer_count for row in rows
                )
                / len(rows),
                "mean_total_retrieval_calls": sum(
                    row.total_retrieval_calls for row in rows
                )
                / len(rows),
                "mean_total_generation_calls": sum(
                    row.total_generation_calls for row in rows
                )
                / len(rows),
                "mean_total_grounding_pairs": sum(
                    row.total_grounding_pairs for row in rows
                )
                / len(rows),
                "mean_trajectories": float(k),
            }
    return {
        "summaries": summaries,
        "selections": {
            key: [row.to_dict() for row in rows] for key, rows in selections.items()
        },
    }


def paired_bootstrap_delta_v64(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    iterations: int = 2000,
    seed: int = 20260723,
) -> Dict[str, float]:
    if set(left) != set(right) or not left:
        raise ValueError("paired bootstrap mappings must share non-empty ids")
    if iterations <= 0:
        raise ValueError("paired bootstrap iterations must be positive")
    qids = sorted(left)
    deltas = [float(left[qid]) - float(right[qid]) for qid in qids]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    samples = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(iterations)
    )
    return {
        "delta": observed,
        "ci95_low": samples[int(0.025 * (len(samples) - 1))],
        "ci95_high": samples[int(0.975 * (len(samples) - 1))],
    }


def evaluate_policy_controller_factorial_v64(
    arms: Mapping[str, Sequence[TrajectoryCandidateV64]],
    gold_answers: Mapping[str, Sequence[str]],
    *,
    k_values: Sequence[int],
    modes: Sequence[str],
    artifact: Optional[AnswerVerifierArtifactV64] = None,
    processed: bool = False,
) -> Dict[str, Any]:
    if not arms:
        raise ValueError("factorial evaluation requires at least one arm")
    return {
        arm_name: evaluate_best_of_n_v64(
            candidates,
            gold_answers,
            k_values=k_values,
            modes=modes,
            artifact=artifact,
            processed=processed,
        )
        for arm_name, candidates in arms.items()
    }


__all__ = [
    "AnswerVerifierArtifactV64",
    "BestOfNSelectionV64",
    "TrajectoryCandidateV64",
    "VERIFIER_FEATURES",
    "VERIFIER_SCHEMA",
    "VERIFIER_VERSION",
    "VerifierExampleV64",
    "answer_exact_match",
    "answer_f1",
    "evaluate_best_of_n_v64",
    "evaluate_policy_controller_factorial_v64",
    "fit_answer_verifier_v64",
    "load_answer_verifier_artifact_v64",
    "normalize_answer",
    "paired_bootstrap_delta_v64",
    "save_answer_verifier_artifact_v64",
    "select_candidate_v64",
    "verifier_quality_metrics",
]
