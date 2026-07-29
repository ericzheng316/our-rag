"""Offline listwise training for the ACEC v7 selector artifact."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .contracts_v7 import assert_sf_label_free, stable_id_hash
from .set_selector_v7 import (
    SELECTOR_FEATURE_NAMES_V7,
    SelectorArtifactV7,
)


@dataclass(frozen=True)
class SelectorTrainingExampleV7:
    question_id: str
    state_id: str
    candidate_ids: Tuple[str, ...]
    features: Tuple[Mapping[str, float], ...]
    target_utilities: Tuple[float, ...]
    retriever_logits: Tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.candidate_ids)
        if not self.question_id or not self.state_id or size < 2:
            raise ValueError("selector training example requires ids and >=2 candidates")
        if len(set(self.candidate_ids)) != size:
            raise ValueError("selector training candidate ids must be unique")
        if not (
            len(self.features)
            == len(self.target_utilities)
            == len(self.retriever_logits)
            == size
        ):
            raise ValueError("selector training rows do not align")
        expected = set(SELECTOR_FEATURE_NAMES_V7)
        for row in self.features:
            if set(row) != expected:
                raise ValueError("selector training feature contract mismatch")
            assert_sf_label_free(row, path="selector_training.features")
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise ValueError("selector training features must be finite")
        if not all(
            math.isfinite(float(value))
            for value in (*self.target_utilities, *self.retriever_logits)
        ):
            raise ValueError("selector targets/logits must be finite")


@dataclass(frozen=True)
class SelectorFitConfigV7:
    hidden_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 100
    target_temperature: float = 0.7
    selector_temperature: float = 0.7
    kl_coefficient: float = 0.02
    entropy_coefficient: float = 0.01
    pairwise_coefficient: float = 0.5
    l2: float = 1e-5
    seed: int = 0
    disabled_features: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.epochs <= 0:
            raise ValueError("hidden size/epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if self.target_temperature <= 0.0 or self.selector_temperature <= 0.0:
            raise ValueError("selector temperatures must be positive")
        for name in (
            "kl_coefficient",
            "entropy_coefficient",
            "pairwise_coefficient",
            "l2",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        unknown_disabled = set(self.disabled_features) - set(
            SELECTOR_FEATURE_NAMES_V7
        )
        if unknown_disabled:
            raise ValueError(
                f"unknown disabled selector features: {sorted(unknown_disabled)}"
            )


@dataclass(frozen=True)
class SelectorSlateStepV7:
    candidate_ids: Tuple[str, ...]
    features: Tuple[Mapping[str, float], ...]
    retriever_logits: Tuple[float, ...]
    selected_id: str

    def __post_init__(self) -> None:
        size = len(self.candidate_ids)
        if size == 0 or len(set(self.candidate_ids)) != size:
            raise ValueError("slate step candidate ids must be non-empty/unique")
        if len(self.features) != size or len(self.retriever_logits) != size:
            raise ValueError("slate step rows do not align")
        if self.selected_id not in self.candidate_ids:
            raise ValueError("slate selected id is outside the step candidates")
        expected = set(SELECTOR_FEATURE_NAMES_V7)
        for row in self.features:
            if set(row) != expected:
                raise ValueError("slate step feature contract mismatch")
            assert_sf_label_free(row, path="selector_slate.features")


@dataclass(frozen=True)
class SelectorSlateExampleV7:
    question_id: str
    state_id: str
    slate_id: str
    steps: Tuple[SelectorSlateStepV7, ...]
    answer_utility: float

    def __post_init__(self) -> None:
        if not self.question_id or not self.state_id or not self.slate_id:
            raise ValueError("selector slate requires question/state/slate ids")
        if not self.steps:
            raise ValueError("selector slate requires at least one step")
        if not math.isfinite(float(self.answer_utility)):
            raise ValueError("selector slate utility must be finite")


def _feature_matrix(
    examples: Sequence[SelectorTrainingExampleV7],
) -> np.ndarray:
    return np.asarray(
        [
            [float(row[name]) for name in SELECTOR_FEATURE_NAMES_V7]
            for example in examples
            for row in example.features
        ],
        dtype=np.float64,
    )


def _disable_features(
    matrix: np.ndarray, disabled_features: Sequence[str]
) -> np.ndarray:
    if not disabled_features:
        return matrix
    result = np.asarray(matrix).copy()
    for name in disabled_features:
        result[..., SELECTOR_FEATURE_NAMES_V7.index(name)] = 0.0
    return result


def _evaluate_numpy(
    examples: Sequence[SelectorTrainingExampleV7],
    artifact: SelectorArtifactV7,
) -> Dict[str, float]:
    correct = 0
    total = 0
    utility_gain = []
    entropy = []
    for example in examples:
        scores = np.asarray(
            [artifact.score(row) for row in example.features], dtype=np.float64
        )
        selected = int(np.argmax(scores))
        baseline = int(np.argmax(np.asarray(example.retriever_logits)))
        utility_gain.append(
            float(example.target_utilities[selected] - example.target_utilities[baseline])
        )
        for left in range(len(scores)):
            for right in range(left + 1, len(scores)):
                gold = example.target_utilities[left] - example.target_utilities[right]
                if gold == 0.0:
                    continue
                total += 1
                correct += int((scores[left] - scores[right]) * gold > 0.0)
        probability = np.exp((scores - scores.max()) / artifact.temperature)
        probability /= probability.sum()
        entropy.append(
            -float(np.sum(probability * np.log(np.maximum(probability, 1e-12))))
        )
    return {
        "pairwise_accuracy": correct / max(total, 1),
        "mean_utility_gain_vs_retriever": float(np.mean(utility_gain)),
        "selector_entropy": float(np.mean(entropy)),
        "examples": float(len(examples)),
    }


def fit_selector_v7(
    fit_examples: Sequence[SelectorTrainingExampleV7],
    validation_examples: Sequence[SelectorTrainingExampleV7],
    *,
    config: SelectorFitConfigV7 | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SelectorArtifactV7:
    """Fit the small selector MLP with listwise and pairwise objectives.

    PyTorch is imported lazily so runtime selector use and CPU contract tests
    retain the package's NumPy-only dependency.
    """

    if not fit_examples or not validation_examples:
        raise ValueError("selector fit/validation examples must be non-empty")
    fit_ids = {example.question_id for example in fit_examples}
    validation_ids = {example.question_id for example in validation_examples}
    overlap = fit_ids & validation_ids
    if overlap:
        raise ValueError(f"selector question split leaks {len(overlap)} ids")
    cfg = config or SelectorFitConfigV7()
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("selector fitting requires PyTorch") from exc

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    raw_fit = _disable_features(
        _feature_matrix(fit_examples), cfg.disabled_features
    )
    center = np.median(raw_fit, axis=0)
    q75 = np.percentile(raw_fit, 75, axis=0)
    q25 = np.percentile(raw_fit, 25, axis=0)
    scale = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)

    model = torch.nn.Sequential(
        torch.nn.Linear(len(SELECTOR_FEATURE_NAMES_V7), cfg.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg.hidden_size, 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2
    )

    generator = np.random.default_rng(cfg.seed)
    for _ in range(cfg.epochs):
        order = generator.permutation(len(fit_examples))
        for example_index in order:
            example = fit_examples[int(example_index)]
            raw = np.asarray(
                [
                    [float(row[name]) for name in SELECTOR_FEATURE_NAMES_V7]
                    for row in example.features
                ],
                dtype=np.float32,
            )
            raw = _disable_features(raw, cfg.disabled_features)
            features = torch.tensor(
                (raw - center) / scale, dtype=torch.float32
            )
            target_utility = torch.tensor(
                example.target_utilities, dtype=torch.float32
            )
            retriever_logits = torch.tensor(
                example.retriever_logits, dtype=torch.float32
            )
            selector_logits = model(features).squeeze(-1)
            target_probability = functional.softmax(
                target_utility / cfg.target_temperature, dim=0
            ).detach()
            selector_log_probability = functional.log_softmax(
                selector_logits / cfg.selector_temperature, dim=0
            )
            selector_probability = selector_log_probability.exp()
            listwise = -(target_probability * selector_log_probability).sum()

            retriever_probability = functional.softmax(
                retriever_logits / cfg.selector_temperature, dim=0
            ).detach()
            kl = (
                selector_probability
                * (
                    selector_log_probability
                    - torch.log(torch.clamp(retriever_probability, min=1e-12))
                )
            ).sum()
            entropy = -(selector_probability * selector_log_probability).sum()

            pair_terms = []
            for left in range(len(example.candidate_ids)):
                for right in range(left + 1, len(example.candidate_ids)):
                    sign = torch.sign(target_utility[left] - target_utility[right])
                    if float(sign) == 0.0:
                        continue
                    pair_terms.append(
                        functional.softplus(
                            -sign * (selector_logits[left] - selector_logits[right])
                        )
                    )
            pairwise = (
                torch.stack(pair_terms).mean()
                if pair_terms
                else selector_logits.sum() * 0.0
            )
            loss = (
                listwise
                + cfg.kl_coefficient * kl
                - cfg.entropy_coefficient * entropy
                + cfg.pairwise_coefficient * pairwise
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    first = model[0]
    output = model[2]
    provisional = SelectorArtifactV7(
        feature_names=SELECTOR_FEATURE_NAMES_V7,
        center=tuple(float(v) for v in center),
        scale=tuple(float(v) for v in scale),
        hidden_weights=tuple(
            tuple(float(v) for v in row)
            for row in first.weight.detach().cpu().numpy()
        ),
        hidden_bias=tuple(float(v) for v in first.bias.detach().cpu().numpy()),
        output_weights=tuple(
            float(v) for v in output.weight.detach().cpu().numpy()[0]
        ),
        output_bias=float(output.bias.detach().cpu().numpy()[0]),
        temperature=cfg.selector_temperature,
        fit_question_id_sha256=stable_id_hash(tuple(fit_ids)),
        validation_question_id_sha256=stable_id_hash(tuple(validation_ids)),
        metrics={},
        metadata={
            **dict(metadata or {}),
            "disabled_features": list(cfg.disabled_features),
        },
    )
    metrics = {
        **{
            f"fit_{key}": value
            for key, value in _evaluate_numpy(fit_examples, provisional).items()
        },
        **{
            f"validation_{key}": value
            for key, value in _evaluate_numpy(validation_examples, provisional).items()
        },
    }
    return SelectorArtifactV7(
        **{
            **provisional.__dict__,
            "metrics": metrics,
            "metadata": {
                **dict(metadata or {}),
                "evidence_membership_supervision_used": False,
                "fit_config": cfg.__dict__,
                "disabled_features": list(cfg.disabled_features),
            },
        }
    )


def fit_selector_from_slates_v7(
    fit_slates: Sequence[SelectorSlateExampleV7],
    validation_slates: Sequence[SelectorSlateExampleV7],
    *,
    config: SelectorFitConfigV7 | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SelectorArtifactV7:
    """Policy-gradient fit from counterfactual slates with within-state baselines."""

    if not fit_slates or not validation_slates:
        raise ValueError("selector slate fit/validation inputs must be non-empty")
    fit_ids = {example.question_id for example in fit_slates}
    validation_ids = {example.question_id for example in validation_slates}
    overlap = fit_ids & validation_ids
    if overlap:
        raise ValueError(f"selector slate split leaks {len(overlap)} question ids")
    cfg = config or SelectorFitConfigV7()
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("selector fitting requires PyTorch") from exc

    all_fit_rows = np.asarray(
        [
            [float(row[name]) for name in SELECTOR_FEATURE_NAMES_V7]
            for slate in fit_slates
            for step in slate.steps
            for row in step.features
        ],
        dtype=np.float64,
    )
    all_fit_rows = _disable_features(
        all_fit_rows, cfg.disabled_features
    )
    center = np.median(all_fit_rows, axis=0)
    q75 = np.percentile(all_fit_rows, 75, axis=0)
    q25 = np.percentile(all_fit_rows, 25, axis=0)
    scale = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)

    torch.manual_seed(cfg.seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(len(SELECTOR_FEATURE_NAMES_V7), cfg.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg.hidden_size, 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.l2
    )
    by_state: Dict[str, List[SelectorSlateExampleV7]] = {}
    for slate in fit_slates:
        by_state.setdefault(slate.state_id, []).append(slate)
    generator = np.random.default_rng(cfg.seed)

    for _ in range(cfg.epochs):
        state_ids = list(by_state)
        generator.shuffle(state_ids)
        for state_id in state_ids:
            slates = by_state[state_id]
            baseline = float(np.mean([slate.answer_utility for slate in slates]))
            slate_log_probabilities = []
            slate_utilities = []
            regularizers = []
            for slate in slates:
                log_probability = None
                step_regularizer = None
                for step in slate.steps:
                    raw = np.asarray(
                        [
                            [float(row[name]) for name in SELECTOR_FEATURE_NAMES_V7]
                            for row in step.features
                        ],
                        dtype=np.float32,
                    )
                    raw = _disable_features(raw, cfg.disabled_features)
                    features = torch.tensor(
                        (raw - center) / scale, dtype=torch.float32
                    )
                    logits = model(features).squeeze(-1)
                    log_probs = functional.log_softmax(
                        logits / cfg.selector_temperature, dim=0
                    )
                    probs = log_probs.exp()
                    chosen = step.candidate_ids.index(step.selected_id)
                    selected_log_probability = log_probs[chosen]
                    log_probability = (
                        selected_log_probability
                        if log_probability is None
                        else log_probability + selected_log_probability
                    )
                    retriever = torch.tensor(
                        step.retriever_logits, dtype=torch.float32
                    )
                    retriever_prob = functional.softmax(
                        retriever / cfg.selector_temperature, dim=0
                    ).detach()
                    kl = (
                        probs
                        * (
                            log_probs
                            - torch.log(torch.clamp(retriever_prob, min=1e-12))
                        )
                    ).sum()
                    entropy = -(probs * log_probs).sum()
                    current = cfg.kl_coefficient * kl - cfg.entropy_coefficient * entropy
                    step_regularizer = (
                        current
                        if step_regularizer is None
                        else step_regularizer + current
                    )
                assert log_probability is not None and step_regularizer is not None
                slate_log_probabilities.append(log_probability)
                slate_utilities.append(float(slate.answer_utility))
                regularizers.append(step_regularizer / len(slate.steps))

            loss = None
            for log_probability, utility, regularizer in zip(
                slate_log_probabilities, slate_utilities, regularizers
            ):
                term = -(utility - baseline) * log_probability + regularizer
                loss = term if loss is None else loss + term
            assert loss is not None
            loss = loss / len(slates)
            pair_terms = []
            for left in range(len(slates)):
                for right in range(left + 1, len(slates)):
                    sign = math.copysign(
                        1.0, slate_utilities[left] - slate_utilities[right]
                    ) if slate_utilities[left] != slate_utilities[right] else 0.0
                    if sign == 0.0:
                        continue
                    pair_terms.append(
                        functional.softplus(
                            -sign
                            * (
                                slate_log_probabilities[left]
                                - slate_log_probabilities[right]
                            )
                        )
                    )
            if pair_terms:
                loss = loss + cfg.pairwise_coefficient * torch.stack(pair_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    first = model[0]
    output = model[2]
    provisional = SelectorArtifactV7(
        feature_names=SELECTOR_FEATURE_NAMES_V7,
        center=tuple(float(v) for v in center),
        scale=tuple(float(v) for v in scale),
        hidden_weights=tuple(
            tuple(float(v) for v in row)
            for row in first.weight.detach().cpu().numpy()
        ),
        hidden_bias=tuple(float(v) for v in first.bias.detach().cpu().numpy()),
        output_weights=tuple(
            float(v) for v in output.weight.detach().cpu().numpy()[0]
        ),
        output_bias=float(output.bias.detach().cpu().numpy()[0]),
        temperature=cfg.selector_temperature,
        fit_question_id_sha256=stable_id_hash(tuple(fit_ids)),
        validation_question_id_sha256=stable_id_hash(tuple(validation_ids)),
        metrics={},
        metadata={
            **dict(metadata or {}),
            "disabled_features": list(cfg.disabled_features),
        },
    )

    def slate_metrics(slates: Sequence[SelectorSlateExampleV7]) -> Dict[str, float]:
        groups: Dict[str, List[Tuple[float, float, str]]] = {}
        for slate in slates:
            log_probability = 0.0
            for step in slate.steps:
                scores = np.asarray(
                    [provisional.score(row) for row in step.features],
                    dtype=np.float64,
                )
                probability = np.exp(
                    (scores - scores.max()) / provisional.temperature
                )
                probability /= probability.sum()
                chosen = step.candidate_ids.index(step.selected_id)
                log_probability += math.log(max(float(probability[chosen]), 1e-12))
            groups.setdefault(slate.state_id, []).append(
                (log_probability, slate.answer_utility, slate.slate_id)
            )
        wins = 0
        utility_delta = []
        ties = 0
        for values in groups.values():
            chosen = max(values, key=lambda row: row[0])
            relevance = next(
                (row for row in values if row[2] == "relevance"),
                values[0],
            )
            delta = float(chosen[1] - relevance[1])
            utility_delta.append(delta)
            wins += int(delta > 0.0)
            ties += int(delta == 0.0)
        return {
            "state_win_rate_vs_relevance": wins / max(len(groups), 1),
            "state_tie_rate_vs_relevance": ties / max(len(groups), 1),
            "mean_utility_gain_vs_relevance": float(np.mean(utility_delta)),
            "states": float(len(groups)),
        }

    metrics = {
        **{f"fit_{key}": value for key, value in slate_metrics(fit_slates).items()},
        **{
            f"validation_{key}": value
            for key, value in slate_metrics(validation_slates).items()
        },
    }
    return SelectorArtifactV7(
        **{
            **provisional.__dict__,
            "metrics": metrics,
            "metadata": {
                **dict(metadata or {}),
                "evidence_membership_supervision_used": False,
                "fit_config": cfg.__dict__,
                "disabled_features": list(cfg.disabled_features),
                "training_objective": "within_state_slate_policy_gradient",
            },
        }
    )


__all__ = [
    "SelectorFitConfigV7",
    "SelectorSlateExampleV7",
    "SelectorSlateStepV7",
    "SelectorTrainingExampleV7",
    "fit_selector_from_slates_v7",
    "fit_selector_v7",
]
