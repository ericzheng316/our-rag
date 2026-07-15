"""Shared GRPO-v2 math and reward configuration.

This module is intentionally small and side-effect free so the estimator can
be unit-tested without loading a model, a retriever, or vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class RewardConfig:
    """One reward scale shared by gold-SF and ACEC coverage modes."""

    answer: float = 1.0
    coverage: float = 0.3
    format_error: float = 0.1
    retrieval_cost: float = 0.05

    def answer_reward(self, correct: bool) -> float:
        return self.answer if correct else 0.0

    def retrieval_reward(self, delta_coverage: float) -> float:
        return self.coverage * delta_coverage - self.retrieval_cost


def completion_token_logprobs(
    logits: torch.Tensor,
    output_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Log-probabilities of completion tokens under the tempered policy.

    ``logits`` has shape ``(1, prompt+completion, vocab)`` and follows the
    causal-LM convention where position ``i`` predicts token ``i+1``.
    """

    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    n_out = int(output_ids.numel())
    if n_out == 0:
        return logits.new_empty((0,), dtype=torch.float32)
    # Keep the full (sequence, vocab) tensor in the model dtype; promoting it
    # to fp32 here roughly doubles the dominant activation and can OOM the
    # real G=8/r=64 configuration.  Only the gathered token log-probs are
    # promoted below.
    scaled = logits[0] / temperature
    log_probs = torch.log_softmax(scaled, dim=-1)
    predicted = log_probs[-(n_out + 1) : -1]
    return predicted.gather(1, output_ids.unsqueeze(1)).squeeze(1).float()


def clipped_grpo_token_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Token-level clipped GRPO objective with DeepSeekMath's KL estimator.

    The KL estimator is ``exp(log pi_ref - log pi) - (log pi_ref - log pi)
    - 1``.  It is non-negative pointwise and has the correct non-zero
    gradient when the policy has moved away from the reference model.
    """

    if policy_logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"policy/old logprob shape mismatch: {policy_logprobs.shape} vs {old_logprobs.shape}"
        )
    if policy_logprobs.shape != reference_logprobs.shape:
        raise ValueError(
            "policy/reference logprob shape mismatch: "
            f"{policy_logprobs.shape} vs {reference_logprobs.shape}"
        )
    if not 0.0 < clip_eps < 1.0:
        raise ValueError(f"clip_eps must be in (0, 1), got {clip_eps}")

    log_ratio = policy_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    policy_loss = -surrogate.mean()

    log_ref_over_policy = reference_logprobs - policy_logprobs
    kl_per_token = torch.exp(log_ref_over_policy) - log_ref_over_policy - 1.0
    kl = kl_per_token.mean()
    loss = policy_loss + kl_coef * kl

    stats = {
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "ratio_mean": ratio.detach().mean(),
        "clip_fraction": (ratio.detach() != clipped_ratio.detach()).float().mean(),
        "old_logprob_mae": (policy_logprobs.detach() - old_logprobs.detach()).abs().mean(),
    }
    return loss, stats
