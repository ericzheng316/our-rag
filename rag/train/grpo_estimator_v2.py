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
) -> torch.Tensor:
    """Raw-model log-probabilities of the sampled completion tokens.

    ``logits`` has shape ``(1, prompt+completion, vocab)`` and follows the
    causal-LM convention where position ``i`` predicts token ``i+1``.

    Sampling temperature is deliberately absent.  vLLM's default
    ``logprobs_mode=raw_logprobs`` reports probabilities before temperature
    and top-k/top-p processors, and the GRPO policy/reference distributions
    use those same raw model probabilities.  Temperature controls rollout
    exploration only.
    """

    n_out = int(output_ids.numel())
    if n_out == 0:
        return logits.new_empty((0,), dtype=torch.float32)
    # Slice completion positions before the fp32 normalization.  vLLM
    # normalizes raw logits in fp32; matching that precision materially
    # improves engine/HF alignment without promoting prompt positions or the
    # entire model output tensor.
    predicted_logits = logits[0, -(n_out + 1) : -1]
    predicted_logits_fp32 = predicted_logits.float()
    token_logits = predicted_logits_fp32.gather(1, output_ids.unsqueeze(1)).squeeze(1)
    log_normalizer = torch.logsumexp(predicted_logits_fp32, dim=-1)
    return token_logits - log_normalizer


def engine_hf_logprob_mae(
    policy_logprobs: torch.Tensor,
    engine_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute raw-logprob difference used only as an alignment gate."""

    if policy_logprobs.shape != engine_logprobs.shape:
        raise ValueError(
            "policy/engine logprob shape mismatch: "
            f"{policy_logprobs.shape} vs {engine_logprobs.shape}"
        )
    if policy_logprobs.numel() == 0:
        raise ValueError("policy/engine logprobs must contain at least one token")
    if not torch.isfinite(policy_logprobs).all():
        raise ValueError("policy logprobs contain NaN or Inf")
    if not torch.isfinite(engine_logprobs).all():
        raise ValueError("engine logprobs contain NaN or Inf")
    return (policy_logprobs.detach() - engine_logprobs.detach()).abs().mean()


def snapshot_old_policy_logprobs(policy_logprobs: torch.Tensor) -> torch.Tensor:
    """Snapshot pi_old for the single optimizer update on a rollout batch."""

    return policy_logprobs.detach()


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
