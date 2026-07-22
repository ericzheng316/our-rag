"""ACEC-v6 estimator extensions: adaptive KL and answer-turn anchoring."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from acec_v6_contract import AdaptiveKLController, effective_kl_coefficient
from grpo_estimator_v2 import (
    clipped_grpo_token_loss,
    completion_token_logprobs,
    engine_hf_logprob_mae,
    snapshot_old_policy_logprobs,
)
from grpo_rsf_simple import turn_level_advantages, turn_level_returns


def unpack_turn_v6(turn: Tuple[Any, ...]):
    if len(turn) == 6:
        inp, out, reward, engine_lp, temperature, kind = turn
        return inp, out, float(reward), engine_lp, float(temperature), str(kind)
    if len(turn) == 5:
        inp, out, reward, engine_lp, temperature = turn
        return inp, out, float(reward), engine_lp, float(temperature), "unknown"
    raise ValueError(f"unexpected ACEC-v6 turn tuple length: {len(turn)}")


def grpo_loss_v6(
    model,
    trajs_per_question: List[List[List]],
    controller: AdaptiveKLController,
    answer_kl_multiplier: float = 2.0,
    kl_coef: Optional[float] = None,
    clip_eps: float = 0.2,
    max_engine_logprob_mae: Optional[float] = None,
    return_stats: bool = False,
):
    # ``base_trainer.train`` always forwards its legacy kl_coef argument.
    # The controller is authoritative in v6; reject a mismatched value rather
    # than silently running a different experiment contract.
    if kl_coef is not None and abs(float(kl_coef) - controller.base_coef) > 1e-12:
        raise ValueError("forwarded kl_coef does not match the v6 controller base")
    if answer_kl_multiplier < 1.0:
        raise ValueError("answer_kl_multiplier must be >= 1")

    total_loss = 0.0
    total_kl = 0.0
    answer_kl = 0.0
    process_kl = 0.0
    answer_turns = 0
    process_turns = 0
    total_ratio = 0.0
    total_clip = 0.0
    total_engine_mae = 0.0
    engine_turns = 0
    n = 0
    applied_coef = float(controller.coef)

    for question_trajs in trajs_per_question:
        reward_lists = [
            [unpack_turn_v6(turn)[2] for turn in trajectory]
            for trajectory in question_trajs
        ]
        group_returns = [turn_level_returns(rewards) for rewards in reward_lists]
        group_advantages = turn_level_advantages(group_returns)

        for trajectory, advantages in zip(question_trajs, group_advantages):
            for turn_index, turn in enumerate(trajectory):
                inp_ids, out_ids, _, engine_lp, _, kind = unpack_turn_v6(turn)
                if out_ids.numel() == 0:
                    continue
                advantage = torch.tensor(
                    advantages[turn_index], dtype=torch.float32, device="cuda"
                )
                full = torch.cat([inp_ids, out_ids]).unsqueeze(0)

                model.enable_adapter_layers()
                logits = model(full).logits
                policy_lp = completion_token_logprobs(logits, out_ids)
                if engine_lp is not None:
                    engine_lp = engine_lp.to(policy_lp.device, dtype=policy_lp.dtype)
                    mae = float(engine_hf_logprob_mae(policy_lp, engine_lp))
                    if max_engine_logprob_mae is not None and mae > max_engine_logprob_mae:
                        raise RuntimeError(
                            "vLLM/HF raw log-probs are not aligned: "
                            f"turn_MAE={mae:.4f} > {max_engine_logprob_mae:.4f}"
                        )
                    total_engine_mae += mae
                    engine_turns += 1

                old_lp = snapshot_old_policy_logprobs(policy_lp)
                with torch.no_grad():
                    model.disable_adapter_layers()
                    ref_logits = model(full).logits
                    ref_lp = completion_token_logprobs(ref_logits, out_ids)
                    del ref_logits
                model.enable_adapter_layers()

                effective_coef = effective_kl_coefficient(
                    applied_coef,
                    turn_kind=kind,
                    answer_multiplier=answer_kl_multiplier,
                )
                turn_loss, turn_stats = clipped_grpo_token_loss(
                    policy_lp,
                    old_lp,
                    ref_lp,
                    advantage,
                    clip_eps=clip_eps,
                    kl_coef=effective_coef,
                )
                turn_loss.backward()
                kl_value = float(turn_stats["kl"])
                if kind == "answer":
                    answer_kl += kl_value
                    answer_turns += 1
                else:
                    process_kl += kl_value
                    process_turns += 1
                total_loss += float(turn_loss.item())
                total_kl += kl_value
                total_ratio += float(turn_stats["ratio_mean"])
                total_clip += float(turn_stats["clip_fraction"])
                n += 1
                del logits, policy_lp, ref_lp, turn_loss

    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad /= max(n, 1)

    mean_kl = total_kl / max(n, 1)
    next_coef = controller.update(mean_kl)
    stats = {
        "kl": mean_kl,
        "ratio": total_ratio / max(n, 1),
        "clip_fraction": total_clip / max(n, 1),
        "engine_logprob_mae": total_engine_mae / max(engine_turns, 1),
        "answer_kl": answer_kl / max(answer_turns, 1),
        "process_kl": process_kl / max(process_turns, 1),
        "kl_coef_applied": applied_coef,
        "kl_coef_next": next_coef,
        "answer_turns": answer_turns,
        "process_turns": process_turns,
    }
    # Keep the additional control signal visible in the existing train.log
    # without modifying the v1-v5 training loop.
    from grpo_rsf_vllm import log

    log.info(
        "[ACEC-v6] KL applied=%.5f next=%.5f answer=%.5f process=%.5f "
        "answer_turns=%d process_turns=%d",
        applied_coef,
        next_coef,
        stats["answer_kl"],
        stats["process_kl"],
        answer_turns,
        process_turns,
    )
    loss_value = total_loss / max(n, 1)
    return (loss_value, stats) if return_stats else loss_value
