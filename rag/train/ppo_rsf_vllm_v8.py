"""ACEC v8 PPO trainer — GRPO replaced by PPO with a learned critic.

Why (owner decision 2026-08-05, evidence-backed): the v5 GRPO runs showed the
group-mean baseline extracting nothing from the shaped reward (flat curve over
100 episodes) while remaining the sole variance-control device; with G as
small as 8 the group baseline is noisy and credit assignment leans entirely on
turn_level_advantages. A learned value baseline (critic) replaces the group
statistic, enables n_samples=1 rollouts, and provides GAE(γ,λ) credit
assignment over turns — the standard remedy for exactly this failure shape.

Design:
- Turn-level MDP, identical trajectory format to grpo_rsf_vllm's
  ``vllm_rollout_batch``: per turn ``(input_ids, output_ids, reward,
  engine_logprobs, temperature)``. Engine logprobs are vLLM's *raw* model
  logprobs (untempered) and serve as the frozen rollout-policy logprobs for
  PPO ratios across all ppo_epochs.
- Critic = fp32 linear ValueHead on the policy trunk's last hidden state at
  the final prompt token of each turn (state value before acting). Values for
  GAE come from a no-grad pre-pass with the pre-update weights; the value
  loss is clipped against those same pre-update values.
- Token-level clipped ratios with the turn advantage broadcast to the turn's
  tokens; per-turn backward (the memory pattern grpo_loss_fn established);
  KL penalty against the LoRA-disabled reference; optional entropy bonus.
- Rewards come from the rollout unchanged (terminal exact-match vs
  golden_answers — MuSiQue aliases ride in golden_answers — minus retrieval
  cost per non-answer turn). belief_factory stays None in v8: outcome reward
  only, judge-shaped rewards are a later, explicitly-flagged extension.

vLLM 0.19.1 spawns its EngineCore by re-importing __main__, so everything
executable lives under ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch


# ── 纯数学（CPU 可测）────────────────────────────────────────────────────────

def gae_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """GAE over turns; terminal bootstrap V_T = 0.

    Returns (advantages, returns) with returns_t = advantages_t + values_t —
    the value-regression target.
    """
    if len(rewards) != len(values):
        raise ValueError(f"rewards/values length mismatch: {len(rewards)} vs {len(values)}")
    T = len(rewards)
    advantages = [0.0] * T
    running = 0.0
    for t in range(T - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def normalize_advantages(flat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if flat.numel() < 2:
        return flat
    return (flat - flat.mean()) / (flat.std(unbiased=False) + eps)


def ppo_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Token-level clipped surrogate with a scalar turn advantage broadcast."""
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return -torch.minimum(unclipped, clipped).mean()


def clipped_value_loss(
    v_new: torch.Tensor,
    v_old: torch.Tensor,
    ret: torch.Tensor,
    clip: float = 0.2,
) -> torch.Tensor:
    """PPO-style value clipping against the pre-update estimate."""
    v_clipped = v_old + torch.clamp(v_new - v_old, -clip, clip)
    return 0.5 * torch.maximum((v_new - ret) ** 2, (v_clipped - ret) ** 2).mean()


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    var = returns.var(unbiased=False)
    if var.item() < 1e-12:
        return float("nan")
    return float(1.0 - (returns - values).var(unbiased=False) / var)


class ValueHead(torch.nn.Module):
    """fp32 linear state-value head on the trunk's hidden state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 1, dtype=torch.float32)
        # 零初始化：V≡0 起步。随机初始化在未归一的隐状态（范数可达几十）上
        # 会给出 ±3 量级的初始值，首个 episode 实测 value_loss 5.34、
        # explained_variance -1596；零初始化让首步 value_loss = 0.5·E[ret²]，
        # 梯度经 dL/dW=(V-ret)·h 正常流动。
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden.float()).squeeze(-1)


# ── 训练（重依赖全部函数内导入）──────────────────────────────────────────────

def _turn_forward(model, vhead, inp_ids, out_ids, need_grad: bool):
    """One policy forward yielding (token_logprobs, state_value)."""
    from grpo_estimator_v2 import completion_token_logprobs

    full = torch.cat([inp_ids, out_ids]).unsqueeze(0)
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        out = model(full, output_hidden_states=True)
        token_lp = completion_token_logprobs(out.logits, out_ids)
        state_hidden = out.hidden_states[-1][0, inp_ids.numel() - 1]
        value = vhead(state_hidden)
    return token_lp, value


def _reference_logprobs(model, inp_ids, out_ids):
    from grpo_estimator_v2 import completion_token_logprobs

    with torch.no_grad(), model.disable_adapter():
        logits = model(torch.cat([inp_ids, out_ids]).unsqueeze(0)).logits
        return completion_token_logprobs(logits, out_ids)


def ppo_update(
    model,
    vhead,
    optimizer,
    trajs_per_question,
    *,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    value_clip: float,
    value_coef: float,
    kl_coef: float,
    entropy_coef: float,
    ppo_epochs: int,
    max_grad_norm: float,
    max_engine_logprob_mae: Optional[float],
) -> dict:
    """One PPO update over a rollout batch. Backward per turn (memory)."""
    from grpo_rsf_simple import _unpack_turn
    from grpo_estimator_v2 import engine_hf_logprob_mae

    trajs = [traj for qt in trajs_per_question for traj in qt if traj]
    if not trajs:
        return {"skipped": True}

    # 值预扫（更新前权重）→ GAE
    turn_meta = []  # (traj_idx, t, inp, out, old_lp, adv, ret, v_old)
    model.eval()
    with torch.no_grad():
        for ti, traj in enumerate(trajs):
            rewards, values = [], []
            unpacked = []
            for turn in traj:
                inp, out, r, engine_lp, _temp = _unpack_turn(turn)
                lp, v = _turn_forward(model, vhead, inp, out, need_grad=False)
                if engine_lp is not None and max_engine_logprob_mae is not None:
                    mae = float(engine_hf_logprob_mae(lp, engine_lp))
                    if mae > max_engine_logprob_mae:
                        raise RuntimeError(
                            f"engine/HF logprob MAE {mae:.4f} > {max_engine_logprob_mae}"
                        )
                old_lp = engine_lp if engine_lp is not None else lp
                rewards.append(float(r))
                values.append(float(v))
                unpacked.append((inp, out, old_lp.detach()))
            advs, rets = gae_advantages(rewards, values, gamma, gae_lambda)
            for t, (inp, out, old_lp) in enumerate(unpacked):
                turn_meta.append([ti, t, inp, out, old_lp, advs[t], rets[t], values[t]])

    adv_flat = torch.tensor([m[5] for m in turn_meta], dtype=torch.float32)
    adv_norm = normalize_advantages(adv_flat)
    for m, a in zip(turn_meta, adv_norm.tolist()):
        m[5] = a

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "n_turns": len(turn_meta)}
    model.train()
    device = next(model.parameters()).device
    for _ in range(ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        for ti, t, inp, out, old_lp, adv, ret, v_old in turn_meta:
            new_lp, v_new = _turn_forward(model, vhead, inp, out, need_grad=True)
            adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
            p_loss = ppo_policy_loss(new_lp, old_lp.to(device), adv_t, clip_eps)
            v_loss = clipped_value_loss(
                v_new,
                torch.tensor(v_old, dtype=torch.float32, device=device),
                torch.tensor(ret, dtype=torch.float32, device=device),
                value_clip,
            )
            loss = p_loss + value_coef * v_loss
            if kl_coef > 0.0:
                ref_lp = _reference_logprobs(model, inp, out)
                loss = loss + kl_coef * (new_lp - ref_lp.to(device)).mean()
                stats["kl"] += float((new_lp - ref_lp.to(device)).mean()) / max(len(turn_meta), 1)
            if entropy_coef > 0.0:
                full = torch.cat([inp, out]).unsqueeze(0)
                logits = model(full).logits[0, -(out.numel() + 1):-1].float()
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(-1).mean()
                loss = loss - entropy_coef * entropy
            (loss / len(turn_meta)).backward()
            with torch.no_grad():
                ratio = torch.exp(new_lp - old_lp.to(device))
                stats["clip_frac"] += float(
                    ((ratio - 1.0).abs() > clip_eps).float().mean()
                ) / max(len(turn_meta), 1)
            stats["policy_loss"] += float(p_loss) / max(len(turn_meta), 1)
            stats["value_loss"] += float(v_loss) / max(len(turn_meta), 1)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + list(vhead.parameters()),
            max_grad_norm,
        )
        optimizer.step()
    rets_t = torch.tensor([m[6] for m in turn_meta])
    vals_t = torch.tensor([m[7] for m in turn_meta])
    stats["explained_variance"] = explained_variance(rets_t, vals_t)
    return stats


def train(args) -> None:
    import json
    import os
    import random

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    import grpo_rsf_simple
    from grpo_estimator_v2 import RewardConfig
    from grpo_rsf_vllm import vllm_rollout_batch

    grpo_rsf_simple.RETRIEVE_URL = args.retrieve_url or os.environ.get("RETRIEVE_URL", "")

    rows = []
    with open(args.data_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            q = r.get("question")
            golds = r.get("golden_answers") or ([r["answer"]] if r.get("answer") else [])
            if not q or not golds:
                continue
            rows.append({"question": str(q), "golden_answers": [str(g) for g in golds],
                         "sf_titles": r.get("sf_titles") or []})
    random.Random(args.seed).shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    print(f"[ppo-v8] questions={len(rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0},
    )
    lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                      lora_dropout=0.0, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    vhead = ValueHead(model.config.hidden_size).to(model.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
            {"params": vhead.parameters(), "lr": args.critic_lr},
        ]
    )

    llm = LLM(model=args.model_path, gpu_memory_utilization=args.vllm_gpu_mem_frac,
              enable_lora=True, max_lora_rank=args.lora_rank, dtype="bfloat16")

    os.makedirs(args.save_path, exist_ok=True)
    scratch = os.path.join(args.save_path, "_lora_scratch")
    step = 0
    prev = os.path.join(scratch, f"step_{step}")
    model.save_pretrained(prev)
    lora_request = LoRARequest(f"policy-{step}", step + 1, prev)
    reward_config = RewardConfig(retrieval_cost=args.retrieval_cost)

    metrics_path = os.path.join(args.save_path, "online_metrics_ppo_v8.jsonl")
    for episode in range(args.num_episodes):
        lo = (episode * args.batch_size) % len(rows)
        batch = rows[lo: lo + args.batch_size] or rows[:args.batch_size]
        trajs, online = vllm_rollout_batch(
            llm, lora_request, batch, args.n_samples, args.n_turns,
            belief_factory=None,
            rollout_temperature=args.rollout_temperature,
            reward_config=reward_config, collect_metrics=True,
        )
        stats = ppo_update(
            model, vhead, optimizer, trajs,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps, value_clip=args.value_clip,
            value_coef=args.value_coef, kl_coef=args.kl_coef,
            entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs,
            max_grad_norm=args.max_grad_norm,
            max_engine_logprob_mae=args.max_engine_logprob_mae,
        )
        mean_r = (sum(sum(t[2] for t in traj) for qt in trajs for traj in qt)
                  / max(sum(len(qt) for qt in trajs), 1))
        step += 1
        new_dir = os.path.join(scratch, f"step_{step}")
        model.save_pretrained(new_dir)
        lora_request = LoRARequest(f"policy-{step}", step + 1, new_dir)
        record = {"episode": episode + 1, "mean_R": mean_r, **stats,
                  **{f"rollout_{k}": v for k, v in (online or {}).items()
                     if isinstance(v, (int, float))}}
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[ppo-v8] ep {episode+1:4d} mean_R={mean_r:.3f} "
              f"pl={stats.get('policy_loss', float('nan')):.4f} "
              f"vl={stats.get('value_loss', float('nan')):.4f} "
              f"ev={stats.get('explained_variance', float('nan')):.3f} "
              f"clip={stats.get('clip_frac', float('nan')):.3f}", flush=True)
        if step % args.save_steps == 0:
            ckpt = os.path.join(args.save_path, f"step_{step}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            torch.save(vhead.state_dict(), os.path.join(ckpt, "value_head.pt"))
    model.save_pretrained(args.save_path)
    torch.save(vhead.state_dict(), os.path.join(args.save_path, "value_head.pt"))
    print("[ppo-v8] done", flush=True)


def build_parser():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--retrieve_url", default="")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_samples", type=int, default=2, help="rollouts per question (PPO 不再需要组基线, 1 亦可)")
    p.add_argument("--n_turns", type=int, default=5)
    p.add_argument("--rollout_temperature", type=float, default=0.7)
    p.add_argument("--vllm_gpu_mem_frac", type=float, default=0.55)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--value_clip", type=float, default=0.2)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--retrieval_cost", type=float, default=0.05)
    p.add_argument("--max_engine_logprob_mae", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260805)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
