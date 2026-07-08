"""
GRPO experience maker with R_sf (supporting-facts marginal) process reward.

Step reward  : r_sf_t = |new SF titles found at turn t| / |total SF titles|
Final reward : R_ans  = 1.0 if EM-correct, 0.0 otherwise  (no LLM judge)

Turn-level GRPO advantages (design doc Section 3.2 — load-bearing):
  G_{i,t} = sum_{t' >= t} r_{i,t'}                         (turn_level_returns)
  A_{i,t} = (G_{i,t} - mu_t) / (sigma_t + eps)              (turn_level_advantages)
  where mu_t/sigma_t are computed over rollouts in the group still alive at
  turn t. Each turn's tokens are trained on A_{i,t}, not one trajectory-wide
  scalar. This matters because a shared-question GRPO group's baseline
  cancels the initial state identically across rollouts — under plain
  trajectory-level advantages (one scalar per rollout, broadcast to every
  turn) all of the dense per-turn R_sf signal collapses to a final-coverage
  bonus and the per-turn structure becomes invisible to the gradient. Ported
  from rag/src/belief/acec/reward.py (same semantics, not imported directly:
  that package eager-imports torch/sentence-transformers via its __init__ for
  the belief-specific modules, which this R_sf-only path has no reason to
  depend on).

Key advantages over the existing PPO experience_maker.py:
  - Zero LLM-judge API calls per step → 5-10x faster rollout
  - Dense reward signal at every retrieval turn
  - No critic network needed
"""

from __future__ import annotations

import os
import re
import requests
import string
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import torch

from openrlhf.trainer.ppo_utils.experience_maker import (
    NaiveExperienceMaker,
    Experience,
    Samples,
    split_response,
    ApplyChatTemplate,
    num_search_one_attempt,
    try_format_wrong,
    format_wrong_factor,
    fail_penalty_factor,
    temperature_factor,
)

# ── Retrieval (reads URLs from env so no hardcoded placeholders) ───────────────
_RETRIEVE_URL = os.environ.get("RETRIEVE_URL", "")
_SPLIT_URL    = os.environ.get("SPLIT_URL", "")
_RETRIEVE_K   = int(os.environ.get("RETRIEVE_K", "5"))   # docs per query
_MAX_DOCS     = int(os.environ.get("MAX_DOCS", "10"))     # cap across all turns


def _retrieve(query: str, k: int = _RETRIEVE_K) -> List[dict]:
    """Call the dense retriever. Returns list of {id, contents, score}."""
    assert _RETRIEVE_URL, "RETRIEVE_URL env var not set"
    for _ in range(3):
        try:
            resp = requests.post(
                _RETRIEVE_URL,
                json={"querys": [query]},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code == 200:
                results = resp.json()
                # inference_new.py uses batch API returning list-of-lists
                return results[0][:k] if results and isinstance(results[0], list) else results[:k]
        except Exception as e:
            print(f"[GRPO-RSF] retrieve error: {e}")
    return []

# ── Reward hyper-params ────────────────────────────────────────────────────────
LAMBDA_ANS = 1.0   # weight for final answer EM reward
LAMBDA_SF  = 0.5   # weight for cumulative supporting-facts reward
LAMBDA_FMT = 0.1   # penalty for format errors

# ── Supporting-facts reward ────────────────────────────────────────────────────

def compute_rsf_marginal(
    doc_text: str,
    sf_titles: List[str],
    found_so_far: Set[str],
) -> Tuple[float, Set[str]]:
    """
    Check which supporting-fact titles appear (case-insensitive substring) in
    the concatenated retrieved-document text for this turn.

    Returns (marginal_reward, updated_found_set).
    marginal_reward = |newly found titles| / |total titles|
    """
    if not sf_titles:
        return 0.0, found_so_far

    newly_found: Set[str] = set()
    doc_lower = doc_text.lower()
    for title in sf_titles:
        if title not in found_so_far and title.lower() in doc_lower:
            newly_found.add(title)

    total = len(sf_titles)
    marginal = len(newly_found) / total
    return marginal, found_so_far | newly_found


def compute_em(prediction: str, golden_answers: List[str]) -> bool:
    """Exact match after lowercasing, article removal, and punctuation stripping."""
    def normalize(s: str) -> str:
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = s.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(s.split())

    pred_norm = normalize(prediction)
    return any(normalize(ans) == pred_norm for ans in golden_answers)


# ── Turn-level GRPO advantages (design doc Section 3.2) ────────────────────────

def turn_level_returns(rewards_per_turn: List[float]) -> List[float]:
    """G_{i,t} = sum_{t' >= t} r_{i,t'} for one rollout."""
    returns = [0.0] * len(rewards_per_turn)
    running = 0.0
    for t in range(len(rewards_per_turn) - 1, -1, -1):
        running += rewards_per_turn[t]
        returns[t] = running
    return returns


def turn_level_advantages(group_returns: List[List[float]], eps: float = 1e-6) -> List[List[float]]:
    """
    A_{i,t} = (G_{i,t} - mu_t) / (sigma_t + eps), with mu_t/sigma_t computed
    over rollouts in the group still alive at turn t. Falls back to the raw
    return when fewer than 2 rollouts remain alive at t.

    `group_returns[i]` is `turn_level_returns(reward_list_i)` for rollout i in
    a GRPO group (all rollouts sharing the same question/prompt).
    """
    import math
    max_len = max((len(r) for r in group_returns), default=0)
    advantages = [[0.0] * len(r) for r in group_returns]
    for t in range(max_len):
        alive = [(i, r[t]) for i, r in enumerate(group_returns) if len(r) > t]
        if len(alive) < 2:
            for i, val in alive:
                advantages[i][t] = val
            continue
        vals = [v for _, v in alive]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sigma = math.sqrt(var)
        for i, v in alive:
            advantages[i][t] = (v - mu) / (sigma + eps)
    return advantages


# ── Experience maker ───────────────────────────────────────────────────────────

class GRPORsfExperienceMaker(NaiveExperienceMaker):
    """
    Drop-in replacement for NaiveExperienceMaker that:
      1. Uses R_sf step rewards instead of LLM-judge scoring.
      2. Implements GRPO group-relative normalization (no critic).
    """

    @torch.no_grad()
    def generate_step_samples(
        self,
        all_prompts: dict,
        **generate_kwargs,
    ) -> Tuple[List[Samples], List[float]]:
        """
        Roll out G trajectories per question, compute R_sf + R_ans, return
        (samples_list, rewards_list) in the same flat format as the base class.

        all_prompts must contain:
          - "prompt"         : List[str]   — initial context strings
          - "golden_answers" : after collate → shape (n_answers, batch)
          - "sf_titles_json" : List[str]   — JSON-encoded SF title lists
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()

        n_questions = len(all_prompts["prompt"])

        # Keyed by question index: list of (samples_for_traj, rewards_for_traj, R_total)
        # Each entry = one complete rollout for question q.
        traj_store: Dict[int, List[Tuple[List[Samples], List[float], float]]] = {
            q: [] for q in range(n_questions)
        }

        for i_sample in range(args.n_samples_per_prompt):
            temperature = 0.1 + i_sample * temperature_factor
            generate_kwargs_cur = {**generate_kwargs, "temperature": temperature}

            for q in range(n_questions):
                import json as _json
                prompts_q      = all_prompts["prompt"][q]
                # golden_answers after collate: shape (n_ans, batch) — take first answer slot
                golden_answers = all_prompts["golden_answers"][0][q]
                sf_titles: List[str] = _json.loads(all_prompts["sf_titles_json"][q])

                sample_list: List[Samples] = []
                reward_list: List[float]   = []
                found_sf: Set[str]         = set()

                state        = "undo"
                SystemInput  = prompts_q
                answer       = ""
                retrieved_ids: List[str]  = []
                documents:     List[str]  = []
                r_ans        = 0.0

                for search_counter in range(num_search_one_attempt):
                    # ── Generate one step ──────────────────────────────────────
                    sys_input_templated = ApplyChatTemplate(SystemInput)
                    inputs = self.tokenize_fn(
                        sys_input_templated, self.prompt_max_len, device="cuda"
                    )
                    flag                = True
                    fmt_wrong_count     = 0
                    sequences = attention_mask = action_mask = None

                    while True:
                        fmt_wrong_count += 1
                        sequences, attention_mask, action_mask = self.actor.generate(
                            **inputs, **generate_kwargs_cur
                        )
                        decoded = self.tokenizer.batch_decode(
                            sequences[0].unsqueeze(0), skip_special_tokens=True
                        )[0]
                        step_marker = f"Step {search_counter + 1}:\n"
                        if decoded.count(step_marker) == 1:
                            break
                        if fmt_wrong_count >= try_format_wrong:
                            flag = False
                            break

                    samples_step = Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                    )

                    # ── Format error ───────────────────────────────────────────
                    if not flag:
                        sample_list.append(samples_step)
                        reward_list.append(-LAMBDA_FMT)
                        state = "wrong"
                        break

                    # ── Parse step output ──────────────────────────────────────
                    start_idx = decoded.find(step_marker)
                    this_step = decoded[start_idx:] if start_idx != -1 else decoded
                    mydict = split_response(this_step)

                    if not mydict.get("analysis"):
                        sample_list.append(samples_step)
                        reward_list.append(-LAMBDA_FMT)
                        state = "wrong"
                        break

                    # ── Final answer ───────────────────────────────────────────
                    if mydict.get("answer"):
                        answer = mydict["answer"]
                        r_ans = LAMBDA_ANS if compute_em(answer, [golden_answers]) else 0.0
                        sample_list.append(samples_step)
                        reward_list.append(r_ans)
                        state = "done"
                        break

                    # ── Retrieval step ─────────────────────────────────────────
                    if mydict.get("query"):
                        raw_docs = _retrieve(mydict["query"])
                        new_docs = []
                        for d in raw_docs:
                            if d.get("id") not in retrieved_ids and len(retrieved_ids) < _MAX_DOCS:
                                retrieved_ids.append(d["id"])
                                new_docs.append(d.get("contents", ""))
                        doc_text: str = "\n".join(new_docs)

                        # R_sf: marginal supporting facts found this turn
                        r_sf, found_sf = compute_rsf_marginal(doc_text, sf_titles, found_sf)
                        step_reward = LAMBDA_SF * r_sf

                        sample_list.append(samples_step)
                        reward_list.append(step_reward)

                        # Update context for next turn
                        step_str = (
                            f"Step {search_counter + 1}:\n"
                            f"The problem analysis: {mydict['analysis']}\n"
                            f"The retrieval query: {mydict['query']}\n"
                            f"The retrieval documents: {doc_text[:512]}"
                        )
                        SystemInput = SystemInput + "\n" + step_str

                        if search_counter == num_search_one_attempt - 1:
                            state = "fail"
                    else:
                        sample_list.append(samples_step)
                        reward_list.append(-LAMBDA_FMT)
                        state = "wrong"
                        break

                # ── Trajectory-level adjustments ───────────────────────────────
                if state == "fail":
                    reward_list = [r * (1 - fail_penalty_factor) for r in reward_list]
                elif state in ("wrong",):
                    pass  # keep as-is

                R_total = sum(reward_list)
                traj_store[q].append((sample_list, reward_list, R_total))

        # ── Turn-level GRPO normalization per question (Section 3.2) ───────────
        all_samples: List[Samples] = []
        all_rewards: List[float]   = []
        turn_counts: List[int]     = []   # sanity dashboard: turns per rollout
        r_total_log: List[float]   = []   # sanity dashboard: trajectory returns

        for q in range(n_questions):
            trajs = traj_store[q]
            if not trajs:
                continue

            group_returns = [turn_level_returns(reward_list) for _, reward_list, _ in trajs]
            group_advantages = turn_level_advantages(group_returns)

            for (sample_list, _, r_total), advantages in zip(trajs, group_advantages):
                # advantages[t] is A_{i,t}, computed across the group at turn t
                # (not a single trajectory-wide scalar) — this is what keeps the
                # dense per-turn R_sf signal from collapsing under the shared
                # group baseline. See module docstring.
                all_samples.extend(sample_list)
                all_rewards.extend(advantages)
                turn_counts.append(len(sample_list))
                r_total_log.append(r_total)

        if turn_counts:
            avg_turns = sum(turn_counts) / len(turn_counts)
            avg_r_total = sum(r_total_log) / len(r_total_log)
            print(f"[GRPO-RSF] batch: {len(turn_counts)} rollouts, "
                  f"avg_turns={avg_turns:.2f}, avg_R_total={avg_r_total:.3f}")

        # ── Pad to expected total size (mirrors base class behaviour) ──────────
        target = num_search_one_attempt * n_questions * args.n_samples_per_prompt
        while len(all_samples) < target:
            idx = len(all_samples) % max(1, len(all_samples))
            all_samples.append(all_samples[idx])
            all_rewards.append(all_rewards[idx])

        assert len(all_samples) == len(all_rewards)
        return all_samples, all_rewards

    @torch.no_grad()
    def process_experiences(
        self,
        experiences: List[Experience],
    ) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        GRPO: rewards are already turn-level group-normalized in
        generate_step_samples, so we just pass them through (no additional
        baseline subtraction).
        """
        return experiences, [exp.info["reward"] for exp in experiences]
