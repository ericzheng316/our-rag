"""
Converts BeliefState into a natural-language prefix injected before the LLM prompt.

Design principle: the prefix tells the model *what the system currently believes*
about the environment, so the policy can adapt its next action accordingly.

Example output:
  [Belief] Retrieval: uncertain (quality=0.52). LLM reliability: high (0.78).
  Corpus noise: low (0.18). Query difficulty: hard.
  Recommendation: continue retrieving with a refined query; verify evidence before answering.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .belief_state import BeliefState


# Action recommendations based on belief regime combinations
_ACTION_ADVICE = {
    # (retrieval_regime, is_hard, high_noise)
    ("reliable",   False, False): "Evidence looks sufficient. Consider answering now.",
    ("reliable",   True,  False): "Hard query but retrieval is working. Retrieve one more targeted sub-question.",
    ("reliable",   False, True ): "Docs may be noisy. Filter or verify evidence before answering.",
    ("reliable",   True,  True ): "Hard query with noisy corpus. Re-rank docs and verify before answering.",
    ("uncertain",  False, False): "Retrieval quality unclear. One more retrieval attempt is warranted.",
    ("uncertain",  True,  False): "Hard query, uncertain retrieval. Decompose and retrieve sub-questions.",
    ("uncertain",  False, True ): "Noisy corpus. Prefer filtering/reranking over additional retrieval.",
    ("uncertain",  True,  True ): "Hard query + noisy corpus. Decompose query and filter retrieved docs.",
    ("unreliable", False, False): "Retrieval seems poor. Consider rephrasing the query or stopping.",
    ("unreliable", True,  False): "Hard query + poor retrieval. Rephrase sub-questions or abstain.",
    ("unreliable", False, True ): "Poor retrieval + noisy corpus. High risk of hallucination — consider abstaining.",
    ("unreliable", True,  True ): "Poor retrieval + hard query + noisy corpus. Strong signal to abstain.",
}


def belief_to_prompt_prefix(belief: "BeliefState") -> str:
    """
    Returns a concise natural-language summary of the current belief state
    to be prepended to the LLM's context window at each reasoning step.
    """
    reg = belief.retrieval_regime
    hard = belief.is_hard_query
    high_noise = belief.corpus_noise > 0.5

    advice = _ACTION_ADVICE.get(
        (reg, hard, high_noise),
        "Belief state inconclusive. Use standard retrieval strategy."
    )

    prefix = (
        f"[Belief State @ step {belief.step}] "
        f"Retrieval: {reg} (E[θ_ret]={belief.ret_quality:.2f}, "
        f"uncertainty={belief.ret_uncertainty:.3f}). "
        f"LLM reliability: E[θ_llm]={belief.llm_reliability:.2f}. "
        f"Corpus noise: E[θ_noise]={belief.corpus_noise:.2f}. "
        f"Query difficulty: {'hard' if hard else 'easy'} (E[θ_diff]={belief.query_difficulty:.2f}). "
        f"→ {advice}"
    )
    return prefix


def belief_to_short_tag(belief: "BeliefState") -> str:
    """
    Ultra-compact version for token-budget-sensitive prompts (< 20 tokens).
    Example: [B: ret=0.65↑ llm=0.80 noise=0.12 diff=hard]
    """
    arrow = "↑" if belief.retrieval_regime == "reliable" else ("↓" if belief.retrieval_regime == "unreliable" else "~")
    return (
        f"[B: ret={belief.ret_quality:.2f}{arrow} "
        f"llm={belief.llm_reliability:.2f} "
        f"noise={belief.corpus_noise:.2f} "
        f"diff={'hard' if belief.is_hard_query else 'easy'}]"
    )
