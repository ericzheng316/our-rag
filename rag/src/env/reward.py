"""Reward functions for dynamic RAG RL."""

from typing import Dict


def compute_reward(metrics: Dict[str, float], answer_weight: float = 1.0, latency_weight: float = 0.1) -> float:
    answer_score = metrics.get("answer_score", 0.0)
    latency = metrics.get("latency", 0.0)
    return answer_weight * answer_score - latency_weight * latency
