"""Gym-like environment for retrieval decision making."""

from dataclasses import dataclass
from typing import Dict, Tuple

from .features import extract_features
from .reward import compute_reward


@dataclass
class EnvState:
    query: str
    step: int = 0


class RetrievalEnv:
    def __init__(self, max_steps: int = 4) -> None:
        self.max_steps = max_steps
        self.state = EnvState(query="")

    def reset(self, query: str) -> Dict[str, float]:
        self.state = EnvState(query=query, step=0)
        return extract_features(query=query, docs=[], step=0)

    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, float], float, bool, Dict[str, float]]:
        self.state.step += 1
        metrics = {
            "answer_score": action.get("answer_score", 0.0),
            "latency": action.get("latency", 0.0),
        }
        reward = compute_reward(metrics)
        done = self.state.step >= self.max_steps
        obs = extract_features(query=self.state.query, docs=[], step=self.state.step)
        return obs, reward, done, metrics
