#!/usr/bin/env python3
"""Placeholder PPO training entrypoint."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.run_episode import run_episode


if __name__ == "__main__":
    result = run_episode(query="what is dynamic rag?", index_root=Path("data/indices"))
    print("[train_ppo placeholder] single episode metrics:", result)
