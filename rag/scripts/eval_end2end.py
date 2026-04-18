#!/usr/bin/env python3
"""Placeholder end-to-end evaluation script."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.run_episode import run_episode


if __name__ == "__main__":
    result = run_episode(query="evaluate rag retrieval", index_root=Path("data/indices"))
    print("[eval_end2end]", result)
