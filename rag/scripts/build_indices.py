#!/usr/bin/env python3
"""Build shard indices from simple in-memory examples."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.index_manager import IndexManager


if __name__ == "__main__":
    manager = IndexManager(index_root=Path("data/indices"))
    manager.register_shard("shard_a", ["dynamic rag introduction", "ppo policy update"])
    manager.register_shard("shard_b", ["retrieval augmentation", "reward shaping tips"])
    print("Built shards:", manager.list_shards())
