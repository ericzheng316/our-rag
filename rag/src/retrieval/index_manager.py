"""Manage multiple retrieval shards."""

from pathlib import Path
from typing import Dict, Iterable, List

from .shards import ShardSpec, build_shard, discover_shards


class IndexManager:
    def __init__(self, index_root: Path) -> None:
        self.index_root = index_root
        self._shards: Dict[str, ShardSpec] = {}

    def register_shard(self, name: str, records: Iterable[str]) -> Path:
        spec = ShardSpec(
            name=name,
            source_path=self.index_root / f"{name}.jsonl",
            index_path=self.index_root / f"{name}.idx",
        )
        index_path = build_shard(spec, records)
        self._shards[name] = spec
        return index_path

    def load_existing(self) -> List[ShardSpec]:
        specs = discover_shards(self.index_root)
        self._shards.update({s.name: s for s in specs})
        return specs

    def list_shards(self) -> List[str]:
        return sorted(self._shards.keys())
