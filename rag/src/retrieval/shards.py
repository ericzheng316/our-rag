"""Shard building logic for dynamic RAG indices."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class ShardSpec:
    name: str
    source_path: Path
    index_path: Path


def build_shard(spec: ShardSpec, records: Iterable[str]) -> Path:
    """Persist a lightweight shard artifact as a placeholder index."""
    spec.index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(records)
    spec.index_path.write_text(payload, encoding="utf-8")
    return spec.index_path


def discover_shards(root: Path) -> List[ShardSpec]:
    """Discover all `.idx` shard files under root."""
    return [
        ShardSpec(name=p.stem, source_path=p, index_path=p)
        for p in root.glob("*.idx")
    ]
