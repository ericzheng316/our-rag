"""Basic retriever over shard artifacts."""

from pathlib import Path
from typing import Dict, List


class Retriever:
    def __init__(self, index_root: Path) -> None:
        self.index_root = index_root

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Return mock retrieval docs from available shard files."""
        hits: List[Dict[str, str]] = []
        for idx_file in sorted(self.index_root.glob("*.idx")):
            for line in idx_file.read_text(encoding="utf-8").splitlines():
                if query.lower() in line.lower() or not query.strip():
                    hits.append({"shard": idx_file.stem, "text": line})
                if len(hits) >= top_k:
                    return hits
        return hits
