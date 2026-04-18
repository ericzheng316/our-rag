from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict
import os


class QueryRequest(BaseModel):
    query: str


app = FastAPI(title="Simple Retriever Service", version="1.0")

ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = Path(os.environ.get("R3RAG_INDEX_ROOT", str(ROOT / "data/indices")))
TOP_K = int(os.environ.get("R3RAG_TOP_K", "5"))


class SimpleRetriever:
    def __init__(self, index_root: Path) -> None:
        self.index_root = index_root
        self.docs: List[Dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        if not self.index_root.exists():
            raise FileNotFoundError(f"Index root not found: {self.index_root}")
        for idx_file in sorted(self.index_root.glob("*.idx")):
            for i, line in enumerate(idx_file.read_text(encoding="utf-8").splitlines()):
                text = line.strip()
                if not text:
                    continue
                self.docs.append({"id": f"{idx_file.stem}:{i}", "contents": text})
        if not self.docs:
            print(f"[retriever] No docs found under {self.index_root}")
        else:
            print(f"[retriever] Loaded {len(self.docs)} docs from {self.index_root}")

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
        q = query.strip().lower()
        if not q:
            return self.docs[:top_k]
        hits = [doc for doc in self.docs if q in doc["contents"].lower()]
        return hits[:top_k] if hits else self.docs[:top_k]


retriever: SimpleRetriever | None = None


@app.on_event("startup")
def load_retriever() -> None:
    global retriever
    if retriever is None:
        print("Loading simple retriever...")
        retriever = SimpleRetriever(INDEX_ROOT)
        print("Retriever loaded successfully.")


@app.post("/search")
def search(query_request: QueryRequest):
    query = query_request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        return retriever.search(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
