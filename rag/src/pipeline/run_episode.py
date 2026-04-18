"""End-to-end pipeline: policy -> retrieve -> LLM -> reward."""

from pathlib import Path
from typing import Dict

from src.env.retrieval_env import RetrievalEnv
from src.llm.qwen_client import QwenClient
from src.retrieval.retriever import Retriever


def run_episode(query: str, index_root: Path) -> Dict[str, float]:
    env = RetrievalEnv()
    retriever = Retriever(index_root=index_root)
    llm = QwenClient()

    obs = env.reset(query)
    docs = retriever.retrieve(query=query, top_k=3)
    prompt = f"Question: {query}\nContext: {docs}"
    response = llm.generate(prompt)

    action = {
        "answer_score": 1.0 if response.get("text") else 0.0,
        "latency": float(len(docs)),
    }
    _, reward, _, info = env.step(action)

    return {"reward": reward, **obs, **info}
