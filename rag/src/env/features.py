"""State feature extraction for retrieval policy."""

from typing import Dict, List


def extract_features(query: str, docs: List[Dict[str, str]], step: int) -> Dict[str, float]:
    return {
        "query_len": float(len(query)),
        "num_docs": float(len(docs)),
        "step": float(step),
    }
