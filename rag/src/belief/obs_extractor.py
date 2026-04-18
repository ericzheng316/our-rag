"""
Extracts observation signals o_t from R3-RAG's step outputs,
to drive the BeliefState Bayesian update.

Maps R3-RAG's internal data structures → (retrieval_relevance, answer_consistency,
doc_contradiction_rate, query_hop_count) for BeliefState.update().

Stage 1: no NLI models. All signals derived from retriever scores + embedder.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class E5Embedder:
    """
    Thin wrapper around a local E5-base-v2 checkpoint that exposes the same
    .encode(texts) -> np.ndarray interface used throughout obs_extractor.

    E5 encode: mean-pool last hidden states over non-padding tokens, then L2-normalize.
    The model is loaded once and reused across all extractor calls.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode texts to unit-normalized embeddings, shape (len(texts), dim)."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            output = self.model(**encoded)
            # Mean pool over non-padding tokens
            attention_mask = encoded["attention_mask"]               # (b, seq)
            token_embs = output.last_hidden_state                    # (b, seq, dim)
            mask_expanded = attention_mask.unsqueeze(-1).float()     # (b, seq, 1)
            summed = (token_embs * mask_expanded).sum(dim=1)         # (b, dim)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)        # (b, 1)
            mean_pooled = summed / counts                            # (b, dim)
            # L2 normalize → unit vectors, cosine sim = dot product
            normed = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(normed.cpu().float().numpy())
        return np.vstack(all_embeddings)

# Stopwords for query content-word extraction (minimal English set)
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "and", "or", "but", "not", "no",
    "that", "this", "it", "its", "what", "which", "who", "how", "when",
    "where", "why", "if", "as", "up", "out", "about", "into", "than",
    "then", "there", "their", "they", "he", "she", "we", "you", "i",
    "me", "him", "her", "us", "them", "my", "your", "his", "our",
}

# E5-base-v2 cosine similarity range (empirical, used for calibration)
_E5_SCORE_MIN = 0.50
_E5_SCORE_MAX = 0.95

# Minimum cosine similarity to count a query word as "covered" by a doc
_COVERAGE_SIM_THRESHOLD = 0.60


def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute n×n cosine similarity matrix from row-stacked embeddings.
    Assumes embeddings may not be unit-normalized (safe to re-normalize).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1e-9, norms)   # avoid division by zero
    normed = embeddings / norms
    return normed @ normed.T                         # (n, n), values in [-1, 1]


def extract_retrieval_relevance(
    docs: List[Dict[str, Any]],
    query: str,
    embedder,
) -> float:
    """
    Composite retrieval-quality signal from three sub-signals.

    (a) calibrated_relevance  — mean E5 cosine score, mapped to [0,1]
    (b) coverage              — pairwise diversity of retrieved doc embeddings
    (c) query_coverage        — fraction of query content-words semantically covered

    Returns x_ret = 0.50 * calibrated_relevance + 0.30 * coverage + 0.20 * query_coverage
    """
    if not docs:
        return 0.0

    # ── (a) Calibrated relevance ──────────────────────────────────────────────
    scores = [float(d["score"]) for d in docs if "score" in d]
    if scores:
        raw = sum(scores) / len(scores)
    else:
        raw = (_E5_SCORE_MIN + _E5_SCORE_MAX) / 2   # mid-range fallback

    calibrated_relevance = (raw - _E5_SCORE_MIN) / (_E5_SCORE_MAX - _E5_SCORE_MIN)
    calibrated_relevance = float(np.clip(calibrated_relevance, 0.0, 1.0))

    # ── (b) Coverage (diversity of retrieved set) ─────────────────────────────
    doc_texts = [d.get("contents", "")[:200] for d in docs]
    doc_embs = embedder.encode(doc_texts, show_progress_bar=False)   # (n, dim)

    if len(doc_embs) < 2:
        coverage = 0.0
    else:
        sim_mat = _cosine_sim_matrix(doc_embs)       # (n, n)
        n = len(doc_embs)
        # Collect upper-triangle distances (i < j)
        pairwise_distances = [
            1.0 - float(sim_mat[i, j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        # High mean distance → diverse set → better coverage signal
        coverage = float(np.mean(pairwise_distances))
        coverage = float(np.clip(coverage, 0.0, 1.0))

    # ── (c) Query coverage ────────────────────────────────────────────────────
    # Extract content words (non-stopword tokens)
    import re
    tokens = re.findall(r"[a-z]+", query.lower())
    content_words = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    if not content_words:
        query_coverage = 0.5    # no content words → neutral
    else:
        # Encode each content word individually and check against doc embeddings
        word_embs = embedder.encode(content_words, show_progress_bar=False)  # (w, dim)
        sim_mat_wd = _cosine_sim_matrix(
            np.vstack([word_embs, doc_embs])
        )
        # sub-matrix: words (rows 0..w-1) vs docs (rows w..w+n-1)
        w = len(content_words)
        word_doc_sim = sim_mat_wd[:w, w:]   # (w, n)
        max_sim_per_word = word_doc_sim.max(axis=1)   # (w,)
        covered = int((max_sim_per_word >= _COVERAGE_SIM_THRESHOLD).sum())
        query_coverage = covered / len(content_words)

    x_ret = 0.50 * calibrated_relevance + 0.30 * coverage + 0.20 * query_coverage
    return float(np.clip(x_ret, 0.0, 1.0))


def extract_answer_consistency(
    response_dict: Dict[str, Any],
    docs: List[Dict[str, Any]],
    embedder,
) -> Optional[float]:
    """
    Embedding-based grounding score: how well the model's answer is
    supported by the retrieved documents.

    grounding = max cosine similarity between answer embedding and any doc embedding.
    Returns None if no answer text is available (caller should skip the update).
    """
    answer_text = (
        response_dict.get("answer")
        or response_dict.get("analysis")
        or ""
    )
    if not answer_text or not docs:
        return None   # skip update — do NOT default to 0.6

    doc_texts = [d.get("contents", "")[:300] for d in docs]
    all_texts = [answer_text] + doc_texts
    embs = embedder.encode(all_texts, show_progress_bar=False)  # (1+n, dim)

    answer_emb = embs[0:1]       # (1, dim)
    doc_embs = embs[1:]          # (n, dim)

    sim_mat = _cosine_sim_matrix(np.vstack([answer_emb, doc_embs]))
    # Row 0 is answer, cols 1..n are docs
    grounding = float(sim_mat[0, 1:].max())
    return float(np.clip(grounding, 0.0, 1.0))


def extract_doc_contradiction_rate(
    docs: List[Dict[str, Any]],
    embedder,
) -> float:
    """
    Topic drift variance as a proxy for corpus noise.

    High variance in doc-to-centroid similarity → some docs are far from
    the topic center → likely distractors or contradictory content.

    noise_signal = min(1.0, variance_of_centroid_similarities * 10)
    """
    if len(docs) < 2:
        return 0.0

    doc_texts = [d.get("contents", "")[:200] for d in docs]
    embs = embedder.encode(doc_texts, show_progress_bar=False)  # (n, dim)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1e-9, norms)
    normed = embs / norms                              # (n, dim), unit vectors

    centroid = normed.mean(axis=0)                    # (dim,)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-9:
        return 0.0
    centroid = centroid / centroid_norm               # unit centroid

    # Cosine similarity of each doc to the centroid
    sim_to_centroid = normed @ centroid               # (n,)

    # High variance → heterogeneous set → more likely distractors/noise
    drift_variance = float(np.var(sim_to_centroid))
    noise_signal = min(1.0, drift_variance * 10)
    return float(np.clip(noise_signal, 0.0, 1.0))


def extract_query_hop_count(split_queries: List[str]) -> float:
    """
    Map number of sub-queries to a difficulty signal in [0, 1].

    Uses a steeper exponential curve than the previous 1-1/(1+h) sigmoid,
    better separating 1-hop (≈0) from 2-hop (≈0.50) from 3+-hop (≈0.75+).

    x_diff = 1 - exp(-0.7 * (h - 1))
    """
    h = max(1, len(split_queries))
    x_diff = 1.0 - math.exp(-0.7 * (h - 1))
    return float(np.clip(x_diff, 0.0, 1.0))


def extract_observation(
    docs: List[Dict[str, Any]],
    query: str,
    embedder,
    response_dict: Optional[Dict[str, Any]] = None,
    split_queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract all observation signals in one call.

    Returns a dict whose keys exactly match BeliefState.update() parameter names,
    suitable for **kwargs unpacking.

    Args:
        docs:          Retrieved documents (list of dicts with "score", "contents")
        query:         The retrieval query string for this step
        embedder:      A SentenceTransformer instance (reused from retriever pipeline)
        response_dict: R3-RAG parsed response with "answer" and/or "analysis" keys
        split_queries: Sub-queries generated by the split module this step
    """
    obs: Dict[str, Any] = {}

    obs["retrieval_relevance"] = extract_retrieval_relevance(docs, query, embedder)

    if response_dict is not None:
        consistency = extract_answer_consistency(response_dict, docs, embedder)
        if consistency is not None:
            obs["answer_consistency"] = consistency

    obs["doc_contradiction_rate"] = extract_doc_contradiction_rate(docs, embedder)

    if split_queries is not None:
        obs["query_hop_count"] = extract_query_hop_count(split_queries)

    return obs


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from belief_state import BeliefState

    E5_MODEL_PATH = "/home/boyuz5/models/e5-base-v2"
    print(f"Loading E5Embedder from {E5_MODEL_PATH} ...")
    embedder = E5Embedder(E5_MODEL_PATH)

    mock_docs = [
        {
            "score": 0.82,
            "contents": (
                "Scott Derrickson is an American director and screenwriter, "
                "best known for Sinister and Doctor Strange."
            ),
            "title": "Scott Derrickson",
        },
        {
            "score": 0.74,
            "contents": (
                "Ed Wood was an American filmmaker notorious for low-budget productions "
                "in the 1950s and 1960s."
            ),
            "title": "Ed Wood",
        },
        {
            "score": 0.61,
            "contents": (
                "The 1999 Odisha cyclone was a devastating tropical storm that struck "
                "the eastern coast of India."
            ),
            "title": "1999 Odisha cyclone",
        },
    ]

    mock_response = {
        "analysis": "Both Scott Derrickson and Ed Wood are American, so they share the same nationality.",
        "answer": "yes",
    }

    mock_split_queries = [
        "What is the nationality of Scott Derrickson?",
        "What is the nationality of Ed Wood?",
    ]

    query = "Were Scott Derrickson and Ed Wood of the same nationality?"

    print("\n--- Extracting observation ---")
    obs = extract_observation(
        docs=mock_docs,
        query=query,
        embedder=embedder,
        response_dict=mock_response,
        split_queries=mock_split_queries,
    )
    print("obs =", obs)

    print("\n--- Updating BeliefState ---")
    belief = BeliefState()
    belief.update(**obs)
    import json
    print(json.dumps(belief.to_dict(), indent=2))
    print("\nbelief_vector =", belief.belief_vector)
    print("\nRepr:", belief)
