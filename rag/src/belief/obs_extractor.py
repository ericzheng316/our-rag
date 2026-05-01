"""
Extracts observation signals o_t from R3-RAG's step outputs,
to drive the BeliefState Bayesian update.

Maps R3-RAG's internal data structures → (retrieval_relevance, answer_consistency,
doc_contradiction_rate, query_hop_count) for BeliefState.update().
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
            attention_mask = encoded["attention_mask"]
            token_embs = output.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (token_embs * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts
            normed = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(normed.cpu().float().numpy())
        return np.vstack(all_embeddings)


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

_E5_SCORE_MIN = 0.50
_E5_SCORE_MAX = 0.95
_COVERAGE_SIM_THRESHOLD = 0.60


def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1e-9, norms)
    normed = embeddings / norms
    return normed @ normed.T


# ── Task 1: Z-score calibrated sigmoid retrieval signal ───────────────────────

def extract_retrieval_relevance(
    docs: List[Dict[str, Any]],
    query: str,
    embedder,
    background_docs: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    Task 1: Z-score calibrated sigmoid retrieval quality signal.

    Uses background pool (top-100 retrieved docs) to compute distribution μ, σ.
    Compares top-k mean score against background via Z-score, then maps to (0,1)
    with temperature-scaled sigmoid.

    P = sigmoid(T * (Z - τ)),  T=5.0, τ=1.5

    Falls back to static normalization if no background pool is available.

    # [OLD ALGO - static normalization]:
    # calibrated_relevance = (raw - _E5_SCORE_MIN) / (_E5_SCORE_MAX - _E5_SCORE_MIN)
    # x_ret = 0.50 * calibrated_relevance + 0.30 * coverage + 0.20 * query_coverage
    """
    if not docs:
        return 0.5

    scores_topk = [float(d["score"]) for d in docs if "score" in d]
    if not scores_topk:
        return 0.5

    s_topk = float(np.mean(scores_topk))

    # Use background pool for dynamic calibration when available
    if background_docs and len(background_docs) > len(docs):
        bg_scores = [float(d["score"]) for d in background_docs if "score" in d]
        if len(bg_scores) >= 5:
            mu = float(np.mean(bg_scores))
            sigma = float(np.std(bg_scores))
            if sigma < 1e-6:
                # All scores identical → no discrimination → neutral
                z = 0.0
            else:
                z = (s_topk - mu) / sigma
            T, tau = 5.0, 1.5
            p = 1.0 / (1.0 + math.exp(-T * (z - tau)))
            return float(np.clip(p, 0.01, 0.99))

    # Fallback: static range normalization (original behavior)
    calibrated = (s_topk - _E5_SCORE_MIN) / (_E5_SCORE_MAX - _E5_SCORE_MIN)
    return float(np.clip(calibrated, 0.01, 0.99))


# ── Task 2: LLM reliability via logit margin ──────────────────────────────────

def extract_llm_logit_margin(logprobs_per_token) -> Optional[float]:
    """
    Task 2: LLM reliability signal from logit margin of first 5 generated tokens.

    Margin = P(top1) - P(top2) for each token.
    O_llm = 1 - exp(-k * avg_margin),  k=5.0

    High margin → model is decisive → high reliability.
    Low margin  → model is uncertain → low reliability.

    Returns None if logprobs unavailable (caller skips update).

    # [OLD ALGO - embedding grounding]:
    # answer_consistency = max cosine similarity between answer embedding and doc embeddings
    # See extract_answer_consistency() below (kept for reference).
    """
    if not logprobs_per_token:
        return None

    margins = []
    for token_lps in logprobs_per_token[:5]:
        if not token_lps or len(token_lps) < 2:
            continue
        sorted_lps = sorted(token_lps.values(), key=lambda x: x.logprob, reverse=True)
        try:
            p1 = math.exp(sorted_lps[0].logprob)
            p2 = math.exp(sorted_lps[1].logprob)
            margins.append(p1 - p2)
        except (IndexError, OverflowError):
            continue

    if not margins:
        return None

    avg_margin = float(np.mean(margins))
    k = 5.0
    o_llm = 1.0 - math.exp(-k * avg_margin)
    return float(np.clip(o_llm, 0.01, 0.99))


# ── Legacy signals (kept for reference / distractor mode) ─────────────────────

def extract_answer_consistency(
    response_dict: Dict[str, Any],
    docs: List[Dict[str, Any]],
    embedder,
) -> Optional[float]:
    """
    [LEGACY] Embedding-based grounding score.
    Used in distractor mode (no logprobs available).
    Replaced by logit margin in real-retrieval mode (Task 2).
    """
    answer_text = (
        response_dict.get("answer")
        or response_dict.get("analysis")
        or ""
    )
    if not answer_text or not docs:
        return None

    doc_texts = [d.get("contents", "")[:300] for d in docs]
    all_texts = [answer_text] + doc_texts
    embs = embedder.encode(all_texts, show_progress_bar=False)

    answer_emb = embs[0:1]
    doc_embs = embs[1:]

    sim_mat = _cosine_sim_matrix(np.vstack([answer_emb, doc_embs]))
    grounding = float(sim_mat[0, 1:].max())
    return float(np.clip(grounding, 0.0, 1.0))


def extract_doc_contradiction_rate(
    docs: List[Dict[str, Any]],
    embedder,
) -> float:
    """Topic drift variance as a proxy for corpus noise."""
    if len(docs) < 2:
        return 0.0

    doc_texts = [d.get("contents", "")[:200] for d in docs]
    embs = embedder.encode(doc_texts, show_progress_bar=False)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1e-9, norms)
    normed = embs / norms

    centroid = normed.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-9:
        return 0.0
    centroid = centroid / centroid_norm

    sim_to_centroid = normed @ centroid
    drift_variance = float(np.var(sim_to_centroid))
    noise_signal = min(1.0, drift_variance * 10)
    return float(np.clip(noise_signal, 0.0, 1.0))


def extract_query_hop_count(split_queries: List[str]) -> float:
    """Map number of sub-queries to difficulty signal in [0, 1]."""
    h = max(1, len(split_queries))
    x_diff = 1.0 - math.exp(-0.7 * (h - 1))
    return float(np.clip(x_diff, 0.0, 1.0))


def extract_evidence_novelty(
    new_docs: List[Dict[str, Any]],
    pool_docs: List[Dict[str, Any]],
    embedder,
) -> float:
    """
    How much new information do new_docs add relative to the existing evidence pool.

    novelty = 1 - mean(max cosine similarity of each new doc to any pool doc)
    High (→1): new docs are fresh → retrieval still useful
    Low (→0):  new docs are redundant → retrieval saturated → stop
    """
    if not pool_docs:
        return 1.0
    if not new_docs:
        return 0.0

    new_texts  = [d.get("contents", "")[:200] for d in new_docs]
    pool_texts = [d.get("contents", "")[:200] for d in pool_docs]

    all_embs  = embedder.encode(new_texts + pool_texts, show_progress_bar=False)
    new_embs  = all_embs[:len(new_texts)]
    pool_embs = all_embs[len(new_texts):]

    sims     = new_embs @ pool_embs.T
    max_sims = sims.max(axis=1)
    novelty  = 1.0 - float(max_sims.mean())
    return float(np.clip(novelty, 0.0, 1.0))


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_observation(
    docs: List[Dict[str, Any]],
    query: str,
    embedder,
    response_dict: Optional[Dict[str, Any]] = None,
    split_queries: Optional[List[str]] = None,
    pool_docs: Optional[List[Dict[str, Any]]] = None,
    background_docs: Optional[List[Dict[str, Any]]] = None,
    logprobs_per_token=None,
) -> Dict[str, Any]:
    """
    Extract all observation signals in one call.

    Args:
        docs:              Top-k retrieved documents (used for inference + belief update)
        query:             Retrieval query string for this step
        embedder:          E5Embedder instance
        response_dict:     R3-RAG parsed response with "answer" and/or "analysis" keys
        split_queries:     Sub-queries generated by split module this step
        pool_docs:         Documents from previous turns (for novelty computation)
        background_docs:   Full top-100 retrieval results (Task 1: Z-score background)
        logprobs_per_token: vllm logprobs output (Task 2: logit margin)
    """
    obs: Dict[str, Any] = {}

    # ── Retrieval quality signal ──────────────────────────────────────────────
    # Use novelty when pool exists (turns 2+): measures incremental information gain.
    # Use Z-score calibrated score (Task 1) for first turn or when no pool.
    if pool_docs:  # non-empty pool → turns 2+: use novelty; empty/None → turn 1: use Z-score
        obs["retrieval_relevance"] = extract_evidence_novelty(docs, pool_docs, embedder)
    else:
        # Task 1: Z-score with background pool
        obs["retrieval_relevance"] = extract_retrieval_relevance(
            docs, query, embedder, background_docs=background_docs
        )

    # ── LLM reliability signal ────────────────────────────────────────────────
    # Task 2: Use logit margin when logprobs available (real-retrieval mode).
    # Fallback to embedding grounding in distractor mode (no logprobs).
    if logprobs_per_token is not None:
        llm_rel = extract_llm_logit_margin(logprobs_per_token)
        if llm_rel is not None:
            obs["answer_consistency"] = llm_rel
    elif response_dict is not None and embedder is not None:
        # [LEGACY fallback] embedding-based grounding
        consistency = extract_answer_consistency(response_dict, docs, embedder)
        if consistency is not None:
            obs["answer_consistency"] = consistency

    # ── Corpus noise ──────────────────────────────────────────────────────────
    obs["doc_contradiction_rate"] = extract_doc_contradiction_rate(docs, embedder)

    # ── Query difficulty ──────────────────────────────────────────────────────
    if split_queries is not None:
        obs["query_hop_count"] = extract_query_hop_count(split_queries)

    return obs


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os, json
    sys.path.insert(0, os.path.dirname(__file__))
    from belief_state import BeliefState

    E5_MODEL_PATH = "/root/models/e5-base-v2"
    print(f"Loading E5Embedder from {E5_MODEL_PATH} ...")
    embedder = E5Embedder(E5_MODEL_PATH)

    mock_docs = [
        {"score": 0.82, "contents": "Scott Derrickson is an American director."},
        {"score": 0.74, "contents": "Ed Wood was an American filmmaker."},
        {"score": 0.61, "contents": "The 1999 Odisha cyclone struck eastern India."},
    ]
    mock_background = mock_docs + [
        {"score": 0.65, "contents": "Some other doc."},
        {"score": 0.60, "contents": "Another doc."},
    ]
    obs = extract_observation(
        docs=mock_docs,
        query="Were Scott Derrickson and Ed Wood of the same nationality?",
        embedder=embedder,
        background_docs=mock_background,
    )
    print("obs =", obs)
    belief = BeliefState()
    belief.update(**obs)
    print(json.dumps(belief.to_dict(), indent=2))
