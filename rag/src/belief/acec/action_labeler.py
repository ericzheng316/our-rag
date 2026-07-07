"""
Action labeler lambda(query_t) -> a_t — design doc Section 1.1.

The policy LLM emits free-form text; this deterministic labeler assigns the
abstract action (mode, target slot) from the emitted query, using E5 cosine
similarity against slot hypotheses. No constrained decoding is required:
a_t is a measurable function of the policy's token-level output, which keeps
ACEC drop-in compatible with R3-RAG's free-form format.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: List[str], **kwargs) -> np.ndarray: ...


class ActionMode(str, Enum):
    EXPAND = "EXPAND"
    REWRITE = "REWRITE"
    DECOMPOSE = "DECOMPOSE"
    ANSWER = "ANSWER"


@dataclass
class ActionLabel:
    mode: ActionMode
    target_slot: Optional[int]  # None for ANSWER; set for EXPAND/REWRITE/DECOMPOSE


def _cosine_to_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    vec_n = vec / (np.linalg.norm(vec) + 1e-9)
    mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return mat_n @ vec_n


def _cosine_pair(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


class ActionLabeler:
    """
    target j = argmax_j cos(E5(query), E5(hyp_j)) if max >= tau_new, else DECOMPOSE
    (spawn a new slot). mode = EXPAND if the query is paraphrase-similar
    (cos >= tau_para) to the last query that targeted slot j, else REWRITE.
    """

    def __init__(self, embedder: Embedder, tau_new: float = 0.55, tau_para: float = 0.85):
        self.embedder = embedder
        self.tau_new = tau_new
        self.tau_para = tau_para
        self._last_query_per_slot: Dict[int, str] = {}
        # Diagnostics only — the max cosine similarity seen at the most
        # recent label() call (None if there were no existing slots to
        # compare against). Lets callers inspect what's actually driving the
        # DECOMPOSE-vs-REWRITE/EXPAND decision without changing label()'s
        # return type.
        self.last_max_sim: Optional[float] = None

    def label(self, query: str, slot_hypotheses: List[str], is_answer: bool = False) -> ActionLabel:
        if is_answer:
            return ActionLabel(ActionMode.ANSWER, None)
        if not query:
            self.last_max_sim = None
            return ActionLabel(ActionMode.DECOMPOSE, None)
        if not slot_hypotheses:
            self.last_max_sim = None
            return ActionLabel(ActionMode.DECOMPOSE, None)

        embs = self.embedder.encode([query] + list(slot_hypotheses))
        q_emb, slot_embs = embs[0], embs[1:]
        sims = _cosine_to_matrix(q_emb, slot_embs)
        j = int(np.argmax(sims))
        self.last_max_sim = float(sims[j])
        if float(sims[j]) < self.tau_new:
            return ActionLabel(ActionMode.DECOMPOSE, None)

        mode = ActionMode.REWRITE
        prev_query = self._last_query_per_slot.get(j)
        if prev_query is not None:
            prev_emb, cur_emb = self.embedder.encode([prev_query, query])
            if _cosine_pair(prev_emb, cur_emb) >= self.tau_para:
                mode = ActionMode.EXPAND

        self._last_query_per_slot[j] = query
        return ActionLabel(mode, j)

    def reset(self) -> None:
        self._last_query_per_slot.clear()

    def register_new_slot_query(self, slot_idx: int, query: str) -> None:
        """Record the query that spawned a DECOMPOSE'd slot, so a later
        re-probe of the same slot can be labeled EXPAND vs REWRITE."""
        self._last_query_per_slot[slot_idx] = query
