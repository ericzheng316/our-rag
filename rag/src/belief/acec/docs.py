"""
Doc-dict helpers shared between the live orchestrator (belief_state.py) and
offline replay (run_scripts/build_acec_calibration.py) — kept in one place so
title extraction can't drift between the two call sites.
"""

from __future__ import annotations

from typing import Any, Dict


def title_from_content(content: str) -> str:
    """Distractor-mode paragraphs are formatted 'Title: sentence...' by
    prep_full_distractor.py's hf_context_to_paras(); real-retrieval mode docs
    won't match this convention, hence the fallback."""
    return content.split(": ", 1)[0] if ": " in content else content[:40]


def doc_title(doc: Dict[str, Any]) -> str:
    """Best-effort title for a retrieved doc dict. Prefers an explicit
    'title' key (real-retrieval mode can supply this directly) and falls
    back to parsing 'contents' via the distractor-mode convention."""
    if doc.get("title"):
        return doc["title"]
    return title_from_content(doc.get("contents", ""))
