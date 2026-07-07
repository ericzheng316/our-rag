"""
Query -> declarative hypothesis conversion (design doc Section 1.1: each slot
"carries a declarative hypothesis template hyp_j", not the raw query).

NLI cross-encoders are trained on short declarative fragments (SNLI/MNLI-style
hypotheses), not interrogative sentences. Scoring entailment against a raw
retrieval query ("What is the nationality of Scott Derrickson?") gives the
model something far outside its training distribution and destroys most of
its discriminative power — this was empirically confirmed by the Week-1
pilot's first real run: per-slot hit AUC came back ~0.5 (chance) once the
trivial-full-recall bug (num_of_docs=10) was fixed, isolating hypothesis
quality as the next suspect.

This is a cheap, rule-based fragment extraction (strip leading WH-word/
auxiliary tokens), not full sentence realization — deliberately, since Week 1
must not require new GPU inference (design doc Section 9) and R3-RAG's own
emitted queries are sometimes truncated/malformed already (its query stop
condition cuts on the first "?"), so a fragile grammar-complete rewrite would
likely fail more often than this simple strip.
"""

from __future__ import annotations

import re

_LEADING_WORDS = re.compile(
    r"^(what|which|who|whom|when|where|why|how|"
    r"is|was|are|were|did|does|do|can|could|will|would|has|have|had)\s+",
    re.IGNORECASE,
)


def query_to_hypothesis(query: str) -> str:
    """
    Strip up to two leading interrogative/auxiliary tokens and the trailing
    "?", producing a short declarative-style fragment. Falls back to the
    original query (minus "?") if nothing is stripped.

    Examples:
        "What is the nationality of Scott Derrickson?" -> "The nationality of Scott Derrickson."
        "Who directed Sinister?"                       -> "Directed Sinister."
        "Did Scott Derrickson direct Sinister?"        -> "Scott Derrickson direct Sinister."
    """
    q = query.strip().rstrip("?").strip()
    if not q:
        return query

    for _ in range(2):
        m = _LEADING_WORDS.match(q)
        if not m:
            break
        q = q[m.end():].strip()

    if not q:
        return query.strip().rstrip("?")

    return q[0].upper() + q[1:] + "."


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    examples = [
        "What is the nationality of Scott Derrickson?",
        "Who directed Sinister?",
        "Did Scott Derrickson direct Sinister?",
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "How many Grammy awards has the composer won?",
        "government position",  # already-fragment query, no leading WH word
    ]
    for q in examples:
        print(f"{q!r} -> {query_to_hypothesis(q)!r}")
