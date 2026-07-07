"""
Entity binding — Algorithm 1, line 4: "bind hypotheses: substitute entities
confirmed in covered slots into hyp_j."

Diagnosed from the Week-1 pilot's debug dump: a slot hypothesis built from a
raw sub-query (e.g. "The actor known for his role in Rebel Without a Cause.")
names a *relation*, not a specific entity — any document that merely mentions
the same film scores high NLI entailment regardless of which actor it's
actually about (observed: 0.999 entailment for a document about the wrong,
co-starring actor). Once another slot in the same question has been resolved
(its coverage posterior p_j crossed a confidence threshold), the concrete
entity that resolved it is real information other slots don't have access to
otherwise — injecting it as context lets the NLI scorer disambiguate "a
document about this specific, now-named entity" from "a document merely
on-topic."

This is a simple context-injection, not real coreference resolution: it does
not try to detect *which* pronoun/phrase in the hypothesis the entity replaces
(that would need a coreference model, which is more machinery than a Week-1
calibration pass should need). It's a cheap prior — extra context concatenated
onto the hypothesis string — that a competent NLI cross-encoder can make use
of even without explicit slot alignment.
"""

from __future__ import annotations

from typing import Sequence


def augment_hypothesis_with_bound_entities(hypothesis: str, bound_entities: Sequence[str]) -> str:
    """Prefix `hypothesis` with any already-confirmed entities from other
    slots in the same question, deduplicated and order-preserving."""
    if not bound_entities:
        return hypothesis
    context = "; ".join(dict.fromkeys(bound_entities))
    return f"{context}. {hypothesis}"
