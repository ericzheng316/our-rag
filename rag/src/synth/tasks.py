"""Chain -> task record: requirements, closed candidate pool, template question.

Every task carries:
  * ``evidence_specification`` — the canonical v5 manifest (consumed unchanged
    by CanonicalEvidenceAdapterV5, annotation_scope=closed_candidate_pool);
  * ``pool`` — the closed candidate pool: gold chunk per hop + confusable
    distractors, shuffled.  Calibration and distractor-mode rollouts read
    documents from here only, which is what licenses confident zero-gain
    labels under the v5 standard;
  * ``question`` — a deterministic template question so the CPU pipeline is
    end-to-end runnable; verbalize.py replaces it with a natural one;
  * ``slot_hints`` — mention-masked hop sentences for the SFT planner, which
    must never name the entity the hop is supposed to discover;
  * ``answer`` — the display title of the final entity (identify-the-entity
    task family; EM/F1-friendly short string);
  * ``closed_book`` — null placeholder, filled by the later screen script.

Leakage rule enforced here: neither the question nor any slot hint may
contain the answer string.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Optional

from .corpus import CorpusIndex, normalize_title
from .graph import MentionEdge

SYNTH_VERSION = "synth_chain_v0.2"
MASK = "[?]"


def mask_mention(sentence: str, mention: str) -> str:
    """Punctuation-tolerant masking: the extracted span joins tokens with
    single spaces, but the sentence may separate them with commas/periods
    ("Pleasanton California" vs "Pleasanton, California"), so match tokens
    with any punctuation/whitespace run in between."""
    tokens = [re.escape(token) for token in mention.split()]
    pattern = r"[\s,;:.'\"()\-]+".join(tokens)
    return re.sub(pattern, MASK, sentence, flags=re.IGNORECASE)


def _leaks_entity(text: str, display_title: str) -> bool:
    """Normalized containment check: does ``text`` still name the entity?"""
    norm_text = f" {normalize_title(text)} "
    norm_title = f" {normalize_title(display_title)} "
    return norm_title.strip() != "" and norm_title in norm_text


def _task_id(chain: List[MentionEdge]) -> str:
    canonical = "|".join(f"{e.src_chunk_id}->{e.dst_title}" for e in chain)
    return "synth:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_task(
    chain: List[MentionEdge],
    index: CorpusIndex,
    chunks: Dict[str, Dict[str, str]],
    rng: random.Random,
    n_distractors: int = 10,
    max_pool: int = 20,
) -> Optional[Dict[str, Any]]:
    """Assemble one task record; returns None if the chain leaks or is broken."""
    final_key = chain[-1].dst_title
    answer = index.display_title[final_key]
    final_chunk_id = index.title_chunks[final_key][0]

    gold_ids = [edge.src_chunk_id for edge in chain] + [final_chunk_id]
    if any(chunk_id not in chunks for chunk_id in gold_ids):
        return None

    requirements: List[Dict[str, Any]] = []
    slot_hints: List[str] = []
    for hop, edge in enumerate(chain):
        hint = mask_mention(edge.sentence, edge.mention_text)
        # The hint must hide what this hop is supposed to discover: drop the
        # task if the destination entity survives masking anywhere in the
        # sentence (second unmasked mention, alias overlap), or if the final
        # answer appears in any hint.
        if _leaks_entity(hint, index.display_title[edge.dst_title]) or _contains(hint, answer):
            return None
        slot_hints.append(hint)
        requirements.append(
            {
                "requirement_id": f"hop:{hop}",
                "description": (
                    f"{index.display_title[edge.src_title]}: {edge.sentence}"
                ),
                "evidence_units": [
                    {
                        "unit_id": f"hop{hop}:{edge.src_chunk_id}",
                        "text": edge.sentence,
                        "source_id": index.display_title[edge.src_title],
                        "metadata": {"chunk_id": edge.src_chunk_id, "mention": edge.mention_text},
                    }
                ],
                "satisfaction_rule": "all",
                "metadata": {"hop": hop, "bridge_to": index.display_title[edge.dst_title]},
            }
        )
    final_text = chunks[final_chunk_id]["contents"]
    requirements.append(
        {
            "requirement_id": f"hop:{len(chain)}",
            "description": f"{answer}: {_first_sentence(final_text)}",
            "evidence_units": [
                {
                    "unit_id": f"hop{len(chain)}:{final_chunk_id}",
                    "text": _first_sentence(final_text),
                    "source_id": answer,
                    "metadata": {"chunk_id": final_chunk_id},
                }
            ],
            "satisfaction_rule": "all",
            "metadata": {"hop": len(chain), "is_answer_entity": True},
        }
    )

    pool = _closed_pool(gold_ids, chain, final_key, index, chunks, rng, n_distractors, max_pool)
    question = template_question(index.display_title[chain[0].src_title], slot_hints)
    if _contains(question, answer):
        return None

    return {
        "task_id": _task_id(chain),
        "synth_version": SYNTH_VERSION,
        "k": len(requirements),
        "question": question,
        "question_source": "template",
        "answer": answer,
        "answer_aliases": [],
        "chain": [
            {
                "src_title": index.display_title[e.src_title],
                "dst_title": index.display_title[e.dst_title],
                "src_chunk_id": e.src_chunk_id,
                "mention": e.mention_text,
                "sentence": e.sentence,
            }
            for e in chain
        ],
        "slot_hints": slot_hints,
        "gold_chunk_ids": gold_ids,
        "pool": pool,
        "closed_book": None,
        "evidence_specification": {
            "question_id": _task_id(chain),
            "requirements": requirements,
            "annotation_scope": "closed_candidate_pool",
            "adapter_name": "canonical_evidence_manifest_v1",
            "annotation_version": SYNTH_VERSION,
            "metadata": {"family": "identify_final_entity"},
        },
    }


def template_question(start_display_title: str, slot_hints: List[str]) -> str:
    """Deterministic fallback question — clunky but unambiguous and leak-free."""
    steps = []
    for hop, hint in enumerate(slot_hints):
        steps.append(f"({hop + 1}) it is the entity marked {MASK} in: \"{hint}\"")
    joined = "; then ".join(steps)
    return (
        "Follow the reference chain and name the final entity. "
        f"Starting point: the article about {start_display_title}. {joined}. "
        "What is the final entity?"
    )


def pool_candidate_ids(chain: List[MentionEdge], index: CorpusIndex) -> List[str]:
    """Every chunk id a task's pool may draw from (golds + distractor candidates).

    The entry script unions this over all chains to form the ``wanted`` set for
    the second corpus pass; _closed_pool draws from the same candidates so the
    two can never drift apart.
    """
    final_key = chain[-1].dst_title
    ids = [edge.src_chunk_id for edge in chain] + [index.title_chunks[final_key][0]]
    for title in [e.src_title for e in chain] + [final_key]:
        for other in index.confusable_titles(title, limit=6):
            ids.extend(index.title_chunks[other][:1])
    seen: List[str] = []
    for chunk_id in ids:
        if chunk_id not in seen:
            seen.append(chunk_id)
    return seen


def _closed_pool(
    gold_ids: List[str],
    chain: List[MentionEdge],
    final_key: str,
    index: CorpusIndex,
    chunks: Dict[str, Dict[str, str]],
    rng: random.Random,
    n_distractors: int,
    max_pool: int,
) -> List[Dict[str, str]]:
    del final_key
    distractor_ids = [
        chunk_id for chunk_id in pool_candidate_ids(chain, index) if chunk_id not in gold_ids
    ]
    rng.shuffle(distractor_ids)
    distractor_ids = [d for d in distractor_ids if d in chunks][:n_distractors]
    pool_ids = gold_ids + distractor_ids
    pool = [chunks[chunk_id] for chunk_id in pool_ids][:max_pool]
    rng.shuffle(pool)
    return pool


def _first_sentence(contents: str) -> str:
    body = contents.split(": ", 1)[1] if ": " in contents else contents
    for mark in (". ", "! ", "? "):
        if mark in body:
            return body.split(mark, 1)[0] + mark.strip()
    return body[:200]


def _contains(haystack: str, needle: str) -> bool:
    return needle.casefold() in haystack.casefold()
