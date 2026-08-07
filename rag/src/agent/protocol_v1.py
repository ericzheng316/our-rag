"""Action protocol v1 — FINALIZED 2026-08-04 (was render_sft's provisional v0).

Grammar per assistant turn (thinking disabled everywhere):
  first turn:   <plan>numbered slot lines</plan> then exactly one search
  search turn:  <search slot="N">free-text query</search>
  final turn:   <answer>short answer text</answer>
Tool feedback arrives as a user turn: <result>doc\n---\ndoc</result>.

Design notes:
  * the slot number in <search slot="N"> IS the native action label — the
    E5-cosine ActionLabeler is retired from the v8 pipeline; routing is
    exact, not inferred.
  * <answer> content is the deterministic official-answer field (the v6.3
    Gate-1 contract, baked in by SFT rather than begged for by prompts).
  * parser is deterministic and total: anything that doesn't match yields
    InvalidAction with a reason, which downstream treats as a format error
    (reward penalty at training time, episode abort at eval time).

Validated end-to-end before finalization: SFT dev compliance 8/8 first turn,
serve-side 6/6 first turn + 6/6 turn-2 (see CLAUDE.md 2026-08-04 entries).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Union

PROTOCOL_VERSION = "action_protocol_v1.1"

SYSTEM_PROMPT = """You answer multi-hop questions by searching a corpus.
Protocol, in order:
1. First message: a <plan> block with one numbered line per evidence slot you
   need, then exactly one <search slot="N">query</search> for the first slot.
2. After each <result> block, either issue the next <search slot="N">query</search>
   (bind entities you have already confirmed into the query) or finish.
   If the results do not contain what you need, you may search the SAME slot
   again with a rewritten query.
3. Finish with <answer>short answer</answer> — the answer text only, no
   explanation inside the tags. Never issue a search after you can answer.
Before each action you may think briefly inside <think>...</think> (one or
two sentences at most)."""

_SEARCH = re.compile(r"<search slot=\"(\d+)\">(.*?)</search>", re.DOTALL)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_PLAN = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass(frozen=True)
class SearchAction:
    slot: int
    query: str
    plan: str | None  # non-None only when the turn carried a <plan> block


@dataclass(frozen=True)
class AnswerAction:
    answer: str


@dataclass(frozen=True)
class InvalidAction:
    reason: str
    raw_text: str


Action = Union[SearchAction, AnswerAction, InvalidAction]


def parse_assistant_turn(text: str) -> Action:
    """Deterministic, total parse of one assistant turn.

    v1.1: an optional brief <think>...</think> may precede the action — it is
    stripped before action parsing (never a format error; multiple thinks or
    a think alone still yield the underlying action / no_action_tag verdict).
    """
    text = _THINK.sub("", text)
    searches = _SEARCH.findall(text)
    answers = _ANSWER.findall(text)
    if answers and searches:
        return InvalidAction("both_search_and_answer", text)
    if len(searches) > 1:
        return InvalidAction("multiple_searches", text)
    if len(answers) > 1:
        return InvalidAction("multiple_answers", text)
    if answers:
        answer = answers[0].strip()
        if not answer:
            return InvalidAction("empty_answer", text)
        return AnswerAction(answer)
    if searches:
        slot_str, query = searches[0]
        query = " ".join(query.split())
        if not query:
            return InvalidAction("empty_query", text)
        plan_match = _PLAN.search(text)
        return SearchAction(
            slot=int(slot_str),
            query=query,
            plan=plan_match.group(1).strip() if plan_match else None,
        )
    return InvalidAction("no_action_tag", text)


def format_result_block(docs: List[str]) -> str:
    return "<result>\n" + "\n---\n".join(docs) + "\n</result>"


__all__ = [
    "PROTOCOL_VERSION",
    "SYSTEM_PROMPT",
    "Action",
    "AnswerAction",
    "InvalidAction",
    "SearchAction",
    "format_result_block",
    "parse_assistant_turn",
]
