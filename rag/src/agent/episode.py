"""Multi-turn episode runner + the v8 rollout log record.

The runner is engine-agnostic: ``generate_fn(prompt_text) -> str`` may be a
local HF model, one vLLM engine, or the engine pool's remote call.  Retrieval
is likewise ``retrieve_fn(query, k) -> List[doc dict]`` — the smoke path
serves it from a task's closed candidate pool (distractor mode, which is also
the calibration-friendly regime); the live path plugs the FAISS retriever
client with the same signature.

One EpisodeResult is one line of the rollout log — the Phase-2 calibration
replay consumes exactly these fields (native actions, per-turn doc ids,
preseedable slot plan), so the schema here IS the log schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .protocol_v1 import (
    PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    AnswerAction,
    InvalidAction,
    SearchAction,
    format_result_block,
    parse_assistant_turn,
)


@dataclass
class TurnRecord:
    turn_index: int
    action_type: str            # search | answer | invalid
    slot: Optional[int]
    query: Optional[str]
    plan: Optional[str]
    doc_ids: List[str]
    raw_text: str
    invalid_reason: Optional[str] = None
    doc_titles: List[str] = field(default_factory=list)  # wiki18 contents 首行


def _title_of(contents: str) -> str:
    """FlashRAG wiki18 格式 '"Title"\ntext' 的首行 title，规范化小写。"""
    head = str(contents).split("\n", 1)[0].strip().strip('"')
    return " ".join(head.split()).lower()


@dataclass
class EpisodeResult:
    task_id: str
    protocol_version: str
    question: str
    gold_answer: str
    final_answer: Optional[str]
    turns: List[TurnRecord] = field(default_factory=list)
    n_searches: int = 0
    protocol_error: bool = False
    hit_turn_limit: bool = False
    messages: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_episode(
    task: Dict[str, Any],
    generate_fn: Callable[[str], str],
    retrieve_fn: Callable[[str, int], List[Dict[str, str]]],
    chat_template_fn: Callable[[List[Dict[str, str]]], str],
    max_turns: int = 8,
    docs_per_search: int = 3,
) -> EpisodeResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["question"]},
    ]
    result = EpisodeResult(
        task_id=task["task_id"],
        protocol_version=PROTOCOL_VERSION,
        question=task["question"],
        gold_answer=task["answer"],
        final_answer=None,
    )

    for turn_index in range(max_turns):
        raw = generate_fn(chat_template_fn(messages))
        action = parse_assistant_turn(raw)
        messages.append({"role": "assistant", "content": raw})

        if isinstance(action, AnswerAction):
            result.turns.append(TurnRecord(
                turn_index, "answer", None, None, None, [], raw))
            result.final_answer = action.answer
            break
        if isinstance(action, InvalidAction):
            result.turns.append(TurnRecord(
                turn_index, "invalid", None, None, None, [], raw,
                invalid_reason=action.reason))
            result.protocol_error = True
            break

        docs = retrieve_fn(action.query, docs_per_search)
        result.turns.append(TurnRecord(
            turn_index, "search", action.slot, action.query, action.plan,
            [str(d.get("id", "")) for d in docs], raw))
        result.n_searches += 1
        messages.append({
            "role": "user",
            "content": format_result_block([d["contents"] for d in docs]),
        })
    else:
        result.hit_turn_limit = True

    result.messages = messages
    return result


def run_episodes_batched(
    tasks: List[Dict[str, Any]],
    batched_generate_fn: Callable[[List[str]], List[str]],
    retriever_factory: Optional[
        Callable[[Dict[str, Any]], Callable[[str, int], List[Dict[str, str]]]]] = None,
    chat_template_fn: Callable[[List[Dict[str, str]]], str] = None,
    max_turns: int = 8,
    docs_per_search: int = 3,
    *,
    batched_retrieve_fn: Optional[
        Callable[[List[str], int], List[List[Dict[str, str]]]]] = None,
) -> List[EpisodeResult]:
    """Lockstep turn-boundary batching: all active episodes generate together.

    This is the driver shape the engine pool wants — one large prompt batch
    per turn boundary, sharded across engines — and the same loop the GRPO
    trainer will use, so it lives here rather than in a script.
    Semantics are identical to run_episode.

    Retrieval takes one of two shapes (exactly one must be given):
    - ``retriever_factory``: per-task retriever, called serially per episode
      (the closed-pool path, where retrieval is local and cheap).
    - ``batched_retrieve_fn(queries, k) -> list of doc-lists``: all active
      searches at a turn boundary go out in ONE call, in input order — the
      FAISS-server path, whose /search endpoint natively batch-encodes; the
      per-query serial loop was measured at ~13 s per boundary at batch 256
      vs ~2 s batched.
    """
    if (retriever_factory is None) == (batched_retrieve_fn is None):
        raise ValueError("provide exactly one of retriever_factory / batched_retrieve_fn")
    states = []
    for task in tasks:
        result = EpisodeResult(
            task_id=task["task_id"],
            protocol_version=PROTOCOL_VERSION,
            question=task["question"],
            gold_answer=task["answer"],
            final_answer=None,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["question"]},
        ]
        states.append({
            "result": result,
            "messages": messages,
            "retrieve": retriever_factory(task) if retriever_factory else None,
            "done": False,
        })

    for turn_index in range(max_turns):
        active = [s for s in states if not s["done"]]
        if not active:
            break
        prompts = [chat_template_fn(s["messages"]) for s in active]
        texts = batched_generate_fn(prompts)

        # 先解析全部动作，收集本 turn 的 search，好一次批发
        parsed = []
        searching = []  # (state, action) needing docs, in order
        for state, raw in zip(active, texts):
            action = parse_assistant_turn(raw)
            state["messages"].append({"role": "assistant", "content": raw})
            parsed.append((state, raw, action))
            if isinstance(action, SearchAction):
                searching.append((state, action))

        docs_by_id: Dict[int, List[Dict[str, str]]] = {}
        if searching and batched_retrieve_fn is not None:
            doc_lists = batched_retrieve_fn(
                [a.query for _, a in searching], docs_per_search)
            if len(doc_lists) != len(searching):
                raise RuntimeError(
                    f"batched_retrieve_fn returned {len(doc_lists)} lists "
                    f"for {len(searching)} queries")
            for (state, _a), docs in zip(searching, doc_lists):
                docs_by_id[id(state)] = docs

        for state, raw, action in parsed:
            result: EpisodeResult = state["result"]
            if isinstance(action, AnswerAction):
                result.turns.append(TurnRecord(
                    turn_index, "answer", None, None, None, [], raw))
                result.final_answer = action.answer
                state["done"] = True
            elif isinstance(action, InvalidAction):
                result.turns.append(TurnRecord(
                    turn_index, "invalid", None, None, None, [], raw,
                    invalid_reason=action.reason))
                result.protocol_error = True
                state["done"] = True
            else:
                if batched_retrieve_fn is not None:
                    docs = docs_by_id[id(state)]
                else:
                    docs = state["retrieve"](action.query, docs_per_search)
                result.turns.append(TurnRecord(
                    turn_index, "search", action.slot, action.query, action.plan,
                    [str(d.get("id", "")) for d in docs], raw,
                    doc_titles=[_title_of(d.get("contents", "")) for d in docs]))
                result.n_searches += 1
                state["messages"].append({
                    "role": "user",
                    "content": format_result_block([d["contents"] for d in docs]),
                })

    for state in states:
        if not state["done"]:
            state["result"].hit_turn_limit = True
        state["result"].messages = state["messages"]
    return [state["result"] for state in states]


def pool_retriever(task: Dict[str, Any]) -> Callable[[str, int], List[Dict[str, str]]]:
    """Distractor-mode retrieval over the task's own closed candidate pool.

    Deterministic lexical scoring (token F1 of query vs contents) — no
    embedder, no server; matches the closed-pool calibration regime.  The
    live-corpus path replaces this with the FAISS retriever client.
    """
    import re
    from collections import Counter

    def tokens(text: str) -> List[str]:
        return re.findall(r"[\w'-]+", text.casefold())

    pool = task["pool"]
    doc_tokens = [(doc, Counter(tokens(doc["contents"]))) for doc in pool]

    def retrieve(query: str, k: int) -> List[Dict[str, str]]:
        q_tokens = Counter(tokens(query))
        scored = []
        for doc, d_tokens in doc_tokens:
            common = sum((q_tokens & d_tokens).values())
            if not common:
                scored.append((0.0, doc))
                continue
            precision = common / max(sum(q_tokens.values()), 1)
            recall = common / max(sum(d_tokens.values()), 1)
            scored.append((2 * precision * recall / (precision + recall), doc))
        scored.sort(key=lambda pair: -pair[0])
        return [doc for _, doc in scored[:k]]

    return retrieve
