"""PROVISIONAL SFT trajectory renderer — action taxonomy v0, NOT finalized.

Renders ideal cold-start trajectories from synthesized tasks in a structured
action format.  THE FORMAT BELOW IS A PLACEHOLDER for the open "action
taxonomy" design decision; regenerating SFT data after the format is finalized
is deliberately cheap (this module is the only thing that changes).

v0 protocol (chat messages, thinking disabled):
  system   — protocol definition (slots, actions, answer contract)
  user     — the question
  assistant— first turn: <plan> with one line per slot (mention-masked hints),
             then a single <search slot="1">query</search>
  user     — <result>...docs...</result> (tool feedback rendered as user turn)
  ...      — one search per remaining slot, in chain order
  assistant— final turn: <answer>short answer</answer>

Rendered files are LLaMA-Factory `sharegpt`-style jsonl (messages list), one
trajectory per task.  Queries for hop i are built from the masked hint plus
the entity discovered at hop i-1 (bridge binding made explicit in the ideal
trajectory, so the policy sees worked examples of slot-targeted, entity-bound
queries — the behavior the belief controller later reinforces).
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List

# Single source of truth since the 2026-08-04 finalization: the renderer
# and the rollout environment must never drift apart on the protocol text.
from agent.protocol_v1 import SYSTEM_PROMPT  # noqa: E402


def render_task(task: Dict[str, Any], rng: random.Random, distractors_per_result: int = 1) -> Dict[str, Any]:
    chain = task["chain"]
    hints = task["slot_hints"]
    pool_by_id = {doc["id"]: doc for doc in task["pool"]}
    gold_ids: List[str] = task["gold_chunk_ids"]
    distractor_ids = [d for d in pool_by_id if d not in set(gold_ids)]

    plan_lines = [f"{i + 1}. find the entity marked [?] in: {hint}" for i, hint in enumerate(hints)]
    plan_lines.append(f"{len(hints) + 1}. confirm the final entity's own article")
    plan = "<plan>\n" + "\n".join(plan_lines) + "\n</plan>"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["question"]},
    ]

    discovered: List[str] = []
    for hop, edge in enumerate(chain):
        query_terms = [edge["src_title"]] + discovered[-1:]
        query = " ".join(query_terms) + " " + hints[hop].replace("[?]", "").strip()
        action = f'<search slot="{hop + 1}">{query.strip()}</search>'
        content = f"{plan}\n{action}" if hop == 0 else action
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {"role": "user", "content": _result_block(gold_ids[hop], distractor_ids, pool_by_id, rng, distractors_per_result)}
        )
        discovered.append(edge["dst_title"])

    final_slot = len(chain) + 1
    final_query = f"{task['answer']}"
    messages.append({"role": "assistant", "content": f'<search slot="{final_slot}">{final_query}</search>'})
    messages.append(
        {"role": "user", "content": _result_block(gold_ids[-1], distractor_ids, pool_by_id, rng, distractors_per_result)}
    )
    messages.append({"role": "assistant", "content": f"<answer>{task['answer']}</answer>"})
    return {"task_id": task["task_id"], "k": task["k"], "messages": messages}


def _result_block(
    gold_id: str,
    distractor_ids: List[str],
    pool_by_id: Dict[str, Dict[str, str]],
    rng: random.Random,
    n_distractors: int,
) -> str:
    ids = [gold_id] + rng.sample(distractor_ids, min(n_distractors, len(distractor_ids)))
    rng.shuffle(ids)
    docs = [pool_by_id[i]["contents"] for i in ids if i in pool_by_id]
    return "<result>\n" + "\n---\n".join(docs) + "\n</result>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    n = 0
    with open(args.out, "w", encoding="utf-8") as sink:
        for line in open(args.tasks, encoding="utf-8"):
            task = json.loads(line)
            sink.write(json.dumps(render_task(task, rng), ensure_ascii=False) + "\n")
            n += 1
    print(f"rendered {n} SFT trajectories (action taxonomy v0 — PROVISIONAL)")


if __name__ == "__main__":
    main()
