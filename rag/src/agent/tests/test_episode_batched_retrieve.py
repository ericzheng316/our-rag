"""batched_retrieve_fn 路径必须与 per-task retriever 路径产出完全一致。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.episode import run_episodes_batched


def _scripted_generate(script):
    """按 (task 序, turn 序) 回放固定输出。"""
    turn = {"i": 0}

    def gen(prompts):
        outs = []
        for j in range(len(prompts)):
            outs.append(script[turn["i"]][j])
        turn["i"] += 1
        return outs
    return gen


def _template(messages):
    return "|".join(m["content"][:20] for m in messages)


TASKS = [
    {"task_id": "a", "question": "qa?", "answer": "ans-a"},
    {"task_id": "b", "question": "qb?", "answer": "ans-b"},
]

# turn0: 两个都 search；turn1: a 答题，b 再 search；turn2: b 答题
SCRIPT = [
    ['<plan>\n1. x\n</plan>\n<search slot="1">alpha</search>',
     '<plan>\n1. y\n</plan>\n<search slot="1">beta</search>'],
    ['<answer>ans-a</answer>',
     '<search slot="2">gamma</search>'],
    ['<answer>ans-b</answer>'],
]


def _doc(q, j):
    return {"id": f"{q}#{j}", "contents": f"doc({q},{j})"}


class BatchedRetrieveEquivalence(unittest.TestCase):
    def _run_factory(self):
        def factory(_task):
            return lambda q, k: [_doc(q, j) for j in range(k)]
        return run_episodes_batched(
            TASKS, _scripted_generate(SCRIPT), factory, _template,
            max_turns=4, docs_per_search=2)

    def _run_batched(self):
        calls = []

        def batched(queries, k):
            calls.append(list(queries))
            return [[_doc(q, j) for j in range(k)] for q in queries]
        eps = run_episodes_batched(
            TASKS, _scripted_generate(SCRIPT), None, _template,
            max_turns=4, docs_per_search=2, batched_retrieve_fn=batched)
        return eps, calls

    def test_identical_results(self):
        a = self._run_factory()
        b, calls = self._run_batched()
        self.assertEqual([e.to_dict() for e in a], [e.to_dict() for e in b])
        # turn0 两条 query 合成一次调用；turn1 只剩 b 的一条
        self.assertEqual(calls, [["alpha", "beta"], ["gamma"]])

    def test_exactly_one_retrieval_shape(self):
        with self.assertRaises(ValueError):
            run_episodes_batched(TASKS, _scripted_generate(SCRIPT), None, _template)

    def test_length_mismatch_raises(self):
        def bad(queries, k):
            return [[_doc(q, 0)] for q in queries][:-1]
        with self.assertRaises(RuntimeError):
            run_episodes_batched(
                TASKS, _scripted_generate(SCRIPT), None, _template,
                max_turns=4, docs_per_search=2, batched_retrieve_fn=bad)


if __name__ == "__main__":
    unittest.main()
