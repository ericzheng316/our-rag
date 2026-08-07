"""协议 v1.1:think 剥离与改写场景的解析行为。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.protocol_v1 import (SYSTEM_PROMPT, AnswerAction, InvalidAction,
                               SearchAction, format_result_block,
                               parse_assistant_turn)


class ThinkStripTest(unittest.TestCase):
    def test_think_before_search(self):
        a = parse_assistant_turn(
            '<think>Results lack the album.</think>\n'
            '<search slot="2">Nevermind release year</search>')
        self.assertIsInstance(a, SearchAction)
        self.assertEqual(a.slot, 2)
        self.assertEqual(a.query, "Nevermind release year")

    def test_think_before_answer(self):
        a = parse_assistant_turn("<think>ok</think><answer>1991</answer>")
        self.assertIsInstance(a, AnswerAction)
        self.assertEqual(a.answer, "1991")

    def test_plain_turn_unchanged(self):
        a = parse_assistant_turn('<search slot="1">q</search>')
        self.assertIsInstance(a, SearchAction)

    def test_think_alone_is_no_action(self):
        a = parse_assistant_turn("<think>hmm</think>")
        self.assertIsInstance(a, InvalidAction)
        self.assertEqual(a.reason, "no_action_tag")

    def test_think_wrapping_action_is_stripped_whole(self):
        # think 里嵌动作标签：整块被剥,不产生动作（防 think 走私动作）
        a = parse_assistant_turn(
            '<think><search slot="1">x</search></think>')
        self.assertIsInstance(a, InvalidAction)

    def test_same_slot_rewrite_is_legal_and_prompt_says_so(self):
        a1 = parse_assistant_turn('<search slot="2">vague relation</search>')
        a2 = parse_assistant_turn('<search slot="2">rewritten specific</search>')
        self.assertIsInstance(a1, SearchAction)
        self.assertIsInstance(a2, SearchAction)
        self.assertIn("SAME slot", SYSTEM_PROMPT)
        self.assertIn("<think>", SYSTEM_PROMPT)


class BudgetLineTest(unittest.TestCase):
    """剩余预算行(状态完备性修复,见 ACEC_V8_STOPPING_DESIGN_NOTES.md)。"""

    def test_absent_by_default(self):
        b = format_result_block(["doc a", "doc b"])
        self.assertTrue(b.endswith("</result>"))
        self.assertNotIn("searches left", b)

    def test_present_and_clamped(self):
        self.assertIn("[searches left: 3]", format_result_block(["d"], turns_left=3))
        self.assertIn("[searches left: 0]", format_result_block(["d"], turns_left=-1))

    def test_block_body_unchanged(self):
        docs = ["alpha", "beta"]
        base = format_result_block(docs)
        with_b = format_result_block(docs, turns_left=2)
        self.assertTrue(with_b.startswith(base))

    def test_prompt_documents_the_signal(self):
        self.assertIn("searches left", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
