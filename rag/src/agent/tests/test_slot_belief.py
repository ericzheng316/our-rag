"""C 臂 slot 覆盖信念(非特权)的行为契约。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.slot_belief import CoverageBelief, SlotBelief, parse_plan_slots


class ParsePlanTest(unittest.TestCase):
    def test_numbered_lines(self):
        slots = parse_plan_slots(
            "1. Find the performer of the song\n"
            "2. Find that performer's birthplace\n"
            "junk line without number")
        self.assertEqual(len(slots), 2)
        self.assertTrue(slots[0].startswith("Find the performer"))

    def test_paren_style_and_empty(self):
        self.assertEqual(len(parse_plan_slots("1) a\n2) b")), 2)
        self.assertEqual(parse_plan_slots(""), [])


class BeliefUpdateTest(unittest.TestCase):
    def test_matching_doc_raises_posterior(self):
        s = SlotBelief("birthplace of Vincent van Gogh", p=0.2)
        p0 = s.p
        s.observe(["Vincent van Gogh was born in Zundert, birthplace museum"])
        self.assertGreater(s.p, p0)

    def test_irrelevant_doc_lowers_posterior(self):
        s = SlotBelief("birthplace of Vincent van Gogh", p=0.2)
        p0 = s.p
        s.observe(["Cricket stadium capacity and ticket prices"])
        self.assertLess(s.p, p0)

    def test_stopwords_do_not_count_as_evidence(self):
        s = SlotBelief("find the country of the person", p=0.2)
        p0 = s.p
        s.observe(["the of the a an in on for and or is was"])
        self.assertLessEqual(s.p, p0)

    def test_bounded(self):
        s = SlotBelief("alpha beta gamma", p=0.2)
        for _ in range(50):
            s.observe(["alpha beta gamma"])
        self.assertLess(s.p, 1.0)
        for _ in range(80):
            s.observe(["zzz"])
        self.assertGreater(s.p, 0.0)


class CoverageBeliefTest(unittest.TestCase):
    def _cb(self):
        cb = CoverageBelief()
        cb.set_plan("1. performer of the song Nevermind\n2. birthplace of that performer")
        return cb

    def test_plan_set_once(self):
        cb = self._cb()
        cb.set_plan("1. something else entirely")   # 后续 plan 不覆盖
        self.assertEqual(len(cb.slots), 2)

    def test_slot_scoped_update(self):
        cb = self._cb()
        before = [s.p for s in cb.slots]
        cb.observe(["Nevermind is a song performed by Kurt"], slot=1)
        self.assertGreater(cb.slots[0].p, before[0])
        self.assertEqual(cb.slots[1].p, before[1])   # 未指定槽不动

    def test_line_format_and_empty(self):
        self.assertEqual(CoverageBelief().as_line(), "")   # 无 plan 不输出
        cb = self._cb()
        line = cb.as_line()
        self.assertTrue(line.startswith("[evidence: "))
        self.assertIn("1:", line)
        self.assertIn("2:", line)
        for tag in ("covered", "partial", "open"):
            line.count(tag)   # 标记词必须来自这三类
        self.assertTrue(all(t in ("covered", "partial", "open")
                            for t in [p.split(":")[1] for p in
                                      line[len("[evidence: "):-1].split(", ")]))

    def test_no_gold_dependency(self):
        """接口层面不接受任何 gold 参数——推理期可算是 C 臂的前提。"""
        import inspect
        for fn in (CoverageBelief.observe, CoverageBelief.set_plan,
                   SlotBelief.observe):
            params = set(inspect.signature(fn).parameters)
            self.assertFalse(params & {"gold", "gold_titles", "golds",
                                       "is_supporting"})

    def test_snapshot(self):
        cb = self._cb()
        snap = cb.snapshot()
        self.assertEqual(set(snap), {"slot1", "slot2"})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in snap.values()))


if __name__ == "__main__":
    unittest.main()
