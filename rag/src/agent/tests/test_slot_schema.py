"""slot_schema CPU 单测:schema 校验、状态机、缓存、级联翻转、门控。"""

import pytest

from agent.slot_schema import (
    Binding,
    JudgeCache,
    PlanSchemaError,
    SlotGraph,
    instantiate,
    parse_canonical_plan,
)

PLAN = """1. Who performed the album Green? (person)
2. Where was #1 born? (city)"""


def test_parse_ok():
    slots = parse_canonical_plan(PLAN)
    assert [s.index for s in slots] == [1, 2]
    assert slots[0].refs == () and slots[1].refs == (1,)
    assert slots[1].answer_type == "city" and slots[1].extractive


@pytest.mark.parametrize("bad", [
    "1. only one line? (person)",                                   # 行数
    "1. a? (x)\n3. b? (y)",                                         # 编号断裂
    "1. asks #2? (x)\n2. b? (y)",                                   # 前向引用
    "1. a? (x)\n2. no type",                                        # 缺类型
    "\n".join(f"{i}. q? (t)" for i in range(1, 8)),                 # 超 6 行
])
def test_parse_rejects(bad):
    with pytest.raises(PlanSchemaError):
        parse_canonical_plan(bad)


def test_instantiate():
    slots = parse_canonical_plan(PLAN)
    r = {1: Binding("Steve Hillage", "s", "p0", 0.9)}
    assert instantiate(slots[1].question, r) == "Where was Steve Hillage born?"


def _judge_factory(table):
    """table: {(命题, 段落文本) -> (val, span, score)};缺省判否。"""
    def judge(prop, text):
        return table.get((prop, text), (None, "", 0.0))
    return judge


P1 = ("p1", "Green is an album by Steve Hillage.")
P2 = ("p2", "Steve Hillage was born in Chingford, London.")
DISTRACTOR = ("p9", "Green Day is an American rock band.")


def test_cascade_and_states():
    slots = parse_canonical_plan(PLAN)
    g = SlotGraph(slots)
    judge = _judge_factory({
        ("Who performed the album Green?", P1[1]):
            ("Steve Hillage", "album by Steve Hillage", 0.92),
        ("Where was Steve Hillage born?", P2[1]):
            ("Chingford", "born in Chingford", 0.88),
    })
    # turn 1:只有 P1 + 干扰段 → 槽1 filled、槽2 open(已解锁但无证据)
    p = g.recompute([P1, DISTRACTOR], judge)
    assert p is not None and 1 in p.resolved and 2 not in p.resolved
    assert g.coverage_summary().startswith("[slots: 1✓(Steve Hillage) 2…")
    # turn 2:P2 到场 → 全量重算级联,槽2 filled
    p = g.recompute([P1, DISTRACTOR, P2], judge)
    assert 2 in p.resolved and p.resolved[2].value == "Chingford"
    # 缓存:旧对零重算(3 段×2 命题=6 对,其中 4 对首 turn 已算)
    assert g.cache.misses == 6


def test_span_verification_forces_negative():
    c = JudgeCache()
    val, span, score = c.get_or_call(
        "q", "p", "real paragraph text",
        lambda prop, text: ("fake", "hallucinated span", 0.99))
    assert val is None and score == 0.0


def test_blocked_propagation():
    slots = parse_canonical_plan(PLAN)
    g = SlotGraph(slots)
    p = g.recompute([DISTRACTOR], _judge_factory({}))
    assert p is not None and not p.resolved
    assert "1…" in g.coverage_summary() and "2✗blocked" in g.coverage_summary()


def test_gate():
    slots = parse_canonical_plan(PLAN)
    g = SlotGraph(slots)
    judge = _judge_factory({
        ("Who performed the album Green?", P1[1]):
            ("Steve Hillage", "album by Steve Hillage", 0.92),
        ("Where was Steve Hillage born?", P2[1]):
            ("Chingford", "born in Chingford", 0.88),
    })
    g.recompute([P1, P2], judge)
    alias = lambda a, golds: a.strip().lower() in {x.lower() for x in golds}
    assert g.gate("Chingford", alias, tau=0.5)
    assert not g.gate("London", alias, tau=0.5)       # 值不匹配路径末端
    assert not g.gate("Chingford", alias, tau=0.95)   # 联合分不过阈


def test_gate_non_extractive_bypass():
    slots = parse_canonical_plan(
        "1. Who directed Inception? (person)\n"
        "2. Are #1 and Christopher Nolan the same person? (yes/no)")
    g = SlotGraph(slots)
    judge = _judge_factory({
        ("Who directed Inception?", "Inception was directed by Christopher Nolan."):
            ("Christopher Nolan", "directed by Christopher Nolan", 0.9),
        ("Are Christopher Nolan and Christopher Nolan the same person?",
         "Inception was directed by Christopher Nolan."):
            ("yes", "Christopher Nolan", 0.8),
    })
    g.recompute([("p1", "Inception was directed by Christopher Nolan.")], judge)
    assert g.gate("yes", lambda a, g_: False, tau=0.5)   # 旁路:不查值匹配
