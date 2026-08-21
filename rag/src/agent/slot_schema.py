"""协议 v1.2:规范 plan 的 schema 解析 + 槽状态机(Judge-C 通道,见
ACEC_JUDGE_SPEC.md)。

plan 行格式:``N. <原子单跳问句,可含 #k 引用>? (答案类型)``,例::

    1. Who performed the album Green? (person)
    2. Where was #1 born? (city)

设计要点(全部来自批准规格):
- 校验:2-6 行、行可解析、#k 只允许指向更早的行(构造性无环);
- 三态:blocked(含未解 #k)→ open(引用全绑定)→ filled(判定覆盖);
- 绑定值必须来自可见上下文且 span 回指段落;参数记忆绑定无效——该约束由
  judge 端 span 校验执行,本模块只携带记录;
- 自适应候选:保留与 top-1 分差 ≤ delta 的候选,路径总预算封顶;
- 逐 turn 全量重算语义由上层驱动;(命题, 段落) 缓存在 JudgeCache;
- 本模块零 GPU 依赖:judge_fn 为注入接口,便于 CPU 测试与后续替换。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# 非抽取型答案:门控走类型旁路(只查路径一致性,不查值匹配)
NON_EXTRACTIVE_TYPES = {"yes/no", "yesno", "boolean", "comparison", "count"}

_LINE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*\(([^()]+)\)\s*$")
_REF = re.compile(r"#(\d+)")
_COREF = re.compile(
    r"\b(?:this|these|those|the same)\s+\w+"
    r"|\bthat\s+(?!is\b|are\b|was\b|were\b|has\b|have\b|had\b|did\b"
    r"|does\b|will\b|would\b|can\b|could\b|aired\b|won\b)\w+"
    r"|^\s*(?:he|she|it|they)\b", re.IGNORECASE)


class PlanSchemaError(ValueError):
    """不合法的规范 plan——上层据此走 0.2 协议罚。"""


@dataclass(frozen=True)
class SlotSpec:
    index: int                      # 1 起
    question: str                   # 含 #k 占位的原子问句
    answer_type: str                # 归一化小写
    refs: Tuple[int, ...]           # 引用的更早槽号(升序)

    @property
    def extractive(self) -> bool:
        return self.answer_type not in NON_EXTRACTIVE_TYPES


def parse_canonical_plan(plan_text: str) -> List[SlotSpec]:
    """解析并校验规范 plan;任何违规抛 PlanSchemaError。"""
    lines = [l for l in str(plan_text).splitlines() if l.strip()]
    if not 2 <= len(lines) <= 6:
        raise PlanSchemaError(f"plan 行数 {len(lines)} 不在 2-6")
    slots: List[SlotSpec] = []
    for want, line in enumerate(lines, 1):
        m = _LINE.match(line)
        if not m:
            raise PlanSchemaError(f"行不可解析: {line!r}")
        idx = int(m.group(1))
        if idx != want:
            raise PlanSchemaError(f"编号断裂: 期望 {want} 得到 {idx}")
        q = m.group(2).strip()
        if not q:
            raise PlanSchemaError(f"空问句: {line!r}")
        refs = sorted({int(r) for r in _REF.findall(q)})
        for r in refs:
            if r >= idx or r < 1:
                raise PlanSchemaError(f"槽 {idx} 引用 #{r} 非法(须指向更早行)")
        # 隐式共指探测:非首行且无 #k 时,指示代词式回指断绑定链,拒收。
        # 关系从句 that+动词 由小动词停用表放行;误杀成本仅为丢一条教材。
        if idx > 1 and not refs and _COREF.search(q):
            raise PlanSchemaError(f"槽 {idx} 隐式共指(须用 #k): {q!r}")
        slots.append(SlotSpec(idx, q, m.group(3).strip().lower(), tuple(refs)))
    return slots


@dataclass(frozen=True)
class Binding:
    """一次候选绑定:值 + 逐字 span + 来源段落。span 校验由 judge 端保证。"""
    value: str
    span: str
    para_id: str
    score: float                    # judge 支持概率(已校准)


@dataclass
class SlotNode:
    spec: SlotSpec
    candidates: List[Binding] = field(default_factory=list)

    def state(self, resolved: Dict[int, Binding]) -> str:
        """给定上游已选绑定,本槽在该路径前缀下的状态。"""
        if any(r not in resolved for r in self.spec.refs):
            return "blocked"
        return "filled" if self.candidates else "open"


def instantiate(question: str, resolved: Dict[int, Binding]) -> str:
    """把 #k 替换为已选绑定值 → 完全绑定的原子命题(judge 输入/缓存键)。"""
    def sub(m: re.Match) -> str:
        return resolved[int(m.group(1))].value
    return _REF.sub(sub, question)


class JudgeCache:
    """(命题串, 段落 id) → (value|None, span, score)。跨 turn 持久。"""

    def __init__(self) -> None:
        self._d: Dict[Tuple[str, str], Tuple[Optional[str], str, float]] = {}
        self.misses = 0

    def get_or_call(self, prop: str, para_id: str, para_text: str,
                    judge_fn: Callable[[str, str], Tuple[Optional[str], str, float]],
                    ) -> Tuple[Optional[str], str, float]:
        key = (prop, para_id)
        if key not in self._d:
            self.misses += 1
            val, span, score = judge_fn(prop, para_text)
            # 逐字 span 校验:段落中不存在 → 强制判否(grounding 结构保证)
            if val is not None and span not in para_text:
                val, span, score = None, "", 0.0
            self._d[key] = (val, span, score)
        return self._d[key]


@dataclass
class Path:
    """DAG 上一条一致路径:每槽至多一个绑定,按槽序前缀展开。"""
    resolved: Dict[int, Binding]
    log_score: float

    def joint_states(self, nodes: Sequence[SlotNode]) -> List[str]:
        out = []
        for n in nodes:
            if n.spec.index in self.resolved:
                out.append("filled")
            else:
                out.append(n.state(self.resolved))
        return out


class SlotGraph:
    """槽 DAG + 逐 turn 全量重算驱动。

    ``recompute(paragraphs, judge_fn)``:paragraphs 为当前 C_t 全量
    [(para_id, text)];缓存使重复对零成本。返回联合分最高的一致路径。
    """

    def __init__(self, slots: List[SlotSpec], *, delta: float = 0.15,
                 per_slot_cap: int = 4, path_budget: int = 32,
                 support_threshold: float = 0.5) -> None:
        self.nodes = [SlotNode(s) for s in slots]
        self.delta = delta
        self.per_slot_cap = per_slot_cap
        self.path_budget = path_budget
        self.support_threshold = support_threshold
        self.cache = JudgeCache()
        self.best_path: Optional[Path] = None

    def _candidates_for(self, node: SlotNode, resolved: Dict[int, Binding],
                        paragraphs, judge_fn) -> List[Binding]:
        prop = instantiate(node.spec.question, resolved)
        cands: List[Binding] = []
        for pid, text in paragraphs:
            val, span, score = self.cache.get_or_call(prop, pid, text, judge_fn)
            if val is not None and score >= self.support_threshold:
                cands.append(Binding(val, span, pid, score))
        cands.sort(key=lambda b: -b.score)
        if not cands:
            return []
        keep = [b for b in cands if b.score >= cands[0].score - self.delta]
        return keep[: self.per_slot_cap]

    def recompute(self, paragraphs: Sequence[Tuple[str, str]],
                  judge_fn) -> Optional[Path]:
        """全量重算:beam 式按槽序展开路径,预算封顶。"""
        import math
        paths: List[Path] = [Path({}, 0.0)]
        for node in self.nodes:
            nxt: List[Path] = []
            node.candidates = []
            for p in paths:
                if node.state(p.resolved) == "blocked":
                    nxt.append(p)                      # 断链:路径原样延续
                    continue
                cands = self._candidates_for(node, p.resolved, paragraphs, judge_fn)
                node.candidates.extend(cands)
                if not cands:
                    nxt.append(p)                      # open 未覆盖
                    continue
                for b in cands:
                    r = dict(p.resolved); r[node.spec.index] = b
                    nxt.append(Path(r, p.log_score + math.log(max(b.score, 1e-9))))
            nxt.sort(key=lambda p: (-len(p.resolved), -p.log_score))
            paths = nxt[: self.path_budget]
        self.best_path = paths[0] if paths else None
        return self.best_path

    # ── 下游消费口 ─────────────────────────────────────────────────────
    def coverage_summary(self) -> str:
        """Phase-2 环境行,如 [slots: 1✓(Nolan) 2… 3✗blocked]。"""
        if self.best_path is None:
            return ""
        marks = []
        states = self.best_path.joint_states(self.nodes)
        for n, st in zip(self.nodes, states):
            i = n.spec.index
            if st == "filled":
                marks.append(f"{i}✓({self.best_path.resolved[i].value[:24]})")
            elif st == "open":
                marks.append(f"{i}…")
            else:
                marks.append(f"{i}✗blocked")
        return "[slots: " + " ".join(marks) + "]"

    def gate(self, answer: str, alias_match, tau: float) -> bool:
        """Phase-1b 放行判定(budget=0 的无条件放行由上层负责)。"""
        import math
        p = self.best_path
        if p is None or len(p.resolved) < len(self.nodes):
            return False
        if math.exp(p.log_score / max(len(self.nodes), 1)) < tau:
            return False
        last = self.nodes[-1].spec
        if not last.extractive:
            return True                                # 类型旁路
        return bool(alias_match(answer, [p.resolved[last.index].value]))
