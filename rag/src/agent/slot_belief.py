"""C 臂:非特权的 slot 覆盖信念,格式化进上下文(默认关闭)。

设计依据见 ACEC_V8_STOPPING_DESIGN_NOTES.md 第 3 节:最优停机规则
``停 ⟺ p_t ≥ E[V(s_{t+1})] − c_r``,其中 ``E[V(s')] − p_t`` 即信息价值
(VOI);slot 覆盖后验是估计该量的状态表示。本模块提供其**推理期可算**的
版本 —— 只用 plan 文本与已见 result,**不碰 gold**,因此推理与训练同分布
(与 critic 侧的特权版本区分:那一版只在训练期喂给影子 critic)。

与 B 臂(剩余预算)的关系:B 补的是"还剩多少步",C 补的是"还差哪些证据"
—— 两者都是停机决策的状态变量,可叠加。C 不新增动作(``<answer>`` 本就是
停机动作),故**不需要新 SFT**;它是观测扩展,由 RL 学习使用。

伪后验的校准缺口是已知的:词面匹配 ≠ 真后验,充分性 gap 是实证量,由
预注册消融(A / B / B+C)测定,不做先验断言。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_WORD = re.compile(r"[\w'-]+")

# 内容词判定用的极简停用表:slot 描述与文档共有的功能词不构成证据
_STOP = set(_WORD.findall(
    "the a an of in on at for from to and or is was are were who what which "
    "where when did does do by with that this it its their his her name "
    "answer step find identify determine confirm locate about into over "
    "person people place country city state year date"))


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall(str(text).casefold()) if w not in _STOP}


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def parse_plan_slots(plan_text: str) -> List[str]:
    """从 <plan> 正文提取编号行,返回每个 slot 的描述文本。"""
    slots = []
    for line in str(plan_text).splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
        if m and m.group(2).strip():
            slots.append(m.group(2).strip())
    return slots


@dataclass
class SlotBelief:
    """一个 slot 的覆盖信念(非特权)。

    ``p`` 的更新是 odds 形式的 Bayes:证据分 s∈(0,1) 经 logit 变成对数似然
    比后累加。s=0.5 无信息。与 belief.acec.lookahead_v8.SlotV8 同构,但证据
    分来自词面覆盖而非 gold 标签。
    """

    description: str
    p: float = 0.2                      # 先验 = 实测绑定基率量级
    weight: float = 1.0

    def observe(self, docs: Sequence[str]) -> float:
        """用本轮检索结果更新;返回新后验。"""
        need = _tokens(self.description)
        if not need or not docs:
            return self.p
        best = 0.0
        for d in docs:
            have = _tokens(d)
            if not have:
                continue
            best = max(best, len(need & have) / len(need))
        # 覆盖比 → 证据分:0 覆盖给 0.35(弱否定),全覆盖给 0.9(强肯定)。
        # 区间刻意保守:词面匹配是伪后验,不该给出接近确定的似然比。
        score = 0.35 + 0.55 * min(best, 1.0)
        self.p = min(max(_sigmoid(_logit(self.p) + self.weight * _logit(score)),
                         1e-4), 1 - 1e-4)
        return self.p


@dataclass
class CoverageBelief:
    """整条轨迹的 slot 覆盖信念 + 上下文格式化。"""

    slots: List[SlotBelief] = field(default_factory=list)
    prior: float = 0.2

    def set_plan(self, plan_text: Optional[str]) -> None:
        if not plan_text or self.slots:
            return
        self.slots = [SlotBelief(d, self.prior) for d in parse_plan_slots(plan_text)]

    def observe(self, docs: Sequence[str], slot: Optional[int] = None) -> None:
        """更新信念。``slot`` 给定时只更新该槽(动作条件),否则全体更新。"""
        if not self.slots:
            return
        if slot is not None and 1 <= slot <= len(self.slots):
            self.slots[slot - 1].observe(docs)
        else:
            for s in self.slots:
                s.observe(docs)

    def as_line(self, threshold: float = 0.5) -> str:
        """上下文行。用离散标记而非裸概率:伪后验的精度不足以支撑数字语义,
        且离散标记对 LLM 更易用(covered / partial / open)。"""
        if not self.slots:
            return ""
        marks = []
        for i, s in enumerate(self.slots, 1):
            tag = ("covered" if s.p >= threshold + 0.2
                   else "partial" if s.p >= threshold else "open")
            marks.append(f"{i}:{tag}")
        return "[evidence: " + ", ".join(marks) + "]"

    def snapshot(self) -> Dict[str, float]:
        return {f"slot{i}": round(s.p, 4) for i, s in enumerate(self.slots, 1)}


__all__ = ["CoverageBelief", "SlotBelief", "parse_plan_slots"]
