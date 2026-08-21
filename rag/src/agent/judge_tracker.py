"""J8 观测注入:judge 版槽覆盖追踪器(轻量,单路径 top-1 语义与 Phase-1a 一致)。

与 SlotGraph 的差别:为 rollout 环内批处理而生——wave_update 对一批追踪器
按拓扑深度做跨轨迹批判定(一次 HTTP 一个深度),judge 调用按 (命题, 段落id)
缓存,逐 turn 全量重算语义由缓存兜底(旧对零成本)。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .slot_schema import SlotSpec, instantiate as _inst, Binding

JudgeBatchFn = Callable[[List[Tuple[str, str]]], List[dict]]
# 输入 [(bound_question, para_text)] → [{"val","span","score"}]


class JudgeTracker:
    def __init__(self, slots: List[SlotSpec]) -> None:
        self.slots = slots
        self.paras: List[Tuple[str, str]] = []        # (pid=标题行, text)
        self._seen_pid: set = set()
        self.resolved: Dict[int, Binding] = {}
        self.cache: Dict[Tuple[str, str], Tuple[Optional[str], float]] = {}

    def add_docs(self, docs: Sequence[str]) -> None:
        for d in docs:
            pid = d.split("\n")[0].strip()
            if pid and pid not in self._seen_pid:
                self._seen_pid.add(pid)
                self.paras.append((pid, d))

    def state_of(self, spec: SlotSpec) -> str:
        if spec.index in self.resolved:
            return "filled"
        if any(r not in self.resolved for r in spec.refs):
            return "blocked"
        return "open"

    def line(self) -> str:
        marks = []
        for s in self.slots:
            st = self.state_of(s)
            if st == "filled":
                marks.append(f"{s.index}✓({self.resolved[s.index].value[:24]})")
            elif st == "open":
                marks.append(f"{s.index}…")
            else:
                marks.append(f"{s.index}✗blocked")
        return "[slots: " + " ".join(marks) + "]"


def wave_update(trackers: Sequence[JudgeTracker],
                judge_batch: JudgeBatchFn,
                threshold: float = 0.5) -> None:
    """跨追踪器按深度波次判定;每深度一次批调用。"""
    if not trackers:
        return
    max_depth = max(len(t.slots) for t in trackers)
    for depth in range(max_depth):
        pairs: List[Tuple[str, str]] = []
        owners: List[Tuple[JudgeTracker, int, str]] = []
        for t in trackers:
            if depth >= len(t.slots):
                continue
            spec = t.slots[depth]
            if spec.index in t.resolved or t.state_of(spec) == "blocked":
                continue
            prop = _inst(spec.question, t.resolved)
            for pid, text in t.paras:
                if (prop, pid) not in t.cache:
                    pairs.append((prop, text))
                    owners.append((t, prop, pid))
        if pairs:
            results = judge_batch(pairs)
            for (t, prop, pid), r in zip(owners, results):
                t.cache[(prop, pid)] = (r.get("val"), float(r.get("score", 0.0)))
        # 结算本深度:每追踪器取缓存中过阈最优
        for t in trackers:
            if depth >= len(t.slots):
                continue
            spec = t.slots[depth]
            if spec.index in t.resolved or t.state_of(spec) == "blocked":
                continue
            prop = _inst(spec.question, t.resolved)
            best = None
            for pid, _text in t.paras:
                v = t.cache.get((prop, pid))
                if v and v[0] is not None and v[1] >= threshold:
                    if best is None or v[1] > best[1]:
                        best = (v[0], v[1], pid)
            if best:
                t.resolved[spec.index] = Binding(best[0], "", best[2], best[1])


def make_belief_update_fn(judge_url: str = "", judge_batch: JudgeBatchFn = None):
    """episode.run_episodes_batched(belief_update_fn=...) 的工厂。

    judge_batch 可注入(CPU 测试用);缺省用 judge_url 的 HTTP 批接口。
    约定:pending 状态携带 _pending_plan/_pending_docs;回调负责建追踪器、
    波次判定,并把注入行写进 st["belief_line"](空串 = 不注入)。
    """
    from .slot_schema import PlanSchemaError, parse_canonical_plan

    if judge_batch is None:
        import requests

        def judge_batch(pairs, _chunk=512):
            out = []
            for s0 in range(0, len(pairs), _chunk):
                r = requests.post(judge_url,
                                  json={"pairs": pairs[s0:s0 + _chunk]},
                                  timeout=600)
                r.raise_for_status()
                out.extend(r.json())
            return out

    def update(pending) -> None:
        live = []
        for st in pending:
            st["belief_line"] = ""
            if st.get("belief_dead"):
                continue
            if not isinstance(st.get("belief"), JudgeTracker):
                try:
                    st["belief"] = JudgeTracker(
                        parse_canonical_plan(st.get("_pending_plan") or ""))
                except PlanSchemaError:
                    st["belief_dead"] = True
                    continue
            st["belief"].add_docs(st.get("_pending_docs") or [])
            live.append(st)
        wave_update([st["belief"] for st in live], judge_batch)
        for st in live:
            st["belief_line"] = st["belief"].line()

    return update
