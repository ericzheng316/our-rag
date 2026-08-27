"""组内信号分析:outcome 打平时覆盖信号的可用性(条件①的定量机制证据)。

数据:rerank_v3(v3-RL ep100, temp0.7, 240 题 × 4 采样, 闭池)。
Caveat(报告时必须带):组大小 4(训练为 8);策略为 RL 后而非 SFT 起点——
tied 比例相对训练初期偏保守(RL 后 all-correct 更多)。

四问:
  A. 各 hop 的 all-correct / all-wrong / mixed 组比例(tied = 前两者,
     outcome 在 LOO 基线下梯度为零);
  B. outcome 打平的组内,coverage 的组内标准差(塑形能否在 outcome 沉默处供差);
  C. coverage 是否预测答对:混合组内 正确/错误 rollout 的配对覆盖差 + 全体 AUC;
  D. tie-break 信号强度随 hop 的变化。
"""
import glob
import json
import re
import statistics
from pathlib import Path

D = "/home/boyuz5/acec/logs/rerank_v3_20260817T040016Z"
POOL = "/home/boyuz5/acec/musique/dev_answerable_pool.jsonl"
OUT = Path("/home/boyuz5/our-rag/paper_assets/out/analysis_group_signal.md")

_RESULT = re.compile(r"<result>\n?(.*?)\n?</result>", re.DOTALL)


def retrieved_titles(msgs):
    seen = set()
    for m in msgs:
        if m["role"] != "user":
            continue
        for block in _RESULT.findall(m["content"]):
            for para in block.split("\n\n"):
                if para.strip():
                    seen.add(para.strip().split("\n")[0].strip().strip('"').casefold())
    return seen


def main():
    golds = {}
    for line in open(POOL):
        r = json.loads(line)
        golds[r["id"]] = [t.casefold() for t in r["gold_titles"]]

    # (id) → {"hop": h, "em": [4], "cov": [4]}
    g = {}
    for i in (1, 2, 3, 4):
        cov_by_id = {}
        for line in open(f"{D}/msgs_r{i}.jsonl"):
            r = json.loads(line)
            gt = golds[r["id"]]
            hits = retrieved_titles(r["messages"]) & set(gt)
            cov_by_id[r["id"]] = len(hits) / len(gt)
        for line in open(f"{D}/hops_r{i}.jsonl"):
            r = json.loads(line)
            e = g.setdefault(r["id"], {"hop": r["hop"], "em": [], "cov": []})
            e["em"].append(int(r["alias_em"]))
            e["cov"].append(cov_by_id[r["id"]])
    n_bad = sum(1 for e in g.values() if len(e["em"]) != 4)
    assert n_bad == 0, f"{n_bad} 组不足 4 采样"

    lines = ["# 组内信号分析(rerank_v3: v3-RL ep100, n=4, temp0.7, 240 题闭池)",
             "", "Caveat: 组大小 4(训练 8);策略为 RL 后,tied 比例相对训练初期偏保守。", ""]

    def sec(t):
        lines.append(t)
        print(t)

    # ── A. 组构成 ─────────────────────────────────────────────────────
    sec("## A. 组 outcome 构成(all-correct / all-wrong / mixed)")
    sec("| hop | n组 | all-correct | all-wrong | mixed | tied合计(零梯度) |")
    sec("|---|---|---|---|---|---|")
    for hop in (2, 3, 4):
        gs = [e for e in g.values() if e["hop"] == hop]
        ac = sum(1 for e in gs if sum(e["em"]) == 4)
        aw = sum(1 for e in gs if sum(e["em"]) == 0)
        mx = len(gs) - ac - aw
        sec(f"| {hop} | {len(gs)} | {ac/len(gs):.1%} | {aw/len(gs):.1%} "
            f"| {mx/len(gs):.1%} | **{(ac+aw)/len(gs):.1%}** |")
    lines.append("")

    # ── B+D. tied 组内覆盖离散 ────────────────────────────────────────
    sec("## B+D. outcome 打平的组:组内 coverage 标准差(塑形的残余信号)")
    sec("| hop | tied组数 | cov std 均值 | std>0 的组占比 | all-wrong组 cov std |")
    sec("|---|---|---|---|---|")
    for hop in (2, 3, 4):
        tied = [e for e in g.values()
                if e["hop"] == hop and sum(e["em"]) in (0, 4)]
        aw = [e for e in tied if sum(e["em"]) == 0]
        if not tied:
            continue
        stds = [statistics.pstdev(e["cov"]) for e in tied]
        aws = [statistics.pstdev(e["cov"]) for e in aw] or [float("nan")]
        sec(f"| {hop} | {len(tied)} | {statistics.mean(stds):.4f} "
            f"| {sum(1 for s in stds if s > 0)/len(stds):.1%} "
            f"| {statistics.mean(aws):.4f} (n={len(aw)}) |")
    lines.append("")

    # ── C. coverage 预测答对 ──────────────────────────────────────────
    sec("## C. coverage → 答对的预测力")
    # 混合组内配对:正确 rollout 均值覆盖 vs 错误 rollout 均值覆盖
    sec("| hop | 混合组数 | 正确侧 cov | 错误侧 cov | 差>0 / =0 / <0 |")
    sec("|---|---|---|---|---|")
    for hop in (2, 3, 4):
        mixed = [e for e in g.values()
                 if e["hop"] == hop and 0 < sum(e["em"]) < 4]
        if not mixed:
            continue
        diffs, cs, ws = [], [], []
        for e in mixed:
            c = statistics.mean(v for v, m in zip(e["cov"], e["em"]) if m)
            w = statistics.mean(v for v, m in zip(e["cov"], e["em"]) if not m)
            cs.append(c); ws.append(w); diffs.append(c - w)
        gt = sum(1 for d in diffs if d > 1e-9)
        eq = sum(1 for d in diffs if abs(d) <= 1e-9)
        sec(f"| {hop} | {len(mixed)} | {statistics.mean(cs):.3f} "
            f"| {statistics.mean(ws):.3f} "
            f"| {gt}/{eq}/{len(diffs)-gt-eq} |")

    def auc_of(pairs):
        pairs = sorted(pairs)
        pos = sum(m for _, m in pairs)
        neg = len(pairs) - pos
        rank = sum(i + 1 for i, (_, m) in enumerate(pairs) if m)
        return (rank - pos * (pos + 1) / 2) / max(pos * neg, 1), len(pairs), pos

    lines.append("")
    sec("分 hop AUC(去 hop 混杂)+ 池化:")
    for hop in (2, 3, 4):
        a, n, p = auc_of([(v, m) for e in g.values() if e["hop"] == hop
                          for v, m in zip(e["cov"], e["em"])])
        sec(f"- {hop}hop AUC = {a:.4f} (n={n}, 正 {p})")
    a, n, p = auc_of([(v, m) for e in g.values()
                      for v, m in zip(e["cov"], e["em"])])
    sec(f"- 池化 AUC = **{a:.4f}** (n={n}, 正 {p};含 hop 混杂,偏乐观)")

    # ── 合成:有梯度可学的组占比(outcome 或 覆盖至少一路有差)──────────
    lines.append("")
    sec("## 合成指标:组内存在非零梯度信号的组占比")
    sec("| hop | 仅 outcome(=mixed) | outcome ∪ coverage | 提升 |")
    sec("|---|---|---|---|")
    for hop in (2, 3, 4):
        gs = [e for e in g.values() if e["hop"] == hop]
        mixed = sum(1 for e in gs if 0 < sum(e["em"]) < 4)
        either = sum(1 for e in gs if 0 < sum(e["em"]) < 4
                     or statistics.pstdev(e["cov"]) > 0)
        sec(f"| {hop} | {mixed/len(gs):.1%} | {either/len(gs):.1%} "
            f"| ×{either/max(mixed,1):.2f} |")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
