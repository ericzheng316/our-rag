"""tie-aware AUC 补算(§7.2 用):v2 分析的 AUC 列并列处理有偏,此处修正。

两个量:
1) 跨题同 hop AUC(cov→EM):并列取平均秩(Mann-Whitney 标准做法);
2) 混合组内 AUC:每个 mixed 组内,(对,错) rollout 两两配对,
   对者覆盖更高计 1、并列计 0.5,汇总所有组内配对——回答
   "组内按覆盖排序能否挑出对的那条",这才是 J7 重排失败的相关量。
数据与覆盖抽取复用 analysis_group_signal_v2 的口径。
"""
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analysis_group_signal_v2 import D, POOL, retrieved_titles


def tie_aware_auc(pairs):
    """pairs: [(score, label)] → AUC, 并列平均秩。"""
    pos = sum(l for _, l in pairs)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    by_score = defaultdict(list)
    for s, l in pairs:
        by_score[s].append(l)
    rank, rank_sum = 0, 0.0
    for s in sorted(by_score):
        labels = by_score[s]
        avg_rank = rank + (len(labels) + 1) / 2
        rank_sum += avg_rank * sum(labels)
        rank += len(labels)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def load(name):
    golds = {}
    for line in open(POOL):
        r = json.loads(line)
        golds[r["id"]] = set(t.casefold() for t in r["gold_titles"])
    ks = sorted(int(f.rsplit("_k", 1)[1].split(".")[0])
                for f in glob.glob(f"{D}/hops_{name}_k*.jsonl")
                if Path(f).stat().st_size > 0)
    g = {}
    for k in ks:
        cov = {}
        for line in open(f"{D}/msgs_{name}_k{k}.jsonl"):
            r = json.loads(line)
            gt = golds[r["id"]]
            cov[r["id"]] = len(retrieved_titles(r["messages"]) & gt) / len(gt)
        for line in open(f"{D}/hops_{name}_k{k}.jsonl"):
            r = json.loads(line)
            e = g.setdefault(r["id"], {"hop": r["hop"], "em": [], "cov": []})
            e["em"].append(int(r["alias_em"]))
            e["cov"].append(cov.get(r["id"], 0.0))
    K = len(ks)
    return {i: e for i, e in g.items() if len(e["em"]) == K}, K


def main():
    out = []
    for name in ("sftv22", "run7s100"):
        g, K = load(name)
        out.append(f"\n# {name} (K={K}, 完整组 {len(g)})")
        out.append("| hop | 跨题AUC(tie-aware) | 混合组数 | 组内配对数 | 组内AUC |")
        out.append("|---|---|---|---|---|")
        for hop in (2, 3, 4):
            gs = [e for e in g.values() if e["hop"] == hop]
            cross = tie_aware_auc([(v, m) for e in gs
                                   for v, m in zip(e["cov"], e["em"])])
            wins = ties = tot = 0
            nmix = 0
            for e in gs:
                if not 0 < sum(e["em"]) < K:
                    continue
                nmix += 1
                for ci, mi in zip(e["cov"], e["em"]):
                    if not mi:
                        continue
                    for cj, mj in zip(e["cov"], e["em"]):
                        if mj:
                            continue
                        tot += 1
                        if ci > cj:
                            wins += 1
                        elif ci == cj:
                            ties += 1
            within = (wins + 0.5 * ties) / tot if tot else float("nan")
            out.append(f"| {hop} | {cross:.4f} | {nmix} | {tot} | {within:.4f} |")
    text = "\n".join(out)
    print(text)
    Path("/home/boyuz5/our-rag/paper_assets/out/analysis_within_auc.md").write_text(
        "# tie-aware AUC 补算(§7.2)\n" + text + "\n")


if __name__ == "__main__":
    main()
