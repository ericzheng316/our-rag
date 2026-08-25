"""六方 wiki18 同台基线表(LaTeX ×2):MuSiQue full dev + HotpotQA 2k 子集。

统一条件(PAPER_HANDOFF §10/10b):同一 wiki18 语料 + e5-base-v2 fp32 Flat
检索服务,temp 0,各系统官方协议逐字复现;度量 alias_em。
配对 McNemar(ours vs 各家,同题集)打印在 check 输出里,供 caption 引用。
"""
import glob
import json
import math
import re
from pathlib import Path

OUT = "/home/boyuz5/our-rag/paper_assets/out"
B = "/home/boyuz5/acec/logs/baseline_evals"
HOTPOT_META = "/home/boyuz5/acec/flashrag_datasets/hotpotqa/dev_distractor.jsonl"

SYSTEMS = [  # (文件前缀, 表内标签)
    ("ours_ep120_wiki18", r"\textbf{ACEC ep120 (ours)}"),
    ("research", "ReSearch"),
    ("searchr1", "Search-R1 (PPO)"),
    ("stepsearch", "StepSearch"),
    ("r1searcher", "R1-Searcher"),
    ("zerosearch", "ZeroSearch"),
]


def load(path):
    d = {}
    for f in glob.glob(path):
        for line in open(f):
            r = json.loads(line)
            grp = r.get("group") or r.get("family") or ""
            # ours 的 family 是细分族(3hop1/4hop2…),归一到 Nhop
            m = re.match(r"(\dhop)", grp)
            if m:
                grp = m.group(1)
            d[r["id"]] = (int(r["alias_em"]), float(r["f1"]),
                          grp, int(r.get("n_search", 0)))
    assert d, f"无文件: {path}"
    return d


def mcnemar(a, b):
    common = set(a) & set(b)
    x = sum(1 for q in common if a[q][0] and not b[q][0])
    y = sum(1 for q in common if not a[q][0] and b[q][0])
    z = (x - y) / math.sqrt(x + y) if x + y else 0.0
    return z, x, y, len(common)


def emit(dataset, suffix, group_cols, out_name):
    data = {}
    for pre, label in SYSTEMS:
        # ours 的 musique 文件名不带 _dev 后缀
        sfx = suffix.removesuffix("_dev") if pre.startswith("ours") else suffix
        data[label] = load(f"{B}/{pre}_{sfx}.jsonl")
    # ours 的 hotpot 文件没有 bridge/comparison 分组,从官方元数据回填
    if dataset == "hotpot":
        qtype = {}
        for line in open(HOTPOT_META):
            r = json.loads(line)
            qtype[f"2hop__hp_{r['id']}"] = r["metadata"]["type"]
        ours = data[SYSTEMS[0][1]]
        data[SYSTEMS[0][1]] = {q: (v[0], v[1], qtype[q], v[3])
                               for q, v in ours.items()}

    head = " & ".join(["System", "EM", "F1"] + group_cols + ["Avg.\\ searches"])
    lines = [r"\begin{tabular}{l" + "c" * (3 + len(group_cols)) + "}",
             r"\toprule", head + r" \\", r"\midrule"]
    ours = data[SYSTEMS[0][1]]
    for label, d in data.items():
        n = len(d)
        em = 100 * sum(v[0] for v in d.values()) / n
        f1 = 100 * sum(v[1] for v in d.values()) / n
        srch = sum(v[3] for v in d.values()) / n
        cells = [label, f"{em:.2f}", f"{f1:.2f}"]
        for g in group_cols:
            key = g.lower().replace("-", "")
            sub = [v[0] for v in d.values() if v[2] == key]
            cells.append(f"{100 * sum(sub) / len(sub):.2f}" if sub else "--")
        cells.append(f"{srch:.2f}")
        lines.append(" & ".join(cells) + r" \\")
        if label != SYSTEMS[0][1]:
            z, x, y, nc = mcnemar(ours, d)
            print(f"[check] {dataset}: ours vs {label}: "
                  f"z={z:+.2f} ({x}:{y}, n={nc})")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(f"{OUT}/{out_name}").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()


def main():
    emit("musique", "musique_dev", ["2hop", "3hop", "4hop"],
         "tab_baselines_musique.tex")
    emit("hotpot", "hotpot2k", ["Bridge", "Comparison"],
         "tab_baselines_hotpot.tex")


if __name__ == "__main__":
    main()
