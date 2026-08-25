"""消融表(LaTeX):塑形 / SFT 初始化,配对 McNemar 全部从预测文件重算。

行定义:
  Full(覆盖塑形, run7 ep100)    —— 主线单种子,与消融同窗口可配对
  w/o coverage shaping           —— outcome-only 对照(同超参同窗口)
  canonical-plan SFT init(v3-RL)—— RL 终点对 SFT 起点不敏感的证据
三种子均值(47.5±1.6)在主表,不进本表。
"""
import glob
import json
import math
from pathlib import Path

OUT = "/home/boyuz5/our-rag/paper_assets/out"
L = "/home/boyuz5/acec/logs"

FULL = f"{L}/eval_r7s100_fulldev_20260811T142930Z/hops_c*.jsonl"
ROWS = [  # (标签, glob;None = Full 自身行)
    ("ACEC (coverage shaping)", None),
    (r"\;\;w/o coverage shaping", f"{L}/eval_noshape_fulldev_20260816T014714Z/hops_c*.jsonl"),
    (r"\;\;canonical-plan SFT init (v3-RL)", f"{L}/eval_v3rl_fulldev_20260819T041512Z/hops_c*.jsonl"),
]


def load(pat):
    d = {}
    n_files = 0
    for f in glob.glob(pat):
        n_files += 1
        for line in open(f):
            r = json.loads(line)
            d[r["id"]] = (int(r["alias_em"]), float(r["f1"]), r["hop"])
    assert n_files, f"无文件: {pat}"
    return d


def summarize(d):
    n = len(d)
    em = 100 * sum(v[0] for v in d.values()) / n
    f1 = 100 * sum(v[1] for v in d.values()) / n
    h4 = [v[0] for v in d.values() if v[2] == 4]
    return em, f1, 100 * sum(h4) / len(h4), n


def mcnemar_z(full, other):
    """配对不一致计数:b = full对/other错, c = full错/other对。"""
    common = set(full) & set(other)
    b = sum(1 for q in common if full[q][0] and not other[q][0])
    c = sum(1 for q in common if not full[q][0] and other[q][0])
    z = (b - c) / math.sqrt(b + c) if b + c else 0.0
    return z, b, c, len(common)


def main():
    full = load(FULL)
    em0, f10, h40, n0 = summarize(full)
    lines = [
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Variant & EM & F1 & 4-hop EM & $\Delta$EM & McNemar $z$ \\",
        r"\midrule",
        f"ACEC (coverage shaping) & {em0:.2f} & {f10:.1f} & {h40:.1f} & -- & -- \\\\"]
    for label, pat in ROWS[1:]:
        d = load(pat)
        em, f1, h4, n = summarize(d)
        z, b, c, nc = mcnemar_z(full, d)
        assert nc == n0 == n, f"配对不齐: {label} common={nc}"
        lines.append(
            f"{label} & {em:.2f} & {f1:.1f} & {h4:.1f} & "
            f"{em - em0:+.2f} & {z:+.2f} ({b}:{c}) \\\\")
        print(f"[check] {label}: EM {em:.2f} Δ{em - em0:+.2f} z={z:+.2f} b:c={b}:{c}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(f"{OUT}/tab_ablation.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
