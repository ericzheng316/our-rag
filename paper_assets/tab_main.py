"""主结果表(LaTeX):同底座阶梯 + 主线 RL,全部从预测文件重算。"""
import glob
import json
import sys
from pathlib import Path

OUT = "/home/boyuz5/our-rag/paper_assets/out"
L = "/home/boyuz5/acec/logs"

ROWS = [  # (标签, 预测文件 glob, 口径备注)
    ("Closed-book (no retrieval)", f"{L}/eval_a_ladder_*/a1_closedbook.jsonl", "open"),
    ("Single-shot RAG (top-10)",   f"{L}/eval_a_ladder_*/a2_singleshot.jsonl", "open"),
    ("Protocol zero-shot",         f"{L}/eval_a_ladder_*/a3_zeroshot.jsonl",   "open"),
    ("SFT (canonical plans)",      f"{L}/eval_sftv3_fulldev_*/hops_c*.jsonl",  "closed"),
    ("+ RL 100 ep (mean of 3 seeds)", None, "closed"),  # 特殊行:三 run 合并
    ("+ RL 120 ep (flagship)",     f"{L}/eval_sweep_20260814T104838Z/hops_s20_c*.jsonl", "closed"),
    ("flagship, open-book",        f"{L}/eval_ob_ep120_*/hops_*.jsonl",        "open"),
]
SEED_GLOBS = [f"{L}/eval_r7s100_fulldev_20260811T142930Z/hops_c*.jsonl",
              f"{L}/eval_seed1_fulldev_*/hops_c*.jsonl",
              f"{L}/eval_seed2_fulldev_*/hops_c*.jsonl"]


def score(globpat):
    hit = {2: 0, 3: 0, 4: 0}
    tot = {2: 0, 3: 0, 4: 0}
    f1s = {2: 0.0, 3: 0.0, 4: 0.0}
    n_files = 0
    for f in glob.glob(globpat):
        n_files += 1
        for line in open(f):
            r = json.loads(line)
            h = r.get("hop", 2)
            em = int(r.get("alias_em", r.get("em", 0)))
            tot[h] = tot.get(h, 0) + 1
            hit[h] = hit.get(h, 0) + em
            f1s[h] = f1s.get(h, 0.0) + float(r.get("f1", 0.0))
    assert n_files, f"无文件: {globpat}"
    n = sum(tot.values())
    return dict(
        em=100 * sum(hit.values()) / n,
        f1=100 * sum(f1s.values()) / n,
        hops={h: 100 * hit[h] / tot[h] for h in tot if tot[h]},
        n=n)


def main():
    lines = [
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Model & EM & F1 & 2hop & 3hop & 4hop \\", r"\midrule"]
    for label, pat, regime in ROWS:
        if pat is None:
            ss = [score(g) for g in SEED_GLOBS]
            import statistics
            em_m = statistics.mean(s["em"] for s in ss)
            em_sd = statistics.stdev(s["em"] for s in ss)
            f1_m = statistics.mean(s["f1"] for s in ss)
            h = {k: statistics.mean(s["hops"][k] for s in ss) for k in (2, 3, 4)}
            lines.append(
                f"{label} & {em_m:.1f}$\\pm${em_sd:.1f} & {f1_m:.1f} & "
                f"{h[2]:.1f} & {h[3]:.1f} & {h[4]:.1f} \\\\")
            continue
        s = score(pat)
        h = s["hops"]
        lines.append(
            f"{label} & {s['em']:.1f} & {s['f1']:.1f} & "
            f"{h.get(2, 0):.1f} & {h.get(3, 0):.1f} & {h.get(4, 0):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(f"{OUT}/tab_main.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
