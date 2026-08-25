"""迁移表(LaTeX):MuSiQue 训练的策略零样本上 HotpotQA / 2Wiki,
SFT v2.2 vs 旗舰 ep120 配对重算(与 fig_transfer 同数据源)。"""
import glob
import json
import math
from pathlib import Path

OUT = "/home/boyuz5/our-rag/paper_assets/out"
L = "/home/boyuz5/acec/logs"

SFT = f"{L}/eval_transfer_sftv22_*/hops_*.jsonl"
RL = f"{L}/eval_transfer_ep120_*/hops_*.jsonl"
HOTPOT_META = "/home/boyuz5/acec/flashrag_datasets/hotpotqa/dev_distractor.jsonl"


def load(pat):
    d = {}
    n_files = 0
    for f in glob.glob(pat):
        n_files += 1
        for line in open(f):
            r = json.loads(line)
            d[r["id"]] = (int(r["alias_em"]), float(r["f1"]))
    assert n_files, f"无文件: {pat}"
    return d


def main():
    qtype = {}
    for line in open(HOTPOT_META):
        r = json.loads(line)
        qtype[f"2hop__hp_{r['id']}"] = r["metadata"]["type"]

    sft, rl = load(SFT), load(RL)
    common = sorted(set(sft) & set(rl))

    groups = [
        ("HotpotQA (all)", [q for q in common if q in qtype]),
        (r"\;\;bridge", [q for q in common if qtype.get(q) == "bridge"]),
        (r"\;\;comparison", [q for q in common if qtype.get(q) == "comparison"]),
        ("2WikiMultihopQA", [q for q in common
                             if "__" in q and q.split("__")[1].startswith("2w")]),
    ]
    lines = [
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Dataset & $n$ & SFT EM & SFT F1 & +RL EM & +RL F1 & $\Delta$EM \\",
        r"\midrule"]
    for label, qs in groups:
        assert qs, f"空组: {label}"
        n = len(qs)
        s_em = 100 * sum(sft[q][0] for q in qs) / n
        s_f1 = 100 * sum(sft[q][1] for q in qs) / n
        r_em = 100 * sum(rl[q][0] for q in qs) / n
        r_f1 = 100 * sum(rl[q][1] for q in qs) / n
        b = sum(1 for q in qs if rl[q][0] and not sft[q][0])
        c = sum(1 for q in qs if not rl[q][0] and sft[q][0])
        z = (b - c) / math.sqrt(b + c) if b + c else 0.0
        lines.append(
            f"{label} & {n} & {s_em:.1f} & {s_f1:.1f} & "
            f"{r_em:.1f} & {r_f1:.1f} & {r_em - s_em:+.1f} \\\\")
        print(f"[check] {label}: n={n} SFT {s_em:.1f} → RL {r_em:.1f} "
              f"(Δ{r_em - s_em:+.1f}, z={z:+.2f}, {b}:{c})")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(f"{OUT}/tab_transfer.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
