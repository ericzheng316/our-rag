"""图:技能/套路分解——增量按检索难度桶(全部从预测文件重算)。"""
import glob
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from style import OUT, plt

L = "/home/boyuz5/acec/logs"
BUCKET = json.load(open("/scratch/boyuz5/acec/musique_openbook/difficulty_buckets.json"))


def load(pat):
    d = {}
    for f in glob.glob(pat):
        for line in open(f):
            r = json.loads(line)
            d[r["id"]] = int(r["alias_em"])
    return d


def delta_by_bucket(a, b):
    out = []
    for bk in ("E", "M", "H"):
        qs = [q for q in BUCKET if BUCKET[q] == bk and q in a and q in b]
        out.append(100 * (sum(b[q] for q in qs) - sum(a[q] for q in qs)) / len(qs))
    return out

sft_ob   = load(f"{L}/eval_sft22_ob_*/hops_*.jsonl")
rl120_ob = load(f"{L}/eval_ob_ep120_*/hops_*.jsonl")
rl100_ob = load(f"{L}/eval_ob_r7s100_fulldev_20260811T142930Z/hops_*.jsonl") or \
           load(f"{L}/eval_ob_r7s100_*/hops_*.jsonl")
rl100_cl = load(f"{L}/eval_r7s100_fulldev_20260811T142930Z/hops_c*.jsonl")
rl120_cl = load(f"{L}/eval_sweep_20260814T104838Z/hops_s20_c*.jsonl")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7))
x = range(3); lab = ["easy", "medium", "hard"]
d1 = delta_by_bucket(sft_ob, rl120_ob)
a1.bar(x, d1, color="#1a63a8", width=0.55)
a1.set_title("SFT → RL gain (open-book)", fontsize=9)
a1.axhline(0, color="k", lw=0.6)
d_cl = delta_by_bucket(rl100_cl, rl120_cl)
d_ob = delta_by_bucket(rl100_ob, rl120_ob)
w = 0.36
a2.bar([i - w / 2 for i in x], d_cl, w, color="#e08a00", label="closed pool")
a2.bar([i + w / 2 for i in x], d_ob, w, color="#2e7d32", label="open-book")
a2.set_title("late-stage peak gain (ep100→120)", fontsize=9)
a2.axhline(0, color="k", lw=0.6)
a2.legend(frameon=False, fontsize=7)
for ax, d in ((a1, [d1]), (a2, [d_cl, d_ob])):
    ax.set_xticks(list(x)); ax.set_xticklabels(lab)
    ax.set_ylabel("Δ EM (pp)")
    ax.set_xlabel("retrieval difficulty of question")
fig.savefig(f"{OUT}/fig_buckets.png"); fig.savefig(f"{OUT}/fig_buckets.pdf")
print("✅ fig_buckets", [round(v,1) for v in d1], [round(v,1) for v in d_cl], [round(v,1) for v in d_ob])
