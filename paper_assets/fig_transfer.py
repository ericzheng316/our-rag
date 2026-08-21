"""图:零样本迁移——RL 增量按数据集/题型(SFT vs ep120 配对重算)。"""
import glob
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from style import OUT, plt

L = "/home/boyuz5/acec/logs"
qtype = {}
for line in open("/home/boyuz5/acec/flashrag_datasets/hotpotqa/dev_distractor.jsonl"):
    r = json.loads(line)
    qtype[f"2hop__hp_{r['id']}"] = r["metadata"]["type"]


def load(pat):
    d = {}
    for f in glob.glob(pat):
        for line in open(f):
            r = json.loads(line)
            d[r["id"]] = int(r["alias_em"])
    return d

sft = load(f"{L}/eval_transfer_sftv22_*/hops_*.jsonl")
rl = load(f"{L}/eval_transfer_ep120_*/hops_*.jsonl")
common = set(sft) & set(rl)

groups = [
    ("2Wiki\n(has 4-hop)", lambda q: q.split("__")[1].startswith("2w")
     if "__" in q else False),
    ("Hotpot\ncomparison", lambda q: qtype.get(q) == "comparison"),
    ("Hotpot\nbridge", lambda q: qtype.get(q) == "bridge"),
]
sft_v, rl_v, labels = [], [], []
for name, sel in groups:
    qs = [q for q in common if sel(q)]
    sft_v.append(100 * sum(sft[q] for q in qs) / len(qs))
    rl_v.append(100 * sum(rl[q] for q in qs) / len(qs))
    labels.append(f"{name}\nn={len(qs)}")

fig, ax = plt.subplots(figsize=(4.0, 2.8))
x = range(3); w = 0.36
ax.bar([i - w / 2 for i in x], sft_v, w, color="#b0b0b0", label="SFT")
ax.bar([i + w / 2 for i in x], rl_v, w, color="#1a63a8", label="+RL (zero-shot)")
for i in x:
    d = rl_v[i] - sft_v[i]
    ax.text(i + w / 2, rl_v[i] + 0.8, f"{d:+.1f}", ha="center", fontsize=7,
            color="#1a63a8" if d > 1 else "#808080")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("EM (%)"); ax.legend(frameon=False, fontsize=7.5)
fig.savefig(f"{OUT}/fig_transfer.png"); fig.savefig(f"{OUT}/fig_transfer.pdf")
print("✅ fig_transfer", [round(v,1) for v in sft_v], [round(v,1) for v in rl_v])
