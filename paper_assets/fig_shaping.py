"""图:塑形消融——贡献沿检索难度单调(逐题配对重算)。"""
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
            d[r["id"]] = (int(r["alias_em"]), r["hop"])
    return d

ns = load(f"{L}/eval_noshape_fulldev_*/hops_c*.jsonl")
sh = load(f"{L}/eval_r7s100_fulldev_20260811T142930Z/hops_c*.jsonl")
common = set(ns) & set(sh)

groups = [("easy", lambda q: BUCKET.get(q) == "E"),
          ("medium", lambda q: BUCKET.get(q) == "M"),
          ("hard", lambda q: BUCKET.get(q) == "H"),
          ("4-hop", lambda q: ns[q][1] == 4)]
vals, ers = [], []
import math
for _, sel in groups:
    qs = [q for q in common if sel(q)]
    d = [sh[q][0] - ns[q][0] for q in qs]
    m = 100 * sum(d) / len(qs)
    se = 100 * math.sqrt(sum((x - sum(d)/len(d))**2 for x in d) / (len(d)-1)) / math.sqrt(len(qs))
    vals.append(m); ers.append(1.96 * se)

fig, ax = plt.subplots(figsize=(3.6, 2.7))
colors = ["#b0b0b0", "#e08a00", "#c23b22", "#1a63a8"]
ax.bar(range(4), vals, yerr=ers, capsize=3, width=0.55, color=colors)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels([g for g, _ in groups])
ax.set_ylabel("shaping contribution to EM (pp)")
ax.text(3, vals[3] + ers[3] + 0.4, "z=3.98", ha="center", fontsize=7)
fig.savefig(f"{OUT}/fig_shaping.png"); fig.savefig(f"{OUT}/fig_shaping.pdf")
print("✅ fig_shaping", [round(v, 1) for v in vals])
