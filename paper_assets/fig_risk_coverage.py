"""图:judge 选择性预测曲线(风险-覆盖,从 phase1a 数据重算)。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from style import OUT, plt

rows = [json.loads(l) for l in
        open("/scratch/boyuz5/acec/judge/phase1a_raw.jsonl")]
pts = sorted(rows, key=lambda r: -(r["coverage"] + 1e-3 * r["joint"]))
n = len(pts)
xs, ys = [], []
for k in range(10, n + 1):
    xs.append(100 * k / n)
    ys.append(100 * sum(r["em"] for r in pts[:k]) / k)
full = [r for r in rows if r["coverage"] >= 0.999]
fx = 100 * len(full) / n
fy = 100 * sum(r["em"] for r in full) / max(len(full), 1)
base = 100 * sum(r["em"] for r in rows) / n

fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.plot(xs, ys, "-", color="#1a63a8", label="coverage-ranked")
ax.axhline(base, color="#808080", ls=":", lw=1, label=f"answer all ({base:.0f}%)")
ax.plot([fx], [fy], "o", color="#c23b22", ms=5)
ax.annotate(f"full DAG resolution\n({fx:.1f}%, {fy:.0f}%)", xy=(fx, fy),
            xytext=(fx + 12, fy - 6), fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.7))
ax.set_xlabel("answering rate (%)"); ax.set_ylabel("precision (%)")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.savefig(f"{OUT}/fig_risk_coverage.png")
fig.savefig(f"{OUT}/fig_risk_coverage.pdf")
print("✅ fig_risk_coverage")
