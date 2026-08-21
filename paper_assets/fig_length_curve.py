"""图:训练长度 vs 全量 dev EM(峰值 + 种子带 + 退化区)。数字来自
PAPER_HANDOFF.md 定稿值(此处内联为常量,来源注明)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from style import C, OUT, plt

eps =    [0,    100,   120,   140,   160,   180]
em =     [34.5, 48.20, 50.27, 47.58, 48.70, 48.53]   # SFT v3 起点用 v2.2 线时为 33.0
seeds_ep100 = [48.20, 48.53, 45.68]

fig, ax = plt.subplots(figsize=(4.2, 2.9))
ax.plot(eps, em, "-o", color=C["main"], ms=4, label="full-dev EM")
lo, hi = min(seeds_ep100), max(seeds_ep100)
ax.fill_between([95, 105], lo, hi, color=C["main"], alpha=0.18,
                label="3-seed range @ep100")
ax.axvspan(180, 200, color=C["bad"], alpha=0.10)
ax.text(190, 36, "syntax\ndegeneration", ha="center", fontsize=7,
        color=C["bad"])
ax.annotate("peak (McNemar z=2.99)", xy=(120, 50.27), xytext=(128, 51.3),
            fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.7))
ax.set_xlabel("RL training episodes")
ax.set_ylabel("MuSiQue full-dev EM (%)")
ax.set_xlim(-5, 205); ax.set_ylim(33, 53)
ax.legend(frameon=False, loc="lower right")
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_length_curve.{ext}")
print("✅ fig_length_curve")
