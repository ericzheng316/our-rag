"""论文图统一风格(所有 fig_*.py 共用)。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C = {"main": "#1a63a8", "bad": "#c23b22", "warn": "#e08a00",
     "ok": "#2e7d32", "gray": "#8a8a8a", "purple": "#6a4fa3"}
OUT = "/home/boyuz5/our-rag/paper_assets/out"

# 论文配色 v2(PALETTE.md; 用户给定八色, 2026-09-16)。PAL 用于柱/线/底纹, INK 用于文字标注。
PAL = {"ours": "#F6D5BD", "key": "#5C84A6", "base": "#C9E1EE", "traj": "#D9D1E8",
       "evidence": "#C8DDD3", "accent": "#F2E5B8", "highlight": "#F3CED6", "ref": "#CFD5DD"}
def _darken(h, k=0.45):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(c * (1 - k)) for c in (r, g, b))
INK = {k: _darken(v) for k, v in PAL.items()}
INK["ours"] = PAL["key"]          # ours 的文字直接用牛仔蓝
