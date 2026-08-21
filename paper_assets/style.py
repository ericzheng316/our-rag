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
