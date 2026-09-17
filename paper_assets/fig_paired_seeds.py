"""Fig: trajectory-level coverage minus outcome-only RL over the four matched pairs
(closed-pool MuSiQue dev): overall and 4-hop EM differences per pair, plus pooled
estimates; annotations give the question-level paired McNemar z.
Data: PAPER_HANDOFF §14 + paper-window recomputation 2026-09-03 (Table tab:main-pair).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from style import OUT, PAL, INK, plt

labels = ["Seed 1", "Seed 2", "Seed 3", "Default", "Pooled\n(1–3)", "Pooled\n(all)"]
d_all = [-0.25, 0.29, 1.82, 1.65, 0.62, 0.88]; z_all = [-0.31, 0.39, 2.48, 2.37, 1.42, 2.36]
d_4h  = [4.20, -2.72, 5.68, 7.16, 2.39, 3.58]; z_4h = [1.99, -1.39, 2.81, 3.98, 2.04, 3.62]

fig, ax = plt.subplots(figsize=(3.4, 2.2))
x = np.arange(len(labels)); w = 0.36
b1 = ax.bar(x - w/2, d_all, w, color=PAL["base"], edgecolor=INK["base"], linewidth=0.8, hatch="///", label="Overall EM", zorder=3)
b2 = ax.bar(x + w/2, d_4h, w, color=PAL["traj"], edgecolor=INK["traj"], linewidth=0.8, label="4-hop EM", zorder=3)
ax.axhline(0, color=INK["ref"], lw=0.8, zorder=2)
ax.axvline(3.5, color=INK["ref"], lw=0.6, ls=":", zorder=2)
for xi, d, z in zip(x + w/2, d_4h, z_4h):
    ax.text(xi, d + (0.35 if d >= 0 else -0.35), f"$z$={z:.1f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=6.5, color=INK["traj"])
for xi, d, z in zip(x - w/2, d_all, z_all):
    ax.text(xi, d + (0.35 if d >= 0 else -0.35), f"{z:.1f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=6.5, color=INK["base"])
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("$\\Delta$EM: traj.-level $-$ outcome-only")
ax.set_ylim(-4.5, 9.5); ax.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
ax.legend(frameon=False, loc="upper left", fontsize=7, handlelength=1.2)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_paired_seeds.{ext}")
print("saved")
