"""Fig: share of K=8 groups that provide any within-group ordering, at the SFT
initialization (799 closed-pool groups), before and after adding coverage.
Stacked bars: mixed-outcome groups (informative under the answer reward alone)
plus all-wrong groups whose rollouts differ in coverage (informative only through
coverage). Annotation: expansion factor. Data: PAPER_HANDOFF §22 (Table tab:repair).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from style import OUT, PAL, LINE, TEXT, MUTED, plt

hops = ["2-hop", "3-hop", "4-hop"]
outcome_only = [29.4, 44.1, 38.2]          # mixed-outcome groups
added = [19.4, 36.2, 50.5]                 # all-wrong groups with coverage variation
total = [48.8, 80.3, 88.7]
expansion = [1.66, 1.82, 2.32]
tied_var = [31.0, 70.4, 84.3]              # tied groups with coverage variation (text)

fig, ax = plt.subplots(figsize=(3.3, 2.3))
x = np.arange(len(hops)); w = 0.55
ax.bar(x, outcome_only, w, color=PAL["base"], edgecolor=LINE["base"], linewidth=0.5, hatch="///", label="Mixed outcomes (answer reward orders)", zorder=3)
ax.bar(x, added, w, bottom=outcome_only, color=PAL["evidence"], edgecolor=LINE["evidence"], linewidth=0.5, label="All wrong, coverage differs (coverage orders)", zorder=3)
for xi, o, t, e in zip(x, outcome_only, total, expansion):
    ax.text(xi, t + 1.5, f"{t:.1f}%\n$\\times${e:.2f}", ha="center", va="bottom", fontsize=7, color=TEXT)
    ax.text(xi, o / 2, f"{o:.1f}", ha="center", va="center", fontsize=7, color=TEXT)
ax.set_xticks(x); ax.set_xticklabels(hops)
ax.set_ylabel("Groups with a within-group ordering (%)")
ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
ax.legend(frameon=False, loc="upper left", fontsize=6.8, handlelength=1.2, bbox_to_anchor=(0, 1.02))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_repair.{ext}")
print("saved fig_repair")
