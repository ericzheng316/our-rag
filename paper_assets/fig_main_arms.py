"""Fig: query-level credit vs outcome-only training (closed-pool MuSiQue dev).
A: EM by hop at step 100, annotated with query-level minus outcome-only (question-level
   paired McNemar z, seed 1 vs seed 1). Light band on a bar = min..max over available seeds.
B: full-dev EM along training (steps 20/50/80/100): line = mean over available seeds,
   light band = min..max over seeds (single-seed steps have no band).
Data: PAPER_HANDOFF §26-§29 (qcredit_evals, grounding_evals re-evaluation), gate50_evals
(outcome-only seeds 1-3 at step 50; seed-1 value 46.50 there vs 46.63 in the qcredit re-eval, greedy drift),
grounding_evals (outcome-only seeds 1-3 at step 100).
Add seeds by appending to the lists below.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from style import OUT, PAL, LINE, INK, TEXT, plt

VARIANTS = [("Outcome-only", PAL["base"]), ("Query-level credit", PAL["ours"])]
INKS = {"Outcome-only": INK["base"], "Query-level credit": INK["ours"]}
LINES = {"Outcome-only": LINE["base"], "Query-level credit": LINE["ours"]}
HATCH = {"Outcome-only": "///", "Query-level credit": ""}
LS = {"Outcome-only": ":", "Query-level credit": "-"}
MARKERS = {"Outcome-only": "s", "Query-level credit": "o"}
BAND = 0.35   # alpha of the min..max band

hops = ["2-hop", "3-hop", "4-hop", "All"]
# step 100, per hop, one value per seed (seed 1 first; paired z below uses seed 1 vs seed 1)
em_seeds = {
    "Outcome-only":       [[55.51, 53.59, 55.19], [43.42, 43.82, 42.24], [31.11, 35.56, 34.32], [47.62, 47.50, 47.62]],
    "Query-level credit": [[57.99], [46.71], [40.00], [51.43]],
}
delta = [2.5, 3.3, 8.9, 3.8]; zs = [2.32, 2.25, 4.08, 4.72]
steps = [20, 50, 80, 100]
curve_seeds = {
    "Outcome-only":       {20: [44.52], 50: [46.50, 47.50, 46.63], 80: [49.03], 100: [47.62, 47.50, 47.62]},
    "Query-level credit": {20: [46.92], 50: [51.68], 80: [52.50], 100: [51.43]},
}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
fig.suptitle("Closed-pool MuSiQue dev, exact match", x=0.05, ha="left", fontsize=9)
x = np.arange(len(hops)); w = 0.36
for k, (name, col) in enumerate(VARIANTS):
    vals = [v[0] for v in em_seeds[name]]            # seed-1 bar (matches the paired z)
    xs = x + (k - 0.5) * w
    ax.bar(xs, vals, w, color=col, edgecolor=LINES[name], linewidth=0.5, hatch=HATCH[name], label=name, zorder=3)
    for xi, seeds in zip(xs, em_seeds[name]):
        if len(seeds) > 1:                              # light band: min..max over seeds
            ax.bar(xi, max(seeds) - min(seeds), w * 0.9, bottom=min(seeds), color=LINES[name], alpha=BAND, linewidth=0, zorder=4)
for i in range(len(hops)):
    top = max(max(em_seeds[n][i]) for n, _ in VARIANTS)
    ax.text(x[i], top + 1.2, f"+{delta[i]:.1f}\n$z$={zs[i]:.1f}", ha="center", va="bottom", fontsize=7, color=TEXT)
ax.set_xticks(x); ax.set_xticklabels(hops); ax.set_ylabel("Exact match (%)")
ax.set_ylim(25, 68)
ax.set_title("A. Accuracy by depth (step 100)", loc="left", fontsize=9)

for name, col in VARIANTS:
    mean = [np.mean(curve_seeds[name][s]) for s in steps]
    lo = [min(curve_seeds[name][s]) for s in steps]; hi = [max(curve_seeds[name][s]) for s in steps]
    bx.fill_between(steps, lo, hi, color=LINES[name], alpha=BAND * 0.6, linewidth=0, zorder=2)
    bx.plot(steps, mean, LS[name], color=LINES[name], lw=1.8, marker=MARKERS[name], ms=4.5,
            mfc=col, mec=LINES[name], mew=0.8, label=name, zorder=3)
    bx.annotate(f"{mean[-1]:.1f}", (steps[-1], mean[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7, color=INKS[name], va="center")
bx.set_xticks(steps); bx.set_xlabel("training step"); bx.set_ylabel("Exact match (%)")
bx.set_ylim(43, 54); bx.set_xlim(15, 112)
bx.set_title("B. Full-dev EM along training", loc="left", fontsize=9)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0), handlelength=1.2)
fig.tight_layout(w_pad=1.5, rect=(0, 0.08, 1, 0.97))
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_main_arms.{ext}")
print("saved", OUT + "/fig_main_arms.{pdf,png}")
