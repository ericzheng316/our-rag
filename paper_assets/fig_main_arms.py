"""Fig: three arms in the training environment (closed-pool MuSiQue dev, seed 1).
A: EM by hop at step 100 (query-level credit / trajectory-level coverage / outcome-only),
   annotated with query-level minus outcome-only.
B: full-dev EM along training (steps 20/50/80/100).
Data: PAPER_HANDOFF §26-§29 (qcredit_evals, grounding_evals re-evaluation, same-question pairing).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from style import OUT, PAL, LINE, INK, TEXT, plt

ARMS = [("Outcome-only", PAL["base"]), ("Trajectory-level coverage", PAL["traj"]), ("Query-level credit", PAL["ours"])]
INKS = {"Outcome-only": INK["base"], "Trajectory-level coverage": INK["traj"], "Query-level credit": INK["ours"]}
LINES = {"Outcome-only": LINE["base"], "Trajectory-level coverage": LINE["traj"], "Query-level credit": LINE["ours"]}
HATCH = {"Outcome-only": "///", "Trajectory-level coverage": "", "Query-level credit": ""}
LS = {"Outcome-only": ":", "Trajectory-level coverage": "--", "Query-level credit": "-"}
MARKERS = {"Outcome-only": "s", "Trajectory-level coverage": "^", "Query-level credit": "o"}
hops = ["2-hop", "3-hop", "4-hop", "All"]
em = {  # step 100
    "Outcome-only":              [55.51, 43.42, 31.11, 47.62],
    "Trajectory-level coverage": [53.67, 43.16, 36.30, 47.46],
    "Query-level credit":        [57.99, 46.71, 40.00, 51.43],
}
delta = [2.5, 3.3, 8.9, 3.8]; zs = [2.32, 2.25, 4.08, 4.72]
steps = [20, 50, 80, 100]
curve = {
    "Outcome-only":              [44.52, 46.63, 49.03, 47.62],
    "Trajectory-level coverage": [44.15, 48.70, 47.54, 47.46],
    "Query-level credit":        [46.92, 51.68, 52.50, 51.43],
}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
fig.suptitle("Closed-pool MuSiQue dev, exact match", x=0.05, ha="left", fontsize=9)
x = np.arange(len(hops)); w = 0.26
for k, (name, col) in enumerate(ARMS):
    ax.bar(x + (k - 1) * w, em[name], w, color=col, edgecolor=LINES[name], linewidth=0.5, hatch=HATCH[name], label=name, zorder=3)
for i in range(len(hops)):
    top = max(em[n][i] for n, _ in ARMS)
    ax.text(x[i], top + 1.2, f"+{delta[i]:.1f}\n$z$={zs[i]:.1f}", ha="center", va="bottom", fontsize=7, color=TEXT)
ax.set_xticks(x); ax.set_xticklabels(hops); ax.set_ylabel("Exact match (%)")
ax.set_ylim(25, 68)
ax.set_title("A. Accuracy by depth (step 100)", loc="left", fontsize=9)

for name, col in ARMS:
    bx.plot(steps, curve[name], LS[name], color=LINES[name], lw=1.8,
            marker=MARKERS[name], ms=4.5, mfc=col, mec=LINES[name], mew=0.8, label=name)
bx.set_xticks(steps); bx.set_xlabel("training step"); bx.set_ylabel("Exact match (%)")
bx.set_ylim(43, 54)
bx.set_title("B. Full-dev EM along training", loc="left", fontsize=9)
offs = {"Outcome-only": 4, "Trajectory-level coverage": -4, "Query-level credit": 0}
for name, col in ARMS:
    bx.annotate(f"{curve[name][-1]:.1f}", (steps[-1], curve[name][-1]), xytext=(5, offs[name]), textcoords="offset points", fontsize=7, color=INKS[name], va="center")
bx.set_xlim(15, 112)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), handlelength=1.2)
fig.tight_layout(w_pad=1.5, rect=(0, 0.08, 1, 0.97))
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_main_arms.{ext}")
print("saved", OUT + "/fig_main_arms.{pdf,png}")
