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
from style import C, OUT, plt

ARMS = [("Outcome-only", C["gray"]), ("Trajectory-level coverage", C["warn"]), ("Query-level credit", C["main"])]
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

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(hops)); w = 0.26
for k, (name, col) in enumerate(ARMS):
    ax.bar(x + (k - 1) * w, em[name], w, color=col, label=name, zorder=3)
for i in range(len(hops)):
    top = max(em[n][i] for n, _ in ARMS)
    ax.text(x[i], top + 1.2, f"+{delta[i]:.1f}\n$z$={zs[i]:.1f}", ha="center", va="bottom", fontsize=7, color=C["main"])
ax.set_xticks(x); ax.set_xticklabels(hops); ax.set_ylabel("Exact match (closed pool, step 100)")
ax.set_ylim(25, 68); ax.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
ax.set_title("A. Accuracy by depth", loc="left")

for name, col in ARMS:
    bx.plot(steps, curve[name], "-o", color=col, ms=4, label=name)
bx.set_xticks(steps); bx.set_xlabel("training step"); bx.set_ylabel("Exact match (closed pool, full dev)")
bx.set_ylim(43, 54); bx.yaxis.grid(True, lw=0.4, alpha=0.5)
bx.set_title("B. Along training", loc="left")
offs = {"Outcome-only": 4, "Trajectory-level coverage": -4, "Query-level credit": 0}
for name, col in ARMS:
    bx.annotate(f"{curve[name][-1]:.1f}", (steps[-1], curve[name][-1]), xytext=(5, offs[name]), textcoords="offset points", fontsize=7, color=col, va="center")
bx.set_xlim(15, 112)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), handlelength=1.2)
fig.tight_layout(w_pad=1.5, rect=(0, 0.06, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_main_arms.{ext}")
print("saved", OUT + "/fig_main_arms.{pdf,png}")
