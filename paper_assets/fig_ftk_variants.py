"""图一(通胀法医学)三种排版候选,供风格拍板。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from style import OUT, plt
import runs_registry as R

RUNS = {
    "no intervention":   ("ppo500_s1_20260808T201116Z", "#999999"),
    "token cost 1e-4":   ("ppo500_s1_20260809T032001Z", "#e08a00"),
    "full grad-mask":    ("ppo500_s1_20260809T151110Z", "#c23b22"),
    "asym grad-mask":    ("ppo500_s1_20260809T231649Z", "#6a4fa3"),
    "re-anchored ref":   ("ppo500_s2_20260812T233826Z", "#b3446c"),
    "two-tier KL (ours)": ("ppo500_s1_20260810T084837Z", "#1a63a8"),
}

def series(rid):
    f = Path(R.LOGS) / rid / "ckpt" / "online_metrics_ppo_proto_v8.jsonl"
    xs, ys = [], []
    if f.exists():
        for line in open(f, errors="ignore"):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "mean_free_tok" in e:
                xs.append(e["episode"]); ys.append(max(e["mean_free_tok"], 0.05))
    return xs, ys

DATA = {k: series(rid) for k, (rid, _) in RUNS.items()}

# ── A. 突出-灰化:失败全灰细线+线端标注,主线独色加粗 ──────────────
fig, ax = plt.subplots(figsize=(4.4, 3.0))
for name, (rid, color) in RUNS.items():
    xs, ys = DATA[name]
    if not xs:
        continue
    if name == "two-tier KL (ours)":
        ax.plot(xs, ys, "-", color="#1a63a8", lw=2.4, zorder=5)
        ax.text(xs[-1] + 1, ys[-1], "ours", color="#1a63a8", fontsize=8,
                va="center", fontweight="bold")
    else:
        ax.plot(xs, ys, "-", color="#b0b0b0", lw=1.0)
        ax.text(xs[-1] + 1, ys[-1], name, color="#808080", fontsize=6,
                va="center")
ax.set_yscale("log"); ax.set_xlim(0, 112)
ax.set_xlabel("training episode"); ax.set_ylabel("free tokens / turn (log)")
fig.savefig(f"{OUT}/fig1_A_emphasis.png")
plt.close(fig)

# ── B. 小面板矩阵:每失败一个面板,主线为参照虚线 ─────────────────
fails = [k for k in RUNS if k != "two-tier KL (ours)"]
fig, axes = plt.subplots(1, 5, figsize=(8.6, 1.9), sharey=True)
ox, oy = DATA["two-tier KL (ours)"]
for ax, name in zip(axes, fails):
    xs, ys = DATA[name]
    ax.plot(ox, oy, "-", color="#1a63a8", lw=1.0, alpha=0.55)
    ax.plot(xs, ys, "-", color=RUNS[name][1], lw=1.6)
    ax.set_yscale("log"); ax.set_title(name, fontsize=7)
    ax.set_xlim(0, 100)
axes[0].set_ylabel("free tok/turn")
for ax in axes:
    ax.set_xlabel("ep", fontsize=7)
fig.savefig(f"{OUT}/fig1_B_facets.png")
plt.close(fig)

# ── C. 双面板:左=计价/屏蔽类,右=锚类对照 ───────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
for name in ("no intervention", "token cost 1e-4", "full grad-mask",
             "asym grad-mask"):
    xs, ys = DATA[name]
    a1.plot(xs, ys, "-", color=RUNS[name][1], lw=1.4, label=name)
for name in ("re-anchored ref", "two-tier KL (ours)"):
    xs, ys = DATA[name]
    a2.plot(xs, ys, "-", color=RUNS[name][1],
            lw=2.2 if "ours" in name else 1.4, label=name)
for ax, t in ((a1, "pricing / masking interventions"),
              (a2, "anchor choice")):
    ax.set_yscale("log"); ax.set_xlim(0, 100)
    ax.set_title(t, fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_xlabel("training episode")
a1.set_ylabel("free tokens / turn (log)")
fig.savefig(f"{OUT}/fig1_C_twopanel.png")
plt.close(fig)
print("✅ 三个候选已渲染")
