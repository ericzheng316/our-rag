"""图:自由 token 通胀法医学——五种干预 vs 双档锚(§9 故事的主图)。
数据源:各 run 的 online_metrics jsonl(mean_free_tok)。可复现:直接运行。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from style import C, OUT, plt
import runs_registry as R

PICK = {  # run id → (图例, 颜色, 线型)
    "ppo500_s1_20260808T201116Z": ("no intervention", C["gray"], ":"),
    "ppo500_s1_20260809T032001Z": ("token cost 1e-4", C["warn"], "--"),
    "ppo500_s1_20260809T151110Z": ("full grad-mask", C["bad"], "--"),
    "ppo500_s1_20260809T231649Z": ("asym grad-mask", C["purple"], "--"),
    "ppo500_s2_20260812T233826Z": ("re-anchored ref", C["bad"], "-."),
    "ppo500_s1_20260810T084837Z": ("two-tier KL anchor (ours)", C["main"], "-"),
}

fig, ax = plt.subplots(figsize=(4.6, 3.0))
for rid, (label, color, ls) in PICK.items():
    f = Path(R.LOGS) / rid / "ckpt" / "online_metrics_ppo_proto_v8.jsonl"
    if not f.exists():
        continue
    xs, ys = [], []
    for line in open(f, errors="ignore"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "mean_free_tok" in e:
            xs.append(e["episode"]); ys.append(e["mean_free_tok"])
    if xs:
        ax.plot(xs, ys, ls, color=color, label=label)
ax.set_yscale("log")
ax.set_xlabel("training episode")
ax.set_ylabel("free tokens per turn (log)")
ax.set_xlim(0, 100)
ax.legend(frameon=False, loc="upper right")
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig_ftk_forensics.{ext}")
print("✅ fig_ftk_forensics")
