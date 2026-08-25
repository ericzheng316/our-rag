"""图:J8 信念观测注入——负结果(注入 vs 对照,同窗口双臂)。

左:训练期无效动作率逐集(注入臂行为退化更快,双臂均触发守护截停)。
右:v3-RL 终点续训 40 集后的全量 dev EM(注入不增益,趋势为负)。
数据:j8_inj_20260821T235038Z / j8_ctrl_20260822T163135Z 的 online_metrics,
eval_j8_ladder_20260823T065004Z 的 full_{inj,ctrl}40_c*.jsonl。
"""
import glob
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from style import C, OUT, plt

L = "/home/boyuz5/acec/logs"
INJ_OM = f"{L}/j8_inj_20260821T235038Z/ckpt/online_metrics_ppo_proto_v8.jsonl"
CTRL_OM = f"{L}/j8_ctrl_20260822T163135Z/ckpt/online_metrics_ppo_proto_v8.jsonl"
LADDER = f"{L}/eval_j8_ladder_20260823T065004Z"
V3RL_INIT_EM = None  # 从预测文件重算,见下


def curve(path, key):
    rows = [json.loads(l) for l in open(path)]
    return [r["episode"] for r in rows], [r[key] for r in rows]


def full_em(pat):
    em = n = 0
    files = glob.glob(pat)
    assert files, f"无文件: {pat}"
    for f in files:
        for line in open(f):
            r = json.loads(line)
            em += int(r["alias_em"])
            n += 1
    return 100 * em / n, n


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6),
                               gridspec_kw={"width_ratios": [1.5, 1]})

# ── 左:无效动作率 ──────────────────────────────────────────────────
for path, color, label in [(CTRL_OM, C["gray"], "control"),
                           (INJ_OM, C["bad"], "belief injection")]:
    ep, inv = curve(path, "invalid_rate")
    ax1.plot(ep, [100 * v for v in inv], color=color, label=label)
ax1.axhline(15, color=C["warn"], lw=0.8, ls="--")
ax1.text(2, 15.6, "guard threshold", fontsize=7, color=C["warn"])
ax1.set_xlabel("episode (resumed from v3-RL ep100)")
ax1.set_ylabel("invalid-action rate (%)")
ax1.legend(frameon=False, loc="upper left")

# ── 右:+40 集全量 dev EM ───────────────────────────────────────────
init_em, _ = full_em(f"{L}/eval_v3rl_fulldev_20260819T041512Z/hops_c*.jsonl")
ctrl_em, n1 = full_em(f"{LADDER}/full_ctrl40_c*.jsonl")
inj_em, n2 = full_em(f"{LADDER}/full_inj40_c*.jsonl")
assert n1 == n2 == 2417
bars = ax2.bar([0, 1], [ctrl_em, inj_em], width=0.55,
               color=[C["gray"], C["bad"]])
ax2.axhline(init_em, color=C["main"], lw=0.9, ls=":")
ax2.text(1.32, init_em, f"v3-RL init\n{init_em:.1f}", fontsize=7,
         color=C["main"], va="center")
for x, v in [(0, ctrl_em), (1, inj_em)]:
    ax2.text(x, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["control\n+40 ep", "injection\n+40 ep"], fontsize=7.5)
ax2.set_ylabel("full-dev EM (%)")
ax2.set_ylim(44, 49)
ax2.text(0.5, 48.6, "McNemar z=$-$1.80 (n.s.)", ha="center", fontsize=7.5)

fig.tight_layout()
fig.savefig(f"{OUT}/fig_j8.png")
fig.savefig(f"{OUT}/fig_j8.pdf")
print(f"✅ fig_j8  init {init_em:.2f} | ctrl {ctrl_em:.2f} | inj {inj_em:.2f}")
