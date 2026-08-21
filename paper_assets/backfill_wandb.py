"""把全部历史训练运行回填成 W&B 离线 runs(超参 + 逐集指标)。

离线模式写入 /scratch/boyuz5/acec/wandb,之后可用
`wandb sync /scratch/boyuz5/acec/wandb/offline-run-*` 上传到账号。
指标优先取 ckpt/online_metrics_ppo_proto_v8.jsonl;缺失则回退解析
train.log 的心跳行。
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runs_registry as R

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "/scratch/boyuz5/acec/wandb"
os.makedirs("/scratch/boyuz5/acec/wandb", exist_ok=True)
import wandb

HB = re.compile(
    r"ep\s+(\d+)\s+R=([+-][\d.]+)\s+em=([\d.]+)\s+ans=([\d.]+)\s+"
    r"inv=([\d.]+)\s+giv=([\d.]+)(?:\s+ftk=([\d.]+))?"
    r"(?:\s+kl=([\d.]+)/([\d.]+))?")


def episodes_from_log(path):
    out = []
    for line in open(path, errors="ignore"):
        m = HB.search(line)
        if m:
            g = m.groups()
            row = {"episode": int(g[0]), "mean_R": float(g[1]),
                   "alias_em": float(g[2]), "answer_rate": float(g[3]),
                   "invalid_rate": float(g[4]), "giveup_rate": float(g[5])}
            if g[6] is not None:
                row["mean_free_tok"] = float(g[6])
            if g[7] is not None:
                row["kl_action"] = float(g[7])
                row["kl_free"] = float(g[8])
            out.append(row)
    return out


def backfill_rl():
    for rid, label, group, delta, outcome, conf in R.RL_RUNS:
        d = Path(R.LOGS) / rid
        if not d.exists():
            print(f"跳过(目录缺失): {rid}")
            continue
        metrics_file = d / "ckpt" / "online_metrics_ppo_proto_v8.jsonl"
        eps = []
        if metrics_file.exists():
            for line in open(metrics_file, errors="ignore"):
                try:
                    eps.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not eps and (d / "train.log").exists():
            eps = episodes_from_log(d / "train.log")
        cfg = dict(R.BASE)
        cfg.update(delta)
        cfg.update(run_dir=str(d), outcome=outcome, confidence=conf,
                   n_episodes=len(eps))
        run = wandb.init(project="acec-v8", group=group, name=label,
                        id=rid.lower().replace(".", "-")[:64],
                        config=cfg, reinit=True)
        for e in eps:
            step = int(e.get("episode", 0))
            run.log({k: v for k, v in e.items()
                     if isinstance(v, (int, float))}, step=step)
        run.finish()
        print(f"✅ {label}: {len(eps)} eps")


def backfill_flat(runs, group, loss_glob):
    for rid, cfg0, outcome in runs:
        d = Path(loss_glob) / rid
        cfg = dict(cfg0); cfg.update(outcome=outcome, run_dir=str(d))
        run = wandb.init(project="acec-v8", group=group, name=rid,
                        id=f"{group}-{rid}"[:64], config=cfg, reinit=True)
        log = d / "train.log"
        if log.exists():
            step = 0
            for line in open(log, errors="ignore"):
                m = re.search(r"step (\d+)/\d+ loss ([\d.]+)", line)
                if m:
                    run.log({"train_loss": float(m.group(2))},
                            step=int(m.group(1)))
                m2 = re.search(r"dev_loss ([\d.]+)", line)
                if m2:
                    step += 1
                    run.log({"dev_loss": float(m2.group(1))}, step=100000 + step)
        run.finish()
        print(f"✅ {group}/{rid}")


if __name__ == "__main__":
    backfill_rl()
    backfill_flat(R.SFT_RUNS, "sft", "/scratch/boyuz5/acec/checkpoints")
    backfill_flat(R.JUDGE_RUNS, "judge", "/scratch/boyuz5/acec/checkpoints")
    print("回填完成; 同步命令: wandb sync /scratch/boyuz5/acec/wandb/wandb/offline-run-*")
