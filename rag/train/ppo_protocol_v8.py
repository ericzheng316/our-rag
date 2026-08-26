"""v8 PPO on the protocol-v1 stack — Qwen3.5-9B policy, real FAISS, dual GPU.

Owner decisions wired here (2026-08-05): exploration actions (query rewrites,
slot pivots) need REAL retrieval, so rollouts hit the FAISS server, not the
closed pool; training is PPO with the critic from ``ppo_rsf_vllm_v8``
(ValueHead + GAE + clipped losses, 11 CPU tests); concurrency comes from a
dedicated engine card — ``--engine_gpu`` hosts the vLLM engine via
``EnginePool`` (worker pins its own CUDA_VISIBLE_DEVICES pre-import) while
``--train_gpu`` holds the HF+PEFT policy, value head and optimizer, so the
78 GiB single-card squeeze from the legacy smoke splits into ~44 + ~34.

Stack contracts honored:
- protocol v1 is canonical (``agent.protocol_v1``); episodes run through
  ``agent.episode.run_episodes_batched`` and the chat template uses
  ``enable_thinking=False`` + add_generation_prompt — the exact SFT
  conditioning.
- every adapter hot-swap goes through ``vllm_adapter_export.export_for_vllm``
  (the language_model-prefix rename; silent no-op otherwise) and
  ``EnginePool.lora_check`` runs at step 1 and every save (load success
  proves nothing).
- old-policy logprobs: the engine returns text only, so the pre-update
  policy's own no-grad forward serves as rollout logprobs — exactly the
  ``engine_lp=None`` fallback ``ppo_update`` already implements. Outputs are
  re-tokenized from text (+ eos); token-split drift is possible but rare for
  protocol-clean text and identical for policy/old sides.

Rewards per turn: search → -retrieval_cost; invalid protocol turn →
-format_penalty (terminal); answer → answer_weight * alias-EM against the
full golden_answers list (MuSiQue aliases included), via the repo-canonical
normalizer in ``belief.acec.answer_verifier_v64``.

vLLM spawns engine cores by re-importing __main__ — everything executable
stays under the guard.
"""

from __future__ import annotations


def build_parser():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--policy_init_adapter", default="",
                   help="策略初始化 adapter(续段用);空 = 与 sft_adapter 相同。"
                        "ref 锚永远取 sft_adapter——段 2 重锚实验证明锚点必须"
                        "是罕见 think 的 SFT 策略,否则自由通道频率失锚(7 集通胀)")
    p.add_argument("--sft_adapter",
                   default="/scratch/boyuz5/acec/checkpoints/sft_synth_v0/adapter_best")
    p.add_argument("--data_path", required=True,
                   help="jsonl: {id, question, golden_answers[]}")
    p.add_argument("--save_path", required=True)
    p.add_argument("--retrieve_url", default="")
    p.add_argument("--show_belief", type=int, default=0,
                   help="J8 观测注入:result 块附 [slots: ...] 覆盖状态行"
                        "(judge 服务计算,环境侧 token,不进奖励不经自由锚)")
    p.add_argument("--judge_url", default="",
                   help="judge 服务地址,如 http://127.0.0.1:18021/judge")
    p.add_argument("--show_budget", type=int, default=1,
                   help="1 = result 块附 [searches left: k]（状态完备性修复："
                        "冒烟实测撞上限率 47.5%→8.3%、EM +5.8pp、协议错不变）")
    p.add_argument("--closed_pool", type=int, default=0,
                   help="1 = MuSiQue 官方闭池设置：题目自带 20 段落做检索池"
                        "（pool_retriever 词法 F1），不需要 FAISS 检索器；"
                        "gold 必在池内 → shaping 与检索天花板 100%")
    p.add_argument("--engine_gpu", type=int, required=True)
    p.add_argument("--train_gpu", type=int, default=-1,
                   help="单卡训练的物理卡号（与 --train_gpus 二选一）")
    p.add_argument("--train_gpus", default="",
                   help="逗号分隔的物理卡号列表；配 torchrun --nproc-per-node=N "
                        "使用，rank i 用第 i 张。rank0 独占 rollout 引擎交互与保存")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8, help="tasks per episode")
    p.add_argument("--n_rollouts", type=int, default=2, help="rollouts per task")
    p.add_argument("--max_turns", type=int, default=8)
    p.add_argument("--docs_per_search", type=int, default=3)
    p.add_argument("--rollout_temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--max_prompt_tokens", type=int, default=6144)
    p.add_argument("--engine_mem_frac", type=float, default=0.85)
    p.add_argument("--lora_rank", type=int, default=32,
                   help="must match the SFT adapter's rank")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--critic_hidden", type=int, default=256,
                   help="critic MLP 隐层宽度；0=线性探针")
    p.add_argument("--gamma", type=float, default=0.99)  # owner 2026-08-05：多跳长视界
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--value_clip", type=float, default=0.2)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--kl_free_coef", type=float, default=0.0,
                   help="自由 token(parser 忽略)的 K3-KL 强锚系数。>0 时"
                        "启用双档锚:动作 token 用 --kl_coef 弱锚,自由 token"
                        "用本系数钉死在 SFT 分布——六次通胀崩塌的收口方案。")
    p.add_argument("--kl_coef", type=float, default=0.05,
                   help="0.01 在 48ep run 里拉不住漂移（KL>0.10 后 em 崩）；无自动早停，KL 由人工盯")
    p.add_argument("--vhead_init", default="",
                   help="critic 热启动权重（fit_value_head.py 产物）；不给则零初始化")
    p.add_argument("--value_feat", default="pool_last_ln",
                   choices=["last", "pool", "pool_last", "pool_last_ln"],
                   help="必须与热启动回归的最优变体一致")
    p.add_argument("--shadow_critic", type=int, default=0,
                   help="1 = critic 影子模式（v_loss 不进主 loss，只吃回放池；"
                        "版本 B 配 --group_baseline 1 用）")
    p.add_argument("--adv_mode", default="loo", choices=["gae", "group_z", "loo"],
                   help="loo = RLOO（同题 K-1 均值基线，一行减法广播）；"
                        "group_z = 组内 z-score；gae = critic 参与")
    p.add_argument("--shaping_beta", type=float, default=0.3,
                   help="gold 支撑段落覆盖增量的系数（title 精确匹配，0 关闭）")
    p.add_argument("--group_baseline", type=int, default=0,
                   help="[兼容] 1 等价 --adv_mode group_z")
    p.add_argument("--replay_eps", type=int, default=10,
                   help="critic 回放池保留最近 N ep 的 (feat,G)")
    p.add_argument("--replay_epochs", type=int, default=3)
    p.add_argument("--replay_batch", type=int, default=512)
    p.add_argument("--sym_feats", type=int, default=1,
                   help="1 = 影子 critic 特征拼符号量[覆盖率,新覆盖,turn进度,"
                        "检索占比](特权侧,仅训练期;asymmetric critic 合法)")
    p.add_argument("--micro_turns", type=int, default=1,
                   help="带梯度 forward+backward 合并 turn 数（实测 >1 在 seq~2k OOM）")
    p.add_argument("--scan_batch", type=int, default=8,
                   help="no_grad 值预扫的批大小")
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--retrieval_cost", type=float, default=0.005,
                   help="EM(k) 推导（2026-08-05）：冷启动边际价值 0.0018/次，"
                        "0.05 会教会策略不检索；训练中期按实测边际再调")
    p.add_argument("--retrieve_chunk", type=int, default=128,
                   help="批量检索单次 POST 的 query 上限（保护 retriever）")
    p.add_argument("--format_penalty", type=float, default=0.2)
    p.add_argument("--grad_mask_free", type=int, default=0,
                   help="0=不屏蔽;1=全屏蔽(自由 token 永不进策略梯度——实测"
                        "缴械纠偏通道,五启 ep40 史上最快崩塌,勿再用);"
                        "2=非对称:自由 token 仅在优势>0 时屏蔽(赢家 think "
                        "拿不到正梯度,输家复读照吃负梯度)。")
    p.add_argument("--token_cost", type=float, default=1e-4,
                   help="parser 忽略 token 的单价(think 内容/标签外裸文本)。"
                        "自由通道计价:外部信息 c_r、内部计算 c_tok、终止 ±EM。"
                        "初始扰动实测≈0(0.7%% turn 有自由 token),448 复读 "
                        "turn 付 0.045=3×c_r。两次崩溃(think 膨胀/复读)的"
                        "统一反压力,替代 ban(ban 会把压力改道成裸散文)。")
    p.add_argument("--turn_limit_penalty", type=float, default=0.3,
                   help="打满 max_turns 仍未作答的终局惩罚（放弃比答错更该罚："
                        "此前无此分支，且 shaping 与检索次数正相关 → 只搜不答"
                        "的 reward 高于尝试作答，正向奖励了无限改写循环）")
    p.add_argument("--answer_weight", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260805)
    return p


def main() -> None:
    import os
    import sys

    args = build_parser().parse_args()
    # 训练进程钉在训练卡上；EnginePool 的 worker 会在自己的 spawn 进程里
    # 用物理编号重设 CUDA_VISIBLE_DEVICES（引擎卡），互不干扰。
    # DDP：torchrun 起 N 进程，rank i pin 到 --train_gpus 的第 i 张物理卡，
    # 之后每进程内自己的卡都是 cuda:0。
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.train_gpus:
        gpus = [int(x) for x in args.train_gpus.split(",") if x.strip()]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[local_rank])
        world = len(gpus)
    else:
        if args.train_gpu < 0:
            raise SystemExit("需要 --train_gpu 或 --train_gpus")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_gpu)
        world = 1

    import json
    import random
    import re
    import time
    from pathlib import Path

    import requests
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = Path(__file__).resolve().parents[2]
    for extra in (repo / "rag" / "train", repo / "rag" / "src"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    import datetime

    import torch.distributed as dist

    dist_group = None
    rank = 0
    if world > 1:
        # rank0 独占引擎加载/export/lora_check（首次可 >10min），其余 rank 在
        # barrier/broadcast 等待 —— 默认 600s 会误杀，放大到 60min。
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
        dist_group = dist.group.WORLD
        rank = dist.get_rank()
        torch.cuda.set_device(0)  # 每进程 CUDA_VISIBLE_DEVICES 已 pin 单卡

    from agent.episode import run_episodes_batched
    _belief_fn = None
    if args.show_belief and args.judge_url:
        from agent.judge_tracker import make_belief_update_fn
        _belief_fn = make_belief_update_fn(judge_url=args.judge_url)
    from belief.acec.answer_verifier_v64 import answer_exact_match
    from engine_pool import EnginePool
    from ppo_rsf_vllm_v8 import coverage_shaping, make_value_head, ppo_update
    from vllm_adapter_export import export_for_vllm

    def log(msg: str) -> None:
        print(f"[ppo-proto-v8] {msg}", flush=True)

    # ── 显式播种 ─────────────────────────────────────────────────────────
    # --seed 统一控制:数据洗牌(下方 random.Random(args.seed))、torch RNG
    # (value head 初始化/dropout;全 rank 同种,参数以 rank0 broadcast 为准)、
    # python 全局 random、vLLM 引擎采样 RNG(EnginePool(seed=...))。
    # 2026-08-25 之前的 run 只播种了数据洗牌,其余为未播种非确定性。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    log(f"seed={args.seed} (data-shuffle + torch + python-random + engine)")

    # ── 数据 ────────────────────────────────────────────────────────────
    rows = []
    with open(args.data_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            golds = [str(g) for g in (r.get("golden_answers") or []) if str(g).strip()]
            if not r.get("question") or not golds:
                continue
            rows.append({"task_id": str(r.get("id", len(rows))),
                         "question": str(r["question"]),
                         "answer": golds[0], "golds": golds,
                         "gold_titles": [str(t) for t in (r.get("gold_titles") or [])],
                         "pool": r.get("pool") or []})
    random.Random(args.seed).shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    golds_by_task = {t["task_id"]: t["golds"] for t in rows}
    gold_titles_by_task = {t["task_id"]: t["gold_titles"] for t in rows}
    log(f"tasks={len(rows)}")

    retrieve_url = args.retrieve_url or os.environ.get("RETRIEVE_URL", "")
    if not retrieve_url and not args.closed_pool:
        raise SystemExit("需要 --retrieve_url（或 --closed_pool 1 走官方闭池）")

    session = requests.Session()

    def batched_retrieve(queries, k):
        """一个 turn 边界的全部 query 一次批发（--retrieve_chunk 分块保护）。

        瞬断重试 3 次（指数退避）后仍失败就抛 —— 静默降级返回空列表是
        2026-07-13 烧过一次的坑（带坏 reward 跑完全程），fail fast 正确。
        """
        out = []
        for s0 in range(0, len(queries), max(args.retrieve_chunk, 1)):
            batch = queries[s0:s0 + max(args.retrieve_chunk, 1)]
            for attempt in range(3):
                try:
                    resp = session.post(retrieve_url, json={"querys": batch},
                                        timeout=300)
                    resp.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
            for docs in resp.json():
                out.append([{"id": str(d.get("id", "")),
                             "contents": str(d.get("contents", ""))}
                            for d in docs[:k]])
        return out

    # ── 策略 + critic（训练卡）────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": 0})
    init_adapter = args.policy_init_adapter or args.sft_adapter
    model = PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
    # KL 参考 = 冻结的 SFT 初始策略（不是裸基座，也不随续段重锚）
    model.load_adapter(args.sft_adapter, adapter_name="ref")
    model.set_adapter("default")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    from ppo_rsf_vllm_v8 import value_feat_dim
    feat_dim = value_feat_dim(model.config.hidden_size, args.value_feat) \
        + (4 if args.sym_feats else 0)
    vhead = make_value_head(feat_dim, args.critic_hidden).to(model.device)
    if args.vhead_init:
        blob = torch.load(args.vhead_init, map_location="cpu")
        # fit_value_head 的 net 是 Sequential(Linear,Tanh,Linear)，与
        # MLPValueHead.net 同构；变体必须一致，维度不匹配会在此直接报错。
        vhead.net.load_state_dict(blob["state_dict"])
        log(f"critic 热启动: variant={blob.get('variant')} "
            f"heldout_ev={blob.get('heldout_ev', float('nan')):.3f}")
    from collections import deque
    replay = deque(maxlen=args.replay_eps)
    vhead_opt = torch.optim.AdamW(vhead.parameters(), lr=args.critic_lr,
                                  weight_decay=1e-4)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": vhead.parameters(), "lr": args.critic_lr},
    ])
    log(f"policy loaded: trainable="
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    def chat_template_fn(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)

    # ── 引擎（引擎卡，仅 rank0）───────────────────────────────────────────
    # torchrun 注入的 dist 环境变量会被 EnginePool 的 spawn worker 继承，
    # vLLM 内部 init 单进程分布式时读到它们就去连 torchrun 的 store（挂死在
    # MASTER_PORT 的 TCP 重试上）。自己的进程组 init 完后这些 env 已无用，
    # spawn 前清掉。
    # 无条件清理：world==1 时我们不 init 进程组，但 torchrun 仍注入了这些
    # 变量，engine worker 照样会去连它的 store 并超时（2026-08-08 单卡踩到）。
    if True:
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                  "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK",
                  "ROLE_WORLD_SIZE", "ROLE_NAME", "MASTER_ADDR", "MASTER_PORT",
                  "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
                  "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE"):
            os.environ.pop(k, None)
    pool = None
    if rank == 0:
        pool = EnginePool(args.base_model, [args.engine_gpu],
                          max_lora_rank=args.lora_rank,
                          gpu_memory_utilization=args.engine_mem_frac,
                          seed=args.seed)
        log(f"engine ready on GPU {args.engine_gpu}")

    os.makedirs(args.save_path, exist_ok=True)
    scratch = Path(args.save_path) / "_lora_scratch"
    step = 0

    def export_step_adapter() -> str:
        hf_dir = scratch / f"step_{step}"
        model.save_pretrained(str(hf_dir))
        vllm_dir = export_for_vllm(hf_dir, scratch / f"step_{step}_vllm")
        return str(vllm_dir)

    def check_adapter(path: str) -> None:
        probe = chat_template_fn([
            {"role": "system", "content": "You answer questions."},
            {"role": "user", "content": "Name any color."}])
        if not pool.lora_check(probe, path, step + 1):
            raise RuntimeError(f"lora_check FAILED for {path} — adapter 静默未生效")
        log("lora_check PASS")

    adapter_path = ""
    if rank == 0:
        adapter_path = export_step_adapter()
        check_adapter(adapter_path)
    if dist_group is not None:
        dist.barrier()

    # 动作闭合即截停:防止策略在 turn 内自我模拟环境(伪造 <result> 后继续
    # 生成,ep31 诊断实锤),也把动作之后的死 token 从梯度里剔掉。
    sampling = {"temperature": args.rollout_temperature,
                "max_tokens": args.max_new_tokens,
                "stop": ["</search>", "</answer>"],
                "include_stop_str_in_output": True}
    eos_id = tokenizer.eos_token_id
    metrics_path = Path(args.save_path) / "online_metrics_ppo_proto_v8.jsonl"

    # 自由通道 = parser 解析之外的一切(think 内容、标签外裸文本)。
    free_strip = re.compile(
        r'<search slot="\d+">.*?</search>|<answer>.*?</answer>|<plan>.*?</plan>',
        re.DOTALL)

    def free_token_count(turn_text: str) -> int:
        free = free_strip.sub("", turn_text).strip()
        if not free:
            return 0
        return len(tokenizer(free, add_special_tokens=False).input_ids)

    def grad_mask_for_turn(turn_text: str, out_len: int) -> torch.Tensor:
        """与 out(含末尾 EOS)等长的掩码:动作标签内 + EOS = 1,其余 = 0。"""
        enc = tokenizer(turn_text, add_special_tokens=False,
                        return_offsets_mapping=True)
        keep = [mm.span() for mm in free_strip.finditer(turn_text)]
        mask = torch.zeros(out_len, dtype=torch.float32)
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            if i >= out_len - 1:
                break
            if any(a < e and b > s for s, e in keep):
                mask[i] = 1.0
        mask[out_len - 1] = 1.0  # EOS:收束信号必须保留梯度
        return mask

    def episodes_to_trajs(episodes):
        trajs, traj_tasks, traj_syms, skipped_long = [], [], [], 0
        stats = {"answered": 0, "em_sum": 0.0, "searches": 0, "invalid": 0,
                 "shaping_sum": 0.0, "gave_up": 0, "free_tok": 0, "turns": 0}
        for ep in episodes:
            golds = golds_by_task.get(ep.task_id, [ep.gold_answer])
            g_titles = set(gold_titles_by_task.get(ep.task_id, []))
            # 打满 turn 未作答 = 放弃：不发过程分（否则"只搜不答"净收益最高），
            # 并在末 turn 记终局惩罚。
            gave_up = bool(ep.hit_turn_limit) and not ep.final_answer
            shp = ([0.0] * len(ep.turns) if gave_up else coverage_shaping(
                [t.doc_titles for t in ep.turns],
                gold_titles_by_task.get(ep.task_id, []), args.shaping_beta))
            covered = set()
            sym_seq = []
            n_search_so_far = 0
            for ti_, tr_ in enumerate(ep.turns):
                new_g = (set(tr_.doc_titles) & g_titles) - covered
                covered |= new_g
                if tr_.action_type == "search":
                    n_search_so_far += 1
                sym_seq.append([
                    len(covered) / max(len(g_titles), 1),
                    len(new_g) / max(len(g_titles), 1),
                    ti_ / max(args.max_turns, 1),
                    n_search_so_far / max(args.max_turns, 1)])
            turn_tuples = []
            turn_syms = []
            asst_positions = [i for i, m in enumerate(ep.messages)
                              if m["role"] == "assistant"]
            for i, pos in enumerate(asst_positions):
                prompt = chat_template_fn(ep.messages[:pos])
                inp = tokenizer(prompt, return_tensors="pt",
                                truncation=False).input_ids[0]
                if inp.numel() > args.max_prompt_tokens:
                    skipped_long += 1
                    break
                out = tokenizer(ep.messages[pos]["content"], return_tensors="pt",
                                add_special_tokens=False).input_ids[0]
                out = torch.cat([out, torch.tensor([eos_id])])
                turn = ep.turns[i] if i < len(ep.turns) else None
                if turn is not None and turn.action_type == "answer":
                    em = float(answer_exact_match(ep.final_answer or "", golds))
                    reward = args.answer_weight * em
                    stats["answered"] += 1
                    stats["em_sum"] += em
                elif turn is not None and turn.action_type == "invalid":
                    reward = -args.format_penalty
                    stats["invalid"] += 1
                else:
                    reward = -args.retrieval_cost + shp[i]
                    stats["searches"] += 1
                    stats["shaping_sum"] += shp[i]
                    if gave_up and i == len(asst_positions) - 1:
                        reward -= args.turn_limit_penalty
                        stats["gave_up"] += 1
                # 内部计算计价:parser 忽略的每个 token 付 c_tok(所有 turn
                # 类型统一适用——invalid 的复读体、answer 前的 think 都在内)。
                n_free = free_token_count(ep.messages[pos]["content"])
                reward -= args.token_cost * n_free
                stats["free_tok"] += n_free
                stats["turns"] += 1
                if args.grad_mask_free or args.kl_free_coef > 0:
                    gmask = grad_mask_for_turn(ep.messages[pos]["content"],
                                               out.numel())
                    turn_tuples.append((inp.to(model.device),
                                        out.to(model.device), reward, None,
                                        args.rollout_temperature, gmask))
                else:
                    turn_tuples.append((inp.to(model.device),
                                        out.to(model.device), reward, None,
                                        args.rollout_temperature))
                turn_syms.append(sym_seq[i] if i < len(sym_seq) else [0.0] * 4)
            if turn_tuples:
                trajs.append(turn_tuples)
                traj_syms.append(turn_syms)
                traj_tasks.append(ep.task_id)
        return trajs, traj_tasks, traj_syms, stats, skipped_long

    for episode_idx in range(args.num_episodes):
        lo = (episode_idx * args.batch_size) % len(rows)
        batch = rows[lo: lo + args.batch_size] or rows[: args.batch_size]
        dup = [t for t in batch for _ in range(args.n_rollouts)]

        t0 = time.time()
        episodes, trajs, traj_tasks, traj_syms, ep_stats, skipped_long = [], [], [], [], None, 0
        if rank == 0:
            from agent.episode import pool_retriever
            episodes = run_episodes_batched(
                dup,
                lambda prompts: [r["text"] for r in pool.generate(
                    prompts, dict(sampling), adapter_path=adapter_path,
                    adapter_version=step + 1)],
                pool_retriever if args.closed_pool else None,
                chat_template_fn,
                max_turns=args.max_turns,
                docs_per_search=args.docs_per_search,
                batched_retrieve_fn=None if args.closed_pool else batched_retrieve,
                show_budget=bool(args.show_budget),
                show_belief=bool(args.show_belief),
                belief_update_fn=(_belief_fn if args.show_belief else None),
            )
            trajs, traj_tasks, traj_syms, ep_stats, skipped_long = episodes_to_trajs(episodes)
        rollout_s = time.time() - t0

        t1 = time.time()
        stats = ppo_update(
            model, vhead, optimizer, [[t] for t in trajs],
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps, value_clip=args.value_clip,
            value_coef=args.value_coef, kl_coef=args.kl_coef,
            entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs,
            max_grad_norm=args.max_grad_norm,
            max_engine_logprob_mae=None,
            micro_turns=args.micro_turns, scan_batch=args.scan_batch,
            dist_group=dist_group,
            group_ids=traj_tasks, group_baseline=bool(args.group_baseline),
            adv_mode=args.adv_mode,
            value_variant=args.value_feat,
            train_value=not args.shadow_critic,
            collect_value_samples=True,
            grad_mask_mode={0: "none", 1: "always", 2: "pos"}[args.grad_mask_free],
            kl_free_coef=args.kl_free_coef,
        )
        update_s = time.time() - t1

        # ── critic 跨 ep 回放池（owner 2026-08-06 批准）────────────────────
        vs_new = stats.pop("value_samples", None)
        replay_ev = float("nan")
        gen_ev = float("nan")
        if rank == 0 and vs_new is not None:
            feats_h, G_new = vs_new
            if args.sym_feats:
                sym_flat = torch.tensor(
                    [v for ts in traj_syms for v in ts], dtype=torch.float16)
                if sym_flat.shape[0] == feats_h.shape[0]:
                    feats_h = torch.cat([feats_h, sym_flat], dim=1)
            vs_new = (feats_h, G_new)
            # 新 ev 定义:上一轮回放训练后的 vhead 对本 ep 未见样本的泛化
            with torch.no_grad():
                Xn = feats_h.float().to(model.device)
                pred = vhead(Xn)
                yn = G_new.to(model.device)
                gen_ev = float(1 - (yn - pred).var() / (yn.var() + 1e-12))
            replay.append(vs_new)
            X = torch.cat([f for f, _g in replay]).float().to(model.device)
            y = torch.cat([g for _f, g in replay]).to(model.device)
            for _ in range(args.replay_epochs):
                perm = torch.randperm(X.shape[0], device=X.device)
                for s0 in range(0, X.shape[0], args.replay_batch):
                    idx = perm[s0:s0 + args.replay_batch]
                    loss_v = (vhead(X[idx]) - y[idx]) ** 2
                    vhead_opt.zero_grad()
                    loss_v.mean().backward()
                    vhead_opt.step()
            with torch.no_grad():
                pred = vhead(X[:4096])
                yy = y[:4096]
                replay_ev = float(1 - (yy - pred).var() / (yy.var() + 1e-12))
        if dist_group is not None:
            for p_ in vhead.parameters():
                dist.broadcast(p_.data, src=0)

        step += 1
        if rank != 0:
            continue  # 非 rank0 直接进下一轮（broadcast 是天然同步点）
        adapter_path = export_step_adapter()

        n_ep = max(len(episodes), 1)
        mean_r = sum(sum(t[2] for t in traj) for traj in trajs) / max(len(trajs), 1)
        record = {
            "episode": episode_idx + 1,
            "mean_R": mean_r,
            "answer_rate": ep_stats["answered"] / n_ep,
            "alias_em": ep_stats["em_sum"] / max(ep_stats["answered"], 1),
            "mean_searches": ep_stats["searches"] / n_ep,
            "mean_shaping": ep_stats["shaping_sum"] / n_ep,
            "invalid_rate": ep_stats["invalid"] / n_ep,
            "giveup_rate": ep_stats["gave_up"] / n_ep,
            "mean_free_tok": ep_stats["free_tok"] / max(ep_stats["turns"], 1),
            "skipped_long_prompts": skipped_long,
            "rollout_s": round(rollout_s, 1),
            "update_s": round(update_s, 1),
            "replay_ev": replay_ev,
            "gen_ev": gen_ev,
            **{k: v for k, v in stats.items() if isinstance(v, (int, float))},
        }
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"ep {episode_idx+1:4d} R={mean_r:+.3f} em={record['alias_em']:.3f} "
            f"ans={record['answer_rate']:.2f} inv={record['invalid_rate']:.2f} "
            f"giv={record['giveup_rate']:.2f} "
            f"ftk={record['mean_free_tok']:.1f} "
            f"kl={stats.get('kl', 0):.3f}/{stats.get('kl_free', 0):.3f} "
            f"ev={gen_ev:.3f} "
            f"rev={replay_ev:.3f} shp={record['mean_shaping']:.3f} "
            f"[{rollout_s:.0f}s+{update_s:.0f}s]")

        if step % args.save_steps == 0 or episode_idx + 1 == args.num_episodes:
            ckpt = Path(args.save_path) / f"step_{step}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            torch.save(vhead.state_dict(), ckpt / "value_head.pt")
            export_for_vllm(ckpt, Path(str(ckpt) + "_vllm"))
            check_adapter(str(Path(str(ckpt) + "_vllm")))

    if pool is not None:
        pool.shutdown()
    if dist_group is not None:
        dist.barrier()
        dist.destroy_process_group()
    log("done")


if __name__ == "__main__":
    main()
