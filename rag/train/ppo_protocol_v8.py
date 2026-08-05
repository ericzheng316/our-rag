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
    p.add_argument("--sft_adapter",
                   default="/scratch/boyuz5/acec/checkpoints/sft_synth_v0/adapter_best")
    p.add_argument("--data_path", required=True,
                   help="jsonl: {id, question, golden_answers[]}")
    p.add_argument("--save_path", required=True)
    p.add_argument("--retrieve_url", default="")
    p.add_argument("--engine_gpu", type=int, required=True)
    p.add_argument("--train_gpu", type=int, required=True)
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
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--micro_turns", type=int, default=4,
                   help="update 阶段每次 forward+backward 合并的 turn 数")
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--retrieval_cost", type=float, default=0.005,
                   help="EM(k) 推导（2026-08-05）：冷启动边际价值 0.0018/次，"
                        "0.05 会教会策略不检索；训练中期按实测边际再调")
    p.add_argument("--retrieve_chunk", type=int, default=128,
                   help="批量检索单次 POST 的 query 上限（保护 retriever）")
    p.add_argument("--format_penalty", type=float, default=0.2)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_gpu)

    import json
    import random
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

    from agent.episode import run_episodes_batched
    from belief.acec.answer_verifier_v64 import answer_exact_match
    from engine_pool import EnginePool
    from ppo_rsf_vllm_v8 import make_value_head, ppo_update
    from vllm_adapter_export import export_for_vllm

    def log(msg: str) -> None:
        print(f"[ppo-proto-v8] {msg}", flush=True)

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
                         "answer": golds[0], "golds": golds})
    random.Random(args.seed).shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    golds_by_task = {t["task_id"]: t["golds"] for t in rows}
    log(f"tasks={len(rows)}")

    retrieve_url = args.retrieve_url or os.environ.get("RETRIEVE_URL", "")
    if not retrieve_url:
        raise SystemExit("需要 --retrieve_url（动作探索依赖真实 FAISS 检索）")

    session = requests.Session()

    def batched_retrieve(queries, k):
        """一个 turn 边界的全部 query 一次批发（--retrieve_chunk 分块保护）。"""
        out = []
        for s0 in range(0, len(queries), max(args.retrieve_chunk, 1)):
            batch = queries[s0:s0 + max(args.retrieve_chunk, 1)]
            resp = session.post(retrieve_url, json={"querys": batch}, timeout=300)
            resp.raise_for_status()
            for docs in resp.json():
                out.append([{"id": str(d.get("id", "")),
                             "contents": str(d.get("contents", ""))}
                            for d in docs[:k]])
        return out

    # ── 策略 + critic（训练卡）────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    # KL 参考 = 冻结的 SFT 初始策略（不是裸基座）——第二份同源 adapter，命名 "ref"
    model.load_adapter(args.sft_adapter, adapter_name="ref")
    model.set_adapter("default")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    vhead = make_value_head(model.config.hidden_size, args.critic_hidden).to(model.device)
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

    # ── 引擎（引擎卡）────────────────────────────────────────────────────
    pool = EnginePool(args.base_model, [args.engine_gpu],
                      max_lora_rank=args.lora_rank,
                      gpu_memory_utilization=args.engine_mem_frac)
    log(f"engine ready on GPU {args.engine_gpu}")

    os.makedirs(args.save_path, exist_ok=True)
    scratch = Path(args.save_path) / "_lora_scratch"
    step = 0

    def export_step_adapter() -> str:
        hf_dir = scratch / f"step_{step}"
        model.save_pretrained(str(hf_dir))
        vllm_dir = export_for_vllm(hf_dir, scratch / f"step_{step}_vllm")
        return str(vllm_dir)

    adapter_path = export_step_adapter()

    def check_adapter(path: str) -> None:
        probe = chat_template_fn([
            {"role": "system", "content": "You answer questions."},
            {"role": "user", "content": "Name any color."}])
        if not pool.lora_check(probe, path, step + 1):
            raise RuntimeError(f"lora_check FAILED for {path} — adapter 静默未生效")
        log("lora_check PASS")

    check_adapter(adapter_path)

    sampling = {"temperature": args.rollout_temperature,
                "max_tokens": args.max_new_tokens}
    eos_id = tokenizer.eos_token_id
    metrics_path = Path(args.save_path) / "online_metrics_ppo_proto_v8.jsonl"

    def episodes_to_trajs(episodes):
        trajs, skipped_long = [], 0
        stats = {"answered": 0, "em_sum": 0.0, "searches": 0, "invalid": 0}
        for ep in episodes:
            golds = golds_by_task.get(ep.task_id, [ep.gold_answer])
            turn_tuples = []
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
                    reward = -args.retrieval_cost
                    stats["searches"] += 1
                turn_tuples.append((inp.to(model.device), out.to(model.device),
                                    reward, None, args.rollout_temperature))
            if turn_tuples:
                trajs.append(turn_tuples)
        return trajs, stats, skipped_long

    for episode_idx in range(args.num_episodes):
        lo = (episode_idx * args.batch_size) % len(rows)
        batch = rows[lo: lo + args.batch_size] or rows[: args.batch_size]
        dup = [t for t in batch for _ in range(args.n_rollouts)]

        t0 = time.time()
        episodes = run_episodes_batched(
            dup,
            lambda prompts: [r["text"] for r in pool.generate(
                prompts, dict(sampling), adapter_path=adapter_path,
                adapter_version=step + 1)],
            None,
            chat_template_fn,
            max_turns=args.max_turns,
            docs_per_search=args.docs_per_search,
            batched_retrieve_fn=batched_retrieve,
        )
        rollout_s = time.time() - t0

        trajs, ep_stats, skipped_long = episodes_to_trajs(episodes)
        t1 = time.time()
        stats = ppo_update(
            model, vhead, optimizer, [[t] for t in trajs],
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps, value_clip=args.value_clip,
            value_coef=args.value_coef, kl_coef=args.kl_coef,
            entropy_coef=args.entropy_coef, ppo_epochs=args.ppo_epochs,
            max_grad_norm=args.max_grad_norm,
            max_engine_logprob_mae=None,
            micro_turns=args.micro_turns,
        )
        update_s = time.time() - t1

        step += 1
        adapter_path = export_step_adapter()

        n_ep = max(len(episodes), 1)
        mean_r = sum(sum(t[2] for t in traj) for traj in trajs) / max(len(trajs), 1)
        record = {
            "episode": episode_idx + 1,
            "mean_R": mean_r,
            "answer_rate": ep_stats["answered"] / n_ep,
            "alias_em": ep_stats["em_sum"] / max(ep_stats["answered"], 1),
            "mean_searches": ep_stats["searches"] / n_ep,
            "invalid_rate": ep_stats["invalid"] / n_ep,
            "skipped_long_prompts": skipped_long,
            "rollout_s": round(rollout_s, 1),
            "update_s": round(update_s, 1),
            **{k: v for k, v in stats.items() if isinstance(v, (int, float))},
        }
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"ep {episode_idx+1:4d} R={mean_r:+.3f} em={record['alias_em']:.3f} "
            f"ans={record['answer_rate']:.2f} inv={record['invalid_rate']:.2f} "
            f"ev={stats.get('explained_variance', float('nan')):.3f} "
            f"[{rollout_s:.0f}s+{update_s:.0f}s]")

        if step % args.save_steps == 0 or episode_idx + 1 == args.num_episodes:
            ckpt = Path(args.save_path) / f"step_{step}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            torch.save(vhead.state_dict(), ckpt / "value_head.pt")
            export_for_vllm(ckpt, Path(str(ckpt) + "_vllm"))
            check_adapter(str(Path(str(ckpt) + "_vllm")))

    pool.shutdown()
    log("done")


if __name__ == "__main__":
    main()
