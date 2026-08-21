"""Judge-C:Qwen3.5-4B LoRA 微调(ACEC_JUDGE_SPEC.md §3/§4)。

样本 = 六族数据的 messages;loss 只算 assistant 段(Yes/No 首 token 供
logit 读数,正例带逐字 quote)。模式沿用 sft_synth_v0 的已验证做法:
chat template + enable_thinking=False、梯度检查点、手写循环、flush 日志、
adapter 双导出(HF + vLLM 前缀重命名,fail-closed)。
"""

from __future__ import annotations

import argparse


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data", default="/scratch/boyuz5/acec/judge/judge_train.jsonl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_samples", type=int, default=0, help="0=全量")
    p.add_argument("--dev_frac", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260815)
    return p


def main() -> None:
    import json
    import math
    import random
    import sys
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = build_parser().parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(f"[judge] {m}", flush=True)

    rows = []
    for line in open(args.data):
        rows.append(json.loads(line))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_samples:
        rows = rows[: args.max_samples]
    n_dev = max(int(len(rows) * args.dev_frac), 200)
    dev, train = rows[:n_dev], rows[n_dev:]
    log(f"train {len(train)} / dev {n_dev}")

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def encode(sample):
        msgs = sample["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        full = prompt + msgs[-1]["content"] + tok.eos_token
        ids = tok(full, truncation=True, max_length=args.max_len,
                  add_special_tokens=False).input_ids
        n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
        labels = [-100] * min(n_prompt, len(ids)) + ids[n_prompt:]
        return ids, labels[: len(ids)]

    def collate(batch):
        enc = [encode(b) for b in batch]
        L = max(len(i) for i, _ in enc)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(enc), L), pad, dtype=torch.long)
        lab = torch.full((len(enc), L), -100, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for k, (i, l) in enumerate(enc):
            ids[k, : len(i)] = torch.tensor(i)
            lab[k, : len(l)] = torch.tensor(l)
            att[k, : len(i)] = 1
        return ids, lab, att

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": 0})
    lconf = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                       lora_dropout=0.05, task_type="CAUSAL_LM",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lconf)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr)
    steps_total = math.ceil(len(train) / (args.batch * args.accum)) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True,
                        collate_fn=collate)

    def dev_loss():
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for k in range(0, min(len(dev), 800), args.batch):
                ids, lab, att = collate(dev[k: k + args.batch])
                o = model(input_ids=ids.cuda(), attention_mask=att.cuda(),
                          labels=lab.cuda())
                tot += float(o.loss)
                n += 1
        model.train()
        return tot / max(n, 1)

    best = float("inf")
    gstep = 0
    model.train()
    for ep in range(args.epochs):
        opt.zero_grad()
        for bi, (ids, lab, att) in enumerate(loader):
            o = model(input_ids=ids.cuda(), attention_mask=att.cuda(),
                      labels=lab.cuda())
            (o.loss / args.accum).backward()
            if (bi + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
                if gstep % 50 == 0:
                    log(f"ep{ep} step {gstep}/{steps_total} "
                        f"loss {float(o.loss):.4f} lr {sched.get_last_lr()[0]:.2e}")
                if gstep % 500 == 0:
                    dl = dev_loss()
                    log(f"dev_loss {dl:.4f}")
                    if dl < best:
                        best = dl
                        model.save_pretrained(out / "adapter_best")
        dl = dev_loss()
        log(f"epoch {ep} dev_loss {dl:.4f}")
        if dl < best:
            best = dl
            model.save_pretrained(out / "adapter_best")

    model.save_pretrained(out / "adapter_final")
    sys.path.insert(0, str(Path(__file__).parent))
    from vllm_adapter_export import export_for_vllm
    for name in ("adapter_best", "adapter_final"):
        if (out / name).exists():
            export_for_vllm(str(out / name), str(out / f"{name}_vllm"))
    log(f"done; best dev_loss {best:.4f}")
    print("JUDGE_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
