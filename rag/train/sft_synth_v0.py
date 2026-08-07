"""SFT cold start on synthetic trajectories — Qwen3.5-9B + LoRA, pinned stack.

Purpose: bake the v0 action protocol (<plan> / <search slot="N"> / <answer>)
and the tagged-answer contract into the backbone before any RL, so the entire
v6.3 Gate-1 class of output-contract failures dies at the root.

Method notes:
  * every assistant turn becomes one training sample: prompt =
    chat-template(messages so far, add_generation_prompt, thinking off),
    target = that turn's content + <|im_end|>.  This exactly matches the
    inference-time conditioning (empty-<think> prefix included), which
    per-conversation masking approximations do not guarantee.
  * LoRA targets the standard 7 projections only — the vLLM round-trip was
    verified for that set; all-linear (which would touch the fla layers) was
    not, so it stays out of the pipeline until separately verified.
  * after training, the adapter is exported for vLLM via
    vllm_adapter_export.export_for_vllm (mandatory key rewrite) and a quick
    first-turn format-compliance check runs on the dev split.

Vendored LLaMA-Factory was rejected: no qwen3 template, pins transformers
<=4.46 against the venv's 5.13 — same conflict class as the OpenRLHF path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def expand_trajectories(path: str, tokenizer, max_len: int):
    """messages-per-trajectory -> one sample per assistant turn."""
    samples, skipped = [], 0
    for line in open(path, encoding="utf-8"):
        traj = json.loads(line)
        messages = traj["messages"]
        # skip_turns（改写示范数据）：这些 assistant turn 是"坏查询"，只做
        # 上下文，绝不做监督目标 —— 否则等于正向教模型发坏查询。
        skip = set(traj.get("skip_turns") or [])
        asst_i = -1
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            asst_i += 1
            if asst_i in skip:
                continue
            prompt = tokenizer.apply_chat_template(
                messages[:i], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            target = msg["content"] + "<|im_end|>"
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            if len(prompt_ids) + len(target_ids) > max_len:
                skipped += 1
                continue
            samples.append(
                {
                    "task_id": traj["task_id"],
                    "input_ids": prompt_ids + target_ids,
                    "labels": [-100] * len(prompt_ids) + target_ids,
                }
            )
    return samples, skipped


class TurnDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch, pad_id: int):
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention.append([1] * (width - pad) + [0] * pad)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attention),
    )


@torch.no_grad()
def dev_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for input_ids, labels, attention in loader:
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=attention.to(device),
            labels=labels.to(device),
        )
        n = int((labels != -100).sum())
        total += float(out.loss) * n
        count += n
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def format_check(model, tokenizer, tasks_path: str, dev_task_ids, device, n: int = 8):
    """Greedy first-turn generation on dev questions: does the protocol hold?"""
    from synth.render_sft import SYSTEM_PROMPT  # noqa: PLC0415

    model.eval()
    model.config.use_cache = True  # re-enable KV cache for generation
    pattern = re.compile(r"<plan>.*?</plan>\s*<search slot=\"1\">.+?</search>", re.DOTALL)
    checked = passed = 0
    for line in open(tasks_path, encoding="utf-8"):
        task = json.loads(line)
        if task["task_id"] not in dev_task_ids:
            continue
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["question"]},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs, max_new_tokens=320, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        checked += 1
        if pattern.search(text):
            passed += 1
        if checked >= n:
            break
    model.train()
    return passed, checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", required=True)
    parser.add_argument("--tasks", required=True, help="tasks_final.jsonl for the format check")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    # batch 1 + accum 16 (same effective batch): the HF fallback for the
    # linear-attention layers (no flash-linear-attention installed) keeps
    # large autograd intermediates and OOMs an 80G A100 at batch 2 x 4096.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--dev-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from vllm_adapter_export import export_for_vllm  # noqa: PLC0415  (sibling module)

    device = "cuda:0"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    samples, skipped = expand_trajectories(args.sft_data, tokenizer, args.max_len)
    task_ids = sorted({s["task_id"] for s in samples})
    rng = random.Random(args.seed)
    rng.shuffle(task_ids)
    n_dev = max(1, int(len(task_ids) * args.dev_frac))
    dev_ids = set(task_ids[:n_dev])
    train = [s for s in samples if s["task_id"] not in dev_ids]
    dev = [s for s in samples if s["task_id"] in dev_ids]
    print(f"samples: {len(train)} train / {len(dev)} dev turns "
          f"({len(task_ids)} tasks, {skipped} over-length skipped)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa"
    )
    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    # Activation memory is the binding constraint (see --batch-size note):
    # checkpointing recomputes the slow linear-attention path in backward,
    # trading ~2x compute for fitting comfortably on one 80G card.
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    train_loader = DataLoader(
        TurnDataset(train), batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    dev_loader = DataLoader(
        TurnDataset(dev), batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, pad_id),
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / args.warmup)
        * 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / max(total_steps, 1))),
    )

    print(f"training: {total_steps} optimizer steps ({args.epochs} epochs)")
    best = float("inf")
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for micro, (input_ids, labels, attention) in enumerate(train_loader):
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                labels=labels.to(device),
            )
            (out.loss / args.grad_accum).backward()
            if (micro + 1) % args.grad_accum == 0 or micro + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    # flush=True: piped stdout is block-buffered and a running
                    # job looks hung otherwise (same lesson as grpo_rsf_simple).
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {float(out.loss.detach()):.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        loss = dev_loss(model, dev_loader, device)
        print(f"epoch {epoch} dev_loss {loss:.4f}")
        if loss < best:
            best = loss
            model.save_pretrained(str(out_dir / "adapter_best"))
    model.save_pretrained(str(out_dir / "adapter_final"))

    passed, checked = format_check(model, tokenizer, args.tasks, dev_ids, device)
    print(f"first-turn format compliance (dev, greedy): {passed}/{checked}")

    for name in ("adapter_best", "adapter_final"):
        if (out_dir / name).exists():
            export_for_vllm(out_dir / name, out_dir / f"{name}_vllm")
    print(f"done; adapters + vLLM exports in {out_dir}")
    print("SFT_SYNTH_OK")


if __name__ == "__main__":
    main()
