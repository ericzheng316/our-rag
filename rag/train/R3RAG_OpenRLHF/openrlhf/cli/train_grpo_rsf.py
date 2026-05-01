"""
GRPO + R_sf training entry point.

Differences from train_ppo.py:
  1. Uses PromptDatasetRsf (adds supporting_facts to each prompt).
  2. Replaces the trainer's NaiveExperienceMaker with GRPORsfExperienceMaker.
  3. No critic / reward model needed.
"""

import argparse
import itertools
import math
import os
from datetime import datetime

import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.datasets.prompts_dataset_rsf import PromptDatasetRsf
from openrlhf.models import Actor
from openrlhf.trainer import PPOTrainer
from openrlhf.trainer.ppo_utils.experience_maker_grpo_rsf import GRPORsfExperienceMaker
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # ── Actor only (no critic, no reward model for GRPO) ──────────────────────
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    initial_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=False),
    )

    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )

    tokenizer = get_tokenizer(
        args.pretrain, actor.model, "left", strategy,
        use_fast=not args.disable_fast_tokenizer,
    )

    # ── Dataset (with supporting_facts) ───────────────────────────────────────
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=False,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDatasetRsf(prompts_data, tokenizer, strategy)

    pretrain_dataloader = None
    if args.pretrain_data:
        pretrain_data = blending_datasets(
            args.pretrain_data, args.pretrain_data_probs, strategy, args.seed,
            return_eval=False, train_split=args.pretrain_split,
        )
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data.select(range(min(
                len(pretrain_data),
                args.max_epochs * len(prompts_dataset) * args.n_samples_per_prompt,
            ))),
            tokenizer, pretrain_max_len, strategy, pretrain_mode=True,
        )
        pretrain_dataloader = itertools.cycle(iter(strategy.setup_dataloader(
            pretrain_dataset, args.micro_train_batch_size, True, True,
            pretrain_dataset.collate_fn,
        )))

    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset, args.rollout_batch_size // strategy.world_size, True, True,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    num_update_steps_per_episodes = (
        len(prompts_dataset) * args.n_samples_per_prompt
        // args.train_batch_size * args.max_epochs
    )
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
    actor_scheduler = get_scheduler(
        "cosine_with_min_lr",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )

    (
        (actor, actor_optim, actor_scheduler),
        (_, _, _),          # critic placeholder (None triple)
        _,                  # reward_model placeholder
        initial_model,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        (None, None, None),
        None,
        initial_model,
        is_rlhf=True,
    )

    os.makedirs(args.save_path, exist_ok=True)

    # ── Trainer: swap in GRPORsfExperienceMaker ───────────────────────────────
    trainer = PPOTrainer(
        strategy,
        actor,
        critic=None,
        reward_model=None,
        initial_model=initial_model,
        ema_model=None,
        actor_optim=actor_optim,
        critic_optim=None,
        actor_scheduler=actor_scheduler,
        critic_scheduler=None,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remote_rm_url=None,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    # Replace experience maker after trainer construction (avoids subclassing PPOTrainer)
    from openrlhf.trainer.ppo_utils.kl_controller import FixedKLController, AdaptiveKLController
    if trainer.kl_target:
        kl_ctl = AdaptiveKLController(args.init_kl_coef, trainer.kl_target, getattr(args, "kl_horizon", 10000))
    else:
        kl_ctl = FixedKLController(args.init_kl_coef)

    trainer.experience_maker = GRPORsfExperienceMaker(
        actor=actor,
        critic=None,
        reward_model=None,
        initial_model=initial_model,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        kl_controller=kl_ctl,
        strategy=strategy,
        remote_rm_url=None,
        reward_fn=None,
    )

    trainer.fit(args, prompts_dataloader, pretrain_dataloader, 0, num_update_steps_per_episodes)

    strategy.save_model(actor, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO + R_sf training")

    # Model
    parser.add_argument("--pretrain", type=str, required=True)
    parser.add_argument("--critic_pretrain", type=str, default=None)  # unused, kept for compat
    parser.add_argument("--reward_pretrain", type=str, default=None)  # unused
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)

    # Optimization
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=2048)
    parser.add_argument("--generate_max_len", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--ptx_coef", type=float, default=0)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.2)
    parser.add_argument("--lambd", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1)
    parser.add_argument("--n_samples_per_prompt", type=int, default=4)
    parser.add_argument("--save_value_network", action="store_true", default=False)
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))

    # KL
    parser.add_argument("--init_kl_coef", type=float, default=0.01)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--kl_horizon", type=int, default=10000)
    parser.add_argument("--use_kl_estimator_k3", action="store_true", default=False)

    # Learning rates
    parser.add_argument("--actor_learning_rate", type=float, default=5e-7)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)  # unused
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)

    # Precision
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # Flash attention / gradient checkpointing
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # Dataset
    parser.add_argument("--prompt_data", type=str, required=True)
    parser.add_argument("--prompt_data_probs", type=str, default="1.0")
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None)
    parser.add_argument("--pretrain_data_probs", type=str, default="1.0")
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="question")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)

    # Distributed / DeepSpeed
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--advantage_estimator", type=str, default="reinforce",
                        choices=["gae", "reinforce", "rloo"])

    # Logging
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="grpo_rsf")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--use_tensorboard", type=str, default=None)

    # EMA / aux loss
    parser.add_argument("--enable_ema", action="store_true", default=False)
    parser.add_argument("--aux_loss_coef", type=float, default=0)

    # Packing
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # Remote RM (unused but required by PPOTrainer compat)
    parser.add_argument("--remote_rm_url", type=str, default=None)
    parser.add_argument("--reward_clip_range", type=float, default=None)

    args = parser.parse_args()
    train(args)
