"""Strict outcome-only GRPO entry point for the ACEC ablation.

The rollout engine, estimator, optimizer, dataset order, and checkpoint
format are inherited unchanged from ``grpo_rsf_vllm``.  This entry point
removes every process reward and refuses ACEC runtime construction:

    R(trajectory) = normalized terminal answer exact match

Retrieval turns and format failures receive zero immediate reward.  Through
the existing return-to-go estimator, a correct terminal outcome credits all
earlier actions in that rollout without using supporting-fact labels,
coverage posteriors, retrieval cost, or format shaping.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from outcome_only_contract import OUTCOME_ONLY_REWARD, validate_outcome_only_reward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--n_turns", type=int, default=5)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--max_engine_logprob_mae", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--max_samples", type=int, default=1600)
    parser.add_argument("--vllm_gpu_mem_frac", type=float, default=0.60)
    return parser


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_run_contract(args: argparse.Namespace, repo_root: Path) -> Path:
    output_root = Path(args.save_path).resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "run_contract.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prior run contract: {path}")
    payload = {
        "schema": "outcome_only_grpo_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "reward": asdict(OUTCOME_ONLY_REWARD),
        "reward_equation": "terminal_normalized_answer_em",
        "process_signals_disabled": [
            "gold_supporting_fact_coverage",
            "acec_posterior_coverage",
            "retrieval_cost",
            "format_penalty",
        ],
        "training": vars(args),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = build_parser().parse_args()
    if args.num_episodes <= 0 or args.batch_size <= 0 or args.n_samples < 2:
        raise ValueError("episodes/batch_size must be positive and n_samples must be >= 2")
    if args.save_steps <= 0:
        raise ValueError("save_steps must be positive")
    validate_outcome_only_reward(OUTCOME_ONLY_REWARD)

    # Attributes consumed by the shared trainer.  They are assigned here,
    # rather than exposed as CLI knobs, so a baseline run cannot silently
    # acquire process shaping.
    args.lambda_ans = OUTCOME_ONLY_REWARD.answer
    args.lambda_cov = OUTCOME_ONLY_REWARD.coverage
    args.lambda_fmt = OUTCOME_ONLY_REWARD.format_error
    args.retrieval_cost = OUTCOME_ONLY_REWARD.retrieval_cost
    args.use_acec = False

    repo_root = Path(__file__).resolve().parents[2]
    contract = write_run_contract(args, repo_root)
    print(f"[outcome-only] contract={contract}")
    print(f"[outcome-only] reward={asdict(OUTCOME_ONLY_REWARD)}")

    # Heavy CUDA/vLLM imports occur only after contract validation.
    import grpo_rsf_vllm as base_trainer

    base_trainer.train(args)


if __name__ == "__main__":
    main()
