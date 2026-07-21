"""Isolated GRPO entry point for the ACEC v5 marginal-utility runtime.

The optimized rollout and estimator implementation stays in
``grpo_rsf_vllm.py``.  This entry point replaces only its belief factory, so
v1-v4 command lines and runtime semantics remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Callable

import grpo_rsf_vllm as base_trainer


REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
RAG_SRC = os.path.join(REPO_ROOT, "rag", "src")
if RAG_SRC not in sys.path:
    sys.path.insert(0, RAG_SRC)


def _load_checked_artifact(path: str) -> Any:
    from belief.acec.calibration_v5 import load_calibration_artifact_v5

    artifact = load_calibration_artifact_v5(path)
    gate = dict(artifact.metrics.get("gate") or {})
    if gate.get("pass") is not True:
        raise ValueError("ACEC v5 runtime refuses an artifact whose validity gate did not pass")
    if not math.isfinite(float(artifact.binding_threshold)):
        raise ValueError("ACEC v5 artifact has a non-finite binding threshold")
    return artifact


def make_belief_factory_v5(args: argparse.Namespace) -> Callable[[str], Any]:
    from belief.obs_extractor import E5Embedder
    from belief.acec import ACECConfig, CrossEncoderNLIScorer, E5QueryEmbedder
    from belief.acec.calibration_v2 import FixedKPredictor
    from belief.acec.k_posterior import UniformKPredictor
    from belief.acec.k_strategy_v5 import build_k_predictor_v5
    from belief.acec.offline_fit import hit_rates_to_beta_priors
    from belief.acec.runtime_v5 import ACECBeliefStateV5

    artifact = _load_checked_artifact(args.acec_artifact_v5)
    base_trainer.log.info(
        f"[GRPO-RSF-vLLM-v5] Loading E5 action embedder: {args.e5_model_path}"
    )
    embedder = E5QueryEmbedder(E5Embedder(args.e5_model_path))
    base_trainer.log.info(
        f"[GRPO-RSF-vLLM-v5] Loading NLI cross-encoder: {args.acec_nli_model}"
    )
    nli_scorer = CrossEncoderNLIScorer(args.acec_nli_model)

    config = ACECConfig()
    calibrated_tau_new = artifact.metadata.get("tau_new")
    if calibrated_tau_new is None:
        raise ValueError("ACEC v5 artifact is missing calibrated tau_new")
    config.tau_new = float(calibrated_tau_new)
    if args.acec_tau_new is not None:
        config.tau_new = float(args.acec_tau_new)
    config.bound_threshold = float(artifact.binding_threshold)
    config.hit_prior_alpha0, config.hit_prior_beta0 = hit_rates_to_beta_priors(
        artifact.gain_rates
    )

    strategy = artifact.k_strategy
    if strategy.mode == "fixed":
        if strategy.fixed_k is None:
            raise ValueError("fixed ACEC v5 K strategy is missing fixed_k")
        # Fixed K is a structural runtime support, not merely a one-hot prior
        # inside a larger K_max.  This prevents a spurious extra slot from
        # truncating away all prior mass and triggering KPosterior fallback.
        config.k_max = int(strategy.fixed_k)
        k_predictor = FixedKPredictor(config.k_max, config.k_max)
    elif strategy.mode == "uniform":
        config.k_max = int(strategy.k_max)
        k_predictor = UniformKPredictor(config.k_max)
    else:
        config.k_max = int(strategy.k_max)
        k_predictor = build_k_predictor_v5(strategy, embedder)

    calibration_nli = artifact.metadata.get("nli_model")
    if calibration_nli and str(calibration_nli) != str(args.acec_nli_model):
        base_trainer.log.warning(
            "[GRPO-RSF-vLLM-v5] WARNING: runtime NLI identifier differs from "
            f"calibration metadata ({args.acec_nli_model!r} vs {calibration_nli!r})"
        )
    base_trainer.log.info(
        "[GRPO-RSF-vLLM-v5] Loaded gate-passing artifact "
        f"tau_new={config.tau_new} binding={config.bound_threshold:.6f} "
        f"k_mode={strategy.mode} k_max={config.k_max}"
    )

    def factory(question: str) -> ACECBeliefStateV5:
        belief = ACECBeliefStateV5(
            embedder,
            nli_scorer,
            artifact.observation_model,
            config=config,
            k_predictor=k_predictor,
        )
        belief.reset(question)
        return belief

    return factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRPO-v2 with the isolated ACEC v5 marginal-utility runtime"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="/root/logs/grpo_rsf_vllm_v5_ckpt")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--n_turns", type=int, default=5)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument(
        "--max_engine_logprob_mae",
        "--max_old_logprob_mae",
        dest="max_engine_logprob_mae",
        type=float,
        default=None,
    )
    parser.add_argument("--lambda_ans", type=float, default=base_trainer.LAMBDA_ANS)
    parser.add_argument("--lambda_cov", type=float, default=base_trainer.LAMBDA_COV)
    parser.add_argument("--lambda_fmt", type=float, default=base_trainer.LAMBDA_FMT)
    parser.add_argument(
        "--retrieval_cost", type=float, default=base_trainer.RETRIEVAL_COST
    )
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--vllm_gpu_mem_frac", type=float, default=0.45)
    parser.add_argument("--acec_artifact_v5", type=str, required=True)
    parser.add_argument(
        "--e5_model_path",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "models", "e5-base-v2"),
    )
    parser.add_argument(
        "--acec_nli_model",
        type=str,
        default="cross-encoder/nli-deberta-v3-base",
    )
    parser.add_argument("--acec_tau_new", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.use_acec = True
    _load_checked_artifact(args.acec_artifact_v5)

    from belief.acec.runtime_v5 import RUNTIME_DIAGNOSTICS_V5

    RUNTIME_DIAGNOSTICS_V5.reset()
    base_trainer.make_belief_factory = make_belief_factory_v5
    base_trainer.train(args)
    print(
        "[GRPO-RSF-vLLM-v5] runtime_metrics="
        + json.dumps(RUNTIME_DIAGNOSTICS_V5.summary(), sort_keys=True)
    )


if __name__ == "__main__":
    main()
