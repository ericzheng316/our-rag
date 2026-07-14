"""
GRPO + R_sf trainer — vLLM for rollout generation, lightweight HF+PEFT for
the gradient step. Single GPU.

Why this exists: grpo_rsf_simple.py generates rollouts one at a time via
HF transformers.generate() — correct (see its docstring for the eager-vs-
sdpa saga this project already hit) but ~17s/rollout serial, measured on a
20-sample/2-episode/G=4 dry run (32 rollouts, ~9 minutes). Design doc
D11-14's smoke run (100 steps, 5k prompts, G=8) is ~6400 rollouts — at that
rate, ~30 hours on a single GPU. vLLM already confirmed, on this exact
model, to be both fast (sub-second for several samples) and *correct*
(reliably reproduces R3-RAG-Qwen's literal "Step N:\nThe problem analysis:
..." template, unlike eager attention) — this script routes rollout
generation through it instead, while keeping the gradient step identical
to grpo_rsf_simple.py.

Architecture:
  - vLLM (enable_lora=True) holds ONE frozen copy of the base model and
    generates all rollouts for an episode in batched, turn-boundary calls:
    within a turn, every still-active rollout (across all questions x all
    G samples) is generated in one vllm.generate() call; rollouts that
    finish (answer / format-error) drop out before the next turn. vLLM's
    own scheduler handles the batching internally — no hand-rolled
    continuous-batching/padding logic here.
  - After every gradient step, the just-updated LoRA adapter is saved to
    disk and handed to vLLM as a *new* LoRARequest (fresh lora_int_id +
    path each step) for the next episode's rollouts — this is vLLM's
    built-in multi-LoRA serving doing the hot-swap, no engine restart.
  - The actual policy/reference log-probs + backward pass still run on a
    *separate*, regular HF+PEFT model (via the imported grpo_loss_fn,
    unchanged from grpo_rsf_simple.py) — vLLM's own forward pass is
    generation-only and never part of the gradient.

UNTESTED on real hardware as of authoring (the GPU box was off). Known risk
areas to check on the first real run:
  - vLLM's LoRA support for target_modules="all-linear" — historically
    vLLM's LoRA layer coverage has been narrower than PEFT's for some
    modules; if this errors, may need to scope target_modules down to
    ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    instead of "all-linear" on both the PEFT config and the vLLM engine.
  - max_lora_rank must be >= lora_rank and vLLM only supports certain rank
    values (commonly a fixed set like 8/16/32/64) — check the error message
    if construction fails.
  - GPU memory headroom for holding both the vLLM engine and the separate
    HF+PEFT training model at once — tune --vllm_gpu_mem_frac down if OOM.
  - Token id fidelity: this script takes input_ids/output_ids directly from
    vLLM's RequestOutput (prompt_token_ids / token_ids) rather than
    re-tokenizing decoded text, on the assumption vLLM uses the exact same
    tokenizer as the HF+PEFT side (same model_path) — should be safe, but
    worth a spot-check against grpo_rsf_simple.py's token ids on first run.

VALIDATED (2026-07-13): the vLLM+LoRA-hotswap generation/training loop
itself ran end to end (see experiments/2026-07-13_grpo_rsf_vllm_first_
validation/) — the risk areas above turned out fine. --use_acec below is
new, unvalidated code layered on top of that working base.

--use_acec: reward mode. Design doc Section 8 risk #1 calls ablation (b)
(gold-SF reward, no belief — the default here) "the most dangerous row in
the paper": it might already capture most of the achievable gain, in which
case belief's marginal value is narrow. Running (b) alone doesn't test
that risk either way — only running the belief-shaped R_cov arm does. When
set, each rollout gets its own ACECBeliefState (no gold labels used at
rollout time, per the design's own principle); reward per retrieval turn
becomes potential_shaped_reward(coverage_before, coverage_after) instead of
the gold supporting-facts marginal. Needs three more things loaded onto the
same GPU (E5Embedder for the action labeler, the NLI cross-encoder, and
optionally the Week-1 calibrated observation model) — real added GPU memory
and per-turn latency (an extra NLI forward pass per retrieved doc) on top
of an already GPU-bound pipeline; tune --vllm_gpu_mem_frac down if tight.
The Week-1 calibration artifact (run_scripts/15_build_acec_calibration.sh's
--acec_observation_model output, observation_model.json) was NOT found
under $HOME/logs/acec_calibration_pilot/ on this persistent volume as of
2026-07-13 (empty directory) despite CLAUDE.md recording a passing Week-1
gate — that calibration may have been produced on a different box/run
that was never copied here. Verify it exists (or re-run 14/15) before
trusting --acec_observation_model; omitting it falls back to uncalibrated
ACECConfig() defaults with a printed warning, which is a real regression
from the calibrated Week-1 numbers, not a silent equivalent.
"""

import argparse, os, shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from grpo_rsf_simple import (
    LAMBDA_ANS, LAMBDA_FMT, LAMBDA_SF, MAX_DOCS, STOP_TOKEN,
    _retrieve, apply_chat_template, exact_match, grpo_loss_fn, parse_step,
    rsf_marginal,
)

BeliefFactory = Callable[[str], Any]   # str (question) -> ACECBeliefState


def make_belief_factory(args) -> BeliefFactory:
    """
    Lazily imports rag/src/belief/acec (only when --use_acec is set, so the
    default gold-SF path stays exactly as dependency-light as before) and
    returns a per-question ACECBeliefState factory, mirroring the exact
    construction inference_new.py already validated for the Week-2 VOI gate
    probe (same NLI scorer, same calibration-loading path).
    """
    import sys
    rag_src = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    sys.path.insert(0, rag_src)

    from belief.obs_extractor import E5Embedder
    from belief.acec import ACECBeliefState, ACECConfig, CrossEncoderNLIScorer, E5QueryEmbedder
    from belief.acec.offline_fit import hit_rates_to_beta_priors, load_calibrated_observation_model

    print(f"[GRPO-RSF-vLLM] [ACEC] Loading E5Embedder for action labeler: {args.e5_model_path}")
    acec_embedder = E5QueryEmbedder(E5Embedder(args.e5_model_path))

    print(f"[GRPO-RSF-vLLM] [ACEC] Loading NLI cross-encoder: {args.acec_nli_model}")
    acec_nli_scorer = CrossEncoderNLIScorer(args.acec_nli_model)

    acec_config = ACECConfig()
    if args.acec_observation_model:
        obs_model, hit_rates = load_calibrated_observation_model(args.acec_observation_model)
        acec_config.hit_prior_alpha0, acec_config.hit_prior_beta0 = hit_rates_to_beta_priors(hit_rates)
        print(f"[GRPO-RSF-vLLM] [ACEC] Loaded calibrated observation model from {args.acec_observation_model}")
    else:
        obs_model = None
        print("[GRPO-RSF-vLLM] [ACEC] WARNING: no --acec_observation_model given — "
              "using UNCALIBRATED ACECConfig() defaults (not the Week-1 gate's fitted "
              "tau_new/hit-rate priors). This is a real regression, not a safe fallback.")
    if args.acec_tau_new is not None:
        acec_config.tau_new = args.acec_tau_new

    def factory(question: str):
        belief = ACECBeliefState(acec_embedder, acec_nli_scorer, config=acec_config, obs_model=obs_model)
        belief.reset(question)
        return belief

    return factory

# ── Batched, turn-boundary rollout collection via vLLM ──────────────────────────

def vllm_rollout_batch(
    llm: LLM,
    lora_request: LoRARequest,
    questions: List[Dict],
    n_samples: int,
    n_turns: int = 5,
    belief_factory: Optional[BeliefFactory] = None,
) -> List[List[List[Tuple[torch.Tensor, torch.Tensor, float]]]]:
    """
    questions: list of {"question": str, "sf_titles": List[str],
                         "golden_answers": List[str]}

    Returns trajs_per_question: list (over questions) of lists (over G
    samples) of lists (over turns) of (input_ids, output_ids, reward)
    tuples — exactly grpo_loss_fn's expected shape, and exactly what
    grpo_rsf_simple.py's rollout() returns per-rollout, just collected
    turn-batched across the whole (questions x n_samples) pool instead of
    one rollout at a time.

    belief_factory: when given, per-turn reward for retrieval steps comes
    from ACECBeliefState.reward() (potential-shaped coverage delta) instead
    of the gold supporting-facts marginal — see module docstring's
    --use_acec section. Each rollout gets its own independent belief state
    (same lifetime as its context/retrieved_ids).
    """
    n_q = len(questions)
    state = []
    for qi, item in enumerate(questions):
        for g in range(n_samples):
            state.append({
                "qi": qi,
                "context": f"The question: {item['question']}",
                "retrieved_ids": [],
                "found_sf": set(),
                "belief": belief_factory(item["question"]) if belief_factory else None,
                "temperature": 0.3 + g * 0.2,   # same diversity schedule as grpo_rsf_simple.py
                "turns": [],
                "done": False,
            })

    for t in range(n_turns):
        active = [i for i, s in enumerate(state) if not s["done"]]
        if not active:
            break

        prompts = [apply_chat_template(state[i]["context"]) for i in active]
        sampling_params = [
            SamplingParams(temperature=state[i]["temperature"], max_tokens=512,
                            stop_token_ids=[STOP_TOKEN])
            for i in active
        ]
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)

        pending_retrieval = []   # indices into `active` that need a retrieval call this turn
        for j, i in enumerate(active):
            s = state[i]
            completion = outputs[j].outputs[0]
            input_ids = torch.tensor(outputs[j].prompt_token_ids, dtype=torch.long, device="cuda")
            new_ids = torch.tensor(completion.token_ids, dtype=torch.long, device="cuda")

            step_text = f"Step {t + 1}:\n{completion.text}"
            d = parse_step(step_text)

            if not d.get("analysis"):
                # Format error is a code-level penalty, orthogonal to reward
                # mode — no belief.turn() call either way, there's no action
                # to label when the model didn't produce parseable output.
                s["turns"].append((input_ids, new_ids, -LAMBDA_FMT))
                s["done"] = True
                continue

            if d.get("answer"):
                golds = questions[s["qi"]]["golden_answers"]
                r_ans = LAMBDA_ANS if exact_match(d["answer"], golds) else 0.0
                if s["belief"] is not None:
                    result = s["belief"].turn(query=None, new_docs=[], is_answer=True)
                    r_ans = s["belief"].reward(result, answer_reward=r_ans)
                s["turns"].append((input_ids, new_ids, r_ans))
                s["done"] = True
                continue

            if d.get("query"):
                s["_pending"] = (input_ids, new_ids, d)
                pending_retrieval.append(i)
            else:
                s["turns"].append((input_ids, new_ids, -LAMBDA_FMT))
                s["done"] = True

        # Retrieval calls, serial. Tried concurrent.futures.ThreadPoolExecutor
        # here first (up to 16 workers) on the theory that these are I/O-bound
        # HTTP calls, not GPU work — that many simultaneous requests crashed
        # the retriever server instead ("BLAS: Program is Terminated. Because
        # you tried to allocate too many memory regions.", retrieve_server.py
        # runs single-request OpenBLAS-threaded E5 encoding, OMP/OPENBLAS/MKL_
        # NUM_THREADS=8 each — it was never built to take concurrent requests,
        # and retrieval was never the bottleneck grpo_rsf_vllm.py exists to
        # fix in the first place, so just don't risk it.
        if pending_retrieval:
            queries = [state[i]["_pending"][2]["query"] for i in pending_retrieval]
            docs_list = [_retrieve(q) for q in queries]

            for i, docs in zip(pending_retrieval, docs_list):
                s = state[i]
                input_ids, new_ids, d = s.pop("_pending")
                sf_titles = questions[s["qi"]]["sf_titles"]

                new_doc_dicts = []
                for doc in docs:
                    if doc.get("id") not in s["retrieved_ids"] and len(s["retrieved_ids"]) < MAX_DOCS:
                        s["retrieved_ids"].append(doc["id"])
                        new_doc_dicts.append(doc)
                doc_text = "\n".join(doc.get("contents", "") for doc in new_doc_dicts)

                if s["belief"] is not None:
                    # No gold labels here by design (ACECBeliefState's own
                    # principle) — new_doc_dicts (raw {"contents": ...}
                    # dicts, not the concatenated doc_text string) go straight
                    # to the NLI-scored coverage update.
                    result = s["belief"].turn(query=d["query"], new_docs=new_doc_dicts)
                    r_turn = s["belief"].reward(result)
                else:
                    r_sf, s["found_sf"] = rsf_marginal(doc_text, sf_titles, s["found_sf"])
                    r_turn = LAMBDA_SF * r_sf
                s["turns"].append((input_ids, new_ids, r_turn))

                step_str = (f"Step {t + 1}:\nThe problem analysis: {d['analysis']}\n"
                            f"The retrieval query: {d['query']}\n"
                            f"The retrieval documents: {doc_text[:512]}")
                s["context"] = s["context"] + "\n" + step_str

                if t == n_turns - 1:
                    last = s["turns"][-1]
                    s["turns"][-1] = (last[0], last[1], last[2] - LAMBDA_FMT)
                    s["done"] = True

    trajs_per_question: List[List[List]] = [[] for _ in range(n_q)]
    for s in state:
        trajs_per_question[s["qi"]].append(s["turns"])
    return trajs_per_question

# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[GRPO-RSF-vLLM] Loading vLLM engine (generation only): {args.model_path}")
    llm = LLM(
        model=args.model_path,
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        max_loras=2,
        gpu_memory_utilization=args.vllm_gpu_mem_frac,
        max_model_len=4096,
        dtype="bfloat16",
    )

    print(f"[GRPO-RSF-vLLM] Loading HF+PEFT model (gradient step only): {args.model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="sdpa",
    )
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules="all-linear", lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )

    belief_factory = make_belief_factory(args) if args.use_acec else None

    ds = load_dataset("json", data_files=args.data_path)["train"]
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    data = list(ds)
    print(f"[GRPO-RSF-vLLM] Training on {len(data)} samples")

    os.makedirs(args.save_path, exist_ok=True)
    lora_scratch_dir = os.path.join(args.save_path, "_lora_scratch")
    os.makedirs(lora_scratch_dir, exist_ok=True)

    # Save the untrained adapter so vLLM has something to load for episode 0's
    # rollouts. PEFT LoRA B matrices are zero-initialized, so this is
    # mathematically identical to the frozen base model — matches what
    # grpo_rsf_simple.py's first episode does too.
    global_step = 0
    prev_lora_dir = os.path.join(lora_scratch_dir, f"step_{global_step}")
    model.save_pretrained(prev_lora_dir)
    lora_request = LoRARequest(f"policy-{global_step}", global_step + 1, prev_lora_dir)

    for episode in range(args.num_episodes):
        start = (episode * args.batch_size) % len(data)
        batch = data[start: start + args.batch_size]
        if not batch:
            batch = data[:args.batch_size]

        questions = []
        for item in batch:
            sf = item.get("supporting_facts") or {}
            questions.append({
                "question": item["question"],
                "sf_titles": sf.get("title", []) if isinstance(sf, dict) else [],
                "golden_answers": item["golden_answers"],
            })

        trajs_per_question = vllm_rollout_batch(
            llm, lora_request, questions, args.n_samples, args.n_turns,
            belief_factory=belief_factory,
        )

        r_totals_log = [sum(r for _, _, r in traj) for qt in trajs_per_question for traj in qt]
        turn_counts_log = [len(traj) for qt in trajs_per_question for traj in qt]

        model.train()
        optimizer.zero_grad()
        loss = grpo_loss_fn(model, trajs_per_question, kl_coef=args.kl_coef)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.empty_cache()

        global_step += 1
        mean_r = sum(r_totals_log) / max(len(r_totals_log), 1)
        avg_turns = sum(turn_counts_log) / max(len(turn_counts_log), 1)
        print(f"Episode {episode + 1:4d} | loss={loss:.4f} | mean_R={mean_r:.3f} | "
              f"avg_turns={avg_turns:.2f} | batch={len(batch)}q × {args.n_samples}samples")

        # Hot-swap: save the just-updated adapter under a fresh id/path (not
        # load_inplace on a reused path — untested live, prefer the
        # unambiguous "new id + new path every step" form) and hand it to
        # vLLM for the next episode's rollouts.
        new_lora_dir = os.path.join(lora_scratch_dir, f"step_{global_step}")
        model.save_pretrained(new_lora_dir)
        lora_request = LoRARequest(f"policy-{global_step}", global_step + 1, new_lora_dir)
        shutil.rmtree(prev_lora_dir, ignore_errors=True)
        prev_lora_dir = new_lora_dir

        if global_step % args.save_steps == 0:
            ckpt = os.path.join(args.save_path, f"step_{global_step}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"[GRPO-RSF-vLLM] Saved checkpoint → {ckpt}")

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    shutil.rmtree(lora_scratch_dir, ignore_errors=True)
    print(f"[GRPO-RSF-vLLM] Training done. Final model → {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--data_path",    type=str, required=True)
    parser.add_argument("--save_path",    type=str, default="/root/logs/grpo_rsf_vllm_ckpt")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--batch_size",   type=int, default=8,   help="questions per episode")
    parser.add_argument("--n_samples",    type=int, default=8,   help="rollouts per question (G)")
    parser.add_argument("--n_turns",      type=int, default=5,   help="max retrieval turns")
    parser.add_argument("--lora_rank",    type=int, default=64)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--kl_coef",      type=float, default=0.01)
    parser.add_argument("--save_steps",   type=int, default=25)
    parser.add_argument("--max_samples",  type=int, default=5000)
    parser.add_argument("--vllm_gpu_mem_frac", type=float, default=0.45,
                         help="fraction of GPU memory for the vLLM engine; "
                              "the rest is left for the separate HF+PEFT "
                              "training model. Tune down if OOM.")
    parser.add_argument("--use_acec", action="store_true", default=False,
                         help="Use ACECBeliefState's potential-shaped coverage "
                              "reward (R_cov) instead of the gold supporting-"
                              "facts marginal for retrieval-turn rewards. See "
                              "module docstring's --use_acec section.")
    parser.add_argument("--e5_model_path", type=str,
                         default=os.path.join(os.path.expanduser("~"), "models", "e5-base-v2"),
                         help="Only used when --use_acec is set (ACEC's action labeler).")
    parser.add_argument("--acec_nli_model", type=str, default="cross-encoder/nli-deberta-v3-base",
                         help="Only used when --use_acec is set.")
    parser.add_argument("--acec_observation_model", type=str, default=None,
                         help="Path to the Week-1 calibration's observation_model.json "
                              "(run_scripts/15_build_acec_calibration.sh's --out_dir). "
                              "Only used when --use_acec is set; omitting it falls back "
                              "to uncalibrated ACECConfig() defaults (see module docstring).")
    parser.add_argument("--acec_tau_new", type=float, default=None,
                         help="Optional override for ACECConfig.tau_new. Only used when "
                              "--use_acec is set.")
    args = parser.parse_args()
    train(args)
