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
"""

import argparse, os, shutil
from typing import Dict, List, Tuple

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

# ── Batched, turn-boundary rollout collection via vLLM ──────────────────────────

def vllm_rollout_batch(
    llm: LLM,
    lora_request: LoRARequest,
    questions: List[Dict],
    n_samples: int,
    n_turns: int = 5,
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
                s["turns"].append((input_ids, new_ids, -LAMBDA_FMT))
                s["done"] = True
                continue

            if d.get("answer"):
                golds = questions[s["qi"]]["golden_answers"]
                r_ans = LAMBDA_ANS if exact_match(d["answer"], golds) else 0.0
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

                new_docs = []
                for doc in docs:
                    if doc.get("id") not in s["retrieved_ids"] and len(s["retrieved_ids"]) < MAX_DOCS:
                        s["retrieved_ids"].append(doc["id"])
                        new_docs.append(doc.get("contents", ""))
                doc_text = "\n".join(new_docs)

                r_sf, s["found_sf"] = rsf_marginal(doc_text, sf_titles, s["found_sf"])
                s["turns"].append((input_ids, new_ids, LAMBDA_SF * r_sf))

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
        )

        r_totals_log = [sum(r for _, _, r in traj) for qt in trajs_per_question for traj in qt]
        turn_counts_log = [len(traj) for qt in trajs_per_question for traj in qt]

        model.train()
        optimizer.zero_grad()
        loss = grpo_loss_fn(model, trajs_per_question, kl_coef=args.kl_coef)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.empty_cache()

        global_step += 1
        mean_r = sum(r_totals_log) / max(len(r_totals_log), 1)
        avg_turns = sum(turn_counts_log) / max(len(turn_counts_log), 1)
        print(f"Episode {episode + 1:4d} | loss={loss.item():.4f} | mean_R={mean_r:.3f} | "
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
    args = parser.parse_args()
    train(args)
