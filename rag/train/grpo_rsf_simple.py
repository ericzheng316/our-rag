"""
Minimal standalone GRPO + R_sf trainer.
No OpenRLHF dependency — uses transformers + peft only. No deepspeed/ray/
flash-attn (attn_implementation="sdpa", built into torch), so this is the
cheap path for a correctness-only smoke test of the turn-level-advantage
wiring; reach for the OpenRLHF path (train/R3RAG_OpenRLHF) once
distributed/multi-GPU training at real scale is actually needed.

attn_implementation must be "sdpa" (or flash-attn), not "eager": R3-RAG-Qwen
is fine-tuned to reproduce an exact literal template ("Step N:\nThe problem
analysis: ...\nThe retrieval query: ..."), and eager attention's bf16
numerics diverge from vLLM's (what inference_new.py actually uses) just
enough, compounding token-by-token, that the model reliably stops emitting
the literal "The problem analysis:" label — 100% format-error rollouts
across every temperature tested (0.001 to 0.9), confirmed by direct A/B
against vLLM (4/4 correct) and sdpa (2/2 correct) on the same prompt. Not a
parsing bug, not a chat-template bug — a numerics-sensitivity issue specific
to eager attention on this narrowly-formatted model.

R_sf  : marginal supporting-fact title hits per retrieval turn
R_ans : EM exact-match at final answer
GRPO  : turn-level advantages (design doc Section 3.2), not one scalar per
        trajectory — see turn_level_returns/turn_level_advantages below.
"""

import argparse, json, logging, math, os, re, requests, string
from typing import List, Set, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# print() to a pipe (`python ... | tee log`) is block-buffered by default —
# lines sit in Python's internal buffer and don't reach the log file until it
# fills or the process exits, making a live run look hung for its whole
# duration. logging.StreamHandler flushes after every record regardless of
# the underlying stream's buffering, so every log call below uses it instead.
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("grpo_rsf")

# ── Env / config ───────────────────────────────────────────────────────────────
RETRIEVE_URL = os.environ.get("RETRIEVE_URL", "")
RETRIEVE_K   = int(os.environ.get("RETRIEVE_K", "5"))
MAX_DOCS     = int(os.environ.get("MAX_DOCS", "10"))

LAMBDA_ANS = 1.0
LAMBDA_SF  = 0.5
LAMBDA_FMT = 0.1
STOP_TOKEN = 151645   # Qwen2.5 <|im_end|>

# ── Reward helpers ─────────────────────────────────────────────────────────────

def _retrieve(query: str) -> List[dict]:
    assert RETRIEVE_URL, "RETRIEVE_URL not set"
    for _ in range(3):
        try:
            r = requests.post(RETRIEVE_URL,
                              json={"querys": [query]},
                              headers={"Content-Type": "application/json"},
                              timeout=30)
            if r.status_code == 200:
                data = r.json()
                return data[0][:RETRIEVE_K] if data and isinstance(data[0], list) else data[:RETRIEVE_K]
        except Exception as e:
            log.warning(f"[retrieve] {e}")
    return []


def _retrieve_batch(queries: List[str]) -> List[List[dict]]:
    """
    Same server call as _retrieve, but for N queries in one request instead
    of N serial requests. retrieve_server.py's /search endpoint natively
    batches (dense_retriever.batch_search over the whole `querys` list, one
    FAISS call), so this is the intended usage — not the same thing as the
    ThreadPoolExecutor concurrency that crashed the retriever in 2026-07-13
    (that was N *simultaneous* independent requests each spinning up their
    own OpenBLAS thread pool; this is one request, processed by the server's
    single fixed 8-thread pool, just with more work in it). Cuts N HTTP
    round-trips down to 1 and lets FAISS batch the distance computation
    instead of N separate matrix-vector products — this is the actual lever
    for retrieval throughput, not adding more training-side GPUs (the
    2026-07-14 head-to-head run's bottleneck was CPU-bound serial FAISS
    search, not GPU compute — see retrive_server.py's faiss_gpu=False).
    """
    if not queries:
        return []
    assert RETRIEVE_URL, "RETRIEVE_URL not set"
    for _ in range(3):
        try:
            r = requests.post(RETRIEVE_URL,
                              json={"querys": queries},
                              headers={"Content-Type": "application/json"},
                              timeout=60)
            if r.status_code == 200:
                data = r.json()
                return [docs[:RETRIEVE_K] for docs in data]
        except Exception as e:
            log.warning(f"[retrieve_batch] {e}")
    return [[] for _ in queries]


def rsf_marginal(doc_text: str, sf_titles: List[str], found: Set[str]) -> Tuple[float, Set[str]]:
    if not sf_titles:
        return 0.0, found
    doc_lower = doc_text.lower()
    newly = {t for t in sf_titles if t not in found and t.lower() in doc_lower}
    return len(newly) / len(sf_titles), found | newly


def exact_match(pred: str, golds: List[str]) -> bool:
    def norm(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = s.translate(str.maketrans("", "", string.punctuation))
        return " ".join(s.split())
    return any(norm(pred) == norm(g) for g in golds)


# ── Turn-level GRPO advantages (design doc Section 3.2) ────────────────────────
# Same semantics as rag/src/belief/acec/reward.py's turn_level_returns /
# turn_level_advantages (verified bit-for-bit identical), duplicated here
# rather than imported so this script keeps its zero-heavy-dependency property
# (acec/__init__.py eager-imports torch/sentence-transformers extras this
# R_sf-only path doesn't need).

def turn_level_returns(rewards_per_turn: List[float]) -> List[float]:
    """G_{i,t} = sum_{t' >= t} r_{i,t'} for one rollout."""
    returns = [0.0] * len(rewards_per_turn)
    running = 0.0
    for t in range(len(rewards_per_turn) - 1, -1, -1):
        running += rewards_per_turn[t]
        returns[t] = running
    return returns


def turn_level_advantages(group_returns: List[List[float]], eps: float = 1e-6) -> List[List[float]]:
    """
    A_{i,t} = (G_{i,t} - mu_t) / (sigma_t + eps), mu_t/sigma_t over rollouts
    still alive at turn t. Falls back to the raw return when fewer than 2
    rollouts remain alive at t.
    """
    max_len = max((len(r) for r in group_returns), default=0)
    advantages = [[0.0] * len(r) for r in group_returns]
    for t in range(max_len):
        alive = [(i, r[t]) for i, r in enumerate(group_returns) if len(r) > t]
        if len(alive) < 2:
            for i, val in alive:
                advantages[i][t] = val
            continue
        vals = [v for _, v in alive]
        mu = sum(vals) / len(vals)
        sigma = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        for i, v in alive:
            advantages[i][t] = (v - mu) / (sigma + eps)
    return advantages

# ── Response parser (mirrors inference_new.py) ─────────────────────────────────

def parse_step(text: str) -> dict:
    d = {}
    for key, tag in [("analysis", "The problem analysis:"),
                     ("query",    "The retrieval query:"),
                     ("answer",   "The final answer:")]:
        idx = text.find(tag)
        if idx == -1:
            continue
        start = idx + len(tag)
        end = len(text)
        for stop in ["The problem analysis:", "The retrieval query:",
                     "The final answer:", "The retrieval documents:", "###"]:
            si = text.find(stop, start)
            if si != -1 and si < end:
                end = si
        val = text[start:end].strip().strip("\\n").strip("#")
        if val:
            d[key] = val
    return d

# ── Chat template ──────────────────────────────────────────────────────────────
# R3-RAG-Qwen was fine-tuned to expect its input wrapped in this Qwen chat
# template (matches ApplyChatTemplate in the OpenRLHF experience makers /
# inference_new.py, ported locally rather than imported to keep this script
# dependency-light). Without it the model doesn't reliably produce the
# "Step N:\nThe problem analysis: ..." format at all — every turn falls
# through to the format-error branch, which looks like the model "always
# answers in one turn" (avg_turns stuck at 1.00) when it's really just never
# emitting parseable output.

def apply_chat_template(user_content: str) -> str:
    # "You are a helpful assistant" with no trailing period, matching the
    # system message inference_new.py actually sends — confirmed identical to
    # tokenizer.apply_chat_template()'s own default aside from this, and this
    # model is sensitive enough to exact template text that it's worth matching.
    return (
        "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# ── Single trajectory rollout ──────────────────────────────────────────────────

@torch.no_grad()
def rollout(model, tokenizer, question: str, sf_titles: List[str],
            golden_answers: List[str], n_turns: int = 5,
            temperature: float = 0.7) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
    """
    Returns list of (input_ids, output_ids, step_reward) for each turn.
    input_ids  : context tokens fed to model this turn (1D, on GPU)
    output_ids : newly generated tokens (1D, on GPU)
    step_reward: R_sf_t for query steps, R_ans for answer step
    """
    turns = []
    context = f"The question: {question}"
    retrieved_ids: List[str] = []
    found_sf: Set[str] = set()

    for t in range(n_turns):
        encoding = tokenizer(apply_chat_template(context), return_tensors="pt",
                             truncation=True, max_length=1024)
        input_ids = encoding.input_ids.cuda()

        out = model.generate(
            input_ids,
            attention_mask=encoding.attention_mask.cuda(),
            max_new_tokens=512,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=STOP_TOKEN,
        )
        new_tokens = out[0, input_ids.shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        step_text = f"Step {t + 1}:\n{output_text}"
        d = parse_step(step_text)

        if not d.get("analysis"):
            turns.append((input_ids[0], new_tokens, -LAMBDA_FMT))
            break

        if d.get("answer"):
            r_ans = LAMBDA_ANS if exact_match(d["answer"], golden_answers) else 0.0
            turns.append((input_ids[0], new_tokens, r_ans))
            break

        if d.get("query"):
            docs = _retrieve(d["query"])
            new_docs = []
            for doc in docs:
                if doc.get("id") not in retrieved_ids and len(retrieved_ids) < MAX_DOCS:
                    retrieved_ids.append(doc["id"])
                    new_docs.append(doc.get("contents", ""))
            doc_text = "\n".join(new_docs)

            r_sf, found_sf = rsf_marginal(doc_text, sf_titles, found_sf)
            turns.append((input_ids[0], new_tokens, LAMBDA_SF * r_sf))

            step_str = (f"Step {t + 1}:\nThe problem analysis: {d['analysis']}\n"
                        f"The retrieval query: {d['query']}\n"
                        f"The retrieval documents: {doc_text[:512]}")
            context = context + "\n" + step_str

            if t == n_turns - 1:
                # Ran out of turns — small penalty for not answering
                turns[-1] = (turns[-1][0], turns[-1][1], turns[-1][2] - LAMBDA_FMT)
        else:
            turns.append((input_ids[0], new_tokens, -LAMBDA_FMT))
            break

    return turns

# ── GRPO loss ──────────────────────────────────────────────────────────────────

def grpo_loss_fn(model, trajs_per_question: List[List[List]], kl_coef: float = 0.01):
    """
    trajs_per_question : list (over questions) of lists (over G samples) of
                         list of (input_ids, output_ids, reward) tuples

    Uses a single model: LoRA enabled for policy, disabled for reference.
    This halves VRAM vs loading two separate model copies.

    Calls .backward() per turn and returns a float (not a loss tensor) —
    accumulating one autograd graph across a whole episode (up to
    batch_size x n_samples x n_turns forward passes) OOMs at the real
    design-doc scale (G=8, lora_rank=64: hit CUDA OOM on the very first
    episode, 2026-07-14). Backward per turn frees each turn's graph
    immediately; gradients accumulate in .grad across calls (matches
    autograd's default behavior), then get scaled by 1/n at the end so the
    net effect on the gradient is identical to the old single
    backward-on-the-mean call. Caller must NOT call loss.backward() again —
    just optimizer.step() after this returns.
    """
    total_loss = 0.0
    n = 0

    for question_trajs in trajs_per_question:
        reward_lists = [[r for _, _, r in traj] for traj in question_trajs]
        group_returns = [turn_level_returns(rl) for rl in reward_lists]
        group_advantages = turn_level_advantages(group_returns)

        for traj, advantages in zip(question_trajs, group_advantages):
            for t, (inp_ids, out_ids, _) in enumerate(traj):
                if out_ids.numel() == 0:
                    continue

                # A_{i,t}: per-turn advantage, not one scalar for the whole
                # trajectory — see turn_level_advantages docstring.
                adv_t = torch.tensor(advantages[t], dtype=torch.float32, device="cuda")

                full = torch.cat([inp_ids, out_ids]).unsqueeze(0)   # (1, L)
                n_out = out_ids.shape[0]

                # ── policy log-probs (LoRA enabled) ───────────────────────────
                model.enable_adapter_layers()
                logits = model(full).logits                           # (1, L, V)
                lp = F.log_softmax(logits[0], dim=-1)                # (L, V)
                pred_lp = lp[-(n_out + 1):-1]                       # (n_out, V)
                token_lp = pred_lp.gather(1, out_ids.unsqueeze(1)).squeeze(1)  # (n_out,)

                # ── ref log-probs (LoRA disabled → base model) ────────────────
                with torch.no_grad():
                    model.disable_adapter_layers()
                    ref_logits = model(full).logits
                    ref_lp = F.log_softmax(ref_logits[0], dim=-1)
                    ref_pred_lp = ref_lp[-(n_out + 1):-1]
                    ref_token_lp = ref_pred_lp.gather(1, out_ids.unsqueeze(1)).squeeze(1)
                model.enable_adapter_layers()

                pg  = -adv_t * token_lp.mean()
                kl  = (token_lp - ref_token_lp).mean()   # approx KL
                turn_loss = pg + kl_coef * kl
                turn_loss.backward()
                total_loss += turn_loss.item()
                n += 1

    for p in model.parameters():
        if p.grad is not None:
            p.grad /= max(n, 1)

    return total_loss / max(n, 1)

# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    log.info(f"[GRPO-RSF] Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Policy model (LoRA) ────────────────────────────────────────────────────
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

    # ── Data ──────────────────────────────────────────────────────────────────
    ds = load_dataset("json", data_files=args.data_path)["train"]
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    data = list(ds)
    log.info(f"[GRPO-RSF] Training on {len(data)} samples")

    os.makedirs(args.save_path, exist_ok=True)
    global_step = 0

    for episode in range(args.num_episodes):
        # Sample a batch of questions
        start = (episode * args.batch_size) % len(data)
        batch = data[start: start + args.batch_size]
        if not batch:
            batch = data[:args.batch_size]

        trajs_per_question = []
        r_totals_log = []
        turn_counts_log = []   # sanity dashboard: turns per rollout

        for item in batch:
            sf    = item.get("supporting_facts") or {}
            sf_titles = sf.get("title", []) if isinstance(sf, dict) else []
            golds = item["golden_answers"]
            q     = item["question"]

            question_trajs = []
            for g in range(args.n_samples):
                temp = 0.3 + g * 0.2       # diverse temperatures: 0.3, 0.5, 0.7, 0.9
                model.eval()
                traj = rollout(model, tokenizer, q, sf_titles, golds,
                               n_turns=args.n_turns, temperature=temp)
                question_trajs.append(traj)
                r_totals_log.append(sum(r for _, _, r in traj))
                turn_counts_log.append(len(traj))
                torch.cuda.empty_cache()

            trajs_per_question.append(question_trajs)

        # ── Gradient step ──────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        loss = grpo_loss_fn(model, trajs_per_question, kl_coef=args.kl_coef)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.empty_cache()

        global_step += 1
        mean_r = sum(r_totals_log) / max(len(r_totals_log), 1)
        avg_turns = sum(turn_counts_log) / max(len(turn_counts_log), 1)
        log.info(f"Episode {episode + 1:4d} | loss={loss:.4f} | mean_R={mean_r:.3f} | "
                 f"avg_turns={avg_turns:.2f} | batch={len(batch)}q × {args.n_samples}samples")

        if global_step % args.save_steps == 0:
            ckpt = os.path.join(args.save_path, f"step_{global_step}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            log.info(f"[GRPO-RSF] Saved checkpoint → {ckpt}")

    # Final save
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    log.info(f"[GRPO-RSF] Training done. Final model → {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--data_path",    type=str, required=True)
    parser.add_argument("--save_path",    type=str, default="/root/logs/grpo_rsf_ckpt")
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--batch_size",   type=int, default=4,   help="questions per episode")
    parser.add_argument("--n_samples",    type=int, default=4,   help="rollouts per question (G)")
    parser.add_argument("--n_turns",      type=int, default=5,   help="max retrieval turns")
    parser.add_argument("--lora_rank",    type=int, default=16)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--kl_coef",      type=float, default=0.01)
    parser.add_argument("--save_steps",   type=int, default=50)
    parser.add_argument("--max_samples",  type=int, default=None)
    args = parser.parse_args()
    train(args)
