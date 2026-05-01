"""
Minimal standalone GRPO + R_sf trainer.
No OpenRLHF dependency — uses transformers + peft only.

R_sf  : marginal supporting-fact title hits per retrieval turn
R_ans : EM exact-match at final answer
GRPO  : group-relative advantage normalization (no critic)
"""

import argparse, json, math, os, re, requests, string
from typing import List, Set, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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
            print(f"[retrieve] {e}")
    return []


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


def grpo_normalize(totals: List[float], eps: float = 1e-8) -> List[float]:
    if len(totals) <= 1:
        return [0.0] * len(totals)
    mu = sum(totals) / len(totals)
    sigma = (sum((r - mu) ** 2 for r in totals) / len(totals)) ** 0.5
    return [(r - mu) / (sigma + eps) for r in totals]

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
        input_ids = tokenizer(context, return_tensors="pt",
                              truncation=True, max_length=1024).input_ids.cuda()

        out = model.generate(
            input_ids,
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
    """
    total_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
    n = 0

    for question_trajs in trajs_per_question:
        r_totals = [sum(r for _, _, r in traj) for traj in question_trajs]
        advantages = grpo_normalize(r_totals)

        for traj, adv in zip(question_trajs, advantages):
            adv_t = torch.tensor(adv, dtype=torch.float32, device="cuda")

            for inp_ids, out_ids, _ in traj:
                if out_ids.numel() == 0:
                    continue

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
                total_loss = total_loss + pg + kl_coef * kl
                n += 1

    return total_loss / max(n, 1)

# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    print(f"[GRPO-RSF] Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Policy model (LoRA) ────────────────────────────────────────────────────
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
        attn_implementation="eager",
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
    print(f"[GRPO-RSF] Training on {len(data)} samples")

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
                torch.cuda.empty_cache()

            trajs_per_question.append(question_trajs)

        # ── Gradient step ──────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        loss = grpo_loss_fn(model, trajs_per_question, kl_coef=args.kl_coef)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.empty_cache()

        global_step += 1
        mean_r = sum(r_totals_log) / max(len(r_totals_log), 1)
        print(f"Episode {episode + 1:4d} | loss={loss.item():.4f} | mean_R={mean_r:.3f} | "
              f"batch={len(batch)}q × {args.n_samples}samples")

        if global_step % args.save_steps == 0:
            ckpt = os.path.join(args.save_path, f"step_{global_step}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"[GRPO-RSF] Saved checkpoint → {ckpt}")

    # Final save
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"[GRPO-RSF] Training done. Final model → {args.save_path}")


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
