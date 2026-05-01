"""
Standalone IRCoT baseline for HotpotQA.

Algorithm (Trivedi et al. ACL 2023):
  for each question:
    retrieve top-K docs using the question
    repeat up to max_turns:
      generate one CoT sentence given [docs, question, prev_cot]
      if "answer is" in sentence → extract answer, stop
      else retrieve K more docs using last CoT sentence as query

Runs all questions in a batched fashion: each turn processes
all unfinished questions together, minimising GPU idle time.

LLM:  Qwen2.5-7B-Instruct via vllm (chat mode)
Retriever: our FAISS server  POST {retrieve_url}  {"querys": [...]}
Input:  FlashRAG hotpotqa/dev.jsonl  (fields: id, question, golden_answers)
Output: JSONL with id, question, golden_answers, pred, cot, num_turns
"""

import argparse
import json
import os
import re
import time
import requests
from tqdm import tqdm
from vllm import LLM, SamplingParams

ANSWER_RE = re.compile(r"(?:so |therefore |thus )?the answer is[:\s]+(.+?)\.?\s*$", re.I)
POST_BATCH = 64   # passages per retrieval HTTP call


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_corpus_entry(contents: str):
    """Split FlashRAG contents into (title, paragraph_text).
    Format: '"Title"\nParagraph text …'
    """
    parts = contents.split("\n", 1)
    title = parts[0].strip().strip('"')
    para  = parts[1].strip() if len(parts) > 1 else ""
    return title, para


def retrieve_batch(retrieve_url: str, queries: list[str], topk: int) -> list[list[dict]]:
    """Call the FAISS retriever server with a batch of queries.
    Returns list of lists: outer = queries, inner = topk passage dicts
    with keys 'title', 'paragraph_text', 'id', 'score'.
    """
    all_results: list[list[dict]] = [[] for _ in queries]
    for start in range(0, len(queries), POST_BATCH):
        batch = queries[start:start + POST_BATCH]
        resp = requests.post(retrieve_url, json={"querys": batch},
                             headers={"Content-Type": "application/json"}, timeout=120)
        resp.raise_for_status()
        raw = resp.json()   # list of lists of dicts with 'id','contents','score'
        for qi, hits in enumerate(raw):
            parsed = []
            for h in hits[:topk]:
                title, para = parse_corpus_entry(h.get("contents", ""))
                parsed.append({
                    "id":             h.get("id", ""),
                    "title":          title,
                    "paragraph_text": para,
                    "score":          h.get("score", 0.0),
                })
            all_results[start + qi] = parsed
    return all_results


def build_context_str(docs: list[dict], max_words_per_para: int = 200) -> str:
    parts = []
    for d in docs:
        para = " ".join(d["paragraph_text"].split()[:max_words_per_para])
        parts.append(f"Wikipedia Title: {d['title']}\n{para}")
    return "\n\n".join(parts)


def build_messages(question: str, context_str: str, cot_so_far: str) -> list[dict]:
    system = (
        "You are a helpful research assistant. Answer multi-hop questions step by step. "
        "You will be given Wikipedia passages and must reason one sentence at a time. "
        'End your reasoning when you know the answer by writing "So the answer is: <answer>."'
    )
    user_parts = []
    if context_str:
        user_parts.append(f"Passages:\n{context_str}")
    user_parts.append(f"Question: {question}")
    if cot_so_far:
        user_parts.append(
            f"Reasoning so far: {cot_so_far}\n\n"
            "Continue with exactly ONE more reasoning sentence. "
            'If you can now answer, write "So the answer is: <answer>." as the last sentence.'
        )
    else:
        user_parts.append(
            "Begin your step-by-step reasoning with ONE sentence. "
            'If you can already answer, write "So the answer is: <answer>."'
        )
    return [
        {"role": "system",  "content": system},
        {"role": "user",    "content": "\n\n".join(user_parts)},
    ]


def extract_answer(sentence: str):
    m = ANSWER_RE.search(sentence)
    if m:
        ans = m.group(1).strip().rstrip(".")
        return ans
    return None


def dedup_docs(existing: list[dict], new_docs: list[dict]) -> list[dict]:
    seen_ids = {d["id"] for d in existing}
    added = [d for d in new_docs if d["id"] not in seen_ids]
    return existing + added


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--retrieve_url",  required=True,
                        help="e.g. http://10.0.0.1:8001/search")
    parser.add_argument("--dev_file",      required=True,
                        help="FlashRAG hotpotqa/dev.jsonl")
    parser.add_argument("--output_file",   required=True)
    parser.add_argument("--topk",          type=int, default=5)
    parser.add_argument("--max_turns",     type=int, default=6)
    parser.add_argument("--max_new_tokens",type=int, default=150)
    parser.add_argument("--tp",            type=int, default=1)
    args = parser.parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=["\n\n", "Question:", "Passages:"],
    )

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Loading data: {args.dev_file}")
    records = []
    with open(args.dev_file) as f:
        for line in f:
            d = json.loads(line)
            records.append({
                "id":             d["id"],
                "question":       d["question"],
                "golden_answers": d.get("golden_answers", []),
            })
    print(f"  {len(records)} questions loaded")

    # ── resume if interrupted ─────────────────────────────────────────────────
    done_ids: set[str] = set()
    out_f = open(args.output_file, "a")
    if os.path.exists(args.output_file):
        with open(args.output_file) as rf:
            for line in rf:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    records = [r for r in records if r["id"] not in done_ids]
    print(f"  {len(records)} remaining after resume")

    # ── per-question state ────────────────────────────────────────────────────
    states = []
    for r in records:
        states.append({
            "id":             r["id"],
            "question":       r["question"],
            "golden_answers": r["golden_answers"],
            "docs":           [],       # accumulated retrieved passages
            "cot":            [],       # CoT sentences generated so far
            "pred":           None,     # final answer (None = not yet found)
            "done":           False,
        })

    active_idx = list(range(len(states)))

    # ── IRCoT main loop ───────────────────────────────────────────────────────
    for turn in range(args.max_turns):
        if not active_idx:
            break
        print(f"\n[{time.strftime('%H:%M:%S')}] Turn {turn+1}/{args.max_turns} — "
              f"{len(active_idx)} active questions")

        # 1. Build retrieval queries
        queries = []
        for i in active_idx:
            s = states[i]
            if s["cot"]:
                q = s["cot"][-1]   # last CoT sentence as query
            else:
                q = s["question"]
            queries.append(q)

        # 2. Batch retrieve
        print(f"  Retrieving (topk={args.topk}) …")
        retrieved = retrieve_batch(args.retrieve_url, queries, args.topk)
        for li, i in enumerate(active_idx):
            states[i]["docs"] = dedup_docs(states[i]["docs"], retrieved[li])

        # 3. Build prompts and generate
        prompts = []
        for i in active_idx:
            s = states[i]
            ctx = build_context_str(s["docs"])
            cot_str = " ".join(s["cot"])
            msgs = build_messages(s["question"], ctx, cot_str)
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        print(f"  Generating for {len(prompts)} questions …")
        outputs = llm.generate(prompts, sampling)

        # 4. Update state
        still_active = []
        for li, i in enumerate(active_idx):
            text = outputs[li].outputs[0].text.strip()
            # Take only first sentence
            first_sent = re.split(r"(?<=[.!?])\s", text)[0] if text else text
            states[i]["cot"].append(first_sent)

            ans = extract_answer(first_sent)
            if ans:
                states[i]["pred"] = ans
                states[i]["done"] = True
            else:
                still_active.append(i)

        active_idx = still_active

    # ── finalize ──────────────────────────────────────────────────────────────
    # Force-extract answer from last CoT for unfinished questions
    for i in range(len(states)):
        if not states[i]["done"] or states[i]["pred"] is None:
            # Try to extract from any CoT sentence
            for sent in reversed(states[i]["cot"]):
                ans = extract_answer(sent)
                if ans:
                    states[i]["pred"] = ans
                    break
            if states[i]["pred"] is None:
                # Fall back: last CoT sentence as-is
                states[i]["pred"] = states[i]["cot"][-1] if states[i]["cot"] else ""

    # ── write results ─────────────────────────────────────────────────────────
    for s in states:
        result = {
            "id":             s["id"],
            "question":       s["question"],
            "golden_answers": s["golden_answers"],
            "pred":           s["pred"],
            "cot":            s["cot"],
            "num_turns":      len(s["cot"]),
        }
        out_f.write(json.dumps(result) + "\n")
    out_f.close()

    print(f"\n[{time.strftime('%H:%M:%S')}] Done. Results → {args.output_file}")
    print(f"  Total questions: {len(records)}")
    turns = [len(s['cot']) for s in states]
    print(f"  Avg turns: {sum(turns)/len(turns):.2f}")


if __name__ == "__main__":
    main()
