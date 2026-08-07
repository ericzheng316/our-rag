"""Closed-book screen: can the base model answer WITHOUT retrieval?

Fills each task's ``closed_book`` field and prints the dataset-level
retrieval-necessity numbers that the narrative depends on ("questions are
unanswerable from parametric knowledge by construction" is a claim this
script turns into a measurement).

Per task:  closed_book = {
    "model":   the screened model id,
    "answers": every sampled closed-book answer,
    "correct": True if ANY sample matches the gold answer (conservative:
               one parametric hit marks the task as closed-book answerable),
}

Default is one greedy sample; ``--n-samples 4 --temperature 0.7`` gives a
stricter pass@4-style screen for eval-slice construction.  Matching reuses
verbalize._answers_match (NFKC + casefold containment).

Run under a __main__ guard (vLLM v1 spawns engine-core processes):
    CUDA_VISIBLE_DEVICES=<free> $PYTHON rag/src/synth/screen.py \
        --tasks .../tasks_verbalized.jsonl --out .../tasks_screened.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, List

try:
    from .verbalize import MODEL, _answers_match
except ImportError:  # direct-path invocation: python rag/src/synth/screen.py
    from verbalize import MODEL, _answers_match

_PROMPT = """{question}

Answer from your own knowledge, without searching. Reply with the exact
entity name only, nothing else. If you do not know, reply exactly: UNKNOWN"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 with n-samples=1 is the greedy screen; use "
                             "0.7 with n-samples>1 for a pass@k-style screen")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tasks: List[Dict[str, Any]] = [json.loads(line) for line in open(args.tasks, encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": _PROMPT.format(question=task["question"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for task in tasks
    ]

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096)
    outs = llm.generate(
        prompts,
        SamplingParams(temperature=args.temperature, max_tokens=32, n=args.n_samples),
    )

    n_correct = 0
    by_k: Counter = Counter()
    by_k_correct: Counter = Counter()
    by_source_correct: Counter = Counter()
    with open(args.out, "w", encoding="utf-8") as sink:
        for task, out in zip(tasks, outs):
            answers = [c.text.strip().split("\n")[0].strip() for c in out.outputs]
            correct = any(
                answer.upper() != "UNKNOWN" and _answers_match(answer, task["answer"])
                for answer in answers
            )
            task["closed_book"] = {"model": MODEL, "answers": answers, "correct": correct}
            by_k[task["k"]] += 1
            if correct:
                n_correct += 1
                by_k_correct[task["k"]] += 1
                by_source_correct[task["question_source"]] += 1
            sink.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"closed-book answerable: {n_correct}/{len(tasks)} "
          f"({100.0 * n_correct / max(len(tasks), 1):.1f}%) "
          f"[n_samples={args.n_samples}, T={args.temperature}]")
    for k in sorted(by_k):
        print(f"  K={k}: {by_k_correct.get(k, 0)}/{by_k[k]}")
    if by_source_correct:
        print("  by question_source:", dict(by_source_correct))


if __name__ == "__main__":
    main()
