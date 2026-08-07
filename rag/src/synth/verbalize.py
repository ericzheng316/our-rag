"""GPU stage: turn chain specs into natural questions with local Qwen3.5-9B.

Reads a tasks.jsonl produced by run_scripts/60_build_synth_tasks.py, batches
one vLLM pass (thinking disabled), and writes tasks_verbalized.jsonl where
``question``/``question_source`` are replaced for every task that passes the
leakage check; failures keep the deterministic template question.

Validity gates per task, all of which must pass or the template question is
kept:
  * lexical leakage — the question must not contain the answer string or any
    bridge entity name (case-folded containment);
  * form — a single interrogative ending with '?';
  * round-trip answerability (second vLLM pass) — the model, given the
    question plus the UNMASKED gold evidence, must produce the gold answer.
    This is the gate that catches semantically scrambled questions (wrong
    chain direction, swapped relations), which lexical checks cannot see.
    Caveat: the checker is the same base model that will later be trained,
    so this filters for answerable-by-this-model, not answerable-in-general
    — acceptable for cold-start data, revisit for eval sets.

Run under a __main__ guard (vLLM v1 spawns engine-core processes):
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=<free> $PYTHON -m synth.verbalize \
        --tasks .../tasks.jsonl --out .../tasks_verbalized.jsonl
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

MODEL = "Qwen/Qwen3.5-9B"

_PROMPT = """You are writing one quiz question for a retrieval benchmark.

The hidden answer is an entity reached by following this reference chain
through Wikipedia articles. Chain facts (each sentence is from the article
named before the colon; [?] marks the entity the next step must uncover):

{facts}

Write ONE natural-sounding English question whose unique answer is exactly:
{answer}

Hard rules:
- Do NOT mention the answer itself or any of these names: {forbidden}.
- The question must ask for the FINAL entity of the chain (the one reached
  last), never the starting entity.
- Keep every relation faithful to the chain facts exactly as stated — do not
  swap who acquired/wrote/founded what.
- The question must force the solver to follow every step of the chain (use
  descriptive references like "the person who ...", "the organization that ...").
- One sentence, ending with a question mark. Output only the question."""

_CHECK_PROMPT = """Answer the question using ONLY the evidence below.

Evidence:
{evidence}

Question: {question}

Reply with the exact entity name only, nothing else."""


def _facts_block(task: Dict[str, Any]) -> str:
    lines = []
    for hop, (edge, hint) in enumerate(zip(task["chain"], task["slot_hints"])):
        lines.append(f"{hop + 1}. {edge['src_title']}: {hint}")
    return "\n".join(lines)


def _forbidden(task: Dict[str, Any]) -> List[str]:
    return [edge["dst_title"] for edge in task["chain"]]


def leaks(question: str, task: Dict[str, Any]) -> bool:
    lowered = question.casefold()
    if task["answer"].casefold() in lowered:
        return True
    return any(name.casefold() in lowered for name in _forbidden(task))


def _evidence_block(task: Dict[str, Any]) -> str:
    """Unmasked gold evidence, exactly what retrieval would surface."""
    lines = [
        f"- {edge['src_title']}: {edge['sentence']}" for edge in task["chain"]
    ]
    final_req = task["evidence_specification"]["requirements"][-1]
    for unit in final_req["evidence_units"]:
        lines.append(f"- {unit['source_id']}: {unit['text']}")
    return "\n".join(lines)


def _answers_match(model_answer: str, gold: str) -> bool:
    """NFKC-normalize before containment: the model may emit decomposed
    accents (e + U+0301) while corpus titles are precomposed (U+00E9), and
    casefold alone does not unify them."""
    import unicodedata

    a = unicodedata.normalize("NFKC", model_answer).strip().casefold()
    g = unicodedata.normalize("NFKC", gold).strip().casefold()
    return bool(a) and (a in g or g in a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tasks = [json.loads(line) for line in open(args.tasks, encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for task in tasks:
        user = _PROMPT.format(
            facts=_facts_block(task),
            answer=task["answer"],
            forbidden=", ".join(_forbidden(task) + [task["answer"]]),
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192)
    outs = llm.generate(
        prompts, SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    )

    # Pass 1 gates: form + lexical leakage.
    candidates: List[int] = []
    questions: Dict[int, str] = {}
    for idx, (task, out) in enumerate(zip(tasks, outs)):
        question = out.outputs[0].text.strip().split("\n")[0].strip()
        if question.endswith("?") and not leaks(question, task):
            candidates.append(idx)
            questions[idx] = question

    # Pass 2 gate: round-trip answerability against unmasked gold evidence.
    check_prompts = []
    for idx in candidates:
        user = _CHECK_PROMPT.format(
            evidence=_evidence_block(tasks[idx]), question=questions[idx]
        )
        check_prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        )
    check_outs = llm.generate(
        check_prompts, SamplingParams(temperature=0.0, max_tokens=32)
    ) if check_prompts else []

    kept = 0
    failed_check = 0
    for idx, out in zip(candidates, check_outs):
        model_answer = out.outputs[0].text.strip().split("\n")[0].strip()
        passed = _answers_match(model_answer, tasks[idx]["answer"])
        tasks[idx]["verbalize_check"] = {"passed": passed, "model_answer": model_answer}
        if passed:
            tasks[idx]["question"] = questions[idx]
            tasks[idx]["question_source"] = "qwen3.5-9b_verbalized"
            kept += 1
        else:
            failed_check += 1

    with open(args.out, "w", encoding="utf-8") as sink:
        for task in tasks:
            sink.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(
        f"verbalized {kept}/{len(tasks)} tasks "
        f"(lexical-gate fails: {len(tasks) - len(candidates)}, "
        f"round-trip fails: {failed_check}; all fails keep template questions)"
    )


if __name__ == "__main__":
    main()
