"""Engine-pool correctness pass: N engines, SFT adapter, 16 live episodes.

Validates, in order:
  1. race-tolerant GPU picking (gpu_pick strict-idle double sample);
  2. pool init: one vLLM engine per picked GPU (spawn workers);
  3. the mandatory anti-silent-no-op lora_check;
  4. 16 multi-turn episodes in lockstep turn batching, distractor-mode
     retrieval over each task's closed pool, SFT adapter driving;
  5. writes the FIRST v8-schema rollout log
     (`episodes.jsonl`) — the artifact Phase-2 replay consumes.

Also prints the first honest throughput number (episodes/min, turns/min)
for the 4-vs-8-card decision.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rag" / "src"))
sys.path.insert(0, str(REPO_ROOT / "rag" / "train"))

MODEL = "Qwen/Qwen3.5-9B"
ADAPTER = "/scratch/boyuz5/acec/checkpoints/sft_synth_v0/adapter_best_vllm"
TASKS = "/scratch/boyuz5/acec/synth/full_v1/tasks_final.jsonl"
OUT_DIR = Path("/scratch/boyuz5/acec/engine_pool_smoke")


def main() -> None:
    from transformers import AutoTokenizer
    from agent.episode import pool_retriever, run_episodes_batched
    from synth.verbalize import _answers_match
    from engine_pool import EnginePool
    from gpu_pick import pick_gpus

    n_want = 2
    gpus = pick_gpus(n_want, min_free_gib=70.0, retries=2, retry_wait_s=10.0)
    print(f"picked GPUs: {gpus}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def chat(messages):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    t0 = time.time()
    pool = EnginePool(MODEL, gpus, max_lora_rank=32)
    print(f"pool up ({len(gpus)} engines) in {time.time()-t0:.0f}s", flush=True)

    probe = chat([{"role": "user", "content": "Briefly describe the city of Paris."}])
    assert pool.lora_check(probe, ADAPTER, adapter_version=1), \
        "LORA_CHECK FAILED — adapter is a silent no-op, do not proceed"
    print("lora_check: PASS", flush=True)

    tasks = [json.loads(line) for line in open(TASKS, encoding="utf-8")]
    random.Random(11).shuffle(tasks)
    tasks = tasks[:16]

    def batched_generate(prompts):
        results = pool.generate(
            prompts, {"temperature": 0.0, "max_tokens": 320},
            adapter_path=ADAPTER, adapter_version=1,
        )
        return [r["text"] for r in results]

    t1 = time.time()
    episodes = run_episodes_batched(
        tasks, batched_generate, pool_retriever, chat, max_turns=8, docs_per_search=3
    )
    wall = time.time() - t1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "episodes.jsonl", "w", encoding="utf-8") as sink:
        for episode in episodes:
            sink.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    n = len(episodes)
    em = sum(
        1 for e in episodes
        if e.final_answer and _answers_match(e.final_answer, e.gold_answer)
    )
    proto_err = sum(1 for e in episodes if e.protocol_error)
    limit = sum(1 for e in episodes if e.hit_turn_limit)
    total_turns = sum(len(e.turns) for e in episodes)
    avg_searches = sum(e.n_searches for e in episodes) / n
    print(f"episodes: {n} | answered-correct (containment): {em}/{n} "
          f"| protocol errors: {proto_err} | turn-limit: {limit}", flush=True)
    print(f"avg searches/episode: {avg_searches:.2f}", flush=True)
    print(f"wall: {wall:.0f}s -> {n / wall * 60:.1f} episodes/min, "
          f"{total_turns / wall * 60:.0f} turns/min on {len(gpus)} engines", flush=True)
    print(f"rollout log -> {OUT_DIR/'episodes.jsonl'}", flush=True)
    pool.shutdown()
    print("ENGINE_POOL_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
