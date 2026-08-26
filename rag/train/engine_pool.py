"""Data-parallel vLLM engine pool: N rollout engines, one per GPU.

Architecture (per the 2026-08 distribution decision: distribute rollouts,
not gradients):

  driver process ──req queue──▶ worker 0 (GPU a, vLLM engine)
        ▲                      worker 1 (GPU b, vLLM engine)   ...
        └──────rsp queue◀──────┘

  * each worker pins its GPU via CUDA_VISIBLE_DEVICES *before* importing
    torch/vllm (spawn context), hosts one engine, and serves generate
    requests from a shared queue;
  * adapter hot-swap: every request names (adapter_version, adapter_path);
    workers build a fresh LoRARequest per version — same mechanism the
    single-engine grpo_rsf_vllm validated, broadcast via filesystem;
  * generate() shards a prompt batch round-robin and reassembles order;
  * behavioral discipline: use ``lora_check()`` after wiring changes — it
    asserts adapter-vs-base first-token logprobs actually differ (the
    silent-no-op trap from 2026-08-04).

The pool serves TEXT generation only; training-side logprob/backward stays
in the driver's HF+PEFT model (not managed here).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class _GenRequest:
    req_id: int
    prompts: List[str]
    sampling: Dict[str, Any]
    adapter_version: int
    adapter_path: Optional[str]
    want_logprobs: bool = False


def _worker_main(
    gpu_index: int,
    model: str,
    engine_kwargs: Dict[str, Any],
    req_queue: mp.Queue,
    rsp_queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(model=model, **engine_kwargs)
        rsp_queue.put(("ready", gpu_index, None))
    except Exception:  # noqa: BLE001
        rsp_queue.put(("init_error", gpu_index, traceback.format_exc()))
        return

    while True:
        request = req_queue.get()
        if request is None:
            break
        try:
            lora_request = None
            if request.adapter_path is not None:
                lora_request = LoRARequest(
                    f"adapter_v{request.adapter_version}",
                    request.adapter_version,
                    request.adapter_path,
                )
            params = SamplingParams(**request.sampling)
            if request.want_logprobs:
                params.logprobs = 1
            outs = llm.generate(request.prompts, params, lora_request=lora_request,
                                use_tqdm=False)
            payload = []
            for out in outs:
                first = out.outputs[0]
                logprob = None
                if request.want_logprobs and first.logprobs:
                    logprob = next(iter(first.logprobs[0].values())).logprob
                payload.append({"text": first.text, "first_logprob": logprob})
            rsp_queue.put(("ok", request.req_id, payload))
        except Exception:  # noqa: BLE001
            rsp_queue.put(("error", request.req_id, traceback.format_exc()))


class EnginePool:
    def __init__(
        self,
        model: str,
        gpu_indices: Sequence[int],
        max_lora_rank: int = 32,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        init_timeout_s: float = 600.0,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.gpu_indices = list(gpu_indices)
        ctx = mp.get_context("spawn")
        self._req_queue: mp.Queue = ctx.Queue()
        self._rsp_queue: mp.Queue = ctx.Queue()
        engine_kwargs = {
            "dtype": "bfloat16",
            "enable_lora": True,
            "max_lora_rank": max_lora_rank,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
        }
        # 显式播种引擎采样 RNG(vLLM 引擎级 seed);None 保持 vLLM 默认,
        # 与既往 run 行为一致
        if seed is not None:
            engine_kwargs["seed"] = seed
        # daemon=False is load-bearing: vLLM v1 engines spawn their own
        # engine-core child processes, and daemonic processes may not have
        # children. Cleanup is handled by shutdown() + the atexit hook.
        self._workers = [
            ctx.Process(
                target=_worker_main,
                args=(gpu, model, engine_kwargs, self._req_queue, self._rsp_queue),
                daemon=False,
            )
            for gpu in self.gpu_indices
        ]
        self._shut_down = False
        import atexit
        atexit.register(self.shutdown)
        for worker in self._workers:
            worker.start()
        deadline = time.time() + init_timeout_s
        ready = 0
        while ready < len(self._workers):
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError("engine pool init timeout")
            status, gpu, detail = self._rsp_queue.get(timeout=remaining)
            if status == "init_error":
                raise RuntimeError(f"engine on GPU {gpu} failed to init:\n{detail}")
            ready += 1
        self._next_req_id = 0

    def generate(
        self,
        prompts: List[str],
        sampling: Optional[Dict[str, Any]] = None,
        adapter_path: Optional[str] = None,
        adapter_version: int = 0,
        want_logprobs: bool = False,
    ) -> List[Dict[str, Any]]:
        """Shard ``prompts`` across workers; returns results in input order."""
        if not prompts:
            return []
        sampling = dict(sampling or {"temperature": 0.0, "max_tokens": 320})
        n_workers = len(self._workers)
        shards: List[List[int]] = [[] for _ in range(n_workers)]
        for position in range(len(prompts)):
            shards[position % n_workers].append(position)
        pending = {}
        for shard in shards:
            if not shard:
                continue
            req_id = self._next_req_id
            self._next_req_id += 1
            pending[req_id] = shard
            self._req_queue.put(_GenRequest(
                req_id, [prompts[i] for i in shard], sampling,
                adapter_version, adapter_path, want_logprobs,
            ))
        results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
        while pending:
            status, req_id, payload = self._rsp_queue.get()
            if status == "error":
                raise RuntimeError(f"worker generate failed:\n{payload}")
            for position, item in zip(pending.pop(req_id), payload):
                results[position] = item
        return results  # type: ignore[return-value]

    def lora_check(self, prompt: str, adapter_path: str, adapter_version: int) -> bool:
        """Anti-silent-no-op assertion: adapter must change the distribution."""
        sampling = {"temperature": 0.0, "max_tokens": 8}
        base = self.generate([prompt], sampling, want_logprobs=True)[0]
        tuned = self.generate([prompt], sampling, adapter_path=adapter_path,
                              adapter_version=adapter_version, want_logprobs=True)[0]
        if base["first_logprob"] is None or tuned["first_logprob"] is None:
            return base["text"] != tuned["text"]
        return (abs(base["first_logprob"] - tuned["first_logprob"]) > 1e-3
                or base["text"] != tuned["text"])

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        for _ in self._workers:
            self._req_queue.put(None)
        for worker in self._workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()
