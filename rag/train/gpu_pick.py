"""Race-tolerant GPU selection for the shared idealab boxes.

The naive "pick a card that looks idle" one-liner lost a real race on
2026-08-04 (card grabbed between pick and engine start; a 0.85-utilization
vLLM engine needs ~67 GiB free).  Rules here:

  * strict idle = < 200 MiB used AND < 5% utilization, sampled twice with a
    short gap (a card someone is actively loading onto fails the second
    sample);
  * required headroom is stated by the caller in GiB and re-checked at claim
    time;
  * callers that spawn engines should still treat engine-init OOM as a
    retryable event (pick again, excluding the failed card) — this module
    shrinks the race window, nothing can close it on a shared box.
"""

from __future__ import annotations

import subprocess
import time
from typing import List, Optional, Sequence


def _query() -> List[tuple]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        idx, used, total, util = [int(x.strip()) for x in line.split(",")]
        rows.append((idx, used, total, util))
    return rows


def free_gpus(
    min_free_gib: float = 70.0,
    samples: int = 2,
    gap_s: float = 2.0,
    exclude: Sequence[int] = (),
) -> List[int]:
    """Indices that look strictly idle across ``samples`` observations."""
    candidates: Optional[set] = None
    for attempt in range(samples):
        if attempt:
            time.sleep(gap_s)
        now = set()
        for idx, used, total, util in _query():
            if idx in exclude:
                continue
            free_gib = (total - used) / 1024.0
            if used < 200 and util < 5 and free_gib >= min_free_gib:
                now.add(idx)
        candidates = now if candidates is None else (candidates & now)
    return sorted(candidates or [])


def pick_gpus(
    n: int,
    min_free_gib: float = 70.0,
    retries: int = 3,
    retry_wait_s: float = 20.0,
    exclude: Sequence[int] = (),
) -> List[int]:
    """Pick ``n`` strictly-idle GPUs or raise after retries."""
    tried_exclude = list(exclude)
    for _ in range(retries):
        cards = free_gpus(min_free_gib=min_free_gib, exclude=tried_exclude)
        if len(cards) >= n:
            return cards[:n]
        time.sleep(retry_wait_s)
    raise RuntimeError(
        f"could not find {n} idle GPUs with >= {min_free_gib} GiB free "
        f"(excluded {tried_exclude}); node is busy — rerun later or lower "
        "gpu_memory_utilization"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=70.0)
    args = parser.parse_args()
    print(",".join(str(i) for i in pick_gpus(args.n, args.min_free_gib)))
