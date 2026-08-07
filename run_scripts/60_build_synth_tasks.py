"""Build synthetic multi-hop tasks from the local wiki18 corpus (CPU only).

Stages: title index -> mention-edge mining -> K-hop chain sampling -> task
assembly (canonical v5 evidence manifest + closed candidate pool + template
question).  GPU stages (verbalize, SFT render, closed-book screen) are
separate scripts; see rag/src/synth/__init__.py for the map.

Smoke:  python 60_build_synth_tasks.py --max-lines 400000 --n-tasks 20 \
            --k 2,3,4 --out-dir /scratch/$USER/acec/synth/smoke
Full corpus mining is an hours-scale single-core pass — shard by --line-range
or reuse a mined edges.jsonl via --edges (both TODO markers below).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rag" / "src"))

from synth.corpus import CorpusIndex, load_chunks  # noqa: E402
from synth.graph import mine_mentions, sample_chains  # noqa: E402
from synth.tasks import SYNTH_VERSION, build_task, pool_candidate_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=os.path.expanduser("~/acec/flashrag_datasets/retrieval-corpus/wiki18_100w.jsonl"),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-lines", type=int, default=None,
                        help="corpus prefix to index/mine (smoke); omit for full corpus")
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--k", default="2,3,4", help="comma list of K values to sample")
    parser.add_argument("--n-distractors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-in-degree", type=int, default=12)
    parser.add_argument("--max-chunk-count", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    k_choices = [int(k) for k in args.k.split(",")]

    t0 = time.time()
    print(f"[1/5] building title index (max_lines={args.max_lines}) ...")
    index = CorpusIndex(args.corpus, max_lines=args.max_lines).build()
    print(f"      {len(index.title_chunks)} linkable titles ({time.time()-t0:.0f}s)")

    print("[2/5] mining mention edges ...")
    edges = mine_mentions(
        args.corpus, index, max_lines=args.max_lines,
        out_path=str(out_dir / "edges.jsonl"),
    )
    print(f"      {len(edges)} edges ({time.time()-t0:.0f}s)")

    print(f"[3/5] sampling chains (K in {k_choices}) ...")
    chains = list(
        sample_chains(
            edges, index, n_chains=args.n_tasks * 2, k_choices=k_choices, rng=rng,
            max_in_degree=args.max_in_degree, max_chunk_count=args.max_chunk_count,
        )
    )
    print(f"      {len(chains)} chains sampled")

    print("[4/5] loading pool chunks (second corpus pass) ...")
    wanted = set()
    for chain in chains:
        wanted.update(pool_candidate_ids(chain, index))
    chunks = load_chunks(args.corpus, wanted, max_lines=args.max_lines)
    print(f"      {len(chunks)}/{len(wanted)} chunks materialized ({time.time()-t0:.0f}s)")

    print("[5/5] assembling tasks ...")
    tasks, k_hist = [], {}
    for chain in chains:
        task = build_task(chain, index, chunks, rng, n_distractors=args.n_distractors)
        if task is None:
            continue
        tasks.append(task)
        k_hist[task["k"]] = k_hist.get(task["k"], 0) + 1
        if len(tasks) >= args.n_tasks:
            break

    tasks_path = out_dir / "tasks.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as sink:
        for task in tasks:
            sink.write(json.dumps(task, ensure_ascii=False) + "\n")

    stats = {
        "synth_version": SYNTH_VERSION,
        "corpus": args.corpus,
        "max_lines": args.max_lines,
        "linkable_titles": len(index.title_chunks),
        "edges": len(edges),
        "chains_sampled": len(chains),
        "tasks": len(tasks),
        "k_histogram": dict(sorted(k_hist.items())),
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as sink:
        json.dump(stats, sink, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(tasks)} tasks -> {tasks_path}")


if __name__ == "__main__":
    main()
