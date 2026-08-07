"""Mention-edge mining and K-hop chain sampling.

An edge (src_title -> dst_title) exists when a chunk of the article
``src_title`` contains a lexical mention of the linkable title ``dst_title``.
Edges carry the exact sentence containing the mention: that sentence is the
gold evidence unit for the hop, and (mention-masked) the slot hint shown to
the agent.  Chains prefer low-in-degree, low-chunk-count destinations so the
resulting question is answerable only by actually walking the chain.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from .corpus import CorpusIndex, extract_mention_spans, normalize_title, parse_corpus_line, sentences


@dataclass(frozen=True)
class MentionEdge:
    src_title: str        # normalized
    dst_title: str        # normalized
    src_chunk_id: str
    mention_text: str     # surface form as it appears in the sentence
    sentence: str         # the sentence containing the mention


def mine_mentions(
    corpus_path: str,
    index: CorpusIndex,
    max_lines: Optional[int] = None,
    max_edges_per_chunk: int = 4,
    out_path: Optional[str] = None,
) -> List[MentionEdge]:
    """One streaming pass over the corpus emitting mention edges.

    Runtime is regex-bound (~a few ms per chunk); the full 21M-chunk corpus is
    an hours-scale single-core job, so real runs shard by line range and this
    function accepts ``max_lines`` for the smoke path.  With ``out_path`` the
    edges are also appended to a jsonl for reuse across sampling runs.
    """
    edges: List[MentionEdge] = []
    sink = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        with open(corpus_path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if max_lines is not None and line_no >= max_lines:
                    break
                chunk_id, title, text = parse_corpus_line(line)
                src_key = normalize_title(title)
                if src_key not in index.title_chunks:
                    continue
                found = 0
                for sentence in sentences(text):
                    if found >= max_edges_per_chunk:
                        break
                    for span in extract_mention_spans(sentence):
                        dst_key = index.lookup(span)
                        if dst_key is None or dst_key == src_key:
                            continue
                        edge = MentionEdge(src_key, dst_key, chunk_id, span, sentence)
                        edges.append(edge)
                        if sink:
                            sink.write(json.dumps(edge.__dict__, ensure_ascii=False) + "\n")
                        found += 1
                        if found >= max_edges_per_chunk:
                            break
    finally:
        if sink:
            sink.close()
    return edges


def _group_by_src(edges: List[MentionEdge]) -> Dict[str, List[MentionEdge]]:
    grouped: Dict[str, List[MentionEdge]] = {}
    for edge in edges:
        grouped.setdefault(edge.src_title, []).append(edge)
    return grouped


def _in_degree(edges: List[MentionEdge]) -> Dict[str, int]:
    degree: Dict[str, int] = {}
    for edge in edges:
        degree[edge.dst_title] = degree.get(edge.dst_title, 0) + 1
    return degree


def sample_chains(
    edges: List[MentionEdge],
    index: CorpusIndex,
    n_chains: int,
    k_choices: List[int],
    rng: random.Random,
    max_in_degree: int = 12,
    max_chunk_count: int = 4,
    max_attempts_factor: int = 200,
) -> Iterator[List[MentionEdge]]:
    """Random-walk K-hop chains with obscurity filters.

    A chain of K requirements uses K-1 mention edges plus the final entity's
    own article: hop i's chunk mentions hop i+1's title.  Destinations must be
    obscure (bounded in-degree and chunk count) and titles must be distinct.
    ``k_choices`` is sampled uniformly per chain, giving the K-label variance
    the K-posterior needs.
    """
    by_src = _group_by_src(edges)
    in_degree = _in_degree(edges)
    seeds = [src for src in by_src if index.chunk_count.get(src, 0) <= max_chunk_count]
    if not seeds:
        return
    produced = 0
    attempts = 0
    max_attempts = n_chains * max_attempts_factor
    while produced < n_chains and attempts < max_attempts:
        attempts += 1
        k = rng.choice(k_choices)
        chain: List[MentionEdge] = []
        visited = {rng.choice(seeds)}
        current = next(iter(visited))
        ok = True
        for _ in range(k - 1):
            options = [
                e for e in by_src.get(current, [])
                if e.dst_title not in visited
                and in_degree.get(e.dst_title, 0) <= max_in_degree
                and index.chunk_count.get(e.dst_title, 0) <= max_chunk_count
                and e.dst_title in index.title_chunks
            ]
            if not options:
                ok = False
                break
            edge = rng.choice(options)
            chain.append(edge)
            visited.add(edge.dst_title)
            current = edge.dst_title
        if ok and len(chain) == k - 1:
            produced += 1
            yield chain
