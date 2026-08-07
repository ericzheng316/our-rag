"""wiki18_100w corpus access and the linkable-title index.

Corpus line format (FlashRAG): {"id": "0", "contents": "\"Title\"\ntext..."}.
One Wikipedia article is split into several ~100-word chunks that share the
same title, so the index maps a normalized title to *all* its chunk ids.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

# Lowercase connectors allowed inside a capitalized mention span.
_CONNECTORS = {"of", "the", "and", "de", "la", "le", "du", "von", "van", "der", "in", "for"}
_CAP_TOKEN = re.compile(r"^[A-Z][\w'.\-]*$")
_WORD = re.compile(r"[\w'.\-]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_WORD.findall(normalized))


def parse_corpus_line(line: str) -> Tuple[str, str, str]:
    """Return (chunk_id, title, text) for one corpus jsonl line."""
    payload = json.loads(line)
    contents = payload["contents"]
    if contents.startswith('"'):
        end = contents.find('"', 1)
        title = contents[1:end] if end > 0 else ""
        text = contents[end + 1 :].lstrip("\n ") if end > 0 else contents
    else:
        head, _, rest = contents.partition("\n")
        title, text = head, rest
    return str(payload["id"]), title.strip(), text


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def extract_mention_spans(text: str, max_len: int = 6) -> Iterator[str]:
    """Yield capitalized multi-token spans that could be entity mentions.

    A span is 2..max_len whitespace tokens, starts and ends with a
    capitalized token, and may contain lowercase connectors in between.
    Single-token mentions are deliberately excluded: they collide with
    sentence-initial words and generic nouns, and precision matters more
    than recall for edge mining.
    """
    tokens = text.split()
    n = len(tokens)
    i = 0
    while i < n:
        word = _strip_edge(tokens[i])
        if not word or not _CAP_TOKEN.match(word):
            i += 1
            continue
        j = i + 1
        last_cap = i
        while j < n and j - i < max_len:
            nxt = _strip_edge(tokens[j])
            if nxt and _CAP_TOKEN.match(nxt):
                last_cap = j
            elif nxt in _CONNECTORS:
                pass
            else:
                break
            j += 1
        if last_cap > i:
            for end in range(last_cap, i, -1):
                span_tokens = [_strip_edge(t) for t in tokens[i : end + 1]]
                if _strip_edge(tokens[end]) and _CAP_TOKEN.match(_strip_edge(tokens[end])):
                    yield " ".join(t for t in span_tokens if t)
        i = max(last_cap, i) + 1


def _strip_edge(token: str) -> str:
    return token.strip("\"'()[]{},.;:!?")


@dataclass
class CorpusIndex:
    """Linkable-title index over the corpus, built in one streaming pass.

    Only multi-token titles are linkable (single-token titles are far too
    ambiguous as lexical mentions).  ``title_chunks`` keeps at most
    ``max_chunks_per_title`` chunk ids per title; ``chunk_count`` keeps the
    true count so obscurity filters can still see it.
    """

    corpus_path: str
    max_lines: Optional[int] = None
    max_chunks_per_title: int = 3
    title_chunks: Dict[str, List[str]] = field(default_factory=dict)
    chunk_count: Dict[str, int] = field(default_factory=dict)
    display_title: Dict[str, str] = field(default_factory=dict)
    token_to_titles: Dict[str, List[str]] = field(default_factory=dict)
    max_titles_per_token: int = 50

    def build(self) -> "CorpusIndex":
        with open(self.corpus_path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if self.max_lines is not None and line_no >= self.max_lines:
                    break
                chunk_id, title, _ = parse_corpus_line(line)
                key = normalize_title(title)
                if len(key.split()) < 2:
                    continue
                self.chunk_count[key] = self.chunk_count.get(key, 0) + 1
                chunks = self.title_chunks.setdefault(key, [])
                if len(chunks) < self.max_chunks_per_title:
                    chunks.append(chunk_id)
                if key not in self.display_title:
                    self.display_title[key] = title
                    for token in set(key.split()):
                        if len(token) < 4:
                            continue
                        bucket = self.token_to_titles.setdefault(token, [])
                        if len(bucket) < self.max_titles_per_token:
                            bucket.append(key)
        return self

    def lookup(self, span: str) -> Optional[str]:
        key = normalize_title(span)
        return key if key in self.title_chunks else None

    def confusable_titles(self, key: str, limit: int) -> List[str]:
        """Titles sharing a (>=4 char) token with ``key`` — distractor pool."""
        out: List[str] = []
        for token in key.split():
            for other in self.token_to_titles.get(token, []):
                if other != key and other not in out:
                    out.append(other)
                    if len(out) >= limit:
                        return out
        return out


def load_chunks(corpus_path: str, wanted: set, max_lines: Optional[int] = None) -> Dict[str, Dict[str, str]]:
    """Second streaming pass: materialize contents for the chunk ids in ``wanted``."""
    found: Dict[str, Dict[str, str]] = {}
    with open(corpus_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if max_lines is not None and line_no >= max_lines:
                break
            chunk_id, title, text = parse_corpus_line(line)
            if chunk_id in wanted:
                found[chunk_id] = {"id": chunk_id, "title": title, "contents": f"{title}: {text}"}
                if len(found) == len(wanted):
                    break
    return found
