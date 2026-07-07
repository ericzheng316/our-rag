"""
HotpotQA gold-evidence adapter — the only dataset in active use.

Reads the 'supporting_facts' field ({'title': [...], 'sent_id': [...]})
that prep_full_distractor.py now merges into dev_distractor.jsonl and that
inference_new.py's solve_init() now carries through into records.jsonl.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class HotpotQAAdapter:
    name: str = "hotpotqa"

    def gold_titles(self, record: Dict[str, Any]) -> List[str]:
        sf = record.get("supporting_facts")
        if not sf or not sf.get("title"):
            return []
        return list(dict.fromkeys(sf["title"]))
