"""
Dataset-specific gold-evidence extraction for ACEC offline calibration
(design doc Section 2.2). Every multi-hop QA dataset with supporting-evidence
annotations exposes gold evidence differently — HotpotQA's supporting_facts
{'title': [...], 'sent_id': [...]}, 2WikiMultiHopQA's evidences list,
MuSiQue's decomposition chain — but offline_fit.py's slot-to-gold-evidence
matching only needs a flat list of titles/ids per question. Implementing this
interface once per dataset keeps that matching logic dataset-agnostic.

HotpotQA is the only dataset in active use (see hotpotqa.py); design doc
Section 7 also evaluates on 2WikiMultiHopQA, MuSiQue-Ans and Bamboogle — add
one module per dataset here, implementing the same interface, when those runs
start. Bamboogle has no supporting-fact annotations at all (it's the fully
OOD generalization test), so its adapter would always return [].
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class GoldEvidenceAdapter(Protocol):
    name: str

    def gold_titles(self, record: Dict[str, Any]) -> List[str]:
        """Return the (deduped, order-preserving) titles/ids of the gold
        evidence pieces required to answer this question, or [] if the
        record carries no gold annotation (e.g. an OOD/unlabeled dataset)."""
        ...
