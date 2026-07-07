from typing import Dict

from .base import GoldEvidenceAdapter
from .hotpotqa import HotpotQAAdapter

_REGISTRY: Dict[str, GoldEvidenceAdapter] = {
    "hotpotqa": HotpotQAAdapter(),
}


def get_adapter(name: str) -> GoldEvidenceAdapter:
    """Look up a dataset's gold-evidence adapter by name. HotpotQA is the
    only one implemented today; add a module per dataset in this package
    (same GoldEvidenceAdapter interface) for 2WikiMultiHopQA / MuSiQue-Ans /
    Bamboogle when those evaluation runs start (design doc Section 7)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"No GoldEvidenceAdapter registered for dataset '{name}'. "
            f"Available: {list(_REGISTRY)}. Add one in rag/src/belief/acec/datasets/."
        )


__all__ = ["GoldEvidenceAdapter", "HotpotQAAdapter", "get_adapter"]
