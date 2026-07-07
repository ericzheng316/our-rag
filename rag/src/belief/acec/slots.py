"""
Evidence slots — the working set S_t = {hat s_1, ..., hat s_K_t} of ACEC design
Section 1.1. Each slot is a sub-goal seeded by the query decomposer and grown
when a discovered bridge entity spawns a new dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List


@dataclass
class Slot:
    """One evidence slot hat s_j with its declarative hypothesis template.

    `bound` tracks whether the slot's dependencies (e.g. a bridge entity from
    an earlier slot) have been resolved into a concrete hypothesis — see
    design doc Section 1.1, "may be bound (dependencies resolved) or unbound".
    """

    idx: int
    hypothesis: str
    bound: bool = False
    bound_entities: List[str] = field(default_factory=list)
    created_at_turn: int = 0

    def bind(self, entity: str, turn: int) -> None:
        if entity not in self.bound_entities:
            self.bound_entities.append(entity)
        self.bound = True

    def substitute(self, template_slot_ref: str, entity: str) -> None:
        """Substitute a confirmed entity into the hypothesis template in place
        (Algorithm 1, line 4: "bind hypotheses: substitute entities confirmed
        in covered slots into hyp_j")."""
        if template_slot_ref in self.hypothesis:
            self.hypothesis = self.hypothesis.replace(template_slot_ref, entity)
        self.bind(entity, self.created_at_turn)


class SlotStore:
    """Ordered collection of slots, grown monotonically via DECOMPOSE actions."""

    def __init__(self) -> None:
        self._slots: List[Slot] = []

    def spawn(self, hypothesis: str, turn: int) -> Slot:
        slot = Slot(idx=len(self._slots), hypothesis=hypothesis, created_at_turn=turn)
        self._slots.append(slot)
        return slot

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self) -> Iterator[Slot]:
        return iter(self._slots)

    def __getitem__(self, i: int) -> Slot:
        return self._slots[i]

    def hypotheses(self) -> List[str]:
        return [s.hypothesis for s in self._slots]
