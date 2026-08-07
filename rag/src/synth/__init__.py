"""Synthetic multi-hop task construction over the local wiki18 corpus.

Pipeline stages (see run_scripts/60_build_synth_tasks.py for the entry point):

  1. corpus.py    — parse wiki18_100w.jsonl, build the linkable-title index
  2. graph.py     — mine title-mention edges, sample K-hop chains
  3. tasks.py     — chain -> task record: requirements (canonical v5
                    evidence_specification, closed candidate pool, distractors)
  4. verbalize.py — GPU stage: turn chain specs into natural questions with
                    the local Qwen3.5-9B via vLLM (leakage-checked)
  5. render_sft.py— PROVISIONAL trajectory renderer for SFT cold start
                    (action taxonomy is still an open design decision)

Design contract (from the 2026-08 narrative refactor):
  * retrieval-mandatory by construction: chains walk low-in-degree titles so
    parametric answering is suppressed; a separate closed-book screen fills
    task["closed_book"] later.
  * K (number of evidence requirements) is exact and variable, set by the
    synthesis graph — the K-posterior finally gets label variance.
  * calibration happens in CLOSED_CANDIDATE_POOL scope: every task carries its
    own pool (gold chunks + confusable distractors), which is what licenses
    confident zero-gain labels under the v5 evidence standard.
  * the synthesis graph doubles as free gold slot supervision: requirements
    are emitted in the canonical v5 manifest consumed unchanged by
    CanonicalEvidenceAdapterV5 — no new adapter code.
"""

from .corpus import CorpusIndex, parse_corpus_line
from .graph import mine_mentions, sample_chains
from .tasks import build_task

__all__ = [
    "CorpusIndex",
    "parse_corpus_line",
    "mine_mentions",
    "sample_chains",
    "build_task",
]
