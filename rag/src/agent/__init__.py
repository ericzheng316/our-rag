"""Agent-side contracts for the v8 (Qwen3.5 backbone) pipeline.

protocol_v1  — the FINALIZED action protocol (decision 2026-08-04): system
               prompt, action grammar, deterministic parser.  Single source
               of truth shared by the SFT renderer, the rollout environment,
               and the future calibration replay.
episode      — multi-turn episode runner producing rollout log records
               (native actions; the old E5 action labeler is retired here).
"""

from .protocol_v1 import (
    PROTOCOL_VERSION,
    SYSTEM_PROMPT,
    AnswerAction,
    InvalidAction,
    SearchAction,
    format_result_block,
    parse_assistant_turn,
)
from .episode import EpisodeResult, run_episode

__all__ = [
    "PROTOCOL_VERSION",
    "SYSTEM_PROMPT",
    "AnswerAction",
    "InvalidAction",
    "SearchAction",
    "format_result_block",
    "parse_assistant_turn",
    "EpisodeResult",
    "run_episode",
]
