"""
Week-1 ACEC calibration builder (ACEC_Belief_RAG_design.md Section 2.2 / Section 9).

Replays a distractor-mode records.jsonl (produced by inference_new.py, which
now carries a 'supporting_facts' field per record — see prep_full_distractor.py
and inference_new.py's solve_init()) through ACECBeliefState, matches the
dynamically-discovered slots to the gold evidence titles (via the dataset's
GoldEvidenceAdapter — rag/src/belief/acec/datasets/, HotpotQA today) using E5
cosine, and emits the (slot, action-mode, NLI score, gold-hit) table that
rag/src/belief/acec/offline_fit.py needs. Prints the go/no-go gate: per-slot
hit AUC >= 0.80 and K-accuracy >= 75% on a held-out split (Section 9, Week 1).

This is pure replay of already-logged trajectories — no new GPU inference.

Usage (on the machine with models + records.jsonl, e.g. conda `rag` env):
    python build_acec_calibration.py \
        --records /root/logs/r3rag-qwen-distractor-acec-v1/records.jsonl \
        --e5_model_path /root/models/e5-base-v2 \
        --nli_model cross-encoder/nli-deberta-v3-base \
        --out_dir /root/logs/acec_calibration_v1

Without --nli_model, falls back to an E5-cosine stand-in for the NLI scorer
(CosineNLIScorer) — good enough to sanity-check the replay logic end-to-end,
but the real go/no-go gate should be evaluated with a real cross-encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.expanduser("~/rag/src"))
from belief.obs_extractor import E5Embedder  # noqa: E402
from belief.acec import (  # noqa: E402
    ACECBeliefState,
    ACECConfig,
    E5QueryEmbedder,
    GoldEvidenceAdapter,
    get_adapter,
    title_from_content,
)
from belief.acec.offline_fit import (  # noqa: E402
    EmpiricalKPredictor,
    HitExample,
    KExample,
    evaluate_hit_auc,
    fit_observation_model,
)


class CosineNLIScorer:
    """E5-cosine stand-in for a calibrated NLI cross-encoder. Use only to
    sanity-check the replay pipeline — the real go/no-go gate needs a real
    cross-encoder (--nli_model), since cosine cannot separate topically-similar
    non-entailing docs from true hits (the exact distractor-mode failure mode
    the design doc's observation model is meant to fix)."""

    def __init__(self, embedder: E5Embedder):
        self.embedder = embedder
        self._cache: Dict[Tuple[str, str], float] = {}

    def score(self, premise: str, hypothesis: str) -> float:
        key = (premise[:200], hypothesis)
        if key not in self._cache:
            embs = self.embedder.encode([premise[:200], hypothesis], show_progress_bar=False)
            sim = float(
                embs[0] @ embs[1] / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9)
            )
            self._cache[key] = (sim + 1.0) / 2.0  # cosine [-1,1] -> [0,1]
        return self._cache[key]


class CrossEncoderNLIScorer:
    """Real calibrated NLI observation model (design doc Section 1.2)."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        # Most nli-* CrossEncoder checkpoints order labels
        # [contradiction, neutral, entailment] — entailment is last.
        self._entail_idx = -1

    def score(self, premise: str, hypothesis: str) -> float:
        probs = self.model.predict([(premise, hypothesis)], apply_softmax=True)[0]
        return float(probs[self._entail_idx])


def match_slots_to_sf(
    slot_hyps: List[str], sf_titles: List[str], embedder: E5Embedder, threshold: float = 0.30
) -> Dict[int, str]:
    """Greedy bipartite match: each slot -> its best unclaimed SF title, only
    if similarity clears `threshold` (Section 2.2: 'Hungarian-matches')."""
    if not slot_hyps or not sf_titles:
        return {}
    embs = embedder.encode(slot_hyps + sf_titles, show_progress_bar=False)
    slot_embs, sf_embs = embs[: len(slot_hyps)], embs[len(slot_hyps) :]
    sims = slot_embs @ sf_embs.T
    pairs = sorted(
        (
            (float(sims[i, j]), i, j)
            for i in range(len(slot_hyps))
            for j in range(len(sf_titles))
        ),
        reverse=True,
    )
    assignment: Dict[int, str] = {}
    claimed = set()
    for sim, i, j in pairs:
        if i in assignment or j in claimed or sim < threshold:
            continue
        assignment[i] = sf_titles[j]
        claimed.add(j)
    return assignment


def replay_record(
    record: Dict[str, Any],
    belief: ACECBeliefState,
    adapter: GoldEvidenceAdapter,
    debug_records: Optional[List[Dict[str, Any]]] = None,
    sim_stats: Optional[List[float]] = None,
) -> Tuple[List[HitExample], Optional[KExample]]:
    sf_titles = adapter.gold_titles(record)
    question = record["problem"]
    belief.reset(question)

    split_querys = record.get("split_querys", [])
    docs_per_turn = record.get("docs", [])

    per_turn: List[Tuple[int, Any]] = []
    routing_debug: List[Dict[str, Any]] = []
    for t, docs in enumerate(docs_per_turn):
        query = split_querys[t][0] if t < len(split_querys) and split_querys[t] else None
        num_slots_before = len(belief.coverage_belief.slots)
        result = belief.turn(query=query, new_docs=[{"contents": d} for d in docs], is_answer=False)
        per_turn.append((t, result))
        if sim_stats is not None and result.label_max_sim is not None:
            sim_stats.append(result.label_max_sim)
        routing_debug.append(
            {
                "turn": t,
                "query": query,
                "num_slots_before": num_slots_before,
                "action_mode": result.action.mode.value,
                "target_slot": result.action.target_slot,
                "label_max_sim": result.label_max_sim,
                "tau_new": belief.config.tau_new,
            }
        )
    if "answer" in record and docs_per_turn:
        belief.turn(query=None, new_docs=[], is_answer=True)

    slot_hyps = [s.hypothesis for s in belief.coverage_belief.slots]
    slot_to_sf = match_slots_to_sf(slot_hyps, sf_titles, belief.labeler.embedder)

    examples: List[HitExample] = []
    debug_turns: List[Dict[str, Any]] = []
    for t, result in per_turn:
        docs = docs_per_turn[t]
        doc_titles = {title_from_content(d) for d in docs}
        for j, m in result.slot_scores.items():
            sf_title = slot_to_sf.get(j)
            if sf_title is None:
                continue  # slot has no confident SF match — excluded per Section 2.2
            role = "tgt" if j == result.action.target_slot else "inc"
            bound = belief.coverage_belief.slots[j].bound
            is_hit = sf_title in doc_titles
            examples.append(HitExample(result.action.mode.value, role, bound, m, is_hit))
            if debug_records is not None:
                debug_turns.append(
                    {
                        "turn": t,
                        "action_mode": result.action.mode.value,
                        "role": role,
                        "slot_idx": j,
                        "slot_hypothesis": slot_hyps[j] if j < len(slot_hyps) else None,
                        "matched_sf_title": sf_title,
                        "doc_titles": sorted(doc_titles),
                        "nli_score": m,
                        "is_hit": is_hit,
                    }
                )

    if debug_records is not None:
        debug_records.append(
            {
                "question": question,
                "sf_titles": sf_titles,
                "slot_hypotheses": slot_hyps,
                "slot_to_sf": slot_to_sf,
                "routing": routing_debug,
                "turns": debug_turns,
            }
        )

    k_example = None
    if sf_titles:
        q_emb = belief.labeler.embedder.encode([question], show_progress_bar=False)[0]
        k_example = KExample(q_emb, len(sf_titles))
    return examples, k_example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Path to a distractor-mode records.jsonl")
    ap.add_argument("--e5_model_path", required=True)
    ap.add_argument(
        "--nli_model",
        default=None,
        help="sentence-transformers CrossEncoder name, e.g. cross-encoder/nli-deberta-v3-base. "
        "Omit to use the E5-cosine stand-in (sanity check only, not the real gate).",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k_max", type=int, default=4)
    ap.add_argument("--held_out_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tau_new",
        type=float,
        default=ACECConfig().tau_new,
        help="ActionLabeler DECOMPOSE-vs-REWRITE/EXPAND cosine threshold (design "
        "doc default 0.55, but that was never calibrated against real E5 output — "
        "the Week-1 pilot found E5-with-query-prefix cosine similarity between "
        "genuinely different sub-questions in the same HotpotQA question starts "
        "around 0.66, so 0.55 never once triggered DECOMPOSE past the first slot. "
        "Sweep this (e.g. 0.75/0.85/0.92/0.95) and watch the "
        "'avg slots spawned' / 'frac >= tau_new' diagnostics below.",
    )
    ap.add_argument(
        "--dataset",
        default="hotpotqa",
        help="Gold-evidence adapter name (rag/src/belief/acec/datasets/). "
        "HotpotQA is the only one implemented today; add a module there for "
        "2WikiMultiHopQA / MuSiQue-Ans / Bamboogle when those runs start.",
    )
    ap.add_argument(
        "--debug_dump",
        default=None,
        help="Optional path to write a JSONL of per-record diagnostics (slot "
        "hypotheses, matched SF titles, doc titles, NLI scores, hit labels) "
        "for manual inspection when the gate result looks off.",
    )
    ap.add_argument(
        "--debug_dump_limit",
        type=int,
        default=50,
        help="Max number of records to write to --debug_dump.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    adapter = get_adapter(args.dataset)
    # E5 requires a 'query: ' prefix for well-calibrated similarity (see
    # rag/src/belief/acec/e5_adapter.py) — without it, the action labeler's
    # cosine routing can't tell unrelated queries apart (diagnosed from the
    # Week-1 pilot: avg slots spawned was exactly 1.00 across all 500 records
    # despite K_true=2 always).
    embedder = E5QueryEmbedder(E5Embedder(args.e5_model_path))
    nli_scorer = (
        CrossEncoderNLIScorer(args.nli_model) if args.nli_model else CosineNLIScorer(embedder)
    )
    if args.nli_model is None:
        print("[WARN] No --nli_model given — using E5-cosine stand-in. "
              "Treat the AUC gate result below as a pipeline sanity check, not the real gate.")

    belief = ACECBeliefState(embedder, nli_scorer, config=ACECConfig(k_max=args.k_max, tau_new=args.tau_new))

    records = []
    with open(args.records, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.records}")

    random.Random(args.seed).shuffle(records)
    n_held = max(1, int(len(records) * args.held_out_frac))
    held_out, fit_split = records[:n_held], records[n_held:]

    debug_records: Optional[List[Dict[str, Any]]] = [] if args.debug_dump else None
    slot_coverage_stats: List[Tuple[int, int]] = []  # (num_slots_spawned, k_true)
    sim_stats: List[float] = []  # every label_max_sim seen, across all records/turns

    def run_split(split, collect_debug: bool):
        all_examples, all_k = [], []
        for record in split:
            sf_titles = adapter.gold_titles(record)
            if not sf_titles or not record.get("docs"):
                continue
            dbg = debug_records if (collect_debug and debug_records is not None and len(debug_records) < args.debug_dump_limit) else None
            examples, k_example = replay_record(record, belief, adapter, debug_records=dbg, sim_stats=sim_stats)
            slot_coverage_stats.append((len(belief.coverage_belief.slots), len(sf_titles)))
            all_examples.extend(examples)
            if k_example is not None:
                all_k.append(k_example)
        return all_examples, all_k

    fit_examples, fit_k = run_split(fit_split, collect_debug=True)
    held_examples, held_k = run_split(held_out, collect_debug=True)
    print(f"Fit split: {len(fit_split)} records -> {len(fit_examples)} hit examples, {len(fit_k)} K examples")
    print(f"Held-out split: {len(held_out)} records -> {len(held_examples)} hit examples, {len(held_k)} K examples")

    if slot_coverage_stats:
        avg_slots = sum(n for n, _ in slot_coverage_stats) / len(slot_coverage_stats)
        avg_k = sum(k for _, k in slot_coverage_stats) / len(slot_coverage_stats)
        under_frac = sum(1 for n, k in slot_coverage_stats if n < k) / len(slot_coverage_stats)
        print(
            f"  [debug] slot identification: avg slots spawned={avg_slots:.2f}, "
            f"avg K_true={avg_k:.2f}, under-decomposed (spawned < K_true) in "
            f"{under_frac:.1%} of records"
        )

    if sim_stats:
        arr = np.array(sim_stats)
        print(
            f"  [debug] label_max_sim (all turns with an existing slot to compare "
            f"against, n={len(arr)}): min={arr.min():.3f} p25={np.percentile(arr, 25):.3f} "
            f"median={np.median(arr):.3f} p75={np.percentile(arr, 75):.3f} max={arr.max():.3f} "
            f"| tau_new={args.tau_new} "
            f"| frac >= tau_new (routed REWRITE/EXPAND, not DECOMPOSE)={float((arr >= args.tau_new).mean()):.1%}"
        )

    if args.debug_dump and debug_records:
        with open(args.debug_dump, "w", encoding="utf-8") as f:
            for rec in debug_records:
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
        print(f"Wrote {len(debug_records)} debug records to {args.debug_dump}")

    model, hit_rates = fit_observation_model(fit_examples)
    held_auc = evaluate_hit_auc(held_examples)
    fit_auc = evaluate_hit_auc(fit_examples)

    for role in ("tgt", "inc"):
        role_examples = [ex for ex in held_examples if ex.role == role]
        n_hit = sum(1 for ex in role_examples if ex.is_hit)
        role_auc = evaluate_hit_auc(role_examples)
        print(f"  [debug] held-out role={role}: n={len(role_examples)}, hits={n_hit}, AUC={role_auc:.3f}")

    k_predictor = EmpiricalKPredictor(args.k_max, fit_k)
    held_k_acc = k_predictor.evaluate_k_accuracy(held_k)

    print("\n=== ACEC Week-1 go/no-go gate (design doc Section 9) ===")
    print(f"  per-slot hit AUC   (fit / held-out): {fit_auc:.3f} / {held_auc:.3f}   [need >= 0.80]")
    print(f"  K-accuracy         (held-out):        {held_k_acc:.3f}   [need >= 0.75]")
    print(f"  empirical pi_a per action mode:        {hit_rates}")
    gate_pass = (held_auc >= 0.80) and (held_k_acc >= 0.75)
    print(f"  GATE: {'PASS — proceed to Algorithm 1 replay / GRPO scaffold' if gate_pass else 'FAIL — observation model too weak, stop and re-scope'}")

    out_path = os.path.join(args.out_dir, "observation_model.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "observation_model": model.to_dict(),
                "hit_rates": hit_rates,
                "fit_auc": fit_auc,
                "held_out_auc": held_auc,
                "held_out_k_accuracy": held_k_acc,
                "n_fit_examples": len(fit_examples),
                "n_held_out_examples": len(held_examples),
            },
            f,
            indent=2,
        )
    print(f"\nWrote calibrated observation model + gate metrics to {out_path}")


if __name__ == "__main__":
    main()
