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
from belief.acec import ACECBeliefState, ACECConfig, GoldEvidenceAdapter, get_adapter  # noqa: E402
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


def recover_title(doc_content: str) -> str:
    """Distractor paragraphs are formatted 'Title: sentence...' by
    prep_full_distractor.py's hf_context_to_paras()."""
    return doc_content.split(": ", 1)[0] if ": " in doc_content else doc_content[:40]


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
    record: Dict[str, Any], belief: ACECBeliefState, adapter: GoldEvidenceAdapter
) -> Tuple[List[HitExample], Optional[KExample]]:
    sf_titles = adapter.gold_titles(record)
    question = record["problem"]
    belief.reset(question)

    split_querys = record.get("split_querys", [])
    docs_per_turn = record.get("docs", [])

    per_turn: List[Tuple[int, Any]] = []
    for t, docs in enumerate(docs_per_turn):
        query = split_querys[t][0] if t < len(split_querys) and split_querys[t] else None
        result = belief.turn(query=query, new_docs=[{"contents": d} for d in docs], is_answer=False)
        per_turn.append((t, result))
    if "answer" in record and docs_per_turn:
        belief.turn(query=None, new_docs=[], is_answer=True)

    slot_hyps = [s.hypothesis for s in belief.coverage_belief.slots]
    slot_to_sf = match_slots_to_sf(slot_hyps, sf_titles, belief.labeler.embedder)

    examples: List[HitExample] = []
    for t, result in per_turn:
        docs = docs_per_turn[t]
        doc_titles = {recover_title(d) for d in docs}
        for j, m in result.slot_scores.items():
            sf_title = slot_to_sf.get(j)
            if sf_title is None:
                continue  # slot has no confident SF match — excluded per Section 2.2
            role = "tgt" if j == result.action.target_slot else "inc"
            bound = belief.coverage_belief.slots[j].bound
            is_hit = sf_title in doc_titles
            examples.append(HitExample(result.action.mode.value, role, bound, m, is_hit))

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
        "--dataset",
        default="hotpotqa",
        help="Gold-evidence adapter name (rag/src/belief/acec/datasets/). "
        "HotpotQA is the only one implemented today; add a module there for "
        "2WikiMultiHopQA / MuSiQue-Ans / Bamboogle when those runs start.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    adapter = get_adapter(args.dataset)
    embedder = E5Embedder(args.e5_model_path)
    nli_scorer = (
        CrossEncoderNLIScorer(args.nli_model) if args.nli_model else CosineNLIScorer(embedder)
    )
    if args.nli_model is None:
        print("[WARN] No --nli_model given — using E5-cosine stand-in. "
              "Treat the AUC gate result below as a pipeline sanity check, not the real gate.")

    belief = ACECBeliefState(embedder, nli_scorer, config=ACECConfig(k_max=args.k_max))

    records = []
    with open(args.records, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.records}")

    random.Random(args.seed).shuffle(records)
    n_held = max(1, int(len(records) * args.held_out_frac))
    held_out, fit_split = records[:n_held], records[n_held:]

    def run_split(split):
        all_examples, all_k = [], []
        for record in split:
            if not adapter.gold_titles(record) or not record.get("docs"):
                continue
            examples, k_example = replay_record(record, belief, adapter)
            all_examples.extend(examples)
            if k_example is not None:
                all_k.append(k_example)
        return all_examples, all_k

    fit_examples, fit_k = run_split(fit_split)
    held_examples, held_k = run_split(held_out)
    print(f"Fit split: {len(fit_split)} records -> {len(fit_examples)} hit examples, {len(fit_k)} K examples")
    print(f"Held-out split: {len(held_out)} records -> {len(held_examples)} hit examples, {len(held_k)} K examples")

    model, hit_rates = fit_observation_model(fit_examples)
    held_auc = evaluate_hit_auc(held_examples)
    fit_auc = evaluate_hit_auc(fit_examples)

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
