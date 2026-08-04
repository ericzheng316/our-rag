"""Factorial ACEC v6.4 inference: policy x controller x verifier x K.

One vLLM engine evaluates frozen R3 and requested LoRA policies.  For every
policy/controller arm it samples the same number of trajectories, applies the
belief controller between turns, and then reports K=1/2/4/8 oracle, majority,
generic-NLI, and ACEC-verifier results.  Runtime traces contain no gold answers;
correctness labels are written only to a separate evaluator-side file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RAG_SRC = REPO_ROOT / "rag" / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from belief.acec.answer_verifier_v64 import (  # noqa: E402
    TrajectoryCandidateV64,
    answer_exact_match,
    evaluate_policy_controller_factorial_v64,
    load_answer_verifier_artifact_v64,
    normalize_answer,
    paired_bootstrap_delta_v64,
)
from belief.acec.evidence_selector_minimal_v63 import answer_hypothesis  # noqa: E402
from belief.acec.inference_controller_v64 import (  # noqa: E402
    BRIDGE_ENTITY_SCHEMA,
    BeliefInferenceControllerV64,
    BeliefSnapshotV64,
    ControllerAction,
    ControllerConfigV64,
    ControllerMode,
    FORCE_ANSWER_INSTRUCTION,
    SlotDecompositionV64,
    load_controller_artifact_v64,
    parse_slot_decomposition,
    preseed_belief_slots,
    slot_decomposition_prompt,
)
try:  # Direct script execution puts rag/train on sys.path.
    from short_answer_contract_v63 import (  # type: ignore
        ANSWER_CONTRACT_VERSION,
        SHORT_ANSWER_INSTRUCTION,
        parse_tagged_short_answer,
    )
except ImportError:  # Package import is used by the contract tests.
    from rag.train.short_answer_contract_v63 import (
        ANSWER_CONTRACT_VERSION,
        SHORT_ANSWER_INSTRUCTION,
        parse_tagged_short_answer,
    )


logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("infer_belief_controller_v64")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _id_hash(ids: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(set(str(value) for value in ids))).encode("utf-8")
    ).hexdigest()


def parse_adapter(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("adapter must be NAME=/absolute/path")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("adapter must be NAME=/absolute/path")
    return name.strip(), path.strip()


def _supporting_titles(row: Mapping[str, Any]) -> List[str]:
    supporting = row.get("supporting_facts")
    if not supporting:
        supporting = (row.get("metadata") or {}).get("supporting_facts")
    if isinstance(supporting, Mapping):
        return [str(value) for value in supporting.get("title", [])]
    if isinstance(supporting, list):
        return [str(value[0]) for value in supporting if len(value) == 2]
    return []


def _read_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(4096).lstrip()[:1]
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"JSON data is not an array: {path}")
            return [dict(row) for row in payload]
        return [json.loads(line) for line in handle if line.strip()]


def load_questions(
    path: Path,
    *,
    max_questions: int,
    seed: int,
    question_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    rows = _read_json_records(path)
    questions = []
    for index, row in enumerate(rows):
        qid = str(row.get("_id", row.get("id", f"row-{index}")))
        if question_ids is not None and qid not in question_ids:
            continue
        question = row.get("question", row.get("problem"))
        answers = row.get("golden_answers")
        if not answers and row.get("answer") is not None:
            answers = [row["answer"]]
        if not question or not answers:
            continue
        questions.append(
            {
                "question_id": qid,
                "question": str(question),
                "golden_answers": [str(answer) for answer in answers],
                "sf_titles": _supporting_titles(row),
                "question_type": row.get(
                    "type", row.get("question_type", (row.get("metadata") or {}).get("type"))
                ),
            }
        )
    if not questions:
        raise ValueError("held-out data produced no usable questions")
    if 0 < max_questions < len(questions):
        questions = random.Random(seed).sample(questions, max_questions)
    return questions


def _read_id_file(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    values = set()
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    value = payload.get("question_id", payload.get("_id", payload.get("id")))
                else:
                    value = payload
            except json.JSONDecodeError:
                value = line.strip()
            if value is not None:
                values.add(str(value))
    return values


def load_slot_manifest(path: Path) -> Dict[str, SlotDecompositionV64]:
    manifest = {}
    for row in _read_json_records(path):
        decomposition = SlotDecompositionV64.from_dict(row)
        if decomposition.question_id in manifest:
            raise ValueError(f"duplicate slot decomposition id {decomposition.question_id}")
        manifest[decomposition.question_id] = decomposition
    return manifest


def load_oracle_entities(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    entities = {}
    for row in _read_json_records(Path(path).expanduser()):
        if (
            row.get("schema") != BRIDGE_ENTITY_SCHEMA
            or int(row.get("version", -1)) != 64
        ):
            raise ValueError("oracle entity manifest contains a non-v6.4 row")
        if row.get("assessable") is False:
            continue
        entity = row.get("entity")
        qid = row.get("question_id", row.get("_id", row.get("id")))
        if qid is not None and entity:
            entities[str(qid)] = str(entity)
    return entities


def generate_slot_manifest(
    llm: Any,
    questions: Sequence[Mapping[str, Any]],
    *,
    expected_k: int,
    generator_name: str,
    batch_size: int,
) -> Tuple[Dict[str, SlotDecompositionV64], List[Dict[str, Any]]]:
    from vllm import SamplingParams
    import grpo_rsf_simple

    parsed: Dict[str, SlotDecompositionV64] = {}
    audit = []
    for offset in range(0, len(questions), batch_size):
        batch = questions[offset : offset + batch_size]
        prompts = [
            grpo_rsf_simple.apply_chat_template(
                slot_decomposition_prompt(item["question"], expected_k)
            )
            for item in batch
        ]
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=256),
            lora_request=None,
            use_tqdm=False,
        )
        for item, output in zip(batch, outputs):
            raw = output.outputs[0].text
            status = "ok"
            try:
                decomposition = parse_slot_decomposition(
                    raw,
                    question_id=item["question_id"],
                    question=item["question"],
                    expected_k=expected_k,
                    generator=generator_name,
                )
                parsed[item["question_id"]] = decomposition
            except ValueError as error:
                status = str(error)
            audit.append(
                {
                    "question_id": item["question_id"],
                    "raw_generation": raw,
                    "status": status,
                }
            )
    return parsed, audit


def _document_label(document: Mapping[str, Any]) -> str:
    if document.get("title"):
        return str(document["title"])
    contents = str(document.get("contents") or "")
    return contents.splitlines()[0][:200] if contents else ""


def _snapshot_payload(decision: Any) -> Optional[Dict[str, Any]]:
    snapshot = decision.snapshot
    if snapshot is None:
        return None
    return {
        "coverage": snapshot.coverage,
        "coverage_std": snapshot.coverage_std,
        "stop_voi": snapshot.stop_voi,
        "target_slot": snapshot.target_slot,
        "target_hypothesis": snapshot.target_hypothesis,
        "target_probability": snapshot.target_probability,
        "bound_entities": list(snapshot.bound_entities),
        "slot_probabilities": list(snapshot.slot_probabilities),
        "slot_bound": list(snapshot.slot_bound),
    }


def _initial_context(question: str) -> str:
    return (
        f"The question: {question}\n"
        f"{SHORT_ANSWER_INSTRUCTION}\n"
        "Keep the existing R3 step format. Put the tagged span after "
        "'The final answer:'."
    )


def _free_text_prediction(row: Mapping[str, Any]) -> str:
    """取被选轨迹的原始自由文本，供 processed-EM 的外部抽取器使用。

    契约解析成功时 selected_raw_answer 与 selected_answer 内容一致；缺 <answer>
    标签时只有前者非空。回退到 selected_answer 是为了兼容尚未携带原文的旧记录。
    """
    raw = str(row.get("selected_raw_answer") or "").strip()
    return raw or str(row.get("selected_answer") or "").strip()


def rollout_controller_batch(
    llm: Any,
    lora_request: Any,
    questions: Sequence[Mapping[str, Any]],
    *,
    policy_name: str,
    controller_config: ControllerConfigV64,
    belief_factory: Any,
    slot_manifest: Mapping[str, SlotDecompositionV64],
    oracle_entities: Mapping[str, str],
    n_samples: int,
    n_turns: int,
    temperature: float,
    max_docs: int,
) -> List[Dict[str, Any]]:
    from vllm import SamplingParams
    import grpo_rsf_simple

    states: List[Dict[str, Any]] = []
    for question_index, item in enumerate(questions):
        for sample_index in range(n_samples):
            belief = belief_factory(item["question"])
            decomposition = slot_manifest.get(item["question_id"])
            if decomposition is not None:
                preseed_belief_slots(belief, decomposition.slots)
            states.append(
                {
                    "question_index": question_index,
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "sample_index": sample_index,
                    "context": _initial_context(item["question"]),
                    "belief": belief,
                    "controller": BeliefInferenceControllerV64(controller_config),
                    "oracle_entity": (
                        oracle_entities.get(item["question_id"])
                        if controller_config.uses_gold_entity
                        else None
                    ),
                    "retrieved_ids": [],
                    "retrieved_documents": [],
                    "retrieval_calls": 0,
                    "generation_calls": 0,
                    "empty_retrievals": 0,
                    "events": [],
                    "answer": "",
                    "raw_answer": "",
                    "answered": False,
                    "format_valid": True,
                    "answer_parse_status": "not_answered",
                    "done": False,
                    "force_answer_pending": False,
                }
            )

    # One extra generation opportunity is reserved for a controller-forced
    # answer after the final permitted retrieval action.
    for turn_index in range(n_turns + 1):
        if turn_index == n_turns:
            for state in states:
                if not state["done"] and not state["force_answer_pending"]:
                    state["answer_parse_status"] = "max_model_turns_without_answer"
                    state["done"] = True
        active = [index for index, state in enumerate(states) if not state["done"]]
        if not active:
            break
        prompts = [grpo_rsf_simple.apply_chat_template(states[index]["context"]) for index in active]
        params = SamplingParams(
            temperature=temperature,
            max_tokens=512,
            stop_token_ids=[grpo_rsf_simple.STOP_TOKEN],
        )
        outputs = llm.generate(
            prompts, params, lora_request=lora_request, use_tqdm=False
        )
        for state_index in active:
            states[state_index]["generation_calls"] += 1
        pending_retrieval = []
        for output_index, state_index in enumerate(active):
            state = states[state_index]
            completion = outputs[output_index].outputs[0]
            parsed = grpo_rsf_simple.parse_step(
                f"Step {turn_index + 1}:\n{completion.text}"
            )
            if not parsed.get("analysis"):
                state["format_valid"] = False
                state["answer_parse_status"] = "missing_problem_analysis"
                state["events"].append(
                    {
                        "turn": turn_index + 1,
                        "kind": "format_error",
                        "raw": completion.text,
                    }
                )
                state["done"] = True
                continue

            if parsed.get("answer"):
                state["raw_answer"] = str(parsed["answer"])
                short = parse_tagged_short_answer(parsed["answer"])
                if not short.valid:
                    state["format_valid"] = False
                    # 策略确实产出了答案文本，只是不满足 v6.3 的标签契约。
                    # 以前这里让 answered 保持 False，连带把 grounding_pairs、
                    # grounding_score、zero_shot_acec_score 全部清零（实测 98–100%
                    # 的候选受影响），使 nli/acec_zero_shot 在全零打分上排序、
                    # 退化成"取第一个"，verifier 阶梯因此从未被真正检验过。
                    # answered 表示"是否作答"，契约合规性由 format_valid 承载。
                    state["answered"] = True
                    state["answer_parse_status"] = short.status
                    state["events"].append(
                        {
                            "turn": turn_index + 1,
                            "kind": "answer_contract_error",
                            "raw_answer": parsed["answer"],
                            "status": short.status,
                        }
                    )
                    state["done"] = True
                    continue
                if state["force_answer_pending"]:
                    snapshot = BeliefSnapshotV64.from_belief(state["belief"])
                    state["events"].append(
                        {
                            "turn": turn_index + 1,
                            "kind": "policy_answer",
                            "answer": short.answer,
                            "controller_action": (
                                ControllerAction.ACCEPT_POLICY_ANSWER.value
                            ),
                            "reason": "forced_answer_completion",
                            "snapshot": {
                                "coverage": snapshot.coverage,
                                "coverage_std": snapshot.coverage_std,
                                "stop_voi": snapshot.stop_voi,
                                "target_slot": snapshot.target_slot,
                                "target_hypothesis": snapshot.target_hypothesis,
                                "target_probability": snapshot.target_probability,
                                "bound_entities": list(snapshot.bound_entities),
                                "slot_probabilities": list(
                                    snapshot.slot_probabilities
                                ),
                                "slot_bound": list(snapshot.slot_bound),
                            },
                        }
                    )
                    state["answer"] = short.answer
                    state["answered"] = True
                    state["answer_parse_status"] = "ok"
                    state["done"] = True
                    continue
                decision = state["controller"].decide(
                    question=state["question"],
                    belief=state["belief"],
                    retrieval_calls=state["retrieval_calls"],
                    policy_answer=short.answer,
                    gold_entity=state["oracle_entity"],
                )
                state["events"].append(
                    {
                        "turn": turn_index + 1,
                        "kind": "policy_answer",
                        "answer": short.answer,
                        "controller_action": decision.action.value,
                        "reason": decision.reason,
                        "snapshot": _snapshot_payload(decision),
                    }
                )
                if decision.action == ControllerAction.ACCEPT_POLICY_ANSWER:
                    state["answer"] = short.answer
                    state["answered"] = True
                    state["answer_parse_status"] = "ok"
                    state["done"] = True
                elif decision.action == ControllerAction.OVERRIDE_ANSWER_WITH_QUERY:
                    if turn_index == n_turns - 1:
                        state["force_answer_pending"] = True
                    state["_pending"] = {
                        "analysis": parsed["analysis"],
                        "policy_query": None,
                        "executed_query": decision.query,
                        "decision": decision,
                    }
                    pending_retrieval.append(state_index)
                else:
                    raise RuntimeError(
                        f"unexpected answer decision {decision.action.value}"
                    )
                continue

            if parsed.get("query"):
                if turn_index == n_turns:
                    state["format_valid"] = False
                    state["answer_parse_status"] = "forced_answer_emitted_query"
                    state["events"].append(
                        {
                            "turn": turn_index + 1,
                            "kind": "forced_answer_error",
                            "policy_query": parsed["query"],
                        }
                    )
                    state["done"] = True
                    continue
                decision = state["controller"].decide(
                    question=state["question"],
                    belief=state["belief"],
                    retrieval_calls=state["retrieval_calls"],
                    policy_query=str(parsed["query"]),
                    gold_entity=state["oracle_entity"],
                )
                state["events"].append(
                    {
                        "turn": turn_index + 1,
                        "kind": "policy_query",
                        "policy_query": parsed["query"],
                        "executed_query": decision.query,
                        "controller_action": decision.action.value,
                        "reason": decision.reason,
                        "injected_entity": decision.injected_entity,
                        "target_slot": decision.target_slot,
                        "snapshot": _snapshot_payload(decision),
                    }
                )
                if decision.action == ControllerAction.FORCE_ANSWER:
                    state["context"] += (
                        f"\nStep {turn_index + 1}:\n"
                        f"The problem analysis: {parsed['analysis']}\n"
                        f"The proposed retrieval query: {parsed['query']}\n"
                        "Controller: "
                        f"{FORCE_ANSWER_INSTRUCTION}"
                    )
                    state["force_answer_pending"] = True
                elif decision.action == ControllerAction.EXECUTE_QUERY:
                    state["_pending"] = {
                        "analysis": parsed["analysis"],
                        "policy_query": parsed["query"],
                        "executed_query": decision.query,
                        "decision": decision,
                    }
                    pending_retrieval.append(state_index)
                else:
                    raise RuntimeError(
                        f"unexpected query decision {decision.action.value}"
                    )
                continue

            state["format_valid"] = False
            state["answer_parse_status"] = "missing_query_or_answer"
            state["done"] = True

        if pending_retrieval:
            queries = [states[index]["_pending"]["executed_query"] for index in pending_retrieval]
            documents_by_state = grpo_rsf_simple._retrieve_batch(queries)
            for state_index, documents in zip(pending_retrieval, documents_by_state):
                state = states[state_index]
                pending = state.pop("_pending")
                state["retrieval_calls"] += 1
                if not documents:
                    state["empty_retrievals"] += 1
                new_documents = []
                for document in documents:
                    document_id = document.get("id")
                    if (
                        document_id not in state["retrieved_ids"]
                        and len(state["retrieved_ids"]) < max_docs
                    ):
                        state["retrieved_ids"].append(document_id)
                        new_documents.append(document)
                        state["retrieved_documents"].append(document)
                document_text = "\n".join(
                    str(document.get("contents") or "") for document in new_documents
                )
                result = state["belief"].turn(
                    query=str(pending["executed_query"]), new_docs=new_documents
                )
                state["events"].append(
                    {
                        "turn": turn_index + 1,
                        "kind": "retrieval",
                        "policy_query": pending["policy_query"],
                        "executed_query": pending["executed_query"],
                        "documents": [
                            {
                                "id": document.get("id"),
                                "title": _document_label(document),
                                "provenance_status": document.get(
                                    "provenance_status", "unmappable"
                                ),
                            }
                            for document in new_documents
                        ],
                        "action": result.action.mode.value,
                        "target_slot": result.action.target_slot,
                        "coverage_before": result.coverage_before,
                        "coverage_after": result.coverage_after,
                        "delta_coverage": result.delta_coverage,
                        "stop_voi": result.stop_voi,
                        "slot_p": list(result.features.slot_p),
                        "slot_bound": [
                            bool(slot.bound)
                            for slot in state["belief"].coverage_belief.slots
                        ],
                    }
                )
                state["context"] += (
                    f"\nStep {turn_index + 1}:\n"
                    f"The problem analysis: {pending['analysis']}\n"
                    f"The retrieval query: {pending['executed_query']}\n"
                    f"The retrieval documents: {document_text[:512]}"
                )
                if state["force_answer_pending"]:
                    state["context"] += (
                        f"\nController after step {turn_index + 1}: "
                        f"{FORCE_ANSWER_INSTRUCTION}"
                    )
        if turn_index == n_turns:
            for state in states:
                if not state["done"]:
                    state["answer_parse_status"] = "forced_answer_not_completed"
                    state["done"] = True
    return states


def score_grounding_batch(states: Sequence[Mapping[str, Any]], batch_size: int) -> List[float]:
    """Batch answer-conditioned NLI once per run instead of per trajectory."""

    pairs = []
    owners = []
    for state_index, state in enumerate(states):
        if not state["answered"]:
            continue
        # 契约不合规时 state["answer"] 是空串，用原始输出构造假设，
        # 否则 NLI 会对着空答案打分，grounding_score 依旧无意义。
        answer_text = str(state["answer"] or state.get("raw_answer") or "").strip()
        if not answer_text:
            continue
        hypothesis = answer_hypothesis(state["question"], answer_text)
        for document in state["retrieved_documents"]:
            contents = str(document.get("contents") or "")
            if contents:
                owners.append(state_index)
                pairs.append((contents, hypothesis))
    scores = [0.0] * len(states)
    if not pairs:
        return scores
    scorer = next(
        state["belief"].nli_scorer
        for state in states
        if state.get("belief") is not None
    )
    probabilities = scorer.model.predict(
        pairs, apply_softmax=True, batch_size=batch_size, show_progress_bar=True
    )
    entailment_index = int(getattr(scorer, "_entail_idx", 1))
    for owner, probability in zip(owners, probabilities):
        scores[owner] = max(
            scores[owner], float(probability[entailment_index])
        )
    return scores


def states_to_candidates(
    states: Sequence[Mapping[str, Any]],
    grounding_scores: Sequence[float],
    *,
    grounding_scored: bool,
) -> List[TrajectoryCandidateV64]:
    candidates = []
    for state, grounding in zip(states, grounding_scores):
        belief = state["belief"].coverage_belief
        candidates.append(
            TrajectoryCandidateV64(
                question_id=state["question_id"],
                sample_index=int(state["sample_index"]),
                answer=str(state["answer"]),
                grounding_score=float(grounding),
                coverage=float(belief.coverage()),
                coverage_std=float(
                    max(belief.coverage_variance(), 0.0) ** 0.5
                ),
                retrieval_calls=int(state["retrieval_calls"]),
                generation_calls=int(state["generation_calls"]),
                grounding_pairs=(
                    len(state["retrieved_documents"])
                    if grounding_scored and state["answered"]
                    else 0
                ),
                answered=bool(state["answered"]),
                format_valid=bool(state["format_valid"]),
                raw_answer=str(state.get("raw_answer") or ""),
                metadata={
                    "answer_parse_status": state["answer_parse_status"],
                    "bound_slot_count": sum(
                        bool(slot.bound) for slot in belief.slots
                    ),
                    "bound_entities": list(
                        dict.fromkeys(
                            str(entity)
                            for slot in belief.slots
                            if slot.bound
                            for entity in slot.bound_entities
                            if str(entity).strip()
                        )
                    ),
                },
            )
        )
    return candidates


def _controller_config(
    args: argparse.Namespace,
    mode: ControllerMode,
    artifact: Optional[Any] = None,
) -> ControllerConfigV64:
    if artifact is not None:
        calibrated = artifact.config
        return ControllerConfigV64(
            mode=mode,
            min_retrievals=calibrated.min_retrievals,
            max_retrievals=calibrated.max_retrievals,
            injection_after_retrievals=calibrated.injection_after_retrievals,
            stop_coverage_min=calibrated.stop_coverage_min,
            stop_std_max=calibrated.stop_std_max,
            query_strategy=calibrated.query_strategy,
            block_low_coverage_answer=calibrated.block_low_coverage_answer,
        )
    return ControllerConfigV64(
        mode=mode,
        min_retrievals=args.min_retrievals,
        max_retrievals=args.max_retrievals,
        injection_after_retrievals=args.injection_after_retrievals,
        stop_coverage_min=args.stop_coverage_min,
        stop_std_max=args.stop_std_max,
        query_strategy=args.query_strategy,
        block_low_coverage_answer=bool(args.block_low_coverage_answer),
    )


def summarize_runtime_arm(
    states: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    oracle_entities: Mapping[str, str],
) -> Dict[str, Any]:
    if not states:
        raise ValueError("runtime arm summary requires non-empty states")
    question_lookup = {str(item["question_id"]): item for item in questions}
    bound_states = 0
    bound_slot_total = 0
    assessable_bindings = 0
    correct_bindings = 0
    predicted_assessable_bindings = 0
    interventions = 0
    sf_recalls = []
    for state in states:
        belief = state["belief"].coverage_belief
        entities = {
            normalize_answer(entity)
            for slot in belief.slots
            if slot.bound
            for entity in slot.bound_entities
            if normalize_answer(entity)
        }
        bound_slot_count = sum(bool(slot.bound) for slot in belief.slots)
        bound_states += int(bool(entities))
        bound_slot_total += bound_slot_count
        oracle = oracle_entities.get(str(state["question_id"]))
        if oracle:
            assessable_bindings += 1
            predicted_assessable_bindings += int(bool(entities))
            correct_bindings += int(normalize_answer(oracle) in entities)
        item = question_lookup[str(state["question_id"])]
        gold_titles = {
            normalize_answer(title) for title in (item.get("sf_titles") or [])
        }
        if gold_titles:
            retrieved_titles = {
                normalize_answer(_document_label(document))
                for document in state["retrieved_documents"]
                if _document_label(document)
            }
            sf_recalls.append(
                len(retrieved_titles & gold_titles) / len(gold_titles)
            )
        interventions += int(
            any(
                event.get("controller_action")
                in {
                    ControllerAction.OVERRIDE_ANSWER_WITH_QUERY.value,
                    ControllerAction.FORCE_ANSWER.value,
                }
                or bool(event.get("injected_entity"))
                or (
                    event.get("kind") == "policy_query"
                    and event.get("executed_query") != event.get("policy_query")
                )
                for event in state["events"]
            )
        )
    count = len(states)
    retrieval_values = [int(state["retrieval_calls"]) for state in states]
    generation_values = [int(state["generation_calls"]) for state in states]
    return {
        "trajectories": count,
        "questions": len({str(state["question_id"]) for state in states}),
        "answer_rate": sum(bool(state["answered"]) for state in states) / count,
        "strict_format_valid_answer_rate": sum(
            bool(state["answered"] and state["format_valid"]) for state in states
        )
        / count,
        "mean_retrieval_calls": sum(retrieval_values) / count,
        "max_retrieval_calls_observed": max(retrieval_values),
        "mean_generation_calls": sum(generation_values) / count,
        "max_generation_calls_observed": max(generation_values),
        "empty_retrieval_rate": sum(
            int(state["empty_retrievals"]) for state in states
        )
        / max(sum(retrieval_values), 1),
        "controller_intervention_rate": interventions / count,
        "binding_trajectory_rate": bound_states / count,
        "mean_bound_slots": bound_slot_total / count,
        "oracle_binding_assessable_trajectories": assessable_bindings,
        "oracle_entity_binding_precision": (
            correct_bindings / predicted_assessable_bindings
            if predicted_assessable_bindings
            else None
        ),
        "oracle_entity_binding_recall": (
            correct_bindings / assessable_bindings if assessable_bindings else None
        ),
        "mean_gold_sf_title_recall": (
            sum(sf_recalls) / len(sf_recalls) if sf_recalls else None
        ),
    }


def paired_selection_metrics(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    left_by_id = {
        str(row["question_id"]): row
        for row in left_rows
    }
    right_by_id = {
        str(row["question_id"]): row
        for row in right_rows
    }
    return {
        metric: paired_bootstrap_delta_v64(
            {qid: float(row[metric]) for qid, row in left_by_id.items()},
            {qid: float(row[metric]) for qid, row in right_by_id.items()},
            iterations=iterations,
            seed=seed + offset,
        )
        for offset, metric in enumerate(("em", "f1"))
    }


def paired_oracle_headroom(
    oracle_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    oracle_by_id = {
        str(row["question_id"]): row for row in oracle_rows
    }
    baseline_by_id = {
        str(row["question_id"]): row for row in baseline_rows
    }
    return {
        metric: paired_bootstrap_delta_v64(
            {
                qid: float(row[f"oracle_{metric}"])
                for qid, row in oracle_by_id.items()
            },
            {
                qid: float(row[metric])
                for qid, row in baseline_by_id.items()
            },
            iterations=iterations,
            seed=seed + offset,
        )
        for offset, metric in enumerate(("em", "f1"))
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _parse_csv_ints(value: str) -> List[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def _parse_csv_strings(value: str) -> List[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_label", default="")
    parser.add_argument("--acec_artifact_v5", required=True)
    parser.add_argument("--adapter", action="append", type=parse_adapter, default=[])
    parser.add_argument("--controller", action="append", choices=[mode.value for mode in ControllerMode])
    parser.add_argument("--verifier_modes", type=_parse_csv_strings, default=[
        "first", "majority", "nli", "acec_zero_shot"
    ])
    parser.add_argument("--verifier_artifact", default=None)
    parser.add_argument("--controller_artifact", default=None)
    parser.add_argument("--k_values", type=_parse_csv_ints, default=[1, 2, 4, 8])
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--max_questions", type=int, default=256)
    parser.add_argument("--question_ids", default=None)
    parser.add_argument("--question_batch_size", type=int, default=16)
    parser.add_argument("--n_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--retrieve_url", default=os.environ.get("RETRIEVE_URL", ""))
    parser.add_argument("--max_docs", type=int, default=10)
    parser.add_argument("--slot_manifest", default=None)
    parser.add_argument("--preseed_slots", type=int, choices=[0, 1], default=1)
    parser.add_argument("--slot_k", type=int, default=2)
    parser.add_argument("--slot_parse_min_coverage", type=float, default=1.0)
    parser.add_argument("--oracle_entity_manifest", default=None)
    parser.add_argument("--min_retrievals", type=int, default=1)
    parser.add_argument("--max_retrievals", type=int, default=5)
    parser.add_argument("--injection_after_retrievals", type=int, default=1)
    parser.add_argument("--stop_coverage_min", type=float, default=0.75)
    parser.add_argument("--stop_std_max", type=float, default=0.25)
    parser.add_argument(
        "--query_strategy",
        choices=["append_entity", "slot_only", "slot_plus_entity"],
        default="slot_plus_entity",
    )
    parser.add_argument("--block_low_coverage_answer", type=int, choices=[0, 1], default=1)
    parser.add_argument("--e5_model_path", required=True)
    parser.add_argument(
        "--acec_nli_model", default="cross-encoder/nli-deberta-v3-base"
    )
    parser.add_argument("--acec_tau_new", type=float, default=None)
    parser.add_argument("--nli_batch_size", type=int, default=128)
    parser.add_argument("--vllm_gpu_mem_frac", type=float, default=0.60)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    parser.add_argument("--bootstrap_iterations", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.retrieve_url:
        raise ValueError("--retrieve_url or RETRIEVE_URL is required")
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v6.4 inference: {output_dir}")
    required_paths = [
        Path(args.model_path).expanduser(),
        Path(args.data_path).expanduser(),
        Path(args.acec_artifact_v5).expanduser(),
        Path(args.e5_model_path).expanduser(),
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"required v6.4 input is missing: {path}")
    if max(args.k_values) > args.n_samples:
        raise ValueError("largest K cannot exceed --n_samples")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("--temperature must be in [0, 2]")
    if (
        args.n_samples <= 0
        or args.question_batch_size <= 0
        or args.n_turns <= 0
        or args.max_docs <= 0
        or args.slot_k <= 0
        or args.nli_batch_size <= 0
        or args.bootstrap_iterations <= 0
    ):
        raise ValueError("sample, batch, turn, document, slot, and NLI sizes must be positive")
    if not 0.0 <= args.slot_parse_min_coverage <= 1.0:
        raise ValueError("--slot_parse_min_coverage must be in [0, 1]")
    if not 0.0 < args.vllm_gpu_mem_frac <= 1.0:
        raise ValueError("--vllm_gpu_mem_frac must be in (0, 1]")
    controllers = [
        ControllerMode(value)
        for value in (args.controller or ["none", "belief_gap", "belief_adaptive"])
    ]
    if len(controllers) != len(set(controllers)):
        raise ValueError("controller arms must be unique")
    if len(args.verifier_modes) != len(set(args.verifier_modes)):
        raise ValueError("verifier modes must be unique")
    supported_verifiers = {
        "first",
        "majority",
        "nli",
        "acec_zero_shot",
        "acec_calibrated",
    }
    unsupported_verifiers = set(args.verifier_modes) - supported_verifiers
    if unsupported_verifiers:
        raise ValueError(
            f"unsupported verifier modes: {sorted(unsupported_verifiers)}"
        )
    adapter_names = [name for name, _ in args.adapter]
    if "frozen_r3" in adapter_names or len(adapter_names) != len(set(adapter_names)):
        raise ValueError("adapter policy names must be unique and not frozen_r3")
    oracle_entities = load_oracle_entities(args.oracle_entity_manifest)
    if any(
        mode in {ControllerMode.GOLD_ENTITY, ControllerMode.GOLD_ENTITY_ADAPTIVE}
        for mode in controllers
    ) and not oracle_entities:
        raise ValueError("gold-entity controller requires --oracle_entity_manifest")

    question_ids = _read_id_file(args.question_ids)
    questions = load_questions(
        Path(args.data_path).expanduser(),
        max_questions=args.max_questions,
        seed=args.seed,
        question_ids=question_ids,
    )
    if oracle_entities:
        missing_oracle = [
            item["question_id"]
            for item in questions
            if any(
                mode in {
                    ControllerMode.GOLD_ENTITY,
                    ControllerMode.GOLD_ENTITY_ADAPTIVE,
                }
                for mode in controllers
            )
            and item["question_id"] not in oracle_entities
        ]
        if missing_oracle:
            raise ValueError(
                f"oracle controller questions lack bridge entities: {missing_oracle[:5]}"
            )

    verifier_artifact = (
        load_answer_verifier_artifact_v64(args.verifier_artifact)
        if args.verifier_artifact
        else None
    )
    if "acec_calibrated" in args.verifier_modes and verifier_artifact is None:
        raise ValueError("acec_calibrated mode requires --verifier_artifact")
    controller_artifact = (
        load_controller_artifact_v64(args.controller_artifact)
        if args.controller_artifact
        else None
    )
    if controller_artifact is not None and (
        not controller_artifact.calibration_split_id_hash
        or controller_artifact.expected_mean_retrievals is None
        or (controller_artifact.metadata.get("gate") or {}).get("pass") is not True
    ):
        raise ValueError(
            "controller artifact must pass its cost-match gate and record its split"
        )
    heldout_ids = {str(item["question_id"]) for item in questions}
    if verifier_artifact is not None:
        verifier_calibration_ids = {
            str(value)
            for key in ("fit_question_ids", "validation_question_ids")
            for value in verifier_artifact.metadata.get(key, [])
        }
        if not verifier_calibration_ids:
            raise ValueError(
                "verifier artifact must record explicit fit/validation question ids"
            )
        overlap = heldout_ids & verifier_calibration_ids
        if overlap:
            raise ValueError(
                f"verifier calibration overlaps held-out ids: {sorted(overlap)[:5]}"
            )
    if controller_artifact is not None:
        controller_calibration_ids = {
            str(value)
            for value in controller_artifact.metadata.get(
                "calibration_question_ids", []
            )
        }
        if not controller_calibration_ids:
            raise ValueError(
                "controller artifact must record explicit calibration question ids"
            )
        overlap = heldout_ids & controller_calibration_ids
        if overlap:
            raise ValueError(
                f"controller calibration overlaps held-out ids: {sorted(overlap)[:5]}"
            )
    for mode in controllers:
        _controller_config(args, mode, controller_artifact)

    validated_adapters: List[Tuple[str, Path]] = []
    adapter_metadata: Dict[str, Dict[str, str]] = {}
    for name, path_string in args.adapter:
        path = Path(path_string).expanduser()
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (path / filename).is_file():
                raise FileNotFoundError(f"adapter {name} lacks {filename}: {path}")
        validated_adapters.append((name, path))
        adapter_metadata[name] = {
            "path": str(path),
            "adapter_config_sha256": _sha256(path / "adapter_config.json"),
            "adapter_model_sha256": _sha256(path / "adapter_model.safetensors"),
        }
    for optional_path in (args.slot_manifest, args.oracle_entity_manifest):
        if optional_path and not Path(optional_path).expanduser().is_file():
            raise FileNotFoundError(f"optional v6.4 input is missing: {optional_path}")

    output_dir.mkdir(parents=True, exist_ok=False)
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    import grpo_rsf_simple
    import grpo_rsf_vllm_v5

    grpo_rsf_simple.RETRIEVE_URL = args.retrieve_url
    grpo_rsf_simple.MAX_DOCS = args.max_docs
    variants: List[Tuple[str, Any]] = [("frozen_r3", None)]
    for adapter_index, (name, path) in enumerate(validated_adapters, start=1):
        variants.append((name, LoRARequest(name, adapter_index, str(path))))

    llm = LLM(
        model=args.model_path,
        enable_lora=bool(args.adapter),
        max_lora_rank=args.max_lora_rank,
        max_loras=max(2, len(args.adapter)),
        gpu_memory_utilization=args.vllm_gpu_mem_frac,
        max_model_len=4096,
        dtype="bfloat16",
        seed=args.seed,
        disable_log_stats=True,
    )
    belief_factory = grpo_rsf_vllm_v5.make_belief_factory_v5(args)

    slot_manifest: Dict[str, SlotDecompositionV64] = {}
    slot_audit: List[Dict[str, Any]] = []
    if args.preseed_slots:
        if args.slot_manifest:
            slot_manifest = load_slot_manifest(Path(args.slot_manifest).expanduser())
        else:
            slot_manifest, slot_audit = generate_slot_manifest(
                llm,
                questions,
                expected_k=args.slot_k,
                generator_name=f"{args.model_path}:base_temperature_0",
                batch_size=args.question_batch_size,
            )
        coverage = sum(
            item["question_id"] in slot_manifest for item in questions
        ) / len(questions)
        if coverage < args.slot_parse_min_coverage:
            _write_jsonl(output_dir / "slot_generation_audit.jsonl", slot_audit)
            raise ValueError(
                f"slot parse coverage {coverage:.4f} is below "
                f"{args.slot_parse_min_coverage:.4f}"
            )
        missing = [
            item["question_id"]
            for item in questions
            if item["question_id"] not in slot_manifest
        ]
        if missing:
            raise ValueError(f"questions missing slot decompositions: {missing[:5]}")
        _write_jsonl(
            output_dir / "slot_manifest.jsonl",
            (slot_manifest[item["question_id"]].to_dict() for item in questions),
        )
        if slot_audit:
            _write_jsonl(output_dir / "slot_generation_audit.jsonl", slot_audit)

    all_arms: Dict[str, List[TrajectoryCandidateV64]] = {}
    runtime_diagnostics: Dict[str, Dict[str, Any]] = {}
    grounding_scored = any(
        mode in {"nli", "acec_zero_shot", "acec_calibrated"}
        for mode in args.verifier_modes
    )
    started = time.monotonic()
    for policy_name, lora_request in variants:
        for controller_mode in controllers:
            arm_name = f"{policy_name}__{controller_mode.value}"
            config = _controller_config(args, controller_mode, controller_artifact)
            arm_states: List[Dict[str, Any]] = []
            log.info("[v6.4] START arm=%s", arm_name)
            for offset in range(0, len(questions), args.question_batch_size):
                batch = questions[offset : offset + args.question_batch_size]
                states = rollout_controller_batch(
                    llm,
                    lora_request,
                    batch,
                    policy_name=policy_name,
                    controller_config=config,
                    belief_factory=belief_factory,
                    slot_manifest=slot_manifest,
                    oracle_entities=oracle_entities,
                    n_samples=args.n_samples,
                    n_turns=args.n_turns,
                    temperature=args.temperature,
                    max_docs=args.max_docs,
                )
                arm_states.extend(states)
                log.info(
                    "[v6.4] arm=%s questions=%d/%d",
                    arm_name,
                    min(offset + len(batch), len(questions)),
                    len(questions),
                )
            grounding = (
                score_grounding_batch(arm_states, args.nli_batch_size)
                if grounding_scored
                else [0.0] * len(arm_states)
            )
            candidates = states_to_candidates(
                arm_states,
                grounding,
                grounding_scored=grounding_scored,
            )
            all_arms[arm_name] = candidates
            runtime_diagnostics[arm_name] = summarize_runtime_arm(
                arm_states, questions, oracle_entities
            )
            runtime_diagnostics[arm_name]["grounding_scored"] = grounding_scored
            runtime_diagnostics[arm_name]["grounding_nli_pairs"] = sum(
                candidate.grounding_pairs for candidate in candidates
            )
            _write_jsonl(
                output_dir / f"runtime_{arm_name}.jsonl",
                (
                    {
                        "schema": "acec_controller_runtime",
                        "version": 64,
                        "question_id": state["question_id"],
                        "sample_index": state["sample_index"],
                        "question": state["question"],
                        "policy": policy_name,
                        "controller": controller_mode.value,
                        "answer": state["answer"],
                        "raw_answer": state["raw_answer"],
                        "answered": state["answered"],
                        "format_valid": state["format_valid"],
                        "answer_parse_status": state["answer_parse_status"],
                        "answer_contract_version": ANSWER_CONTRACT_VERSION,
                        "retrieval_calls": state["retrieval_calls"],
                        "generation_calls": state["generation_calls"],
                        "events": state["events"],
                    }
                    for state in arm_states
                ),
            )
            question_lookup = {item["question_id"]: item for item in questions}
            _write_jsonl(
                output_dir / f"evaluator_features_{arm_name}.jsonl",
                (
                    {
                        **candidate.to_dict(),
                        "answer_correct": bool(
                            answer_exact_match(
                                candidate.answer,
                                question_lookup[candidate.question_id][
                                    "golden_answers"
                                ],
                            )
                        ),
                    }
                    for candidate in candidates
                ),
            )

    gold_answers = {
        item["question_id"]: item["golden_answers"] for item in questions
    }
    _write_jsonl(
        output_dir / "manifest.jsonl",
        (
            {
                "id": item["question_id"],
                "question": item["question"],
                "golden_answers": item["golden_answers"],
                "question_type": item.get("question_type"),
                "sf_titles": item.get("sf_titles", []),
            }
            for item in questions
        ),
    )
    factorial = evaluate_policy_controller_factorial_v64(
        all_arms,
        gold_answers,
        k_values=args.k_values,
        modes=args.verifier_modes,
        artifact=verifier_artifact,
    )
    sampling_headroom: Dict[str, Any] = {}
    for arm_index, (arm_name, arm_result) in enumerate(factorial.items()):
        selections = arm_result["selections"]
        baseline_key = next(
            (
                key
                for key in selections
                if key.endswith("@1")
            ),
            None,
        )
        if baseline_key is None:
            continue
        sampling_headroom[arm_name] = {}
        for k_index, k in enumerate(sorted(set(args.k_values))):
            oracle_key = next(
                (
                    key
                    for key in selections
                    if key.endswith(f"@{k}")
                ),
                None,
            )
            if oracle_key is None:
                continue
            sampling_headroom[arm_name][f"oracle@{k}_minus_pass@1"] = {
                "oracle_em": arm_result["summaries"][oracle_key]["oracle_em"],
                "oracle_f1": arm_result["summaries"][oracle_key]["oracle_f1"],
                "paired_headroom": paired_oracle_headroom(
                    selections[oracle_key],
                    selections[baseline_key],
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + arm_index * 100 + k_index * 10,
                ),
            }
    selection_dir = output_dir / "selected_predictions"
    selection_dir.mkdir()
    for arm_name, arm_result in factorial.items():
        for selector_key, rows in arm_result["selections"].items():
            safe_key = selector_key.replace("@", "_k")
            # `prediction` 必须是**原始自由文本**，不是标签契约裁剪后的结果。
            # v6.3 的 tagged_short_answer 契约要求策略输出 <answer>...</answer>，
            # 而 R3-RAG-Qwen 是按 `Step N:` 模板刚性微调的，实测 96.5–98.8% 的
            # 轨迹判为 missing_answer_tag —— 但 raw_answer 里躺着 "Selma."
            # 这类完全正确的短答案。若此处沿用契约结果，写出的全是空串，
            # eval_r3_processed_answers_api 的抽取器拿不到任何可抽取的文本，
            # processed-EM 这条兜底通路就形同虚设（v5 管线保留的正是自由文本）。
            #
            # 严格 Answer EM 不受影响：它由 results.json 里的 em/f1 承载，
            # 这里额外落 contract_answer/contract_valid 以便两种口径并列报告。
            # golden_answers 是 load_inputs() 的硬性要求，缺了会直接 ValueError。
            _write_jsonl(
                selection_dir / f"{arm_name}__{safe_key}.jsonl",
                (
                    {
                        "question_id": row["question_id"],
                        "id": row["question_id"],
                        "answer": _free_text_prediction(row),
                        "prediction": _free_text_prediction(row),
                        "answered": bool(_free_text_prediction(row)),
                        "golden_answers": list(
                            gold_answers.get(row["question_id"], [])
                        ),
                        "em": float(row["em"]),
                        "f1": float(row["f1"]),
                        "contract_answer": row["selected_answer"],
                        "contract_valid": bool(row["selected_answer"]),
                        "selected_sample_index": row["selected_sample_index"],
                        "policy_controller_arm": arm_name,
                        "verifier": row["mode"],
                        "k": row["k"],
                    }
                    for row in rows
                ),
            )

    paired_vs_no_controller: Dict[str, Any] = {}
    for policy_name, _ in variants:
        baseline_arm = f"{policy_name}__none"
        if baseline_arm not in factorial:
            continue
        for controller_mode in controllers:
            controlled_arm = f"{policy_name}__{controller_mode.value}"
            if controlled_arm == baseline_arm:
                continue
            for selector_key in factorial[baseline_arm]["selections"]:
                baseline_rows = factorial[baseline_arm]["selections"][selector_key]
                controlled_rows = factorial[controlled_arm]["selections"][selector_key]
                paired_vs_no_controller[
                    f"{controlled_arm}__{selector_key}"
                ] = paired_selection_metrics(
                    controlled_rows,
                    baseline_rows,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed,
                )

    global_reference_arms = []
    frozen_reference = "frozen_r3__none"
    if frozen_reference in factorial:
        global_reference_arms.append(frozen_reference)
    outcome_none_arms = [
        arm
        for arm in sorted(factorial)
        if "outcome" in arm.casefold() and arm.endswith("__none")
    ]
    global_reference_arms.extend(outcome_none_arms)
    paired_vs_references: Dict[str, Any] = {}
    for target_arm, target_result in factorial.items():
        target_policy, target_controller = target_arm.rsplit("__", 1)
        if target_policy == "frozen_r3" or "outcome" in target_policy.casefold():
            continue
        matched_outcome_arms = [
            arm
            for arm in sorted(factorial)
            if "outcome" in arm.casefold()
            and arm.endswith(f"__{target_controller}")
        ]
        reference_arms = list(
            dict.fromkeys(global_reference_arms + matched_outcome_arms)
        )
        for reference_index, reference_arm in enumerate(reference_arms):
            if target_arm == reference_arm:
                continue
            shared_selectors = sorted(
                set(target_result["selections"])
                & set(factorial[reference_arm]["selections"])
            )
            for selector_index, selector_key in enumerate(shared_selectors):
                comparison_key = (
                    f"{target_arm}__minus__{reference_arm}__{selector_key}"
                )
                paired_vs_references[comparison_key] = paired_selection_metrics(
                    target_result["selections"][selector_key],
                    factorial[reference_arm]["selections"][selector_key],
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + reference_index * 1000 + selector_index * 10,
                )

    contract = {
        "schema": "acec_v64_factorial_run",
        "version": 64,
        "git_commit": _git_commit(),
        "run_label": args.run_label,
        "model_path": args.model_path,
        "data_path": args.data_path,
        "data_sha256": _sha256(Path(args.data_path).expanduser()),
        "acec_artifact_v5": args.acec_artifact_v5,
        "acec_artifact_sha256": _sha256(
            Path(args.acec_artifact_v5).expanduser()
        ),
        "policies": [name for name, _ in variants],
        "policy_adapters": adapter_metadata,
        "controllers": [mode.value for mode in controllers],
        "verifier_modes": args.verifier_modes,
        "k_values": args.k_values,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "seed": args.seed,
        "question_count": len(questions),
        "question_ids": sorted(heldout_ids),
        "question_id_sha256": _id_hash(
            item["question_id"] for item in questions
        ),
        "preseed_slots": bool(args.preseed_slots),
        "slot_k": args.slot_k,
        "slot_parse_min_coverage": args.slot_parse_min_coverage,
        "slot_manifest_sha256": (
            _sha256(output_dir / "slot_manifest.jsonl")
            if (output_dir / "slot_manifest.jsonl").is_file()
            else None
        ),
        "question_batch_size": args.question_batch_size,
        "n_turns": args.n_turns,
        "max_docs": args.max_docs,
        "nli_batch_size": args.nli_batch_size,
        "vllm_gpu_mem_frac": args.vllm_gpu_mem_frac,
        "max_lora_rank": args.max_lora_rank,
        "bootstrap_iterations": args.bootstrap_iterations,
        "e5_model_path": args.e5_model_path,
        "acec_nli_model": args.acec_nli_model,
        "oracle_entity_manifest_sha256": (
            _sha256(Path(args.oracle_entity_manifest).expanduser())
            if args.oracle_entity_manifest
            else None
        ),
        "controller_configs": {
            mode.value: {
                **vars(_controller_config(args, mode, controller_artifact)),
                "mode": mode.value,
            }
            for mode in controllers
        },
        "controller_threshold_source": (
            "question_disjoint_calibration_artifact"
            if controller_artifact is not None
            else "command_line_uncalibrated"
        ),
        "controller_artifact_sha256": (
            _sha256(Path(args.controller_artifact).expanduser())
            if args.controller_artifact
            else None
        ),
        "controller_expected_mean_retrievals": (
            controller_artifact.expected_mean_retrievals
            if controller_artifact is not None
            else None
        ),
        "answer_contract_version": ANSWER_CONTRACT_VERSION,
        "verifier_artifact_sha256": (
            _sha256(Path(args.verifier_artifact).expanduser())
            if args.verifier_artifact
            else None
        ),
        "elapsed_seconds": time.monotonic() - started,
        "factorial": {
            arm: result["summaries"] for arm, result in factorial.items()
        },
        "runtime_diagnostics": runtime_diagnostics,
        "sampling_oracle_headroom": sampling_headroom,
        "paired_controller_minus_none": paired_vs_no_controller,
        "paired_target_minus_references": paired_vs_references,
    }
    with (output_dir / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(contract, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
