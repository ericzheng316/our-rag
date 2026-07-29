from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stdout
from io import StringIO
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import types
import tempfile
import unittest
from unittest.mock import patch

from belief.acec.inference_controller_v64 import (
    BeliefInferenceControllerV64,
    ControllerAction,
    ControllerArtifactV64,
    ControllerConfigV64,
    ControllerMode,
    build_gap_directed_query,
    derive_gold_bridge_entity_v64,
    load_controller_artifact_v64,
    parse_slot_decomposition,
    preseed_belief_slots,
    save_controller_artifact_v64,
)


@dataclass
class _Slot:
    hypothesis: str
    bound: bool = False
    bound_entities: tuple[str, ...] = ()


class _Coverage:
    def __init__(
        self,
        *,
        coverage: float = 0.4,
        variance: float = 0.04,
        stop: bool = False,
        target: int | None = 0,
    ) -> None:
        self.config = SimpleNamespace(k_max=3)
        self.slots: list[_Slot] = [
            _Slot("identify the intermediate work", True, ("Bridge Work",)),
            _Slot("find the creator of that work"),
        ]
        self.p = [0.9, 0.2]
        self.turn = 1
        self._coverage = coverage
        self._variance = variance
        self._stop = stop
        self._target = target

    def coverage(self) -> float:
        return self._coverage

    def coverage_variance(self) -> float:
        return self._variance

    def should_stop_voi(self) -> bool:
        return self._stop

    def suggest_target_slot(self) -> int | None:
        return self._target

    def spawn_slot(self, hypothesis: str) -> None:
        self.slots.append(_Slot(hypothesis))
        self.p.append(0.0)


class _RuntimeBelief:
    def __init__(self) -> None:
        self.coverage_belief = _Coverage()
        self.labeler = SimpleNamespace(tau_new=0.8)
        self.nli_scorer = SimpleNamespace()

    def turn(self, query, new_docs):
        self.coverage_belief.turn += 1
        return SimpleNamespace(
            action=SimpleNamespace(
                mode=SimpleNamespace(value="rewrite"),
                target_slot=0,
            ),
            coverage_before=self.coverage_belief.coverage(),
            coverage_after=self.coverage_belief.coverage(),
            delta_coverage=0.0,
            stop_voi=False,
            features=SimpleNamespace(slot_p=list(self.coverage_belief.p)),
        )


def _belief(**kwargs):
    return SimpleNamespace(
        coverage_belief=_Coverage(**kwargs),
        labeler=SimpleNamespace(tau_new=0.8),
    )


class InferenceControllerV64Test(unittest.TestCase):
    def test_parse_and_preseed_fixed_slots(self):
        decomposition = parse_slot_decomposition(
            "<slot>find the bridge</slot>\n<slot>find its author</slot>",
            question_id="q1",
            question="Who wrote the work?",
            expected_k=2,
            generator="frozen-r3",
        )
        self.assertEqual(
            decomposition.slots, ("find the bridge", "find its author")
        )

        belief = _belief()
        belief.coverage_belief.slots = []
        belief.coverage_belief.p = []
        belief.coverage_belief.turn = 0
        preseed_belief_slots(belief, decomposition.slots)
        self.assertEqual(
            [slot.hypothesis for slot in belief.coverage_belief.slots],
            list(decomposition.slots),
        )
        self.assertEqual(belief.labeler.tau_new, -1.0)
        self.assertTrue(belief.v64_slot_count_frozen)

        with self.assertRaisesRegex(ValueError, "expected 2"):
            parse_slot_decomposition(
                "<slot>only one</slot>",
                question_id="q1",
                question="Question",
                expected_k=2,
                generator="frozen-r3",
            )

    def test_none_controller_is_a_true_noop(self):
        controller = BeliefInferenceControllerV64(
            ControllerConfigV64(mode=ControllerMode.NONE)
        )
        query = controller.decide(
            question="q",
            belief=None,
            retrieval_calls=0,
            policy_query="original query",
        )
        answer = controller.decide(
            question="q",
            belief=None,
            retrieval_calls=0,
            policy_answer="answer",
        )
        self.assertEqual(query.action, ControllerAction.EXECUTE_QUERY)
        self.assertEqual(query.query, "original query")
        self.assertEqual(answer.action, ControllerAction.ACCEPT_POLICY_ANSWER)
        monitor = BeliefInferenceControllerV64(
            ControllerConfigV64(
                mode=ControllerMode.MONITOR_PRESEED,
                max_retrievals=1,
            )
        ).decide(
            question="q",
            belief=_belief(),
            retrieval_calls=1,
            policy_query="still unchanged",
        )
        self.assertEqual(monitor.action, ControllerAction.EXECUTE_QUERY)
        self.assertEqual(monitor.query, "still unchanged")
        self.assertEqual(monitor.reason, "monitor_only")

    def test_gap_controller_targets_slot_and_injects_bound_entity(self):
        controller = BeliefInferenceControllerV64(
            ControllerConfigV64(
                mode=ControllerMode.BELIEF_GAP,
                injection_after_retrievals=1,
            )
        )
        decision = controller.decide(
            question="q",
            belief=_belief(),
            retrieval_calls=1,
            policy_query="generic query",
        )
        self.assertEqual(decision.action, ControllerAction.EXECUTE_QUERY)
        self.assertEqual(decision.injected_entity, "Bridge Work")
        self.assertEqual(decision.target_slot, 0)
        self.assertEqual(
            decision.query, "identify the intermediate work Bridge Work"
        )

    def test_adaptive_rejects_low_coverage_then_stops_when_ready(self):
        controller = BeliefInferenceControllerV64(
            ControllerConfigV64(
                mode=ControllerMode.BELIEF_ADAPTIVE,
                min_retrievals=1,
                stop_coverage_min=0.75,
                stop_std_max=0.25,
            )
        )
        rejected = controller.decide(
            question="q",
            belief=_belief(coverage=0.4, variance=0.04, stop=False),
            retrieval_calls=1,
            policy_answer="premature",
        )
        self.assertEqual(
            rejected.action, ControllerAction.OVERRIDE_ANSWER_WITH_QUERY
        )
        self.assertEqual(rejected.query, "identify the intermediate work")

        accepted = controller.decide(
            question="q",
            belief=_belief(coverage=0.9, variance=0.01, stop=True),
            retrieval_calls=2,
            policy_answer="ready",
        )
        self.assertEqual(accepted.action, ControllerAction.ACCEPT_POLICY_ANSWER)
        self.assertEqual(accepted.reason, "belief_stop_accepts_answer")

        forced = controller.decide(
            question="q",
            belief=_belief(coverage=0.9, variance=0.01, stop=True),
            retrieval_calls=2,
            policy_query="another query",
        )
        self.assertEqual(forced.action, ControllerAction.FORCE_ANSWER)

    def test_max_budget_and_explicit_gold_mode(self):
        belief = _belief()
        maxed = BeliefInferenceControllerV64(
            ControllerConfigV64(
                mode=ControllerMode.BELIEF_GAP,
                max_retrievals=2,
            )
        ).decide(
            question="q",
            belief=belief,
            retrieval_calls=2,
            policy_query="query",
        )
        self.assertEqual(maxed.action, ControllerAction.FORCE_ANSWER)

        gold = BeliefInferenceControllerV64(
            ControllerConfigV64(
                mode=ControllerMode.GOLD_ENTITY,
                injection_after_retrievals=1,
            )
        )
        with self.assertRaisesRegex(ValueError, "oracle entity"):
            gold.decide(
                question="q",
                belief=belief,
                retrieval_calls=1,
                policy_query="query",
            )
        injected = gold.decide(
            question="q",
            belief=belief,
            retrieval_calls=1,
            policy_query="query",
            gold_entity="Gold Bridge",
        )
        self.assertEqual(injected.query, "query Gold Bridge")

    def test_query_dedup_and_immutable_controller_artifact(self):
        self.assertEqual(
            build_gap_directed_query(
                "Bridge Work",
                "Bridge Work",
                "Bridge Work",
                strategy="slot_plus_entity",
            ),
            "Bridge Work",
        )
        artifact = ControllerArtifactV64(
            config=ControllerConfigV64(mode=ControllerMode.BELIEF_ADAPTIVE),
            calibration_split_id_hash="abc",
            expected_mean_retrievals=1.5,
            metadata={"source": "test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.json"
            save_controller_artifact_v64(path, artifact)
            self.assertEqual(load_controller_artifact_v64(path), artifact)
            with self.assertRaises(FileExistsError):
                save_controller_artifact_v64(path, artifact)

    def test_bridge_oracle_uses_annotated_cross_title_mention(self):
        record = {
            "_id": "bridge-1",
            "type": "bridge",
            "question": "Who directed the film based on the novel?",
            "answer": "A Director",
            "supporting_facts": [["Novel A", 0], ["Film B", 0]],
            "context": [
                ["Novel A", ["Novel A was adapted into Film B."]],
                ["Film B", ["Film B was directed by A Director."]],
            ],
        }
        label = derive_gold_bridge_entity_v64(record)
        self.assertTrue(label.assessable)
        self.assertEqual(label.entity, "Film B")
        self.assertEqual(label.basis, "supporting_sentence_cross_title")
        self.assertEqual(label.schema, "acec_bridge_entity_label")
        self.assertEqual(label.version, 64)

        rejected = derive_gold_bridge_entity_v64(dict(record, answer="Film B"))
        self.assertFalse(rejected.assessable)
        self.assertEqual(rejected.basis, "bridge_entity_equals_gold_answer")

    def test_cost_matched_sweep_freezes_passing_artifact(self):
        from rag.train.fit_controller_artifact_v64 import main as fit_main

        calibration_ids = ["cal-1", "cal-2"]
        calibration_hash = hashlib.sha256(
            "\n".join(calibration_ids).encode("utf-8")
        ).hexdigest()
        base = {
            "schema": "acec_v64_factorial_run",
            "question_id_sha256": calibration_hash,
            "question_ids": calibration_ids,
            "data_sha256": "data-hash",
            "controller_configs": {
                "belief_adaptive": {
                    "mode": "belief_adaptive",
                    "min_retrievals": 1,
                    "max_retrievals": 5,
                    "injection_after_retrievals": 1,
                    "stop_coverage_min": 0.75,
                    "stop_std_max": 0.25,
                    "query_strategy": "slot_plus_entity",
                    "block_low_coverage_answer": True,
                }
            },
            "runtime_diagnostics": {
                "acec_ep50__belief_adaptive": {"mean_retrieval_calls": 1.0}
            },
            "factorial": {
                "acec_ep50__belief_adaptive": {"first@1": {"em": 0.4}}
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            output = root / "controller.json"
            first.write_text(json.dumps(base), encoding="utf-8")
            second_payload = json.loads(json.dumps(base))
            second_payload["runtime_diagnostics"]["acec_ep50__belief_adaptive"][
                "mean_retrieval_calls"
            ] = 1.5
            second_payload["factorial"]["acec_ep50__belief_adaptive"]["first@1"][
                "em"
            ] = 0.42
            second.write_text(json.dumps(second_payload), encoding="utf-8")
            argv = [
                "fit_controller_artifact_v64.py",
                "--sweep_result",
                str(first),
                "--sweep_result",
                str(second),
                "--arm",
                "acec_ep50__belief_adaptive",
                "--target_mean_retrievals",
                "1.48",
                "--max_mean_cost_gap",
                "0.05",
                "--output",
                str(output),
            ]
            with patch("sys.argv", argv), redirect_stdout(StringIO()):
                fit_main()
            artifact = load_controller_artifact_v64(output)
            self.assertEqual(artifact.expected_mean_retrievals, 1.5)
            self.assertTrue(artifact.metadata["gate"]["pass"])

    def test_rollout_preserves_baseline_turn_limit_and_completes_forced_answer(
        self,
    ):
        from rag.train.infer_belief_controller_v64 import rollout_controller_batch

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kwargs: kwargs
        fake_r3 = types.ModuleType("grpo_rsf_simple")
        fake_r3.STOP_TOKEN = 151645
        fake_r3.apply_chat_template = lambda value: value
        fake_r3.parse_step = lambda text: (
            {
                "analysis": "retrieve",
                "query": "query",
            }
            if "QUERY" in text
            else {
                "analysis": "answer",
                "answer": "<answer>right</answer>",
            }
        )
        fake_r3._retrieve_batch = lambda queries: [
            [{"id": f"d{index}", "title": "Doc", "contents": "evidence"}]
            for index, _ in enumerate(queries)
        ]

        class _FakeLLM:
            def __init__(self, responses):
                self.responses = list(responses)

            def generate(self, prompts, params, **kwargs):
                response = self.responses.pop(0)
                return [
                    SimpleNamespace(
                        outputs=[SimpleNamespace(text=response)]
                    )
                    for _ in prompts
                ]

        question = {
            "question_id": "q",
            "question": "Question?",
            "golden_answers": ["right"],
            "sf_titles": [],
        }
        modules = {"vllm": fake_vllm, "grpo_rsf_simple": fake_r3}
        with patch.dict(sys.modules, modules):
            baseline = rollout_controller_batch(
                _FakeLLM(["QUERY"]),
                None,
                [question],
                policy_name="frozen_r3",
                controller_config=ControllerConfigV64(
                    mode=ControllerMode.NONE
                ),
                belief_factory=lambda _: _RuntimeBelief(),
                slot_manifest={},
                oracle_entities={},
                n_samples=1,
                n_turns=1,
                temperature=0.0,
                max_docs=5,
            )[0]
            self.assertFalse(baseline["answered"])
            self.assertEqual(
                baseline["answer_parse_status"],
                "max_model_turns_without_answer",
            )
            self.assertEqual(baseline["generation_calls"], 1)

            adaptive = rollout_controller_batch(
                _FakeLLM(["ANSWER", "ANSWER"]),
                None,
                [question],
                policy_name="acec_ep50",
                controller_config=ControllerConfigV64(
                    mode=ControllerMode.BELIEF_ADAPTIVE,
                    min_retrievals=1,
                    max_retrievals=5,
                ),
                belief_factory=lambda _: _RuntimeBelief(),
                slot_manifest={},
                oracle_entities={},
                n_samples=1,
                n_turns=1,
                temperature=0.0,
                max_docs=5,
            )[0]
            self.assertTrue(adaptive["answered"])
            self.assertEqual(adaptive["answer"], "right")
            self.assertEqual(adaptive["retrieval_calls"], 1)
            self.assertEqual(adaptive["generation_calls"], 2)
            self.assertEqual(
                adaptive["events"][-1]["reason"],
                "forced_answer_completion",
            )


if __name__ == "__main__":
    unittest.main()
