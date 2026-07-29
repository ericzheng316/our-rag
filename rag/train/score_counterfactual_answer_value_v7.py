"""Score ACEC v7 slate prefixes by frozen gold-answer log likelihood.

This is answer supervision, not supporting-fact supervision.  Every slate
prefix is scored so the chosen document receives a signed marginal target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _prompt(question: str, history: str, documents: Sequence[str]) -> str:
    evidence = "\n\n".join(
        f"[Evidence {index + 1}]\n{text}"
        for index, text in enumerate(documents)
    )
    parts = [
        "Answer the question with the shortest correct answer.",
        f"Question: {question}",
    ]
    if history.strip():
        parts.append(f"Previous reasoning/retrieval history:\n{history.strip()}")
    if evidence:
        parts.append(f"Evidence:\n{evidence}")
    parts.append("Answer:")
    return "\n\n".join(parts)


class FrozenAnswerLikelihoodScorerV7:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str,
        max_prompt_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_value = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype_value,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_prompt_tokens = int(max_prompt_tokens)

    def _item(self, prompt: str, answer: str) -> Tuple[List[int], int, List[int]]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(
            " " + str(answer).strip(), add_special_tokens=False
        )
        prompt_ids = prompt_ids[-self.max_prompt_tokens :]
        if not prompt_ids or not answer_ids:
            raise ValueError("answer likelihood prompt/answer tokenization is empty")
        return prompt_ids + answer_ids, len(prompt_ids), answer_ids

    def score_many(
        self, requests: Sequence[Tuple[str, Sequence[str]]], *, batch_size: int
    ) -> List[float]:
        expanded = []
        request_ranges = []
        for prompt, answers in requests:
            start = len(expanded)
            for answer in answers:
                expanded.append(self._item(prompt, str(answer)))
            request_ranges.append((start, len(expanded)))
        scores: List[float] = []
        torch = self.torch
        for offset in range(0, len(expanded), batch_size):
            batch = expanded[offset : offset + batch_size]
            max_length = max(len(item[0]) for item in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                int(self.tokenizer.pad_token_id),
                dtype=torch.long,
                device=self.device,
            )
            attention = torch.zeros_like(input_ids)
            for row, (ids, _, _) in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device=self.device
                )
                attention[row, : len(ids)] = 1
            with torch.inference_mode():
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention
                ).logits.float()
                log_probs = torch.log_softmax(logits, dim=-1)
            for row, (_, prompt_length, answer_ids) in enumerate(batch):
                positions = torch.arange(
                    prompt_length - 1,
                    prompt_length + len(answer_ids) - 1,
                    device=self.device,
                )
                tokens = torch.tensor(
                    answer_ids, dtype=torch.long, device=self.device
                )
                values = log_probs[row, positions, tokens]
                scores.append(float(values.mean().item()))
        result = []
        for start, end in request_ranges:
            if start == end:
                raise ValueError("each likelihood request needs a gold answer")
            result.append(max(scores[start:end]))
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_tokens", type=int, default=3072)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    rows = list(_read_jsonl(input_path))
    if not rows:
        raise ValueError("counterfactual input is empty")
    scorer = FrozenAnswerLikelihoodScorerV7(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    for row in rows:
        gold_answers = tuple(str(value) for value in row.get("gold_answers") or ())
        if not gold_answers:
            raise ValueError("answer-value scoring requires gold_answers")
        by_id = {
            str(candidate["candidate_id"]): str(candidate["contents"])
            for candidate in row["candidates"]
        }
        requests: List[Tuple[str, Sequence[str]]] = []
        request_keys: List[Tuple[str, int]] = []
        requests.append(
            (
                _prompt(
                    str(row.get("question") or ""),
                    str(row.get("history") or ""),
                    (),
                ),
                gold_answers,
            )
        )
        request_keys.append(("baseline", 0))
        for slate in row["slates"]:
            prefix: List[str] = []
            for prefix_index, candidate_id in enumerate(slate["selected_ids"], start=1):
                prefix.append(by_id[str(candidate_id)])
                requests.append(
                    (
                        _prompt(
                            str(row.get("question") or ""),
                            str(row.get("history") or ""),
                            prefix,
                        ),
                        gold_answers,
                    )
                )
                request_keys.append((str(slate["slate_id"]), prefix_index))
        values = scorer.score_many(requests, batch_size=args.batch_size)
        keyed = dict(zip(request_keys, values))
        baseline = keyed[("baseline", 0)]
        row["baseline_answer_logprob"] = baseline
        for slate in row["slates"]:
            prefix_values = [
                keyed[(str(slate["slate_id"]), prefix_index)]
                for prefix_index in range(1, len(slate["selected_ids"]) + 1)
            ]
            slate["prefix_answer_logprobs"] = prefix_values
            slate["answer_utility"] = (
                prefix_values[-1] - baseline if prefix_values else 0.0
            )
            slate["prefix_marginal_utilities"] = [
                value - (baseline if index == 0 else prefix_values[index - 1])
                for index, value in enumerate(prefix_values)
            ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"states": len(rows), "output": str(output_path)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
