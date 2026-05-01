"""
Simple EM / F1 eval for IRCoT output JSONL.
Matches the normalisation used in FlashRAG / hotpotqa official eval.
"""

import argparse
import json
import re
import string
import os
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match(pred: str, golds: list) -> int:
    p = normalize(pred)
    return int(any(p == normalize(g) for g in golds))


def token_f1(pred: str, golds: list) -> float:
    p_tokens = Counter(normalize(pred).split())
    best = 0.0
    for g in golds:
        g_tokens = Counter(normalize(g).split())
        common = sum((p_tokens & g_tokens).values())
        if not common:
            continue
        precision = common / sum(p_tokens.values())
        recall    = common / sum(g_tokens.values())
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_file", required=True)
    parser.add_argument("--output_dir",   required=True)
    args = parser.parse_args()

    records = []
    with open(args.records_file) as f:
        for line in f:
            records.append(json.loads(line))

    em_scores, f1_scores, turns = [], [], []
    for r in records:
        pred   = r.get("pred", "")
        golds  = r.get("golden_answers", [])
        em_scores.append(exact_match(pred, golds))
        f1_scores.append(token_f1(pred, golds))
        turns.append(r.get("num_turns", 0))

    n = len(records)
    results = {
        "n":         n,
        "EM":        round(sum(em_scores) / n * 100, 2),
        "F1":        round(sum(f1_scores) / n * 100, 2),
        "avg_turns": round(sum(turns) / n, 2),
    }

    print(f"\n=== IRCoT Eval ({n} examples) ===")
    print(f"  EM:        {results['EM']:.1f}%")
    print(f"  F1:        {results['F1']:.1f}%")
    print(f"  Avg turns: {results['avg_turns']:.2f}")

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
