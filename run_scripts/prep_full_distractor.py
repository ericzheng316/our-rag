"""
Merge flashrag dev.jsonl + HF distractor dev.jsonl into a single file
with distractor_paras field, suitable for distractor-mode inference.

Output: /home/boyuz5/data/flashrag_datasets/hotpotqa/dev_distractor.jsonl
Fields: id, question, golden_answers, metadata, distractor_paras
"""

import json

FLASHRAG_PATH = "/home/boyuz5/data/flashrag_datasets/hotpotqa/dev.jsonl"
HF_PATH = "/home/boyuz5/data/datasets/hotpotqa/distractor_jsonl/dev.jsonl"
OUTPUT_PATH = "/home/boyuz5/data/flashrag_datasets/hotpotqa/dev_distractor.jsonl"


def hf_context_to_paras(context: dict) -> list[str]:
    """Convert HF context dict → list of 'Title: sentence1 sentence2 ...' strings."""
    titles = context["title"]
    sentences_list = context["sentences"]
    return [
        f"{title}: {' '.join(sents)}"
        for title, sents in zip(titles, sentences_list)
    ]


def main():
    # Load HF data keyed by question text
    hf_by_question: dict[str, list[str]] = {}
    with open(HF_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            paras = hf_context_to_paras(d["context"])
            hf_by_question[d["question"]] = paras

    print(f"Loaded {len(hf_by_question)} questions from HF distractor file")

    matched = 0
    total = 0
    with open(FLASHRAG_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            total += 1
            paras = hf_by_question.get(d["question"], [])
            if paras:
                matched += 1
            d["distractor_paras"] = paras
            json.dump(d, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Written {total} records to {OUTPUT_PATH}")
    print(f"Matched distractor paras: {matched}/{total}")
    if total - matched > 0:
        print(f"WARNING: {total - matched} records had no matching distractor paras")


if __name__ == "__main__":
    main()
