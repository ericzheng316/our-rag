"""
Merge flashrag dev.jsonl + HF distractor dev.jsonl into a single file
with distractor_paras field, suitable for distractor-mode inference.

Output: $HOME/data/flashrag_datasets/hotpotqa/dev_distractor.jsonl
Fields: id, question, golden_answers, metadata, distractor_paras,
        distractor_titles, supporting_facts

`distractor_titles[j]` is the title of `distractor_paras[j]` (parallel arrays),
and `supporting_facts` is HotpotQA's raw {'title': [...], 'sent_id': [...]}
dict — both are needed by rag/src/belief/acec/offline_fit.py to derive gold
hit labels for ACEC calibration (they were previously dropped, which is why
inference_new.py's records.jsonl could never support that calibration step).
"""

import json
import os

_HOME = os.path.expanduser("~")
FLASHRAG_PATH = os.path.join(_HOME, "data/flashrag_datasets/hotpotqa/dev.jsonl")
HF_PATH = os.path.join(_HOME, "data/datasets/hotpotqa/distractor_jsonl/dev.jsonl")
OUTPUT_PATH = os.path.join(_HOME, "data/flashrag_datasets/hotpotqa/dev_distractor.jsonl")


def hf_context_to_paras(context: dict) -> tuple[list[str], list[str]]:
    """Convert HF context dict → (paras, titles), both 'Title: sentence...'
    strings and the bare titles, as parallel lists."""
    titles = context["title"]
    sentences_list = context["sentences"]
    paras = [
        f"{title}: {' '.join(sents)}"
        for title, sents in zip(titles, sentences_list)
    ]
    return paras, list(titles)


def main():
    # Load HF data keyed by question text
    hf_by_question: dict[str, dict] = {}
    with open(HF_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            paras, titles = hf_context_to_paras(d["context"])
            hf_by_question[d["question"]] = {
                "distractor_paras": paras,
                "distractor_titles": titles,
                "supporting_facts": d.get("supporting_facts"),
            }

    print(f"Loaded {len(hf_by_question)} questions from HF distractor file")

    matched = 0
    total = 0
    with open(FLASHRAG_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            total += 1
            hf_entry = hf_by_question.get(d["question"])
            if hf_entry:
                matched += 1
                d.update(hf_entry)
            else:
                d["distractor_paras"] = []
                d["distractor_titles"] = []
                d["supporting_facts"] = None
            json.dump(d, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Written {total} records to {OUTPUT_PATH}")
    print(f"Matched distractor paras + supporting_facts: {matched}/{total}")
    if total - matched > 0:
        print(f"WARNING: {total - matched} records had no matching distractor paras")


if __name__ == "__main__":
    main()
