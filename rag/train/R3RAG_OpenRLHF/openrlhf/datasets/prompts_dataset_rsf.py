"""
PromptDataset variant that passes supporting_facts alongside prompts.

Expected JSONL format (train_sf.jsonl):
  {
    "id":               "...",
    "question":         "Were Scott Derrickson and Ed Wood of the same nationality?",
    "golden_answers":   ["yes"],
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
    "type":             "comparison"
  }

Usage:
  from openrlhf.datasets.prompts_dataset_rsf import PromptDatasetRsf
  dataset = PromptDatasetRsf(raw_dataset, tokenizer, strategy)
"""

import json

from torch.utils.data import Dataset
from tqdm import tqdm


_QUESTION_TEMPLATE = "The question: {question}"


class PromptDatasetRsf(Dataset):
    """PPO/GRPO prompt dataset that carries SF titles for R_sf reward.

    SF titles are serialised as a JSON string to survive PyTorch's
    default_collate (which can't stack variable-length lists of strings).
    """

    def __init__(self, dataset, tokenizer, strategy) -> None:
        super().__init__()
        self.prompts: list = []

        for data in tqdm(dataset, desc="Loading prompts+SF", disable=not strategy.is_rank_0()):
            prompt = _QUESTION_TEMPLATE.format(question=data["question"])
            sf = data.get("supporting_facts") or {}
            sf_titles = sf.get("title", []) if isinstance(sf, dict) else []
            self.prompts.append({
                "prompt":         prompt,
                "golden_answers": data["golden_answers"],
                # JSON string → survives collate, no variable-length list issue
                "sf_titles_json": json.dumps(sf_titles),
            })

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
