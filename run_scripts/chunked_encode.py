"""
Phase 1: Chunked corpus encoding → fp16 memmap on disk.

Encodes 17.3M passages with E5-base-v2 in chunks of CHUNK_SIZE,
writing results directly to a numpy memmap file.
Peak CPU RAM: ~3 GB (one chunk at a time).
GPU VRAM: ~500 MB (E5-base-v2 model).

Output: /home/boyuz5/data/indices/e5_full_emb/embeddings.bin  (fp16, shape: [N, 768])
        /home/boyuz5/data/indices/e5_full_emb/meta.json       (N, dim, dtype)
"""

import json
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
CORPUS_PATH  = "/home/boyuz5/data/flashrag_datasets/retrieval-corpus/wiki18_100w_clean.jsonl"
MODEL_PATH   = "/home/boyuz5/models/e5-base-v2"
SAVE_DIR     = "/home/boyuz5/data/indices/e5_full_emb"
CHUNK_SIZE   = 500_000   # passages per chunk  (~0.75 GB fp16 embeddings)
BATCH_SIZE   = 512       # encoder batch size
MAX_LEN      = 180
DIM          = 768
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────────────────


def mean_pool_normalize(model_out, attention_mask):
    token_embs = model_out.last_hidden_state          # (B, seq, dim)
    mask = attention_mask.unsqueeze(-1).float()       # (B, seq, 1)
    summed = (token_embs * mask).sum(dim=1)           # (B, dim)
    counts = mask.sum(dim=1).clamp(min=1e-9)          # (B, 1)
    mean   = summed / counts                           # (B, dim)
    return torch.nn.functional.normalize(mean, p=2, dim=1)  # unit vectors


def encode_batch(texts, tokenizer, model):
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    embs = mean_pool_normalize(out, enc["attention_mask"])
    return embs.cpu().float().numpy()


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    emb_path  = os.path.join(SAVE_DIR, "embeddings.bin")
    meta_path = os.path.join(SAVE_DIR, "meta.json")

    print(f"Counting corpus lines …")
    N = count_lines(CORPUS_PATH)
    print(f"Corpus: {N:,} passages")

    # Pre-allocate memmap
    emb_map = np.memmap(emb_path, dtype=np.float16, mode="w+", shape=(N, DIM))
    print(f"Memmap created: {emb_path}  ({N * DIM * 2 / 1e9:.1f} GB)")

    print(f"Loading E5-base-v2 from {MODEL_PATH} onto {DEVICE} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    t0 = time.time()
    written = 0

    with open(CORPUS_PATH, encoding="utf-8") as f:
        chunk_texts = []
        pbar = tqdm(total=N, desc="Encoding", unit="passages")

        for line in f:
            obj  = json.loads(line)
            text = obj.get("contents", "")
            chunk_texts.append(text)

            if len(chunk_texts) == CHUNK_SIZE:
                # Encode this chunk batch-by-batch
                chunk_embs = []
                for i in range(0, len(chunk_texts), BATCH_SIZE):
                    batch = chunk_texts[i : i + BATCH_SIZE]
                    chunk_embs.append(encode_batch(batch, tokenizer, model))
                chunk_arr = np.concatenate(chunk_embs, axis=0).astype(np.float16)

                end = written + len(chunk_arr)
                emb_map[written:end] = chunk_arr
                emb_map.flush()

                pbar.update(len(chunk_arr))
                written = end
                chunk_texts = []

        # Remainder
        if chunk_texts:
            chunk_embs = []
            for i in range(0, len(chunk_texts), BATCH_SIZE):
                batch = chunk_texts[i : i + BATCH_SIZE]
                chunk_embs.append(encode_batch(batch, tokenizer, model))
            chunk_arr = np.concatenate(chunk_embs, axis=0).astype(np.float16)
            end = written + len(chunk_arr)
            emb_map[written:end] = chunk_arr
            emb_map.flush()
            pbar.update(len(chunk_arr))
            written = end

        pbar.close()

    elapsed = time.time() - t0
    print(f"\nEncoding done: {written:,} passages in {elapsed/60:.1f} min  "
          f"({written/elapsed:.0f} passages/sec)")

    # Save metadata
    with open(meta_path, "w") as f:
        json.dump({"n": written, "dim": DIM, "dtype": "float16"}, f)
    print(f"Meta saved: {meta_path}")
    print(f"Embeddings: {emb_path}  ({os.path.getsize(emb_path)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
