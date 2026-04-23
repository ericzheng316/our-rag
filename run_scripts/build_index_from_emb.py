"""
Phase 2: Build FAISS SQ8 index from fp16 embedding memmap.

Reads embeddings in chunks → trains SQ8 quantizer → adds all vectors.
Saves a 13.3 GB SQ8 index file compatible with the existing retriever server.

Peak CPU RAM: ~15 GB (SQ8 index grows to 13.3 GB + 1.5 GB chunk).
No GPU needed for this phase.

Input:  /home/boyuz5/data/indices/e5_full_emb/embeddings.bin
Output: /home/boyuz5/data/indices/e5_Flat/e5_SQ8.index
"""

import json
import os
import time

import faiss
import numpy as np
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
EMB_DIR      = "/home/boyuz5/data/indices/e5_full_emb"
INDEX_PATH   = "/home/boyuz5/data/indices/e5_Flat/e5_SQ8.index"
CHUNK_SIZE   = 500_000   # passages per chunk during add
TRAIN_SIZE   = 1_000_000  # passages used for SQ8 training
DIM          = 768
# ─────────────────────────────────────────────────────────────────────────────


def load_meta(emb_dir):
    with open(os.path.join(emb_dir, "meta.json")) as f:
        return json.load(f)


def main():
    meta     = load_meta(EMB_DIR)
    N        = meta["n"]
    emb_path = os.path.join(EMB_DIR, "embeddings.bin")

    print(f"Loading memmap: {N:,} passages × {DIM} dims (fp16) …")
    emb_map = np.memmap(emb_path, dtype=np.float16, mode="r", shape=(N, DIM))

    # ── Train SQ8 on first TRAIN_SIZE vectors ────────────────────────────────
    print(f"Training SQ8 quantizer on {TRAIN_SIZE:,} vectors …")
    train_data = np.array(emb_map[:TRAIN_SIZE], dtype=np.float32)

    index = faiss.IndexScalarQuantizer(DIM, faiss.ScalarQuantizer.QT_8bit,
                                       faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    index.train(train_data)
    del train_data
    print(f"Training done in {time.time()-t0:.1f}s")

    # ── Add all vectors in chunks ─────────────────────────────────────────────
    print(f"Adding {N:,} vectors in chunks of {CHUNK_SIZE:,} …")
    t0 = time.time()
    for start in tqdm(range(0, N, CHUNK_SIZE), desc="Adding chunks"):
        end   = min(start + CHUNK_SIZE, N)
        chunk = np.array(emb_map[start:end], dtype=np.float32)
        index.add(chunk)
        del chunk

    elapsed = time.time() - t0
    print(f"Add done in {elapsed/60:.1f} min  |  index.ntotal = {index.ntotal:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    print(f"Writing index → {INDEX_PATH} …")
    faiss.write_index(index, INDEX_PATH)
    size_gb = os.path.getsize(INDEX_PATH) / 1e9
    print(f"Done. Index size: {size_gb:.1f} GB")


if __name__ == "__main__":
    main()
