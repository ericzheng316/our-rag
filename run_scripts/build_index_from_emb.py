"""
Phase 2: Build FAISS Flat (exact) index from fp16 embedding memmap.

Reads embeddings in chunks → adds all vectors to IndexFlatIP.
Saves a ~53 GB Flat index compatible with the retriever server (02_start_retriever.sh).

Peak CPU RAM: ~53 GB (Flat index in memory) + 1.5 GB per chunk.
Requires ≥80 GB RAM.  Run on dedicated A100 node, NOT on shared ICRN H200.
No GPU needed for this phase (index building is CPU-only).

Input:  /home/boyuz5/data/indices/e5_full_emb/embeddings.bin
Output: /home/boyuz5/data/indices/e5_Flat/e5_Flat.index
"""

import json
import os
import time

import faiss
import numpy as np
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
EMB_DIR      = "/home/boyuz5/data/indices/e5_full_emb"
INDEX_PATH   = "/home/boyuz5/data/indices/e5_Flat/e5_Flat.index"
CHUNK_SIZE   = 500_000   # passages per chunk during add
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

    # ── Build Flat (exact inner-product) index ────────────────────────────────
    print(f"Creating IndexFlatIP (exact, dim={DIM}) …")
    index = faiss.IndexFlatIP(DIM)

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
