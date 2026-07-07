# Legacy code

Archived, not part of the active pipeline. Kept for reference only — nothing
under `run_scripts/` or `rag/benchmark/` calls into this directory.

- `startup/` — an earlier Gradio-based interactive demo (`R3-RAG.py`,
  `RRAG.py`) plus a placeholder retriever microservice, predating the
  batch command-line pipeline now used everywhere else
  (`rag/benchmark/R3-RAG/src/inference_new.py` for inference,
  `rag/benchmark/retriever/src/retrive_server.py` for retrieval). `RRAG.py`
  is an earlier, more minimal draft of `R3-RAG.py`. If you need an
  interactive demo again, start here rather than from scratch — but verify
  it still matches the current prompt format in `inference_new.py` first,
  since the two have diverged.
