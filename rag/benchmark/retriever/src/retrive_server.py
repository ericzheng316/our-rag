from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
import logging
import sys
import faiss
faiss.omp_set_num_threads(8)
from flashrag.retriever import DenseRetriever
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

app = FastAPI(title="Dense Retriever Service", version="1.0")
dense_retriever: DenseRetriever = None

class QueryRequest(BaseModel):
    querys: list

@app.post("/search")
def search(query_request: QueryRequest):
    if not query_request.querys:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        results, scores = dense_retriever.batch_search(query_request.querys, return_score=True)
        output = []
        for doc_list, score_list in zip(results, scores):
            output.append([
                {**doc, "score": float(score_list[j])}
                for j, doc in enumerate(doc_list)
            ])
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_args():
    parser = argparse.ArgumentParser(description='retriever server')
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--corpus_path', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    import threading, time
    args = get_args()

    _loading_done = threading.Event()
    def _heartbeat():
        start = time.time()
        while not _loading_done.wait(15):
            print(f"  ... still loading ({int(time.time()-start)}s elapsed)", flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()

    log.info("Loading Retriever...")
    retrieval_config = {
        'retrieval_method': 'e5',
        'retrieval_model_path': args.model_path,
        'retrieval_query_max_length': 256,
        'retrieval_use_fp16': True,
        'retrieval_topk': 100,  # Task1: top-100 for background Z-score calibration (top-k sliced in inference)
        'retrieval_batch_size': 32,
        'index_path': args.index_path,
        'corpus_path': args.corpus_path,
        'save_retrieval_cache': False,
        'use_retrieval_cache': False,
        'retrieval_cache_path': None,
        'use_reranker': False,
        'faiss_gpu': False,
        'use_sentence_transformer': False,
        'retrieval_pooling_method': 'mean',
        'instruction': None,
    }
    try:
        dense_retriever = DenseRetriever(retrieval_config)
    except Exception:
        log.exception("FAILED to load DenseRetriever")
        sys.exit(1)
    _loading_done.set()

    log.info("=" * 50)
    log.info("RETRIEVER READY — http://%s:%s", args.host, args.port)
    log.info("=" * 50)

    uvicorn.run(app, host=args.host, port=args.port)
