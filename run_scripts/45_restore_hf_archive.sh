#!/usr/bin/env bash
# Restore the ACEC project and experiment record from its Hugging Face archive.

set -euo pipefail

HF_REPO_ID="${1:-${HF_REPO_ID:-}}"
MODE="${2:---experiments-only}"
DATA_ROOT="${DATA_ROOT:-/root/data}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$DATA_ROOT/acec_archive_download}"
PROJECT_ROOT="${PROJECT_ROOT:-$DATA_ROOT/our-rag}"
HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"

if [[ -z "$HF_REPO_ID" || "$HF_REPO_ID" != */* ]]; then
    echo "usage: bash $0 OWNER/REPO [--experiments-only|--full]" >&2
    exit 2
fi
if [[ "$MODE" != "--experiments-only" && "$MODE" != "--full" ]]; then
    echo "[restore] unsupported mode: $MODE" >&2
    exit 2
fi
for tool in hf git rsync sha256sum; do
    command -v "$tool" >/dev/null || {
        echo "[restore] required tool is unavailable: $tool" >&2
        exit 2
    }
done

mkdir -p "$DATA_ROOT" "$HF_HOME"
echo "[restore] downloading private experiment archive"
HF_HOME="$HF_HOME" hf download \
    "$HF_REPO_ID" \
    --repo-type dataset \
    --local-dir "$ARCHIVE_ROOT"

echo "[restore] verifying SHA-256 checksums"
(
    cd "$ARCHIVE_ROOT"
    sha256sum -c manifest/files.sha256
)

if [[ -e "$PROJECT_ROOT/.git/index.lock" ]]; then
    echo "[restore] refusing to modify project while .git/index.lock exists" >&2
    exit 2
fi
if [[ -d "$PROJECT_ROOT/.git" ]] && [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
    echo "[restore] refusing to overwrite a dirty existing project: $PROJECT_ROOT" >&2
    exit 2
fi
if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
    git clone "$ARCHIVE_ROOT/project/our-rag.bundle" "$PROJECT_ROOT"
fi
archived_head="$(cat "$ARCHIVE_ROOT/project/git_head.txt")"
git -C "$PROJECT_ROOT" checkout --detach "$archived_head"
if [[ -s "$ARCHIVE_ROOT/project/index.patch" ]]; then
    git -C "$PROJECT_ROOT" apply --index "$ARCHIVE_ROOT/project/index.patch"
fi
if [[ -s "$ARCHIVE_ROOT/project/working_tree.patch" ]]; then
    git -C "$PROJECT_ROOT" apply "$ARCHIVE_ROOT/project/working_tree.patch"
fi
if [[ -s "$ARCHIVE_ROOT/project/current_local_worktree.tar.gz" ]]; then
    tar -xzf "$ARCHIVE_ROOT/project/current_local_worktree.tar.gz" -C "$PROJECT_ROOT"
fi

mkdir -p \
    "$DATA_ROOT/logs" \
    "$DATA_ROOT/setup_logs" \
    "$DATA_ROOT/flashrag_datasets/hotpotqa" \
    "$DATA_ROOT/datasets/hotpotqa/distractor_jsonl"
rsync -a "$ARCHIVE_ROOT/experiments/logs/" "$DATA_ROOT/logs/"
rsync -a "$ARCHIVE_ROOT/experiments/setup_logs/" "$DATA_ROOT/setup_logs/"
rsync -a \
    "$ARCHIVE_ROOT/datasets/hotpotqa/flashrag/" \
    "$DATA_ROOT/flashrag_datasets/hotpotqa/"
rsync -a \
    "$ARCHIVE_ROOT/datasets/hotpotqa/distractor_jsonl/" \
    "$DATA_ROOT/datasets/hotpotqa/distractor_jsonl/"
rsync -a "$ARCHIVE_ROOT/project/historical_bundles/" "$DATA_ROOT/"

if [[ "$MODE" == "--experiments-only" ]]; then
    echo "[restore] COMPLETE (experiments only): $DATA_ROOT"
    exit 0
fi

command -v wget >/dev/null || {
    echo "[restore] wget is required for --full" >&2
    exit 2
}
command -v unzip >/dev/null || {
    echo "[restore] unzip is required for --full" >&2
    exit 2
}

echo "[restore] downloading public model dependencies"
mkdir -p "$DATA_ROOT/models"
HF_HOME="$HF_HOME" hf download \
    Yuan-Li-FNLP/R3-RAG-Qwen \
    --local-dir "$DATA_ROOT/models/R3-RAG-Qwen"
HF_HOME="$HF_HOME" hf download \
    intfloat/e5-base-v2 \
    --local-dir "$DATA_ROOT/models/e5-base-v2"
HF_HOME="$HF_HOME" hf download cross-encoder/nli-deberta-v3-base

echo "[restore] downloading public retrieval corpus"
mkdir -p "$DATA_ROOT/flashrag_datasets/retrieval-corpus"
wget -c \
    -O "$DATA_ROOT/flashrag_datasets/retrieval-corpus/wiki18_100w.jsonl" \
    "https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w.jsonl"

echo "[restore] downloading and normalizing public E5 FAISS index"
index_zip="$DATA_ROOT/wiki18_100w_e5_index.zip"
index_extract="$DATA_ROOT/wiki18_100w_e5_index_extract"
index_target="$DATA_ROOT/indices/e5_Flat/e5_Flat.index"
mkdir -p "$index_extract" "$(dirname "$index_target")"
wget -c \
    -O "$index_zip" \
    "https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/resolve/master/retrieval_corpus/wiki18_100w_e5_index.zip"
unzip -o "$index_zip" -d "$index_extract"
extracted_index="$(find "$index_extract" -type f -name '*.index' -print -quit)"
if [[ -z "$extracted_index" ]]; then
    echo "[restore] downloaded archive did not contain a FAISS .index file" >&2
    exit 4
fi
mv "$extracted_index" "$index_target"
actual_size="$(stat -c '%s' "$index_target")"
if [[ "$actual_size" != "64559075373" ]]; then
    echo "[restore] unexpected E5 index size: $actual_size" >&2
    exit 4
fi

echo "[restore] COMPLETE (full data/assets): $DATA_ROOT"
