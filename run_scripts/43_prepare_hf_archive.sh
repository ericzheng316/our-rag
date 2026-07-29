#!/usr/bin/env bash
# Prepare a browseable, checksum-verified Hugging Face dataset folder.
# Large experiment files are hard-linked, so staging consumes almost no
# additional persistent-disk blocks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/data}"
PROJECT_ROOT="${PROJECT_ROOT:-$DATA_ROOT/our-rag}"
STAGING_ROOT="${STAGING_ROOT:-$DATA_ROOT/acec_hf_archive}"
PYTHON="${PYTHON:-$PROJECT_ROOT/rag/.venv/bin/python}"

if [[ -e "$PROJECT_ROOT/.git/index.lock" ]]; then
    echo "[archive] refusing to run while $PROJECT_ROOT/.git/index.lock exists" >&2
    exit 2
fi
if [[ ! -d "$PROJECT_ROOT/.git" || ! -d "$DATA_ROOT/logs" ]]; then
    echo "[archive] missing project Git repository or experiment logs under $DATA_ROOT" >&2
    exit 2
fi
if [[ -e "$STAGING_ROOT" ]]; then
    echo "[archive] refusing to overwrite existing staging directory: $STAGING_ROOT" >&2
    exit 2
fi

required_tools=(cp find git grep sha256sum sort)
for tool in "${required_tools[@]}"; do
    command -v "$tool" >/dev/null || {
        echo "[archive] required tool is unavailable: $tool" >&2
        exit 2
    }
done

scan_file="$(mktemp)"
trap 'rm -f "$scan_file"' EXIT
find "$DATA_ROOT/logs" "$DATA_ROOT/setup_logs" \
    -type f \
    \( -name '*.log' -o -name '*.json' -o -name '*.jsonl' -o \
       -name '*.md' -o -name '*.sh' -o -name '*.status' -o \
       -name '*.csv' -o -name '*.tsv' -o -name '*.jinja' \) \
    -print0 > "$scan_file"

secret_hits="$(mktemp)"
trap 'rm -f "$scan_file" "$secret_hits"' EXIT
if xargs -0 grep -IlE \
    'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|QC-[A-Za-z0-9_-]{20,}' \
    < "$scan_file" > "$secret_hits" 2>/dev/null; then
    echo "[archive] possible credential material found; upload is blocked:" >&2
    sed -n '1,100p' "$secret_hits" >&2
    exit 3
fi

mkdir -p \
    "$STAGING_ROOT/experiments" \
    "$STAGING_ROOT/datasets/hotpotqa" \
    "$STAGING_ROOT/project/historical_bundles" \
    "$STAGING_ROOT/environment" \
    "$STAGING_ROOT/system" \
    "$STAGING_ROOT/setup" \
    "$STAGING_ROOT/manifest" \
    "$STAGING_ROOT/run_scripts"

echo "[archive] hard-linking complete experiment records"
cp -a -l "$DATA_ROOT/logs" "$STAGING_ROOT/experiments/logs"
cp -a -l "$DATA_ROOT/setup_logs" "$STAGING_ROOT/experiments/setup_logs"

if [[ -d "$PROJECT_ROOT/experiments" ]]; then
    cp -a -l "$PROJECT_ROOT/experiments" "$STAGING_ROOT/experiments/repo_summaries"
fi
if [[ -d "$DATA_ROOT/flashrag_datasets/hotpotqa" ]]; then
    cp -a -l \
        "$DATA_ROOT/flashrag_datasets/hotpotqa" \
        "$STAGING_ROOT/datasets/hotpotqa/flashrag"
fi
if [[ -d "$DATA_ROOT/datasets/hotpotqa/distractor_jsonl" ]]; then
    cp -a -l \
        "$DATA_ROOT/datasets/hotpotqa/distractor_jsonl" \
        "$STAGING_ROOT/datasets/hotpotqa/distractor_jsonl"
fi

shopt -s nullglob
historical_bundles=("$DATA_ROOT"/acec-*.bundle)
if (( ${#historical_bundles[@]} )); then
    cp -a -l "${historical_bundles[@]}" "$STAGING_ROOT/project/historical_bundles/"
fi
shopt -u nullglob

for source_path in \
    "$DATA_ROOT/bootstrap.sh" \
    "$DATA_ROOT/_flashrag_dl/run_downloads.sh" \
    "$DATA_ROOT/src/faiss-v1.14.3.tar.gz"; do
    if [[ -f "$source_path" ]]; then
        cp -a -l "$source_path" "$STAGING_ROOT/setup/"
    fi
done

cp -a "$REPO_ROOT/archive/acec_hf_archive.yaml" "$STAGING_ROOT/"
cp -a "$REPO_ROOT/archive/HF_DATASET_CARD.md" "$STAGING_ROOT/README.md"
cp -a "$REPO_ROOT/run_scripts/45_restore_hf_archive.sh" "$STAGING_ROOT/run_scripts/"

echo "[archive] recording source state"
git -C "$PROJECT_ROOT" bundle create "$STAGING_ROOT/project/our-rag.bundle" --all
git -C "$PROJECT_ROOT" rev-parse HEAD > "$STAGING_ROOT/project/git_head.txt"
git -C "$PROJECT_ROOT" branch --show-current > "$STAGING_ROOT/project/git_branch.txt"
git -C "$PROJECT_ROOT" remote -v > "$STAGING_ROOT/project/git_remotes.txt"
git -C "$PROJECT_ROOT" status --short --branch > "$STAGING_ROOT/project/git_status.txt"
git -C "$PROJECT_ROOT" diff --binary > "$STAGING_ROOT/project/working_tree.patch"
git -C "$PROJECT_ROOT" diff --binary --cached > "$STAGING_ROOT/project/index.patch"

echo "[archive] recording software environment"
if [[ -x "$PYTHON" ]]; then
    "$PYTHON" --version > "$STAGING_ROOT/environment/python_version.txt" 2>&1
    "$PYTHON" -m pip freeze --all > "$STAGING_ROOT/environment/pip_freeze.txt"
    "$PYTHON" -m pip list --format=json > "$STAGING_ROOT/environment/pip_list.json"
fi
if command -v conda >/dev/null; then
    conda env export > "$STAGING_ROOT/environment/conda_environment.yaml"
    conda list --explicit > "$STAGING_ROOT/environment/conda_explicit.txt"
fi

{
    uname -a
    printf '\n'
    cat /etc/os-release 2>/dev/null || true
} > "$STAGING_ROOT/system/os.txt"
lscpu > "$STAGING_ROOT/system/lscpu.txt" 2>&1 || true
free -h > "$STAGING_ROOT/system/memory.txt" 2>&1 || true
df -h "$DATA_ROOT" > "$STAGING_ROOT/system/storage.txt" 2>&1 || true
if command -v nvidia-smi >/dev/null; then
    nvidia-smi -q > "$STAGING_ROOT/system/nvidia_smi.txt" 2>&1 || true
fi
{
    printf 'HF_HOME=%s\n' "${HF_HOME:-}"
    printf 'HF_ENDPOINT=%s\n' "${HF_ENDPOINT:-}"
    printf 'TRANSFORMERS_CACHE=%s\n' "${TRANSFORMERS_CACHE:-}"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
} > "$STAGING_ROOT/environment/nonsecret_runtime_variables.txt"

file_count="$(find "$STAGING_ROOT" -type f | wc -l)"
byte_count="$(du -sb "$STAGING_ROOT" | awk '{print $1}')"
{
    printf 'archive_id=acec-experiments-2026-07\n'
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'payload_file_count_before_manifests=%s\n' "$file_count"
    printf 'payload_logical_bytes_before_manifests=%s\n' "$byte_count"
    printf 'staging_root=%s\n' "$STAGING_ROOT"
} > "$STAGING_ROOT/manifest/archive_summary.txt"

echo "[archive] generating inventory and SHA-256 manifest"
inventory_tmp="$(mktemp)"
checksum_tmp="$(mktemp)"
trap 'rm -f "$scan_file" "$secret_hits" "$inventory_tmp" "$checksum_tmp"' EXIT
(
    cd "$STAGING_ROOT"
    find . -type f \
        ! -path './manifest/files.tsv' \
        ! -path './manifest/files.sha256' \
        -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\t%p\n' \
        | LC_ALL=C sort > "$inventory_tmp"
    mv "$inventory_tmp" manifest/files.tsv
    find . -type f ! -path './manifest/files.sha256' -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum > "$checksum_tmp"
    mv "$checksum_tmp" manifest/files.sha256
)

echo "[archive] ready: $STAGING_ROOT"
cat "$STAGING_ROOT/manifest/archive_summary.txt"
