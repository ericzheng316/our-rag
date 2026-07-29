#!/usr/bin/env bash
# Upload the prepared archive as a private Hugging Face dataset repository.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/data}"
STAGING_ROOT="${STAGING_ROOT:-$DATA_ROOT/acec_hf_archive}"
HF_REPO_ID="${HF_REPO_ID:-${1:-}}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ -z "$HF_REPO_ID" || "$HF_REPO_ID" != */* ]]; then
    echo "usage: HF_REPO_ID=OWNER/REPO bash $0" >&2
    exit 2
fi
if [[ ! -s "$STAGING_ROOT/manifest/files.sha256" ]]; then
    echo "[upload] archive has not been prepared: $STAGING_ROOT" >&2
    exit 2
fi
if ! hf auth whoami >/dev/null 2>&1; then
    echo "[upload] Hugging Face login required; run: hf auth login" >&2
    exit 3
fi

echo "[upload] repo=$HF_REPO_ID source=$STAGING_ROOT workers=$NUM_WORKERS"
echo "[upload] visibility=private repo_type=dataset"
hf upload-large-folder \
    "$HF_REPO_ID" \
    "$STAGING_ROOT" \
    --repo-type dataset \
    --private \
    --num-workers "$NUM_WORKERS"

echo "[upload] verifying remote repository identity"
python - "$HF_REPO_ID" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
info = HfApi().repo_info(repo_id=repo_id, repo_type="dataset")
files = {item.rfilename for item in info.siblings}
required = {"README.md", "manifest/files.sha256", "acec_hf_archive.yaml"}
missing = required - files
if missing:
    raise SystemExit(f"remote archive is missing required files: {sorted(missing)}")
print(f"[upload] remote files={len(files)} commit={info.sha}")
PY
echo "[upload] COMPLETE: https://huggingface.co/datasets/$HF_REPO_ID"
