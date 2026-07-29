---
pretty_name: ACEC Complete Experiment Archive
language:
  - en
task_categories:
  - question-answering
tags:
  - rag
  - grpo
  - hotpotqa
  - acec
---

# ACEC Complete Experiment Archive

This private dataset repository is the complete, auditable record of the ACEC
experiments. It intentionally keeps successful runs, failed runs, smoke tests,
calibration artifacts, raw and processed evaluations, launch logs, trajectories,
and every locally retained LoRA checkpoint.

## Contents

- `experiments/logs/`: every file from the persistent experiment log tree,
  including intermediate checkpoints and `_lora_scratch`.
- `experiments/setup_logs/`: machine setup, dependency, download, service, and
  diagnostic logs.
- `experiments/repo_summaries/`: human-readable experiment records tracked in
  the source repository.
- `datasets/hotpotqa/`: exact prepared HotpotQA inputs used by these runs.
- `project/`: a self-contained Git bundle, historical bundles, Git state, and
  an exact tracked/untracked non-ignored source overlay for the uncommitted
  v6.3/v6.4 implementation present when the archive was finalized.
- `environment/` and `system/`: package locks and machine/runtime information.
- `manifest/`: per-file sizes, timestamps, and SHA-256 checksums.

## Deliberate exclusions

Public upstream assets are not duplicated: R3-RAG-Qwen, E5, NLI, the
`wiki18_100w` corpus, and its 64.6 GB E5 FAISS index. Their immutable
identifiers or download URLs are recorded in `acec_hf_archive.yaml`, and the
restore script can download them.

Package caches, the virtual environment, and the FAISS build directory are also
excluded because they are machine-specific. The package inventory, FAISS source
archive, and build logs are retained.

## Restore

Download the repository and run:

```bash
bash run_scripts/45_restore_hf_archive.sh OWNER/REPO --experiments-only
```

Use `--full` to additionally restore public models, retrieval corpus, and index.
The restore process verifies `manifest/files.sha256` before installing files.

No API keys, access tokens, or passwords are intentionally stored here.
