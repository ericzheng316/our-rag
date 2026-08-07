#!/usr/bin/env bash
# Synthetic task construction (CPU). Smoke defaults; override via env.
#   MAX_LINES=  corpus prefix (empty = full corpus, hours-scale)
#   N_TASKS=    number of tasks
#   K=          comma list of K values
set -euo pipefail
source "$HOME/acec/env.sh"

# ${VAR-default} (no colon): an explicitly EMPTY MAX_LINES= means full corpus;
# the default applies only when the variable is entirely unset.
MAX_LINES="${MAX_LINES-400000}"
N_TASKS="${N_TASKS:-50}"
K="${K:-2,3,4}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/acec/synth/$(date +%Y%m%dT%H%M%S)}"

exec "$PYTHON" "$(dirname "$0")/60_build_synth_tasks.py" \
    ${MAX_LINES:+--max-lines "$MAX_LINES"} \
    --n-tasks "$N_TASKS" --k "$K" --out-dir "$OUT_DIR" "$@"
