#!/usr/bin/env bash
# ACEC v7 G3 ablation: retrain the selector without conditional delta-C.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${INPUT:?set INPUT to counterfactual_scored.jsonl}"
OUTPUT="${OUTPUT:-$(dirname "$INPUT")/set_selector_v7_no_delta_c.json}"

DISABLE_FEATURES="conditional_coverage_gain" \
OUTPUT="$OUTPUT" \
    "$SCRIPT_DIR/53_train_v7_selector.sh"
