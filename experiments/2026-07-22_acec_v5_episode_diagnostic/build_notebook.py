#!/usr/bin/env python3
"""Build the reader-facing ACEC checkpoint diagnostic notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def build_notebook(output: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ACEC diagnostics)",
            "language": "python",
            "name": "acec-diagnostics",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# ACEC-v5 episode 50/75/100 diagnostic

## tl;dr

- **Episode 50 is the current stopping point.** From episode 50 to 100,
  held-out processed EM fell 3.13 points, processed F1 fell 3.27 points,
  Gold supporting-fact recall fell 5.08 points, and retrieval calls fell from
  1.461 to 1.297 per question.
- **The answer regression is mostly semantic replacement, not formatting.**
  Step 100 lost 17 exact answers that step 50 got right and gained only nine;
  11 of the 17 losses had zero F1 after the change. Mean answer length rose
  slightly rather than collapsing.
- **Retrieval compression is real but does not fully explain answer loss.**
  Fourteen of sixteen held-out batches used fewer retrievals at step 100 and
  eleven also lost Gold-SF recall, but batch-level SF changes did not track
  processed-F1 changes closely.
- **The training proxies diverged from fixed held-out behavior.** The final
  10 training episodes had higher mean reward, online EM, and online SF recall
  than the window ending at step 50, but those windows contain different
  sequential training questions and are not a fixed validation set.
"""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

This diagnostic aligns the same 256 HotpotQA dev questions across ACEC-v5
steps 50, 75, and 100. Answer metrics are available per question. Retrieval
calls and Gold supporting-fact recall were logged only as 16-question batch
aggregates, so retrieval relationships are descriptive at batch grain and
must not be interpreted as per-question causality.

### Key assumptions

- The fixed held-out manifest and all three processed-prediction files have
  one unique row per question and matching ids.
- R3 processed EM/F1 includes the original answer plus extracted candidate
  answers, following the saved evaluator implementation.
- Question-type labels are deterministic lexical heuristics for diagnostic
  slicing, not authoritative HotpotQA categories.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd

ROOT = Path.cwd()
summary = json.loads((ROOT / "analysis_results.json").read_text())
per_question = pd.read_csv(ROOT / "per_question_diagnostic.csv")
drivers = pd.read_csv(ROOT / "question_type_drivers.csv")
batches = pd.read_csv(ROOT / "batch_diagnostic.csv")
training = pd.read_csv(ROOT / "training_episode_metrics.csv")
transitions = pd.read_csv(ROOT / "transition_examples.csv")

assert len(per_question) == 256
assert per_question["id"].nunique() == 256
assert len(training) == 100
print("validated: 256 unique held-out questions, 100 training episodes")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Results

### Step 50 leads the saved checkpoints on fixed held-out quality

Both direct and processed metrics peak at step 50. The drop is not a product
of one metric: direct F1, processed F1, substring EM, and Gold-SF recall all
move downward by step 100 while retrieval calls continue to fall.
"""
        ),
        nbf.v4.new_code_cell(
            """overall = pd.DataFrame(summary["overall"]).T[
    ["direct_em", "direct_f1", "processed_em", "processed_f1",
     "substring_em", "gold_sf_recall", "retrieval_calls", "mean_prediction_words"]
]
overall.round(4)"""
        ),
        nbf.v4.new_markdown_cell(
            """### Later checkpoints replace correct entities with wrong ones

Step 75 loses 20 processed-exact answers from step 50 and gains 10. Step 100
loses 17 and gains nine. The paired bootstrap intervals cross zero on this
256-question subset, so the ranking is not statistically resolved, but the
point estimates and multiple answer metrics consistently favor step 50.
"""
        ),
        nbf.v4.new_code_cell(
            """comparison_rows = []
for name, values in summary["comparisons"].items():
    comparison_rows.append({"comparison": name, **values})
pd.DataFrame(comparison_rows)[
    ["comparison", "processed_em_delta", "processed_em_ci95",
     "processed_f1_delta", "processed_f1_ci95", "wins", "losses",
     "prediction_changed", "lost_exact_with_zero_after_f1"]
]"""
        ),
        nbf.v4.new_markdown_cell(
            """### Most questions are checkpoint-stable; a small unstable set drives the ranking

Across all three checkpoints, 73 questions are always processed-exact and 142
are always wrong. Only 41 questions change exactness at least once. Step 50
already captures 101 of the 114 questions that any of these checkpoints can
answer exactly.
"""
        ),
        nbf.v4.new_code_cell(
            """pd.DataFrame(
    [{"pattern (50/75/100)": key, "questions": value}
     for key, value in summary["stability"]["correctness_pattern_counts"].items()]
).sort_values("pattern (50/75/100)")"""
        ),
        nbf.v4.new_markdown_cell(
            """### Who and what questions explain more than the net step-100 F1 loss

`who` and `what` slices contribute approximately -3.57 points to the overall
step100-minus-step50 F1 movement, more than the observed -3.27 points because
`which` and numeric questions partially offset the decline. The answer-type
slice is heuristic, and small slices such as numeric and where should not be
treated as stable estimates.
"""
        ),
        nbf.v4.new_code_cell(
            """driver_view = drivers[
    ["question_type", "n", "step50_processed_f1", "step100_processed_f1",
     "step100_vs_step50_f1_delta", "step100_vs_step50_f1_contribution"]
].copy()
driver_view.sort_values("step100_vs_step50_f1_contribution").round(4)"""
        ),
        nbf.v4.new_markdown_cell(
            """### Retrieval compression accompanies lower evidence coverage

From step 50 to 100, retrieval calls decline in 14 of 16 held-out batches;
11 of those batches also lose Gold-SF recall. The batch-level correlation
between retrieval-call change and SF-recall change is 0.58. However, SF-recall
change has near-zero correlation with processed-F1 change at this coarse
grain, so evidence loss is a likely contributor but not a complete answer.
"""
        ),
        nbf.v4.new_code_cell(
            """pd.DataFrame(summary["batch_diagnostics"]).T.round(4)"""
        ),
        nbf.v4.new_markdown_cell(
            """### Online training proxies do not validate the fixed held-out trend

The ten-episode window ending at step 100 has higher online reward, online EM,
and online SF recall than the window ending at step 50, while the fixed
held-out evaluation worsens. Because each training window contains different
sequential questions, this is evidence of proxy/generalization mismatch but
not by itself proof of causal reward hacking.
"""
        ),
        nbf.v4.new_code_cell(
            """pd.DataFrame(summary["training_last10_windows"]).T.round(4)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Limitations, uncertainty, and robustness checks

- The fixed held-out set has 256 questions. Step 50's advantage over steps 75
  and 100 is directionally consistent but its paired 95% intervals include
  zero.
- Retrieval diagnostics are limited to 16 batches, not per-question traces.
  The original evaluator discarded individual trajectories and retrieved
  document ids.
- The API extractor was called separately for identical question/prediction
  pairs. Of 914 calls, 282 could have been reused; two duplicate groups
  disagreed on processed EM and five on F1. Caching identical inputs would
  remove this avoidable measurement noise and reduce cost. The noise audit
  does not overturn the step-50 conclusion.
"""
        ),
        nbf.v4.new_code_cell(
            """pd.Series(summary["extractor_reuse_audit"], name="value").to_frame()"""
        ),
        nbf.v4.new_markdown_cell(
            """## Recommended next steps

1. Treat step 50 as the current checkpoint and do not extend v5 unchanged to
   500/800 episodes.
2. Save per-question retrieval traces in the next evaluator: queries, actions,
   document ids/titles, Gold-SF hits, and stopping turn.
3. Cache processed-answer extraction by `(question, prediction)` across model
   variants.
4. Build v6 around two protections: calibrate/guard the early-answer decision
   against Gold-SF loss, and anchor the answer generator with stronger KL,
   answer-token masking, or a small supervised rehearsal set.
5. Gate the next run at 25-episode intervals on fixed held-out processed EM/F1,
   semantic ACC, Gold-SF recall, and retrieval calls. Scale toward 500 only if
   quality does not deteriorate while efficiency improves.

## Further questions

- Do the 17 step50-to-step100 exact losses correspond to premature stopping,
  wrong retrieved evidence, or failure to use adequate evidence?
- Can an answer-head anchor preserve entity selection while the action policy
  continues learning lower-cost retrieval?
- Does the step-50 advantage replicate on a larger or second fixed dev sample?
"""
        ),
    ]
    nbf.write(notebook, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_notebook(args.output)


if __name__ == "__main__":
    main()
