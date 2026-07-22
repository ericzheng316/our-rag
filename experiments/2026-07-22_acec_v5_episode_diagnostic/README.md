# ACEC-v5 Episode 50/75/100 diagnostic

## Decision

Do not extend the unchanged ACEC-v5 recipe to 500 or 800 episodes. On the same 256-question HotpotQA held-out sample, Episode 50 is the best of the three saved checkpoints. Episode 100 is more retrieval-efficient, but it loses answer quality and supporting-fact coverage.

| Checkpoint | Processed EM | Processed F1 | Direct F1 | Gold-SF recall | Retrievals/question |
| --- | ---: | ---: | ---: | ---: | ---: |
| Episode 50 | 39.45% | 50.08% | 47.73% | 73.24% | 1.461 |
| Episode 75 | 35.55% | 47.08% | 45.62% | 70.12% | 1.418 |
| Episode 100 | 36.33% | 46.81% | 44.98% | 68.16% | 1.297 |

Episode 100 versus 50 changes processed EM by -3.13 percentage points and processed F1 by -3.27 points. Their paired 95% bootstrap intervals, `[-7.03, +0.78]` and `[-6.99, +0.48]` points respectively, include zero. The point estimates therefore support Episode 50 as the current stopping choice but do not establish a statistically conclusive population ordering from 256 questions.

## Diagnostic interpretation

Two effects occur together:

1. The policy becomes more economical. Retrievals fall by 0.164 per question, or 11.2%, from Episode 50 to 100. Fourteen of sixteen evaluation batches reduce retrieval, and eleven reduce both retrieval and Gold-SF recall. Their batch-level deltas correlate at 0.58.
2. The final-answer behavior also drifts. Episode 100 wins 9 and loses 17 exact answers relative to Episode 50; 11 of those 17 losses fall to zero F1. Mean answer length rises from 3.23 to 3.42 words, so truncation is not the explanation. Lexically classified `who` and `what` questions contribute -3.57 F1 points, more than the -3.27-point net decline because other types partly offset them.

Online metrics do not contradict the held-out result. The ten episodes ending at 100 have higher reward, online EM, and online SF than the ten ending at 50, while using different sequential training questions. They are training observations rather than a fixed validation series. This is evidence of proxy/generalization mismatch, not by itself proof of causal reward hacking.

## Recommended next experiment

Treat Episode 50 as the v5 candidate checkpoint. Before a longer run:

1. Save per-question retrieval actions, queries, document ids/titles, Gold-SF hits, and stop positions.
2. Cache the processed evaluator by `(question, prediction)` across variants. Of 914 extractor calls, 282 were reusable; identical inputs disagreed on processed EM in two duplicate groups.
3. Protect the answer/stop decision with an evidence-coverage guard and protect final-answer behavior with adaptive KL, answer-token masking, or a small high-quality SFT rehearsal set.
4. Evaluate every 25 episodes on a fixed set using processed EM/F1, semantic accuracy, Gold-SF recall, and retrieval calls. Expand to 500 only after answer quality is non-decreasing while efficiency improves.

## Reproducible artifacts

- `episode_diagnostic.ipynb`: executed reader-facing notebook; 18 cells, no execution errors.
- `analysis_results.json`: canonical results, source hashes, paired intervals, and audit counts.
- `report_artifact.json`: validated and rendered Data Analytics report payload.
- `queries/`: seven executable DuckDB queries backing the report's cards, charts, and tables.
- `analyze_episode_checkpoints.py`: source-to-result analysis pipeline.
- `build_notebook.py` and `build_report_artifact.py`: reproducible presentation builders.

Run the Python analysis with the held-out result directory, processed-evaluation directory, and training log shown by `--help`. Run every report query from the repository root so its relative CSV paths resolve correctly.
