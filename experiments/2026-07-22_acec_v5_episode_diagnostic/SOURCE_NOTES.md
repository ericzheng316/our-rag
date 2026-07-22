# Source and validation notes

## Scope and audience

- Question: how Episodes 50, 75, and 100 differ, what drives the late regression, and whether unchanged v5 should run for 500/800 episodes.
- Audience: ACEC/RAG research and training implementation team.
- Evaluation population: the same 256 HotpotQA development questions, seed `20260721`, temperature 0, one rollout per checkpoint.
- Training grain: 100 episodes, each with 16 sequential questions and 8 samples per question.
- Retrieval grain available in the historical evaluation logs: 16-question batch, not individual question.
- Analysis date: 2026-07-22, Asia/Shanghai.

## Source integrity

The source files were read from `/root/data` on the CPU instance and analyzed locally. Their SHA-256 hashes are stored in `analysis_results.json`. Headline inputs include:

- `train.log`: `8c7f861c1042734d4715d831754e2ec602570363cf76a426cd452660ff66bb2a`
- `eval.log`: `25c83a0cc9bdf537065d61222bc248f7af1484f02df2668f32c351b20a0fc906`
- Step 50 processed predictions: `3bd98d8c751459f8f39aa2fab0e2bf76b9859c4ff6a7199731f2cbc438b6dccb`
- Step 75 processed predictions: `89600ec9e7e5b0af9b69773a212203db82ae4d6a460d24980f91f611a51ff75d`
- Step 100 processed predictions: `01b71d8bb630f3d94b746a017d8f66de7ab6f0c308b496707729dc48ae8d3f79`

All three variants contain the same 256 question ids. No question-level join loss or row multiplication was observed.

## Metric definitions

- Direct EM/F1: normalized HotpotQA answer metric applied directly to the generated answer.
- Processed EM/F1: maximum normalized answer metric over the original response and the recorded R3-extracted candidates.
- Exact win/loss: a paired processed-EM transition on the same question id.
- Gold-SF recall: fraction of the two gold supporting-fact titles found; historical logging only preserves its 16-question batch mean.
- Retrieval calls: mean retrieval actions per question.
- Question type: deterministic lexical prefix heuristic for diagnostics only; it is not an official HotpotQA field.
- Confidence intervals: 10,000 paired bootstrap resamples of the 256 question ids.

## Report map

| Report element | Dataset | SQL source |
| --- | --- | --- |
| Headline cards | `headline` | `queries/headline.sql` |
| Checkpoint chart/table | `checkpoint_long`, `checkpoint_metrics` | `queries/checkpoint_metrics_long.sql`, `queries/checkpoint_metrics.sql` |
| Question-type contribution | `question_drivers` | `queries/question_type_drivers.sql` |
| Exact-loss table | `step100_losses` | `queries/step100_losses.sql` |
| Retrieval/SF scatter | `batch_scatter` | `queries/batch_scatter.sql` |
| Online training trend | `training_long` | `queries/training_trend.sql` |

## Validation report

Overall assessment: **share with caveats**.

Verified checks:

- All seven SQL files execute successfully in DuckDB and produce the expected 1, 3, 6, 8, 17, 16, and 300 rows.
- The three checkpoint means independently reproduce the recorded processed and direct metrics.
- Exact transition counts reconcile: Episode 50→100 has 9 wins, 17 losses, 84 both-correct, and 146 both-wrong, totaling 256.
- The eight mutually exclusive stability patterns total 256 questions.
- Question-type counts total 256 and their weighted F1 contributions sum to the overall Episode-100-minus-50 processed-F1 delta.
- Batch counts are equal (16 questions each), so the mean of batch means has the intended weighting.
- The executed notebook contains 18 cells, eight stored outputs, and no error outputs.
- The native report artifact passed validation with seven bounded datasets and eight registered sources, then rendered successfully.
- No remote password or API key is stored in this experiment directory.

Required caveats:

- The paired confidence intervals cross zero; the current result chooses a checkpoint but is not a definitive statistical claim that Episode 50 dominates in the HotpotQA population.
- Retrieval-to-answer causality cannot be identified because per-question retrieval traces were not saved. Batch correlations are descriptive only.
- The processed-answer extractor is nondeterministic on a small number of identical inputs. Caching would remove this noise and slightly reduce evaluation cost; direct and substring metrics still agree with the overall direction.
- Online training windows use changing questions and are not comparable fixed-validation cohorts.
- Lexical question-type slices, especially the small numeric and where groups, are exploratory.

No handoff blocker remains for the checkpoint decision. A stronger paper claim requires a second fixed sample or a larger held-out evaluation plus per-question retrieval traces.
