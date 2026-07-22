WITH per_question AS (
  SELECT *
  FROM read_csv_auto('experiments/2026-07-22_acec_v5_episode_diagnostic/per_question_diagnostic.csv')
),
batch_means AS (
  SELECT
    variant,
    avg(gold_sf_recall) AS gold_sf_recall,
    avg(retrieval_calls) AS retrieval_calls
  FROM read_csv_auto('experiments/2026-07-22_acec_v5_episode_diagnostic/batch_diagnostic.csv')
  WHERE variant IN ('step50', 'step100')
  GROUP BY variant
)
SELECT
  50 AS best_episode,
  avg(step100_processed_f1 - step50_processed_f1) AS step100_vs_step50_processed_f1_delta,
  (SELECT gold_sf_recall FROM batch_means WHERE variant = 'step100')
    - (SELECT gold_sf_recall FROM batch_means WHERE variant = 'step50')
    AS step100_vs_step50_gold_sf_delta,
  (SELECT retrieval_calls FROM batch_means WHERE variant = 'step100')
    - (SELECT retrieval_calls FROM batch_means WHERE variant = 'step50')
    AS step100_vs_step50_retrieval_delta,
  count(*) AS heldout_questions
FROM per_question;
