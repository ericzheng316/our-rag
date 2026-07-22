WITH per_question AS (
  SELECT *
  FROM read_csv_auto('experiments/2026-07-22_acec_v5_episode_diagnostic/per_question_diagnostic.csv')
),
answer_metrics AS (
  SELECT 'step50' AS variant, 'Episode 50' AS checkpoint, 50 AS episode,
         avg(step50_direct_em) AS direct_em,
         avg(step50_direct_f1) AS direct_f1,
         avg(step50_processed_em) AS processed_em,
         avg(step50_processed_f1) AS processed_f1,
         avg(step50_prediction_words) AS mean_prediction_words
  FROM per_question
  UNION ALL
  SELECT 'step75', 'Episode 75', 75,
         avg(step75_direct_em), avg(step75_direct_f1),
         avg(step75_processed_em), avg(step75_processed_f1),
         avg(step75_prediction_words)
  FROM per_question
  UNION ALL
  SELECT 'step100', 'Episode 100', 100,
         avg(step100_direct_em), avg(step100_direct_f1),
         avg(step100_processed_em), avg(step100_processed_f1),
         avg(step100_prediction_words)
  FROM per_question
),
batch_metrics AS (
  SELECT variant,
         avg(gold_sf_recall) AS gold_sf_recall,
         avg(retrieval_calls) AS retrieval_calls
  FROM read_csv_auto('experiments/2026-07-22_acec_v5_episode_diagnostic/batch_diagnostic.csv')
  WHERE variant IN ('step50', 'step75', 'step100')
  GROUP BY variant
),
metrics AS (
  SELECT a.*, b.gold_sf_recall, b.retrieval_calls
  FROM answer_metrics a
  JOIN batch_metrics b USING (variant)
)
SELECT *, 'Processed EM' AS metric, processed_em AS value
FROM metrics
UNION ALL
SELECT *, 'Processed F1' AS metric, processed_f1 AS value
FROM metrics
ORDER BY episode, metric;
