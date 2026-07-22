WITH batches AS (
  SELECT *
  FROM read_csv_auto('experiments/2026-07-22_acec_v5_episode_diagnostic/batch_diagnostic.csv')
  WHERE variant IN ('step50', 'step100')
),
step50 AS (SELECT * FROM batches WHERE variant = 'step50'),
step100 AS (SELECT * FROM batches WHERE variant = 'step100')
SELECT
  'Batch ' || step50.batch AS batch,
  step50.batch AS batch_number,
  step50.question_count,
  step100.retrieval_calls - step50.retrieval_calls AS retrieval_delta,
  step100.gold_sf_recall - step50.gold_sf_recall AS gold_sf_delta,
  step100.processed_f1 - step50.processed_f1 AS processed_f1_delta,
  step50.retrieval_calls AS step50_retrieval_calls,
  step100.retrieval_calls AS step100_retrieval_calls,
  step50.gold_sf_recall AS step50_gold_sf_recall,
  step100.gold_sf_recall AS step100_gold_sf_recall
FROM step50
JOIN step100 USING (batch)
ORDER BY batch_number;
