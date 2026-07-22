SELECT
  id,
  question_type,
  question,
  golden_answers AS gold,
  step50_prediction AS step50_answer,
  step100_prediction AS step100_answer,
  step100_processed_f1 AS step100_f1,
  pattern_50_75_100 AS pattern
FROM read_csv_auto('experiments/2026-07-22_acec_v5_episode_diagnostic/transition_examples.csv')
WHERE starts_with(pattern_50_75_100, '1')
  AND ends_with(pattern_50_75_100, '0')
ORDER BY step100_f1, id;
