SELECT
  question_type,
  n,
  step50_processed_f1,
  step100_processed_f1,
  step100_vs_step50_f1_delta AS f1_delta,
  step100_vs_step50_f1_contribution AS f1_contribution,
  step100_vs_step50_em_delta AS em_delta
FROM read_csv_auto('experiments/2026-07-22_acec_v5_episode_diagnostic/question_type_drivers.csv')
ORDER BY f1_contribution;
